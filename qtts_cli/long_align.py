"""Pure planning, matching and attribution for long-audio alignment.

FR-P1-11 splits audio longer than the forced aligner's five-minute limit into
silence-bounded windows.  This module contains only the deterministic,
model-free stages:

* :func:`plan_windows` -> non-overlapping cores plus overlapping audio windows;
* :func:`monotonic_window_ranges` -> ASR transcripts to global reference spans;
* :func:`merge_window_items` -> one timing per display character.

No torch / qwen-asr imports happen here, so every rule below is unit-testable
without downloading a model.
"""

from __future__ import annotations

import math
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Optional, Sequence

import numpy as np

from .errors import InputError, UsageError
from .mapped_text import MappedText, mapped_normalize
from .qc import normalize_for_qc

DEFAULT_WINDOW_S = 270.0
MIN_WINDOW_S = 240.0
MAX_WINDOW_S = 285.0
DEFAULT_OVERLAP_S = 3.0
MIN_OVERLAP_S = 2.0
DEFAULT_SILENCE_S = 0.6
SILENCE_DBFS = -45.0
FRAME_MS = 20.0
MAX_ALIGN_S = 300.0
TIME_EPSILON = 1e-6
MIN_MATCH_RATIO = 0.5


@dataclass(frozen=True)
class AudioWindow:
    """One aligner call: a core ownership range plus surrounding audio context.

    Sample positions are authoritative.  Seconds are derived from ``sr`` so no
    window boundary is ever reconstructed from a rounded float.
    """

    index: int
    sr: int
    audio_start_sample: int
    audio_end_sample: int
    core_start_sample: int
    core_end_sample: int
    display_start: Optional[int] = None
    display_end: Optional[int] = None
    spoken_start: Optional[int] = None
    spoken_end: Optional[int] = None

    @property
    def audio_start_s(self) -> float:
        return self.audio_start_sample / self.sr

    @property
    def audio_end_s(self) -> float:
        return self.audio_end_sample / self.sr

    @property
    def core_start_s(self) -> float:
        return self.core_start_sample / self.sr

    @property
    def core_end_s(self) -> float:
        return self.core_end_sample / self.sr

    @property
    def audio_duration_s(self) -> float:
        return (self.audio_end_sample - self.audio_start_sample) / self.sr

    @property
    def core_duration_s(self) -> float:
        return (self.core_end_sample - self.core_start_sample) / self.sr

    def with_text_ranges(
        self,
        *,
        display_start: Optional[int] = None,
        display_end: Optional[int] = None,
        spoken_start: Optional[int] = None,
        spoken_end: Optional[int] = None,
    ) -> "AudioWindow":
        return AudioWindow(
            index=self.index,
            sr=self.sr,
            audio_start_sample=self.audio_start_sample,
            audio_end_sample=self.audio_end_sample,
            core_start_sample=self.core_start_sample,
            core_end_sample=self.core_end_sample,
            display_start=self.display_start if display_start is None else display_start,
            display_end=self.display_end if display_end is None else display_end,
            spoken_start=self.spoken_start if spoken_start is None else spoken_start,
            spoken_end=self.spoken_end if spoken_end is None else spoken_end,
        )


class WindowPlan(list[AudioWindow]):
    """A window list tagged with how the cuts were chosen."""

    def __init__(self, windows, kind: str) -> None:
        super().__init__(windows)
        self.kind = kind


@dataclass(frozen=True)
class WindowItem:
    """Aligner item already projected to global display character offsets."""

    text: str
    start_s: float
    end_s: float
    display_start: int
    display_end: int


@dataclass(frozen=True)
class WindowAlignment:
    window: AudioWindow
    items: Sequence[WindowItem]


@dataclass(frozen=True)
class CharTiming:
    index: int
    char: str
    start_s: float
    end_s: float


def _usage_number(name: str, value: float) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise UsageError(f"{name} must be finite", details={name: value})
    return number


def detect_silence_runs(
    samples: np.ndarray,
    sr: int,
    *,
    frame_ms: float = FRAME_MS,
    threshold_dbfs: float = SILENCE_DBFS,
    min_silence_s: float = DEFAULT_SILENCE_S,
) -> list[tuple[int, int]]:
    """Return ``[start_sample, end_sample)`` runs of qualifying silence.

    The last partial frame is evaluated on its real samples instead of being
    dropped (``audio_io.frame_rms`` truncates it), so audio ending in silence
    cannot hide the final 0-19 ms from the planner.
    """
    if sr <= 0:
        raise UsageError("sr must be a positive integer", details={"sr": sr})
    frame_ms = _usage_number("frame_ms", frame_ms)
    threshold_dbfs = _usage_number("threshold_dbfs", threshold_dbfs)
    min_silence_s = _usage_number("min_silence_s", min_silence_s)
    if frame_ms <= 0 or min_silence_s < 0:
        raise UsageError("frame_ms must be positive and min_silence_s >= 0")
    data = np.asarray(samples, dtype=np.float32)
    if data.ndim > 1:
        data = data.mean(axis=1).astype(np.float32)
    if data.size == 0:
        return []
    frame = max(1, int(round(sr * frame_ms / 1000.0)))
    minimum = int(math.ceil(min_silence_s * sr - 1e-9))

    runs: list[tuple[int, int]] = []
    run_start: Optional[int] = None
    for start in range(0, data.size, frame):
        end = min(start + frame, data.size)
        chunk = data[start:end]
        rms = float(np.sqrt(np.mean(np.square(chunk, dtype=np.float64))))
        silent = 20.0 * math.log10(max(rms, 1e-12)) <= threshold_dbfs
        if silent and run_start is None:
            run_start = start
        elif not silent and run_start is not None:
            if end - run_start >= minimum:
                runs.append((run_start, start))
            run_start = None
    if run_start is not None and data.size - run_start >= minimum:
        runs.append((run_start, data.size))
    return runs


def _sample(moment_s: float, sr: int) -> int:
    return int(round(moment_s * sr))


def _silence_midpoints(
    samples: np.ndarray, sr: int, min_silence_s: float
) -> list[int]:
    return [
        (start + end) // 2
        for start, end in detect_silence_runs(
            samples, sr, min_silence_s=min_silence_s
        )
    ]


def validate_srt_cues(
    cues: Sequence[tuple[float, float]],
    duration_s: float,
    *,
    max_window_s: float = MAX_WINDOW_S,
) -> None:
    """Validate SRT cue spans before they become hard window anchors."""
    duration_s = _usage_number("duration_s", duration_s)
    max_window_s = _usage_number("max_window_s", max_window_s)
    if duration_s <= 0:
        raise InputError("audio duration must be positive")
    if not cues:
        raise InputError("SRT input contains no cues")
    previous_end = 0.0
    for index, cue in enumerate(cues):
        try:
            start, end = float(cue[0]), float(cue[1])
        except (TypeError, ValueError, IndexError) as exc:
            raise InputError(f"invalid SRT cue at index {index}") from exc
        if not math.isfinite(start) or not math.isfinite(end):
            raise InputError(f"SRT cue {index} has non-finite times")
        if start < 0 or end <= start:
            raise InputError(f"SRT cue {index} has invalid start/end times")
        if start < previous_end - TIME_EPSILON:
            raise InputError(f"SRT cue {index} overlaps or is out of order")
        if end > duration_s + TIME_EPSILON:
            raise InputError(f"SRT cue {index} ends after the audio")
        if end - start > max_window_s + TIME_EPSILON:
            raise InputError(
                "cue exceeds aligner window limit",
                details={
                    "cue_index": index,
                    "duration_s": round(end - start, 3),
                    "limit_s": max_window_s,
                },
            )
        previous_end = end


def _cue_candidate_samples(
    cue_boundaries: Sequence[tuple[float, int]],
    *,
    samples: np.ndarray,
    sr: int,
    min_silence_s: float,
) -> list[int]:
    if not cue_boundaries:
        raise InputError("SRT input contains no cue boundaries")
    duration = len(samples) / float(sr)
    parsed: list[tuple[float, int]] = []
    for index, boundary in enumerate(cue_boundaries):
        try:
            time_s, char_index = float(boundary[0]), int(boundary[1])
        except (TypeError, ValueError, IndexError) as exc:
            raise InputError(
                f"invalid SRT cue boundary at index {index}"
            ) from exc
        if not math.isfinite(time_s) or time_s < 0:
            raise InputError(f"SRT cue boundary {index} has an invalid time")
        if time_s > duration + TIME_EPSILON:
            raise InputError(f"SRT cue boundary {index} is after the audio")
        if char_index < 0:
            raise InputError(f"SRT cue boundary {index} has a negative character index")
        if parsed and (
            time_s < parsed[-1][0] - TIME_EPSILON
            or char_index < parsed[-1][1]
        ):
            raise InputError("SRT cue boundaries are not monotonic")
        parsed.append((time_s, char_index))

    runs = detect_silence_runs(samples, sr, min_silence_s=min_silence_s)
    candidates: list[int] = []
    for time_s, _char_index in parsed:
        sample = _sample(time_s, sr)
        if sample <= 0 or sample >= len(samples):
            continue
        if any(start <= sample <= end for start, end in runs):
            candidates.append(sample)
    return sorted(set(candidates))


def _plan_cuts(
    total_samples: int,
    candidates: Sequence[int],
    *,
    min_samples: int,
    max_samples: int,
    target_samples: int,
    overlap_samples: int,
    max_align_samples: int,
    failure_reason: str,
) -> list[int]:
    usable = sorted(
        {sample for sample in candidates if 0 < sample < total_samples}
    )
    best: dict[int, tuple[float, tuple[int, ...]]] = {0: (0.0, ())}
    for sample in usable:
        chosen = None
        for previous in [0, *[p for p in usable if p < sample]]:
            state = best.get(previous)
            if state is None:
                continue
            step = sample - previous
            if step < min_samples or step > max_samples:
                continue
            cost = state[0] + abs(step - target_samples)
            path = state[1] + (sample,)
            key = (cost, path)
            if chosen is None or key < chosen:
                chosen = key
        if chosen is not None:
            best[sample] = chosen

    final: Optional[tuple[float, tuple[int, ...]]] = None
    for previous, state in best.items():
        if previous == 0:
            continue
        audio_start = max(0, previous - overlap_samples)
        if total_samples - audio_start > max_align_samples:
            continue
        step = total_samples - previous
        if step <= 0 or step > max_samples:
            continue
        key = (state[0] + abs(step - target_samples), state[1])
        if final is None or key < final:
            final = key
    if final is None:
        raise InputError(
            failure_reason,
            details={"candidate_count": len(usable)},
        )
    return list(final[1])


def plan_windows(
    samples: np.ndarray,
    sr: int,
    *,
    window_s: float = DEFAULT_WINDOW_S,
    min_window_s: float = MIN_WINDOW_S,
    max_window_s: float = MAX_WINDOW_S,
    overlap_s: float = DEFAULT_OVERLAP_S,
    min_silence_s: float = DEFAULT_SILENCE_S,
    cue_boundaries: Optional[Sequence[tuple[float, int]]] = None,
    max_align_s: float = MAX_ALIGN_S,
    approximate: bool = False,
) -> WindowPlan:
    """Plan non-overlapping cores with overlapping audio context windows.

    Non-final cores are 240-285 s and are cut at a qualified silence midpoint
    (or an SRT cue boundary inside qualified silence).  A global dynamic
    program avoids greedy dead ends.  With ``approximate=True`` only, a failed
    silence plan falls back to deterministic fixed-interval cuts.  The final
    core may be shorter than ``min_window_s``; every audio window is validated
    against ``max_align_s``.  Text ranges are intentionally left ``None``.
    """
    if sr <= 0:
        raise UsageError("sr must be a positive integer", details={"sr": sr})
    window_s = _usage_number("window_s", window_s)
    min_window_s = _usage_number("min_window_s", min_window_s)
    max_window_s = _usage_number("max_window_s", max_window_s)
    overlap_s = _usage_number("overlap_s", overlap_s)
    min_silence_s = _usage_number("min_silence_s", min_silence_s)
    max_align_s = _usage_number("max_align_s", max_align_s)
    if min_window_s <= 0 or window_s <= 0 or max_window_s <= 0:
        raise UsageError("window durations must be positive")
    if min_window_s > window_s or window_s > max_window_s:
        raise UsageError("require min_window_s <= window_s <= max_window_s")
    if overlap_s < MIN_OVERLAP_S or overlap_s >= min_window_s:
        raise UsageError(
            f"overlap_s must be >= {MIN_OVERLAP_S:g}s and smaller than min_window_s",
            details={"overlap_s": overlap_s, "min_window_s": min_window_s},
        )
    if min_silence_s < 0:
        raise UsageError("min_silence_s must be >= 0")
    if max_window_s + overlap_s > max_align_s + TIME_EPSILON:
        raise UsageError(
            "max_window_s + overlap_s exceeds the aligner limit",
            details={
                "max_window_s": max_window_s,
                "overlap_s": overlap_s,
                "max_align_s": max_align_s,
            },
        )

    data = np.asarray(samples, dtype=np.float32)
    if data.ndim > 1:
        data = data.mean(axis=1).astype(np.float32)
    if data.size == 0:
        raise InputError("audio is empty; cannot plan alignment windows")
    total = int(data.size)
    max_align_samples = int(math.floor(max_align_s * sr + TIME_EPSILON))
    overlap_samples = _sample(overlap_s, sr)

    if total <= max_align_samples:
        kind = "cue" if cue_boundaries is not None else "silence"
        return WindowPlan([AudioWindow(0, sr, 0, total, 0, total)], kind)

    minimum = int(math.ceil(min_window_s * sr - TIME_EPSILON))
    maximum = int(math.floor(max_window_s * sr + TIME_EPSILON))
    target = _sample(window_s, sr)
    kind = "cue" if cue_boundaries is not None else "silence"
    if cue_boundaries is None:
        cuts = _silence_midpoints(data, sr, min_silence_s)
        failure = (
            "no valid silence window plan; audio exceeds the aligner limit "
            "and no 240-285s alignment of qualifying silences exists"
        )
        if not approximate:
            failure += " (use --approximate to override)"
    else:
        cuts = _cue_candidate_samples(
            cue_boundaries,
            samples=data,
            sr=sr,
            min_silence_s=min_silence_s,
        )
        failure = (
            "no valid SRT cue window plan: every cut must be a cue boundary "
            "inside qualifying silence with 240-285s cores"
        )

    try:
        planned = _plan_cuts(
            total,
            cuts,
            min_samples=minimum,
            max_samples=maximum,
            target_samples=target,
            overlap_samples=overlap_samples,
            max_align_samples=max_align_samples,
            failure_reason=failure,
        )
    except InputError:
        if not approximate or cue_boundaries is not None:
            raise
        planned = _fixed_cut_samples(
            total,
            step=target,
            max_core_samples=maximum,
        )
        kind = "fixed"

    boundaries = [0, *planned, total]
    windows: list[AudioWindow] = []
    for index in range(len(boundaries) - 1):
        core_start = boundaries[index]
        core_end = boundaries[index + 1]
        core_length = core_end - core_start
        last = index == len(boundaries) - 2
        if core_length <= 0:
            raise InputError(
                "planned window core is empty",
                details={"window_index": index},
            )
        if core_length > maximum:
            raise InputError(
                "planned window core exceeds the maximum window length",
                details={
                    "window_index": index,
                    "core_duration_s": round(core_length / float(sr), 3),
                    "max_window_s": max_window_s,
                },
            )
        if not last and core_length < minimum:
            raise InputError(
                "planned window core is shorter than the minimum window length",
                details={
                    "window_index": index,
                    "core_duration_s": round(core_length / float(sr), 3),
                    "min_window_s": min_window_s,
                },
            )
        audio_start = max(0, core_start - overlap_samples)
        audio_end = core_end
        if audio_end - audio_start > max_align_samples:
            raise UsageError(
                "planned audio window exceeds the aligner limit",
                details={
                    "window_index": index,
                    "audio_duration_s": round(
                        (audio_end - audio_start) / float(sr), 3
                    ),
                },
            )
        windows.append(
            AudioWindow(
                index=index,
                sr=sr,
                audio_start_sample=audio_start,
                audio_end_sample=audio_end,
                core_start_sample=core_start,
                core_end_sample=core_end,
            )
        )
    return WindowPlan(windows, kind)


def _fixed_cut_samples(
    total_samples: int,
    *,
    step: int,
    max_core_samples: int,
) -> list[int]:
    """Deterministic fixed-interval cuts used only by approximate mode.

    Cuts are derived from sample positions, never from rounded seconds.  A new
    cut is added while the remaining core would exceed the maximum core size;
    this also keeps the eventual final audio window (core plus left overlap)
    inside the aligner limit because ``max_window_s + overlap_s <= max_align_s``.
    """
    if step <= 0 or max_core_samples <= 0:
        raise UsageError("fixed window step must be positive")
    step = min(step, max_core_samples)
    cuts: list[int] = []
    core_start = 0
    while total_samples - core_start > max_core_samples:
        next_cut = core_start + step
        if next_cut >= total_samples:
            break
        cuts.append(next_cut)
        core_start = next_cut
    return cuts


def _occurrences(text: str, needle: str) -> list[int]:
    if not needle:
        return []
    found: list[int] = []
    start = 0
    while True:
        position = text.find(needle, start)
        if position < 0:
            break
        found.append(position)
        start = position + 1
    return found


def _unique_sequence(
    reference: str, spans: Sequence[tuple[int, int]]
) -> list[tuple[int, int]]:
    """Ensure repeated spans have exactly one monotonic occurrence sequence.

    Each window span is searched in the normalized reference.  A first window
    must start at 0, a last window must end at the reference end.  Adjacent
    occurrences must advance end offsets and may overlap but not leave a gap.
    """
    candidates_by_window: list[list[tuple[int, int]]] = []
    for index, (start, end) in enumerate(spans):
        span_text = reference[start:end]
        occurrences = _occurrences(reference, span_text)
        options = [(position, position + len(span_text)) for position in occurrences]
        if index == 0:
            options = [option for option in options if option[0] == 0]
        if index == len(spans) - 1:
            options = [option for option in options if option[1] == len(reference)]
        if not options:
            raise InputError(
                "window text range does not have a unique monotonic anchor",
                details={"reason": "ambiguous_alignment_anchor", "window": index},
            )
        candidates_by_window.append(options)

    first_options = candidates_by_window[0]
    if len(first_options) > 1:
        raise InputError(
            "multiple equivalent alignment anchors",
            details={"reason": "ambiguous_alignment_anchor", "window": 0},
        )
    paths: list[tuple[tuple[int, int], ...]] = [
        (option,) for option in first_options
    ]
    for index in range(1, len(candidates_by_window)):
        options = candidates_by_window[index]
        next_paths: list[tuple[tuple[int, int], ...]] = []
        for path in paths:
            previous_start, previous_end = path[-1]
            for start, end in options:
                if start < previous_start:
                    continue
                if start > previous_end:
                    continue
                if end <= previous_end:
                    continue
                next_paths.append(path + ((start, end),))
        unique = sorted(set(next_paths))
        if not unique:
            raise InputError(
                "window text ranges cannot cover the reference monotonically",
                details={"reason": "monotonic_coverage", "window": index},
            )
        if len(unique) > 1:
            raise InputError(
                "multiple equivalent alignment anchors",
                details={"reason": "ambiguous_alignment_anchor", "window": index},
            )
        paths = unique
    return list(paths[0])


def monotonic_window_ranges(
    reference: str,
    window_transcripts: Sequence[str],
    *,
    language: str = "auto",
    windows: Optional[Sequence[AudioWindow]] = None,
    window_cores: Optional[Sequence[tuple[float, float]]] = None,
    duration_s: Optional[float] = None,
) -> list[tuple[int, int]]:
    """Map one ASR transcript per audio window to global source spans.

    Matching runs on normalized ``PreparedText.spoken_text`` (passed as the
    caller), and every returned boundary is converted through
    :class:`MappedText` provenance, never by reusing normalized offsets or
    length ratios.  When window/core timing is supplied, each window must also
    advance a plausible share of the reference and the final window must match
    close to the end of its suffix.
    """
    if not window_transcripts:
        raise InputError("no window transcripts were produced")
    cores: Optional[list[tuple[float, float]]] = None
    if windows is not None:
        cores = [(window.core_start_s, window.core_end_s) for window in windows]
    elif window_cores is not None:
        cores = [(float(start), float(end)) for start, end in window_cores]
    if cores is not None:
        if len(cores) != len(window_transcripts):
            raise InputError(
                "window core count does not match transcript count",
                details={
                    "cores": len(cores),
                    "transcripts": len(window_transcripts),
                },
            )
        if duration_s is None:
            duration_s = max(end for _start, end in cores)
        if duration_s <= 0:
            raise UsageError("duration_s must be positive")
    mapped: MappedText = mapped_normalize(reference, language)
    if not mapped.text:
        raise InputError("reference text is empty after normalization")

    transcripts = [
        normalize_for_qc(transcript, mapped.language)
        for transcript in window_transcripts
    ]
    for index, transcript in enumerate(transcripts):
        if not transcript:
            raise InputError(
                "window transcript is empty after normalization",
                details={"window": index, "reason": "empty_transcript"},
            )

    reference_units = list(mapped.text)
    spans: list[tuple[int, int]] = []
    previous_end = 0
    for index, transcript in enumerate(transcripts):
        # Each window owns new text from the previous end onward.  Audio
        # overlap appears as transcript prefix/suffix insertions here; it does
        # not move the ownership cursor backwards.
        search_start = previous_end
        suffix = reference_units[search_start:]
        matcher = SequenceMatcher(None, suffix, list(transcript), autojunk=False)
        blocks = [block for block in matcher.get_matching_blocks() if block.size]
        if not blocks:
            raise InputError(
                "window transcript does not match the reference text",
                details={"window": index, "reason": "no_match"},
            )
        matched = sum(block.size for block in blocks)
        if matched / max(1, len(transcript)) < MIN_MATCH_RATIO:
            raise InputError(
                "window transcript matches too little of the reference text",
                details={"window": index, "reason": "insufficient_match"},
            )
        last = index == len(transcripts) - 1
        block_end = max(block.a + block.size for block in blocks)
        if last:
            tolerance = max(2, math.ceil(0.15 * len(suffix)))
            if block_end < len(suffix) - tolerance:
                raise InputError(
                    "last window transcript leaves the reference tail uncovered",
                    details={
                        "window": index,
                        "reason": "uncovered_reference_tail",
                        "suffix_chars": len(suffix),
                        "matched_end": block_end,
                    },
                )
        start = previous_end
        end = search_start + block_end
        if last:
            end = len(mapped.text)
        if end <= previous_end:
            raise InputError(
                "window transcript does not cover new reference text",
                details={"window": index, "reason": "monotonic_coverage"},
            )
        if cores is not None and duration_s is not None:
            core_start, core_end = cores[index]
            expected = max(
                1,
                round(len(mapped.text) * max(0.0, core_end - core_start) / duration_s),
            )
            advance = end - previous_end
            if advance < 0.5 * expected:
                raise InputError(
                    "window transcript advances too little of the reference text",
                    details={
                        "window": index,
                        "reason": "insufficient_reference_coverage",
                        "advance_chars": advance,
                        "expected_chars": expected,
                    },
                )
        spans.append((start, end))
        previous_end = end

    unique_spans = _unique_sequence(mapped.text, spans)

    source_ranges: list[tuple[int, int]] = []
    previous_source_end = 0
    for index, (_start, end) in enumerate(unique_spans):
        # Start at the previous source end so punctuation removed by
        # normalization cannot fall into a coverage hole between windows.
        source_start = previous_source_end
        source_end = (
            len(reference)
            if end == len(mapped.text)
            else mapped.end_boundary(end)
        )
        if source_end <= source_start:
            raise InputError(
                "normalized window boundary maps to an empty source range",
                details={"window": index, "reason": "empty_source_range"},
            )
        source_ranges.append((source_start, source_end))
        previous_source_end = source_end

    if source_ranges[0][0] != 0 or source_ranges[-1][1] != len(reference):
        raise InputError(
            "window text ranges do not cover the full reference",
            details={"reason": "monotonic_coverage"},
        )
    previous_start, previous_end = source_ranges[0]
    for start, end in source_ranges[1:]:
        if start < previous_start or end <= previous_end or start > previous_end:
            raise InputError(
                "window source ranges are not monotonic",
                details={"reason": "non_monotonic_source_ranges"},
            )
        previous_start, previous_end = start, end
    return source_ranges


def _is_fillable(char: str) -> bool:
    return char.isspace() or unicodedata.category(char)[0] in ("P", "S")


@dataclass
class _MergeState:
    non_core: int
    windows: tuple[int, ...]
    times: tuple[tuple[float, float], ...]
    last_start: float
    last_end: float
    chosen: list[tuple[int, CharTiming, float, float]]

    def objective(self) -> tuple:
        return self.non_core, self.windows, self.times


def merge_window_items(
    display_text: str,
    window_alignments: Sequence[WindowAlignment],
    *,
    duration_s: float,
    approximate: bool = False,
    stats: Optional[dict] = None,
    tail_allowance_s: float = 0.0,
) -> list[CharTiming]:
    """Choose exactly one timing per display character (character ownership).

    A candidate whose time lies inside its own window core wins over one that
    merely overlaps the core.  Candidates are selected with a global DP whose
    objective is (non-core candidates, window sequence, time sequence), so a
    locally attractive candidate cannot dead-end a later character.

    ``approximate=True`` clamps regressing times to a monotonic sequence and
    records the count in ``stats["monotonic_adjustments"]``.  The final window
    may extend at most ``tail_allowance_s`` beyond the audio, and is clipped.
    """
    duration_s = _usage_number("duration_s", duration_s)
    tail_allowance_s = _usage_number("tail_allowance_s", tail_allowance_s)
    if duration_s <= 0:
        raise InputError("audio duration must be positive")
    if tail_allowance_s < 0:
        raise UsageError("tail_allowance_s must be >= 0")
    if not display_text:
        return []

    ordered = sorted(window_alignments, key=lambda value: value.window.index)
    last_window_index = ordered[-1].window.index if ordered else -1
    candidates: list[list[tuple[int, int, float, float]]] = [
        [] for _ in range(len(display_text))
    ]
    for alignment in ordered:
        window = alignment.window
        allowance = tail_allowance_s if window.index == last_window_index else 0.0
        for item in alignment.items:
            if item.display_start < 0 or item.display_end > len(display_text):
                raise InputError(
                    "window item display range is outside the reference",
                    details={
                        "window": window.index,
                        "display_start": item.display_start,
                        "display_end": item.display_end,
                    },
                )
            if item.display_end <= item.display_start:
                raise InputError(
                    "window item display range is empty",
                    details={"window": window.index},
                )
            raw_start = float(item.start_s)
            raw_end = float(item.end_s)
            if not math.isfinite(raw_start) or not math.isfinite(raw_end):
                raise InputError(
                    "window item has non-finite times",
                    details={"window": window.index},
                )
            if raw_end < raw_start - TIME_EPSILON:
                raise InputError(
                    "window item end precedes start",
                    details={"window": window.index},
                )
            window_start = window.audio_start_s
            window_end = window.audio_end_s
            if raw_start >= duration_s:
                raise InputError(
                    "window item start is outside the audio duration",
                    details={
                        "window": window.index,
                        "start_s": raw_start,
                        "duration_s": duration_s,
                    },
                )
            if (
                raw_start < window_start - TIME_EPSILON
                or raw_end < window_start - TIME_EPSILON
                or raw_start >= window_end + TIME_EPSILON
                or raw_end > window_end + allowance + TIME_EPSILON
            ):
                raise InputError(
                    "window item is outside the audio range of its window",
                    details={
                        "reason": "item_outside_window_audio",
                        "window": window.index,
                        "start_s": raw_start,
                        "end_s": raw_end,
                        "window_audio_start_s": round(window_start, 6),
                        "window_audio_end_s": round(window_end, 6),
                        "allowance_s": allowance,
                    },
                )
            start = min(max(window_start, raw_start), window_end)
            end = min(max(start, raw_end), window_end)
            inside_core = (
                start >= window.core_start_s - TIME_EPSILON
                and end <= window.core_end_s + TIME_EPSILON
            )
            preference = 0 if inside_core else 1
            for char_index in range(item.display_start, item.display_end):
                candidates[char_index].append(
                    (preference, window.index, start, end)
                )

    states = [
        _MergeState(
            non_core=0,
            windows=(),
            times=(),
            last_start=0.0,
            last_end=0.0,
            chosen=[],
        )
    ]
    for index, char in enumerate(display_text):
        options = sorted(candidates[index])
        if not options:
            if not approximate and not _is_fillable(char):
                raise InputError(
                    "no aligner item covers this reference character",
                    details={"char_index": index, "reason": "missing_coverage"},
                )
            continue
        next_states: dict[tuple[float, float], _MergeState] = {}
        for state in states:
            for preference, window_index, raw_start, raw_end in options:
                if approximate:
                    start = max(raw_start, state.last_end)
                    end = max(raw_end, start)
                else:
                    if (
                        raw_start < state.last_start - TIME_EPSILON
                        or raw_end < state.last_end - TIME_EPSILON
                    ):
                        continue
                    start = max(raw_start, state.last_start)
                    end = max(raw_end, state.last_end)
                timing = CharTiming(index, char, start, end)
                updated = _MergeState(
                    non_core=state.non_core + preference,
                    windows=state.windows + (window_index,),
                    times=state.times + ((start, end),),
                    last_start=start,
                    last_end=end,
                    chosen=state.chosen
                    + [(index, timing, raw_start, raw_end)],
                )
                key = (start, end)
                current = next_states.get(key)
                if current is None or updated.objective() < current.objective():
                    next_states[key] = updated
        if not next_states:
            raise InputError(
                "window items cannot be ordered monotonically",
                details={"char_index": index, "reason": "non_monotonic_times"},
            )
        states = list(next_states.values())

    best = min(states, key=lambda state: state.objective())
    chosen: list[Optional[CharTiming]] = [None] * len(display_text)
    adjustments = 0
    previous_end = 0.0
    for index, timing, raw_start, raw_end in best.chosen:
        if approximate and (
            raw_start < previous_end - TIME_EPSILON
            or raw_end < previous_end - TIME_EPSILON
        ):
            adjustments += 1
        chosen[index] = timing
        previous_end = timing.end_s

    previous: Optional[CharTiming] = None
    for index, timing in enumerate(chosen):
        if timing is not None:
            previous = timing
            continue
        following = next(
            (value for value in chosen[index + 1:] if value is not None), None
        )
        source = previous or following
        if source is None:
            chosen[index] = CharTiming(index, display_text[index], 0.0, 0.0)
        else:
            chosen[index] = CharTiming(
                index, display_text[index], source.start_s, source.end_s
            )

    result = [timing for timing in chosen if timing is not None]
    last_start = 0.0
    last_end = 0.0
    for timing in result:
        if (
            timing.start_s < last_start - TIME_EPSILON
            or timing.end_s < last_end - TIME_EPSILON
            or timing.end_s < timing.start_s - TIME_EPSILON
        ):
            raise InputError(
                "merged character times are not monotonic",
                details={"char_index": timing.index},
            )
        last_start = max(last_start, timing.start_s)
        last_end = max(last_end, timing.end_s)
    if stats is not None:
        stats["monotonic_adjustments"] = adjustments
    return result
