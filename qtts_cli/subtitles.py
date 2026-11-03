"""Subtitle cue construction and SRT/VTT/ASS/JSON rendering (FR-P0-05).

Content invariant: cues are carved out of the *normalized source text* by index,
so ``"".join(cue.text for cue in cues) == text``. The aligner contributes timing
only; it never supplies subtitle text (PRD §4.1).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence

from .errors import InputError, UsageError
from .segmenter import _cut_inside_token, _quote_depths, segment_text

SUBTITLE_FORMATS = ("srt", "vtt", "ass", "json")
GRANULARITIES = ("sentence", "char", "word")
MIN_CUE_SECONDS = 0.001

# P1 subtitle policy defaults (FR-P1-05a/05b). Display width counts CJK and
# full-width characters as 2 cells.
DEFAULT_MAX_CUE_CHARS = 42
DEFAULT_MAX_LINE_WIDTH = 42
DEFAULT_MAX_CPS = 20.0
DEFAULT_MIN_CUE_SECONDS = 0.8
DEFAULT_MAX_CUE_SECONDS = 7.0
DEFAULT_MAX_MERGE_GAP_MS = 300

_SPLIT_PUNCT = set("，,、：:；;—–-")
_SPLIT_SPACE = set(" \t　")
_HANGING_CLOSE = set("，。！？、；：,.!?;:)]}）】》」』”’")
_HANGING_OPEN = set("([{（【《「『“‘")


def display_width_char(ch: str) -> int:
    """Terminal display width: CJK/full-width/emoji = 2, combining = 0."""
    if ch in "\r\n":
        return 0
    if unicodedata.combining(ch):
        return 0
    return 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1


def display_width(text: str, *, ignore_whitespace: bool = False) -> int:
    if ignore_whitespace:
        return sum(display_width_char(ch) for ch in text if not ch.isspace())
    return sum(display_width_char(ch) for ch in text)


@dataclass(frozen=True)
class SubtitlePolicy:
    """Display/timing constraints shared by segmentation and QC."""

    max_cue_chars: int = DEFAULT_MAX_CUE_CHARS
    max_line_width: int = DEFAULT_MAX_LINE_WIDTH
    max_cps: float = DEFAULT_MAX_CPS
    min_cue_seconds: float = DEFAULT_MIN_CUE_SECONDS
    max_cue_seconds: float = DEFAULT_MAX_CUE_SECONDS
    max_merge_gap_ms: int = DEFAULT_MAX_MERGE_GAP_MS
    qc_enabled: bool = True

    def __post_init__(self) -> None:
        if self.max_cue_chars <= 0:
            raise UsageError("--max-cue-chars must be a positive integer")
        if self.max_line_width <= 0:
            raise UsageError("--max-line-width must be a positive integer")
        if self.max_cps <= 0:
            raise UsageError("--max-cps must be positive")
        if self.min_cue_seconds < 0:
            raise UsageError("--min-cue-seconds must be >= 0")
        if self.max_cue_seconds <= 0:
            raise UsageError("--max-cue-seconds must be positive")
        if self.min_cue_seconds > self.max_cue_seconds:
            raise UsageError("--min-cue-seconds cannot exceed --max-cue-seconds")
        if self.max_merge_gap_ms < 0:
            raise UsageError("--max-merge-gap-ms must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_cue_chars": self.max_cue_chars,
            "max_line_width": self.max_line_width,
            "max_cps": self.max_cps,
            "min_cue_seconds": self.min_cue_seconds,
            "max_cue_seconds": self.max_cue_seconds,
            "max_merge_gap_ms": self.max_merge_gap_ms,
            "qc_enabled": self.qc_enabled,
        }


@dataclass(frozen=True)
class SourceSpan:
    """A character interval plus the boundaries it descends from.

    ``parent_ids`` carries the three hard boundaries from t10: SRT input cue >
    TTS plan segment > original sentence. Cues may only merge when every parent
    id matches; ``--srt`` ranges therefore can never be regrouped.
    """

    start: int
    end: int
    parent_ids: tuple[str, ...] = ()


def _cue_width_limit(policy: SubtitlePolicy) -> int:
    """The stricter of cue width and line width: no renderer auto-wraps."""
    return min(policy.max_cue_chars, policy.max_line_width)


def policy_from_args(args: Any) -> SubtitlePolicy:
    return SubtitlePolicy(
        max_cue_chars=int(getattr(args, "max_cue_chars", DEFAULT_MAX_CUE_CHARS)),
        max_line_width=int(getattr(args, "max_line_width", DEFAULT_MAX_LINE_WIDTH)),
        max_cps=float(getattr(args, "max_cps", DEFAULT_MAX_CPS)),
        min_cue_seconds=float(getattr(args, "min_cue_seconds", DEFAULT_MIN_CUE_SECONDS)),
        max_cue_seconds=float(getattr(args, "max_cue_seconds", DEFAULT_MAX_CUE_SECONDS)),
        max_merge_gap_ms=int(
            getattr(args, "max_merge_gap_ms", DEFAULT_MAX_MERGE_GAP_MS)
        ),
        qc_enabled=not bool(getattr(args, "no_subtitle_qc", False)),
    )


@dataclass
class AlignItem:
    text: str
    start: float
    end: float


@dataclass
class Cue:
    start: float
    end: float
    text: str
    index: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "text": self.text,
        }
        if self.index is not None:
            payload["index"] = self.index
        return payload

    def shifted(self, offset: float) -> "Cue":
        return Cue(self.start + offset, self.end + offset, self.text, self.index)


@dataclass
class SrtCue:
    index: int
    start: float
    end: float
    text: str


_SRT_TIME = re.compile(
    r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[,.](\d{3})"
)


def _srt_seconds(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def parse_srt(raw: str) -> list[SrtCue]:
    """Parse SRT into cues, preserving indices, boundaries and text."""
    lines = raw.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    cues: list[SrtCue] = []
    idx = 0
    while idx < len(lines):
        line = lines[idx].strip()
        if not line:
            idx += 1
            continue
        number = None
        if line.isdigit():
            number = int(line)
            idx += 1
            if idx >= len(lines):
                break
        match = _SRT_TIME.search(lines[idx])
        if not match:
            idx += 1
            continue
        times = match.groups()
        idx += 1
        text_lines: list[str] = []
        while idx < len(lines) and lines[idx].strip():
            text_lines.append(lines[idx])
            idx += 1
        cues.append(
            SrtCue(
                index=number if number is not None else len(cues) + 1,
                start=_srt_seconds(*times[:4]),
                end=_srt_seconds(*times[4:]),
                text="\n".join(text_lines),
            )
        )
    return cues


def normalize_formats(value: str | Iterable[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        parts = [p.strip().lower() for p in value.replace(";", ",").split(",")]
    else:
        parts = [str(p).strip().lower() for p in value]
    out: list[str] = []
    for part in parts:
        if not part:
            continue
        if part not in SUBTITLE_FORMATS:
            raise UsageError(
                f"unsupported subtitle format: {part!r}; expected {', '.join(SUBTITLE_FORMATS)}"
            )
        if part not in out:
            out.append(part)
    return out


# --------------------------------------------------------------------------- #
# Mapping aligner items back onto the source text
# --------------------------------------------------------------------------- #
def _find_span(text: str, needle: str, cursor: int) -> int:
    idx = text.find(needle, cursor)
    if idx >= 0:
        return idx
    return text.lower().find(needle.lower(), cursor)


def _is_cjk(ch: str) -> bool:
    return "一" <= ch <= "鿿" or "㐀" <= ch <= "䶿"


def map_items(text: str, items: Sequence[AlignItem]) -> list[tuple[int, int, float, float]]:
    """Return (char_start, char_end, time_start, time_end) aligned to ``text``.

    Items that cannot be located are skipped; callers can inspect
    :func:`match_ratio` to detect a poor alignment.
    """
    spans: list[tuple[int, int, float, float]] = []
    cursor = 0
    for item in items:
        needle = item.text or ""
        if not needle:
            continue
        idx = _find_span(text, needle, cursor)
        if idx < 0:
            continue
        end = idx + len(needle)
        spans.append((idx, end, float(item.start), float(item.end)))
        cursor = end
    return spans


def _is_content(ch: str) -> bool:
    return ch.isalnum() or _is_cjk(ch)


def match_ratio(text: str, spans: Sequence[tuple[int, int, float, float]]) -> float:
    """Fraction of content characters (letters/digits/CJK) covered by spans.

    Punctuation is excluded because the aligner deliberately drops it; counting
    it would make a perfectly aligned ``你好，世界。`` look only 67% covered.
    """
    content_positions = [i for i, ch in enumerate(text) if _is_content(ch)]
    if not content_positions:
        return 1.0
    covered = 0
    for pos in content_positions:
        if any(start <= pos < end for start, end, _, _ in spans):
            covered += 1
    return min(1.0, covered / len(content_positions))


def coverage(text: str, items: Sequence[AlignItem]) -> float:
    return match_ratio(text, map_items(text, items))


def _char_times(
    text: str, spans: Sequence[tuple[int, int, float, float]]
) -> list[Optional[tuple[float, float]]]:
    """Per-character (start, end) where known; None for unmatched characters."""
    times: list[Optional[tuple[float, float]]] = [None] * len(text)
    for start, end, t_start, t_end in spans:
        length = max(1, end - start)
        step = (t_end - t_start) / length
        for offset in range(length):
            if start + offset < len(times):
                times[start + offset] = (t_start + step * offset, t_start + step * (offset + 1))
    return times


def _atoms_from_spans(
    text: str,
    spans: Sequence[tuple[int, int, float, float]],
    granularity: str,
) -> list[tuple[int, int, float, float]]:
    atoms: list[tuple[int, int, float, float]] = []
    for start, end, t_start, t_end in spans:
        piece = text[start:end]
        splittable = end - start > 1 and all(_is_cjk(ch) for ch in piece)
        if granularity == "word" or end - start <= 1 or not splittable:
            # The aligner has no per-letter timings for Latin text; fabricating
            # them would produce meaningless one-letter cues.
            atoms.append((start, end, t_start, t_end))
            continue
        length = end - start
        step = (t_end - t_start) / length if length else 0.0
        for offset in range(length):
            atoms.append(
                (start + offset, start + offset + 1, t_start + step * offset, t_start + step * (offset + 1))
            )
    return atoms


def _bump(stats: Optional[dict[str, int]], key: str) -> None:
    if stats is not None:
        stats[key] = stats.get(key, 0) + 1


def _clean_cues(
    cues: Sequence[Cue],
    audio_duration: Optional[float],
    repair_stats: Optional[dict[str, int]] = None,
) -> list[Cue]:
    """Clamp to the audio window, enforce ``end > start`` and monotonicity.

    When aligner timings saturate the end of the audio, the whole block is
    shifted back; if it still does not fit, durations are compressed
    proportionally. Zero-length cues are never emitted (codex P0-7 follow-up).
    Structural repairs are counted so ``--no-subtitle-qc`` can still report
    that it repaired overlaps/out-of-range/non-positive timings.
    """
    if not cues:
        return []
    # [start, end, index, text] so the block can be shifted/compressed in place.
    prepared: list[list[Any]] = []
    last_end = 0.0
    for cue in cues:
        raw_start, raw_end = float(cue.start), float(cue.end)
        if raw_start < -1e-9:
            _bump(repair_stats, "out_of_range")
        if raw_start < last_end - 1e-9:
            _bump(repair_stats, "overlap")
        if raw_end <= raw_start + 1e-9:
            _bump(repair_stats, "non_positive")
        start = max(0.0, raw_start, last_end)
        end = max(raw_end, start + MIN_CUE_SECONDS)
        prepared.append([start, end, cue.index, cue.text])
        last_end = end

    if audio_duration is None or audio_duration <= 0:
        return [Cue(start, end, text, index) for start, end, index, text in prepared]

    limit = float(audio_duration)
    if prepared[-1][1] > limit + 1e-9:
        _bump(repair_stats, "out_of_range")
    overflow = prepared[-1][1] - limit
    if overflow > 0:
        if prepared[0][0] >= overflow:
            for item in prepared:
                item[0] -= overflow
                item[1] -= overflow
        else:
            # Not enough headroom to shift: lay the block out from zero, then
            # scale the durations so the tail lands exactly on the audio end.
            first_start = prepared[0][0]
            for item in prepared:
                item[0] -= first_start
                item[1] -= first_start
            total = prepared[-1][1]
            if total > limit:
                if limit < len(prepared) * MIN_CUE_SECONDS:
                    raise InputError(
                        f"cannot fit {len(prepared)} cues into {limit:.3f}s of audio; "
                        "alignment produced unusable timing",
                        details={"cues": len(prepared), "audio_duration_s": round(limit, 3)},
                    )
                base = len(prepared) * MIN_CUE_SECONDS
                scale = (limit - base) / (total - base) if total > base else 0.0
                cursor = 0.0
                for item in prepared:
                    duration = (item[1] - item[0] - MIN_CUE_SECONDS) * scale + MIN_CUE_SECONDS
                    item[0] = cursor
                    item[1] = cursor + duration
                    cursor = item[1]
                total = cursor
            delta = limit - total
            for item in prepared:
                item[0] += delta
                item[1] += delta

    out: list[Cue] = []
    last_end = 0.0
    for start, end, index, text in prepared:
        start = max(0.0, float(start), last_end)
        if float(end) > limit + 1e-9:
            _bump(repair_stats, "out_of_range")
        end = min(float(end), limit)
        if end <= start:
            _bump(repair_stats, "non_positive")
            # Only floating-point fuzz should reach here; keep the cue inside
            # the window or fail rather than emit a zero-length cue.
            if limit - start >= MIN_CUE_SECONDS:
                end = start + MIN_CUE_SECONDS
            else:
                raise InputError(
                    "alignment timing cannot be placed inside the audio window",
                    details={"start": round(start, 6), "audio_duration_s": round(limit, 3)},
                )
        out.append(Cue(start, end, str(text), index))
        last_end = end
    return out


def _sentence_ranges(text: str) -> list[tuple[int, int]]:
    """Character ranges of sentence-like segments; concatenation is ``text``."""
    ranges: list[tuple[int, int]] = []
    cursor = 0
    for sentence in segment_text(text):
        end = cursor + len(sentence)
        ranges.append((cursor, end))
        cursor = end
    return ranges


def _span_duration(times: Sequence[Optional[tuple[float, float]]], span: SourceSpan) -> float:
    known = [t for t in times[span.start : span.end] if t is not None]
    if not known:
        return 0.0
    return max(t[1] for t in known) - min(t[0] for t in known)


def _cues_from_spans(
    text: str,
    spans: Sequence[SourceSpan],
    times: Sequence[Optional[tuple[float, float]]],
) -> list[Cue]:
    cues: list[Cue] = []
    previous_end = 0.0
    for span in spans:
        if span.end <= span.start:
            continue
        known = [t for t in times[span.start : span.end] if t is not None]
        if known:
            t_start = min(t[0] for t in known)
            t_end = max(t[1] for t in known)
        else:
            t_start = previous_end
            t_end = previous_end + MIN_CUE_SECONDS
        cues.append(Cue(t_start, max(t_end, t_start + MIN_CUE_SECONDS), text[span.start : span.end]))
        previous_end = cues[-1].end
    return cues


def _max_width_cut(text: str, start: int, end: int, max_width: int) -> int:
    width = 0
    for idx in range(start, end):
        width += display_width_char(text[idx])
        if width > max_width:
            return idx
    return end


def _natural_cut(
    text: str,
    start: int,
    end: int,
    max_cut: int,
    depths: Sequence[int],
) -> Optional[int]:
    """Latest secondary-punctuation/space boundary at the shallowest quote depth."""
    for depth_rank in (0, 1):
        for kind in (0, 1):
            for cut in range(max_cut, start, -1):
                if cut >= end:
                    continue
                idx = cut - 1
                if kind == 0 and text[idx] not in _SPLIT_PUNCT:
                    continue
                if kind == 1 and text[idx] not in _SPLIT_SPACE:
                    continue
                if depth_rank == 0 and depths[cut] != 0:
                    continue
                return cut
    return None


def _hard_cut(
    text: str,
    start: int,
    end: int,
    max_cut: int,
    depths: Sequence[int],
) -> Optional[int]:
    """Token/quote-safe hard cut; never inside a number, Latin word or quote."""
    for cut in range(max_cut, start, -1):
        if cut >= end:
            continue
        if depths[cut] != 0:
            continue
        if _cut_inside_token(text, cut):
            continue
        if text[cut] in _HANGING_CLOSE or text[cut - 1] in _HANGING_OPEN:
            continue
        return cut
    return None


def _split_text_by_width(
    text: str,
    start: int,
    end: int,
    policy: SubtitlePolicy,
    warnings: Optional[list[str]],
    stats: Optional[dict[str, int]],
) -> list[tuple[int, int]]:
    limit = _cue_width_limit(policy)
    if display_width(text[start:end]) <= limit:
        return [(start, end)]
    depths = _quote_depths(text)
    pieces: list[tuple[int, int]] = []
    cursor = start
    while display_width(text[cursor:end]) > limit:
        max_cut = _max_width_cut(text, cursor, end, limit)
        if max_cut <= cursor:
            max_cut = cursor + 1
        cut = _natural_cut(text, cursor, end, max_cut, depths)
        if cut is None:
            cut = _hard_cut(text, cursor, end, max_cut, depths)
            if cut is None:
                if warnings is not None:
                    warnings.append(
                        "subtitle cue exceeds --max-cue-chars but has no safe split "
                        f"point (quote/number/Latin word): {text[cursor:end]!r}"
                    )
                break
            _bump(stats, "hard_cuts")
            if warnings is not None:
                warnings.append(
                    "hard-cut subtitle cue at a token/quote-safe boundary: "
                    f"{text[cursor:cut]!r} | {text[cut:end]!r}"
                )
        pieces.append((cursor, cut))
        cursor = cut
    if cursor < end:
        pieces.append((cursor, end))
    return pieces or [(start, end)]


def plan_sentence_spans(
    text: str,
    policy: SubtitlePolicy,
    *,
    parent_ids: tuple[str, ...] = (),
    warnings: Optional[list[str]] = None,
    stats: Optional[dict[str, int]] = None,
) -> list[SourceSpan]:
    """Initial text split: sentence ranges, further split by display width.

    Timing is deliberately not available here; duration/CPS balancing happens
    after alignment in :func:`balance_spans` (t10 hard constraint).
    """
    spans: list[SourceSpan] = []
    for sentence_id, (start, end) in enumerate(_sentence_ranges(text)):
        pids = tuple(parent_ids) + (f"sent:{sentence_id}",)
        pieces = _split_text_by_width(text, start, end, policy, warnings, stats)
        if stats is not None:
            stats["sentences"] = stats.get("sentences", 0) + 1
            stats["splits"] = stats.get("splits", 0) + max(0, len(pieces) - 1)
        spans.extend(SourceSpan(a, b, pids) for a, b in pieces)
    return spans


def _duration_cut(
    text: str,
    span: SourceSpan,
    times: Sequence[Optional[tuple[float, float]]],
    policy: SubtitlePolicy,
) -> tuple[Optional[int], bool]:
    """Cut an overlong timed span at a natural boundary; hard cut as fallback."""
    depths = _quote_depths(text)
    for depth_rank in (0, 1):
        for kind in (0, 1):
            for cut in range(span.end - 1, span.start, -1):
                idx = cut - 1
                if kind == 0 and text[idx] not in _SPLIT_PUNCT:
                    continue
                if kind == 1 and text[idx] not in _SPLIT_SPACE:
                    continue
                if depth_rank == 0 and depths[cut] != 0:
                    continue
                if _span_duration(times, SourceSpan(span.start, cut)) <= policy.max_cue_seconds:
                    return cut, False
    for cut in range(span.end - 1, span.start, -1):
        if depths[cut] != 0 or _cut_inside_token(text, cut):
            continue
        if text[cut] in _HANGING_CLOSE or text[cut - 1] in _HANGING_OPEN:
            continue
        if _span_duration(times, SourceSpan(span.start, cut)) <= policy.max_cue_seconds:
            return cut, True
    return None, False


def _split_spans_by_duration(
    text: str,
    spans: Sequence[SourceSpan],
    times: Sequence[Optional[tuple[float, float]]],
    policy: SubtitlePolicy,
    warnings: Optional[list[str]],
    stats: Optional[dict[str, int]],
) -> list[SourceSpan]:
    out: list[SourceSpan] = []
    for span in spans:
        cursor = span.start
        while _span_duration(times, SourceSpan(cursor, span.end)) > policy.max_cue_seconds:
            cut, hard = _duration_cut(text, SourceSpan(cursor, span.end), times, policy)
            if cut is None or cut <= cursor:
                if warnings is not None:
                    warnings.append(
                        "subtitle cue exceeds --max-cue-seconds and cannot be split "
                        f"without breaking a token/quote: {text[cursor:span.end]!r}"
                    )
                break
            out.append(SourceSpan(cursor, cut, span.parent_ids))
            _bump(stats, "splits")
            if hard:
                _bump(stats, "hard_cuts")
                if warnings is not None:
                    warnings.append(
                        "hard-cut overlong subtitle cue by time at a token-safe boundary: "
                        f"{text[cursor:cut]!r} | {text[cut:span.end]!r}"
                    )
            cursor = cut
        out.append(SourceSpan(cursor, span.end, span.parent_ids))
    return out


def _span_bounds(
    times: Sequence[Optional[tuple[float, float]]],
    span: SourceSpan,
) -> Optional[tuple[float, float]]:
    known = [t for t in times[span.start : span.end] if t is not None]
    if not known:
        return None
    return min(t[0] for t in known), max(t[1] for t in known)


def _merge_gap(
    times: Sequence[Optional[tuple[float, float]]],
    left: SourceSpan,
    right: SourceSpan,
) -> Optional[float]:
    """Silence between two spans; None when either side has no timing."""
    left_bounds = _span_bounds(times, left)
    right_bounds = _span_bounds(times, right)
    if left_bounds is None or right_bounds is None:
        return None
    return max(0.0, right_bounds[0] - left_bounds[1])


def _merge_short_spans(
    text: str,
    spans: Sequence[SourceSpan],
    times: Sequence[Optional[tuple[float, float]]],
    policy: SubtitlePolicy,
    stats: Optional[dict[str, int]],
) -> list[SourceSpan]:
    merged = list(spans)
    index = 0
    while index < len(merged) - 1:
        left, right = merged[index], merged[index + 1]
        if left.parent_ids != right.parent_ids:
            index += 1
            continue
        left_d = _span_duration(times, left)
        right_d = _span_duration(times, right)
        if min(left_d, right_d) >= policy.min_cue_seconds:
            index += 1
            continue
        gap = _merge_gap(times, left, right)
        if gap is None or gap > policy.max_merge_gap_ms / 1000.0:
            index += 1
            continue
        candidate = SourceSpan(left.start, right.end, left.parent_ids)
        if display_width(text[candidate.start : candidate.end]) > _cue_width_limit(policy):
            index += 1
            continue
        if _span_duration(times, candidate) > policy.max_cue_seconds:
            index += 1
            continue
        merged[index] = candidate
        del merged[index + 1]
        _bump(stats, "merges")
    return merged


def _span_cps(
    text: str,
    span: SourceSpan,
    times: Sequence[Optional[tuple[float, float]]],
) -> float:
    """CPS using the rendered millisecond duration (same formula as QC)."""
    duration_ms = int(round(_span_duration(times, span) * 1000.0))
    seconds = max(0.0, duration_ms / 1000.0)
    visible = display_width(text[span.start : span.end], ignore_whitespace=True)
    if seconds <= 0:
        return float(visible) * 1_000_000_000.0 if visible else 0.0
    return visible / seconds


def _merge_for_cps(
    text: str,
    spans: Sequence[SourceSpan],
    times: Sequence[Optional[tuple[float, float]]],
    policy: SubtitlePolicy,
    stats: Optional[dict[str, int]],
) -> list[SourceSpan]:
    """Post-timing rebalance: merge a too-fast cue with a same-parent neighbour.

    Splitting cannot lower CPS (both width and duration divide out); merging a
    dense cue with an adjacent sparse cue can. Boundaries and text order are
    preserved, and the same-parent rule still prevents crossing sentences/SRT
    cues.
    """
    merged = list(spans)
    index = 0
    while index < len(merged):
        current = merged[index]
        if _span_cps(text, current, times) <= policy.max_cps:
            index += 1
            continue
        for neighbour_index in (index - 1, index + 1):
            if not 0 <= neighbour_index < len(merged):
                continue
            neighbour = merged[neighbour_index]
            if neighbour.parent_ids != current.parent_ids:
                continue
            if neighbour_index < index:
                left, right = neighbour, current
            else:
                left, right = current, neighbour
            gap = _merge_gap(times, left, right)
            if gap is None or gap > policy.max_merge_gap_ms / 1000.0:
                continue
            candidate = SourceSpan(left.start, right.end, left.parent_ids)
            if display_width(text[candidate.start : candidate.end]) > _cue_width_limit(policy):
                continue
            if _span_duration(times, candidate) > policy.max_cue_seconds:
                continue
            if _span_cps(text, candidate, times) > policy.max_cps:
                continue
            target = min(index, neighbour_index)
            merged[target] = candidate
            del merged[max(index, neighbour_index)]
            _bump(stats, "cps_merges")
            index = target + 1
            break
        else:
            index += 1
    return merged


def balance_spans(
    text: str,
    spans: Sequence[SourceSpan],
    times: Sequence[Optional[tuple[float, float]]],
    policy: SubtitlePolicy,
    *,
    warnings: Optional[list[str]] = None,
    stats: Optional[dict[str, int]] = None,
) -> list[SourceSpan]:
    """Post-alignment balancing: split long cues, merge short siblings.

    Merging is limited to spans with identical ``parent_ids`` (same SRT cue /
    TTS segment / original sentence), so it can never cross a hard boundary.
    """
    split = _split_spans_by_duration(text, spans, times, policy, warnings, stats)
    merged = _merge_short_spans(text, split, times, policy, stats)
    return _merge_for_cps(text, merged, times, policy, stats)


def build_cues(
    text: str,
    items: Sequence[AlignItem],
    *,
    granularity: str = "sentence",
    fallback_duration: Optional[float] = None,
    audio_duration: Optional[float] = None,
    policy: Optional[SubtitlePolicy] = None,
    parent_ids: tuple[str, ...] = (),
    warnings: Optional[list[str]] = None,
    stats: Optional[dict[str, int]] = None,
    repair_stats: Optional[dict[str, int]] = None,
) -> list[Cue]:
    """Build cues whose concatenated text equals ``text`` exactly."""
    if granularity not in GRANULARITIES:
        raise UsageError(f"unsupported granularity: {granularity!r}")
    if not text:
        return []
    cfg = policy or SubtitlePolicy()
    aligned = map_items(text, items)
    if not aligned:
        duration = (
            fallback_duration
            if fallback_duration and fallback_duration > 0
            else max(0.3, len(text) * 0.12)
        )
        end = min(duration, audio_duration) if audio_duration else duration
        if granularity == "sentence":
            spans = plan_sentence_spans(
                text, cfg, parent_ids=parent_ids, warnings=warnings, stats=stats
            )
            total_width = sum(display_width(text[s.start : s.end]) for s in spans) or 1
            cues = []
            cursor = 0.0
            for span in spans:
                share = end * display_width(text[span.start : span.end]) / total_width
                cues.append(Cue(cursor, cursor + share, text[span.start : span.end]))
                cursor += share
            return _clean_cues(cues, audio_duration, repair_stats)
        return [Cue(0.0, max(MIN_CUE_SECONDS, end), text)]

    if granularity == "sentence":
        spans = plan_sentence_spans(
            text, cfg, parent_ids=parent_ids, warnings=warnings, stats=stats
        )
        times = _char_times(text, aligned)
        spans = balance_spans(text, spans, times, cfg, warnings=warnings, stats=stats)
        return _clean_cues(_cues_from_spans(text, spans, times), audio_duration, repair_stats)

    atoms = _atoms_from_spans(text, aligned, granularity)
    boundaries = [0]
    for atom in atoms:
        if atom[0] > boundaries[-1]:
            boundaries.append(atom[0])
    if boundaries[-1] != len(text):
        boundaries.append(len(text))

    atom_index = 0
    cues: list[Cue] = []
    for i in range(len(boundaries) - 1):
        b_start, b_end = boundaries[i], boundaries[i + 1]
        while atom_index < len(atoms) and atoms[atom_index][1] <= b_start:
            atom_index += 1
        if atom_index < len(atoms):
            t_start = atoms[atom_index][2]
        else:
            t_start = cues[-1].end if cues else 0.0
        t_end = t_start
        j = atom_index
        while j < len(atoms) and atoms[j][0] < b_end:
            t_end = max(t_end, atoms[j][3])
            j += 1
        cues.append(Cue(t_start, max(t_end, t_start + MIN_CUE_SECONDS), text[b_start:b_end]))
    return _clean_cues(cues, audio_duration, repair_stats)


def build_cues_for_ranges(
    text: str,
    items: Sequence[AlignItem],
    ranges: Sequence[tuple[int, int]],
    *,
    audio_duration: Optional[float] = None,
    repair_stats: Optional[dict[str, int]] = None,
) -> list[Cue]:
    """Preserve caller-provided text ranges (used by `align --srt`).

    Each range gets the time of its first/last matched character; ranges with no
    matched characters inherit the previous cue's end so the timeline stays
    monotonic instead of collapsing to zero.
    """
    times = _char_times(text, map_items(text, items))
    cues: list[Cue] = []
    previous_end = 0.0
    for start_idx, end_idx in ranges:
        start_idx = max(0, min(start_idx, len(text)))
        end_idx = max(start_idx, min(end_idx, len(text)))
        known = [t for t in times[start_idx:end_idx] if t is not None]
        if known:
            t_start = min(t[0] for t in known)
            t_end = max(t[1] for t in known)
        else:
            t_start = previous_end
            t_end = previous_end + MIN_CUE_SECONDS
        cues.append(Cue(t_start, max(t_end, t_start + MIN_CUE_SECONDS), text[start_idx:end_idx]))
        previous_end = cues[-1].end
    return _clean_cues(cues, audio_duration, repair_stats)


def shift_cues(cues: Sequence[Cue], offset: float) -> list[Cue]:
    return [cue.shifted(offset) for cue in cues]


def monotonic(
    cues: Sequence[Cue],
    audio_duration: Optional[float] = None,
    repair_stats: Optional[dict[str, int]] = None,
) -> list[Cue]:
    """Clamp overlaps while preserving every cue text."""
    return _clean_cues(cues, audio_duration, repair_stats)


def full_text(cues: Sequence[Cue]) -> str:
    return "".join(cue.text for cue in cues)


def ensure_full_text(cues: Sequence[Cue], expected: str) -> None:
    """The subtitle contract is fail-closed: cue text must equal the source."""
    actual = full_text(cues)
    if actual != expected:
        raise InputError(
            "subtitle text does not match the normalized source text",
            details={
                "expected_chars": len(expected),
                "actual_chars": len(actual),
                "first_mismatch": next(
                    (
                        index
                        for index, (left, right) in enumerate(zip(expected, actual))
                        if left != right
                    ),
                    min(len(expected), len(actual)),
                ),
            },
        )


# --------------------------------------------------------------------------- #
# Renderers
# --------------------------------------------------------------------------- #
def _ts_srt(t: float) -> str:
    ms = max(0, int(round(t * 1000)))
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _ts_vtt(t: float) -> str:
    return _ts_srt(t).replace(",", ".")


def _ts_ass(t: float) -> str:
    cs = max(0, int(round(t * 100)))
    h, rem = divmod(cs, 360_000)
    m, rem = divmod(rem, 6000)
    s, cs = divmod(rem, 100)
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"


def to_srt(cues: Sequence[Cue]) -> str:
    blocks = []
    for idx, cue in enumerate(cues, 1):
        number = cue.index if cue.index is not None else idx
        blocks.append(f"{number}\n{_ts_srt(cue.start)} --> {_ts_srt(cue.end)}\n{cue.text}\n")
    return "\n".join(blocks)


def to_vtt(cues: Sequence[Cue]) -> str:
    lines = ["WEBVTT", ""]
    for cue in cues:
        lines.append(f"{_ts_vtt(cue.start)} --> {_ts_vtt(cue.end)}")
        lines.append(cue.text)
        lines.append("")
    return "\n".join(lines)


_ASS_HEADER = """[Script Info]
ScriptType: v4.00+
WrapStyle: 0
ScaledBorderAndShadow: yes
YCbCr Matrix: None

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,48,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,0,2,10,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def to_ass(cues: Sequence[Cue]) -> str:
    lines = [_ASS_HEADER.rstrip("\n")]
    for cue in cues:
        text = cue.text.replace("\n", "\\N")
        lines.append(f"Dialogue: 0,{_ts_ass(cue.start)},{_ts_ass(cue.end)},Default,,0,0,0,,{text}")
    return "\n".join(lines) + "\n"


def to_json(cues: Sequence[Cue], *, language: Optional[str] = None, granularity: str = "sentence") -> str:
    import json

    payload = {
        "schema_version": 1,
        "language": language,
        "granularity": granularity,
        "cues": [cue.to_dict() for cue in cues],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def render(cues: Sequence[Cue], fmt: str, *, language: Optional[str] = None, granularity: str = "sentence") -> str:
    if fmt == "srt":
        return to_srt(cues)
    if fmt == "vtt":
        return to_vtt(cues)
    if fmt == "ass":
        return to_ass(cues)
    if fmt == "json":
        return to_json(cues, language=language, granularity=granularity)
    raise UsageError(f"unsupported subtitle format: {fmt!r}")
