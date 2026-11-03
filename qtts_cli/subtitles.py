"""Subtitle cue construction and SRT/VTT/ASS/JSON rendering (FR-P0-05).

Content invariant: cues are carved out of the *normalized source text* by index,
so ``"".join(cue.text for cue in cues) == text``. The aligner contributes timing
only; it never supplies subtitle text (PRD §4.1).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence

from .errors import InputError, UsageError
from .segmenter import segment_text

SUBTITLE_FORMATS = ("srt", "vtt", "ass", "json")
GRANULARITIES = ("sentence", "char", "word")
MIN_CUE_SECONDS = 0.001


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


def _clean_cues(cues: Sequence[Cue], audio_duration: Optional[float]) -> list[Cue]:
    """Clamp to the audio window, enforce ``end > start`` and monotonicity.

    When aligner timings saturate the end of the audio, the whole block is
    shifted back; if it still does not fit, durations are compressed
    proportionally. Zero-length cues are never emitted (codex P0-7 follow-up).
    """
    if not cues:
        return []
    # [start, end, index, text] so the block can be shifted/compressed in place.
    prepared: list[list[Any]] = []
    last_end = 0.0
    for cue in cues:
        start = max(0.0, float(cue.start), last_end)
        end = max(float(cue.end), start + MIN_CUE_SECONDS)
        prepared.append([start, end, cue.index, cue.text])
        last_end = end

    if audio_duration is None or audio_duration <= 0:
        return [Cue(start, end, text, index) for start, end, index, text in prepared]

    limit = float(audio_duration)
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
        end = min(float(end), limit)
        if end <= start:
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


def build_cues(
    text: str,
    items: Sequence[AlignItem],
    *,
    granularity: str = "sentence",
    fallback_duration: Optional[float] = None,
    audio_duration: Optional[float] = None,
) -> list[Cue]:
    """Build cues whose concatenated text equals ``text`` exactly."""
    if granularity not in GRANULARITIES:
        raise UsageError(f"unsupported granularity: {granularity!r}")
    if not text:
        return []
    spans = map_items(text, items)
    if not spans:
        duration = (
            fallback_duration
            if fallback_duration and fallback_duration > 0
            else max(0.3, len(text) * 0.12)
        )
        end = min(duration, audio_duration) if audio_duration else duration
        return [Cue(0.0, max(MIN_CUE_SECONDS, end), text)]

    if granularity == "sentence":
        # One cue per sentence-like segment, timed from its first/last matched
        # character. P1 adds secondary splitting and CPS/line-length gates.
        return build_cues_for_ranges(
            text, items, _sentence_ranges(text), audio_duration=audio_duration
        )

    atoms = _atoms_from_spans(text, spans, granularity)
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
    return _clean_cues(cues, audio_duration)


def build_cues_for_ranges(
    text: str,
    items: Sequence[AlignItem],
    ranges: Sequence[tuple[int, int]],
    *,
    audio_duration: Optional[float] = None,
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
    return _clean_cues(cues, audio_duration)


def shift_cues(cues: Sequence[Cue], offset: float) -> list[Cue]:
    return [cue.shifted(offset) for cue in cues]


def monotonic(cues: Sequence[Cue], audio_duration: Optional[float] = None) -> list[Cue]:
    """Clamp overlaps while preserving every cue text."""
    return _clean_cues(cues, audio_duration)


def full_text(cues: Sequence[Cue]) -> str:
    return "".join(cue.text for cue in cues)


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
