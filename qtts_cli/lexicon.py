"""Pronunciation lexicon and director markers (FR-P1-06).

Projection order is fixed: raw text -> director markers -> lexicon-protected
display projection -> spoken projection -> sentence/budget splitting.  The
display text is the subtitle truth; TTS receives the spoken text and
:func:`spoken_items_to_display` projects aligner times back onto display
characters.  Mapping is explicit character ranges (no ``str.find`` reuse).
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

from .errors import InputError, UsageError
from .mapped_text import mapped_normalize_text
from .segmenter import _quote_depths
from .subtitles import AlignItem

LEXICON_SCHEMA_VERSION = 1
_MARKER_RE = re.compile(r"\[\[([^\[\]]*)\]\]")
_PAUSE_VALUE_RE = re.compile(r"^pause\s*:\s*(\d+)$", re.IGNORECASE)
_UNSUPPORTED_DIRECTOR_RE = re.compile(
    r"^(?:speed|rate|pitch|emphasis|stress)\s*:", re.IGNORECASE
)
_SENTENCE_PUNCT = set("。！？.!?\n")


def _normalized_chars(text: str) -> list[tuple[str, int]]:
    """NFKC+casefold per original character, preserving original indices."""
    out: list[tuple[str, int]] = []
    for index, char in enumerate(text):
        for folded in unicodedata.normalize("NFKC", char).casefold():
            out.append((folded, index))
    return out


def _match_key(text: str) -> str:
    return "".join(char for char, _ in _normalized_chars(text))


@dataclass(frozen=True)
class LexiconEntry:
    term: str
    spoken: str
    key: str


@dataclass
class Lexicon:
    entries: list[LexiconEntry] = field(default_factory=list)
    sha256: Optional[str] = None

    @classmethod
    def load(cls, path: str | Path | None) -> Optional["Lexicon"]:
        if path is None:
            return None
        file = Path(path).expanduser()
        if not file.is_file():
            raise InputError(f"lexicon not found: {file}", details={"lexicon": str(file)})
        try:
            raw = json.loads(file.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise InputError(f"invalid lexicon {file}: {exc}") from exc
        if not isinstance(raw, dict):
            raise UsageError("lexicon must be a JSON object")
        unknown = set(raw) - {"schema_version", "entries"}
        if unknown:
            raise UsageError(f"unknown lexicon field(s): {', '.join(sorted(unknown))}")
        if raw.get("schema_version") != LEXICON_SCHEMA_VERSION:
            raise UsageError(
                f"unsupported lexicon schema_version: {raw.get('schema_version')!r}"
            )
        entries_raw = raw.get("entries")
        if not isinstance(entries_raw, list) or not entries_raw:
            raise UsageError("lexicon.entries must be a non-empty list")
        entries: list[LexiconEntry] = []
        seen: dict[str, str] = {}
        for index, item in enumerate(entries_raw):
            if not isinstance(item, dict):
                raise UsageError(f"lexicon.entries[{index}] must be an object")
            unknown_entry = set(item) - {"term", "spoken"}
            if unknown_entry:
                raise UsageError(
                    f"lexicon.entries[{index}] unknown field(s): "
                    f"{', '.join(sorted(unknown_entry))}"
                )
            term = item.get("term")
            spoken = item.get("spoken")
            if not isinstance(term, str) or not term:
                raise UsageError(f"lexicon.entries[{index}].term must be a non-empty string")
            if not isinstance(spoken, str) or not spoken:
                raise UsageError(f"lexicon.entries[{index}].spoken must be a non-empty string")
            key = _match_key(term)
            if not key:
                raise UsageError(f"lexicon.entries[{index}].term normalizes to empty")
            if key in seen:
                raise InputError(
                    f"duplicate lexicon term: {term!r} conflicts with {seen[key]!r}",
                    details={"term": term, "conflicts_with": seen[key]},
                )
            seen[key] = term
            entries.append(LexiconEntry(term=term, spoken=spoken, key=key))
        # Longest match first; deterministic tie-break by canonical key.
        entries.sort(key=lambda entry: (-len(entry.key), entry.key))
        canonical = {
            "schema_version": LEXICON_SCHEMA_VERSION,
            "entries": [
                {"term": entry.term, "spoken": entry.spoken}
                for entry in sorted(entries, key=lambda e: (e.key, e.term))
            ],
        }
        digest = hashlib.sha256(
            json.dumps(
                canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        return cls(entries=entries, sha256=digest)


@dataclass(frozen=True)
class MapBlock:
    display_start: int
    display_end: int
    spoken_start: int
    spoken_end: int
    term: Optional[str] = None

    def shifted(self, display_delta: int, spoken_delta: int) -> "MapBlock":
        return MapBlock(
            self.display_start + display_delta,
            self.display_end + display_delta,
            self.spoken_start + spoken_delta,
            self.spoken_end + spoken_delta,
            self.term,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "display_start": self.display_start,
            "display_end": self.display_end,
            "spoken_start": self.spoken_start,
            "spoken_end": self.spoken_end,
        }
        if self.term is not None:
            payload["term"] = self.term
        return payload


@dataclass
class PreparedText:
    display_text: str
    spoken_text: str
    mapping: list[MapBlock]
    pauses: list[dict[str, int]] = field(default_factory=list)
    # Raw-source span for every display character; lets callers map original
    # cue boundaries through the one global lexicon projection.
    source_spans: Optional[list[tuple[int, int]]] = None


def parse_director_markers(text: str) -> tuple[str, list[dict[str, int]]]:
    """Remove supported pause markers; reject unsupported/invalid markers."""
    pauses: list[dict[str, int]] = []
    pieces: list[str] = []
    cursor = 0
    clean_length = 0
    for match in _MARKER_RE.finditer(text):
        before = text[cursor : match.start()]
        pieces.append(before)
        clean_length += len(before)
        body = match.group(1).strip()
        pause_match = _PAUSE_VALUE_RE.match(body)
        if pause_match:
            pauses.append({"offset": clean_length, "ms": int(pause_match.group(1))})
        elif _UNSUPPORTED_DIRECTOR_RE.match(body):
            raise UsageError(
                f"director marker {match.group(0)!r} is not supported by the "
                "deterministic transformers backend; use [[pause:MS]] only"
            )
        else:
            raise UsageError(f"unknown director marker: {match.group(0)!r}")
        cursor = match.end()
    pieces.append(text[cursor:])
    clean = "".join(pieces)
    # There is no escape syntax: a leftover bracket pair is always a typo.
    if "[[" in clean or "]]" in clean:
        raise UsageError(
            f"malformed director marker: unmatched '[[' or ']]' in {clean!r}"
        )
    return clean, pauses


def _validate_mapping(
    display_text: str, spoken_text: str, mapping: Sequence[MapBlock], *, source: str
) -> None:
    display_cursor = 0
    spoken_cursor = 0
    for block in mapping:
        if block.display_end <= block.display_start or block.spoken_end <= block.spoken_start:
            raise InputError(
                f"internal {source} mapping error: empty block",
                details={"display": display_text, "spoken": spoken_text},
            )
        if block.display_start != display_cursor or block.spoken_start != spoken_cursor:
            raise InputError(
                f"internal {source} mapping error: non-contiguous range",
                details={"display": display_text, "spoken": spoken_text},
            )
        display_cursor = block.display_end
        spoken_cursor = block.spoken_end
    if display_cursor != len(display_text) or spoken_cursor != len(spoken_text):
        raise InputError(
            f"internal {source} mapping error: incomplete coverage",
            details={
                "display_chars": len(display_text),
                "display_covered": display_cursor,
                "spoken_chars": len(spoken_text),
                "spoken_covered": spoken_cursor,
            },
        )


def project_text(
    text: str, *, language: str, lexicon: Optional[Lexicon] = None
) -> PreparedText:
    """Project raw text into normalized display text and spoken text.

    Lexicon terms are matched against the raw text's canonical (NFKC+casefold)
    form *before* numeric/date normalization.  The matched term is preserved
    verbatim for subtitles; the replacement is used by TTS.
    """
    if lexicon is None or not lexicon.entries:
        mapped = mapped_normalize_text(text, language)
        display = mapped.text
        mapping = [MapBlock(i, i + 1, i, i + 1) for i in range(len(display))]
        _validate_mapping(display, display, mapping, source="display")
        return PreparedText(
            display_text=display,
            spoken_text=display,
            mapping=mapping,
            source_spans=[(unit.start, unit.end) for unit in mapped.chars],
        )

    normalized: list[tuple[str, int]] = []
    for index, char in enumerate(text):
        normalized.extend(
            (folded, index)
            for folded in unicodedata.normalize("NFKC", char).casefold()
        )
    normalized_text = "".join(char for char, _ in normalized)
    by_first: dict[str, list[LexiconEntry]] = {}
    for entry in lexicon.entries:
        by_first.setdefault(entry.key[0], []).append(entry)
    for bucket in by_first.values():
        bucket.sort(key=lambda entry: (-len(entry.key), entry.key))

    depths = _quote_depths(text)
    display_parts: list[str] = []
    spoken_parts: list[str] = []
    mapping: list[MapBlock] = []
    source_spans: list[tuple[int, int]] = []
    display_cursor = 0
    display_length = 0
    spoken_length = 0
    match_cursor = 0

    def append_identity(raw: str, raw_offset: int) -> None:
        nonlocal display_length, spoken_length
        mapped = mapped_normalize_text(raw, language)
        normalized_span = mapped.text
        if not normalized_span:
            return
        display_parts.append(normalized_span)
        spoken_parts.append(normalized_span)
        for offset, unit in enumerate(mapped.chars):
            mapping.append(
                MapBlock(
                    display_length + offset,
                    display_length + offset + 1,
                    spoken_length + offset,
                    spoken_length + offset + 1,
                )
            )
            source_spans.append((raw_offset + unit.start, raw_offset + unit.end))
        display_length += len(normalized_span)
        spoken_length += len(normalized_span)

    while match_cursor < len(normalized_text):
        chosen: Optional[LexiconEntry] = None
        start_index = -1
        end_index = -1
        for entry in by_first.get(normalized_text[match_cursor], []):
            last = match_cursor + len(entry.key) - 1
            if last >= len(normalized_text):
                continue
            if normalized_text[match_cursor : last + 1] != entry.key:
                continue
            # A match must start and end on original-character boundaries.
            candidate_start = normalized[match_cursor][1]
            candidate_end = normalized[last][1] + 1
            if match_cursor > 0 and normalized[match_cursor - 1][1] >= candidate_start:
                continue
            if last + 1 < len(normalized) and normalized[last + 1][1] < candidate_end:
                continue
            chosen = entry
            start_index = candidate_start
            end_index = candidate_end
            break
        if chosen is None:
            match_cursor += 1
            continue

        term_display = text[start_index:end_index]
        if "\n" in term_display:
            raise InputError(
                f"lexicon term {chosen.term!r} crosses a segment/newline boundary",
                details={"term": chosen.term, "text": term_display},
            )
        if len(term_display) > 1 and any(ch in _SENTENCE_PUNCT for ch in term_display):
            raise InputError(
                f"lexicon term {chosen.term!r} crosses a sentence boundary",
                details={"term": chosen.term, "text": term_display},
            )
        if any(depths[i] != depths[start_index] for i in range(start_index + 1, end_index + 1)):
            raise InputError(
                f"lexicon term {chosen.term!r} crosses a quote boundary",
                details={"term": chosen.term, "text": term_display},
            )
        if display_cursor < start_index:
            append_identity(text[display_cursor:start_index], display_cursor)
        display_parts.append(term_display)
        spoken_parts.append(chosen.spoken)
        mapping.append(
            MapBlock(
                display_length,
                display_length + len(term_display),
                spoken_length,
                spoken_length + len(chosen.spoken),
                chosen.term,
            )
        )
        source_spans.extend(
            (start_index + offset, start_index + offset + 1)
            for offset in range(len(term_display))
        )
        display_length += len(term_display)
        spoken_length += len(chosen.spoken)
        display_cursor = end_index
        match_cursor += len(chosen.key)
        while match_cursor < len(normalized) and normalized[match_cursor][1] < end_index:
            match_cursor += 1

    if display_cursor < len(text):
        append_identity(text[display_cursor:], display_cursor)

    display_text = "".join(display_parts)
    spoken_text = "".join(spoken_parts)
    _validate_mapping(display_text, spoken_text, mapping, source="lexicon")
    return PreparedText(
        display_text=display_text,
        spoken_text=spoken_text,
        mapping=mapping,
        source_spans=source_spans,
    )


def protected_ranges(mapping: Sequence[MapBlock]) -> list[tuple[int, int]]:
    return sorted(
        (block.display_start, block.display_end)
        for block in mapping
        if block.term is not None
    )


def slice_prepared(prepared: PreparedText, start: int, end: int) -> PreparedText:
    """Slice a projection by display range, preserving term blocks whole."""
    if start < 0 or end > len(prepared.display_text) or start >= end:
        raise InputError(
            "internal projection error: invalid slice",
            details={"start": start, "end": end, "chars": len(prepared.display_text)},
        )
    spoken_parts: list[str] = []
    mapping: list[MapBlock] = []
    spoken_cursor = 0
    for block in prepared.mapping:
        if block.display_end <= start:
            continue
        if block.display_start >= end:
            break
        local_start = max(block.display_start, start)
        local_end = min(block.display_end, end)
        if block.term is not None and (
            local_start != block.display_start or local_end != block.display_end
        ):
            raise InputError(
                f"internal projection error: split inside lexicon term {block.term!r}",
                details={"term": block.term, "start": start, "end": end},
            )
        if block.term is None:
            spoken_start = block.spoken_start + (local_start - block.display_start)
            spoken_end = block.spoken_start + (local_end - block.display_start)
        else:
            spoken_start = block.spoken_start
            spoken_end = block.spoken_end
        piece = prepared.spoken_text[spoken_start:spoken_end]
        spoken_parts.append(piece)
        mapping.append(
            MapBlock(
                local_start - start,
                local_end - start,
                spoken_cursor,
                spoken_cursor + len(piece),
                block.term,
            )
        )
        spoken_cursor += len(piece)
    display_text = prepared.display_text[start:end]
    spoken_text = "".join(spoken_parts)
    _validate_mapping(display_text, spoken_text, mapping, source="slice")
    spans = prepared.source_spans
    return PreparedText(
        display_text=display_text,
        spoken_text=spoken_text,
        mapping=mapping,
        source_spans=None if spans is None else list(spans[start:end]),
    )


def _is_content_char(char: str) -> bool:
    return char.isalnum() or "一" <= char <= "鿿" or "㐀" <= char <= "䶿"


def _match_item_at(spoken_text: str, start: int, needle: str) -> int:
    """Return the index where ``needle`` starts at/after ``start``, or -1."""
    position = start
    matched_start = -1
    for char in needle:
        if not char.isspace():
            while position < len(spoken_text) and spoken_text[position].isspace():
                position += 1
        if position >= len(spoken_text) or spoken_text[position] != char:
            return -1
        if matched_start < 0:
            matched_start = position
        position += 1
    return matched_start if matched_start >= 0 else start


def _locate_item(spoken_text: str, cursor: int, needle: str) -> int:
    """Locate the next aligner item sequentially, without ``str.find``.

    The aligner drops punctuation, so the search may skip punctuation and
    whitespace between items. Skipping a content character means the item
    stream is unreliable; the item is left unmapped so coverage/QC exposes it.
    """
    if not needle:
        return cursor
    position = cursor
    while position < len(spoken_text):
        matched = _match_item_at(spoken_text, position, needle)
        if matched >= 0:
            return matched
        if _is_content_char(spoken_text[position]):
            return -1
        position += 1
    return -1


def display_boundary_for_spoken(
    mapping: Sequence[MapBlock],
    spoken_index: int,
    *,
    edge: str,
) -> int:
    """Map a spoken boundary to display, snapping through lexicon blocks.

    ``edge="end"`` snaps a window end forward to the end of a lexicon term;
    ``edge="start"`` snaps a start backward to its beginning.  Split points can
    therefore never produce half a term.
    """
    if edge not in ("start", "end"):
        raise UsageError(f"invalid boundary edge: {edge!r}")
    if not mapping:
        if spoken_index == 0:
            return 0
        raise InputError(
            "internal projection error: spoken boundary outside mapping",
            details={"spoken_index": spoken_index},
        )
    total = mapping[-1].spoken_end
    if spoken_index <= 0:
        return 0
    if spoken_index >= total:
        return mapping[-1].display_end
    for block in mapping:
        if spoken_index == block.spoken_end:
            return block.display_end
        if spoken_index == block.spoken_start:
            return block.display_start
        if block.spoken_start < spoken_index < block.spoken_end:
            if block.term is not None or (
                block.display_end - block.display_start
                != block.spoken_end - block.spoken_start
            ):
                return (
                    block.display_start if edge == "start" else block.display_end
                )
            return block.display_start + (spoken_index - block.spoken_start)
    raise InputError(
        "internal projection error: spoken boundary not covered",
        details={"spoken_index": spoken_index},
    )


def spoken_items_to_display_positions(
    display_text: str,
    spoken_text: str,
    mapping: Sequence[MapBlock],
    items: Sequence[AlignItem],
) -> list[tuple[int, AlignItem]]:
    """Map items to ``(display_char_index, AlignItem)`` pairs, preserving slots."""
    spoken_times: list[Optional[tuple[float, float]]] = [None] * len(spoken_text)
    cursor = 0
    for item in items:
        needle = item.text or ""
        if not needle:
            continue
        index = _locate_item(spoken_text, cursor, needle)
        if index < 0:
            continue
        step = (float(item.end) - float(item.start)) / max(1, len(needle))
        for offset in range(len(needle)):
            position = index + offset
            if position < len(spoken_times):
                spoken_times[position] = (
                    float(item.start) + step * offset,
                    float(item.start) + step * (offset + 1),
                )
        cursor = index + len(needle)

    positioned: list[tuple[int, AlignItem]] = []
    for block in mapping:
        known = [
            value
            for value in spoken_times[block.spoken_start : block.spoken_end]
            if value is not None
        ]
        if not known:
            continue
        start = min(value[0] for value in known)
        end = max(value[1] for value in known)
        length = block.display_end - block.display_start
        if length <= 0:
            continue
        step = (end - start) / length
        for offset in range(length):
            char_index = block.display_start + offset
            if char_index >= len(display_text):
                break
            positioned.append(
                (
                    char_index,
                    AlignItem(
                        display_text[char_index],
                        start + step * offset,
                        start + step * (offset + 1),
                    ),
                )
            )
    return positioned


def spoken_items_to_display(
    display_text: str,
    spoken_text: str,
    mapping: Sequence[MapBlock],
    items: Sequence[AlignItem],
) -> list[AlignItem]:
    """Convert aligner items on spoken text into per-display-character items."""
    return [
        item
        for _char_index, item in spoken_items_to_display_positions(
            display_text, spoken_text, mapping, items
        )
    ]
