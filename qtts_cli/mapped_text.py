"""Normalization that keeps a source range for every output character.

Long-audio alignment matches ASR transcripts against the reference in
normalized space, but every resulting boundary must be reported against the
original display text.  ``qc.normalize_for_qc`` is deliberately lossy (dates,
numbers, URLs and emoji are rewritten or deleted), so its output offsets cannot
be reused as source offsets.  :class:`MappedText` runs the exact same rules
while carrying ``[start, end)`` provenance from the original string.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Optional, Sequence

from . import textnorm as tn
from .qc import qc_language


@dataclass(frozen=True)
class MappedChar:
    char: str
    start: int
    end: int


@dataclass(frozen=True)
class MappedText:
    """Normalized text plus a source range for every normalized character."""

    source: str
    text: str
    chars: tuple[MappedChar, ...]
    language: str

    def __len__(self) -> int:
        return len(self.chars)

    def boundary(self, index: int) -> int:
        """Map a normalized start boundary to an original source offset."""
        if index <= 0:
            return 0
        if index >= len(self.chars):
            return len(self.source)
        return self.chars[index].start

    def end_boundary(self, index: int) -> int:
        """Map a normalized exclusive end boundary to a source offset."""
        if index <= 0:
            return 0
        if index > len(self.chars):
            return len(self.source)
        return self.chars[index - 1].end

    def source_span(self, start: int, end: int) -> tuple[int, int]:
        """Map normalized ``[start, end)`` to an original source span."""
        if start < 0 or end < start or end > len(self.chars):
            raise ValueError("normalized span out of range")
        if start == end:
            point = self.boundary(start)
            return point, point
        return self.boundary(start), self.end_boundary(end)


def _replace(
    units: Sequence[MappedChar],
    pattern: re.Pattern[str],
    replacement,
) -> list[MappedChar]:
    """Regex substitution with provenance.

    Unchanged matches keep their per-character spans; a rewritten or deleted
    match produces replacement characters whose span is the union of the
    matched source range.
    """
    current = "".join(unit.char for unit in units)
    output: list[MappedChar] = []
    cursor = 0
    for match in pattern.finditer(current):
        output.extend(units[cursor:match.start()])
        matched = units[match.start():match.end()]
        if not matched:
            cursor = match.end()
            continue
        text = replacement(match)
        if text == match.group(0):
            output.extend(matched)
        else:
            start = min(unit.start for unit in matched)
            end = max(unit.end for unit in matched)
            for char in text:
                output.append(MappedChar(char, start, end))
        cursor = match.end()
    output.extend(units[cursor:])
    return output


def _date_replacement(match: re.Match[str], language: str) -> str:
    year, month, day = int(match.group(1)), int(match.group(2)), int(match.group(3))
    if not (1 <= month <= 12 and 1 <= day <= 31):
        return match.group(0)
    if language == "Chinese":
        year_txt = tn._digits_to_words(match.group(1), "Chinese")
        return (
            f"{year_txt}年{tn.int_to_chinese(month)}月"
            f"{tn.int_to_chinese(day)}日"
        )
    return (
        f"{tn._MONTHS_EN[month - 1]} {tn.int_to_english(day)}, "
        f"{tn.int_to_english(year)}"
    )


def _time_replacement(match: re.Match[str], language: str) -> str:
    hour, minute = int(match.group(1)), int(match.group(2))
    if hour > 23 or minute > 59:
        return match.group(0)
    if language == "Chinese":
        hour_txt = tn.int_to_chinese(hour) + "点"
        return hour_txt if minute == 0 else f"{hour_txt}{tn.int_to_chinese(minute)}分"
    return f"{tn.int_to_english(hour)} {tn.int_to_english(minute)}"


def _normalize_language_line(units: Sequence[MappedChar], language: str) -> list[MappedChar]:
    if language not in ("Chinese", "English"):
        return list(units)
    chinese = language == "Chinese"

    def percent(match: re.Match[str]) -> str:
        number = tn._number_to_words(match.group(1), language)
        return f"百分之{number}" if chinese else f"{number} percent"

    def currency(match: re.Match[str]) -> str:
        number = tn._number_to_words(match.group(2), language)
        if chinese:
            return f"{number}{tn._CURRENCY_CN.get(match.group(1), '')}"
        return f"{number} {tn._CURRENCY_EN.get(match.group(1), '')}"

    def unit(match: re.Match[str]) -> str:
        number = tn._number_to_words(match.group(1), language)
        cn_unit, en_unit = tn._UNITS.get(match.group(2), (match.group(2), match.group(2)))
        return f"{number}{cn_unit}" if chinese else f"{number} {en_unit}"

    result = _replace(units, tn._URL, lambda _match: " ")
    result = _replace(result, tn._EMAIL, lambda _match: " ")
    result = _replace(result, tn._EMOJI, lambda _match: "")
    result = _replace(
        result, tn._DATE_YMD, lambda match: _date_replacement(match, language)
    )
    if chinese:
        result = _replace(
            result,
            tn._YEAR_CN,
            lambda match: tn._digits_to_words(match.group(1), "Chinese"),
        )
    result = _replace(
        result, tn._TIME, lambda match: _time_replacement(match, language)
    )
    result = _replace(result, tn._PERCENT, percent)
    result = _replace(result, tn._CURRENCY, currency)
    result = _replace(result, tn._UNIT, unit)
    result = _replace(
        result,
        tn._NUMBER,
        lambda match: tn._number_to_words(match.group(0), language),
    )
    return result


def _strip_units(units: Sequence[MappedChar]) -> list[MappedChar]:
    start = 0
    end = len(units)
    while start < end and units[start].char.isspace():
        start += 1
    while end > start and units[end - 1].char.isspace():
        end -= 1
    return list(units[start:end])


def _collapse_line_spaces(units: Sequence[MappedChar]) -> list[MappedChar]:
    collapsed = _replace(units, re.compile(r"[ \t ]+"), lambda _match: " ")
    return _strip_units(collapsed)


def _split_lines(
    units: Sequence[MappedChar],
) -> tuple[list[list[MappedChar]], list[MappedChar]]:
    parts: list[list[MappedChar]] = [[]]
    newlines: list[MappedChar] = []
    for unit in units:
        if unit.char == "\n":
            newlines.append(unit)
            parts.append([])
        else:
            parts[-1].append(unit)
    return parts, newlines


def _nfkc_units(source: str) -> list[MappedChar]:
    units: list[MappedChar] = []
    index = 0
    while index < len(source):
        if source.startswith("\r\n", index):
            units.append(MappedChar("\n", index, index + 2))
            index += 2
            continue
        char = source[index]
        if char == "\r":
            units.append(MappedChar("\n", index, index + 1))
            index += 1
            continue
        normalized = (
            char
            if char in tn._CJK_PUNCT
            else unicodedata.normalize("NFKC", char)
        )
        for output in normalized:
            units.append(MappedChar(output, index, index + 1))
        index += 1
    return [unit for unit in units if unit.char not in ("​", "﻿")]


def mapped_normalize_text(text: str, language: str = "auto") -> MappedText:
    """Return ``textnorm.normalize_text(text, language)`` plus provenance.

    This mirrors the display normalization used by ``lexicon.project_text``
    (no punctuation stripping, whitespace collapsing or casefolding) so a
    single global projection can still expose where every normalized character
    came from.
    """
    if not text:
        resolved = qc_language(language, "")
        return MappedText("", "", (), resolved)
    units = _nfkc_units(text)
    parts, newlines = _split_lines(units)
    normalized_parts: list[list[MappedChar]] = []
    for part in parts:
        line = _collapse_line_spaces(part)
        if not line:
            normalized_parts.append([])
            continue
        line_text = "".join(unit.char for unit in line)
        resolved = (
            tn.detect_language(line_text) if language.lower() == "auto" else language
        )
        if resolved not in ("Chinese", "English"):
            normalized_parts.append(list(line))
        else:
            normalized_parts.append(_strip_units(_normalize_language_line(line, resolved)))
    joined: list[MappedChar] = []
    for index, part in enumerate(normalized_parts):
        joined.extend(part)
        if index < len(newlines):
            joined.append(newlines[index])
    joined = _replace(joined, re.compile(r"\n{3,}"), lambda _match: "\n\n")
    joined = _strip_units(joined)
    normalized_text = "".join(unit.char for unit in joined)
    return MappedText(text, normalized_text, tuple(joined), language)


def mapped_normalize(text: str, language: str = "auto") -> MappedText:
    """Return ``normalize_for_qc(text, language)`` plus source provenance."""
    if not text:
        resolved = qc_language(language, "")
        return MappedText("", "", (), resolved)
    resolved = qc_language(language, text)
    units = _nfkc_units(text)

    parts, newlines = _split_lines(units)
    normalized_parts: list[list[MappedChar]] = []
    for part in parts:
        line = _collapse_line_spaces(part)
        if not line:
            normalized_parts.append([])
            continue
        normalized = _normalize_language_line(line, resolved)
        normalized_parts.append(_strip_units(normalized))

    joined: list[MappedChar] = []
    for index, part in enumerate(normalized_parts):
        joined.extend(part)
        if index < len(newlines):
            joined.append(newlines[index])
    joined = _replace(joined, re.compile(r"\n{3,}"), lambda _match: "\n\n")
    joined = _strip_units(joined)

    visible = [
        unit
        for unit in joined
        if unicodedata.category(unit.char)[0] not in ("P", "S")
    ]
    collapsed = _replace(visible, re.compile(r"\s+"), lambda _match: " ")
    collapsed = _strip_units(collapsed)

    folded: list[MappedChar] = []
    for unit in collapsed:
        for char in unit.char.casefold():
            folded.append(MappedChar(char, unit.start, unit.end))

    normalized_text = "".join(unit.char for unit in folded)
    return MappedText(text, normalized_text, tuple(folded), resolved)


def mapped_span_to_source(
    mapped: MappedText, start: int, end: int
) -> tuple[int, int]:
    """Convenience wrapper matching the public mapping terminology."""
    return mapped.source_span(start, end)


def source_boundary(mapped: MappedText, index: int) -> Optional[int]:
    """Return the original offset for a normalized boundary (None if invalid)."""
    if index < 0 or index > len(mapped.chars):
        return None
    return mapped.boundary(index)
