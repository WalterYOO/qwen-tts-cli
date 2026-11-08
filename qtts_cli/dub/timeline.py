"""Strict SRT / CMX3600-subset timeline parsing (FR-P2-04 §2).

The parser is pure: it never imports a model runtime and performs no I/O
beyond reading the timeline file.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..errors import InputError, UsageError
from ..utils import sha256_file

MAX_TIMELINE_BYTES = 8_000_000
MAX_CUES = 20_000
MAX_CUE_CHARS = 200_000
MAX_TOTAL_CHARS = 2_000_000

_SPEAKER = re.compile(r"^(?P<speaker>[A-Za-z0-9_-]{1,64}):\s*(?P<text>.*)$")
_SRT_TIME = re.compile(
    r"^(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*"
    r"(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*$"
)
_EDL_TIME = re.compile(r"^(\d{2}):(\d{2}):(\d{2}):(\d{2})$")
_POSITIVE_INT = re.compile(r"^[0-9]+$")


@dataclass
class Cue:
    index: int
    start_s: float
    end_s: float
    text: str
    speaker: Optional[str] = None
    instruct: Optional[str] = None


@dataclass
class ParsedTimeline:
    path: Path
    format: str
    sha256: str
    cues: list[Cue]

    @property
    def text_chars(self) -> int:
        return sum(len(cue.text) for cue in self.cues)


def _error(message: str, path: Optional[Path] = None) -> InputError:
    return InputError(message, details={"timeline": str(path)} if path is not None else None)


def _check_limits(cues: list[Cue], path: Path) -> None:
    if not cues:
        raise _error(f"timeline has no cues: {path}", path)
    if len(cues) > MAX_CUES:
        raise _error(f"timeline has too many cues ({len(cues)}; limit {MAX_CUES})", path)
    total = 0
    for cue in cues:
        count = len(cue.text)
        if count > MAX_CUE_CHARS:
            raise _error(
                f"cue {cue.index} text is too long ({count} characters; limit {MAX_CUE_CHARS})",
                path,
            )
        total += count
        if total > MAX_TOTAL_CHARS:
            raise _error(
                f"timeline text is too long ({total} characters; limit {MAX_TOTAL_CHARS})",
                path,
            )


def _split_lines(text: str) -> list[str]:
    if text.startswith("﻿"):
        text = text[1:]
    return text.replace("\r\n", "\n").replace("\r", "\n").split("\n")


def _srt_timestamp(groups: tuple[str, ...], path: Path) -> float:
    hours, minutes, seconds, millis = (int(value) for value in groups)
    if minutes > 59 or seconds > 59:
        raise _error(f"invalid SRT timecode in {path}", path)
    return round(hours * 3600 + minutes * 60 + seconds + millis / 1000.0, 3)


def _srt_text(lines: list[str], index: int, path: Path) -> tuple[str, Optional[str]]:
    pieces = [line.strip() for line in lines]
    pieces = [piece for piece in pieces if piece]
    if not pieces:
        raise _error(f"cue {index} has empty text", path)
    match = _SPEAKER.match(pieces[0])
    if match:
        speaker = match.group("speaker")
        remainder = match.group("text").strip()
        rest = [remainder] if remainder else []
        rest.extend(pieces[1:])
        if not rest:
            raise _error(f"cue {index} has empty text after the speaker prefix", path)
        return " ".join(rest), speaker
    return " ".join(pieces), None


def _srt_blocks(lines: list[str]) -> list[list[str]]:
    """Split on blank or whitespace-only separator lines (§2.1)."""
    blocks: list[list[str]] = []
    block: list[str] = []
    for line in lines:
        if line.strip() == "":
            if block:
                blocks.append(block)
                block = []
        else:
            block.append(line)
    if block:
        blocks.append(block)
    return blocks


def parse_srt(text: str, path: Path) -> list[Cue]:
    cues: list[Cue] = []
    previous_index = 0
    previous_end = -1.0
    for block in _srt_blocks(_split_lines(text)):
        first = block[0].strip()
        if not _POSITIVE_INT.match(first) or int(first) <= 0:
            raise _error(f"invalid SRT cue index {first!r} in {path}", path)
        index = int(first)
        if index <= previous_index:
            raise _error(
                f"SRT cue indexes must be unique and strictly increasing ({index})", path
            )
        if len(block) < 2:
            raise _error(f"cue {index} is missing its timing line", path)
        timing = block[1].strip()
        match = _SRT_TIME.match(timing)
        if match is None:
            raise _error(f"cue {index} has a malformed timing line: {timing!r}", path)
        body = block[2:]
        if not body:
            raise _error(f"cue {index} has empty text", path)
        for raw in body:
            stripped = raw.strip()
            if _SRT_TIME.match(stripped):
                raise _error(
                    f"cue {index} is missing its blank separator before another timing line",
                    path,
                )
            if _POSITIVE_INT.match(stripped) and int(stripped) > 0:
                raise _error(
                    f"cue {index} has an index-like body line {stripped!r}; "
                    "separate cues with a blank line",
                    path,
                )
        cue_text, speaker = _srt_text(body, index, path)
        start = _srt_timestamp(match.groups()[:4], path)
        end = _srt_timestamp(match.groups()[4:], path)
        if end <= start:
            raise _error(f"cue {index} must end after it starts", path)
        if start < previous_end:
            raise _error(f"cue {index} overlaps the previous cue", path)
        cues.append(Cue(index, start, end, cue_text, speaker))
        previous_index = index
        previous_end = end
    _check_limits(cues, path)
    return cues


def _edl_frames(value: str, fps: int, path: Path, label: str) -> int:
    match = _EDL_TIME.match(value)
    if match is None:
        raise _error(f"invalid EDL {label} timecode {value!r} in {path}", path)
    hours, minutes, seconds, frames = (int(part) for part in match.groups())
    if minutes > 59 or seconds > 59 or frames >= fps:
        raise _error(f"invalid EDL {label} timecode {value!r} for {fps} fps", path)
    return ((hours * 60 + minutes) * 60 + seconds) * fps + frames


def _edl_speaker(value: str, path: Path) -> str:
    name = value.strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name):
        raise _error(f"invalid EDL speaker name {name!r} in {path}", path)
    return name


def parse_edl(text: str, path: Path, *, fps: int) -> list[Cue]:
    if not isinstance(fps, int) or isinstance(fps, bool) or not 1 <= fps <= 120:
        raise _error(f"EDL frame rate must be an integer in 1..120, got {fps!r}", path)
    lines = _split_lines(text)
    cues: list[Cue] = []
    seen_title = False
    seen_fcm = False
    started = False
    previous_index = 0
    previous_start_frames = -1
    previous_end_frames = -1
    current: Optional[dict] = None

    def finish() -> None:
        nonlocal current
        if current is None:
            return
        pieces = [piece for piece in current["text"] if piece.strip()]
        if not pieces:
            raise _error(f"EDL event {current['index']} has no text", path)
        text_value = " ".join(piece.strip() for piece in pieces)
        cues.append(
            Cue(
                index=current["index"],
                start_s=current["recin"] / float(fps),
                end_s=current["recout"] / float(fps),
                text=text_value,
                speaker=current["speaker"],
            )
        )
        current = None

    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        upper = line.upper()
        if upper.startswith("TITLE:"):
            if started or seen_title:
                raise _error(f"duplicate or late EDL TITLE in {path}", path)
            if not line[6:].strip():
                raise _error(f"EDL TITLE must not be empty in {path}", path)
            seen_title = True
            continue
        if upper.startswith("FCM:"):
            if started or seen_fcm:
                raise _error(f"duplicate or late EDL FCM in {path}", path)
            value = line[4:].strip().upper()
            if value == "DROP FRAME":
                raise _error("EDL drop-frame timecode is not supported", path)
            if value != "NON-DROP FRAME":
                raise _error(f"unsupported EDL FCM value {line[4:].strip()!r}", path)
            seen_fcm = True
            continue
        if line.startswith("*"):
            body = line[1:].strip()
            if not started:
                continue
            if current is None:
                # Comment blocks only follow an event; a comment after an
                # event has been consumed still belongs to that event in CMX
                # terms, but without an open event there is nothing to attach.
                continue
            if body == "SPEAKER" or body.startswith("SPEAKER "):
                current["speaker"] = _edl_speaker(body[len("SPEAKER") :], path)
            elif body == "TEXT" or body.startswith("TEXT "):
                current["text"].append(body[len("TEXT") :].strip())
            continue
        # Event line.
        fields = line.split()
        if len(fields) != 8:
            raise _error(f"unrecognised EDL line {raw!r}", path)
        number = fields[0]
        if not _POSITIVE_INT.match(number) or int(number) <= 0:
            raise _error(f"invalid EDL event number {number!r}", path)
        index = int(number)
        if index <= previous_index:
            raise _error(
                f"EDL event numbers must be unique and strictly increasing ({index})", path
            )
        track = fields[2]
        if "V" not in track:
            raise _error(f"EDL event {index} is not a video track ({track!r})", path)
        if fields[3] != "C":
            raise _error(f"EDL event {index} type must be 'C', got {fields[3]!r}", path)
        srcin = _edl_frames(fields[4], fps, path, "SRCIN")
        srcout = _edl_frames(fields[5], fps, path, "SRCOUT")
        recin = _edl_frames(fields[6], fps, path, "RECIN")
        recout = _edl_frames(fields[7], fps, path, "RECOUT")
        if srcout - srcin != recout - recin:
            raise _error(
                f"EDL event {index} has a source/record duration mismatch", path
            )
        if recout <= recin:
            raise _error(f"EDL event {index} is zero-length or reversed", path)
        if recin < previous_end_frames or recin <= previous_start_frames:
            raise _error(f"EDL event {index} overlaps the previous event", path)
        finish()
        started = True
        previous_index = index
        previous_start_frames = recin
        previous_end_frames = recout
        current = {
            "index": index,
            "recin": recin,
            "recout": recout,
            "speaker": None,
            "text": [],
        }
    finish()
    _check_limits(cues, path)
    return cues


def load_timeline(path_value: str | Path, *, fps: int) -> ParsedTimeline:
    """Read and parse a timeline; the suffix selects the format."""
    path = Path(path_value).expanduser().resolve(strict=False)
    suffix = path.suffix.lower()
    if suffix not in (".srt", ".edl"):
        raise UsageError(
            f"timeline must have a .srt or .edl suffix: {path}",
            details={"timeline": str(path)},
        )
    if not path.is_file():
        raise _error(f"timeline file not found: {path}", path)
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise _error(f"cannot stat timeline {path}: {exc}", path) from exc
    if size > MAX_TIMELINE_BYTES:
        raise _error(
            f"timeline is too large ({size} bytes; limit {MAX_TIMELINE_BYTES})", path
        )
    try:
        raw = path.read_bytes()
        text = raw.decode("utf-8-sig")
    except (OSError, UnicodeDecodeError) as exc:
        raise _error(f"cannot read timeline {path} as UTF-8: {exc}", path) from exc
    if "\x00" in text:
        raise _error(f"timeline {path} contains NUL bytes", path)
    fmt = "srt" if suffix == ".srt" else "edl"
    cues = parse_srt(text, path) if fmt == "srt" else parse_edl(text, path, fps=fps)
    return ParsedTimeline(path=path, format=fmt, sha256=sha256_file(path), cues=cues)
