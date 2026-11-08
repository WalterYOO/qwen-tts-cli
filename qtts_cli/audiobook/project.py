"""Strict audiobook project loader (FR-P2-03 §2)."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from ..capabilities import parse_voice_ref
from ..errors import InputError, UsageError
from ..script import DuplicateJSONKey, _yaml_load_strict
from ..utils import sha256_file, sha256_text

MAX_PROJECT_BYTES = 1_000_000
MAX_SOURCE_BYTES = 8_000_000
MAX_BGM_BYTES = 512_000_000
MAX_CHAPTERS = 500
MAX_PARAGRAPHS_PER_CHAPTER = 5000
MAX_PARAGRAPHS = 20_000
MAX_PARAGRAPH_CHARS = 200_000
MAX_CHAPTER_TEXT_CHARS = 2_000_000
MAX_DEPTH = 100
MAX_NODES = 100_000

_TOP_KEYS = {
    "schema_version",
    "title",
    "author",
    "album",
    "language",
    "voice",
    "instruct",
    "sample_rate",
    "pause",
    "loudness",
    "bgm",
    "chapters",
}
_CHAPTER_KEYS = {"title", "source", "text", "voice", "language", "instruct"}
_PAUSE_KEYS = {"paragraph_ms", "chapter_ms"}
_LOUDNESS_KEYS = {"lufs", "true_peak"}
_BGM_KEYS = {"file", "volume_db", "ducking_db", "fade_ms"}
_BLANK_LINE = re.compile(r"\n\s*\n")


@dataclass
class Chapter:
    index: int
    title: str
    paragraphs: list[str]
    voice: str
    language: str
    instruct: Optional[str]
    source_path: Optional[Path]
    source_sha256: str
    inline: bool
    text: str

    @property
    def paragraph_count(self) -> int:
        return len(self.paragraphs)


@dataclass
class BgmSpec:
    path: Path
    volume_db: float
    ducking_db: float
    fade_ms: int


@dataclass
class Project:
    path: Path
    sha256: str
    title: str
    author: Optional[str]
    album: str
    language: str
    voice: str
    instruct: Optional[str]
    sample_rate: Optional[int]
    paragraph_ms: int
    chapter_ms: int
    lufs: Optional[float]
    true_peak: Optional[float]
    bgm: Optional[BgmSpec]
    chapters: list[Chapter]

    @property
    def paragraph_count(self) -> int:
        return sum(chapter.paragraph_count for chapter in self.chapters)


def _invalid(message: str, path: Path | None = None) -> InputError:
    details = {"project": str(path)} if path is not None else None
    return InputError(message, details=details)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _number(
    value: Any,
    name: str,
    *,
    path: Path | None = None,
    minimum: float | None = None,
    maximum: float | None = None,
    integer: bool = False,
) -> float | int:
    if not _is_number(value):
        raise _invalid(f"{name} must be a number", path)
    if not math.isfinite(float(value)):
        raise _invalid(f"{name} must be finite", path)
    if integer and float(value) != int(value):
        raise _invalid(f"{name} must be an integer", path)
    if minimum is not None and float(value) < minimum:
        raise _invalid(f"{name} must be >= {minimum}", path)
    if maximum is not None and float(value) > maximum:
        raise _invalid(f"{name} must be <= {maximum}", path)
    return int(value) if integer else float(value)


def _string(
    value: Any,
    name: str,
    *,
    path: Path | None = None,
    allow_empty: bool = False,
) -> str:
    if not isinstance(value, str):
        raise _invalid(f"{name} must be a string", path)
    text = value.strip()
    if not text and not allow_empty:
        raise _invalid(f"{name} must not be empty", path)
    return text


def _mapping(value: Any, name: str, *, path: Path | None = None) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise _invalid(f"{name} must be a mapping", path)
    return value


def _unknown(mapping: dict[str, Any], allowed: set[str], name: str, path: Path) -> None:
    unknown = set(mapping) - allowed
    if unknown:
        raise _invalid(f"unknown {name} field: {sorted(unknown)[0]}", path)


def _check_tree(value: Any, path: Path) -> None:
    """Iterative depth/node checker; avoids Python recursion limits."""
    stack: list[tuple[Any, int]] = [(value, 1)]
    nodes = 0
    while stack:
        node, depth = stack.pop()
        nodes += 1
        if nodes > MAX_NODES:
            raise _invalid("project has too many nodes", path)
        if depth > MAX_DEPTH:
            raise _invalid("project nesting is too deep", path)
        if isinstance(node, dict):
            for key, item in node.items():
                if not isinstance(key, str):
                    raise _invalid("project object keys must be strings", path)
                stack.append((item, depth + 1))
        elif isinstance(node, list):
            for item in node:
                stack.append((item, depth + 1))


def _duplicate_json_key(pairs):
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise DuplicateJSONKey(f"duplicate key {key!r}")
        out[key] = value
    return out


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _load_project_data(path: Path, raw: str) -> Any:
    suffix = path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        try:
            return _yaml_load_strict(raw, path)
        except RecursionError as exc:
            raise _invalid("project YAML nesting is too deep", path) from exc
        except UsageError as exc:
            raise _invalid(exc.message, path) from exc
    if suffix == ".json":
        try:
            return json.loads(
                raw,
                object_pairs_hook=_duplicate_json_key,
                parse_constant=_reject_constant,
            )
        except RecursionError as exc:
            raise _invalid("project JSON nesting is too deep", path) from exc
        except (DuplicateJSONKey, json.JSONDecodeError, ValueError) as exc:
            raise _invalid(f"invalid JSON in {path}: {exc}", path) from exc
    raise _invalid(f"project must be .yaml/.yml/.json: {path}", path)


def _read_text(path: Path, limit: int, label: str) -> str:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise _invalid(f"cannot stat {label} {path}: {exc}", path) from exc
    if size > limit:
        raise _invalid(f"{label} {path} is too large ({size} bytes; limit {limit})", path)
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise _invalid(f"cannot read {label} {path} as UTF-8: {exc}", path) from exc
    if "\x00" in raw:
        raise _invalid(f"{label} {path} contains NUL bytes", path)
    return raw


def split_paragraphs(text: str) -> list[str]:
    pieces = _BLANK_LINE.split(text)
    return [piece.strip() for piece in pieces if piece.strip()]


def _resolve_input(path_text: str, base: Path, label: str, limit: int) -> tuple[Path, str]:
    candidate = Path(path_text).expanduser()
    if not candidate.is_absolute():
        candidate = base / candidate
    candidate = candidate.resolve(strict=False)
    if not candidate.is_file():
        raise _invalid(f"{label} not found: {candidate}", candidate)
    raw = _read_text(candidate, limit, label)
    return candidate, raw


def _validate_bgm(mapping: dict[str, Any], base: Path, path: Path) -> Optional[BgmSpec]:
    if mapping is None:
        return None
    data = _mapping(mapping, "bgm", path=path)
    _unknown(data, _BGM_KEYS, "bgm", path)
    if "file" not in data:
        raise _invalid("bgm.file is required", path)
    raw_file = _string(data["file"], "bgm.file", path=path)
    candidate = Path(raw_file).expanduser()
    if not candidate.is_absolute():
        candidate = base / candidate
    candidate = candidate.resolve(strict=False)
    if not candidate.is_file():
        raise _invalid(f"BGM file not found: {candidate}", path)
    try:
        size = candidate.stat().st_size
    except OSError as exc:
        raise _invalid(f"cannot stat BGM {candidate}: {exc}", path) from exc
    if size > MAX_BGM_BYTES:
        raise _invalid(f"BGM file is too large ({size} bytes; limit {MAX_BGM_BYTES})", path)
    volume = _number(data.get("volume_db", -20.0), "bgm.volume_db", path=path, minimum=-60, maximum=0)
    ducking = _number(data.get("ducking_db", -10.0), "bgm.ducking_db", path=path, minimum=-60, maximum=0)
    fade = _number(data.get("fade_ms", 500), "bgm.fade_ms", path=path, minimum=0, maximum=60000, integer=True)
    return BgmSpec(candidate, float(volume), float(ducking), int(fade))


def load_project(path_value: str | Path) -> Project:
    path = Path(path_value).expanduser().resolve(strict=False)
    if not path.is_file():
        raise _invalid(f"project file not found: {path}", path)
    raw = _read_text(path, MAX_PROJECT_BYTES, "project")
    data = _load_project_data(path, raw)
    if not isinstance(data, dict):
        raise _invalid("project must be a mapping", path)
    _check_tree(data, path)
    _unknown(data, _TOP_KEYS, "project", path)

    schema = data.get("schema_version")
    if isinstance(schema, bool) or not isinstance(schema, int) or schema != 1:
        raise _invalid("schema_version must be the integer 1", path)

    stem = path.stem
    title = data.get("title", stem)
    title = _string(title, "title", path=path)
    author = data.get("author")
    if author is not None:
        author = _string(author, "author", path=path)
    album = data.get("album", title)
    album = _string(album, "album", path=path)
    language = _string(data.get("language", "auto"), "language", path=path)
    instruct = data.get("instruct")
    if instruct is not None:
        instruct = _string(instruct, "instruct", path=path, allow_empty=True)

    voice_raw = data.get("voice")
    voice = None
    if voice_raw is not None:
        voice = _string(voice_raw, "voice", path=path)
        try:
            parse_voice_ref(voice)
        except Exception as exc:
            raise _invalid(f"invalid voice reference: {voice!r}", path) from exc

    sample_rate = None
    if "sample_rate" in data:
        sample_rate = int(
            _number(
                data["sample_rate"],
                "sample_rate",
                path=path,
                minimum=8000,
                maximum=48000,
                integer=True,
            )
        )

    pause = _mapping(data.get("pause", {}), "pause", path=path)
    _unknown(pause, _PAUSE_KEYS, "pause", path)
    paragraph_ms = int(
        _number(pause.get("paragraph_ms", 400), "pause.paragraph_ms", path=path, minimum=0, maximum=60000, integer=True)
    )
    chapter_ms = int(
        _number(pause.get("chapter_ms", 1500), "pause.chapter_ms", path=path, minimum=0, maximum=60000, integer=True)
    )

    loudness = _mapping(data.get("loudness", {}), "loudness", path=path)
    _unknown(loudness, _LOUDNESS_KEYS, "loudness", path)
    lufs = None
    if "lufs" in loudness:
        lufs = float(_number(loudness["lufs"], "loudness.lufs", path=path, minimum=-40, maximum=-5))
    true_peak = None
    if "true_peak" in loudness:
        true_peak = float(
            _number(loudness["true_peak"], "loudness.true_peak", path=path, minimum=-9, maximum=0)
        )

    bgm = _validate_bgm(data.get("bgm"), path.parent, path)

    raw_chapters = data.get("chapters")
    if not isinstance(raw_chapters, list) or not raw_chapters:
        raise _invalid("chapters must be a non-empty list", path)
    if len(raw_chapters) > MAX_CHAPTERS:
        raise _invalid(f"too many chapters ({len(raw_chapters)}; limit {MAX_CHAPTERS})", path)

    chapters: list[Chapter] = []
    total_paragraphs = 0
    total_text_chars = 0
    for index, raw_chapter in enumerate(raw_chapters, 1):
        chapter_data = _mapping(raw_chapter, f"chapters[{index}]", path=path)
        _unknown(chapter_data, _CHAPTER_KEYS, f"chapters[{index}]", path)
        chapter_title = _string(chapter_data.get("title"), f"chapters[{index}].title", path=path)
        has_source = "source" in chapter_data
        has_text = "text" in chapter_data
        if has_source == has_text:
            raise _invalid(f"chapters[{index}] must have exactly one of source/text", path)
        source_path: Optional[Path] = None
        inline = has_text
        if has_source:
            source_name = _string(chapter_data["source"], f"chapters[{index}].source", path=path)
            source_path, body = _resolve_input(source_name, path.parent, "source", MAX_SOURCE_BYTES)
            source_sha = sha256_file(source_path)
        else:
            body = _string(chapter_data["text"], f"chapters[{index}].text", path=path)
            source_sha = sha256_text(body)
        paragraphs = split_paragraphs(body)
        if not paragraphs:
            raise _invalid(f"chapters[{index}] has no paragraphs", path)
        if len(paragraphs) > MAX_PARAGRAPHS_PER_CHAPTER:
            raise _invalid(
                f"chapters[{index}] has too many paragraphs "
                f"({len(paragraphs)}; limit {MAX_PARAGRAPHS_PER_CHAPTER})",
                path,
            )
        for paragraph in paragraphs:
            if len(paragraph) > MAX_PARAGRAPH_CHARS:
                raise _invalid("a paragraph exceeds 200000 characters", path)
        total_paragraphs += len(paragraphs)
        total_text_chars += len(body)
        if total_paragraphs > MAX_PARAGRAPHS:
            raise _invalid(f"too many paragraphs ({total_paragraphs}; limit {MAX_PARAGRAPHS})", path)
        if total_text_chars > MAX_CHAPTER_TEXT_CHARS:
            raise _invalid(f"chapter text exceeds {MAX_CHAPTER_TEXT_CHARS} characters", path)

        chapter_voice_raw = chapter_data.get("voice", voice)
        if chapter_voice_raw is None:
            raise _invalid(f"chapters[{index}] has no voice (project voice is unset)", path)
        chapter_voice = _string(chapter_voice_raw, f"chapters[{index}].voice", path=path)
        try:
            parse_voice_ref(chapter_voice)
        except Exception as exc:
            raise _invalid(f"invalid voice reference: {chapter_voice!r}", path) from exc
        chapter_language = _string(
            chapter_data.get("language", language),
            f"chapters[{index}].language",
            path=path,
        )
        chapter_instruct = chapter_data.get("instruct", instruct)
        if chapter_instruct is not None:
            chapter_instruct = _string(
                chapter_instruct,
                f"chapters[{index}].instruct",
                path=path,
                allow_empty=True,
            )
        chapters.append(
            Chapter(
                index=index,
                title=chapter_title,
                paragraphs=paragraphs,
                voice=chapter_voice,
                language=chapter_language,
                instruct=chapter_instruct,
                source_path=source_path,
                source_sha256=source_sha,
                inline=inline,
                text="\n\n".join(paragraphs),
            )
        )

    return Project(
        path=path,
        sha256=sha256_file(path),
        title=title,
        author=author,
        album=album,
        language=language,
        voice=voice or chapters[0].voice,
        instruct=instruct,
        sample_rate=sample_rate,
        paragraph_ms=paragraph_ms,
        chapter_ms=chapter_ms,
        lufs=lufs,
        true_peak=true_peak,
        bgm=bgm,
        chapters=chapters,
    )
