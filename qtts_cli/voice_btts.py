"""Read and verify ``btts voice export --mode full`` bundles (FR-P2-09).

The CLI command that consumes this module lives in
:mod:`qtts_cli.commands.voice`.  This module only parses the bundle: it never
loads a model, writes to the voice store, or trusts a path without validating
it.

Bundle layout (btts 0.2.x)::

    manifest.json
    LICENSE.txt
    README.txt
    <voice>/profile.json
    <voice>/LICENSE.txt
    <voice>/preview.wav
    <voice>/ref.wav + ref.txt          # kind=clone
    <voice>/design.yaml               # kind=design (optional in btts)

``btts voice export --zip`` wraps the same tree in one top-level directory.
Every file is hashed; when ``manifest.json`` is present it must list exactly
the files in the bundle and every hash must match.  Symlinks, special files,
absolute/backslash/``..`` paths, casefold collisions and unknown files are
rejected before anything is read into the qtts store.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import stat
import struct
import warnings
import zipfile
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

from .errors import InputError
from .voice_store import validate_name

MANIFEST = "manifest.json"
ROOT_FILES = frozenset({"manifest.json", "LICENSE.txt", "README.txt"})
FIXED_VOICE_FILES = frozenset({"profile.json", "LICENSE.txt", "preview.wav", "design.yaml"})
PROFILE_SCHEMA_VERSION = 1
EXPORT_SCHEMA_VERSION = 1

MAX_FILES = 512
MAX_MEMBER_BYTES = 64 * 1024 * 1024
MAX_TOTAL_BYTES = 512 * 1024 * 1024
MAX_PROFILE_BYTES = 1024 * 1024
MAX_TEXT_BYTES = 1024 * 1024
MAX_COMPRESSION_RATIO = 500.0

# Exceptions the stdlib zip reader can raise while parsing an untrusted
# central directory/local header.  They must surface as InputError (exit 4)
# with bundle context instead of escaping to the CLI as a traceback or being
# mistaken for a usage error.
_ZIP_PARSE_ERRORS = (
    zipfile.BadZipFile,
    zipfile.LargeZipFile,
    NotImplementedError,
    UnicodeError,
    ValueError,
    UserWarning,
    OSError,
    EOFError,
    RuntimeError,
    struct.error,
    zlib.error,
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

# btts profiles use ISO-639-1 codes; qtts uses display names.  Accept the
# qtts names too so a hand-repaired profile is not rejected unnecessarily.
_LANGUAGE_CODES = {
    "zh": "Chinese",
    "zh-cn": "Chinese",
    "zh-hans": "Chinese",
    "cmn": "Chinese",
    "en": "English",
    "en-us": "English",
    "en-gb": "English",
    "ja": "Japanese",
    "ko": "Korean",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
}
_LANGUAGE_NAMES = {
    "Chinese",
    "English",
    "Japanese",
    "Korean",
    "German",
    "French",
    "Spanish",
    "Italian",
    "Portuguese",
    "Russian",
}
_LANGUAGE_BY_CASEFOLD = {name.casefold(): name for name in _LANGUAGE_NAMES}


@dataclass(frozen=True)
class FileRef:
    rel: str
    size: int
    sha256: str


@dataclass(frozen=True)
class BundleVoice:
    name: str
    kind: str
    language: str
    btts_language: str
    voice_id: str
    profile: dict[str, Any]
    profile_rel: str
    profile_sha256: str
    files: dict[str, FileRef]
    ref_audio: Optional[str]
    ref_transcript: Optional[str]
    ref_text: Optional[str]
    preview: str
    design_yaml: Optional[str]
    tags: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class BttsBundle:
    path: Path
    format: str
    root_prefix: str
    manifest_present: bool
    manifest_verified: bool
    manifest_sha256: Optional[str]
    manifest: Optional[dict[str, Any]]
    voices: dict[str, BundleVoice]
    files: dict[str, FileRef]
    warnings: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def load_bundle(path: str | Path) -> BttsBundle:
    """Parse, validate and hash a btts export directory or zip bundle."""
    source = Path(path).expanduser()
    if source.is_dir():
        scanned, warnings = _scan_directory(source)
        return _finalize(source, "directory", "", scanned, warnings)
    if source.is_file():
        scanned, root_prefix, warnings = _scan_zip(source)
        return _finalize(source, "zip", root_prefix, scanned, warnings)
    raise InputError(
        f"btts bundle not found: {source}",
        details={"bundle": str(source)},
    )


def read_bundle_file(bundle: BttsBundle, rel: str) -> bytes:
    """Read one already-verified member and re-check its size and hash."""
    ref = bundle.files.get(rel)
    if ref is None:
        _fail("bundle file is not listed", bundle, file=rel)
    limit = MAX_MEMBER_BYTES
    if rel == MANIFEST:
        limit = MAX_PROFILE_BYTES
    data = _read_rel(bundle, rel, limit)
    if len(data) != ref.size:
        _fail("bundle file changed while importing", bundle, file=rel)
    if _sha256_bytes(data) != ref.sha256:
        _fail("bundle file changed while importing", bundle, file=rel)
    return data


def map_language(value: Any, *, bundle: Path | str, voice: str) -> tuple[str, str]:
    """Return ``(qtts_language, original_btts_language)`` or raise InputError."""
    if not isinstance(value, str) or not value.strip():
        _fail(
            "profile.json language must be a non-empty string",
            bundle,
            voice=voice,
            file=f"{voice}/profile.json",
        )
    raw = value.strip()
    mapped = _LANGUAGE_BY_CASEFOLD.get(raw.casefold())
    if mapped is None:
        mapped = _LANGUAGE_CODES.get(raw.casefold())
    if mapped is None:
        _fail(
            f"unsupported btts language {raw!r}",
            bundle,
            voice=voice,
            file=f"{voice}/profile.json",
            supported=sorted(_LANGUAGE_NAMES),
        )
    return mapped, raw


# --------------------------------------------------------------------------- #
# directory scan
# --------------------------------------------------------------------------- #
def _scan_directory(root: Path) -> tuple[dict[str, int], list[str]]:
    warnings: list[str] = []
    scanned: dict[str, int] = {}
    seen_paths: dict[str, str] = {}
    seen_entries: dict[str, str] = {}

    try:
        entries = sorted(root.iterdir(), key=lambda item: item.name)
    except OSError as exc:
        raise InputError(
            f"cannot read btts bundle {root}: {exc}",
            details={"bundle": str(root)},
        ) from exc

    for entry in entries:
        name = entry.name
        _check_component(name, root, rel=name)
        folded = name.casefold()
        other = seen_entries.get(folded)
        if other is not None:
            _fail(
                "bundle entries collide on a case-insensitive filesystem",
                root,
                file=name,
                other=other,
            )
        seen_entries[folded] = name
        try:
            info = entry.lstat()
        except OSError as exc:
            raise InputError(
                f"cannot inspect bundle entry {name!r}: {exc}",
                details={"bundle": str(root), "file": name},
            ) from exc
        if stat.S_ISLNK(info.st_mode):
            _fail("bundle contains a symlink", root, file=name)
        if stat.S_ISDIR(info.st_mode):
            _validate_voice_name(name, root, file=name)
            _scan_voice_directory(root, name, scanned, seen_paths, warnings)
        elif stat.S_ISREG(info.st_mode):
            # Root files are validated after manifest parsing in ``_finalize``
            # so a real ``--mode wav`` export (root ``*.wav`` files) gets the
            # actionable "re-export with --mode full" error.
            _record_file(scanned, seen_paths, root, rel=name, size=info.st_size)
        else:
            _fail("bundle contains a special file", root, file=name)

    if sum(scanned.values()) > MAX_TOTAL_BYTES:
        _fail("bundle expands beyond the size limit", root)
    if MANIFEST not in scanned:
        warnings.append("manifest.json is missing; file hashes were not verified")
    return scanned, warnings


def _scan_voice_directory(
    root: Path,
    voice: str,
    scanned: dict[str, int],
    seen_paths: dict[str, str],
    warnings: list[str],
) -> None:
    directory = root / voice
    try:
        entries = sorted(directory.iterdir(), key=lambda item: item.name)
    except OSError as exc:
        raise InputError(
            f"cannot read voice directory {voice!r}: {exc}",
            details={"bundle": str(root), "voice": voice},
        ) from exc
    if not entries:
        _fail("voice directory is empty", root, voice=voice)
    for entry in entries:
        name = entry.name
        _check_component(name, root, voice=voice, rel=f"{voice}/{name}")
        try:
            info = entry.lstat()
        except OSError as exc:
            raise InputError(
                f"cannot inspect voice file {voice}/{name}: {exc}",
                details={"bundle": str(root), "voice": voice, "file": f"{voice}/{name}"},
            ) from exc
        if stat.S_ISLNK(info.st_mode):
            _fail("voice directory contains a symlink", root, voice=voice,
                  file=f"{voice}/{name}")
        if stat.S_ISDIR(info.st_mode):
            _fail("voice directory contains a nested directory", root, voice=voice,
                  file=f"{voice}/{name}")
        if not stat.S_ISREG(info.st_mode):
            _fail("voice directory contains a special file", root, voice=voice,
                  file=f"{voice}/{name}")
        _record_file(scanned, seen_paths, root, rel=f"{voice}/{name}", size=info.st_size)


def _record_file(
    scanned: dict[str, int],
    seen_paths: dict[str, str],
    root: Path,
    *,
    rel: str,
    size: int,
) -> None:
    if len(scanned) + 1 > MAX_FILES:
        _fail(f"bundle has too many files (> {MAX_FILES})", root, file=rel)
    if size <= 0:
        _fail("bundle contains an empty file", root, file=rel)
    if size > MAX_MEMBER_BYTES:
        _fail("bundle file is too large", root, file=rel, size=size)
    folded = rel.casefold()
    other = seen_paths.get(folded)
    if other is not None:
        _fail(
            "bundle paths collide on a case-insensitive filesystem",
            root,
            file=rel,
            other=other,
        )
    seen_paths[folded] = rel
    scanned[rel] = size


# --------------------------------------------------------------------------- #
# zip scan
# --------------------------------------------------------------------------- #
@contextlib.contextmanager
def _zip_warnings_as_errors() -> Iterator[None]:
    """Turn stdlib zip warnings (e.g. overlapped entries) into exceptions.

    Without this an untrusted bundle can print a bare ``UserWarning`` to
    stderr (breaking the ``--json`` "exactly one object" contract) or crash
    the CLI when ``PYTHONWARNINGS=error`` is set.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        yield


def _scan_zip(path: Path) -> tuple[dict[str, int], str, list[str]]:
    try:
        return _scan_zip_members(path)
    except InputError:
        raise
    except _ZIP_PARSE_ERRORS as exc:
        raise InputError(
            f"invalid btts zip bundle: {exc}",
            details={"bundle": str(path), "file": None},
        ) from exc


def _scan_zip_members(path: Path) -> tuple[dict[str, int], str, list[str]]:
    warnings_list: list[str] = []
    with _zip_warnings_as_errors(), zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
    if not infos:
        _fail("empty btts zip bundle", path)

    file_infos: list[zipfile.ZipInfo] = []
    directory_infos: list[zipfile.ZipInfo] = []
    seen: set[str] = set()
    total = 0
    for info in infos:
        name = info.filename
        if name in seen:
            _fail("duplicate zip member", path, file=name)
        seen.add(name)
        _check_zip_member(info, path)
        if info.is_dir():
            directory_infos.append(info)
            continue
        file_infos.append(info)
        total += info.file_size
        if total > MAX_TOTAL_BYTES:
            _fail("bundle expands beyond the size limit", path)

    if not file_infos:
        _fail("empty btts zip bundle", path)

    parts_list: list[list[str]] = []
    for info in file_infos:
        parts = info.filename.split("/")
        _check_parts(parts, path, rel=info.filename)
        parts_list.append(parts)
    has_root_files = any(len(parts) == 1 for parts in parts_list)
    if has_root_files:
        root_prefix = ""
    else:
        tops = {parts[0] for parts in parts_list}
        if len(tops) != 1:
            _fail(
                "btts zip bundle must have a single top-level directory or a flat layout",
                path,
            )
        root_prefix = f"{next(iter(tops))}/"

    _validate_zip_directories(directory_infos, path, root_prefix)

    scanned: dict[str, int] = {}
    seen_paths: dict[str, str] = {}
    for info in file_infos:
        name = info.filename
        if root_prefix:
            if not name.startswith(root_prefix):
                _fail("zip members do not share one top-level directory", path, file=name)
            rel = name[len(root_prefix) :]
        else:
            rel = name
        parts = rel.split("/")
        _check_parts(parts, path, rel=name)
        if len(parts) > 2:
            _fail("nested voice directories are not allowed", path, file=name)
        # Root files are checked after manifest parsing in ``_finalize`` (see
        # the directory scanner) so ``--mode wav`` exports get a clear hint.
        _record_zip_file(scanned, seen_paths, path, rel=rel, info=info)

    if MANIFEST not in scanned:
        _fail("btts zip bundle has no manifest.json", path)
    return scanned, root_prefix, warnings_list


def _check_zip_member(info: zipfile.ZipInfo, path: Path) -> None:
    name = info.filename
    if not name or name != name.strip() or "\x00" in name:
        _fail("invalid zip member name", path, file=repr(name))
    if name.startswith(("/", "\\")) or "\\" in name or ":" in name.split("/")[0]:
        _fail("zip member uses an absolute or drive path", path, file=name)
    if info.flag_bits & 0x1:
        _fail("encrypted zip members are not allowed", path, file=name)
    mode = info.external_attr >> 16
    file_type = stat.S_IFMT(mode)
    if file_type and file_type not in (stat.S_IFREG, stat.S_IFDIR):
        _fail("non-regular zip member is not allowed", path, file=name)
    if info.file_size > MAX_MEMBER_BYTES:
        _fail("bundle file is too large", path, file=name, size=info.file_size)
    if info.file_size and info.compress_size == 0:
        _fail("suspicious zip compression", path, file=name)
    if info.compress_size and info.file_size / info.compress_size > MAX_COMPRESSION_RATIO:
        _fail("zip member compression ratio too high", path, file=name)
    if not info.is_dir() and info.file_size <= 0:
        _fail("bundle contains an empty file", path, file=name)


def _validate_zip_directories(
    directory_infos: list[zipfile.ZipInfo], path: Path, root_prefix: str
) -> None:
    if not directory_infos:
        return
    if not root_prefix:
        _fail("flat btts zip bundles must not contain directory members", path)
    expected = root_prefix.rstrip("/")
    if len(directory_infos) != 1 or directory_infos[0].filename.rstrip("/") != expected:
        _fail("zip bundle has an unexpected directory member", path)


def _record_zip_file(
    scanned: dict[str, int],
    seen_paths: dict[str, str],
    path: Path,
    *,
    rel: str,
    info: zipfile.ZipInfo,
) -> None:
    if len(scanned) + 1 > MAX_FILES:
        _fail(f"bundle has too many files (> {MAX_FILES})", path, file=rel)
    folded = rel.casefold()
    other = seen_paths.get(folded)
    if other is not None:
        _fail(
            "bundle paths collide on a case-insensitive filesystem",
            path,
            file=rel,
            other=other,
        )
    seen_paths[folded] = rel
    scanned[rel] = info.file_size


def _check_parts(parts: list[str], path: Path, *, rel: str) -> None:
    if not parts or any(part in ("", ".", "..") for part in parts):
        _fail("bundle member escapes the bundle root", path, file=rel)
    for part in parts:
        _check_component(part, path, rel=rel)


def _validate_voice_name(name: str, path: Path, *, file: Optional[str] = None) -> None:
    try:
        validate_name(name)
    except InputError as exc:
        extra = {
            key: value
            for key, value in exc.details.items()
            if key not in ("bundle", "voice", "file")
        }
        _fail(exc.message, path, voice=name, file=file, **extra)


def _check_component(
    name: str,
    path: Path,
    *,
    rel: str,
    voice: Optional[str] = None,
) -> None:
    if (
        not name
        or name in (".", "..")
        or name != name.strip()
        or any(ord(ch) < 32 or ord(ch) == 127 for ch in name)
        or "\x00" in name
        or "/" in name
        or "\\" in name
    ):
        _fail("invalid bundle entry name", path, voice=voice, file=rel)


# --------------------------------------------------------------------------- #
# hashing / verification
# --------------------------------------------------------------------------- #
def _finalize(
    path: Path,
    fmt: str,
    root_prefix: str,
    scanned: dict[str, int],
    warnings: list[str],
) -> BttsBundle:
    refs: dict[str, FileRef] = {}
    for rel, expected_size in sorted(scanned.items()):
        digest, size = _hash_rel(path, fmt, root_prefix, rel)
        if size != expected_size:
            _fail("bundle file changed while reading", path, file=rel)
        refs[rel] = FileRef(rel=rel, size=size, sha256=digest)

    manifest: Optional[dict[str, Any]] = None
    manifest_sha256: Optional[str] = None
    manifest_verified = False
    if MANIFEST in refs:
        manifest_ref = refs[MANIFEST]
        if manifest_ref.size > MAX_PROFILE_BYTES:
            _fail(
                "manifest.json is too large",
                path,
                file=MANIFEST,
                size=manifest_ref.size,
            )
        raw = _read_ref(path, fmt, root_prefix, manifest_ref)
        manifest = _parse_manifest(raw, path)
        manifest_sha256 = manifest_ref.sha256
        _verify_manifest(path, manifest, refs)
        manifest_verified = True
    elif fmt == "zip":
        _fail("btts zip bundle has no manifest.json", path)
    else:
        # No manifest: only the documented root files are allowed, and a
        # ``--mode wav`` export (root ``*.wav`` files, no profiles) gets a
        # targeted error instead of a generic "unexpected root file".
        _check_manifestless_root_files(path, refs)

    voice_names = sorted({rel.split("/")[0] for rel in refs if "/" in rel})
    voices: dict[str, BundleVoice] = {}
    for voice_name in voice_names:
        voice = _build_voice(path, fmt, root_prefix, refs, voice_name)
        voices[voice_name] = voice
        if voice.kind == "design" and voice.design_yaml is None:
            warnings.append(
                f"{voice.name}: design.yaml is missing; profile.json still "
                "contains the design recipe"
            )
    if not voices:
        _fail("btts bundle contains no voices", path)

    if manifest is not None:
        _cross_check_manifest_voices(path, manifest, refs, voices)

    return BttsBundle(
        path=path,
        format=fmt,
        root_prefix=root_prefix,
        manifest_present=manifest is not None,
        manifest_verified=manifest_verified,
        manifest_sha256=manifest_sha256,
        manifest=manifest,
        voices=voices,
        files=refs,
        warnings=warnings,
    )


def _check_manifestless_root_files(path: Path, refs: dict[str, FileRef]) -> None:
    unexpected = sorted(
        rel
        for rel in refs
        if "/" not in rel and rel not in ROOT_FILES - {MANIFEST}
    )
    if not unexpected:
        return
    wav_files = [rel for rel in unexpected if rel.lower().endswith(".wav")]
    if wav_files:
        _fail(
            "btts bundle looks like a `--mode wav` export (found root .wav "
            "files without profiles); re-export with "
            "`btts voice export --mode full`",
            path,
            file=wav_files[0],
            files=unexpected,
        )
    _fail(f"unexpected root file in btts bundle: {unexpected[0]!r}", path,
          file=unexpected[0], files=unexpected)


def _build_voice(
    path: Path,
    fmt: str,
    root_prefix: str,
    refs: dict[str, FileRef],
    name: str,
) -> BundleVoice:
    _validate_voice_name(name, path, file=f"{name}/profile.json")
    profile_rel = f"{name}/profile.json"
    ref = refs.get(profile_rel)
    if ref is None:
        _fail("voice directory is missing profile.json", path, voice=name, file=profile_rel)
    if ref.size > MAX_PROFILE_BYTES:
        _fail("profile.json is too large", path, voice=name, file=profile_rel)
    profile = _parse_profile(_read_ref(path, fmt, root_prefix, ref), path, name)

    voice_id = profile.get("voice_id")
    if not isinstance(voice_id, str) or not voice_id.strip():
        _fail("profile.json voice_id must be a non-empty string", path, voice=name,
              file=profile_rel)
    kind = profile.get("kind")
    if kind not in ("clone", "design"):
        _fail("profile.json kind must be clone or design", path, voice=name,
              file=profile_rel)
    language, btts_language = map_language(profile.get("language"), bundle=path, voice=name)

    fixed = set(FIXED_VOICE_FILES)
    ref_audio: Optional[str] = None
    ref_transcript: Optional[str] = None
    ref_text: Optional[str] = None
    design_yaml = f"{name}/design.yaml" if f"{name}/design.yaml" in refs else None

    if kind == "clone":
        reference = profile.get("reference")
        if not isinstance(reference, dict):
            _fail("clone profile.json must contain a reference object", path, voice=name,
                  file=profile_rel)
        ref_audio = _safe_profile_filename(reference.get("audio", "ref.wav"), path, name)
        ref_transcript = _safe_profile_filename(
            reference.get("transcript", "ref.txt"), path, name
        )
        fixed.update({ref_audio, ref_transcript})
        for filename in (ref_audio, ref_transcript, "preview.wav"):
            rel = f"{name}/{filename}"
            if rel not in refs:
                _fail(f"clone bundle is missing {filename}", path, voice=name, file=rel)
        expected_audio = _profile_hash(reference.get("audio_sha256"), path, name, "reference.audio_sha256")
        if refs[f"{name}/{ref_audio}"].sha256 != expected_audio:
            _fail("reference audio hash does not match profile.json", path, voice=name,
                  file=f"{name}/{ref_audio}")
        expected_text = _profile_hash(
            reference.get("transcript_sha256"), path, name, "reference.transcript_sha256"
        )
        text_ref = refs[f"{name}/{ref_transcript}"]
        if text_ref.size > MAX_TEXT_BYTES:
            _fail("reference transcript is too large", path, voice=name,
                  file=f"{name}/{ref_transcript}")
        raw_text = _read_ref(path, fmt, root_prefix, text_ref)
        ref_text = _decode_transcript(raw_text, path, name, ref_transcript)
        if _sha256_text(ref_text) != expected_text:
            _fail("reference transcript hash does not match profile.json", path,
                  voice=name, file=f"{name}/{ref_transcript}")
        if isinstance(profile.get("design"), dict):
            _fail("clone profile.json must not contain a design block", path, voice=name,
                  file=profile_rel)
    else:
        reference = profile.get("reference")
        if reference is not None:
            _fail("design profile.json must not contain a reference block", path,
                  voice=name, file=profile_rel)
        if not isinstance(profile.get("design"), dict):
            _fail("design profile.json must contain a design object", path, voice=name,
                  file=profile_rel)
        if f"{name}/preview.wav" not in refs:
            _fail("design bundle is missing preview.wav", path, voice=name,
                  file=f"{name}/preview.wav")

    allowed = fixed | {"profile.json", "preview.wav"}
    for rel in refs:
        if not rel.startswith(f"{name}/"):
            continue
        filename = rel[len(name) + 1 :]
        if filename not in allowed:
            _fail(f"unexpected voice file: {filename!r}", path, voice=name, file=rel)

    tags = _profile_tags(profile.get("tags"), path, name)
    files = {
        rel[len(name) + 1 :]: refs[rel]
        for rel in refs
        if rel.startswith(f"{name}/")
    }
    return BundleVoice(
        name=name,
        kind=kind,
        language=language,
        btts_language=btts_language,
        voice_id=voice_id,
        profile=profile,
        profile_rel=profile_rel,
        profile_sha256=ref.sha256,
        files=files,
        ref_audio=ref_audio,
        ref_transcript=ref_transcript,
        ref_text=ref_text,
        preview=f"{name}/preview.wav",
        design_yaml=design_yaml,
        tags=tags,
    )


def _verify_manifest(
    path: Path, manifest: dict[str, Any], refs: dict[str, FileRef]
) -> None:
    listed = _manifest_file_map(path, manifest)
    actual = set(refs) - {MANIFEST}
    expected = set(listed)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        _fail(
            "bundle files do not exactly match manifest.json",
            path,
            missing=missing,
            extra=extra,
        )
    for rel, entry in listed.items():
        ref = refs[rel]
        if ref.size != entry["size"]:
            _fail("manifest size does not match the bundle file", path, file=rel,
                  expected=entry["size"], actual=ref.size)
        if ref.sha256 != entry["sha256"]:
            _fail("manifest sha256 does not match the bundle file", path, file=rel,
                  expected=entry["sha256"], actual=ref.sha256)


def _manifest_file_map(path: Path, manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}
    root_files = manifest.get("root_files")
    if not isinstance(root_files, list):
        _fail("manifest.root_files must be a list", path)
    for entry in root_files:
        _manifest_entry(entries, path, entry, root=True)
    voices = manifest.get("voices")
    if not isinstance(voices, list) or not voices:
        _fail("manifest.voices must be a non-empty list", path)
    for voice in voices:
        if not isinstance(voice, dict):
            _fail("manifest.voices entries must be objects", path)
        name = voice.get("name")
        if not isinstance(name, str):
            _fail("manifest voice name must be a string", path)
        _validate_voice_name(name, path)
        files = voice.get("files")
        if not isinstance(files, list) or not files:
            _fail("manifest voice files must be a non-empty list", path, voice=name)
        for entry in files:
            _manifest_entry(entries, path, entry, voice=name)
    return entries


def _manifest_entry(
    entries: dict[str, dict[str, Any]],
    path: Path,
    entry: Any,
    *,
    voice: Optional[str] = None,
    root: bool = False,
) -> None:
    if not isinstance(entry, dict):
        _fail("manifest file entries must be objects", path, voice=voice)
    rel = entry.get("path")
    if not isinstance(rel, str) or not rel:
        _fail("manifest file paths must be strings", path, voice=voice)
    _check_parts(rel.split("/"), path, rel=rel)
    if len(rel.split("/")) > 2:
        _fail("nested voice directories are not allowed", path, file=rel)
    if root:
        if "/" in rel or rel not in ROOT_FILES - {MANIFEST}:
            _fail(f"unexpected root file in manifest: {rel!r}", path, file=rel)
    else:
        assert voice is not None
        if not rel.startswith(f"{voice}/") or rel.count("/") != 1:
            _fail("manifest voice file is outside its voice directory", path,
                  voice=voice, file=rel)
    if rel in entries:
        _fail("manifest lists a file more than once", path, file=rel)
    digest = _normalize_hash(entry.get("sha256"), path, rel)
    size = entry.get("bytes")
    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
        _fail("manifest file size must be a non-negative integer", path, file=rel)
    entries[rel] = {"sha256": digest, "size": size}


def _cross_check_manifest_voices(
    path: Path,
    manifest: dict[str, Any],
    refs: dict[str, FileRef],
    voices: dict[str, BundleVoice],
) -> None:
    entries = manifest.get("voices")
    assert isinstance(entries, list)
    seen: dict[str, str] = {}
    listed_names: set[str] = set()
    for entry in entries:
        name = entry["name"]
        folded = name.casefold()
        if folded in seen:
            _fail("manifest has duplicate voice names", path, voice=name, other=seen[folded])
        seen[folded] = name
        listed_names.add(name)
        voice = voices.get(name)
        if voice is None:
            _fail("manifest lists a voice that is not in the bundle", path, voice=name)
        if entry.get("kind") != voice.kind:
            _fail("manifest kind does not match profile.json", path, voice=name,
                  file=voice.profile_rel)
        if entry.get("language") != voice.profile.get("language"):
            _fail("manifest language does not match profile.json", path, voice=name,
                  file=voice.profile_rel)
        expected_profile = _normalize_hash(
            entry.get("profile_sha256"), path, voice.profile_rel
        )
        if expected_profile != voice.profile_sha256:
            _fail("manifest profile_sha256 does not match profile.json", path,
                  voice=name, file=voice.profile_rel)
    missing = sorted(set(voices) - listed_names)
    if missing:
        _fail("bundle voices are missing from manifest.json", path, voices=missing)


# --------------------------------------------------------------------------- #
# profile parsing
# --------------------------------------------------------------------------- #
def _parse_profile(raw: bytes, path: Path, name: str) -> dict[str, Any]:
    try:
        profile = json.loads(raw.decode("utf-8"), parse_constant=_reject_constant)
    except (UnicodeDecodeError, ValueError) as exc:
        _fail(f"invalid profile.json: {exc}", path, voice=name,
              file=f"{name}/profile.json")
    if not isinstance(profile, dict):
        _fail("profile.json must be an object", path, voice=name,
              file=f"{name}/profile.json")
    version = profile.get("schema_version")
    if not isinstance(version, int) or isinstance(version, bool) or version != PROFILE_SCHEMA_VERSION:
        _fail(
            f"unsupported profile.json schema_version: {version!r}",
            path,
            voice=name,
            file=f"{name}/profile.json",
        )
    if profile.get("name") != name:
        _fail("profile.json name does not match its directory", path, voice=name,
              file=f"{name}/profile.json")
    return profile


def _parse_manifest(raw: bytes, path: Path) -> dict[str, Any]:
    try:
        manifest = json.loads(raw.decode("utf-8"), parse_constant=_reject_constant)
    except (UnicodeDecodeError, ValueError) as exc:
        _fail(f"invalid manifest.json: {exc}", path, file=MANIFEST)
    if not isinstance(manifest, dict):
        _fail("manifest.json must be an object", path, file=MANIFEST)
    version = manifest.get("schema_version")
    if not isinstance(version, int) or isinstance(version, bool) or version != EXPORT_SCHEMA_VERSION:
        _fail(f"unsupported manifest schema_version: {version!r}", path, file=MANIFEST)
    if manifest.get("kind") != "btts-voice-export":
        _fail("manifest.json is not a btts voice export", path, file=MANIFEST)
    if manifest.get("mode") != "full":
        _fail(
            "btts bundle was exported without profiles; re-export with "
            "`btts voice export --mode full`",
            path,
            file=MANIFEST,
            mode=manifest.get("mode"),
        )
    return manifest


def _decode_transcript(raw: bytes, path: Path, voice: str, filename: str) -> str:
    try:
        text = raw.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
    except UnicodeDecodeError as exc:
        _fail(f"reference transcript is not UTF-8: {exc}", path, voice=voice,
              file=f"{voice}/{filename}")
    text = text.rstrip("\n")
    if not text.strip():
        _fail("reference transcript is empty", path, voice=voice,
              file=f"{voice}/{filename}")
    return text


def _safe_profile_filename(value: Any, path: Path, voice: str) -> str:
    if not isinstance(value, str) or not value:
        _fail("profile reference filenames must be non-empty strings", path,
              voice=voice, file=f"{voice}/profile.json")
    if (
        value in (".", "..")
        or value != value.strip()
        or any(ord(ch) < 32 or ord(ch) == 127 for ch in value)
        or "\x00" in value
        or "/" in value
        or "\\" in value
        or Path(value).is_absolute()
    ):
        _fail("profile reference filename must be a plain basename", path,
              voice=voice, file=f"{voice}/profile.json")
    return value


def _profile_hash(value: Any, path: Path, voice: str, field_name: str) -> str:
    return _normalize_hash(value, path, f"{voice}/profile.json")


def _normalize_hash(value: Any, path: Path, rel: str) -> str:
    if not isinstance(value, str):
        _fail("manifest/profile hash must be a string", path, file=rel)
    digest = value.strip().lower()
    if digest.startswith("sha256:"):
        digest = digest[len("sha256:") :]
    if not _SHA256_RE.match(digest):
        _fail("invalid sha256 value", path, file=rel, value=value)
    return digest


def _profile_tags(value: Any, path: Path, voice: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        _fail("profile.json tags must be a list", path, voice=voice,
              file=f"{voice}/profile.json")
    tags: list[str] = []
    seen: set[str] = set()
    for tag in value:
        if not isinstance(tag, str) or not tag.strip() or len(tag) > 64:
            _fail("profile.json tags must be non-empty strings (max 64 chars)",
                  path, voice=voice, file=f"{voice}/profile.json")
        key = tag.casefold()
        if key in seen:
            continue
        seen.add(key)
        tags.append(tag)
    if len(tags) > 64:
        _fail("profile.json has too many tags", path, voice=voice,
              file=f"{voice}/profile.json")
    return tags


# --------------------------------------------------------------------------- #
# raw file access
# --------------------------------------------------------------------------- #
def _hash_rel(path: Path, fmt: str, root_prefix: str, rel: str) -> tuple[str, int]:
    digest = hashlib.sha256()
    total = 0
    with _open_rel(path, fmt, root_prefix, rel) as handle:
        while True:
            chunk = handle.read(1 << 20)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_MEMBER_BYTES:
                _fail("bundle file is too large", path, file=rel)
            digest.update(chunk)
    return digest.hexdigest(), total


def _read_ref(path: Path, fmt: str, root_prefix: str, ref: FileRef) -> bytes:
    data = _read_rel_path(path, fmt, root_prefix, ref.rel, ref.size)
    if len(data) != ref.size or _sha256_bytes(data) != ref.sha256:
        _fail("bundle file changed while importing", path, file=ref.rel)
    return data


def _read_rel(bundle: BttsBundle, rel: str, limit: int) -> bytes:
    return _read_rel_path(bundle.path, bundle.format, bundle.root_prefix, rel, limit)


def _read_rel_path(path: Path, fmt: str, root_prefix: str, rel: str, limit: int) -> bytes:
    with _open_rel(path, fmt, root_prefix, rel) as handle:
        data = handle.read(limit + 1)
    if len(data) > limit:
        _fail("bundle file is too large", path, file=rel)
    return data


@contextlib.contextmanager
def _open_rel(path: Path, fmt: str, root_prefix: str, rel: str) -> Iterator[Any]:
    if fmt == "directory":
        target = path.joinpath(*rel.split("/"))
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            fd = os.open(target, flags)
        except OSError as exc:
            raise InputError(
                f"cannot read bundle file {rel}: {exc}",
                details={"bundle": str(path), "file": rel},
            ) from exc
        try:
            try:
                info = os.fstat(fd)
            except OSError as exc:
                raise InputError(
                    f"cannot read bundle file {rel}: {exc}",
                    details={"bundle": str(path), "file": rel},
                ) from exc
            if not stat.S_ISREG(info.st_mode):
                _fail("bundle file is not a regular file", path, file=rel)
            with os.fdopen(fd, "rb") as handle:
                fd = -1
                yield handle
        finally:
            if fd >= 0:
                os.close(fd)
        return

    try:
        with _zip_warnings_as_errors(), zipfile.ZipFile(path) as archive:
            member = f"{root_prefix}{rel}"
            try:
                info = archive.getinfo(member)
            except KeyError:
                _fail("zip member is missing", path, file=rel)
            with archive.open(info) as handle:
                yield handle
    except InputError:
        raise
    except _ZIP_PARSE_ERRORS as exc:
        raise InputError(
            f"invalid btts zip bundle: {exc}",
            details={"bundle": str(path), "file": rel},
        ) from exc


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _reject_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON number: {value}")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_text(text: str) -> str:
    return _sha256_bytes(text.encode("utf-8"))


def _fail(
    message: str,
    bundle: Path | str,
    *,
    voice: Optional[str] = None,
    file: Optional[str] = None,
    **extra: Any,
) -> None:
    details: dict[str, Any] = {"bundle": str(bundle)}
    if voice is not None:
        details["voice"] = voice
    if file is not None:
        details["file"] = file
    details.update(extra)
    raise InputError(message, details=details)
