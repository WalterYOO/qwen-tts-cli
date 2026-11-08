"""`.qttsvoice` voice archive export/import (FR-P1-10).

Security contract (t10 pre-review):
- fixed member whitelist, no directory entries, no absolute/drive/backslash/
  ``..``/duplicate/casefold-colliding/encrypted/special members;
- no ``extractall``: every member is streamed to a staging directory while its
  sha256 is computed and compared with the manifest;
- safetensors is validated through the safe API, never unpickled;
- install is staging-dir + rename, with the old profile moved aside and
  restored on failure.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
import uuid
import zipfile
import zlib
from pathlib import Path, PurePosixPath
from typing import Any, Optional

from . import __version__
from .errors import InputError
from .utils import atomic_write_json, sha256_file
from .voice_store import SCHEMA_VERSION as VOICE_SCHEMA_VERSION
from .voice_store import VoiceStore, utc_now, validate_name

ARCHIVE_SUFFIX = ".qttsvoice"
ARCHIVE_SCHEMA_VERSION = 1
MANIFEST = "manifest.json"
ALLOWED_MEMBERS = (MANIFEST, "voice.json", "prompt.safetensors", "ref.wav", "preview.wav")
PROFILE_MEMBERS = frozenset(ALLOWED_MEMBERS) - {MANIFEST}
MAX_MEMBERS = 16
MAX_MEMBER_BYTES = 64 * 1024 * 1024
MAX_TOTAL_BYTES = 256 * 1024 * 1024
MAX_COMPRESSION_RATIO = 500.0
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _json_bytes(obj: Any) -> bytes:
    return (json.dumps(obj, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _entry(path: str, data: bytes) -> dict[str, Any]:
    return {"path": path, "sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}


def export_voice(
    store: VoiceStore,
    name: str,
    out: str | Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    meta = store.require(name)
    kind = str(meta.get("kind"))
    if kind == "finetune":
        raise InputError(
            "finetune profiles are machine-local; copy the checkpoint and re-register "
            "on the target machine",
            details={"voice": name, "kind": kind},
        )
    voice_dir = store.dir_for(name)
    members: dict[str, bytes] = {}

    members["voice.json"] = (voice_dir / "voice.json").read_bytes()
    if kind == "clone":
        prompt = voice_dir / "prompt.safetensors"
        if not prompt.is_file():
            raise InputError(
                f"voice {name} has no prompt.safetensors; export would be unusable",
                details={"voice": name, "path": str(prompt)},
            )
        members["prompt.safetensors"] = prompt.read_bytes()
    for optional in ("ref.wav", "preview.wav"):
        candidate = voice_dir / optional
        if candidate.is_file():
            members[optional] = candidate.read_bytes()

    files = [_entry(member, members[member]) for member in sorted(members)]
    manifest = {
        "schema_version": ARCHIVE_SCHEMA_VERSION,
        "qtts_version": __version__,
        "created_at": utc_now(),
        "name": name,
        "kind": kind,
        "consent": meta.get("consent"),
        "files": files,
    }

    out_path = Path(out).expanduser()
    if out_path.exists() and not force:
        raise InputError(
            f"archive already exists: {out_path}; pass --force to overwrite",
            details={"archive": str(out_path)},
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{out_path.name}.", suffix=".tmp",
                                    dir=str(out_path.parent))
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_DEFLATED,
                             compresslevel=6) as archive:
            archive.writestr(MANIFEST, _json_bytes(manifest))
            for member in sorted(members):
                archive.writestr(member, members[member])
        with open(tmp_path, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(tmp_path, out_path)
    except BaseException:
        try:
            tmp_path.unlink()
        except OSError:
            pass
        raise

    return {
        "ok": True,
        "archive": str(out_path),
        "voice": meta,
        "kind": kind,
        "files": files,
        "archive_sha256": sha256_file(out_path),
    }


def _validate_members(infos: list[zipfile.ZipInfo]) -> None:
    if not infos:
        raise InputError("empty .qttsvoice archive")
    if len(infos) > MAX_MEMBERS:
        raise InputError(f"archive has too many members: {len(infos)} > {MAX_MEMBERS}")
    seen: dict[str, str] = {}
    total = 0
    for info in infos:
        name = info.filename
        if not name or name != name.strip():
            raise InputError(f"invalid archive member name: {name!r}")
        pure = PurePosixPath(name)
        if name.startswith(("/", "\\")) or "\\" in name or ":" in pure.parts[0]:
            raise InputError(f"archive member uses an absolute or drive path: {name!r}")
        if any(part in ("", ".", "..") for part in pure.parts):
            raise InputError(f"archive member escapes the archive root: {name!r}")
        if name not in ALLOWED_MEMBERS:
            raise InputError(
                f"unexpected archive member: {name!r}; allowed: " + ", ".join(ALLOWED_MEMBERS)
            )
        folded = name.casefold()
        if folded in seen:
            raise InputError(
                f"duplicate/casefold-colliding archive member: {name!r} vs {seen[folded]!r}"
            )
        seen[folded] = name
        if info.is_dir():
            raise InputError(f"directory members are not allowed: {name!r}")
        if info.flag_bits & 0x1:
            raise InputError(f"encrypted archive member is not allowed: {name!r}")
        mode = info.external_attr >> 16
        file_type = stat.S_IFMT(mode)
        if file_type and file_type != stat.S_IFREG:
            raise InputError(f"non-regular archive member is not allowed: {name!r}")
        if info.file_size > MAX_MEMBER_BYTES:
            raise InputError(f"archive member too large: {name!r} ({info.file_size} bytes)")
        total += info.file_size
        if total > MAX_TOTAL_BYTES:
            raise InputError(f"archive expands beyond {MAX_TOTAL_BYTES} bytes")
        if info.file_size and info.compress_size == 0:
            raise InputError(f"suspicious compression for member: {name!r}")
        if info.compress_size and info.file_size / info.compress_size > MAX_COMPRESSION_RATIO:
            raise InputError(f"archive member compression ratio too high: {name!r}")
    if MANIFEST not in {info.filename for info in infos}:
        raise InputError("archive has no manifest.json")


def _read_member(archive: zipfile.ZipFile, name: str) -> bytes:
    info = archive.getinfo(name)
    if info.file_size > MAX_MEMBER_BYTES:
        raise InputError(f"archive member too large: {name!r}")
    with archive.open(info) as handle:
        data = handle.read(MAX_MEMBER_BYTES + 1)
    if len(data) != info.file_size:
        raise InputError(f"truncated archive member: {name!r}")
    return data


def _parse_manifest(raw: bytes) -> dict[str, Any]:
    try:
        manifest = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InputError(f"invalid manifest.json: {exc}") from exc
    if not isinstance(manifest, dict):
        raise InputError("manifest.json must be an object")
    if manifest.get("schema_version") != ARCHIVE_SCHEMA_VERSION:
        raise InputError(
            f"unsupported archive schema_version: {manifest.get('schema_version')!r}"
        )
    name = manifest.get("name")
    if not isinstance(name, str):
        raise InputError("manifest.name must be a string")
    validate_name(name)
    if manifest.get("kind") not in ("clone", "design"):
        raise InputError(
            "manifest.kind must be clone or design; finetune profiles are "
            "machine-local and cannot be imported"
        )
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise InputError("manifest.files must be a non-empty list")
    entries: dict[str, dict[str, Any]] = {}
    for entry in files:
        if not isinstance(entry, dict):
            raise InputError("manifest.files entries must be objects")
        member = entry.get("path")
        if not isinstance(member, str):
            raise InputError("manifest file paths must be strings")
        if member not in PROFILE_MEMBERS:
            raise InputError(f"manifest lists an unsupported profile file: {member!r}")
        if member in entries:
            raise InputError(f"manifest lists {member!r} more than once")
        digest = entry.get("sha256")
        if not isinstance(digest, str) or not _SHA256_RE.match(digest):
            raise InputError(f"manifest has an invalid sha256 for {member!r}")
        size = entry.get("size")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise InputError(f"manifest has an invalid size for {member!r}")
        entries[member] = entry
    if "voice.json" not in entries:
        raise InputError("manifest.files must include voice.json")
    if manifest["kind"] == "clone" and "prompt.safetensors" not in entries:
        raise InputError("clone archives must include prompt.safetensors")
    return manifest


def _validate_voice_meta(meta: Any, *, expected_name: str, expected_kind: str) -> dict[str, Any]:
    if not isinstance(meta, dict):
        raise InputError("voice.json must be an object")
    if meta.get("schema_version") != VOICE_SCHEMA_VERSION:
        raise InputError(f"unsupported voice schema_version: {meta.get('schema_version')!r}")
    if meta.get("name") != expected_name:
        raise InputError("voice.json name does not match manifest.name")
    if meta.get("kind") != expected_kind:
        raise InputError("voice.json kind does not match manifest.kind")
    if not isinstance(meta.get("source"), dict):
        raise InputError("voice.json source must be an object")
    if not isinstance(meta.get("model"), dict):
        raise InputError("voice.json model must be an object")
    consent = meta.get("consent")
    if consent is not None and not isinstance(consent, dict):
        raise InputError("voice.json consent must be an object")
    if expected_kind == "clone":
        source = meta["source"]
        mode = str(source.get("mode") or "icl").lower()
        if mode not in ("icl", "xvec"):
            raise InputError("clone voice source.mode must be icl or xvec")
        if mode == "icl" and not str(source.get("ref_text") or "").strip():
            raise InputError("ICL clone profile requires source.ref_text")
    return meta


def _validate_safetensors(path: Path) -> None:
    try:
        from safetensors import safe_open

        with safe_open(str(path), framework="numpy") as handle:
            keys = set(handle.keys())
    except ImportError:  # pragma: no cover - safetensors is a runtime dependency
        raise InputError("safetensors is unavailable; cannot validate prompt.safetensors")
    except Exception as exc:
        raise InputError(f"invalid prompt.safetensors: {exc}") from exc
    if "ref_spk_embedding" not in keys:
        raise InputError("prompt.safetensors is missing ref_spk_embedding")


def _validate_profile(directory: Path, meta: dict[str, Any], name: str) -> None:
    if meta.get("name") != name:
        raise InputError("imported voice.json name does not match the install name")
    if meta.get("kind") == "clone":
        prompt = directory / "prompt.safetensors"
        if not prompt.is_file():
            raise InputError("clone profile is missing prompt.safetensors")
        _validate_safetensors(prompt)
    for member in ("ref.wav", "preview.wav"):
        path = directory / member
        if not path.is_file():
            continue
        with open(path, "rb") as handle:
            header = handle.read(4)
        if header != b"RIFF":
            raise InputError(f"{member} is not a valid WAV file")


def _stream_member(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    destination: Path,
) -> tuple[str, int]:
    digest = hashlib.sha256()
    total = 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    with archive.open(info) as source, open(destination, "wb") as out:
        while True:
            chunk = source.read(1 << 20)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_MEMBER_BYTES:
                raise InputError(f"archive member too large while reading: {info.filename!r}")
            digest.update(chunk)
            out.write(chunk)
        out.flush()
        os.fsync(out.fileno())
    if total != info.file_size:
        raise InputError(f"truncated archive member: {info.filename!r}")
    return digest.hexdigest(), total


def _install_profile(store: VoiceStore, staging: Path, target_name: str, force: bool) -> bool:
    target = store.dir_for(target_name)
    if not target.exists():
        os.replace(staging, target)
        return False
    if not force:
        raise InputError(f"voice already exists: {target_name}; pass --force to replace")
    backup = target.with_name(f".{target.name}.backup.{uuid.uuid4().hex[:8]}")
    os.replace(target, backup)
    try:
        os.replace(staging, target)
    except BaseException:
        os.replace(backup, target)
        raise
    shutil.rmtree(backup, ignore_errors=True)
    return True


def import_voice(
    store: VoiceStore,
    archive_path: str | Path,
    *,
    name: Optional[str] = None,
    force: bool = False,
) -> dict[str, Any]:
    path = Path(archive_path).expanduser()
    if not path.is_file():
        raise InputError(f"archive not found: {path}", details={"archive": str(path)})
    replaced = False
    staging: Optional[Path] = None
    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            _validate_members(infos)
            manifest = _parse_manifest(_read_member(archive, MANIFEST))
            by_name = {info.filename: info for info in infos if info.filename != MANIFEST}
            listed = {entry["path"]: entry for entry in manifest["files"]}
            if set(by_name) != set(listed):
                raise InputError(
                    "archive members do not exactly match manifest.files",
                    details={"members": sorted(by_name), "manifest": sorted(listed)},
                )

            voice_raw = _read_member(archive, "voice.json")
            try:
                meta = json.loads(voice_raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise InputError(f"invalid voice.json: {exc}") from exc
            meta = _validate_voice_meta(
                meta, expected_name=manifest["name"], expected_kind=manifest["kind"]
            )

            target_name = validate_name(name or manifest["name"])
            target = store.dir_for(target_name)
            if target.exists() and not force:
                raise InputError(
                    f"voice already exists: {target_name}; pass --force to replace",
                    details={"voice": target_name},
                )

            staging = store.staging_dir(target_name)
            for member, entry in listed.items():
                info = by_name[member]
                digest, size = _stream_member(archive, info, staging / member)
                if digest != entry["sha256"]:
                    raise InputError(
                        f"sha256 mismatch for {member!r}",
                        details={"member": member, "expected": entry["sha256"], "actual": digest},
                    )
                if size != entry["size"]:
                    raise InputError(
                        f"size mismatch for {member!r}",
                        details={"member": member, "expected": entry["size"], "actual": size},
                    )

        new_meta = dict(meta)
        new_meta["name"] = target_name
        _validate_voice_meta(new_meta, expected_name=target_name, expected_kind=manifest["kind"])
        _validate_profile(staging, new_meta, target_name)
        atomic_write_json(staging / "voice.json", new_meta)
        _validate_profile(staging, new_meta, target_name)
        replaced = _install_profile(store, staging, target_name, force)
        staging = None
        return {
            "ok": True,
            "archive": str(path),
            "output": str(store.dir_for(target_name)),
            "voice": new_meta,
            "replaced": replaced,
            "files": sorted(listed),
        }
    except (zipfile.BadZipFile, zlib.error, EOFError, NotImplementedError) as exc:
        # Corrupt archives and CRC failures must be input errors (exit 4), not
        # tracebacks. The staging directory is removed by the finally block.
        raise InputError(f"invalid .qttsvoice archive: {exc}") from exc
    finally:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)
