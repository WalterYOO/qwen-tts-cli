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
from .utils import atomic_write_json, read_json, sha256_file
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

BUNDLE_FORMAT = "qtts-voice-bundle"
BUNDLE_SCHEMA_VERSION = 2
BUNDLE_MANIFEST = "bundle.json"
BUNDLE_MAX_MANIFEST_BYTES = 1024 * 1024
BUNDLE_MAX_VOICES = 256
BUNDLE_JOURNAL = ".bundle-import.json"
BUNDLE_EXPORT_JOURNAL = ".bundle-export.json"
_ARCHIVE_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}\.qttsvoice$")


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
    dry_run: bool = False,
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
    if voice_dir.is_symlink():
        raise InputError(
            f"voice directory is a symlink; refusing to export: {name}",
            details={"voice": name},
        )
    meta_path = voice_dir / "voice.json"
    if meta_path.is_symlink():
        raise InputError(
            f"voice.json is a symlink; refusing to export: {name}",
            details={"voice": name},
        )
    if kind == "clone":
        store.validate_reference(meta, voice_dir)
    members: dict[str, bytes] = {}

    members["voice.json"] = meta_path.read_bytes()
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
        if candidate.is_symlink():
            raise InputError(
                f"profile file is a symlink; refusing to export: {candidate.name}",
                details={"voice": name, "file": candidate.name},
            )
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
    try:
        out_path.resolve().relative_to(store.root.resolve())
    except ValueError:
        pass
    else:
        raise InputError(
            f"refusing to write an archive inside the voice store: {out_path}",
            details={"archive": str(out_path), "store": str(store.root)},
        )
    if out_path.exists() and not force:
        raise InputError(
            f"archive already exists: {out_path}; pass --force to overwrite",
            details={"archive": str(out_path)},
        )
    if dry_run:
        return {
            "ok": True,
            "dry_run": True,
            "archive": str(out_path),
            "voice": meta,
            "kind": kind,
            "files": files,
            "archive_sha256": None,
        }
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


def _read_archive_manifest(path: Path) -> dict[str, Any]:
    """Validate members and return the archive manifest without extracting."""
    try:
        with zipfile.ZipFile(path) as archive:
            _validate_members(archive.infolist())
            return _parse_manifest(_read_member(archive, MANIFEST))
    except (zipfile.BadZipFile, zlib.error, EOFError, NotImplementedError) as exc:
        raise InputError(f"invalid .qttsvoice archive: {exc}") from exc


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


def _stage_archive(
    store: VoiceStore,
    path: Path,
    target_name: str,
) -> tuple[Path, dict[str, Any], list[str]]:
    """Validate one archive and extract it into a staging directory.

    Nothing is published here; callers may stage several archives before making
    a single all-or-nothing install decision.
    """
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
        return staging, new_meta, sorted(listed)
    except (zipfile.BadZipFile, zlib.error, EOFError, NotImplementedError) as exc:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)
        raise InputError(f"invalid .qttsvoice archive: {exc}") from exc
    except BaseException:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)
        raise


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
    staging: Optional[Path] = None
    manifest = _read_archive_manifest(path)
    try:
        target_name = validate_name(name or manifest["name"])
        actual = store.existing_names().get(target_name.casefold())
        if actual is not None and actual != target_name:
            raise InputError(
                f"voice name collides case-insensitively with existing {actual!r}: "
                f"{target_name!r}"
            )
        if actual is not None and not force:
            raise InputError(
                f"voice already exists: {target_name}; pass --force to replace",
                details={"voice": target_name},
            )
        staging, new_meta, files = _stage_archive(store, path, target_name)
        replaced = _install_profile(store, staging, target_name, force)
        staging = None
        return {
            "ok": True,
            "archive": str(path),
            "output": str(store.dir_for(target_name)),
            "voice": new_meta,
            "replaced": replaced,
            "files": files,
        }
    finally:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)


# --------------------------------------------------------------------------- #
# Multi-voice bundles (schema 2, directory of single-voice archives)
# --------------------------------------------------------------------------- #
def _bundle_bytes(manifest: dict[str, Any]) -> bytes:
    return (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _read_bundle_manifest(directory: Path) -> dict[str, Any]:
    if directory.is_symlink():
        raise InputError(
            f"bundle directory is a symlink: {directory}",
            details={"bundle": str(directory)},
        )
    if not directory.is_dir():
        raise InputError(f"bundle directory not found: {directory}")
    path = directory / BUNDLE_MANIFEST
    if not path.is_file() or path.is_symlink():
        raise InputError(
            f"bundle has no regular {BUNDLE_MANIFEST}",
            details={"bundle": str(directory)},
        )
    try:
        with path.open("rb") as handle:
            raw = handle.read(BUNDLE_MAX_MANIFEST_BYTES + 1)
    except OSError as exc:
        raise InputError(f"cannot read {BUNDLE_MANIFEST}: {exc}") from exc
    if len(raw) > BUNDLE_MAX_MANIFEST_BYTES:
        raise InputError(f"{BUNDLE_MANIFEST} is too large")
    try:
        manifest = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InputError(f"invalid {BUNDLE_MANIFEST}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise InputError(f"{BUNDLE_MANIFEST} must be an object")
    if manifest.get("format") != BUNDLE_FORMAT:
        raise InputError(
            f"unsupported bundle format: {manifest.get('format')!r}; expected {BUNDLE_FORMAT}"
        )
    if manifest.get("bundle_schema_version") != BUNDLE_SCHEMA_VERSION:
        raise InputError(
            "unsupported bundle_schema_version: "
            f"{manifest.get('bundle_schema_version')!r}"
        )
    voices = manifest.get("voices")
    if not isinstance(voices, list) or not voices:
        raise InputError(f"{BUNDLE_MANIFEST}.voices must be a non-empty list")
    if len(voices) > BUNDLE_MAX_VOICES:
        raise InputError(f"bundle has too many voices: {len(voices)} > {BUNDLE_MAX_VOICES}")
    names: set[str] = set()
    archives: set[str] = set()
    for entry in voices:
        if not isinstance(entry, dict):
            raise InputError(f"{BUNDLE_MANIFEST}.voices entries must be objects")
        name = validate_name(entry.get("name"))
        archive = entry.get("archive")
        if not isinstance(archive, str) or not _ARCHIVE_NAME_RE.fullmatch(archive):
            raise InputError(f"invalid bundle archive name for {name!r}: {archive!r}")
        if entry.get("kind") not in ("clone", "design"):
            raise InputError(f"invalid bundle voice kind for {name!r}")
        digest = entry.get("sha256")
        if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
            raise InputError(f"invalid bundle sha256 for {name!r}")
        size = entry.get("size")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise InputError(f"invalid bundle size for {name!r}")
        folded = name.casefold()
        if folded in names:
            raise InputError(f"duplicate bundle voice name: {name!r}")
        names.add(folded)
        if archive.casefold() in archives:
            raise InputError(f"duplicate bundle archive: {archive!r}")
        archives.add(archive.casefold())
    return manifest


def _safe_bundle_child(directory: Path, name: Any, prefix: str) -> Optional[Path]:
    if (
        not isinstance(name, str)
        or not name.startswith(prefix)
        or "/" in name
        or "\\" in name
        or name in (".", "..")
    ):
        return None
    candidate = directory / name
    if candidate.is_symlink():
        return None
    try:
        candidate.resolve().relative_to(directory.resolve())
    except ValueError:
        return None
    return candidate


def _recover_export(directory: Path) -> None:
    path = directory / BUNDLE_EXPORT_JOURNAL
    if not path.is_file():
        return
    try:
        journal = read_json(path)
    except (OSError, ValueError):
        journal = None
    if not isinstance(journal, dict):
        try:
            path.unlink()
        except OSError:
            pass
        return
    backup = _safe_bundle_child(directory, journal.get("backup_dir"), ".bundle-backup-")
    staging = _safe_bundle_child(directory, journal.get("staging_dir"), ".bundle-stage-")
    committed = journal.get("state") == "committed"
    managed = bool(journal.get("managed"))
    backed_up = journal.get("backed_up") is True
    backed_up_files = set(journal.get("backed_up_files") or [])
    restoring_files = set(journal.get("restoring_files") or [])
    restored_files = set(journal.get("restored_files") or [])
    if not committed and managed and backed_up:
        valid = backup is not None and bool(backed_up_files)
        present: set[str] = set()
        if valid and backup.is_dir():
            present = {
                child.name
                for child in backup.iterdir()
                if child.is_file() and not child.is_symlink()
            }
        missing = backed_up_files - present
        valid = valid and all(
            (directory / name).is_file()
            and (name in restoring_files or name in restored_files)
            for name in missing
        )
        if not valid:
            raise InputError(
                "bundle export journal is inconsistent; the backup directory is "
                f"missing or incomplete. Inspect {directory} and delete {path} "
                "to discard the journal",
                details={"journal": str(path), "backup_dir": journal.get("backup_dir")},
            )
    if not committed:
        # Restore everything we moved into the backup directory first.  This
        # also covers a crash halfway through the backup phase, where only the
        # ``backed_up`` flag would be missing.
        restored: set[str] = set()
        if backup is not None and backup.is_dir():
            for child in sorted(backup.iterdir()):
                if not child.is_file() or child.is_symlink():
                    continue
                restoring_files.add(child.name)
                journal["restoring_files"] = sorted(restoring_files)
                atomic_write_json(path, journal)
                os.replace(child, directory / child.name)
                restoring_files.discard(child.name)
                restored_files.add(child.name)
                restored.add(child.name)
                journal["restoring_files"] = sorted(restoring_files)
                journal["restored_files"] = sorted(restored_files)
                atomic_write_json(path, journal)
        recovered = restored | restored_files | {
            name for name in restoring_files if (directory / name).is_file()
        }
        if managed and backed_up and not backed_up_files.issubset(recovered):
            raise InputError(
                "bundle export recovery could not restore every backed-up file; "
                f"inspect {directory} and delete {path} to discard the journal",
                details={"journal": str(path), "restored": sorted(restored)},
            )
        for archive in journal.get("archives") or []:
            if (
                not isinstance(archive, str)
                or "/" in archive
                or "\\" in archive
                or archive in backed_up_files
                or archive in restored
            ):
                continue
            destination = directory / archive
            # Only delete an archive we know was just published: either the
            # destination had no previous bundle, or the backup phase had
            # completed (so every old archive is in the backup directory).
            may_delete = (not managed) or (backup is not None and backed_up)
            if may_delete and destination.is_file() and not destination.is_symlink():
                destination.unlink()
        if not managed:
            new_manifest = directory / BUNDLE_MANIFEST
            if new_manifest.is_file() and not new_manifest.is_symlink():
                new_manifest.unlink()
    if backup is not None:
        shutil.rmtree(backup, ignore_errors=True)
    if staging is not None:
        shutil.rmtree(staging, ignore_errors=True)
    try:
        path.unlink()
    except OSError:
        pass


def _bundle_policy(directory: Path, *, force: bool) -> set[str]:
    """Validate an existing output directory; return names we may replace."""
    if not directory.exists():
        return set()
    if directory.is_symlink() or not directory.is_dir():
        raise InputError(f"bundle output is not a directory: {directory}")
    children = sorted(directory.iterdir())
    if not children:
        return set()
    manifest_path = directory / BUNDLE_MANIFEST
    if not manifest_path.is_file():
        raise InputError(
            f"bundle output directory is not empty: {directory}; use an empty directory"
        )
    previous = _read_bundle_manifest(directory)
    managed = {BUNDLE_MANIFEST} | {entry["archive"] for entry in previous["voices"]}
    internal = {BUNDLE_EXPORT_JOURNAL}
    internal.update(
        child.name
        for child in children
        if child.name.startswith((".bundle-stage-", ".bundle-backup-"))
    )
    extras = [
        child.name
        for child in children
        if child.name not in managed and child.name not in internal
    ]
    if extras:
        raise InputError(
            f"bundle output directory contains unmanaged files: {', '.join(extras)}"
        )
    if not force:
        existing = [entry["archive"] for entry in previous["voices"]
                    if (directory / entry["archive"]).exists()]
        if existing:
            raise InputError(
                f"bundle archives already exist; pass --force to replace: {', '.join(existing)}"
            )
    return managed


def export_bundle(
    store: VoiceStore,
    out_dir: str | Path,
    *,
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    directory = Path(out_dir).expanduser()
    try:
        directory.resolve().relative_to(store.root.resolve())
    except ValueError:
        pass
    else:
        raise InputError(
            f"refusing to write a bundle inside the voice store: {directory}",
            details={"bundle": str(directory), "store": str(store.root)},
        )
    if directory.is_symlink():
        raise InputError(f"bundle output is a symlink: {directory}")
    if not dry_run and directory.is_dir():
        _recover_export(directory)
    managed = _bundle_policy(directory, force=force)
    scan = store.scan_entries()
    warnings = list(scan.get("warnings") or [])
    selected: list[dict[str, Any]] = []
    for entry in scan.get("voices") or []:
        name = entry.get("name")
        meta = entry.get("meta") or {}
        if entry.get("status") != "ok" or not meta:
            warnings.append(f"skipped voice {name}: {entry.get('status')}")
            continue
        kind = str(meta.get("kind"))
        if kind == "finetune":
            warnings.append(f"skipped finetune voice {name}: machine-local")
            continue
        selected.append({"name": name, "kind": kind, "meta": meta})
    if not selected:
        raise InputError("no exportable user voices")
    folded: set[str] = set()
    for entry in selected:
        key = str(entry["name"]).casefold()
        if key in folded:
            raise InputError(
                f"voice names collide case-insensitively: {entry['name']!r}; "
                "rename one of them before exporting a bundle"
            )
        folded.add(key)
        entry["archive"] = f"{entry['name']}{ARCHIVE_SUFFIX}"
    if len(selected) > BUNDLE_MAX_VOICES:
        raise InputError(f"too many voices to export: {len(selected)}")

    if dry_run:
        return {
            "ok": True,
            "dry_run": True,
            "bundle": str(directory / BUNDLE_MANIFEST),
            "out_dir": str(directory),
            "voices": [
                {"name": entry["name"], "kind": entry["kind"],
                 "archive": entry["archive"]}
                for entry in selected
            ],
            "sha256": None,
            "warnings": warnings,
        }

    directory.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    staging = directory / f".bundle-stage-{token}"
    backup = directory / f".bundle-backup-{token}"
    journal = {
        "state": "publishing",
        "managed": bool(managed),
        "backed_up": False,
        "staging_dir": staging.name,
        "backup_dir": backup.name if managed else None,
        "archives": [entry["archive"] for entry in selected],
        "restoring_files": [],
        "restored_files": [],
    }
    try:
        # Write the journal before creating staging so a crash at any point
        # leaves recovery enough information to clean up.
        atomic_write_json(directory / BUNDLE_EXPORT_JOURNAL, journal)
        staging.mkdir()
        for entry in selected:
            archive = staging / entry["archive"]
            export_voice(store, entry["name"], archive, force=True)
            entry["sha256"] = sha256_file(archive)
            entry["size"] = archive.stat().st_size
        manifest = {
            "format": BUNDLE_FORMAT,
            "bundle_schema_version": BUNDLE_SCHEMA_VERSION,
            "qtts_version": __version__,
            "created_at": utc_now(),
            "voices": [
                {
                    "name": entry["name"],
                    "kind": entry["kind"],
                    "archive": entry["archive"],
                    "sha256": entry["sha256"],
                    "size": entry["size"],
                }
                for entry in selected
            ],
        }
        if managed:
            backup.mkdir()
            for child in sorted(directory.iterdir()):
                if child.name in managed:
                    os.replace(child, backup / child.name)
        journal["backed_up"] = True
        journal["backed_up_files"] = sorted(
            child.name for child in backup.iterdir()
        ) if managed else []
        atomic_write_json(directory / BUNDLE_EXPORT_JOURNAL, journal)
        for entry in selected:
            os.replace(staging / entry["archive"], directory / entry["archive"])
        atomic_write_json(directory / BUNDLE_MANIFEST, manifest)
        journal["state"] = "committed"
        atomic_write_json(directory / BUNDLE_EXPORT_JOURNAL, journal)
    except BaseException:
        # The journal is on disk from the first step; reuse the same crash
        # recovery path instead of keeping a second, divergent rollback.
        _recover_export(directory)
        raise
    shutil.rmtree(staging, ignore_errors=True)
    shutil.rmtree(backup, ignore_errors=True)
    try:
        (directory / BUNDLE_EXPORT_JOURNAL).unlink()
    except OSError:
        pass
    return {
        "ok": True,
        "dry_run": False,
        "bundle": str(directory / BUNDLE_MANIFEST),
        "out_dir": str(directory),
        "voices": [
            {"name": entry["name"], "kind": entry["kind"],
             "archive": entry["archive"], "sha256": entry["sha256"],
             "size": entry["size"]}
            for entry in selected
        ],
        "sha256": sha256_file(directory / BUNDLE_MANIFEST),
        "warnings": warnings,
    }


def _recover_bundle_journal(store: VoiceStore) -> None:
    path = store.root / BUNDLE_JOURNAL
    if not path.is_file():
        return
    try:
        journal = read_json(path)
    except (OSError, ValueError):
        journal = None
    if not isinstance(journal, dict):
        try:
            path.unlink()
        except OSError:
            pass
        return
    raw_backup = journal.get("backup_dir")
    backup_dir = _safe_bundle_child(store.root, raw_backup, ".bundle-backup-")
    committed = journal.get("state") == "committed"
    raw_entries = journal.get("entries")
    if not committed and not isinstance(raw_entries, list):
        raise InputError(
            "bundle import journal is corrupt; inspect the store and delete "
            f"{path} to discard the journal",
            details={"journal": str(path), "backup_dir": raw_backup},
        )
    entries: list[dict[str, Any]] = []
    if not committed:
        for item in raw_entries:
            if not isinstance(item, dict) or not isinstance(item.get("target"), str):
                raise InputError(
                    "bundle import journal has a malformed entry; inspect the "
                    f"store and delete {path} to discard the journal",
                    details={"journal": str(path), "entry": item},
                )
            entries.append(item)
        try:
            for entry in entries:
                try:
                    target = store.dir_for(entry["target"])
                except InputError as exc:
                    if entry.get("installed"):
                        raise InputError(
                            "bundle import journal targets an invalid voice name; "
                            f"backup and journal are preserved: {path}",
                            details={"journal": str(path), "target": entry["target"]},
                        ) from exc
                    continue
                if entry.get("restored"):
                    continue
                backup = backup_dir / target.name if backup_dir is not None else None
                if backup is not None and backup.exists():
                    if target.exists():
                        shutil.rmtree(target)
                    # Write ``restoring`` before consuming the backup so a
                    # crash between the replace and the next journal write can
                    # be recognized as "the old profile is already in place".
                    entry["restoring"] = True
                    atomic_write_json(path, journal)
                    os.replace(backup, target)
                    entry["installed"] = False
                    entry["restoring"] = False
                    entry["restored"] = True
                    atomic_write_json(path, journal)
                    continue
                if entry.get("restoring"):
                    if not target.exists():
                        raise InputError(
                            "bundle import journal needs manual recovery: the "
                            f"restore target is missing: {target}",
                            details={"journal": str(path), "target": target.name},
                        )
                    entry["installed"] = False
                    entry["restoring"] = False
                    entry["restored"] = True
                    atomic_write_json(path, journal)
                    continue
                if not entry.get("installed"):
                    continue
                if entry.get("replaced", True):
                    # No backup, no restore marker: we cannot tell whether the
                    # target is the old or the new profile. Fail closed.
                    raise InputError(
                        "bundle import journal needs manual recovery; the backup "
                        f"is missing for replaced voice {target.name}. Inspect "
                        f"the store and delete {path} to discard the journal",
                        details={"journal": str(path), "target": target.name},
                    )
                if target.exists():
                    # Newly installed voice with no previous profile: roll back.
                    shutil.rmtree(target)
                entry["installed"] = False
                entry["restored"] = True
                atomic_write_json(path, journal)
        except OSError as exc:
            raise InputError(
                "bundle import journal recovery failed; backup and journal are "
                f"preserved for manual recovery: {path}",
                details={"journal": str(path), "backup_dir": raw_backup},
            ) from exc
    if backup_dir is not None and backup_dir.exists():
        try:
            shutil.rmtree(backup_dir)
        except OSError as exc:
            raise InputError(
                "bundle import journal backup could not be removed; journal is "
                f"preserved for manual recovery: {path}",
                details={"journal": str(path), "backup_dir": raw_backup},
            ) from exc
    try:
        path.unlink()
    except OSError:
        pass


def import_bundle(
    store: VoiceStore,
    bundle_dir: str | Path,
    *,
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    directory = Path(bundle_dir).expanduser()
    if not dry_run:
        _recover_bundle_journal(store)
    manifest = _read_bundle_manifest(directory)
    warnings: list[str] = []
    entries = manifest["voices"]
    listed = {entry["archive"] for entry in entries}
    for child in sorted(directory.iterdir()):
        if child.name.endswith(ARCHIVE_SUFFIX) and child.name not in listed:
            warnings.append(f"ignored archive not listed in {BUNDLE_MANIFEST}: {child.name}")

    existing = store.existing_names()
    targets: list[dict[str, Any]] = []
    for entry in entries:
        name = entry["name"]
        archive = directory / entry["archive"]
        if archive.is_symlink():
            raise InputError(f"bundle archive is a symlink: {archive}")
        if not archive.is_file():
            raise InputError(f"bundle archive not found: {archive}")
        if archive.stat().st_size != entry["size"]:
            raise InputError(f"bundle archive size mismatch: {entry['archive']}")
        if sha256_file(archive) != entry["sha256"]:
            raise InputError(f"bundle archive sha256 mismatch: {entry['archive']}")
        archive_manifest = _read_archive_manifest(archive)
        if archive_manifest["name"] != name:
            raise InputError(
                "bundle voice name does not match its archive manifest: "
                f"{name!r} != {archive_manifest['name']!r}"
            )
        if archive_manifest["kind"] != entry["kind"]:
            raise InputError(
                "bundle voice kind does not match its archive manifest: "
                f"{name!r} ({entry['kind']!r} != {archive_manifest['kind']!r})"
            )
        actual = existing.get(name.casefold())
        if actual is not None and actual != name:
            raise InputError(
                f"voice name collides case-insensitively with existing {actual!r}: {name!r}"
            )
        replaced = actual is not None
        if replaced and not force:
            raise InputError(f"voice already exists: {name}; pass --force to replace")
        targets.append({"entry": entry, "name": name, "archive": archive,
                        "replaced": replaced})
    if dry_run:
        return {
            "ok": True,
            "dry_run": True,
            "bundle_dir": str(directory),
            "voices": [
                {"name": item["name"], "replaced": item["replaced"]}
                for item in targets
            ],
            "warnings": warnings,
        }

    staged: list[tuple[str, Path, dict[str, Any], list[str]]] = []
    try:
        for item in targets:
            staging_path, meta, files = _stage_archive(store, item["archive"], item["name"])
            staged.append((item["name"], staging_path, meta, files))
    except BaseException:
        for _, staging_path, _, _ in staged:
            shutil.rmtree(staging_path, ignore_errors=True)
        raise

    token = uuid.uuid4().hex
    backup_dir = store.root / f".bundle-backup-{token}"
    backup_dir.mkdir(exist_ok=False)
    journal_entries = [
        {
            "target": name,
            "installed": False,
            "replaced": store.dir_for(name).exists(),
        }
        for name, _, _, _ in staged
    ]
    journal = {
        "backup_dir": backup_dir.name,
        "entries": journal_entries,
    }
    atomic_write_json(store.root / BUNDLE_JOURNAL, journal)
    try:
        for index, (name, staging_path, _meta, _files) in enumerate(staged):
            target = store.dir_for(name)
            if target.exists():
                os.replace(target, backup_dir / name)
            journal_entries[index]["installed"] = True
            atomic_write_json(store.root / BUNDLE_JOURNAL, journal)
            os.replace(staging_path, target)
    except BaseException:
        # The on-disk journal is authoritative; run the same idempotent
        # recovery used by the next invocation.
        try:
            _recover_bundle_journal(store)
        finally:
            for _, staging_path, _, _ in staged:
                shutil.rmtree(staging_path, ignore_errors=True)
        raise
    journal["state"] = "committed"
    atomic_write_json(store.root / BUNDLE_JOURNAL, journal)
    shutil.rmtree(backup_dir, ignore_errors=True)
    try:
        (store.root / BUNDLE_JOURNAL).unlink()
    except OSError:
        pass
    return {
        "ok": True,
        "bundle_dir": str(directory),
        "voices": [
            {"name": item["name"], "replaced": item["replaced"],
             "files": files}
            for item, (_, _, _, files) in zip(targets, staged)
        ],
        "warnings": warnings,
    }
