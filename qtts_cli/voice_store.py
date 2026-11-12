"""Voice asset store (FR-P0-02, PRD §8.1).

Layout::

    <data-dir>/voices/<name>/
      voice.json          metadata (no binary payloads)
      prompt.safetensors  clone prompt tensors (never pickle)
      ref.wav             reference audio
      preview.wav         fixed audition line
"""

from __future__ import annotations

import os
import re
import secrets
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from .errors import InputError, UsageError
from .utils import atomic_write_json, read_json, sha256_file

SCHEMA_VERSION = 1
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_RESERVED = {"builtin", "design", "voice"}
TAG_MAX_LENGTH = 64
TOMBSTONE_PREFIX = ".deleting-"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def validate_name(name: str) -> str:
    if not name or not _NAME_RE.match(name):
        raise InputError(
            f"invalid voice name: {name!r}; use letters, digits, '_', '-', '.' (max 64, must not start with '.')",
            details={"name": name},
        )
    if name.lower() in _RESERVED:
        raise InputError(f"voice name {name!r} is reserved")
    return name


def normalize_tag(tag: str) -> str:
    """Trim one tag and reject values that cannot be stored safely."""
    value = str(tag).strip()
    if not value:
        raise UsageError("tag must not be empty")
    if len(value) > TAG_MAX_LENGTH:
        raise UsageError(f"tag must be at most {TAG_MAX_LENGTH} characters")
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise UsageError(f"tag contains control characters: {value!r}")
    return value


def normalize_tags(tags: Iterable[str]) -> list[str]:
    """Normalize, trim and case-insensitively de-duplicate while keeping order."""
    seen: set[str] = set()
    out: list[str] = []
    for tag in tags:
        value = normalize_tag(tag)
        key = value.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out


def _require_real_directory(path: Path, *, what: str) -> None:
    """Refuse symlinks for destructive operations."""
    if path.is_symlink():
        raise InputError(
            f"{what} is a symlink; refusing to touch it",
            details={"path": str(path)},
        )


class VoiceStore:
    def __init__(self, data_dir: str | Path) -> None:
        self.root = Path(data_dir).expanduser() / "voices"

    # -- paths -------------------------------------------------------------- #
    def dir_for(self, name: str) -> Path:
        return self.root / validate_name(name)

    def exists(self, name: str) -> bool:
        return (self.dir_for(name) / "voice.json").is_file()

    def existing_names(self) -> dict[str, str]:
        """Map casefolded names to on-disk directory names.

        Import conflict checks use this instead of ``Path.exists`` so they do
        not depend on whether the filesystem is case-sensitive. Hidden staging
        and backup directories are ignored.
        """
        if not self.root.is_dir():
            return {}
        found: dict[str, str] = {}
        for child in sorted(self.root.iterdir()):
            if child.name.startswith(".") or not child.is_dir():
                continue
            found.setdefault(child.name.casefold(), child.name)
        return found

    def require(self, name: str) -> dict[str, Any]:
        meta = self.get(name)
        if meta is None:
            raise InputError(
                f"voice not found: {name}",
                details={"voice": name, "store": str(self.root)},
            )
        return meta

    # -- read --------------------------------------------------------------- #
    def get(self, name: str) -> Optional[dict[str, Any]]:
        path = self.dir_for(name) / "voice.json"
        if not path.is_file():
            return None
        try:
            meta = read_json(path)
        except (OSError, ValueError) as exc:
            raise InputError(f"corrupt voice metadata: {path}: {exc}") from exc
        if not isinstance(meta, dict) or meta.get("name") != name:
            raise InputError(f"invalid voice metadata: {path}")
        return meta

    def list(self) -> list[dict[str, Any]]:
        if not self.root.is_dir():
            return []
        out: list[dict[str, Any]] = []
        for child in sorted(self.root.iterdir()):
            if not child.is_dir():
                continue
            try:
                meta = self.get(child.name)
            except InputError:
                continue
            if meta is not None:
                out.append(meta)
        return out

    def scan_entries(self) -> dict[str, Any]:
        """Inventory every profile directory without raising on bad entries.

        ``status`` is one of ``ok``, ``mismatch``, ``corrupt`` or ``symlink``.
        Tombstones and hidden staging directories are skipped and reported as
        warnings so ``voice list`` can stay usable after a crashed delete.
        """
        voices: list[dict[str, Any]] = []
        warnings: list[str] = []
        if not self.root.is_dir():
            return {"voices": voices, "warnings": warnings}
        for child in sorted(self.root.iterdir()):
            if child.name.startswith(TOMBSTONE_PREFIX):
                warnings.append(f"skipped tombstone: {child.name}")
                continue
            if child.name.startswith(".rename-"):
                warnings.append(
                    f"incomplete rename: {child.name}; re-run the voice rename"
                )
                continue
            if child.name.startswith("."):
                continue
            if child.is_symlink():
                warnings.append(f"skipped symlink voice directory: {child.name}")
                voices.append(
                    {"name": child.name, "meta": None, "status": "symlink",
                     "warnings": ["voice directory is a symlink"]}
                )
                continue
            if not child.is_dir():
                continue
            meta_path = child / "voice.json"
            if not meta_path.is_file():
                warnings.append(f"skipped corrupt voice {child.name}: missing voice.json")
                voices.append(
                    {"name": child.name, "meta": None, "status": "corrupt",
                     "warnings": ["missing voice.json"]}
                )
                continue
            try:
                meta = read_json(meta_path)
            except (OSError, ValueError) as exc:
                warnings.append(f"skipped corrupt voice {child.name}: {exc}")
                voices.append(
                    {"name": child.name, "meta": None, "status": "corrupt",
                     "warnings": [str(exc)]}
                )
                continue
            if not isinstance(meta, dict):
                warnings.append(f"skipped corrupt voice {child.name}: not an object")
                voices.append(
                    {"name": child.name, "meta": None, "status": "corrupt",
                     "warnings": ["voice.json is not an object"]}
                )
                continue
            stored = meta.get("name")
            if not isinstance(stored, str) or not stored:
                warnings.append(
                    f"skipped corrupt voice {child.name}: name is missing or invalid"
                )
                voices.append(
                    {"name": child.name, "meta": meta, "status": "corrupt",
                     "warnings": ["name is missing or invalid"]}
                )
                continue
            if stored != child.name:
                # Rename from the on-disk directory name to the recorded name.
                hint = (
                    f"directory {child.name!r} contains name {stored!r}; "
                    f"run: qtts voice rename {child.name} {stored}"
                )
                warnings.append(hint)
                voices.append(
                    {"name": child.name, "meta": meta, "status": "mismatch",
                     "warnings": [hint]}
                )
                continue
            voices.append({"name": child.name, "meta": meta, "status": "ok", "warnings": []})
        return {"voices": voices, "warnings": warnings}

    def resolve_dir(self, name: str) -> Optional[Path]:
        """Resolve a profile directory by exact name, then case-insensitively.

        The returned path carries the on-disk spelling, not the caller's, so
        metadata checks stay correct on case-insensitive filesystems.
        """
        validate_name(name)
        if not self.root.is_dir():
            return None
        children = [child for child in self.root.iterdir() if not child.name.startswith(".")]
        for child in children:
            if child.name == name:
                return child
        for child in children:
            if child.name.casefold() == name.casefold():
                return child
        return None

    # -- write -------------------------------------------------------------- #
    def staging_dir(self, name: str) -> Path:
        """A temp directory beside the final profile for an atomic commit (P1-2)."""
        name = validate_name(name)
        self.root.mkdir(parents=True, exist_ok=True)
        return Path(tempfile.mkdtemp(prefix=f".{name}.", dir=str(self.root)))

    def commit_staging(self, staging: str | Path, name: str) -> Path:
        """Atomically publish a fully written staging directory."""
        name = validate_name(name)
        target = self.dir_for(name)
        if target.exists():
            raise InputError(f"voice already exists: {name}")
        os.replace(Path(staging), target)
        return target

    def save_meta(self, meta: dict[str, Any], directory: Optional[Path] = None) -> Path:
        name = validate_name(str(meta.get("name", "")))
        base = Path(directory) if directory is not None else self.dir_for(name)
        path = base / "voice.json"
        atomic_write_json(path, meta)
        return path

    def create(
        self,
        *,
        name: str,
        kind: str,
        source: dict[str, Any],
        model: dict[str, Any],
        tokenizer: Optional[dict[str, Any]] = None,
        dtype: Optional[str] = None,
        language: Optional[str] = None,
        default_instruct: Optional[str] = None,
        tags: Optional[list[str]] = None,
        consent: Optional[dict[str, Any]] = None,
        qtts_version: str = "0.1.0",
        staging: Optional[Path] = None,
        allow_existing: bool = False,
    ) -> dict[str, Any]:
        name = validate_name(name)
        if self.exists(name) and not allow_existing:
            raise InputError(
                f"voice already exists: {name}; remove it first or choose another name",
                details={"voice": name},
            )
        meta = {
            "schema_version": SCHEMA_VERSION,
            "name": name,
            "kind": kind,
            "created_at": utc_now(),
            "source": source,
            "model": model,
            "tokenizer": tokenizer or {"type": "12hz", "codec_rate_hz": 12.5},
            "dtype": dtype,
            "language": language,
            "default_instruct": default_instruct,
            "tags": list(tags or []),
            "consent": consent or {"rights_confirmed": False, "license": None, "note": ""},
            "qtts_version": qtts_version,
        }
        base = Path(staging) if staging is not None else self.dir_for(name)
        base.mkdir(parents=True, exist_ok=True)
        self.save_meta(meta, directory=base)
        return meta

    def copy_ref_audio(self, name: str, src: str | Path, directory: Optional[Path] = None) -> Path:
        base = Path(directory) if directory is not None else self.dir_for(name)
        target = base / "ref.wav"
        _atomic_copy(Path(src).expanduser(), target)
        return target

    def save_prompt_tensors(
        self,
        name: str,
        *,
        ref_spk_embedding: Any,
        ref_code: Any = None,
        x_vector_only_mode: bool = False,
        icl_mode: bool = True,
        directory: Optional[Path] = None,
    ) -> Path:
        """Persist clone prompt tensors with safetensors (no pickle)."""
        import torch
        from safetensors.torch import save_file

        tensors: dict[str, Any] = {
            "ref_spk_embedding": _to_cpu_contiguous(ref_spk_embedding),
            "flags": torch.tensor(
                [1.0 if x_vector_only_mode else 0.0, 1.0 if icl_mode else 0.0], dtype=torch.float32
            ),
        }
        if ref_code is not None:
            tensors["ref_code"] = _to_cpu_contiguous(ref_code)
        base = Path(directory) if directory is not None else self.dir_for(name)
        base.mkdir(parents=True, exist_ok=True)
        path = base / "prompt.safetensors"
        # safetensors writes in place; stage a unique temp then replace so a
        # concurrent reader never sees a partial file (codex P1-8).
        fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(base))
        os.close(fd)
        try:
            save_file(tensors, tmp)
            with open(tmp, "rb") as fh:
                os.fsync(fh.fileno())
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        return path

    def load_prompt_tensors(self, name: str) -> dict[str, Any]:
        from safetensors.torch import load_file

        path = self.dir_for(name) / "prompt.safetensors"
        if not path.is_file():
            raise InputError(
                f"voice {name} has no prompt.safetensors; re-add the voice or run --materialize-clone",
                details={"voice": name, "path": str(path)},
            )
        tensors = load_file(str(path))
        flags = tensors.pop("flags", None)
        if flags is not None:
            tensors["x_vector_only_mode"] = bool(flags[0].item())
            tensors["icl_mode"] = bool(flags[1].item())
        else:
            tensors["x_vector_only_mode"] = False
            tensors["icl_mode"] = True
        return tensors

    def set_preview(self, name: str, src: str | Path, directory: Optional[Path] = None) -> Path:
        base = Path(directory) if directory is not None else self.dir_for(name)
        target = base / "preview.wav"
        _atomic_copy(Path(src).expanduser(), target)
        return target

    def prompt_sha256(self, name: str) -> Optional[str]:
        path = self.dir_for(name) / "prompt.safetensors"
        return sha256_file(path) if path.is_file() else None

    def rename(self, old: str, new: str, *, dry_run: bool = False) -> dict[str, Any]:
        """Rename a profile directory and its ``voice.json`` name atomically.

        Crash recovery: if a previous run already wrote ``name=new`` but died
        before moving the directory, re-running ``rename old new`` finishes
        the move instead of failing the metadata check.
        """
        validate_name(old)
        validate_name(new)
        recovered = self._recover_renames()
        target = self.root / new
        source = self.resolve_dir(old)
        if source is None or not source.is_dir():
            # Only a journal left by *this* rename proves the move completed;
            # an unrelated profile that happens to be called `new` must not
            # turn a typo into exit 0.
            completed = [
                info
                for info in recovered
                if info.get("old") == old and info.get("new") == new
            ]
            if completed and self._is_renamed_target(target, new):
                self._consume_pending_rename(old, new)
                return {
                    "old": old,
                    "new": new,
                    "path": str(target),
                    "dry_run": False,
                    "recovered": True,
                }
            raise UsageError(f"voice not found: {old}", details={"voice": old})
        _require_real_directory(source, what="voice directory")
        case_only = source.name.casefold() == new.casefold()
        if target.exists() or target.is_symlink():
            same_directory = (
                case_only
                and target.exists()
                and not target.is_symlink()
                and target.samefile(source)
            )
            if not same_directory:
                raise UsageError(
                    f"voice already exists: {new}", details={"voice": new}
                )

        meta_path = source / "voice.json"
        if not meta_path.is_file():
            raise InputError(
                f"cannot rename {source.name}: missing voice.json",
                details={"voice": source.name},
            )
        if meta_path.is_symlink():
            raise InputError(
                f"cannot rename {source.name}: voice.json is a symlink",
                details={"voice": source.name},
            )
        try:
            meta = read_json(meta_path)
        except (OSError, ValueError) as exc:
            raise InputError(
                f"cannot rename {source.name}: corrupt voice.json: {exc}",
                details={"voice": source.name},
            ) from exc
        if not isinstance(meta, dict) or meta.get("name") not in (source.name, new):
            raise InputError(
                f"cannot rename {source.name}: voice.json name is {meta.get('name')!r}",
                details={"voice": source.name, "name": meta.get("name") if isinstance(meta, dict) else None},
            )

        payload = {
            "old": source.name,
            "new": new,
            "path": str(target),
            "dry_run": bool(dry_run),
        }
        if dry_run:
            return payload
        if source.name == new:
            if meta.get("name") != new:
                meta["name"] = new
                atomic_write_json(meta_path, meta)
            payload["path"] = str(source)
            return payload

        journal = self._create_rename_journal(source.name, new)
        try:
            if meta.get("name") != new:
                meta["name"] = new
                atomic_write_json(meta_path, meta)
            if case_only:
                _replace_case_only(source, target, journal)
            else:
                os.replace(source, target)
        except OSError as exc:
            raise InputError(
                f"cannot rename {source.name} to {new}: {exc}",
                details={"voice": source.name, "new": new},
            ) from exc
        shutil.rmtree(journal, ignore_errors=True)
        payload["path"] = str(target)
        return payload

    def _is_renamed_target(self, target: Path, name: str) -> bool:
        if target.is_symlink() or not target.is_dir():
            return False
        meta_path = target / "voice.json"
        if meta_path.is_symlink() or not meta_path.is_file():
            return False
        try:
            meta = read_json(meta_path)
        except (OSError, ValueError):
            return False
        return isinstance(meta, dict) and meta.get("name") == name

    def _create_rename_journal(self, old: str, new: str) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        journal = self.root / f".rename-{secrets.token_hex(8)}"
        journal.mkdir()
        atomic_write_json(journal / "rename.json", {"old": old, "new": new})
        return journal

    def _iter_rename_journals(self):
        if not self.root.is_dir():
            return
        for child in sorted(self.root.iterdir()):
            if not child.name.startswith(".rename-") or child.is_symlink():
                continue
            if not child.is_dir():
                continue
            try:
                info = read_json(child / "rename.json")
            except (OSError, ValueError):
                continue
            if not isinstance(info, dict):
                continue
            old, new = info.get("old"), info.get("new")
            if not isinstance(old, str) or not isinstance(new, str):
                continue
            try:
                validate_name(new)
            except InputError:
                continue
            yield child, {"old": old, "new": new}

    def _has_pending_rename(self, old: str, new: str) -> bool:
        return any(
            info["old"] == old and info["new"] == new
            for _, info in self._iter_rename_journals()
        )

    def _consume_pending_rename(self, old: str, new: str) -> None:
        for journal, info in self._iter_rename_journals():
            if info["old"] == old and info["new"] == new:
                shutil.rmtree(journal, ignore_errors=True)

    def _recover_renames(self) -> list[dict[str, Any]]:
        """Finish interrupted renames recorded in ``.rename-*`` journals."""
        recovered: list[dict[str, Any]] = []
        for journal, info in self._iter_rename_journals():
            new = info["new"]
            target = self.root / new
            profile = journal / "profile"
            if (
                profile.is_dir()
                and not profile.is_symlink()
                and not (target.exists() or target.is_symlink())
            ):
                os.replace(profile, target)
            if not profile.exists() and self._is_renamed_target(target, new):
                shutil.rmtree(journal, ignore_errors=True)
                recovered.append(info)
        return recovered

    def set_tags(
        self,
        name: str,
        *,
        add: Iterable[str] = (),
        remove: Iterable[str] = (),
        replace: Optional[Iterable[str]] = None,
        clear: bool = False,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Edit tags atomically. With no operations, only reads the profile."""
        additions = list(add)
        removals = list(remove)
        if replace is not None and (additions or removals):
            raise UsageError("--set cannot be combined with --add or --remove")
        directory, meta = self._writable_profile(name)
        before = normalize_tags(meta.get("tags") or [])
        if replace is None and not additions and not removals and not clear:
            return {"voice": name, "before": before, "after": before,
                    "changed": False, "tags": before, "dry_run": bool(dry_run)}

        if replace is not None:
            after = normalize_tags(replace)
        else:
            after = [] if clear else list(before)
            dropped = {normalize_tag(tag).casefold() for tag in removals}
            after = [tag for tag in after if tag.casefold() not in dropped]
            present = {tag.casefold() for tag in after}
            for tag in additions:
                value = normalize_tag(tag)
                if value.casefold() not in present:
                    present.add(value.casefold())
                    after.append(value)

        changed = after != before
        if changed:
            # Dry runs must surface a broken clone too, otherwise "would set"
            # is a promise the real run cannot keep.
            self._check_reference(meta, directory)
            if not dry_run:
                meta["tags"] = after
                self.save_meta(meta, directory=directory)
        return {"voice": name, "before": before, "after": after,
                "changed": changed, "tags": after, "dry_run": bool(dry_run)}

    def _writable_profile(self, name: str) -> tuple[Path, dict[str, Any]]:
        """Resolve a real (non-symlink) profile directory and its metadata."""
        directory = self.resolve_dir(name)
        if directory is None:
            raise InputError(f"voice not found: {name}", details={"voice": name})
        _require_real_directory(directory, what="voice directory")
        meta_path = directory / "voice.json"
        if meta_path.is_symlink():
            raise InputError(
                f"voice {directory.name} has a symlinked voice.json; refusing to write",
                details={"voice": directory.name},
            )
        if not meta_path.is_file():
            raise InputError(
                f"voice {directory.name} has no voice.json",
                details={"voice": directory.name},
            )
        try:
            meta = read_json(meta_path)
        except (OSError, ValueError) as exc:
            raise InputError(
                f"corrupt voice metadata: {meta_path}: {exc}",
                details={"voice": directory.name},
            ) from exc
        if not isinstance(meta, dict):
            raise InputError(
                f"invalid voice metadata: {meta_path}",
                details={"voice": directory.name},
            )
        if meta.get("name") not in (directory.name, name):
            raise InputError(
                f"voice directory {directory.name!r} contains name {meta.get('name')!r}",
                details={"voice": directory.name},
            )
        return directory, meta

    def delete(self, name: str, *, dry_run: bool = False) -> dict[str, Any]:
        """Remove a profile via a tombstone rename, refusing symlinks."""
        validate_name(name)
        path = self.root / name
        if path.is_symlink():
            raise InputError(
                f"voice directory is a symlink; refusing to delete: {name}",
                details={"voice": name},
            )
        if not path.is_dir():
            raise InputError(f"voice not found: {name}", details={"voice": name})
        _require_real_directory(path, what="voice directory")
        meta_path = path / "voice.json"
        if meta_path.is_symlink():
            raise InputError(
                f"voice.json is a symlink; refusing to delete: {name}",
                details={"voice": name},
            )
        tombstone = self.root / f"{TOMBSTONE_PREFIX}{name}-{secrets.token_hex(4)}"
        payload = {
            "voice": name,
            "path": str(path),
            "tombstone": str(tombstone),
            "dry_run": bool(dry_run),
        }
        if dry_run:
            return payload
        try:
            os.replace(path, tombstone)
        except OSError as exc:
            raise InputError(
                f"cannot delete voice {name}: {exc}", details={"voice": name}
            ) from exc
        try:
            shutil.rmtree(tombstone)
        except OSError as exc:
            raise InputError(
                f"cannot delete voice {name}; left tombstone {tombstone.name}",
                details={"voice": name, "tombstone": str(tombstone)},
            ) from exc
        return payload

    def remove(self, name: str) -> None:
        self.delete(name)

    def validate_reference(self, meta: dict[str, Any], base: Path) -> None:
        """Public read path for clone profiles (used by show and export)."""
        self._check_reference(meta, base)

    def _check_reference(self, meta: dict[str, Any], base: Path) -> None:
        """A clone profile must keep a usable prompt in the same directory."""
        if meta.get("kind") != "clone":
            return
        name = base.name
        prompt = base / "prompt.safetensors"
        ref = base / "ref.wav"
        if not prompt.is_file() and not ref.is_file():
            raise InputError(
                f"clone voice {name} has neither prompt.safetensors nor ref.wav",
                details={"voice": name},
            )
        for candidate in (prompt, ref):
            if candidate.is_symlink():
                raise InputError(
                    f"clone voice {name} reference is a symlink: {candidate.name}",
                    details={"voice": name},
                )
            if not candidate.exists():
                continue
            try:
                candidate.resolve(strict=True).relative_to(base.resolve(strict=True))
            except (OSError, ValueError) as exc:
                raise InputError(
                    f"clone voice {name} reference escapes its profile: {candidate.name}",
                    details={"voice": name},
                ) from exc


def _replace_case_only(source: Path, target: Path, journal: Path) -> None:
    """Case-only renames need a recoverable second hop on case-insensitive FS."""
    profile = journal / "profile"
    os.replace(source, profile)
    os.replace(profile, target)


def _atomic_copy(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{dest.name}.", suffix=".tmp", dir=str(dest.parent))
    try:
        with os.fdopen(fd, "wb") as out, open(src, "rb") as inp:
            shutil.copyfileobj(inp, out)
            out.flush()
            os.fsync(out.fileno())
        os.replace(tmp, dest)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _to_cpu_contiguous(tensor: Any) -> Any:
    try:
        return tensor.detach().to("cpu").contiguous()
    except AttributeError:  # already a plain value
        return tensor
