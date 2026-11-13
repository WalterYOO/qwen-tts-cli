"""Crash-recoverable multi-voice install for ``voice import-btts``.

The journal is deliberately small and path-based.  Every entry records the
target voice directory, the real on-disk old path (which may differ only by
case), the hidden staging/backup directories and a set of progress flags.  A
crash between any two filesystem operations can therefore be replayed without
guessing whether the old profile or the new one is currently installed.

This module only touches paths below ``VoiceStore.root`` (``<data>/voices``).
It never receives an arbitrary user path.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import sys
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .errors import InputError
from .utils import atomic_write_json, read_json
from .voice_store import VoiceStore, validate_name

JOURNAL_NAME = ".btts-import.json"
LOCK_NAME = ".btts-import.lock"
_BACKUP_PREFIX = ".btts-backup."
_FLAGS = (
    "backup_done",
    "installing",
    "installed",
    "replacing",
    "restoring",
    "restored",
    "staging_removed",
)


def _safe_child(root: Path, name: Any, *, allow_missing: bool = True) -> Path:
    if not isinstance(name, str) or not name:
        raise InputError("invalid import journal path")
    if name in (".", "..") or "\x00" in name or "/" in name or "\\" in name:
        raise InputError("invalid import journal path")
    if Path(name).is_absolute() or Path(name).name != name:
        raise InputError("invalid import journal path")
    path = root / name
    try:
        resolved_root = root.resolve()
        resolved = path.resolve(strict=False)
    except OSError as exc:
        raise InputError(f"cannot resolve import path: {exc}") from exc
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise InputError("import path escapes the voice store")
    if path.is_symlink():
        raise InputError(f"refusing symlinked import path: {path}")
    if not allow_missing and not path.exists():
        raise InputError(f"import path disappeared: {path}")
    return path


def _voice_child(root: Path, name: Any) -> Path:
    if not isinstance(name, str):
        raise InputError("invalid voice name in import journal")
    try:
        validate_name(name)
    except InputError as exc:
        raise InputError("invalid voice name in import journal") from exc
    return _safe_child(root, name)


def _remove(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if path.is_symlink() or path.is_file():
        path.unlink()
    else:
        shutil.rmtree(path)


def _same_path_identity(left: Path, right: Path) -> bool:
    if left == right:
        return True
    try:
        if left.exists() and right.exists():
            return os.path.samefile(left, right)
    except OSError:
        return False
    return False


class ImportLock:
    """Non-blocking cross-platform lock for the import journal."""

    def __init__(self, store: VoiceStore) -> None:
        self.store = store

    def __enter__(self):
        self.store.root.mkdir(parents=True, exist_ok=True)
        self._handle = (self.store.root / LOCK_NAME).open("a+")
        try:
            if sys.platform.startswith("win"):
                import msvcrt

                self._handle.seek(0)
                msvcrt.locking(self._handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self._handle.close()
            raise InputError(f"another btts import is running: {self.store.root}") from exc
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            if sys.platform.startswith("win"):
                import msvcrt

                self._handle.seek(0)
                msvcrt.locking(self._handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        finally:
            self._handle.close()
        return False


def journal_path(store: VoiceStore) -> Path:
    return store.root / JOURNAL_NAME


def has_pending_journal(store: VoiceStore) -> bool:
    return journal_path(store).is_file()


def _write_journal(store: VoiceStore, payload: dict[str, Any]) -> None:
    atomic_write_json(journal_path(store), payload)


def _read_journal(store: VoiceStore) -> dict[str, Any]:
    path = journal_path(store)
    if path.is_symlink():
        raise InputError(f"refusing symlinked btts import journal: {path}")
    try:
        payload = read_json(path)
    except (OSError, ValueError) as exc:
        raise InputError(f"corrupt btts import journal: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise InputError(f"invalid btts import journal: {path}")
    return payload


def _validate_entries(store: VoiceStore, payload: dict[str, Any]) -> list[dict[str, Any]]:
    if type(payload.get("schema_version")) is not int or payload["schema_version"] != 1:
        raise InputError("invalid btts import journal: unsupported schema_version")
    if type(payload.get("committed")) is not bool:
        raise InputError("invalid btts import journal: committed must be a boolean")
    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        raise InputError("invalid btts import journal: entries must be a non-empty list")
    checked: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in entries:
        if not isinstance(raw, dict):
            raise InputError("invalid btts import journal entry")
        target = raw.get("target")
        _voice_child(store.root, target)
        folded = str(target).casefold()
        if folded in seen:
            raise InputError("duplicate target in btts import journal")
        seen.add(folded)
        for flag in ("had_existing", *_FLAGS):
            if type(raw.get(flag)) is not bool:
                raise InputError(
                    f"invalid btts import journal: {flag} must be a boolean"
                )
        had_existing = raw["had_existing"]
        existing = raw.get("existing_path")
        if had_existing:
            if not isinstance(existing, str):
                raise InputError("btts import journal is missing existing_path")
            _voice_child(store.root, existing)
            if existing.casefold() != folded:
                raise InputError(
                    "btts import journal existing_path does not match target"
                )
        elif existing is not None:
            raise InputError("btts import journal has existing_path without had_existing")
        staging = raw.get("staging")
        _safe_child(store.root, staging)
        if not isinstance(staging, str) or not staging.startswith(f".{target}."):
            raise InputError("invalid btts import journal staging path")
        backup = raw.get("backup")
        if had_existing:
            if not isinstance(backup, str) or not backup.startswith(_BACKUP_PREFIX):
                raise InputError("invalid btts import journal backup path")
            _safe_child(store.root, backup)
        elif backup is not None:
            raise InputError("btts import journal has backup without had_existing")
        checked.append(dict(raw))
    if payload["committed"]:
        for entry in checked:
            if not entry["installed"]:
                raise InputError(
                    "invalid btts import journal: committed entry is not installed"
                )
    return checked


def _update_entry(
    store: VoiceStore, payload: dict[str, Any], index: int, **updates: Any
) -> dict[str, Any]:
    entry = dict(payload["entries"][index])
    entry.update(updates)
    payload["entries"][index] = entry
    _write_journal(store, payload)
    return entry


def _rollback_entry(
    store: VoiceStore,
    payload: dict[str, Any],
    index: int,
) -> None:
    root = store.root
    entry = dict(payload["entries"][index])
    target = _voice_child(root, entry["target"])
    existing = (
        _voice_child(root, entry["existing_path"])
        if entry.get("had_existing")
        else None
    )
    backup = (
        _safe_child(root, entry["backup"]) if entry.get("backup") else None
    )
    staging = _safe_child(root, entry["staging"]) if entry.get("staging") else None

    def update(**values: Any) -> None:
        nonlocal entry
        entry = _update_entry(store, payload, index, **values)

    if entry.get("restored") or entry.get("staging_removed"):
        if backup is not None and backup.exists():
            # A restored entry must have moved its backup back.  If the backup
            # is still present the journal is inconsistent; fail closed rather
            # than delete the only copy of the old profile in the cleanup pass.
            raise InputError(
                "cannot recover btts import: backup is still present after restore",
                details={"voice": entry["target"]},
            )
        if staging is not None:
            _remove(staging)
        update(restored=True, staging_removed=True, restoring=False, installing=False)
        return

    if entry.get("restoring"):
        if entry.get("had_existing"):
            if backup is not None and backup.exists():
                # The backup is authoritative: target (if present) is the new
                # profile, even when it resolves to the same path as existing
                # on a case-insensitive filesystem.
                if target.exists():
                    _remove(target)
                    update(restoring=True)
                if existing.exists():
                    raise InputError(
                        "cannot recover btts import: existing path reappeared "
                        "while the backup is still present",
                        details={"voice": entry["target"]},
                    )
                os.replace(backup, existing)
                update(
                    restored=True,
                    staging_removed=True,
                    restoring=False,
                    installing=False,
                    installed=False,
                )
                if staging is not None:
                    _remove(staging)
                return
            if existing is not None and existing.exists():
                update(
                    restored=True,
                    staging_removed=True,
                    restoring=False,
                    installing=False,
                    installed=False,
                )
                if staging is not None:
                    _remove(staging)
                return
            raise InputError(
                "cannot recover btts import: backup disappeared before restore",
                details={"voice": entry["target"]},
            )
        # New voice: there is no old profile to restore.
        if target.exists():
            update(restoring=True)
            _remove(target)
        update(
            restored=True,
            staging_removed=True,
            restoring=False,
            installing=False,
            installed=False,
        )
        if staging is not None:
            _remove(staging)
        return

    if entry.get("installed"):
        if entry.get("had_existing"):
            if backup is not None and backup.exists():
                update(restoring=True)
                if target.exists():
                    _remove(target)
                if existing.exists():
                    raise InputError(
                        "cannot recover btts import: existing path reappeared "
                        "while the backup is still present",
                        details={"voice": entry["target"]},
                    )
                os.replace(backup, existing)
                update(
                    restored=True,
                    staging_removed=True,
                    restoring=False,
                    installing=False,
                    installed=False,
                )
                if staging is not None:
                    _remove(staging)
                return
            raise InputError(
                "cannot recover btts import: old profile backup is missing",
                details={"voice": entry["target"]},
            )
        if target.exists():
            _remove(target)
        update(
            restored=True,
            staging_removed=True,
            restoring=False,
            installing=False,
            installed=False,
        )
        if staging is not None:
            _remove(staging)
        return

    # installing=true / installed=false (or backup happened but replace did not).
    if entry.get("installing"):
        if staging is not None and staging.exists() and not target.exists():
            if entry.get("had_existing"):
                if backup is None or not backup.exists():
                    if existing is None or not existing.exists():
                        raise InputError(
                            "cannot recover btts import: old profile is missing",
                            details={"voice": entry["target"]},
                        )
                else:
                    os.replace(backup, existing)
            _remove(staging)
            update(
                restored=True,
                staging_removed=True,
                restoring=False,
                installing=False,
                installed=False,
            )
            return
        if not target.exists() and (staging is None or not staging.exists()):
            if backup is not None and backup.exists():
                if existing is not None and not existing.exists():
                    os.replace(backup, existing)
                update(
                    restored=True,
                    staging_removed=True,
                    restoring=False,
                    installing=False,
                    installed=False,
                )
                if staging is not None:
                    _remove(staging)
                return
            if not entry.get("had_existing"):
                update(
                    restored=True,
                    staging_removed=True,
                    restoring=False,
                    installing=False,
                    installed=False,
                )
                return
            if existing is not None and existing.exists():
                update(
                    restored=True,
                    staging_removed=True,
                    restoring=False,
                    installing=False,
                    installed=False,
                )
                return
            raise InputError(
                "cannot recover btts import: old profile is missing",
                details={"voice": entry["target"]},
            )
        if target.exists() and staging is not None and not staging.exists():
            # The rename completed but the installed flag did not.
            update(installed=True, installing=False)
            return _rollback_entry(store, payload, index)
        raise InputError(
            "cannot determine btts import state",
            details={"voice": entry["target"]},
        )

    if entry.get("backup_done"):
        if backup is not None and backup.exists():
            # The backup is authoritative.  Neither the target nor the
            # original name may exist when the old profile is restored.
            if target.exists() or (existing is not None and existing.exists()):
                raise InputError(
                    "cannot determine btts import state",
                    details={"voice": entry["target"]},
                )
            os.replace(backup, existing)
            update(
                restored=True,
                staging_removed=True,
                restoring=False,
                installing=False,
                installed=False,
            )
            if staging is not None:
                _remove(staging)
            return
        if existing is not None and existing.exists() and not target.exists():
            update(
                restored=True,
                staging_removed=True,
                restoring=False,
                installing=False,
                installed=False,
            )
            if staging is not None:
                _remove(staging)
            return
        raise InputError(
            "cannot determine btts import state",
            details={"voice": entry["target"]},
        )

    # No backup was recorded.  The old directory was never moved.
    if backup is not None and backup.exists():
        if target.exists():
            raise InputError(
                "cannot determine btts import state",
                details={"voice": entry["target"]},
            )
        os.replace(backup, existing)
        update(
            restored=True,
            staging_removed=True,
            restoring=False,
            installing=False,
            installed=False,
        )
        if staging is not None:
            _remove(staging)
        return
    if target.exists() and existing is not None and not _same_path_identity(target, existing):
        raise InputError(
            "cannot determine btts import state",
            details={"voice": entry["target"]},
        )
    if staging is not None:
        _remove(staging)
    update(
        restored=True,
        staging_removed=True,
        restoring=False,
        installing=False,
        installed=False,
    )


def recover_import(
    store: VoiceStore, *, dry_run: bool = False, _locked: bool = False
) -> list[str]:
    """Recover a pending journal.  Returns recovered target names."""
    if not has_pending_journal(store):
        return []
    if dry_run:
        raise InputError(
            "a previous btts import needs recovery; rerun without --dry-run",
            details={"journal": str(journal_path(store))},
        )
    if _locked:
        return _recover_locked(store)
    with ImportLock(store):
        return _recover_locked(store)


def _recover_locked(store: VoiceStore) -> list[str]:
    payload = _read_journal(store)
    entries = _validate_entries(store, payload)
    payload["entries"] = entries
    recovered: list[str] = []
    if payload.get("committed"):
        for entry in entries:
            target = _voice_child(store.root, entry["target"])
            if entry.get("installed") and not target.exists():
                raise InputError(
                    "committed btts import is missing a voice directory",
                    details={"voice": entry["target"]},
                )
            staging = _safe_child(store.root, entry["staging"])
            backup = (
                _safe_child(store.root, entry["backup"])
                if entry.get("backup")
                else None
            )
            _remove(staging)
            if backup is not None:
                _remove(backup)
            recovered.append(entry["target"])
        journal_path(store).unlink(missing_ok=True)
        return recovered
    for index in range(len(entries) - 1, -1, -1):
        _rollback_entry(store, payload, index)
        recovered.append(payload["entries"][index]["target"])
    for entry in payload["entries"]:
        staging = _safe_child(store.root, entry["staging"])
        _remove(staging)
        if entry.get("backup"):
            backup = _safe_child(store.root, entry["backup"])
            if backup.exists():
                # Every rollback path either moves the backup back or raises.
                # Reaching here means the journal is inconsistent, so do not
                # delete the only remaining copy of the old profile.
                raise InputError(
                    "cannot recover btts import: backup is still present after rollback",
                    details={"voice": entry["target"]},
                )
    journal_path(store).unlink(missing_ok=True)
    return recovered


def commit_import(
    store: VoiceStore,
    items: Iterable[tuple[str, Path, str | None]],
    *,
    dry_run: bool = False,
) -> set[str]:
    """Install staged voices atomically, with journal recovery on a crash.

    ``items`` is ``(target_name, staging_dir, existing_name_or_none)``.
    Returns the target names that replaced an existing voice.
    """
    if dry_run:
        return set()
    if has_pending_journal(store):
        recover_import(store)
    prepared = [
        (validate_name(target), Path(staging), existing)
        for target, staging, existing in items
    ]
    if not prepared:
        return set()
    with ImportLock(store):
        if has_pending_journal(store):
            raise InputError("a btts import journal appeared while waiting for the lock")
        current = store.existing_names()
        for target, _staging, existing in prepared:
            if current.get(target.casefold()) != existing:
                raise InputError(
                    "voice store changed while waiting for the btts import lock; "
                    "rerun the import",
                    details={"voice": target},
                )
        entries: list[dict[str, Any]] = []
        replaced: set[str] = set()
        for target, staging, existing in prepared:
            if existing is not None:
                validate_name(existing)
            backup = (
                f"{_BACKUP_PREFIX}{uuid.uuid4().hex[:12]}"
                if existing is not None
                else None
            )
            entries.append(
                {
                    "target": target,
                    "existing_path": existing,
                    "staging": staging.name,
                    "backup": backup,
                    "had_existing": existing is not None,
                    "backup_done": False,
                    "installing": False,
                    "installed": False,
                    "replacing": existing is not None,
                    "restoring": False,
                    "restored": False,
                    "staging_removed": False,
                }
            )
        payload: dict[str, Any] = {"schema_version": 1, "committed": False, "entries": entries}
        _write_journal(store, payload)
        try:
            for index, (target, staging, existing) in enumerate(prepared):
                entry = entries[index]
                if existing is not None:
                    existing_path = _voice_child(store.root, existing)
                    backup_path = _safe_child(store.root, entry["backup"])
                    os.replace(existing_path, backup_path)
                    entry = _update_entry(store, payload, index, backup_done=True)
                entry = _update_entry(store, payload, index, installing=True)
                os.replace(staging, store.root / target)
                _update_entry(
                    store,
                    payload,
                    index,
                    installing=False,
                    installed=True,
                )
                if existing is not None:
                    replaced.add(target)
            payload["committed"] = True
            _write_journal(store, payload)
            for entry in payload["entries"]:
                _remove(_safe_child(store.root, entry["staging"]))
                if entry.get("backup"):
                    _remove(_safe_child(store.root, entry["backup"]))
            journal_path(store).unlink(missing_ok=True)
            return replaced
        except BaseException:
            with contextlib.suppress(BaseException):
                recover_import(store, _locked=True)
            raise
