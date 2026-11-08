"""Managed paths, build lock, publication journal and recovery (FR-P2-03 §8)."""

from __future__ import annotations

import errno
import getpass
import os
import shutil
import signal
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from ..commands import post as post_cmd
from ..errors import InputError
from ..utils import atomic_write_json, read_json

WORK_DIR_NAMES = ("segments", "chapters", "recipes")
JOURNAL_NAME = ".audiobook.publish.json"
LOCK_NAME = ".audiobook.lock"


def canonical(path: str | os.PathLike[str]) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def collision_key(path: str | os.PathLike[str]) -> str:
    return post_cmd._collision_key(canonical(path))


def _parts(path: Path) -> list[str]:
    return [part.casefold() for part in canonical(path).parts]


def is_within(path: Path, directory: Path) -> bool:
    child = _parts(path)
    parent = _parts(directory)
    return len(child) >= len(parent) and child[: len(parent)] == parent


def work_dirs(out_dir: Path) -> dict[str, Path]:
    return {name: canonical(out_dir) / name for name in WORK_DIR_NAMES}


def validate_paths(
    managed: dict[str, list[Path]],
    dirs: Iterable[Path],
) -> None:
    """Reject cross-role collisions and paths inside builder work dirs."""
    keys: dict[str, str] = {}
    for role, paths in managed.items():
        for path in paths:
            key = collision_key(path)
            if key in keys:
                other = keys[key]
                if other == role:
                    continue
                raise InputError(
                    f"{role} path {path} collides with {other} path",
                    details={"path": str(canonical(path)), "roles": sorted({role, other})},
                )
            keys[key] = role
    work = [canonical(directory) for directory in dirs]
    for role, paths in managed.items():
        for path in paths:
            for directory in work:
                if is_within(path, directory):
                    raise InputError(
                        f"{role} path {path} is inside the builder work dir {directory}",
                        details={"path": str(canonical(path)), "work_dir": str(directory)},
                    )


def _check_parent(path: Path) -> None:
    if path.exists() and path.is_dir():
        raise InputError(f"target path is a directory: {path}", details={"path": str(path)})
    ancestor = path.parent
    while True:
        if ancestor.exists():
            if not ancestor.is_dir():
                raise InputError(
                    f"parent of {path} is not a directory: {ancestor}",
                    details={"path": str(path), "parent": str(ancestor)},
                )
            if not os.access(ancestor, os.W_OK):
                raise InputError(
                    f"parent directory is not writable: {ancestor}",
                    details={"path": str(path), "parent": str(ancestor)},
                )
            return
        if ancestor.parent == ancestor:
            raise InputError(f"cannot find a parent directory for {path}", details={"path": str(path)})
        ancestor = ancestor.parent


def preflight_parents(
    *,
    final: Path,
    manifest: Path,
    lock: Path,
    journal: Path,
    out_dir: Path,
) -> None:
    for path in (final, manifest, lock, journal):
        _check_parent(path)
    if not is_within(final, out_dir):
        parent = final.parent
        if not parent.exists() or not parent.is_dir():
            raise InputError(
                f"--out parent must already exist: {parent}",
                details={"parent": str(parent)},
            )
        if not os.access(parent, os.W_OK):
            raise InputError(
                f"--out parent is not writable: {parent}",
                details={"parent": str(parent)},
            )


def ensure_out_dir(out_dir: Path) -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Lock
# --------------------------------------------------------------------------- #


@dataclass
class BuildLock:
    path: Path
    acquired: bool = False
    stale_removed: bool = False

    def release(self) -> None:
        if not self.acquired:
            return
        try:
            os.unlink(self.path)
        except FileNotFoundError:
            pass
        except OSError:
            pass
        self.acquired = False


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as exc:
        if exc.errno == errno.ESRCH:
            return False
        if exc.errno == errno.EPERM:
            return True
        raise
    return True


def acquire_lock(path: Path) -> BuildLock:
    lock = BuildLock(canonical(path))
    payload = {
        "pid": os.getpid(),
        "user": getpass.getuser(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        _create_lock(lock.path, payload)
        lock.acquired = True
        return lock
    except FileExistsError:
        pass

    try:
        existing = read_json(lock.path)
    except (OSError, ValueError) as exc:
        raise InputError(
            f"another audiobook build holds the lock {lock.path} (unreadable)",
            details={"lock": str(lock.path)},
        ) from exc
    if not isinstance(existing, dict):
        raise InputError(
            f"audiobook lock {lock.path} is malformed",
            details={"lock": str(lock.path)},
        )
    owner = existing.get("user")
    if not isinstance(owner, str) or owner != getpass.getuser():
        raise InputError(
            f"audiobook lock {lock.path} is owned by another user; refusing to remove it",
            details={"lock": str(lock.path), "owner": owner},
        )
    pid = existing.get("pid")
    if not isinstance(pid, int) or isinstance(pid, bool):
        raise InputError(
            f"audiobook lock {lock.path} is malformed",
            details={"lock": str(lock.path)},
        )
    try:
        alive = _pid_alive(pid)
    except OSError as exc:
        raise InputError(f"cannot inspect audiobook lock pid {pid}: {exc}", details={"lock": str(lock.path)}) from exc
    if alive:
        raise InputError(
            "another audiobook build is running",
            details={"lock": str(lock.path), "pid": pid},
        )
    try:
        os.unlink(lock.path)
    except FileNotFoundError:
        pass
    except OSError as exc:
        raise InputError(f"cannot remove stale audiobook lock {lock.path}: {exc}", details={"lock": str(lock.path)}) from exc
    lock.stale_removed = True
    try:
        _create_lock(lock.path, payload)
        lock.acquired = True
    except FileExistsError as exc:
        raise InputError(
            "another audiobook build is running",
            details={"lock": str(lock.path)},
        ) from exc
    return lock


def _create_lock(path: Path, payload: dict[str, Any]) -> None:
    descriptor = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    try:
        import json

        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            os.unlink(path)
        except OSError:
            pass
        raise


# --------------------------------------------------------------------------- #
# Journal / recovery
# --------------------------------------------------------------------------- #


@dataclass
class RecoveryResult:
    warnings: list[str] = field(default_factory=list)


def read_journal(path: Path) -> Optional[dict[str, Any]]:
    if not Path(path).is_file():
        return None
    try:
        payload = read_json(path)
    except (OSError, ValueError) as exc:
        raise InputError(
            f"cannot read publication journal {path}: {exc}",
            details={"journal": str(path)},
        ) from exc
    if not isinstance(payload, dict):
        raise InputError(
            f"publication journal {path} is malformed",
            details={"journal": str(path)},
        )
    return payload


def _journal_targets(journal: dict[str, Any]) -> tuple[Path, Path]:
    try:
        final = canonical(journal["final"])
        manifest = canonical(journal["manifest"])
    except (KeyError, TypeError) as exc:
        raise InputError("publication journal is missing final/manifest targets") from exc
    return final, manifest


def _optional_path(raw: Any) -> Optional[Path]:
    if raw is None or raw == "":
        return None
    if not isinstance(raw, (str, os.PathLike)):
        raise InputError(
            "publication journal contains a malformed path",
            details={"value": repr(raw)},
        )
    return canonical(raw)


def _restore(
    target: Path,
    backup: Optional[Path],
    old_existed: bool,
) -> tuple[bool, Optional[str], bool]:
    """Return (ok, warning, missing_unrestorable)."""
    if backup is not None and backup.exists():
        try:
            os.replace(backup, target)
        except OSError as exc:
            return False, f"cannot restore {target} from {backup}: {exc}", True
        return True, None, False
    if old_existed:
        if target.exists():
            return True, None, False
        return True, f"cannot restore {target}: no backup and the original is gone", True
    try:
        if target.exists():
            os.unlink(target)
    except OSError as exc:
        return False, f"cannot remove partial publish {target}: {exc}", True
    return True, None, False


def _remove(path: Any, warnings: list[str], label: str) -> bool:
    if path is None:
        return True
    if not isinstance(path, (str, os.PathLike)):
        warnings.append(f"cannot remove {label}: malformed journal path {path!r}")
        return False
    try:
        if Path(path).exists():
            os.unlink(path)
    except OSError as exc:
        warnings.append(f"cannot remove {label} {path}: {exc}")
        return False
    return True


def recover(
    journal_path: Path,
    *,
    expected_final: Optional[Path],
    expected_manifest: Optional[Path],
    dry_run: bool = False,
) -> RecoveryResult:
    """Apply the §8 recovery table for an existing journal."""
    journal_path = canonical(journal_path)
    journal = read_journal(journal_path)
    result = RecoveryResult()
    if journal is None:
        return result
    final, manifest = _journal_targets(journal)
    if expected_final is not None and expected_manifest is not None:
        if collision_key(final) != collision_key(expected_final) or collision_key(manifest) != collision_key(
            expected_manifest
        ):
            message = (
                f"an interrupted build for {final} must be recovered first "
                f"(journal {journal_path}, targets {final} / {manifest})"
            )
            if dry_run:
                result.warnings.append(message)
                return result
            raise InputError(
                message,
                details={"journal": str(journal_path), "final": str(final), "manifest": str(manifest)},
            )
    if dry_run:
        return result

    state = str(journal.get("state") or "prepared")
    backup = _optional_path(journal.get("backup"))
    manifest_backup = _optional_path(journal.get("manifest_backup"))
    old_final_existed = bool(journal.get("old_final_existed"))
    old_manifest_existed = bool(journal.get("old_manifest_existed"))

    fatal = False
    cleanup_ok = True
    if state == "published" and final.exists() and manifest.exists():
        cleanup_ok &= _remove(backup, result.warnings, "final backup")
        cleanup_ok &= _remove(manifest_backup, result.warnings, "manifest backup")
        cleanup_ok &= _remove(journal.get("final_tmp"), result.warnings, "final temp")
        cleanup_ok &= _remove(journal.get("manifest_tmp"), result.warnings, "manifest temp")
    else:
        ok, warning, missing = _restore(final, backup, old_final_existed)
        fatal = fatal or missing
        cleanup_ok &= ok and not missing
        if warning:
            result.warnings.append(warning)
        ok2, warning2, missing2 = _restore(manifest, manifest_backup, old_manifest_existed)
        fatal = fatal or missing2
        cleanup_ok &= ok2 and not missing2
        if warning2:
            result.warnings.append(warning2)
        cleanup_ok &= _remove(journal.get("final_tmp"), result.warnings, "final temp")
        cleanup_ok &= _remove(journal.get("manifest_tmp"), result.warnings, "manifest temp")

    if state == "published" and final.exists() and not manifest.exists():
        # Never leave a new final without a manifest.
        fatal = True
        result.warnings.append(f"published final {final} has no manifest")
    if fatal:
        # Keep the journal: the next build must not lose the recovery trail.
        result.warnings.append(f"kept publication journal {journal_path} for the next build")
        raise InputError(
            f"publication journal {journal_path} cannot be fully recovered",
            details={
                "journal": str(journal_path),
                "final": str(final),
                "manifest": str(manifest),
            },
        )
    if not cleanup_ok:
        result.warnings.append(f"kept publication journal {journal_path} for the next build")
        return result
    _remove(journal_path, result.warnings, "journal")
    return result


# --------------------------------------------------------------------------- #
# Publication
# --------------------------------------------------------------------------- #


@dataclass
class PublishResult:
    warnings: list[str] = field(default_factory=list)
    interrupted_after_publish: bool = False


def _hook(hooks: Optional[dict[str, Callable[[], None]]], name: str) -> None:
    if hooks and name in hooks:
        hooks[name]()


def _install_deferring(record: list[int]) -> Optional[dict[int, Any]]:
    if threading.current_thread() is not threading.main_thread():
        return None
    previous: dict[int, Any] = {}

    def handler(signum: int, frame: Any) -> None:  # pragma: no cover - signal timing
        record.append(signum)

    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            try:
                previous[signum] = signal.getsignal(signum)
                signal.signal(signum, handler)
            except (ValueError, OSError):
                previous.pop(signum, None)
    except BaseException:
        _restore_handlers(previous)
        raise
    return previous


def _restore_handlers(previous: Optional[dict[int, Any]]) -> None:
    if not previous:
        return
    for signum, handler in previous.items():
        try:
            signal.signal(signum, handler)
        except (ValueError, OSError):
            pass


def _rollback(
    journal_path: Path,
    journal: dict[str, Any],
    *,
    final: Path,
    manifest: Path,
    warnings: list[str],
) -> None:
    backup = _optional_path(journal.get("backup"))
    manifest_backup = _optional_path(journal.get("manifest_backup"))
    ok, _, missing = _restore(final, backup, bool(journal.get("old_final_existed")))
    ok2, _, missing2 = _restore(manifest, manifest_backup, bool(journal.get("old_manifest_existed")))
    temp_ok = _remove(journal.get("final_tmp"), warnings, "final temp")
    temp_ok &= _remove(journal.get("manifest_tmp"), warnings, "manifest temp")
    if ok and ok2 and not missing and not missing2 and temp_ok:
        _remove(journal_path, warnings, "journal")
    else:
        warnings.append(f"kept publication journal {journal_path} for the next build")


def publish(
    *,
    final_tmp: Path,
    manifest_tmp: Path,
    final: Path,
    manifest: Path,
    journal_path: Path,
    work_dirs_to_remove: Iterable[Path] = (),
    keep_work_dirs: bool = False,
    hooks: Optional[dict[str, Callable[[], None]]] = None,
) -> PublishResult:
    """Run the §8 publication transaction."""
    final = canonical(final)
    manifest = canonical(manifest)
    journal_path = canonical(journal_path)
    token = uuid.uuid4().hex
    backup = final.parent / f"{final.name}.{token}.bak"
    manifest_backup = manifest.parent / f"{manifest.name}.{token}.mbak"
    old_final_existed = final.exists()
    old_manifest_existed = manifest.exists()
    journal: dict[str, Any] = {
        "schema": 1,
        "state": "prepared",
        "final": str(final),
        "manifest": str(manifest),
        "final_tmp": str(canonical(final_tmp)),
        "manifest_tmp": str(canonical(manifest_tmp)),
        "backup": str(backup),
        "manifest_backup": str(manifest_backup),
        "old_final_existed": bool(old_final_existed),
        "old_manifest_existed": bool(old_manifest_existed),
        "pid": os.getpid(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(journal_path, journal)

    deferred: list[int] = []
    previous: Optional[dict[int, Any]] = None
    try:
        _hook(hooks, "after_journal")
        if old_final_existed:
            os.replace(final, backup)
            _hook(hooks, "after_final_backup")
        if old_manifest_existed:
            os.replace(manifest, manifest_backup)
            _hook(hooks, "after_manifest_backup")
        os.replace(final_tmp, final)
        _hook(hooks, "after_final_replace")
        os.replace(manifest_tmp, manifest)
        _hook(hooks, "after_manifest_replace")
        # Installing the deferring handler is the effective commit point.
        # Signals before this line roll back; the call itself is covered by
        # this try so no window is left between step 6 and the commit point.
        previous = _install_deferring(deferred)
    except BaseException:
        warnings: list[str] = []
        _rollback(
            journal_path,
            journal,
            final=final,
            manifest=manifest,
            warnings=warnings,
        )
        _restore_handlers(previous)
        raise

    warnings = []
    try:
        try:
            _hook(hooks, "before_published_write")
            journal["state"] = "published"
            atomic_write_json(journal_path, journal)
        except BaseException:
            _rollback(
                journal_path,
                journal,
                final=final,
                manifest=manifest,
                warnings=warnings,
            )
            raise
        _hook(hooks, "after_published")
        failed_cleanup = False
        for path in (backup, manifest_backup):
            try:
                if Path(path).exists():
                    os.unlink(path)
            except OSError as exc:
                failed_cleanup = True
                warnings.append(f"cannot remove backup {path}: {exc}")
        for key in ("final_tmp", "manifest_tmp"):
            raw = journal.get(key)
            if raw:
                try:
                    if Path(raw).exists():
                        os.unlink(raw)
                except OSError as exc:
                    warnings.append(f"cannot remove temporary {raw}: {exc}")
        if failed_cleanup:
            warnings.append(f"kept publication journal {journal_path} for the next build")
        else:
            try:
                os.unlink(journal_path)
            except FileNotFoundError:
                pass
            except OSError as exc:
                warnings.append(f"cannot remove publication journal {journal_path}: {exc}")
        if not keep_work_dirs:
            for directory in work_dirs_to_remove:
                try:
                    shutil.rmtree(directory, ignore_errors=False)
                except FileNotFoundError:
                    pass
                except OSError as exc:
                    warnings.append(f"cannot remove work dir {directory}: {exc}")
        _hook(hooks, "cleanup")
    finally:
        _restore_handlers(previous)
    return PublishResult(warnings=warnings, interrupted_after_publish=bool(deferred))
