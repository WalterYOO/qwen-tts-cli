"""Managed paths, flock, fixed-order publication and manifests (FR-P2-04 §7)."""

from __future__ import annotations

import errno
import json
import os
import signal
import threading
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from ..commands import post as post_cmd
from ..errors import InputError, ModelError
from ..utils import eprint


def canonical(path: str | os.PathLike[str]) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def collision_key(path: str | os.PathLike[str]) -> str:
    return post_cmd._collision_key(canonical(path))


def _parts(path: Path) -> list[str]:
    return [
        unicodedata.normalize("NFC", part).casefold()
        for part in canonical(path).parts
    ]


def is_within(path: Path, directory: Path) -> bool:
    child = _parts(path)
    parent = _parts(directory)
    return len(child) >= len(parent) and child[: len(parent)] == parent


def validate_paths(
    *,
    files: dict[str, Path],
    dirs: dict[str, Path],
    inputs: dict[str, Path],
    work_dir: Path,
    stems_dir: Optional[Path],
) -> None:
    """Reject cross-role collisions and unsafe containment (§7)."""
    roles: dict[str, str] = {}
    for role, path in {**files, **dirs, **inputs}.items():
        key = collision_key(path)
        if key in roles:
            raise InputError(
                f"{role} path {path} collides with {roles[key]} path",
                details={"path": str(canonical(path)), "roles": sorted({role, roles[key]})},
            )
        roles[key] = role

    work = canonical(work_dir)
    for role, path in files.items():
        if is_within(path, work):
            raise InputError(
                f"{role} path {path} is inside the work dir {work}",
                details={"path": str(canonical(path)), "work_dir": str(work)},
            )
    if stems_dir is not None:
        stems = canonical(stems_dir)
        if is_within(stems, work) or is_within(work, stems):
            raise InputError(
                f"work dir {work} and stems dir {stems} must not contain each other",
                details={"work_dir": str(work), "stems_dir": str(stems)},
            )
    for role, path in inputs.items():
        if is_within(path, work):
            raise InputError(
                f"{role} input {path} is inside the work dir {work}",
                details={"path": str(canonical(path)), "work_dir": str(work)},
            )
        if stems_dir is not None and is_within(path, stems_dir):
            raise InputError(
                f"{role} input {path} is inside the stems dir {stems_dir}",
                details={"path": str(canonical(path)), "stems_dir": str(canonical(stems_dir))},
            )


def _nearest_ancestor(path: Path) -> Path:
    ancestor = canonical(path).parent
    while True:
        if ancestor.exists():
            return ancestor
        if ancestor.parent == ancestor:
            raise InputError(
                f"cannot find a parent directory for {path}", details={"path": str(path)}
            )
        ancestor = ancestor.parent


def _check_file_target(path: Path) -> None:
    target = canonical(path)
    if target.exists() and target.is_dir():
        raise InputError(
            f"target path is a directory: {target}", details={"path": str(target)}
        )
    ancestor = _nearest_ancestor(target)
    if not ancestor.is_dir():
        raise InputError(
            f"parent of {target} is not a directory: {ancestor}",
            details={"path": str(target), "parent": str(ancestor)},
        )
    if not os.access(ancestor, os.W_OK):
        raise InputError(
            f"parent directory is not writable: {ancestor}",
            details={"path": str(target), "parent": str(ancestor)},
        )


def _check_dir_target(path: Path) -> None:
    target = canonical(path)
    if target.exists():
        if not target.is_dir():
            raise InputError(
                f"directory role already exists as a file: {target}",
                details={"path": str(target)},
            )
        if not os.access(target, os.W_OK):
            raise InputError(
                f"directory is not writable: {target}", details={"path": str(target)}
            )
        return
    ancestor = _nearest_ancestor(target)
    if not ancestor.is_dir():
        raise InputError(
            f"parent of {target} is not a directory: {ancestor}",
            details={"path": str(target), "parent": str(ancestor)},
        )
    if not os.access(ancestor, os.W_OK):
        raise InputError(
            f"parent directory is not writable: {ancestor}",
            details={"path": str(target), "parent": str(ancestor)},
        )


def preflight_parents(files: dict[str, Path], dirs: dict[str, Path]) -> None:
    for path in files.values():
        _check_file_target(path)
    for path in dirs.values():
        _check_dir_target(path)


def check_existing_targets(
    files: dict[str, Path],
    dirs: dict[str, Path],
    *,
    force: bool,
    dry_run: bool,
    exclude_dirs: Iterable[str] = (),
) -> None:
    if force or dry_run:
        return
    excluded = set(exclude_dirs)
    for role, path in files.items():
        if canonical(path).exists():
            raise InputError(
                f"output already exists: {path}; pass --force to overwrite",
                details={"path": str(canonical(path)), "role": role},
            )
    for role, path in dirs.items():
        if role in excluded:
            continue
        if canonical(path).exists():
            raise InputError(
                f"output already exists: {path}; pass --force to overwrite",
                details={"path": str(canonical(path)), "role": role},
            )


def create_directories(paths: Iterable[Path]) -> None:
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# flock
# --------------------------------------------------------------------------- #


def acquire_lock(work_dir: Path) -> int:
    """Non-blocking exclusive flock on ``<work>/.lock``.

    Contention is exit 4; any other locking failure is exit 3.  The kernel
    releases the lock on process exit, including SIGKILL.
    """
    try:
        import fcntl
    except ImportError as exc:  # pragma: no cover - POSIX-only project
        raise ModelError("fcntl is unavailable; cannot lock the work dir") from exc
    lock_path = Path(work_dir) / ".lock"
    try:
        descriptor = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o644)
    except OSError as exc:
        raise ModelError(f"cannot open build lock {lock_path}: {exc}") from exc
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        os.close(descriptor)
        raise InputError(
            f"another dub build holds the lock {lock_path}",
            details={"lock": str(lock_path)},
        ) from exc
    except OSError as exc:
        os.close(descriptor)
        if exc.errno in (errno.EACCES, errno.EAGAIN):
            raise InputError(
                f"another dub build holds the lock {lock_path}",
                details={"lock": str(lock_path)},
            ) from exc
        raise ModelError(f"cannot lock the work dir {work_dir}: {exc}") from exc
    return descriptor


def release_lock(descriptor: Optional[int]) -> None:
    if descriptor is None:
        return
    try:
        os.close(descriptor)
    except OSError:
        pass


# --------------------------------------------------------------------------- #
# Publication
# --------------------------------------------------------------------------- #


def _install_deferring(record: list[int]) -> Optional[dict[int, Any]]:
    if threading.current_thread() is not threading.main_thread():
        return None
    previous: dict[int, Any] = {}

    def handler(signum: int, frame: Any) -> None:  # pragma: no cover - signal timing
        if record:
            # A second signal exits immediately, as required by the contract.
            try:
                signal.signal(signum, signal.SIG_DFL)
            except (ValueError, OSError):
                pass
            os.kill(os.getpid(), signum)
            return
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


@dataclass
class PublishResult:
    warnings: list[str] = field(default_factory=list)
    interrupted_after_publish: bool = False


def publish(
    *,
    final_tmp: Path,
    final: Path,
    later: list[tuple[Path, Path]],
    manifest_tmp: Path,
    manifest: Path,
    warnings: list[str],
    quiet: bool = False,
) -> PublishResult:
    """Atomically replace artifacts in the contract's fixed order.

    ``later`` contains (temp, destination) pairs for stems and timeline.json
    (and the muxed mp4); the manifest is always replaced last.
    """
    temps = [final_tmp, manifest_tmp] + [Path(tmp) for tmp, _ in later]
    deferred: list[int] = []
    previous: Optional[dict[int, Any]] = None
    try:
        # Installing the deferring handler is the effective commit point.
        previous = _install_deferring(deferred)
        try:
            os.replace(final_tmp, final)
        except BaseException as exc:
            _restore_handlers(previous)
            for path in temps:
                _remove(path)
            if deferred:
                message = (
                    "cancellation signal was recorded during a failed final replace"
                )
                warnings.append(message)
                if not quiet:
                    eprint(f"warning: {message}")
            raise ModelError(
                f"failed to publish final output {final}: {exc}",
                details={"path": str(final)},
            ) from exc
        for tmp, destination in list(later) + [(manifest_tmp, manifest)]:
            try:
                os.replace(tmp, destination)
            except BaseException as exc:
                _restore_handlers(previous)
                for path in temps:
                    _remove(path)
                if deferred:
                    message = (
                        "cancellation was recorded but publication did not complete; "
                        f"already replaced files are kept ({destination})"
                    )
                    warnings.append(message)
                    if not quiet:
                        eprint(f"warning: {message}")
                raise ModelError(
                    f"failed to publish {destination}: {exc}",
                    details={"path": str(destination)},
                ) from exc
    except BaseException:
        _restore_handlers(previous)
        raise
    _restore_handlers(previous)
    return PublishResult(warnings=[], interrupted_after_publish=bool(deferred))


def _remove(path: Path) -> None:
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
    except OSError:
        pass


def write_json_temp(path: Path, document: dict[str, Any]) -> None:
    payload = json.dumps(document, ensure_ascii=False, sort_keys=True) + "\n"
    try:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise ModelError(
            f"cannot write {path}: {exc}", details={"path": str(path)}
        ) from exc


def build_document(
    *,
    timeline: dict[str, Any],
    project: Optional[dict[str, Any]],
    video: Optional[dict[str, Any]],
    request: dict[str, Any],
    voices: dict[str, Any],
    cues: dict[str, int],
    speakers: list[str],
    sample_rate: int,
    output: dict[str, Any],
    mux: Optional[dict[str, Any]],
    stems: Optional[dict[str, Any]],
    bed: Optional[dict[str, Any]],
    loudness: dict[str, Any],
    stretch: dict[str, Any],
    warnings: list[str],
    clipped: bool,
) -> dict[str, Any]:
    from .. import __version__

    return {
        "schema_version": 1,
        "qtts_version": __version__,
        "timeline": timeline,
        "project": project,
        "video": video,
        "request": request,
        "voices": voices,
        "cues": dict(cues),
        "speakers": list(speakers),
        "sample_rate": int(sample_rate),
        "output": output,
        "mux": mux,
        "stems": stems,
        "bed": bed,
        "loudness": dict(loudness),
        "stretch": dict(stretch),
        "warnings": list(warnings),
        "clipped": bool(clipped),
    }
