"""External runner, flock helpers, streaming and signal handling (§4.2/§5.2/§7)."""

from __future__ import annotations

import contextlib
import errno
import fcntl
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterator, Optional, Sequence

from ..errors import InputError, ModelError
from ..utils import eprint

RING_LINES = 50


class CommitInterrupt(KeyboardInterrupt):
    """A deferred signal delivered during/after a commit phase."""

    def __init__(
        self,
        *,
        published: bool = False,
        registered: bool = False,
        partial: Optional[Path] = None,
    ) -> None:
        super().__init__("interrupted during commit")
        self.published = published
        self.registered = registered
        self.partial = partial

    def status_line(self) -> str:
        return f"published: {str(self.published).lower()}, registered: {str(self.registered).lower()}"


def build_env(offline: bool = False) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    if offline:
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
    return env


# --------------------------------------------------------------------------- #
# Locks
# --------------------------------------------------------------------------- #
def project_lock_path(project_file: Path) -> Path:
    return project_file.parent / f".{project_file.stem}.finetune.lock"


def _open_lock(path: Path) -> int:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        return os.open(str(path), os.O_RDWR | os.O_CREAT, 0o600)
    except OSError as exc:
        raise ModelError(
            f"cannot create lock file {path}: {exc}", details={"path": str(path)}
        ) from exc


def acquire_lock(path: Path, *, shared: bool = False) -> int:
    """Acquire one non-blocking flock.  Contention is exit 4, I/O is exit 3."""
    fd = _open_lock(path)
    flags = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
    try:
        fcntl.flock(fd, flags | fcntl.LOCK_NB)
    except OSError as exc:
        os.close(fd)
        if isinstance(exc, BlockingIOError) or exc.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
            raise InputError(
                "another finetune command is using this project/output",
                details={"lock": str(path)},
            ) from exc
        raise ModelError(
            f"cannot lock {path}: {exc}", details={"path": str(path)}
        ) from exc
    return fd


def hold_locks(
    paths: Sequence[Path], *, shared: bool = False
) -> contextlib.AbstractContextManager[list[int]]:
    return hold_lock_spec([(path, shared) for path in paths])


@contextlib.contextmanager
def hold_lock_spec(specs: Sequence[tuple[Path, bool]]) -> Iterator[list[int]]:
    fds: list[int] = []
    try:
        for path, shared in specs:
            fds.append(acquire_lock(path, shared=shared))
        yield fds
    finally:
        for fd in fds:
            with contextlib.suppress(OSError):
                os.close(fd)


# --------------------------------------------------------------------------- #
# External subprocess
# --------------------------------------------------------------------------- #
def _command_summary(argv: Sequence[str]) -> str:
    return " ".join(str(part) for part in argv)


def run_external(
    argv: Sequence[str],
    *,
    cwd: Path,
    env: dict[str, str],
    quiet: bool = False,
) -> int:
    """Run one external command; stdout/stderr are merged and streamed.

    Raises ``ModelError`` on spawn/wait failures and non-zero exit, and
    ``KeyboardInterrupt`` after an interrupted child exits.
    """
    tail: list[str] = []
    interrupted = False
    forwarded = False

    def handle(signum, frame):
        nonlocal interrupted, forwarded
        if not forwarded:
            forwarded = True
            interrupted = True
            if process is not None:
                try:
                    os.killpg(process.pid, signum)
                except (ProcessLookupError, PermissionError, OSError):
                    pass
            return
        # Second signal: restore the default and re-raise immediately.
        with contextlib.suppress(OSError):
            signal.signal(signum, signal.SIG_DFL)
        raise KeyboardInterrupt

    previous: dict[int, Any] = {}
    process: Optional[subprocess.Popen] = None
    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.signal(signum, handle)
        try:
            process = subprocess.Popen(
                [str(part) for part in argv],
                cwd=str(cwd),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except OSError as exc:
            raise ModelError(
                f"cannot start external command: {exc}",
                details={"command": _command_summary(argv), "cwd": str(cwd)},
            ) from exc
        assert process.stdout is not None
        for raw in process.stdout:
            line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
            if line:
                tail.append(line)
                if len(tail) > RING_LINES:
                    del tail[0]
                if not quiet:
                    eprint(line)
        returncode = process.wait()
    finally:
        for signum, handler in previous.items():
            with contextlib.suppress(OSError, ValueError):
                signal.signal(signum, handler)
    if interrupted:
        raise KeyboardInterrupt
    if returncode != 0:
        message = (
            f"external command failed with exit code {returncode}: "
            f"{_command_summary(argv)}"
        )
        raise ModelError(
            message,
            details={
                "returncode": returncode,
                "command": _command_summary(argv),
                "cwd": str(cwd),
                "tail": tail,
            },
        )
    return returncode


# --------------------------------------------------------------------------- #
# Commit-phase signal deferral
# --------------------------------------------------------------------------- #
class DeferringHandlers:
    """Record INT/TERM without raising until the rename sequence is finished."""

    def __init__(self) -> None:
        self.received: Optional[int] = None
        self._previous: dict[int, Any] = {}

    def __enter__(self) -> "DeferringHandlers":
        for signum in (signal.SIGINT, signal.SIGTERM):
            self._previous[signum] = signal.signal(signum, self._handle)
        return self

    def _handle(self, signum, frame) -> None:
        if self.received is None:
            self.received = signum

    def __exit__(self, *exc: Any) -> None:
        for signum, handler in self._previous.items():
            with contextlib.suppress(OSError, ValueError):
                signal.signal(signum, handler)
        self._previous.clear()

    def raise_if_deferred(
        self, *, published: bool = False, registered: bool = False,
        partial: Optional[Path] = None,
    ) -> None:
        if self.received is not None:
            raise CommitInterrupt(
                published=published, registered=registered, partial=partial
            )


def report_commit_interrupt(exc: CommitInterrupt) -> None:
    """Write the §5.3 status line to stderr before main prints ``interrupted``."""
    eprint(exc.status_line())
    if exc.partial is not None:
        eprint(f"partial kept at {exc.partial}")


def report_interrupt_state(
    *, published: bool, registered: bool, partial: Optional[Path] = None
) -> None:
    eprint(f"published: {str(published).lower()}, registered: {str(registered).lower()}")
    if partial is not None:
        eprint(f"partial kept at {partial}")


def stderr_flush() -> None:
    with contextlib.suppress(Exception):
        sys.stderr.flush()
