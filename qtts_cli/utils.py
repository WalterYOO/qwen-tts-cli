"""Small shared helpers: hashing, canonical JSON, atomic writes, output guards."""

from __future__ import annotations

import contextlib
import errno
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterator, Optional


# --------------------------------------------------------------------------- #
# Hashing / canonical JSON
# --------------------------------------------------------------------------- #
def canonical_json(obj: Any) -> str:
    """Deterministic JSON used for hashing (sorted keys, no spaces)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: str | os.PathLike[str]) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(obj: Any) -> str:
    return sha256_text(canonical_json(obj))


# --------------------------------------------------------------------------- #
# Atomic writes (PRD §9: tmp + rename)
# --------------------------------------------------------------------------- #
def _atomic_replace(path: Path, write_fn) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as fh:
            write_fn(fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise


def atomic_write_bytes(path: str | os.PathLike[str], data: bytes) -> None:
    _atomic_replace(Path(path), lambda fh: fh.write(data))


def atomic_write_text(path: str | os.PathLike[str], text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: str | os.PathLike[str], obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def read_json(path: str | os.PathLike[str]) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- #
# stdout hygiene
# --------------------------------------------------------------------------- #
@contextlib.contextmanager
def stdout_to_stderr() -> Iterator[None]:
    """Redirect file descriptor 1 to stderr for the duration of the block.

    Third-party imports (soundfile/sox banners, transformers warnings) write to
    fd 1. ``--json`` and ``--out -`` require stdout to stay machine-readable, so
    every heavy call runs under this guard.
    """
    try:
        sys.stdout.flush()
    except Exception:
        pass
    saved = os.dup(1)
    try:
        os.dup2(2, 1)
        yield
    finally:
        with contextlib.suppress(Exception):
            sys.stdout.flush()
        os.dup2(saved, 1)
        os.close(saved)


class StdoutMuted:
    """Neutralize fd 1 during module import without swallowing errors."""

    def __enter__(self) -> "StdoutMuted":
        try:
            sys.stdout.flush()
        except Exception:
            pass
        self._saved = os.dup(1)
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 1)
        os.close(devnull)
        return self

    def __exit__(self, *exc: Any) -> None:
        with contextlib.suppress(Exception):
            sys.stdout.flush()
        os.dup2(self._saved, 1)
        os.close(self._saved)


# --------------------------------------------------------------------------- #
# CLI output
# --------------------------------------------------------------------------- #
def emit_json(obj: Any) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")
    sys.stdout.flush()


def eprint(*args: Any, **kw: Any) -> None:
    print(*args, file=sys.stderr, **kw)
    sys.stderr.flush()


def is_tty() -> bool:
    try:
        return sys.stdin.isatty() and sys.stderr.isatty()
    except Exception:
        return False


def confirm(question: str, assume_yes: bool = False) -> bool:
    """Interactive confirmation; non-TTY callers must pass --yes."""
    if assume_yes:
        return True
    if not is_tty():
        return False
    sys.stderr.write(question.rstrip() + " [y/N] ")
    sys.stderr.flush()
    try:
        answer = sys.stdin.readline().strip().lower()
    except Exception:
        return False
    return answer in ("y", "yes")


def human_bytes(n: int) -> str:
    step = 1024.0
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < step or unit == "TiB":
            return f"{n:.1f}{unit}" if unit != "B" else f"{n}B"
        n /= step
    return f"{n:.1f}TiB"


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    p = Path(path).expanduser()
    p.mkdir(parents=True, exist_ok=True)
    return p


def safe_unlink(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError as exc:  # pragma: no cover - unusual fs errors
        if exc.errno != errno.ENOENT:
            raise


def rel_or_abs(path: Optional[str], base: Optional[Path] = None) -> Optional[str]:
    if path is None:
        return None
    p = Path(path).expanduser()
    try:
        return str(p.resolve())
    except OSError:
        return str(p)
