"""Idempotent marker-block installation for shell completion.

The only thing this module writes is the region between two marker lines.  A
malformed or duplicated marker always fails closed: guessing how to repair an
rc file is worse than telling the user exactly which line is wrong.
"""

from __future__ import annotations

import hashlib
import os
import re
import stat
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from ..errors import EXIT_INPUT, InputError, QttsError, UsageError
from .detect import SUPPORTED_SHELLS
from .shells import generate_script

MARKER_VERSION = 1

BEGIN_RE = re.compile(rb"^# >>> qtts completion: (bash|zsh|fish) v([0-9]+) >>>$")
END_RE = re.compile(rb"^# <<< qtts completion: (bash|zsh|fish) v([0-9]+) <<<$")
_MARKER_PREFIXES = (b"# >>> qtts completion", b"# <<< qtts completion")


class SymlinkRefused(InputError):
    """The rc target is a symlink and ``--follow-symlinks`` was not given."""


class MarkerError(InputError):
    """The rc file contains an incomplete, duplicated or inconsistent block."""


@dataclass(frozen=True)
class MarkerBlock:
    shell: str
    version: int
    start: int
    end: int
    content: bytes


@dataclass(frozen=True)
class Target:
    requested: Path
    path: Path
    symlink: bool


def _io_error(message: str, *, details: Optional[dict[str, Any]] = None) -> QttsError:
    return QttsError(message, code=3, kind="io", details=details)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def default_target(shell: str) -> Path:
    if shell not in SUPPORTED_SHELLS:
        raise UsageError(f"unsupported shell {shell!r}; supported shells: bash, zsh, fish")
    try:
        home = Path.home()
    except RuntimeError as exc:  # HOME is unset and cannot be expanded
        raise InputError(
            "cannot locate the home directory; pass --rc-file with an absolute path"
        ) from exc
    if shell == "bash":
        return home / ".bashrc"
    if shell == "zsh":
        return home / ".zshrc"
    xdg = os.environ.get("XDG_CONFIG_HOME", "").strip()
    base = Path(xdg).expanduser() if xdg else home / ".config"
    return base / "fish" / "conf.d" / "qtts.fish"


def _expand_rc_file(rc_file: Optional[str], shell: str) -> Path:
    if rc_file is None:
        return default_target(shell)
    text = str(rc_file)
    if not text.strip():
        raise UsageError("--rc-file must not be empty")
    if "\x00" in text:
        raise UsageError("--rc-file must not contain NUL")
    try:
        expanded = Path(text).expanduser()
    except RuntimeError as exc:
        raise InputError(
            "cannot expand '~' in --rc-file; pass an absolute path"
        ) from exc
    if text.startswith("~") and str(expanded).startswith("~"):
        raise InputError(
            "cannot expand '~' in --rc-file; pass an absolute path"
        )
    return expanded


def resolve_target(
    shell: str, rc_file: Optional[str] = None, *, follow_symlinks: bool = False
) -> Target:
    requested = _expand_rc_file(rc_file, shell)
    if requested.is_symlink():
        if not follow_symlinks:
            raise SymlinkRefused(
                f"refusing to write through symlink {requested}; "
                "pass --follow-symlinks to update its target",
                details={"rc_file": str(requested)},
            )
        try:
            path = requested.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise InputError(
                f"cannot resolve symlink {requested}: {exc}",
                details={"rc_file": str(requested)},
            ) from exc
        if not path.exists():
            raise InputError(
                f"symlink target does not exist: {requested} -> {path}",
                details={"rc_file": str(requested)},
            )
    else:
        path = requested

    try:
        info = path.stat()
    except FileNotFoundError:
        return Target(requested=requested, path=path, symlink=requested.is_symlink())
    except OSError as exc:
        raise _io_error(f"cannot stat {path}: {exc}", details={"rc_file": str(path)}) from exc
    if not stat.S_ISREG(info.st_mode):
        raise InputError(
            f"rc target is not a regular file: {path}",
            details={"rc_file": str(path)},
        )
    return Target(requested=requested, path=path, symlink=requested.is_symlink())


def _nearest_existing_parent(path: Path) -> Path:
    parent = path.parent
    while parent != parent.parent and not parent.exists():
        parent = parent.parent
    return parent


def check_writable(target: Target) -> None:
    if target.path.exists():
        try:
            mode = stat.S_IMODE(target.path.stat().st_mode)
        except OSError as exc:
            raise _io_error(f"cannot stat {target.path}: {exc}") from exc
        if not mode & stat.S_IWUSR:
            raise _io_error(
                f"rc file is not writable: {target.path}",
                details={"rc_file": str(target.path), "mode": oct(mode)},
            )
    parent = _nearest_existing_parent(target.path)
    if not parent.exists() or not parent.is_dir():
        raise InputError(
            f"parent of rc file is not a directory: {parent}",
            details={"rc_file": str(target.path)},
        )
    if not os.access(parent, os.W_OK | os.X_OK):
        raise _io_error(
            f"parent directory is not writable: {parent}",
            details={"rc_file": str(target.path)},
        )


def read_target(target: Target) -> bytes:
    if not target.path.exists():
        return b""
    try:
        return target.path.read_bytes()
    except OSError as exc:
        raise _io_error(f"cannot read {target.path}: {exc}") from exc


def _malformed(message: str, line_no: int, path: Path) -> MarkerError:
    return MarkerError(
        f"malformed qtts completion marker in {path}: {message} (line {line_no})",
        details={"rc_file": str(path), "line": line_no},
    )


def parse_blocks(data: bytes, path: str | Path = "<rc>") -> list[MarkerBlock]:
    """Parse qtts marker blocks; any ambiguity raises :class:`MarkerError`."""
    lines = data.splitlines(keepends=True)
    blocks: list[MarkerBlock] = []
    open_block: Optional[tuple[str, int, int, int, int]] = None
    # (shell, version, start_offset, begin_index, line_number)
    offset = 0
    seen_shells: set[str] = set()
    for index, line in enumerate(lines):
        stripped = line.rstrip(b"\r\n")
        offset_after = offset + len(line)
        start_match = BEGIN_RE.match(stripped)
        end_match = END_RE.match(stripped)
        if stripped.startswith(_MARKER_PREFIXES) and not (start_match or end_match):
            raise _malformed("marker line is incomplete or has extra text", index + 1, Path(path))
        if start_match:
            if open_block is not None:
                raise _malformed("nested BEGIN marker", index + 1, Path(path))
            shell = start_match.group(1).decode("ascii")
            version = int(start_match.group(2))
            open_block = (shell, version, offset, index, index + 1)
        elif end_match:
            if open_block is None:
                raise _malformed("END marker without BEGIN", index + 1, Path(path))
            shell, version, start, begin_index, begin_line = open_block
            end_shell = end_match.group(1).decode("ascii")
            end_version = int(end_match.group(2))
            if shell != end_shell or version != end_version:
                raise _malformed(
                    "BEGIN/END shell or version mismatch", index + 1, Path(path)
                )
            if shell in seen_shells:
                raise _malformed(
                    f"duplicate completion block for {shell}", index + 1, Path(path)
                )
            seen_shells.add(shell)
            content = b"".join(lines[begin_index + 1:index])
            blocks.append(
                MarkerBlock(
                    shell=shell,
                    version=version,
                    start=start,
                    end=offset_after,
                    content=content,
                )
            )
            open_block = None
        offset = offset_after
    if open_block is not None:
        raise _malformed("BEGIN marker without END", open_block[4], Path(path))
    return blocks


def render_block(shell: str, script: str) -> bytes:
    if shell not in SUPPORTED_SHELLS:
        raise UsageError(f"unsupported shell {shell!r}")
    if not script.endswith("\n"):
        script += "\n"
    begin = f"# >>> qtts completion: {shell} v{MARKER_VERSION} >>>\n"
    end = f"# <<< qtts completion: {shell} v{MARKER_VERSION} <<<\n"
    if f"# <<< qtts completion: {shell} v{MARKER_VERSION} <<<" in script:
        raise QttsError("generated completion script contains its own END marker", code=EXIT_INPUT)
    return begin.encode("ascii") + script.encode("utf-8") + end.encode("ascii")


def _created_directories(path: Path) -> list[Path]:
    created: list[Path] = []
    parent = path.parent
    while parent != parent.parent and not parent.exists():
        created.append(parent)
        parent = parent.parent
    return created


def atomic_write(path: Path, data: bytes, mode: int) -> None:
    created = _created_directories(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise _io_error(f"cannot create {path.parent}: {exc}") from exc

    fd: Optional[int] = None
    tmp: Optional[str] = None
    try:
        fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
        os.fchmod(fd, mode)
        with os.fdopen(fd, "wb") as handle:
            fd = None
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        tmp = None
        try:
            dir_fd = os.open(str(path.parent), os.O_RDONLY)
        except OSError:
            dir_fd = None
        if dir_fd is not None:
            try:
                os.fsync(dir_fd)
            except OSError:
                pass
            finally:
                os.close(dir_fd)
    except BaseException as exc:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
        if tmp is not None:
            try:
                os.unlink(tmp)
            except FileNotFoundError:
                pass
        for directory in created:
            try:
                directory.rmdir()
            except OSError:
                break
        if isinstance(exc, QttsError):
            raise
        if isinstance(exc, OSError):
            raise _io_error(f"cannot write {path}: {exc}", details={"rc_file": str(path)}) from exc
        raise


def _mode_for(target: Target) -> int:
    if target.path.exists():
        try:
            return stat.S_IMODE(target.path.stat().st_mode)
        except OSError as exc:
            raise _io_error(f"cannot stat {target.path}: {exc}") from exc
    return 0o644


def install(
    shell: str,
    rc_file: Optional[str] = None,
    *,
    dry_run: bool = False,
    follow_symlinks: bool = False,
) -> dict[str, Any]:
    target = resolve_target(shell, rc_file, follow_symlinks=follow_symlinks)
    old = read_target(target)
    blocks = parse_blocks(old, target.path)
    same = [block for block in blocks if block.shell == shell]
    if len(same) > 1:  # parse_blocks already rejects this; kept as a guard
        raise MarkerError(f"duplicate completion block for {shell} in {target.path}")
    if same and same[0].version > MARKER_VERSION:
        raise InputError(
            f"unsupported qtts completion marker version v{same[0].version} in "
            f"{target.path}; upgrade qtts or remove the block manually",
            details={"rc_file": str(target.path), "marker_version": same[0].version},
        )

    script = generate_script(shell)
    block = render_block(shell, script)
    if same:
        existing = same[0]
        new = old[: existing.start] + block + old[existing.end :]
        block_found = True
    else:
        separator = b"" if not old or old.endswith(b"\n") else b"\n"
        new = old + separator + block
        block_found = False

    changed = new != old
    existed = target.path.exists()
    if changed:
        check_writable(target)
        if not dry_run:
            atomic_write(target.path, new, _mode_for(target))

    warnings: list[str] = []
    if shell == "bash" and sys.platform == "darwin" and target.requested == default_target("bash"):
        warnings.append(
            "macOS Terminal may start a login shell and read ~/.bash_profile; "
            "if that file does not source ~/.bashrc, install with "
            "--rc-file ~/.bash_profile instead"
        )

    return {
        "ok": True,
        "command": "completion",
        "action": "install",
        "shell": shell,
        "rc_file": str(target.requested),
        "resolved_rc_file": str(target.path),
        "changed": changed,
        "created": changed and not existed,
        "block_found": block_found,
        "marker_version": MARKER_VERSION,
        "before_sha256": sha256_bytes(old) if existed else None,
        "after_sha256": sha256_bytes(new),
        "script_sha256": sha256_bytes(script.encode("utf-8")),
        "dry_run": dry_run,
        "warnings": warnings,
    }


def _is_default_fish_target(target: Target) -> bool:
    try:
        return target.path == default_target("fish")
    except (InputError, UsageError):
        return False


def uninstall(
    shell: str,
    rc_file: Optional[str] = None,
    *,
    dry_run: bool = False,
    follow_symlinks: bool = False,
    explicit_rc: bool = False,
) -> dict[str, Any]:
    target = resolve_target(shell, rc_file, follow_symlinks=follow_symlinks)
    if not target.path.exists():
        return {
            "ok": True,
            "command": "completion",
            "action": "uninstall",
            "shell": shell,
            "rc_file": str(target.requested),
            "resolved_rc_file": str(target.path),
            "changed": False,
            "removed": False,
            "block_found": False,
            "marker_version": None,
            "before_sha256": None,
            "after_sha256": None,
            "deleted_file": False,
            "dry_run": dry_run,
        }

    old = read_target(target)
    blocks = parse_blocks(old, target.path)
    same = [block for block in blocks if block.shell == shell]
    if not same:
        return {
            "ok": True,
            "command": "completion",
            "action": "uninstall",
            "shell": shell,
            "rc_file": str(target.requested),
            "resolved_rc_file": str(target.path),
            "changed": False,
            "removed": False,
            "block_found": False,
            "marker_version": None,
            "before_sha256": sha256_bytes(old),
            "after_sha256": sha256_bytes(old),
            "deleted_file": False,
            "dry_run": dry_run,
        }

    existing = same[0]
    if existing.version > MARKER_VERSION:
        raise InputError(
            f"unsupported qtts completion marker version v{existing.version} in "
            f"{target.path}; upgrade qtts or remove the block manually",
            details={"rc_file": str(target.path), "marker_version": existing.version},
        )
    new = old[: existing.start] + old[existing.end :]
    deleted_file = (
        not new.strip()
        and rc_file is None
        and not explicit_rc
        and _is_default_fish_target(target)
    )
    check_writable(target)
    if not dry_run:
        if deleted_file:
            try:
                target.path.unlink()
            except OSError as exc:
                raise _io_error(f"cannot remove {target.path}: {exc}") from exc
        else:
            atomic_write(target.path, new, _mode_for(target))

    return {
        "ok": True,
        "command": "completion",
        "action": "uninstall",
        "shell": shell,
        "rc_file": str(target.requested),
        "resolved_rc_file": str(target.path),
        "changed": True,
        "removed": True,
        "block_found": True,
        "marker_version": existing.version,
        "before_sha256": sha256_bytes(old),
        "after_sha256": None if deleted_file else sha256_bytes(new),
        "deleted_file": deleted_file,
        "dry_run": dry_run,
    }


def _status_payload(
    shell: str,
    target: Target,
    *,
    state: str,
    expected_sha256: str,
    installed: bool,
    marker_version: Optional[int] = None,
    actual_sha256: Optional[str] = None,
    other_blocks: Optional[Iterable[MarkerBlock]] = None,
    message: Optional[str] = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "ok": state == "current",
        "command": "completion",
        "action": "status",
        "shell": shell,
        "rc_file": str(target.requested),
        "resolved_rc_file": str(target.path),
        "installed": installed,
        "state": state,
        "marker_version": marker_version,
        "expected_sha256": expected_sha256,
        "actual_sha256": actual_sha256,
        "other_blocks": [
            {"shell": block.shell, "version": block.version}
            for block in (other_blocks or [])
        ],
        "exit_code": 0 if state == "current" else EXIT_INPUT,
    }
    if message:
        payload["message"] = message
    return payload


def status(
    shell: str,
    rc_file: Optional[str] = None,
    *,
    follow_symlinks: bool = False,
) -> dict[str, Any]:
    expected_sha256 = sha256_bytes(generate_script(shell).encode("utf-8"))
    requested = _expand_rc_file(rc_file, shell)
    if requested.is_symlink() and not follow_symlinks:
        return _status_payload(
            shell,
            Target(requested=requested, path=requested, symlink=True),
            state="symlink_refused",
            expected_sha256=expected_sha256,
            installed=False,
        )

    try:
        target = resolve_target(shell, rc_file, follow_symlinks=follow_symlinks)
    except SymlinkRefused:
        return _status_payload(
            shell,
            Target(requested=requested, path=requested, symlink=True),
            state="symlink_refused",
            expected_sha256=expected_sha256,
            installed=False,
        )
    except InputError as exc:
        return _status_payload(
            shell,
            Target(requested=requested, path=requested, symlink=requested.is_symlink()),
            state="malformed",
            expected_sha256=expected_sha256,
            installed=False,
            message=exc.message,
        )

    if not target.path.exists():
        return _status_payload(
            shell,
            target,
            state="not_installed",
            expected_sha256=expected_sha256,
            installed=False,
        )

    try:
        data = read_target(target)
    except QttsError:
        raise
    try:
        blocks = parse_blocks(data, target.path)
    except MarkerError as exc:
        return _status_payload(
            shell,
            target,
            state="malformed",
            expected_sha256=expected_sha256,
            installed=False,
            actual_sha256=sha256_bytes(data),
            message=exc.message,
        )

    same = [block for block in blocks if block.shell == shell]
    if not same:
        return _status_payload(
            shell,
            target,
            state="not_installed",
            expected_sha256=expected_sha256,
            installed=False,
            actual_sha256=sha256_bytes(data),
            other_blocks=blocks,
        )

    block = same[0]
    actual_sha256 = sha256_bytes(block.content)
    if block.version != MARKER_VERSION:
        state = "unsupported_version"
    elif block.content == generate_script(shell).encode("utf-8"):
        state = "current"
    else:
        state = "stale"
    return _status_payload(
        shell,
        target,
        state=state,
        expected_sha256=expected_sha256,
        installed=True,
        marker_version=block.version,
        actual_sha256=actual_sha256,
        other_blocks=[item for item in blocks if item.shell != shell],
    )
