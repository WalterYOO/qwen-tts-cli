"""Idempotent installation for shell completion.

Completion scripts live in their own files (``_qtts`` for zsh, a
``bash-completion`` drop-in for bash, or ``fish/completions/qtts.fish``).
Only when the shell cannot autoload that directory do we add a small marker
block to the user's rc file, and that block contains two setup lines rather
than the whole completion script.  A malformed or duplicated marker always
fails closed: guessing how to repair an rc file is worse than telling the user
exactly which line is wrong.
"""

from __future__ import annotations

import hashlib
import os
import re
import shlex
import stat
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from ..errors import EXIT_INPUT, InputError, QttsError, UsageError
from .detect import SUPPORTED_SHELLS
from .shells import generate_script

MARKER_VERSION = 2
_SCRIPT_MARKER = b"# qtts-generated completion file"

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


def _home() -> Path:
    try:
        return Path.home()
    except RuntimeError as exc:  # HOME is unset and cannot be expanded
        raise InputError(
            "cannot locate the home directory; pass --rc-file with an absolute path"
        ) from exc


def _xdg_data() -> Path:
    value = os.environ.get("XDG_DATA_HOME", "").strip()
    return Path(value).expanduser() if value else _home() / ".local" / "share"


def _xdg_config() -> Path:
    value = os.environ.get("XDG_CONFIG_HOME", "").strip()
    return Path(value).expanduser() if value else _home() / ".config"


def _zsh_base() -> Path:
    value = os.environ.get("ZDOTDIR", "").strip()
    return Path(value).expanduser() if value else _home()


def default_rc_path(shell: str) -> Path:
    """Default rc file for the small setup block, not the completion script."""
    if shell not in SUPPORTED_SHELLS:
        raise UsageError(f"unsupported shell {shell!r}; supported shells: bash, zsh, fish")
    if shell == "bash":
        rc = _home() / ".bashrc"
        if rc.exists():
            return rc
        profile = _home() / ".bash_profile"
        return profile if profile.exists() else rc
    if shell == "zsh":
        return _zsh_base() / ".zshrc"
    return _xdg_config() / "fish" / "config.fish"


def _script_name(shell: str) -> str:
    return {"bash": "qtts", "zsh": "_qtts", "fish": "qtts.fish"}[shell]


def _writable_dir(directory: Path) -> bool:
    try:
        if directory.exists():
            return directory.is_dir() and os.access(directory, os.W_OK | os.X_OK)
        return directory.parent.is_dir() and os.access(directory.parent, os.W_OK | os.X_OK)
    except OSError:
        return False


def _zsh_config_files() -> list[Path]:
    base = _zsh_base()
    return [base / name for name in (".zshrc", ".zshenv", ".zprofile", ".zlogin")]


_FPATH_RE = re.compile(r"fpath\s*\+?=\s*\(([^)]*)\)")


def _strip_managed_blocks(text: str) -> str:
    return re.sub(
        r"^# >>> qtts completion: zsh v[0-9]+ >>>.*?^# <<< qtts completion: zsh v[0-9]+ <<<\n?",
        "",
        text,
        flags=re.DOTALL | re.MULTILINE,
    )


def _fpath_dirs_from_text(text: str, config: Path) -> list[Path]:
    directories: list[Path] = []
    for match in _FPATH_RE.finditer(text):
        try:
            parts = shlex.split(match.group(1))
        except ValueError:
            continue
        for part in parts:
            if part in {"$fpath", "${fpath}"}:
                continue
            expanded = os.path.expandvars(os.path.expanduser(part))
            if "$" in expanded:
                continue
            path = Path(expanded)
            if not path.is_absolute():
                path = config.parent / path
            directories.append(path)
    return directories


def _zsh_fpath_dirs() -> list[Path]:
    directories: list[Path] = []
    seen: set[str] = set()
    for config in _zsh_config_files():
        try:
            text = config.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for directory in _fpath_dirs_from_text(_strip_managed_blocks(text), config):
            key = str(directory)
            if key not in seen:
                seen.add(key)
                directories.append(directory)
    return directories


def _oh_my_zsh_dir() -> Optional[Path]:
    custom = os.environ.get("ZSH_CUSTOM", "").strip()
    if custom:
        return Path(custom).expanduser() / "completions"
    omz = _home() / ".oh-my-zsh"
    if omz.is_dir():
        return omz / "custom" / "completions"
    return None


_BASH_COMPLETION_MARKERS = (
    "/usr/share/bash-completion/bash_completion",
    "/etc/bash_completion",
    "/opt/homebrew/etc/profile.d/bash_completion.sh",
    "/usr/local/etc/profile.d/bash_completion.sh",
    "/usr/share/bash-completion/completions",
)


def _bash_completion_installed() -> bool:
    return any(Path(marker).exists() for marker in _BASH_COMPLETION_MARKERS)


def _bash_script_path() -> tuple[Path, bool]:
    """Return ``(script_path, needs_rc_block)``."""
    user_dir = os.environ.get("BASH_COMPLETION_USER_DIR", "").strip()
    if user_dir:
        return Path(user_dir).expanduser() / "completions" / "qtts", False
    if _bash_completion_installed():
        return _xdg_data() / "bash-completion" / "completions" / "qtts", False
    return _xdg_data() / "qtts" / "completion.bash", True


def _zsh_script_path() -> tuple[Path, bool]:
    """Return ``(script_path, needs_rc_block)``."""
    omz = _oh_my_zsh_dir()
    if omz is not None:
        return omz / "_qtts", False
    for directory in _zsh_fpath_dirs():
        if _writable_dir(directory):
            return directory / "_qtts", False
    return _home() / ".zfunc" / "_qtts", True


def default_script_path(shell: str) -> Path:
    if shell not in SUPPORTED_SHELLS:
        raise UsageError(f"unsupported shell {shell!r}; supported shells: bash, zsh, fish")
    if shell == "bash":
        return _bash_script_path()[0]
    if shell == "zsh":
        return _zsh_script_path()[0]
    return _xdg_config() / "fish" / "completions" / "qtts.fish"


def _expand_rc_file(rc_file: Optional[str], shell: str) -> Path:
    if rc_file is None:
        return default_rc_path(shell)
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


def _resolve_path(
    requested: Path, *, follow_symlinks: bool, label: str
) -> Target:
    if requested.is_symlink():
        if not follow_symlinks:
            raise SymlinkRefused(
                f"refusing to write through symlink {requested}; "
                "pass --follow-symlinks to update its target",
                details={label: str(requested)},
            )
        try:
            path = requested.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise InputError(
                f"cannot resolve symlink {requested}: {exc}",
                details={label: str(requested)},
            ) from exc
        if not path.exists():
            raise InputError(
                f"symlink target does not exist: {requested} -> {path}",
                details={label: str(requested)},
            )
    else:
        path = requested

    try:
        info = path.stat()
    except FileNotFoundError:
        return Target(requested=requested, path=path, symlink=requested.is_symlink())
    except OSError as exc:
        raise _io_error(f"cannot stat {path}: {exc}", details={label: str(path)}) from exc
    if not stat.S_ISREG(info.st_mode):
        raise InputError(
            f"{label} is not a regular file: {path}",
            details={label: str(path)},
        )
    return Target(requested=requested, path=path, symlink=requested.is_symlink())


def resolve_target(
    shell: str, rc_file: Optional[str] = None, *, follow_symlinks: bool = False
) -> Target:
    return _resolve_path(
        _expand_rc_file(rc_file, shell),
        follow_symlinks=follow_symlinks,
        label="rc_file",
    )


@dataclass(frozen=True)
class Plan:
    shell: str
    script: Target
    rc: Optional[Target]
    warnings: tuple[str, ...] = ()
    directory_only: bool = False


def resolve_plan(
    shell: str,
    rc_file: Optional[str] = None,
    *,
    script_dir: Optional[str] = None,
    follow_symlinks: bool = False,
) -> Plan:
    if script_dir is not None and rc_file is not None:
        raise UsageError("--dir and --rc-file are mutually exclusive")
    if script_dir is not None:
        text = str(script_dir)
        if not text.strip():
            raise UsageError("--dir must not be empty")
        if "\x00" in text:
            raise UsageError("--dir must not contain NUL")
        try:
            directory = Path(text).expanduser()
        except RuntimeError as exc:
            raise InputError("cannot expand '~' in --dir; pass an absolute path") from exc
        if text.startswith("~") and str(directory).startswith("~"):
            raise InputError("cannot expand '~' in --dir; pass an absolute path")
        script_path = directory / _script_name(shell)
        script = _resolve_path(
            script_path, follow_symlinks=follow_symlinks, label="completion file"
        )
        return Plan(shell=shell, script=script, rc=None, directory_only=True)

    warnings: list[str] = []
    if shell == "bash":
        path, needs_rc = _bash_script_path()
    elif shell == "zsh":
        path, needs_rc = _zsh_script_path()
    else:
        path, needs_rc = _xdg_config() / "fish" / "completions" / "qtts.fish", False

    rc: Optional[Target] = None
    if rc_file is not None:
        rc = resolve_target(shell, rc_file, follow_symlinks=follow_symlinks)
    elif needs_rc:
        rc = resolve_target(shell, None, follow_symlinks=follow_symlinks)

    if shell == "bash" and sys.platform == "darwin" and rc is not None:
        if rc.requested == default_rc_path("bash"):
            warnings.append(
                "macOS Terminal may start a login shell and read ~/.bash_profile; "
                "if that file does not source ~/.bashrc, install with "
                "--rc-file ~/.bash_profile instead"
            )

    script = _resolve_path(
        path, follow_symlinks=follow_symlinks, label="completion file"
    )
    return Plan(shell=shell, script=script, rc=rc, warnings=tuple(warnings))


def _nearest_existing_parent(path: Path) -> Path:
    parent = path.parent
    while parent != parent.parent and not parent.exists():
        parent = parent.parent
    return parent


def check_writable(target: Target, *, label: str = "rc file") -> None:
    if target.path.exists():
        try:
            mode = stat.S_IMODE(target.path.stat().st_mode)
        except OSError as exc:
            raise _io_error(f"cannot stat {target.path}: {exc}") from exc
        if not mode & stat.S_IWUSR:
            raise _io_error(
                f"{label} is not writable: {target.path}",
                details={"rc_file": str(target.path), "mode": oct(mode)},
            )
    parent = _nearest_existing_parent(target.path)
    if not parent.exists() or not parent.is_dir():
        raise InputError(
            f"parent of {label} is not a directory: {parent}",
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


def script_payload(shell: str) -> bytes:
    """Return the standalone completion file, including its ownership marker."""
    script = generate_script(shell)
    if not script.endswith("\n"):
        script += "\n"
    marker = (
        f"# qtts-generated completion file; remove with: "
        f"qtts completion --uninstall {shell}\n"
    )
    return (script + marker).encode("utf-8")


def render_script(shell: str) -> bytes:
    return script_payload(shell)


def block_body(shell: str, script: Path) -> str:
    quoted = shlex.quote(str(script))
    if shell == "zsh":
        directory = shlex.quote(str(script.parent))
        return (
            f"fpath=({directory} $fpath)\n"
            "autoload -Uz compinit && compinit\n"
        )
    if shell == "fish":
        return f"source {quoted}\n"
    return f"if [ -r {quoted} ]; then . {quoted}; fi\n"


def render_block(shell: str, body: str) -> bytes:
    if shell not in SUPPORTED_SHELLS:
        raise UsageError(f"unsupported shell {shell!r}")
    if not body.endswith("\n"):
        body += "\n"
    begin = f"# >>> qtts completion: {shell} v{MARKER_VERSION} >>>\n"
    end = f"# <<< qtts completion: {shell} v{MARKER_VERSION} <<<\n"
    if f"# <<< qtts completion: {shell} v{MARKER_VERSION} <<<" in body:
        raise QttsError("generated completion block contains its own END marker", code=EXIT_INPUT)
    return begin.encode("ascii") + body.encode("utf-8") + end.encode("ascii")


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


def _read_bytes(path: Path) -> bytes:
    if not path.exists():
        return b""
    try:
        return path.read_bytes()
    except OSError as exc:
        raise _io_error(f"cannot read {path}: {exc}") from exc


def _remove_script_file(path: Path, *, dry_run: bool) -> bool:
    """Remove a completion file only when it carries the qtts marker."""
    if path.is_symlink() or not path.is_file():
        return False
    try:
        data = path.read_bytes()
    except OSError:
        return False
    if _SCRIPT_MARKER not in data and b"# >>> qtts completion:" not in data:
        return False
    if not dry_run:
        try:
            path.unlink()
        except FileNotFoundError:
            return False
        except OSError as exc:
            raise _io_error(f"cannot remove {path}: {exc}") from exc
    return True


def _known_script_targets(shell: str) -> list[Path]:
    targets: list[Path] = []
    if shell == "zsh":
        targets.extend(
            [
                _home() / ".zfunc" / "_qtts",
                _home() / ".zsh" / "completions" / "_qtts",
                _home() / ".config" / "zsh" / "completions" / "_qtts",
            ]
        )
        omz = _oh_my_zsh_dir()
        if omz is not None:
            targets.append(omz / "_qtts")
    elif shell == "bash":
        targets.extend(
            [
                _xdg_data() / "bash-completion" / "completions" / "qtts",
                _xdg_data() / "qtts" / "completion.bash",
            ]
        )
        user_dir = os.environ.get("BASH_COMPLETION_USER_DIR", "").strip()
        if user_dir:
            targets.append(Path(user_dir).expanduser() / "completions" / "qtts")
    else:
        targets.extend(
            [
                _xdg_config() / "fish" / "completions" / "qtts.fish",
                _xdg_config() / "fish" / "conf.d" / "qtts.fish",
            ]
        )
    return targets


def _rc_files_for(shell: str) -> list[Path]:
    if shell == "zsh":
        return _zsh_config_files()
    if shell == "bash":
        return [_home() / ".bashrc", _home() / ".bash_profile"]
    return []


def _remove_rc_block(
    target: Target, shell: str, *, dry_run: bool
) -> tuple[bool, bool, Optional[int]]:
    """Remove one shell's block. Return ``(changed, found, version)``."""
    if not target.path.exists():
        return False, False, None
    old = read_target(target)
    blocks = parse_blocks(old, target.path)
    same = [block for block in blocks if block.shell == shell]
    if not same:
        return False, False, None
    existing = same[0]
    if existing.version > MARKER_VERSION:
        raise InputError(
            f"unsupported qtts completion marker version v{existing.version} in "
            f"{target.path}; upgrade qtts or remove the block manually",
            details={"rc_file": str(target.path), "marker_version": existing.version},
        )
    new = old[: existing.start] + old[existing.end :]
    if new != old:
        check_writable(target)
        if not dry_run:
            atomic_write(target.path, new, _mode_for(target))
    return new != old, True, existing.version


def _cleanup_stale(
    shell: str,
    keep_script: Path,
    *,
    dry_run: bool,
    keep_rc: Optional[Path] = None,
) -> tuple[bool, list[Path], list[Path]]:
    """Remove completion files/blocks left by an earlier install layout."""
    changed = False
    removed_scripts: list[Path] = []
    removed_rc: list[Path] = []
    # Validate every candidate rc before deleting any older script file, so a
    # malformed or newer block cannot leave the install half-migrated.
    rc_targets: list[tuple[Path, Target]] = []
    for rc_path in _rc_files_for(shell):
        try:
            # Cleanup only removes qtts-owned marker blocks; following a
            # symlinked rc here is what lets a dotfiles-managed .zshrc shed
            # an older inline block after migrating to a fpath/OMZ install.
            target = _resolve_path(
                rc_path, follow_symlinks=True, label="rc_file"
            )
        except QttsError:
            continue
        if keep_rc is not None and (rc_path == keep_rc or target.path == keep_rc):
            continue
        if not target.path.exists():
            continue
        blocks = parse_blocks(read_target(target), target.path)
        same = [block for block in blocks if block.shell == shell]
        if same and same[0].version > MARKER_VERSION:
            raise InputError(
                f"unsupported qtts completion marker version v{same[0].version} "
                f"in {target.path}; upgrade qtts or remove the block manually",
                details={
                    "rc_file": str(target.path),
                    "marker_version": same[0].version,
                },
            )
        if same:
            if not dry_run:
                check_writable(target, label="rc file")
            rc_targets.append((rc_path, target))
    for path in _known_script_targets(shell):
        if path != keep_script and _remove_script_file(path, dry_run=dry_run):
            changed = True
            removed_scripts.append(path)
    for rc_path, target in rc_targets:
        removed, _, _ = _remove_rc_block(target, shell, dry_run=dry_run)
        if removed:
            changed = True
            removed_rc.append(rc_path)
    return changed, removed_scripts, removed_rc


def _stale_rc_blocks(
    shell: str, *, keep_rc: Optional[Path] = None
) -> tuple[list[Path], list[Path]]:
    """Find this shell's blocks left in rc files other than ``keep_rc``."""
    stale: list[Path] = []
    malformed: list[Path] = []
    for rc_path in _rc_files_for(shell):
        try:
            target = _resolve_path(rc_path, follow_symlinks=True, label="rc_file")
        except QttsError:
            continue
        if keep_rc is not None and (rc_path == keep_rc or target.path == keep_rc):
            continue
        if not target.path.exists():
            continue
        try:
            blocks = parse_blocks(read_target(target), target.path)
        except MarkerError:
            malformed.append(rc_path)
            continue
        except QttsError:
            continue
        if any(block.shell == shell for block in blocks):
            stale.append(rc_path)
    return stale, malformed


def install(
    shell: str,
    rc_file: Optional[str] = None,
    *,
    dry_run: bool = False,
    follow_symlinks: bool = False,
    script_dir: Optional[str] = None,
) -> dict[str, Any]:
    plan = resolve_plan(
        shell,
        rc_file,
        script_dir=script_dir,
        follow_symlinks=follow_symlinks,
    )
    old_script = _read_bytes(plan.script.path)
    script_existed = plan.script.path.exists()
    if script_existed and _SCRIPT_MARKER not in old_script:
        raise InputError(
            f"completion file already exists and was not created by qtts: "
            f"{plan.script.path}",
            details={"script_file": str(plan.script.path)},
        )
    new_script = script_payload(shell)
    script_changed = new_script != old_script

    rc_changed = False
    block_found = False
    new_rc: Optional[bytes] = None
    if plan.rc is not None:
        old_rc = read_target(plan.rc)
        blocks = parse_blocks(old_rc, plan.rc.path)
        same = [block for block in blocks if block.shell == shell]
        if len(same) > 1:  # parse_blocks already rejects this
            raise MarkerError(f"duplicate completion block for {shell} in {plan.rc.path}")
        if same and same[0].version > MARKER_VERSION:
            raise InputError(
                f"unsupported qtts completion marker version v{same[0].version} in "
                f"{plan.rc.path}; upgrade qtts or remove the block manually",
                details={"rc_file": str(plan.rc.path), "marker_version": same[0].version},
            )
        block = render_block(shell, block_body(shell, plan.script.path))
        if same:
            existing = same[0]
            new_rc = old_rc[: existing.start] + block + old_rc[existing.end :]
            block_found = True
        else:
            separator = b"" if not old_rc or old_rc.endswith(b"\n") else b"\n"
            new_rc = old_rc + separator + block
        rc_changed = new_rc != old_rc

    if not dry_run:
        # Validate every destination before cleanup or any write, so a
        # read-only rc cannot leave the install half-migrated.
        if script_changed:
            check_writable(plan.script, label="completion file")
        if plan.rc is not None and rc_changed:
            check_writable(plan.rc)

    changed = script_changed or rc_changed
    if not plan.directory_only and rc_file is None:
        stale_changed, _, _ = _cleanup_stale(
            shell,
            plan.script.path,
            dry_run=dry_run,
            keep_rc=plan.rc.path if plan.rc is not None else None,
        )
        changed = changed or stale_changed
    if not dry_run:
        if script_changed:
            atomic_write(plan.script.path, new_script, _mode_for(plan.script))
        if plan.rc is not None and rc_changed:
            assert new_rc is not None
            atomic_write(plan.rc.path, new_rc, _mode_for(plan.rc))

    return {
        "ok": True,
        "command": "completion",
        "action": "install",
        "shell": shell,
        "script_file": str(plan.script.requested),
        "resolved_script_file": str(plan.script.path),
        "rc_file": str(plan.rc.requested) if plan.rc is not None else None,
        "resolved_rc_file": str(plan.rc.path) if plan.rc is not None else None,
        "changed": changed,
        "created": script_changed and not script_existed,
        "script_changed": script_changed,
        "script_created": script_changed and not script_existed,
        "rc_changed": rc_changed,
        "block_found": block_found,
        "marker_version": MARKER_VERSION,
        "before_sha256": sha256_bytes(old_script) if script_existed else None,
        "after_sha256": sha256_bytes(new_script),
        "script_sha256": sha256_bytes(new_script),
        "dry_run": dry_run,
        "warnings": list(plan.warnings),
        "directory_only": plan.directory_only,
    }


def uninstall(
    shell: str,
    rc_file: Optional[str] = None,
    *,
    dry_run: bool = False,
    follow_symlinks: bool = False,
    explicit_rc: bool = False,
    script_dir: Optional[str] = None,
) -> dict[str, Any]:
    plan = resolve_plan(
        shell,
        rc_file,
        script_dir=script_dir,
        follow_symlinks=follow_symlinks,
    )
    script_removed = _remove_script_file(plan.script.path, dry_run=dry_run)
    rc_removed = False
    block_found = False
    marker_version: Optional[int] = None
    rc_files_removed: list[str] = []
    if plan.rc is not None:
        rc_removed, block_found, marker_version = _remove_rc_block(
            plan.rc, shell, dry_run=dry_run
        )
        if rc_removed:
            rc_files_removed.append(str(plan.rc.requested))
    stale_changed = False
    if rc_file is None and script_dir is None:
        stale_changed, stale_scripts, stale_rc = _cleanup_stale(
            shell, plan.script.path, dry_run=dry_run
        )
        for path in stale_scripts:
            script_removed = True
        for path in stale_rc:
            if str(path) not in rc_files_removed:
                rc_files_removed.append(str(path))
        rc_removed = rc_removed or bool(stale_rc)

    changed = script_removed or rc_removed or stale_changed
    return {
        "ok": True,
        "command": "completion",
        "action": "uninstall",
        "shell": shell,
        "script_file": str(plan.script.requested),
        "resolved_script_file": str(plan.script.path),
        "rc_file": str(plan.rc.requested) if plan.rc is not None else None,
        "resolved_rc_file": str(plan.rc.path) if plan.rc is not None else None,
        "changed": changed,
        "removed": changed,
        "script_removed": script_removed,
        "rc_removed": rc_removed,
        "block_found": block_found,
        "marker_version": marker_version,
        "before_sha256": None,
        "after_sha256": None,
        "deleted_file": script_removed,
        "dry_run": dry_run,
        "rc_files_removed": rc_files_removed,
    }


def _status_payload(
    shell: str,
    *,
    script_file: Optional[Path],
    resolved_script_file: Optional[Path],
    rc_file: Optional[Path],
    resolved_rc_file: Optional[Path],
    state: str,
    expected_sha256: str,
    installed: bool,
    script_installed: bool,
    rc_required: bool,
    rc_block_found: bool,
    rc_state: str,
    marker_version: Optional[int] = None,
    actual_sha256: Optional[str] = None,
    other_blocks: Optional[Iterable[MarkerBlock]] = None,
    stale_rc_files: Optional[Iterable[Path]] = None,
    malformed_rc_files: Optional[Iterable[Path]] = None,
    message: Optional[str] = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "ok": state == "current",
        "command": "completion",
        "action": "status",
        "shell": shell,
        "script_file": str(script_file) if script_file is not None else None,
        "resolved_script_file": (
            str(resolved_script_file) if resolved_script_file is not None else None
        ),
        "rc_file": str(rc_file) if rc_file is not None else None,
        "resolved_rc_file": (
            str(resolved_rc_file) if resolved_rc_file is not None else None
        ),
        "installed": installed,
        "script_installed": script_installed,
        "rc_required": rc_required,
        "rc_block_found": rc_block_found,
        "rc_state": rc_state,
        "state": state,
        "marker_version": marker_version,
        "expected_sha256": expected_sha256,
        "actual_sha256": actual_sha256,
        "other_blocks": [
            {"shell": block.shell, "version": block.version}
            for block in (other_blocks or [])
        ],
        "stale_rc_files": [str(path) for path in (stale_rc_files or [])],
        "malformed_rc_files": [
            str(path) for path in (malformed_rc_files or [])
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
    expected_sha256 = sha256_bytes(script_payload(shell))
    try:
        plan = resolve_plan(shell, rc_file, follow_symlinks=follow_symlinks)
    except SymlinkRefused as exc:
        return _status_payload(
            shell,
            script_file=default_script_path(shell),
            resolved_script_file=None,
            rc_file=_expand_rc_file(rc_file, shell),
            resolved_rc_file=None,
            state="symlink_refused",
            expected_sha256=expected_sha256,
            installed=False,
            script_installed=False,
            rc_required=False,
            rc_block_found=False,
            rc_state="symlink_refused",
            message=exc.message,
        )
    except InputError as exc:
        return _status_payload(
            shell,
            script_file=default_script_path(shell),
            resolved_script_file=None,
            rc_file=_expand_rc_file(rc_file, shell),
            resolved_rc_file=None,
            state="malformed",
            expected_sha256=expected_sha256,
            installed=False,
            script_installed=False,
            rc_required=rc_file is not None,
            rc_block_found=False,
            rc_state="malformed",
            message=exc.message,
        )

    rc_required = plan.rc is not None
    if not plan.script.path.exists():
        return _status_payload(
            shell,
            script_file=plan.script.requested,
            resolved_script_file=plan.script.path,
            rc_file=plan.rc.requested if plan.rc is not None else None,
            resolved_rc_file=plan.rc.path if plan.rc is not None else None,
            state="not_installed",
            expected_sha256=expected_sha256,
            installed=False,
            script_installed=False,
            rc_required=rc_required,
            rc_block_found=False,
            rc_state="not_required" if not rc_required else "not_installed",
        )

    data = _read_bytes(plan.script.path)
    script_installed = _SCRIPT_MARKER in data
    script_current = data == script_payload(shell)

    rc_state = "not_required"
    rc_block_found = False
    rc_ok = True
    marker_version: Optional[int] = None
    other_blocks: list[MarkerBlock] = []
    if plan.rc is not None:
        rc_state = "not_installed"
        rc_ok = False
        if plan.rc.path.exists():
            rc_data = read_target(plan.rc)
            try:
                blocks = parse_blocks(rc_data, plan.rc.path)
            except MarkerError as exc:
                return _status_payload(
                    shell,
                    script_file=plan.script.requested,
                    resolved_script_file=plan.script.path,
                    rc_file=plan.rc.requested,
                    resolved_rc_file=plan.rc.path,
                    state="malformed",
                    expected_sha256=expected_sha256,
                    installed=False,
                    script_installed=script_installed,
                    rc_required=True,
                    rc_block_found=False,
                    rc_state="malformed",
                    actual_sha256=sha256_bytes(data),
                    message=exc.message,
                )
            same = [block for block in blocks if block.shell == shell]
            other_blocks = [item for item in blocks if item.shell != shell]
            if same:
                rc_block_found = True
                marker_version = same[0].version
                expected_body = (
                    block_body(shell, plan.script.path).encode("utf-8")
                )
                if same[0].version > MARKER_VERSION:
                    rc_state = "unsupported_version"
                elif same[0].content == expected_body:
                    rc_state = "current"
                    rc_ok = True
                else:
                    rc_state = "stale"
            else:
                rc_state = "not_installed"

    if rc_file is None:
        stale_rc_files, malformed_rc_files = _stale_rc_blocks(
            shell, keep_rc=plan.rc.path if plan.rc is not None else None
        )
    else:
        stale_rc_files, malformed_rc_files = [], []
    if not script_current:
        state = "stale"
        message = None
        if not script_installed:
            message = "completion file was not created by qtts"
    elif not rc_ok:
        state = (
            rc_state
            if rc_state in ("unsupported_version", "malformed")
            else "stale"
        )
        message = None
    elif malformed_rc_files:
        state = "malformed"
        message = "malformed completion marker in " + ", ".join(
            str(path) for path in malformed_rc_files
        )
    elif stale_rc_files:
        state = "stale"
        message = "stale completion block in " + ", ".join(
            str(path) for path in stale_rc_files
        )
    else:
        state = "current"
        message = None

    return _status_payload(
        shell,
        script_file=plan.script.requested,
        resolved_script_file=plan.script.path,
        rc_file=plan.rc.requested if plan.rc is not None else None,
        resolved_rc_file=plan.rc.path if plan.rc is not None else None,
        state=state,
        expected_sha256=expected_sha256,
        installed=(
            script_installed
            and rc_ok
            and not stale_rc_files
            and not malformed_rc_files
        ),
        script_installed=script_installed,
        rc_required=rc_required,
        rc_block_found=rc_block_found,
        rc_state=rc_state,
        marker_version=marker_version,
        actual_sha256=sha256_bytes(data),
        other_blocks=other_blocks,
        stale_rc_files=stale_rc_files,
        malformed_rc_files=malformed_rc_files,
        message=message,
    )
