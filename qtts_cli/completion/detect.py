"""Shell detection for ``qtts completion``.

Detection is deliberately conservative: an explicit shell (positional or
``QTTS_COMPLETION_SHELL``) always wins, then ``$SHELL``, then the parent
process chain, then the platform default.  We never guess that ``sh`` means
bash; a wrong adapter silently breaks completion.
"""

from __future__ import annotations

import os
import platform
import subprocess
from pathlib import Path
from typing import Optional

from ..errors import UsageError

SUPPORTED_SHELLS = ("bash", "zsh", "fish")
PLANNED_SHELLS = ("powershell", "pwsh")
_ALL_NAMES = set(SUPPORTED_SHELLS) | set(PLANNED_SHELLS)


def _basename(value: str) -> str:
    text = str(value).strip()
    if not text:
        return ""
    # Login shells are often reported as "-zsh"; Windows uses "bash.exe".
    name = Path(text).name.lower()
    if name.endswith(".exe"):
        name = name[:-4]
    return name.lstrip("-")


def _supported_message() -> str:
    return "supported shells: bash, zsh, fish"


def normalize_shell(value: str) -> str:
    """Return a supported shell name or raise a usage error (exit 2)."""
    name = _basename(value)
    if name in SUPPORTED_SHELLS:
        return name
    if name in PLANNED_SHELLS:
        raise UsageError(
            f"unsupported shell {name!r}; {_supported_message()} "
            "(PowerShell is planned but not implemented in this release)"
        )
    raise UsageError(f"unknown shell {value!r}; {_supported_message()}")


def _parent_shell() -> Optional[str]:
    """Walk the parent process chain (max four hops) looking for a shell."""
    pid = os.getppid()
    for _ in range(4):
        if pid <= 1:
            return None
        try:
            completed = subprocess.run(
                ["ps", "-p", str(pid), "-o", "comm="],
                capture_output=True,
                text=True,
                timeout=0.2,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        name = _basename(completed.stdout)
        if name in SUPPORTED_SHELLS:
            return name
        try:
            parent = subprocess.run(
                ["ps", "-p", str(pid), "-o", "ppid="],
                capture_output=True,
                text=True,
                timeout=0.2,
                check=False,
            )
            pid = int(parent.stdout.strip() or "0")
        except (OSError, ValueError, subprocess.SubprocessError):
            return None
    return None


def _platform_shell() -> Optional[str]:
    system = platform.system()
    if system == "Darwin":
        return "zsh"
    if system == "Linux":
        return "bash"
    return None


def detect_shell(explicit: Optional[str] = None) -> str:
    """Resolve ``auto``/``None`` to one of the supported shells."""
    if explicit and explicit.strip().lower() != "auto":
        return normalize_shell(explicit)

    env_shell = os.environ.get("QTTS_COMPLETION_SHELL", "").strip()
    if env_shell:
        # An explicit environment override must not silently fall back.
        return normalize_shell(env_shell)

    shell_env = os.environ.get("SHELL", "").strip()
    if shell_env:
        name = _basename(shell_env)
        if name in SUPPORTED_SHELLS:
            return name
        if name in PLANNED_SHELLS:
            return normalize_shell(name)
        # A real $SHELL the user chose (sh/dash/ksh/...) is authoritative:
        # falling back to the platform default would install into the wrong
        # file while claiming success.
        raise UsageError(
            f"cannot detect a supported shell (SHELL={shell_env}); "
            f"pass e.g. 'qtts completion bash' ({_supported_message()})"
        )

    parent = _parent_shell()
    if parent:
        return parent

    fallback = _platform_shell()
    if fallback:
        return fallback

    raise UsageError(
        "cannot detect a supported shell; pass e.g. 'qtts completion bash' "
        f"or set QTTS_COMPLETION_SHELL ({_supported_message()})"
    )
