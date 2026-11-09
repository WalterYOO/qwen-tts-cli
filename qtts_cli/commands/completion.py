"""``qtts completion`` command surface."""

from __future__ import annotations

import os
import sys
from typing import Any

from ..completion import install as install_mod
from ..completion.detect import detect_shell
from ..completion.engine import complete_main
from ..completion.shells import generate_script
from ..errors import UsageError

_ACTION_NAMES = ("install", "uninstall", "status")


def _shell_argument(args: Any) -> str:
    value = str(getattr(args, "shell", "auto") or "auto")
    lowered = value.strip().lower()
    if lowered in _ACTION_NAMES:
        raise UsageError(
            f"unknown shell {value!r}; --{lowered} is a flag: "
            f"try 'qtts completion --{lowered}'"
        )
    return value


def _rc_file(args: Any) -> tuple[str | None, bool]:
    explicit = getattr(args, "rc_file", None)
    if explicit is not None:
        return str(explicit), True
    env = os.environ.get("QTTS_COMPLETION_RC", "").strip()
    return (env or None), bool(env)


def _wants_json(cfg: Any, args: Any) -> bool:
    if cfg is not None and getattr(cfg, "json", False):
        return True
    return bool(getattr(args, "json", False))


def run(cfg: Any, args: Any) -> dict[str, Any]:
    shell_arg = _shell_argument(args)
    action = "print"
    for name in _ACTION_NAMES:
        if getattr(args, name, False):
            action = name
            break

    rc_file, rc_explicit = _rc_file(args)
    flag_rc = getattr(args, "rc_file", None) is not None
    follow = bool(getattr(args, "follow_symlinks", False))
    dry_run = bool(getattr(args, "dry_run", False))
    quiet = bool(getattr(args, "quiet", False))

    if action == "print":
        # QTTS_COMPLETION_RC is an install-time override; printing must remain
        # usable in shells that set it globally.
        if flag_rc:
            raise UsageError("--rc-file is only valid with --install, --uninstall or --status")
        if dry_run:
            raise UsageError("--dry-run is only valid with --install or --uninstall")
        if follow:
            raise UsageError(
                "--follow-symlinks is only valid with --install, --uninstall or --status"
            )
        shell = detect_shell(shell_arg)
        script = generate_script(shell)
        if _wants_json(cfg, args):
            return {
                "ok": True,
                "command": "completion",
                "action": "print",
                "shell": shell,
                "marker_version": install_mod.MARKER_VERSION,
                "script": script,
                "script_sha256": install_mod.sha256_bytes(script.encode("utf-8")),
            }
        sys.stdout.write(script)
        sys.stdout.flush()
        return {
            "ok": True,
            "command": "completion",
            "action": "print",
            "shell": shell,
            "raw_stdout": True,
            "stdout_used": True,
        }

    if action == "status" and dry_run:
        raise UsageError("--dry-run is only valid with --install or --uninstall")

    shell = detect_shell(shell_arg)
    if action == "install":
        payload = install_mod.install(
            shell, rc_file, dry_run=dry_run, follow_symlinks=follow
        )
    elif action == "uninstall":
        payload = install_mod.uninstall(
            shell,
            rc_file,
            dry_run=dry_run,
            follow_symlinks=follow,
            explicit_rc=rc_explicit,
        )
    else:
        payload = install_mod.status(shell, rc_file, follow_symlinks=follow)
    if not _wants_json(cfg, args):
        # Human-only formatting flag; keep the machine payload schema stable.
        payload["quiet"] = quiet
    return payload


def format_result(payload: dict[str, Any]) -> str:
    if payload.get("quiet"):
        return ""
    action = payload.get("action")
    shell = payload.get("shell", "?")
    rc_file = payload.get("rc_file", "?")
    dry_run = bool(payload.get("dry_run"))
    if action == "install":
        if not payload.get("changed"):
            return f"installed {shell} completion in {rc_file} (already up to date)"
        prefix = "would install" if dry_run else "installed"
        reason = "created" if payload.get("created") else "updated"
        text = f"{prefix} {shell} completion in {rc_file} ({reason})"
        warnings = payload.get("warnings") or []
        if warnings:
            text += "\n" + "\n".join(f"warning: {item}" for item in warnings)
        return text
    if action == "uninstall":
        if not payload.get("block_found"):
            return f"no {shell} completion block in {rc_file}; nothing to do"
        prefix = "would remove" if dry_run else "removed"
        text = f"{prefix} {shell} completion from {rc_file}"
        if payload.get("deleted_file"):
            text += " (deleted empty drop-in)"
        return text
    if action == "status":
        text = f"{shell} completion: {payload.get('state')} ({rc_file})"
        message = payload.get("message")
        if message:
            text += f"\n{message}"
        return text
    return ""


__all__ = ["complete_main", "format_result", "run"]
