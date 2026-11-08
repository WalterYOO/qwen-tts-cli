"""`qtts plugins list` (FR-P2-08)."""

from __future__ import annotations

from typing import Any

from ..errors import EXIT_MODEL
from ..plugins import PLUGIN_API_VERSION, discover_plugins


def list_plugins(cfg, args) -> dict[str, Any]:
    records, warnings = discover_plugins()
    plugins = [record.to_dict() for record in records]
    strict_failed = bool(getattr(args, "strict", False)) and any(
        record["status"] != "ok" for record in plugins
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "plugin_api_version": PLUGIN_API_VERSION,
        "plugins": plugins,
        "warnings": warnings,
        "ok": not strict_failed,
    }
    if strict_failed:
        payload["exit_code"] = EXIT_MODEL
    return payload


def format_result(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    for plugin in payload.get("plugins") or []:
        row = (
            f"{plugin['kind']} {plugin['name']} "
            f"{plugin.get('version') or '-'} {plugin['status']}"
        )
        if plugin.get("reason"):
            row += f" {plugin['reason']}"
        lines.append(row)
    if not lines:
        lines.append("no plugins installed")
    for warning in payload.get("warnings") or []:
        lines.append(f"warning: {warning}")
    if not payload.get("ok", True):
        lines.append("strict: one or more plugins are not usable")
    return "\n".join(lines)
