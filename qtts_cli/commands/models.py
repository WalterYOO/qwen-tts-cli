"""`qtts models list|pull|prune` (FR-P0-08)."""

from __future__ import annotations

from typing import Any

from .. import models_ops
from ..errors import UsageError


def run(cfg, args) -> dict[str, Any]:
    command = args.models_command
    if command == "list":
        return {"ok": True, "model_dir": str(cfg.model_dir) if cfg.model_dir else None,
                "models": models_ops.list_models(cfg)}
    if command == "pull":
        return models_ops.pull(cfg, args.model_id, source=args.source, revision=args.revision)
    if command == "prune":
        return models_ops.prune(cfg, yes=args.yes, dry_run=args.dry_run)
    raise UsageError(f"unknown models command: {command}")


def format_result(payload: dict[str, Any]) -> str:
    if "models" in payload:
        lines = [f"model dir: {payload['model_dir'] or '(not configured)'}"]
        if not payload["models"]:
            lines.append("  (no models)")
        for model in payload["models"]:
            status = "complete" if model.get("complete") else "INCOMPLETE"
            tier = model.get("size_tier") or "-"
            lines.append(
                f"  {model['name']:38s} {model.get('kind', '?'):8s} {tier:>6s} "
                f"{status:10s} {model.get('size', '')}"
            )
        return "\n".join(lines)
    if "removed" in payload:
        if payload.get("dry_run"):
            lines = ["prune (dry run):"]
            for item in payload.get("candidates", []):
                lines.append(f"  would remove {item['name']} ({item['size']})")
            if not payload.get("candidates"):
                lines.append("  nothing to remove")
            return "\n".join(lines)
        removed = payload.get("removed") or []
        if payload.get("cancelled"):
            return "prune cancelled"
        return "removed: " + (", ".join(removed) if removed else "(nothing)")
    if payload.get("path"):
        return f"pulled {payload['model']} ({payload['source']}) -> {payload['path']} [{payload.get('size')}]"
    return str(payload)
