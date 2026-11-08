"""`qtts serve` CLI surface (FR-P2-01)."""

from __future__ import annotations

import argparse
from typing import Any

from .. import serve as serve_mod


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model", help="TTS model id or path override (transformers only)"
    )
    parser.add_argument(
        "--api",
        choices=["native", "openai", "both"],
        default="both",
        help="HTTP API surface (default: both)",
    )
    parser.add_argument(
        "--host", default="127.0.0.1", help="bind host (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="bind port, 0 = ephemeral (default: 8000)"
    )
    parser.add_argument(
        "--allowed-host",
        action="append",
        default=[],
        metavar="HOST",
        help="additional accepted Host value (repeatable; no port/scheme)",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=1,
        help="worker count for remote backends (default: 1; local is always 1)",
    )
    parser.add_argument(
        "--queue-size", type=int, default=32, help="waiting queue capacity (default: 32)"
    )
    parser.add_argument(
        "--max-connections",
        type=int,
        default=32,
        help="concurrent HTTP connections (default: 32)",
    )
    parser.add_argument(
        "--keep-jobs",
        type=int,
        default=100,
        help="finished job directories to retain, 0 = keep all (default: 100)",
    )
    parser.add_argument(
        "--max-body-bytes",
        type=int,
        default=1048576,
        help="maximum request body size (default: 1048576)",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="bearer token (default: QTTS_SERVE_TOKEN)",
    )
    parser.add_argument(
        "--shutdown-timeout",
        type=float,
        default=10.0,
        help="seconds to wait for running jobs on shutdown (default: 10.0)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="resolve backend metadata and exit"
    )
    parser.set_defaults(func=run, formatter=format_result)


def run(cfg, args) -> dict[str, Any]:
    return serve_mod.run(cfg, args)


def format_result(payload: dict[str, Any]) -> str:
    if not payload.get("dry_run"):
        return ""
    backend = payload.get("backend") or {}
    models = ", ".join(backend.get("models") or []) or "(none)"
    kind = "remote" if backend.get("remote") else "local"
    return "\n".join(
        [
            "qtts serve (dry run)",
            f"  backend: {backend.get('name')} ({backend.get('identity')}, {kind})",
            f"  models: {models}",
            f"  configured: {str(bool(backend.get('configured'))).lower()}",
            f"  api: {payload.get('api')}",
            f"  bind: {payload.get('host')}:{payload.get('port')}",
            "  concurrency: "
            f"{payload.get('max_concurrency')} "
            f"(effective {payload.get('effective_max_concurrency')})",
        ]
    )
