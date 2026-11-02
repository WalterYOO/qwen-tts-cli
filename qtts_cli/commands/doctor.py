"""`qtts doctor` diagnostics and `--bench` baseline (FR-P0-08)."""

from __future__ import annotations

from typing import Any

from .. import doctor


def run(cfg, args) -> dict[str, Any]:
    report = doctor.gather(cfg)
    # `doctor` succeeded as a diagnosis even when it found problems; the report
    # carries the problems explicitly.
    report["healthy"] = report.get("ok", True)
    report["ok"] = True
    if getattr(args, "bench", False):
        bench = doctor.run_bench(cfg)
        report["bench"] = bench
        report["healthy"] = report["healthy"] and bool(bench.get("ok"))
    return report


def format_result(payload: dict[str, Any]) -> str:
    text = doctor.format_report(payload)
    if "bench" in payload:
        text += "\n\nbench:\n" + doctor.format_bench(payload["bench"])
    return text
