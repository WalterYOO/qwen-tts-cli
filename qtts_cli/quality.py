"""Shared QC report serialization and threshold gates (FR-P1-03/07).

Thresholds are opt-in: argparse defaults are ``None`` and only explicitly
supplied values participate. Report JSON is canonical so ``--json`` and
``--report`` are byte-identical.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .errors import InputError, UsageError
from .utils import atomic_write_text

THRESHOLD_NAMES = (
    "max_clipping_ratio",
    "target_lufs",
    "lufs_tolerance",
    "max_true_peak_dbtp",
    "max_dc_offset",
    "max_silence_ratio",
    "min_speech_rate",
    "max_speech_rate",
)


def canonical_report_text(payload: dict[str, Any]) -> str:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )


def write_report(path: str | Path, payload: dict[str, Any]) -> Path:
    target = Path(path).expanduser()
    try:
        atomic_write_text(target, canonical_report_text(payload))
    except OSError as exc:
        raise InputError(
            f"cannot write report {target}: {exc}", details={"path": str(target)}
        ) from exc
    return target


def thresholds_from_args(args: Any) -> dict[str, float]:
    values: dict[str, float] = {}
    for name in THRESHOLD_NAMES:
        raw = getattr(args, name, None)
        if raw is not None:
            values[name] = float(raw)
    return values


def validate_thresholds(thresholds: dict[str, float], *, has_text: bool) -> None:
    for name, value in thresholds.items():
        if not math.isfinite(value):
            raise UsageError(f"--{name.replace('_', '-')} must be finite")
    if ("target_lufs" in thresholds) != ("lufs_tolerance" in thresholds):
        raise UsageError("--target-lufs and --lufs-tolerance must be given together")
    for name in ("max_clipping_ratio", "max_silence_ratio"):
        if name in thresholds and not 0.0 <= thresholds[name] <= 1.0:
            raise UsageError(f"--{name.replace('_', '-')} must be between 0 and 1")
    for name in (
        "lufs_tolerance",
        "max_dc_offset",
        "min_speech_rate",
        "max_speech_rate",
    ):
        if name in thresholds and thresholds[name] < 0:
            raise UsageError(f"--{name.replace('_', '-')} must be >= 0")
    if (
        "min_speech_rate" in thresholds
        and "max_speech_rate" in thresholds
        and thresholds["min_speech_rate"] > thresholds["max_speech_rate"]
    ):
        raise UsageError("--min-speech-rate must not exceed --max-speech-rate")
    if (
        "min_speech_rate" in thresholds or "max_speech_rate" in thresholds
    ) and not has_text:
        raise UsageError("speech-rate thresholds require --text or --text-file")


def _violation(
    metric: str,
    op: str,
    value: Optional[float],
    threshold: float,
    reason: Optional[str] = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "metric": metric,
        "op": op,
        "value": value,
        "threshold": threshold,
    }
    if reason:
        payload["reason"] = reason
    return payload


def evaluate(
    metrics: dict[str, Any], thresholds: dict[str, float]
) -> list[dict[str, Any]]:
    """Compare metrics against explicit thresholds only."""
    violations: list[dict[str, Any]] = []
    unavailable = metrics.get("unavailable") or {}

    def max_gate(
        name: str, metric: str, value: Optional[float], reason_key: str
    ) -> None:
        if name not in thresholds:
            return
        limit = thresholds[name]
        if value is None:
            violations.append(
                _violation(
                    metric, "unavailable", None, limit, unavailable.get(reason_key)
                )
            )
        elif value > limit:
            violations.append(_violation(metric, ">", float(value), limit))

    max_gate(
        "max_clipping_ratio", "clipping_ratio", metrics.get("clipping_ratio"), "clipping"
    )
    max_gate(
        "max_true_peak_dbtp", "true_peak_dbtp", metrics.get("true_peak_dbtp"), "true_peak"
    )
    max_gate(
        "max_silence_ratio", "silence_ratio", metrics.get("silence_ratio"), "silence"
    )

    if "max_dc_offset" in thresholds:
        limit = thresholds["max_dc_offset"]
        value = metrics.get("dc_offset")
        if value is None:
            violations.append(_violation("dc_offset", "unavailable", None, limit))
        elif abs(float(value)) > limit:
            violations.append(_violation("dc_offset", "abs>", abs(float(value)), limit))

    if "target_lufs" in thresholds:
        target = thresholds["target_lufs"]
        tolerance = thresholds["lufs_tolerance"]
        value = metrics.get("lufs_integrated")
        if value is None:
            violations.append(
                _violation(
                    "lufs_integrated",
                    "unavailable",
                    None,
                    target,
                    unavailable.get("lufs"),
                )
            )
        elif abs(float(value) - target) > tolerance:
            violations.append(
                _violation("lufs_integrated", "outside_tolerance", float(value), target)
            )

    for name, op in (("min_speech_rate", "<"), ("max_speech_rate", ">")):
        if name not in thresholds:
            continue
        limit = thresholds[name]
        speech = metrics.get("speech_rate")
        value = None if speech is None else speech.get("chars_per_s")
        if value is None:
            violations.append(
                _violation(
                    "chars_per_s",
                    "unavailable",
                    None,
                    limit,
                    unavailable.get("speech_rate"),
                )
            )
        elif op == "<" and float(value) < limit:
            violations.append(_violation("chars_per_s", op, float(value), limit))
        elif op == ">" and float(value) > limit:
            violations.append(_violation("chars_per_s", op, float(value), limit))
    return violations


def embedding_array(value: Any) -> Optional[np.ndarray]:
    """Convert a torch tensor / numpy array / sequence to a flat float32 vector."""
    if value is None:
        return None
    for attribute in ("detach", "cpu", "float", "numpy"):
        if hasattr(value, attribute):
            try:
                value = getattr(value, attribute)()
            except Exception:  # pragma: no cover - tensor backend dependent
                return None
    try:
        array = np.asarray(value, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        return None
    return array if array.size else None
