"""Subtitle quality gate (FR-P1-05b).

CPS is measured against the *rendered* millisecond duration::

    cps = non-whitespace visible display width / max(0, duration_ms / 1000)

CJK/full-width characters count as 2, combining marks as 0, and equality with
a configured limit is not a violation. Short-but-unmergeable sentences are
warnings only (t10 pre-review); structural problems remain repairable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from .errors import InputError, QualityError
from .subtitles import (
    Cue,
    SubtitlePolicy,
    _HANGING_CLOSE,
    _HANGING_OPEN,
    display_width,
)

EPSILON_MS = 0


@dataclass(frozen=True)
class QcViolation:
    code: str
    message: str
    cue: Optional[int] = None
    severity: str = "error"
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
        }
        if self.cue is not None:
            payload["cue"] = self.cue
        if self.details:
            payload["details"] = self.details
        return payload


@dataclass
class QcReport:
    ok: bool
    enabled: bool
    violations: list[QcViolation]
    metrics: dict[str, Any]
    repairs: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "enabled": self.enabled,
            "violations": [v.to_dict() for v in self.violations],
            "metrics": self.metrics,
            "repairs": dict(self.repairs),
        }

    @property
    def errors(self) -> list[QcViolation]:
        return [v for v in self.violations if v.severity == "error"]

    @property
    def warnings(self) -> list[QcViolation]:
        return [v for v in self.violations if v.severity == "warning"]


def _ms(seconds: float) -> int:
    return int(round(float(seconds) * 1000.0))


def _cps(text: str, duration_ms: int) -> float:
    visible = display_width(text, ignore_whitespace=True)
    seconds = max(0.0, duration_ms / 1000.0)
    if seconds <= 0:
        return float(visible) * 1_000_000_000.0 if visible else 0.0
    return visible / seconds


def _line_width(text: str) -> int:
    lines = text.split("\n") or [""]
    return max(display_width(line) for line in lines)


def _hanging(text: str) -> Optional[str]:
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped[0] in _HANGING_CLOSE:
            return "line starts with closing punctuation"
        if stripped[-1] in _HANGING_OPEN:
            return "line ends with opening punctuation"
    return None


def check_subtitles(
    cues: Sequence[Cue],
    *,
    granularity: str,
    audio_duration: Optional[float],
    policy: SubtitlePolicy,
    repairs: Optional[dict[str, int]] = None,
    formatting_strict: bool = True,
) -> QcReport:
    """Validate cues and return a machine-readable report.

    ``formatting_strict=False`` (used for ``align --srt``) downgrades CPS,
    duration and line-width findings to warnings because the caller-supplied
    cue boundaries must never be regrouped.
    """
    violations: list[QcViolation] = []
    repair_counts = dict(repairs or {})
    enabled = bool(policy.qc_enabled)
    limit_ms = _ms(audio_duration) if audio_duration is not None else None
    duration_values: list[int] = []
    cps_values: list[float] = []
    width_values: list[int] = []

    previous_start: Optional[int] = None
    previous_end: Optional[int] = None
    for index, cue in enumerate(cues, 1):
        start_ms = _ms(cue.start)
        end_ms = _ms(cue.end)
        duration_ms = end_ms - start_ms
        duration_values.append(duration_ms)
        text = cue.text
        width = _line_width(text)
        width_values.append(width)
        duration_s = max(0, duration_ms) / 1000.0
        cps = _cps(text, duration_ms)
        cps_values.append(cps)

        def add(code: str, message: str, severity: str = "error", **details: Any) -> None:
            violations.append(
                QcViolation(code, message, cue=index, severity=severity, details=details)
            )

        if end_ms <= start_ms:
            add("end_not_after_start", "cue end is not after its start",
                start_ms=start_ms, end_ms=end_ms)
        if start_ms < 0:
            add("out_of_range", "cue starts before zero", start_ms=start_ms)
        if limit_ms is not None and end_ms > limit_ms + EPSILON_MS:
            add("out_of_range", "cue ends after the audio",
                end_ms=end_ms, audio_duration_ms=limit_ms)
        if previous_end is not None and start_ms < previous_end - EPSILON_MS:
            add("overlap", "cue starts before the previous cue ends",
                start_ms=start_ms, previous_end_ms=previous_end)
        if previous_start is not None and (
            start_ms < previous_start or (previous_end is not None and end_ms < previous_end)
        ):
            add("not_monotonic", "cue timeline goes backwards",
                start_ms=start_ms, previous_start_ms=previous_start)
        previous_start, previous_end = start_ms, end_ms

        if not enabled or granularity != "sentence":
            continue

        severity = "error" if formatting_strict else "warning"
        if policy.min_cue_seconds > 0 and duration_s < policy.min_cue_seconds:
            # t10: a legal short sentence that cannot merge must not fail the gate.
            add(
                "duration_too_short",
                "cue displays for less than the minimum duration",
                "warning",
                duration_ms=duration_ms,
                min_ms=_ms(policy.min_cue_seconds),
            )
        if duration_s > policy.max_cue_seconds:
            add(
                "duration_too_long",
                "cue displays for longer than the maximum duration",
                severity,
                duration_ms=duration_ms,
                max_ms=_ms(policy.max_cue_seconds),
            )
        # CPS is always checked. A short cue may also produce a
        # duration_too_short warning; long text squeezed into a short window
        # must not hide behind that warning (t10 fix #1).
        if cps > policy.max_cps + 1e-9:
            # If the cue would fit within CPS at the minimum display time,
            # the defect is its duration, not its text density. Such a cue is
            # unmergeable ("好。") and must not fail the gate (fix #1 round 2).
            capacity = policy.max_cps * policy.min_cue_seconds
            short_but_affordable = (
                duration_s < policy.min_cue_seconds
                and display_width(text, ignore_whitespace=True) <= capacity
            )
            add(
                "cps_too_high",
                "cue reading speed exceeds --max-cps",
                "warning" if short_but_affordable else severity,
                cps=round(cps, 3),
                max_cps=policy.max_cps,
            )
        if width > policy.max_line_width:
            add(
                "line_too_wide",
                "cue line exceeds --max-line-width",
                severity,
                width=width,
                max_width=policy.max_line_width,
            )
        hanging = _hanging(text)
        if hanging:
            add("punctuation_hanging", hanging, severity)

    ok = not any(v.severity == "error" for v in violations)
    finite_cps = [value for value in cps_values if value < 1e8]
    metrics = {
        "cue_count": len(cues),
        "duration_ms_min": min(duration_values) if duration_values else None,
        "duration_ms_max": max(duration_values) if duration_values else None,
        "cps_max": round(max(finite_cps), 3) if finite_cps else None,
        "line_width_max": max(width_values) if width_values else None,
        "audio_duration_ms": limit_ms,
        "error_count": sum(1 for v in violations if v.severity == "error"),
        "warning_count": sum(1 for v in violations if v.severity == "warning"),
    }
    return QcReport(ok=ok, enabled=enabled, violations=violations, metrics=metrics,
                    repairs=repair_counts)


STRUCTURAL_CODES = frozenset(
    {"end_not_after_start", "out_of_range", "overlap", "not_monotonic"}
)


def raise_quality_error(report: QcReport) -> None:
    errors = report.errors
    first = errors[0] if errors else None
    detail = f"; first: {first.message}" if first else ""
    message = f"subtitle quality gate failed with {len(errors)} violation(s){detail}"
    # Threshold failures are a quality-gate result (exit 5). Structural defects
    # are not fixable by the writer: they mean the timeline itself is unusable
    # (exit 4), including when --no-subtitle-qc disabled threshold checks.
    if any(v.code in STRUCTURAL_CODES for v in errors):
        raise InputError(message, details={"subtitle_qc": report.to_dict()})
    raise QualityError(message, details={"subtitle_qc": report.to_dict()})
