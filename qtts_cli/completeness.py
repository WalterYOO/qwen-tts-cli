"""Heuristic truncation / incompleteness assessment (no model required).

``TtsEngine.generate`` does not expose ``finish_reason`` yet, so recipes may
carry ``None``. This module therefore treats missing evidence as ``unknown``,
never as ``complete``: a clip is only ``complete`` when a stop reason or a
plausible duration/tail combination supports it.
"""

from __future__ import annotations

import math
from typing import Any, Optional

from .textnorm import detect_language

STATUS_COMPLETE = "complete"
STATUS_UNKNOWN = "unknown"
STATUS_SUSPECT = "suspect_truncated"
STATUS_TOO_SHORT = "too_short"
STATUS_FAILED = "failed"

SEVERITY = {
    STATUS_COMPLETE: 0,
    STATUS_UNKNOWN: 1,
    STATUS_SUSPECT: 2,
    STATUS_TOO_SHORT: 3,
    STATUS_FAILED: 4,
}
FAILURE_STATUSES = frozenset({STATUS_SUSPECT, STATUS_TOO_SHORT, STATUS_FAILED})

_CJK_RATE = 4.5  # characters per second, conversational Mandarin/Japanese
_KOREAN_RATE = 4.0
_LATIN_RATE = 14.0
_DEFAULT_RATE = 6.0
_STOP_REASONS = frozenset({"stop", "eos", "end", "end_turn"})
_LENGTH_REASONS = frozenset(
    {"length", "max_new_tokens", "max_tokens", "token_budget_exhausted"}
)
DEFAULT_MAX_NEW_TOKENS = 2048
DEFAULT_CODEC_RATE = 12.5


def _positive(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number <= 0:
        return None
    return number


def recipe_budget(recipe: Optional[dict[str, Any]]) -> tuple[Optional[float], Optional[float]]:
    """Return ``(max_new_tokens, codec_rate)`` from a recipe, if present.

    ``say`` recipes keep the budget under ``sampling``; batch/takes jobs may
    also put it at the top level. Unknown or invalid values return ``None``
    so the defaults apply.
    """
    if not isinstance(recipe, dict):
        return None, None
    sampling = recipe.get("sampling")
    if not isinstance(sampling, dict):
        sampling = {}
    tokenizer = recipe.get("tokenizer")
    if not isinstance(tokenizer, dict):
        tokenizer = {}
    # First valid value wins: an explicit top-level null must not mask a
    # usable nested value.
    tokens = None
    for value in (recipe.get("max_new_tokens"), sampling.get("max_new_tokens")):
        tokens = _positive(value)
        if tokens is not None:
            break
    rate = None
    for value in (
        recipe.get("codec_rate"),
        sampling.get("codec_rate"),
        tokenizer.get("codec_rate_hz"),
    ):
        rate = _positive(value)
        if rate is not None:
            break
    return tokens, rate


def _round(value: Optional[float], digits: int = 3) -> Optional[float]:
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value):
        return None
    return round(value, digits)


def speech_rate_for_language(language: Optional[str], text: str = "") -> float:
    """Approximate characters per second used to estimate expected duration."""
    resolved = (language or "auto").strip()
    if not resolved or resolved.lower() == "auto":
        resolved = detect_language(text)
    key = resolved.lower()
    if key in ("chinese", "japanese", "cantonese", "zh", "ja", "yue"):
        return _CJK_RATE
    if key in ("korean", "ko"):
        return _KOREAN_RATE
    if key in ("english", "en", "german", "french", "spanish", "italian",
               "portuguese", "russian"):
        return _LATIN_RATE
    return _DEFAULT_RATE


def estimate_duration_s(text: Optional[str], language: Optional[str] = None) -> Optional[float]:
    """Estimate speech duration from visible character count."""
    if not text or not str(text).strip():
        return None
    visible = sum(1 for char in str(text) if not char.isspace())
    if visible == 0:
        return None
    return _round(visible / speech_rate_for_language(language, str(text)), 3)


def assess(
    *,
    duration_s: float,
    text: Optional[str] = None,
    language: Optional[str] = None,
    recipe: Optional[dict[str, Any]] = None,
    max_new_tokens: Optional[float] = None,
    codec_rate: Optional[float] = None,
    tail_rms_dbfs: Optional[float] = None,
    reference_rms_dbfs: Optional[float] = None,
    tail_silence_s: Optional[float] = None,
) -> dict[str, Any]:
    """Classify a clip as complete / unknown / suspect_truncated / too_short.

    ``failed`` is intentionally not produced here; the caller injects it when
    the audio itself is missing or unreadable.
    """
    reasons: list[str] = []
    duration = float(duration_s)
    finish_reason = None
    if isinstance(recipe, dict):
        raw = recipe.get("finish_reason")
        finish_reason = str(raw) if raw is not None else None

    expected = estimate_duration_s(text, language)
    ratio = None
    if expected and expected > 0:
        ratio = _round(duration / expected)

    recipe_tokens, recipe_rate = recipe_budget(recipe)
    budget_tokens = _positive(max_new_tokens) or recipe_tokens or DEFAULT_MAX_NEW_TOKENS
    budget_rate = _positive(codec_rate) or recipe_rate or DEFAULT_CODEC_RATE
    max_audio_s = _round(float(budget_tokens) / float(budget_rate))

    tail_abrupt = False
    if (
        tail_rms_dbfs is not None
        and reference_rms_dbfs is not None
        and tail_rms_dbfs > -45.0
        and tail_rms_dbfs > reference_rms_dbfs - 6.0
        and (tail_silence_s is None or tail_silence_s < 0.05)
    ):
        tail_abrupt = True

    status = STATUS_UNKNOWN
    if finish_reason in _LENGTH_REASONS:
        status = STATUS_SUSPECT
        reasons.append("finish_reason_length")
    elif finish_reason in _STOP_REASONS:
        status = STATUS_COMPLETE
    elif duration < 0.5:
        status = STATUS_TOO_SHORT
        reasons.append("duration_below_0_5s")
    elif max_audio_s is not None and duration >= 0.98 * max_audio_s:
        status = STATUS_SUSPECT
        reasons.append("token_budget_exhausted")
    elif expected is not None and expected >= 2.0 and ratio is not None and ratio < 0.5:
        status = STATUS_UNKNOWN
        reasons.append("shorter_than_expected")
    elif tail_abrupt:
        status = STATUS_UNKNOWN
        reasons.append("abrupt_tail")
        if max_audio_s is not None and duration >= 0.9 * max_audio_s:
            status = STATUS_SUSPECT
            reasons.append("token_budget_near_exhausted")
    elif not recipe:
        status = STATUS_UNKNOWN
        reasons.append("no_recipe")
    elif ratio is not None and ratio >= 0.8:
        status = STATUS_COMPLETE
    else:
        status = STATUS_UNKNOWN
        reasons.append("short_or_unassessable")

    return {
        "status": status,
        "reasons": reasons,
        "finish_reason": finish_reason,
        "expected_duration_s": expected,
        "duration_ratio": ratio,
        "tail_energy_dbfs": _round(tail_rms_dbfs, 2),
        "max_audio_s": max_audio_s,
    }
