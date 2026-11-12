"""Heuristic completion/truncation assessment (no model required)."""

from __future__ import annotations

import pytest

from qtts_cli.completeness import (
    FAILURE_STATUSES,
    SEVERITY,
    assess,
    estimate_duration_s,
    speech_rate_for_language,
)


def test_finish_reason_length_is_suspect_and_stop_is_complete():
    truncated = assess(duration_s=3.0, recipe={"finish_reason": "length"})
    assert truncated["status"] == "suspect_truncated"
    assert "finish_reason_length" in truncated["reasons"]

    stopped = assess(duration_s=3.0, recipe={"finish_reason": "stop"})
    assert stopped["status"] == "complete"
    assert stopped["reasons"] == []


def test_clip_shorter_than_half_a_second_is_too_short():
    # finish_reason is authoritative: an explicit stop wins over the
    # duration heuristic, but an unlabelled 0.3 s clip is never complete.
    report = assess(duration_s=0.3)
    assert report["status"] == "too_short"
    assert report["status"] in FAILURE_STATUSES
    assert assess(duration_s=0.3, recipe={"finish_reason": "stop"})["status"] == "complete"


def test_token_budget_exhaustion_is_suspect():
    report = assess(
        duration_s=163.8,
        recipe={"finish_reason": None},
        max_new_tokens=2048,
        codec_rate=12.5,
    )
    assert report["status"] == "suspect_truncated"
    assert "token_budget_exhausted" in report["reasons"]


def test_token_budget_is_read_from_the_recipe():
    # say keeps the budget under sampling; duration >= 0.8 s with 10 tokens
    # at 1 token/s means the budget was exhausted.
    report = assess(
        duration_s=9.9,
        recipe={"max_new_tokens": 10, "codec_rate": 1, "finish_reason": None},
    )
    assert report["max_audio_s"] == pytest.approx(10.0)
    assert report["status"] == "suspect_truncated"
    assert "token_budget_exhausted" in report["reasons"]

    nested = assess(
        duration_s=0.9,
        recipe={"sampling": {"max_new_tokens": 10, "codec_rate": 12.5}},
    )
    assert nested["max_audio_s"] == pytest.approx(0.8)
    assert nested["status"] == "suspect_truncated"


def test_explicit_budget_overrides_recipe():
    report = assess(
        duration_s=10.0,
        recipe={"max_new_tokens": 10, "codec_rate": 1},
        max_new_tokens=2048,
        codec_rate=12.5,
    )
    assert report["max_audio_s"] == pytest.approx(163.84)
    assert "token_budget_exhausted" not in report["reasons"]


def test_invalid_recipe_budget_falls_back_to_defaults():
    for recipe in (
        {"max_new_tokens": 0},
        {"max_new_tokens": -5, "codec_rate": 0},
        {"sampling": {"max_new_tokens": "many"}},
    ):
        report = assess(duration_s=10.0, recipe=recipe)
        assert report["max_audio_s"] == pytest.approx(163.84)


def test_null_top_level_budget_does_not_mask_nested_value():
    report = assess(
        duration_s=9.9,
        recipe={
            "max_new_tokens": None,
            "sampling": {"max_new_tokens": 10, "codec_rate": 1},
        },
    )
    assert report["max_audio_s"] == pytest.approx(10.0)
    assert report["status"] == "suspect_truncated"


def test_shorter_than_expected_is_unknown_not_truncated():
    text = "你好" * 20  # expected 40 / 4.5 ~ 8.9s
    report = assess(duration_s=2.0, text=text, language="Chinese")
    assert report["status"] == "unknown"
    assert "shorter_than_expected" in report["reasons"]
    assert report["status"] not in FAILURE_STATUSES


def test_abrupt_tail_is_unknown_and_near_budget_escalates():
    common = dict(
        duration_s=100.0,
        tail_rms_dbfs=-20.0,
        reference_rms_dbfs=-20.0,
        tail_silence_s=0.0,
    )
    unknown = assess(**common)
    assert unknown["status"] == "unknown"
    assert "abrupt_tail" in unknown["reasons"]

    escalated = assess(**common, max_new_tokens=1300, codec_rate=12.5)
    assert escalated["status"] == "suspect_truncated"
    assert "token_budget_near_exhausted" in escalated["reasons"]


def test_natural_tail_silence_prevents_abrupt_tail():
    report = assess(
        duration_s=5.0,
        tail_rms_dbfs=-20.0,
        reference_rms_dbfs=-20.0,
        tail_silence_s=0.3,
    )
    assert "abrupt_tail" not in report["reasons"]


def test_missing_recipe_is_never_complete():
    report = assess(duration_s=5.0)
    assert report["status"] == "unknown"
    assert "no_recipe" in report["reasons"]


def test_recipe_without_finish_reason_uses_duration_ratio():
    report = assess(duration_s=5.0, recipe={})
    assert report["status"] in ("unknown", "complete")
    assert report["finish_reason"] is None


def test_expected_duration_and_rates_by_language():
    assert speech_rate_for_language("Chinese") == 4.5
    assert speech_rate_for_language("Korean") == 4.0
    assert speech_rate_for_language("English") == 14.0
    assert speech_rate_for_language("Dutch") == 6.0
    assert estimate_duration_s("你好世界") == pytest.approx(4 / 4.5, abs=1e-3)
    assert estimate_duration_s("hello world") == pytest.approx(10 / 14.0, abs=1e-3)
    assert estimate_duration_s("   ") is None
    assert estimate_duration_s("") is None


def test_auto_language_uses_text_detection():
    assert speech_rate_for_language("auto", "你好世界") == 4.5
    assert speech_rate_for_language(None, "hello world") == 14.0


def test_severity_ordering():
    assert SEVERITY["complete"] < SEVERITY["unknown"]
    assert SEVERITY["unknown"] < SEVERITY["suspect_truncated"]
    assert SEVERITY["suspect_truncated"] < SEVERITY["too_short"]
    assert SEVERITY["too_short"] < SEVERITY["failed"]
