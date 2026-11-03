"""FR-P1-03 text QC math and FR-P1-07 audio metrics (no models)."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from qtts_cli.audio_qc import (
    analyze,
    cosine_similarity,
    integrated_loudness,
    true_peak_dbtp,
)
from qtts_cli.errors import InputError, UsageError
from qtts_cli.qc import (
    error_blocks,
    error_ratio,
    levenshtein,
    normalize_for_qc,
    qc_language,
    resolve_metric,
    text_units,
)


# --------------------------------------------------------------------------- #
# Text normalization / metric selection
# --------------------------------------------------------------------------- #
def test_normalization_pipeline_chinese_and_english():
    assert normalize_for_qc("你好，世界！", "Chinese") == "你好世界"
    assert normalize_for_qc("Hello, WORLD!", "English") == "hello world"
    assert normalize_for_qc("2026年", "Chinese") == "二零二六年"
    assert normalize_for_qc("", "auto") == ""


def test_lowercase_languages_expand_numbers_like_canonical_names():
    for language in ("chinese", "CHINESE", " Chinese "):
        assert normalize_for_qc("2026年", language) == normalize_for_qc(
            "2026年", "Chinese"
        )
    for language in ("english", "ENGLISH"):
        assert normalize_for_qc("$12.50", language) == normalize_for_qc(
            "$12.50", "English"
        )
    assert normalize_for_qc("2026-11-03", "chinese") == normalize_for_qc(
        "2026-11-03", "Chinese"
    )
    assert normalize_for_qc("2026年", "chinese") == "二零二六年"


def test_qc_language_is_resolved_once_from_reference():
    assert qc_language("auto", "你好世界hi") == "Chinese"
    assert qc_language("auto", "hello 你好 world") == "English"
    assert qc_language("auto", "こんにちは世界") == "Japanese"
    assert qc_language("english", "你好") == "English"
    assert qc_language("chinese", "你好") == "Chinese"


def test_resolve_metric_auto_and_explicit():
    assert resolve_metric("auto", "你好世界") == "cer"
    assert resolve_metric("auto", "hello world") == "wer"
    assert resolve_metric("Chinese", "") == "cer"
    assert resolve_metric("English", "你好") == "wer"
    assert resolve_metric("English", "", "cer") == "cer"
    with pytest.raises(UsageError) as bad:
        resolve_metric("auto", "", "bleu")
    assert bad.value.code == 2


def test_text_units():
    assert text_units("你好 世界", "cer") == ["你", "好", "世", "界"]
    assert text_units("hello world", "wer") == ["hello", "world"]
    assert text_units("", "wer") == []


def test_levenshtein_and_ratio():
    assert levenshtein([], []) == 0
    assert levenshtein("kitten", "sitting") == 3
    assert levenshtein(["a"], []) == 1
    assert levenshtein(["a", "b"], ["a", "c"]) == 1
    assert error_ratio(0, []) == 0.0
    assert error_ratio(1, []) == 1.0
    assert error_ratio(1, ["a", "b"]) == 0.5


def test_error_blocks_replace_insert_delete():
    replace = error_blocks(list("你好世界"), list("你号世界"))
    assert replace == [
        {
            "op": "replace",
            "ref": "好",
            "hyp": "号",
            "ref_span": [1, 2],
            "hyp_span": [1, 2],
            "start": None,
            "end": None,
        }
    ]

    insert = error_blocks(list("你好"), list("你好啊"))
    assert insert == [
        {
            "op": "insert",
            "ref": "",
            "hyp": "啊",
            "ref_span": [2, 2],
            "hyp_span": [2, 3],
            "start": None,
            "end": None,
        }
    ]

    delete = error_blocks(list("你好啊"), list("你好"))
    assert delete[0]["op"] == "delete"
    assert delete[0]["ref"] == "啊"
    assert delete[0]["hyp"] == ""

    empty_hyp = error_blocks(list("你好"), [])
    assert empty_hyp[0]["op"] == "delete"
    assert empty_hyp[0]["ref"] == "你好"


def test_error_blocks_map_hypothesis_spans_to_optional_times():
    items = [
        {"text": "你", "start": 0.0, "end": 0.1},
        {"text": "好", "start": 0.1, "end": 0.2},
        {"text": "啊", "start": 0.2, "end": 0.3},
    ]
    blocks = error_blocks(list("你好"), list("你好啊"), items=items)
    assert blocks[0]["start"] == 0.2
    assert blocks[0]["end"] == 0.3


# --------------------------------------------------------------------------- #
# Audio QC
# --------------------------------------------------------------------------- #
def _tone(
    freq: float = 997.0,
    amplitude: float = 0.1,
    seconds: float = 2.0,
    sr: int = 48000,
) -> np.ndarray:
    t = np.arange(int(round(sr * seconds)), dtype=np.float64) / float(sr)
    return (amplitude * np.sin(2.0 * math.pi * freq * t)).astype(np.float32)


def test_lufs_matches_bs1770_reference_tone():
    value, reason = integrated_loudness(_tone(amplitude=0.1), 48000)
    assert reason is None
    assert value == pytest.approx(-23.01, abs=0.5)


def test_true_peak_oversampling():
    value, reason = true_peak_dbtp(_tone(amplitude=0.5), 48000)
    assert reason is None
    assert value == pytest.approx(-6.02, abs=0.5)


def test_silence_ratio_covers_full_duration():
    sr = 8000
    tone = _tone(amplitude=0.2, seconds=1.0, sr=sr)
    silence = np.zeros(int(0.5 * sr), dtype=np.float32)
    samples = np.concatenate([tone, silence, tone])
    report = analyze(samples, sr)
    assert report["duration_s"] == pytest.approx(2.5, abs=0.001)
    assert report["silence_ratio"] == pytest.approx(0.2, abs=0.02)
    assert report["longest_silence_s"] == pytest.approx(0.5, abs=0.021)


def test_short_gap_is_not_long_silence():
    sr = 8000
    tone = _tone(amplitude=0.2, seconds=0.5, sr=sr)
    gap = np.zeros(int(0.1 * sr), dtype=np.float32)
    report = analyze(np.concatenate([tone, gap, tone]), sr)
    assert report["silence_ratio"] == 0.0
    assert report["longest_silence_s"] == 0.0


def test_clipping_peak_and_stereo_channels():
    sr = 1000
    left = np.full(1000, 0.1, dtype=np.float32)
    right = np.full(1000, -0.02, dtype=np.float32)
    stereo = np.stack([left, right], axis=1)
    stereo[0, 0] = 1.0
    stereo[1, 1] = -1.0
    report = analyze(stereo, sr)
    assert report["channels"] == 2
    assert report["peak_dbfs"] == 0.0
    assert report["clipping_count"] == 2
    assert report["clipping_ratio"] == pytest.approx(2 / 2000, abs=1e-6)
    # Per-channel DC; the larger absolute mean wins, not the global mean.
    assert report["dc_offset"] == pytest.approx(0.1009, abs=0.001)


def test_more_than_two_channels_is_rejected():
    with pytest.raises(InputError) as error:
        analyze(np.zeros((100, 3), dtype=np.float32), 1000)
    assert error.value.code == 4


def test_non_finite_and_empty_audio_are_rejected():
    samples = np.zeros(100, dtype=np.float32)
    samples[10] = np.nan
    with pytest.raises(InputError):
        analyze(samples, 1000)
    with pytest.raises(InputError):
        analyze(np.zeros(0, dtype=np.float32), 1000)


def test_digital_silence_never_fabricates_numbers():
    report = analyze(np.zeros(24000, dtype=np.float32), 24000)
    assert report["peak_dbfs"] is None
    assert report["lufs_integrated"] is None
    assert report["true_peak_dbtp"] is None
    assert report["unavailable"]["peak"] == "digital_silence"
    assert report["unavailable"]["lufs"] == "digital_silence"
    assert report["unavailable"]["true_peak"] == "digital_silence"
    json.dumps(report, allow_nan=False)


def test_short_audio_lufs_unavailable_but_true_peak_present():
    report = analyze(_tone(seconds=0.2), 48000)
    assert report["lufs_integrated"] is None
    assert report["unavailable"]["lufs"] == "too_short_for_lufs"
    assert report["true_peak_dbtp"] is not None


def test_speech_rate_chinese_and_english():
    sr = 24000
    tone = _tone(amplitude=0.2, seconds=1.0, sr=sr)
    chinese = analyze(tone, sr, text="你好世界", language="Chinese")
    assert chinese["speech_rate"]["chars_per_s"] == pytest.approx(4.0, abs=0.01)
    assert chinese["speech_rate"]["words_per_min"] is None
    assert chinese["unavailable"]["words_per_min"] == "cjk_no_word_boundaries"

    english = analyze(tone, sr, text="Hello, world!", language="English")
    assert english["speech_rate"]["chars_per_s"] == pytest.approx(10.0, abs=0.01)
    assert english["speech_rate"]["words_per_min"] == pytest.approx(120.0, abs=1.0)

    no_text = analyze(tone, sr)
    assert no_text["speech_rate"] is None
    assert no_text["unavailable"]["speech_rate"] == "no_text"


def test_speaker_similarity_fake_embeddings():
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    tone = _tone(seconds=1.0)
    same = analyze(tone, 48000, reference_embedding=([1.0, 2.0], [1.0, 2.0]))
    assert same["speaker_similarity"] == pytest.approx(1.0)
    orthogonal = analyze(tone, 48000, reference_embedding=([1.0, 0.0], [0.0, 1.0]))
    assert orthogonal["speaker_similarity"] == pytest.approx(0.0)
    mismatch = analyze(tone, 48000, reference_embedding=([1.0, 0.0], [1.0, 0.0, 0.0]))
    assert mismatch["speaker_similarity"] is None
    assert mismatch["unavailable"]["speaker_similarity"] == "embedding_shape_mismatch"
    only_ref = analyze(tone, 48000, reference_embedding=[1.0, 0.0])
    assert only_ref["speaker_similarity"] is None
    assert (
        only_ref["unavailable"]["speaker_similarity"]
        == "reference_embedding_unavailable"
    )


def test_analysis_is_deterministic_and_json_safe():
    samples = _tone(amplitude=0.3, seconds=1.2)
    first = analyze(samples, 48000, text="hello", language="English")
    second = analyze(samples, 48000, text="hello", language="English")
    assert first == second
    encoded = json.dumps(first, sort_keys=True, allow_nan=False)
    assert "-inf" not in encoded and "NaN" not in encoded
