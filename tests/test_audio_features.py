"""Acoustic feature extraction for ``qc --metrics full`` (FR-P1-07)."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from qtts_cli.audio_features import analyze_features

SR = 24000


def _tone(freq: float, seconds: float = 2.0, amplitude: float = 0.5, sr: int = SR):
    t = np.arange(int(round(sr * seconds)), dtype=np.float64) / float(sr)
    return (amplitude * np.sin(2.0 * math.pi * freq * t)).astype(np.float32)


@pytest.mark.parametrize("freq", [60.0, 80.0, 120.0, 220.0, 440.0, 500.0])
def test_yin_tracks_pure_tones(freq: float):
    report = analyze_features(_tone(freq), SR)
    assert report["f0_median_hz"] == pytest.approx(freq, abs=1.5)
    assert report["f0_mean_hz"] == pytest.approx(freq, abs=1.5)
    assert report["f0_std_hz"] == pytest.approx(0.0, abs=1.5)
    assert report["f0_p10_hz"] == pytest.approx(freq, abs=1.5)
    assert report["f0_p90_hz"] == pytest.approx(freq, abs=1.5)
    assert report["voiced_ratio"] == pytest.approx(1.0)
    assert report["analysis"]["f0_backend"] == "yin"


def test_harmonic_complex_tracks_fundamental_not_overtone():
    tone = _tone(200.0) + 0.3 * _tone(400.0) + 0.1 * _tone(600.0)
    report = analyze_features(tone, SR)
    assert report["f0_median_hz"] == pytest.approx(200.0, abs=2.0)


def test_stereo_is_downmixed_instead_of_rejected():
    mono = _tone(220.0)
    stereo = np.stack([mono, mono * 0.5], axis=1)
    report = analyze_features(stereo, SR)
    assert report["f0_median_hz"] == pytest.approx(220.0, abs=1.5)
    assert report["analysis"]["analysis_channels"] == "downmix"


def test_anti_phase_stereo_cancels_to_silence():
    mono = _tone(220.0)
    stereo = np.stack([mono, -mono], axis=1)
    report = analyze_features(stereo, SR)
    assert report["f0_median_hz"] is None
    assert report["voiced_ratio"] is None
    assert report["unavailable"]["voiced_ratio"] == "no_non_silent_audio"


def test_white_noise_is_not_reported_as_voiced():
    noise = np.random.default_rng(0).normal(0.0, 0.1, SR * 2).astype(np.float32)
    report = analyze_features(noise, SR)
    assert report["voiced_ratio"] == 0.0
    assert "voiced_ratio" not in report["unavailable"]


def test_digital_silence_returns_nulls_and_reasons():
    report = analyze_features(np.zeros(SR, dtype=np.float32), SR)
    assert report["f0_median_hz"] is None
    assert report["voiced_ratio"] is None
    assert report["spectral_centroid_hz"] is None
    assert report["unavailable"]["f0_median_hz"] == "no_non_silent_audio"
    assert report["unavailable"]["voiced_ratio"] == "no_non_silent_audio"
    json.dumps(report, allow_nan=False)


def test_percentiles_need_ten_voiced_frames():
    report = analyze_features(_tone(220.0, seconds=0.1), SR)
    assert report["f0_median_hz"] is not None
    assert report["f0_p10_hz"] is None
    assert report["f0_p90_hz"] is None
    reason = "insufficient_voiced_frames_for_percentiles"
    assert report["unavailable"]["f0_p10_hz"] == reason
    assert report["unavailable"]["f0_p90_hz"] == reason


def test_spectral_centroid_follows_spectral_content():
    low = analyze_features(_tone(300.0), SR)["spectral_centroid_hz"]
    high = analyze_features(_tone(3000.0), SR)["spectral_centroid_hz"]
    assert low is not None and high is not None
    assert high > low * 3.0


def test_hf_lf_separates_bright_and_dark_signals():
    dark = analyze_features(_tone(300.0), SR)["hf_lf_db"]
    bright = analyze_features(_tone(6000.0), SR)["hf_lf_db"]
    assert dark is not None and bright is not None
    assert bright > dark + 20.0


def test_ltas_marks_too_short_clips_and_measures_long_ones():
    short = analyze_features(_tone(440.0, seconds=0.3), SR)
    assert short["ltas_slope_db_per_octave"] is None
    assert short["unavailable"]["ltas_slope_db_per_octave"] == "too_short_for_ltas"
    # HF/LF is still measurable from a short clip; only the slope needs 0.5 s.
    assert short["hf_lf_db"] is not None
    assert "hf_lf_db" not in short["unavailable"]

    noise = np.random.default_rng(1).normal(0.0, 0.1, SR * 2).astype(np.float32)
    long = analyze_features(noise, SR)
    assert long["ltas_slope_db_per_octave"] is not None
    assert long["hf_lf_db"] is not None


@pytest.mark.parametrize("sr", [16000, 48000])
def test_hf_lf_available_at_16k_and_48k(sr: int):
    report = analyze_features(_tone(1000.0, sr=sr), sr)
    assert report["hf_lf_db"] is not None
    assert "hf_lf_db" not in report["unavailable"]


def test_hf_lf_band_unavailable_at_8k():
    report = analyze_features(_tone(300.0, sr=8000), 8000)
    assert report["hf_lf_db"] is None
    assert report["unavailable"]["hf_lf_db"] == "band_unavailable"


def test_hnr_is_praat_or_explicitly_unavailable():
    try:
        import parselmouth  # noqa: F401
    except Exception:
        missing = True
    else:
        missing = False

    report = analyze_features(_tone(220.0), SR)
    if missing:
        assert report["hnr_db"] is None
        assert report["unavailable"]["hnr_db"] == "dependency_missing:praat-parselmouth"
        assert report["analysis"]["hnr_backend"] == "unavailable"
    else:
        assert report["hnr_db"] is not None
        assert report["hnr_db"] > 10.0
        assert report["analysis"]["hnr_backend"] == "praat"


def test_missing_praat_import_is_dependency_missing(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "parselmouth", None)
    report = analyze_features(_tone(220.0), SR)
    assert report["hnr_db"] is None
    assert report["unavailable"]["hnr_db"] == "dependency_missing:praat-parselmouth"


def test_broken_praat_import_is_analysis_failed(monkeypatch):
    import builtins
    import sys

    monkeypatch.delitem(sys.modules, "parselmouth", raising=False)
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "parselmouth":
            raise RuntimeError("ABI mismatch")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    report = analyze_features(_tone(220.0), SR)
    assert report["hnr_db"] is None
    assert report["unavailable"]["hnr_db"] == "analysis_failed"


def test_analysis_records_backend_versions():
    report = analyze_features(_tone(220.0, seconds=1.0), SR)
    analysis = report["analysis"]
    assert analysis["f0_backend"] == "yin"
    assert analysis["f0_min_hz"] == 60.0
    assert analysis["f0_max_hz"] == 500.0
    assert analysis["yin_threshold"] == 0.15
    assert analysis["numpy"]
    assert "scipy" in analysis


def test_invalid_inputs_raise_value_error():
    with pytest.raises(ValueError):
        analyze_features(np.zeros(100, dtype=np.float32), 0)
    with pytest.raises(ValueError):
        analyze_features(_tone(220.0), SR, f0_min=500.0, f0_max=60.0)
    with pytest.raises(ValueError):
        analyze_features(_tone(220.0), SR, hnr_backend="yin")
    with pytest.raises(ValueError):
        analyze_features(np.zeros(0, dtype=np.float32), SR)
