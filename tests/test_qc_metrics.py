"""``qc --metrics full`` wiring: metric merge, gates, completion, reports."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from qtts_cli.audio_qc import analyze
from qtts_cli.cli import main
from qtts_cli.commands import qc as qc_cmd
from qtts_cli.errors import UsageError
from qtts_cli.quality import evaluate, validate_thresholds

SR = 24000
BASIC_KEYS = {
    "duration_s",
    "sample_rate",
    "channels",
    "rms_dbfs",
    "peak_dbfs",
    "clipping_ratio",
    "clipping_count",
    "dc_offset",
    "silence_ratio",
    "longest_silence_s",
    "lufs_integrated",
    "true_peak_dbtp",
    "speech_rate",
    "speaker_similarity",
    "completion",
    "unavailable",
}
FULL_ONLY = {
    "f0_median_hz",
    "f0_mean_hz",
    "f0_std_hz",
    "f0_p10_hz",
    "f0_p90_hz",
    "voiced_ratio",
    "hnr_db",
    "spectral_centroid_hz",
    "ltas_slope_db_per_octave",
    "hf_lf_db",
}
MAX_F0_STD = "max_" + "f0_" + "std" + "_hz"


def _tone(freq: float = 220.0, seconds: float = 2.0, sr: int = SR):
    t = np.arange(int(round(sr * seconds)), dtype=np.float64) / float(sr)
    return (0.3 * np.sin(2.0 * math.pi * freq * t)).astype(np.float32)


def _write_wav(path: Path, *, seconds: float = 1.0, frequency: float = 220.0) -> Path:
    sf.write(path, _tone(frequency, seconds), SR)
    return path


def _run(tmp_path: Path, model_dir: Path, *args: str) -> int:
    return main(
        [
            *args,
            "--model-dir",
            str(model_dir),
            "--data-dir",
            str(tmp_path / "data"),
            "--cache-dir",
            str(tmp_path / "cache"),
        ]
    )


# --------------------------------------------------------------------------- #
# Metric merge
# --------------------------------------------------------------------------- #
def test_basic_mode_keeps_the_old_metric_set():
    report = analyze(_tone(), SR)
    assert report["completion"]["status"] in ("unknown", "complete")
    assert set(report) == BASIC_KEYS
    assert FULL_ONLY.isdisjoint(report)


def test_full_mode_adds_acoustic_features():
    report = analyze(_tone(220.0), SR, metrics="full")
    assert report["f0_median_hz"] == pytest.approx(220.0, abs=1.5)
    assert report["voiced_ratio"] == pytest.approx(1.0)
    assert report["analysis"]["f0_backend"] == "yin"
    assert isinstance(report["unavailable"], dict)
    json.dumps(report, sort_keys=True, allow_nan=False)


def test_full_mode_merges_unavailable_reasons_at_top_level():
    report = analyze(np.zeros(SR, dtype=np.float32), SR, metrics="full")
    assert report["f0_median_hz"] is None
    assert report["unavailable"]["f0_median_hz"] == "no_non_silent_audio"
    assert report["unavailable"]["voiced_ratio"] == "no_non_silent_audio"


def test_recipe_finish_reason_reaches_completion():
    report = analyze(_tone(seconds=3.0), SR, recipe={"finish_reason": "length"})
    assert report["completion"]["status"] == "suspect_truncated"
    assert report["completion"]["finish_reason"] == "length"


def test_unknown_metrics_mode_is_rejected():
    from qtts_cli.errors import InputError

    with pytest.raises(InputError) as error:
        analyze(_tone(), SR, metrics="extreme")
    assert error.value.code == 4


# --------------------------------------------------------------------------- #
# Thresholds
# --------------------------------------------------------------------------- #
def test_min_max_gates_across_full_metrics():
    metrics = {
        "duration_s": 5.0,
        "voiced_ratio": 0.4,
        "hnr_db": 8.0,
        "f0_median_hz": 300.0,
        "f0_std_hz": 60.0,
        "spectral_centroid_hz": 9000.0,
        "ltas_slope_db_per_octave": -15.0,
        "hf_lf_db": -12.0,
        "unavailable": {},
    }
    thresholds = {
        "min_voiced_ratio": 0.5,
        "min_hnr_db": 10.0,
        "min_f0_hz": 120.0,
        "max_f0_hz": 280.0,
        "max_f0_std_hz": 50.0,
        "max_spectral_centroid_hz": 8000.0,
        "min_ltas_slope_db_per_octave": -12.0,
        "max_hf_lf_db": -15.0,
    }
    violations = {v["metric"] for v in evaluate(metrics, thresholds)}
    assert violations == {
        "voiced_ratio",
        "hnr_db",
        "f0_median_hz",
        "f0_std_hz",
        "spectral_centroid_hz",
        "ltas_slope_db_per_octave",
        "hf_lf_db",
    }


def test_missing_full_metric_is_an_unavailable_violation():
    metrics = {"unavailable": {"hnr_db": "dependency_missing:praat-parselmouth"}}
    violations = evaluate(metrics, {"min_hnr_db": 10.0})
    assert violations == [
        {
            "metric": "hnr_db",
            "op": "unavailable",
            "value": None,
            "threshold": 10.0,
            "reason": "dependency_missing:praat-parselmouth",
        }
    ]


@pytest.mark.parametrize(
    "thresholds, has_text",
    [
        ({"min_f0_hz": 300.0, "max_f0_hz": 200.0}, False),
        ({"min_spectral_centroid_hz": 900.0, "max_spectral_centroid_hz": 800.0}, False),
        ({"min_hf_lf_db": 1.0, "max_hf_lf_db": 0.0}, False),
        ({"min_voiced_ratio": 1.5}, False),
        ({"min_hnr_db": float("nan")}, False),
        ({"max_duration_s": -1.0}, False),
        ({"min_duration_s": -1.0}, False),
        ({"min_f0_hz": 0.0}, False),
        ({"max_f0_hz": 0.0}, False),
        ({MAX_F0_STD: -1.0}, False),
        ({"min_spectral_centroid_hz": -10.0}, False),
    ],
)
def test_invalid_full_thresholds_are_usage_errors(thresholds, has_text):
    with pytest.raises(UsageError):
        validate_thresholds(thresholds, has_text=has_text)


def test_require_full_violations_lists_every_missing_metric():
    metrics = {"unavailable": {"hnr_db": "dependency_missing:praat-parselmouth"}}
    violations = qc_cmd._require_full_violations(metrics)
    reported = {item["metric"] for item in violations}
    assert reported == set(qc_cmd.FULL_METRIC_KEYS)
    hnr = [item for item in violations if item["metric"] == "hnr_db"][0]
    assert hnr["op"] == "unavailable"
    assert hnr["reason"] == "dependency_missing:praat-parselmouth"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def test_qc_full_report_is_canonical(capsys, tmp_path: Path, model_dir: Path):
    audio = _write_wav(tmp_path / "tone.wav")
    report = tmp_path / "full.json"
    rc = _run(
        tmp_path, model_dir,
        "qc", str(audio), "--metrics", "full", "--json", "--report", str(report),
    )
    assert rc == 0
    stdout = capsys.readouterr().out
    assert report.read_bytes() == stdout.encode("utf-8")
    payload = json.loads(stdout)
    assert payload["metrics_version"] == 1
    assert payload["metrics_mode"] == "full"
    assert payload["metrics"]["f0_median_hz"] == pytest.approx(220.0, abs=1.5)
    assert payload["analysis"]["f0_backend"] == "yin"
    assert payload["completion"]["status"] in ("unknown", "complete")
    assert payload["completion_checked"] is True
    assert "completion" not in payload["metrics"]
    assert "analysis" not in payload["metrics"]


def test_qc_recipe_finish_reason_and_fail_on_incomplete(
    capsys, tmp_path: Path, model_dir: Path
):
    audio = _write_wav(tmp_path / "tone.wav")
    recipe = tmp_path / "recipe.json"
    recipe.write_text(json.dumps({"finish_reason": "length"}), encoding="utf-8")
    rc = _run(
        tmp_path, model_dir,
        "qc", str(audio), "--recipe", str(recipe), "--fail-on-incomplete", "--json",
    )
    assert rc == 5
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["completion"]["status"] == "suspect_truncated"
    assert payload["violations"] == [
        {
            "metric": "completion",
            "op": "status",
            "value": "suspect_truncated",
            "threshold": "complete",
            "reason": "finish_reason_length",
        }
    ]


def test_qc_full_only_threshold_requires_full_mode(
    capsys, tmp_path: Path, model_dir: Path
):
    audio = _write_wav(tmp_path / "tone.wav")
    rc = _run(tmp_path, model_dir, "qc", str(audio), "--min-hnr-db", "5")
    assert rc == 2
    assert "requires --metrics full" in capsys.readouterr().err


def test_qc_basic_text_output_keeps_no_completion_line(
    capsys, tmp_path: Path, model_dir: Path
):
    audio = _write_wav(tmp_path / "tone.wav")
    assert _run(tmp_path, model_dir, "qc", str(audio)) == 0
    assert "completion:" not in capsys.readouterr().out

    assert _run(tmp_path, model_dir, "qc", str(audio), "--metrics", "full") == 0
    assert "completion:" in capsys.readouterr().out


def test_qc_rejects_three_channels(capsys, tmp_path: Path, model_dir: Path):
    three = tmp_path / "three.wav"
    sf.write(three, np.zeros((SR, 3), dtype=np.float32), SR)
    assert _run(tmp_path, model_dir, "qc", str(three)) == 4
    assert "mono/stereo" in capsys.readouterr().err


def test_qc_rejects_non_finite_audio(capsys, tmp_path: Path, model_dir: Path):
    broken = tmp_path / "nan.wav"
    samples = _tone()
    samples[100] = np.nan
    # FLOAT preserves the NaN; PCM_16 would quantize it away.
    sf.write(broken, samples, SR, subtype="FLOAT")
    assert _run(tmp_path, model_dir, "qc", str(broken)) == 4
    assert "NaN" in capsys.readouterr().err


def test_qc_require_full_metrics_needs_full_mode(capsys, tmp_path: Path, model_dir: Path):
    audio = _write_wav(tmp_path / "tone.wav")
    rc = _run(
        tmp_path, model_dir, "qc", str(audio), "--require-full-metrics"
    )
    assert rc == 2
    assert "--require-full-metrics requires --metrics full" in capsys.readouterr().err


def test_qc_missing_recipe_is_an_input_error(capsys, tmp_path: Path, model_dir: Path):
    audio = _write_wav(tmp_path / "tone.wav")
    rc = _run(
        tmp_path, model_dir, "qc", str(audio), "--recipe", str(tmp_path / "nope.json")
    )
    assert rc == 4
    assert "recipe file not found" in capsys.readouterr().err


def test_qc_max_f0_std_hz_gate_is_reachable_from_cli(
    capsys, tmp_path: Path, model_dir: Path
):
    sr = SR
    seconds = 2.0
    t = np.arange(int(sr * seconds), dtype=np.float64) / float(sr)
    block = int(0.05 * sr)
    alternating = np.where(
        (np.arange(t.size) // block) % 2 == 0, np.sin(2 * np.pi * 150 * t),
        np.sin(2 * np.pi * 350 * t),
    ).astype(np.float32) * 0.3
    audio = tmp_path / "alternating.wav"
    sf.write(audio, alternating, sr)

    rc = _run(
        tmp_path, model_dir,
        "qc", str(audio), "--metrics", "full", "--max-f0-std-hz", "10", "--json",
    )
    assert rc == 5
    payload = json.loads(capsys.readouterr().out)
    assert payload["metrics"]["f0_std_hz"] > 10
    assert payload["violations"][0]["metric"] == "f0_std_hz"


def test_qc_acoustic_gate_fails_on_silence(capsys, tmp_path: Path, model_dir: Path):
    audio = tmp_path / "silence.wav"
    sf.write(audio, np.zeros(SR, dtype=np.float32), SR)
    rc = _run(
        tmp_path, model_dir,
        "qc", str(audio), "--metrics", "full", "--min-voiced-ratio", "0.5", "--json",
    )
    assert rc == 5
    payload = json.loads(capsys.readouterr().out)
    assert payload["violations"][0]["metric"] == "voiced_ratio"
    assert payload["violations"][0]["op"] == "unavailable"
