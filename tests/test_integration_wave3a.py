"""Real-model Wave 3a checks (set RUN_INTEGRATION=1).

One TTS clip is reused by ASR quality checking, audio QC, and automatic
reference transcription so the suite stays inside the 3-4 minute budget.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from qtts_cli.cli import main

_raw_models = os.environ.get("QTTS_SMOKE_MODEL_DIR")
MODELS = Path(_raw_models).expanduser() if _raw_models else None
ASR_MODEL = MODELS / "Qwen3-ASR-0.6B" if MODELS is not None else None

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("RUN_INTEGRATION") != "1",
        reason="set RUN_INTEGRATION=1 to run real-model integration tests",
    ),
    pytest.mark.skipif(
        MODELS is None or not MODELS.is_dir(),
        reason="set QTTS_SMOKE_MODEL_DIR to the local model directory",
    ),
    pytest.mark.skipif(
        ASR_MODEL is None or not ASR_MODEL.is_dir(),
        reason="Qwen3-ASR-0.6B not found under QTTS_SMOKE_MODEL_DIR",
    ),
]

TEXT = "你好，世界。今天天气很好。"
ALTERED = "你好，世界。今天天汽很好。"


def _run(tmp_path: Path, *args: str) -> int:
    return main(
        [
            *args,
            "--model-dir",
            str(MODELS),
            "--data-dir",
            str(tmp_path / "data"),
            "--cache-dir",
            str(tmp_path / "cache"),
        ]
    )


@pytest.fixture(scope="module")
def generated(tmp_path_factory) -> tuple[Path, Path]:
    tmp_path = tmp_path_factory.mktemp("wave3a-integration")
    out = tmp_path / "speech.wav"
    rc = main(
        [
            "say",
            TEXT,
            "--voice",
            "builtin:Vivian",
            "--language",
            "Chinese",
            "--out",
            str(out),
            "--model-dir",
            str(MODELS),
            "--data-dir",
            str(tmp_path / "data"),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--json",
        ]
    )
    assert rc == 0
    assert out.is_file() and out.stat().st_size > 0
    return tmp_path, out


def test_check_real_asr_cer_below_threshold(generated, capsys):
    tmp_path, audio = generated
    rc = _run(
        tmp_path,
        "check",
        str(audio),
        "--text",
        TEXT,
        "--language",
        "Chinese",
        "--threshold",
        "0.2",
        "--json",
    )
    payload = json.loads(capsys.readouterr().out)
    assert rc == 0, payload
    assert payload["ok"] is True
    assert payload["metric"] == "cer"
    assert payload["ratio"] < 0.2
    assert payload["asr"]["model"] == "Qwen3-ASR-0.6B"
    assert payload["asr"]["language"] == "Chinese"
    assert payload["audio"]["path"] == "speech.wav"
    assert payload["errors"] == []
    print(f"\n[wave3a] CER={payload['ratio']:.4f} ({payload['distance']}/"
          f"{payload['reference_units']}) hypothesis={payload['hypothesis']!r}")


def test_check_altered_text_fails_the_threshold(generated, capsys):
    tmp_path, audio = generated
    rc = _run(
        tmp_path,
        "check",
        str(audio),
        "--text",
        ALTERED,
        "--language",
        "Chinese",
        "--threshold",
        "0.0",
        "--json",
    )
    payload = json.loads(capsys.readouterr().out)
    assert rc == 5, payload
    assert payload["ok"] is False
    assert payload["ratio"] > 0.0
    print(f"\n[wave3a] altered CER={payload['ratio']:.4f}")


def test_qc_real_audio_metrics_are_plausible(generated, capsys):
    tmp_path, audio = generated
    rc = _run(tmp_path, "qc", str(audio), "--json")
    payload = json.loads(capsys.readouterr().out)
    assert rc == 0, payload
    metrics = payload["metrics"]
    assert metrics["lufs_integrated"] is not None
    assert -40.0 < metrics["lufs_integrated"] < 0.0
    assert metrics["true_peak_dbtp"] is not None
    assert -40.0 < metrics["true_peak_dbtp"] <= 6.0
    assert 0.0 <= metrics["silence_ratio"] <= 1.0
    assert metrics["clipping_ratio"] >= 0.0
    assert payload["violations"] == []
    print(
        f"\n[wave3a] LUFS={metrics['lufs_integrated']} "
        f"TP={metrics['true_peak_dbtp']}dBTP "
        f"silence={metrics['silence_ratio']}"
    )


def test_voice_add_auto_ref_text_uses_real_asr(generated, capsys):
    tmp_path, audio = generated
    rc = _run(
        tmp_path,
        "voice",
        "add",
        "auto_ref",
        "--kind",
        "clone",
        "--ref-audio",
        str(audio),
        "--auto-ref-text",
        "--language",
        "Chinese",
        "--i-have-rights",
        "--json",
    )
    payload = json.loads(capsys.readouterr().out)
    assert rc == 0, payload
    source = payload["voice"]["source"]
    assert source["ref_text_source"] == "asr"
    assert isinstance(source["ref_text"], str) and source["ref_text"].strip()
    assert source["asr_model"]["id"] == "Qwen3-ASR-0.6B"
    assert (tmp_path / "data" / "voices" / "auto_ref" / "prompt.safetensors").is_file()
    print(f"\n[wave3a] ref_text={source['ref_text']!r}")
