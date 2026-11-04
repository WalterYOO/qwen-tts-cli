"""Real-model FR-P1-04 smoke tests (RUN_INTEGRATION=1)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from qtts_cli.catalog import scan
from qtts_cli.cli import main

_MODEL_DIR = os.environ.get("QTTS_SMOKE_MODEL_DIR")
MODELS = Path(_MODEL_DIR) if _MODEL_DIR else None

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("RUN_INTEGRATION") != "1",
        reason="set RUN_INTEGRATION=1 to run real-model smoke tests",
    ),
    pytest.mark.skipif(
        MODELS is None or not MODELS.is_dir(),
        reason="set QTTS_SMOKE_MODEL_DIR to an existing model directory",
    ),
]

TEXT = "你好，这是多 take 测试。"


@pytest.fixture(autouse=True)
def _warm_model_imports(capsys):
    """Emit third-party import warnings before the test captures stdout.

    ``qwen_tts``/``qwen_asr`` print a banner while importing; without a warm
    import it lands in front of the first ``--json`` document.
    """
    for name in ("qwen_tts", "qwen_asr"):
        try:
            __import__(name)
        except ImportError:
            pass
    capsys.readouterr()


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


def _say(tmp_path: Path, *extra: str) -> int:
    return _run(
        tmp_path,
        "say",
        TEXT,
        "--voice",
        "builtin:Vivian",
        "--language",
        "Chinese",
        "--out",
        str(tmp_path / "out.wav"),
        *extra,
    )


def test_integration_takes_lufs_library_roundtrip(tmp_path: Path, capsys):
    rc = _say(tmp_path, "--takes", "2", "--select", "lufs", "--json")
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    job_id = payload["takes"]["job_id"]
    job_dir = tmp_path / "data" / "takes" / job_id
    job = json.loads((job_dir / "job.json").read_text(encoding="utf-8"))
    assert job["state"] == "complete"
    assert job["selected"] in (1, 2)
    for take in job["takes"]:
        assert take["metrics"]["lufs"] is not None
        assert (job_dir / take["file"]).is_file()

    assert _run(tmp_path, "takes", "list", job_id, "--json") == 0
    listed = json.loads(capsys.readouterr().out)
    assert listed["job_id"] == job_id

    assert _run(tmp_path, "takes", "compare", job_id, "--json") == 0
    compared = json.loads(capsys.readouterr().out)
    assert len(compared["takes"]) == 2

    target = tmp_path / "promoted.wav"
    assert (
        _run(
            tmp_path,
            "takes",
            "promote",
            job_id,
            "--id",
            "2",
            "--out",
            str(target),
            "--json",
        )
        == 0
    )
    json.loads(capsys.readouterr().out)
    assert target.read_bytes() == (job_dir / "take-02.wav").read_bytes()


def test_integration_takes_cer(tmp_path: Path, capsys):
    if not any(model.kind == "asr" for model in scan(MODELS)):
        pytest.skip("no local ASR model for --select cer")
    rc = _say(tmp_path, "--takes", "2", "--select", "cer", "--json")
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    job = json.loads(
        (tmp_path / "data" / "takes" / payload["takes"]["job_id"] / "job.json").read_text(
            encoding="utf-8"
        )
    )
    assert job["state"] == "complete"
    assert job["selected"] in (1, 2)
    assert all(take["metrics"]["cer"] is not None for take in job["takes"])
