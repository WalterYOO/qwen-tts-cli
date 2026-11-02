"""Real-model smoke tests (set RUN_INTEGRATION=1; local models required).

Covers PRD §11.3, §11.5, §11.11 and §11.13: Chinese and English `say`, subtitle
text fidelity, and same-process TTS -> forced alignment.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from qtts_cli.cli import main

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS = REPO_ROOT / "models"
MODELS = Path(os.environ.get("QTTS_SMOKE_MODEL_DIR", DEFAULT_MODELS))

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("RUN_INTEGRATION") != "1",
        reason="set RUN_INTEGRATION=1 to run real-model smoke tests",
    ),
    pytest.mark.skipif(not MODELS.is_dir(), reason=f"model dir not found: {MODELS}"),
]

CHINESE_TEXT = "你好，世界。这是一次中文冒烟测试。"
ENGLISH_TEXT = "Hello world. This is an English smoke test."


def _say(tmp_path: Path, text: str, language: str, speaker: str, name: str):
    out = tmp_path / f"{name}.wav"
    started = time.perf_counter()
    rc = main(
        [
            "say",
            text,
            "--voice",
            f"builtin:{speaker}",
            "--language",
            language,
            "--model-dir",
            str(MODELS),
            "--data-dir",
            str(tmp_path / "data"),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--out",
            str(out),
            "--subtitle",
            "srt,json",
            "--json",
        ]
    )
    elapsed = time.perf_counter() - started
    assert rc == 0
    assert out.is_file() and out.stat().st_size > 0
    recipe_path = out.with_suffix(".recipe.json")
    assert recipe_path.is_file()
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    assert recipe["seed"] == 1234
    assert recipe["normalization"]["version"] >= 1
    for segment in recipe["segments"]:
        assert segment["language"]
        assert "pause_after_ms" in segment
        assert segment["text_sha256"]
    return out, recipe, elapsed


def test_chinese_say_with_subtitles(tmp_path: Path):
    out, recipe, elapsed = _say(tmp_path, CHINESE_TEXT, "Chinese", "Vivian", "zh")
    srt = out.with_suffix(".srt")
    assert srt.is_file()
    cues = [
        line
        for line in srt.read_text(encoding="utf-8").splitlines()
        if line and not line.isdigit() and "-->" not in line
    ]
    assert "".join(cues) == CHINESE_TEXT
    assert recipe["language"] == "Chinese"
    assert recipe["timing"]["total_ms"] > 0
    print(
        f"\n[smoke] zh: duration={_duration(out):.2f}s wall={elapsed:.2f}s "
        f"load={recipe['timing']['model_load_ms']}ms generate={recipe['timing']['generate_ms']}ms"
    )


def test_english_say_with_subtitles(tmp_path: Path):
    out, recipe, elapsed = _say(tmp_path, ENGLISH_TEXT, "English", "Ryan", "en")
    srt = out.with_suffix(".srt")
    cues = [
        line
        for line in srt.read_text(encoding="utf-8").splitlines()
        if line and not line.isdigit() and "-->" not in line
    ]
    assert "".join(cues) == ENGLISH_TEXT
    assert recipe["language"] == "English"
    print(
        f"\n[smoke] en: duration={_duration(out):.2f}s wall={elapsed:.2f}s "
        f"load={recipe['timing']['model_load_ms']}ms generate={recipe['timing']['generate_ms']}ms"
    )


def test_align_same_process_after_tts(tmp_path: Path):
    out, _, _ = _say(tmp_path, CHINESE_TEXT, "Chinese", "Vivian", "align_src")
    started = time.perf_counter()
    rc = main(
        [
            "align",
            str(out),
            "--text",
            CHINESE_TEXT,
            "--language",
            "Chinese",
            "--model-dir",
            str(MODELS),
            "--data-dir",
            str(tmp_path / "data"),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--format",
            "srt,vtt,ass,json",
            "--out-prefix",
            str(tmp_path / "aligned"),
            "--json",
        ]
    )
    elapsed = time.perf_counter() - started
    assert rc == 0, "same-process TTS -> align must succeed"
    data = json.loads((tmp_path / "aligned.json").read_text(encoding="utf-8"))
    assert "".join(cue["text"] for cue in data["cues"]) == CHINESE_TEXT
    assert data["cues"][0]["start"] >= 0
    assert data["cues"][-1]["end"] <= _duration(out) + 0.01
    assert data["cues"]
    for prev, cur in zip(data["cues"], data["cues"][1:]):
        assert cur["start"] >= prev["end"] - 1e-6
    print(f"\n[smoke] align: wall={elapsed:.2f}s cues={len(data['cues'])}")


def test_say_stdout_audio_is_not_polluted(tmp_path: Path, capsysbinary):
    rc = main(
        [
            "say", "你好。", "--voice", "builtin:Vivian", "--language", "Chinese",
            "--model-dir", str(MODELS), "--data-dir", str(tmp_path / "data"),
            "--cache-dir", str(tmp_path / "cache"), "--out", "-",
        ]
    )
    out, err = capsysbinary.readouterr()
    assert rc == 0, err
    assert out[:4] == b"RIFF"
    assert b"wrote" not in out
    assert b"wrote -" in err


def test_align_preserves_srt_cue_boundaries(tmp_path: Path):
    out, _, _ = _say(tmp_path, CHINESE_TEXT, "Chinese", "Vivian", "srt_src")
    source_srt = tmp_path / "source.srt"
    source_srt.write_text(
        "3\n00:00:00,000 --> 00:00:01,000\n你好，世界。\n\n"
        "8\n00:00:01,000 --> 00:00:02,000\n这是一次中文冒烟测试。\n",
        encoding="utf-8",
    )
    rc = main(
        [
            "align", str(out), "--srt", str(source_srt), "--language", "Chinese",
            "--model-dir", str(MODELS), "--data-dir", str(tmp_path / "data"),
            "--cache-dir", str(tmp_path / "cache"), "--format", "srt",
            "--out-prefix", str(tmp_path / "rewritten"), "--json",
        ]
    )
    assert rc == 0
    text = (tmp_path / "rewritten.srt").read_text(encoding="utf-8")
    assert text.startswith("3\n")
    assert "\n8\n" in text
    assert "你好，世界。" in text
    assert "这是一次中文冒烟测试。" in text


def _duration(path: Path) -> float:
    import soundfile as sf

    info = sf.info(str(path))
    return float(info.frames) / float(info.samplerate)
