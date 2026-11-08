"""``say.run(cancel_check=...)`` checkpoints (§4)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from qtts_cli.cli import build_parser
from qtts_cli.commands import say as say_cmd
from qtts_cli.config import build_config
from qtts_cli.errors import CancelledError


class FakeSayEngine:
    loads = 0
    generated = 0

    def __init__(self, cfg, model):
        self.cfg = cfg
        self.model_info = model
        self.device = "cpu"
        self.dtype_name = "fp32"
        self.load_ms = 0.0
        self.model = None

    def load(self):
        type(self).loads += 1
        self.model = object()
        return self

    def generate(self, **kwargs):
        type(self).generated += 1
        t = np.arange(2400, dtype=np.float64) / 24000.0
        audio = (0.2 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
        return audio, 24000, {"generate_ms": 1.0, "sample_rate": 24000, "duration_s": 0.1}

    def build_clone_prompt_items(self, *args, **kwargs):
        return []

    def extract_speaker_embedding(self, samples, sr):
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)


class CancelAfter:
    def __init__(self, count: int):
        self.remaining = count
        self.calls = 0

    def __call__(self) -> bool:
        self.calls += 1
        return self.calls >= self.remaining


def _args(
    tmp_path: Path,
    model_dir: Path,
    text: str,
    out: Path,
    *extra: str,
):
    argv = [
        "say",
        text,
        "--voice",
        "builtin:Vivian",
        "--out",
        str(out),
        "--model-dir",
        str(model_dir),
        "--data-dir",
        str(tmp_path / "data"),
        "--cache-dir",
        str(tmp_path / "cache"),
        *extra,
    ]
    args = build_parser().parse_args(argv)
    return build_config(args), args


@pytest.fixture()
def engine(monkeypatch):
    FakeSayEngine.loads = 0
    FakeSayEngine.generated = 0
    monkeypatch.setattr(say_cmd, "TtsEngine", FakeSayEngine)
    return FakeSayEngine


def test_cancel_before_any_model_access(engine, tmp_path: Path, model_dir: Path):
    out = tmp_path / "out.wav"
    cfg, args = _args(tmp_path, model_dir, "你好", out)
    with pytest.raises(CancelledError):
        say_cmd.run(cfg, args, cancel_check=lambda: True)
    assert engine.loads == 0
    assert engine.generated == 0
    assert not out.exists()


def test_cancel_after_preflight_before_acquire(engine, tmp_path: Path, model_dir: Path):
    out = tmp_path / "out.wav"
    cfg, args = _args(tmp_path, model_dir, "你好", out)
    check = CancelAfter(2)
    with pytest.raises(CancelledError):
        say_cmd.run(cfg, args, cancel_check=check)
    assert check.calls >= 2
    assert engine.loads == 0
    assert engine.generated == 0
    assert not out.exists()


def test_cancel_inside_segment_loop_leaves_no_output(engine, tmp_path: Path, model_dir: Path):
    out = tmp_path / "out.wav"
    cfg, args = _args(tmp_path, model_dir, "你好。世界。再见。", out)
    check = CancelAfter(4)
    with pytest.raises(CancelledError):
        say_cmd.run(cfg, args, cancel_check=check)
    assert engine.generated >= 1
    assert not out.exists()


def test_cancel_before_subtitles(engine, tmp_path: Path, model_dir: Path):
    out = tmp_path / "out.wav"
    cfg, args = _args(tmp_path, model_dir, "你好。世界。", out, "--subtitle", "srt")

    def cancel() -> bool:
        return engine.generated > 0

    with pytest.raises(CancelledError):
        say_cmd.run(cfg, args, cancel_check=cancel)
    assert not out.exists()
    assert not out.with_suffix(".srt").exists()


def test_cancel_before_audio_write(engine, tmp_path: Path, model_dir: Path):
    out = tmp_path / "out.wav"
    cfg, args = _args(tmp_path, model_dir, "你好", out)

    def cancel() -> bool:
        return engine.generated > 0

    with pytest.raises(CancelledError):
        say_cmd.run(cfg, args, cancel_check=cancel)
    assert engine.generated >= 1
    assert not out.exists()


def test_cancel_before_recipe_after_audio(engine, tmp_path: Path, model_dir: Path):
    out = tmp_path / "out.wav"
    recipe = tmp_path / "recipe.json"
    cfg, args = _args(
        tmp_path, model_dir, "你好", out, "--recipe-out", str(recipe)
    )

    def cancel() -> bool:
        return out.exists()

    with pytest.raises(CancelledError):
        say_cmd.run(cfg, args, cancel_check=cancel)
    assert out.exists()
    assert not recipe.exists()


def test_cancel_with_all_cache_hits(engine, tmp_path: Path, model_dir: Path):
    out = tmp_path / "out.wav"
    cfg, args = _args(tmp_path, model_dir, "你好", out)
    payload = say_cmd.run(cfg, args)
    assert payload["ok"] is True and out.exists()
    generated = engine.generated

    out2 = tmp_path / "out2.wav"
    cfg2, args2 = _args(tmp_path, model_dir, "你好", out2)
    with pytest.raises(CancelledError):
        say_cmd.run(cfg2, args2, cancel_check=CancelAfter(3))
    assert engine.generated == generated
    assert not out2.exists()


def test_cancel_check_none_is_unchanged(engine, tmp_path: Path, model_dir: Path):
    out = tmp_path / "out.wav"
    cfg, args = _args(tmp_path, model_dir, "你好", out)
    payload = say_cmd.run(cfg, args, cancel_check=None)
    assert payload["ok"] is True
    assert out.is_file()
