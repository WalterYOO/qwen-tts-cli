"""Regression tests for codex P0 review findings 1-10."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from qtts_cli import catalog
from qtts_cli.commands import voice as voice_cmd
from qtts_cli.config import Config
from qtts_cli.errors import CapabilityError, InputError
from qtts_cli.plan import build_plan
from qtts_cli.sidecar import build_recipe, recipe_path
from qtts_cli.subtitles import (
    AlignItem,
    build_cues,
    build_cues_for_ranges,
    coverage,
    full_text,
    monotonic,
    parse_srt,
    render,
)
from qtts_cli.voice_store import VoiceStore

REPO_ROOT = Path(__file__).resolve().parents[1]


def run_cli(model_dir: Path, tmp_path: Path, *args: str):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)
    cmd = [
        sys.executable,
        "-m",
        "qtts_cli",
        *args,
        "--model-dir",
        str(model_dir),
        "--data-dir",
        str(tmp_path / "data"),
        "--cache-dir",
        str(tmp_path / "cache"),
    ]
    return subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(tmp_path))


# --------------------------------------------------------------------------- #
# P0-1: aligner prerequisites before TTS load
# --------------------------------------------------------------------------- #
def test_dry_run_reports_missing_aligner(small_model_dir: Path, tmp_path: Path):
    completed = run_cli(
        small_model_dir, tmp_path,
        "say", "你好。", "--voice", "builtin:Vivian", "--subtitle", "srt", "--dry-run",
    )
    assert completed.returncode == 3, completed.stderr
    assert "aligner" in completed.stderr.lower()
    assert "Traceback" not in completed.stderr


def test_dry_run_reports_aligner_and_dependencies(model_dir: Path, tmp_path: Path):
    completed = run_cli(
        model_dir, tmp_path,
        "say", "你好。", "--voice", "builtin:Vivian", "--subtitle", "srt", "--dry-run", "--json",
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert "ForcedAligner" in payload["aligner"]["model"]
    assert isinstance(payload["dependencies"]["align"], bool)
    assert isinstance(payload["dependencies"]["tts"], bool)


# --------------------------------------------------------------------------- #
# P0-2: stdout has exactly one consumer
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "args",
    [
        ("say", "你好。", "--voice", "builtin:Vivian", "--out", "-",
         "--subtitle", "srt", "--subtitle-out-prefix", "-", "--dry-run"),
        ("say", "你好。", "--voice", "builtin:Vivian", "--json",
         "--subtitle", "srt", "--subtitle-out-prefix", "-", "--dry-run"),
        ("say", "你好。", "--voice", "builtin:Vivian", "--out", "-",
         "--subtitle", "srt", "--dry-run"),
        ("say", "你好。", "--voice", "builtin:Vivian", "--dry-run",
         "--recipe-out", "-", "--json"),
    ],
)
def test_say_stdout_conflicts(model_dir: Path, tmp_path: Path, args):
    completed = run_cli(model_dir, tmp_path, *args)
    assert completed.returncode == 2, completed.stderr
    assert "stdout" in completed.stderr.lower() or "subtitle" in completed.stderr.lower()


def test_multi_format_subtitles_to_stdout_rejected(model_dir: Path, tmp_path: Path):
    completed = run_cli(
        model_dir, tmp_path,
        "say", "你好。", "--voice", "builtin:Vivian", "--dry-run",
        "--subtitle", "srt,vtt", "--subtitle-out-prefix", "-",
    )
    assert completed.returncode == 2
    assert "one subtitle format" in completed.stderr


def test_align_stdout_json_conflict_precedes_model_load(model_dir: Path, tmp_path: Path):
    completed = run_cli(
        model_dir, tmp_path,
        "align", str(tmp_path / "missing.wav"), "--text", "你好", "--out-prefix", "-", "--json",
    )
    assert completed.returncode == 2, completed.stderr
    assert "stdout" in completed.stderr.lower()


def test_align_multi_format_stdout_rejected(model_dir: Path, tmp_path: Path):
    completed = run_cli(
        model_dir, tmp_path,
        "align", str(tmp_path / "missing.wav"), "--text", "你好",
        "--out-prefix", "-", "--format", "srt,vtt",
    )
    assert completed.returncode == 2
    assert "one format" in completed.stderr


# --------------------------------------------------------------------------- #
# P0-4: plain builtins become exit-code errors, never tracebacks
# --------------------------------------------------------------------------- #
def test_unknown_subtitle_format_is_usage_error(model_dir: Path, tmp_path: Path):
    completed = run_cli(
        model_dir, tmp_path, "say", "你好。", "--voice", "builtin:Vivian",
        "--subtitle", "mp4", "--dry-run",
    )
    assert completed.returncode == 2, completed.stderr
    assert "unsupported subtitle format" in completed.stderr
    assert "Traceback" not in completed.stderr


def test_missing_config_file_is_input_error(model_dir: Path, tmp_path: Path):
    completed = run_cli(
        model_dir, tmp_path, "say", "你好。", "--voice", "builtin:Vivian",
        "--dry-run", "--config", str(tmp_path / "nope.json"),
    )
    assert completed.returncode == 4, completed.stderr
    assert "config file not found" in completed.stderr
    assert "Traceback" not in completed.stderr


def test_invalid_config_json_is_usage_error(model_dir: Path, tmp_path: Path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    completed = run_cli(
        model_dir, tmp_path, "say", "你好。", "--voice", "builtin:Vivian",
        "--dry-run", "--config", str(bad),
    )
    assert completed.returncode == 2, completed.stderr
    assert "invalid JSON" in completed.stderr
    assert "Traceback" not in completed.stderr


# --------------------------------------------------------------------------- #
# P0-5: --json error stream is exactly one JSON object
# --------------------------------------------------------------------------- #
def test_json_error_emits_one_object_on_stderr(model_dir: Path, tmp_path: Path):
    completed = run_cli(
        model_dir, tmp_path, "say", "你好。", "--voice", "builtin:Vivian",
        "--json", "--subtitle", "mp4",
    )
    assert completed.returncode == 2
    assert completed.stdout == ""
    payload = json.loads(completed.stderr)
    assert payload["ok"] is False
    assert payload["error"] == "usage"


def test_argparse_error_with_json_is_single_object(model_dir: Path, tmp_path: Path):
    completed = run_cli(
        model_dir, tmp_path, "say", "你好", "--voice", "builtin:Vivian",
        "--format", "mp4", "--json", "--dry-run",
    )
    assert completed.returncode == 2, completed.stderr
    assert completed.stdout == ""
    payload = json.loads(completed.stderr)
    assert payload["ok"] is False
    assert payload["error"] == "usage"


def test_argparse_error_without_json_has_no_traceback(model_dir: Path, tmp_path: Path):
    completed = run_cli(
        model_dir, tmp_path, "say", "你好", "--voice", "builtin:Vivian",
        "--format", "mp4", "--dry-run",
    )
    assert completed.returncode == 2
    assert "error:" in completed.stderr
    assert "Traceback" not in completed.stderr


# --------------------------------------------------------------------------- #
# P0-6: sidecar path, seed, stdout recipe
# --------------------------------------------------------------------------- #
def test_recipe_path_replaces_audio_suffix():
    assert recipe_path("out.wav") == Path("out.recipe.json")
    assert recipe_path("dir/clip.flac") == Path("dir/clip.recipe.json")


def test_recipe_has_top_level_seed():
    recipe = build_recipe(
        qtts_version="0.0.0", text="hi", source="test", voice_ref="builtin:Ryan",
        voice=None, model={"id": "m"}, sampling={}, segments=[], created_at="2026-11-03T00:00:00Z",
        seed=99, normalization={"version": 2},
    )
    assert recipe["seed"] == 99
    assert recipe["normalization"]["version"] == 2


def test_dry_run_will_write_recipe_sidecar(model_dir: Path, tmp_path: Path):
    completed = run_cli(
        model_dir, tmp_path, "say", "你好。", "--voice", "builtin:Vivian",
        "--out", str(tmp_path / "clip.wav"), "--dry-run", "--json",
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert str(tmp_path / "clip.recipe.json") in payload["will_write"]


def test_stdout_audio_recipe_falls_back_to_data_dir(model_dir: Path, tmp_path: Path):
    completed = run_cli(
        model_dir, tmp_path, "say", "你好。", "--voice", "builtin:Vivian",
        "--out", "-", "--dry-run",
    )
    assert completed.returncode == 0, completed.stderr
    assert "recipes" in completed.stdout and ".recipe.json" in completed.stdout


# --------------------------------------------------------------------------- #
# P0-7: subtitle mapping robustness
# --------------------------------------------------------------------------- #
def test_partial_alignment_is_filled_and_clamped():
    text = "ABCDE"
    cues = build_cues(text, [AlignItem("C", 0.1, 0.2)], granularity="char", audio_duration=1.0)
    assert full_text(cues) == text
    assert all(0.0 <= cue.start <= cue.end <= 1.0 for cue in cues)
    for prev, cur in zip(cues, cues[1:]):
        assert cur.start >= prev.end - 1e-9
    assert coverage(text, [AlignItem("C", 0.1, 0.2)]) == pytest.approx(0.2)


def test_out_of_range_times_are_clamped():
    text = "你好，世界。"
    items = [AlignItem("你好", 0.0, 5.0), AlignItem("世界", 5.0, 9.0)]
    cues = monotonic(build_cues(text, items, audio_duration=0.5), audio_duration=0.5)
    assert full_text(cues) == text
    assert all(0.0 <= cue.start <= cue.end <= 0.5 for cue in cues)


def test_zero_duration_at_audio_tail_is_reallocated():
    cues = build_cues(
        "AB",
        [AlignItem("A", 2.0, 2.1), AlignItem("B", 2.1, 2.2)],
        granularity="char",
        audio_duration=2.0,
    )
    assert [cue.text for cue in cues] == ["A", "B"]
    assert all(cue.end > cue.start for cue in cues)
    assert all(cue.end <= 2.0 + 1e-9 for cue in cues)
    assert cues[0].end <= cues[1].start + 1e-9


def test_impossible_cue_fit_raises():
    text = "你好世界你好世界"
    with pytest.raises(InputError, match="cannot fit"):
        build_cues(text, [AlignItem(text, 0.0, 1.0)], granularity="char", audio_duration=0.005)


def test_repeated_text_maps_sequentially():
    text = "abcabc"
    items = [AlignItem("a", 0.0, 0.1), AlignItem("b", 0.1, 0.2), AlignItem("c", 0.2, 0.3),
             AlignItem("a", 0.3, 0.4), AlignItem("b", 0.4, 0.5), AlignItem("c", 0.5, 0.6)]
    cues = build_cues(text, items, granularity="char")
    assert coverage(text, items) == pytest.approx(1.0)
    assert [c.text for c in cues] == ["a", "b", "c", "a", "b", "c"]


def test_punctuation_does_not_lower_coverage():
    text = "你好，世界。"
    items = [AlignItem("你", 0, 1), AlignItem("好", 1, 2), AlignItem("世", 2, 3), AlignItem("界", 3, 4)]
    assert coverage(text, items) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# P0-8 / P0-9: capability checks before loading
# --------------------------------------------------------------------------- #
def test_voice_add_clone_rejects_wrong_model_kind_before_load(model_dir: Path, tmp_path: Path):
    import numpy as np
    import soundfile as sf

    ref = tmp_path / "ref.wav"
    t = np.linspace(0, 3.0, int(24000 * 3.0), endpoint=False)
    sf.write(ref, (0.3 * np.sin(2 * np.pi * 220 * t)).astype("float32"), 24000)
    completed = run_cli(
        model_dir, tmp_path,
        "voice", "add", "bob", "--kind", "clone", "--ref-audio", str(ref),
        "--ref-text", "你好", "--i-have-rights",
        "--model", "Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    )
    assert completed.returncode == 4, completed.stderr
    assert "cannot run mode" in completed.stderr
    assert "Traceback" not in completed.stderr
    assert not (tmp_path / "data" / "voices" / "bob").exists()


def test_base_model_rejects_instruct_before_load(model_dir: Path, tmp_path: Path):
    voice_dir = tmp_path / "data" / "voices" / "alice"
    voice_dir.mkdir(parents=True)
    (voice_dir / "voice.json").write_text(json.dumps({
        "schema_version": 1,
        "name": "alice",
        "kind": "clone",
        "source": {"type": "clone", "mode": "xvec"},
        "model": {"id": "m", "revision": None},
    }), encoding="utf-8")
    (voice_dir / "prompt.safetensors").write_bytes(b"fake")
    completed = run_cli(
        model_dir, tmp_path,
        "say", "你好。", "--voice", "voice:alice", "--instruct", "温柔", "--dry-run",
    )
    assert completed.returncode == 4, completed.stderr
    assert "instruct" in completed.stderr.lower()
    assert "Traceback" not in completed.stderr


def test_preview_missing_prompt_rejected_before_load(monkeypatch, model_dir: Path, tmp_path: Path):
    voice_dir = tmp_path / "data" / "voices" / "alice"
    voice_dir.mkdir(parents=True)
    (voice_dir / "voice.json").write_text(json.dumps({
        "schema_version": 1,
        "name": "alice",
        "kind": "clone",
        "source": {"type": "clone", "mode": "icl", "ref_text": "你好"},
        "model": {"id": "m", "revision": None},
    }), encoding="utf-8")
    calls = {"loaded": False}

    def fake_load(self):
        calls["loaded"] = True
        return self

    monkeypatch.setattr(voice_cmd.TtsEngine, "load", fake_load)
    cfg = Config(data_dir=tmp_path / "data", model_dir=model_dir, device="cpu", dtype="fp32", seed=1)
    args = SimpleNamespace(name="alice", text=None, out=None, format=None, language="Chinese")
    with pytest.raises(InputError, match="prompt.safetensors"):
        voice_cmd.preview(cfg, args)
    assert calls["loaded"] is False


def test_per_segment_language_is_validated(tmp_path: Path):
    root = tmp_path / "models"
    model = root / "Qwen3-TTS-12Hz-1.7B-CustomVoice"
    model.mkdir(parents=True)
    (model / "config.json").write_text(json.dumps({
        "model_type": "qwen3_tts",
        "tts_model_type": "custom_voice",
        "tts_model_size": "1b7",
        "talker_config": {"spk_id": {"vivian": 1}, "codec_language_id": {"chinese": 1}},
    }), encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"fake")
    cfg = Config(data_dir=tmp_path / "data", model_dir=root, dtype="fp32", device="cpu", seed=1)
    with pytest.raises(CapabilityError, match="English"):
        build_plan(
            cfg, text="Hello world.", voice_ref="builtin:Vivian", language="auto",
            models=catalog.scan(root), store=VoiceStore(cfg.data_dir),
        )


# --------------------------------------------------------------------------- #
# P0-10: format resolution
# --------------------------------------------------------------------------- #
def test_unknown_output_suffix_is_usage_error(model_dir: Path, tmp_path: Path):
    completed = run_cli(
        model_dir, tmp_path, "say", "你好。", "--voice", "builtin:Vivian",
        "--out", str(tmp_path / "out.mp3"), "--dry-run",
    )
    assert completed.returncode == 2, completed.stderr
    assert "unknown suffix" in completed.stderr


def test_explicit_format_must_match_suffix(model_dir: Path, tmp_path: Path):
    completed = run_cli(
        model_dir, tmp_path, "say", "你好。", "--voice", "builtin:Vivian",
        "--out", str(tmp_path / "out.flac"), "--format", "wav", "--dry-run",
    )
    assert completed.returncode == 2, completed.stderr
    assert "does not match" in completed.stderr


def test_no_suffix_defaults_to_wav(model_dir: Path, tmp_path: Path):
    completed = run_cli(
        model_dir, tmp_path, "say", "你好。", "--voice", "builtin:Vivian",
        "--out", str(tmp_path / "clip"), "--dry-run", "--json",
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["plan"]["output"]["format"] == "wav"


# --------------------------------------------------------------------------- #
# P1-3 / P1-9: SRT boundaries and indices survive alignment
# --------------------------------------------------------------------------- #
def test_srt_parser_preserves_index_and_boundaries():
    raw = "7\n00:00:01,000 --> 00:00:02,500\n你好\n\n9\n00:00:03,000 --> 00:00:04,000\n世界\n"
    cues = parse_srt(raw)
    assert [c.index for c in cues] == [7, 9]
    assert cues[0].start == pytest.approx(1.0) and cues[1].end == pytest.approx(4.0)


def test_range_builder_preserves_requested_ranges_and_indices():
    text = "你好\n世界"
    items = [AlignItem("你", 0.0, 0.2), AlignItem("好", 0.2, 0.4),
             AlignItem("世", 0.6, 0.8), AlignItem("界", 0.8, 1.0)]
    cues = build_cues_for_ranges(text, items, [(0, 2), (3, 5)], audio_duration=1.0)
    assert [c.text for c in cues] == ["你好", "世界"]
    cues[0].index = 7
    cues[1].index = 9
    srt = render(cues, "srt")
    assert srt.startswith("7\n")
    assert "\n9\n" in srt
    assert cues[1].start >= cues[0].end
