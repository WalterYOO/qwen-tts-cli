"""FR-P2-03 §12.2/§12.3/§12.5/§12.6b: fingerprints, cache, adapter, cancel."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from audiobook_utils import FakeRunner, inline_project, make_args, read_audio, run_build
from qtts_cli import audiobook
from qtts_cli.audiobook.segments import SaySegmentRunner, SegmentCache
from qtts_cli.config import build_config
from qtts_cli.errors import CancelledError
from qtts_cli.voice_store import VoiceStore


def _clear_outputs(project: Path) -> None:
    build = project.parent / "build"
    for name in ("book.wav", "book.json"):
        (build / name).unlink(missing_ok=True)


def _build(project: Path, model_dir: Path, runner, *extra: str, data_dir=None):
    return run_build(project, model_dir, runner, "--keep-stems", *extra, data_dir=data_dir)


def _make_clone(tmp_path: Path) -> Path:
    data_dir = tmp_path / "data"
    store = VoiceStore(data_dir)
    store.create(
        name="alice",
        kind="clone",
        source={"ref_text": "hello there", "mode": "icl"},
        model={"id": "Qwen3-TTS-12Hz-1.7B-Base"},
    )
    (store.dir_for("alice") / "prompt.safetensors").write_bytes(b"prompt")
    sf.write(str(store.dir_for("alice") / "ref.wav"), np.zeros(2400, dtype=np.float32), 24000)
    return data_dir


def test_voice_precedence_project_chapter_cli(tmp_path: Path, model_dir: Path):
    project = inline_project(
        tmp_path,
        chapters=[
            {"title": "A", "text": "one"},
            {"title": "B", "voice": "builtin:Serena", "text": "two"},
        ],
    )
    runner = FakeRunner()
    _build(project, model_dir, runner)
    assert [call["voice"] for call in runner.calls] == ["builtin:Vivian", "builtin:Serena"]

    _clear_outputs(project)
    runner2 = FakeRunner()
    _build(project, model_dir, runner2, "--voice", "builtin:Ryan")
    assert {call["voice"] for call in runner2.calls} == {"builtin:Ryan"}


def test_language_instruct_seed_and_one_call_per_paragraph(tmp_path: Path, model_dir: Path):
    project = inline_project(
        tmp_path,
        chapters=[
            {"title": "A", "language": "English", "instruct": "soft", "text": "one\n\ntwo"},
        ],
    )
    runner = FakeRunner()
    _build(project, model_dir, runner, "--seed", "77")
    assert len(runner.calls) == 2
    for call in runner.calls:
        assert call["language"] == "English"
        assert call["instruct"] == "soft"
        assert call["seed"] == 77
        assert call["out"].parent.name == "segments"
    assert [call["text"] for call in runner.calls] == ["one", "two"]


def test_segment_reuse_and_master_only_changes(tmp_path: Path, model_dir: Path):
    project = inline_project(tmp_path, chapters=[{"title": "A", "text": "one\n\ntwo"}])
    runner = FakeRunner()
    first = _build(project, model_dir, runner, "--sample-rate", "24000")
    assert first["segments"] == {"total": 2, "synthesized": 2, "reused": 0}

    _clear_outputs(project)
    second = FakeRunner()
    reused = _build(
        project,
        model_dir,
        second,
        "--sample-rate",
        "16000",
        "--paragraph-pause-ms",
        "100",
        "--out",
        str(project.parent / "other.flac"),
    )
    assert reused["segments"] == {"total": 2, "synthesized": 0, "reused": 2}
    assert second.calls == []
    (project.parent / "other.flac").unlink()


def test_text_and_seed_changes_invalidate(tmp_path: Path, model_dir: Path):
    project = inline_project(tmp_path, chapters=[{"title": "A", "text": "one"}])
    _build(project, model_dir, FakeRunner())
    _clear_outputs(project)
    changed_seed = FakeRunner()
    _build(project, model_dir, changed_seed, "--seed", "99")
    assert changed_seed.calls
    _clear_outputs(project)
    changed_project = inline_project(tmp_path, chapters=[{"title": "A", "text": "changed"}])
    changed_text = FakeRunner()
    _build(changed_project, model_dir, changed_text)
    assert changed_text.calls


def test_dtype_change_invalidates(tmp_path: Path, model_dir: Path):
    project = inline_project(tmp_path, chapters=[{"title": "A", "text": "one"}])
    _build(project, model_dir, FakeRunner(), "--device", "cpu", "--dtype", "fp32")
    _clear_outputs(project)
    runner = FakeRunner()
    _build(project, model_dir, runner, "--device", "cpu", "--dtype", "bf16")
    assert runner.calls


def test_clone_ref_audio_and_model_mtime_invalidate(tmp_path: Path, model_dir: Path):
    data_dir = _make_clone(tmp_path)
    project = inline_project(tmp_path, chapters=[{"title": "A", "text": "one"}])
    # Replace the project voice with the clone reference.
    project.write_text(project.read_text().replace("builtin:Vivian", "voice:alice"), encoding="utf-8")
    _build(project, model_dir, FakeRunner(), data_dir=data_dir)
    _clear_outputs(project)
    reused = FakeRunner()
    second = _build(project, model_dir, reused, data_dir=data_dir)
    assert second["segments"]["reused"] == 1

    ref = VoiceStore(data_dir).dir_for("alice") / "ref.wav"
    data, sr = read_audio(ref)
    sf.write(str(ref), np.full_like(data, 0.1), sr)  # same size, different bytes
    _clear_outputs(project)
    changed_ref = FakeRunner()
    third = _build(project, model_dir, changed_ref, data_dir=data_dir)
    assert third["segments"]["synthesized"] == 1

    weight = next(Path(model_dir / "Qwen3-TTS-12Hz-1.7B-Base").glob("*.safetensors"))
    stat = weight.stat()
    os.utime(weight, (stat.st_atime + 5, stat.st_mtime + 5))
    _clear_outputs(project)
    changed_model = FakeRunner()
    fourth = _build(project, model_dir, changed_model, data_dir=data_dir)
    assert fourth["segments"]["synthesized"] == 1


def _segment_paths(project: Path) -> tuple[Path, Path]:
    root = project.parent / "build" / "segments"
    return root / "0001-0001.wav", root / "0001-0001.json"


def _corrupt_and_rebuild(project: Path, model_dir: Path, mutate, *, expect_synth=True):
    wav, sidecar = _segment_paths(project)
    mutate(wav, sidecar)
    _clear_outputs(project)
    runner = FakeRunner()
    payload = _build(project, model_dir, runner)
    assert payload["segments"]["total"] == 1
    if expect_synth:
        assert payload["segments"]["synthesized"] == 1
    return payload


def test_corrupt_cache_is_a_miss_not_an_error(tmp_path: Path, model_dir: Path):
    project = inline_project(tmp_path, chapters=[{"title": "A", "text": "one"}])
    _build(project, model_dir, FakeRunner())

    def truncate(wav, sidecar):
        wav.write_bytes(wav.read_bytes()[:20])

    assert _corrupt_and_rebuild(project, model_dir, truncate)["ok"] is True

    def malformed(wav, sidecar):
        sidecar.write_text("{not json", encoding="utf-8")

    assert _corrupt_and_rebuild(project, model_dir, malformed)["ok"] is True

    def header(wav, sidecar):
        sf.write(str(wav), np.zeros(100, dtype=np.float32), 16000, subtype="PCM_16")

    assert _corrupt_and_rebuild(project, model_dir, header)["ok"] is True

    def sha(wav, sidecar):
        raw = bytearray(wav.read_bytes())
        raw[-10] ^= 0xFF
        wav.write_bytes(bytes(raw))

    assert _corrupt_and_rebuild(project, model_dir, sha)["ok"] is True

    def missing_sidecar(wav, sidecar):
        sidecar.unlink()

    assert _corrupt_and_rebuild(project, model_dir, missing_sidecar)["ok"] is True


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sample_rate", "oops"),
        ("frames", None),
        ("channels", 1.0),
        ("channels", True),
        ("languages_actual", 123),
        ("sha256", 5),
    ],
)
def test_corrupt_sidecar_types_are_a_miss(
    tmp_path: Path, model_dir: Path, field: str, value
):
    project = inline_project(tmp_path, chapters=[{"title": "A", "text": "one"}])
    _build(project, model_dir, FakeRunner())
    _, sidecar = _segment_paths(project)
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    payload[field] = value
    sidecar.write_text(json.dumps(payload), encoding="utf-8")
    _clear_outputs(project)
    runner = FakeRunner()
    result = _build(project, model_dir, runner)
    assert result["segments"] == {"total": 1, "synthesized": 1, "reused": 0}
    assert len(runner.calls) == 1


def test_sidecar_languages_requires_string_list(tmp_path: Path):
    root = tmp_path / "segments"
    root.mkdir()
    cache = SegmentCache(root, enabled=True)
    _, sidecar = cache.paths(1, 1)
    sidecar.write_text('{"languages_actual": 123}', encoding="utf-8")
    assert audiobook._sidecar_languages(cache, 1, 1) == []
    sidecar.write_text('{"languages_actual": ["Chinese"]}', encoding="utf-8")
    assert audiobook._sidecar_languages(cache, 1, 1) == ["Chinese"]


def test_force_and_no_cache_bypass_reuse(tmp_path: Path, model_dir: Path):
    project = inline_project(tmp_path, chapters=[{"title": "A", "text": "one"}])
    _build(project, model_dir, FakeRunner())
    _clear_outputs(project)
    forced = FakeRunner()
    assert _build(project, model_dir, forced, "--force")["segments"]["reused"] == 0
    assert forced.calls
    _clear_outputs(project)
    nocache = FakeRunner()
    assert _build(project, model_dir, nocache, "--no-cache")["segments"]["reused"] == 0
    assert nocache.calls


def test_cancellation_before_commit_keeps_no_output(tmp_path: Path, model_dir: Path):
    project = inline_project(tmp_path, chapters=[{"title": "A", "text": "one\n\ntwo"}])

    def stop(runner, call):
        if call == 2:
            raise CancelledError("stop")

    with pytest.raises(KeyboardInterrupt):
        _build(project, model_dir, FakeRunner(on_call=stop))
    build = project.parent / "build"
    assert not (build / "book.wav").exists()
    assert not (build / "book.json").exists()
    assert not (build / ".audiobook.publish.json").exists()
    assert not (build / ".audiobook.lock").exists()
    assert (build / "segments").is_dir()


def test_say_adapter_arguments(tmp_path: Path, model_dir: Path, monkeypatch):
    import qtts_cli.commands.say as say_mod

    captured = {}

    def fake_say(cfg, args, **kwargs):
        captured["cfg"] = cfg
        captured["args"] = args
        captured["kwargs"] = kwargs
        sf.write(str(args.out), np.zeros(240, dtype=np.float32), 24000, subtype="PCM_16")
        return {
            "sample_rate": 24000,
            "duration_s": 0.01,
            "dtype": "bf16",
            "device": "cpu",
            "segments": [{"language": "Chinese"}, {"language": "English"}],
        }

    monkeypatch.setattr(say_mod, "run", fake_say)
    project = inline_project(tmp_path, chapters=[{"title": "A", "text": "one"}])
    args = make_args(project, model_dir)
    cfg = build_config(args)
    runner = SaySegmentRunner(cfg, args, tmp_path / "recipes")
    result = runner.synthesize(
        text="one",
        out=tmp_path / "one.wav",
        voice="builtin:Vivian",
        language="auto",
        instruct=None,
        seed=1234,
        cancel_check=lambda: False,
    )
    assert captured["cfg"].no_cache is True
    assert captured["cfg"].json is False
    assert captured["cfg"].seed == 1234
    assert captured["args"].pause_ms == 0
    assert captured["args"].format == "wav"
    assert captured["args"].recipe_out == str(tmp_path / "recipes" / "one.json")
    assert captured["kwargs"]["cancel_check"] is not None
    assert result["languages_actual"] == ["Chinese", "English"]


def test_fresh_invalid_output_fails_with_exit_3(tmp_path: Path, model_dir: Path):
    project = inline_project(tmp_path, chapters=[{"title": "A", "text": "one"}])

    class Broken(FakeRunner):
        def synthesize(self, **kwargs):
            kwargs["out"].write_bytes(b"not a wav")
            return {"sample_rate": 24000, "duration_s": 0.1, "languages_actual": []}

    from qtts_cli.errors import ModelError

    with pytest.raises(ModelError):
        _build(project, model_dir, Broken())
