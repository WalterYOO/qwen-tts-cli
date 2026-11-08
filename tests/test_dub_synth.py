"""FR-P2-04 §10.5/§10.6: synthesis, cache, fingerprints and timing."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from dub_utils import FakeRunner, read_audio, run_build, write_project, write_srt
from qtts_cli.dub import mix
from qtts_cli.errors import CancelledError, InputError, ModelError, QualityError


def _timeline(tmp_path: Path, *texts: str, seconds: float = 1.0) -> Path:
    entries = [
        (index, (index - 1) * seconds, index * seconds, text)
        for index, text in enumerate(texts, 1)
    ]
    return write_srt(tmp_path / "t.srt", entries)


def _build(timeline: Path, model_dir: Path, runner=None, *extra: str):
    return run_build(
        timeline,
        model_dir,
        runner,
        "--out",
        str(timeline.parent / "out.wav"),
        "--voice",
        "builtin:Vivian",
        "--no-loudness",
        "--keep-work",
        *extra,
    )


def _clear(out: Path, *, stems: bool = False) -> None:
    out.unlink(missing_ok=True)
    (out.parent / f"{out.stem}.dub.json").unlink(missing_ok=True)


def _cue_files(out: Path, index: int = 1) -> tuple[Path, Path]:
    root = out.parent / f"{out.stem}.dub-work" / "cues"
    return root / f"{index:04d}.wav", root / f"{index:04d}.json"


def test_one_call_per_cue_and_exact_say_adapter(tmp_path: Path, model_dir: Path, monkeypatch):
    import qtts_cli.commands.say as say_mod

    captured: list[dict] = []

    def fake_say(cfg, args, **kwargs):
        captured.append({"cfg": cfg, "args": args, "kwargs": kwargs})
        sf.write(str(args.out), np.zeros(2400, dtype=np.float32), 24000, subtype="PCM_16")
        return {
            "sample_rate": 24000,
            "duration_s": 0.1,
            "dtype": "bf16",
            "device": "cpu",
            "segments": [{"language": "English"}],
        }

    monkeypatch.setattr(say_mod, "run", fake_say)
    timeline = _timeline(tmp_path, "one", "two")
    project = tmp_path / "dub.yaml"
    project.write_text(
        "schema_version: 1\nvoice: builtin:Vivian\nlanguage: English\ninstruct: calm\n",
        encoding="utf-8",
    )
    payload = run_build(
        timeline,
        model_dir,
        None,
        "--out",
        str(tmp_path / "out.wav"),
        "--project",
        str(project),
        "--device",
        "cpu",
        "--no-loudness",
        "--keep-work",
        "--seed",
        "77",
    )
    assert len(captured) == 2
    assert payload["segments"] == {"total": 2, "synthesized": 2, "reused": 0}
    for index, item in enumerate(captured, 1):
        args = item["args"]
        assert item["cfg"].no_cache is True
        assert item["cfg"].json is False
        assert item["cfg"].seed == 77
        assert args.text in {"one", "two"}
        assert args.pause_ms == 0
        assert args.subtitle == []
        assert args.qc is False
        assert args.takes is None and args.select is None
        assert args.yes is True and args.dry_run is False
        assert args.language == "English"
        assert args.instruct == "calm"
        assert args.recipe_out == str(
            tmp_path / "out.dub-work" / "recipes" / f"{index:04d}.json"
        )
        assert item["kwargs"]["cancel_check"] is not None
    assert captured[0]["kwargs"]["engine_cache"] is captured[1]["kwargs"]["engine_cache"]
    assert captured[0]["kwargs"]["aligner_cache"] is captured[1]["kwargs"]["aligner_cache"]


def test_cache_reuse_and_excluded_changes(tmp_path: Path, model_dir: Path):
    timeline = _timeline(tmp_path, "one", "two")
    out = tmp_path / "out.wav"
    first = _build(timeline, model_dir, FakeRunner(seconds=0.5))
    assert first["segments"] == {"total": 2, "synthesized": 2, "reused": 0}
    _clear(out)
    second = FakeRunner(seconds=0.5)
    payload = _build(
        timeline,
        model_dir,
        second,
        "--sample-rate",
        "16000",
        "--min-speed",
        "0.5",
        "--max-speed",
        "1.5",
        "--stems-out",
        str(tmp_path / "stems"),
    )
    assert payload["segments"] == {"total": 2, "synthesized": 0, "reused": 2}
    assert second.calls == []


def test_fingerprint_invalidation(tmp_path: Path, model_dir: Path):
    # Pin device/dtype so the baseline is independent of the host's auto
    # runtime (CPU-only environments must resolve the same context).
    runtime = ("--device", "cpu", "--dtype", "bf16")
    timeline = _timeline(tmp_path, "one")
    out = tmp_path / "out.wav"
    _build(timeline, model_dir, FakeRunner(), *runtime)

    _clear(out)
    changed_text = _timeline(tmp_path, "changed")
    runner = FakeRunner()
    _build(changed_text, model_dir, runner, *runtime)
    assert runner.calls

    _clear(out)
    changed_seed = FakeRunner()
    _build(changed_text, model_dir, changed_seed, "--seed", "99", *runtime)
    assert changed_seed.calls

    _clear(out)
    changed_voice = FakeRunner()
    run_build(
        changed_text,
        model_dir,
        changed_voice,
        "--out",
        str(out),
        "--voice",
        "builtin:Serena",
        "--no-loudness",
        "--keep-work",
        *runtime,
    )
    assert changed_voice.calls

    _clear(out)
    changed_dtype = FakeRunner()
    run_build(
        changed_text,
        model_dir,
        changed_dtype,
        "--out",
        str(out),
        "--voice",
        "builtin:Serena",
        "--device",
        "cpu",
        "--dtype",
        "fp32",
        "--no-loudness",
        "--keep-work",
    )
    assert changed_dtype.calls


def test_language_and_instruct_invalidate_fingerprint(tmp_path: Path, model_dir: Path):
    timeline = _timeline(tmp_path, "one")
    project = write_project(
        tmp_path / "dub.yaml",
        "schema_version: 1\nvoice: builtin:Vivian\n"
        "language: English\ninstruct: calm\n",
    )
    out = tmp_path / "out.wav"
    common = (
        "--out",
        str(out),
        "--project",
        str(project),
        "--device",
        "cpu",
        "--dtype",
        "bf16",
        "--no-loudness",
        "--keep-work",
    )
    run_build(timeline, model_dir, FakeRunner(), *common)

    _clear(out)
    language_runner = FakeRunner()
    run_build(
        timeline,
        model_dir,
        language_runner,
        *common,
        "--language",
        "Chinese",
    )
    assert language_runner.calls
    assert language_runner.calls[0]["language"] == "Chinese"

    _clear(out)
    instruct_runner = FakeRunner()
    run_build(
        timeline,
        model_dir,
        instruct_runner,
        *common,
        "--language",
        "Chinese",
        "--instruct",
        "bright",
    )
    assert instruct_runner.calls
    assert instruct_runner.calls[0]["instruct"] == "bright"


def test_model_file_change_invalidates(tmp_path: Path, model_dir: Path):
    timeline = _timeline(tmp_path, "one")
    out = tmp_path / "out.wav"
    _build(timeline, model_dir, FakeRunner())
    weight = next((model_dir / "Qwen3-TTS-12Hz-1.7B-CustomVoice").glob("*.safetensors"))
    stat = weight.stat()
    os.utime(weight, (stat.st_atime + 5, stat.st_mtime + 5))
    _clear(out)
    runner = FakeRunner()
    _build(timeline, model_dir, runner)
    assert runner.calls


@pytest.mark.parametrize(
    "mutate",
    [
        lambda wav, sidecar: wav.write_bytes(wav.read_bytes()[:20]),
        lambda wav, sidecar: sidecar.write_text("{not json", encoding="utf-8"),
        lambda wav, sidecar: sf.write(str(wav), np.zeros(100, dtype=np.float32), 16000, subtype="PCM_16"),
        lambda wav, sidecar: _flip_byte(wav),
        lambda wav, sidecar: sidecar.unlink(),
    ],
)
def test_corrupt_cache_is_a_miss(tmp_path: Path, model_dir: Path, mutate):
    timeline = _timeline(tmp_path, "one")
    out = tmp_path / "out.wav"
    _build(timeline, model_dir, FakeRunner())
    wav, sidecar = _cue_files(out)
    mutate(wav, sidecar)
    _clear(out)
    runner = FakeRunner()
    payload = _build(timeline, model_dir, runner)
    assert payload["segments"] == {"total": 1, "synthesized": 1, "reused": 0}
    assert len(runner.calls) == 1


def _flip_byte(wav: Path) -> None:
    raw = bytearray(wav.read_bytes())
    raw[-10] ^= 0xFF
    wav.write_bytes(bytes(raw))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sample_rate", "oops"),
        ("frames", None),
        ("channels", 1.0),
        ("channels", True),
        ("languages_actual", 123),
        ("sha256", 5),
        ("duration_s", float("inf")),
    ],
)
def test_corrupt_sidecar_types_are_a_miss(tmp_path: Path, model_dir: Path, field: str, value):
    timeline = _timeline(tmp_path, "one")
    out = tmp_path / "out.wav"
    _build(timeline, model_dir, FakeRunner())
    _, sidecar = _cue_files(out)
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    payload[field] = value
    sidecar.write_text(json.dumps(payload), encoding="utf-8")
    _clear(out)
    runner = FakeRunner()
    result = _build(timeline, model_dir, runner)
    assert result["segments"]["synthesized"] == 1


def test_wav_is_written_before_the_sidecar(tmp_path: Path, model_dir: Path, monkeypatch):
    from qtts_cli.dub import synth as synth_mod

    timeline = _timeline(tmp_path, "one")
    out = tmp_path / "out.wav"
    real_write = synth_mod.CueCache.write

    def checked_write(self, index, **kwargs):
        wav, _ = self.paths(index)
        assert wav.is_file()
        real_write(self, index, **kwargs)
        raise ModelError("stop after a complete pair")

    monkeypatch.setattr(synth_mod.CueCache, "write", checked_write)
    with pytest.raises(ModelError):
        _build(timeline, model_dir, FakeRunner())
    wav, sidecar = _cue_files(out)
    assert wav.is_file() and sidecar.is_file()


def test_force_and_no_cache_bypass_reuse(tmp_path: Path, model_dir: Path):
    timeline = _timeline(tmp_path, "one")
    out = tmp_path / "out.wav"
    _build(timeline, model_dir, FakeRunner())
    _clear(out)
    forced = FakeRunner()
    assert _build(timeline, model_dir, forced, "--force")["segments"]["synthesized"] == 1
    assert forced.calls
    _clear(out)
    nocache = FakeRunner()
    assert _build(timeline, model_dir, nocache, "--no-cache")["segments"]["synthesized"] == 1
    assert nocache.calls


def test_fresh_invalid_output_is_model_error(tmp_path: Path, model_dir: Path):
    timeline = _timeline(tmp_path, "one")

    class Broken(FakeRunner):
        def synthesize(self, **kwargs):
            kwargs["out"].write_bytes(b"not a wav")
            return {"sample_rate": 24000, "duration_s": 0.1, "languages_actual": []}

    with pytest.raises(ModelError):
        _build(timeline, model_dir, Broken())


def test_cancellation_keeps_work_and_publishes_nothing(tmp_path: Path, model_dir: Path):
    timeline = _timeline(tmp_path, "one", "two")
    out = tmp_path / "out.wav"

    def stop(runner, call):
        if call == 2:
            raise CancelledError("stop")

    with pytest.raises(KeyboardInterrupt):
        _build(timeline, model_dir, FakeRunner(on_call=stop))
    assert not out.exists()
    assert not (tmp_path / "out.dub.json").exists()
    assert (tmp_path / "out.dub-work" / "cues").is_dir()


def test_inconsistent_synthesis_rate_is_model_error(tmp_path: Path, model_dir: Path):
    timeline = _timeline(tmp_path, "one", "two")
    out = tmp_path / "out.wav"
    with pytest.raises(ModelError):
        _build(timeline, model_dir, FakeRunner(sample_rates=[24000, 16000]))
    assert not out.exists()
    assert (tmp_path / "out.dub-work").is_dir()


def test_unsupported_rate_ratio_is_input_error(tmp_path: Path, model_dir: Path):
    timeline = _timeline(tmp_path, "one")
    out = tmp_path / "out.wav"
    with pytest.raises(InputError):
        _build(timeline, model_dir, FakeRunner(sample_rate=96000))
    assert not out.exists()
    assert (tmp_path / "out.dub-work" / "cues").is_dir()


def test_no_stretch_fit(tmp_path: Path, model_dir: Path):
    timeline = _timeline(tmp_path, "one")
    payload = _build(timeline, model_dir, FakeRunner(seconds=1.0))
    data, sr = read_audio(Path(payload["output"]["path"]))
    assert sr == 24000 and data.shape[0] == 24000
    assert payload["stretch"]["mean"] == 1.0
    assert payload["stretch"]["overflow_count"] == 0


def test_stretch_down_and_padding(tmp_path: Path, model_dir: Path):
    timeline = _timeline(tmp_path, "one")
    payload = _build(
        timeline,
        model_dir,
        FakeRunner(seconds=0.5),
        "--min-speed",
        "0.5",
    )
    data, sr = read_audio(Path(payload["output"]["path"]))
    assert data.shape[0] == 24000
    assert payload["stretch"]["mean"] == pytest.approx(0.5, abs=1e-6)
    assert not any("overflow" in warning for warning in payload["warnings"])


def test_clamp_and_overflow_warn(tmp_path: Path, model_dir: Path):
    timeline = _timeline(tmp_path, "one")
    payload = _build(timeline, model_dir, FakeRunner(seconds=2.0))
    assert payload["stretch"]["mean"] == pytest.approx(1.25, abs=1e-6)
    assert payload["stretch"]["overflow_count"] == 1
    assert any("overflows" in warning for warning in payload["warnings"])
    assert Path(payload["output"]["path"]).is_file()


def test_dead_band_never_warns(tmp_path: Path, model_dir: Path):
    timeline = _timeline(tmp_path, "one")
    payload = _build(timeline, model_dir, FakeRunner(seconds=1.004))
    assert payload["stretch"]["overflow_count"] == 0
    assert not any("overflow" in warning for warning in payload["warnings"])


def test_trailing_padding_has_no_leading_offset(tmp_path: Path, model_dir: Path):
    timeline = _timeline(tmp_path, "one")
    payload = _build(timeline, model_dir, FakeRunner(seconds=0.5))
    data, _ = read_audio(Path(payload["output"]["path"]))
    assert np.allclose(data[16000:], 0.0, atol=1e-6)


def test_overflow_fail_exits_5_and_publishes_nothing(tmp_path: Path, model_dir: Path):
    timeline = _timeline(tmp_path, "one", "two")
    out = tmp_path / "out.wav"
    with pytest.raises(QualityError) as caught:
        _build(
            timeline,
            model_dir,
            FakeRunner(seconds=2.0),
            "--overflow",
            "fail",
        )
    assert caught.value.code == 5
    assert caught.value.details["cues"] == [1, 2]
    assert not out.exists()
    assert (tmp_path / "out.dub-work" / "cues").is_dir()


def test_cue_beyond_video_is_input_error(tmp_path: Path, model_dir: Path):
    timeline = _timeline(tmp_path, "one", seconds=2.0)
    video = tmp_path / "audio-only.wav"
    sf.write(str(video), np.zeros(24000, dtype=np.float32), 24000, subtype="PCM_16")
    runner = FakeRunner(seconds=2.0)
    with pytest.raises(InputError):
        _build(timeline, model_dir, runner, "--video", str(video))
    assert runner.calls == []
    assert not (tmp_path / "out.wav").exists()


def test_rendered_video_overflow_warn_truncates(tmp_path: Path, model_dir: Path):
    # The cue itself fits the 1 s video, but its rendered audio (2 s, clamped
    # to 1.25x) extends beyond the video tail.
    timeline = _timeline(tmp_path, "one", seconds=1.0)
    video = tmp_path / "audio-only.wav"
    sf.write(str(video), np.zeros(24000, dtype=np.float32), 24000, subtype="PCM_16")
    payload = _build(
        timeline,
        model_dir,
        FakeRunner(seconds=2.0),
        "--video",
        str(video),
    )
    data, sr = read_audio(Path(payload["output"]["path"]))
    assert data.shape[0] == 24000
    assert any("beyond the video duration" in warning for warning in payload["warnings"])


def test_non_master_source_rate_is_resampled(tmp_path: Path, model_dir: Path):
    timeline = _timeline(tmp_path, "one")
    for rate in (12000, 44100):
        out = tmp_path / f"out_{rate}.wav"
        run_build(
            timeline,
            model_dir,
            FakeRunner(sample_rate=rate, seconds=1.0),
            "--out",
            str(out),
            "--voice",
            "builtin:Vivian",
            "--no-loudness",
            "--keep-work",
        )
        data, sr = read_audio(out)
        assert sr == 24000
        assert data.shape[0] == 24000


def test_half_up_frame_geometry():
    assert mix.cue_geometry(0.0005, 1.0005, 12000, 1) == mix.CueGeometry(6, 12006, 12000)
    assert mix.cue_geometry(0.5, 1.5, 24000, 1).start_frame == 12000
    assert mix.cue_geometry(0.5, 1.5, 44100, 1).start_frame == 22050
    with pytest.raises(InputError):
        mix.cue_geometry(1.0, 1.0, 24000, 1)


def test_overlapping_placement_sums():
    master = np.zeros(10, dtype=np.float32)
    mix.place(master, np.full(6, 0.25, dtype=np.float32), 2)
    mix.place(master, np.full(6, 0.5, dtype=np.float32), 4)
    assert np.allclose(master[2:4], 0.25)
    assert np.allclose(master[4:8], 0.75)
    assert np.allclose(master[8:], 0.5)
