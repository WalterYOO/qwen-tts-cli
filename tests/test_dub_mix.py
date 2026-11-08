"""FR-P2-04 §10.7/§10.8: mixing, stems, bed, loudness and timeline.json."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from dub_utils import FakeRunner, manifest_of, read_audio, run_build, write_srt
from qtts_cli import audio_qc
from qtts_cli.dub import mix
from qtts_cli.errors import InputError
from qtts_cli.utils import sha256_file, sha256_text

HAVE_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
ffmpeg_only = pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg/ffprobe not installed")


def _timeline(tmp_path: Path, *texts: str, seconds: float = 1.0) -> Path:
    entries = [
        (index, (index - 1) * seconds, index * seconds, text)
        for index, text in enumerate(texts, 1)
    ]
    return write_srt(tmp_path / "t.srt", entries)


def _build(timeline: Path, model_dir: Path, *extra: str, runner=None):
    return run_build(
        timeline,
        model_dir,
        runner or FakeRunner(seconds=1.0),
        "--out",
        str(timeline.parent / "out.wav"),
        "--voice",
        "builtin:Vivian",
        "--keep-work",
        *extra,
    )


def test_stems_length_offsets_and_timeline_json(tmp_path: Path, model_dir: Path):
    timeline = _timeline(tmp_path, "plain", "alice: hello", "bob: hi")
    payload = _build(
        timeline,
        model_dir,
        "--no-loudness",
        "--stems-out",
        str(tmp_path / "stems"),
    )
    master, sr = read_audio(Path(payload["output"]["path"]))
    anonymous, _ = read_audio(tmp_path / "stems" / "__anonymous__.wav")
    alice, _ = read_audio(tmp_path / "stems" / "alice.wav")
    bob, _ = read_audio(tmp_path / "stems" / "bob.wav")
    for stem in (anonymous, alice, bob):
        assert stem.shape == master.shape
    assert np.max(np.abs(anonymous[0:24000])) > 0.05
    assert np.allclose(anonymous[24000:], 0.0, atol=1e-6)
    assert np.allclose(alice[0:24000], 0.0, atol=1e-6)
    assert np.max(np.abs(alice[24000:48000])) > 0.05
    assert np.allclose(bob[0:48000], 0.0, atol=1e-6)

    document = json.loads((tmp_path / "stems" / "timeline.json").read_text(encoding="utf-8"))
    assert document["schema_version"] == 1
    assert document["sample_rate"] == 24000
    assert document["timeline"]["format"] == "srt"
    assert document["counts"]["cues"] == 3
    assert document["cues"][0]["text_sha256"] == sha256_text("plain")
    assert document["cues"][1]["speaker"] == "alice"
    assert document["files"]["alice"] == "alice.wav"
    for name, digest in document["hashes"].items():
        assert sha256_file(tmp_path / "stems" / name) == digest
    raw = (tmp_path / "stems" / "timeline.json").read_text(encoding="utf-8")
    assert raw.endswith("\n")
    assert raw == json.dumps(document, ensure_ascii=False, sort_keys=True) + "\n"


def test_stems_are_not_normalized(tmp_path: Path, model_dir: Path):
    timeline = _timeline(tmp_path, "plain")
    stems_dir = tmp_path / "stems"
    _build(timeline, model_dir, "--no-loudness", "--stems-out", str(stems_dir))
    before = sha256_file(stems_dir / "__anonymous__.wav")

    (tmp_path / "out.wav").unlink()
    (tmp_path / "out.dub.json").unlink()
    shutil.rmtree(stems_dir)
    payload = _build(timeline, model_dir, "--lufs", "-10", "--stems-out", str(stems_dir))
    after = sha256_file(stems_dir / "__anonymous__.wav")
    assert payload["segments"]["reused"] == 1
    assert before == after


def test_video_length_sets_stem_length(tmp_path: Path, model_dir: Path):
    timeline = _timeline(tmp_path, "plain", seconds=1.0)
    video = tmp_path / "audio.wav"
    sf.write(str(video), np.zeros(48000, dtype=np.float32), 24000, subtype="PCM_16")
    payload = _build(
        timeline,
        model_dir,
        "--no-loudness",
        "--video",
        str(video),
        "--stems-out",
        str(tmp_path / "stems"),
    )
    master, _ = read_audio(Path(payload["output"]["path"]))
    stem, _ = read_audio(tmp_path / "stems" / "__anonymous__.wav")
    assert master.shape[0] == stem.shape[0] == 48000


@ffmpeg_only
def test_bed_gain_and_trim(tmp_path: Path, model_dir: Path):
    timeline = _timeline(tmp_path, "plain", seconds=2.0)
    video = tmp_path / "audio-only.wav"
    bed = (0.25 * np.sin(2 * np.pi * 440 * np.arange(48000) / 24000)).astype(np.float32)
    sf.write(str(video), bed, 24000, subtype="PCM_16")
    payload = _build(
        timeline,
        model_dir,
        "--no-loudness",
        "--video",
        str(video),
        "--bed-db",
        "-6",
        runner=FakeRunner(seconds=2.0, silent=True),
    )
    data, sr = read_audio(Path(payload["output"]["path"]))
    assert data.shape[0] == 48000
    expected = 0.25 * 10 ** (-6 / 20)
    assert np.max(np.abs(data)) == pytest.approx(expected, abs=0.01)
    manifest = manifest_of(tmp_path / "out.wav")
    assert manifest["bed"]["volume_db"] == -6
    assert manifest["bed"]["sha256"] == sha256_file(video)


@ffmpeg_only
def test_bed_missing_audio_stream(tmp_path: Path, model_dir: Path):
    timeline = _timeline(tmp_path, "plain")
    video = tmp_path / "video.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=black:s=32x32:d=1",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video)],
        check=True,
    )
    with pytest.raises(InputError):
        _build(
            timeline,
            model_dir,
            "--no-loudness",
            "--video",
            str(video),
            "--bed-db",
            "-6",
        )


def test_estimated_budget_rejected_before_synthesis(tmp_path: Path, model_dir: Path, monkeypatch):
    timeline = _timeline(tmp_path, "plain")
    monkeypatch.setattr(mix, "MAX_MASTER_BYTES", 1000)
    runner = FakeRunner(seconds=1.0)
    with pytest.raises(InputError):
        _build(timeline, model_dir, "--no-loudness", runner=runner)
    assert runner.calls == []


def test_bed_and_master_budget_rejected_before_synthesis(
    tmp_path: Path, model_dir: Path, monkeypatch
):
    timeline = _timeline(tmp_path, "plain")
    video = tmp_path / "audio-only.wav"
    sf.write(str(video), np.zeros(24000, dtype=np.float32), 24000, subtype="PCM_16")
    monkeypatch.setattr(mix, "MAX_MASTER_BYTES", 1000)
    runner = FakeRunner(seconds=1.0)
    with pytest.raises(InputError):
        _build(
            timeline,
            model_dir,
            "--no-loudness",
            "--video",
            str(video),
            "--bed-db",
            "-6",
            runner=runner,
        )
    assert runner.calls == []
    assert not (tmp_path / "out.wav").exists()


def test_actual_budget_rejected_after_overflow(tmp_path: Path, model_dir: Path, monkeypatch):
    timeline = _timeline(tmp_path, "one", "two")
    monkeypatch.setattr(mix, "MAX_MASTER_BYTES", 200_000)
    runner = FakeRunner(seconds=3.0)
    with pytest.raises(InputError):
        _build(
            timeline,
            model_dir,
            "--no-loudness",
            "--max-speed",
            "2.0",
            runner=runner,
        )
    assert runner.calls
    assert not (tmp_path / "out.wav").exists()
    assert (tmp_path / "out.dub-work" / "cues").is_dir()


def test_normalize_limiter_and_measurements():
    sr = 24000
    loud = (0.99 * np.sin(2 * np.pi * 440 * np.arange(sr) / sr)).astype(np.float32)
    result = mix.normalize(loud, sr, enabled=True, target_lufs=-18, true_peak=-1.5)
    assert result.limiter_ran is True
    assert result.verified_ok is True
    measured = audio_qc.true_peak_dbtp(result.mix, sr)[0]
    assert measured <= -1.5 + 0.1
    lufs, _ = audio_qc.integrated_loudness(result.mix, sr)
    assert lufs == pytest.approx(-18.0, abs=0.3)


def test_normalize_silent_lufs_none_leaves_mix_unchanged():
    sr = 24000
    silent = np.zeros(sr, dtype=np.float32)
    result = mix.normalize(silent, sr, enabled=True, target_lufs=-18, true_peak=-1.5)
    assert result.limiter_ran is False
    assert np.array_equal(result.mix, silent)
    assert result.verified_ok is None
    assert any("loudness unavailable" in warning for warning in result.warnings)


def test_no_loudness_above_ceiling_warns_but_keeps_audio():
    sr = 24000
    loud = np.full(sr, 1.2, dtype=np.float32)
    result = mix.normalize(loud, sr, enabled=False, target_lufs=-18, true_peak=-1.5)
    assert result.limiter_ran is False
    assert result.verified_ok is False
    assert result.actual_true_peak_dbtp is not None
    assert np.array_equal(result.mix, loud)
    assert any("exceeds" in warning for warning in result.warnings)


def test_no_loudness_below_ceiling_is_verified():
    sr = 24000
    signal = (0.25 * np.sin(2 * np.pi * 440 * np.arange(sr) / sr)).astype(np.float32)
    result = mix.normalize(signal, sr, enabled=False, target_lufs=-18, true_peak=-1.5)
    assert result.verified_ok is True
    assert result.actual_true_peak_dbtp is not None


def test_clipped_detection():
    assert mix.is_clipped(np.array([0.0, 1.2], dtype=np.float32))
    assert mix.is_clipped(np.array([-1.2, 0.0], dtype=np.float32))
    assert not mix.is_clipped(np.array([0.5, -0.5], dtype=np.float32))


def test_integration_loudness_and_true_peak(tmp_path: Path, model_dir: Path):
    timeline = _timeline(tmp_path, "plain")
    payload = _build(timeline, model_dir)
    data, sr = read_audio(Path(payload["output"]["path"]))
    lufs, _ = audio_qc.integrated_loudness(data, sr)
    assert lufs == pytest.approx(-18.0, abs=0.3)
    peak, _ = audio_qc.true_peak_dbtp(data, sr)
    assert peak <= -1.5 + 0.1
    assert payload["loudness"]["verified_ok"] is True


class _HotRunner(FakeRunner):
    def synthesize(self, **kwargs):
        frames = 24000
        signal = np.full(frames, 0.99, dtype=np.float32)
        sf.write(str(kwargs["out"]), signal, 24000, subtype="PCM_16")
        return {"sample_rate": 24000, "duration_s": 1.0, "languages_actual": []}


def test_no_loudness_records_actual_peak_and_warns(tmp_path: Path, model_dir: Path):
    timeline = _timeline(tmp_path, "plain")
    payload = _build(timeline, model_dir, "--no-loudness", runner=_HotRunner(seconds=1.0))
    assert payload["ok"] is True
    assert payload["loudness"]["verified_ok"] is False
    assert payload["loudness"]["true_peak_dbtp"] > -1.5
    assert any("exceeds" in warning for warning in payload["warnings"])


def test_silent_mix_lufs_none_branch(tmp_path: Path, model_dir: Path):
    timeline = _timeline(tmp_path, "plain")
    payload = _build(timeline, model_dir, runner=FakeRunner(seconds=1.0, silent=True))
    assert payload["ok"] is True
    assert any("loudness unavailable" in warning for warning in payload["warnings"])
    assert payload["loudness"]["verified_ok"] is None


@ffmpeg_only
def test_lossy_encoded_peak_recorded(tmp_path: Path, model_dir: Path):
    timeline = _timeline(tmp_path, "plain")
    payload = _build(
        timeline,
        model_dir,
        "--format",
        "m4a",
        "--out",
        str(tmp_path / "out.m4a"),
    )
    assert Path(payload["output"]["path"]).is_file()
    assert payload["loudness"]["encoded_true_peak_dbtp"] is not None


@ffmpeg_only
def test_lossy_decode_failure_warns_and_succeeds(tmp_path: Path, model_dir: Path, monkeypatch):
    from qtts_cli.dub import encode as dub_encode

    monkeypatch.setattr(dub_encode, "decode_peak", lambda *args, **kwargs: None)
    timeline = _timeline(tmp_path, "plain")
    payload = _build(
        timeline,
        model_dir,
        "--format",
        "m4a",
        "--out",
        str(tmp_path / "out.m4a"),
    )
    assert payload["ok"] is True
    assert payload["loudness"]["encoded_true_peak_dbtp"] is None
    assert payload["loudness"]["verified_ok"] is False
    assert any("encoded true peak is unavailable" in warning for warning in payload["warnings"])


def test_stems_are_materialized_one_speaker_at_a_time(
    tmp_path: Path, model_dir: Path, monkeypatch
):
    import weakref

    real_allocate = mix.allocate
    state = {"live": 0, "peak": 0}

    def released():
        state["live"] -= 1

    def tracked(frames):
        array = real_allocate(frames)
        state["live"] += 1
        state["peak"] = max(state["peak"], state["live"])
        weakref.finalize(array, released)
        return array

    monkeypatch.setattr(mix, "allocate", tracked)
    timeline = _timeline(tmp_path, "plain", "alice: hello", "bob: hi")
    payload = _build(
        timeline,
        model_dir,
        "--no-loudness",
        "--stems-out",
        str(tmp_path / "stems"),
    )
    master, sr = read_audio(Path(payload["output"]["path"]))
    for key in ("__anonymous__", "alice", "bob"):
        stem, stem_sr = read_audio(tmp_path / "stems" / f"{key}.wav")
        assert stem_sr == sr
        assert stem.shape == master.shape
    # One master plus at most the stem currently being materialized.
    assert state["peak"] <= 2


def test_no_loudness_between_ceiling_and_tolerance_is_not_verified(monkeypatch):
    monkeypatch.setattr(
        audio_qc, "true_peak_dbtp", lambda data, sr: (-1.45, None)
    )
    result = mix.normalize(
        np.zeros(24000, dtype=np.float32),
        24000,
        enabled=False,
        target_lufs=-18,
        true_peak=-1.5,
    )
    assert result.limiter_ran is False
    assert result.verified_ok is False
    assert any("exceeds -1.50" in warning for warning in result.warnings)


def test_lufs_unavailable_between_ceiling_and_tolerance_is_not_verified(monkeypatch):
    monkeypatch.setattr(
        audio_qc, "integrated_loudness", lambda data, sr: (None, "silent")
    )
    monkeypatch.setattr(
        audio_qc, "true_peak_dbtp", lambda data, sr: (-1.45, None)
    )
    result = mix.normalize(
        np.zeros(24000, dtype=np.float32),
        24000,
        enabled=True,
        target_lufs=-18,
        true_peak=-1.5,
    )
    assert result.limiter_ran is False
    assert result.verified_ok is False
    assert any("exceeds -1.50" in warning for warning in result.warnings)


@ffmpeg_only
def test_encoded_peak_above_ceiling_is_not_verified(
    tmp_path: Path, model_dir: Path, monkeypatch
):
    from qtts_cli.dub import encode as dub_encode

    monkeypatch.setattr(
        dub_encode,
        "decode_peak",
        lambda path, sr: {"true_peak_dbtp": -1.4},
    )
    timeline = _timeline(tmp_path, "plain")
    payload = _build(
        timeline,
        model_dir,
        "--format",
        "m4a",
        "--out",
        str(tmp_path / "out.m4a"),
    )
    assert payload["loudness"]["verified_ok"] is False
    assert payload["loudness"]["encoded_true_peak_dbtp"] == -1.4
    assert any("exceeds -1.50" in warning for warning in payload["warnings"])
