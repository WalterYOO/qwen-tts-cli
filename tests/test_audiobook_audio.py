"""FR-P2-03 §12.6/§12.7: assembly, ducking, fading and BGM fallbacks."""

from __future__ import annotations

import io
import math
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from qtts_cli.audiobook import audio
from qtts_cli.errors import InputError, ModelError


def _seg(seconds: float, sr: int, value: float = 0.1) -> np.ndarray:
    return np.full(int(round(seconds * sr)), value, dtype=np.float32)


def test_frames_and_ms_rounding():
    assert audio.frames_for_ms(1, 1000) == 1
    assert audio.frames_for_ms(1, 8000) == 8
    assert audio.frames_for_ms(0, 24000) == 0
    assert audio.ms_for_frame(120, 24000) == 5
    assert audio.ms_for_frame(1, 24000) == 0
    assert audio.ms_for_frame(2, 24000) == 0
    assert audio.ms_for_frame(3, 24000) == 0
    assert audio.ms_for_frame(12, 24000) == 1
    assert audio.ms_for_frame(1, 3) == 333


def test_assembly_exact_silence_and_offsets():
    sr = 1000
    assembled = audio.assemble(
        [[_seg(0.1, sr), _seg(0.2, sr)], [_seg(0.1, sr)]],
        [[sr, sr], [sr]],
        sr,
        paragraph_ms=100,
        chapter_ms=200,
    )
    assert assembled.full.shape[0] == 700
    assert assembled.start_frames == [0, 600]
    assert assembled.end_frames == [400, 700]
    assert assembled.start_ms == [0, 600]
    assert assembled.end_ms == [400, 700]
    # Chapter 1 is 0.1 + 0.1 pause + 0.2 = 0.4 s; chapter 2 is 0.1 s.
    assert np.all(assembled.full[0:100] == 0.1)
    assert np.all(assembled.full[100:200] == 0)
    assert np.all(assembled.full[200:400] == 0.1)
    # The chapter pause is silent and belongs to the chapter transition.
    assert np.all(assembled.full[400:600] == 0)
    assert np.all(assembled.full[600:700] == 0.1)


def test_resampled_length_matches_resampler():
    for in_rate, out_rate in ((16000, 24000), (44100, 48000), (22050, 24000), (48000, 24000)):
        for frames in (1, 2, 3, 7, 100, 999, 1001, 12345):
            expected = audio.resampled_length(frames, in_rate, out_rate)
            actual = audio.resample_segment(
                np.zeros(frames, dtype=np.float32), in_rate, out_rate
            ).shape[0]
            assert actual == expected


def test_frame_levels_matches_naive_across_chunks():
    rng = np.random.default_rng(0)
    total = (1 << 20) + 12345  # forces a second internal chunk
    speech = (rng.standard_normal(total) * 0.1).astype(np.float32)
    speech[1000:1300] = 0
    frame = 160
    frames = math.ceil(total / frame)
    got = audio._frame_levels(speech, frame, frames)
    expected = np.empty(frames, dtype=np.float64)
    for k in range(frames):
        chunk = speech[k * frame : min((k + 1) * frame, total)]
        rms = math.sqrt(float(np.mean(np.square(chunk, dtype=np.float64))))
        expected[k] = 20.0 * math.log10(rms + 1e-12)
    assert np.allclose(got, expected, atol=1e-9)


def test_resample_and_ratio_guard():
    sr = 16000
    data = np.ones(1600, dtype=np.float32)
    out = audio.resample_segment(data, sr, 24000)
    assert out.shape[0] == 2400
    with pytest.raises(InputError):
        audio.resample_segment(data, sr, 1000)
    with pytest.raises(InputError):
        audio.resample_segment(data, sr, 40000)


def _reference_duck(speech, sr, ducking_db):
    total = speech.shape[0]
    frame = max(1, int(round(0.020 * sr)))
    count = math.ceil(total / frame)
    levels = []
    for k in range(count):
        chunk = speech[k * frame : min((k + 1) * frame, total)]
        rms = math.sqrt(float(np.mean(np.square(chunk, dtype=np.float64))))
        levels.append(20.0 * math.log10(rms + 1e-12))
    targets = [ducking_db if level >= -35.0 else 0.0 for level in levels]
    ys = [targets[0]]
    for k in range(1, count):
        tau = 0.05 if targets[k] < targets[k - 1] else 0.30
        a = math.exp(-frame / (tau * sr))
        ys.append(a * ys[-1] + (1 - a) * targets[k])
    centres = np.array([k * frame + frame / 2 for k in range(count)])
    return np.interp(np.arange(total), centres, np.array(ys))


def _reference_mix(speech, bgm, sr, volume_db, ducking_db, fade_ms):
    total = speech.shape[0]
    duck_db = _reference_duck(speech, sr, ducking_db)
    duck = 10.0 ** (duck_db / 20.0)
    if total < 3 or fade_ms == 0:
        fd = 0
    else:
        fd = min(int(round(fade_ms * sr / 1000.0)), max(1, (total - 1) // 2))
    fade = np.ones(total)
    if fd:
        for n in range(fd):
            fade[n] = n / fd
        for n in range(total - fd, total):
            fade[n] = (total - 1 - n) / fd
    g = 10.0 ** (volume_db / 20.0)
    length = bgm.shape[0]
    return np.array(
        [
            float(speech[n]) + float(bgm[n % length]) * g * fade[n] * duck[n]
            for n in range(total)
        ],
        dtype=np.float32,
    )


def test_ducking_matches_reference_formula():
    sr = 8000
    total = sr
    speech = 0.2 * np.sin(2 * np.pi * 220 * np.arange(total) / sr)
    speech = speech.astype(np.float32)
    # The second half is a pause, so the ducking releases.
    speech[sr // 2 :] = 0
    bgm = np.full(1000, 0.5, dtype=np.float32)
    reference = _reference_mix(speech.copy(), bgm, sr, -6.0, -10.0, 50)
    master = speech.copy()
    mixed = audio.mix_bgm(master, bgm, sr, volume_db=-6.0, ducking_db=-10.0, fade_ms=50)
    assert mixed is master  # writable float32 master is mixed in place
    assert mixed.dtype == np.float32
    assert np.allclose(mixed, reference, atol=1e-5)

    readonly = speech.copy()
    readonly.setflags(write=False)
    copied = audio.mix_bgm(
        readonly, bgm, sr, volume_db=-6.0, ducking_db=-10.0, fade_ms=50
    )
    assert copied is not readonly
    assert np.allclose(copied, reference, atol=1e-5)

    duck_db = audio.duck_envelope_db(speech, sr, -10.0)
    during = duck_db[sr // 4 : sr // 4 + sr // 8]
    assert abs(float(np.mean(during)) - (-10.0)) < 1.5
    release = duck_db[sr // 2 + sr // 3 : sr // 2 + sr // 3 + sr // 8]
    # The one-pole release is gradual (tau 0.30 s), so after 1/3 s it has
    # recovered well above the -10 dB attack level without reaching 0 yet.
    assert -4.0 < float(np.mean(release)) < -1.0


def test_bgm_phase_continuity_and_looping():
    sr = 1000
    total = 500
    speech = np.zeros(total, dtype=np.float32)
    bgm = np.arange(1, 101, dtype=np.float32) / 100.0
    mixed = audio.mix_bgm(
        speech, bgm, sr, volume_db=0.0, ducking_db=0.0, fade_ms=0
    )
    for n in (0, 99, 100, 101, 250, 499):
        assert mixed[n] == pytest.approx(float(bgm[n % 100]), rel=0, abs=1e-6)


def test_fade_and_clamp():
    sr = 1000
    total = 1000
    gain = audio.fade_gain(total, sr, 100)
    assert gain[0] == pytest.approx(0.0)
    assert gain[50] == pytest.approx(0.5)
    assert gain[total // 2] == pytest.approx(1.0)
    assert gain[-1] == pytest.approx(0.0)
    clamped = audio.fade_gain(101, sr, 10000)
    assert clamped[50] == pytest.approx(1.0)
    assert np.all(audio.fade_gain(2, sr, 100) == 1.0)
    assert np.all(audio.fade_gain(10, sr, 0) == 1.0)


def _write_bgm(path: Path, data: np.ndarray, sr: int) -> Path:
    sf.write(str(path), data, sr, subtype="PCM_16")
    return path


def test_load_bgm_multichannel_downmix_and_loop_flag(tmp_path: Path):
    sr = 8000
    data = np.zeros((8000, 4), dtype=np.float32)
    data[:, 0] = 0.5
    data[:, 1] = 0.5
    data[:, 2] = 0.5
    data[:, 3] = 0.5
    path = _write_bgm(tmp_path / "four.wav", data, sr)
    loaded = audio.load_bgm(path, sr)
    assert loaded is not None
    assert loaded.source_channels == 4
    assert loaded.samples.shape[0] == 8000
    assert float(np.max(np.abs(loaded.samples - 0.5))) < 0.01


def test_empty_bgm_is_an_error(tmp_path: Path):
    path = tmp_path / "empty.wav"
    path.write_bytes(b"")
    with pytest.raises(InputError):
        audio.load_bgm(path, 8000)
    with pytest.raises(InputError):
        audio.mix_bgm(
            np.zeros(10, dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            8000,
            volume_db=0,
            ducking_db=0,
            fade_ms=0,
        )


def test_ffmpeg_fallback(monkeypatch, tmp_path: Path):
    sr = 8000
    path = _write_bgm(tmp_path / "tone.wav", np.full(8000, 0.25, dtype=np.float32), sr)
    monkeypatch.setattr(audio, "_soundfile_info", lambda _: None)
    loaded = audio.load_bgm(path, 4000)
    assert loaded is not None
    assert loaded.samples.shape[0] == pytest.approx(4000, abs=2)
    assert loaded.source_sample_rate == sr

    monkeypatch.setattr(audio, "BGM_MAX_SECONDS", 0.1)
    with pytest.raises(InputError):
        audio.load_bgm(path, 4000)


class _FakeStdout:
    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = list(chunks)
        self.reads = 0

    def read(self, size: int) -> bytes:
        self.reads += 1
        return self.chunks.pop(0) if self.chunks else b""

    def close(self) -> None:
        pass


class _FakeProcess:
    def __init__(self, chunks: list[bytes], *, returncode: int = 0, stderr: bytes = b"") -> None:
        self.stdout = _FakeStdout(chunks)
        self.stderr = io.BytesIO(stderr) if stderr is not None else None
        self.returncode = None
        self.killed = False
        self.final_returncode = returncode

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9

    def wait(self) -> int:
        if self.returncode is None:
            self.returncode = self.final_returncode
        return self.returncode


def test_ffmpeg_fallback_streams_bounded_chunks(monkeypatch):
    monkeypatch.setattr(audio.shutil, "which", lambda _: "/usr/bin/ffmpeg")
    payload = np.array([0.25, -0.5, 0.0, 1.0], dtype="<f4").tobytes()
    process = _FakeProcess([payload, b""])
    monkeypatch.setattr(audio.subprocess, "Popen", lambda *args, **kwargs: process)
    decoded = audio._decode_ffmpeg(Path("bgm.ogg"), 8000, {"duration_s": 10.0})
    assert np.array_equal(decoded, np.array([0.25, -0.5, 0.0, 1.0], dtype=np.float32))
    assert process.stdout.reads == 2
    assert not process.killed


def test_ffmpeg_fallback_kills_on_overrun(monkeypatch):
    monkeypatch.setattr(audio.shutil, "which", lambda _: "/usr/bin/ffmpeg")
    oversized = b"\x00" * 8000
    process = _FakeProcess([oversized, b"more"])
    monkeypatch.setattr(audio.subprocess, "Popen", lambda *args, **kwargs: process)
    with pytest.raises(InputError):
        # 0.001 s at 1 kHz caps the decode at ~1001 frames (4004 bytes).
        audio._decode_ffmpeg(Path("bgm.ogg"), 1000, {"duration_s": 0.001})
    assert process.killed
    assert process.stdout.reads == 1


def test_ffmpeg_fallback_failure_and_empty_output(monkeypatch):
    monkeypatch.setattr(audio.shutil, "which", lambda _: "/usr/bin/ffmpeg")
    failing = _FakeProcess([b""], returncode=1, stderr=b"decode failed")
    monkeypatch.setattr(audio.subprocess, "Popen", lambda *args, **kwargs: failing)
    with pytest.raises(InputError) as error:
        audio._decode_ffmpeg(Path("bgm.ogg"), 8000, {"duration_s": 1.0})
    assert "decode failed" in error.value.details.get("stderr", "")

    empty = _FakeProcess([b""])
    monkeypatch.setattr(audio.subprocess, "Popen", lambda *args, **kwargs: empty)
    with pytest.raises(InputError):
        audio._decode_ffmpeg(Path("bgm.ogg"), 8000, {"duration_s": 1.0})


def test_ffmpeg_missing_on_fallback_is_exit_3(monkeypatch, tmp_path: Path):
    path = _write_bgm(tmp_path / "tone.wav", np.full(800, 0.25, dtype=np.float32), 8000)
    monkeypatch.setattr(audio, "_soundfile_info", lambda _: None)
    monkeypatch.setattr(audio.shutil, "which", lambda _: None)
    with pytest.raises(ModelError):
        audio.load_bgm(path, 8000)


def test_dry_run_probe_does_not_decode(monkeypatch, tmp_path: Path):
    sr = 8000
    path = _write_bgm(tmp_path / "tone.wav", np.full(800, 0.25, dtype=np.float32), sr)
    monkeypatch.setattr(
        audio.audio_io,
        "read_audio",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("decoded")),
    )
    meta = audio.probe_bgm(path, sr)
    assert meta["sample_rate"] == sr
    assert audio.load_bgm(path, sr, dry_run=True) is None
