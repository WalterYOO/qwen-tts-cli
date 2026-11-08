"""FR-P2-03 §12.8/§12.9: loudness, lossless encoding, ffmpeg and M4B chapters."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from audiobook_utils import FakeRunner, inline_project, read_audio, run_build
from qtts_cli import audio_qc
from qtts_cli.audiobook import encode
from qtts_cli.errors import ModelError

HAVE_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
ffmpeg_only = pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg/ffprobe not installed")


def _build(project: Path, model_dir: Path, runner=None, *extra: str):
    return run_build(project, model_dir, runner or FakeRunner(), *extra)


class _PeakStdout:
    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = list(chunks)
        self.read_sizes: list[int] = []

    def read(self, size: int) -> bytes:
        self.read_sizes.append(size)
        return self.chunks.pop(0) if self.chunks else b""

    def close(self) -> None:
        pass


class _PeakProcess:
    def __init__(self, chunks: list[bytes], *, returncode: int = 0) -> None:
        self.stdout = _PeakStdout(chunks)
        self.stderr = None
        self.returncode = None
        self.final_returncode = returncode

    def kill(self) -> None:
        self.returncode = -9

    def wait(self) -> int:
        if self.returncode is None:
            self.returncode = self.final_returncode
        return self.returncode


def test_decode_peak_streams_popen_chunks(monkeypatch):
    sr = 8000
    signal = (0.5 * np.sin(2 * np.pi * 440 * np.arange(sr) / sr)).astype("<f4")
    split = sr // 3
    chunks = [signal[:split].tobytes(), signal[split:].tobytes(), b""]
    process = _PeakProcess(chunks)
    monkeypatch.setattr(encode.shutil, "which", lambda _: "/usr/bin/ffmpeg")
    monkeypatch.setattr(encode.subprocess, "Popen", lambda *args, **kwargs: process)
    measured = encode.decode_peak(Path("audio.m4a"), sr)
    assert measured is not None
    expected, _ = audio_qc.true_peak_dbtp(signal.astype(np.float32), sr)
    assert measured["true_peak_dbtp"] == pytest.approx(expected, abs=0.05)
    assert measured["lufs_integrated"] is None
    assert process.stdout.read_sizes and all(
        size <= encode._DECODE_PEAK_CHUNK * 4 for size in process.stdout.read_sizes
    )


@pytest.mark.parametrize("size", [1, 2, 3, 5, 7])
def test_decode_peak_reassembles_partial_chunks(monkeypatch, size: int):
    sr = 8000
    signal = (0.5 * np.sin(2 * np.pi * 440 * np.arange(sr // 2) / sr)).astype("<f4")
    payload = signal.tobytes()
    chunks = [payload[i : i + size] for i in range(0, len(payload), size)] + [b""]
    process = _PeakProcess(chunks)
    monkeypatch.setattr(encode.shutil, "which", lambda _: "/usr/bin/ffmpeg")
    monkeypatch.setattr(encode.subprocess, "Popen", lambda *args, **kwargs: process)
    measured = encode.decode_peak(Path("audio.m4a"), sr)
    assert measured is not None
    expected, _ = audio_qc.true_peak_dbtp(signal.astype(np.float32), sr)
    assert measured["true_peak_dbtp"] == pytest.approx(expected, abs=0.05)


def test_decode_peak_failure_and_empty_are_none(monkeypatch):
    monkeypatch.setattr(encode.shutil, "which", lambda _: "/usr/bin/ffmpeg")
    failing = _PeakProcess([], returncode=1)
    monkeypatch.setattr(encode.subprocess, "Popen", lambda *args, **kwargs: failing)
    assert encode.decode_peak(Path("audio.m4a"), 8000) is None

    empty = _PeakProcess([b""])
    monkeypatch.setattr(encode.subprocess, "Popen", lambda *args, **kwargs: empty)
    assert encode.decode_peak(Path("audio.m4a"), 8000) is None


def test_chapter_windows_exact_formula():
    windows = encode.chapter_windows([0, 1234, 1234, 5000], 6000)
    assert windows[0] == (0, 1234)
    assert windows[1] == (1234, 1234 + 1)  # same-ms chapter still gets 1 ms
    assert windows[2] == (1235, 5000)
    assert windows[3] == (5000, 6000)
    for start, end in windows:
        assert start < end


def test_chapter_windows_sub_millisecond():
    windows = encode.chapter_windows([0, 0, 0], 1)
    assert windows == [(0, 1), (1, 2), (2, 3)]


def test_wav_and_flac_readback(tmp_path: Path, model_dir: Path):
    project = inline_project(tmp_path, chapters=[{"title": "A", "text": "one"}])
    payload = _build(project, model_dir)
    data, sr = read_audio(Path(payload["output"]["path"]))
    assert sr == 24000
    assert data.shape[0] == 12000
    (tmp_path / "build" / "book.wav").unlink()
    (tmp_path / "build" / "book.json").unlink()
    flac = _build(project, model_dir, None, "--out", str(tmp_path / "book.flac"))
    data, sr = read_audio(Path(flac["output"]["path"]))
    assert sr == 24000
    assert data.shape[0] == 12000


def test_no_loudness_leaves_samples_unchanged(tmp_path: Path, model_dir: Path):
    project = inline_project(
        tmp_path,
        chapters=[{"title": "A", "text": "one"}],
        extra="pause: {paragraph_ms: 0, chapter_ms: 0}",
    )
    payload = _build(project, model_dir, None, "--no-loudness", "--keep-stems")
    final, sr = read_audio(Path(payload["output"]["path"]))
    segment, seg_sr = read_audio(tmp_path / "build" / "segments" / "0001-0001.wav")
    assert sr == seg_sr
    assert np.array_equal(final, segment)


def test_per_chapter_loudness_target(tmp_path: Path, model_dir: Path):
    project = inline_project(
        tmp_path,
        chapters=[{"title": "A", "text": "one"}, {"title": "B", "text": "two"}],
        extra="pause: {paragraph_ms: 0, chapter_ms: 0}",
    )
    payload = _build(project, model_dir, None, "--lufs", "-18", "--true-peak", "-1.5")
    assert payload["loudness"]["verified_ok"] is True
    data, sr = read_audio(Path(payload["output"]["path"]))
    chapters = json.loads((tmp_path / "build" / "book.json").read_text())["chapters"]
    for chapter in chapters:
        start = int(chapter["start_ms"] * sr / 1000)
        stop = int(chapter["end_ms"] * sr / 1000)
        lufs, _ = audio_qc.integrated_loudness(data[start:stop], sr)
        assert lufs is not None
        assert abs(lufs - (-18.0)) <= 0.3
        peak, _ = audio_qc.true_peak_dbtp(data[start:stop], sr)
        assert peak is not None
        assert peak <= -1.5 + 0.1


def test_silent_chapter_warns_and_continues(tmp_path: Path, model_dir: Path):
    class Silence(FakeRunner):
        def synthesize(self, **kwargs):
            import soundfile as sf

            sf.write(str(kwargs["out"]), np.zeros(12000, dtype=np.float32), 24000, subtype="PCM_16")
            return {
                "sample_rate": 24000,
                "duration_s": 0.5,
                "dtype_actual": None,
                "device_actual": "cpu",
                "languages_actual": ["English"],
            }

    project = inline_project(tmp_path, chapters=[{"title": "A", "text": "one"}])
    payload = _build(project, model_dir, Silence(), "--lufs", "-18")
    assert payload["ok"] is True
    assert any("loudness unavailable" in warning for warning in payload["warnings"])
    manifest = json.loads((tmp_path / "build" / "book.json").read_text())
    assert manifest["chapters"][0]["loudness"] is None


@ffmpeg_only
@pytest.mark.parametrize("fmt", ["mp3", "m4a"])
def test_lossy_encode(tmp_path: Path, model_dir: Path, fmt: str):
    project = inline_project(tmp_path, chapters=[{"title": "A", "text": "one"}])
    payload = _build(project, model_dir, None, "--format", fmt, "--bitrate", "64")
    assert payload["ok"] is True
    path = Path(payload["output"]["path"])
    assert path.is_file() and path.stat().st_size > 0
    duration = float(
        subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", str(path)],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    )
    assert abs(duration - 0.5) <= 0.5 * 0.01 + 0.5
    manifest = json.loads((tmp_path / "build" / "book.json").read_text())
    assert manifest["output"]["encoder"] in {"aac", "libmp3lame"}
    assert manifest["loudness"]["enabled"] is True
    assert manifest["loudness"]["encoded_true_peak_dbtp"] is not None


@ffmpeg_only
def test_m4b_chapter_metadata_and_escaping(tmp_path: Path, model_dir: Path):
    project = inline_project(
        tmp_path,
        chapters=[
            {"title": "One = a; b # c \\ d", "text": "one"},
            {"title": "Two", "text": "two"},
        ],
        extra="pause: {paragraph_ms: 0, chapter_ms: 500}",
    )
    payload = _build(project, model_dir, None, "--m4b")
    path = Path(payload["output"]["path"])
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_chapters", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    chapters = json.loads(result.stdout)["chapters"]
    assert len(chapters) == 2
    assert chapters[0]["tags"]["title"] == "One = a; b # c \\ d"
    manifest = json.loads((tmp_path / "build" / "book.json").read_text())
    starts = [chapter["start_ms"] for chapter in manifest["chapters"]]
    total = manifest["output"]["duration_s"] * 1000
    windows = []
    previous_end = 0
    for index, start in enumerate(starts):
        start = 0 if index == 0 else max(start, previous_end)
        end = max(start + 1, int(round(total))) if index == len(starts) - 1 else max(start + 1, starts[index + 1])
        windows.append((start, end))
        previous_end = end
    for chapter, (start, end) in zip(chapters, windows):
        assert int(round(float(chapter["start_time"]) * 1000)) == start
        assert int(round(float(chapter["end_time"]) * 1000)) == end


@ffmpeg_only
def test_encoded_duration_sanity_fails(tmp_path: Path, model_dir: Path, monkeypatch):
    project = inline_project(tmp_path, chapters=[{"title": "A", "text": "one"}])
    monkeypatch.setattr(encode, "encoded_duration_s", lambda _: 99.0)
    with pytest.raises(ModelError):
        _build(project, model_dir, None, "--format", "m4a")
    assert not (tmp_path / "build" / "book.m4a").exists()
    assert not (tmp_path / "build" / "book.json").exists()


@ffmpeg_only
def test_unavailable_encoded_peak(tmp_path: Path, model_dir: Path, monkeypatch):
    project = inline_project(tmp_path, chapters=[{"title": "A", "text": "one"}])
    monkeypatch.setattr(encode, "decode_peak", lambda *args, **kwargs: None)
    payload = _build(project, model_dir, None, "--format", "m4b")
    assert payload["loudness"]["verified_ok"] is None
    assert any("encoded true peak is unavailable" in warning for warning in payload["warnings"])
