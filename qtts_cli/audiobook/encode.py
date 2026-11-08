"""Lossless streaming and ffmpeg encoding for audiobooks (FR-P2-03 §7)."""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .. import audio_qc
from ..errors import InputError, ModelError

LOSSY_FORMATS = {"m4a", "m4b", "mp3"}
LOSSLESS_FORMATS = {"wav", "flac"}
_FFMPEG_ENCODER = {"m4a": "aac", "m4b": "aac", "mp3": "libmp3lame"}
_FFMPEG_CONTAINER = {"m4a": "ipod", "m4b": "ipod", "mp3": "mp3"}
_DURATION_ABS_TOLERANCE_S = 0.5
_DURATION_REL_TOLERANCE = 0.01
_DECODE_PEAK_CHUNK = 1 << 20  # samples per read
_TRUE_PEAK_BLOCK = 1 << 20  # samples per measurement block
_TRUE_PEAK_OVERLAP = 256  # filter context carried between chunks


def format_of(encoder: str) -> str:
    return encoder


def require_ffmpeg(fmt: str) -> str:
    """Same semantics as ``commands.post._preflight_ffmpeg`` for one format."""
    if fmt not in _FFMPEG_ENCODER:
        raise InputError(f"{fmt} is not an ffmpeg-encoded format")
    executable = shutil.which("ffmpeg")
    if executable is None:
        raise ModelError(f"ffmpeg is required for {fmt} output but was not found on PATH")
    try:
        completed = subprocess.run(
            [executable, "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise ModelError(f"cannot execute ffmpeg: {exc}") from exc
    if completed.returncode != 0:
        raise ModelError(
            "ffmpeg -encoders failed",
            details={"returncode": completed.returncode},
        )
    reported = f"{completed.stdout}\n{completed.stderr}".lower()
    encoder = _FFMPEG_ENCODER[fmt]
    if encoder not in reported:
        raise ModelError(f"ffmpeg does not provide the {encoder} encoder required for {fmt}")
    return executable


def write_lossless(path: Path, master: np.ndarray, sr: int, fmt: str) -> None:
    import soundfile as sf

    if fmt not in LOSSLESS_FORMATS:
        raise InputError(f"{fmt} is not a lossless format")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    data = np.asarray(master, dtype=np.float32)
    with sf.SoundFile(
        str(output),
        mode="w",
        samplerate=int(sr),
        channels=1,
        format="WAV" if fmt == "wav" else "FLAC",
        subtype="PCM_16",
    ) as handle:
        step = 1 << 20
        for start in range(0, data.shape[0], step):
            handle.write(data[start : start + step])


def _escape_metadata(value: str) -> str:
    text = str(value)
    text = text.replace("\\", "\\\\")
    text = text.replace("\n", "\\\n")
    text = text.replace("=", "\\=")
    text = text.replace(";", "\\;")
    text = text.replace("#", "\\#")
    return text


def chapter_windows(start_ms: list[int], total_ms: int) -> list[tuple[int, int]]:
    """Exact monotonic M4B START/END formula from §7."""
    windows: list[tuple[int, int]] = []
    previous_end = 0
    for index, raw_start in enumerate(start_ms):
        start = 0 if index == 0 else max(int(raw_start), previous_end)
        if index == len(start_ms) - 1:
            end = max(start + 1, int(total_ms))
        else:
            end = max(start + 1, int(start_ms[index + 1]))
        windows.append((start, end))
        previous_end = end
    return windows


def write_m4b_metadata(
    path: Path,
    *,
    title: str,
    author: Optional[str],
    album: str,
    chapters: list[dict[str, Any]],
    windows: list[tuple[int, int]],
) -> None:
    lines = [";FFMETADATA1", f"title={_escape_metadata(title)}"]
    if author:
        lines.append(f"artist={_escape_metadata(author)}")
    lines.append(f"album={_escape_metadata(album)}")
    for chapter, (start, end) in zip(chapters, windows):
        lines.extend(
            [
                "[CHAPTER]",
                "TIMEBASE=1/1000",
                f"START={int(start)}",
                f"END={int(end)}",
                f"title={_escape_metadata(chapter['title'])}",
            ]
        )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def encode_lossy(
    *,
    master_wav: Path,
    output: Path,
    fmt: str,
    bitrate_kbps: int,
    metadata_path: Optional[Path] = None,
) -> None:
    executable = require_ffmpeg(fmt)
    command = [executable, "-v", "error", "-y", "-i", str(master_wav)]
    if metadata_path is not None:
        command.extend(["-i", str(metadata_path)])
    if metadata_path is not None:
        command.extend(["-map", "0:a:0", "-map_metadata", "1", "-map_chapters", "1"])
    command.extend(["-c:a", _FFMPEG_ENCODER[fmt], "-b:a", f"{int(bitrate_kbps)}k"])
    if fmt in {"m4a", "m4b"}:
        command.extend(["-movflags", "+faststart"])
    command.extend(["-f", _FFMPEG_CONTAINER[fmt], str(output)])
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise ModelError(
            f"ffmpeg failed to encode {fmt}: {completed.stderr.strip()}",
            details={"format": fmt, "returncode": completed.returncode},
        )
    if not Path(output).is_file() or Path(output).stat().st_size == 0:
        raise ModelError(f"ffmpeg produced no {fmt} output")


def _probe(path: Path, entries: str) -> str:
    probe = shutil.which("ffprobe")
    if probe is None:
        raise ModelError("ffprobe is required to verify encoded output but was not found on PATH")
    completed = subprocess.run(
        [probe, "-v", "error", "-show_entries", entries, "-of", "json", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise ModelError(
            f"ffprobe failed for {path}",
            details={"stderr": completed.stderr.strip()},
        )
    return completed.stdout


def encoded_duration_s(path: Path) -> float:
    payload = _probe(path, "format=duration")
    try:
        return float(json.loads(payload or "{}")["format"]["duration"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ModelError(f"cannot read encoded duration for {path}") from exc


def chapter_metadata(path: Path) -> list[dict[str, Any]]:
    payload = _probe(path, "chapter=start_time,end_time:chapter_tags=title")
    try:
        chapters = json.loads(payload or "{}").get("chapters") or []
    except ValueError as exc:
        raise ModelError(f"cannot read chapter metadata for {path}") from exc
    result = []
    for chapter in chapters:
        result.append(
            {
                "start_ms": int(round(float(chapter.get("start_time", 0.0)) * 1000)),
                "end_ms": int(round(float(chapter.get("end_time", 0.0)) * 1000)),
                "title": (chapter.get("tags") or {}).get("title"),
            }
        )
    return result


def verify_duration(path: Path, expected_s: float) -> float:
    actual = encoded_duration_s(path)
    tolerance = max(_DURATION_ABS_TOLERANCE_S, expected_s * _DURATION_REL_TOLERANCE)
    if abs(actual - expected_s) > tolerance:
        raise ModelError(
            f"encoded duration {actual:.3f}s differs from expected {expected_s:.3f}s "
            f"(tolerance {tolerance:.3f}s)",
            details={"path": str(path), "actual_s": actual, "expected_s": expected_s},
        )
    return actual


def _kill_process(process) -> None:
    try:
        process.kill()
    except OSError:
        pass


def _close_process_streams(process) -> None:
    for stream in (getattr(process, "stdout", None), getattr(process, "stderr", None)):
        close = getattr(stream, "close", None)
        if close is None:
            continue
        try:
            close()
        except (OSError, ValueError):
            pass


def _measure_peak_block(
    block: np.ndarray,
    tail: np.ndarray,
    sr: int,
    max_peak: Optional[float],
) -> tuple[Optional[float], np.ndarray]:
    window = np.concatenate((tail, block)) if tail.size else block
    peak, _ = audio_qc.true_peak_dbtp(window, int(sr))
    if peak is not None:
        max_peak = peak if max_peak is None else max(max_peak, peak)
    return max_peak, window[-_TRUE_PEAK_OVERLAP:].copy()


def decode_peak(path: Path, sr: int) -> Optional[dict[str, Optional[float]]]:
    """Measure the decoded true peak with bounded memory.

    ffmpeg streams ``f32le`` on stdout; bytes are reassembled across read
    boundaries, measurement blocks are oversampled together with a short
    overlap from the previous block, and the whole file is never buffered.
    Only the true-peak field is meaningful for the audit path.
    """
    executable = shutil.which("ffmpeg")
    if executable is None:
        return None
    try:
        process = subprocess.Popen(
            [
                executable,
                "-v",
                "error",
                "-i",
                str(path),
                "-map",
                "0:a:0",
                "-ac",
                "1",
                "-ar",
                str(int(sr)),
                "-f",
                "f32le",
                "-",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return None
    max_peak: Optional[float] = None
    tail = np.zeros(0, dtype=np.float32)
    pending = bytearray()
    block_bytes = _TRUE_PEAK_BLOCK * 4
    try:
        stdout = process.stdout
        if stdout is None:
            _kill_process(process)
            return None
        while True:
            raw = stdout.read(_DECODE_PEAK_CHUNK * 4)
            if not raw:
                break
            pending.extend(raw)
            while len(pending) >= block_bytes:
                block = np.frombuffer(bytes(pending[:block_bytes]), dtype="<f4")
                del pending[:block_bytes]
                max_peak, tail = _measure_peak_block(block, tail, sr, max_peak)
        usable = len(pending) - (len(pending) % 4)
        if usable:
            block = np.frombuffer(bytes(pending[:usable]), dtype="<f4")
            max_peak, tail = _measure_peak_block(block, tail, sr, max_peak)
    except (OSError, ValueError):
        _kill_process(process)
        return None
    finally:
        returncode = process.wait()
        _close_process_streams(process)
    if returncode != 0 or max_peak is None:
        return None
    return {
        "lufs_integrated": None,
        "true_peak_dbtp": max_peak,
        "peak_dbfs": None,
    }


def uuid_path(target: Path, suffix: str) -> Path:
    import uuid

    return Path(target).parent / f"{Path(target).name}.{uuid.uuid4().hex}{suffix}"


def remove_quietly(path: Path) -> bool:
    try:
        os.unlink(path)
        return True
    except FileNotFoundError:
        return True
    except OSError:
        return False


def duration_seconds_of_frames(frames: int, sr: int) -> float:
    return round(int(frames) / float(sr), 3)


def ceil_frames(duration: float, sr: int) -> int:
    return int(math.ceil(float(duration) * int(sr)))
