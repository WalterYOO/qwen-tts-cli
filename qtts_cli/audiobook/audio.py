"""Audio assembly, BGM ducking and mixing helpers (FR-P2-03 §4–§5)."""

from __future__ import annotations

import math
import shutil
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .. import audio_io, postprocess
from ..errors import InputError, ModelError
from ..utils import sha256_file

BGM_MAX_SECONDS = 7200.0
BGM_MAX_BUFFER_BYTES = 256 * 1024 * 1024
BGM_READ_CHUNK = 4 * 1024 * 1024
BGM_STDERR_TAIL = 8 * 1024
DUCK_FRAME_SECONDS = 0.020
DUCK_THRESHOLD_DB = -35.0
DUCK_ATTACK_TAU = 0.05
DUCK_RELEASE_TAU = 0.30


def frames_for_ms(ms: int, sr: int) -> int:
    """Half-up ``round(ms * sr / 1000)`` for non-negative milliseconds."""
    return (int(ms) * int(sr) + 500) // 1000


def ms_for_frame(frame: int, sr: int) -> int:
    """Half-up ``round(1000 * frame / sr)``."""
    return (2000 * int(frame) + int(sr)) // (2 * int(sr))


def duration_s(frames: int, sr: int) -> float:
    return round(int(frames) / float(sr), 3)


def resample_segment(samples: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    """Resample a mono buffer, enforcing the contract's 0.5–2.0 ratio band."""
    data = np.asarray(samples, dtype=np.float32)
    if data.ndim != 1:
        data = data.mean(axis=1) if data.ndim == 2 else data.reshape(-1)
    if int(sr) <= 0:
        raise InputError("segment sample rate must be positive")
    ratio = float(target_sr) / float(sr)
    if ratio < 0.5 or ratio > 2.0:
        raise InputError(
            f"segment sample rate {sr} Hz cannot be resampled to {target_sr} Hz "
            f"(ratio {ratio:.3f} outside 0.5-2.0)",
            details={"from": int(sr), "to": int(target_sr)},
        )
    if int(sr) == int(target_sr):
        return np.ascontiguousarray(data, dtype=np.float32)
    out, _ = postprocess.resample(data, int(sr), int(target_sr))
    if out.ndim == 2:
        out = out[:, 0]
    return np.ascontiguousarray(out, dtype=np.float32)


@dataclass
class Assembled:
    full: np.ndarray
    start_frames: list[int]
    end_frames: list[int]
    start_ms: list[int]
    end_ms: list[int]


def resampled_length(frames: int, in_rate: int, out_rate: int) -> int:
    """Frame count produced by ``postprocess.resample`` (ceil ratio)."""
    if int(in_rate) <= 0 or int(out_rate) <= 0:
        raise InputError("sample rates must be positive")
    if int(in_rate) == int(out_rate):
        return int(frames)
    ratio = float(out_rate) / float(in_rate)
    if ratio < 0.5 or ratio > 2.0:
        raise InputError(
            f"segment sample rate {in_rate} Hz cannot be resampled to {out_rate} Hz "
            f"(ratio {ratio:.3f} outside 0.5-2.0)",
            details={"from": int(in_rate), "to": int(out_rate)},
        )
    return (int(frames) * int(out_rate) + int(in_rate) - 1) // int(in_rate)


def assemble(
    chapter_segments: list[list[np.ndarray]],
    segment_rates: list[list[int]],
    sr: int,
    paragraph_ms: int,
    chapter_ms: int,
) -> Assembled:
    """Assemble the full speech master with bounded intermediate memory.

    The exact length of every chapter is computed first so the master is
    allocated once; chapters are then written into it and their intermediate
    arrays released.  No chapter-buffer list is retained.
    """
    paragraph_pad = frames_for_ms(paragraph_ms, sr)
    chapter_pad = frames_for_ms(chapter_ms, sr)

    chapter_lengths: list[int] = []
    for segments, rates in zip(chapter_segments, segment_rates):
        length = 0
        for index, (segment, rate) in enumerate(zip(segments, rates)):
            if index:
                length += paragraph_pad
            length += resampled_length(int(segment.shape[0]), int(rate), sr)
        chapter_lengths.append(length)

    start_frames: list[int] = []
    end_frames: list[int] = []
    start_ms: list[int] = []
    end_ms: list[int] = []
    cursor = 0
    for index, length in enumerate(chapter_lengths):
        start_frames.append(cursor)
        end_frames.append(cursor + length)
        start_ms.append(ms_for_frame(cursor, sr))
        end_ms.append(ms_for_frame(cursor + length, sr))
        cursor += length
        if index != len(chapter_lengths) - 1:
            cursor += chapter_pad

    full = np.zeros(max(0, cursor), dtype=np.float32)
    for chapter_index, (segments, rates) in enumerate(zip(chapter_segments, segment_rates)):
        position = start_frames[chapter_index]
        for index, (segment, rate) in enumerate(zip(segments, rates)):
            if index:
                position += paragraph_pad
            expected = resampled_length(int(segment.shape[0]), int(rate), sr)
            block = resample_segment(segment, int(rate), sr)
            if int(block.shape[0]) != expected:
                raise InputError(
                    f"resampler produced {block.shape[0]} frames, expected {expected}",
                    details={"from": int(rate), "to": int(sr)},
                )
            full[position : position + expected] = block
            position += expected
    return Assembled(full, start_frames, end_frames, start_ms, end_ms)


# --------------------------------------------------------------------------- #
# BGM
# --------------------------------------------------------------------------- #


@dataclass
class BgmData:
    samples: np.ndarray
    sample_rate: int
    source_sample_rate: int
    source_channels: int
    bytes: int
    sha256: str
    duration_s: float


def _buffer_frames_ok(frames: int, bytes_per_sample: int = 4) -> bool:
    return int(frames) * bytes_per_sample <= BGM_MAX_BUFFER_BYTES


def _soundfile_info(path: Path) -> Optional[dict]:
    try:
        return audio_io.audio_info(path)
    except InputError:
        return None


def _ffprobe(path: Path) -> dict:
    probe = shutil.which("ffprobe")
    if probe is None:
        raise ModelError(
            f"ffprobe is required to read BGM {path} but was not found on PATH",
            details={"bgm": str(path)},
        )
    completed = subprocess.run(
        [
            probe,
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "format=duration:stream=sample_rate,channels",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise InputError(
            f"cannot read BGM {path}: ffprobe failed",
            details={"bgm": str(path), "stderr": completed.stderr.strip()},
        )
    import json

    try:
        payload = json.loads(completed.stdout or "{}")
        duration = float(payload["format"]["duration"])
        stream = payload["streams"][0]
        sample_rate = int(stream["sample_rate"])
        channels = int(stream["channels"])
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise InputError(
            f"cannot read BGM {path}: ffprobe returned no usable audio metadata",
            details={"bgm": str(path)},
        ) from exc
    if not math.isfinite(duration) or duration <= 0 or sample_rate <= 0 or channels <= 0:
        raise InputError(f"cannot read BGM {path}: unusable audio metadata", details={"bgm": str(path)})
    return {
        "duration_s": duration,
        "sample_rate": sample_rate,
        "channels": channels,
        "frames": int(math.ceil(duration * sample_rate)),
    }


def _check_metadata(meta: dict, master_sr: int, path: Path, *, source_buffer: bool) -> None:
    duration = float(meta["duration_s"])
    if duration > BGM_MAX_SECONDS:
        raise InputError(
            f"BGM {path} is {duration:.1f}s; the limit is {BGM_MAX_SECONDS:.0f}s",
            details={"bgm": str(path), "duration_s": round(duration, 3)},
        )
    master_frames = int(math.ceil(duration * int(master_sr)))
    if not _buffer_frames_ok(master_frames):
        raise InputError(
            f"BGM {path} would need {master_frames * 4} bytes at {master_sr} Hz "
            f"(limit {BGM_MAX_BUFFER_BYTES})",
            details={"bgm": str(path), "frames": master_frames, "sample_rate": int(master_sr)},
        )
    if source_buffer:
        source_bytes = int(meta.get("frames", 0)) * int(meta.get("channels", 1)) * 4
        if source_bytes > BGM_MAX_BUFFER_BYTES:
            raise InputError(
                f"BGM {path} source buffer is {source_bytes} bytes "
                f"(limit {BGM_MAX_BUFFER_BYTES})",
                details={"bgm": str(path), "bytes": source_bytes},
            )


def probe_bgm(path: Path, master_sr: int) -> dict:
    """Read-only metadata validation used by dry-run (never decodes)."""
    info = _soundfile_info(path)
    if info is not None:
        meta = {
            "duration_s": info["duration_s"],
            "sample_rate": info["sample_rate"],
            "channels": info["channels"],
            "frames": info["frames"],
        }
        _check_metadata(meta, master_sr, path, source_buffer=True)
        return meta
    meta = _ffprobe(path)
    _check_metadata(meta, master_sr, path, source_buffer=False)
    return meta


def _kill_quietly(process) -> None:
    try:
        process.kill()
    except OSError:
        pass


def _drain_stderr(stream, holder: list[bytes]) -> None:
    """Keep only a bounded tail of stderr so a chatty decoder cannot fill RAM."""
    try:
        while True:
            block = stream.read(4096)
            if not block:
                break
            holder.append(block)
            total = sum(len(item) for item in holder)
            while holder and total > BGM_STDERR_TAIL:
                total -= len(holder.pop(0))
    except (OSError, ValueError):
        pass


def _close_quietly(stream) -> None:
    close = getattr(stream, "close", None)
    if close is None:
        return
    try:
        close()
    except (OSError, ValueError):
        pass


def _decode_ffmpeg(path: Path, master_sr: int, meta: dict) -> np.ndarray:
    executable = shutil.which("ffmpeg")
    if executable is None:
        raise ModelError(
            f"ffmpeg is required to decode BGM {path} but was not found on PATH",
            details={"bgm": str(path)},
        )
    duration = float(meta["duration_s"])
    cap_frames = int(math.ceil(min(BGM_MAX_SECONDS, duration + 1.0) * master_sr))
    cap_bytes = cap_frames * 4
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
            str(int(master_sr)),
            "-f",
            "f32le",
            "-",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stderr = getattr(process, "stderr", None)
    tail: list[bytes] = []
    reader = None
    if stderr is not None:
        reader = threading.Thread(target=_drain_stderr, args=(stderr, tail), daemon=True)
        reader.start()

    buffer = bytearray()
    total = 0
    overrun = False
    try:
        stdout = process.stdout
        if stdout is None:
            raise InputError(
                f"cannot decode BGM {path}: ffmpeg stdout is unavailable",
                details={"bgm": str(path)},
            )
        while True:
            chunk = stdout.read(BGM_READ_CHUNK)
            if not chunk:
                break
            if total + len(chunk) > cap_bytes:
                overrun = True
                break
            buffer.extend(chunk)
            total += len(chunk)
    except BaseException:
        _kill_quietly(process)
        raise
    finally:
        if overrun:
            _kill_quietly(process)
        returncode = process.wait()
        if reader is not None:
            reader.join(timeout=5.0)
        _close_quietly(getattr(process, "stdout", None))
        _close_quietly(stderr)

    stderr_text = b"".join(tail).decode("utf-8", "replace").strip()
    if overrun:
        raise InputError(
            f"BGM {path} decoded more than the allowed {cap_frames} frames",
            details={"bgm": str(path), "frames": cap_frames},
        )
    if returncode != 0:
        raise InputError(
            f"cannot decode BGM {path}: ffmpeg failed",
            details={"bgm": str(path), "stderr": stderr_text},
        )
    raw = memoryview(buffer)
    raw = raw[: len(raw) - (len(raw) % 4)]
    samples = np.frombuffer(raw, dtype="<f4")
    if samples.size == 0:
        raise InputError(f"BGM {path} decoded to empty audio", details={"bgm": str(path)})
    if samples.size > cap_frames:
        raise InputError(
            f"BGM {path} decoded more than the allowed {cap_frames} frames",
            details={"bgm": str(path), "frames": int(samples.size)},
        )
    return np.ascontiguousarray(samples.astype(np.float32))


def load_bgm(path: Path, master_sr: int, *, dry_run: bool = False) -> Optional[BgmData]:
    """Read and validate BGM, with the contract's single ffmpeg fallback."""
    info = _soundfile_info(path)
    if dry_run:
        probe_bgm(path, master_sr)
        return None
    if info is not None:
        meta = {
            "duration_s": info["duration_s"],
            "sample_rate": info["sample_rate"],
            "channels": info["channels"],
            "frames": info["frames"],
        }
        _check_metadata(meta, master_sr, path, source_buffer=True)
        samples, sr = audio_io.read_audio(path, mono=True)
        source_channels = int(info["channels"])
    else:
        meta = _ffprobe(path)
        _check_metadata(meta, master_sr, path, source_buffer=False)
        samples = _decode_ffmpeg(path, master_sr, meta)
        sr = int(master_sr)
        source_channels = int(meta["channels"])

    if int(sr) != int(master_sr):
        resampled, _ = postprocess.resample(samples, int(sr), int(master_sr))
        samples = resampled[:, 0] if resampled.ndim == 2 else resampled
    samples = np.ascontiguousarray(samples, dtype=np.float32)
    if samples.size == 0:
        raise InputError(f"BGM {path} is empty", details={"bgm": str(path)})
    if not _buffer_frames_ok(samples.size):
        raise InputError(
            f"BGM {path} buffer exceeds {BGM_MAX_BUFFER_BYTES} bytes",
            details={"bgm": str(path), "frames": int(samples.size)},
        )
    return BgmData(
        samples=samples,
        sample_rate=int(master_sr),
        source_sample_rate=int(meta["sample_rate"]),
        source_channels=source_channels,
        bytes=int(path.stat().st_size),
        sha256=sha256_file(path),
        duration_s=float(meta["duration_s"]),
    )


# --------------------------------------------------------------------------- #
# Ducking / mixing
# --------------------------------------------------------------------------- #


def _frame_levels(speech: np.ndarray, frame: int, frames: int) -> np.ndarray:
    """Per-frame RMS in dB, accumulated in bounded float64 chunks."""
    total = int(speech.shape[0])
    sums = np.zeros(int(frames), dtype=np.float64)
    if total == 0 or frames == 0:
        return np.zeros(0, dtype=np.float64)
    step = 1 << 20
    for start in range(0, total, step):
        stop = min(start + step, total)
        chunk = np.asarray(speech[start:stop], dtype=np.float64)
        square = np.square(chunk)
        first = start // frame
        last = min(int(frames) - 1, (stop - 1) // frame)
        if last < first:  # pragma: no cover - defensive
            continue
        boundaries = np.maximum(start, np.arange(first, last + 1, dtype=np.int64) * frame)
        sums[first : last + 1] += np.add.reduceat(square, boundaries - start)
    lengths = np.minimum(
        (np.arange(int(frames), dtype=np.int64) + 1) * frame, total
    ) - np.arange(int(frames), dtype=np.int64) * frame
    rms = np.sqrt(sums / lengths.astype(np.float64))
    return 20.0 * np.log10(rms + 1e-12)


def _smooth_targets(targets: np.ndarray, frame: int, sr: int) -> np.ndarray:
    count = targets.shape[0]
    out = np.empty(count, dtype=np.float64)
    out[0] = targets[0]
    if count == 1:
        return out
    attack = targets[1:] < targets[:-1]
    a_attack = math.exp(-frame / (DUCK_ATTACK_TAU * sr))
    a_release = math.exp(-frame / (DUCK_RELEASE_TAU * sr))
    changes = np.flatnonzero(attack[1:] != attack[:-1]) + 1
    starts = np.concatenate(([0], changes))
    ends = np.concatenate((changes, [attack.shape[0]]))
    from scipy.signal import lfilter

    previous = targets[0]
    for start, stop in zip(starts.tolist(), ends.tolist()):
        a = a_attack if bool(attack[start]) else a_release
        chunk = targets[1 + start : 1 + stop]
        filtered, _ = lfilter([1.0 - a], [1.0, -a], chunk, zi=[a * previous])
        out[1 + start : 1 + stop] = filtered
        previous = out[stop]
    return out


def _duck_curve(speech: np.ndarray, sr: int, ducking_db: float) -> tuple[np.ndarray, np.ndarray]:
    total = int(speech.shape[0])
    frame = max(1, int(round(DUCK_FRAME_SECONDS * sr)))
    count = int(math.ceil(total / frame)) if total else 0
    if count == 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)
    levels = _frame_levels(np.asarray(speech, dtype=np.float32), frame, count)
    targets = np.where(levels >= DUCK_THRESHOLD_DB, float(ducking_db), 0.0)
    smoothed = _smooth_targets(targets, frame, sr)
    centres = np.arange(count, dtype=np.float64) * frame + frame / 2.0
    return centres, smoothed


def duck_envelope_db(speech: np.ndarray, sr: int, ducking_db: float) -> np.ndarray:
    """Frame-centred ducking envelope in dB, one value per sample."""
    total = int(speech.shape[0])
    centres, smoothed = _duck_curve(speech, sr, ducking_db)
    if centres.size == 0:
        return np.zeros(0, dtype=np.float64)
    return np.interp(np.arange(total, dtype=np.float64), centres, smoothed)


def fade_gain(total: int, sr: int, fade_ms: int) -> np.ndarray:
    if total < 3 or fade_ms == 0:
        return np.ones(max(total, 0), dtype=np.float32)
    fade = min(int(round(fade_ms * sr / 1000.0)), max(1, (total - 1) // 2))
    if fade <= 0:
        return np.ones(total, dtype=np.float32)
    gain = np.ones(total, dtype=np.float32)
    ramp = np.arange(fade, dtype=np.float32) / float(fade)
    gain[:fade] = ramp
    gain[total - fade :] = ramp[::-1]
    return gain


def mix_bgm(
    speech: np.ndarray,
    bgm: np.ndarray,
    sr: int,
    *,
    volume_db: float,
    ducking_db: float,
    fade_ms: int,
) -> np.ndarray:
    """Exact single-pass mix from §5 (float32, no limiting).

    The speech master is mixed in place when it is a writable float32 array
    (the builder passes an exclusive master), so a long book never needs a
    second full-length copy.  A read-only or non-float32 input is copied.
    """
    total = int(speech.shape[0])
    length = int(bgm.shape[0])
    if length == 0:
        raise InputError("BGM is empty")
    gain = np.float32(10.0 ** (float(volume_db) / 20.0))
    centres, smoothed = _duck_curve(speech, sr, ducking_db)
    fade = fade_gain(total, sr, fade_ms)
    out = np.asarray(speech, dtype=np.float32)
    if not out.flags.writeable:
        out = np.array(out, dtype=np.float32, copy=True)
    step = 1 << 20
    log10_over_20 = math.log(10.0) / 20.0
    for start in range(0, total, step):
        stop = min(start + step, total)
        n = np.arange(start, stop, dtype=np.int64)
        # Interpolate the envelope per chunk so a long book never needs a
        # full-length float64 copy.
        duck_db = np.interp(n.astype(np.float64), centres, smoothed)
        duck = np.exp(duck_db * log10_over_20).astype(np.float32)
        out[start:stop] += (bgm[n % length] * gain) * (fade[start:stop] * duck)
    return out
