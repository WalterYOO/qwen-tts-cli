"""Frame placement, stems, bed, loudness and timeline.json (FR-P2-04 §4–§6)."""

from __future__ import annotations

import math
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .. import audio_qc, postprocess
from ..errors import InputError, ModelError, UsageError
from ..utils import sha256_text
from .synth import MAX_MASTER_BYTES

ANONYMOUS = "__anonymous__"
MAX_VOICE_SECONDS = 2 * 60 * 60


def half_up(value: float) -> int:
    return int(math.floor(float(value) + 0.5))


@dataclass
class CueGeometry:
    start_frame: int
    end_frame: int
    target_frames: int


def cue_geometry(start_s: float, end_s: float, sample_rate: int, index: int) -> CueGeometry:
    start = half_up(float(start_s) * int(sample_rate))
    end = half_up(float(end_s) * int(sample_rate))
    target = end - start
    if target <= 0:
        raise InputError(
            f"cue {index} has a non-positive frame duration",
            details={"index": int(index), "start_frame": start, "end_frame": end},
        )
    return CueGeometry(start_frame=start, end_frame=end, target_frames=target)


def estimate_frames(
    cues: list[Any], sample_rate: int, video_frames: Optional[int]
) -> int:
    last_end = half_up(float(cues[-1].end_s) * int(sample_rate)) if cues else 0
    return max(last_end, int(video_frames or 0))


def check_master_budget(frames: int, label: str = "voice master") -> None:
    if int(frames) < 0:
        raise InputError(f"{label} length is negative")
    if int(frames) * 4 > MAX_MASTER_BYTES:
        raise InputError(
            f"{label} would need {int(frames) * 4} bytes "
            f"(limit {MAX_MASTER_BYTES}); shorten the timeline or lower --sample-rate",
            details={"frames": int(frames), "bytes": int(frames) * 4},
        )


def allocate(frames: int) -> np.ndarray:
    return np.zeros(int(frames), dtype=np.float32)


def place(master: np.ndarray, data: np.ndarray, start_frame: int) -> None:
    start = max(0, int(start_frame))
    if start >= master.shape[0]:
        return
    stop = min(master.shape[0], start + int(data.shape[0]))
    if stop <= start:
        return
    master[start:stop] += data[: stop - start]


def clip_to(master: np.ndarray, frames: int) -> np.ndarray:
    frames = max(0, int(frames))
    if master.shape[0] == frames:
        return master
    if master.shape[0] > frames:
        return np.ascontiguousarray(master[:frames])
    out = np.zeros(frames, dtype=np.float32)
    out[: master.shape[0]] = master
    return out


# --------------------------------------------------------------------------- #
# Bed
# --------------------------------------------------------------------------- #


def decode_bed(video: Path, sample_rate: int, length_frames: int) -> np.ndarray:
    """Decode the first audio stream to mono float32 with a frame cap."""
    executable = shutil.which("ffmpeg")
    if executable is None:
        raise ModelError("ffmpeg is required for --bed-db but was not found on PATH")
    out = np.zeros(int(length_frames), dtype=np.float32)
    command = [
        executable,
        "-v",
        "error",
        "-i",
        str(video),
        "-map",
        "0:a:0",
        "-ac",
        "1",
        "-ar",
        str(int(sample_rate)),
        "-f",
        "f32le",
        "-",
    ]
    # ffmpeg writes diagnostics to stderr; using a pipe while draining stdout
    # can deadlock once the pipe buffer fills.  A temporary file is bounded
    # only by ffmpeg's own output, and we read just the tail for the error.
    error_file = tempfile.TemporaryFile()
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=error_file,
        )
    except OSError as exc:
        error_file.close()
        raise ModelError(f"cannot execute ffmpeg for --bed-db: {exc}") from exc
    written = 0
    pending = bytearray()
    capped = False
    try:
        stdout = process.stdout
        if stdout is None:  # pragma: no cover - defensive
            process.kill()
            raise InputError("cannot read the video audio stream")
        while written < out.shape[0]:
            raw = stdout.read(1 << 16)
            if not raw:
                break
            pending.extend(raw)
            usable = len(pending) - (len(pending) % 4)
            if not usable:
                continue
            chunk = np.frombuffer(bytes(pending[:usable]), dtype="<f4")
            del pending[:usable]
            take = min(chunk.shape[0], out.shape[0] - written)
            out[written : written + take] = chunk[:take]
            written += take
        capped = written >= out.shape[0]
    finally:
        if capped:
            try:
                process.kill()
            except OSError:
                pass
        try:
            process.stdout.close()  # type: ignore[union-attr]
        except (OSError, ValueError):
            pass
        stderr = b""
        try:
            error_file.seek(0, 2)
            size = error_file.tell()
            error_file.seek(max(0, size - 8192))
            stderr = error_file.read(8192)
        except (OSError, ValueError):
            pass
        finally:
            try:
                error_file.close()
            except OSError:
                pass
        returncode = process.wait()
    if returncode != 0 and not capped:
        message = stderr.decode("utf-8", "replace").strip()
        raise InputError(
            f"cannot decode the video audio stream: {message}", details={"video": str(video)}
        )
    return out


def apply_bed(master: np.ndarray, bed: np.ndarray, volume_db: float) -> np.ndarray:
    gain = np.float32(10.0 ** (float(volume_db) / 20.0))
    master += bed[: master.shape[0]] * gain
    return master


# --------------------------------------------------------------------------- #
# Loudness
# --------------------------------------------------------------------------- #


@dataclass
class LoudnessResult:
    mix: np.ndarray
    enabled: bool
    verified_ok: Optional[bool]
    warnings: list[str]
    actual_true_peak_dbtp: Optional[float] = None
    limiter_ran: bool = False


def normalize(
    mix: np.ndarray,
    sample_rate: int,
    *,
    enabled: bool,
    target_lufs: float,
    true_peak: float,
) -> LoudnessResult:
    warnings: list[str] = []
    if not enabled:
        peak, reason = audio_qc.true_peak_dbtp(mix, int(sample_rate))
        if peak is None:
            verified: Optional[bool] = None
            warnings.append(f"true peak is unavailable ({reason})")
        else:
            verified = peak <= float(true_peak)
            if not verified:
                warnings.append(
                    f"true peak {peak:.2f} dBTP exceeds {float(true_peak):.2f} dBTP "
                    "(loudness normalization is disabled)"
                )
        return LoudnessResult(
            mix, False, verified, warnings, actual_true_peak_dbtp=peak, limiter_ran=False
        )

    lufs, reason = audio_qc.integrated_loudness(mix, int(sample_rate))
    if lufs is None:
        peak, peak_reason = audio_qc.true_peak_dbtp(mix, int(sample_rate))
        if peak is None:
            verified = None
            warnings.append(f"true peak is unavailable ({peak_reason})")
        else:
            verified = peak <= float(true_peak)
            if not verified:
                warnings.append(
                    f"true peak {peak:.2f} dBTP exceeds {float(true_peak):.2f} dBTP"
                )
        warnings.append(f"loudness unavailable ({reason}); left unchanged")
        return LoudnessResult(
            mix, True, verified, warnings, actual_true_peak_dbtp=peak, limiter_ran=False
        )

    try:
        processed, out_rate, _applied, _before, _after, dsp_warnings = postprocess.process(
            mix,
            int(sample_rate),
            {"lufs": float(target_lufs), "true_peak": float(true_peak)},
        )
    except UsageError:
        raise
    except (InputError, ImportError, RuntimeError, OSError, ValueError) as exc:
        message = exc.message if isinstance(exc, InputError) else str(exc)
        raise ModelError(f"loudness processing failed: {message}") from exc
    warnings.extend(dsp_warnings)
    processed = np.asarray(processed, dtype=np.float32)
    processed = processed[:, 0] if processed.ndim == 2 else processed.reshape(-1)
    peak = postprocess.measure(processed, int(out_rate)).get("true_peak_dbtp")
    if peak is None:
        verified = None
        warnings.append("true peak is unavailable after normalization")
    elif peak > float(true_peak) + postprocess.TRUE_PEAK_TOLERANCE_DB:
        raise ModelError(
            f"true peak {peak:.2f} dBTP exceeds the ceiling "
            f"{float(true_peak) + postprocess.TRUE_PEAK_TOLERANCE_DB:.2f} dBTP"
        )
    else:
        verified = True
    return LoudnessResult(
        np.ascontiguousarray(processed, dtype=np.float32),
        True,
        verified,
        warnings,
        actual_true_peak_dbtp=peak,
        limiter_ran=True,
    )


def is_clipped(mix: np.ndarray) -> bool:
    return bool(mix.size and (float(np.max(mix)) > 1.0 or float(np.min(mix)) < -1.0))


# --------------------------------------------------------------------------- #
# timeline.json
# --------------------------------------------------------------------------- #


def timeline_document(
    *,
    timeline_path: Path,
    timeline_format: str,
    timeline_sha256: str,
    sample_rate: int,
    cues: list[Any],
    metadata: list[dict[str, Any]],
    files: dict[str, str],
    hashes: dict[str, str],
    warnings: list[str],
) -> dict[str, Any]:
    records = []
    for cue, meta in zip(cues, metadata):
        records.append(
            {
                "index": int(cue.index),
                "start_s": round(float(cue.start_s), 6),
                "end_s": round(float(cue.end_s), 6),
                "text_sha256": sha256_text(cue.text),
                "speaker": cue.speaker,
                "voice": meta["voice"],
                "instruct": meta["instruct"],
                "synth_duration_s": round(float(meta["synth_duration_s"]), 3),
                "speed": round(float(meta["speed"]), 6),
                "rendered_duration_s": round(
                    float(meta["rendered_frames"]) / float(sample_rate), 3
                ),
                "overflow": bool(meta["overflow"]),
            }
        )
    return {
        "schema_version": 1,
        "timeline": {
            "path": str(timeline_path),
            "format": timeline_format,
            "sha256": timeline_sha256,
        },
        "sample_rate": int(sample_rate),
        "cues": records,
        "files": dict(files),
        "counts": {
            "cues": len(cues),
            "text_chars": sum(len(cue.text) for cue in cues),
        },
        "warnings": list(warnings),
        "hashes": dict(hashes),
    }


def stretch_stats(metadata: list[dict[str, Any]], overflow_count: int) -> dict[str, Any]:
    speeds = [float(meta["speed"]) for meta in metadata]
    if not speeds:
        return {"min": None, "max": None, "mean": None, "overflow_count": int(overflow_count)}
    return {
        "min": round(min(speeds), 6),
        "max": round(max(speeds), 6),
        "mean": round(sum(speeds) / len(speeds), 6),
        "overflow_count": int(overflow_count),
    }
