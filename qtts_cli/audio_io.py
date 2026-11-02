"""Audio reading/writing with atomic replacement (PRD §9)."""

from __future__ import annotations

import io
import math
import sys
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from .errors import InputError, UsageError
from .utils import atomic_write_bytes

SUPPORTED_FORMATS = ("wav", "flac")
_SUBTYPES = {"wav": "PCM_16", "flac": "PCM_16"}


def _sf():
    import soundfile  # lazy: keeps CLI imports light

    return soundfile


def normalize_format(fmt: str) -> str:
    value = (fmt or "").lower().strip().lstrip(".")
    if value not in SUPPORTED_FORMATS:
        raise UsageError(
            f"unsupported audio format: {fmt!r}; P0 supports wav and flac",
            details={"format": fmt, "supported": list(SUPPORTED_FORMATS)},
        )
    return value


def format_from_path(path: str | Path, default: str = "wav") -> str:
    """Infer a container from the suffix.

    Only a *missing* suffix falls back to ``default``. An unknown suffix is a
    usage error: silently writing WAV bytes into ``out.mp3`` is worse than
    failing (codex P0-10).
    """
    suffix = Path(path).suffix.lower().lstrip(".")
    if not suffix:
        return default
    if suffix not in SUPPORTED_FORMATS:
        raise UsageError(
            f"cannot infer audio format from {str(path)!r}: unknown suffix '.{suffix}'; "
            f"use .wav/.flac or pass --format",
            details={"path": str(path), "suffix": suffix, "supported": list(SUPPORTED_FORMATS)},
        )
    return suffix


def resolve_output_format(path: str | Path | None, explicit: Optional[str] = None, default: str = "wav") -> str:
    """Resolve the output container for ``path``, validating ``--format`` agreement."""
    if path in (None, "-"):
        return normalize_format(explicit) if explicit else default
    path_fmt = format_from_path(path)
    if explicit:
        fmt = normalize_format(explicit)
        if fmt != path_fmt:
            raise UsageError(
                f"--format {fmt} does not match output suffix .{path_fmt} for {str(path)!r}"
            )
        return fmt
    return path_fmt


def read_audio(path: str | Path, *, mono: bool = True) -> tuple[np.ndarray, int]:
    p = Path(path).expanduser()
    if not p.is_file():
        raise InputError(f"audio file not found: {p}", details={"path": str(p)})
    try:
        data, sr = _sf().read(str(p), dtype="float32", always_2d=True)
    except Exception as exc:  # soundfile raises various backend errors
        raise InputError(f"cannot read audio file {p}: {exc}", details={"path": str(p)}) from exc
    if mono:
        if data.shape[1] > 1:
            data = data.mean(axis=1, keepdims=True)
        return np.ascontiguousarray(data[:, 0], dtype=np.float32), int(sr)
    return np.ascontiguousarray(data, dtype=np.float32), int(sr)


def audio_info(path: str | Path) -> dict:
    p = Path(path).expanduser()
    if not p.is_file():
        raise InputError(f"audio file not found: {p}", details={"path": str(p)})
    try:
        info = _sf().info(str(p))
    except Exception as exc:
        raise InputError(f"cannot inspect audio file {p}: {exc}", details={"path": str(p)}) from exc
    duration = float(info.frames) / float(info.samplerate) if info.samplerate else 0.0
    return {
        "path": str(p),
        "format": info.format,
        "subtype": info.subtype,
        "sample_rate": int(info.samplerate),
        "channels": int(info.channels),
        "frames": int(info.frames),
        "duration_s": round(duration, 4),
    }


def encode_audio(samples: np.ndarray, sr: int, fmt: str) -> bytes:
    fmt = normalize_format(fmt)
    buf = io.BytesIO()
    data = np.asarray(samples, dtype=np.float32)
    if data.ndim == 1:
        data = data[:, None]
    _sf().write(
        buf,
        data,
        int(sr),
        format="WAV" if fmt == "wav" else "FLAC",
        subtype=_SUBTYPES[fmt],
    )
    return buf.getvalue()


def write_audio(path: str | Path, samples: np.ndarray, sr: int, fmt: Optional[str] = None) -> Path:
    target = Path(path)
    fmt = normalize_format(fmt or format_from_path(target))
    atomic_write_bytes(target, encode_audio(samples, sr, fmt))
    return target


def write_audio_stdout(samples: np.ndarray, sr: int, fmt: str) -> None:
    payload = encode_audio(samples, sr, fmt)
    sys.stdout.buffer.write(payload)
    sys.stdout.buffer.flush()


def write_pcm16_wav(path: str | Path, samples: np.ndarray, sr: int) -> Path:
    return write_audio(path, samples, sr, "wav")


# --------------------------------------------------------------------------- #
# Manipulation
# --------------------------------------------------------------------------- #
def duration_s(samples: np.ndarray, sr: int) -> float:
    return float(len(samples)) / float(sr) if sr else 0.0


def silence_ms(ms: int, sr: int) -> np.ndarray:
    count = max(0, int(round(sr * ms / 1000.0)))
    return np.zeros(count, dtype=np.float32)


def concat(parts: Iterable[np.ndarray], sr: int, pause_after_ms: Optional[Iterable[int]] = None) -> np.ndarray:
    """Concatenate mono float32 arrays, inserting silence between them."""
    items = [np.asarray(p, dtype=np.float32) for p in parts]
    if not items:
        return np.zeros(0, dtype=np.float32)
    pauses = list(pause_after_ms) if pause_after_ms is not None else []
    out: list[np.ndarray] = []
    for idx, item in enumerate(items):
        out.append(item)
        if idx < len(items) - 1:
            pause = pauses[idx] if idx < len(pauses) else 0
            if pause:
                out.append(silence_ms(int(pause), sr))
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


def to_mono(samples: np.ndarray) -> np.ndarray:
    data = np.asarray(samples, dtype=np.float32)
    if data.ndim == 1:
        return data
    return data.mean(axis=1).astype(np.float32)


def frame_rms(samples: np.ndarray, sr: int, frame_ms: float = 20.0) -> np.ndarray:
    """RMS per frame in dBFS (0 dBFS == full scale)."""
    x = np.asarray(samples, dtype=np.float32)
    if x.size == 0:
        return np.zeros(0, dtype=np.float32)
    frame = max(1, int(sr * frame_ms / 1000.0))
    usable = (x.size // frame) * frame
    if usable == 0:
        usable = x.size
        frame = x.size
    trimmed = x[:usable].reshape(-1, frame)
    rms = np.sqrt(np.mean(np.square(trimmed, dtype=np.float64), axis=1))
    with np.errstate(divide="ignore"):
        return (20.0 * np.log10(np.maximum(rms, 1e-12))).astype(np.float32)


def dbfs_from_rms(rms: float) -> float:
    return float(20.0 * math.log10(max(rms, 1e-12)))


def peak_dbfs(samples: np.ndarray) -> float:
    x = np.asarray(samples, dtype=np.float32)
    if x.size == 0:
        return -120.0
    return float(20.0 * math.log10(max(float(np.max(np.abs(x))), 1e-12)))
