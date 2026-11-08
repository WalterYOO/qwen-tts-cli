"""Pure audio post-processing primitives (FR-P2-05).

This module performs no I/O and imports no model runtime. Loudness and true
peak measurements reuse :mod:`qtts_cli.audio_qc` (BS.1770-4), never a copied
K-weighting implementation.

The true-peak limiter is the frozen ``lookahead-min-4x`` algorithm from the
P2-1a contract addendum: a 4x oversampled gain envelope, a 5 ms attack hold,
a 50 ms release, and a final measured static safety attenuation. It never uses
sample clipping.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np

from .audio_io import peak_dbfs
from .audio_qc import integrated_loudness, true_peak_dbtp
from .errors import InputError, ModelError, UsageError

POST_DSP_VERSION = 1
LIMITER_ALGORITHM = "lookahead-min-4x"
FRAME_MS = 20.0
MAX_CHANNELS = 2
DEFAULT_LOUDNESS_TRUE_PEAK = -1.0
TRUE_PEAK_TOLERANCE_DB = 0.1
OVERSAMPLE = 4
ATTACK_MS = 5.0
RELEASE_MS = 50.0


def _as_2d(samples: np.ndarray) -> np.ndarray:
    data = np.asarray(samples, dtype=np.float32)
    if data.ndim == 1:
        data = data[:, None]
    if data.ndim != 2:
        raise InputError("audio must be a 1-D or 2-D array")
    if not np.all(np.isfinite(data)):
        raise InputError("audio contains NaN or Inf samples")
    return np.ascontiguousarray(data, dtype=np.float32)


def _peak(samples: np.ndarray) -> Optional[float]:
    return round(float(peak_dbfs(samples)), 2)


def measure(samples: np.ndarray, sr: int) -> dict[str, Optional[float]]:
    """Measure the report metrics for a finite stereo/mono float array."""
    lufs, _ = integrated_loudness(samples, sr)
    peak, _ = true_peak_dbtp(samples, sr)
    return {
        "lufs_integrated": lufs,
        "true_peak_dbtp": peak,
        "peak_dbfs": _peak(samples),
    }


_metrics = measure


def _frame_rms_db(samples: np.ndarray, sr: int) -> np.ndarray:
    """20 ms frame levels; the tail uses its actual length.

    For stereo, a frame is silent only when every channel is below threshold,
    i.e. the level is the maximum per-channel frame RMS.
    """
    data = _as_2d(samples)
    frame = max(1, int(round(sr * FRAME_MS / 1000.0)))
    frames = int(math.ceil(data.shape[0] / float(frame))) if data.shape[0] else 0
    values = np.empty(frames, dtype=np.float64)
    for index in range(frames):
        chunk = data[index * frame : (index + 1) * frame].astype(np.float64)
        rms = np.sqrt(np.mean(np.square(chunk), axis=0))
        values[index] = 20.0 * math.log10(max(float(np.max(rms)), 1e-12))
    return values


def trim_silence(
    samples: np.ndarray,
    sr: int,
    threshold_dbfs: float,
    min_silence_s: float,
    padding_ms: int,
    mode: str,
) -> tuple[np.ndarray, dict[str, float]]:
    """Trim leading/trailing silence without touching internal pauses."""
    data = _as_2d(samples)
    if data.shape[0] == 0:
        raise InputError("cannot trim silence from empty audio")
    if mode not in ("leading", "trailing", "both"):
        raise UsageError(f"invalid trim mode: {mode!r}")
    if min_silence_s < 0 or padding_ms < 0:
        raise UsageError("trim silence durations must be >= 0")

    if not np.any(data):
        raise InputError("audio is entirely silent; nothing remains after trimming")
    levels = _frame_rms_db(data, sr)
    silent = levels <= float(threshold_dbfs)
    if not np.any(~silent):
        raise InputError("audio is entirely silent; nothing remains after trimming")

    frame_samples = max(1, int(round(sr * FRAME_MS / 1000.0)))
    leading = 0
    while leading < silent.size and silent[leading]:
        leading += 1
    trailing = 0
    while trailing < silent.size and silent[silent.size - 1 - trailing]:
        trailing += 1
    leading_samples = min(data.shape[0], leading * frame_samples)
    # A trailing partial frame is shorter than ``frame_samples``; count only
    # the samples that actually belong to the silent tail.
    last_active_frame = silent.size - trailing
    trailing_samples = max(
        0,
        min(data.shape[0], data.shape[0] - last_active_frame * frame_samples),
    )

    min_samples = float(min_silence_s) * float(sr)
    padding_samples = int(round(float(padding_ms) * sr / 1000.0))
    start_cut = 0
    end_cut = 0
    if mode in ("leading", "both") and leading_samples >= min_samples:
        start_cut = max(0, leading_samples - padding_samples)
    if mode in ("trailing", "both") and trailing_samples >= min_samples:
        end_cut = max(0, trailing_samples - padding_samples)
    start_cut = min(start_cut, data.shape[0])
    end_cut = min(end_cut, data.shape[0] - start_cut)
    out = data[start_cut : data.shape[0] - end_cut]
    if out.shape[0] == 0:
        raise InputError("audio is entirely silent; nothing remains after trimming")
    info = {
        "start_s": round(start_cut / float(sr), 4),
        "end_s": round(end_cut / float(sr), 4),
        "removed_s": round((start_cut + end_cut) / float(sr), 4),
    }
    return np.ascontiguousarray(out, dtype=np.float32), info


def convert_channels(samples: np.ndarray, mode: str) -> tuple[np.ndarray, dict[str, int]]:
    """Convert between mono and stereo; ``keep`` preserves the input."""
    data = _as_2d(samples)
    if data.shape[1] > MAX_CHANNELS:
        raise InputError(
            f"audio has {data.shape[1]} channels; post supports at most {MAX_CHANNELS}"
        )
    before = int(data.shape[1])
    if mode == "keep":
        out = data
    elif mode == "mono":
        out = data.mean(axis=1, keepdims=True, dtype=np.float64).astype(np.float32)
    elif mode == "stereo":
        if before == 1:
            out = np.repeat(data, 2, axis=1)
        else:
            out = data
    else:
        raise UsageError(f"invalid channel mode: {mode!r}")
    return np.ascontiguousarray(out, dtype=np.float32), {
        "from": before,
        "to": int(out.shape[1]),
    }


def resample(samples: np.ndarray, sr: int, target_sr: int) -> tuple[np.ndarray, dict[str, int]]:
    """Polyphase resampling; identical rates are a no-op."""
    data = _as_2d(samples)
    if target_sr <= 0:
        raise UsageError("--sample-rate must be positive")
    if int(sr) <= 0:
        raise InputError("input sample rate must be positive")
    if int(sr) == int(target_sr):
        return data, {"from": int(sr), "to": int(target_sr)}
    try:
        from scipy.signal import resample_poly
    except ImportError as exc:  # pragma: no cover - scipy is a base dependency
        raise InputError("scipy is required for resampling") from exc
    divisor = math.gcd(int(sr), int(target_sr))
    up = int(target_sr) // divisor
    down = int(sr) // divisor
    channels = [
        resample_poly(data[:, channel].astype(np.float64), up, down)
        for channel in range(data.shape[1])
    ]
    out = np.stack(channels, axis=1).astype(np.float32)
    return np.ascontiguousarray(out), {"from": int(sr), "to": int(target_sr)}


def apply_gain(samples: np.ndarray, gain_db: float) -> np.ndarray:
    data = _as_2d(samples)
    if not math.isfinite(float(gain_db)):
        raise UsageError("gain must be finite")
    out = data * np.float32(10.0 ** (float(gain_db) / 20.0))
    if not np.all(np.isfinite(out)):
        raise InputError("gain produced non-finite samples")
    return np.ascontiguousarray(out, dtype=np.float32)


def _limiter_envelope(data: np.ndarray, sr: int, ceiling_dbtp: float) -> np.ndarray:
    """Return one linear gain per original sample, never above the raw envelope."""
    from scipy.ndimage import minimum_filter1d
    from scipy.signal import lfilter, resample_poly

    linear_ceiling = 10.0 ** (float(ceiling_dbtp) / 20.0)
    channels = []
    for channel in range(data.shape[1]):
        upsampled = resample_poly(data[:, channel].astype(np.float64), OVERSAMPLE, 1)
        target_length = data.shape[0] * OVERSAMPLE
        if upsampled.size < target_length:
            upsampled = np.pad(upsampled, (0, target_length - upsampled.size))
        else:
            upsampled = upsampled[:target_length]
        gains = np.minimum(
            1.0, linear_ceiling / np.maximum(np.abs(upsampled), 1e-12)
        )
        channels.append(gains.reshape(data.shape[0], OVERSAMPLE))
    # Link channels so the stereo image is preserved.
    raw = np.minimum.reduce(channels).min(axis=1).astype(np.float64)

    attack = max(1, int(round(sr * ATTACK_MS / 1000.0)))
    release = max(1, int(round(sr * RELEASE_MS / 1000.0)))
    hold = minimum_filter1d(raw, size=attack, mode="nearest")
    alpha = math.exp(-1.0 / float(release))
    smoothed = lfilter([1.0 - alpha], [1.0, -alpha], hold, zi=[alpha * hold[0]])[0]
    # A one-pole release may rise above the current envelope after a sharp
    # drop. Clamp to the held envelope, which itself is <= the raw envelope.
    return np.minimum(smoothed, hold)


def limit_true_peak(
    samples: np.ndarray, sr: int, ceiling_dbtp: float
) -> tuple[np.ndarray, float, Optional[float], float]:
    """Apply ``lookahead-min-4x`` and a measured static safety attenuation.

    Returns ``(limited, limiter_reduction_db, achieved_dbtp,
    safety_attenuation_db)``. This function never sample-clips.
    """
    data = _as_2d(samples)
    if not math.isfinite(float(ceiling_dbtp)):
        raise UsageError("--true-peak must be finite")
    current, reason = true_peak_dbtp(data, sr)
    if current is None:
        raise InputError(
            f"true peak is unavailable ({reason}); cannot enforce a ceiling"
        )
    if current <= ceiling_dbtp:
        return data, 0.0, current, 0.0

    envelope = _limiter_envelope(data, sr, ceiling_dbtp)
    minimum = float(np.min(envelope))
    reduction = -20.0 * math.log10(minimum) if minimum > 0.0 else 0.0
    limited = np.ascontiguousarray(data * envelope[:, None].astype(np.float32))
    if not np.all(np.isfinite(limited)):
        raise InputError("true-peak limiting produced non-finite samples")

    achieved, reason = true_peak_dbtp(limited, sr)
    if achieved is None:
        raise InputError(f"true peak became unavailable while limiting ({reason})")

    safety = 0.0
    for _ in range(4):
        if achieved <= ceiling_dbtp + TRUE_PEAK_TOLERANCE_DB:
            break
        step = max(0.0, achieved - ceiling_dbtp)
        limited = apply_gain(limited, -step)
        safety += step
        achieved, reason = true_peak_dbtp(limited, sr)
        if achieved is None:
            raise InputError(f"true peak became unavailable while limiting ({reason})")
    if achieved > ceiling_dbtp + TRUE_PEAK_TOLERANCE_DB:
        raise InputError(
            f"could not meet true-peak ceiling {ceiling_dbtp} dBTP "
            f"(achieved {achieved} dBTP)"
        )
    return limited, round(reduction, 3), achieved, round(safety, 3)


def _plugin_process(plugin, data: np.ndarray, sr: int, params: dict[str, Any]):
    """Run a post-process plugin and validate its ``(samples, sr)`` result."""
    handler = getattr(plugin, "process", None)
    if handler is None:
        handler = getattr(getattr(plugin, "obj", None), "process", None)
    if not callable(handler):
        raise ModelError("post-process plugin has no callable process()")
    try:
        result = handler(data, sr, dict(params))
    except ModelError:
        raise
    except Exception as exc:
        raise ModelError(
            f"post-process plugin failed: {type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(result, (tuple, list)) or len(result) != 2:
        raise InputError("post-process plugin must return (samples, sample_rate)")
    samples, out_sr = result
    expected_channels = int(data.shape[1])
    try:
        array = np.asarray(samples, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise InputError(f"post-process plugin returned non-numeric audio: {exc}") from exc
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2:
        raise InputError("post-process plugin must return 1-D or 2-D audio")
    if array.shape[1] != expected_channels:
        raise InputError(
            "post-process plugin changed the channel count "
            f"({expected_channels} -> {array.shape[1]})"
        )
    if array.shape[0] == 0:
        raise InputError("post-process plugin returned empty audio")
    if not np.all(np.isfinite(array)):
        raise InputError("post-process plugin returned NaN or Inf samples")
    try:
        rate = int(out_sr)
        exact = float(out_sr) == float(rate)
    except (TypeError, ValueError) as exc:
        raise InputError(f"post-process plugin returned an invalid sample rate: {out_sr!r}") from exc
    if not exact or rate <= 0 or rate != int(sr):
        raise InputError(
            f"post-process plugin must keep the sample rate at {int(sr)} Hz "
            f"(got {out_sr!r})"
        )
    ratio = float(array.shape[0]) / float(data.shape[0])
    if not (0.5 <= ratio <= 2.0):
        raise InputError(
            "post-process plugin changed duration by "
            f"{ratio:.2f}x; allowed range is 0.5x..2.0x"
        )
    return np.ascontiguousarray(array, dtype=np.float32), rate


def _plugin_applied(plugin, params: dict[str, Any]) -> dict[str, Any]:
    payload = getattr(plugin, "applied_payload", None)
    if callable(payload):
        return payload(dict(params))
    return {
        "name": str(getattr(plugin, "name", "plugin")),
        "params": dict(params),
        "api_version": int(getattr(plugin, "api_version", 1)),
        "distribution": getattr(plugin, "distribution", None),
        "version": getattr(plugin, "version", None),
    }


def process(
    samples: np.ndarray,
    sr: int,
    options: dict[str, Any],
    *,
    plugin: Any = None,
) -> tuple[np.ndarray, int, dict[str, Any], dict[str, Any], dict[str, Any], list[str]]:
    """Apply the fixed processing order and return measurements for reporting."""
    data = _as_2d(samples)
    if data.shape[0] == 0:
        raise InputError("audio is empty")
    if data.shape[1] > MAX_CHANNELS:
        raise InputError(
            f"audio has {data.shape[1]} channels; post supports at most {MAX_CHANNELS}"
        )
    before = measure(data, sr)

    applied: dict[str, Any] = {
        "gain_db": 0.0,
        "limiter_reduction_db": 0.0,
        "safety_attenuation_db": 0.0,
        "resample": None,
        "channels": {"from": int(data.shape[1]), "to": int(data.shape[1])},
        "trim": None,
        "true_peak_ceiling_dbtp": None,
        "limiter_algorithm": LIMITER_ALGORITHM,
    }
    warnings: list[str] = []

    if options.get("trim"):
        data, trim_info = trim_silence(
            data,
            sr,
            float(options.get("trim_threshold_dbfs", -50.0)),
            float(options.get("trim_min_silence_s", 0.2)),
            int(options.get("trim_padding_ms", 50)),
            str(options.get("trim_mode", "both")),
        )
        applied["trim"] = trim_info

    channel_mode = str(options.get("channels", "keep"))
    data, channel_info = convert_channels(data, channel_mode)
    applied["channels"] = channel_info

    target_sr = options.get("sample_rate")
    out_sr = int(sr)
    if target_sr is not None and int(target_sr) != int(sr):
        data, resample_info = resample(data, sr, int(target_sr))
        out_sr = int(target_sr)
        applied["resample"] = resample_info

    # Plugin DSP runs after trim/channels/resample and before loudness gain,
    # so the LUFS/true-peak guarantees are measured on the plugin's output.
    if plugin is not None:
        params = dict(options.get("plugin_params") or {})
        data, out_sr = _plugin_process(plugin, data, out_sr, params)
        applied["postprocess_plugin"] = _plugin_applied(plugin, params)

    lufs_target = options.get("lufs")
    if lufs_target is not None:
        current, reason = integrated_loudness(data, out_sr)
        if current is None:
            raise InputError(
                f"integrated loudness is unavailable ({reason}); cannot normalize"
            )
        gain_db = float(lufs_target) - float(current)
        data = apply_gain(data, gain_db)
        applied["gain_db"] = round(gain_db, 3)

    ceiling = options.get("true_peak")
    if ceiling is None and lufs_target is not None:
        ceiling = DEFAULT_LOUDNESS_TRUE_PEAK
    if ceiling is not None:
        data, reduction, _achieved, safety = limit_true_peak(
            data, out_sr, float(ceiling)
        )
        applied["true_peak_ceiling_dbtp"] = float(ceiling)
        applied["limiter_reduction_db"] = round(float(reduction), 3)
        applied["safety_attenuation_db"] = round(float(safety), 3)

    after_dsp = measure(data, out_sr)
    return (
        np.ascontiguousarray(data, dtype=np.float32),
        int(out_sr),
        applied,
        before,
        after_dsp,
        warnings,
    )
