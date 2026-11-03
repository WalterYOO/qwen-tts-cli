"""Audio quality metrics for FR-P1-07 (pure numpy/scipy, no TTS models).

``analyze()`` only measures. Thresholds and exit codes live in the CLI layer
so the same metrics can feed ``qtts qc``, ``say --qc`` and reports.

Channel semantics (t10 H-D01): the caller must read audio with
``mono=False``. Mono and stereo are kept as ``(frames, channels)``; more than
two channels is an input error, never a silent downmix. K-weighting, true
peak, peak, clipping and DC offset are all computed per channel.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Sequence

import numpy as np

from .errors import InputError
from .qc import normalize_for_qc
from .textnorm import detect_language

DEFAULT_SILENCE_DBFS = -50.0
DEFAULT_MIN_SILENCE_S = 0.2
DEFAULT_FRAME_MS = 20.0
LUFS_BLOCK_S = 0.4
LUFS_HOP_S = 0.1
LUFS_SAMPLE_RATE = 48000
CLIP_LEVEL = 0.999
MAX_CHANNELS = 2
_CJK_LANGUAGES = {"chinese", "japanese", "cantonese"}

# ITU-R BS.1770-4 K-weighting biquads at 48 kHz.
_K_STAGE1_B = (1.53512485958697, -2.69169618940638, 1.19839281085285)
_K_STAGE1_A = (1.0, -1.69065929318241, 0.73248077421585)
_K_STAGE2_B = (1.0, -2.0, 1.0)
_K_STAGE2_A = (1.0, -1.99004745483398, 0.99007225036621)


def _scipy_signal():
    try:
        from scipy import signal  # lazy: only metrics that need it import scipy
    except Exception:  # pragma: no cover - environment dependent
        return None
    return signal


def _as_channels(samples: Sequence[float] | np.ndarray) -> np.ndarray:
    """Return finite ``(frames, channels)`` float64 with 1 or 2 channels."""
    data = np.asarray(samples, dtype=np.float64)
    if data.ndim == 1:
        data = data[:, None]
    if data.ndim != 2:
        raise InputError(
            "audio must be mono or stereo", details={"shape": list(data.shape)}
        )
    if data.shape[0] == 0 or data.shape[1] == 0:
        raise InputError("cannot analyze empty audio")
    if data.shape[1] > MAX_CHANNELS:
        raise InputError(
            f"audio has {data.shape[1]} channels; qc supports mono/stereo only",
            details={"channels": int(data.shape[1])},
        )
    if not np.all(np.isfinite(data)):
        raise InputError("audio contains NaN or Inf samples")
    return np.ascontiguousarray(data)


def _is_digital_silence(data: np.ndarray) -> bool:
    return not bool(np.any(data))


def _resample_channels(data: np.ndarray, sr: int, target: int):
    signal = _scipy_signal()
    if signal is None:
        return None
    if sr == target:
        return data
    divisor = math.gcd(int(sr), target)
    up, down = target // divisor, int(sr) // divisor
    return np.stack(
        [signal.resample_poly(data[:, ch], up, down) for ch in range(data.shape[1])],
        axis=1,
    )


def integrated_loudness(
    samples: Sequence[float] | np.ndarray, sr: int
) -> tuple[Optional[float], Optional[str]]:
    """ITU-R BS.1770-4 integrated loudness in LUFS, or ``(None, reason)``.

    K-weighting runs at its native 48 kHz coefficients after polyphase
    resampling. Blocks are 400 ms with 100 ms hops; the absolute gate is
    -70 LUFS and the relative gate -10 LU. Mono/stereo channel weights are
    both 1.0, so block energy is summed over channels.
    """
    signal = _scipy_signal()
    if signal is None:
        return None, "scipy_unavailable"
    data = _as_channels(samples)
    if _is_digital_silence(data):
        return None, "digital_silence"
    resampled = _resample_channels(data, int(sr), LUFS_SAMPLE_RATE)
    if resampled is None:
        return None, "scipy_unavailable"
    weighted = np.empty_like(resampled, dtype=np.float64)
    for ch in range(resampled.shape[1]):
        channel = signal.lfilter(_K_STAGE1_B, _K_STAGE1_A, resampled[:, ch])
        weighted[:, ch] = signal.lfilter(_K_STAGE2_B, _K_STAGE2_A, channel)

    block = int(round(LUFS_BLOCK_S * LUFS_SAMPLE_RATE))
    hop = int(round(LUFS_HOP_S * LUFS_SAMPLE_RATE))
    if weighted.shape[0] < block:
        return None, "too_short_for_lufs"
    mean_squares = np.array(
        [
            float(np.sum(np.mean(weighted[start : start + block, :] ** 2, axis=0)))
            for start in range(0, weighted.shape[0] - block + 1, hop)
        ],
        dtype=np.float64,
    )
    loudness = -0.691 + 10.0 * np.log10(np.maximum(mean_squares, 1e-12))

    gated = loudness > -70.0
    if not np.any(gated):
        return None, "below_absolute_gate"
    relative_gate = -0.691 + 10.0 * np.log10(float(np.mean(mean_squares[gated]))) - 10.0
    keep = gated & (loudness > relative_gate)
    if not np.any(keep):
        return None, "below_relative_gate"
    value = -0.691 + 10.0 * math.log10(float(np.mean(mean_squares[keep])))
    return round(float(value), 2), None


def true_peak_dbtp(
    samples: Sequence[float] | np.ndarray,
    sr: int,
    *,
    oversample: int = 4,
) -> tuple[Optional[float], Optional[str]]:
    """Peak after polyphase oversampling (>=4x), in dBTP."""
    signal = _scipy_signal()
    if signal is None:
        return None, "scipy_unavailable"
    if oversample < 4:
        raise ValueError("true peak requires at least 4x oversampling")
    data = _as_channels(samples)
    if _is_digital_silence(data):
        return None, "digital_silence"
    try:
        peaks = [
            float(np.max(np.abs(signal.resample_poly(data[:, ch], oversample, 1))))
            for ch in range(data.shape[1])
        ]
    except Exception:  # pragma: no cover - unusual backend error
        return None, "resample_failed"
    return round(20.0 * math.log10(max(max(peaks), 1e-12)), 2), None


def cosine_similarity(
    first: Sequence[float], second: Sequence[float]
) -> Optional[float]:
    a = np.asarray(first, dtype=np.float64).reshape(-1)
    b = np.asarray(second, dtype=np.float64).reshape(-1)
    if a.size == 0 or a.shape != b.shape:
        return None
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0.0 or norm_b == 0.0:
        return None
    return float(np.clip(np.dot(a, b) / (norm_a * norm_b), -1.0, 1.0))


def _embedding_pair(reference_embedding: Any, audio_embedding: Any) -> tuple[Any, Any]:
    """Accept a single reference vector, ``(reference, audio)``, or a dict."""
    if audio_embedding is not None:
        return reference_embedding, audio_embedding
    if isinstance(reference_embedding, dict):
        reference = (
            reference_embedding.get("reference") or reference_embedding.get("ref")
        )
        audio = (
            reference_embedding.get("audio")
            or reference_embedding.get("hypothesis")
        )
        return reference, audio
    if isinstance(reference_embedding, (tuple, list)) and len(reference_embedding) == 2:
        first, second = reference_embedding
        try:
            shape_first = np.asarray(first).shape
            shape_second = np.asarray(second).shape
        except Exception:  # pragma: no cover - ragged input
            return None, None
        if len(shape_first) == 1 and len(shape_second) == 1:
            return first, second
    return None, None


def _speech_rate(
    text: Optional[str],
    language: str,
    non_silent_s: float,
    unavailable: dict[str, str],
) -> Optional[dict[str, Optional[float]]]:
    if not text or not str(text).strip():
        unavailable["speech_rate"] = "no_text"
        return None
    normalized = normalize_for_qc(str(text), language or "auto")
    chars = len([ch for ch in normalized if not ch.isspace()])
    if non_silent_s <= 0 or chars == 0:
        unavailable["speech_rate"] = "no_non_silent_audio"
        return None
    lang = (language or "auto").strip()
    if lang.lower() == "auto":
        lang = detect_language(normalized)
    cjk = lang.lower() in _CJK_LANGUAGES
    if cjk:
        unavailable["words_per_min"] = "cjk_no_word_boundaries"
        words_per_min = None
    else:
        words = len(normalized.split())
        if words == 0:
            unavailable["words_per_min"] = "no_words"
            words_per_min = None
        else:
            words_per_min = round(words / (non_silent_s / 60.0), 2)
    return {
        "chars_per_s": round(chars / non_silent_s, 2),
        "words_per_min": words_per_min,
    }


def _frame_rms_channels(
    data: np.ndarray, sr: int, frame_ms: float
) -> tuple[np.ndarray, float]:
    """Sample-accurate frame RMS in dBFS, zero-padded to the full duration.

    Unlike ``audio_io.frame_rms`` this never drops a trailing partial frame
    (t10 H-D03).
    """
    frame = max(1, int(round(sr * frame_ms / 1000.0)))
    padded = (-data.shape[0]) % frame
    if padded:
        data = np.pad(data, ((0, padded), (0, 0)), mode="constant")
    frames = data.reshape(-1, frame, data.shape[1])
    mean_square = np.mean(frames**2, axis=(1, 2))
    rms_db = 20.0 * np.log10(np.maximum(mean_square, 1e-12))
    return rms_db.astype(np.float64), float(frame) / float(sr)


def _silence_mask(
    silent: np.ndarray, frame_s: float, min_silence_s: float
) -> tuple[np.ndarray, float]:
    """Keep silent runs of at least ``min_silence_s``; return mask + longest."""
    mask = np.zeros(silent.shape, dtype=bool)
    longest = 0.0
    index = 0
    total = int(silent.size)
    while index < total:
        if not bool(silent[index]):
            index += 1
            continue
        end = index
        while end < total and bool(silent[end]):
            end += 1
        run_s = float(end - index) * frame_s
        if run_s + 1e-9 >= float(min_silence_s):
            mask[index:end] = True
            longest = max(longest, run_s)
        index = end
    return mask, longest


def analyze(
    samples: Sequence[float] | np.ndarray,
    sr: int,
    *,
    text: Optional[str] = None,
    language: str = "auto",
    reference_embedding: Any = None,
    audio_embedding: Any = None,
    silence_dbfs: float = DEFAULT_SILENCE_DBFS,
    min_silence_s: float = DEFAULT_MIN_SILENCE_S,
    frame_ms: float = DEFAULT_FRAME_MS,
) -> dict[str, Any]:
    """Measure audio quality. Missing measurements are null, never fabricated."""
    if int(sr) <= 0:
        raise InputError("sample rate must be positive", details={"sample_rate": sr})
    data = _as_channels(samples)
    rate = int(sr)
    duration = float(data.shape[0]) / float(rate)
    unavailable: dict[str, str] = {}

    absolute = np.abs(data)
    peak = float(np.max(absolute))
    digital_silence = peak == 0.0
    peak_db: Optional[float] = None
    if digital_silence:
        unavailable["peak"] = "digital_silence"
    else:
        peak_db = round(20.0 * math.log10(max(peak, 1e-12)), 2)

    clipping = absolute >= CLIP_LEVEL
    clipping_count = int(np.count_nonzero(clipping))

    channel_means = np.mean(data, axis=0)
    dc_offset = float(channel_means[int(np.argmax(np.abs(channel_means)))])

    rms_db, frame_s = _frame_rms_channels(data, rate, frame_ms)
    silent = rms_db < float(silence_dbfs)
    silence_mask, longest_silence_s = _silence_mask(
        silent, frame_s, min_silence_s
    )
    silent_s = float(np.count_nonzero(silence_mask)) * frame_s
    silence_ratio = min(1.0, silent_s / duration) if duration > 0 else 0.0
    non_silent_s = max(0.0, duration - silent_s)

    lufs, lufs_reason = integrated_loudness(data, rate)
    if lufs is None:
        unavailable["lufs"] = lufs_reason or "unavailable"
    true_peak, true_peak_reason = true_peak_dbtp(data, rate)
    if true_peak is None:
        unavailable["true_peak"] = true_peak_reason or "unavailable"

    speech_rate = _speech_rate(text, language, non_silent_s, unavailable)

    similarity: Optional[float] = None
    reference, compared = _embedding_pair(reference_embedding, audio_embedding)
    if reference is None or compared is None:
        if reference_embedding is not None or audio_embedding is not None:
            unavailable["speaker_similarity"] = "reference_embedding_unavailable"
    else:
        first = np.asarray(reference, dtype=np.float64).reshape(-1)
        second = np.asarray(compared, dtype=np.float64).reshape(-1)
        if first.shape != second.shape:
            unavailable["speaker_similarity"] = "embedding_shape_mismatch"
        elif first.size == 0 or not np.any(first) or not np.any(second):
            unavailable["speaker_similarity"] = "zero_norm_embedding"
        else:
            value = cosine_similarity(first, second)
            if value is None:
                unavailable["speaker_similarity"] = "zero_norm_embedding"
            else:
                similarity = round(float(value), 4)

    return {
        "duration_s": round(duration, 4),
        "sample_rate": rate,
        "channels": int(data.shape[1]),
        "peak_dbfs": peak_db,
        "clipping_ratio": round(float(clipping_count) / float(data.size), 6),
        "clipping_count": clipping_count,
        "dc_offset": round(dc_offset, 6),
        "silence_ratio": round(silence_ratio, 4),
        "longest_silence_s": round(longest_silence_s, 3),
        "lufs_integrated": lufs,
        "true_peak_dbtp": true_peak,
        "speech_rate": speech_rate,
        "speaker_similarity": similarity,
        "unavailable": unavailable,
    }
