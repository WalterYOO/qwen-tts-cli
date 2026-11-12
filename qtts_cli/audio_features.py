"""Acoustic voice features for ``qc --metrics full`` and design reports.

Pure numpy/scipy except HNR, which uses ``praat-parselmouth`` when installed.
Every unavailable metric returns ``None`` plus a reason in ``unavailable``;
values are never fabricated (see docs/qtts-design-metrics-voice-plan.md).
"""

from __future__ import annotations

import math
from typing import Any, Optional, Sequence

import numpy as np

DEFAULT_F0_MIN_HZ = 60.0
DEFAULT_F0_MAX_HZ = 500.0
YIN_THRESHOLD = 0.15
MIN_WINDOW_S = 0.04
HOP_S = 0.01
MIN_VOICED_FRAMES = 3
MIN_PERCENTILE_FRAMES = 10
CENTROID_N_FFT = 1024
CENTROID_HOP = 256


def _mono(samples: Sequence[float] | np.ndarray) -> np.ndarray:
    data = np.asarray(samples, dtype=np.float64)
    if data.ndim == 1:
        return np.ascontiguousarray(data)
    if data.ndim != 2 or data.shape[1] == 0:
        raise ValueError("audio must be mono or (frames, channels)")
    return np.ascontiguousarray(data.mean(axis=1))


def _round(value: Optional[float], digits: int = 2) -> Optional[float]:
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value):
        return None
    return round(value, digits)


def _frame_rms_db(frames: np.ndarray) -> np.ndarray:
    mean_square = np.mean(frames**2, axis=1)
    return 20.0 * np.log10(np.maximum(mean_square, 1e-20))


def _frames(signal: np.ndarray, length: int, hop: int) -> np.ndarray:
    if signal.size < length:
        signal = np.pad(signal, (0, length - signal.size))
    view = np.lib.stride_tricks.sliding_window_view(signal, length)
    return view[::hop]


def _yin_frame(
    frame: np.ndarray,
    sr: int,
    *,
    f0_min: float,
    f0_max: float,
    threshold: float = YIN_THRESHOLD,
) -> tuple[Optional[float], Optional[float]]:
    """Return ``(f0_hz, cmndf_min)`` for one frame; ``(None, None)`` if too short."""
    x = np.asarray(frame, dtype=np.float64)
    x = x - x.mean()
    n = x.size
    tau_min = max(1, int(math.floor(sr / f0_max)))
    tau_max = min(int(math.ceil(sr / f0_min)), n - 2)
    if tau_max <= tau_min:
        return None, None

    size = 1 << (2 * n - 1).bit_length()
    spec = np.fft.rfft(x, size)
    auto = np.fft.irfft(spec * np.conj(spec), size)[:n]

    csum = np.concatenate(([0.0], np.cumsum(x * x)))
    taus = np.arange(1, tau_max + 1)
    # d(tau) = sum_{j=0}^{n-1-tau} x[j]^2 + sum_{j=tau}^{n-1} x[j]^2 - 2*acf(tau)
    e0 = csum[n - taus]
    e1 = csum[n] - csum[taus]
    diff = np.maximum(e0 + e1 - 2.0 * auto[taus], 0.0)
    cumulative = np.cumsum(diff)
    cmndf = diff * taus / np.maximum(cumulative, 1e-12)

    search = cmndf[tau_min - 1 :]
    if search.size == 0:
        return None, None
    # YIN wants the first *local minimum* below the threshold. Taking the
    # first sample below it would stop on the descending slope of a pure
    # tone (e.g. tau=99 instead of 109 for 220 Hz at 24 kHz).
    candidates: list[int] = []
    if search.size == 1:
        candidates.append(0)
    elif search.size >= 2:
        # `search[0]` is tau_min itself; a 500 Hz tone at 24 kHz has its true
        # lag exactly there, so skipping it would report the octave below.
        if search[0] <= search[1]:
            candidates.append(0)
        for offset in range(1, search.size - 1):
            if search[offset] <= search[offset - 1] and search[offset] <= search[offset + 1]:
                candidates.append(offset)
        if search[-1] <= search[-2]:
            candidates.append(search.size - 1)
    below = [offset for offset in candidates if search[offset] < threshold]
    if not below:
        # YIN treats a frame as unvoiced when no dip crosses the threshold.
        # Returning the global argmin here would label white noise as a
        # pitch of ~60 Hz and inflate voiced_ratio.
        return None, _round(float(search[int(np.argmin(search))]))
    index = below[0]
    tau = tau_min + index

    # Parabolic interpolation around the chosen lag.
    if 0 < index < search.size - 1:
        left, mid, right = search[index - 1], search[index], search[index + 1]
        denominator = left - 2.0 * mid + right
        if abs(denominator) > 1e-12:
            delta = float(np.clip(0.5 * (left - right) / denominator, -0.5, 0.5))
            tau = tau + delta
    if tau <= 0:
        return None, None
    f0 = float(sr) / float(tau)
    if not math.isfinite(f0) or not (f0_min <= f0 <= f0_max):
        return None, _round(float(search[index]))
    return f0, _round(float(search[index]))


def _f0_stats(values: list[float], unavailable: dict[str, str]) -> dict[str, Optional[float]]:
    result: dict[str, Optional[float]] = {
        "f0_median_hz": None,
        "f0_mean_hz": None,
        "f0_std_hz": None,
        "f0_p10_hz": None,
        "f0_p90_hz": None,
    }
    if len(values) < MIN_VOICED_FRAMES:
        # Keep a more specific upstream reason such as no_non_silent_audio.
        unavailable.setdefault("f0_median_hz", "insufficient_voiced_frames")
        return result
    array = np.asarray(values, dtype=np.float64)
    result["f0_median_hz"] = _round(float(np.median(array)))
    result["f0_mean_hz"] = _round(float(np.mean(array)))
    result["f0_std_hz"] = _round(float(np.std(array)))
    if array.size >= MIN_PERCENTILE_FRAMES:
        result["f0_p10_hz"] = _round(float(np.percentile(array, 10)))
        result["f0_p90_hz"] = _round(float(np.percentile(array, 90)))
    else:
        reason = "insufficient_voiced_frames_for_percentiles"
        unavailable.setdefault("f0_p10_hz", reason)
        unavailable.setdefault("f0_p90_hz", reason)
    return result


def _yin(
    mono: np.ndarray,
    sr: int,
    *,
    f0_min: float,
    f0_max: float,
    silence_dbfs: float,
) -> tuple[dict[str, Optional[float]], dict[str, str]]:
    unavailable: dict[str, str] = {}
    window = max(int(round(MIN_WINDOW_S * sr)), int(round(3.0 * sr / f0_min)))
    hop = max(1, int(round(HOP_S * sr)))
    frames = _frames(mono, window, hop)
    if frames.size == 0:
        unavailable["f0_median_hz"] = "no_non_silent_audio"
        unavailable["voiced_ratio"] = "no_non_silent_audio"
        return _f0_stats([], unavailable) | {"voiced_ratio": None}, unavailable

    frame_db = _frame_rms_db(frames)
    analyzable = frame_db > silence_dbfs
    if not np.any(analyzable):
        unavailable["f0_median_hz"] = "no_non_silent_audio"
        unavailable["voiced_ratio"] = "no_non_silent_audio"
        return _f0_stats([], unavailable) | {"voiced_ratio": None}, unavailable

    voiced_values: list[float] = []
    for frame in frames[analyzable]:
        f0, _ = _yin_frame(frame, sr, f0_min=f0_min, f0_max=f0_max)
        if f0 is not None:
            voiced_values.append(f0)

    voiced_ratio = len(voiced_values) / float(int(np.count_nonzero(analyzable)))
    stats = _f0_stats(voiced_values, unavailable)
    # 0.0 is a measured ratio, not a missing value; only a zero denominator
    # (handled above) makes it unavailable.
    stats["voiced_ratio"] = _round(voiced_ratio, 4)
    return stats, unavailable


def _praat_hnr(
    mono: np.ndarray, sr: int, *, f0_min: float
) -> tuple[Optional[float], Optional[str], Optional[str]]:
    try:
        import parselmouth  # type: ignore
    except ImportError:
        return None, "dependency_missing:praat-parselmouth", None
    except Exception:
        return None, "analysis_failed", _praat_version()
    try:
        sound = parselmouth.Sound(mono, sampling_frequency=sr)
        try:
            harmonicity = sound.to_harmonicity_cc(
                time_step=0.01,
                minimum_pitch=float(f0_min),
                silence_threshold=0.1,
                periods_per_window=4.5,
            )
        except TypeError:
            harmonicity = sound.to_harmonicity_cc(
                time_step=0.01,
                minimum_pitch=float(f0_min),
                silence_threshold=0.1,
            )
        values = np.asarray(harmonicity.values, dtype=np.float64).reshape(-1)
        values = values[np.isfinite(values) & (values > -100.0)]
        if values.size == 0:
            return None, "no_voiced_frames", _praat_version()
        return _round(float(np.median(values))), None, _praat_version()
    except Exception:
        return None, "analysis_failed", _praat_version()


def _praat_version() -> Optional[str]:
    try:
        from importlib.metadata import version

        return version("praat-parselmouth")
    except Exception:
        return None


def _spectral_centroid(
    mono: np.ndarray, sr: int, *, silence_dbfs: float
) -> tuple[Optional[float], Optional[str]]:
    n_fft = CENTROID_N_FFT
    frames = _frames(mono, n_fft, CENTROID_HOP)
    if frames.size == 0:
        return None, "no_non_silent_audio"
    frame_db = _frame_rms_db(frames)
    keep = frame_db > silence_dbfs
    if not np.any(keep):
        return None, "no_non_silent_audio"
    window = np.hanning(n_fft)
    spectrum = np.abs(np.fft.rfft(frames[keep] * window, axis=1))
    frequencies = np.fft.rfftfreq(n_fft, d=1.0 / float(sr))
    denominators = spectrum.sum(axis=1)
    valid = denominators > 0.0
    if not np.any(valid):
        return None, "no_non_silent_audio"
    centroids = (spectrum[valid] * frequencies).sum(axis=1) / denominators[valid]
    return _round(float(np.median(centroids)), 1), None


def _ltas(
    mono: np.ndarray, sr: int
) -> tuple[
    Optional[float],
    Optional[float],
    dict[str, str],
]:
    unavailable: dict[str, str] = {}
    if mono.size < 16:
        unavailable["ltas_slope_db_per_octave"] = "too_short_for_ltas"
        unavailable["hf_lf_db"] = "band_unavailable"
        return None, None, unavailable
    try:
        from scipy import signal
    except Exception:  # pragma: no cover - scipy is a core dependency
        unavailable["ltas_slope_db_per_octave"] = "analysis_failed"
        unavailable["hf_lf_db"] = "analysis_failed"
        return None, None, unavailable

    nperseg = min(4096, mono.size)
    try:
        frequencies, power = signal.welch(
            mono,
            fs=sr,
            window="hann",
            nperseg=nperseg,
            noverlap=nperseg // 2,
            detrend="constant",
            scaling="density",
        )
    except Exception:
        unavailable["ltas_slope_db_per_octave"] = "analysis_failed"
        unavailable["hf_lf_db"] = "analysis_failed"
        return None, None, unavailable

    upper = min(8000.0, float(sr) / 2.0)
    mask = (frequencies >= 50.0) & (frequencies <= upper)
    slope: Optional[float] = None
    if mono.size < int(0.5 * sr):
        unavailable["ltas_slope_db_per_octave"] = "too_short_for_ltas"
    elif int(np.count_nonzero(mask)) < 10:
        unavailable["ltas_slope_db_per_octave"] = "too_short_for_ltas"
    else:
        f = frequencies[mask]
        p = np.maximum(power[mask], 1e-20)
        try:
            slope = _round(float(np.polyfit(np.log2(f), 10.0 * np.log10(p), 1)[0]))
        except Exception:
            unavailable["ltas_slope_db_per_octave"] = "analysis_failed"

    hf_lf: Optional[float] = None
    low = (frequencies >= 500.0) & (frequencies < 4000.0)
    high = (frequencies >= 4000.0) & (frequencies <= upper)
    # A single Nyquist bin is not a meaningful 4-8 kHz band (8 kHz audio),
    # and a band needs at least a couple of bins to be comparable.
    if int(np.count_nonzero(low)) < 2 or int(np.count_nonzero(high)) < 2:
        unavailable["hf_lf_db"] = "band_unavailable"
    else:
        delta = float(frequencies[1] - frequencies[0]) if frequencies.size > 1 else 0.0
        low_energy = float(np.sum(power[low]) * delta)
        high_energy = float(np.sum(power[high]) * delta)
        if low_energy <= 0.0 or delta <= 0.0:
            unavailable["hf_lf_db"] = "band_unavailable"
        else:
            hf_lf = _round(10.0 * math.log10(max(high_energy, 1e-20) / low_energy))
    return slope, hf_lf, unavailable


def analyze_features(
    samples: Sequence[float] | np.ndarray,
    sr: int,
    *,
    silence_dbfs: float = -50.0,
    f0_min: float = DEFAULT_F0_MIN_HZ,
    f0_max: float = DEFAULT_F0_MAX_HZ,
    hnr_backend: str = "praat",
) -> dict[str, Any]:
    """Measure F0, voicing, HNR and spectral shape for one audio clip.

    Missing measurements are ``None`` and explained in ``unavailable``.
    """
    if int(sr) <= 0:
        raise ValueError("sample rate must be positive")
    if f0_min <= 0 or f0_max <= f0_min:
        raise ValueError("f0_min must be positive and below f0_max")
    backend = (hnr_backend or "praat").lower()
    if backend not in ("praat", "auto"):
        raise ValueError("hnr_backend must be 'praat'")

    mono = _mono(samples)
    if mono.size == 0:
        raise ValueError("cannot analyze empty audio")

    unavailable: dict[str, str] = {}
    result: dict[str, Any] = {}
    f0_values, f0_unavailable = _yin(
        mono, sr, f0_min=f0_min, f0_max=f0_max, silence_dbfs=silence_dbfs
    )
    unavailable.update(f0_unavailable)
    result.update(f0_values)

    hnr, reason, praat_version = _praat_hnr(mono, sr, f0_min=f0_min)
    result["hnr_db"] = hnr
    if reason:
        unavailable["hnr_db"] = reason

    centroid, centroid_reason = _spectral_centroid(mono, sr, silence_dbfs=silence_dbfs)
    result["spectral_centroid_hz"] = centroid
    if centroid_reason:
        unavailable["spectral_centroid_hz"] = centroid_reason

    slope, hf_lf, ltas_unavailable = _ltas(mono, sr)
    result["ltas_slope_db_per_octave"] = slope
    result["hf_lf_db"] = hf_lf
    unavailable.update(ltas_unavailable)

    result["unavailable"] = unavailable
    result["analysis"] = {
        "analysis_channels": "downmix",
        "f0_backend": "yin",
        "hnr_backend": "praat" if praat_version else "unavailable",
        "f0_min_hz": f0_min,
        "f0_max_hz": f0_max,
        "yin_threshold": YIN_THRESHOLD,
        "numpy": np.__version__,
        "scipy": _scipy_version(),
        "praat_parselmouth": praat_version,
    }
    return result


def _scipy_version() -> Optional[str]:
    try:
        import scipy

        return getattr(scipy, "__version__", None)
    except Exception:  # pragma: no cover
        return None
