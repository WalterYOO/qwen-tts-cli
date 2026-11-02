"""Reference-audio admission checks (FR-P0-03).

Every check returns an actionable status so `voice add` can explain *why* a
reference clip is rejected before any model is loaded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .audio_io import audio_info, duration_s, frame_rms, peak_dbfs, read_audio

OK = "ok"
WARN = "warn"
ERROR = "error"


@dataclass
class Check:
    name: str
    value: Any
    status: str = OK
    advice: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"name": self.name, "value": self.value, "status": self.status}
        if self.advice:
            out["advice"] = self.advice
        return out


@dataclass
class AdmissionReport:
    checks: list[Check] = field(default_factory=list)
    path: Optional[str] = None

    @property
    def errors(self) -> list[Check]:
        return [c for c in self.checks if c.status == ERROR]

    @property
    def warnings(self) -> list[Check]:
        return [c for c in self.checks if c.status == WARN]

    @property
    def ok(self) -> bool:
        return not self.errors

    def add(self, check: Check) -> None:
        self.checks.append(check)

    def get(self, name: str) -> Optional[Check]:
        for check in self.checks:
            if check.name == name:
                return check
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "path": self.path,
            "errors": [c.to_dict() for c in self.errors],
            "warnings": [c.to_dict() for c in self.warnings],
            "checks": [c.to_dict() for c in self.checks],
        }


def _coverage_rate(ref_text: Optional[str], speech_s: float, language: Optional[str]) -> Optional[float]:
    if not ref_text or speech_s <= 0:
        return None
    visible = sum(1 for ch in ref_text if not ch.isspace())
    return visible / speech_s


def _coverage_range(language: Optional[str]) -> tuple[float, float]:
    lang = (language or "Chinese").lower()
    if lang.startswith("en"):
        return (6.0, 24.0)
    if lang.startswith(("zh", "chinese")):
        return (2.0, 9.0)
    return (2.0, 24.0)


def check_samples(
    samples: np.ndarray,
    sr: int,
    *,
    channels: int = 1,
    ref_text: Optional[str] = None,
    language: Optional[str] = None,
    fmt: Optional[str] = None,
) -> AdmissionReport:
    report = AdmissionReport()
    x = np.asarray(samples, dtype=np.float32)
    total_s = duration_s(x, sr)

    # Duration
    if total_s < 2.0:
        report.add(Check("duration_s", round(total_s, 3), ERROR, "record at least 3 seconds of continuous speech"))
    elif total_s < 3.0:
        report.add(Check("duration_s", round(total_s, 3), WARN, "3-30 seconds is recommended for stable cloning"))
    elif total_s > 60.0:
        report.add(Check("duration_s", round(total_s, 3), WARN, "trim to 3-30 seconds; long clips add little quality"))
    else:
        report.add(Check("duration_s", round(total_s, 3), OK))

    # Sample rate
    if sr < 16000:
        report.add(Check("sample_rate", sr, WARN, "use a 16 kHz or 24 kHz source; low rates hurt clone quality"))
    else:
        report.add(Check("sample_rate", sr, OK))

    # Channels
    if channels > 2:
        report.add(Check("channels", channels, WARN, "downmix to mono before cloning"))
    elif channels == 2:
        report.add(Check("channels", channels, OK, "stereo will be downmixed to mono"))
    else:
        report.add(Check("channels", channels, OK))

    if x.size == 0:
        report.add(Check("signal", 0, ERROR, "audio contains no samples"))
        return report

    # Clipping
    clipped = float(np.mean(np.abs(x) >= 0.999))
    if clipped > 0.01:
        report.add(Check("clipping_ratio", round(clipped, 5), ERROR, "re-record or lower the input gain"))
    elif clipped > 0.001:
        report.add(Check("clipping_ratio", round(clipped, 5), WARN, "light clipping detected; normalize or re-record"))
    else:
        report.add(Check("clipping_ratio", round(clipped, 5), OK))

    # Silence / speech activity, using 20 ms frames.
    frames = frame_rms(x, sr, frame_ms=20.0)
    if frames.size:
        silence_ratio = float(np.mean(frames < -50.0))
        speech_ratio = float(np.mean(frames > -35.0))
        noise_floor = float(np.percentile(frames, 10))
        peak = peak_dbfs(x)
    else:
        silence_ratio, speech_ratio, noise_floor, peak = 1.0, 0.0, -120.0, -120.0

    if silence_ratio > 0.6:
        report.add(Check("silence_ratio", round(silence_ratio, 4), WARN, "trim leading/trailing silence and long pauses"))
    else:
        report.add(Check("silence_ratio", round(silence_ratio, 4), OK))

    if speech_ratio < 0.3:
        report.add(Check("speech_ratio", round(speech_ratio, 4), ERROR, "too little speech; record a continuous clean passage"))
    elif speech_ratio < 0.5:
        report.add(Check("speech_ratio", round(speech_ratio, 4), WARN, "expected more continuous speech; consider trimming"))
    else:
        report.add(Check("speech_ratio", round(speech_ratio, 4), OK))

    if noise_floor > -35.0:
        report.add(Check("noise_floor_dbfs", round(noise_floor, 2), WARN, "high noise floor; record in a quieter room or denoise"))
    elif noise_floor > -45.0:
        report.add(Check("noise_floor_dbfs", round(noise_floor, 2), WARN, "audible background noise; denoise before cloning"))
    else:
        report.add(Check("noise_floor_dbfs", round(noise_floor, 2), OK))

    if peak < -30.0:
        report.add(Check("peak_dbfs", round(peak, 2), WARN, "level is very low; increase gain without clipping"))
    else:
        report.add(Check("peak_dbfs", round(peak, 2), OK))

    dc = float(np.mean(x))
    if abs(dc) > 0.02:
        report.add(Check("dc_offset", round(dc, 5), WARN, "apply a high-pass/DC filter"))
    else:
        report.add(Check("dc_offset", round(dc, 5), OK))

    # Text/speech coverage: catches transcripts that do not match the audio.
    rate = _coverage_rate(ref_text, total_s, language)
    if rate is not None:
        low, high = _coverage_range(language)
        if not (low <= rate <= high):
            report.add(
                Check(
                    "ref_text_rate_chars_per_s",
                    round(rate, 2),
                    WARN,
                    f"transcript looks mismatched with the audio (expected {low}-{high} chars/s); "
                    "check --ref-text or use --mode xvec",
                )
            )
        else:
            report.add(Check("ref_text_rate_chars_per_s", round(rate, 2), OK))

    if fmt:
        report.add(Check("format", fmt, OK))
    return report


def check_ref_audio(
    path: str | Path,
    *,
    ref_text: Optional[str] = None,
    language: Optional[str] = None,
) -> AdmissionReport:
    info = audio_info(path)
    samples, sr = read_audio(path)
    report = check_samples(
        samples,
        sr,
        channels=int(info["channels"]),
        ref_text=ref_text,
        language=language,
        fmt=str(info.get("format") or ""),
    )
    report.path = str(Path(path).expanduser())
    return report
