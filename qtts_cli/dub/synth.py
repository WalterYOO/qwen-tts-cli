"""Synthesis adapter, cue cache, fingerprints and duration fit (FR-P2-04 §4)."""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from .. import audio_io
from ..audiobook import audio as audio_mod
from ..audiobook.segments import (
    SaySegmentRunner,
    build_fingerprint_context,
    effective_runtime,
    fingerprint,
    load_models,
    resolve_chapter_voice,
    runtime_is_resolved,
)
from ..backends.registry import preflight_voice_ref
from ..errors import InputError, ModelError
from ..utils import atomic_write_json, sha256_file
from ..voice_store import VoiceStore

EPS = 0.005
MAX_MASTER_BYTES = 1 << 30


def resolve_voice_ref(
    *,
    models: list[Any],
    store: VoiceStore,
    ref: str,
    spec: Any,
    model_override: Optional[str],
) -> Any:
    """Preflight then resolve exactly as audiobook does (syntax errors are 4)."""
    preflight_voice_ref(ref, spec)
    return resolve_chapter_voice(
        models=models,
        store=store,
        ref=ref,
        spec=spec,
        model_override=model_override,
    )


def build_context(*, cfg, spec, store, resolution, device, dtype, seed):
    return build_fingerprint_context(
        cfg=cfg,
        spec=spec,
        store=store,
        resolution=resolution,
        device=device,
        dtype=dtype,
        seed=seed,
    )


# --------------------------------------------------------------------------- #
# Cue cache
# --------------------------------------------------------------------------- #


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def valid_sidecar(sidecar: dict[str, Any], expected: str) -> bool:
    if not isinstance(sidecar.get("fingerprint"), str) or sidecar["fingerprint"] != expected:
        return False
    if not _positive_int(sidecar.get("sample_rate")) or not _positive_int(
        sidecar.get("frames")
    ):
        return False
    channels = sidecar.get("channels")
    if not _positive_int(channels) or channels != 1:
        return False
    if not isinstance(sidecar.get("sha256"), str) or not _finite_number(
        sidecar.get("duration_s")
    ):
        return False
    languages = sidecar.get("languages_actual")
    if not isinstance(languages, list) or not all(
        isinstance(value, str) for value in languages
    ):
        return False
    return True


class CueCache:
    def __init__(self, root: Path, *, enabled: bool) -> None:
        self.root = Path(root)
        self.enabled = bool(enabled)

    def paths(self, index: int) -> tuple[Path, Path]:
        stem = f"{int(index):04d}"
        return self.root / f"{stem}.wav", self.root / f"{stem}.json"

    def lookup(
        self, index: int, expected: str
    ) -> Optional[tuple[np.ndarray, int, list[str]]]:
        if not self.enabled:
            return None
        wav_path, sidecar_path = self.paths(index)
        if not wav_path.is_file() or not sidecar_path.is_file():
            return None
        try:
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, UnicodeDecodeError):
            return None
        if not isinstance(sidecar, dict) or not valid_sidecar(sidecar, expected):
            return None
        try:
            info = audio_io.audio_info(wav_path)
            if (
                int(info.get("channels", 0)) != 1
                or int(info.get("frames", 0)) <= 0
                or int(info.get("sample_rate", 0)) != sidecar["sample_rate"]
                or int(info.get("frames", -1)) != sidecar["frames"]
            ):
                return None
            digest = sha256_file(wav_path)
            data, sr = audio_io.read_audio(wav_path, mono=True)
        except (OSError, TypeError, ValueError, OverflowError, InputError):
            return None
        if digest != sidecar.get("sha256"):
            return None
        if data.size == 0 or int(sr) != sidecar["sample_rate"]:
            return None
        if not np.all(np.isfinite(data)):
            return None
        languages = [str(value) for value in sidecar.get("languages_actual") or []]
        return data, int(sr), languages

    def write(
        self,
        index: int,
        *,
        fingerprint_value: str,
        sample_rate: int,
        frames: int,
        duration_s: float,
        sha256: str,
        device_actual: Any,
        dtype_actual: Any,
        languages_actual: list[str],
    ) -> None:
        _, sidecar_path = self.paths(index)
        try:
            atomic_write_json(
                sidecar_path,
                {
                    "fingerprint": fingerprint_value,
                    "sample_rate": int(sample_rate),
                    "channels": 1,
                    "frames": int(frames),
                    "duration_s": float(duration_s),
                    "sha256": sha256,
                    "device_actual": device_actual,
                    "dtype_actual": dtype_actual,
                    "languages_actual": sorted(set(languages_actual)),
                },
            )
        except OSError as exc:
            raise ModelError(
                f"cannot write the cue cache sidecar {sidecar_path}: {exc}",
                details={"path": str(sidecar_path)},
            ) from exc


# --------------------------------------------------------------------------- #
# say.run adapter (same shape as audiobook)
# --------------------------------------------------------------------------- #


def synthesize_cue(
    runner: Any,
    *,
    text: str,
    out: Path,
    voice: str,
    language: str,
    instruct: Optional[str],
    seed: int,
    cancel_check: Callable[[], bool],
    recipe_dir: Path,
    index: int,
) -> dict[str, Any]:
    kwargs = dict(
        text=text,
        out=out,
        voice=voice,
        language=language,
        instruct=instruct,
        seed=seed,
        cancel_check=cancel_check,
    )
    if isinstance(runner, SaySegmentRunner):
        kwargs["recipe_out"] = recipe_dir / f"{int(index):04d}.json"
    return runner.synthesize(**kwargs)


def validate_fresh_audio(path: Path) -> tuple[np.ndarray, int]:
    try:
        data, sr = audio_io.read_audio(path, mono=True)
    except InputError as exc:
        raise ModelError(
            f"fresh synthesis output is unreadable: {path}: {exc.message}",
            details={"path": str(path)},
        ) from exc
    if data.size == 0 or int(sr) <= 0 or not np.all(np.isfinite(data)):
        raise ModelError(
            f"fresh synthesis output is invalid: {path}", details={"path": str(path)}
        )
    return data, int(sr)


# --------------------------------------------------------------------------- #
# Duration fit / atempo
# --------------------------------------------------------------------------- #


def _require_stretch_ffmpeg() -> str:
    executable = shutil.which("ffmpeg")
    if executable is None:
        raise ModelError(
            "ffmpeg is required for time-stretching but was not found on PATH"
        )
    return executable


def stretch_file(source: Path, destination: Path, speed: float) -> None:
    executable = _require_stretch_ffmpeg()
    command = [
        executable,
        "-v",
        "error",
        "-y",
        "-i",
        str(source),
        "-filter:a",
        f"atempo={float(speed):.9f}",
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        str(destination),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise ModelError(
            f"ffmpeg time-stretch failed: {completed.stderr.strip()}",
            details={"path": str(source), "speed": float(speed)},
        )
    if not destination.is_file() or destination.stat().st_size == 0:
        raise ModelError(f"ffmpeg produced no stretched audio for {source}")


@dataclass
class RenderFit:
    data: np.ndarray
    source_frames: int
    source_rate: int
    speed: float
    rendered_frames: int
    synth_duration_s: float
    overflow: bool


def fit_duration(
    *,
    source: np.ndarray,
    source_path: Path,
    source_rate: int,
    target_frames: int,
    master_rate: int,
    min_speed: float,
    max_speed: float,
    stretch_dir: Path,
    index: int,
) -> RenderFit:
    """Apply the §4 duration-fit algorithm, returning master-rate audio."""
    source_frames = int(source.shape[0])
    base_frames = audio_mod.resampled_length(source_frames, int(source_rate), int(master_rate))
    ratio = float(base_frames) / float(target_frames)
    speed = min(max(ratio, float(min_speed)), float(max_speed))
    stretched = abs(speed - 1.0) > EPS
    if stretched:
        stretch_path = Path(stretch_dir) / f"{int(index):04d}.{uuid.uuid4().hex}.wav"
        stretch_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            stretch_file(source_path, stretch_path, speed)
            rendered_source, rendered_rate = audio_io.read_audio(stretch_path, mono=True)
        except InputError as exc:
            raise ModelError(
                f"cannot read stretched cue {index}: {exc.message}",
                details={"path": str(stretch_path)},
            ) from exc
        finally:
            try:
                os.unlink(stretch_path)
            except OSError:
                pass
        if int(rendered_rate) != int(source_rate):
            raise ModelError(
                f"time-stretch changed the sample rate for cue {index} "
                f"({source_rate} -> {rendered_rate})",
                details={"path": str(source_path)},
            )
    else:
        rendered_source = source
    data = audio_mod.resample_segment(rendered_source, int(source_rate), int(master_rate))
    actual = int(data.shape[0])
    if actual < int(target_frames):
        data = np.pad(data, (0, int(target_frames) - actual))
        actual = int(target_frames)
    tolerance = max(int(math.ceil(EPS * float(target_frames))), 1)
    overflow = (actual - int(target_frames)) > tolerance
    return RenderFit(
        data=np.ascontiguousarray(data, dtype=np.float32),
        source_frames=source_frames,
        source_rate=int(source_rate),
        speed=float(speed),
        rendered_frames=actual,
        synth_duration_s=source_frames / float(source_rate),
        overflow=bool(overflow),
    )


def write_rendered(path: Path, data: np.ndarray, sr: int) -> None:
    import soundfile as sf

    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(path), np.asarray(data, dtype=np.float32), int(sr), subtype="FLOAT")
    except (RuntimeError, OSError) as exc:
        raise ModelError(
            f"cannot write rendered cue {path}: {exc}", details={"path": str(path)}
        ) from exc
    except Exception as exc:
        if isinstance(exc, sf.SoundFileError):
            raise ModelError(
                f"cannot write rendered cue {path}: {exc}", details={"path": str(path)}
            ) from exc
        raise


def require_ffmpeg_probe() -> str:
    probe = shutil.which("ffprobe")
    if probe is None:
        raise ModelError("ffprobe is required but was not found on PATH")
    return probe


__all__ = [
    "CueCache",
    "RenderFit",
    "SaySegmentRunner",
    "build_context",
    "effective_runtime",
    "fingerprint",
    "fit_duration",
    "load_models",
    "require_ffmpeg_probe",
    "resolve_voice_ref",
    "runtime_is_resolved",
    "synthesize_cue",
    "validate_fresh_audio",
    "write_rendered",
]
