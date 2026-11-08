"""Synthesis adapter, segment fingerprints and the segment cache (FR-P2-03 §3)."""

from __future__ import annotations

import dataclasses
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Optional

import numpy as np

from .. import audio_io
from ..catalog import scan
from ..errors import InputError, ModelError
from ..plan import SAMPLING_DEFAULTS, VoiceResolution, resolve_voice
from ..textnorm import NORMALIZATION_VERSION
from ..utils import atomic_write_json, sha256_file, sha256_text
from ..voice_store import VoiceStore

_DTYPE_SPELLINGS = {
    "float16": "fp16",
    "fp16": "fp16",
    "bfloat16": "bf16",
    "bf16": "bf16",
    "float32": "fp32",
    "fp32": "fp32",
}


def normalize_dtype_name(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    return _DTYPE_SPELLINGS.get(str(value), str(value))


def effective_runtime(cfg, spec: Any) -> tuple[str, Optional[str]]:
    """Resolve the effective device/dtype once, without loading weights."""
    if getattr(spec, "remote", False):
        return "remote", None
    from ..plan import resolve_dtype_name

    device = cfg.resolve_device()
    dtype = normalize_dtype_name(resolve_dtype_name(dataclasses.replace(cfg, device=device)))
    return device, dtype


def runtime_is_resolved(cfg, spec: Any) -> bool:
    """True when device/dtype can be resolved without importing torch."""
    if getattr(spec, "remote", False):
        return True
    return str(getattr(cfg, "device", "auto")) != "auto"


def model_file_list(model_path: Path) -> list[dict[str, Any]]:
    root = Path(model_path)
    entries: list[dict[str, Any]] = []
    try:
        for current, dirs, names in os.walk(root):
            dirs.sort()
            for name in sorted(names):
                path = Path(current) / name
                try:
                    stat = path.stat()
                except OSError:
                    continue
                if not path.is_file():
                    continue
                entries.append(
                    {
                        "path": path.relative_to(root).as_posix(),
                        "size": int(stat.st_size),
                        "mtime_ns": int(stat.st_mtime_ns),
                    }
                )
    except OSError as exc:
        raise InputError(f"cannot inspect model directory {root}: {exc}") from exc
    entries.sort(key=lambda item: item["path"])
    return entries


@dataclass
class FingerprintContext:
    spec: Any
    resolution: VoiceResolution
    voice_ref_audio_sha256: Optional[str]
    model_files: list[dict[str, Any]]
    device: str
    dtype: Optional[str]
    sampling: dict[str, Any]
    seed: int


def build_fingerprint_context(
    *,
    cfg,
    spec: Any,
    store: VoiceStore,
    resolution: VoiceResolution,
    device: str,
    dtype: Optional[str],
    seed: int,
) -> FingerprintContext:
    remote = bool(getattr(spec, "remote", False))
    model = resolution.model
    ref_audio = None
    if resolution.mode == "clone" and resolution.voice_name:
        ref = store.dir_for(resolution.voice_name) / "ref.wav"
        if ref.is_file():
            ref_audio = sha256_file(ref)
    sampling = dict(SAMPLING_DEFAULTS)
    from ..budget import DEFAULT_MAX_NEW_TOKENS

    sampling["max_new_tokens"] = int(DEFAULT_MAX_NEW_TOKENS)
    return FingerprintContext(
        spec=spec,
        resolution=resolution,
        voice_ref_audio_sha256=ref_audio,
        model_files=[] if remote else model_file_list(Path(model.path)),
        device=device,
        dtype=dtype,
        sampling=sampling,
        seed=int(seed),
    )


def fingerprint_payload(
    context: FingerprintContext,
    *,
    text: str,
    language: str,
    instruct: Optional[str],
) -> dict[str, Any]:
    spec = context.spec
    resolution = context.resolution
    model = resolution.model
    remote = bool(getattr(spec, "remote", False))
    return {
        "schema": 1,
        "text": text,
        "voice_ref": resolution.ref,
        "voice_identity": resolution.identity,
        "voice_ref_audio_sha256": context.voice_ref_audio_sha256,
        "model_id": model.model_id,
        "model_revision": model.revision,
        "model_fingerprint": model.fingerprint,
        "model_root": None if remote else str(Path(model.path).resolve()),
        "model_files": context.model_files,
        "backend_name": spec.name,
        "backend_identity": spec.identity,
        "device": context.device,
        "dtype": context.dtype,
        "language": language,
        "instruct": instruct,
        "sampling": context.sampling,
        "seed": context.seed,
        "normalization_version": NORMALIZATION_VERSION,
    }


def fingerprint(context: FingerprintContext, *, text: str, language: str, instruct: Optional[str]) -> str:
    payload = fingerprint_payload(context, text=text, language=language, instruct=instruct)
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "sha256:" + sha256_text(canonical)


def resolve_chapter_voice(
    *,
    models: list[Any],
    store: VoiceStore,
    ref: str,
    spec: Any,
    model_override: Optional[str],
) -> VoiceResolution:
    from ..capabilities import parse_voice_ref

    return resolve_voice(
        models,
        store,
        parse_voice_ref(ref),
        model_override=model_override,
        backend_identity=spec.identity,
        require_clone_prompt=bool(getattr(spec, "require_clone_prompt", True)),
        speaker_policy=spec.speaker_policy() if hasattr(spec, "speaker_policy") else "strict",
    )


def load_models(cfg, spec: Any) -> list[Any]:
    models = list(spec.models)
    if models:
        return models
    return scan(cfg.model_dir)


# --------------------------------------------------------------------------- #
# Segment cache
# --------------------------------------------------------------------------- #


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _valid_sidecar(sidecar: dict[str, Any], expected: str) -> bool:
    """Strict sidecar schema; any type damage is a cache miss, never an error."""
    if not isinstance(sidecar.get("fingerprint"), str) or sidecar["fingerprint"] != expected:
        return False
    if not _positive_int(sidecar.get("sample_rate")) or not _positive_int(sidecar.get("frames")):
        return False
    channels = sidecar.get("channels")
    if not _positive_int(channels) or channels != 1:
        return False
    if not isinstance(sidecar.get("sha256"), str) or not _finite_number(sidecar.get("duration_s")):
        return False
    languages = sidecar.get("languages_actual")
    if not isinstance(languages, list) or not all(isinstance(value, str) for value in languages):
        return False
    return True


class SegmentCache:
    def __init__(self, root: Path, *, enabled: bool) -> None:
        self.root = Path(root)
        self.enabled = bool(enabled)

    def paths(self, chapter: int, paragraph: int) -> tuple[Path, Path]:
        stem = f"{int(chapter):04d}-{int(paragraph):04d}"
        return self.root / f"{stem}.wav", self.root / f"{stem}.json"

    def lookup(self, chapter: int, paragraph: int, expected: str) -> Optional[tuple[np.ndarray, int]]:
        if not self.enabled:
            return None
        wav_path, sidecar_path = self.paths(chapter, paragraph)
        if not wav_path.is_file() or not sidecar_path.is_file():
            return None
        try:
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, UnicodeDecodeError):
            return None
        if not isinstance(sidecar, dict):
            return None
        try:
            if not _valid_sidecar(sidecar, expected):
                return None
            info = audio_io.audio_info(wav_path)
        except (TypeError, ValueError, OverflowError):
            return None
        except InputError:
            return None
        try:
            if (
                int(info.get("channels", 0)) != 1
                or int(info.get("frames", 0)) <= 0
                or int(info.get("sample_rate", 0)) != sidecar["sample_rate"]
                or int(info.get("frames", -1)) != sidecar["frames"]
            ):
                return None
        except (TypeError, ValueError, OverflowError):
            return None
        try:
            digest = sha256_file(wav_path)
        except OSError:
            return None
        if digest != sidecar.get("sha256"):
            return None
        try:
            data, sr = audio_io.read_audio(wav_path, mono=True)
        except InputError:
            return None
        if data.size == 0:
            return None
        return data, int(sr)

    def write_sidecar(
        self,
        chapter: int,
        paragraph: int,
        *,
        fingerprint: str,
        sample_rate: int,
        frames: int,
        duration_s: float,
        sha256: str,
        device_actual: Any,
        dtype_actual: Any,
        languages_actual: list[str],
    ) -> None:
        _, sidecar_path = self.paths(chapter, paragraph)
        atomic_write_json(
            sidecar_path,
            {
                "fingerprint": fingerprint,
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


# --------------------------------------------------------------------------- #
# say.run adapter
# --------------------------------------------------------------------------- #


class SaySegmentRunner:
    """Default runner: one ``commands.say.run`` call per paragraph."""

    def __init__(self, cfg, args, recipe_dir: Path) -> None:
        self.cfg = cfg
        self.args = args
        self.recipe_dir = Path(recipe_dir)
        import threading

        self.engine_cache: dict[str, Any] = {}
        self.aligner_cache: dict[str, Any] = {}
        self._cancel_event = threading.Event()

    def cancel_check(self) -> bool:
        return self._cancel_event.is_set()

    def request_cancel(self) -> None:
        self._cancel_event.set()

    def _say_args(
        self,
        *,
        text: str,
        out: Path,
        voice: str,
        language: str,
        instruct: Optional[str],
        seed: int,
        recipe_out: Path,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            text=text,
            stdin=False,
            jsonl=False,
            file=None,
            script=None,
            voice=voice,
            language=language,
            instruct=instruct,
            out=str(out),
            out_dir=None,
            format="wav",
            subtitle=[],
            subtitle_out_prefix=None,
            recipe_out=str(recipe_out),
            granularity="sentence",
            max_new_tokens=None,
            pause_ms=0,
            model=getattr(self.args, "model", None),
            backend=getattr(self.args, "backend", None) or getattr(self.cfg, "backend", None),
            backend_model=getattr(self.args, "backend_model", None)
            or getattr(self.cfg, "backend_model", None),
            lexicon=None,
            qc=False,
            reference_voice=None,
            takes=None,
            select=None,
            asr_model=None,
            max_cue_chars=42,
            max_line_width=42,
            max_cps=20.0,
            min_cue_seconds=0.8,
            max_cue_seconds=7.0,
            max_merge_gap_ms=300,
            no_subtitle_qc=False,
            max_clipping_ratio=None,
            target_lufs=None,
            lufs_tolerance=None,
            max_true_peak_dbtp=None,
            max_dc_offset=None,
            max_silence_ratio=None,
            min_speech_rate=None,
            max_speech_rate=None,
            dry_run=False,
            yes=True,
        )

    def synthesize(
        self,
        *,
        text: str,
        out: Path,
        voice: str,
        language: str,
        instruct: Optional[str],
        seed: int,
        cancel_check: Callable[[], bool],
        recipe_out: Optional[Path] = None,
    ) -> dict[str, Any]:
        from ..commands.say import run as say_run

        def check() -> bool:
            return bool(cancel_check()) or self.cancel_check()

        out = Path(out)
        recipe = Path(recipe_out) if recipe_out is not None else (
            self.recipe_dir / f"{out.stem}.json"
        )
        cfg = dataclasses.replace(self.cfg, no_cache=True, json=False, seed=int(seed))
        payload = say_run(
            cfg,
            self._say_args(
                text=text,
                out=out,
                voice=voice,
                language=language,
                instruct=instruct,
                seed=int(seed),
                recipe_out=recipe,
            ),
            engine_cache=self.engine_cache,
            aligner_cache=self.aligner_cache,
            cancel_check=check,
        )
        try:
            info = audio_io.audio_info(out)
        except InputError as exc:
            raise ModelError(
                f"say produced no readable audio for {out}: {exc.message}",
                details={"out": str(out)},
            ) from exc
        if int(info.get("frames", 0)) <= 0 or int(info.get("sample_rate", 0)) <= 0:
            raise ModelError(f"say produced invalid audio for {out}", details={"out": str(out)})
        languages = sorted(
            {
                str(segment.get("language"))
                for segment in payload.get("segments") or []
                if segment.get("language")
            }
        )
        return {
            "sample_rate": int(payload.get("sample_rate") or info["sample_rate"]),
            "duration_s": float(payload.get("duration_s") or info["duration_s"]),
            "dtype_actual": payload.get("dtype"),
            "device_actual": payload.get("device"),
            "languages_actual": languages,
        }


def synth_result_is_valid(data: np.ndarray, sr: int) -> bool:
    return bool(int(sr) > 0 and data.size > 0 and np.all(np.isfinite(data)))


def frames_after_resample(data: np.ndarray, sr: int, target_sr: int) -> int:
    if int(sr) == int(target_sr):
        return int(data.shape[0])
    return int(math.ceil(data.shape[0] * float(target_sr) / float(sr)))
