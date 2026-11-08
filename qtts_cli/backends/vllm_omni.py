"""vLLM-Omni offline backend (FR-P2-02).

Offline batch only.  ``vllm_omni`` is imported lazily inside the Omni factory;
module import, ``--dry-run`` and ``plugins list`` never touch it.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..catalog import ModelInfo
from ..errors import InputError, ModelError, QttsError
from ..voice_store import VoiceStore
from .base import BackendCapabilities

PROMPT_TOKEN_LENGTH = 2048
TASKS = {"builtin": "CustomVoice", "design": "VoiceDesign", "clone": "Base"}

CAPABILITIES = BackendCapabilities(
    modes=frozenset({"builtin", "clone", "design"}),
    instruct_modes=frozenset({"builtin", "design"}),
    sampling=frozenset({"max_new_tokens"}),
    remote=False,
    notes=("offline batch only", "Linux GPU", "requires vllm-omni"),
)


def _default_omni_factory(model: str, extra: dict[str, Any]):
    from vllm_omni import Omni

    return Omni(model=model, **extra)


# Tests replace this seam.
_omni_factory = _default_omni_factory


def _to_numpy(chunk: Any) -> np.ndarray:
    if hasattr(chunk, "detach"):
        try:
            chunk = chunk.detach().cpu().float()
        except Exception as exc:  # pragma: no cover - backend dependent
            raise ModelError(f"cannot move vllm-omni audio to CPU: {exc}") from exc
    try:
        return np.asarray(chunk, dtype=np.float32)
    except Exception as exc:
        raise ModelError(f"vllm-omni returned non-numeric audio: {exc}") from exc


def _mono_chunk(chunk: Any, index: int) -> np.ndarray:
    array = _to_numpy(chunk)
    if array.ndim == 1:
        mono = array
    elif array.ndim == 2:
        if array.shape[1] <= 2:
            mono = array.mean(axis=1)
        elif array.shape[0] <= 2:
            mono = array.mean(axis=0)
        else:
            raise ModelError(
                "vllm-omni returned an audio chunk with unsupported shape",
                details={"index": index, "shape": list(array.shape)},
            )
    else:
        raise ModelError(
            "vllm-omni returned an audio chunk with unsupported rank",
            details={"index": index, "shape": list(array.shape)},
        )
    mono = np.asarray(mono, dtype=np.float32).reshape(-1)
    if mono.size == 0:
        raise ModelError(
            "vllm-omni returned an empty audio chunk", details={"index": index}
        )
    if not np.all(np.isfinite(mono)):
        raise ModelError(
            "vllm-omni returned non-finite audio", details={"index": index}
        )
    return mono


def _sample_rate(mm: dict[str, Any]) -> int:
    value = mm.get("sr")
    if isinstance(value, (list, tuple)):
        if not value:
            raise ModelError("vllm-omni returned no sample rate")
        value = value[-1]
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ModelError(f"vllm-omni returned an invalid sample rate: {value!r}") from exc
    if not math.isfinite(number) or number <= 0 or not number.is_integer():
        raise ModelError(f"vllm-omni returned an invalid sample rate: {value!r}")
    return int(number)


def _ref_audio_path(store: VoiceStore, name: str) -> Path:
    meta = store.require(name)
    source = meta.get("source") or {}
    ref = source.get("ref_audio")
    if not ref:
        raise InputError(
            f"voice {name} has no reference audio for vllm-omni cloning",
            details={"voice": name},
        )
    base = store.dir_for(name).resolve()
    candidate = (base / str(ref)).resolve()
    if candidate != base and base not in candidate.parents:
        raise InputError(
            f"voice {name} reference audio escapes the voice directory",
            details={"voice": name, "path": str(candidate)},
        )
    if not candidate.is_file():
        raise InputError(
            f"voice {name} reference audio not found: {candidate}",
            details={"voice": name, "path": str(candidate)},
        )
    return candidate


def preflight(plan, cfg, args) -> None:
    """Validate every clone voice's ref audio before any create/load."""
    store = VoiceStore(cfg.data_dir)
    voices = [plan.voice, *[seg.voice for seg in plan.segments if seg.voice is not None]]
    seen: set[int] = set()
    for voice in voices:
        if id(voice) in seen:
            continue
        seen.add(id(voice))
        if voice.mode != "clone":
            continue
        if not voice.voice_name:
            raise InputError("clone voice is missing its profile name")
        _ref_audio_path(store, voice.voice_name)
        if voice.icl and not (voice.ref_text and voice.ref_text.strip()):
            raise InputError(
                f"ICL voice {voice.voice_name} requires ref_text for vllm-omni",
                details={"voice": voice.voice_name, "mode": "icl"},
            )


class VllmOmniBackend:
    name = "vllm-omni"
    capabilities = CAPABILITIES

    def __init__(self, cfg, model: ModelInfo) -> None:
        self.cfg = cfg
        self.model_info = model
        self.model: Any = None
        self.device: Optional[str] = None
        self.dtype_name: Optional[str] = None
        self.load_ms: Optional[float] = None

    def load(self) -> "VllmOmniBackend":
        if self.model is not None:
            return self
        extra: dict[str, Any] = {}
        started = time.perf_counter()
        try:
            self.model = _omni_factory(str(self.model_info.path), extra)
        except QttsError:
            raise
        except Exception as exc:
            raise ModelError(
                f"cannot create vllm-omni pipeline: {exc}; install vllm-omni "
                "(Linux GPU) or use --backend transformers",
                details={"model": self.model_info.name},
            ) from exc
        self.load_ms = (time.perf_counter() - started) * 1000.0
        return self

    def build_clone_prompt_items(self, store, voice_name: str) -> Any:
        meta = store.require(voice_name)
        source = meta.get("source") or {}
        mode = str(source.get("mode") or "icl").lower()
        return {
            "ref_audio": str(_ref_audio_path(store, voice_name)),
            "ref_text": source.get("ref_text"),
            "x_vector_only_mode": mode != "icl",
        }

    def _additional(
        self,
        *,
        text: str,
        language: str,
        mode: str,
        speaker: Optional[str],
        instruct: Optional[str],
        prompt_items: Any,
        max_new_tokens: int,
    ) -> dict[str, Any]:
        task = TASKS.get(mode)
        if task is None:
            raise ModelError(f"unknown generation mode: {mode}")
        additional: dict[str, Any] = {
            "task_type": [task],
            "text": [text],
            "language": [language],
            "max_new_tokens": [int(max_new_tokens)],
            "full_utterance_decode": [True],
        }
        if mode == "builtin":
            if not speaker:
                raise InputError("builtin generation requires a speaker")
            additional["speaker"] = [speaker]
            additional["instruct"] = [instruct or ""]
        elif mode == "design":
            additional["instruct"] = [instruct or ""]
            additional["non_streaming_mode"] = [True]
        else:
            if not isinstance(prompt_items, dict):
                raise ModelError("clone mode requires a vllm-omni ref audio prompt")
            additional["ref_audio"] = [str(prompt_items["ref_audio"])]
            additional["ref_text"] = [prompt_items.get("ref_text") or ""]
            additional["x_vector_only_mode"] = [
                bool(prompt_items.get("x_vector_only_mode", False))
            ]
        return additional

    def generate(
        self,
        *,
        text: str,
        language: str,
        mode: str,
        speaker: Optional[str] = None,
        instruct: Optional[str] = None,
        prompt_items: Any = None,
        sampling: Optional[dict[str, Any]] = None,
        seed: int = 1234,
    ) -> tuple[np.ndarray, int, dict[str, Any]]:
        self.load()
        sampling = dict(sampling or {})
        max_new_tokens = int(sampling.get("max_new_tokens", PROMPT_TOKEN_LENGTH))
        additional = self._additional(
            text=text,
            language=language,
            mode=mode,
            speaker=speaker,
            instruct=instruct,
            prompt_items=prompt_items,
            max_new_tokens=max_new_tokens,
        )
        inputs = {
            "prompt_token_ids": [0] * PROMPT_TOKEN_LENGTH,
            "additional_information": additional,
        }
        started = time.perf_counter()
        audio_chunks: list[np.ndarray] = []
        rate: Optional[int] = None
        try:
            outputs = self.model.generate([inputs])
            for stage_outputs in outputs:
                items = getattr(stage_outputs, "outputs", None)
                if not items:
                    raise ModelError("vllm-omni returned no outputs")
                mm = getattr(items[0], "multimodal_output", None)
                if not isinstance(mm, dict) or "audio" not in mm:
                    raise ModelError("vllm-omni output has no multimodal audio")
                chunk_rate = _sample_rate(mm)
                if rate is None:
                    rate = chunk_rate
                elif rate != chunk_rate:
                    raise ModelError(
                        "vllm-omni returned inconsistent sample rates",
                        details={"first": rate, "later": chunk_rate},
                    )
                raw = mm["audio"]
                chunks = list(raw) if isinstance(raw, (list, tuple)) else [raw]
                if not chunks:
                    raise ModelError("vllm-omni returned an empty audio list")
                audio_chunks.extend(
                    _mono_chunk(chunk, index) for index, chunk in enumerate(chunks)
                )
        except QttsError:
            raise
        except Exception as exc:
            raise ModelError(f"vllm-omni generation failed: {exc}") from exc
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if not audio_chunks or rate is None:
            raise ModelError("vllm-omni returned no audio")
        audio = np.ascontiguousarray(np.concatenate(audio_chunks), dtype=np.float32)
        if not np.all(np.isfinite(audio)):
            raise ModelError("vllm-omni returned non-finite audio")
        info = {
            "generate_ms": round(elapsed_ms, 1),
            "sample_rate": rate,
            "duration_s": round(len(audio) / float(rate), 3),
            "mode": mode,
            "language": language,
            "backend": self.name,
        }
        return audio, rate, info

    def release(self) -> None:
        self.model = None
