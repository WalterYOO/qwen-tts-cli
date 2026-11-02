"""Lazy Qwen3-TTS runtime wrapper.

Importing this module is cheap and side-effect free. ``torch``/``qwen_tts`` are
imported only inside :meth:`TtsEngine.load`, which is never reached by
``--dry-run``.
"""

from __future__ import annotations

import gc
import os
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .audio_io import read_audio
from .catalog import ModelInfo
from .errors import ModelError
from .utils import StdoutMuted, stdout_to_stderr

_MODEL_CACHE: dict[tuple[str, str, str], Any] = {}


def _runtime():
    try:
        with StdoutMuted():
            import torch  # noqa: F401
            from qwen_tts import Qwen3TTSModel, VoiceClonePromptItem
    except Exception as exc:  # pragma: no cover - depends on environment
        raise ModelError(
            "qwen-tts runtime is not available in this Python environment; "
            "install the 'tts' extra (qwen-tts + torch)",
            details={"error": f"{type(exc).__name__}: {exc}"},
        ) from exc
    return torch, Qwen3TTSModel, VoiceClonePromptItem


def clear_model_cache() -> None:
    _MODEL_CACHE.clear()
    try:
        import torch

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()


class TtsEngine:
    def __init__(self, cfg, model: ModelInfo) -> None:
        self.cfg = cfg
        self.model_info = model
        self.device: Optional[str] = None
        self.dtype_name: Optional[str] = None
        self.model: Any = None
        self.load_ms: Optional[float] = None

    @property
    def cache_key(self) -> tuple[str, str, str]:
        device = self.device or self.cfg.device
        dtype = self.dtype_name or self.cfg.dtype
        return (str(Path(self.model_info.path)), device, dtype)

    def load(self) -> "TtsEngine":
        if self.model is not None:
            return self
        torch, Qwen3TTSModel, _ = _runtime()
        self.device = self.cfg.resolve_device()
        torch_dtype = self.cfg.torch_dtype(self.device)
        self.dtype_name = {
            torch.bfloat16: "bf16",
            torch.float16: "fp16",
            torch.float32: "fp32",
        }[torch_dtype]
        key = self.cache_key
        cached = _MODEL_CACHE.get(key)
        if cached is not None:
            self.model = cached
            self.load_ms = 0.0
            return self
        # Free other models first: 1.7B + 0.6B can exceed MPS memory.
        if _MODEL_CACHE:
            clear_model_cache()
        # --offline must reach the loader, not just block `models pull`
        # (codex P1-1).
        if self.cfg.offline:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        extra = {"local_files_only": True} if self.cfg.offline else {}
        started = time.perf_counter()
        try:
            with stdout_to_stderr():
                try:
                    model = Qwen3TTSModel.from_pretrained(
                        str(self.model_info.path),
                        device_map=self.device,
                        dtype=torch_dtype,
                        **extra,
                    )
                except TypeError as exc:
                    if extra and "local_files_only" in str(exc):
                        model = Qwen3TTSModel.from_pretrained(
                            str(self.model_info.path),
                            device_map=self.device,
                            dtype=torch_dtype,
                        )
                    else:
                        raise
        except Exception as exc:
            raise ModelError(
                f"failed to load TTS model {self.model_info.name}: {exc}",
                details={"model": self.model_info.name, "path": str(self.model_info.path)},
            ) from exc
        self.load_ms = (time.perf_counter() - started) * 1000.0
        self.model = model
        _MODEL_CACHE[key] = model
        return self

    # -- prompts ------------------------------------------------------------ #
    def build_clone_prompt_items(self, store, voice_name: str) -> list[Any]:
        torch, _, VoiceClonePromptItem = _runtime()  # noqa: F841
        meta = store.require(voice_name)
        tensors = store.load_prompt_tensors(voice_name)
        return [
            VoiceClonePromptItem(
                ref_code=tensors.get("ref_code"),
                ref_spk_embedding=tensors["ref_spk_embedding"],
                x_vector_only_mode=bool(tensors.get("x_vector_only_mode", False)),
                icl_mode=bool(tensors.get("icl_mode", True)),
                ref_text=(meta.get("source") or {}).get("ref_text"),
            )
        ]

    def create_clone_prompt_items(
        self,
        ref_audio: str | Path,
        *,
        ref_text: Optional[str] = None,
        x_vector_only_mode: bool = False,
    ) -> list[Any]:
        self.load()
        samples, sr = read_audio(ref_audio)
        try:
            with stdout_to_stderr():
                return self.model.create_voice_clone_prompt(
                    ref_audio=(samples, sr),
                    ref_text=ref_text,
                    x_vector_only_mode=x_vector_only_mode,
                )
        except Exception as exc:
            raise ModelError(f"failed to build clone prompt: {exc}") from exc

    # -- generation --------------------------------------------------------- #
    def _seed(self, seed: int) -> None:
        torch, _, _ = _runtime()
        torch.manual_seed(int(seed))
        try:
            if self.device and self.device.startswith("cuda"):
                torch.cuda.manual_seed_all(int(seed))
            if self.device == "mps":
                torch.mps.manual_seed(int(seed))
        except Exception:
            pass

    def generate(
        self,
        *,
        text: str,
        language: str,
        mode: str,
        speaker: Optional[str] = None,
        instruct: Optional[str] = None,
        prompt_items: Optional[list[Any]] = None,
        sampling: Optional[dict[str, Any]] = None,
        seed: int = 1234,
    ) -> tuple[np.ndarray, int, dict[str, Any]]:
        self.load()
        sampling = dict(sampling or {})
        self._seed(seed)
        started = time.perf_counter()
        kwargs = {
            "do_sample": sampling.get("do_sample", True),
            "top_k": sampling.get("top_k", 50),
            "top_p": sampling.get("top_p", 1.0),
            "temperature": sampling.get("temperature", 0.9),
            "repetition_penalty": sampling.get("repetition_penalty", 1.05),
            "max_new_tokens": int(sampling.get("max_new_tokens", 2048)),
        }
        try:
            with stdout_to_stderr():
                if mode == "builtin":
                    wavs, sr = self.model.generate_custom_voice(
                        text=text,
                        language=language,
                        speaker=speaker,
                        instruct=instruct or None,
                        **kwargs,
                    )
                elif mode == "design":
                    wavs, sr = self.model.generate_voice_design(
                        text=text,
                        language=language,
                        instruct=instruct or "",
                        **kwargs,
                    )
                elif mode == "clone":
                    if not prompt_items:
                        raise ModelError("clone mode requires a voice prompt")
                    wavs, sr = self.model.generate_voice_clone(
                        text=text,
                        language=language,
                        voice_clone_prompt=prompt_items,
                        **kwargs,
                    )
                else:
                    raise ModelError(f"unknown generation mode: {mode}")
        except ModelError:
            raise
        except Exception as exc:
            raise ModelError(f"generation failed: {exc}", details={"mode": mode}) from exc
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if not wavs:
            raise ModelError("model returned no audio")
        audio = np.asarray(wavs[0], dtype=np.float32).reshape(-1)
        info = {
            "generate_ms": round(elapsed_ms, 1),
            "sample_rate": int(sr),
            "duration_s": round(len(audio) / float(sr), 3),
            "mode": mode,
            "language": language,
        }
        return audio, int(sr), info
