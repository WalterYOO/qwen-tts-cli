"""Lazy Qwen3-ForcedAligner wrapper (qwen-asr)."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .catalog import ModelInfo
from .errors import InputError, ModelError
from .subtitles import AlignItem
from .textnorm import detect_language
from .utils import StdoutMuted, stdout_to_stderr

# Hard limit from the aligner model card (PRD §3, §6.3).
MAX_ALIGN_SECONDS = 300.0

_ALIGNER_CACHE: dict[tuple[str, str, str], Any] = {}


def _runtime():
    try:
        with StdoutMuted():
            import torch  # noqa: F401
            from qwen_asr import Qwen3ForcedAligner
    except Exception as exc:  # pragma: no cover - environment dependent
        raise ModelError(
            "qwen-asr is not available in this Python environment; install the 'align' extra "
            "(qwen-asr + torch) or point QTTS_ASR_PYTHON at an interpreter that has it",
            details={"error": f"{type(exc).__name__}: {exc}"},
        ) from exc
    return torch, Qwen3ForcedAligner


def resolve_align_language(text: str, language: Optional[str]) -> str:
    lang = (language or "auto").strip()
    if lang.lower() == "auto":
        lang = detect_language(text)
    if lang == "Auto":
        lang = "Chinese" if any("一" <= ch <= "鿿" for ch in text) else "English"
    return lang


class AlignerEngine:
    def __init__(self, cfg, model: ModelInfo) -> None:
        self.cfg = cfg
        self.model_info = model
        self.model: Any = None
        self.device: Optional[str] = None
        self.dtype_name: Optional[str] = None
        self.load_ms: Optional[float] = None

    def load(self) -> "AlignerEngine":
        if self.model is not None:
            return self
        torch, Qwen3ForcedAligner = _runtime()
        self.device = self.cfg.resolve_device()
        torch_dtype = self.cfg.torch_dtype(self.device)
        self.dtype_name = {
            torch.bfloat16: "bf16",
            torch.float16: "fp16",
            torch.float32: "fp32",
        }[torch_dtype]
        key = (str(Path(self.model_info.path)), self.device, self.dtype_name)
        cached = _ALIGNER_CACHE.get(key)
        if cached is not None:
            self.model = cached
            self.load_ms = 0.0
            return self
        if self.cfg.offline:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        extra = {"local_files_only": True} if self.cfg.offline else {}
        started = time.perf_counter()
        try:
            with stdout_to_stderr():
                try:
                    model = Qwen3ForcedAligner.from_pretrained(
                        str(self.model_info.path),
                        device_map=self.device,
                        dtype=torch_dtype,
                        **extra,
                    )
                except TypeError as exc:
                    if extra and "local_files_only" in str(exc):
                        model = Qwen3ForcedAligner.from_pretrained(
                            str(self.model_info.path),
                            device_map=self.device,
                            dtype=torch_dtype,
                        )
                    else:
                        raise
        except Exception as exc:
            raise ModelError(
                f"failed to load forced aligner {self.model_info.name}: {exc}",
                details={"model": self.model_info.name, "path": str(self.model_info.path)},
            ) from exc
        self.load_ms = (time.perf_counter() - started) * 1000.0
        self.model = model
        _ALIGNER_CACHE[key] = model
        return self

    def align(
        self,
        samples: np.ndarray,
        sr: int,
        text: str,
        language: Optional[str] = None,
    ) -> tuple[list[AlignItem], dict[str, Any]]:
        duration = len(samples) / float(sr) if sr else 0.0
        if duration <= 0:
            raise InputError("cannot align an empty audio file")
        if duration > MAX_ALIGN_SECONDS:
            raise InputError(
                f"audio is {duration:.1f}s; the forced aligner supports at most "
                f"{MAX_ALIGN_SECONDS:.0f}s per sample in P0 (long-audio windowing is P1)",
                details={"duration_s": round(duration, 3), "limit_s": MAX_ALIGN_SECONDS},
            )
        lang = resolve_align_language(text, language)
        self.load()
        started = time.perf_counter()
        try:
            with stdout_to_stderr():
                results = self.model.align((np.asarray(samples, dtype=np.float32), int(sr)), text, lang)
        except Exception as exc:
            raise ModelError(f"forced alignment failed: {exc}", details={"language": lang}) from exc
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if not results:
            raise ModelError("aligner returned no result")
        items = [AlignItem(str(it.text), float(it.start_time), float(it.end_time)) for it in results[0]]
        info = {
            "align_ms": round(elapsed_ms, 1),
            "language": lang,
            "items": len(items),
            "duration_s": round(duration, 3),
        }
        return items, info
