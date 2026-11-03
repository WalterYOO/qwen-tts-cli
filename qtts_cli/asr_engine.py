"""Lazy Qwen3-ASR wrapper (qwen-asr).

Mirrors :mod:`qtts_cli.align_engine`: importing this module never imports
torch/qwen_asr. ``--dry-run`` and unit tests can use the engine's output
normalisation without the runtime installed.
"""

from __future__ import annotations

import gc
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .catalog import ModelInfo
from .errors import InputError, ModelError, UsageError
from .utils import StdoutMuted, stdout_to_stderr

_ASR_CACHE: dict[tuple[Any, ...], Any] = {}
_ASR_LOCK = threading.RLock()

# Frozen Qwen3-ASR language table. Validating here keeps bad language input
# independent of model availability (t10 Wave 3a P1-1).
ASR_LANGUAGES = (
    "Chinese",
    "English",
    "Cantonese",
    "Arabic",
    "German",
    "French",
    "Spanish",
    "Portuguese",
    "Indonesian",
    "Italian",
    "Korean",
    "Russian",
    "Thai",
    "Vietnamese",
    "Japanese",
    "Turkish",
    "Hindi",
    "Malay",
    "Dutch",
    "Swedish",
    "Danish",
    "Finnish",
    "Polish",
    "Czech",
    "Filipino",
    "Persian",
    "Greek",
    "Romanian",
    "Hungarian",
    "Macedonian",
)
_ASR_LANGUAGE_MAP = {name.lower(): name for name in ASR_LANGUAGES}


def canonical_asr_language(language: Optional[str]) -> Optional[str]:
    """Map ``auto`` to ``None`` and canonicalise/validate an ASR language."""
    raw = (language or "").strip()
    if not raw or raw.lower() == "auto":
        return None
    canonical = _ASR_LANGUAGE_MAP.get(raw.lower())
    if canonical is None:
        raise UsageError(
            f"unsupported ASR language {raw!r}",
            details={"language": raw, "supported": list(ASR_LANGUAGES)},
        )
    return canonical


def clear_asr_cache() -> None:
    """Drop the process-level ASR cache and release accelerator memory."""
    with _ASR_LOCK:
        _ASR_CACHE.clear()
    torch = sys.modules.get("torch")
    if torch is None:
        return
    for backend_name in ("mps", "cuda"):
        try:
            backend = getattr(torch, backend_name, None)
            if backend is not None and backend.is_available():
                backend.empty_cache()
        except Exception:  # pragma: no cover - backend dependent
            continue


def _runtime():
    try:
        with StdoutMuted():
            import torch  # noqa: F401
            from qwen_asr import Qwen3ASRModel
    except Exception as exc:  # pragma: no cover - environment dependent
        raise ModelError(
            "qwen-asr is not available in this Python environment; install the 'align' extra "
            "(qwen-asr + torch) or point QTTS_ASR_PYTHON at an interpreter that has it",
            details={"error": f"{type(exc).__name__}: {exc}"},
        ) from exc
    return torch, Qwen3ASRModel


def _get(obj: Any, *names: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        for name in names:
            if name in obj:
                return obj[name]
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return round(float(value), 3)
    except (TypeError, ValueError):
        return None


def _normalize_items(raw: Any) -> Optional[list[dict[str, Any]]]:
    """Normalise forced-aligner output across qwen-asr versions.

    Accepts ``ForcedAlignResult`` (``.items`` / iterable), a dict with
    ``items``, or a plain iterable of item objects/dicts.
    """
    if raw is None:
        return None
    if hasattr(raw, "items") and not isinstance(raw, dict):
        raw = raw.items
    elif isinstance(raw, dict):
        raw = raw.get("items")
    if raw is None:
        return None
    items: list[dict[str, Any]] = []
    for item in raw:
        text = _get(item, "text", "word", default="")
        start = _get(item, "start", "start_time")
        end = _get(item, "end", "end_time")
        items.append(
            {
                "text": str(text or ""),
                "start": _as_float(start),
                "end": _as_float(end),
            }
        )
    return items


def normalize_transcription(result: Any, *, timestamps: bool = False) -> dict[str, Any]:
    """Return ``{"text", "language", "items"}`` from a qwen-asr result.

    qwen-asr returns ``list[ASRTranscription]`` with exactly one item for one
    input audio. Anything else is a broken contract, not something to paper
    over (t10 H-A04). A single object/dict/string is accepted for tests and
    older versions; an empty string is a legal transcription.
    """
    if isinstance(result, (list, tuple)):
        if len(result) != 1:
            raise ModelError(
                f"ASR returned {len(result)} results for one audio; expected 1",
                details={"results": len(result)},
            )
        result = result[0]
    if isinstance(result, str):
        return {"text": result, "language": None, "items": None}
    if result is None:
        raise ModelError("ASR returned no result")

    text = _get(result, "text", default=None)
    if not isinstance(text, str):
        raise ModelError(
            "ASR returned a non-string transcript",
            details={"text_type": type(text).__name__},
        )
    language = _get(result, "language", "lang")
    language = str(language) if language else None
    items = None
    if timestamps:
        items = _normalize_items(_get(result, "time_stamps", "timestamps", "items"))
    return {"text": text, "language": language, "items": items}


def _resolve_language(model: ModelInfo, language: Optional[str]) -> Optional[str]:
    """Canonicalise the frozen language table, then check model support."""
    canonical = canonical_asr_language(language)
    if canonical is None:
        return None
    if model.languages and not any(
        supported.lower() == canonical.lower() for supported in model.languages
    ):
        raise UsageError(
            f"ASR model {model.name} does not support language {canonical!r}",
            details={
                "language": canonical,
                "supported": model.languages,
                "model": model.name,
            },
        )
    return canonical


class AsrEngine:
    def __init__(
        self,
        cfg,
        model: ModelInfo,
        *,
        aligner: Optional[ModelInfo] = None,
    ) -> None:
        self.cfg = cfg
        self.model_info = model
        self.aligner = aligner
        self.model: Any = None
        self.device: Optional[str] = None
        self.dtype_name: Optional[str] = None
        self.load_ms: Optional[float] = None

    def load(self) -> "AsrEngine":
        if self.model is not None:
            return self
        torch, Qwen3ASRModel = _runtime()
        self.device = self.cfg.resolve_device()
        torch_dtype = self.cfg.torch_dtype(self.device)
        self.dtype_name = {
            torch.bfloat16: "bf16",
            torch.float16: "fp16",
            torch.float32: "fp32",
        }[torch_dtype]
        aligner_key = (
            (str(Path(self.aligner.path)), self.aligner.fingerprint)
            if self.aligner is not None
            else None
        )
        key = (
            str(Path(self.model_info.path)),
            self.device,
            self.dtype_name,
            self.model_info.fingerprint,
            aligner_key,
        )
        with _ASR_LOCK:
            cached = _ASR_CACHE.get(key)
            if cached is not None:
                self.model = cached
                self.load_ms = 0.0
                return self
            if self.cfg.offline:
                os.environ.setdefault("HF_HUB_OFFLINE", "1")
                os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            kwargs: dict[str, Any] = {"device_map": self.device, "dtype": torch_dtype}
            if self.cfg.offline:
                kwargs["local_files_only"] = True
            if self.aligner is not None:
                kwargs["forced_aligner"] = str(self.aligner.path)
                kwargs["forced_aligner_kwargs"] = {
                    "device_map": self.device,
                    "dtype": torch_dtype,
                }
            started = time.perf_counter()
            try:
                with stdout_to_stderr():
                    try:
                        model = Qwen3ASRModel.from_pretrained(
                            str(self.model_info.path), **kwargs
                        )
                    except TypeError as exc:
                        if "local_files_only" in str(exc) and "local_files_only" in kwargs:
                            kwargs.pop("local_files_only")
                            model = Qwen3ASRModel.from_pretrained(
                                str(self.model_info.path), **kwargs
                            )
                        else:
                            raise
            except Exception as exc:
                raise ModelError(
                    f"failed to load ASR model {self.model_info.name}: {exc}",
                    details={
                        "model": self.model_info.name,
                        "path": str(self.model_info.path),
                    },
                ) from exc
            self.load_ms = (time.perf_counter() - started) * 1000.0
            self.model = model
            _ASR_CACHE[key] = model
            return self

    def transcribe_samples(
        self,
        samples: np.ndarray,
        sr: int,
        *,
        language: Optional[str] = None,
    ) -> dict[str, Any]:
        """Transcribe an in-memory mono array without a temporary file.

        Long-audio windowing uses plain ASR only; timestamps stay disabled so
        qwen-asr never constructs a resident forced aligner.
        """
        if int(sr) <= 0:
            raise InputError("sample rate must be positive", details={"sr": int(sr)})
        data = np.asarray(samples, dtype=np.float32)
        if data.ndim > 1:
            data = data.mean(axis=1).astype(np.float32)
        if data.size == 0:
            raise InputError("cannot transcribe empty audio")
        if not np.all(np.isfinite(data)):
            raise InputError("audio contains NaN or Inf samples")
        # Validate language before loading any weights.
        lang_value = _resolve_language(self.model_info, language)
        self.load()
        try:
            with stdout_to_stderr():
                results = self.model.transcribe(
                    (np.ascontiguousarray(data, dtype=np.float32), int(sr)),
                    language=lang_value,
                    return_time_stamps=False,
                )
        except Exception as exc:
            raise ModelError(
                f"ASR transcription failed: {exc}",
                details={"model": self.model_info.name, "language": lang_value},
            ) from exc
        return normalize_transcription(results, timestamps=False)

    def release(self) -> None:
        """Drop this engine's model reference and the shared cache/backends."""
        self.model = None
        gc.collect()
        clear_asr_cache()
        gc.collect()

    def transcribe(
        self,
        audio_path: str | Path,
        *,
        language: Optional[str] = None,
        timestamps: bool = False,
    ) -> dict[str, Any]:
        path = Path(audio_path).expanduser()
        if not path.is_file():
            raise InputError(f"audio file not found: {path}", details={"path": str(path)})
        # Validate/normalise the language before any model is touched.
        lang_value = _resolve_language(self.model_info, language)
        self.load()
        if timestamps and getattr(self.model, "forced_aligner", None) is None:
            raise ModelError(
                "timestamped ASR requires a forced aligner; construct AsrEngine(..., aligner=...)",
                details={"model": self.model_info.name},
            )
        try:
            with stdout_to_stderr():
                results = self.model.transcribe(
                    str(path),
                    language=lang_value,
                    return_time_stamps=timestamps,
                )
        except Exception as exc:
            raise ModelError(
                f"ASR transcription failed: {exc}",
                details={"model": self.model_info.name, "language": lang_value},
            ) from exc
        return normalize_transcription(results, timestamps=timestamps)
