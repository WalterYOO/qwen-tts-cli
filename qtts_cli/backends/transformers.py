"""Thin adapter around the existing local :class:`TtsEngine` (FR-P2-02).

Default behavior is unchanged: the adapter delegates every call and adds
``info["backend"] = "transformers"``.  Only this adapter clears the shared
model cache on release.
"""

from __future__ import annotations

from typing import Any, Optional

from ..catalog import ModelInfo
from ..tts_engine import TtsEngine, clear_model_cache
from .base import BackendCapabilities

CAPABILITIES = BackendCapabilities(
    modes=frozenset({"builtin", "clone", "design"}),
    instruct_modes=frozenset({"builtin", "design"}),
    sampling=frozenset(
        {"do_sample", "top_k", "top_p", "temperature", "repetition_penalty", "max_new_tokens"}
    ),
    remote=False,
)


class TransformersBackend:
    name = "transformers"
    capabilities = CAPABILITIES

    def __init__(self, cfg, model: ModelInfo, *, engine: Any = None) -> None:
        self._engine = engine if engine is not None else TtsEngine(cfg, model)
        self.model_info = model

    # -- delegated engine attributes ---------------------------------------- #
    @property
    def model(self) -> Any:
        return self._engine.model

    @model.setter
    def model(self, value: Any) -> None:
        self._engine.model = value

    @property
    def device(self) -> Optional[str]:
        return self._engine.device

    @property
    def dtype_name(self) -> Optional[str]:
        return self._engine.dtype_name

    @property
    def load_ms(self) -> Optional[float]:
        return self._engine.load_ms

    # -- protocol ----------------------------------------------------------- #
    def load(self) -> "TransformersBackend":
        self._engine.load()
        return self

    def generate(self, **kwargs):
        audio, sr, info = self._engine.generate(**kwargs)
        info = dict(info)
        info["backend"] = self.name
        return audio, sr, info

    def build_clone_prompt_items(self, store, voice_name: str) -> Any:
        return self._engine.build_clone_prompt_items(store, voice_name)

    def require_speaker_embedding(self) -> None:
        self._engine.require_speaker_embedding()

    def extract_speaker_embedding(self, samples, sr):
        return self._engine.extract_speaker_embedding(samples, sr)

    def release(self) -> None:
        self._engine.model = None
        for attribute in ("prompt", "prompt_items", "_prompt"):
            if hasattr(self._engine, attribute):
                try:
                    setattr(self._engine, attribute, None)
                except Exception:
                    pass
        clear_model_cache()
