"""Backend protocol and capability descriptors (FR-P2-02).

This module is import-light: it never imports ``torch``, ``dashscope`` or
``vllm_omni``.  Concrete backends keep their optional runtimes behind lazy
imports inside ``load()``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Protocol, runtime_checkable

import numpy as np

from ..catalog import ModelInfo

BACKEND_API_VERSION = 1

TTS_MODES = frozenset({"builtin", "clone", "design"})


@dataclass(frozen=True)
class BackendCapabilities:
    """What a backend can do; ``instruct_modes`` is not a per-model promise."""

    modes: frozenset[str]
    instruct_modes: frozenset[str]
    sampling: frozenset[str]
    languages: tuple[str, ...] = ()
    remote: bool = False
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "modes": sorted(self.modes),
            "instruct_modes": sorted(self.instruct_modes),
            "sampling": sorted(self.sampling),
            "languages": list(self.languages),
            "remote": bool(self.remote),
            "notes": list(self.notes),
        }


@runtime_checkable
class TtsBackend(Protocol):
    """Structural protocol shared by local, remote and plugin backends."""

    name: str
    model_info: ModelInfo
    model: Any
    device: Optional[str]
    dtype_name: Optional[str]
    load_ms: Optional[float]
    capabilities: BackendCapabilities

    def load(self) -> "TtsBackend": ...

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
    ) -> tuple[np.ndarray, int, dict[str, Any]]: ...

    def build_clone_prompt_items(self, store, voice_name: str) -> Any: ...

    def release(self) -> None: ...


@dataclass
class BackendSpec:
    """Resolved backend: model list plus creation hook used by say/batch."""

    name: str
    identity: str
    models: list[ModelInfo]
    capabilities: BackendCapabilities
    create: Callable[[Any, ModelInfo], TtsBackend]
    remote: bool
    strict_speakers: bool = True
    require_clone_prompt: bool = True
    preflight_voice_ref: Optional[Callable[[str], None]] = None
    preflight: Optional[Callable[..., None]] = None
    # Extras used by dry-run reporting; not part of the frozen protocol.
    runtime_module: Optional[str] = None
    configured: Optional[Callable[[], bool]] = None

    def speaker_policy(self) -> str:
        return "strict" if self.strict_speakers else "canonical_or_passthrough"


def validate_spec_models(spec: BackendSpec) -> None:
    """Every model a backend offers must fit its declared modes."""
    for model in spec.models:
        if model.kind not in spec.capabilities.modes:
            from ..errors import ModelError

            raise ModelError(
                f"backend {spec.name} declares modes "
                f"{sorted(spec.capabilities.modes)} but model {model.name} "
                f"has kind {model.kind!r}",
                details={
                    "backend": spec.name,
                    "model": model.name,
                    "kind": model.kind,
                    "modes": sorted(spec.capabilities.modes),
                },
            )


_MISSING = object()


_REQUIRED_INSTANCE_ATTRIBUTES = (
    "name",
    "model_info",
    "model",
    "device",
    "dtype_name",
    "load_ms",
    "capabilities",
)
_REQUIRED_INSTANCE_METHODS = (
    "load",
    "generate",
    "build_clone_prompt_items",
    "release",
)


def ensure_instance_protocol(instance: Any, spec: BackendSpec) -> None:
    """Reject a created instance that does not implement ``TtsBackend``."""
    from ..errors import ModelError

    missing = [
        attribute
        for attribute in _REQUIRED_INSTANCE_ATTRIBUTES
        if getattr(instance, attribute, _MISSING) is _MISSING
    ]
    missing.extend(
        f"{method}()"
        for method in _REQUIRED_INSTANCE_METHODS
        if not callable(getattr(instance, method, None))
    )
    invalid: list[str] = []
    model_info = getattr(instance, "model_info", _MISSING)
    if model_info is not _MISSING and not isinstance(model_info, ModelInfo):
        invalid.append("model_info is not a ModelInfo")
    if not missing and not invalid:
        return
    problems = list(missing) + invalid
    raise ModelError(
        f"backend {spec.name} create() returned an invalid instance "
        f"({'; '.join(problems)})",
        details={
            "backend": spec.name,
            "returned": type(instance).__name__,
            "missing": missing,
            "invalid": invalid,
        },
    )


def ensure_instance_capabilities(instance: Any, spec: BackendSpec) -> None:
    """Reject a created instance that promises less than its spec."""
    from ..errors import ModelError

    caps = getattr(instance, "capabilities", _MISSING)
    if caps is _MISSING or caps is None:
        raise ModelError(
            f"backend instance for {spec.name} has no capabilities",
            details={"backend": spec.name},
        )
    if callable(caps):
        try:
            caps = caps()
        except Exception as exc:
            raise ModelError(
                f"backend instance for {spec.name} capabilities() failed: "
                f"{type(exc).__name__}: {exc}",
                details={"backend": spec.name},
            ) from exc
    if not isinstance(caps, BackendCapabilities):
        raise ModelError(
            f"backend instance for {spec.name} capabilities must be "
            "BackendCapabilities",
            details={
                "backend": spec.name,
                "returned": type(caps).__name__,
            },
        )
    for attribute in ("modes", "instruct_modes", "sampling"):
        declared = set(getattr(spec.capabilities, attribute, ()))
        actual = set(getattr(caps, attribute, ()))
        if not declared.issubset(actual):
            raise ModelError(
                f"backend instance for {spec.name} is weaker than its spec "
                f"({attribute}: {sorted(actual)} < {sorted(declared)})",
                details={
                    "backend": spec.name,
                    "attribute": attribute,
                    "declared": sorted(declared),
                    "actual": sorted(actual),
                },
            )
