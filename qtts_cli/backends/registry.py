"""Backend registry and built-in specs (FR-P2-02).

``resolve_backend`` is metadata-only: it scans local model directories, builds
synthetic remote models and reads plugin objects, but never imports a runtime
or loads weights.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Optional

from ..catalog import ModelInfo, scan
from ..errors import CapabilityError, ModelError, UsageError
from . import dashscope as dashscope_backend
from . import vllm_omni as vllm_backend
from .base import TTS_MODES, BackendSpec, validate_spec_models
from .transformers import CAPABILITIES as TRANSFORMERS_CAPABILITIES
from .transformers import TransformersBackend

BUILTIN_NAMES = ("transformers", "dashscope", "vllm-omni")


def builtin_backends() -> dict[str, Callable[[Any, Any], BackendSpec]]:
    return {
        "transformers": _transformers_spec,
        "dashscope": _dashscope_spec,
        "vllm-omni": _vllm_spec,
    }


def _local_tts_models(cfg) -> list[ModelInfo]:
    return [model for model in scan(cfg.model_dir) if model.kind in TTS_MODES]


def _requested_name(name: Optional[str], cfg, args) -> str:
    value = name
    if value is None:
        value = getattr(args, "backend", None)
    if value is None:
        value = getattr(cfg, "backend", None)
    if value is None:
        value = "transformers"
    value = str(value).strip()
    if not value:
        raise UsageError("--backend must not be empty")
    return value


def _check_model_flags(name: str, args, cfg=None) -> Optional[str]:
    """Validate --model/--backend-model for a backend name; return backend model."""
    local_model = getattr(args, "model", None)
    backend_model = getattr(args, "backend_model", None)
    if backend_model is None and cfg is not None:
        backend_model = getattr(cfg, "backend_model", None)
    if backend_model is not None:
        backend_model = str(backend_model)
        if not backend_model.strip():
            raise UsageError("--backend-model must not be empty")
    if local_model is not None and backend_model is not None:
        raise UsageError("--model and --backend-model are mutually exclusive")
    if name == "transformers":
        if backend_model is not None:
            raise UsageError("--backend-model is not valid for the transformers backend")
        return None
    if name == "vllm-omni":
        if backend_model is not None:
            raise UsageError("--backend-model is not valid for the vllm-omni backend")
        return None
    if name == "dashscope":
        if local_model is not None:
            raise UsageError(
                "--model selects a local model; use --backend-model for dashscope"
            )
        return backend_model
    if local_model is not None:
        raise UsageError(
            f"--model selects a local model; use --backend-model for backend {name!r}"
        )
    return backend_model


def _transformers_spec(cfg, args) -> BackendSpec:
    models = _local_tts_models(cfg)
    if not models:
        raise ModelError(
            "no local models found; pass --model-dir DIR or run `qtts models pull`",
            details={"model_dir": str(cfg.model_dir) if cfg.model_dir else None},
        )
    return BackendSpec(
        name="transformers",
        identity="transformers",
        models=models,
        capabilities=TRANSFORMERS_CAPABILITIES,
        create=lambda cfg_, model: TransformersBackend(cfg_, model),
        remote=False,
        strict_speakers=True,
        require_clone_prompt=True,
        runtime_module="qwen_tts",
    )


def _dashscope_spec(cfg, args) -> BackendSpec:
    model_id = (
        _check_model_flags("dashscope", args, cfg)
        or dashscope_backend.DEFAULT_MODEL_ID
    )
    region = os.environ.get("DASHSCOPE_REGION") or dashscope_backend.DEFAULT_REGION
    workspace = os.environ.get("DASHSCOPE_WORKSPACE") or None
    model = dashscope_backend.synthetic_model(model_id)
    return BackendSpec(
        name="dashscope",
        identity=dashscope_backend.backend_identity(model_id, region, workspace),
        models=[model],
        capabilities=dashscope_backend.CAPABILITIES,
        create=lambda cfg_, model_: dashscope_backend.DashScopeBackend(cfg_, model_),
        remote=True,
        strict_speakers=False,
        require_clone_prompt=True,  # clone is blocked by modes before this matters
        preflight_voice_ref=dashscope_backend.preflight_voice_ref,
        preflight=dashscope_backend.preflight,
        runtime_module="dashscope",
        configured=lambda: bool(os.environ.get("DASHSCOPE_API_KEY")),
    )


def _vllm_spec(cfg, args) -> BackendSpec:
    models = _local_tts_models(cfg)
    if not models:
        raise ModelError(
            "no local models found for vllm-omni; pass --model-dir DIR, install "
            "vllm-omni on Linux GPU, or use --backend transformers",
            details={
                "model_dir": str(cfg.model_dir) if cfg.model_dir else None,
                "hint": "pip install vllm-omni",
            },
        )
    return BackendSpec(
        name="vllm-omni",
        identity="vllm-omni",
        models=models,
        capabilities=vllm_backend.CAPABILITIES,
        create=lambda cfg_, model: vllm_backend.VllmOmniBackend(cfg_, model),
        remote=False,
        strict_speakers=True,
        require_clone_prompt=False,
        preflight=vllm_backend.preflight,
        runtime_module="vllm_omni",
    )


def _plugin_spec(cfg, args, name: str) -> BackendSpec:
    from .. import plugins

    backend_model = _check_model_flags(name, args, cfg)
    records, _warnings = plugins.discover_plugins()
    record = plugins.resolve_backend_plugin(name, records)
    capabilities = plugins.plugin_default_capabilities(record)
    try:
        returned_models = record.obj.models(cfg, backend_model)
        models = list(returned_models)
    except Exception as exc:
        raise ModelError(
            f"backend plugin {name!r} models() failed: "
            f"{type(exc).__name__}: {exc}",
            details={"backend": name},
        ) from exc
    for index, model in enumerate(models):
        if not isinstance(model, ModelInfo):
            raise ModelError(
                f"backend plugin {name!r} models() must return ModelInfo "
                f"objects, got {type(model).__name__} at index {index}",
                details={
                    "backend": name,
                    "index": index,
                    "returned": type(model).__name__,
                },
            )
    version = record.version or "?"
    spec = BackendSpec(
        name=name,
        identity=f"plugin:{name}@{version}:{backend_model or '-'}",
        models=models,
        capabilities=capabilities,
        create=record.obj.create,
        remote=bool(capabilities.remote),
        strict_speakers=False,
        require_clone_prompt=False,
        preflight=getattr(record.obj, "preflight", None),
        runtime_module=None,
        configured=None,
    )
    validate_spec_models(spec)
    return spec


def resolve_backend(
    name: Optional[str] = None, cfg: Any = None, args: Any = None
) -> BackendSpec:
    """Resolve a backend name to a metadata-only :class:`BackendSpec`."""
    if cfg is None:
        raise UsageError("resolve_backend requires a config")
    requested = _requested_name(name, cfg, args)
    if requested in BUILTIN_NAMES:
        # Flag compatibility is validated even before building local specs.
        _check_model_flags(requested, args, cfg)
        spec = builtin_backends()[requested](cfg, args)
    else:
        spec = _plugin_spec(cfg, args, requested)
    validate_spec_models(spec)
    return spec


def preflight_voice_ref(raw_ref: str, spec: BackendSpec) -> None:
    """Reject a voice kind before ``resolve_voice``/``select`` runs."""
    from ..capabilities import parse_voice_ref

    parsed = parse_voice_ref(raw_ref)
    mode = "clone" if parsed.kind == "voice" else parsed.kind
    if mode not in spec.capabilities.modes:
        raise CapabilityError(
            f"backend {spec.name} does not support {parsed.kind}: voices",
            details={
                "backend": spec.name,
                "mode": mode,
                "supported": sorted(spec.capabilities.modes),
            },
        )
    if spec.preflight_voice_ref is not None:
        spec.preflight_voice_ref(raw_ref)


def preflight_instruct(plan, spec: BackendSpec) -> None:
    """A backend may accept a mode but still not forward instruction.

    Only voices that actually carry an instruction are checked; a structured
    script may mix an instructed voice with a clone segment that has none.
    """
    affected: list[Any] = []
    if getattr(plan, "instruct", None):
        affected.append(plan.voice)
    for seg in getattr(plan, "segments", ()):
        if seg.instruct and seg.voice is not None:
            affected.append(seg.voice)
    seen: set[int] = set()
    for voice in affected:
        if id(voice) in seen:
            continue
        seen.add(id(voice))
        if voice.mode not in spec.capabilities.instruct_modes:
            raise CapabilityError(
                f"backend {spec.name} does not forward instruction for "
                f"mode {voice.mode!r}",
                details={
                    "backend": spec.name,
                    "mode": voice.mode,
                    "supported": sorted(spec.capabilities.instruct_modes),
                },
            )


def plan_voices(plan) -> list[Any]:
    voices = [plan.voice]
    voices.extend(seg.voice for seg in plan.segments if seg.voice is not None)
    unique: list[Any] = []
    seen: set[int] = set()
    for voice in voices:
        if id(voice) in seen:
            continue
        seen.add(id(voice))
        unique.append(voice)
    return unique


def run_preflight(plan, cfg, args, spec: BackendSpec) -> None:
    preflight_instruct(plan, spec)
    if spec.preflight is not None:
        from ..errors import QttsError

        try:
            spec.preflight(plan, cfg, args)
        except QttsError:
            raise
        except Exception as exc:
            raise ModelError(
                f"backend {spec.name} preflight failed: "
                f"{type(exc).__name__}: {exc}",
                details={"backend": spec.name},
            ) from exc
