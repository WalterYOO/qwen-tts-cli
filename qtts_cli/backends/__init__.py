"""Pluggable TTS backends (FR-P2-02).

Importing this package never imports optional runtimes.
"""

from .base import (
    BACKEND_API_VERSION,
    BackendCapabilities,
    BackendSpec,
    TtsBackend,
    ensure_instance_capabilities,
    ensure_instance_protocol,
)
from .registry import (
    BUILTIN_NAMES,
    builtin_backends,
    preflight_instruct,
    preflight_voice_ref,
    resolve_backend,
    run_preflight,
)

__all__ = [
    "BACKEND_API_VERSION",
    "BUILTIN_NAMES",
    "BackendCapabilities",
    "BackendSpec",
    "TtsBackend",
    "builtin_backends",
    "ensure_instance_capabilities",
    "ensure_instance_protocol",
    "preflight_instruct",
    "preflight_voice_ref",
    "resolve_backend",
    "run_preflight",
]
