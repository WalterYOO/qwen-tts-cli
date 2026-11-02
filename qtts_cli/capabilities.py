"""Voice reference parsing and the model capability matrix (PRD §3.1, §4.2).

All checks run before any model weight is loaded. Violations raise
``CapabilityError`` (exit code 4).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .catalog import ModelInfo, language_supported
from .errors import CapabilityError, InputError

VOICE_KINDS = ("builtin", "voice", "design")


@dataclass(frozen=True)
class VoiceRef:
    kind: str  # builtin | voice | design
    value: str

    @property
    def raw(self) -> str:
        return f"{self.kind}:{self.value}"

    @property
    def mode(self) -> str:
        """Generation mode: builtin (CustomVoice) / clone (Base) / design."""
        return "clone" if self.kind == "voice" else self.kind


def parse_voice_ref(text: str) -> VoiceRef:
    """Parse ``builtin:Vivian`` / ``voice:alice`` / ``design:<description>``."""
    if not text or not text.strip():
        raise InputError("--voice is required (builtin:NAME, voice:NAME or design:DESCRIPTION)")
    raw = text.strip()
    if ":" not in raw:
        raise InputError(
            f"invalid --voice value: {text!r}; expected builtin:NAME, voice:NAME or design:DESCRIPTION",
            details={"voice": text},
        )
    prefix, _, value = raw.partition(":")
    prefix = prefix.strip().lower()
    value = value.strip()
    if prefix not in VOICE_KINDS:
        raise InputError(
            f"unknown voice kind: {prefix!r}; expected one of {', '.join(VOICE_KINDS)}",
            details={"voice": text, "supported_kinds": list(VOICE_KINDS)},
        )
    if not value:
        raise InputError(f"empty voice reference value in {text!r}", details={"voice": text})
    if prefix == "voice" and ("/" in value or "\\" in value or value in (".", "..")):
        raise InputError(f"invalid voice name: {value!r}")
    return VoiceRef(kind=prefix, value=value)


def validate_capabilities(
    model: ModelInfo,
    *,
    mode: str,
    instruct: Optional[str] = None,
    language: Optional[str] = None,
    icl: bool = False,
    ref_text: Optional[str] = None,
    description: Optional[str] = None,
) -> None:
    """Reject known-unsupported combinations before loading weights."""
    expected_kind = {"builtin": "builtin", "clone": "clone", "design": "design"}[mode]
    if model.kind != expected_kind:
        raise CapabilityError(
            f"model {model.name} cannot run mode '{mode}' (model kind is '{model.kind}')",
            details={"model": model.name, "mode": mode, "model_kind": model.kind},
        )

    if instruct and not model.supports_instruct:
        detail = {
            "model": model.name,
            "model_type": model.tts_model_type,
            "size": model.size,
            "mode": mode,
        }
        if model.tts_model_type == "custom_voice":
            raise CapabilityError(
                f"{model.size or 'this'} CustomVoice model does not support --instruct; "
                "use the 1.7B CustomVoice model or omit --instruct",
                details=detail,
            )
        raise CapabilityError(
            f"model {model.name} does not support --instruct",
            details=detail,
        )

    if mode == "design" and not (description and description.strip()):
        raise CapabilityError(
            "voice design requires a non-empty description (--voice design:...)",
            details={"model": model.name},
        )

    if mode == "clone" and icl and not (ref_text and ref_text.strip()):
        raise CapabilityError(
            "ICL voice cloning requires --ref-text; use --mode xvec to clone without it",
            details={"model": model.name, "mode": "icl"},
        )

    if language and not language_supported(model, language):
        raise CapabilityError(
            f"language {language!r} is not supported by model {model.name}",
            details={"language": language, "model": model.name, "supported": model.languages},
        )
