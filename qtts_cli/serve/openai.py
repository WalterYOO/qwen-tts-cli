"""OpenAI-compatible request mapping for ``qtts serve`` (FR-P2-01 §9)."""

from __future__ import annotations

from typing import Any, Optional

from ..capabilities import parse_voice_ref
from ..errors import InputError

VOICE_ALIASES = {
    "alloy": "builtin:Vivian",
    "echo": "builtin:Ethan",
    "fable": "builtin:Momo",
    "onyx": "builtin:Marcus",
    "nova": "builtin:Serena",
    "shimmer": "builtin:Chelsie",
    "ash": "builtin:Kai",
    "sage": "builtin:Eldric Sage",
    "coral": "builtin:Jada",
    "ballad": "builtin:Seren",
    "verse": "builtin:Dylan",
    "marin": "builtin:Mia",
}

_ALLOWED_FIELDS = frozenset(
    {"model", "input", "voice", "response_format", "speed", "instructions", "stream_format"}
)
_FORMATS = frozenset({"wav", "flac"})


class OpenAIRequestError(Exception):
    """A client-visible OpenAI request error (mapped to §7.4 shape)."""

    def __init__(
        self,
        message: str,
        *,
        status: int = 400,
        code: str = "invalid_request",
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status = status
        self.code = code
        self.details = details


def resolve_voice(raw: str) -> str:
    """Apply the §9.1 alias table (case-insensitive; prefixes win)."""
    lowered = raw.lower()
    if lowered.startswith(("builtin:", "voice:", "design:")):
        return raw
    alias = VOICE_ALIASES.get(lowered)
    if alias is not None:
        return alias
    return f"builtin:{raw}"


def _string(
    body: dict[str, Any], name: str, *, maximum: Optional[int] = None
) -> str:
    value = body.get(name)
    if not isinstance(value, str) or not value:
        raise OpenAIRequestError(
            f"{name} must be a non-empty string", code="invalid_request"
        )
    if maximum is not None and len(value) > maximum:
        raise OpenAIRequestError(
            f"{name} exceeds {maximum} characters", code="invalid_request"
        )
    return value


def map_speech_request(body: dict[str, Any]) -> dict[str, Any]:
    unknown = set(body) - _ALLOWED_FIELDS
    if unknown:
        raise OpenAIRequestError(
            f"unknown field: {sorted(unknown)[0]}", code="invalid_request"
        )
    if "model" in body:
        model = body["model"]
        if not isinstance(model, str) or len(model) > 256:
            raise OpenAIRequestError(
                "model must be a string of at most 256 characters",
                code="invalid_request",
            )
    text = _string(body, "input", maximum=200000)
    voice = resolve_voice(_string(body, "voice"))
    try:
        parse_voice_ref(voice)
    except InputError as exc:
        raise OpenAIRequestError(
            exc.message, code="invalid_request"
        ) from None
    response_format = body.get("response_format", "wav")
    if not isinstance(response_format, str) or response_format not in _FORMATS:
        raise OpenAIRequestError(
            "response_format must be wav or flac",
            code="invalid_request",
            details={"supported": sorted(_FORMATS)},
        )
    if "speed" in body:
        speed = body["speed"]
        if isinstance(speed, bool) or not isinstance(speed, (int, float)):
            raise OpenAIRequestError(
                "speed must be 1 or 1.0", code="invalid_request"
            )
        if speed != 1:
            raise OpenAIRequestError(
                "speed must be 1 or 1.0", code="invalid_request"
            )
    instructions = body.get("instructions")
    if instructions is not None:
        if not isinstance(instructions, str):
            raise OpenAIRequestError(
                "instructions must be a string or null",
                code="invalid_request",
            )
        if len(instructions) > 20000:
            raise OpenAIRequestError(
                "instructions exceeds 20000 characters",
                code="invalid_request",
            )
    stream_format = body.get("stream_format")
    if stream_format is not None and stream_format != "audio":
        raise OpenAIRequestError(
            "stream_format must be 'audio'", code="invalid_request"
        )
    return {
        "text": text,
        "voice": voice,
        "language": "auto",
        "instruct": instructions,
        "format": response_format,
        "subtitle": [],
        "max_new_tokens": None,
        "pause_ms": None,
        "seed": None,
        "metadata": None,
    }
