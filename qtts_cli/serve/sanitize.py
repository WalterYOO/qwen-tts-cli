"""The single outbound sanitizer for ``qtts serve`` (FR-P2-01 §7.1).

Every string returned to an HTTP client, written to ``result.json`` or appended
to ``run.log`` goes through :func:`redact_text` (or one of the two builders that
call it).  Caller-provided ``request.text/voice/metadata`` is exempt: it is
echoed verbatim.
"""

from __future__ import annotations

import math
import os
import re
import traceback
from pathlib import Path
from typing import Any, Iterable

from ..errors import QttsError

_POSIX_PATH = re.compile(r"""(?<![\w])/(?:[^\s"'<>/]+/)*[^\s"'<>]*""")
_WINDOWS_PATH = re.compile(r"""[A-Za-z]:[\\/][^\s"'<>]*""")
_UNC_PATH = re.compile(r"""\\\\[^\s"'<>]+""")

_PAYLOAD_KEYS = (
    "ok",
    "format",
    "sample_rate",
    "duration_s",
    "sampling",
    "cache",
    "backend",
    "device",
    "dtype",
    "timing",
    "warnings",
    "align_coverage",
)
_SEGMENT_KEYS = (
    "id",
    "source_segment_id",
    "text_sha256",
    "display_sha256",
    "language",
    "instruct",
    "seed",
    "max_new_tokens",
    "cached",
    "generate_ms",
    "sampling_forwarded",
)


def redact_text(
    text: str,
    secrets: Iterable[str] = (),
    host_dirs: Iterable[str | os.PathLike[str]] = (),
) -> str:
    """Replace secrets, host directories and absolute paths with ``<redacted>``."""
    if not isinstance(text, str):
        text = str(text)
    for secret in secrets:
        secret = str(secret or "")
        if secret:
            text = text.replace(secret, "<redacted>")
    for directory in host_dirs:
        value = str(directory or "")
        if value and value != os.sep:
            text = text.replace(value, "<redacted>")
    text = _POSIX_PATH.sub("<redacted>", text)
    text = _WINDOWS_PATH.sub("<redacted>", text)
    text = _UNC_PATH.sub("<redacted>", text)
    return text


def _redact_value(value: Any, secrets, host_dirs) -> Any:
    if isinstance(value, str):
        return redact_text(value, secrets, host_dirs)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, list):
        return [_redact_value(item, secrets, host_dirs) for item in value]
    if isinstance(value, tuple):
        return [_redact_value(item, secrets, host_dirs) for item in value]
    if isinstance(value, dict):
        return {
            redact_text(key, secrets, host_dirs)
            if isinstance(key, str)
            else key: _redact_value(item, secrets, host_dirs)
            for key, item in value.items()
        }
    return value


def _finite_value(value: Any) -> Any:
    """Replace non-finite floats so every outbound JSON body is valid."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, list):
        return [_finite_value(item) for item in value]
    if isinstance(value, tuple):
        return [_finite_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _finite_value(item) for key, item in value.items()}
    return value


def _redact_details(value: Any, secrets, host_dirs) -> Any:
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            lowered = str(key).lower()
            if lowered.endswith(("path", "dir", "file", "store")):
                continue
            redacted_key = (
                redact_text(key, secrets, host_dirs)
                if isinstance(key, str)
                else key
            )
            candidate = redacted_key
            suffix = 2
            while candidate in result:
                candidate = f"{redacted_key}#{suffix}"
                suffix += 1
            result[candidate] = _redact_details(item, secrets, host_dirs)
        return result
    if isinstance(value, (list, tuple)):
        return [_redact_details(item, secrets, host_dirs) for item in value]
    if isinstance(value, str):
        return redact_text(value, secrets, host_dirs)
    return value


def redacted_traceback(exc: BaseException, *, secrets: Iterable[str] = ()) -> str:
    text = "".join(
        traceback.format_exception(type(exc), exc, exc.__traceback__)
    )
    return redact_text(text, secrets)


def public_error(
    exc: BaseException,
    *,
    secrets: Iterable[str] = (),
    host_dirs: Iterable[str | os.PathLike[str]] = (),
) -> dict[str, Any]:
    """Return the public error body; never includes a traceback."""
    if isinstance(exc, QttsError):
        payload = exc.to_dict()
        payload["message"] = redact_text(payload.get("message", ""), secrets, host_dirs)
        payload["details"] = _redact_details(
            payload.get("details") or {}, secrets, host_dirs
        )
        return _finite_value(payload)
    return {
        "ok": False,
        "error": "internal",
        "message": "internal server error",
    }


def public_payload(
    say_payload: dict[str, Any],
    workdir: str | os.PathLike[str],
    *,
    secrets: Iterable[str] = (),
    host_dirs: Iterable[str | os.PathLike[str]] = (),
) -> dict[str, Any]:
    """Whitelist projection of ``say.run()`` output (no paths, no recipe)."""
    out = {}
    for key in _PAYLOAD_KEYS:
        if key not in say_payload:
            continue
        out[key] = _redact_value(say_payload[key], secrets, host_dirs)
    segments = []
    for segment in say_payload.get("segments") or []:
        if not isinstance(segment, dict):
            continue
        item = {
            key: _redact_value(segment.get(key), secrets, host_dirs)
            for key in _SEGMENT_KEYS
            if key in segment
        }
        segments.append(item)
    out["segments"] = segments
    subtitles = {}
    root = Path(workdir)
    for fmt, path in (say_payload.get("subtitles") or {}).items():
        try:
            relative = os.path.relpath(str(path), str(root))
        except (TypeError, ValueError):
            relative = Path(str(path)).name
        try:
            size = os.path.getsize(path)
        except OSError:
            size = 0
        subtitles[fmt] = {"file": relative.replace(os.sep, "/"), "bytes": size}
    out["subtitles"] = subtitles
    return _finite_value(out)
