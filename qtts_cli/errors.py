"""Exit codes and the single error type used across the CLI (PRD §7.5)."""

from __future__ import annotations

from typing import Any, Optional

# PRD §7.5
EXIT_OK = 0
EXIT_USAGE = 2
EXIT_MODEL = 3
EXIT_INPUT = 4
EXIT_QC = 5
EXIT_BATCH = 6
EXIT_INTERRUPT = 130


class QttsError(Exception):
    """An error that maps onto a stable process exit code.

    ``message`` is always written to stderr; ``details`` are included in
    ``--json`` error payloads.
    """

    def __init__(
        self,
        message: str,
        code: int = EXIT_INPUT,
        *,
        kind: str = "error",
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.kind = kind
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"ok": False, "error": self.kind, "message": self.message}
        if self.details:
            payload["details"] = self.details
        return payload


class UsageError(QttsError):
    def __init__(self, message: str, **kw: Any) -> None:
        super().__init__(message, EXIT_USAGE, kind="usage", **kw)


class InputError(QttsError):
    """Bad user input or an unsupported capability (exit 4)."""

    def __init__(self, message: str, **kw: Any) -> None:
        super().__init__(message, EXIT_INPUT, kind="input", **kw)


class CapabilityError(InputError):
    def __init__(self, message: str, **kw: Any) -> None:
        super().__init__(message, **kw)
        self.kind = "capability"


class ModelError(QttsError):
    def __init__(self, message: str, **kw: Any) -> None:
        super().__init__(message, EXIT_MODEL, kind="model", **kw)


class QCError(QttsError):
    def __init__(self, message: str, **kw: Any) -> None:
        super().__init__(message, EXIT_QC, kind="qc", **kw)
