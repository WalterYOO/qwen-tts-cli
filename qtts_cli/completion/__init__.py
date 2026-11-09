"""Shell completion support for ``qtts``.

The public CLI lives in :mod:`qtts_cli.commands.completion`; this package
contains the three small, testable pieces:

``detect``
    Shell detection and validation.
``shells``
    Deterministic bash/zsh/fish adapter generation.
``install``
    Marker-block parsing plus idempotent install/uninstall/status.
``engine``
    The hidden ``qtts __complete`` runtime used by the shell adapters.
"""

from __future__ import annotations

__all__ = ["detect", "engine", "install", "shells"]
