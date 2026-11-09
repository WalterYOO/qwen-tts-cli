"""Shell detection and validation for `qtts completion`."""

from __future__ import annotations

import pytest

from qtts_cli.completion import detect
from qtts_cli.errors import UsageError


@pytest.mark.parametrize("shell", ["bash", "zsh", "fish"])
def test_explicit_supported_shells(shell: str, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SHELL", "/bin/sh")
    assert detect.detect_shell(shell) == shell


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("/bin/zsh", "zsh"),
        ("-bash", "bash"),
        ("fish.exe", "fish"),
        ("/usr/local/bin/fish", "fish"),
    ],
)
def test_normalize_shell(value: str, expected: str):
    assert detect.normalize_shell(value) == expected


@pytest.mark.parametrize("value", ["pwsh", "powershell", "PowerShell.exe"])
def test_planned_shells_are_rejected(value: str):
    with pytest.raises(UsageError) as excinfo:
        detect.normalize_shell(value)
    assert "PowerShell is planned" in excinfo.value.message


def test_unknown_shell_is_rejected():
    with pytest.raises(UsageError) as excinfo:
        detect.normalize_shell("tcsh")
    assert "supported shells" in excinfo.value.message


def test_env_override_wins(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SHELL", "/bin/bash")
    monkeypatch.setenv("QTTS_COMPLETION_SHELL", "fish")
    assert detect.detect_shell() == "fish"


def test_invalid_env_override_does_not_fall_back(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SHELL", "/bin/bash")
    monkeypatch.setenv("QTTS_COMPLETION_SHELL", "tcsh")
    with pytest.raises(UsageError):
        detect.detect_shell()


def test_shell_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("QTTS_COMPLETION_SHELL", raising=False)
    monkeypatch.setenv("SHELL", "/bin/zsh")
    assert detect.detect_shell() == "zsh"


def test_unsupported_shell_env_is_not_guessed(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("QTTS_COMPLETION_SHELL", raising=False)
    monkeypatch.setenv("SHELL", "/bin/sh")
    with pytest.raises(UsageError) as excinfo:
        detect.detect_shell()
    assert "cannot detect a supported shell" in excinfo.value.message


def test_platform_fallback(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("QTTS_COMPLETION_SHELL", raising=False)
    monkeypatch.delenv("SHELL", raising=False)
    monkeypatch.setattr(detect, "_parent_shell", lambda: None)
    monkeypatch.setattr(detect.platform, "system", lambda: "Darwin")
    assert detect.detect_shell() == "zsh"
    monkeypatch.setattr(detect.platform, "system", lambda: "Linux")
    assert detect.detect_shell() == "bash"


def test_unknown_platform_fails(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("QTTS_COMPLETION_SHELL", raising=False)
    monkeypatch.delenv("SHELL", raising=False)
    monkeypatch.setattr(detect, "_parent_shell", lambda: None)
    monkeypatch.setattr(detect.platform, "system", lambda: "Windows")
    with pytest.raises(UsageError):
        detect.detect_shell()


def test_parent_shell(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("QTTS_COMPLETION_SHELL", raising=False)
    monkeypatch.delenv("SHELL", raising=False)
    monkeypatch.setattr(detect, "_parent_shell", lambda: "fish")
    assert detect.detect_shell() == "fish"
