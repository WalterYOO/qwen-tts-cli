"""Deterministic adapter generation and syntax smoke checks."""

from __future__ import annotations

import shutil
import subprocess

import pytest

from qtts_cli.completion import shells
from qtts_cli.completion.install import render_block
from qtts_cli.errors import UsageError


@pytest.mark.parametrize("shell", ["bash", "zsh", "fish"])
def test_script_is_deterministic_and_newline_terminated(shell: str):
    first = shells.generate_script(shell)
    second = shells.generate_script(shell)
    assert first == second
    assert first.endswith("\n")
    assert not first.endswith("\n\n")
    assert "eval" not in first
    assert "__complete" in first


@pytest.mark.parametrize(
    ("shell", "checker"),
    [("bash", "bash"), ("zsh", "zsh"), ("fish", "fish")],
)
def test_script_passes_native_syntax_check(shell: str, checker: str, tmp_path):
    executable = shutil.which(checker)
    if executable is None:
        pytest.skip(f"{checker} not installed")
    script = tmp_path / f"qtts.{shell}"
    script.write_text(shells.generate_script(shell), encoding="utf-8")
    completed = subprocess.run(
        [executable, "-n", str(script)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def _bash_complete(tmp_path, payload: str) -> str:
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash not installed")
    adapter = tmp_path / "adapter.bash"
    adapter.write_text(shells.generate_script("bash"), encoding="utf-8")
    fake = tmp_path / "fake-qtts"
    fake.write_text(f"#!/bin/sh\nprintf '%b\\n' {payload!r}\n", encoding="utf-8")
    fake.chmod(0o755)
    command = (
        f"source {adapter}; "
        f"QTTS_COMPLETION_BIN={fake}; "
        "COMP_LINE='qtts '; COMP_POINT=5; "
        "_qtts_completion; "
        "if [ ${#COMPREPLY[@]} -gt 0 ]; then printf '%s\\n' \"${COMPREPLY[@]}\"; fi"
    )
    completed = subprocess.run(
        [bash, "-c", command], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout


def test_bash_adapter_does_not_emit_header_as_candidate(tmp_path):
    assert _bash_complete(tmp_path, "#qtts-complete:v1 kind=plain") == ""


def test_bash_adapter_emits_candidates(tmp_path):
    output = _bash_complete(tmp_path, "#qtts-complete:v1 kind=plain\nsay\nscript")
    assert output.splitlines() == ["say", "script"]


def test_unknown_shell_is_rejected():
    with pytest.raises(UsageError):
        shells.generate_script("tcsh")


@pytest.mark.parametrize("shell", ["bash", "zsh", "fish"])
def test_render_block_has_exact_markers(shell: str):
    script = shells.generate_script(shell)
    block = render_block(shell, script)
    begin = f"# >>> qtts completion: {shell} v1 >>>\n".encode()
    end = f"# <<< qtts completion: {shell} v1 <<<\n".encode()
    assert block.startswith(begin)
    assert block.endswith(end)
    assert script.encode() in block
