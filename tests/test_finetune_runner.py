"""FR-P2-07 §7/§12.4: locks, external streaming and signal semantics."""

from __future__ import annotations

import errno
import os
import signal
import sys
from pathlib import Path

import pytest

from qtts_cli.errors import InputError, ModelError
from qtts_cli.finetune import checkpoints
from qtts_cli.finetune import runner as runner_mod


def test_shell_free_argv_and_streaming(tmp_path: Path, capsys):
    script = "import sys; print('OUT', repr(sys.argv[1:])); print('ERR', file=sys.stderr)"
    argv = [sys.executable, "-c", script, "a b", "$(echo pwn)"]
    code = runner_mod.run_external(
        argv, cwd=tmp_path, env=runner_mod.build_env(), quiet=False
    )
    assert code == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "OUT ['a b', '$(echo pwn)']" in captured.err
    assert "ERR" in captured.err


def test_offline_env_additions(tmp_path: Path, capsys):
    script = "import os; print(os.environ.get('HF_HUB_OFFLINE'), os.environ.get('PYTHONUNBUFFERED'))"
    runner_mod.run_external(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=runner_mod.build_env(offline=True),
    )
    assert "1 1" in capsys.readouterr().err


def test_quiet_drains_silently(tmp_path: Path, capsys):
    runner_mod.run_external(
        [sys.executable, "-c", "print('noise')"],
        cwd=tmp_path,
        env=runner_mod.build_env(),
        quiet=True,
    )
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""


def test_nonzero_exit_keeps_only_last_50_nonempty_lines(tmp_path: Path):
    script = (
        "import sys\n"
        "for i in range(60): print(f'line{i}')\n"
        "print()\n"
        "print('boom', file=sys.stderr)\n"
        "sys.exit(7)\n"
    )
    with pytest.raises(ModelError) as exc:
        runner_mod.run_external(
            [sys.executable, "-c", script],
            cwd=tmp_path,
            env=runner_mod.build_env(),
            quiet=True,
        )
    details = exc.value.details
    assert details["returncode"] == 7
    assert len(details["tail"]) == 50
    assert details["tail"][0] == "line11"
    assert details["tail"][-1] == "boom"


def test_spawn_oserror_is_exit3(tmp_path: Path):
    with pytest.raises(ModelError) as exc:
        runner_mod.run_external(
            [str(tmp_path / "missing-python")],
            cwd=tmp_path,
            env=runner_mod.build_env(),
        )
    assert exc.value.code == 3


def test_lock_contention_and_io_errors(tmp_path: Path, monkeypatch):
    lock = tmp_path / "artifact.lock"
    fd = runner_mod.acquire_lock(lock)
    try:
        with pytest.raises(InputError) as exc:
            runner_mod.acquire_lock(lock)
        assert exc.value.code == 4
        assert exc.value.details["lock"] == str(lock)
    finally:
        os.close(fd)

    def fail_open(*args, **kwargs):
        raise OSError(errno.EIO, "disk error")

    monkeypatch.setattr(runner_mod.os, "open", fail_open)
    with pytest.raises(ModelError) as exc:
        runner_mod.acquire_lock(tmp_path / "other.lock")
    assert exc.value.code == 3


def test_shared_eval_vs_exclusive_train(tmp_path: Path):
    lock = tmp_path / "output.lock"
    with runner_mod.hold_lock_spec([(lock, True)]):
        with runner_mod.hold_lock_spec([(lock, True)]):
            pass
        with pytest.raises(InputError):
            runner_mod.acquire_lock(lock)
    fd = runner_mod.acquire_lock(lock)
    try:
        with pytest.raises(InputError):
            runner_mod.acquire_lock(lock, shared=True)
    finally:
        os.close(fd)


def test_project_lock_path(tmp_path: Path):
    path = tmp_path / "p.yaml"
    assert runner_mod.project_lock_path(path) == tmp_path / ".p.finetune.lock"


class FakePopen:
    pid = 4242

    def __init__(self, lines):
        self.returncode = 0
        self.stdout = iter(lines)

    def wait(self):
        return self.returncode


def _patch_popen(monkeypatch, lines):
    popen = FakePopen(lines)
    monkeypatch.setattr(runner_mod.subprocess, "Popen", lambda *a, **k: popen)
    return popen


def test_first_signal_is_forwarded_once(monkeypatch, tmp_path: Path):
    forwarded: list[tuple[int, int]] = []
    monkeypatch.setattr(runner_mod.os, "killpg", lambda pid, sig: forwarded.append((pid, sig)))

    def stream():
        yield b"working\n"
        signal.getsignal(signal.SIGINT)(signal.SIGINT, None)
        yield b"still working\n"

    _patch_popen(monkeypatch, stream())
    with pytest.raises(KeyboardInterrupt):
        runner_mod.run_external(
            ["fake"], cwd=tmp_path, env=runner_mod.build_env(), quiet=True
        )
    assert forwarded == [(4242, signal.SIGINT)]


def test_second_signal_reraises(monkeypatch, tmp_path: Path):
    forwarded: list[tuple[int, int]] = []
    monkeypatch.setattr(runner_mod.os, "killpg", lambda pid, sig: forwarded.append((pid, sig)))

    def stream():
        yield b"working\n"
        handler = signal.getsignal(signal.SIGINT)
        handler(signal.SIGINT, None)
        handler(signal.SIGINT, None)

    _patch_popen(monkeypatch, stream())
    with pytest.raises(KeyboardInterrupt):
        runner_mod.run_external(
            ["fake"], cwd=tmp_path, env=runner_mod.build_env(), quiet=True
        )
    assert forwarded == [(4242, signal.SIGINT)]


def test_deferring_handlers_record_then_raise(monkeypatch):
    deferrer = runner_mod.DeferringHandlers()
    with deferrer:
        handler = signal.getsignal(signal.SIGINT)
        handler(signal.SIGINT, None)
        assert deferrer.received == signal.SIGINT
    with pytest.raises(runner_mod.CommitInterrupt) as exc:
        deferrer.raise_if_deferred(published=True, registered=False)
    assert exc.value.published is True


def test_commit_phase_finishes_before_deferred_signal(tmp_path: Path, monkeypatch):
    partial = tmp_path / "out.partial"
    final = tmp_path / "out"
    partial.mkdir()
    (partial / "checkpoint-epoch-0").mkdir()
    real_replace = os.replace
    fired = {"done": False}

    def replace(src, dst):
        if not fired["done"]:
            fired["done"] = True
            signal.getsignal(signal.SIGINT)(signal.SIGINT, None)
        return real_replace(src, dst)

    monkeypatch.setattr(checkpoints.os, "replace", replace)
    with pytest.raises(runner_mod.CommitInterrupt) as exc:
        checkpoints.publish_output_dir(partial, final)
    assert exc.value.published is True
    assert final.is_dir()
    assert not partial.exists()
