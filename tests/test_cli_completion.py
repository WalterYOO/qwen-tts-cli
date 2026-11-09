"""End-to-end completion command behaviour (subprocess and in-process)."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run(tmp_path: Path, *args: str, env_extra: dict[str, str] | None = None):
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)
    env["HOME"] = str(home)
    env["XDG_CONFIG_HOME"] = str(home / ".config")
    env["XDG_DATA_HOME"] = str(home / ".local" / "share")
    for name in ("QTTS_COMPLETION_SHELL", "QTTS_COMPLETION_RC", "ZDOTDIR", "ZSH_CUSTOM", "BASH_COMPLETION_USER_DIR"):
        env.pop(name, None)
    if env_extra:
        env.update(env_extra)
    completed = subprocess.run(
        [sys.executable, "-m", "qtts_cli", *args],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(tmp_path),
    )
    return completed


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_print_script(tmp_path: Path):
    completed = _run(tmp_path, "completion", "bash")
    assert completed.returncode == 0
    assert completed.stderr == ""
    assert completed.stdout.startswith("# qtts completion for bash")
    assert completed.stdout.endswith("\n")


def test_print_script_json(tmp_path: Path):
    completed = _run(tmp_path, "completion", "bash", "--json")
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["action"] == "print"
    assert payload["shell"] == "bash"
    assert payload["marker_version"] == 2
    assert payload["script"].startswith("# qtts completion for bash")


def test_install_status_uninstall_roundtrip(tmp_path: Path):
    rc = tmp_path / ".bashrc"
    rc.write_text("export USER_LINE=1\n", encoding="utf-8")

    installed = _run(
        tmp_path, "completion", "bash", "--install", "--rc-file", str(rc), "--json"
    )
    assert installed.returncode == 0, installed.stderr
    payload = json.loads(installed.stdout)
    assert payload["changed"] is True
    assert payload["block_found"] is False
    assert Path(payload["script_file"]).is_file()
    assert "complete -F _qtts_completion qtts" not in rc.read_text(encoding="utf-8")
    first = _sha(rc)
    script_before = _sha(Path(payload["script_file"]))

    again = _run(
        tmp_path, "completion", "bash", "--install", "--rc-file", str(rc), "--json"
    )
    assert again.returncode == 0
    assert json.loads(again.stdout)["changed"] is False
    assert _sha(rc) == first
    assert _sha(Path(payload["script_file"])) == script_before

    status = _run(tmp_path, "completion", "bash", "--status", "--rc-file", str(rc), "--json")
    assert status.returncode == 0
    assert json.loads(status.stdout)["state"] == "current"

    removed = _run(
        tmp_path, "completion", "bash", "--uninstall", "--rc-file", str(rc), "--json"
    )
    assert removed.returncode == 0
    assert json.loads(removed.stdout)["removed"] is True
    assert rc.read_text(encoding="utf-8") == "export USER_LINE=1\n"
    assert not Path(payload["script_file"]).exists()


def test_status_not_installed_is_exit_four_with_stdout_payload(tmp_path: Path):
    rc = tmp_path / ".zshrc"
    completed = _run(tmp_path, "completion", "zsh", "--status", "--rc-file", str(rc), "--json")
    assert completed.returncode == 4
    assert completed.stderr == ""
    payload = json.loads(completed.stdout)
    assert payload["state"] == "not_installed"
    assert payload["ok"] is False
    assert payload["exit_code"] == 4


def test_install_rejects_positional_action_name(tmp_path: Path):
    completed = _run(tmp_path, "completion", "install")
    assert completed.returncode == 2
    assert "--install is a flag" in completed.stderr
    assert "Traceback" not in completed.stderr


def test_status_dry_run_is_usage_error(tmp_path: Path):
    completed = _run(tmp_path, "completion", "bash", "--status", "--dry-run")
    assert completed.returncode == 2


def test_unsupported_shell_is_usage_error(tmp_path: Path):
    completed = _run(tmp_path, "completion", "pwsh")
    assert completed.returncode == 2
    assert "PowerShell is planned" in completed.stderr


def test_print_mode_ignores_env_rc(tmp_path: Path):
    completed = _run(
        tmp_path,
        "completion",
        "bash",
        env_extra={"QTTS_COMPLETION_RC": str(tmp_path / ".bashrc")},
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.startswith("# qtts completion for bash")


def test_shell_detection_env_override(tmp_path: Path):
    completed = _run(
        tmp_path,
        "completion",
        env_extra={"QTTS_COMPLETION_SHELL": "fish", "SHELL": "/bin/bash"},
    )
    assert completed.returncode == 0
    assert completed.stdout.startswith("# qtts completion for fish")


def test_unsupported_shell_env_is_not_guessed(tmp_path: Path):
    completed = _run(
        tmp_path,
        "completion",
        env_extra={"SHELL": "/bin/sh"},
    )
    assert completed.returncode == 2
    assert "cannot detect a supported shell" in completed.stderr


def test_default_install_uses_home(tmp_path: Path):
    home = tmp_path / "home"
    home.mkdir()
    completed = _run(
        tmp_path,
        "completion",
        "bash",
        "--install",
        "--json",
        env_extra={"HOME": str(home)},
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["script_file"] == str(
        home / ".local" / "share" / "qtts" / "completion.bash"
    )
    assert payload["rc_file"] == str(home / ".bashrc")
    assert Path(payload["script_file"]).is_file()
    assert (home / ".bashrc").is_file()
    assert not (home / ".zshrc").exists()


def test_dry_run_creates_nothing(tmp_path: Path):
    home = tmp_path / "home"
    home.mkdir()
    completed = _run(
        tmp_path,
        "completion",
        "bash",
        "--install",
        "--dry-run",
        "--json",
        env_extra={"HOME": str(home)},
    )
    assert completed.returncode == 0, completed.stderr
    assert not (home / ".bashrc").exists()
    payload = json.loads(completed.stdout)
    assert payload["changed"] is True
    assert payload["created"] is True
    assert payload["dry_run"] is True
    assert not Path(payload["script_file"]).exists()
    assert not (home / ".local" / "share" / "qtts").exists()


def test_dir_install_writes_only_the_completion_file(tmp_path: Path):
    target = tmp_path / "completions"
    completed = _run(
        tmp_path,
        "completion",
        "zsh",
        "--install",
        "--dir",
        str(target),
        "--json",
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["directory_only"] is True
    assert payload["rc_file"] is None
    assert (target / "_qtts").is_file()
    assert not (tmp_path / "home" / ".zshrc").exists()


def test_dir_install_ignores_env_rc(tmp_path: Path):
    target = tmp_path / "completions"
    env_rc = tmp_path / "env-rc"
    completed = _run(
        tmp_path,
        "completion",
        "bash",
        "--install",
        "--dir",
        str(target),
        "--json",
        env_extra={"QTTS_COMPLETION_RC": str(env_rc)},
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["directory_only"] is True
    assert payload["rc_file"] is None
    assert (target / "qtts").is_file()
    assert not env_rc.exists()


def test_status_rejects_dir(tmp_path: Path):
    completed = _run(tmp_path, "completion", "zsh", "--status", "--dir", str(tmp_path))
    assert completed.returncode == 2
    assert "--dir is only valid with --install or --uninstall" in completed.stderr


def test_json_error_is_single_object_on_stderr(tmp_path: Path):
    completed = _run(tmp_path, "completion", "pwsh", "--json")
    assert completed.returncode == 2
    assert completed.stdout == ""
    payload = json.loads(completed.stderr)
    assert payload["ok"] is False
    assert payload["code"] == 2


def test_quiet_suppresses_human_status(tmp_path: Path):
    rc = tmp_path / ".bashrc"
    completed = _run(
        tmp_path, "completion", "bash", "--status", "--rc-file", str(rc), "--quiet"
    )
    assert completed.returncode == 4
    assert completed.stdout == ""


def test_completion_does_not_build_config(monkeypatch, capsys):
    from qtts_cli import cli

    def boom(_args):
        raise AssertionError("build_config must not run for completion")

    monkeypatch.setattr(cli, "build_config", boom)
    monkeypatch.setenv("QTTS_COMPLETION_SHELL", "bash")
    assert cli.main(["completion", "bash", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["shell"] == "bash"


def test_complete_does_not_build_config(monkeypatch, capsys):
    from qtts_cli import cli

    def boom(_args):
        raise AssertionError("build_config must not run for __complete")

    monkeypatch.setattr(cli, "build_config", boom)
    assert cli.main(["__complete", "--shell", "bash", "--line", "qtts ", "--point", "5"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("#qtts-complete:v1 kind=plain")
    assert "say" in out


def test_completion_does_not_import_model_runtimes():
    script = (
        "import sys\n"
        "from qtts_cli.cli import main\n"
        "rc = main(['__complete','--shell','bash','--line','qtts ','--point','5'])\n"
        "heavy = sorted({m.split('.')[0] for m in sys.modules if m.split('.')[0] in "
        "('torch','qwen_tts','transformers','qwen_asr')})\n"
        "print('RC', rc)\n"
        "print('HEAVY', heavy)\n"
    )
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT))
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
    )
    assert completed.returncode == 0, completed.stderr
    assert "RC 0" in completed.stdout
    assert "HEAVY []" in completed.stdout
