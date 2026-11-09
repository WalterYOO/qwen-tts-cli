"""Completion file installation, rc block and status semantics."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from qtts_cli.completion import install as mod
from qtts_cli.completion.install import MarkerError, SymlinkRefused
from qtts_cli.errors import QttsError


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    base = tmp_path / "home"
    base.mkdir()
    monkeypatch.setenv("HOME", str(base))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(base / ".config"))
    monkeypatch.setenv("XDG_DATA_HOME", str(base / ".local" / "share"))
    monkeypatch.delenv("ZDOTDIR", raising=False)
    monkeypatch.delenv("ZSH_CUSTOM", raising=False)
    monkeypatch.delenv("BASH_COMPLETION_USER_DIR", raising=False)
    return base


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_install_zsh_writes_script_file_and_small_block(home: Path):
    result = mod.install("zsh")
    script = home / ".zfunc" / "_qtts"
    rc = home / ".zshrc"
    assert result["script_file"] == str(script)
    assert result["rc_file"] == str(rc)
    assert result["changed"] is True
    assert result["script_created"] is True
    assert result["rc_changed"] is True
    assert mod.MARKER_VERSION == 2

    script_text = script.read_text(encoding="utf-8")
    assert script_text.startswith("#compdef qtts\n# qtts completion for zsh")
    assert "_qtts() {" in script_text
    assert mod._SCRIPT_MARKER.decode() in script_text

    rc_text = rc.read_text(encoding="utf-8")
    assert rc_text.startswith("# >>> qtts completion: zsh v2 >>>\n")
    assert f"fpath=({script.parent} $fpath)" in rc_text
    assert "autoload -Uz compinit && compinit" in rc_text
    assert "_qtts()" not in rc_text
    assert rc_text.count("# >>> qtts completion: zsh v2 >>>") == 1


def test_install_is_byte_idempotent(home: Path):
    first = mod.install("zsh")
    script = Path(first["script_file"])
    rc = Path(first["rc_file"])
    script_before = _sha(script)
    rc_before = _sha(rc)
    second = mod.install("zsh")
    assert first["changed"] is True
    assert second["changed"] is False
    assert _sha(script) == script_before
    assert _sha(rc) == rc_before


def test_install_uses_existing_fpath_dir_without_touching_rc(home: Path):
    completions = home / "custom-completions"
    completions.mkdir()
    rc = home / ".zshrc"
    rc.write_text(f"fpath=('{completions}' $fpath)\n", encoding="utf-8")
    before = _sha(rc)

    result = mod.install("zsh")
    assert result["script_file"] == str(completions / "_qtts")
    assert result["rc_file"] is None
    assert (completions / "_qtts").is_file()
    assert _sha(rc) == before


def test_install_prefers_oh_my_zsh_custom_completions(home: Path):
    (home / ".oh-my-zsh").mkdir()
    result = mod.install("zsh")
    expected = home / ".oh-my-zsh" / "custom" / "completions" / "_qtts"
    assert result["script_file"] == str(expected)
    assert result["rc_file"] is None
    assert expected.is_file()
    assert not (home / ".zshrc").exists()


def test_install_bash_prefers_bash_completion_dropin(home: Path, monkeypatch):
    monkeypatch.setattr(mod, "_bash_completion_installed", lambda: True)
    result = mod.install("bash")
    expected = home / ".local" / "share" / "bash-completion" / "completions" / "qtts"
    assert result["script_file"] == str(expected)
    assert result["rc_file"] is None
    assert expected.is_file()
    assert not (home / ".bashrc").exists()


def test_install_bash_fallback_writes_source_block(home: Path):
    result = mod.install("bash")
    script = home / ".local" / "share" / "qtts" / "completion.bash"
    rc = home / ".bashrc"
    assert result["script_file"] == str(script)
    assert result["rc_file"] == str(rc)
    rc_text = rc.read_text(encoding="utf-8")
    assert f"if [ -r {script} ]; then . {script}; fi" in rc_text
    assert "complete -F _qtts_completion qtts" not in rc_text


def test_install_fish_writes_completions_file_only(home: Path):
    result = mod.install("fish")
    script = home / ".config" / "fish" / "completions" / "qtts.fish"
    assert result["script_file"] == str(script)
    assert result["rc_file"] is None
    assert result["rc_changed"] is False
    assert script.is_file()
    assert not (home / ".config" / "fish" / "config.fish").exists()


def test_install_directory_only(home: Path, tmp_path: Path):
    target = tmp_path / "completions"
    result = mod.install("bash", script_dir=str(target))
    assert result["directory_only"] is True
    assert result["rc_file"] is None
    assert Path(result["script_file"]) == target / "qtts"
    assert (target / "qtts").is_file()
    assert not (home / ".bashrc").exists()


def test_install_dry_run_writes_nothing(home: Path):
    result = mod.install("zsh", dry_run=True)
    assert result["changed"] is True
    assert result["dry_run"] is True
    assert not Path(result["script_file"]).exists()
    assert not Path(result["rc_file"]).exists()
    assert not (home / ".zfunc").exists()


def test_install_read_only_script_is_io_error(home: Path):
    first = mod.install("zsh")
    script = Path(first["script_file"])
    script.write_text(
        script.read_text(encoding="utf-8") + "# stale edit\n", encoding="utf-8"
    )
    script.chmod(0o444)
    try:
        with pytest.raises(QttsError) as excinfo:
            mod.install("zsh")
        assert excinfo.value.code == 3
    finally:
        script.chmod(0o644)


def test_install_refuses_foreign_completion_file(home: Path):
    script = home / ".zfunc" / "_qtts"
    script.parent.mkdir(parents=True)
    script.write_text("not qtts\n", encoding="utf-8")
    with pytest.raises(QttsError) as excinfo:
        mod.install("zsh")
    assert excinfo.value.code == 4
    assert script.read_text(encoding="utf-8") == "not qtts\n"
    assert not (home / ".zshrc").exists()


def test_readonly_primary_rc_prevents_cleanup(home: Path):
    first = mod.install("zsh")
    rc = Path(first["rc_file"])
    rc.write_text("user only\n", encoding="utf-8")
    zshenv = home / ".zshenv"
    zshenv.write_text(
        "# >>> qtts completion: zsh v1 >>>\n"
        "# old inline script\n"
        "# <<< qtts completion: zsh v1 <<<\n",
        encoding="utf-8",
    )
    rc.chmod(0o444)
    try:
        with pytest.raises(QttsError) as excinfo:
            mod.install("zsh")
        assert excinfo.value.code == 3
    finally:
        rc.chmod(0o644)
    assert "# >>> qtts completion: zsh v1 >>>" in zshenv.read_text(
        encoding="utf-8"
    )


def test_install_atomic_failure_leaves_original(home: Path, monkeypatch):
    first = mod.install("zsh")
    script = Path(first["script_file"])
    script.write_text(
        script.read_text(encoding="utf-8") + "# stale edit\n", encoding="utf-8"
    )
    before = _sha(script)

    def fail_replace(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(mod.os, "replace", fail_replace)
    with pytest.raises(QttsError) as excinfo:
        mod.install("zsh")
    assert excinfo.value.code == 3
    assert _sha(script) == before
    assert list(script.parent.glob(".*.tmp")) == []


@pytest.mark.parametrize(
    "content",
    [
        b"# >>> qtts completion: zsh v2 >>>\n",
        b"# <<< qtts completion: zsh v2 <<<\n",
        b"# >>> qtts completion: zsh v2 >>>\n# <<< qtts completion: bash v2 <<<\n",
        b"# >>> qtts completion: zsh v2\n",
        b"# >>> qtts completion: zsh v2 >>>\n# <<< qtts completion: zsh v2 <<<\n"
        b"# >>> qtts completion: zsh v2 >>>\n# <<< qtts completion: zsh v2 <<<\n",
    ],
)
def test_malformed_markers_fail_closed(home: Path, content: bytes):
    rc = home / ".zshrc"
    rc.write_bytes(b"user\n" + content)
    before = _sha(rc)
    with pytest.raises(MarkerError) as excinfo:
        mod.install("zsh")
    assert excinfo.value.code == 4
    assert _sha(rc) == before
    assert not (home / ".zfunc" / "_qtts").exists()
    assert list(home.rglob(".*.tmp")) == []


def test_uninstall_removes_script_and_block(home: Path):
    first = mod.install("zsh")
    result = mod.uninstall("zsh")
    assert result["script_removed"] is True
    assert result["rc_removed"] is True
    assert result["changed"] is True
    assert not Path(first["script_file"]).exists()
    assert "# >>> qtts completion: zsh v2 >>>" not in Path(first["rc_file"]).read_text(
        encoding="utf-8"
    )
    again = mod.uninstall("zsh")
    assert again["changed"] is False


def test_uninstall_explicit_rc_keeps_rc_file(home: Path):
    rc = home / "custom-rc"
    rc.write_text("keep\n", encoding="utf-8")
    mod.install("zsh", str(rc))
    result = mod.uninstall("zsh", str(rc))
    assert result["rc_removed"] is True
    assert rc.exists()
    assert rc.read_text(encoding="utf-8") == "keep\n"


def test_uninstall_explicit_rc_does_not_remove_other_rc_blocks(home: Path):
    other = home / ".zshrc"
    other.write_text("keep\n", encoding="utf-8")
    custom = home / "custom-rc"
    mod.install("zsh", str(custom))
    # Install a second block in the default rc through the default path.
    mod.install("zsh")
    result = mod.uninstall("zsh", str(custom))
    assert result["rc_removed"] is True
    assert "# >>> qtts completion: zsh v2 >>>" in other.read_text(encoding="utf-8")


def test_uninstall_fish_default_removes_file_without_rc(home: Path):
    first = mod.install("fish")
    result = mod.uninstall("fish")
    assert result["script_removed"] is True
    assert result["rc_removed"] is False
    assert not Path(first["script_file"]).exists()


def test_uninstall_explicit_fish_rc_is_retained(home: Path, monkeypatch, tmp_path: Path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    rc = tmp_path / "fish-config"
    mod.install("fish", str(rc))
    result = mod.uninstall("fish", str(rc))
    assert result["rc_removed"] is True
    assert rc.exists()
    assert rc.read_bytes() == b""


def test_symlink_rc_refused_and_follow_writes_target(home: Path):
    target = home / "real-rc"
    target.write_text("keep\n", encoding="utf-8")
    link = home / "link-rc"
    link.symlink_to(target)
    with pytest.raises(SymlinkRefused):
        mod.install("zsh", str(link))
    result = mod.install("zsh", str(link), follow_symlinks=True)
    assert result["rc_changed"] is True
    assert link.is_symlink()
    assert "# >>> qtts completion: zsh v2 >>>" in target.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("shell", "rc_name"), [("zsh", ".zshrc"), ("bash", ".bashrc")]
)
def test_default_symlinked_rc_is_not_treated_as_stale(
    home: Path, shell: str, rc_name: str
):
    real = home / "dotfiles" / rc_name
    real.parent.mkdir()
    real.write_text("keep\n", encoding="utf-8")
    link = home / rc_name
    link.symlink_to(real)

    first = mod.install(shell, follow_symlinks=True)
    assert first["rc_file"] == str(link)
    assert first["rc_changed"] is True
    assert link.is_symlink()

    current = mod.status(shell, follow_symlinks=True)
    assert current["state"] == "current"
    assert current["stale_rc_files"] == []

    second = mod.install(shell, follow_symlinks=True)
    assert second["changed"] is False
    text = real.read_text(encoding="utf-8")
    assert f"# >>> qtts completion: {shell} v2 >>>" in text


def test_follow_dangling_symlink_is_input_error(home: Path):
    link = home / "link-rc"
    link.symlink_to(home / "missing")
    with pytest.raises(QttsError) as excinfo:
        mod.install("zsh", str(link), follow_symlinks=True)
    assert excinfo.value.code == 4


def test_status_states(home: Path):
    not_installed = mod.status("zsh")
    assert not_installed["state"] == "not_installed"
    assert not_installed["exit_code"] == 4

    first = mod.install("zsh")
    current = mod.status("zsh")
    assert current["state"] == "current"
    assert current["exit_code"] == 0
    assert current["ok"] is True
    assert current["rc_state"] == "current"

    script = Path(first["script_file"])
    script.write_text(script.read_text(encoding="utf-8") + "# edit\n", encoding="utf-8")
    stale = mod.status("zsh")
    assert stale["state"] == "stale"
    assert stale["exit_code"] == 4

    mod.install("zsh")
    rc = Path(first["rc_file"])
    rc.write_text("user only\n", encoding="utf-8")
    stale_rc = mod.status("zsh")
    assert stale_rc["state"] == "stale"
    assert stale_rc["rc_state"] == "not_installed"


def test_status_unsupported_version_and_malformed(home: Path):
    first = mod.install("zsh")
    rc = Path(first["rc_file"])
    rc.write_bytes(
        b"# >>> qtts completion: zsh v99 >>>\n"
        b"fpath=(/tmp $fpath)\n"
        b"autoload -Uz compinit && compinit\n"
        b"# <<< qtts completion: zsh v99 <<<\n"
    )
    result = mod.status("zsh")
    assert result["state"] == "unsupported_version"
    assert result["rc_state"] == "unsupported_version"
    assert result["marker_version"] == 99

    rc.write_bytes(b"# >>> qtts completion: zsh v2 >>>\n")
    assert mod.status("zsh")["state"] == "malformed"


def test_status_symlink_refused(home: Path):
    target = home / "real-rc"
    target.write_text("", encoding="utf-8")
    link = home / "link-rc"
    link.symlink_to(target)
    result = mod.status("zsh", str(link))
    assert result["state"] == "symlink_refused"
    assert result["exit_code"] == 4


def test_higher_marker_version_is_refused_for_install_and_uninstall(home: Path):
    rc = home / ".zshrc"
    rc.write_bytes(
        b"keep\n"
        b"# >>> qtts completion: zsh v99 >>>\n"
        b"fpath=(/tmp $fpath)\n"
        b"# <<< qtts completion: zsh v99 <<<\n"
    )
    before = _sha(rc)
    with pytest.raises(QttsError) as excinfo:
        mod.install("zsh")
    assert excinfo.value.code == 4
    with pytest.raises(QttsError) as excinfo:
        mod.uninstall("zsh")
    assert excinfo.value.code == 4
    assert _sha(rc) == before
    assert not (home / ".zfunc" / "_qtts").exists()


def test_multiple_shell_blocks_coexist(home: Path):
    rc = home / "shared-rc"
    mod.install("bash", str(rc))
    mod.install("zsh", str(rc))
    status = mod.status("bash", str(rc))
    assert status["state"] == "current"
    assert status["other_blocks"] == [{"shell": "zsh", "version": 2}]
    result = mod.uninstall("bash", str(rc))
    assert result["rc_removed"] is True
    text = rc.read_text(encoding="utf-8")
    assert "# >>> qtts completion: zsh v2 >>>" in text
    assert "# >>> qtts completion: bash v2 >>>" not in text


def test_install_cleans_v1_block_from_symlinked_rc(home: Path):
    (home / ".oh-my-zsh").mkdir()
    real = home / "dotfiles" / "zshrc"
    real.parent.mkdir()
    real.write_text(
        "keep\n"
        "# >>> qtts completion: zsh v1 >>>\n"
        "# old inline script\n"
        "# <<< qtts completion: zsh v1 <<<\n",
        encoding="utf-8",
    )
    (home / ".zshrc").symlink_to(real)

    result = mod.install("zsh")
    assert result["rc_file"] is None
    text = real.read_text(encoding="utf-8")
    assert text == "keep\n"
    assert (home / ".zshrc").is_symlink()


def test_install_cleans_v1_block_from_alternate_zsh_rc(home: Path):
    zshenv = home / ".zshenv"
    zshenv.write_text(
        "keep-env\n"
        "# >>> qtts completion: zsh v1 >>>\n"
        "# old inline script\n"
        "# <<< qtts completion: zsh v1 <<<\n",
        encoding="utf-8",
    )

    result = mod.install("zsh")
    assert result["rc_file"] == str(home / ".zshrc")
    assert zshenv.read_text(encoding="utf-8") == "keep-env\n"
    assert "# >>> qtts completion: zsh v2 >>>" in (
        home / ".zshrc"
    ).read_text(encoding="utf-8")


def test_install_cleans_v1_block_from_alternate_bash_rc(home: Path):
    bashrc = home / ".bashrc"
    bashrc.write_text("keep\n", encoding="utf-8")
    profile = home / ".bash_profile"
    profile.write_text(
        "# >>> qtts completion: bash v1 >>>\n"
        "# old inline script\n"
        "# <<< qtts completion: bash v1 <<<\n",
        encoding="utf-8",
    )

    result = mod.install("bash")
    assert result["rc_file"] == str(bashrc)
    assert profile.read_text(encoding="utf-8") == ""
    assert "# >>> qtts completion: bash v2 >>>" in bashrc.read_text(
        encoding="utf-8"
    )


def test_malformed_alternate_rc_does_not_delete_stale_script(home: Path):
    stale = home / ".zsh" / "completions" / "_qtts"
    stale.parent.mkdir(parents=True)
    stale.write_text(
        mod._SCRIPT_MARKER.decode() + "\n# old completion\n", encoding="utf-8"
    )
    zshenv = home / ".zshenv"
    zshenv.write_text("# >>> qtts completion: zsh v2 >>>\n", encoding="utf-8")

    with pytest.raises(MarkerError):
        mod.install("zsh")
    assert stale.is_file()
    assert zshenv.read_text(encoding="utf-8") == "# >>> qtts completion: zsh v2 >>>\n"


def test_readonly_alternate_rc_does_not_delete_stale_script(home: Path):
    stale = home / ".zsh" / "completions" / "_qtts"
    stale.parent.mkdir(parents=True)
    stale.write_text(
        mod._SCRIPT_MARKER.decode() + "\n# old completion\n", encoding="utf-8"
    )
    zshenv = home / ".zshenv"
    zshenv.write_text(
        "# >>> qtts completion: zsh v1 >>>\n"
        "# old inline script\n"
        "# <<< qtts completion: zsh v1 <<<\n",
        encoding="utf-8",
    )
    zshenv.chmod(0o444)
    try:
        with pytest.raises(QttsError) as excinfo:
            mod.install("zsh")
        assert excinfo.value.code == 3
    finally:
        zshenv.chmod(0o644)
    assert stale.is_file()
    assert "# >>> qtts completion: zsh v1 >>>" in zshenv.read_text(
        encoding="utf-8"
    )


def test_status_reports_stale_alternate_rc(home: Path):
    mod.install("zsh")
    zshenv = home / ".zshenv"
    zshenv.write_text(
        "# >>> qtts completion: zsh v1 >>>\n"
        "# old inline script\n"
        "# <<< qtts completion: zsh v1 <<<\n",
        encoding="utf-8",
    )

    result = mod.status("zsh")
    assert result["state"] == "stale"
    assert result["exit_code"] == 4
    assert result["stale_rc_files"] == [str(zshenv)]
    assert "stale completion block" in result["message"]

    mod.install("zsh")
    assert mod.status("zsh")["state"] == "current"
    assert zshenv.read_text(encoding="utf-8") == ""


def test_status_reports_malformed_alternate_rc(home: Path):
    mod.install("zsh")
    zshenv = home / ".zshenv"
    zshenv.write_text("# >>> qtts completion: zsh v2 >>>\n", encoding="utf-8")

    result = mod.status("zsh")
    assert result["state"] == "malformed"
    assert result["malformed_rc_files"] == [str(zshenv)]
    assert "malformed completion marker" in result["message"]


def test_status_explicit_rc_does_not_scan_default_rc(home: Path):
    mod.install("zsh")
    custom = home / "custom-rc"
    mod.install("zsh", str(custom))

    result = mod.status("zsh", str(custom))
    assert result["state"] == "current"
    assert result["stale_rc_files"] == []


def test_legacy_fish_dropin_is_removed_on_install(home: Path):
    legacy = home / ".config" / "fish" / "conf.d" / "qtts.fish"
    legacy.parent.mkdir(parents=True)
    legacy.write_text(
        "# >>> qtts completion: fish v1 >>>\n"
        "# old inline script\n"
        "# <<< qtts completion: fish v1 <<<\n",
        encoding="utf-8",
    )
    result = mod.install("fish")
    assert result["changed"] is True
    assert not legacy.exists()
    assert Path(result["script_file"]).is_file()
