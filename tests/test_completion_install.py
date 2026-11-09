"""Marker-block install/uninstall/status semantics."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from qtts_cli.completion import install as mod
from qtts_cli.completion.install import MarkerError, SymlinkRefused
from qtts_cli.errors import QttsError


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_install_creates_file_and_preserves_existing_content(tmp_path: Path):
    rc = tmp_path / "rc"
    rc.write_text("export USER_LINE=1\n", encoding="utf-8")
    result = mod.install("bash", str(rc))
    assert result["changed"] is True
    assert result["created"] is False
    assert result["block_found"] is False
    text = rc.read_text(encoding="utf-8")
    assert text.startswith("export USER_LINE=1\n")
    assert text.count("# >>> qtts completion: bash v1 >>>") == 1
    assert text.count("# <<< qtts completion: bash v1 <<<") == 1


def test_install_is_byte_idempotent(tmp_path: Path):
    rc = tmp_path / "rc"
    rc.write_text("export USER_LINE=1\n", encoding="utf-8")
    first = mod.install("zsh", str(rc))
    before = _sha(rc)
    second = mod.install("zsh", str(rc))
    assert first["changed"] is True
    assert second["changed"] is False
    assert _sha(rc) == before
    assert rc.read_text(encoding="utf-8").count("# >>> qtts completion: zsh v1 >>>") == 1


def test_install_appends_separator_when_missing_newline(tmp_path: Path):
    rc = tmp_path / "rc"
    rc.write_text("export USER_LINE=1", encoding="utf-8")
    mod.install("bash", str(rc))
    assert rc.read_text(encoding="utf-8").startswith("export USER_LINE=1\n# >>>")


def test_install_updates_only_the_block(tmp_path: Path):
    rc = tmp_path / "rc"
    rc.write_text("before\n", encoding="utf-8")
    mod.install("bash", str(rc))
    text = rc.read_text(encoding="utf-8")
    old_block = mod.render_block("bash", "# old script\n").decode()
    # Replace the generated block with an old one, then reinstall.
    start = text.index("# >>> qtts completion: bash v1 >>>")
    end = text.index("# <<< qtts completion: bash v1 <<<") + len(
        "# <<< qtts completion: bash v1 <<<\n"
    )
    rc.write_text(text[:start] + old_block + "after\n" + text[end:], encoding="utf-8")
    result = mod.install("bash", str(rc))
    assert result["changed"] is True
    new_text = rc.read_text(encoding="utf-8")
    assert new_text.startswith("before\n# >>>")
    assert new_text.endswith("# <<< qtts completion: bash v1 <<<\nafter\n")
    assert "# old script" not in new_text


def test_install_dry_run_writes_nothing(tmp_path: Path):
    target = tmp_path / "nested" / "rc"
    result = mod.install("fish", str(target), dry_run=True)
    assert result["changed"] is True
    assert result["created"] is True
    assert not target.exists()
    assert not target.parent.exists()


def test_install_read_only_file_is_io_error(tmp_path: Path):
    rc = tmp_path / "rc"
    rc.write_text("keep\n", encoding="utf-8")
    rc.chmod(0o444)
    try:
        with pytest.raises(QttsError) as excinfo:
            mod.install("bash", str(rc))
        assert excinfo.value.code == 3
        assert rc.read_text(encoding="utf-8") == "keep\n"
    finally:
        rc.chmod(0o644)


def test_read_only_current_install_is_a_noop(tmp_path: Path):
    rc = tmp_path / "rc"
    mod.install("bash", str(rc))
    rc.chmod(0o444)
    try:
        result = mod.install("bash", str(rc))
        assert result["changed"] is False
    finally:
        rc.chmod(0o644)


def test_read_only_uninstall_without_block_is_a_noop(tmp_path: Path):
    rc = tmp_path / "rc"
    rc.write_text("keep\n", encoding="utf-8")
    rc.chmod(0o444)
    try:
        result = mod.uninstall("bash", str(rc))
        assert result["removed"] is False
        assert result["block_found"] is False
    finally:
        rc.chmod(0o644)


def test_install_rejects_directory_target(tmp_path: Path):
    target = tmp_path / "dir"
    target.mkdir()
    with pytest.raises(QttsError) as excinfo:
        mod.install("bash", str(target))
    assert excinfo.value.code == 4


def test_install_atomic_failure_leaves_original(tmp_path: Path, monkeypatch):
    rc = tmp_path / "rc"
    rc.write_text("keep\n", encoding="utf-8")
    before = _sha(rc)

    def fail_replace(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(mod.os, "replace", fail_replace)
    with pytest.raises(QttsError) as excinfo:
        mod.install("bash", str(rc))
    assert excinfo.value.code == 3
    assert _sha(rc) == before
    assert list(tmp_path.glob(".*.tmp")) == []


@pytest.mark.parametrize(
    "content",
    [
        b"# >>> qtts completion: bash v1 >>>\n",
        b"# <<< qtts completion: bash v1 <<<\n",
        b"# >>> qtts completion: bash v1 >>>\n# <<< qtts completion: zsh v1 <<<\n",
        b"# >>> qtts completion: bash v1\n",
        b"# >>> qtts completion: bash v1 >>>\n# <<< qtts completion: bash v1 <<<\n"
        b"# >>> qtts completion: bash v1 >>>\n# <<< qtts completion: bash v1 <<<\n",
    ],
)
def test_malformed_markers_fail_closed(tmp_path: Path, content: bytes):
    rc = tmp_path / "rc"
    rc.write_bytes(b"user\n" + content)
    before = _sha(rc)
    with pytest.raises(MarkerError) as excinfo:
        mod.install("bash", str(rc))
    assert excinfo.value.code == 4
    assert _sha(rc) == before
    assert list(tmp_path.glob(".*.tmp")) == []


def test_uninstall_removes_only_block_and_is_idempotent(tmp_path: Path):
    rc = tmp_path / "rc"
    rc.write_text("before\nafter\n", encoding="utf-8")
    mod.install("bash", str(rc))
    result = mod.uninstall("bash", str(rc))
    assert result["removed"] is True
    assert result["block_found"] is True
    assert rc.read_text(encoding="utf-8") == "before\nafter\n"
    again = mod.uninstall("bash", str(rc))
    assert again["removed"] is False
    assert again["block_found"] is False


def test_uninstall_explicit_empty_rc_is_retained(tmp_path: Path):
    rc = tmp_path / "rc"
    mod.install("bash", str(rc))
    result = mod.uninstall("bash", str(rc))
    assert result["deleted_file"] is False
    assert rc.exists()
    assert rc.read_bytes() == b""


def test_uninstall_fish_default_dropin_is_deleted(tmp_path: Path, monkeypatch):
    xdg = tmp_path / "xdg"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
    target = mod.default_target("fish")
    mod.install("fish", None)
    assert target.is_file()
    result = mod.uninstall("fish", None)
    assert result["deleted_file"] is True
    assert not target.exists()


def test_uninstall_explicit_default_fish_path_is_retained(tmp_path: Path, monkeypatch):
    xdg = tmp_path / "xdg"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
    target = mod.default_target("fish")
    mod.install("fish", None)
    result = mod.uninstall("fish", str(target))
    assert result["deleted_file"] is False
    assert target.exists()
    assert target.read_bytes() == b""


def test_symlink_default_refused_and_follow_writes_target(tmp_path: Path):
    target = tmp_path / "real"
    target.write_text("keep\n", encoding="utf-8")
    link = tmp_path / "link"
    link.symlink_to(target)
    with pytest.raises(SymlinkRefused):
        mod.install("bash", str(link))
    result = mod.install("bash", str(link), follow_symlinks=True)
    assert result["changed"] is True
    assert link.is_symlink()
    assert "# >>> qtts completion: bash v1 >>>" in target.read_text(encoding="utf-8")


def test_follow_dangling_symlink_is_input_error(tmp_path: Path):
    link = tmp_path / "link"
    link.symlink_to(tmp_path / "missing")
    with pytest.raises(QttsError) as excinfo:
        mod.install("bash", str(link), follow_symlinks=True)
    assert excinfo.value.code == 4


def test_status_states(tmp_path: Path):
    rc = tmp_path / "rc"
    not_installed = mod.status("bash", str(rc))
    assert not_installed["state"] == "not_installed"
    assert not_installed["exit_code"] == 4

    mod.install("bash", str(rc))
    current = mod.status("bash", str(rc))
    assert current["state"] == "current"
    assert current["exit_code"] == 0
    assert current["ok"] is True

    data = rc.read_bytes().replace(b"# qtts completion for bash", b"# edited")
    rc.write_bytes(data)
    stale = mod.status("bash", str(rc))
    assert stale["state"] == "stale"
    assert stale["exit_code"] == 4


def test_status_unsupported_version_and_malformed(tmp_path: Path):
    rc = tmp_path / "rc"
    rc.write_bytes(
        b"# >>> qtts completion: bash v2 >>>\n"
        b"# qtts completion for bash\n"
        b"# <<< qtts completion: bash v2 <<<\n"
    )
    result = mod.status("bash", str(rc))
    assert result["state"] == "unsupported_version"
    assert result["marker_version"] == 2

    rc.write_bytes(b"# >>> qtts completion: bash v1 >>>\n")
    assert mod.status("bash", str(rc))["state"] == "malformed"


def test_status_symlink_refused(tmp_path: Path):
    target = tmp_path / "real"
    target.write_text("", encoding="utf-8")
    link = tmp_path / "link"
    link.symlink_to(target)
    result = mod.status("bash", str(link))
    assert result["state"] == "symlink_refused"
    assert result["exit_code"] == 4


def test_higher_marker_version_is_refused_for_install_and_uninstall(tmp_path: Path):
    rc = tmp_path / "rc"
    rc.write_bytes(
        b"keep\n"
        b"# >>> qtts completion: bash v2 >>>\n"
        b"# qtts completion for bash\n"
        b"# <<< qtts completion: bash v2 <<<\n"
    )
    before = _sha(rc)
    with pytest.raises(QttsError) as excinfo:
        mod.install("bash", str(rc))
    assert excinfo.value.code == 4
    with pytest.raises(QttsError) as excinfo:
        mod.uninstall("bash", str(rc))
    assert excinfo.value.code == 4
    assert _sha(rc) == before


def test_multiple_shell_blocks_coexist(tmp_path: Path):
    rc = tmp_path / "rc"
    mod.install("bash", str(rc))
    mod.install("zsh", str(rc))
    status = mod.status("bash", str(rc))
    assert status["state"] == "current"
    assert status["other_blocks"] == [{"shell": "zsh", "version": 1}]
    result = mod.uninstall("bash", str(rc))
    assert result["removed"] is True
    text = rc.read_text(encoding="utf-8")
    assert "# >>> qtts completion: zsh v1 >>>" in text
    assert "# >>> qtts completion: bash v1 >>>" not in text
