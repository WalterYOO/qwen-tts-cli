"""FR-P2-03 §12.10/§12.11: preflight, lock, journal and recovery."""

from __future__ import annotations

import getpass
import json
import os
import signal
from pathlib import Path

import pytest

from audiobook_utils import FakeRunner, inline_project, run_build
from qtts_cli import audiobook
from qtts_cli.audiobook import manifest as manifest_mod
from qtts_cli.errors import InputError


class Bomb(FakeRunner):
    def synthesize(self, **kwargs):
        raise AssertionError("runner must not be called")


def _build(project: Path, model_dir: Path, runner=None, *extra: str):
    return run_build(project, model_dir, runner or FakeRunner(), *extra)


def _pair(root: Path):
    final = root / "out.wav"
    manifest = root / "out.json"
    journal = root / ".audiobook.publish.json"
    final.write_bytes(b"old-final")
    manifest.write_text("old-manifest", encoding="utf-8")
    return final, manifest, journal


def _temps(root: Path):
    final_tmp = root / "final.tmp"
    manifest_tmp = root / "manifest.tmp"
    final_tmp.write_bytes(b"new-final")
    manifest_tmp.write_text("new-manifest", encoding="utf-8")
    return final_tmp, manifest_tmp


def test_existing_output_blocks_before_runner(tmp_path: Path, model_dir: Path):
    project = inline_project(tmp_path, chapters=[{"title": "A", "text": "one"}])
    _build(project, model_dir)
    with pytest.raises(InputError):
        _build(project, model_dir, Bomb())


def test_parent_preflight_blocks_before_runner(tmp_path: Path, model_dir: Path):
    project = inline_project(tmp_path, chapters=[{"title": "A", "text": "one"}])
    not_a_dir = tmp_path / "notdir"
    not_a_dir.write_text("x", encoding="utf-8")
    with pytest.raises(InputError):
        _build(project, model_dir, Bomb(), "--out", str(not_a_dir / "out.wav"))


def test_lock_contention_and_stale_lock(tmp_path: Path, model_dir: Path):
    project = inline_project(tmp_path, chapters=[{"title": "A", "text": "one"}])
    out_dir = tmp_path / "build"
    out_dir.mkdir()
    lock = out_dir / ".audiobook.lock"
    lock.write_text(
        json.dumps({"pid": os.getpid(), "user": getpass.getuser(), "created_at": "x"}),
        encoding="utf-8",
    )
    with pytest.raises(InputError):
        _build(project, model_dir, Bomb())

    lock.write_text(
        json.dumps({"pid": 99999999, "user": getpass.getuser(), "created_at": "x"}),
        encoding="utf-8",
    )
    assert _build(project, model_dir)["ok"] is True

    lock.write_text(
        json.dumps({"pid": os.getpid(), "user": "someone-else", "created_at": "x"}),
        encoding="utf-8",
    )
    with pytest.raises(InputError):
        _build(project, model_dir, Bomb())
    lock.write_text("{nope", encoding="utf-8")
    with pytest.raises(InputError):
        _build(project, model_dir, Bomb())


def test_manifest_replace_failure_rolls_back(tmp_path: Path):
    root = tmp_path
    final, manifest, journal = _pair(root)
    final_tmp, _ = _temps(root)
    with pytest.raises(FileNotFoundError):
        manifest_mod.publish(
            final_tmp=final_tmp,
            manifest_tmp=root / "missing.tmp",
            final=final,
            manifest=manifest,
            journal_path=journal,
        )
    assert final.read_bytes() == b"old-final"
    assert manifest.read_text() == "old-manifest"
    assert not journal.exists()


def test_published_cleanup_failure_keeps_success(tmp_path: Path, monkeypatch):
    final, manifest, journal = _pair(tmp_path)
    final_tmp, manifest_tmp = _temps(tmp_path)
    real_unlink = os.unlink

    def failing_unlink(path):
        if str(path).endswith(".bak"):
            raise OSError("cannot remove")
        return real_unlink(path)

    monkeypatch.setattr(manifest_mod.os, "unlink", failing_unlink)
    result = manifest_mod.publish(
        final_tmp=final_tmp,
        manifest_tmp=manifest_tmp,
        final=final,
        manifest=manifest,
        journal_path=journal,
    )
    assert final.read_bytes() == b"new-final"
    assert manifest.read_text() == "new-manifest"
    assert result.warnings
    assert journal.exists()
    monkeypatch.setattr(manifest_mod.os, "unlink", real_unlink)
    assert manifest_mod.recover(journal, expected_final=final, expected_manifest=manifest).warnings is not None
    assert not journal.exists()


def test_step7_journal_failure_rolls_back(tmp_path: Path):
    final, manifest, journal = _pair(tmp_path)
    final_tmp, manifest_tmp = _temps(tmp_path)

    def boom():
        raise RuntimeError("journal write failed")

    with pytest.raises(RuntimeError):
        manifest_mod.publish(
            final_tmp=final_tmp,
            manifest_tmp=manifest_tmp,
            final=final,
            manifest=manifest,
            journal_path=journal,
            hooks={"before_published_write": boom},
        )
    assert final.read_bytes() == b"old-final"
    assert manifest.read_text() == "old-manifest"
    assert not journal.exists()


def _raising_handler(signum, frame):
    raise KeyboardInterrupt


def test_signal_before_commit_point_rolls_back(tmp_path: Path):
    final, manifest, journal = _pair(tmp_path)
    final_tmp, manifest_tmp = _temps(tmp_path)
    previous = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, _raising_handler)

    def send():
        signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)

    try:
        with pytest.raises(KeyboardInterrupt):
            manifest_mod.publish(
                final_tmp=final_tmp,
                manifest_tmp=manifest_tmp,
                final=final,
                manifest=manifest,
                journal_path=journal,
                hooks={"after_journal": send},
            )
    finally:
        signal.signal(signal.SIGTERM, previous)
    assert final.read_bytes() == b"old-final"
    assert manifest.read_text() == "old-manifest"
    assert not journal.exists()


def test_signal_after_commit_point_is_deferred(tmp_path: Path):
    final, manifest, journal = _pair(tmp_path)
    final_tmp, manifest_tmp = _temps(tmp_path)
    previous = signal.getsignal(signal.SIGINT)
    previous_term = signal.getsignal(signal.SIGTERM)
    try:
        result = manifest_mod.publish(
            final_tmp=final_tmp,
            manifest_tmp=manifest_tmp,
            final=final,
            manifest=manifest,
            journal_path=journal,
            hooks={
                "before_published_write": lambda: signal.getsignal(signal.SIGTERM)(
                    signal.SIGTERM, None
                )
            },
        )
    finally:
        signal.signal(signal.SIGINT, previous)
        signal.signal(signal.SIGTERM, previous_term)
    assert result.interrupted_after_publish is True
    assert final.read_bytes() == b"new-final"
    assert manifest.read_text() == "new-manifest"
    assert not journal.exists()


# --------------------------------------------------------------------------- #
# Recovery windows from synthetic journals
# --------------------------------------------------------------------------- #


def _journal(final, manifest, *, state="prepared", backup, manifest_backup, old_final, old_manifest):
    return {
        "schema": 1,
        "state": state,
        "final": str(final),
        "manifest": str(manifest),
        "final_tmp": str(final.parent / "final.tmp"),
        "manifest_tmp": str(manifest.parent / "manifest.tmp"),
        "backup": str(backup),
        "manifest_backup": str(manifest_backup),
        "old_final_existed": old_final,
        "old_manifest_existed": old_manifest,
    }


@pytest.mark.parametrize("window", ["journal", "final_backup", "final_replace", "manifest_replace"])
@pytest.mark.parametrize("old_final", [True, False])
def test_recovery_windows(tmp_path: Path, window: str, old_final: bool):
    final = tmp_path / "out.wav"
    manifest = tmp_path / "out.json"
    journal_path = tmp_path / ".audiobook.publish.json"
    backup = tmp_path / "out.wav.bak"
    manifest_backup = tmp_path / "out.json.mbak"

    if old_final:
        final.write_bytes(b"old-final")
    manifest.write_text("old-manifest", encoding="utf-8")

    if window == "final_backup":
        if old_final:
            os.replace(final, backup)
        else:
            os.replace(manifest, manifest_backup)
    elif window == "final_replace":
        if old_final:
            os.replace(final, backup)
        os.replace(manifest, manifest_backup)
        final.write_bytes(b"new-final")
    elif window == "manifest_replace":
        if old_final:
            os.replace(final, backup)
        os.replace(manifest, manifest_backup)
        final.write_bytes(b"new-final")
        manifest.write_text("new-manifest", encoding="utf-8")

    journal_path.write_text(
        json.dumps(
            _journal(
                final,
                manifest,
                state="prepared",
                backup=backup,
                manifest_backup=manifest_backup,
                old_final=old_final,
                old_manifest=True,
            )
        ),
        encoding="utf-8",
    )
    manifest_mod.recover(journal_path, expected_final=final, expected_manifest=manifest)
    assert manifest.read_text() == "old-manifest"
    if old_final:
        assert final.read_bytes() == b"old-final"
    else:
        assert not final.exists()
    assert not journal_path.exists()


@pytest.mark.parametrize("old_final", [True, False])
def test_recovery_published_keeps_new_pair(tmp_path: Path, old_final: bool):
    final = tmp_path / "out.wav"
    manifest = tmp_path / "out.json"
    journal_path = tmp_path / ".audiobook.publish.json"
    backup = tmp_path / "out.wav.bak"
    manifest_backup = tmp_path / "out.json.mbak"
    if old_final:
        backup.write_bytes(b"old-final")
    manifest_backup.write_text("old-manifest", encoding="utf-8")
    final.write_bytes(b"new-final")
    manifest.write_text("new-manifest", encoding="utf-8")
    journal_path.write_text(
        json.dumps(
            _journal(
                final,
                manifest,
                state="published",
                backup=backup,
                manifest_backup=manifest_backup,
                old_final=old_final,
                old_manifest=True,
            )
        ),
        encoding="utf-8",
    )
    manifest_mod.recover(journal_path, expected_final=final, expected_manifest=manifest)
    assert final.read_bytes() == b"new-final"
    assert manifest.read_text() == "new-manifest"
    assert not backup.exists() and not manifest_backup.exists()
    assert not journal_path.exists()


def test_recovery_keeps_journal_when_cleanup_fails(tmp_path: Path, monkeypatch):
    final = tmp_path / "out.wav"
    manifest = tmp_path / "out.json"
    journal_path = tmp_path / ".audiobook.publish.json"
    final_tmp = tmp_path / "final.tmp"
    final.write_bytes(b"old-final")
    manifest.write_text("old-manifest", encoding="utf-8")
    final_tmp.write_bytes(b"new-final")
    journal_path.write_text(
        json.dumps(
            _journal(
                final,
                manifest,
                backup=tmp_path / "out.wav.bak",
                manifest_backup=tmp_path / "out.json.mbak",
                old_final=True,
                old_manifest=True,
            )
        ),
        encoding="utf-8",
    )
    real_unlink = os.unlink

    def failing_unlink(path):
        if Path(path) == final_tmp:
            raise OSError("cannot remove")
        return real_unlink(path)

    monkeypatch.setattr(manifest_mod.os, "unlink", failing_unlink)
    result = manifest_mod.recover(journal_path, expected_final=final, expected_manifest=manifest)
    assert any("kept publication journal" in warning for warning in result.warnings)
    assert journal_path.exists()
    assert final.read_bytes() == b"old-final"


def test_recovery_rejects_malformed_journal_path(tmp_path: Path):
    final = tmp_path / "out.wav"
    manifest = tmp_path / "out.json"
    journal_path = tmp_path / ".audiobook.publish.json"
    payload = _journal(
        final,
        manifest,
        backup=tmp_path / "out.wav.bak",
        manifest_backup=tmp_path / "out.json.mbak",
        old_final=False,
        old_manifest=False,
    )
    payload["backup"] = 42
    journal_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(InputError):
        manifest_mod.recover(journal_path, expected_final=final, expected_manifest=manifest)


def test_fatal_recovery_keeps_journal(tmp_path: Path, model_dir: Path):
    final = tmp_path / "out.wav"
    manifest = tmp_path / "out.json"
    journal_path = tmp_path / ".audiobook.publish.json"
    final.write_bytes(b"new-final")
    journal_path.write_text(
        json.dumps(
            _journal(
                final,
                manifest,
                state="published",
                backup=tmp_path / "out.wav.bak",
                manifest_backup=tmp_path / "out.json.mbak",
                old_final=True,
                old_manifest=True,
            )
        ),
        encoding="utf-8",
    )
    with pytest.raises(InputError):
        manifest_mod.recover(journal_path, expected_final=final, expected_manifest=manifest)
    assert journal_path.exists()
    assert final.read_bytes() == b"new-final"
    assert not manifest.exists()

    # A subsequent build for the same targets is blocked before synthesis.
    build = tmp_path / "build"
    build.mkdir()
    project = inline_project(tmp_path, chapters=[{"title": "A", "text": "one"}])
    (build / "book.wav").write_bytes(b"new-final")
    (build / ".audiobook.publish.json").write_text(
        json.dumps(
            _journal(
                build / "book.wav",
                build / "book.json",
                state="published",
                backup=build / "book.wav.bak",
                manifest_backup=build / "book.json.mbak",
                old_final=True,
                old_manifest=True,
            )
        ),
        encoding="utf-8",
    )
    bomb = Bomb()
    with pytest.raises(InputError):
        _build(project, model_dir, bomb, "--force")
    assert bomb.calls == []
    assert (build / ".audiobook.publish.json").exists()


def test_signal_between_manifest_replace_and_commit_rolls_back(tmp_path: Path):
    final, manifest, journal = _pair(tmp_path)
    final_tmp, manifest_tmp = _temps(tmp_path)

    def boom():
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        manifest_mod.publish(
            final_tmp=final_tmp,
            manifest_tmp=manifest_tmp,
            final=final,
            manifest=manifest,
            journal_path=journal,
            hooks={"after_manifest_replace": boom},
        )
    assert final.read_bytes() == b"old-final"
    assert manifest.read_text() == "old-manifest"
    assert not journal.exists()


def test_journal_for_other_targets_blocks_build(tmp_path: Path, model_dir: Path):
    project = inline_project(tmp_path, chapters=[{"title": "A", "text": "one"}])
    out_dir = tmp_path / "build"
    out_dir.mkdir()
    journal = out_dir / ".audiobook.publish.json"
    journal.write_text(
        json.dumps(
            _journal(
                tmp_path / "other.wav",
                tmp_path / "other.json",
                backup=tmp_path / "other.wav.bak",
                manifest_backup=tmp_path / "other.json.mbak",
                old_final=False,
                old_manifest=False,
            )
        ),
        encoding="utf-8",
    )
    with pytest.raises(InputError):
        _build(project, model_dir, Bomb())

    planned = _build(project, model_dir, Bomb(), "--dry-run")
    assert any("interrupted build" in warning for warning in planned["warnings"])


def test_builder_post_commit_protocol(tmp_path: Path, model_dir: Path, monkeypatch, capsys):
    project = inline_project(tmp_path, chapters=[{"title": "A", "text": "one"}])

    def fake_publish(**kwargs):
        return manifest_mod.PublishResult(warnings=[], interrupted_after_publish=True)

    monkeypatch.setattr(audiobook.manifest_mod, "publish", fake_publish)
    payload = _build(project, model_dir)
    captured = capsys.readouterr()
    assert payload == {"raw_stdout": True, "exit_code": 130}
    assert captured.err == "interrupted after publish\n"
