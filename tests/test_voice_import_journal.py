"""Crash-window tests for the btts multi-voice import journal."""

from __future__ import annotations

import json

import pytest

from qtts_cli.errors import InputError
from qtts_cli.voice_import_journal import (
    JOURNAL_NAME,
    ImportLock,
    commit_import,
    has_pending_journal,
    journal_path,
    recover_import,
)
from qtts_cli.voice_store import VoiceStore


def _store(tmp_path):
    store = VoiceStore(tmp_path / "data")
    store.root.mkdir(parents=True, exist_ok=True)
    return store


def _voice(root, name, marker):
    path = root / name
    path.mkdir(parents=True, exist_ok=True)
    (path / "marker.txt").write_text(marker, encoding="utf-8")
    return path


def _entry(
    target,
    *,
    existing=None,
    backup=None,
    staging=None,
    had_existing=None,
    **flags,
):
    if had_existing is None:
        had_existing = existing is not None
    entry = {
        "target": target,
        "existing_path": existing,
        "staging": staging or f".{target}.stage",
        "backup": backup,
        "had_existing": had_existing,
        "backup_done": False,
        "installing": False,
        "installed": False,
        "replacing": had_existing,
        "restoring": False,
        "restored": False,
        "staging_removed": False,
    }
    entry.update(flags)
    return entry


def _journal(store, entries, *, committed=False):
    path = journal_path(store)
    path.write_text(
        json.dumps({"schema_version": 1, "committed": committed, "entries": entries}),
        encoding="utf-8",
    )
    return path


def _marker(path):
    return (path / "marker.txt").read_text(encoding="utf-8")


def test_new_voice_installed_is_removed_on_recovery(tmp_path):
    store = _store(tmp_path)
    _voice(store.root, "alice", "new")
    _journal(store, [_entry("alice", had_existing=False, installed=True)])

    assert recover_import(store) == ["alice"]
    assert not (store.root / "alice").exists()
    assert not has_pending_journal(store)


def test_installed_replacement_restores_backup(tmp_path):
    store = _store(tmp_path)
    existing = _voice(store.root, "alice", "old")
    backup = _voice(store.root, ".btts-backup.1", "old")
    target = _voice(store.root, "alice", "new")
    assert target == existing
    _journal(
        store,
        [
            _entry(
                "alice",
                existing="alice",
                backup=".btts-backup.1",
                installed=True,
                backup_done=True,
            )
        ],
    )

    assert recover_import(store) == ["alice"]
    assert _marker(existing) == "old"
    assert not backup.exists()
    assert not has_pending_journal(store)


def test_replace_completed_but_installed_flag_missing(tmp_path):
    store = _store(tmp_path)
    backup = _voice(store.root, ".btts-backup.2", "old")
    _voice(store.root, "alice", "new")
    _journal(
        store,
        [
            _entry(
                "alice",
                existing="alice",
                backup=".btts-backup.2",
                backup_done=True,
                installing=True,
            )
        ],
    )

    recover_import(store)
    assert _marker(store.root / "alice") == "old"
    assert not backup.exists()
    assert not has_pending_journal(store)


def test_backup_moved_but_backup_done_flag_missing(tmp_path):
    store = _store(tmp_path)
    backup = _voice(store.root, ".btts-backup.3", "old")
    staging = _voice(store.root, ".alice.stage", "staged")
    _journal(
        store,
        [
            _entry(
                "alice",
                existing="alice",
                backup=".btts-backup.3",
                staging=".alice.stage",
            )
        ],
    )

    recover_import(store)
    assert _marker(store.root / "alice") == "old"
    assert not backup.exists()
    assert not staging.exists()
    assert not has_pending_journal(store)


def test_restoring_crash_removes_new_target_and_restores_backup(tmp_path):
    store = _store(tmp_path)
    backup = _voice(store.root, ".btts-backup.4", "old")
    _voice(store.root, "alice", "new")
    _journal(
        store,
        [
            _entry(
                "alice",
                existing="alice",
                backup=".btts-backup.4",
                backup_done=True,
                installed=True,
                restoring=True,
            )
        ],
    )

    recover_import(store)
    assert _marker(store.root / "alice") == "old"
    assert not backup.exists()
    assert not has_pending_journal(store)


def test_restore_completed_but_restored_flag_missing_is_idempotent(tmp_path):
    store = _store(tmp_path)
    _voice(store.root, "alice", "old")
    _journal(
        store,
        [
            _entry(
                "alice",
                existing="alice",
                backup=".btts-backup.5",
                backup_done=True,
                installed=True,
                restoring=True,
            )
        ],
    )

    recover_import(store)
    assert _marker(store.root / "alice") == "old"
    assert not has_pending_journal(store)


def test_staging_deleted_but_staging_removed_flag_missing_keeps_old(tmp_path):
    store = _store(tmp_path)
    _voice(store.root, "alice", "old")
    _journal(
        store,
        [
            _entry(
                "alice",
                existing="alice",
                backup=".btts-backup.6",
                backup_done=True,
                installed=True,
                restoring=True,
            )
        ],
    )

    recover_import(store)
    assert _marker(store.root / "alice") == "old"
    assert not has_pending_journal(store)


def test_staging_removed_state_never_deletes_restored_profile(tmp_path):
    store = _store(tmp_path)
    _voice(store.root, "alice", "old")
    staging = _voice(store.root, ".alice.stage", "staged")
    _journal(
        store,
        [
            _entry(
                "alice",
                existing="alice",
                backup=".btts-backup.7",
                staging=".alice.stage",
                installed=True,
                restored=True,
                staging_removed=True,
            )
        ],
    )

    recover_import(store)
    assert _marker(store.root / "alice") == "old"
    assert not staging.exists()
    assert not has_pending_journal(store)


def test_case_only_target_restores_existing_path(tmp_path):
    store = _store(tmp_path)
    _voice(store.root, "alice", "new")
    backup = _voice(store.root, ".btts-backup.8", "old")
    _journal(
        store,
        [
            _entry(
                "Alice",
                existing="alice",
                backup=".btts-backup.8",
                installed=True,
                backup_done=True,
            )
        ],
    )

    recover_import(store)
    assert _marker(store.root / "alice") == "old"
    assert not backup.exists()
    assert not has_pending_journal(store)


def test_committed_journal_only_cleans_backups_and_staging(tmp_path):
    store = _store(tmp_path)
    _voice(store.root, "alice", "new")
    staging = _voice(store.root, ".alice.stage", "staged")
    backup = _voice(store.root, ".btts-backup.9", "old")
    _journal(
        store,
        [
            _entry(
                "alice",
                existing="alice",
                backup=".btts-backup.9",
                staging=".alice.stage",
                installed=True,
                backup_done=True,
            )
        ],
        committed=True,
    )

    recover_import(store)
    assert _marker(store.root / "alice") == "new"
    assert not staging.exists()
    assert not backup.exists()
    assert not has_pending_journal(store)


def test_missing_backup_for_replaced_voice_fails_closed(tmp_path):
    store = _store(tmp_path)
    _voice(store.root, "alice", "new")
    path = _journal(
        store,
        [
            _entry(
                "alice",
                existing="alice",
                backup=".btts-backup.10",
                installed=True,
                backup_done=True,
            )
        ],
    )

    with pytest.raises(InputError, match="backup is missing"):
        recover_import(store)
    assert path.exists()
    assert _marker(store.root / "alice") == "new"


def test_new_voice_with_existing_path_null_is_valid(tmp_path):
    store = _store(tmp_path)
    _voice(store.root, "alice", "new")
    _journal(store, [_entry("alice", had_existing=False, installed=True)])

    recover_import(store)
    assert not (store.root / "alice").exists()
    assert not has_pending_journal(store)


def test_traversal_in_journal_entry_fails_closed(tmp_path):
    store = _store(tmp_path)
    path = _journal(store, [_entry("../evil", had_existing=False)])

    with pytest.raises(InputError):
        recover_import(store)
    assert path.exists()


def test_commit_rechecks_existing_names_under_lock(tmp_path):
    store = _store(tmp_path)
    _voice(store.root, "alice", "concurrent")
    staging = _voice(store.root, ".alice.stage", "new")

    with pytest.raises(InputError, match="store changed"):
        commit_import(store, [("alice", staging, None)])

    assert _marker(store.root / "alice") == "concurrent"
    assert not has_pending_journal(store)


def test_commit_rejects_disappeared_replacement_target(tmp_path):
    store = _store(tmp_path)
    staging = _voice(store.root, ".alice.stage", "new")

    with pytest.raises(InputError, match="store changed"):
        commit_import(store, [("alice", staging, "alice")])

    assert not (store.root / "alice").exists()
    assert not has_pending_journal(store)


@pytest.mark.parametrize("mutation", ["committed-string", "staging-target", "backup-real", "flag-string"])
def test_malformed_journal_fields_fail_closed(tmp_path, mutation):
    store = _store(tmp_path)
    _voice(store.root, "alice", "real")
    entry = _entry("alice", had_existing=False, installed=True)
    payload = {"schema_version": 1, "committed": False, "entries": [entry]}
    if mutation == "committed-string":
        payload["committed"] = "false"
    elif mutation == "staging-target":
        entry["staging"] = "alice"
    elif mutation == "backup-real":
        entry["had_existing"] = True
        entry["existing_path"] = "alice"
        entry["backup"] = "alice"
    elif mutation == "flag-string":
        entry["installed"] = "yes"
    path = journal_path(store)
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(InputError):
        recover_import(store)
    assert path.exists()
    assert _marker(store.root / "alice") == "real"


def test_committed_journal_with_uninstalled_entry_fails_closed(tmp_path):
    store = _store(tmp_path)
    _voice(store.root, "alice", "new")
    backup = _voice(store.root, ".btts-backup.11", "old")
    staging = _voice(store.root, ".alice.stage", "staged")
    path = _journal(
        store,
        [
            _entry(
                "alice",
                existing="alice",
                backup=".btts-backup.11",
                staging=".alice.stage",
                installed=False,
                backup_done=True,
            )
        ],
        committed=True,
    )

    with pytest.raises(InputError, match="not installed"):
        recover_import(store)
    assert path.exists()
    assert backup.exists()
    assert staging.exists()
    assert _marker(store.root / "alice") == "new"


def test_restored_entry_with_backup_present_fails_closed(tmp_path):
    store = _store(tmp_path)
    _voice(store.root, "alice", "old")
    backup = _voice(store.root, ".btts-backup.12", "old")
    path = _journal(
        store,
        [
            _entry(
                "alice",
                existing="alice",
                backup=".btts-backup.12",
                installed=True,
                backup_done=True,
                restored=True,
                staging_removed=True,
            )
        ],
    )

    with pytest.raises(InputError, match="backup is still present"):
        recover_import(store)
    assert path.exists()
    assert _marker(backup) == "old"
    assert _marker(store.root / "alice") == "old"


def test_existing_path_pointing_at_another_voice_fails_closed(tmp_path):
    store = _store(tmp_path)
    _voice(store.root, "bob", "untouched")
    _voice(store.root, "alice", "new")
    backup = _voice(store.root, ".btts-backup.13", "backup")
    path = _journal(
        store,
        [
            _entry(
                "alice",
                existing="bob",
                backup=".btts-backup.13",
                installed=True,
                backup_done=True,
            )
        ],
    )

    with pytest.raises(InputError, match="does not match target"):
        recover_import(store)
    assert path.exists()
    assert _marker(store.root / "bob") == "untouched"
    assert _marker(store.root / "alice") == "new"
    assert _marker(backup) == "backup"


def test_backup_done_with_target_and_backup_present_fails_closed(tmp_path):
    store = _store(tmp_path)
    _voice(store.root, "alice", "new")
    backup = _voice(store.root, ".btts-backup.14", "old")
    path = _journal(
        store,
        [
            _entry(
                "alice",
                existing="alice",
                backup=".btts-backup.14",
                backup_done=True,
            )
        ],
    )

    with pytest.raises(InputError, match="cannot determine"):
        recover_import(store)
    assert path.exists()
    assert _marker(store.root / "alice") == "new"
    assert _marker(backup) == "old"


def test_casefold_duplicate_targets_fail_closed(tmp_path):
    store = _store(tmp_path)
    path = _journal(
        store,
        [
            _entry("alice", had_existing=False),
            _entry("Alice", had_existing=False),
        ],
    )

    with pytest.raises(InputError, match="duplicate target"):
        recover_import(store)
    assert path.exists()


def test_symlinked_journal_fails_closed(tmp_path):
    store = _store(tmp_path)
    target = tmp_path / "outside.json"
    target.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "committed": False,
                "entries": [_entry("alice", had_existing=False)],
            }
        ),
        encoding="utf-8",
    )
    path = journal_path(store)
    path.symlink_to(target)

    with pytest.raises(InputError, match="symlinked btts import journal"):
        recover_import(store)
    assert path.is_symlink()


def test_corrupt_journal_fails_closed(tmp_path):
    store = _store(tmp_path)
    path = journal_path(store)
    path.write_text("{not-json", encoding="utf-8")

    with pytest.raises(InputError, match="corrupt btts import journal"):
        recover_import(store)
    assert path.exists()


def test_pending_journal_with_dry_run_is_rejected(tmp_path):
    store = _store(tmp_path)
    path = _journal(store, [_entry("alice", had_existing=False, installed=True)])

    with pytest.raises(InputError, match="needs recovery"):
        recover_import(store, dry_run=True)
    assert path.exists()


def test_import_lock_contention_keeps_journal(tmp_path):
    store = _store(tmp_path)
    _voice(store.root, "alice", "new")
    path = _journal(store, [_entry("alice", had_existing=False, installed=True)])

    with ImportLock(store), pytest.raises(
        InputError, match="another btts import is running"
    ):
        recover_import(store)
    assert path.exists()
    assert (store.root / "alice").exists()


def test_commit_import_rolls_back_all_entries_on_failure(tmp_path, monkeypatch):
    store = _store(tmp_path)
    existing = _voice(store.root, "alice", "old")
    staging_a = _voice(store.root, ".alice.stage", "new")
    staging_b = _voice(store.root, ".bob.stage", "new")
    real_replace = __import__("os").replace

    def flaky(src, dst, *args, **kwargs):
        if str(dst) == str(store.root / "bob"):
            raise OSError("simulated publish failure")
        return real_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr("qtts_cli.voice_import_journal.os.replace", flaky)
    with pytest.raises(OSError, match="simulated publish failure"):
        commit_import(
            store,
            [
                ("alice", staging_a, "alice"),
                ("bob", staging_b, None),
            ],
        )

    assert _marker(existing) == "old"
    assert not (store.root / "bob").exists()
    assert not has_pending_journal(store)


def test_journal_name_is_fixed_and_under_store(tmp_path):
    store = _store(tmp_path)
    assert journal_path(store) == store.root / JOURNAL_NAME
