"""Voice-store management primitives: tags, rename, delete, scan."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from qtts_cli.errors import InputError, UsageError
from qtts_cli.voice_store import (
    TOMBSTONE_PREFIX,
    VoiceStore,
    normalize_tag,
    normalize_tags,
)


def _store(tmp_path: Path) -> VoiceStore:
    return VoiceStore(tmp_path / "data")


def _profile(store: VoiceStore, name: str, *, kind: str = "design", tags=()) -> dict:
    return store.create(
        name=name,
        kind=kind,
        source={"type": kind, "description": "synthetic"},
        model={},
        tags=list(tags),
    )


# --------------------------------------------------------------------------- #
# Tag normalization
# --------------------------------------------------------------------------- #
def test_normalize_tag_trims_and_rejects_unsafe_values():
    assert normalize_tag("  warm  ") == "warm"
    with pytest.raises(UsageError) as empty:
        normalize_tag("   ")
    assert empty.value.code == 2
    with pytest.raises(UsageError):
        normalize_tag("x" * 65)
    with pytest.raises(UsageError):
        normalize_tag("bad\x00tag")


def test_normalize_tags_dedupes_case_insensitively_and_keeps_order():
    assert normalize_tags(["Warm", "narrator", "warm", " calm "]) == [
        "Warm",
        "narrator",
        "calm",
    ]


# --------------------------------------------------------------------------- #
# scan_entries
# --------------------------------------------------------------------------- #
def test_scan_entries_reports_ok_mismatch_corrupt_and_symlink(tmp_path: Path):
    store = _store(tmp_path)
    _profile(store, "good")
    _profile(store, "misnamed")
    misnamed = store.dir_for("misnamed") / "voice.json"
    meta = json.loads(misnamed.read_text(encoding="utf-8"))
    meta["name"] = "other"
    misnamed.write_text(json.dumps(meta), encoding="utf-8")

    missing = store.root / "no-meta"
    missing.mkdir()
    broken = store.root / "broken-json"
    broken.mkdir()
    (broken / "voice.json").write_text("{not json", encoding="utf-8")

    outside = tmp_path / "outside"
    outside.mkdir()
    (store.root / "linked").symlink_to(outside)

    tombstone = store.root / f"{TOMBSTONE_PREFIX}crashed-abcd1234"
    tombstone.mkdir()

    report = store.scan_entries()
    by_name = {entry["name"]: entry for entry in report["voices"]}
    assert by_name["good"]["status"] == "ok"
    assert by_name["misnamed"]["status"] == "mismatch"
    assert "rename misnamed other" in by_name["misnamed"]["warnings"][0]
    # The hint must actually repair the profile.
    store.rename("misnamed", "other")
    assert store.require("other")["name"] == "other"
    assert by_name["no-meta"]["status"] == "corrupt"
    assert by_name["broken-json"]["status"] == "corrupt"
    assert by_name["linked"]["status"] == "symlink"
    assert "crashed-abcd1234" not in by_name
    assert any("tombstone" in warning for warning in report["warnings"])
    assert any("symlink" in warning for warning in report["warnings"])


def test_scan_entries_on_missing_root_is_empty(tmp_path: Path):
    assert _store(tmp_path).scan_entries() == {"voices": [], "warnings": []}


# --------------------------------------------------------------------------- #
# rename
# --------------------------------------------------------------------------- #
def test_rename_moves_directory_and_metadata(tmp_path: Path):
    store = _store(tmp_path)
    _profile(store, "old", tags=["warm"])
    (store.dir_for("old") / "ref.wav").write_bytes(b"RIFF")
    result = store.rename("old", "new")
    assert result["old"] == "old" and result["new"] == "new"
    assert not store.dir_for("old").exists()
    assert store.require("new")["name"] == "new"
    assert store.require("new")["tags"] == ["warm"]
    assert (store.dir_for("new") / "ref.wav").read_bytes() == b"RIFF"


def test_rename_case_only(tmp_path: Path):
    store = _store(tmp_path)
    _profile(store, "Alice")
    store.rename("Alice", "alice")
    assert "alice" in {child.name for child in store.root.iterdir()}
    assert store.require("alice")["name"] == "alice"


def test_rename_accepts_any_case_spelling_of_the_source(tmp_path: Path):
    store = _store(tmp_path)
    _profile(store, "Alice")
    store.rename("alice", "bob")
    assert store.require("bob")["name"] == "bob"


def test_rename_recovers_case_only_crash_between_hops(tmp_path: Path):
    store = _store(tmp_path)
    _profile(store, "Alice")
    meta_path = store.dir_for("Alice") / "voice.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["name"] = "alice"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    staging = store.root / ".rename-crashed"
    staging.mkdir()
    (staging / "rename.json").write_text(
        json.dumps({"old": "Alice", "new": "alice"}), encoding="utf-8"
    )
    store.root.joinpath("Alice").rename(staging / "profile")

    result = store.rename("Alice", "alice")
    assert result["new"] == "alice"
    assert store.require("alice")["name"] == "alice"
    assert not staging.exists()


def test_rename_source_missing_without_journal_is_usage_error(tmp_path: Path):
    store = _store(tmp_path)
    _profile(store, "new")
    with pytest.raises(UsageError) as error:
        store.rename("old", "new")
    assert error.value.code == 2


def test_rename_after_completed_move_recovers_from_journal(tmp_path: Path):
    store = _store(tmp_path)
    _profile(store, "new")
    journal = store.root / ".rename-crashed"
    journal.mkdir()
    (journal / "rename.json").write_text(
        json.dumps({"old": "old", "new": "new"}), encoding="utf-8"
    )
    result = store.rename("old", "new")
    assert result["recovered"] is True
    assert store.require("new")["name"] == "new"
    assert not journal.exists()


@pytest.mark.parametrize("kind", ["file", "broken-symlink", "dir-symlink"])
def test_rename_refuses_non_directory_targets(tmp_path: Path, kind: str):
    store = _store(tmp_path)
    _profile(store, "source")
    target = store.root / "occupied"
    if kind == "file":
        target.write_text("x", encoding="utf-8")
    elif kind == "broken-symlink":
        target.symlink_to(tmp_path / "missing")
    else:
        directory = tmp_path / "elsewhere"
        directory.mkdir()
        target.symlink_to(directory)
    with pytest.raises(UsageError) as error:
        store.rename("source", "occupied")
    assert error.value.code == 2
    assert store.require("source")["name"] == "source"


def test_rename_refuses_existing_target_with_exit_2(tmp_path: Path):
    store = _store(tmp_path)
    _profile(store, "one")
    _profile(store, "two")
    with pytest.raises(UsageError) as error:
        store.rename("one", "two")
    assert error.value.code == 2
    assert store.require("one")["name"] == "one"


def test_rename_missing_source_is_usage_error(tmp_path: Path):
    with pytest.raises(UsageError) as error:
        _store(tmp_path).rename("missing", "newname")
    assert error.value.code == 2


def test_rename_refuses_symlinked_source(tmp_path: Path):
    store = _store(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "voice.json").write_text('{"name": "linked"}', encoding="utf-8")
    store.root.mkdir(parents=True, exist_ok=True)
    (store.root / "linked").symlink_to(outside)
    with pytest.raises(InputError) as error:
        store.rename("linked", "moved")
    assert error.value.code == 4


def test_rename_repairs_crash_after_metadata_write(tmp_path: Path):
    store = _store(tmp_path)
    _profile(store, "old")
    path = store.dir_for("old") / "voice.json"
    meta = json.loads(path.read_text(encoding="utf-8"))
    meta["name"] = "new"
    path.write_text(json.dumps(meta), encoding="utf-8")

    result = store.rename("old", "new")
    assert result["new"] == "new"
    assert store.require("new")["name"] == "new"


def test_rename_refuses_third_party_mismatch(tmp_path: Path):
    store = _store(tmp_path)
    _profile(store, "old")
    path = store.dir_for("old") / "voice.json"
    meta = json.loads(path.read_text(encoding="utf-8"))
    meta["name"] = "intruder"
    path.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(InputError) as error:
        store.rename("old", "new")
    assert error.value.code == 4


def test_rename_dry_run_does_not_write(tmp_path: Path):
    store = _store(tmp_path)
    _profile(store, "old")
    before = (store.dir_for("old") / "voice.json").read_bytes()
    result = store.rename("old", "new", dry_run=True)
    assert result["dry_run"] is True
    assert store.dir_for("old").is_dir()
    assert (store.dir_for("old") / "voice.json").read_bytes() == before


# --------------------------------------------------------------------------- #
# tags
# --------------------------------------------------------------------------- #
def test_tag_read_only_when_no_operation(tmp_path: Path):
    store = _store(tmp_path)
    _profile(store, "alice", tags=["Warm"])
    result = store.set_tags("alice")
    assert result == {
        "voice": "alice",
        "before": ["Warm"],
        "after": ["Warm"],
        "changed": False,
        "tags": ["Warm"],
        "dry_run": False,
    }


def test_tag_add_remove_replace_and_clear(tmp_path: Path):
    store = _store(tmp_path)
    _profile(store, "alice", tags=["warm"])
    assert store.set_tags("alice", add=["Narrator", "WARM"])["tags"] == [
        "warm",
        "Narrator",
    ]
    assert store.set_tags("alice", remove=["WARM", "absent"])["tags"] == ["Narrator"]
    assert store.set_tags("alice", replace=["a", "b"])["tags"] == ["a", "b"]
    assert store.set_tags("alice", clear=True, add=["fresh"])["tags"] == ["fresh"]


def test_tag_set_conflicts_with_add_and_remove(tmp_path: Path):
    store = _store(tmp_path)
    _profile(store, "alice")
    with pytest.raises(UsageError) as error:
        store.set_tags("alice", replace=["x"], add=["y"])
    assert error.value.code == 2


def test_tag_rejects_invalid_value_without_writing(tmp_path: Path):
    store = _store(tmp_path)
    _profile(store, "alice", tags=["keep"])
    with pytest.raises(UsageError):
        store.set_tags("alice", add=[""])
    assert store.require("alice")["tags"] == ["keep"]


def test_tag_edit_refuses_symlinked_profile(tmp_path: Path):
    store = _store(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    meta = {"schema_version": 1, "name": "linked", "kind": "design", "tags": []}
    (outside / "voice.json").write_text(json.dumps(meta), encoding="utf-8")
    store.root.mkdir(parents=True, exist_ok=True)
    (store.root / "linked").symlink_to(outside)

    with pytest.raises(InputError) as error:
        store.set_tags("linked", add=["escaped"])
    assert error.value.code == 4
    unchanged = json.loads((outside / "voice.json").read_text(encoding="utf-8"))
    assert unchanged["tags"] == []


def test_tag_edit_refuses_broken_clone_reference(tmp_path: Path):
    store = _store(tmp_path)
    _profile(store, "clone", kind="clone")
    with pytest.raises(InputError) as error:
        store.set_tags("clone", add=["usable"])
    assert error.value.code == 4
    assert store.require("clone")["tags"] == []


def test_tag_edit_refuses_escaping_clone_reference(tmp_path: Path):
    store = _store(tmp_path)
    _profile(store, "clone", kind="clone")
    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"RIFF")
    (store.dir_for("clone") / "ref.wav").symlink_to(outside)
    with pytest.raises(InputError):
        store.set_tags("clone", add=["usable"])


def test_tag_dry_run_also_checks_clone_reference(tmp_path: Path):
    store = _store(tmp_path)
    _profile(store, "clone", kind="clone")
    with pytest.raises(InputError) as error:
        store.set_tags("clone", add=["usable"], dry_run=True)
    assert error.value.code == 4


def test_tag_edit_accepts_healthy_clone_reference(tmp_path: Path):
    store = _store(tmp_path)
    _profile(store, "clone", kind="clone")
    (store.dir_for("clone") / "ref.wav").write_bytes(b"RIFF")
    assert store.set_tags("clone", add=["usable"])["tags"] == ["usable"]


# --------------------------------------------------------------------------- #
# delete
# --------------------------------------------------------------------------- #
def test_delete_removes_profile_and_tombstone(tmp_path: Path):
    store = _store(tmp_path)
    _profile(store, "alice")
    result = store.delete("alice")
    assert result["dry_run"] is False
    assert not store.dir_for("alice").exists()
    assert not any(
        child.name.startswith(TOMBSTONE_PREFIX) for child in store.root.iterdir()
    )


def test_delete_dry_run_keeps_profile(tmp_path: Path):
    store = _store(tmp_path)
    _profile(store, "alice")
    result = store.delete("alice", dry_run=True)
    assert result["dry_run"] is True
    assert store.require("alice")["name"] == "alice"


def test_delete_corrupt_profile_by_directory_name(tmp_path: Path):
    store = _store(tmp_path)
    store.root.mkdir(parents=True, exist_ok=True)
    broken = store.root / "broken"
    broken.mkdir()
    (broken / "ref.wav").write_bytes(b"RIFF")
    store.delete("broken")
    assert not broken.exists()


def test_delete_refuses_symlinked_directory(tmp_path: Path):
    store = _store(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    store.root.mkdir(parents=True, exist_ok=True)
    (store.root / "linked").symlink_to(outside)
    with pytest.raises(InputError) as error:
        store.delete("linked")
    assert error.value.code == 4
    assert outside.is_dir()


def test_delete_refuses_symlinked_metadata(tmp_path: Path):
    store = _store(tmp_path)
    _profile(store, "alice")
    target = tmp_path / "voice.json"
    target.write_text('{"name": "alice"}', encoding="utf-8")
    meta = store.dir_for("alice") / "voice.json"
    meta.unlink()
    meta.symlink_to(target)
    with pytest.raises(InputError):
        store.delete("alice")


def test_delete_missing_profile_is_input_error(tmp_path: Path):
    with pytest.raises(InputError) as error:
        _store(tmp_path).delete("missing")
    assert error.value.code == 4


def test_delete_failure_keeps_tombstone_hint(tmp_path: Path, monkeypatch):
    import shutil

    store = _store(tmp_path)
    _profile(store, "alice")

    def boom(path, *args, **kwargs):
        raise OSError("disk busy")

    monkeypatch.setattr(shutil, "rmtree", boom)
    with pytest.raises(InputError) as error:
        store.delete("alice")
    assert error.value.code == 4
    tombstones = [
        child.name for child in store.root.iterdir()
        if child.name.startswith(TOMBSTONE_PREFIX)
    ]
    assert len(tombstones) == 1
    assert tombstones[0] in str(error.value) or tombstones[0] in json.dumps(
        error.value.details, ensure_ascii=False
    )


def test_remove_uses_ordered_delete(tmp_path: Path):
    store = _store(tmp_path)
    _profile(store, "alice")
    store.remove("alice")
    assert not store.dir_for("alice").exists()
