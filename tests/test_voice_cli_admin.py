"""Phase 2 voice CLI contracts: list/show/rename/tag/delete/preview/export."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from qtts_cli.cli import main
from qtts_cli.errors import InputError
from qtts_cli.voice_archive import (
    BUNDLE_EXPORT_JOURNAL,
    BUNDLE_JOURNAL,
    _recover_bundle_journal,
    _recover_export,
    export_bundle,
    import_bundle,
)
from qtts_cli.voice_store import TOMBSTONE_PREFIX, VoiceStore


def _run(tmp_path: Path, model_dir: Path, *args: str) -> int:
    return main(
        [
            *args,
            "--model-dir",
            str(model_dir),
            "--data-dir",
            str(tmp_path / "data"),
            "--cache-dir",
            str(tmp_path / "cache"),
        ]
    )


def _store(tmp_path: Path) -> VoiceStore:
    return VoiceStore(tmp_path / "data")


def _profile(
    tmp_path: Path,
    name: str,
    *,
    kind: str = "design",
    tags=(),
    language: str = "Chinese",
) -> dict:
    store = _store(tmp_path)
    meta = store.create(
        name=name,
        kind=kind,
        source={"type": kind, "description": "synthetic"},
        model={},
        tags=list(tags),
        language=language,
    )
    if kind == "clone":
        (store.dir_for(name) / "ref.wav").write_bytes(b"RIFF")
    return meta


# --------------------------------------------------------------------------- #
# list / filter
# --------------------------------------------------------------------------- #
def test_list_keeps_legacy_keys_and_adds_new_ones(
    capsys, tmp_path: Path, model_dir: Path
):
    _profile(tmp_path, "alice", tags=["warm"], language="Chinese")
    assert _run(tmp_path, model_dir, "voice", "list", "--json") == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload["builtin"]) == 9
    assert payload["count"] == 10
    entry = payload["voices"][0]
    for key in (
        "name",
        "kind",
        "language",
        "created_at",
        "model",
        "materialized",
        "has_prompt",
    ):
        assert key in entry
    assert entry["tags"] == ["warm"]
    assert entry["source_kind"] == "design"
    assert payload["schema_version"] == 1
    assert payload["warnings"] == []


def test_list_query_modes(capsys, tmp_path: Path, model_dir: Path):
    _profile(tmp_path, "alice")
    for query, mode, expected in (
        ("ali", "fuzzy", 1),
        ("ali", "prefix", 1),
        ("ali", "exact", 0),
        ("alice", "exact", 1),
        ("VIV", "fuzzy", 1),  # builtin Vivian still matches a name query
    ):
        assert _run(
            tmp_path, model_dir, "voice", "list", query, "--match", mode, "--json"
        ) == 0
        payload = json.loads(capsys.readouterr().out)
        assert len(payload["voices"]) + len(payload["builtin"]) == expected


def test_list_tag_kind_and_language_filters(capsys, tmp_path: Path, model_dir: Path):
    _profile(tmp_path, "alice", tags=["warm", "narration"])
    _profile(tmp_path, "bob", tags=["warm"], kind="clone")
    _profile(tmp_path, "carol", tags=["calm"], language="English")

    assert _run(tmp_path, model_dir, "voice", "list", "--tag", "warm", "--json") == 0
    payload = json.loads(capsys.readouterr().out)
    assert {entry["name"] for entry in payload["voices"]} == {"alice", "bob"}
    assert payload["builtin"] == []

    assert _run(
        tmp_path, model_dir, "voice", "list", "--tag", "warm", "--tag", "narration",
        "--json",
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    assert {entry["name"] for entry in payload["voices"]} == {"alice"}

    assert _run(tmp_path, model_dir, "voice", "list", "--kind", "clone", "--json") == 0
    payload = json.loads(capsys.readouterr().out)
    assert {entry["name"] for entry in payload["voices"]} == {"bob"}

    assert _run(tmp_path, model_dir, "voice", "list", "--language", "en", "--json") == 0
    payload = json.loads(capsys.readouterr().out)
    assert {entry["name"] for entry in payload["voices"]} == {"carol"}
    assert payload["builtin"] == []

    assert _run(
        tmp_path, model_dir, "voice", "list", "--language", "zh",
        "--include-builtin", "--json",
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload["builtin"]) == 9
    assert {entry["name"] for entry in payload["voices"]} == {"alice", "bob"}


def test_list_names_only_and_json_conflict(capsys, tmp_path: Path, model_dir: Path):
    _profile(tmp_path, "alice")
    assert _run(tmp_path, model_dir, "voice", "list", "ali", "--names-only") == 0
    assert capsys.readouterr().out.strip() == "alice"

    assert _run(
        tmp_path, model_dir, "voice", "list", "ali", "--names-only", "--json"
    ) == 2
    assert "--names-only cannot be combined with --json" in capsys.readouterr().err


def test_list_empty_query_and_no_match(capsys, tmp_path: Path, model_dir: Path):
    assert _run(tmp_path, model_dir, "voice", "list", "") == 2
    assert "empty voice query" in capsys.readouterr().err

    assert _run(tmp_path, model_dir, "voice", "list", "zzz-no-such-voice") == 0
    assert "no matching voices." in capsys.readouterr().out


def test_list_warns_about_corrupt_and_tombstone(capsys, tmp_path: Path, model_dir: Path):
    store = _store(tmp_path)
    store.root.mkdir(parents=True, exist_ok=True)
    (store.root / "broken").mkdir()
    (store.root / f"{TOMBSTONE_PREFIX}crashed-abcd1234").mkdir()
    assert _run(tmp_path, model_dir, "voice", "list", "--json") == 0
    payload = json.loads(capsys.readouterr().out)
    assert any("corrupt" in warning for warning in payload["warnings"])
    assert any("tombstone" in warning for warning in payload["warnings"])


# --------------------------------------------------------------------------- #
# show / inspect
# --------------------------------------------------------------------------- #
def test_show_and_inspect_alias(capsys, tmp_path: Path, model_dir: Path):
    _profile(tmp_path, "alice", tags=["warm"])
    for command in ("show", "inspect"):
        assert _run(tmp_path, model_dir, "voice", command, "alice", "--json") == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["voice"]["name"] == "alice"
        assert payload["directory"] == "alice"
        assert {entry["name"] for entry in payload["files"]} == {"voice.json"}
        assert payload["warnings"] == []


def test_show_builtin(capsys, tmp_path: Path, model_dir: Path):
    assert _run(tmp_path, model_dir, "voice", "show", "Vivian", "--json") == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["kind"] == "builtin"
    assert payload["speaker"] == "Vivian"


def test_show_mismatch_is_exit_4_with_repair_hint(
    capsys, tmp_path: Path, model_dir: Path
):
    store = _store(tmp_path)
    _profile(tmp_path, "alice")
    path = store.dir_for("alice") / "voice.json"
    meta = json.loads(path.read_text(encoding="utf-8"))
    meta["name"] = "bob"
    path.write_text(json.dumps(meta), encoding="utf-8")

    assert _run(tmp_path, model_dir, "voice", "show", "alice") == 4
    assert "rename alice bob" in capsys.readouterr().err


def test_show_missing_profile_is_exit_4(capsys, tmp_path: Path, model_dir: Path):
    assert _run(tmp_path, model_dir, "voice", "show", "missing") == 4
    assert "voice not found" in capsys.readouterr().err


def test_show_clone_without_reference_is_exit_4(
    capsys, tmp_path: Path, model_dir: Path
):
    _profile(tmp_path, "clone", kind="clone")
    path = _store(tmp_path).dir_for("clone") / "ref.wav"
    path.unlink()
    assert _run(tmp_path, model_dir, "voice", "show", "clone") == 4
    assert "neither prompt.safetensors nor ref.wav" in capsys.readouterr().err


def test_show_clone_with_escaping_reference_is_exit_4(
    capsys, tmp_path: Path, model_dir: Path
):
    _profile(tmp_path, "clone", kind="clone")
    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"RIFF")
    ref = _store(tmp_path).dir_for("clone") / "ref.wav"
    ref.unlink()
    ref.symlink_to(outside)
    assert _run(tmp_path, model_dir, "voice", "show", "clone") == 4
    assert "symlink" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# rename
# --------------------------------------------------------------------------- #
def test_rename_cli_and_dry_run(capsys, tmp_path: Path, model_dir: Path):
    store = _store(tmp_path)
    _profile(tmp_path, "alice")
    assert _run(
        tmp_path, model_dir, "voice", "rename", "alice", "bob", "--dry-run", "--json"
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["dry_run"] is True
    assert store.require("alice")["name"] == "alice"

    assert _run(tmp_path, model_dir, "voice", "rename", "alice", "bob", "--json") == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["new"] == "bob"
    assert store.require("bob")["name"] == "bob"


def test_rename_existing_target_is_exit_2(capsys, tmp_path: Path, model_dir: Path):
    _profile(tmp_path, "alice")
    _profile(tmp_path, "bob")
    assert _run(tmp_path, model_dir, "voice", "rename", "alice", "bob") == 2
    assert "already exists" in capsys.readouterr().err


def test_rename_symlinked_source_is_exit_4(capsys, tmp_path: Path, model_dir: Path):
    store = _store(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "voice.json").write_text('{"name": "linked"}', encoding="utf-8")
    store.root.mkdir(parents=True, exist_ok=True)
    (store.root / "linked").symlink_to(outside)
    assert _run(tmp_path, model_dir, "voice", "rename", "linked", "moved") == 4
    assert "symlink" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# tag
# --------------------------------------------------------------------------- #
def test_tag_cli_lifecycle(capsys, tmp_path: Path, model_dir: Path):
    store = _store(tmp_path)
    _profile(tmp_path, "alice", tags=["warm"])
    assert _run(tmp_path, model_dir, "voice", "tag", "alice", "--add", "Narrator",
                 "--json") == 0
    assert json.loads(capsys.readouterr().out)["tags"] == ["warm", "Narrator"]

    assert _run(tmp_path, model_dir, "voice", "tag", "alice", "--remove", "WARM",
                 "--json") == 0
    assert json.loads(capsys.readouterr().out)["tags"] == ["Narrator"]

    assert _run(tmp_path, model_dir, "voice", "tag", "alice", "--set", "a", "--set", "b",
                 "--json") == 0
    assert json.loads(capsys.readouterr().out)["tags"] == ["a", "b"]

    assert _run(tmp_path, model_dir, "voice", "tag", "alice", "--clear", "--add", "c",
                 "--json") == 0
    assert json.loads(capsys.readouterr().out)["tags"] == ["c"]
    assert store.require("alice")["tags"] == ["c"]


def test_tag_list_only_and_conflicts(capsys, tmp_path: Path, model_dir: Path):
    _profile(tmp_path, "alice", tags=["warm"])
    assert _run(tmp_path, model_dir, "voice", "tag", "alice") == 0
    assert "alice: warm" in capsys.readouterr().out

    assert _run(tmp_path, model_dir, "voice", "tag", "alice", "--set", "x",
                 "--add", "y") == 2
    assert "--set cannot be combined" in capsys.readouterr().err

    assert _run(tmp_path, model_dir, "voice", "tag", "alice", "--add", " ") == 2
    assert "tag must not be empty" in capsys.readouterr().err


def test_tag_dry_run_does_not_write(capsys, tmp_path: Path, model_dir: Path):
    store = _store(tmp_path)
    _profile(tmp_path, "alice")
    assert _run(tmp_path, model_dir, "voice", "tag", "alice", "--add", "warm",
                 "--dry-run") == 0
    assert "would set" in capsys.readouterr().out
    assert store.require("alice")["tags"] == []


def test_tag_dry_run_surfaces_broken_clone(capsys, tmp_path: Path, model_dir: Path):
    store = _store(tmp_path)
    _profile(tmp_path, "clone", kind="clone")
    (store.dir_for("clone") / "ref.wav").unlink()
    assert _run(tmp_path, model_dir, "voice", "tag", "clone", "--add", "warm",
                 "--dry-run") == 4
    assert "neither prompt.safetensors nor ref.wav" in capsys.readouterr().err


def test_filter_alias_matches_list(capsys, tmp_path: Path, model_dir: Path):
    _profile(tmp_path, "alice", tags=["warm"])
    assert _run(tmp_path, model_dir, "voice", "filter", "--tag", "warm", "--json") == 0
    payload = json.loads(capsys.readouterr().out)
    assert [entry["name"] for entry in payload["voices"]] == ["alice"]


# --------------------------------------------------------------------------- #
# delete / rm
# --------------------------------------------------------------------------- #
def test_delete_requires_yes_in_non_tty(capsys, tmp_path: Path, model_dir: Path):
    _profile(tmp_path, "alice")
    assert _run(tmp_path, model_dir, "voice", "delete", "alice") == 2
    assert "requires --yes" in capsys.readouterr().err


def test_rm_keeps_legacy_input_error_semantics(
    capsys, tmp_path: Path, model_dir: Path
):
    _profile(tmp_path, "alice")
    assert _run(tmp_path, model_dir, "voice", "rm", "alice") == 4
    assert "requires --yes" in capsys.readouterr().err


def test_delete_dry_run_needs_no_confirmation(
    capsys, tmp_path: Path, model_dir: Path
):
    store = _store(tmp_path)
    _profile(tmp_path, "alice")
    assert _run(tmp_path, model_dir, "voice", "delete", "alice", "--dry-run") == 0
    assert "would delete" in capsys.readouterr().out
    assert store.require("alice")["name"] == "alice"


def test_delete_yes_and_dry_run(capsys, tmp_path: Path, model_dir: Path):
    store = _store(tmp_path)
    _profile(tmp_path, "alice")
    assert _run(tmp_path, model_dir, "voice", "delete", "alice", "--yes", "--dry-run") == 0
    assert "would delete" in capsys.readouterr().out
    assert store.require("alice")["name"] == "alice"

    assert _run(tmp_path, model_dir, "voice", "delete", "alice", "--yes") == 0
    assert "removed voice alice" in capsys.readouterr().out
    assert not store.dir_for("alice").exists()


def test_delete_corrupt_profile_by_directory_name(
    capsys, tmp_path: Path, model_dir: Path
):
    store = _store(tmp_path)
    store.root.mkdir(parents=True, exist_ok=True)
    (store.root / "broken").mkdir()
    assert _run(tmp_path, model_dir, "voice", "delete", "broken", "--yes") == 0
    assert not (store.root / "broken").exists()


# --------------------------------------------------------------------------- #
# preview constraints
# --------------------------------------------------------------------------- #
def test_preview_no_save_and_flac_constraints(
    capsys, tmp_path: Path, model_dir: Path
):
    store = _store(tmp_path)
    _profile(tmp_path, "alice")
    preview = store.dir_for("alice") / "preview.wav"

    assert _run(tmp_path, model_dir, "voice", "preview", "alice", "--no-save") == 2
    assert "--no-save requires --out" in capsys.readouterr().err

    assert _run(tmp_path, model_dir, "voice", "preview", "alice",
                 "--format", "flac") == 2
    assert "requires --out" in capsys.readouterr().err

    assert _run(
        tmp_path, model_dir, "voice", "preview", "alice",
        "--no-save", "--out", str(preview),
    ) == 2
    assert "cannot write the profile preview" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# export
# --------------------------------------------------------------------------- #
def test_preview_validates_format_before_loading_the_engine(
    capsys, tmp_path: Path, model_dir: Path, monkeypatch
):
    from qtts_cli.commands import voice as voice_cmd

    def boom(*args, **kwargs):
        raise AssertionError("engine must not be constructed for a usage error")

    monkeypatch.setattr(voice_cmd, "TtsEngine", boom)
    _profile(tmp_path, "alice")
    out = tmp_path / "alice.wav"
    assert _run(
        tmp_path, model_dir, "voice", "preview", "alice",
        "--out", str(out), "--format", "flac",
    ) == 2
    assert "does not match output suffix" in capsys.readouterr().err


def test_export_refuses_symlinked_reference(
    capsys, tmp_path: Path, model_dir: Path
):
    store = _store(tmp_path)
    _profile(tmp_path, "clone", kind="clone")
    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"RIFF")
    ref = store.dir_for("clone") / "ref.wav"
    ref.unlink()
    ref.symlink_to(outside)
    assert _run(
        tmp_path, model_dir, "voice", "export", "clone",
        "-o", str(tmp_path / "clone.qttsvoice"),
    ) == 4
    assert "symlink" in capsys.readouterr().err


def test_export_dry_run_respects_force(capsys, tmp_path: Path, model_dir: Path):
    _profile(tmp_path, "alice")
    archive = tmp_path / "alice.qttsvoice"
    archive.write_bytes(b"existing")
    assert _run(
        tmp_path, model_dir, "voice", "export", "alice", "-o", str(archive), "--dry-run"
    ) == 4
    assert "already exists" in capsys.readouterr().err
    assert _run(
        tmp_path, model_dir, "voice", "export", "alice", "-o", str(archive),
        "--dry-run", "--force",
    ) == 0
    assert "would export" in capsys.readouterr().out
    assert archive.read_bytes() == b"existing"


def test_export_dry_run_writes_nothing(capsys, tmp_path: Path, model_dir: Path):
    _profile(tmp_path, "alice")
    archive = tmp_path / "alice.qttsvoice"
    assert _run(
        tmp_path, model_dir, "voice", "export", "alice", "-o", str(archive), "--dry-run"
    ) == 0
    assert "would export" in capsys.readouterr().out
    assert not archive.exists()


def test_export_refuses_store_path(capsys, tmp_path: Path, model_dir: Path):
    store = _store(tmp_path)
    _profile(tmp_path, "alice")
    inside = store.root / "alice.qttsvoice"
    assert _run(
        tmp_path, model_dir, "voice", "export", "alice", "-o", str(inside)
    ) == 4
    assert "inside the voice store" in capsys.readouterr().err


def _run_with_data(
    tmp_path: Path, data_dir: Path, model_dir: Path, *args: str
) -> int:
    return main(
        [
            *args,
            "--model-dir",
            str(model_dir),
            "--data-dir",
            str(data_dir),
            "--cache-dir",
            str(tmp_path / "cache"),
        ]
    )


def test_export_all_and_import_bundle_roundtrip(
    capsys, tmp_path: Path, model_dir: Path
):
    import zipfile

    _profile(tmp_path, "alice")
    _profile(tmp_path, "bob", kind="design")
    bundle = tmp_path / "bundle"
    assert _run(
        tmp_path, model_dir, "voice", "export", "--all",
        "--out-dir", str(bundle), "--json",
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["out_dir"] == str(bundle)
    assert {entry["name"] for entry in payload["voices"]} == {"alice", "bob"}
    manifest = json.loads((bundle / "bundle.json").read_text())
    assert manifest["format"] == "qtts-voice-bundle"
    assert manifest["bundle_schema_version"] == 2
    assert {entry["archive"] for entry in manifest["voices"]} == {
        "alice.qttsvoice",
        "bob.qttsvoice",
    }
    with zipfile.ZipFile(bundle / "alice.qttsvoice") as archive:
        single = json.loads(archive.read("manifest.json"))
    assert single["schema_version"] == 1

    data2 = tmp_path / "data2"
    assert _run_with_data(
        tmp_path, data2, model_dir, "voice", "import-bundle", str(bundle), "--json"
    ) == 0
    imported = json.loads(capsys.readouterr().out)
    assert {entry["name"] for entry in imported["voices"]} == {"alice", "bob"}
    assert (data2 / "voices" / "alice" / "voice.json").is_file()
    assert (data2 / "voices" / "bob" / "voice.json").is_file()

    assert _run_with_data(
        tmp_path, data2, model_dir, "voice", "import-bundle", str(bundle)
    ) == 4
    assert "already exists" in capsys.readouterr().err
    assert _run_with_data(
        tmp_path, data2, model_dir, "voice", "import-bundle", str(bundle), "--force"
    ) == 0
    assert "replaced" in capsys.readouterr().out


def test_import_bundle_is_all_or_nothing_and_verifies_hashes(
    capsys, tmp_path: Path, model_dir: Path
):
    _profile(tmp_path, "alice")
    _profile(tmp_path, "bob")
    bundle = tmp_path / "bundle"
    assert _run(
        tmp_path, model_dir, "voice", "export", "--all", "--out-dir", str(bundle)
    ) == 0
    capsys.readouterr()
    with open(bundle / "bob.qttsvoice", "ab") as handle:
        handle.write(b"tampered")

    data2 = tmp_path / "data2"
    assert _run_with_data(
        tmp_path, data2, model_dir, "voice", "import-bundle", str(bundle)
    ) == 4
    assert "size mismatch" in capsys.readouterr().err
    assert not (data2 / "voices" / "alice").exists()
    assert not (data2 / "voices" / "bob").exists()


def test_export_all_rejects_unmanaged_files_and_dry_run_is_read_only(
    capsys, tmp_path: Path, model_dir: Path
):
    _profile(tmp_path, "alice")
    new_bundle = tmp_path / "new-bundle"
    assert _run(
        tmp_path, model_dir, "voice", "export", "--all",
        "--out-dir", str(new_bundle), "--dry-run",
    ) == 0
    assert "would export" in capsys.readouterr().out
    assert not new_bundle.exists()

    bundle = tmp_path / "bundle"
    assert _run(
        tmp_path, model_dir, "voice", "export", "--all", "--out-dir", str(bundle)
    ) == 0
    (bundle / "notes.txt").write_text("keep me")
    assert _run(
        tmp_path, model_dir, "voice", "export", "--all",
        "--out-dir", str(bundle), "--force",
    ) == 4
    assert "unmanaged files" in capsys.readouterr().err
    assert (bundle / "notes.txt").read_text() == "keep me"


def test_import_bundle_dry_run_installs_nothing(
    capsys, tmp_path: Path, model_dir: Path
):
    _profile(tmp_path, "alice")
    bundle = tmp_path / "bundle"
    assert _run(
        tmp_path, model_dir, "voice", "export", "--all", "--out-dir", str(bundle)
    ) == 0
    capsys.readouterr()

    data2 = tmp_path / "data2"
    assert _run_with_data(
        tmp_path, data2, model_dir,
        "voice", "import-bundle", str(bundle), "--dry-run", "--json",
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["dry_run"] is True
    assert payload["voices"] == [{"name": "alice", "replaced": False}]
    assert not (data2 / "voices").exists()


def test_export_bundle_rejects_casefold_collisions(tmp_path, monkeypatch):
    store = _store(tmp_path)
    meta = _profile(tmp_path, "alice")
    monkeypatch.setattr(
        VoiceStore,
        "scan_entries",
        lambda self: {
            "voices": [
                {"name": "Alice", "status": "ok", "meta": dict(meta, name="Alice")},
                {"name": "alice", "status": "ok", "meta": meta},
            ],
            "warnings": [],
        },
    )
    with pytest.raises(InputError, match="case-insensitively"):
        export_bundle(store, tmp_path / "bundle")
    assert not (tmp_path / "bundle").exists()


def test_import_bundle_rejects_casefold_collisions(tmp_path, monkeypatch):
    _profile(tmp_path, "alice")
    bundle = tmp_path / "bundle"
    export_bundle(_store(tmp_path), bundle)
    other = VoiceStore(tmp_path / "data2")
    monkeypatch.setattr(other, "existing_names", lambda: {"alice": "Alice"})
    with pytest.raises(InputError, match="case-insensitively"):
        import_bundle(other, bundle)
    assert not other.dir_for("alice").exists()


def test_export_recovery_restores_previous_archives(tmp_path: Path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    staging = bundle / ".bundle-stage-abc"
    backup = bundle / ".bundle-backup-abc"
    staging.mkdir()
    backup.mkdir()
    (bundle / "alice.qttsvoice").write_bytes(b"new")
    (backup / "alice.qttsvoice").write_bytes(b"old")
    (staging / "alice.qttsvoice").write_bytes(b"temp")
    (bundle / BUNDLE_EXPORT_JOURNAL).write_text(json.dumps({
        "state": "publishing",
        "staging_dir": staging.name,
        "backup_dir": backup.name,
        "archives": ["alice.qttsvoice"],
    }))
    _recover_export(bundle)
    assert (bundle / "alice.qttsvoice").read_bytes() == b"old"
    assert not staging.exists()
    assert not backup.exists()
    assert not (bundle / BUNDLE_EXPORT_JOURNAL).exists()


def test_export_recovery_keeps_committed_archives(tmp_path: Path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    backup = bundle / ".bundle-backup-abc"
    backup.mkdir()
    (bundle / "alice.qttsvoice").write_bytes(b"new")
    (backup / "alice.qttsvoice").write_bytes(b"old")
    (bundle / BUNDLE_EXPORT_JOURNAL).write_text(json.dumps({
        "state": "committed",
        "backup_dir": backup.name,
        "archives": ["alice.qttsvoice"],
    }))
    _recover_export(bundle)
    assert (bundle / "alice.qttsvoice").read_bytes() == b"new"
    assert not backup.exists()
    assert not (bundle / BUNDLE_EXPORT_JOURNAL).exists()


def test_export_recovery_rejects_out_of_bounds_journal_paths(tmp_path: Path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep.txt").write_text("keep")
    (bundle / BUNDLE_EXPORT_JOURNAL).write_text(json.dumps({
        "state": "publishing",
        "staging_dir": "../outside",
        "backup_dir": "../outside",
        "archives": ["../outside/keep.txt"],
    }))
    _recover_export(bundle)
    assert (outside / "keep.txt").read_text() == "keep"
    assert not (bundle / BUNDLE_EXPORT_JOURNAL).exists()


def test_import_recovery_discards_corrupt_journal(tmp_path: Path):
    store = _store(tmp_path)
    _profile(tmp_path, "alice")
    (store.root / BUNDLE_JOURNAL).write_text("{not json")
    _recover_bundle_journal(store)
    assert not (store.root / BUNDLE_JOURNAL).exists()
    assert store.exists("alice")


def test_import_recovery_ignores_out_of_bounds_backup_dir(tmp_path: Path):
    store = _store(tmp_path)
    _profile(tmp_path, "alice")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep.txt").write_text("keep")
    for backup_dir in ("", "..", "../outside", "/tmp/not-a-bundle-backup"):
        (store.root / BUNDLE_JOURNAL).write_text(json.dumps({
            "backup_dir": backup_dir,
            "entries": [{"target": "alice", "installed": False}],
        }))
        _recover_bundle_journal(store)
        assert store.exists("alice")
        assert not (store.root / BUNDLE_JOURNAL).exists()
    assert (outside / "keep.txt").read_text() == "keep"


def test_import_recovery_restores_backup_after_crash_before_replace(tmp_path: Path):
    store = _store(tmp_path)
    _profile(tmp_path, "alice")
    (store.dir_for("alice") / "marker").write_text("new")
    backup = store.root / ".bundle-backup-abc"
    (backup / "alice").mkdir(parents=True)
    (backup / "alice" / "marker").write_text("old")
    store.dir_for("alice").rename(store.root / "alice-old")
    (store.root / BUNDLE_JOURNAL).write_text(json.dumps({
        "backup_dir": backup.name,
        "entries": [{"target": "alice", "installed": True}],
    }))
    _recover_bundle_journal(store)
    assert (store.dir_for("alice") / "marker").read_text() == "old"
    assert not backup.exists()
    assert not (store.root / BUNDLE_JOURNAL).exists()


def test_import_recovery_rolls_back_installed_target(tmp_path: Path):
    store = _store(tmp_path)
    _profile(tmp_path, "alice")
    (store.dir_for("alice") / "marker").write_text("new")
    backup = store.root / ".bundle-backup-abc"
    (backup / "alice").mkdir(parents=True)
    (backup / "alice" / "marker").write_text("old")
    (store.root / BUNDLE_JOURNAL).write_text(json.dumps({
        "backup_dir": backup.name,
        "entries": [{"target": "alice", "installed": True}],
    }))
    _recover_bundle_journal(store)
    assert (store.dir_for("alice") / "marker").read_text() == "old"
    assert not (store.root / BUNDLE_JOURNAL).exists()


def test_import_recovery_keeps_committed_journal_installs(tmp_path: Path):
    store = _store(tmp_path)
    _profile(tmp_path, "alice")
    (store.dir_for("alice") / "marker").write_text("new")
    backup = store.root / ".bundle-backup-abc"
    (backup / "alice").mkdir(parents=True)
    (backup / "alice" / "marker").write_text("old")
    (store.root / BUNDLE_JOURNAL).write_text(json.dumps({
        "state": "committed",
        "backup_dir": backup.name,
        "entries": [{"target": "alice", "installed": True}],
    }))
    _recover_bundle_journal(store)
    assert (store.dir_for("alice") / "marker").read_text() == "new"
    assert not backup.exists()
    assert not (store.root / BUNDLE_JOURNAL).exists()


def test_import_recovery_refuses_to_remove_without_a_backup(tmp_path: Path):
    store = _store(tmp_path)
    _profile(tmp_path, "alice")
    (store.dir_for("alice") / "marker").write_text("new")
    journal = store.root / BUNDLE_JOURNAL
    journal.write_text(json.dumps({
        "backup_dir": "../outside",
        "entries": [{"target": "alice", "installed": True}],
    }))
    with pytest.raises(InputError, match="manual recovery"):
        _recover_bundle_journal(store)
    assert (store.dir_for("alice") / "marker").read_text() == "new"
    assert journal.exists()


def test_import_recovery_refuses_when_backup_directory_is_missing(tmp_path: Path):
    store = _store(tmp_path)
    _profile(tmp_path, "alice")
    (store.dir_for("alice") / "marker").write_text("new")
    journal = store.root / BUNDLE_JOURNAL
    journal.write_text(json.dumps({
        "backup_dir": ".bundle-backup-missing",
        "entries": [{"target": "alice", "installed": True}],
    }))
    with pytest.raises(InputError, match="manual recovery"):
        _recover_bundle_journal(store)
    assert (store.dir_for("alice") / "marker").read_text() == "new"
    assert journal.exists()


def test_import_recovery_is_idempotent_after_restore(tmp_path: Path):
    store = _store(tmp_path)
    _profile(tmp_path, "alice")
    (store.dir_for("alice") / "marker").write_text("old")
    journal = store.root / BUNDLE_JOURNAL
    journal.write_text(json.dumps({
        "backup_dir": ".bundle-backup-consumed",
        "entries": [{
            "target": "alice",
            "installed": True,
            "replaced": True,
            "restoring": True,
        }],
    }))
    _recover_bundle_journal(store)
    assert (store.dir_for("alice") / "marker").read_text() == "old"
    assert not journal.exists()

    journal.write_text(json.dumps({
        "backup_dir": ".bundle-backup-consumed",
        "entries": [{
            "target": "alice",
            "installed": False,
            "replaced": True,
            "restored": True,
        }],
    }))
    _recover_bundle_journal(store)
    assert (store.dir_for("alice") / "marker").read_text() == "old"
    assert not journal.exists()


def test_import_recovery_restores_after_crash_before_installed_flag(tmp_path: Path):
    store = _store(tmp_path)
    _profile(tmp_path, "alice")
    store.dir_for("alice").rename(store.root / "alice-moved")
    backup = store.root / ".bundle-backup-abc"
    (backup / "alice").mkdir(parents=True)
    (backup / "alice" / "marker").write_text("old")
    (store.root / BUNDLE_JOURNAL).write_text(json.dumps({
        "backup_dir": backup.name,
        "entries": [{"target": "alice", "installed": False, "replaced": True}],
    }))
    _recover_bundle_journal(store)
    assert (store.dir_for("alice") / "marker").read_text() == "old"
    assert not (store.root / BUNDLE_JOURNAL).exists()


def test_import_recovery_rolls_back_new_voice_without_backup(tmp_path: Path):
    store = _store(tmp_path)
    _profile(tmp_path, "alice")
    (store.dir_for("alice") / "marker").write_text("new")
    (store.root / BUNDLE_JOURNAL).write_text(json.dumps({
        "backup_dir": ".bundle-backup-gone",
        "entries": [{"target": "alice", "installed": True, "replaced": False}],
    }))
    _recover_bundle_journal(store)
    assert not store.dir_for("alice").exists()
    assert not (store.root / BUNDLE_JOURNAL).exists()


def test_import_recovery_rejects_malformed_journal(tmp_path: Path):
    store = _store(tmp_path)
    _profile(tmp_path, "alice")
    (store.dir_for("alice") / "marker").write_text("new")
    backup = store.root / ".bundle-backup-abc"
    (backup / "alice").mkdir(parents=True)
    (backup / "alice" / "marker").write_text("old")
    journal = store.root / BUNDLE_JOURNAL

    for entries in (
        {"alice": {"installed": True}},
        [{"target": "../escape", "installed": True}],
        [{"target": "alice", "installed": True}, "junk"],
    ):
        journal.write_text(json.dumps({
            "backup_dir": backup.name,
            "entries": entries,
        }))
        with pytest.raises(InputError, match="journal"):
            _recover_bundle_journal(store)
        assert backup.is_dir()
        assert journal.exists()
        assert (store.dir_for("alice") / "marker").read_text() == "new"


def test_export_recovery_keeps_archives_when_backup_never_started(tmp_path: Path):
    bundle = tmp_path / "bundle"
    (bundle / ".bundle-stage-abc" / "candidates").mkdir(parents=True)
    (bundle / ".bundle-stage-abc" / "alice.qttsvoice").write_bytes(b"temp")
    (bundle / "alice.qttsvoice").write_bytes(b"old")
    (bundle / BUNDLE_EXPORT_JOURNAL).write_text(json.dumps({
        "state": "publishing",
        "managed": True,
        "backed_up": False,
        "staging_dir": ".bundle-stage-abc",
        "backup_dir": ".bundle-backup-abc",
        "archives": ["alice.qttsvoice"],
    }))
    _recover_export(bundle)
    assert (bundle / "alice.qttsvoice").read_bytes() == b"old"
    assert not (bundle / ".bundle-stage-abc").exists()
    assert not (bundle / BUNDLE_EXPORT_JOURNAL).exists()


def test_export_recovery_restores_partial_backup(tmp_path: Path):
    bundle = tmp_path / "bundle"
    backup = bundle / ".bundle-backup-abc"
    backup.mkdir(parents=True)
    (backup / "alice.qttsvoice").write_bytes(b"old")
    (bundle / BUNDLE_EXPORT_JOURNAL).write_text(json.dumps({
        "state": "publishing",
        "managed": True,
        "backed_up": False,
        "backup_dir": backup.name,
        "archives": ["alice.qttsvoice"],
    }))
    _recover_export(bundle)
    assert (bundle / "alice.qttsvoice").read_bytes() == b"old"
    assert not backup.exists()
    assert not (bundle / BUNDLE_EXPORT_JOURNAL).exists()


def test_export_recovery_refuses_incomplete_backup(tmp_path: Path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "alice.qttsvoice").write_bytes(b"old")
    journal = bundle / BUNDLE_EXPORT_JOURNAL
    journal.write_text(json.dumps({
        "state": "publishing",
        "managed": True,
        "backed_up": True,
        "backed_up_files": ["alice.qttsvoice", "bundle.json"],
        "backup_dir": ".bundle-backup-abc",
        "archives": ["alice.qttsvoice"],
    }))
    with pytest.raises(InputError, match="inconsistent"):
        _recover_export(bundle)
    assert (bundle / "alice.qttsvoice").read_bytes() == b"old"
    assert journal.exists()


def test_export_recovery_restores_previous_manifest(tmp_path: Path):
    bundle = tmp_path / "bundle"
    backup = bundle / ".bundle-backup-abc"
    backup.mkdir(parents=True)
    (backup / "alice.qttsvoice").write_bytes(b"old-audio")
    (backup / "bundle.json").write_text('{"old": true}')
    (bundle / "alice.qttsvoice").write_bytes(b"new-audio")
    (bundle / "bundle.json").write_text('{"new": true}')
    (bundle / BUNDLE_EXPORT_JOURNAL).write_text(json.dumps({
        "state": "publishing",
        "managed": True,
        "backed_up": True,
        "backed_up_files": ["alice.qttsvoice", "bundle.json"],
        "backup_dir": backup.name,
        "archives": ["alice.qttsvoice"],
    }))
    _recover_export(bundle)
    assert (bundle / "alice.qttsvoice").read_bytes() == b"old-audio"
    assert (bundle / "bundle.json").read_text() == '{"old": true}'
    assert not backup.exists()
    assert not (bundle / BUNDLE_EXPORT_JOURNAL).exists()


def test_export_rollback_removes_first_manifest(tmp_path: Path, monkeypatch):
    from qtts_cli import voice_archive as archive_mod

    store = _store(tmp_path)
    _profile(tmp_path, "alice")
    bundle = tmp_path / "bundle"
    real_write = archive_mod.atomic_write_json

    def fail_committed(path, data):
        if str(path).endswith(BUNDLE_EXPORT_JOURNAL) and data.get("state") == "committed":
            raise KeyboardInterrupt
        return real_write(path, data)

    monkeypatch.setattr(archive_mod, "atomic_write_json", fail_committed)
    with pytest.raises(KeyboardInterrupt):
        export_bundle(store, bundle)
    assert not (bundle / "bundle.json").exists()
    assert not (bundle / "alice.qttsvoice").exists()
    assert not list(bundle.glob(".bundle-*"))


def test_import_recovery_keeps_backup_when_cleanup_fails(tmp_path: Path, monkeypatch):
    from qtts_cli import voice_archive as archive_mod

    store = _store(tmp_path)
    _profile(tmp_path, "alice")
    (store.dir_for("alice") / "marker").write_text("new")
    backup = store.root / ".bundle-backup-abc"
    (backup / "alice").mkdir(parents=True)
    (backup / "alice" / "marker").write_text("old")
    (store.root / BUNDLE_JOURNAL).write_text(json.dumps({
        "backup_dir": backup.name,
        "entries": [{"target": "alice", "installed": True}],
    }))

    real_rmtree = archive_mod.shutil.rmtree

    def fail_target(path, *args, **kwargs):
        if str(path).endswith("alice"):
            raise OSError("permission denied")
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(archive_mod.shutil, "rmtree", fail_target)
    with pytest.raises(InputError, match="manual recovery"):
        _recover_bundle_journal(store)
    assert backup.is_dir()
    assert (store.root / BUNDLE_JOURNAL).exists()


def test_export_recovery_resumes_after_partial_restore(tmp_path: Path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "alice.qttsvoice").write_bytes(b"old-audio")
    (bundle / "bundle.json").write_text('{"old": true}')
    (bundle / BUNDLE_EXPORT_JOURNAL).write_text(json.dumps({
        "state": "publishing",
        "managed": True,
        "backed_up": True,
        "backed_up_files": ["alice.qttsvoice", "bundle.json"],
        "restoring_files": ["alice.qttsvoice", "bundle.json"],
        "restored_files": [],
        "backup_dir": ".bundle-backup-gone",
        "archives": ["alice.qttsvoice"],
    }))
    _recover_export(bundle)
    assert (bundle / "alice.qttsvoice").read_bytes() == b"old-audio"
    assert (bundle / "bundle.json").read_text() == '{"old": true}'
    assert not (bundle / BUNDLE_EXPORT_JOURNAL).exists()


def test_import_bundle_rolls_back_from_journal_on_failure(tmp_path: Path, monkeypatch):
    _profile(tmp_path, "alice")
    _profile(tmp_path, "bob")
    bundle = tmp_path / "bundle"
    export_bundle(_store(tmp_path), bundle)

    from qtts_cli import voice_archive as archive_mod

    real_write = archive_mod.atomic_write_json

    def fail_second_install(path, data):
        entries = data.get("entries") if isinstance(data, dict) else None
        if (
            str(path).endswith(BUNDLE_JOURNAL)
            and isinstance(entries, list)
            and len(entries) > 1
            and entries[1].get("installed")
        ):
            raise KeyboardInterrupt
        return real_write(path, data)

    monkeypatch.setattr(archive_mod, "atomic_write_json", fail_second_install)
    other = VoiceStore(tmp_path / "data2")
    with pytest.raises(KeyboardInterrupt):
        import_bundle(other, bundle)
    assert not other.dir_for("alice").exists()
    assert not other.dir_for("bob").exists()
    assert not (other.root / BUNDLE_JOURNAL).exists()
    assert not list(other.root.glob(".bundle-backup-*"))


def test_export_dry_run_does_not_recover(tmp_path: Path):
    store = _store(tmp_path)
    _profile(tmp_path, "alice")
    bundle = tmp_path / "bundle"
    export_bundle(store, bundle)
    backup = bundle / ".bundle-backup-stale"
    backup.mkdir()
    (backup / "alice.qttsvoice").write_bytes(b"old")
    journal = bundle / BUNDLE_EXPORT_JOURNAL
    journal.write_text(json.dumps({
        "state": "publishing",
        "managed": True,
        "backed_up": False,
        "backup_dir": backup.name,
        "archives": ["alice.qttsvoice"],
    }))
    payload = export_bundle(store, bundle, force=True, dry_run=True)
    assert payload["dry_run"] is True
    assert journal.is_file()
    assert backup.is_dir()
    assert (bundle / "alice.qttsvoice").is_file()


def test_import_dry_run_skips_recovery_and_writes_nothing(tmp_path: Path):
    _profile(tmp_path, "alice")
    bundle = tmp_path / "bundle"
    export_bundle(_store(tmp_path), bundle)
    other = VoiceStore(tmp_path / "data2")
    journal = other.root / BUNDLE_JOURNAL
    other.root.mkdir(parents=True, exist_ok=True)
    journal.write_text(json.dumps({
        "backup_dir": "../outside",
        "entries": [{"target": "alice", "installed": True}],
    }))
    payload = import_bundle(other, bundle, dry_run=True)
    assert payload["dry_run"] is True
    assert journal.exists()
    assert not other.dir_for("alice").exists()


def test_import_bundle_rejects_name_and_kind_mismatch(tmp_path: Path):
    store = _store(tmp_path)
    _profile(tmp_path, "alice")
    bundle = tmp_path / "bundle"
    export_bundle(store, bundle)
    manifest = json.loads((bundle / "bundle.json").read_text())
    manifest["voices"][0]["name"] = "bob"
    (bundle / "bundle.json").write_text(json.dumps(manifest))
    other = VoiceStore(tmp_path / "data2")
    with pytest.raises(InputError, match="name does not match"):
        import_bundle(other, bundle)

    manifest["voices"][0]["name"] = "alice"
    manifest["voices"][0]["kind"] = "clone"
    (bundle / "bundle.json").write_text(json.dumps(manifest))
    with pytest.raises(InputError, match="kind does not match"):
        import_bundle(other, bundle)


def test_voice_add_rejects_casefold_collision(
    capsys, tmp_path: Path, model_dir: Path, monkeypatch
):
    monkeypatch.setattr(
        VoiceStore, "existing_names", lambda self: {"alice": "Alice"}
    )
    assert _run(
        tmp_path, model_dir, "voice", "add", "alice",
        "--kind", "design", "--description", "温柔",
    ) == 4
    assert "case-insensitively" in capsys.readouterr().err
