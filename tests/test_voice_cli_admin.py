"""Phase 2 voice CLI contracts: list/show/rename/tag/delete/preview/export."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from qtts_cli.cli import main
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
