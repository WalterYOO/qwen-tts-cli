"""FR-P1-10: `.qttsvoice` export/import security and round trips."""

from __future__ import annotations

import hashlib
import json
import stat
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from safetensors.numpy import save_file

from qtts_cli.commands import voice as voice_cmd
from qtts_cli.config import Config
from qtts_cli.errors import InputError
from qtts_cli.voice_archive import export_voice, import_voice
from qtts_cli.voice_store import VoiceStore

WAV = b"RIFF\x00\x00\x00\x00WAVE"


def _prompt(path: Path) -> None:
    save_file({"ref_spk_embedding": np.zeros((1, 4), dtype="float32")}, str(path))


def _clone(store: VoiceStore, name: str = "alice") -> dict:
    staging = store.staging_dir(name)
    meta = store.create(
        name=name,
        kind="clone",
        source={
            "type": "clone",
            "mode": "icl",
            "ref_text": "你好",
            "ref_audio": "ref.wav",
            "ref_audio_sha256": hashlib.sha256(WAV).hexdigest(),
        },
        model={"id": "m", "revision": None},
        consent={"rights_confirmed": True, "license": "self-owned", "note": "consent"},
        staging=staging,
    )
    _prompt(staging / "prompt.safetensors")
    (staging / "ref.wav").write_bytes(WAV)
    store.commit_staging(staging, name)
    return meta


def _design(store: VoiceStore, name: str = "narrator") -> dict:
    staging = store.staging_dir(name)
    meta = store.create(
        name=name,
        kind="design",
        source={"type": "design", "description": "温柔", "sample_text": "你好"},
        model={"id": "design-model", "revision": None},
        consent={"rights_confirmed": True, "license": "ai-generated-design", "note": ""},
        staging=staging,
    )
    (staging / "preview.wav").write_bytes(WAV)
    store.commit_staging(staging, name)
    return meta


def _manifest(members: dict[str, bytes], name: str, kind: str) -> dict:
    return {
        "schema_version": 1,
        "qtts_version": "test",
        "created_at": "2026-01-01T00:00:00Z",
        "name": name,
        "kind": kind,
        "consent": {},
        "files": [
            {"path": key, "sha256": hashlib.sha256(value).hexdigest(), "size": len(value)}
            for key, value in members.items()
        ],
    }


def _archive(
    path: Path,
    members: dict[str, bytes],
    *,
    name: str = "alice",
    kind: str = "design",
    manifest: dict | None = None,
) -> Path:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "manifest.json", json.dumps(manifest or _manifest(members, name, kind))
        )
        for key, value in members.items():
            archive.writestr(key, value)
    return path


def _voice_meta(name: str = "alice", kind: str = "design") -> dict:
    return {
        "schema_version": 1,
        "name": name,
        "kind": kind,
        "created_at": "2026-01-01T00:00:00Z",
        "source": {"type": kind, "description": "温柔"},
        "model": {"id": "m", "revision": None},
        "consent": {"rights_confirmed": True, "license": "self-owned", "note": ""},
        "qtts_version": "test",
    }


def test_clone_round_trip_preserves_consent_and_payload(tmp_path: Path):
    store = VoiceStore(tmp_path / "source")
    _clone(store)
    archive = tmp_path / "alice.qttsvoice"
    export_voice(store, "alice", archive)
    assert archive.read_bytes()[:2] == b"PK"

    target = VoiceStore(tmp_path / "target")
    result = import_voice(target, archive)
    assert result["voice"]["name"] == "alice"
    assert result["voice"]["consent"]["license"] == "self-owned"
    assert result["voice"]["consent"]["note"] == "consent"
    assert (target.dir_for("alice") / "prompt.safetensors").is_file()
    assert (target.dir_for("alice") / "ref.wav").read_bytes() == WAV
    assert list((target.root).glob(".alice.backup.*")) == []


def test_design_round_trip_without_prompt(tmp_path: Path):
    store = VoiceStore(tmp_path / "source")
    _design(store)
    archive = tmp_path / "narrator.qttsvoice"
    export_voice(store, "narrator", archive)

    target = VoiceStore(tmp_path / "target")
    result = import_voice(target, archive)
    assert result["voice"]["kind"] == "design"
    assert (target.dir_for("narrator") / "preview.wav").is_file()
    assert not (target.dir_for("narrator") / "prompt.safetensors").exists()


def test_export_requires_prompt_for_clone(tmp_path: Path):
    store = VoiceStore(tmp_path / "source")
    staging = store.staging_dir("broken")
    store.create(
        name="broken",
        kind="clone",
        source={"type": "clone", "mode": "xvec"},
        model={"id": "m"},
        staging=staging,
    )
    store.commit_staging(staging, "broken")
    with pytest.raises(InputError, match="prompt.safetensors"):
        export_voice(store, "broken", tmp_path / "broken.qttsvoice")


def test_export_force_overwrites_atomically(tmp_path: Path):
    store = VoiceStore(tmp_path / "source")
    _design(store)
    archive = tmp_path / "narrator.qttsvoice"
    export_voice(store, "narrator", archive)
    with pytest.raises(InputError, match="already exists"):
        export_voice(store, "narrator", archive)
    export_voice(store, "narrator", archive, force=True)
    assert archive.is_file()
    assert list(tmp_path.glob(".narrator.qttsvoice.*.tmp")) == []


@pytest.mark.parametrize("member", ["../evil", "/absolute", "C:/drive", "a\\b"])
def test_import_rejects_traversal_and_absolute_members(tmp_path: Path, member: str):
    archive = _archive(
        tmp_path / "bad.qttsvoice",
        {"voice.json": json.dumps(_voice_meta()).encode(), member: b"x"},
    )
    with pytest.raises(InputError):
        import_voice(VoiceStore(tmp_path / "target"), archive)
    assert not (tmp_path / "evil").exists()


def test_import_rejects_duplicate_extra_and_non_regular_members(tmp_path: Path):
    voice = json.dumps(_voice_meta()).encode()
    duplicate = tmp_path / "duplicate.qttsvoice"
    with zipfile.ZipFile(duplicate, "w") as archive:
        archive.writestr(
            "manifest.json", json.dumps(_manifest({"voice.json": voice}, "alice", "design"))
        )
        archive.writestr("voice.json", voice)
        archive.writestr("voice.json", voice)
    with pytest.raises(InputError, match="duplicate"):
        import_voice(VoiceStore(tmp_path / "t1"), duplicate)

    extra = _archive(tmp_path / "extra.qttsvoice", {"voice.json": voice, "notes.txt": b"x"})
    with pytest.raises(InputError, match="unexpected archive member"):
        import_voice(VoiceStore(tmp_path / "t2"), extra)

    symlink = tmp_path / "symlink.qttsvoice"
    with zipfile.ZipFile(symlink, "w") as archive:
        archive.writestr(
            "manifest.json", json.dumps(_manifest({"voice.json": voice}, "alice", "design"))
        )
        info = zipfile.ZipInfo("voice.json")
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(info, voice)
    with pytest.raises(InputError, match="non-regular"):
        import_voice(VoiceStore(tmp_path / "t3"), symlink)


def test_import_rejects_sha_and_member_manifest_mismatch(tmp_path: Path):
    voice = json.dumps(_voice_meta()).encode()
    manifest = _manifest({"voice.json": voice}, "alice", "design")
    manifest["files"][0]["sha256"] = "0" * 64
    mismatch = _archive(
        tmp_path / "mismatch.qttsvoice", {"voice.json": voice}, manifest=manifest
    )
    with pytest.raises(InputError, match="sha256 mismatch"):
        import_voice(VoiceStore(tmp_path / "t1"), mismatch)

    missing_manifest = _manifest({"voice.json": voice, "ref.wav": WAV}, "alice", "design")
    missing = _archive(
        tmp_path / "missing.qttsvoice", {"voice.json": voice}, manifest=missing_manifest
    )
    with pytest.raises(InputError, match="do not exactly match"):
        import_voice(VoiceStore(tmp_path / "t2"), missing)


def test_import_rejects_non_zip_and_corrupt_zip_as_input_errors(tmp_path: Path):
    not_zip = tmp_path / "not.zip"
    not_zip.write_bytes(b"not a zip archive")
    with pytest.raises(InputError, match="invalid .qttsvoice archive"):
        import_voice(VoiceStore(tmp_path / "t1"), not_zip)

    voice = json.dumps(_voice_meta()).encode()
    archive = tmp_path / "corrupt.qttsvoice"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_STORED) as handle:
        handle.writestr(
            "manifest.json", json.dumps(_manifest({"voice.json": voice}, "alice", "design"))
        )
        handle.writestr("voice.json", voice)
    raw = bytearray(archive.read_bytes())
    marker = voice.index(b"alice")
    raw[raw.index(voice[marker : marker + 5])] ^= 0xFF
    archive.write_bytes(raw)
    with pytest.raises(InputError, match="invalid .qttsvoice archive"):
        import_voice(VoiceStore(tmp_path / "t2"), archive)


def test_import_rejects_non_string_manifest_paths(tmp_path: Path):
    voice = json.dumps(_voice_meta()).encode()
    manifest = _manifest({"voice.json": voice}, "alice", "design")
    manifest["files"][0]["path"] = []
    archive = _archive(tmp_path / "badpath.qttsvoice", {"voice.json": voice},
                       manifest=manifest)
    with pytest.raises(InputError, match="paths must be strings"):
        import_voice(VoiceStore(tmp_path / "t"), archive)


def test_import_rejects_invalid_prompt_and_leaves_no_partial_profile(tmp_path: Path):
    clone_meta = _voice_meta(kind="clone")
    clone_meta["source"] = {"mode": "xvec", "type": "clone"}
    members = {
        "voice.json": json.dumps(clone_meta).encode(),
        "prompt.safetensors": b"not safetensors",
    }
    manifest = _manifest(members, "alice", "clone")
    archive = _archive(tmp_path / "badprompt.qttsvoice", members, manifest=manifest)
    target = VoiceStore(tmp_path / "target")
    with pytest.raises(InputError, match="invalid prompt.safetensors"):
        import_voice(target, archive)
    assert not target.exists("alice")
    assert list(target.root.glob(".alice.*")) == []


def test_import_name_override_and_force_replacement(tmp_path: Path):
    store = VoiceStore(tmp_path / "source")
    _design(store)
    archive = tmp_path / "narrator.qttsvoice"
    export_voice(store, "narrator", archive)

    target = VoiceStore(tmp_path / "target")
    result = import_voice(target, archive, name="bob")
    assert result["voice"]["name"] == "bob"
    assert target.exists("bob")
    assert not target.exists("narrator")
    with pytest.raises(InputError, match="already exists"):
        import_voice(target, archive, name="bob")
    replaced = import_voice(target, archive, name="bob", force=True)
    assert replaced["replaced"] is True
    assert list(target.root.glob(".bob.backup.*")) == []


def test_materialize_clears_design_default_instruct(monkeypatch, tmp_path: Path):
    torch = pytest.importorskip("torch")

    class FakeEngine:
        def __init__(self, cfg, model):
            self.cfg = cfg
            self.model = None
            self.dtype_name = "fp32"

        def load(self):
            self.model = object()
            return self

        def generate(self, **kwargs):
            return np.zeros(2400, dtype=np.float32), 24000, {"generate_ms": 1.0}

        def create_clone_prompt_items(self, *args, **kwargs):
            item = SimpleNamespace(
                ref_spk_embedding=torch.zeros((1, 4), dtype=torch.float32),
                ref_code=None,
            )
            return [item]

    monkeypatch.setattr(voice_cmd, "TtsEngine", FakeEngine)
    monkeypatch.setattr(voice_cmd, "clear_model_cache", lambda: None)
    cfg = Config(data_dir=tmp_path / "data", model_dir=tmp_path / "models", seed=1)
    store = VoiceStore(cfg.data_dir)
    staging = store.staging_dir("designed")
    meta = store.create(
        name="designed",
        kind="design",
        source={"type": "design", "description": "温柔女声", "sample_text": "你好。"},
        model={"id": "design-model", "revision": None},
        default_instruct="温柔女声",
        staging=staging,
    )
    design = SimpleNamespace(model_id="design-model", revision=None,
                             fingerprint="f", path=tmp_path / "design")
    base = SimpleNamespace(model_id="base-model", revision=None,
                           fingerprint="b", path=tmp_path / "base")
    result = voice_cmd._materialize_design(
        cfg, store, staging, meta, "温柔女声", "你好。", "Chinese", design, base
    )
    assert result["voice"]["kind"] == "clone"
    assert result["voice"]["default_instruct"] is None
    on_disk = json.loads((staging / "voice.json").read_text(encoding="utf-8"))
    assert on_disk["kind"] == "clone"
    assert on_disk["default_instruct"] is None
