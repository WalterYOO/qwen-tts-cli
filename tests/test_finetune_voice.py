"""FR-P2-07 §9.4/§12.9: voice commands and archive restrictions."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from qtts_cli.cli import main
from qtts_cli.commands import voice as voice_cmd
from qtts_cli.utils import sha256_file, sha256_text
from qtts_cli.voice_archive import ARCHIVE_SCHEMA_VERSION, import_voice
from qtts_cli.voice_store import VoiceStore
from finetune_utils import write_checkpoint, write_tts_model


def make_profile(tmp_path: Path, *, name: str = "alice") -> tuple[VoiceStore, Path]:
    checkpoint = write_checkpoint(tmp_path / "out", 0, name)
    weights_sha = sha256_file(checkpoint / "model.safetensors")
    config_sha = sha256_file(checkpoint / "config.json")
    checkpoint_sha = sha256_text(f"{weights_sha}:{config_sha}:{name}")
    store = VoiceStore(tmp_path / "data")
    store.create(
        name=name,
        kind="finetune",
        source={
            "type": "finetune",
            "speaker": name,
            "checkpoint": checkpoint.name,
            "weights_sha256": weights_sha,
            "weights_size": (checkpoint / "model.safetensors").stat().st_size,
            "config_sha256": config_sha,
            "checkpoint_sha256": checkpoint_sha,
            "output_dir": str(checkpoint.parent),
            "init_model": None,
        },
        model={
            "id": checkpoint.name,
            "revision": None,
            "fingerprint": checkpoint_sha[:16],
            "path": str(checkpoint),
        },
    )
    return store, checkpoint


def run(tmp_path: Path, *args: str) -> int:
    models = tmp_path / "models"
    if not models.is_dir():
        write_tts_model(
            models,
            "Qwen3-TTS-12Hz-0.6B-CustomVoice",
            "custom_voice",
            speakers={"vivian": 3065},
        )
    return main(
        [
            *args,
            "--model-dir",
            str(models),
            "--data-dir",
            str(tmp_path / "data"),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--no-cache",
        ]
    )


class FakeEngine:
    calls: list[dict] = []

    def __init__(self, cfg, model):
        self.cfg = cfg
        self.model_info = model
        self.device = "cpu"
        self.dtype_name = "fp32"
        self.load_ms = 0.0
        self.model = None

    def load(self):
        self.model = object()
        return self

    def generate(self, **kwargs):
        FakeEngine.calls.append(kwargs)
        return np.zeros(2400, dtype=np.float32), 24000, {"generate_ms": 1.0}

    def build_clone_prompt_items(self, *args, **kwargs):
        return []


def test_list_shows_finetune_without_opening_checkpoint(tmp_path: Path, capsys):
    store, checkpoint = make_profile(tmp_path)
    import shutil

    shutil.rmtree(checkpoint)
    assert run(tmp_path, "voice", "list", "--json") == 0
    payload = json.loads(capsys.readouterr().out)
    entry = next(v for v in payload["voices"] if v["name"] == "alice")
    assert entry["kind"] == "finetune"
    assert entry["model"] == "checkpoint-epoch-0"
    assert entry["has_prompt"] is False


def test_inspect_reports_checkpoint_metadata(tmp_path: Path, capsys):
    store, checkpoint = make_profile(tmp_path)
    assert run(tmp_path, "voice", "inspect", "alice", "--json") == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["checkpoint"] == {
        "path": str(checkpoint),
        "exists": True,
        "speaker": "alice",
        "fingerprint": store.get("alice")["model"]["fingerprint"],
    }

    import shutil

    shutil.rmtree(checkpoint)
    assert run(tmp_path, "voice", "inspect", "alice", "--json") == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["checkpoint"]["exists"] is False
    assert payload["checkpoint"]["path"] == str(checkpoint)


def test_preview_generates_builtin_with_canonical_speaker(tmp_path: Path, capsys, monkeypatch):
    make_profile(tmp_path)
    FakeEngine.calls = []
    monkeypatch.setattr(voice_cmd, "TtsEngine", FakeEngine)
    assert run(tmp_path, "voice", "preview", "alice", "--json") == 0
    payload = json.loads(capsys.readouterr().out)
    preview = tmp_path / "data" / "voices" / "alice" / "preview.wav"
    assert preview.is_file()
    assert payload["output"] == str(preview)
    call = FakeEngine.calls[-1]
    assert call["mode"] == "builtin"
    assert call["speaker"] == "alice"
    assert call["instruct"] is None
    assert call["language"] == "Chinese"
    assert call["prompt_items"] is None


def test_preview_missing_or_changed_checkpoint_is_4(tmp_path: Path, monkeypatch):
    store, checkpoint = make_profile(tmp_path)
    monkeypatch.setattr(voice_cmd, "TtsEngine", FakeEngine)
    weights = checkpoint / "model.safetensors"
    weights.write_bytes(weights.read_bytes() + b"x")
    assert run(tmp_path, "voice", "preview", "alice", "--json") == 4

    import shutil

    shutil.rmtree(checkpoint)
    assert run(tmp_path, "voice", "preview", "alice", "--json") == 4


def test_rm_removes_only_profile_and_reports_checkpoint(tmp_path: Path, capsys):
    store, checkpoint = make_profile(tmp_path)
    assert run(tmp_path, "voice", "rm", "alice", "--yes", "--json") == 0
    out = capsys.readouterr()
    payload = json.loads(out.out)
    assert payload["removed"] == "alice"
    assert payload["checkpoint"] == str(checkpoint)
    assert not store.dir_for("alice").exists()
    assert checkpoint.is_dir()

    assert run(tmp_path, "voice", "rm", "alice", "--yes") == 4


def test_rm_human_summary_mentions_retained_checkpoint(tmp_path: Path, capsys):
    _, checkpoint = make_profile(tmp_path)
    assert run(tmp_path, "voice", "rm", "alice", "--yes") == 0
    out = capsys.readouterr().out
    assert "checkpoint left at" in out
    assert str(checkpoint) in out


def test_export_finetune_is_4(tmp_path: Path):
    make_profile(tmp_path)
    out = tmp_path / "alice.qttsvoice"
    assert run(tmp_path, "voice", "export", "alice", "-o", str(out), "--json") == 4
    assert not out.exists()


def test_import_manifest_with_finetune_kind_is_4(tmp_path: Path):
    voice_bytes = b'{"name":"alice","kind":"finetune"}'
    manifest = {
        "schema_version": ARCHIVE_SCHEMA_VERSION,
        "name": "alice",
        "kind": "finetune",
        "files": [
            {
                "path": "voice.json",
                "sha256": hashlib.sha256(voice_bytes).hexdigest(),
                "size": len(voice_bytes),
            }
        ],
    }
    archive = tmp_path / "finetune.qttsvoice"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("manifest.json", json.dumps(manifest))
        handle.writestr("voice.json", voice_bytes)
    store = VoiceStore(tmp_path / "data")
    from qtts_cli.errors import InputError

    with pytest.raises(InputError) as exc:
        import_voice(store, archive)
    assert exc.value.code == 4
    assert "finetune" in exc.value.message
