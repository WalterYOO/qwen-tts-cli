"""FR-P2-07 §9.2/§12.8: finetune profile resolution and say integration."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from qtts_cli.capabilities import parse_voice_ref
from qtts_cli.cli import main
from qtts_cli.commands import say as say_cmd
from qtts_cli.errors import InputError
from qtts_cli.plan import resolve_voice
from qtts_cli.utils import sha256_file, sha256_text
from qtts_cli.voice_store import VoiceStore
from finetune_utils import write_checkpoint, write_tts_model


def make_checkpoint(tmp_path: Path, *, size: str = "0b6", name: str = "checkpoint-epoch-0") -> Path:
    checkpoint = write_checkpoint(tmp_path / "out", 0, "alice", name=name)
    config_path = checkpoint / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["tts_model_size"] = size
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return checkpoint


def make_profile(
    tmp_path: Path,
    *,
    name: str = "alice",
    size: str = "0b6",
    checkpoint: Path | None = None,
    speaker: str = "alice",
    checkpoint_sha: str | None = None,
    weights_size: int | None = None,
    config_sha: str | None = None,
) -> tuple[VoiceStore, Path, dict]:
    checkpoint = checkpoint or make_checkpoint(tmp_path, size=size)
    weights_sha = sha256_file(checkpoint / "model.safetensors")
    config_hash = sha256_file(checkpoint / "config.json")
    checkpoint_sha = checkpoint_sha or sha256_text(f"{weights_sha}:{config_hash}:{speaker}")
    store = VoiceStore(tmp_path / "data")
    meta = store.create(
        name=name,
        kind="finetune",
        source={
            "type": "finetune",
            "speaker": speaker,
            "checkpoint": checkpoint.name,
            "weights_sha256": weights_sha,
            "weights_size": weights_size
            if weights_size is not None
            else (checkpoint / "model.safetensors").stat().st_size,
            "config_sha256": config_sha if config_sha is not None else config_hash,
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
    return store, checkpoint, meta


def resolve(store: VoiceStore, name: str = "alice"):
    return resolve_voice([], store, parse_voice_ref(f"voice:{name}"))


def test_resolve_returns_builtin_canonical_speaker(tmp_path: Path):
    store, checkpoint, meta = make_profile(tmp_path)
    resolution = resolve(store)
    assert resolution.mode == "builtin"
    assert resolution.voice_name == "alice"
    assert resolution.speaker == "alice"
    assert resolution.model.kind == "builtin"
    assert resolution.model.tts_model_type == "custom_voice"
    assert resolution.model.path == checkpoint
    assert resolution.model.fingerprint == meta["source"]["checkpoint_sha256"][:16]
    assert resolution.identity["type"] == "finetune"
    assert resolution.identity["voice"] == "alice"
    assert resolution.identity["speaker"] == "alice"
    assert (
        resolution.identity["checkpoint_sha256"]
        == meta["source"]["checkpoint_sha256"]
    )


def test_missing_path_config_and_speaker_are_4(tmp_path: Path):
    store, checkpoint, meta = make_profile(tmp_path)
    (checkpoint / "config.json").unlink()
    with pytest.raises(InputError) as exc:
        resolve(store)
    assert exc.value.code == 4

    store2, checkpoint2, _ = make_profile(tmp_path / "two", name="bob")
    meta2 = json.loads((store2.dir_for("bob") / "voice.json").read_text())
    meta2["model"]["path"] = str(tmp_path / "gone")
    (store2.dir_for("bob") / "voice.json").write_text(json.dumps(meta2))
    with pytest.raises(InputError):
        resolve(store2, "bob")

    store3, _, _ = make_profile(tmp_path / "three", name="carol", speaker="nobody")
    with pytest.raises(InputError) as exc:
        resolve(store3, "carol")
    assert "speaker" in exc.value.message


def test_wrong_model_kind_is_4(tmp_path: Path):
    checkpoint = write_checkpoint(tmp_path / "out", 0, "alice")
    config_path = checkpoint / "config.json"
    config = json.loads(config_path.read_text())
    config["tts_model_type"] = "base"
    config_path.write_text(json.dumps(config))
    store, _, _ = make_profile(tmp_path, checkpoint=checkpoint)
    with pytest.raises(InputError) as exc:
        resolve(store)
    assert exc.value.code == 4
    assert "custom_voice" in exc.value.message


def test_weight_size_and_config_hash_mismatch_is_4_with_hint(tmp_path: Path):
    store, checkpoint, _ = make_profile(tmp_path)
    weights = checkpoint / "model.safetensors"
    weights.write_bytes(weights.read_bytes() + b"x")
    with pytest.raises(InputError) as exc:
        resolve(store)
    assert exc.value.code == 4
    assert "re-register" in exc.value.message
    assert "finetune train --register-only --force" in exc.value.message

    store2, checkpoint2, _ = make_profile(tmp_path / "two", name="bob")
    config = checkpoint2 / "config.json"
    config.write_text(config.read_text().replace('"0b6"', '"0b5"'))
    with pytest.raises(InputError) as exc:
        resolve(store2, "bob")
    assert "re-register" in exc.value.message


def test_model_override_is_4(tmp_path: Path):
    store, _, _ = make_profile(tmp_path)
    models = [write_tts_model(tmp_path / "models", "Qwen3-TTS-12Hz-0.6B-CustomVoice", "custom_voice")]
    from qtts_cli.catalog import load_model

    base = load_model(models[0])
    with pytest.raises(InputError) as exc:
        resolve_voice(
            [base],
            store,
            parse_voice_ref("voice:alice"),
            model_override="Qwen3-TTS-12Hz-0.6B-CustomVoice",
        )
    assert exc.value.code == 4
    assert "bound to its finetune checkpoint" in exc.value.message


def test_cache_identity_changes_with_checkpoint_hash(tmp_path: Path):
    checkpoint = make_checkpoint(tmp_path)
    store_a, _, _ = make_profile(
        tmp_path / "a", name="one", checkpoint=checkpoint, checkpoint_sha="a" * 64
    )
    store_b, _, _ = make_profile(
        tmp_path / "b", name="two", checkpoint=checkpoint, checkpoint_sha="b" * 64
    )
    first = resolve(store_a, "one")
    second = resolve(store_b, "two")
    assert first.model.fingerprint != second.model.fingerprint
    assert (
        first.identity["checkpoint_sha256"] != second.identity["checkpoint_sha256"]
    )


class FakeEngine:
    instances: list["FakeEngine"] = []

    def __init__(self, cfg, model):
        self.cfg = cfg
        self.model_info = model
        self.device = "cpu"
        self.dtype_name = "fp32"
        self.load_ms = 0.0
        self.model = None
        self.calls: list[dict] = []
        FakeEngine.instances.append(self)

    def load(self):
        self.model = object()
        return self

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        assert kwargs["speaker"] == "alice"
        return np.zeros(2400, dtype=np.float32), 24000, {"generate_ms": 1.0}

    def build_clone_prompt_items(self, *args, **kwargs):
        return []


def _run(tmp_path: Path, *args: str) -> int:
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
            str(tmp_path / "models"),
            "--data-dir",
            str(tmp_path / "data"),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--no-cache",
        ]
    )


def test_say_dry_run_resolves_profile(tmp_path: Path, capsys):
    store, _, _ = make_profile(tmp_path)
    assert _run(tmp_path, "say", "你好", "--voice", "voice:alice", "--dry-run", "--json") == 0
    out = capsys.readouterr()
    payload = json.loads(out.out)
    voice = payload["plan"]["voice"]
    assert voice["mode"] == "builtin"
    assert voice["speaker"] == "alice"
    assert voice["voice_name"] == "alice"
    assert voice["model_fingerprint"] == store.get("alice")["model"]["fingerprint"]


def test_say_with_fake_engine_succeeds(tmp_path: Path, monkeypatch):
    make_profile(tmp_path)
    FakeEngine.instances = []
    monkeypatch.setattr(say_cmd, "TtsEngine", FakeEngine)
    output = tmp_path / "said.wav"
    code = _run(tmp_path, "say", "你好", "--voice", "voice:alice", "--out", str(output))
    assert code == 0
    assert output.is_file()
    assert FakeEngine.instances[-1].calls[-1]["speaker"] == "alice"


def test_script_segment_resolves_finetune_profile(tmp_path: Path, capsys):
    make_profile(tmp_path)
    script = tmp_path / "script.yaml"
    script.write_text(
        "\n".join(
            [
                "schema_version: 1",
                "defaults:",
                "  language: Chinese",
                "segments:",
                "  - id: one",
                '    text: "你好。"',
                '    voice: "voice:alice"',
            ]
        ),
        encoding="utf-8",
    )
    assert _run(tmp_path, "say", "--script", str(script), "--dry-run", "--json") == 0
    payload = json.loads(capsys.readouterr().out)
    segment = payload["plan"]["segments"][0]
    assert segment["voice"]["mode"] == "builtin"
    assert segment["voice"]["speaker"] == "alice"
    assert segment["voice"]["voice_name"] == "alice"


def test_instruct_follows_capability(tmp_path: Path):
    make_profile(tmp_path, size="1b7")
    assert (
        _run(
            tmp_path,
            "say",
            "你好",
            "--voice",
            "voice:alice",
            "--instruct",
            "开心",
            "--dry-run",
            "--json",
        )
        == 0
    )

    make_profile(tmp_path, name="small", size="0b6")
    assert (
        _run(
            tmp_path,
            "say",
            "你好",
            "--voice",
            "voice:small",
            "--instruct",
            "开心",
            "--dry-run",
            "--json",
        )
        == 4
    )
