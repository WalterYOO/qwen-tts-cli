"""FR-P2-07 §6.3/§12.7: profile construction and atomic registration."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

from qtts_cli import __version__
from qtts_cli.errors import InputError, ModelError
from qtts_cli.finetune import register as registration
from qtts_cli.finetune import train
from qtts_cli.finetune import runner as runner_mod
from qtts_cli.utils import sha256_file, sha256_text, canonical_json
from qtts_cli.voice_store import VoiceStore
from finetune_utils import (
    FakeExternal,
    checkpoint_writer,
    config_for,
    make_official,
    make_project,
    parse_args,
    prepared_line,
    write_checkpoint,
    write_dataset,
    write_tts_model,
)


def setup(
    tmp_path: Path,
    *,
    register_yaml: bool = True,
    prepared: bool = True,
    output: bool = True,
    output_dir: Path | None = None,
    data_dir: Path | None = None,
    register_name: str = "alice",
):
    models = tmp_path / "models"
    models.mkdir()
    base = write_tts_model(models, "Qwen3-TTS-12Hz-1.7B-Base", "base")
    official = make_official(tmp_path / "official")
    _, items, ref = write_dataset(tmp_path / "audio")
    output_dir = output_dir or (tmp_path / "out")
    data_dir = data_dir or (tmp_path / "data")
    extra = ""
    if register_yaml:
        extra = (
            "register:\n"
            f"  name: {register_name}\n"
            "  consent:\n"
            "    rights_confirmed: true\n"
            "    license: CC0\n"
        )
    project = make_project(
        tmp_path,
        data_dir=data_dir,
        init_model=base,
        official=official,
        output_dir=output_dir,
        ref=ref,
        items=items,
        extra=extra,
    )
    prepared_path = tmp_path / "project.train.jsonl"
    if prepared:
        records = [
            prepared_line(
                {"audio": str(audio), "text": f"第 {index} 句", "ref_audio": str(ref)}
            )
            for index, audio in enumerate(items, 1)
        ]
        prepared_path.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
            encoding="utf-8",
        )
    if output:
        write_checkpoint(output_dir, 1, "alice")
    return {
        "project": project,
        "models": models,
        "base": base,
        "official": official,
        "items": items,
        "ref": ref,
        "output": output_dir,
        "data": data_dir,
        "prepared": prepared_path,
    }


def run_train(env, tmp_path: Path, *extra: str):
    argv = [
        "finetune",
        "train",
        "--project",
        str(env["project"]),
        "--python",
        sys.executable,
        *extra,
    ]
    args = parse_args(argv)
    cfg = config_for(args, data_dir=env["data"], model_dir=env["models"])
    return train(cfg, args)


def test_exact_voice_json_fields(tmp_path: Path, monkeypatch):
    env = setup(tmp_path, register_yaml=True, output=True)
    raw = tmp_path / "project.raw.jsonl"
    raw.write_text('{"audio":"a","text":"t","ref_audio":"r"}\n', encoding="utf-8")
    monkeypatch.setattr(runner_mod, "run_external", FakeExternal(checkpoint_writer(epochs=2)))
    payload = run_train(env, tmp_path, "--force")
    assert payload["registered"] is True
    store = VoiceStore(env["data"])
    meta = store.get("alice")
    assert meta is not None
    checkpoint = env["output"] / "checkpoint-epoch-1"
    weights_sha = sha256_file(checkpoint / "model.safetensors")
    config_sha = sha256_file(checkpoint / "config.json")
    checkpoint_sha = sha256_text(f"{weights_sha}:{config_sha}:alice")

    assert meta["schema_version"] == 1
    assert meta["name"] == "alice"
    assert meta["kind"] == "finetune"
    assert meta["created_at"].endswith("Z")
    source = meta["source"]
    assert source["type"] == "finetune"
    assert source["speaker"] == "alice"
    assert source["checkpoint"] == "checkpoint-epoch-1"
    assert source["weights_sha256"] == weights_sha
    assert source["weights_size"] == (checkpoint / "model.safetensors").stat().st_size
    assert source["config_sha256"] == config_sha
    assert source["checkpoint_sha256"] == checkpoint_sha
    assert source["output_dir"] == str(env["output"])
    assert source["init_model"]["id"] == "Qwen3-TTS-12Hz-1.7B-Base"
    assert source["init_model"]["path"] == str(env["base"])
    assert source["init_model"]["fingerprint"]
    assert source["dataset"]["items"] == len(env["items"])
    assert source["dataset"]["items_sha256"] == sha256_text(
        canonical_json(
            [[str(item), f"第 {index} 句测试文本。"] for index, item in enumerate(env["items"], 1)]
        )
    )
    assert source["dataset"]["ref_audio_sha256"] == sha256_file(env["ref"])
    assert source["dataset"]["raw_jsonl_sha256"] == sha256_file(raw)
    assert source["dataset"]["prepared_jsonl_sha256"] == sha256_file(env["prepared"])
    assert source["train"] == {
        "batch_size": 2,
        "lr": 2e-05,
        "num_epochs": 3,
        "epoch": 1,
    }
    assert meta["model"] == {
        "id": "checkpoint-epoch-1",
        "revision": None,
        "fingerprint": checkpoint_sha[:16],
        "path": str(checkpoint),
    }
    assert meta["tokenizer"] == {"type": "12hz", "codec_rate_hz": 12.5}
    assert meta["dtype"] is None
    assert meta["language"] == "Auto"
    assert meta["default_instruct"] is None
    assert meta["tags"] == []
    assert meta["consent"]["rights_confirmed"] is True
    assert meta["consent"]["license"] == "CC0"
    assert meta["qtts_version"] == __version__
    assert payload["selected"]["weights_sha256"] == weights_sha
    assert payload["selected"]["fingerprint"] == checkpoint_sha[:16]


def test_register_only_null_hashes_and_init_model(tmp_path: Path, monkeypatch):
    env = setup(tmp_path, output=True, prepared=False)
    monkeypatch.setattr(runner_mod, "run_external", FakeExternal())
    payload = run_train(env, tmp_path, "--register-only")
    assert payload["registered"] is True
    meta = VoiceStore(env["data"]).get("alice")
    assert meta["source"]["dataset"]["raw_jsonl_sha256"] is None
    assert meta["source"]["dataset"]["prepared_jsonl_sha256"] is None
    # The project configures init_model_path, which register-only still records.
    assert meta["source"]["init_model"]["path"] == str(env["base"])


def test_register_only_without_init_model_records_null(tmp_path: Path, monkeypatch):
    env = setup(tmp_path, output=True, prepared=False)
    project = Path(env["project"])
    text = project.read_text(encoding="utf-8")
    project.write_text(
        "\n".join(line for line in text.splitlines() if "init_model_path:" not in line) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(runner_mod, "run_external", FakeExternal())
    run_train(env, tmp_path, "--register-only")
    assert VoiceStore(env["data"]).get("alice")["source"]["init_model"] is None


def test_consent_required_even_with_cli_register(tmp_path: Path, monkeypatch):
    env = setup(tmp_path, register_yaml=False, output=True)
    fake = FakeExternal()
    monkeypatch.setattr(runner_mod, "run_external", fake)
    with pytest.raises(InputError) as exc:
        run_train(env, tmp_path, "--register", "alice")
    assert exc.value.code == 4
    assert fake.calls == []


@pytest.mark.parametrize("name", ["builtin", "../escape", ".hidden", "a" * 65])
def test_invalid_or_reserved_name_is_4_pre_training(tmp_path: Path, monkeypatch, name):
    env = setup(tmp_path, output=True)
    fake = FakeExternal()
    monkeypatch.setattr(runner_mod, "run_external", fake)
    with pytest.raises(InputError) as exc:
        run_train(env, tmp_path, "--register", name, "--register-only")
    assert exc.value.code == 4
    assert fake.calls == []


def test_output_containing_profile_is_4_pre_training(tmp_path: Path, monkeypatch):
    data = tmp_path / "data"
    env = setup(tmp_path, data_dir=data, output_dir=data / "voices")
    fake = FakeExternal()
    monkeypatch.setattr(runner_mod, "run_external", fake)
    with pytest.raises(InputError) as exc:
        run_train(env, tmp_path)
    assert exc.value.code == 4
    assert fake.calls == []
    assert not (data / "voices.partial").exists()


def test_profile_inside_output_is_4_pre_training(tmp_path: Path, monkeypatch):
    data = tmp_path / "data"
    env = setup(tmp_path, data_dir=data, output_dir=tmp_path / "out")
    # Point the project output at a directory containing the profile store.
    project = Path(env["project"])
    text = project.read_text(encoding="utf-8")
    project.write_text(
        text.replace(f"output_dir: {tmp_path / 'out'}", f"output_dir: {tmp_path}"),
        encoding="utf-8",
    )
    fake = FakeExternal()
    monkeypatch.setattr(runner_mod, "run_external", fake)
    with pytest.raises(InputError) as exc:
        run_train(env, tmp_path)
    assert exc.value.code == 4
    assert fake.calls == []


def test_existing_profile_without_force_is_4(tmp_path: Path, monkeypatch):
    env = setup(tmp_path, output=True, prepared=False)
    voices = env["data"] / "voices" / "alice"
    voices.mkdir(parents=True)
    (voices / "voice.json").write_text('{"name":"alice","kind":"finetune"}')
    fake = FakeExternal()
    monkeypatch.setattr(runner_mod, "run_external", fake)
    with pytest.raises(InputError) as exc:
        run_train(env, tmp_path, "--register-only")
    assert exc.value.code == 4
    assert fake.calls == []


def test_force_replaces_profile_atomically(tmp_path: Path, monkeypatch):
    env = setup(tmp_path, register_yaml=True, output=True)
    old = env["data"] / "voices" / "alice"
    old.mkdir(parents=True)
    (old / "voice.json").write_text('{"name":"alice","kind":"finetune","old":true}')
    monkeypatch.setattr(runner_mod, "run_external", FakeExternal(checkpoint_writer(epochs=1)))
    payload = run_train(env, tmp_path, "--force")
    assert payload["registered"] is True
    meta = VoiceStore(env["data"]).get("alice")
    assert "old" not in meta
    assert meta["model"]["path"].endswith("checkpoint-epoch-0")


def test_registration_rollback_restores_old_profile_on_oserror(tmp_path: Path):
    store = VoiceStore(tmp_path / "data")
    target = store.dir_for("alice")
    target.mkdir(parents=True)
    (target / "voice.json").write_text('{"name":"alice","old":true}')
    staging = store.staging_dir("alice")
    (staging / "voice.json").write_text('{"name":"alice","new":true}')
    real_replace = os.replace
    calls = {"n": 0}

    def fail_second(src, dst):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("rename failed")
        return real_replace(src, dst)

    import qtts_cli.finetune.register as register_mod

    original = register_mod.os.replace
    register_mod.os.replace = fail_second
    try:
        with pytest.raises(ModelError) as exc:
            registration.publish_profile(store, staging, "alice", published=True)
    finally:
        register_mod.os.replace = original
    assert exc.value.code == 3
    assert json.loads((target / "voice.json").read_text())["old"] is True
    assert not staging.exists()


def test_registration_pre_commit_signal_restores_old_profile(tmp_path: Path):
    store = VoiceStore(tmp_path / "data")
    target = store.dir_for("alice")
    target.mkdir(parents=True)
    (target / "voice.json").write_text('{"name":"alice","old":true}')
    staging = store.staging_dir("alice")
    (staging / "voice.json").write_text('{"name":"alice","new":true}')
    real_replace = os.replace
    calls = {"n": 0}

    def interrupt_second(src, dst):
        calls["n"] += 1
        if calls["n"] == 2:
            raise KeyboardInterrupt
        return real_replace(src, dst)

    import qtts_cli.finetune.register as register_mod

    original = register_mod.os.replace
    register_mod.os.replace = interrupt_second
    try:
        with pytest.raises(KeyboardInterrupt):
            registration.publish_profile(store, staging, "alice", published=True)
    finally:
        register_mod.os.replace = original
    assert json.loads((target / "voice.json").read_text())["old"] is True
    assert not staging.exists()


def test_staging_removed_on_oserror(tmp_path: Path, monkeypatch):
    env = setup(tmp_path, register_yaml=True, output=True)
    monkeypatch.setattr(runner_mod, "run_external", FakeExternal(checkpoint_writer(epochs=1)))

    def fail(self, meta, directory=None):
        raise OSError("disk full")

    monkeypatch.setattr(VoiceStore, "save_meta", fail)
    with pytest.raises(ModelError) as exc:
        run_train(env, tmp_path, "--force")
    assert exc.value.code == 3
    assert exc.value.details["published"] is True
    assert exc.value.details["registered"] is False
    voices = env["data"] / "voices"
    assert not [p for p in voices.iterdir() if p.name.startswith(".alice.")]
    assert VoiceStore(env["data"]).get("alice") is None


def test_checkpoint_is_never_touched(tmp_path: Path, monkeypatch):
    env = setup(tmp_path, register_yaml=True, output=True)
    before = {
        str(path.relative_to(env["output"])): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(env["output"].rglob("*"))
        if path.is_file()
    }
    monkeypatch.setattr(runner_mod, "run_external", FakeExternal())
    run_train(env, tmp_path, "--register-only")
    after = {
        str(path.relative_to(env["output"])): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(env["output"].rglob("*"))
        if path.is_file()
    }
    assert before == after
    assert not (env["output"] / "voice.json").exists()
