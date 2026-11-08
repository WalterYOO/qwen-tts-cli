"""FR-P2-07 §5/§12.5: training subprocess, publication, recovery and registration."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from qtts_cli.errors import InputError, ModelError
from qtts_cli.finetune import register as registration
from qtts_cli.finetune import train
from qtts_cli.finetune import runner as runner_mod
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


def write_prepared(path: Path, items, ref, texts):
    records = [
        prepared_line({"audio": str(audio), "text": text, "ref_audio": str(ref)})
        for audio, text in zip(items, texts)
    ]
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def setup(tmp_path: Path, *, register: bool = False, output_dir: Path | None = None,
          prepared: bool = True):
    models = tmp_path / "models"
    models.mkdir()
    base = write_tts_model(models, "Qwen3-TTS-12Hz-1.7B-Base", "base")
    official = make_official(tmp_path / "official")
    _, items, ref = write_dataset(tmp_path / "audio")
    output = output_dir or (tmp_path / "out")
    extra = ""
    if register:
        extra = (
            "register:\n"
            "  name: alice\n"
            "  consent:\n"
            "    rights_confirmed: true\n"
            "    license: CC0\n"
        )
    project = make_project(
        tmp_path,
        data_dir=tmp_path / "data",
        init_model=base,
        official=official,
        output_dir=output,
        ref=ref,
        items=items,
        extra=extra,
    )
    if prepared:
        write_prepared(
            tmp_path / "project.train.jsonl",
            items,
            ref,
            [f"第 {index} 句" for index in range(1, len(items) + 1)],
        )
    return {
        "project": project,
        "official": official,
        "base": base,
        "items": items,
        "ref": ref,
        "output": output,
        "models": models,
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
    cfg = config_for(args, data_dir=tmp_path / "data", model_dir=env["models"])
    return train(cfg, args)


def test_success_publication_and_exact_argv(tmp_path: Path, monkeypatch):
    env = setup(tmp_path)
    fake = FakeExternal(checkpoint_writer(epochs=7))
    monkeypatch.setattr(runner_mod, "run_external", fake)
    payload = run_train(env, tmp_path, "--lr", "0.001", "--batch-size", "8", "--epochs", "7")
    assert payload["ok"] and payload["published"] is True
    assert payload["registered"] is False
    assert [cp["epoch"] for cp in payload["checkpoints"]] == list(range(7))
    assert payload["selected"]["epoch"] == 6
    assert payload["selected"]["speaker"] == "alice"
    assert Path(payload["selected"]["path"]).is_dir()
    assert (tmp_path / "out" / "checkpoint-epoch-6").is_dir()
    assert not (tmp_path / "out" / "checkpoint-epoch-6").is_symlink()
    from qtts_cli.utils import sha256_file

    assert payload["selected"]["weights_sha256"] == sha256_file(
        tmp_path / "out" / "checkpoint-epoch-6" / "model.safetensors"
    )
    call = fake.calls[0]
    partial = tmp_path / "out.partial"
    assert call["argv"] == [
        str(Path(sys.executable).resolve()),
        str(env["official"] / "sft_12hz.py"),
        "--init_model_path",
        str(env["base"]),
        "--output_model_path",
        str(partial),
        "--train_jsonl",
        str(tmp_path / "project.train.jsonl"),
        "--batch_size",
        "8",
        "--lr",
        "0.001",
        "--num_epochs",
        "7",
        "--speaker_name",
        "alice",
    ]
    assert call["cwd"] == env["official"]
    assert call["env"]["PYTHONUNBUFFERED"] == "1"
    assert not partial.exists()
    assert payload["recoveries"] == []


def test_registered_profile_records_cli_epochs(tmp_path: Path, monkeypatch):
    env = setup(tmp_path, register=True)
    fake = FakeExternal(checkpoint_writer(epochs=7))
    monkeypatch.setattr(runner_mod, "run_external", fake)
    payload = run_train(env, tmp_path, "--epochs", "7")
    assert payload["registered"] is True
    meta = json.loads(
        (tmp_path / "data" / "voices" / "alice" / "voice.json").read_text(
            encoding="utf-8"
        )
    )
    assert meta["source"]["train"]["num_epochs"] == 7
    assert payload["selected"]["weights_sha256"] == meta["source"]["weights_sha256"]


def test_default_lr_float_repr(tmp_path: Path, monkeypatch):
    env = setup(tmp_path)
    fake = FakeExternal(checkpoint_writer(epochs=1))
    monkeypatch.setattr(runner_mod, "run_external", fake)
    run_train(env, tmp_path)
    argv = fake.calls[0]["argv"]
    assert argv[argv.index("--lr") + 1] == repr(2e-05)
    assert argv[argv.index("--batch_size") + 1] == "2"
    assert argv[argv.index("--num_epochs") + 1] == "3"


def test_existing_output_without_force_exit4_before_runner(tmp_path: Path, monkeypatch):
    env = setup(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    (out / "sentinel").write_text("x")
    fake = FakeExternal(checkpoint_writer())
    monkeypatch.setattr(runner_mod, "run_external", fake)
    with pytest.raises(InputError):
        run_train(env, tmp_path)
    assert fake.calls == []
    assert (out / "sentinel").is_file()


def test_force_replaces_existing_output(tmp_path: Path, monkeypatch):
    env = setup(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    (out / "sentinel").write_text("x")
    fake = FakeExternal(checkpoint_writer(epochs=1))
    monkeypatch.setattr(runner_mod, "run_external", fake)
    payload = run_train(env, tmp_path, "--force")
    assert payload["published"] is True
    assert not (out / "sentinel").exists()
    assert (out / "checkpoint-epoch-0").is_dir()


def test_stale_partial_force_cleanup(tmp_path: Path, monkeypatch):
    env = setup(tmp_path)
    partial = tmp_path / "out.partial"
    partial.mkdir()
    (partial / "sentinel").write_text("x")
    fake = FakeExternal(checkpoint_writer(epochs=1))
    monkeypatch.setattr(runner_mod, "run_external", fake)
    run_train(env, tmp_path, "--force")
    assert not (tmp_path / "out" / "sentinel").exists()
    assert (tmp_path / "out" / "checkpoint-epoch-0").is_dir()


@pytest.mark.parametrize(
    "kind",
    ["missing_weights", "wrong_tts_type", "wrong_spk", "missing_tokenizer", "bad_epoch"],
)
def test_malformed_checkpoint_is_exit3_and_keeps_partial(tmp_path: Path, monkeypatch, kind):
    env = setup(tmp_path)

    def callback(fake, call):
        argv = call["argv"]
        partial = Path(argv[argv.index("--output_model_path") + 1])
        if kind == "bad_epoch":
            write_checkpoint(partial, 0, "alice")
            (partial / "checkpoint-epoch-x").mkdir()
            return
        extra = {}
        kwargs = {}
        if kind == "missing_weights":
            kwargs["weights"] = b"x"
        if kind == "wrong_tts_type":
            kwargs["tts_model_type"] = "base"
        if kind == "wrong_spk":
            kwargs["speaker_id"] = 999
        if kind == "missing_tokenizer":
            kwargs["tokenizer"] = False
        target = write_checkpoint(partial, 0, "alice", extra_config=extra, **kwargs)
        if kind == "missing_weights":
            (target / "model.safetensors").unlink()

    monkeypatch.setattr(runner_mod, "run_external", FakeExternal(callback))
    with pytest.raises(ModelError) as exc:
        run_train(env, tmp_path)
    assert exc.value.code == 3
    assert (tmp_path / "out.partial").is_dir()
    assert not (tmp_path / "out").exists()


def test_zero_checkpoints_is_exit3(tmp_path: Path, monkeypatch):
    env = setup(tmp_path)

    def empty(fake, call):
        argv = call["argv"]
        Path(argv[argv.index("--output_model_path") + 1]).mkdir(parents=True)

    monkeypatch.setattr(runner_mod, "run_external", FakeExternal(empty))
    with pytest.raises(ModelError) as exc:
        run_train(env, tmp_path)
    assert exc.value.code == 3
    assert (tmp_path / "out.partial").is_dir()
    assert not (tmp_path / "out").exists()


def test_publish_failure_restores_old_and_keeps_partial(tmp_path: Path, monkeypatch):
    env = setup(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    (out / "sentinel").write_text("old")
    monkeypatch.setattr(runner_mod, "run_external", FakeExternal(checkpoint_writer(epochs=1)))
    real_replace = os.replace
    calls = {"n": 0}

    def fail_second(src, dst):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("rename failed")
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", fail_second)
    with pytest.raises(ModelError) as exc:
        run_train(env, tmp_path, "--force")
    assert exc.value.code == 3
    assert (out / "sentinel").read_text() == "old"
    assert (tmp_path / "out.partial").is_dir()


def test_pre_commit_keyboard_interrupt_restores_old(tmp_path: Path, monkeypatch, capsys):
    env = setup(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    (out / "sentinel").write_text("old")
    monkeypatch.setattr(runner_mod, "run_external", FakeExternal(checkpoint_writer(epochs=1)))
    real_replace = os.replace
    calls = {"n": 0}

    def interrupt_second(src, dst):
        calls["n"] += 1
        if calls["n"] == 2:
            raise KeyboardInterrupt
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", interrupt_second)
    with pytest.raises(KeyboardInterrupt):
        run_train(env, tmp_path, "--force")
    assert (out / "sentinel").read_text() == "old"
    assert (tmp_path / "out.partial").is_dir()
    assert "published: false" in capsys.readouterr().err


def test_post_commit_signal_keeps_new_output(tmp_path: Path, monkeypatch, capsys):
    env = setup(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    (out / "sentinel").write_text("old")
    monkeypatch.setattr(runner_mod, "run_external", FakeExternal(checkpoint_writer(epochs=1)))
    real_replace = os.replace
    fired = {"done": False}

    def signal_on_commit(src, dst):
        result = real_replace(src, dst)
        if not fired["done"] and Path(dst) == out:
            fired["done"] = True
            import signal as signal_module

            signal_module.getsignal(signal_module.SIGINT)(signal_module.SIGINT, None)
        return result

    monkeypatch.setattr(os, "replace", signal_on_commit)
    with pytest.raises(runner_mod.CommitInterrupt) as exc:
        run_train(env, tmp_path, "--force")
    assert exc.value.published is True
    assert (out / "checkpoint-epoch-0").is_dir()
    assert not (out / "sentinel").exists()
    assert "published: true" in capsys.readouterr().err


def test_stale_old_recovery_without_force(tmp_path: Path, monkeypatch):
    env = setup(tmp_path)
    out = tmp_path / "out"
    out.with_name("out.old").mkdir()
    (out.with_name("out.old") / "sentinel").write_text("old")
    fake = FakeExternal(checkpoint_writer())
    monkeypatch.setattr(runner_mod, "run_external", fake)
    with pytest.raises(InputError) as exc:
        run_train(env, tmp_path)
    assert exc.value.code == 4
    assert "recoveries" in exc.value.details
    assert (out / "sentinel").read_text() == "old"
    assert fake.calls == []


def test_stale_old_recovery_with_force(tmp_path: Path, monkeypatch):
    env = setup(tmp_path)
    out = tmp_path / "out"
    out.with_name("out.old").mkdir()
    (out.with_name("out.old") / "sentinel").write_text("old")
    monkeypatch.setattr(runner_mod, "run_external", FakeExternal(checkpoint_writer(epochs=1)))
    payload = run_train(env, tmp_path, "--force")
    assert payload["recoveries"] == ["restored_old"]
    assert (out / "checkpoint-epoch-0").is_dir()
    assert not (out / "sentinel").exists()


def test_stale_old_dry_run_writes_nothing(tmp_path: Path, monkeypatch):
    env = setup(tmp_path)
    out = tmp_path / "out"
    out.with_name("out.old").mkdir()
    (out.with_name("out.old") / "sentinel").write_text("old")
    fake = FakeExternal()
    monkeypatch.setattr(runner_mod, "run_external", fake)
    with pytest.raises(InputError) as exc:
        run_train(env, tmp_path, "--dry-run")
    assert exc.value.details["recovery_required"] is True
    assert not out.exists()
    assert (out.with_name("out.old") / "sentinel").is_file()
    assert fake.calls == []


def test_old_and_partial_without_force_restores_old(tmp_path: Path, monkeypatch):
    # `.old` present with final absent means an interrupted publication; the
    # stale `.partial` must not pre-empt recovery of that publication.
    env = setup(tmp_path)
    out = tmp_path / "out"
    out.with_name("out.old").mkdir()
    (out.with_name("out.old") / "sentinel").write_text("old")
    partial = out.with_name("out.partial")
    partial.mkdir()
    (partial / "stale").write_text("x")
    fake = FakeExternal()
    monkeypatch.setattr(runner_mod, "run_external", fake)
    with pytest.raises(InputError) as exc:
        run_train(env, tmp_path)
    assert exc.value.code == 4
    assert exc.value.details["recoveries"] == ["restored_old"]
    # The old publication was restored, not stranded in `.old`.
    assert (out / "sentinel").read_text() == "old"
    assert not out.with_name("out.old").exists()
    assert fake.calls == []


def test_old_and_partial_with_force_recovers_and_replaces(tmp_path: Path, monkeypatch):
    env = setup(tmp_path)
    out = tmp_path / "out"
    out.with_name("out.old").mkdir()
    (out.with_name("out.old") / "sentinel").write_text("old")
    partial = out.with_name("out.partial")
    partial.mkdir()
    (partial / "stale").write_text("x")
    monkeypatch.setattr(
        runner_mod, "run_external", FakeExternal(checkpoint_writer(epochs=1))
    )
    payload = run_train(env, tmp_path, "--force")
    assert payload["recoveries"] == ["restored_old"]
    assert (out / "checkpoint-epoch-0").is_dir()
    assert not (out / "sentinel").exists()
    assert not out.with_name("out.old").exists()
    assert not partial.exists()


def test_old_and_partial_dry_run_writes_nothing(tmp_path: Path, monkeypatch):
    env = setup(tmp_path)
    out = tmp_path / "out"
    out.with_name("out.old").mkdir()
    (out.with_name("out.old") / "sentinel").write_text("old")
    partial = out.with_name("out.partial")
    partial.mkdir()
    (partial / "stale").write_text("x")
    fake = FakeExternal()
    monkeypatch.setattr(runner_mod, "run_external", fake)
    with pytest.raises(InputError) as exc:
        run_train(env, tmp_path, "--dry-run")
    assert exc.value.details["recovery_required"] is True
    assert not out.exists()
    assert (out.with_name("out.old") / "sentinel").is_file()
    assert (partial / "stale").is_file()
    assert fake.calls == []


def test_both_output_and_profile_recoveries(tmp_path: Path, monkeypatch):
    env = setup(tmp_path, register=True)
    out = tmp_path / "out"
    out.with_name("out.old").mkdir()
    (out.with_name("out.old") / "sentinel").write_text("old")
    voices = tmp_path / "data" / "voices"
    backup = voices / ".alice.old-abc"
    backup.mkdir(parents=True)
    (backup / "voice.json").write_text('{"name":"alice","kind":"finetune"}')
    monkeypatch.setattr(runner_mod, "run_external", FakeExternal(checkpoint_writer(epochs=1)))
    payload = run_train(env, tmp_path, "--force")
    assert payload["recoveries"] == ["restored_old", "restored_profile"]
    assert payload["registered"] is True
    assert (voices / "alice" / "voice.json").is_file()


def test_missing_dataset_audio_exit4_before_runner(tmp_path: Path, monkeypatch):
    env = setup(tmp_path)
    env["items"][0].unlink()
    fake = FakeExternal()
    monkeypatch.setattr(runner_mod, "run_external", fake)
    with pytest.raises(InputError) as exc:
        run_train(env, tmp_path)
    assert exc.value.code == 4
    assert fake.calls == []
    assert not (tmp_path / "out.partial").exists()


def test_output_dir_equal_default_raw_jsonl_exit4_before_runner(
    tmp_path: Path, monkeypatch
):
    env = setup(tmp_path)
    raw = tmp_path / "project.raw.jsonl"
    fake = FakeExternal()
    monkeypatch.setattr(runner_mod, "run_external", fake)
    with pytest.raises(InputError) as exc:
        run_train(env, tmp_path, "--output-dir", str(raw))
    assert exc.value.code == 4
    assert fake.calls == []
    assert not raw.exists()


def test_missing_train_script_exit4_before_runner(tmp_path: Path, monkeypatch):
    env = setup(tmp_path)
    (env["official"] / "sft_12hz.py").unlink()
    fake = FakeExternal()
    monkeypatch.setattr(runner_mod, "run_external", fake)
    with pytest.raises(InputError):
        run_train(env, tmp_path)
    assert fake.calls == []
    assert not (tmp_path / "out.partial").exists()
    assert not (tmp_path / "out").exists()


def test_registration_failure_after_publish(tmp_path: Path, monkeypatch):
    env = setup(tmp_path, register=True)
    monkeypatch.setattr(runner_mod, "run_external", FakeExternal(checkpoint_writer(epochs=1)))

    def fail(*args, **kwargs):
        raise ModelError("cannot hash")

    monkeypatch.setattr(registration, "write_staging_profile", fail)
    with pytest.raises(ModelError) as exc:
        run_train(env, tmp_path)
    assert exc.value.code == 3
    assert exc.value.details["published"] is True
    assert exc.value.details["registered"] is False
    assert (tmp_path / "out" / "checkpoint-epoch-0").is_dir()


def test_profile_backup_recovery_without_force(tmp_path: Path, monkeypatch):
    env = setup(tmp_path, register=True)
    backup = tmp_path / "data" / "voices" / ".alice.old-1"
    backup.mkdir(parents=True)
    (backup / "voice.json").write_text('{"name":"alice","kind":"finetune"}')
    fake = FakeExternal(checkpoint_writer())
    monkeypatch.setattr(runner_mod, "run_external", fake)
    with pytest.raises(InputError) as exc:
        run_train(env, tmp_path)
    assert exc.value.code == 4
    assert exc.value.details["recoveries"] == ["restored_profile"]
    assert fake.calls == []
    assert (tmp_path / "data" / "voices" / "alice" / "voice.json").is_file()


def test_profile_backup_recovery_with_force(tmp_path: Path, monkeypatch):
    env = setup(tmp_path, register=True)
    backup = tmp_path / "data" / "voices" / ".alice.old-1"
    backup.mkdir(parents=True)
    (backup / "voice.json").write_text('{"name":"alice","kind":"finetune"}')
    monkeypatch.setattr(runner_mod, "run_external", FakeExternal(checkpoint_writer(epochs=1)))
    payload = run_train(env, tmp_path, "--force")
    assert payload["recoveries"] == ["restored_profile"]
    assert payload["registered"] is True


def test_multiple_profile_backups_exit4(tmp_path: Path, monkeypatch):
    env = setup(tmp_path, register=True)
    for suffix in ("a", "b"):
        backup = tmp_path / "data" / "voices" / f".alice.old-{suffix}"
        backup.mkdir(parents=True)
    fake = FakeExternal()
    monkeypatch.setattr(runner_mod, "run_external", fake)
    with pytest.raises(InputError):
        run_train(env, tmp_path)
    assert fake.calls == []


def test_target_present_with_backup_deletes_stale_backup(tmp_path: Path, monkeypatch):
    env = setup(tmp_path, register=True)
    voices = tmp_path / "data" / "voices"
    (voices / "alice").mkdir(parents=True)
    (voices / "alice" / "voice.json").write_text('{"name":"alice","kind":"finetune"}')
    backup = voices / ".alice.old-a"
    backup.mkdir()
    monkeypatch.setattr(runner_mod, "run_external", FakeExternal(checkpoint_writer(epochs=1)))
    run_train(env, tmp_path, "--force")
    assert (voices / "alice" / "voice.json").is_file()
    assert not backup.exists()
    assert registration.profile_backups(  # no lingering backups
        __import__("qtts_cli.voice_store", fromlist=["VoiceStore"]).VoiceStore(tmp_path / "data"),
        "alice",
    ) == []


def test_register_only_without_request_is_usage_error(tmp_path: Path, monkeypatch):
    env = setup(tmp_path)
    fake = FakeExternal()
    monkeypatch.setattr(runner_mod, "run_external", fake)
    with pytest.raises(Exception) as exc:
        run_train(env, tmp_path, "--register-only")
    assert getattr(exc.value, "code", None) == 2
    assert fake.calls == []
    assert not (tmp_path / ".project.finetune.lock").exists()


def test_interrupt_during_staging_build_keeps_output_and_old_profile(tmp_path: Path, monkeypatch, capsys):
    env = setup(tmp_path, register=True)
    voices = tmp_path / "data" / "voices"
    (voices / "alice").mkdir(parents=True)
    (voices / "alice" / "voice.json").write_text('{"name":"alice","kind":"finetune","old":true}')
    monkeypatch.setattr(runner_mod, "run_external", FakeExternal(checkpoint_writer(epochs=1)))

    from qtts_cli.voice_store import VoiceStore

    def interrupt(self, meta, directory=None):
        Path(directory, "partial").write_text("x")
        raise KeyboardInterrupt

    monkeypatch.setattr(VoiceStore, "save_meta", interrupt)
    with pytest.raises(KeyboardInterrupt):
        run_train(env, tmp_path, "--force")
    assert (tmp_path / "out" / "checkpoint-epoch-0").is_dir()
    old = json.loads((voices / "alice" / "voice.json").read_text())
    assert old.get("old") is True
    assert not [p for p in voices.iterdir() if p.name.startswith(".alice.")]
    assert "published: true" in capsys.readouterr().err


def test_register_only_works_and_takes_locks(tmp_path: Path, monkeypatch):
    env = setup(tmp_path, register=True, prepared=False)
    out = tmp_path / "out"
    write_checkpoint(out, 1, "alice")
    (out / "sentinel").write_text("keep")
    fake = FakeExternal()
    monkeypatch.setattr(runner_mod, "run_external", fake)
    payload = run_train(env, tmp_path, "--register-only")
    assert fake.calls == []
    assert payload["published"] is False
    assert payload["registered"] is True
    assert payload["selected"]["epoch"] == 1
    assert (out / "sentinel").is_file()
    assert (tmp_path / "data" / "voices" / "alice" / "voice.json").is_file()

    fd = runner_mod.acquire_lock(Path(f"{out}.lock"))
    try:
        with pytest.raises(InputError):
            run_train(env, tmp_path, "--register-only", "--force")
    finally:
        os.close(fd)


def test_register_only_output_dir_containing_data_dir_is_4(
    tmp_path: Path, monkeypatch
):
    env = setup(tmp_path, register=True, output_dir=tmp_path)
    write_checkpoint(tmp_path, 0, "alice")
    fake = FakeExternal()
    monkeypatch.setattr(runner_mod, "run_external", fake)
    with pytest.raises(InputError) as exc:
        run_train(env, tmp_path, "--register-only")
    assert exc.value.code == 4
    assert fake.calls == []
    assert not (tmp_path / "data" / "voices" / "alice").exists()


def test_register_only_output_dir_parent_of_profile_is_4(
    tmp_path: Path, monkeypatch
):
    # output_dir is the parent of the (not yet existing) profile dir; exact
    # pairwise equality and profile-exists checks cannot catch this, only
    # containment (profile is under output_dir, data_dir contains both).
    data_dir = tmp_path / "data"
    output_dir = data_dir / "voices"
    output_dir.mkdir(parents=True)
    env = setup(tmp_path, register=True, output_dir=output_dir)
    write_checkpoint(output_dir, 0, "alice")
    fake = FakeExternal()
    monkeypatch.setattr(runner_mod, "run_external", fake)
    with pytest.raises(InputError) as exc:
        run_train(env, tmp_path, "--register-only")
    assert exc.value.code == 4
    assert Path(exc.value.details["conflict"]).resolve() == data_dir.resolve()
    assert fake.calls == []
    assert not (data_dir / "voices" / "alice").exists()


def test_register_only_missing_output_exit4(tmp_path: Path, monkeypatch):
    env = setup(tmp_path, register=True, prepared=False)
    fake = FakeExternal()
    monkeypatch.setattr(runner_mod, "run_external", fake)
    with pytest.raises(InputError):
        run_train(env, tmp_path, "--register-only")
    assert fake.calls == []


def test_register_only_explicit_checkpoint_without_output_dir(tmp_path: Path, monkeypatch):
    env = setup(tmp_path, register=True, prepared=False)
    other = tmp_path / "elsewhere"
    checkpoint = write_checkpoint(other, 4, "alice")
    # Remove output_dir from the project so the explicit checkpoint drives the lock.
    project = Path(env["project"])
    text = project.read_text(encoding="utf-8")
    project.write_text(
        "\n".join(
            line for line in text.splitlines() if "output_dir:" not in line
        )
        + "\n",
        encoding="utf-8",
    )
    fake = FakeExternal()
    monkeypatch.setattr(runner_mod, "run_external", fake)
    fd = runner_mod.acquire_lock(Path(f"{other}.lock"))
    try:
        with pytest.raises(InputError):
            run_train(env, tmp_path, "--register-only", "--checkpoint", str(checkpoint))
    finally:
        os.close(fd)


def test_artifact_lock_contention_before_runner(tmp_path: Path, monkeypatch):
    env = setup(tmp_path)
    out = tmp_path / "out"
    fd = runner_mod.acquire_lock(Path(f"{out}.lock"))
    fake = FakeExternal(checkpoint_writer())
    monkeypatch.setattr(runner_mod, "run_external", fake)
    try:
        with pytest.raises(InputError):
            run_train(env, tmp_path)
        assert fake.calls == []
    finally:
        os.close(fd)
