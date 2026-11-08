"""FR-P2-07 §4/§12.3: prepare subprocess, verification and target rules."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from qtts_cli.errors import InputError, ModelError
from qtts_cli.finetune import prepare
from qtts_cli.finetune import runner as runner_mod
from finetune_utils import (
    FakeExternal,
    config_for,
    make_official,
    make_project,
    parse_args,
    prepared_writer,
    write_dataset,
    write_tts_model,
)


def setup(tmp_path: Path, *, prepare_script: bool = True):
    models = tmp_path / "models"
    models.mkdir()
    base = write_tts_model(models, "Qwen3-TTS-12Hz-1.7B-Base", "base")
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    (tokenizer / "config.json").write_text("{}", encoding="utf-8")
    official = make_official(tmp_path / "official", prepare=prepare_script)
    _, items, ref = write_dataset(tmp_path / "audio")
    project = make_project(
        tmp_path,
        data_dir=tmp_path / "data",
        init_model=base,
        official=official,
        output_dir=tmp_path / "out",
        ref=ref,
        items=items,
    )
    return project, official, tokenizer, base, items, ref


def run_prepare(project, tmp_path, *extra: str):
    argv = [
        "finetune",
        "prepare",
        "--project",
        str(project),
        "--tokenizer-model-path",
        str(tmp_path / "tokenizer"),
        "--device",
        "cpu",
        *extra,
    ]
    args = parse_args(argv)
    cfg = config_for(args, data_dir=tmp_path / "data", model_dir=tmp_path / "models")
    return prepare(cfg, args)


def test_success_exact_argv_and_publication(tmp_path: Path, monkeypatch):
    project, official, tokenizer, base, items, ref = setup(tmp_path)
    fake = FakeExternal(prepared_writer(rows=2))
    monkeypatch.setattr(runner_mod, "run_external", fake)
    payload = run_prepare(project, tmp_path)
    assert payload["ok"] and payload["phase"] == "prepare"
    assert payload["lines"] == 2
    output = tmp_path / "project.train.jsonl"
    raw = tmp_path / "project.raw.jsonl"
    assert Path(payload["output_jsonl"]) == output
    assert Path(payload["raw_jsonl"]) == raw
    assert output.is_file() and raw.is_file()
    assert payload["raw_sha256"] == __import__("hashlib").sha256(raw.read_bytes()).hexdigest()
    assert payload["output_sha256"] == __import__("hashlib").sha256(output.read_bytes()).hexdigest()
    call = fake.calls[0]
    assert call["argv"] == [
        sys.executable,
        str(official / "prepare_data.py"),
        "--device",
        "cpu",
        "--tokenizer_model_path",
        str(tokenizer),
        "--input_jsonl",
        str(raw),
        "--output_jsonl",
        str(output) + ".tmp",
    ]
    assert call["cwd"] == official
    assert call["env"]["PYTHONUNBUFFERED"] == "1"
    assert call["env"]["PYTHONIOENCODING"] == "utf-8"
    assert not Path(str(output) + ".tmp").exists()
    assert payload["init_model_path"] == str(base)
    first_line = json.loads(raw.read_text(encoding="utf-8").splitlines()[0])
    assert set(first_line) == {"audio", "text", "ref_audio"}


def test_offline_env(tmp_path: Path, monkeypatch):
    project, *_ = setup(tmp_path)
    fake = FakeExternal(prepared_writer())
    monkeypatch.setattr(runner_mod, "run_external", fake)
    args = parse_args(
        [
            "finetune",
            "prepare",
            "--project",
            str(project),
            "--tokenizer-model-path",
            str(tmp_path / "tokenizer"),
            "--device",
            "cpu",
            "--offline",
        ]
    )
    cfg = config_for(args, data_dir=tmp_path / "data")
    prepare(cfg, args)
    env = fake.calls[0]["env"]
    assert env["HF_HUB_OFFLINE"] == "1"
    assert env["TRANSFORMERS_OFFLINE"] == "1"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda lines: lines[:-1],
        lambda lines: [dict(lines[0], audio=lines[1]["audio"])] + lines[1:],
        lambda lines: [dict(lines[0], text="edited")] + lines[1:],
        lambda lines: [dict(lines[0], ref_audio="/elsewhere.wav")] + lines[1:],
        lambda lines: [dict(lines[0], audio_codes=[])],
        lambda lines: [dict(lines[0], audio_codes=[[0] * 17])],
        lambda lines: [dict(lines[0], audio_codes=[[-1] + [0] * 15])],
    ],
)
def test_post_verification_failures(tmp_path: Path, monkeypatch, mutate):
    project, *_ = setup(tmp_path)
    output = tmp_path / "project.train.jsonl"
    previous = b"previous-output"
    output.write_bytes(previous)
    fake = FakeExternal(prepared_writer(mutate=mutate))
    monkeypatch.setattr(runner_mod, "run_external", fake)
    with pytest.raises(ModelError) as exc:
        run_prepare(project, tmp_path, "--force")
    assert exc.value.code == 3
    assert not Path(str(output) + ".tmp").exists()
    assert output.read_bytes() == previous


def test_missing_tmp_is_exit3(tmp_path: Path, monkeypatch):
    project, *_ = setup(tmp_path)
    fake = FakeExternal()
    monkeypatch.setattr(runner_mod, "run_external", fake)
    with pytest.raises(ModelError):
        run_prepare(project, tmp_path)
    assert not Path(str(tmp_path / "project.train.jsonl") + ".tmp").exists()


def test_existing_output_requires_force_before_runner(tmp_path: Path, monkeypatch):
    project, *_ = setup(tmp_path)
    output = tmp_path / "project.train.jsonl"
    output.write_text("old\n", encoding="utf-8")
    fake = FakeExternal(prepared_writer())
    monkeypatch.setattr(runner_mod, "run_external", fake)
    with pytest.raises(InputError) as exc:
        run_prepare(project, tmp_path)
    assert exc.value.code == 4
    assert fake.calls == []
    assert output.read_text(encoding="utf-8") == "old\n"


def test_missing_prepare_script_before_any_write(tmp_path: Path, monkeypatch):
    project, official, *_ = setup(tmp_path, prepare_script=False)
    fake = FakeExternal()
    monkeypatch.setattr(runner_mod, "run_external", fake)
    with pytest.raises(InputError):
        run_prepare(project, tmp_path)
    assert fake.calls == []
    assert not (tmp_path / "project.raw.jsonl").exists()


def test_raw_target_directory_rejected(tmp_path: Path, monkeypatch):
    project, *_ = setup(tmp_path)
    raw = tmp_path / "project.raw.jsonl"
    raw.mkdir()
    fake = FakeExternal()
    monkeypatch.setattr(runner_mod, "run_external", fake)
    with pytest.raises(InputError):
        run_prepare(project, tmp_path)
    assert fake.calls == []
    assert not (tmp_path / "project.train.jsonl").exists()


def test_raw_fifo_rejected(tmp_path: Path, monkeypatch):
    project, *_ = setup(tmp_path)
    raw = tmp_path / "project.raw.jsonl"
    os.mkfifo(raw)
    fake = FakeExternal()
    monkeypatch.setattr(runner_mod, "run_external", fake)
    with pytest.raises(InputError):
        run_prepare(project, tmp_path)
    assert fake.calls == []


def test_colliding_raw_and_output_paths(tmp_path: Path, monkeypatch):
    project, *_ = setup(tmp_path)
    output = tmp_path / "collision.tmp"
    output.write_text("previous\n", encoding="utf-8")
    fake = FakeExternal(prepared_writer())
    monkeypatch.setattr(runner_mod, "run_external", fake)
    with pytest.raises(InputError):
        run_prepare(
            project,
            tmp_path,
            "--raw-jsonl",
            str(tmp_path / "collision"),
            "--output-jsonl",
            str(output),
            "--force",
        )
    assert fake.calls == []
    assert output.read_text(encoding="utf-8") == "previous\n"


def test_artifact_lock_contention_before_runner(tmp_path: Path, monkeypatch):
    project, *_ = setup(tmp_path)
    output = tmp_path / "project.train.jsonl"
    fd = runner_mod.acquire_lock(Path(str(output) + ".lock"))
    fake = FakeExternal(prepared_writer())
    monkeypatch.setattr(runner_mod, "run_external", fake)
    try:
        with pytest.raises(InputError) as exc:
            run_prepare(project, tmp_path)
        assert exc.value.code == 4
        assert "lock" in exc.value.details
        assert fake.calls == []
    finally:
        os.close(fd)


def test_dry_run_creates_nothing_and_uses_final_output(tmp_path: Path, monkeypatch):
    project, official, tokenizer, base, *_ = setup(tmp_path)
    fake = FakeExternal()
    monkeypatch.setattr(runner_mod, "run_external", fake)
    data_dir = tmp_path / "data"
    payload = run_prepare(project, tmp_path, "--dry-run")
    output = tmp_path / "project.train.jsonl"
    assert payload["dry_run"] is True
    assert payload["raw_sha256"] is None and payload["output_sha256"] is None
    assert payload["temporary_output"] == str(output) + ".tmp"
    assert payload["command"][-1] == str(output)
    assert payload["will_write"] == {
        "raw_jsonl": str(tmp_path / "project.raw.jsonl"),
        "output_jsonl": str(output),
    }
    assert fake.calls == []
    assert not (tmp_path / "project.raw.jsonl").exists()
    assert not output.exists()
    assert not Path(str(output) + ".tmp").exists()
    assert not Path(str(output) + ".lock").exists()
    assert not data_dir.exists()


def test_dry_run_default_device_is_auto(tmp_path: Path):
    project, *_ = setup(tmp_path)
    args = parse_args(
        [
            "finetune",
            "prepare",
            "--project",
            str(project),
            "--tokenizer-model-path",
            str(tmp_path / "tokenizer"),
            "--dry-run",
        ]
    )
    cfg = config_for(args, data_dir=tmp_path / "data")
    payload = prepare(cfg, args)
    assert payload["device"] == "auto"


def test_child_failure_maps_to_exit3_and_removes_tmp(tmp_path: Path, monkeypatch):
    project, *_ = setup(tmp_path)

    def fail(argv, *, cwd, env, quiet):
        Path(argv[-1]).write_text("partial\n", encoding="utf-8")
        raise ModelError("child failed", details={"returncode": 1})

    monkeypatch.setattr(runner_mod, "run_external", fail)
    with pytest.raises(ModelError) as exc:
        run_prepare(project, tmp_path)
    assert exc.value.code == 3
    assert not Path(str(tmp_path / "project.train.jsonl") + ".tmp").exists()


def test_stale_tmp_is_deleted_before_run(tmp_path: Path, monkeypatch):
    project, *_ = setup(tmp_path)
    output = tmp_path / "project.train.jsonl"
    stale = Path(str(output) + ".tmp")
    stale.write_text("stale", encoding="utf-8")
    fake = FakeExternal(prepared_writer())
    monkeypatch.setattr(runner_mod, "run_external", fake)
    run_prepare(project, tmp_path)
    assert output.is_file()
    assert not stale.exists()


def test_tokenizer_fallback_from_base(tmp_path: Path, monkeypatch):
    project, official, tokenizer, base, *_ = setup(tmp_path)
    fake = FakeExternal(prepared_writer())
    monkeypatch.setattr(runner_mod, "run_external", fake)
    args = parse_args(
        [
            "finetune",
            "prepare",
            "--project",
            str(project),
            "--device",
            "cpu",
        ]
    )
    cfg = config_for(args, data_dir=tmp_path / "data")
    payload = prepare(cfg, args)
    assert payload["tokenizer_model_path"] == str(base / "speech_tokenizer")
    assert payload["init_model_path"] == str(base)


def test_missing_tokenizer_config_exit4(tmp_path: Path, monkeypatch):
    project, *_ = setup(tmp_path)
    (tmp_path / "tokenizer" / "config.json").unlink()
    fake = FakeExternal()
    monkeypatch.setattr(runner_mod, "run_external", fake)
    with pytest.raises(InputError):
        run_prepare(project, tmp_path)
    assert fake.calls == []
