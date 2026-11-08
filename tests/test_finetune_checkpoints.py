"""FR-P2-07 §6/§12.6: checkpoint discovery, validation and selection."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from qtts_cli.errors import InputError, ModelError
from qtts_cli.finetune import checkpoints, train
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


def make_output(root: Path, epochs=(0, 2, 5), speaker="alice") -> Path:
    for epoch in epochs:
        write_checkpoint(root, epoch, speaker)
    return root


def test_last_selects_highest_numeric_epoch(tmp_path: Path):
    output = make_output(tmp_path / "out")
    selected = checkpoints.select_checkpoint("last", output_dir=output, speaker_name="alice")
    assert selected.epoch == 5
    assert selected.path == output / "checkpoint-epoch-5"


def test_integer_selector_is_epoch_not_list_index(tmp_path: Path):
    output = make_output(tmp_path / "out", epochs=(0, 5))
    with pytest.raises(InputError):
        checkpoints.select_checkpoint(2, output_dir=output, speaker_name="alice")
    selected = checkpoints.select_checkpoint(5, output_dir=output, speaker_name="alice")
    assert selected.epoch == 5


def test_missing_epoch_and_output(tmp_path: Path):
    output = make_output(tmp_path / "out", epochs=(0, 1))
    with pytest.raises(InputError) as exc:
        checkpoints.select_checkpoint(9, output_dir=output, speaker_name="alice")
    assert exc.value.code == 4
    with pytest.raises(InputError):
        checkpoints.select_checkpoint("last", output_dir=tmp_path / "missing", speaker_name="alice")
    with pytest.raises(InputError):
        checkpoints.select_checkpoint("", output_dir=output, speaker_name="alice")


def test_explicit_directory_validation(tmp_path: Path):
    output = make_output(tmp_path / "out")
    selected = checkpoints.select_checkpoint(
        output / "checkpoint-epoch-2", output_dir=output, speaker_name="alice"
    )
    assert selected.epoch == 2
    with pytest.raises(InputError):
        checkpoints.select_checkpoint(
            tmp_path / "missing", output_dir=output, speaker_name="alice"
        )
    broken = write_checkpoint(tmp_path / "broken", 1, "alice", tts_model_type="base")
    with pytest.raises(InputError) as exc:
        checkpoints.select_checkpoint(
            broken, output_dir=output, speaker_name="alice"
        )
    assert exc.value.code == 4


def test_explicit_directory_outside_published_output_is_4(tmp_path: Path):
    output = make_output(tmp_path / "out")
    other = write_checkpoint(tmp_path / "elsewhere", 1, "alice")
    with pytest.raises(InputError) as exc:
        checkpoints.select_checkpoint(
            other, output_dir=output, speaker_name="alice", require_inside=output
        )
    assert exc.value.code == 4
    assert "outside" in exc.value.message


def test_speaker_absent_is_4_for_explicit_directory(tmp_path: Path):
    output = make_output(tmp_path / "out")
    checkpoint = write_checkpoint(tmp_path / "other", 0, "bob")
    with pytest.raises(InputError) as exc:
        checkpoints.select_checkpoint(checkpoint, output_dir=output, speaker_name="alice")
    assert exc.value.code == 4


@pytest.mark.parametrize(
    "mutate",
    [
        lambda path: (path / "model.safetensors").unlink(),
        lambda path: (path / "model.safetensors").write_bytes(b""),
        lambda path: (path / "speech_tokenizer" / "config.json").unlink(),
        lambda path: (path / "speech_tokenizer").rename(path / "nope"),
    ],
)
def test_malformed_published_checkpoint_is_3(tmp_path: Path, mutate):
    output = make_output(tmp_path / "out", epochs=(0,))
    mutate(output / "checkpoint-epoch-0")
    with pytest.raises(ModelError) as exc:
        checkpoints.select_checkpoint("last", output_dir=output, speaker_name="alice")
    assert exc.value.code == 3


def test_non_numeric_epoch_directory_is_3(tmp_path: Path):
    output = make_output(tmp_path / "out", epochs=(0,))
    (output / "checkpoint-epoch-x").mkdir()
    with pytest.raises(ModelError) as exc:
        checkpoints.validate_output(output, "alice")
    assert exc.value.code == 3


def test_other_names_are_ignored(tmp_path: Path):
    output = make_output(tmp_path / "out", epochs=(0,))
    (output / "notes").mkdir()
    (output / "checkpoint-epoch-1.txt").write_text("x")
    assert [cp.epoch for cp in checkpoints.validate_output(output, "alice")] == [0]


def test_selection_never_imports_torch():
    code = (
        "import sys\n"
        "from qtts_cli.finetune import checkpoints, project, dataset\n"
        "bad = [m for m in ('torch','qwen_tts','qwen_asr','dashscope',"
        "'vllm_omni','accelerate','librosa','safetensors') if m in sys.modules]\n"
        "print(','.join(bad))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "PYTHONPATH": "."},
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == ""


def _train_env(tmp_path: Path, register: bool = False):
    models = tmp_path / "models"
    models.mkdir()
    base = write_tts_model(models, "Qwen3-TTS-12Hz-1.7B-Base", "base")
    official = make_official(tmp_path / "official")
    _, items, ref = write_dataset(tmp_path / "audio")
    extra = ""
    if register:
        extra = (
            "register:\n"
            "  name: alice\n"
            "  consent:\n"
            "    rights_confirmed: true\n"
        )
    project = make_project(
        tmp_path,
        data_dir=tmp_path / "data",
        init_model=base,
        official=official,
        output_dir=tmp_path / "out",
        ref=ref,
        items=items,
        extra=extra,
    )
    records = [
        prepared_line({"audio": str(audio), "text": f"第 {index} 句", "ref_audio": str(ref)})
        for index, audio in enumerate(items, 1)
    ]
    (tmp_path / "project.train.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    return project, models


def run_train(tmp_path: Path, *extra: str):
    project, models = _train_env(tmp_path)
    argv = ["finetune", "train", "--project", str(project), "--python", sys.executable, *extra]
    args = parse_args(argv)
    cfg = config_for(args, data_dir=tmp_path / "data", model_dir=models)
    return train(cfg, args)


def test_artifact_lock_is_derived_from_output_root(tmp_path: Path, monkeypatch):
    def write(fake, call):
        argv = call["argv"]
        partial = Path(argv[argv.index("--output_model_path") + 1])
        checkpoint_writer(epochs=1)(fake, call)
        assert partial.is_dir()

    monkeypatch.setattr(runner_mod, "run_external", FakeExternal(write))
    run_train(tmp_path)
    assert (tmp_path / "out.lock").is_file()
    assert (tmp_path / "out" / "checkpoint-epoch-0").is_dir()


def test_model_override_is_ignored_by_train_phase(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(runner_mod, "run_external", FakeExternal(checkpoint_writer(epochs=1)))
    payload = run_train(tmp_path, "--model", str(tmp_path / "not-a-model"))
    assert payload["ok"] is True
