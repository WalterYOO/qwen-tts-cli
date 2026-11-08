"""FR-P2-07 §1/§8/§12.11: CLI surface, JSON discipline and exit codes."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from qtts_cli.cli import build_parser, main
from qtts_cli.finetune import evaluate as evaluate_mod
from qtts_cli.finetune import runner as runner_mod
from finetune_utils import (
    FakeExternal,
    make_official,
    make_project,
    prepared_line,
    write_checkpoint,
    write_dataset,
    write_tts_model,
)


def setup(tmp_path: Path, *, output: bool = False, eval_block: bool = False):
    models = tmp_path / "models"
    models.mkdir()
    base = write_tts_model(models, "Qwen3-TTS-12Hz-1.7B-Base", "base")
    official = make_official(tmp_path / "official")
    _, items, ref = write_dataset(tmp_path / "audio")
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    (tokenizer / "config.json").write_text("{}", encoding="utf-8")
    extra = ""
    if eval_block:
        extra = (
            "eval:\n"
            "  items:\n"
            "    - text: hello\n"
            "      language: English\n"
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
        prepared_line(
            {"audio": str(audio), "text": f"第 {index} 句", "ref_audio": str(ref)}
        )
        for index, audio in enumerate(items, 1)
    ]
    prepared = tmp_path / "project.train.jsonl"
    prepared.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    if output:
        write_checkpoint(tmp_path / "out", 0, "alice")
    return project, models, tokenizer, base, items, ref, prepared


def common(tmp_path: Path, models: Path):
    return [
        "--model-dir",
        str(models),
        "--data-dir",
        str(tmp_path / "data"),
        "--cache-dir",
        str(tmp_path / "cache"),
    ]


def test_subcommands_parse_and_project_required():
    parser = build_parser()
    project = "p.yaml"
    for command in ("prepare", "train", "eval"):
        args = parser.parse_args(["finetune", command, "--project", project])
        assert args.project == project
    from qtts_cli.errors import UsageError

    for command in ("prepare", "train", "eval"):
        with pytest.raises(UsageError) as exc:
            parser.parse_args(["finetune", command])
        assert exc.value.code == 2


def test_unknown_flags_exit_2():
    parser = build_parser()
    from qtts_cli.errors import UsageError

    with pytest.raises(UsageError) as exc:
        parser.parse_args(["finetune", "train", "--project", "p.yaml", "--nope"])
    assert exc.value.code == 2


def test_suppress_precedence(tmp_path: Path):
    project, models, _, _, _, _, _ = setup(tmp_path)
    parser = build_parser()
    args = parser.parse_args(["finetune", "train", "--project", str(project)])
    assert not hasattr(args, "batch_size")
    assert not hasattr(args, "lr")
    args = parser.parse_args(
        ["finetune", "train", "--project", str(project), "--batch-size", "8"]
    )
    assert args.batch_size == 8


def test_train_dry_run_uses_project_defaults_and_cli(tmp_path: Path, capsys):
    project, models, _, _, _, _, _ = setup(tmp_path)
    code = main(
        ["finetune", "train", "--project", str(project), "--dry-run", "--json", *common(tmp_path, models)]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    command = payload["command"]
    assert command[command.index("--batch_size") + 1] == "2"
    assert command[command.index("--lr") + 1] == repr(2e-05)
    assert payload["checkpoints"] == []
    assert payload["selected"] is None
    assert payload["will_create_partial"] is True
    assert payload["models_loaded"] is False


def test_prepare_dry_run_json_shape(tmp_path: Path, capsys):
    project, models, tokenizer, _, _, _, _ = setup(tmp_path)
    code = main(
        [
            "finetune",
            "prepare",
            "--project",
            str(project),
            "--tokenizer-model-path",
            str(tokenizer),
            "--dry-run",
            "--force",
            "--json",
            *common(tmp_path, models),
        ]
    )
    assert code == 0
    out = capsys.readouterr()
    payload = json.loads(out.out)
    assert payload["ok"] is True
    assert payload["phase"] == "prepare"
    assert payload["dry_run"] is True
    assert payload["device"] == "auto"
    assert payload["raw_sha256"] is None
    assert set(payload["will_write"]) == {"raw_jsonl", "output_jsonl"}
    assert "prepared" not in out.err and "raw:" not in out.err


def test_eval_dry_run_json_shape(tmp_path: Path, capsys):
    project, models, _, _, _, _, _ = setup(tmp_path, output=True, eval_block=True)
    code = main(
        [
            "finetune",
            "eval",
            "--project",
            str(project),
            "--dry-run",
            "--json",
            *common(tmp_path, models),
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["phase"] == "eval"
    assert payload["models_loaded"] is False
    assert payload["items"] == 1
    assert payload["speaker"] == "alice"
    assert payload["checkpoint"] == str(tmp_path / "out" / "checkpoint-epoch-0")


def error_contract(capsys, code: int):
    out = capsys.readouterr()
    assert out.out == ""
    payload = json.loads(out.err)
    assert payload["ok"] is False
    assert isinstance(payload["code"], int)
    assert "Traceback" not in out.err
    assert code == payload["code"]
    return payload


def test_usage_error_is_exactly_one_json(tmp_path: Path, capsys):
    project, models, _, _, _, _, _ = setup(tmp_path)
    code = main(
        [
            "finetune",
            "train",
            "--project",
            str(project),
            "--register-only",
            "--json",
            *common(tmp_path, models),
        ]
    )
    assert code == 2
    payload = error_contract(capsys, 2)
    assert payload["error"] == "usage"


def test_input_error_is_exactly_one_json(tmp_path: Path, capsys):
    project, models, _, _, _, _, _ = setup(tmp_path)
    (tmp_path / "official" / "sft_12hz.py").unlink()
    code = main(
        [
            "finetune",
            "train",
            "--project",
            str(project),
            "--json",
            *common(tmp_path, models),
        ]
    )
    assert code == 4
    payload = error_contract(capsys, 4)
    assert payload["error"] == "input"


def test_model_error_is_exactly_one_json(tmp_path: Path, monkeypatch, capsys):
    project, models, _, _, _, _, _ = setup(tmp_path)

    def bad(fake, call):
        argv = call["argv"]
        output = Path(argv[argv.index("--output_model_path") + 1])
        write_checkpoint(output, 0, "alice", tts_model_type="base")

    monkeypatch.setattr(runner_mod, "run_external", FakeExternal(bad))
    code = main(
        [
            "finetune",
            "train",
            "--project",
            str(project),
            "--json",
            *common(tmp_path, models),
        ]
    )
    assert code == 3
    payload = error_contract(capsys, 3)
    assert payload["error"] == "model"


def test_interrupt_is_130_with_empty_stdout(tmp_path: Path, monkeypatch, capsys):
    project, models, _, _, _, _, _ = setup(tmp_path)

    def interrupt(argv, *, cwd, env, quiet=False):
        raise KeyboardInterrupt

    monkeypatch.setattr(runner_mod, "run_external", interrupt)
    code = main(
        [
            "finetune",
            "train",
            "--project",
            str(project),
            "--json",
            *common(tmp_path, models),
        ]
    )
    assert code == 130
    out = capsys.readouterr()
    assert out.out == ""
    assert "Traceback" not in out.err


class FakeEngine:
    def __init__(self, cfg, model):
        self.model = None

    def generate(self, **kwargs):
        return np.ones(2400, dtype=np.float32), 24000, {}


class FakeAsr:
    def __init__(self, text):
        self.text = text

    def transcribe_samples(self, audio, sr, *, language):
        return {"text": self.text}

    def release(self):
        pass


def test_gate_failure_exit_5_with_json_payload(tmp_path: Path, monkeypatch, capsys):
    project, models, _, _, _, _, _ = setup(tmp_path, output=True, eval_block=True)
    asr = models / "Qwen3-ASR-0.6B"
    asr.mkdir()
    (asr / "config.json").write_text(
        json.dumps({"model_type": "qwen3_asr", "support_languages": ["english"]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(evaluate_mod, "_engine_factory", lambda cfg, m: FakeEngine(cfg, m))
    monkeypatch.setattr(evaluate_mod, "_asr_factory", lambda cfg, m: FakeAsr("wrong"))
    monkeypatch.setattr(evaluate_mod, "_clear_tts_cache", lambda: None)
    monkeypatch.setattr(evaluate_mod, "_clear_asr_cache", lambda: None)
    code = main(
        [
            "finetune",
            "eval",
            "--project",
            str(project),
            "--asr-model",
            "Qwen3-ASR-0.6B",
            "--max-cer",
            "0",
            "--json",
            *common(tmp_path, models),
        ]
    )
    assert code == 5
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["exit_code"] == 5
    assert payload["gate_failures"]


def test_human_summary_does_not_pollute_json(tmp_path: Path, capsys):
    project, models, _, _, _, _, _ = setup(tmp_path)
    code = main(
        ["finetune", "train", "--project", str(project), "--dry-run", "--json", *common(tmp_path, models)]
    )
    assert code == 0
    out = capsys.readouterr()
    json.loads(out.out)
    assert "would train" not in out.out
    assert "would train" not in out.err


def test_help_lists_finetune():
    assert "finetune" in build_parser().format_help()


def test_finetune_import_and_dry_run_are_torch_free(tmp_path: Path):
    project, models, tokenizer, _, _, _, _ = setup(tmp_path)
    code = (
        "import sys\n"
        "from qtts_cli.cli import main\n"
        "main(['finetune', 'train', '--project', sys.argv[1], '--dry-run', '--json',\n"
        "      '--model-dir', sys.argv[2]])\n"
        "bad = [m for m in ('torch','qwen_tts','qwen_asr','dashscope',"
        "'vllm_omni','accelerate','librosa','safetensors') if m in sys.modules]\n"
        "print('BAD=' + ','.join(bad), file=sys.stderr)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(project), str(models)],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "PYTHONPATH": "."},
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout)["dry_run"] is True
    markers = [line for line in result.stderr.splitlines() if line.startswith("BAD=")]
    assert markers == ["BAD="]
