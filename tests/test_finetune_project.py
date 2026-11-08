"""FR-P2-07 §2/§12.1: strict finetune project schema."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from qtts_cli.errors import InputError
from qtts_cli.finetune.project import DEVICE_CHOICES, load_project

BASE = """
schema_version: 1
dataset:
  ref_audio: ref.wav
  items:
    - audio: utt_1.wav
      text: "你好"
    - audio: utt_2.wav
      text: "hello"
"""


def _write(tmp_path: Path, text: str, name: str = "finetune.yaml") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_minimal_yaml_and_json(tmp_path: Path):
    path = _write(tmp_path, BASE)
    project = load_project(path)
    assert project.dataset.language == "Auto"
    assert len(project.dataset.items) == 2
    assert project.dataset.items[0].text == "你好"
    assert project.prepare.device is None
    assert project.train.speaker_name is None
    assert project.eval is None

    data = {
        "schema_version": 1,
        "dataset": {"ref_audio": "ref.wav", "items": [{"audio": "u.wav", "text": "x"}]},
    }
    json_path = _write(tmp_path, json.dumps(data), "finetune.json")
    project = load_project(json_path)
    assert len(project.dataset.items) == 1


def test_unknown_keys(tmp_path: Path):
    for body in (
        BASE + "\nunknown: 1\n",
        BASE + "\nprepare:\n  unknown: 1\n",
        BASE + "\nregister:\n  nope: 1\n",
        BASE + "\neval:\n  items:\n    - text: x\n      unknown: 1\n",
        BASE + "\ntrain:\n  unknown: 1\n",
        BASE + "\ndataset:\n  ref_audio: r.wav\n  unknown: 1\n  items:\n    - audio: a.wav\n      text: x\n",
    ):
        with pytest.raises(InputError):
            load_project(_write(tmp_path, body))


@pytest.mark.parametrize("value", ["2", "0", "true", "1.0"])
def test_schema_version_invalid(tmp_path: Path, value: str):
    body = BASE.replace("schema_version: 1", f"schema_version: {value}")
    with pytest.raises(InputError):
        load_project(_write(tmp_path, body))


def test_duplicate_json_keys(tmp_path: Path):
    text = (
        '{"schema_version":1,"schema_version":1,"dataset":{"ref_audio":"r.wav",'
        '"items":[{"audio":"a.wav","text":"x"}]}}'
    )
    with pytest.raises(InputError):
        load_project(_write(tmp_path, text, "p.json"))


def test_yaml_deep_nesting(tmp_path: Path):
    nested = "x: " + "[" * 120 + "]" * 120 + "\n"
    with pytest.raises(InputError):
        load_project(_write(tmp_path, BASE + nested))


def test_size_limit(tmp_path: Path):
    body = BASE + "\nprepare:\n  raw_jsonl: " + "a" * 1_000_001 + "\n"
    with pytest.raises(InputError):
        load_project(_write(tmp_path, body))


def test_wrong_types_and_non_finite(tmp_path: Path):
    with pytest.raises(InputError):
        load_project(
            _write(
                tmp_path,
                BASE
                + "\ntrain:\n  batch_size: true\n",
            )
        )
    nan_project = {
        "schema_version": 1,
        "dataset": {"ref_audio": "r.wav", "items": [{"audio": "a.wav", "text": "x"}]},
        "train": {"lr": float("nan")},
    }
    nan_json = json.dumps(nan_project)
    with pytest.raises(InputError):
        load_project(_write(tmp_path, nan_json, "p.json"))
    bool_lr = dict(nan_project, train={"lr": True})
    with pytest.raises(InputError):
        load_project(_write(tmp_path, json.dumps(bool_lr), "p2.json"))


def test_speaker_name_lowercased_and_regex(tmp_path: Path):
    project = load_project(
        _write(tmp_path, BASE + "\ntrain:\n  speaker_name: Alice_01\n")
    )
    assert project.train.speaker_name == "alice_01"
    for bad in ("-alice", "_alice", "a" * 65, "alice/../x", "alice space"):
        with pytest.raises(InputError):
            load_project(
                _write(
                    tmp_path,
                    BASE + '\ntrain:\n  speaker_name: "' + bad + '"\n',
                    "bad.yaml",
                )
            )


def test_reserved_register_name(tmp_path: Path):
    with pytest.raises(InputError):
        load_project(
            _write(tmp_path, BASE + "\nregister:\n  name: builtin\n")
        )


def test_threshold_ranges(tmp_path: Path):
    good = BASE + "\neval:\n  items:\n    - text: x\n  thresholds:\n    max_cer: 0\n    min_speaker_similarity: 1\n"
    load_project(_write(tmp_path, good))
    for body in (
        BASE + "\neval:\n  items:\n    - text: x\n  thresholds:\n    max_cer: -0.1\n",
        BASE + "\neval:\n  items:\n    - text: x\n  thresholds:\n    min_speaker_similarity: 1.1\n",
        BASE + "\neval:\n  items:\n    - text: x\n  thresholds:\n    min_speaker_similarity: -0.1\n",
    ):
        with pytest.raises(InputError):
            load_project(_write(tmp_path, body))


def test_prepare_device_domain(tmp_path: Path):
    assert DEVICE_CHOICES == ("auto", "cpu", "mps", "cuda", "cuda:0", "cuda:1")
    for device in DEVICE_CHOICES:
        load_project(
            _write(tmp_path, BASE + f"\nprepare:\n  device: {device}\n")
        )
    with pytest.raises(InputError):
        load_project(_write(tmp_path, BASE + "\nprepare:\n  device: gpu\n"))


def test_paths_resolve_relative_to_project(tmp_path: Path):
    sub = tmp_path / "sub"
    sub.mkdir()
    path = _write(
        sub,
        """
schema_version: 1
dataset:
  ref_audio: audio/ref.wav
  items:
    - audio: audio/u.wav
      text: x
prepare:
  raw_jsonl: out/raw.jsonl
  output_jsonl: out/train.jsonl
train:
  output_dir: out/model
  init_model_path: ../base
  finetuning_dir: ../official
""",
    )
    project = load_project(path)
    assert project.dataset.ref_audio == (sub / "audio/ref.wav").resolve()
    assert project.dataset.items[0].audio == (sub / "audio/u.wav").resolve()
    assert project.prepare.raw_jsonl == (sub / "out/raw.jsonl").resolve()
    assert project.train.output_dir == (sub / "out/model").resolve()
    assert project.train.init_model_path == (tmp_path / "base").resolve()
    assert project.train.finetuning_dir == (tmp_path / "official").resolve()


def test_nul_rejected(tmp_path: Path):
    path = tmp_path / "nul.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset": {
                    "ref_audio": "ref\u0000.wav",
                    "items": [{"audio": "a.wav", "text": "x"}],
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(InputError):
        load_project(path)


def test_output_dir_containment(tmp_path: Path):
    data_dir = tmp_path / "data"
    official = tmp_path / "official"
    base = tmp_path / "base"
    body = BASE + f"""
train:
  finetuning_dir: {official}
  init_model_path: {base}
  output_dir: {tmp_path / 'out'}
register:
  name: alice
  consent:
    rights_confirmed: true
"""
    path = _write(tmp_path, body)
    load_project(path, data_dir=data_dir)
    # Inside the voice store is forbidden.
    bad = body.replace(str(tmp_path / "out"), str(data_dir / "voices" / "alice"))
    with pytest.raises(InputError):
        load_project(_write(tmp_path, bad, "bad.yaml"), data_dir=data_dir, register_name="alice")
    # Parent of data_dir is also an overlap.
    with pytest.raises(InputError):
        load_project(
            _write(
                tmp_path,
                body.replace(str(tmp_path / "out"), str(tmp_path)),
                "bad2.yaml",
            ),
            data_dir=data_dir,
        )
    # A dataset input may not live inside the output dir.
    inside = """
schema_version: 1
dataset:
  ref_audio: audio/ref.wav
  items:
    - audio: audio/u.wav
      text: x
train:
  finetuning_dir: official
  init_model_path: base
  output_dir: audio
"""
    with pytest.raises(InputError):
        load_project(_write(tmp_path, inside, "bad3.yaml"))
