"""FR-P2-07 §3/§12.2: audio validation and deterministic raw JSONL."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from qtts_cli.errors import InputError, ModelError
from qtts_cli.finetune import dataset as ds
from qtts_cli.finetune.project import load_project
from finetune_utils import write_tone


def _dataset(root: Path, *, item_seconds=(1.0, 1.5), ref_seconds=2.5):
    items = [
        SimpleNamespace(
            audio=write_tone(root / f"u{index}.wav", seconds=seconds), text=f"文本{index}"
        )
        for index, seconds in enumerate(item_seconds, 1)
    ]
    ref = write_tone(root / "ref.wav", seconds=ref_seconds)
    return SimpleNamespace(items=items, ref_audio=ref, language="Auto")


def test_valid_inclusive_durations(tmp_path: Path):
    dataset = _dataset(tmp_path, item_seconds=(0.5, 30.0), ref_seconds=2.0)
    info = ds.validate_dataset_audio(dataset)
    assert info["items"] == 2
    assert info["duration_s"] == pytest.approx(30.5, abs=0.01)
    # The maximum ref duration is also inclusive.
    ref30 = write_tone(tmp_path / "ref30.wav", seconds=30.0)
    dataset.ref_audio = ref30
    ds.validate_dataset_audio(dataset)


def test_missing_and_non_file(tmp_path: Path):
    dataset = _dataset(tmp_path)
    dataset.items[0].audio = tmp_path / "missing.wav"
    with pytest.raises(InputError) as exc:
        ds.validate_dataset_audio(dataset)
    assert "path" in exc.value.details
    directory = tmp_path / "dir.wav"
    directory.mkdir()
    dataset.items[0].audio = directory
    with pytest.raises(InputError):
        ds.validate_dataset_audio(dataset)


def test_wrong_container(tmp_path: Path):
    dataset = _dataset(tmp_path)
    dataset.items[0].audio = write_tone(tmp_path / "u.aiff", fmt="AIFF")
    with pytest.raises(InputError) as exc:
        ds.validate_dataset_audio(dataset)
    assert exc.value.details["check"] == "container"


@pytest.mark.parametrize("rate", [22050, 16000])
def test_wrong_sample_rate(tmp_path: Path, rate: int):
    dataset = _dataset(tmp_path)
    dataset.items[0].audio = write_tone(tmp_path / "u.wav", sample_rate=rate)
    with pytest.raises(InputError) as exc:
        ds.validate_dataset_audio(dataset)
    assert exc.value.details["check"] == "sample_rate"


def test_stereo_rejected(tmp_path: Path):
    dataset = _dataset(tmp_path)
    dataset.items[0].audio = write_tone(tmp_path / "u.wav", channels=2)
    with pytest.raises(InputError) as exc:
        ds.validate_dataset_audio(dataset)
    assert exc.value.details["check"] == "channels"


@pytest.mark.parametrize(("seconds", "check"), [(0.49, "duration"), (30.01, "duration")])
def test_item_duration_bounds(tmp_path: Path, seconds: float, check: str):
    dataset = _dataset(tmp_path)
    dataset.items[0].audio = write_tone(tmp_path / "u.wav", seconds=seconds)
    with pytest.raises(InputError) as exc:
        ds.validate_dataset_audio(dataset)
    assert exc.value.details["check"] == check


def test_ref_duration_bounds(tmp_path: Path):
    dataset = _dataset(tmp_path)
    dataset.ref_audio = write_tone(tmp_path / "ref.wav", seconds=1.99)
    with pytest.raises(InputError) as exc:
        ds.validate_dataset_audio(dataset)
    assert exc.value.details["check"] == "duration"


def test_text_rules(tmp_path: Path):
    base = """
schema_version: 1
dataset:
  ref_audio: ref.wav
  items:
    - audio: u.wav
      text: TEXT
"""
    good = tmp_path / "good.yaml"
    good.write_text(base.replace("TEXT", "'  hello  '"), encoding="utf-8")
    project = load_project(good)
    assert project.dataset.items[0].text == "hello"
    for text in ("'   '", '"' + "字" * 2001 + '"'):
        bad = tmp_path / "bad.yaml"
        bad.write_text(base.replace("TEXT", text), encoding="utf-8")
        with pytest.raises(InputError):
            load_project(bad)
    empty = base.replace("    - audio: u.wav\n      text: TEXT\n", "")
    bad = tmp_path / "empty.yaml"
    bad.write_text(empty, encoding="utf-8")
    with pytest.raises(InputError):
        load_project(bad)


def test_duplicate_audio(tmp_path: Path):
    body = """
schema_version: 1
dataset:
  ref_audio: ref.wav
  items:
    - audio: u.wav
      text: a
    - audio: ./u.wav
      text: b
"""
    path = tmp_path / "p.yaml"
    path.write_text(body, encoding="utf-8")
    with pytest.raises(InputError):
        load_project(path)


def test_exact_raw_bytes_and_deterministic(tmp_path: Path):
    dataset = _dataset(tmp_path)
    target = tmp_path / "out" / "raw.jsonl"
    digest = ds.write_raw_jsonl(target, dataset)
    content = target.read_text(encoding="utf-8")
    expected = "".join(
        json.dumps(
            {
                "audio": str(item.audio),
                "text": item.text,
                "ref_audio": str(dataset.ref_audio),
            },
            ensure_ascii=False,
        )
        + "\n"
        for item in dataset.items
    )
    assert content == expected
    assert content.count("\n") == 2
    assert " " not in content.splitlines()[0].split('"')[0]  # default separators only
    assert digest == ds.write_raw_jsonl(target, dataset)
    assert target.read_text(encoding="utf-8") == content


def test_atomic_temp_cleanup_on_failure(tmp_path: Path, monkeypatch):
    dataset = _dataset(tmp_path)
    target = tmp_path / "raw.jsonl"

    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(ds.os, "replace", fail)
    with pytest.raises(ModelError):
        ds.write_raw_jsonl(target, dataset)
    assert not Path(f"{target}.tmp").exists()
    assert not target.exists()


def test_items_sha_and_hashes(tmp_path: Path):
    dataset = _dataset(tmp_path)
    value = ds.items_sha256(dataset)
    assert len(value) == 64
    raw, prepared = ds.raw_prepared_hashes(tmp_path / "nope", tmp_path / "nope2")
    assert raw is None and prepared is None
