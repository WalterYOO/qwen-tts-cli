"""FR-P1-08: `say --stdin --jsonl` with a fake backend."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from qtts_cli.cli import main
from qtts_cli.commands import say as say_cmd
from qtts_cli.subtitles import AlignItem, Cue


class FakeEngine:
    loads = 0

    def __init__(self, cfg, model):
        self.cfg = cfg
        self.model_info = model
        self.device = "cpu"
        self.dtype_name = "fp32"
        self.load_ms = 0.0
        self.model = None

    def load(self):
        FakeEngine.loads += 1
        self.model = object()
        return self

    def generate(self, **kwargs):
        return np.zeros(2400, dtype=np.float32), 24000, {"generate_ms": 1.0}

    def build_clone_prompt_items(self, *args, **kwargs):
        return []


class TrackingEngine(FakeEngine):
    loads = 0
    live = 0
    max_live = 0
    instances = []

    def __init__(self, cfg, model):
        super().__init__(cfg, model)
        self._model = None
        TrackingEngine.instances.append(self)

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        if value is not None and getattr(self, "_model", None) is None:
            TrackingEngine.live += 1
            TrackingEngine.max_live = max(TrackingEngine.max_live, TrackingEngine.live)
        elif value is None and getattr(self, "_model", None) is not None:
            TrackingEngine.live -= 1
        self._model = value

    def load(self):
        TrackingEngine.loads += 1
        self.model = object()
        return self


class FakeAligner:
    def __init__(self, cfg, model):
        self.load_ms = 0.0

    def align(self, audio, sample_rate, text, language):
        items = [
            AlignItem(ch, i * 0.1, (i + 1) * 0.1)
            for i, ch in enumerate(text)
            if not ch.isspace()
        ]
        return items, {"align_ms": 1.0}


@pytest.fixture()
def fake_engine(monkeypatch):
    FakeEngine.loads = 0
    monkeypatch.setattr(say_cmd, "TtsEngine", FakeEngine)
    return FakeEngine


def _run(
    monkeypatch,
    tmp_path: Path,
    model_dir: Path,
    lines: list[object],
    *extra: str,
    no_cache: bool = True,
):
    if lines:
        stdin = "".join(
            (json.dumps(line) if not isinstance(line, str) else line) + "\n"
            for line in lines
        )
    else:
        stdin = ""
    monkeypatch.setattr(sys, "stdin", io.StringIO(stdin))
    return main(
        [
            "say",
            "--stdin",
            "--jsonl",
            "--voice",
            "builtin:Vivian",
            "--out-dir",
            str(tmp_path / "out"),
            "--model-dir",
            str(model_dir),
            "--data-dir",
            str(tmp_path / "data"),
            *(("--no-cache",) if no_cache else ()),
            *extra,
        ]
    )


def _records(capsys):
    out, _ = capsys.readouterr()
    return [json.loads(line) for line in out.splitlines() if line.strip()]


def test_jsonl_success_reuses_one_model_load(monkeypatch, capsys, tmp_path: Path, model_dir: Path, fake_engine):
    rc = _run(
        monkeypatch,
        tmp_path,
        model_dir,
        [{"id": "a", "text": "你好。"}, {"id": "b", "text": "世界。"}],
    )
    assert rc == 0
    records = _records(capsys)
    assert [record["id"] for record in records] == ["a", "b"]
    assert all(record["status"] == "ok" for record in records)
    assert all(Path(record["output"]).is_file() for record in records)
    assert all(Path(record["recipe"]).is_file() for record in records)
    assert (tmp_path / "out" / "a.wav").is_file()
    assert (tmp_path / "out" / "b.wav").is_file()
    assert fake_engine.loads == 1


def test_jsonl_bad_lines_do_not_stop_the_stream(monkeypatch, capsys, tmp_path: Path, model_dir: Path, fake_engine):
    rc = _run(
        monkeypatch,
        tmp_path,
        model_dir,
        [
            {"id": "a", "text": "你好。"},
            "{not json",
            {"id": "b"},
            {"id": "a", "text": "重复。"},
            {"id": "c", "text": "继续。"},
        ],
    )
    assert rc == 6
    records = _records(capsys)
    assert [record["status"] for record in records] == ["ok", "error", "error", "error", "ok"]
    assert records[1]["id"] is None
    assert records[1]["error"]["code"] == "invalid_json"
    assert records[2]["error"]["code"] == "invalid_request"
    assert records[3]["error"]["code"] == "duplicate_id"
    assert (tmp_path / "out" / "c.wav").is_file()


def test_jsonl_single_failure_continues(monkeypatch, capsys, tmp_path: Path, model_dir: Path, fake_engine):
    rc = _run(
        monkeypatch,
        tmp_path,
        model_dir,
        [
            {"id": "a", "text": "你好。"},
            {"id": "b", "text": "世界。", "voice": "builtin:Nobody"},
            {"id": "c", "text": "继续。"},
        ],
    )
    assert rc == 6
    records = _records(capsys)
    assert [record["status"] for record in records] == ["ok", "error", "ok"]
    assert records[1]["error"]["details"]["exit_code"] == 4


def test_jsonl_empty_stdin_is_usage_error(monkeypatch, capsys, tmp_path: Path, model_dir: Path, fake_engine):
    rc = _run(monkeypatch, tmp_path, model_dir, [])
    assert rc == 2
    out, err = capsys.readouterr()
    assert out == ""
    assert "no requests" in err


def test_jsonl_startup_argument_errors(monkeypatch, capsys, tmp_path: Path, model_dir: Path, fake_engine):
    common = [
        "say", "--model-dir", str(model_dir), "--data-dir", str(tmp_path / "data"),
        "--voice", "builtin:Vivian",
    ]
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    assert main([*common, "--jsonl"]) == 2
    assert main([*common, "--jsonl", "--stdin", "--json"]) == 2
    assert main([*common, "--jsonl", "--stdin", "--out", str(tmp_path / "x.wav")]) == 2
    out, _ = capsys.readouterr()
    assert out == ""


def test_jsonl_rejects_stdout_per_record_and_unsafe_ids(
    monkeypatch, capsys, tmp_path: Path, model_dir: Path, fake_engine
):
    rc = _run(
        monkeypatch,
        tmp_path,
        model_dir,
        [
            {"id": "escape", "text": "你好。", "out": "-"},
            {"id": "../escape", "text": "你好。"},
            {"id": "A", "text": "世界。"},
            {"id": "a", "text": "继续。"},
        ],
    )
    assert rc == 6
    records = _records(capsys)
    assert records[0]["error"]["code"] == "invalid_request"
    assert records[1]["status"] == "ok"
    assert Path(records[1]["output"]).parent == tmp_path / "out"
    assert not (tmp_path / "escape.wav").exists()
    assert records[2]["status"] == "ok"
    assert records[3]["error"]["code"] == "id_filename_conflict"


def test_jsonl_rejects_duplicate_output_paths(
    monkeypatch, capsys, tmp_path: Path, model_dir: Path, fake_engine
):
    shared = tmp_path / "out" / "shared.wav"
    rc = _run(
        monkeypatch,
        tmp_path,
        model_dir,
        [
            {"id": "a", "text": "你好。", "out": str(shared)},
            {"id": "b", "text": "世界。", "out": str(shared)},
        ],
    )
    assert rc == 6
    records = _records(capsys)
    assert records[0]["status"] == "ok"
    assert records[1]["error"]["code"] == "output_path_conflict"
    assert shared.is_file()


def test_jsonl_keeps_only_one_tts_model_resident(
    monkeypatch, capsys, tmp_path: Path, model_dir: Path
):
    TrackingEngine.loads = 0
    TrackingEngine.live = 0
    TrackingEngine.max_live = 0
    TrackingEngine.instances = []
    monkeypatch.setattr(say_cmd, "TtsEngine", TrackingEngine)
    rc = _run(
        monkeypatch,
        tmp_path,
        model_dir,
        [
            {"id": "a", "text": "你好。", "voice": "builtin:Vivian"},
            {"id": "b", "text": "世界。", "voice": "design:温柔"},
            {"id": "c", "text": "继续。", "voice": "builtin:Vivian"},
        ],
    )
    assert rc == 0
    assert TrackingEngine.max_live == 1
    assert TrackingEngine.loads == 3
    _records(capsys)


def test_jsonl_switch_releases_the_previous_engine(
    monkeypatch, capsys, tmp_path: Path, model_dir: Path
):
    TrackingEngine.loads = 0
    TrackingEngine.live = 0
    TrackingEngine.max_live = 0
    TrackingEngine.instances = []
    monkeypatch.setattr(say_cmd, "TtsEngine", TrackingEngine)
    rc = _run(
        monkeypatch,
        tmp_path,
        model_dir,
        [
            {"id": "a", "text": "你好。", "voice": "builtin:Vivian"},
            {"id": "b", "text": "世界。", "voice": "design:温柔"},
        ],
    )
    assert rc == 0
    assert len(TrackingEngine.instances) == 2
    assert TrackingEngine.instances[0].model is None
    assert TrackingEngine.instances[1].model is not None
    _records(capsys)


def test_jsonl_cache_hit_reuses_the_single_engine(
    monkeypatch, capsys, tmp_path: Path, model_dir: Path
):
    TrackingEngine.loads = 0
    TrackingEngine.live = 0
    TrackingEngine.max_live = 0
    TrackingEngine.instances = []
    monkeypatch.setattr(say_cmd, "TtsEngine", TrackingEngine)
    rc = _run(
        monkeypatch,
        tmp_path,
        model_dir,
        [
            {"id": "a", "text": "你好。"},
            {"id": "b", "text": "你好。"},
        ],
        no_cache=False,
    )
    assert rc == 0
    assert TrackingEngine.loads == 1
    assert len(TrackingEngine.instances) == 1
    records = _records(capsys)
    assert records[1]["metrics"]["cache"]["hits"] >= 1


def test_say_join_invariant_failure_exits_4(
    monkeypatch, capsys, tmp_path: Path, model_dir: Path, fake_engine
):
    monkeypatch.setattr(say_cmd, "AlignerEngine", FakeAligner)
    monkeypatch.setattr(
        say_cmd, "build_cues", lambda *args, **kwargs: [Cue(0.0, 0.1, "错误文本")]
    )
    out = tmp_path / "out.wav"
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    rc = main([
        "say", "你好。", "--voice", "builtin:Vivian", "--out", str(out),
        "--subtitle", "srt", "--model-dir", str(model_dir),
        "--data-dir", str(tmp_path / "data"), "--no-cache", "--json",
    ])
    assert rc == 4
    _, err = capsys.readouterr()
    assert "does not match the normalized source text" in err
    assert not out.exists()
    assert not (tmp_path / "out.srt").exists()
