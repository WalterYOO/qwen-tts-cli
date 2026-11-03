"""FR-P1-01: batch manifest plan/run, resume, retry and failure isolation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from qtts_cli.cli import main
from qtts_cli.commands import say as say_cmd
from qtts_cli.errors import ModelError


class FakeEngine:
    loads = 0
    generated = 0
    fail_texts: set[str] = set()
    fail_once_texts: set[str] = set()
    seen_once: set[str] = set()
    fail_exc: BaseException | None = None

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
        text = kwargs["text"]
        if FakeEngine.fail_exc is not None:
            raise FakeEngine.fail_exc
        if text in FakeEngine.fail_texts:
            raise ModelError(f"permanent failure for {text}")
        if text in FakeEngine.fail_once_texts and text not in FakeEngine.seen_once:
            FakeEngine.seen_once.add(text)
            raise ModelError(f"transient failure for {text}")
        FakeEngine.generated += 1
        return np.zeros(2400, dtype=np.float32), 24000, {"generate_ms": 1.0}

    def build_clone_prompt_items(self, *args, **kwargs):
        return []


_ORIGINAL_GENERATE = FakeEngine.generate


@pytest.fixture()
def fake_engine(monkeypatch):
    FakeEngine.loads = 0
    FakeEngine.generated = 0
    FakeEngine.fail_texts = set()
    FakeEngine.fail_once_texts = set()
    FakeEngine.seen_once = set()
    FakeEngine.fail_exc = None
    FakeEngine.generate = _ORIGINAL_GENERATE  # type: ignore[assignment]
    monkeypatch.setattr(say_cmd, "TtsEngine", FakeEngine)
    return FakeEngine


def _manifest(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "manifest.jsonl"
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def _run(tmp_path: Path, model_dir: Path, *args: str) -> int:
    return main(
        [
            *args,
            "--model-dir",
            str(model_dir),
            "--data-dir",
            str(tmp_path / "data"),
            "--cache-dir",
            str(tmp_path / "cache"),
        ]
    )


def _rows(count: int = 2) -> list[dict]:
    return [
        {
            "id": f"job{index}",
            "text": f"这是第{index}段。",
            "voice": "builtin:Vivian",
            "language": "Chinese",
        }
        for index in range(count)
    ]


def _status(out_dir: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in (out_dir / "status.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_batch_plan_is_deterministic_and_loads_no_model(
    monkeypatch, capsys, fake_engine, tmp_path: Path, model_dir: Path
):
    manifest = _manifest(tmp_path, _rows())
    out_dir = tmp_path / "out"

    def boom(self):
        raise AssertionError("batch plan must not load models")

    monkeypatch.setattr(FakeEngine, "load", boom)
    rc = _run(tmp_path, model_dir, "batch", "plan", str(manifest), "--out-dir", str(out_dir))
    assert rc == 0
    capsys.readouterr()
    first = (out_dir / "plan.json").read_bytes()
    rc = _run(tmp_path, model_dir, "batch", "plan", str(manifest), "--out-dir", str(out_dir))
    assert rc == 0
    capsys.readouterr()
    assert (out_dir / "plan.json").read_bytes() == first
    statuses = _status(out_dir)
    assert [item["status"] for item in statuses] == ["pending", "pending"]
    assert [item["id"] for item in statuses] == ["job0", "job1"]


def test_bad_json_duplicate_id_and_unknown_voice_fail_at_startup(
    capsys, tmp_path: Path, model_dir: Path
):
    bad = tmp_path / "bad.jsonl"
    bad.write_text("{not json\n", encoding="utf-8")
    assert _run(
        tmp_path, model_dir, "batch", "plan", str(bad), "--out-dir", str(tmp_path / "o1")
    ) == 2
    capsys.readouterr()

    duplicate = _manifest(tmp_path, [_rows()[0], _rows()[0]])
    assert _run(
        tmp_path, model_dir, "batch", "plan", str(duplicate), "--out-dir", str(tmp_path / "o2")
    ) == 4
    capsys.readouterr()

    unknown = _manifest(
        tmp_path,
        [{"id": "x", "text": "你好。", "voice": "builtin:Nobody"}],
    )
    assert _run(
        tmp_path, model_dir, "batch", "plan", str(unknown), "--out-dir", str(tmp_path / "o3")
    ) == 4
    capsys.readouterr()


def test_run_validates_all_jobs_before_generating(
    fake_engine, tmp_path: Path, model_dir: Path
):
    rows = _rows(1)
    rows.append(
        {"id": "bad", "text": "你好。", "voice": "builtin:Nobody", "language": "Chinese"}
    )
    out_dir = tmp_path / "out"
    rc = _run(
        tmp_path,
        model_dir,
        "batch",
        "run",
        str(_manifest(tmp_path, rows)),
        "--out-dir",
        str(out_dir),
    )
    assert rc == 4
    assert not (out_dir / "job0.wav").exists()
    assert FakeEngine.generated == 0


def test_batch_run_all_success(fake_engine, tmp_path: Path, model_dir: Path):
    manifest = _manifest(tmp_path, _rows())
    out_dir = tmp_path / "out"
    rc = _run(
        tmp_path,
        model_dir,
        "batch",
        "run",
        str(manifest),
        "--out-dir",
        str(out_dir),
        "--json",
    )
    assert rc == 0
    assert [item["status"] for item in _status(out_dir)] == ["done", "done"]
    for index in range(2):
        output = out_dir / f"job{index}.wav"
        assert output.is_file()
        recipe = json.loads(
            output.with_suffix(".recipe.json").read_text(encoding="utf-8")
        )
        assert recipe["job_key"].startswith("sha256:")


def test_batch_partial_failure_exits_6(fake_engine, tmp_path: Path, model_dir: Path):
    rows = _rows(3)
    rows[1]["text"] = "FAIL。"
    FakeEngine.fail_texts = {"FAIL。"}
    out_dir = tmp_path / "out"
    rc = _run(
        tmp_path,
        model_dir,
        "batch",
        "run",
        str(_manifest(tmp_path, rows)),
        "--out-dir",
        str(out_dir),
        "--json",
    )
    assert rc == 6
    statuses = {item["id"]: item for item in _status(out_dir)}
    assert statuses["job0"]["status"] == "done"
    assert statuses["job1"]["status"] == "failed"
    assert statuses["job2"]["status"] == "done"
    assert statuses["job1"]["last_error"]["code"] == 3
    assert statuses["job1"]["transient"] is True
    assert "error" not in statuses["job1"]


def test_batch_fail_fast_marks_unstarted_skipped(
    fake_engine, tmp_path: Path, model_dir: Path
):
    rows = [
        {"id": "fail", "text": "FAIL。", "voice": "builtin:Vivian", "language": "Chinese"},
        {
            "id": "later1",
            "text": "这是一个很长很长的成功句子。",
            "voice": "builtin:Vivian",
            "language": "Chinese",
        },
        {
            "id": "later2",
            "text": "另一个同样很长的成功句子。",
            "voice": "builtin:Vivian",
            "language": "Chinese",
        },
    ]
    FakeEngine.fail_texts = {"FAIL。"}
    out_dir = tmp_path / "out"
    rc = _run(
        tmp_path,
        model_dir,
        "batch",
        "run",
        str(_manifest(tmp_path, rows)),
        "--out-dir",
        str(out_dir),
        "--fail-fast",
        "--jobs",
        "1",
    )
    assert rc == 6
    statuses = {item["id"]: item["status"] for item in _status(out_dir)}
    assert statuses["fail"] == "failed"
    assert statuses["later1"] == "skipped"
    assert statuses["later2"] == "skipped"


def test_batch_retry_only_transient_failures(fake_engine, tmp_path: Path, model_dir: Path):
    rows = [
        {"id": "flaky", "text": "FLAKY。", "voice": "builtin:Vivian", "language": "Chinese"}
    ]
    FakeEngine.fail_once_texts = {"FLAKY。"}
    out_dir = tmp_path / "out"
    rc = _run(
        tmp_path,
        model_dir,
        "batch",
        "run",
        str(_manifest(tmp_path, rows)),
        "--out-dir",
        str(out_dir),
        "--retry",
        "2",
    )
    assert rc == 0
    status = _status(out_dir)[0]
    assert status["status"] == "done"
    assert status["attempts"] == 2


def test_batch_resume_skips_verified_done_and_recovers_running(
    fake_engine, tmp_path: Path, model_dir: Path
):
    manifest = _manifest(tmp_path, _rows())
    out_dir = tmp_path / "out"
    assert _run(
        tmp_path, model_dir, "batch", "run", str(manifest), "--out-dir", str(out_dir)
    ) == 0
    loads = FakeEngine.loads

    def boom(self, **kwargs):
        raise AssertionError("verified done job must not run")

    FakeEngine.generate = boom  # type: ignore[assignment]
    assert _run(
        tmp_path,
        model_dir,
        "batch",
        "run",
        str(manifest),
        "--out-dir",
        str(out_dir),
        "--resume",
    ) == 0
    assert FakeEngine.loads == loads

    # Crash recovery: a stale running row becomes pending on resume.
    statuses = _status(out_dir)
    statuses[0]["status"] = "running"
    (out_dir / "status.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in statuses), encoding="utf-8"
    )

    def generate(self, **kwargs):
        FakeEngine.generated += 1
        return np.zeros(2400, dtype=np.float32), 24000, {"generate_ms": 1.0}

    FakeEngine.generate = generate  # type: ignore[assignment]
    assert _run(
        tmp_path,
        model_dir,
        "batch",
        "run",
        str(manifest),
        "--out-dir",
        str(out_dir),
        "--resume",
    ) == 0
    assert all(item["status"] == "done" for item in _status(out_dir))


def test_jobs_two_matches_jobs_one_mapping(fake_engine, tmp_path: Path, model_dir: Path):
    rows = _rows(4)
    rows[2]["text"] = "FAIL。"
    FakeEngine.fail_texts = {"FAIL。"}
    manifest = _manifest(tmp_path, rows)
    first_dir = tmp_path / "one"
    second_dir = tmp_path / "two"
    assert _run(
        tmp_path,
        model_dir,
        "batch",
        "run",
        str(manifest),
        "--out-dir",
        str(first_dir),
        "--jobs",
        "1",
    ) == 6
    assert _run(
        tmp_path,
        model_dir,
        "batch",
        "run",
        str(manifest),
        "--out-dir",
        str(second_dir),
        "--jobs",
        "2",
    ) == 6
    first = {item["id"]: item["status"] for item in _status(first_dir)}
    second = {item["id"]: item["status"] for item in _status(second_dir)}
    assert first == second
    for index in range(4):
        name = f"job{index}.wav"
        assert (first_dir / name).is_file() == (second_dir / name).is_file()


def _write_lexicon(tmp_path: Path, name: str, spoken: str) -> Path:
    path = tmp_path / name
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "entries": [{"term": "AI", "spoken": spoken}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return path


def test_resume_invalidates_job_key_when_lexicon_changes(
    fake_engine, tmp_path: Path, model_dir: Path
):
    manifest = _manifest(tmp_path, _rows(1))
    out_dir = tmp_path / "out"
    first_lexicon = _write_lexicon(tmp_path, "a.json", "人工智能")
    second_lexicon = _write_lexicon(tmp_path, "b.json", "人工智慧")
    assert _run(
        tmp_path,
        model_dir,
        "batch",
        "run",
        str(manifest),
        "--out-dir",
        str(out_dir),
        "--lexicon",
        str(first_lexicon),
    ) == 0
    first_plan = json.loads((out_dir / "plan.json").read_text(encoding="utf-8"))
    assert first_plan["jobs"][0]["lexicon_sha256"]
    FakeEngine.generated = 0
    assert _run(
        tmp_path,
        model_dir,
        "batch",
        "run",
        str(manifest),
        "--out-dir",
        str(out_dir),
        "--resume",
        "--lexicon",
        str(second_lexicon),
    ) == 0
    second_plan = json.loads((out_dir / "plan.json").read_text(encoding="utf-8"))
    assert second_plan["jobs"][0]["job_key"] != first_plan["jobs"][0]["job_key"]
    assert FakeEngine.generated == 1


def test_unknown_suffix_fails_at_startup_even_with_explicit_format(
    fake_engine, tmp_path: Path, model_dir: Path
):
    rows = _rows(1)
    rows[0]["out"] = str(tmp_path / "x.mp3")
    rows[0]["format"] = "wav"
    out_dir = tmp_path / "out"
    assert _run(
        tmp_path,
        model_dir,
        "batch",
        "run",
        str(_manifest(tmp_path, rows)),
        "--out-dir",
        str(out_dir),
    ) == 4
    assert not (tmp_path / "x.mp3").exists()
    assert not (out_dir / "status.jsonl").exists()
    assert FakeEngine.generated == 0


def test_retry_skips_non_transient_and_resume_keeps_failure(
    fake_engine, tmp_path: Path, model_dir: Path
):
    FakeEngine.fail_exc = ValueError("boom")
    manifest = _manifest(tmp_path, _rows(1))
    out_dir = tmp_path / "out"
    assert _run(
        tmp_path,
        model_dir,
        "batch",
        "run",
        str(manifest),
        "--out-dir",
        str(out_dir),
        "--retry",
        "2",
    ) == 6
    status = _status(out_dir)[0]
    assert status["attempts"] == 1
    assert status["status"] == "failed"
    assert status["transient"] is False
    assert status["last_error"]["kind"] == "internal"
    assert FakeEngine.generated == 0

    assert _run(
        tmp_path,
        model_dir,
        "batch",
        "run",
        str(manifest),
        "--out-dir",
        str(out_dir),
        "--resume",
        "--retry",
        "2",
    ) == 6
    again = _status(out_dir)[0]
    assert again["attempts"] == 1
    assert again["status"] == "failed"
    assert FakeEngine.generated == 0


def test_manifest_duplicate_key_exits_2(tmp_path: Path, model_dir: Path):
    manifest = tmp_path / "duplicate.jsonl"
    manifest.write_text(
        '{"id":"a","id":"b","text":"你好。","voice":"builtin:Vivian"}\n',
        encoding="utf-8",
    )
    assert _run(
        tmp_path,
        model_dir,
        "batch",
        "plan",
        str(manifest),
        "--out-dir",
        str(tmp_path / "out"),
    ) == 2


def test_resume_reads_legacy_error_without_reviving_failure(
    fake_engine, tmp_path: Path, model_dir: Path
):
    manifest = _manifest(tmp_path, _rows(1))
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    legacy = {
        "id": "job0",
        "status": "failed",
        "job_key": "sha256:legacy",
        "output": str(out_dir / "job0.wav"),
        "attempts": 1,
        "output_sha256": None,
        "error": {"code": 3, "kind": "model", "message": "old permanent failure"},
    }
    (out_dir / "status.jsonl").write_text(
        json.dumps(legacy, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    assert _run(
        tmp_path,
        model_dir,
        "batch",
        "run",
        str(manifest),
        "--out-dir",
        str(out_dir),
        "--resume",
        "--retry",
        "2",
    ) == 6
    status = _status(out_dir)[0]
    assert status["status"] == "failed"
    assert status["attempts"] == 1
    assert status["transient"] is None
    assert status["last_error"]["message"] == "old permanent failure"
    assert "error" not in status
    assert FakeEngine.generated == 0


def test_watch_writes_only_stderr(
    fake_engine, capsys, tmp_path: Path, model_dir: Path
):
    manifest = _manifest(tmp_path, _rows(1))
    rc = _run(
        tmp_path,
        model_dir,
        "batch",
        "run",
        str(manifest),
        "--out-dir",
        str(tmp_path / "out"),
        "--watch",
        "--json",
    )
    assert rc == 0
    out, err = capsys.readouterr()
    payload = json.loads(out)
    assert payload["done"] == 1
    assert "[batch]" in err
