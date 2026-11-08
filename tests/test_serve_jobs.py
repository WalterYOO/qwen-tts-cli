"""JobStore state machine, queueing, resources, publication and retention."""

from __future__ import annotations

import json
import os
import shutil
import threading
import time
from pathlib import Path

import pytest

from qtts_cli.errors import CancelledError, ModelError
from qtts_cli.serve import _startup_cleanup
from qtts_cli.serve.jobs import (
    Job,
    JobStore,
    QueueFull,
    StoreClosed,
    cleanup_orphan_jobs,
    cleanup_staging,
)
from qtts_cli.serve.sanitize import redact_text

from test_serve_helpers import AUDIO_BYTES, RecordingRunner, wait_until


def request(**overrides):
    body = {
        "text": "hello /foo",
        "voice": "builtin:Vivian",
        "language": "auto",
        "instruct": None,
        "format": "wav",
        "subtitle": [],
        "max_new_tokens": None,
        "pause_ms": None,
        "seed": None,
        "metadata": None,
    }
    body.update(overrides)
    return body


def make_store(
    tmp_path: Path,
    runner: RecordingRunner,
    *,
    local: bool = True,
    max_concurrency: int = 1,
    queue_size: int = 8,
    keep_jobs: int = 100,
    secrets=(),
    host_dirs=(),
) -> JobStore:
    return JobStore(
        tmp_path / "serve",
        max_concurrency=max_concurrency,
        queue_size=queue_size,
        keep_jobs=keep_jobs,
        local_runtime=local,
        runner_factory=lambda: runner,
        secrets=secrets,
        host_dirs=host_dirs,
    )


def snapshot(store: JobStore, job_id: str) -> dict:
    value = store.snapshot(job_id)
    assert value is not None
    return value


# --------------------------------------------------------------------------- #
# §5 queueing
# --------------------------------------------------------------------------- #
def test_queue_size_zero_hands_off_to_idle_worker_or_rejects(tmp_path: Path):
    release = threading.Event()

    def work(req, wd, cancel):
        release.wait(10)
        (wd / "audio.wav").write_bytes(AUDIO_BYTES)
        return {
            "ok": True,
            "format": "wav",
            "duration_s": 0.01,
            "cache": {},
            "timing": {},
            "segments": [],
            "subtitles": {},
        }

    runner = RecordingRunner(work=work)
    store = make_store(tmp_path, runner, queue_size=0)
    store.start()
    try:
        first = store.submit(request())
        with pytest.raises(QueueFull):
            store.submit(request())
        wait_until(lambda: snapshot(store, first.id)["status"] == "running")
        with pytest.raises(QueueFull):
            store.submit(request())
        release.set()
        finished = store.wait(first.id, timeout=10)
        assert finished is not None and finished.status == "done"
    finally:
        release.set()
        store.shutdown(1.0)


def test_cancelling_queued_job_releases_queue_capacity(tmp_path: Path):
    release = threading.Event()

    def work(req, wd, cancel):
        release.wait(10)
        (wd / "audio.wav").write_bytes(AUDIO_BYTES)
        return {"ok": True, "format": "wav", "subtitles": {}}

    runner = RecordingRunner(work=work)
    store = make_store(tmp_path, runner, queue_size=1)
    store.start()
    try:
        first = store.submit(request())
        wait_until(lambda: snapshot(store, first.id)["status"] == "running")
        second = store.submit(request())
        assert snapshot(store, second.id)["status"] == "queued"
        with pytest.raises(QueueFull):
            store.submit(request())

        outcome, cancelled = store.cancel(second.id)
        assert outcome == "cancelled" and cancelled is not None
        assert snapshot(store, second.id)["status"] == "cancelled"
        assert second.id not in store._pending

        third = store.submit(request())
        assert third.initial_queue_position == 0
        release.set()
        assert store.wait(first.id, timeout=10).status == "done"  # type: ignore[union-attr]
        assert store.wait(third.id, timeout=10).status == "done"  # type: ignore[union-attr]
    finally:
        release.set()
        store.shutdown(1.0)


def test_submit_after_shutdown_is_rejected(tmp_path: Path):
    store = make_store(tmp_path, RecordingRunner())
    with store._cond:
        store._stopping = True
    with pytest.raises(StoreClosed):
        store.submit(request())


def test_health_counts_cancel_requested_as_running(tmp_path: Path):
    started = threading.Event()
    release = threading.Event()

    def work(req, wd, cancel):
        started.set()
        release.wait(10)
        raise CancelledError("cancelled")

    store = make_store(tmp_path, RecordingRunner(work=work))
    store.start()
    try:
        job = store.submit(request())
        assert started.wait(5)
        outcome, _ = store.cancel(job.id)
        assert outcome == "cancel_requested"
        health = store.health()
        assert health["running"] == 1
        assert health["total"] == 1
        assert health["total"] == sum(
            health[name]
            for name in ("queued", "running", "done", "failed", "cancelled")
        )
    finally:
        release.set()
        store.shutdown(1.0)


def test_list_snapshots_jobs_under_the_store_lock(tmp_path: Path, monkeypatch):
    store = make_store(tmp_path, RecordingRunner())
    job = store.submit(request())
    entered = threading.Event()
    release = threading.Event()
    original = Job.public

    def slow_public(self, *args, **kwargs):
        entered.set()
        release.wait(5)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Job, "public", slow_public)
    listed: list[dict] = []
    got: list[Job] = []

    def list_jobs():
        listed.extend(store.list())

    def get_job():
        got.append(store.get(job.id))

    lister = threading.Thread(target=list_jobs, daemon=True)
    lister.start()
    assert entered.wait(5)
    getter = threading.Thread(target=get_job, daemon=True)
    getter.start()
    time.sleep(0.05)
    assert got == []  # blocked behind the public() snapshot inside list()
    release.set()
    lister.join(5)
    getter.join(5)
    assert len(listed) == 1 and listed[0]["id"] == job.id
    assert got and got[0] is job


def test_queue_size_positive_positions_are_unique_and_compact(tmp_path: Path):
    release = threading.Event()

    def work(req, wd, cancel):
        while not release.wait(0.01):
            if cancel():
                raise CancelledError("cancelled")
        (wd / "audio.wav").write_bytes(AUDIO_BYTES)
        return {
            "ok": True,
            "format": "wav",
            "duration_s": 0.01,
            "cache": {},
            "timing": {},
            "segments": [],
            "subtitles": {},
        }

    runner = RecordingRunner(work=work)
    store = make_store(tmp_path, runner, queue_size=1)
    store.start()
    try:
        first = store.submit(request())
        second = store.submit(request())
        with pytest.raises(QueueFull):
            store.submit(request())
        wait_until(lambda: snapshot(store, first.id)["status"] == "running")
        waiting = snapshot(store, second.id)
        assert waiting["status"] == "queued"
        assert waiting["queue_position"] == 0
        outcome, cancelled = store.cancel(second.id)
        assert outcome == "cancelled" and cancelled is not None
        assert snapshot(store, second.id)["status"] == "cancelled"
        release.set()
        assert store.wait(first.id, timeout=10).status == "done"  # type: ignore[union-attr]
        assert runner.started == 1
    finally:
        release.set()
        store.shutdown(1.0)

    final = tmp_path / "serve" / "jobs" / second.id
    assert (final / "request.json").is_file()
    assert (final / "result.json").is_file()
    assert (final / "run.log").is_file()
    assert not (final / "audio.wav").exists()


def test_waiting_lock_is_queued_in_api_and_one_aligner_at_a_time(tmp_path: Path):
    release = threading.Event()

    def work(req, wd, cancel):
        while not release.wait(0.01):
            if cancel():
                raise CancelledError("cancelled")
        (wd / "audio.wav").write_bytes(AUDIO_BYTES)
        return {
            "ok": True,
            "format": "wav",
            "duration_s": 0.01,
            "cache": {},
            "timing": {},
            "segments": [],
            "subtitles": {},
        }

    runner = RecordingRunner(work=work)
    store = make_store(
        tmp_path, runner, local=False, max_concurrency=3, queue_size=8
    )
    store.start()
    try:
        jobs = [store.submit(request(subtitle=["srt"])) for _ in range(3)]
        wait_until(lambda: snapshot(store, jobs[0].id)["status"] == "running")
        wait_until(
            lambda: snapshot(store, jobs[1].id)["status"] == "queued"
            and snapshot(store, jobs[2].id)["status"] == "queued"
        )
        health = store.health()
        assert health["running"] == 1
        assert health["queued"] == 2
        positions = {
            snapshot(store, jobs[1].id)["queue_position"],
            snapshot(store, jobs[2].id)["queue_position"],
        }
        assert positions == {0, 1}
        outcome, _ = store.cancel(jobs[1].id)
        assert outcome == "cancelled"
        release.set()
        assert store.wait(jobs[0].id, timeout=10).status == "done"  # type: ignore[union-attr]
        assert store.wait(jobs[2].id, timeout=10).status == "done"  # type: ignore[union-attr]
        # The cancelled waiting_lock job never entered the runner.
        assert runner.started == 2
        assert snapshot(store, jobs[1].id)["status"] == "cancelled"
    finally:
        release.set()
        store.shutdown(1.0)


# --------------------------------------------------------------------------- #
# §4 resources
# --------------------------------------------------------------------------- #
def test_local_backend_is_always_single_concurrency(tmp_path: Path):
    runner = RecordingRunner(delay=0.05)
    store = make_store(tmp_path, runner, local=True, max_concurrency=4)
    assert store.effective_max_concurrency == 1
    store.start()
    try:
        ids = [store.submit(request()).id for _ in range(3)]
        for job_id in ids:
            assert store.wait(job_id, timeout=10).status == "done"  # type: ignore[union-attr]
        assert runner.peak == 1
    finally:
        store.shutdown(1.0)


def test_remote_backend_runs_up_to_max_concurrency(tmp_path: Path):
    barrier = threading.Barrier(2, timeout=5)

    def work(req, wd, cancel):
        barrier.wait()
        (wd / "audio.wav").write_bytes(AUDIO_BYTES)
        return {
            "ok": True,
            "format": "wav",
            "duration_s": 0.01,
            "cache": {},
            "timing": {},
            "segments": [],
            "subtitles": {},
        }

    runner = RecordingRunner(work=work)
    store = make_store(tmp_path, runner, local=False, max_concurrency=2)
    store.start()
    try:
        ids = [store.submit(request()).id for _ in range(2)]
        for job_id in ids:
            assert store.wait(job_id, timeout=10).status == "done"  # type: ignore[union-attr]
        assert runner.peak == 2
    finally:
        store.shutdown(1.0)


def test_remote_subtitle_jobs_take_the_aligner_lock(tmp_path: Path):
    release = threading.Event()

    def work(req, wd, cancel):
        while not release.wait(0.01):
            if cancel():
                raise CancelledError("cancelled")
        (wd / "audio.wav").write_bytes(AUDIO_BYTES)
        return {
            "ok": True,
            "format": "wav",
            "duration_s": 0.01,
            "cache": {},
            "timing": {},
            "segments": [],
            "subtitles": {},
        }

    runner = RecordingRunner(work=work)
    store = make_store(tmp_path, runner, local=False, max_concurrency=2)
    store.start()
    try:
        first = store.submit(request(subtitle=["srt"]))
        second = store.submit(request(subtitle=["srt"]))
        wait_until(lambda: snapshot(store, first.id)["status"] == "running")
        wait_until(lambda: snapshot(store, second.id)["status"] == "queued")
        assert runner.started == 1
        release.set()
        for job in (first, second):
            assert store.wait(job.id, timeout=10).status == "done"  # type: ignore[union-attr]
        assert runner.started == 2
    finally:
        release.set()
        store.shutdown(1.0)


# --------------------------------------------------------------------------- #
# §3 state machine
# --------------------------------------------------------------------------- #
def test_running_cancel_is_cooperative_and_maps_to_cancelled(tmp_path: Path):
    runner = RecordingRunner(delay=5.0)
    store = make_store(tmp_path, runner)
    store.start()
    try:
        job = store.submit(request())
        wait_until(lambda: snapshot(store, job.id)["status"] == "running")
        outcome, current = store.cancel(job.id)
        assert outcome == "cancel_requested"
        assert current is not None and current.public()["status"] == "cancel_requested"
        finished = store.wait(job.id, timeout=10)
        assert finished is not None
        assert finished.status == "cancelled"
        assert finished.error is None
        outcome, _ = store.cancel(job.id)
        assert outcome == "conflict"
    finally:
        store.shutdown(1.0)


def test_unexpected_cancelled_error_is_never_failed(tmp_path: Path):
    def work(req, wd, cancel):
        raise CancelledError("stop")

    runner = RecordingRunner(work=work)
    store = make_store(tmp_path, runner)
    store.start()
    try:
        job = store.submit(request())
        finished = store.wait(job.id, timeout=10)
        assert finished is not None and finished.status == "cancelled"
        assert finished.error is None
        log = (tmp_path / "serve" / "jobs" / job.id / "run.log").read_text(
            encoding="utf-8"
        )
        assert "Traceback" not in log
    finally:
        store.shutdown(1.0)


def test_failed_job_has_single_sanitized_error_object(tmp_path: Path):
    error = ModelError(
        "failed: /tmp/outside/model (token SECRET)",
        details={
            "path": "/tmp/outside/model",
            "model_dir": "/Users/someone/models",
            "reason": "/usr/local/bin/python",
            "token": "SECRET",
        },
    )
    runner = RecordingRunner(fail=error)
    store = make_store(tmp_path, runner, secrets=("SECRET",))
    store.start()
    try:
        job = store.submit(request())
        finished = store.wait(job.id, timeout=10)
        assert finished is not None and finished.status == "failed"
        assert finished.error is not None
        assert finished.error["ok"] is False
        assert finished.error["code"] == 3
        assert "SECRET" not in json.dumps(finished.error)
        assert "/tmp/outside" not in json.dumps(finished.error)
        assert finished.error["details"]["token"] == "<redacted>"
        assert "path" not in finished.error["details"]
        assert "model_dir" not in finished.error["details"]
        result = json.loads(
            (tmp_path / "serve" / "jobs" / job.id / "result.json").read_text(
                encoding="utf-8"
            )
        )
        assert result["status"] == "failed"
        assert result["error"]["code"] == 3
        log = (tmp_path / "serve" / "jobs" / job.id / "run.log").read_text(
            encoding="utf-8"
        )
        assert "SECRET" not in log
        assert "/tmp/outside" not in log
        assert not (tmp_path / "serve" / "jobs" / job.id / "audio.wav").exists()
    finally:
        store.shutdown(1.0)


@pytest.mark.parametrize("kind", ["failed", "cancelled"])
def test_failed_and_cancelled_remove_generated_artifacts(tmp_path: Path, kind: str):
    def work(req, wd, cancel):
        (wd / "audio.wav").write_bytes(AUDIO_BYTES)
        (wd / "audio.srt").write_text("subtitle", encoding="utf-8")
        (wd / "audio.recipe.json").write_text("{}", encoding="utf-8")
        if kind == "failed":
            raise ModelError("engine exploded")
        raise CancelledError("cancelled")

    runner = RecordingRunner(work=work)
    store = make_store(tmp_path, runner)
    store.start()
    try:
        job = store.submit(request(subtitle=["srt"]))
        finished = store.wait(job.id, timeout=10)
        assert finished is not None and finished.status == kind
        final = tmp_path / "serve" / "jobs" / job.id
        for name in ("audio.wav", "audio.srt", "audio.recipe.json"):
            assert not (final / name).exists()
        assert (final / "request.json").is_file()
        assert (final / "result.json").is_file()
        assert (final / "run.log").is_file()
        assert list((tmp_path / "serve" / "staging").iterdir()) == []
    finally:
        store.shutdown(1.0)


def test_done_job_publishes_audio_subtitles_and_relative_paths(tmp_path: Path):
    runner = RecordingRunner()
    store = make_store(tmp_path, runner)
    store.start()
    try:
        job = store.submit(request(subtitle=["srt"]))
        finished = store.wait(job.id, timeout=10)
        assert finished is not None and finished.status == "done"
        public = finished.public()
        assert public["output"]["file"] == "audio.wav"
        assert public["output"]["bytes"] == len(AUDIO_BYTES)
        assert public["subtitles"]["srt"]["file"] == "audio.srt"
        assert public["request"]["text"] == "hello /foo"
        final = tmp_path / "serve" / "jobs" / job.id
        for name in ("request.json", "result.json", "audio.wav", "audio.srt"):
            assert (final / name).is_file()
        assert not (final / "run.log").exists()
        text = (final / "result.json").read_text(encoding="utf-8")
        assert str(tmp_path) not in text
        assert json.loads(text)["status"] == "done"
        assert list((tmp_path / "serve" / "staging").iterdir()) == []
    finally:
        store.shutdown(1.0)


def test_health_counts_map_waiting_lock_to_queued(tmp_path: Path):
    release = threading.Event()

    def work(req, wd, cancel):
        release.wait(10)
        (wd / "audio.wav").write_bytes(AUDIO_BYTES)
        return {
            "ok": True,
            "format": "wav",
            "duration_s": 0.01,
            "cache": {},
            "timing": {},
            "segments": [],
            "subtitles": {},
        }

    runner = RecordingRunner(work=work)
    store = make_store(tmp_path, runner, local=False, max_concurrency=2)
    store.start()
    try:
        first = store.submit(request(subtitle=["srt"]))
        second = store.submit(request(subtitle=["srt"]))
        wait_until(lambda: snapshot(store, first.id)["status"] == "running")
        wait_until(lambda: snapshot(store, second.id)["status"] == "queued")
        health = store.health()
        assert health["queued"] == 1
        assert health["running"] == 1
        release.set()
        store.wait(first.id, timeout=10)
        store.wait(second.id, timeout=10)
        health = store.health()
        assert health["done"] == 2
        assert health["queued"] == health["running"] == 0
    finally:
        release.set()
        store.shutdown(1.0)


# --------------------------------------------------------------------------- #
# §6 shutdown/forced publication
# --------------------------------------------------------------------------- #
def test_timeout_force_publishes_single_cancelled_version(tmp_path: Path):
    release = threading.Event()

    def stubborn(req, wd, cancel):
        # Deliberately ignores cancel_check, like a backend stuck in generate().
        release.wait(10)
        (wd / "audio.wav").write_bytes(AUDIO_BYTES)
        return {
            "ok": True,
            "format": "wav",
            "duration_s": 0.01,
            "cache": {},
            "timing": {},
            "segments": [],
            "subtitles": {},
        }

    runner = RecordingRunner(work=stubborn)
    store = make_store(tmp_path, runner)
    store.start()
    job = store.submit(request(subtitle=["srt"]))
    wait_until(lambda: snapshot(store, job.id)["status"] == "running")
    started = time.monotonic()
    store.shutdown(0.1)
    assert time.monotonic() - started < 5
    final = tmp_path / "serve" / "jobs" / job.id
    result = json.loads((final / "result.json").read_text(encoding="utf-8"))
    assert result["status"] == "cancelled"
    assert (final / "run.log").is_file()
    assert not (final / "audio.wav").exists()
    assert not (final / "audio.srt").exists()
    # The stuck worker still owns its staging directory until it returns.
    staging = tmp_path / "serve" / "staging" / job.id
    assert staging.exists()
    release.set()
    wait_until(lambda: not staging.exists())
    result = json.loads((final / "result.json").read_text(encoding="utf-8"))
    assert result["status"] == "cancelled"
    assert len([p for p in final.parent.iterdir() if p.name == job.id]) == 1


def test_shutdown_cancels_queued_and_polite_running_jobs(tmp_path: Path):
    runner = RecordingRunner(delay=5.0)
    store = make_store(tmp_path, runner, queue_size=8)
    store.start()
    first = store.submit(request())
    second = store.submit(request())
    wait_until(lambda: snapshot(store, first.id)["status"] == "running")
    store.shutdown(2.0)
    assert snapshot(store, first.id)["status"] == "cancelled"
    assert snapshot(store, second.id)["status"] == "cancelled"
    for job_id in (first.id, second.id):
        final = tmp_path / "serve" / "jobs" / job_id
        assert (final / "request.json").is_file()
        assert (final / "result.json").is_file()
        assert (final / "run.log").is_file()
        assert not (final / "audio.wav").exists()
    assert list((tmp_path / "serve" / "staging").iterdir()) == []


# --------------------------------------------------------------------------- #
# §6 retention + startup cleanup
# --------------------------------------------------------------------------- #
def test_keep_jobs_counts_all_terminal_states(tmp_path: Path):
    runner = RecordingRunner()
    store = make_store(tmp_path, runner, keep_jobs=1)
    store.start()
    ids = []
    try:
        for _ in range(3):
            job = store.submit(request())
            ids.append(job.id)
            assert store.wait(job.id, timeout=10).status == "done"  # type: ignore[union-attr]
        jobs_root = tmp_path / "serve" / "jobs"
        dirs = [p.name for p in jobs_root.iterdir() if p.is_dir()]
        assert ids[2] in dirs
        assert ids[0] not in dirs and ids[1] not in dirs
        assert [item["id"] for item in store.list()] == [ids[2]]
        assert store.get(ids[0]) is None and store.get(ids[1]) is None
        assert store.snapshot(ids[0]) is None
        assert store.health()["total"] == 1
    finally:
        store.shutdown(1.0)

    keep_all = make_store(tmp_path / "keepall", runner, keep_jobs=0)
    keep_all.start()
    try:
        kept = []
        for _ in range(2):
            job = keep_all.submit(request())
            kept.append(job.id)
            keep_all.wait(job.id, timeout=10)
        root = tmp_path / "keepall" / "serve" / "jobs"
        assert sorted(p.name for p in root.iterdir()) == sorted(kept)
    finally:
        keep_all.shutdown(1.0)


def test_failed_atomic_publish_writes_minimal_terminal_artifacts(
    tmp_path: Path, monkeypatch
):
    import qtts_cli.serve.jobs as jobs_module

    class BrokenOS:
        @staticmethod
        def replace(source, target):
            raise OSError("replace failed")

    store = make_store(tmp_path, RecordingRunner())
    store.start()
    try:
        monkeypatch.setattr(jobs_module, "os", BrokenOS())
        job = store.submit(request())
        finished = store.wait(job.id, timeout=10)
        assert finished is not None and finished.status == "failed"
        final_dir = tmp_path / "serve" / "jobs" / job.id
        request_json = json.loads(
            (final_dir / "request.json").read_text(encoding="utf-8")
        )
        result_json = json.loads(
            (final_dir / "result.json").read_text(encoding="utf-8")
        )
        assert request_json["voice"] == "builtin:Vivian"
        assert result_json["status"] == "failed"
        assert result_json["error"]["error"] == "internal"
    finally:
        monkeypatch.undo()
        store.shutdown(1.0)


def test_wait_returns_last_seen_terminal_job_after_retention(
    tmp_path: Path, monkeypatch
):
    store = make_store(tmp_path, RecordingRunner(), keep_jobs=1)
    job = store.submit(request())
    cond = store._cond
    original_wait = cond.wait
    waiting = threading.Event()

    def marked_wait(timeout=None):
        waiting.set()
        return original_wait(timeout)

    monkeypatch.setattr(cond, "wait", marked_wait)
    result: list[Job] = []
    waiter = threading.Thread(
        target=lambda: result.append(store.wait(job.id, timeout=5)),
        daemon=True,
    )
    waiter.start()
    assert waiting.wait(5)
    with cond:
        job.status = "done"
        store._jobs.pop(job.id, None)
        cond.notify_all()
    waiter.join(5)
    assert result == [job]


def test_pinned_job_survives_retention_until_unpinned(tmp_path: Path):
    store = make_store(tmp_path, RecordingRunner(), keep_jobs=1)
    store.start()
    try:
        first = store.submit(request(), pin=True)
        assert store.wait(first.id, timeout=10).status == "done"  # type: ignore[union-attr]
        second = store.submit(request(), pin=True)
        assert store.wait(second.id, timeout=10).status == "done"  # type: ignore[union-attr]

        # Two pinned terminal jobs exceed keep_jobs but neither may be removed.
        assert store.get(first.id) is not None
        assert store.get(second.id) is not None
        assert store.pin("job_" + "0" * 32) is False
        assert (tmp_path / "serve" / "jobs" / first.id).is_dir()
        assert (tmp_path / "serve" / "jobs" / second.id).is_dir()

        store.unpin(first.id)
        assert store.get(first.id) is None
        assert not (tmp_path / "serve" / "jobs" / first.id).exists()
        assert store.get(second.id) is not None

        # unpin is idempotent and never drives the counter below zero.
        store.unpin(first.id)
        assert first.pins == 0
        assert store.pin(second.id) is True
        store.unpin(second.id)
        store.unpin(second.id)
        store.unpin(second.id)
        assert second.pins == 0
        assert store.get(second.id) is not None
    finally:
        store.shutdown(1.0)


def test_runtime_retention_includes_preexisting_job_directories(tmp_path: Path):
    store = make_store(tmp_path, RecordingRunner(), keep_jobs=1)
    jobs_root = tmp_path / "serve" / "jobs"
    old = jobs_root / "job_old"
    older = jobs_root / "job_older"
    old.mkdir()
    older.mkdir()
    os.utime(older, (1, 1))
    os.utime(old, (2, 2))
    cleanup_orphan_jobs(tmp_path / "serve", 1)
    assert [p.name for p in jobs_root.iterdir()] == [old.name]

    store.start()
    try:
        job = store.submit(request())
        assert store.wait(job.id, timeout=10).status == "done"  # type: ignore[union-attr]
        names = sorted(p.name for p in jobs_root.iterdir() if p.is_dir())
        assert names == [job.id]
    finally:
        store.shutdown(1.0)


def test_keep_jobs_bounds_in_memory_metadata(tmp_path: Path):
    store = make_store(tmp_path, RecordingRunner(), keep_jobs=1)
    store.start()
    try:
        for _ in range(20):
            job = store.submit(request())
            assert store.wait(job.id, timeout=10).status == "done"  # type: ignore[union-attr]
            assert len(store._jobs) <= 1
        assert len(store.list()) == 1
    finally:
        store.shutdown(1.0)


def test_startup_cleanup_only_removes_the_staging_snapshot(
    tmp_path: Path, monkeypatch
):
    root = tmp_path / "serve"
    staging = root / "staging"
    old = staging / "job_old"
    old.mkdir(parents=True)
    (old / "audio.wav").write_bytes(b"old")

    entered = threading.Event()
    release = threading.Event()
    seen: dict[str, list[Path]] = {}

    def blocking_cleanup(root_arg, entries=None):
        seen["entries"] = list(entries)
        entered.set()
        release.wait(5)
        for entry in entries:
            if entry.is_dir():
                shutil.rmtree(entry, ignore_errors=True)
            else:
                entry.unlink(missing_ok=True)

    monkeypatch.setattr("qtts_cli.serve.cleanup_staging", blocking_cleanup)
    _startup_cleanup(root, 0)
    assert entered.wait(5)
    new = staging / "job_new"
    new.mkdir()
    (new / "audio.wav").write_bytes(b"new")
    release.set()
    wait_until(lambda: not old.exists())
    assert new.exists()
    assert new not in seen["entries"]
    assert old in seen["entries"]


def test_startup_cleanup_removes_staging_and_orphans(tmp_path: Path):
    root = tmp_path / "serve"
    staging = root / "staging" / "job_orphan"
    staging.mkdir(parents=True)
    (staging / "audio.wav").write_bytes(b"x")
    jobs = root / "jobs"
    import os

    for index, name in enumerate(("old", "middle", "new")):
        directory = jobs / name
        directory.mkdir(parents=True)
        (directory / "request.json").write_text("{}", encoding="utf-8")
        os.utime(directory, (1000 + index, 1000 + index))
    cleanup_staging(root)
    assert list((root / "staging").iterdir()) == []
    cleanup_orphan_jobs(root, 1)
    remaining = sorted(p.name for p in jobs.iterdir())
    assert remaining == ["new"]
    cleanup_orphan_jobs(root, 0)
    assert sorted(p.name for p in jobs.iterdir()) == ["new"]


def test_redact_text_replaces_paths_and_secrets():
    text = redact_text(
        "a /Users/me/secret b C:\\Users\\x c \\\\server\\share token123",
        secrets=("token123",),
    )
    assert "/Users/" not in text
    assert "C:\\Users" not in text
    assert "\\\\server" not in text
    assert "token123" not in text
