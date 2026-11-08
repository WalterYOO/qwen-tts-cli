"""Process lifecycle: SIGINT/SIGTERM, forced shutdown, daemon workers (§1, §12)."""

from __future__ import annotations

import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from test_serve_helpers import Client, wait_until

REPO = Path(__file__).resolve().parents[1]

SCRIPT = r'''
import sys, time
from pathlib import Path
from types import SimpleNamespace

import qtts_cli.backends.registry as registry
from qtts_cli import serve
from qtts_cli.config import Config
from qtts_cli.errors import CancelledError


class Spec:
    name = "fake"
    identity = "fake"
    models = []
    remote = False
    configured = None


registry.resolve_backend = lambda *args, **kwargs: Spec()

data = Path(sys.argv[1])
mode = sys.argv[2]
timeout = float(sys.argv[3])

class Runner:
    def submit(self, request, workdir, cancel_check):
        if mode == "stubborn":
            time.sleep(60)
        while True:
            if cancel_check():
                raise CancelledError("cancelled")
            time.sleep(0.01)

args = SimpleNamespace(
    api="both", host="127.0.0.1", port=0, allowed_host=[],
    max_concurrency=1, queue_size=16, max_connections=16, keep_jobs=100,
    max_body_bytes=1048576, token=None, shutdown_timeout=timeout,
    dry_run=False, model=None,
)
serve.run(
    Config(data_dir=data, cache_dir=data / "cache", json=True),
    args,
    runner_factory=lambda: Runner(),
)
'''


def _env() -> dict[str, str]:
    env = dict(os.environ)
    env.pop("DASHSCOPE_API_KEY", None)
    env.pop("QTTS_SERVE_TOKEN", None)
    env["PYTHONPATH"] = str(REPO)
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _start(tmp_path: Path, mode: str, timeout: float):
    data = tmp_path / "data"
    proc = subprocess.Popen(
        [sys.executable, "-c", SCRIPT, str(data), mode, str(timeout)],
        cwd=str(REPO),
        env=_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    result: queue.Queue[str] = queue.Queue()

    def read() -> None:
        result.put(proc.stdout.readline())

    threading.Thread(target=read, daemon=True).start()
    line = result.get(timeout=15)
    payload = json.loads(line)
    return proc, data, payload["port"]


def _stop(proc: subprocess.Popen, *, timeout: float = 20.0):
    if proc.poll() is None:
        proc.send_signal(signal.SIGINT)
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)
        raise
    return out, err


JOB = {"text": "hi", "voice": "builtin:Vivian"}


def _job_dirs(data: Path) -> list[Path]:
    root = data / "serve" / "jobs"
    return [path for path in root.iterdir() if path.is_dir()]


def test_sigint_cancels_queued_jobs_and_exits_zero(tmp_path: Path):
    proc, data, port = _start(tmp_path, "polite", 5.0)
    try:
        client = Client("127.0.0.1", port)
        ids = []
        for _ in range(3):
            response = client.request("POST", "/v1/jobs", JOB)
            assert response.status == 202
            ids.append(response.json["job"]["id"])
        wait_until(
            lambda: client.request("GET", "/health").json["queue"]["queued"] >= 2
        )
        wait_until(
            lambda: client.request("GET", "/health").json["queue"]["running"] == 1
        )
        out, err = _stop(proc)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)
    assert proc.returncode == 0, err
    assert out == ""
    dirs = _job_dirs(data)
    assert len(dirs) == 3
    for directory in dirs:
        result = json.loads(
            (directory / "result.json").read_text(encoding="utf-8")
        )
        assert result["status"] == "cancelled"
        assert (directory / "request.json").is_file()
        assert (directory / "run.log").is_file()
        assert not (directory / "audio.wav").exists()
    assert list((data / "serve" / "staging").iterdir()) == []


def test_shutdown_timeout_exits_with_daemon_worker_and_leaves_orphan(tmp_path: Path):
    proc, data, port = _start(tmp_path, "stubborn", 0.5)
    try:
        client = Client("127.0.0.1", port)
        created = client.request("POST", "/v1/jobs", JOB)
        job_id = created.json["job"]["id"]
        wait_until(
            lambda: client.request("GET", f"/v1/jobs/{job_id}").json["job"]["status"]
            == "running"
        )
        started = time.monotonic()
        out, err = _stop(proc)
        assert time.monotonic() - started < 10
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)
    assert proc.returncode == 0, err
    directory = data / "serve" / "jobs" / job_id
    result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
    assert result["status"] == "cancelled"
    assert (directory / "request.json").is_file()
    assert (directory / "run.log").is_file()
    assert not (directory / "audio.wav").exists()
    # The stuck worker's staging is an orphan until the next startup.
    orphan = data / "serve" / "staging" / job_id
    assert orphan.is_dir()

    # A fresh process clears staging before listening; it has no live jobs.
    proc2, data2, port2 = _start(tmp_path, "polite", 1.0)
    try:
        assert data2 == data
        out2, err2 = _stop(proc2)
    finally:
        if proc2.poll() is None:
            proc2.kill()
            proc2.wait(timeout=5)
    assert proc2.returncode == 0
    assert list((data / "serve" / "staging").iterdir()) == []


def test_second_sigint_exits_130(tmp_path: Path):
    proc, data, port = _start(tmp_path, "stubborn", 30.0)
    try:
        client = Client("127.0.0.1", port)
        created = client.request("POST", "/v1/jobs", JOB)
        job_id = created.json["job"]["id"]
        wait_until(
            lambda: client.request("GET", f"/v1/jobs/{job_id}").json["job"]["status"]
            == "running"
        )
        proc.send_signal(signal.SIGINT)
        time.sleep(0.2)
        assert proc.poll() is None
        proc.send_signal(signal.SIGINT)
        try:
            out, err = proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
            raise
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)
    assert proc.returncode == 130
