"""Shared helpers for the ``qtts serve`` test modules.

The file matches ``test_serve*.py`` by name but intentionally defines no test
functions: it only provides fakes that never touch torch or the network.
"""

from __future__ import annotations

import http.client
import json
import socket
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Optional

import qtts_cli.backends.registry as registry
from qtts_cli import serve
from qtts_cli.config import Config
from qtts_cli.errors import CancelledError

AUDIO_BYTES = b"RIFF\x00\x00\x00\x00WAVEfake-audio"


class FakeSpec:
    def __init__(self, *, remote: bool = False, configured: Optional[bool] = True):
        self.name = "fake"
        self.identity = "fake:test:model"
        self.models = [SimpleNamespace(model_id="fake-model")]
        self.remote = remote
        self.configured = (
            None if configured is None else (lambda: bool(configured))
        )


class RecordingRunner:
    """A thread-safe fake runner used as the sole worker-side implementation."""

    def __init__(
        self,
        *,
        work: Optional[
            Callable[[dict[str, Any], Path, Callable[[], bool]], dict[str, Any]]
        ] = None,
        fail: Optional[BaseException] = None,
        delay: float = 0.0,
    ) -> None:
        self.work = work
        self.fail = fail
        self.delay = delay
        self._lock = threading.Lock()
        self.active = 0
        self.peak = 0
        self.started = 0
        self.requests: list[dict[str, Any]] = []

    def submit(
        self,
        request: dict[str, Any],
        workdir: Path,
        cancel_check: Callable[[], bool],
    ) -> dict[str, Any]:
        with self._lock:
            self.active += 1
            self.started += 1
            self.peak = max(self.peak, self.active)
            self.requests.append(dict(request))
        try:
            if self.work is not None:
                return self.work(request, workdir, cancel_check)
            if self.delay:
                deadline = time.monotonic() + self.delay
                while time.monotonic() < deadline:
                    if cancel_check():
                        raise CancelledError("cancelled")
                    time.sleep(0.005)
            if self.fail is not None:
                raise self.fail
            return _success_payload(request, workdir)
        finally:
            with self._lock:
                self.active -= 1


def _success_payload(request: dict[str, Any], workdir: Path) -> dict[str, Any]:
    fmt = request.get("format", "wav")
    (workdir / f"audio.{fmt}").write_bytes(AUDIO_BYTES)
    subtitles = {}
    for sub in request.get("subtitle") or []:
        path = workdir / f"audio.{sub}"
        if sub == "vtt":
            path.write_text("WEBVTT\n", encoding="utf-8")
        elif sub == "json":
            path.write_text("[]\n", encoding="utf-8")
        else:
            path.write_text(
                "1\n00:00:00,000 --> 00:00:01,000\nhello\n", encoding="utf-8"
            )
        subtitles[sub] = str(path)
    return {
        "ok": True,
        "format": fmt,
        "sample_rate": 24000,
        "duration_s": 0.01,
        "sampling": {"seed": 1},
        "cache": {"hits": 0},
        "backend": {"name": "fake", "device": "cpu"},
        "device": "cpu",
        "dtype": "fp32",
        "timing": {"model_load_ms": 0.0, "generate_ms": 1.0, "total_ms": 1.0},
        "warnings": [],
        "align_coverage": None,
        "segments": [{"id": 1, "seed": 1}],
        "subtitles": subtitles,
    }


def serve_args(tmp_path: Path, **overrides: Any) -> SimpleNamespace:
    values = dict(
        api="both",
        host="127.0.0.1",
        port=0,
        allowed_host=[],
        max_concurrency=1,
        queue_size=8,
        max_connections=16,
        keep_jobs=100,
        max_body_bytes=65536,
        token=None,
        shutdown_timeout=2.0,
        dry_run=False,
        model=None,
        backend_model=None,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def make_config(tmp_path: Path) -> Config:
    return Config(
        data_dir=tmp_path / "data",
        cache_dir=tmp_path / "cache",
        model_dir=tmp_path / "models",
    )


class ServerHandle:
    def __init__(self, server, state, thread, stop_event):
        self.server = server
        self.state = state
        self.store = state.store
        self.port = server.server_address[1]
        self.host = "127.0.0.1"
        self.thread = thread
        self.stop_event = stop_event

    @property
    def base_path(self) -> str:
        return f"http://{self.host}:{self.port}"

    def client(self) -> "Client":
        return Client(self.host, self.port)

    def close(self) -> None:
        self.stop_event.set()
        self.thread.join(10)
        assert not self.thread.is_alive()


@contextmanager
def live_server(
    monkeypatch,
    tmp_path: Path,
    *,
    runner: Optional[RecordingRunner] = None,
    spec: Optional[FakeSpec] = None,
    **overrides: Any,
):
    spec = spec or FakeSpec()
    runner = runner or RecordingRunner()
    monkeypatch.setattr(registry, "resolve_backend", lambda *a, **k: spec)
    cfg = make_config(tmp_path)
    args = serve_args(tmp_path, **overrides)
    stop = threading.Event()
    ready: dict[str, Any] = {}
    errors: list[BaseException] = []

    def target() -> None:
        try:
            serve.run(
                cfg,
                args,
                runner_factory=lambda: runner,
                stop_event=stop,
                ready_callback=lambda server, state: ready.update(
                    server=server, state=state
                ),
            )
        except BaseException as exc:  # noqa: BLE001 - surfaced to the test
            errors.append(exc)

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while "state" not in ready and time.monotonic() < deadline:
        if errors:
            raise errors[0]
        if not thread.is_alive():
            raise AssertionError("serve thread exited before listening")
        time.sleep(0.01)
    if "state" not in ready:
        stop.set()
        raise AssertionError("serve did not become ready")
    handle = ServerHandle(ready["server"], ready["state"], thread, stop)
    try:
        yield handle
    finally:
        handle.close()
        if errors:
            raise errors[0]


class Client:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port

    def request(
        self,
        method: str,
        path: str,
        body: Any = None,
        *,
        raw_body: Optional[bytes] = None,
        headers: Optional[dict[str, str]] = None,
        params: Optional[dict[str, Any]] = None,
    ):
        conn = http.client.HTTPConnection(self.host, self.port, timeout=15)
        data = raw_body
        merged = dict(headers or {})
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            merged.setdefault("Content-Type", "application/json")
        if params:
            from urllib.parse import urlencode

            path = f"{path}?{urlencode(params, doseq=True)}"
        conn.request(method, path, body=data, headers=merged)
        response = conn.getresponse()
        payload = response.read()
        result = Response(
            response.status,
            {key.lower(): value for key, value in response.getheaders()},
            payload,
        )
        conn.close()
        return result


class Response:
    def __init__(self, status: int, headers: dict[str, str], body: bytes):
        self.status = status
        self.headers = headers
        self.body = body

    @property
    def json(self) -> Any:
        return json.loads(self.body.decode("utf-8"))

    @property
    def text(self) -> str:
        return self.body.decode("utf-8")


def wait_until(predicate: Callable[[], bool], timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition not met before timeout")


def wait_job(
    client: Client,
    job_id: str,
    timeout: float = 10.0,
    *,
    headers: Optional[dict[str, str]] = None,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.request("GET", f"/v1/jobs/{job_id}", headers=headers)
        assert response.status == 200
        job = response.json["job"]
        if job["status"] in ("done", "failed", "cancelled"):
            return job
        time.sleep(0.01)
    raise AssertionError(f"job {job_id} did not finish")


def raw_request(
    host: str, port: int, payload: bytes, *, timeout: float = 5.0
) -> bytes:
    with socket.create_connection((host, port), timeout=timeout) as sock:
        sock.sendall(payload)
        chunks = []
        while True:
            chunk = sock.recv(65536)
            if not chunk:
                break
            chunks.append(chunk)
    return b"".join(chunks)


def response_status(raw: bytes) -> int:
    first = raw.split(b"\r\n", 1)[0]
    return int(first.split()[1])
