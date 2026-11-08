"""Native/OpenAI HTTP surface, framing, auth and connection limits."""

from __future__ import annotations

import json
import socket
import threading
from pathlib import Path

import pytest

from qtts_cli.errors import CancelledError, InputError, ModelError, QCError
from qtts_cli.serve.jobs import StoreClosed

from test_serve_helpers import (
    AUDIO_BYTES,
    FakeSpec,
    RecordingRunner,
    live_server,
    raw_request,
    response_status,
    wait_job,
    wait_until,
)

JOB = {"text": "hello /foo", "voice": "builtin:Vivian"}


def _raw(method: str, path: str, headers: list[tuple[str, str]], body: bytes = b""):
    lines = [f"{method} {path} HTTP/1.1"]
    lines.extend(f"{name}: {value}" for name, value in headers)
    return ("\r\n".join(lines) + "\r\n\r\n").encode("utf-8") + body


def test_create_poll_audio_subtitles_and_list(monkeypatch, tmp_path: Path):
    with live_server(monkeypatch, tmp_path) as server:
        client = server.client()
        response = client.request(
            "POST",
            "/v1/jobs",
            {**JOB, "subtitle": ["srt", "vtt"]},
        )
        assert response.status == 202
        created = response.json
        assert created["ok"] is True
        assert created["job"]["status"] == "queued"
        assert created["job"]["queue_position"] == 0
        assert created["job"]["created_at"].endswith("Z")
        job_id = created["job"]["id"]

        job = wait_job(client, job_id)
        assert job["status"] == "done"
        assert job["request"]["text"] == "hello /foo"  # caller text is exempt
        assert job["output"]["url"] == f"/v1/jobs/{job_id}/audio"
        assert job["output"]["file"] == "audio.wav"
        assert job["output"]["bytes"] == len(AUDIO_BYTES)
        assert job["subtitles"]["srt"]["file"] == "audio.srt"
        assert job["error"] is None
        assert job["timings"]["queue_ms"] is not None
        assert job["timings"]["run_ms"] is not None
        assert not (tmp_path / "data" / "serve" / "jobs" / job_id / "result.json").read_text(
            encoding="utf-8"
        ).count(str(tmp_path))

        audio = client.request("GET", f"/v1/jobs/{job_id}/audio")
        assert audio.status == 200
        assert audio.body == AUDIO_BYTES
        assert audio.headers["content-type"] == "audio/wav"
        assert audio.headers["cache-control"] == "no-store"
        assert audio.headers["content-length"] == str(len(AUDIO_BYTES))

        subtitle = client.request("GET", f"/v1/jobs/{job_id}/subtitles/srt")
        assert subtitle.status == 200
        assert subtitle.headers["content-type"] == "text/plain; charset=utf-8"
        assert "-->".encode() in subtitle.body

        listing = client.request("GET", "/v1/jobs", params={"status": "done"})
        assert listing.status == 200
        assert [item["id"] for item in listing.json["jobs"]] == [job_id]
        repeated = client.request(
            "GET", "/v1/jobs", params={"status": ["done", "failed"]}
        )
        assert repeated.status == 200
        assert [item["id"] for item in repeated.json["jobs"]] == [job_id]
        assert client.request("GET", "/v1/jobs", params={"limit": 0}).status == 400
        assert client.request("GET", "/v1/jobs", params={"limit": "x"}).status == 400
        assert client.request("GET", "/v1/jobs", params={"bad": "1"}).status == 400
        assert client.request("GET", "/v1/jobs", params={"status": "nope"}).status == 400

        conflict = client.request("DELETE", f"/v1/jobs/{job_id}")
        assert conflict.status == 409
        assert conflict.json["error"] == "conflict"
        assert conflict.json["details"]["status"] == "done"
        assert "code" not in conflict.json

        assert client.request("GET", "/v1/jobs/job_" + "0" * 32).status == 404
        assert client.request("GET", "/v1/jobs/not-an-id").status == 404


def test_native_request_validation(monkeypatch, tmp_path: Path):
    with live_server(monkeypatch, tmp_path, max_body_bytes=300000) as server:
        client = server.client()

        def post(**extra):
            return client.request("POST", "/v1/jobs", {**JOB, **extra})

        assert post(nope=1).status == 400
        assert post(text="").status == 400
        assert post(text="x" * 200001).status == 400
        assert post(voice="").status == 400
        assert post(language="x" * 65).status == 400
        assert post(instruct="x" * 20001).status == 400
        assert post(instruct=5).status == 400
        assert post(format="mp3").status == 400
        assert post(subtitle="srt").status == 400
        assert post(subtitle=["srt", "bad"]).status == 400
        assert post(subtitle=["srt", "vtt", "ass", "json"]).status == 202
        assert post(subtitle=["srt", "srt"]).status == 202
        assert post(max_new_tokens=True).status == 400
        assert post(max_new_tokens=0).status == 400
        assert post(max_new_tokens=65537).status == 400
        assert post(pause_ms=-1).status == 400
        assert post(seed=-1).status == 400
        assert post(seed=2**63).status == 400
        assert post(metadata=[]).status == 400
        assert post(metadata={"x": "y" * 5000}).status == 400
        assert post(metadata={f"k{i}": i for i in range(65)}).status == 400

        assert client.request("PUT", "/v1/jobs").status == 405
        assert client.request("PUT", "/v1/jobs").headers["allow"] == "GET, POST"
        assert client.request("POST", "/health").status == 405
        assert client.request("GET", "/nope").status == 404


def test_native_non_string_format_and_invalid_voice_create_no_jobs(
    monkeypatch, tmp_path: Path
):
    with live_server(monkeypatch, tmp_path) as server:
        client = server.client()

        def jobs() -> list[dict]:
            response = client.request("GET", "/v1/jobs")
            assert response.status == 200
            return response.json["jobs"]

        def post(**extra):
            return client.request("POST", "/v1/jobs", {**JOB, **extra})

        assert jobs() == []

        for fmt in ([], {}):
            response = post(format=fmt)
            assert response.status == 400
            assert response.json["error"] == "invalid_request"
            assert response.json["details"]["supported"] == ["flac", "wav"]

        for voice in ("builtin:", "nonsense", "foo:bar", "voice:", "voice:../x",
                      "voice:a/b", "voice:a\\b", "voice:."):
            response = post(voice=voice)
            assert response.status == 400, voice
            assert response.json["error"] == "invalid_request", voice

        assert jobs() == []

        for extra in (
            {"format": "wav"},
            {"format": "flac"},
            {"voice": "builtin:Eldric Sage"},
            {"voice": "design:温柔的女声"},
            {"voice": "voice:alice"},
        ):
            assert post(**extra).status == 202
        assert len(jobs()) == 5


def test_parse_voice_ref_keeps_legal_references_and_rejects_bad_names():
    from qtts_cli.capabilities import parse_voice_ref

    legal = {
        "builtin:Eldric Sage": ("builtin", "Eldric Sage"),
        "design:温柔的女声": ("design", "温柔的女声"),
        "voice:alice": ("voice", "alice"),
    }
    for raw, expected in legal.items():
        ref = parse_voice_ref(raw)
        assert (ref.kind, ref.value) == expected
    for raw in ("builtin:", "nonsense", "foo:bar", "voice:", "voice:../x",
                "voice:a/b", "voice:a\\b", "voice:."):
        with pytest.raises(InputError):
            parse_voice_ref(raw)


def test_framing_json_and_media_type(monkeypatch, tmp_path: Path):
    with live_server(monkeypatch, tmp_path) as server:
        port = server.port
        body = json.dumps(JOB).encode("utf-8")

        good_headers = [
            ("Host", f"127.0.0.1:{port}"),
            ("Content-Type", "application/json; charset=utf-8"),
            ("Content-Length", str(len(body))),
        ]
        raw = raw_request("127.0.0.1", port, _raw("POST", "/v1/jobs", good_headers, body))
        assert response_status(raw) == 202

        # Missing / duplicate / non-numeric Content-Length.
        raw = raw_request(
            "127.0.0.1",
            port,
            _raw(
                "POST",
                "/v1/jobs",
                [("Host", f"127.0.0.1:{port}"), ("Content-Type", "application/json")],
            ),
        )
        assert response_status(raw) == 400
        raw = raw_request(
            "127.0.0.1",
            port,
            _raw(
                "POST",
                "/v1/jobs",
                [
                    ("Host", f"127.0.0.1:{port}"),
                    ("Content-Type", "application/json"),
                    ("Content-Length", "2"),
                    ("Content-Length", "2"),
                ],
                b"{}",
            ),
        )
        assert response_status(raw) == 400
        raw = raw_request(
            "127.0.0.1",
            port,
            _raw(
                "POST",
                "/v1/jobs",
                [
                    ("Host", f"127.0.0.1:{port}"),
                    ("Content-Type", "application/json"),
                    ("Content-Length", "abc"),
                ],
            ),
        )
        assert response_status(raw) == 400

        # Transfer-Encoding and oversized bodies.
        raw = raw_request(
            "127.0.0.1",
            port,
            _raw(
                "POST",
                "/v1/jobs",
                [
                    ("Host", f"127.0.0.1:{port}"),
                    ("Content-Type", "application/json"),
                    ("Transfer-Encoding", "chunked"),
                ],
            ),
        )
        assert response_status(raw) == 400
        raw = raw_request(
            "127.0.0.1",
            port,
            _raw(
                "POST",
                "/v1/jobs",
                [
                    ("Host", f"127.0.0.1:{port}"),
                    ("Content-Type", "application/json"),
                    ("Content-Length", "65537"),
                ],
            ),
        )
        assert response_status(raw) == 413

        # Wrong media type -> 415 before parsing.
        raw = raw_request(
            "127.0.0.1",
            port,
            _raw(
                "POST",
                "/v1/jobs",
                [
                    ("Host", f"127.0.0.1:{port}"),
                    ("Content-Type", "text/plain"),
                    ("Content-Length", str(len(body))),
                ],
                body,
            ),
        )
        assert response_status(raw) == 415

        bad_json = [
            b'{"text":"a","text":"b","voice":"v"}',
            b'{"text":"a","voice":"v","seed":NaN}',
            b'["not","an","object"]',
            b"{",
        ]
        for payload in bad_json:
            raw = raw_request(
                "127.0.0.1",
                port,
                _raw(
                    "POST",
                    "/v1/jobs",
                    [
                        ("Host", f"127.0.0.1:{port}"),
                        ("Content-Type", "application/json"),
                        ("Content-Length", str(len(payload))),
                    ],
                    payload,
                ),
            )
            assert response_status(raw) == 400

        # Short read: declared 50 bytes, sent 3, then half-close.
        with socket.create_connection(("127.0.0.1", port), timeout=5) as sock:
            head = _raw(
                "POST",
                "/v1/jobs",
                [
                    ("Host", f"127.0.0.1:{port}"),
                    ("Content-Type", "application/json"),
                    ("Content-Length", "50"),
                ],
                b"abc",
            )
            sock.sendall(head)
            sock.shutdown(socket.SHUT_WR)
            data = b""
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                data += chunk
        assert response_status(data) == 400

        # Host checks.
        raw = raw_request(
            "127.0.0.1",
            port,
            _raw("GET", "/health", [("Content-Length", "0")]),
        )
        assert response_status(raw) == 400
        raw = raw_request(
            "127.0.0.1",
            port,
            _raw(
                "GET",
                "/health",
                [("Host", f"127.0.0.1:{port}"), ("Host", f"127.0.0.1:{port}")],
            ),
        )
        assert response_status(raw) == 400
        raw = raw_request(
            "127.0.0.1",
            port,
            _raw("GET", "/health", [("Host", "evil.example:1234")]),
        )
        assert response_status(raw) == 403


def test_auth_matrix_and_allowed_hosts(monkeypatch, tmp_path: Path):
    with live_server(monkeypatch, tmp_path, token="serve-test-token") as server:
        client = server.client()
        health = client.request("GET", "/health")
        assert health.status == 200  # loopback health is public
        unauthorized = client.request("GET", "/v1/jobs")
        assert unauthorized.status == 401
        assert unauthorized.headers["www-authenticate"] == "Bearer"
        assert client.request(
            "GET", "/v1/jobs", headers={"Authorization": "Bearer wrong"}
        ).status == 401
        assert client.request(
            "GET", "/v1/jobs", headers={"Authorization": "Bearer serve-test-token"}
        ).status == 200

    # Wildcard bind: health requires the token; only explicit Hosts pass.
    with live_server(
        monkeypatch,
        tmp_path / "wildcard",
        token="serve-test-token",
        host="0.0.0.0",
        allowed_host=["allowed.test"],
    ) as server:
        client = server.client()
        assert client.request("GET", "/health").status == 401
        assert client.request(
            "GET", "/health", headers={"Authorization": "Bearer serve-test-token"}
        ).status == 200
        assert client.request(
            "GET",
            "/v1/jobs",
            headers={
                "Authorization": "Bearer serve-test-token",
                "Host": f"allowed.test:{server.port}",
            },
        ).status == 200
        assert client.request(
            "GET",
            "/v1/jobs",
            headers={
                "Authorization": "Bearer serve-test-token",
                "Host": f"evil.test:{server.port}",
            },
        ).status == 403


def test_queue_full_returns_429_with_retry_after(monkeypatch, tmp_path: Path):
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
    with live_server(monkeypatch, tmp_path, runner=runner, queue_size=0) as server:
        client = server.client()
        first = client.request("POST", "/v1/jobs", JOB)
        assert first.status == 202
        wait_until(lambda: server.store.health()["running"] == 1)
        full = client.request("POST", "/v1/jobs", JOB)
        assert full.status == 429
        assert full.headers["retry-after"] == "1"
        assert full.json["error"] == "queue_full"
        assert "code" not in full.json
        release.set()
        wait_job(client, first.json["job"]["id"])


def test_running_job_cancel_and_audio_conflict(monkeypatch, tmp_path: Path):
    runner = RecordingRunner(delay=5.0)
    with live_server(monkeypatch, tmp_path, runner=runner) as server:
        client = server.client()
        created = client.request("POST", "/v1/jobs", JOB)
        job_id = created.json["job"]["id"]
        wait_until(
            lambda: client.request("GET", f"/v1/jobs/{job_id}").json["job"]["status"]
            == "running"
        )
        conflict = client.request("GET", f"/v1/jobs/{job_id}/audio")
        assert conflict.status == 409
        cancelled = client.request("DELETE", f"/v1/jobs/{job_id}")
        assert cancelled.status == 202
        assert cancelled.json["job"]["status"] == "cancel_requested"
        final = wait_job(client, job_id)
        assert final["status"] == "cancelled"
        assert client.request("DELETE", f"/v1/jobs/{job_id}").status == 409
        assert client.request("GET", f"/v1/jobs/{job_id}/audio").status == 404


def test_openai_speech_models_and_message_mapping(monkeypatch, tmp_path: Path):
    runner = RecordingRunner()
    spec = FakeSpec()
    spec.models = [
        type("M", (), {"model_id": "first-model"})(),
        type("M", (), {"model_id": "second-model"})(),
    ]
    with live_server(monkeypatch, tmp_path, runner=runner, spec=spec) as server:
        client = server.client()
        speech = client.request(
            "POST",
            "/v1/audio/speech",
            {"model": "any-model", "input": "hello", "voice": "alloy"},
        )
        assert speech.status == 200
        assert speech.body == AUDIO_BYTES
        assert speech.headers["content-type"] == "audio/wav"
        assert speech.headers["content-disposition"] == 'attachment; filename="speech.wav"'
        assert runner.requests[-1]["voice"] == "builtin:Vivian"
        assert runner.requests[-1]["text"] == "hello"

        assert client.request(
            "POST",
            "/v1/audio/speech",
            {"model": "x", "input": "hi", "voice": "nova", "response_format": "flac"},
        ).status == 200
        assert runner.requests[-1]["voice"] == "builtin:Serena"
        assert runner.requests[-1]["format"] == "flac"

        assert client.request(
            "POST",
            "/v1/audio/speech",
            {"model": "x", "input": "hi", "voice": "voice:MyVoice"},
        ).status == 200
        assert runner.requests[-1]["voice"] == "voice:MyVoice"
        assert client.request(
            "POST",
            "/v1/audio/speech",
            {"model": "x", "input": "hi", "voice": "Vivian"},
        ).status == 200
        assert runner.requests[-1]["voice"] == "builtin:Vivian"
        assert client.request(
            "POST",
            "/v1/audio/speech",
            {"input": "hi", "voice": "alloy", "instructions": "calm"},
        ).status == 200
        assert runner.requests[-1]["instruct"] == "calm"
        assert client.request(
            "POST",
            "/v1/audio/speech",
            {"input": "hi", "voice": "alloy", "speed": 1.0},
        ).status == 200

        before = len(runner.requests)
        for fmt in ("mp3", "opus", "aac", "pcm", [], {}):
            response = client.request(
                "POST",
                "/v1/audio/speech",
                {"input": "hi", "voice": "alloy", "response_format": fmt},
            )
            assert response.status == 400
            assert response.json["error"]["details"]["supported"] == ["flac", "wav"]
        for voice in ("builtin:", "voice:", "voice:../x", "voice:a/b"):
            response = client.request(
                "POST",
                "/v1/audio/speech",
                {"input": "hi", "voice": voice},
            )
            assert response.status == 400, voice
            assert response.json["error"]["code"] == "invalid_request", voice
        assert len(runner.requests) == before
        assert client.request(
            "POST",
            "/v1/audio/speech",
            {"input": "hi", "voice": "builtin:Eldric Sage"},
        ).status == 200
        assert client.request(
            "POST",
            "/v1/audio/speech",
            {"input": "hi", "voice": "design:温柔的女声"},
        ).status == 200
        assert client.request(
            "POST",
            "/v1/audio/speech",
            {"input": "hi", "voice": "alloy", "speed": 1.5},
        ).status == 400
        assert client.request(
            "POST",
            "/v1/audio/speech",
            {"input": "hi", "voice": "alloy", "speed": True},
        ).status == 400
        assert client.request(
            "POST",
            "/v1/audio/speech",
            {"input": "hi", "voice": "alloy", "stream_format": "sse"},
        ).status == 400
        assert client.request(
            "POST",
            "/v1/audio/speech",
            {"input": "hi", "voice": "alloy", "unknown": 1},
        ).status == 400
        assert client.request(
            "POST",
            "/v1/audio/speech",
            {"input": "hi", "voice": "alloy", "model": "x" * 257},
        ).status == 400
        assert client.request(
            "POST", "/v1/audio/speech", {"voice": "alloy"}
        ).status == 400

        models = client.request("GET", "/v1/models")
        assert models.status == 200
        assert [item["id"] for item in models.json["data"]] == [
            "first-model",
            "second-model",
        ]
        assert models.json["data"][0]["object"] == "model"
        assert models.json["data"][0]["owned_by"] == "qtts"


@pytest.mark.parametrize(
    ("error", "status", "code"),
    [
        (InputError("bad input"), 400, "invalid_request"),
        (ModelError("no model"), 503, "model_error"),
        (QCError("bad qc"), 422, "unprocessable"),
        (RuntimeError("boom"), 500, "internal_error"),
    ],
)
def test_openai_failure_status_mapping(
    monkeypatch, tmp_path: Path, error, status: int, code: str
):
    runner = RecordingRunner(fail=error)
    with live_server(monkeypatch, tmp_path, runner=runner) as server:
        response = server.client().request(
            "POST",
            "/v1/audio/speech",
            {"input": "hi", "voice": "alloy"},
        )
        assert response.status == status
        assert response.json["error"]["code"] == code
        assert "Traceback" not in response.text


def test_openai_cancelled_job_is_503(monkeypatch, tmp_path: Path):
    runner = RecordingRunner(work=lambda req, wd, cancel: (_ for _ in ()).throw(
        CancelledError("cancelled")
    ))
    with live_server(monkeypatch, tmp_path, runner=runner) as server:
        response = server.client().request(
            "POST", "/v1/audio/speech", {"input": "hi", "voice": "alloy"}
        )
        assert response.status == 503
        assert response.json["error"]["code"] == "cancelled"


def test_api_mode_gating(monkeypatch, tmp_path: Path):
    with live_server(monkeypatch, tmp_path / "native", api="native") as server:
        client = server.client()
        assert client.request("GET", "/health").status == 200
        assert client.request("GET", "/v1/models").status == 404
        assert client.request(
            "POST", "/v1/audio/speech", {"input": "hi", "voice": "alloy"}
        ).status == 404
        assert client.request("POST", "/v1/jobs", JOB).status == 202

    with live_server(monkeypatch, tmp_path / "openai", api="openai") as server:
        client = server.client()
        assert client.request("GET", "/health").status == 200
        assert client.request("GET", "/v1/jobs").status == 404
        assert client.request(
            "POST", "/v1/audio/speech", {"input": "hi", "voice": "alloy"}
        ).status == 200


def test_connection_limit_returns_503(monkeypatch, tmp_path: Path):
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
    with live_server(
        monkeypatch,
        tmp_path,
        runner=runner,
        max_connections=1,
    ) as server:
        client = server.client()
        result = {}

        def speech() -> None:
            result["response"] = client.request(
                "POST",
                "/v1/audio/speech",
                {"input": "hi", "voice": "alloy"},
            )

        thread = threading.Thread(target=speech, daemon=True)
        thread.start()
        wait_until(lambda: runner.started == 1)
        busy = client.request("GET", "/health")
        assert busy.status == 503
        assert busy.json["error"] == "server_busy"
        release.set()
        thread.join(10)
        assert not thread.is_alive()
        assert result["response"].status == 200


def test_delete_queued_job_releases_http_queue_capacity(
    monkeypatch, tmp_path: Path
):
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
    with live_server(
        monkeypatch, tmp_path, runner=runner, queue_size=1
    ) as server:
        client = server.client()
        first = client.request("POST", "/v1/jobs", JOB)
        assert first.status == 202
        wait_until(lambda: runner.started == 1)
        second = client.request("POST", "/v1/jobs", JOB)
        assert second.status == 202
        second_id = second.json["job"]["id"]
        wait_until(
            lambda: client.request("GET", f"/v1/jobs/{second_id}").json["job"][
                "status"
            ]
            == "queued"
        )
        deleted = client.request("DELETE", f"/v1/jobs/{second_id}")
        assert deleted.status == 200
        assert deleted.json["job"]["status"] == "cancelled"

        third = client.request("POST", "/v1/jobs", JOB)
        assert third.status == 202
        assert third.json["job"]["queue_position"] == 0
        release.set()
        assert wait_job(client, first.json["job"]["id"])["status"] == "done"
        assert wait_job(client, third.json["job"]["id"])["status"] == "done"


def test_retained_job_returns_404_over_http(monkeypatch, tmp_path: Path):
    with live_server(monkeypatch, tmp_path, keep_jobs=1) as server:
        client = server.client()
        ids = []
        for _ in range(3):
            response = client.request("POST", "/v1/jobs", JOB)
            assert response.status == 202
            job_id = response.json["job"]["id"]
            ids.append(job_id)
            assert wait_job(client, job_id)["status"] == "done"

        assert client.request("GET", f"/v1/jobs/{ids[0]}").status == 404
        assert client.request("GET", f"/v1/jobs/{ids[1]}").status == 404
        listing = client.request("GET", "/v1/jobs")
        assert [item["id"] for item in listing.json["jobs"]] == [ids[2]]


def test_submit_during_shutdown_maps_to_503(
    monkeypatch, tmp_path: Path
):
    with live_server(monkeypatch, tmp_path) as server:
        def closed(request, **kwargs):
            raise StoreClosed()

        monkeypatch.setattr(server.store, "submit", closed)
        native = server.client().request("POST", "/v1/jobs", JOB)
        assert native.status == 503
        assert native.json["error"] == "service_unavailable"

        openai = server.client().request(
            "POST",
            "/v1/audio/speech",
            {"input": "hi", "voice": "alloy"},
        )
        assert openai.status == 503
        assert openai.json["error"]["type"] == "service_unavailable"
        assert openai.json["error"]["code"] == "service_unavailable"


def test_openai_auth_and_host_errors_use_openai_shape(
    monkeypatch, tmp_path: Path
):
    token = "openai-secret"
    with live_server(monkeypatch, tmp_path, token=token) as server:
        client = server.client()
        for method, path, body in (
            ("POST", "/v1/audio/speech", {"input": "hi", "voice": "alloy"}),
            ("GET", "/v1/models", None),
        ):
            response = client.request(method, path, body)
            assert response.status == 401
            assert "ok" not in response.json
            assert response.json["error"]["code"] == "unauthorized"
            assert response.json["error"]["type"] == "unauthorized"

        forbidden = client.request(
            "GET",
            "/v1/models",
            headers={
                "Authorization": f"Bearer {token}",
                "Host": "evil.test",
            },
        )
        assert forbidden.status == 403
        assert forbidden.json["error"]["code"] == "forbidden"
        assert forbidden.json["error"]["type"] == "forbidden"


def test_non_finite_runner_values_are_sanitized(monkeypatch, tmp_path: Path):
    nan = float("nan")
    inf = float("inf")

    def work(req, wd, cancel):
        (wd / "audio.wav").write_bytes(AUDIO_BYTES)
        return {
            "ok": True,
            "format": "wav",
            "duration_s": nan,
            "sampling": {"temperature": nan, "top_p": inf},
            "cache": {"hits": 0},
            "timing": {"generate_ms": inf, "total_ms": nan},
            "align_coverage": inf,
            "segments": [{"id": 1, "generate_ms": nan}],
            "warnings": [],
            "subtitles": {},
        }

    def reject_constant(value: str):
        raise AssertionError(f"non-finite JSON constant: {value}")

    with live_server(
        monkeypatch, tmp_path, runner=RecordingRunner(work=work)
    ) as server:
        client = server.client()
        created = client.request("POST", "/v1/jobs", JOB)
        job_id = created.json["job"]["id"]
        wait_job(client, job_id)
        response = client.request("GET", f"/v1/jobs/{job_id}")
        assert b"NaN" not in response.body and b"Infinity" not in response.body
        job = json.loads(
            response.body, parse_constant=reject_constant
        )["job"]
        assert job["output"]["duration_s"] is None
        assert job["metrics"]["generate_ms"] is None
        assert job["metrics"]["total_ms"] is None

        result_path = (
            tmp_path / "data" / "serve" / "jobs" / job_id / "result.json"
        )
        text = result_path.read_text(encoding="utf-8")
        assert "NaN" not in text and "Infinity" not in text
        result = json.loads(text, parse_constant=reject_constant)
        assert result["output"]["duration_s"] is None
        assert result["metrics"]["generate_ms"] is None
        payload = server.store.get(job_id).result["payload"]
        assert payload["duration_s"] is None
        assert payload["sampling"] == {"temperature": None, "top_p": None}
        assert payload["timing"]["generate_ms"] is None
        assert payload["align_coverage"] is None
        assert payload["segments"][0]["generate_ms"] is None


def test_concurrent_openai_requests_survive_retention(
    monkeypatch, tmp_path: Path
):
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
    with live_server(
        monkeypatch,
        tmp_path,
        runner=runner,
        spec=FakeSpec(remote=True),
        keep_jobs=1,
        max_concurrency=2,
    ) as server:
        responses: dict[str, object] = {}

        def speech(name: str) -> None:
            responses[name] = server.client().request(
                "POST",
                "/v1/audio/speech",
                {"input": "hi", "voice": "alloy"},
            )

        threads = [
            threading.Thread(target=speech, args=(name,), daemon=True)
            for name in ("first", "second")
        ]
        for thread in threads:
            thread.start()
        wait_until(lambda: runner.started == 2)
        release.set()
        for thread in threads:
            thread.join(15)
            assert not thread.is_alive()

        assert responses["first"].status == 200
        assert responses["second"].status == 200
        assert responses["first"].body == AUDIO_BYTES
        assert responses["second"].body == AUDIO_BYTES
        jobs_root = tmp_path / "data" / "serve" / "jobs"
        wait_until(
            lambda: len([p for p in jobs_root.iterdir() if p.is_dir()]) == 1
        )


def test_openai_method_not_allowed_uses_openai_shape(
    monkeypatch, tmp_path: Path
):
    with live_server(monkeypatch, tmp_path) as server:
        client = server.client()
        get_speech = client.request("GET", "/v1/audio/speech")
        assert get_speech.status == 405
        assert get_speech.headers["allow"] == "POST"
        assert "ok" not in get_speech.json
        assert get_speech.json["error"]["code"] == "method_not_allowed"

        post_models = client.request("POST", "/v1/models", {})
        assert post_models.status == 405
        assert post_models.headers["allow"] == "GET"
        assert "ok" not in post_models.json
        assert post_models.json["error"]["type"] == "method_not_allowed"
