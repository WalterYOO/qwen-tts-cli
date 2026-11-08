"""Outbound sanitization: API bodies, result.json and run.log (§7.1)."""

from __future__ import annotations

import json
from pathlib import Path

from qtts_cli.errors import InputError, ModelError
from qtts_cli.serve.sanitize import (
    public_error,
    public_payload,
    redacted_traceback,
)

from test_serve_helpers import AUDIO_BYTES, RecordingRunner, live_server, wait_job


def test_public_payload_is_a_whitelist_and_redacts_free_strings(tmp_path: Path):
    work = tmp_path / "work"
    work.mkdir()
    (work / "audio.srt").write_text("x", encoding="utf-8")
    payload = {
        "ok": True,
        "format": "wav",
        "duration_s": 1.0,
        "warnings": ["bad /Users/me/x token-123"],
        "backend": {"name": "fake", "path": "/tmp/outside"},
        "device": "cpu /usr/local/lib",
        "dtype": "fp32",
        "cache": {"store": "/Users/me/cache"},
        "timing": {"generate_ms": 1.0},
        "sampling": {
            "temperature": 0.5,
            "config": "/Users/me/sampling.json",
            "/Users/me/key": "value",
        },
        "align_coverage": "/Users/me/align",
        "segments": [
            {
                "id": 1,
                "instruct": "say /Users/me/x",
                "language": "/Users/me/lang",
                "text": "not-whitelisted",
                "unknown": "drop",
            }
        ],
        "output": "/Users/me/out.wav",
        "recipe": {"store": "/tmp/x"},
        "voices": ["builtin:Vivian"],
        "models": ["m"],
        "subtitles": {"srt": str(work / "audio.srt")},
    }
    result = public_payload(
        payload,
        work,
        secrets=("token-123",),
        host_dirs=(tmp_path,),
    )
    assert "output" not in result
    assert "recipe" not in result
    assert "voices" not in result
    assert "models" not in result
    assert "text" not in result["segments"][0]
    assert "unknown" not in result["segments"][0]
    assert result["subtitles"] == {"srt": {"file": "audio.srt", "bytes": 1}}
    assert result["sampling"]["temperature"] == 0.5
    assert result["sampling"]["config"] == "<redacted>"
    assert "<redacted>" in result["sampling"]
    assert result["align_coverage"] == "<redacted>"
    assert result["segments"][0]["language"] == "<redacted>"
    text = json.dumps(result)
    assert "/Users/" not in text
    assert "/tmp/outside" not in text
    assert "/usr/local" not in text
    assert "token-123" not in text

    plain = public_payload(payload, work)
    assert plain["segments"][0]["instruct"]


def test_public_error_preserves_code_and_redacts_details():
    error = ModelError(
        "failed: /tmp/outside/model token xyz",
        details={
            "input_path": "/var/secret",
            "cache_dir": "/Users/me/cache",
            "store": "/tmp/store",
            "windows": "C:\\Users\\x\\model",
            "posix": "/usr/local/x",
            "reason": "token xyz",
            "/Users/me/a.wav": "one",
            "/tmp/b.wav": "two",
            "xyz": "three",
        },
    )
    body = public_error(error, secrets=("xyz",))
    assert body["ok"] is False
    assert body["error"] == "model"
    assert body["code"] == 3
    assert "xyz" not in json.dumps(body)
    assert "/tmp/outside" not in json.dumps(body)
    assert "input_path" not in body["details"]
    assert "cache_dir" not in body["details"]
    assert "store" not in body["details"]
    assert "C:\\Users" not in body["details"]["windows"]
    assert "/usr/local" not in body["details"]["posix"]
    assert "Traceback" not in json.dumps(body)
    redacted_keys = {
        key for key in body["details"] if str(key).startswith("<redacted>")
    }
    assert redacted_keys == {"<redacted>", "<redacted>#2", "<redacted>#3"}
    assert sorted(
        body["details"][key] for key in redacted_keys
    ) == ["one", "three", "two"]

    nonfinite = public_error(
        InputError(
            "bad",
            details={"inf": float("inf"), "nan": float("nan")},
        )
    )
    assert nonfinite["details"] == {"inf": None, "nan": None}

    plain = public_error(InputError("bad"))
    assert plain["code"] == 4

    internal = public_error(RuntimeError("boom /Users/me/x"))
    assert internal == {
        "ok": False,
        "error": "internal",
        "message": "internal server error",
    }


def test_redacted_traceback_never_leaks_paths_or_tokens():
    try:
        raise RuntimeError("boom /Users/me/secret token-abc")
    except RuntimeError as exc:
        text = redacted_traceback(exc, secrets=("token-abc",))
    assert "/Users/" not in text
    assert "token-abc" not in text
    assert "RuntimeError" in text


def test_api_and_result_json_are_redacted_but_request_text_is_exempt(
    monkeypatch, tmp_path: Path
):
    token = "serve-token-xyz"

    def work(req, wd, cancel):
        (wd / "audio.wav").write_bytes(AUDIO_BYTES)
        return {
            "ok": True,
            "format": "wav",
            "duration_s": 0.01,
            "warnings": [
                "/Users/me/private.wav",
                "/tmp/outside",
                token,
                str(tmp_path),
            ],
            "backend": {
                "name": "fake",
                "runtime": "C:\\Users\\x\\runtime",
            },
            "device": "cpu /usr/local/lib",
            "dtype": "fp32",
            "cache": {"hits": 0, "dir": "/Users/me/cache"},
            "timing": {"model_load_ms": 0.0, "note": "/var/tmp/x"},
            "align_coverage": None,
            "segments": [
                {
                    "id": 1,
                    "instruct": "read /Users/me/instructions.txt",
                }
            ],
            "subtitles": {},
        }

    runner = RecordingRunner(work=work)
    with live_server(
        monkeypatch, tmp_path, runner=runner, token=token
    ) as server:
        client = server.client()
        auth = {"Authorization": f"Bearer {token}"}
        created = client.request(
            "POST",
            "/v1/jobs",
            {"text": "keep /foo", "voice": "builtin:Vivian"},
            headers=auth,
        )
        job_id = created.json["job"]["id"]
        wait_job(client, job_id, headers=auth)
        response = client.request("GET", f"/v1/jobs/{job_id}", headers=auth)
        assert response.status == 200
        body = response.text
        for forbidden in (
            "/Users/",
            "/tmp/outside",
            "/usr/local",
            "C:\\Users",
            token,
            str(tmp_path),
        ):
            assert forbidden not in body
        job = json.loads(body)["job"]
        assert job["request"]["text"] == "keep /foo"

        result_path = tmp_path / "data" / "serve" / "jobs" / job_id / "result.json"
        result_text = result_path.read_text(encoding="utf-8")
        for forbidden in ("/Users/", "/tmp/outside", token, str(tmp_path)):
            assert forbidden not in result_text

        runner.work = None
        runner.fail = ModelError(
            f"failed /Users/me/model {token}", details={"path": "/tmp/x"}
        )
        failed = client.request(
            "POST",
            "/v1/jobs",
            {"text": "x", "voice": "builtin:Vivian"},
            headers=auth,
        )
        failed_id = failed.json["job"]["id"]
        failed_job = wait_job(client, failed_id, headers=auth)
        assert failed_job["status"] == "failed"
        assert failed_job["error"]["code"] == 3
        log = (
            tmp_path / "data" / "serve" / "jobs" / failed_id / "run.log"
        ).read_text(encoding="utf-8")
        assert token not in log
        assert "/Users/" not in log
        assert "Traceback" not in log


def test_error_detail_keys_are_redacted_end_to_end(
    monkeypatch, tmp_path: Path
):
    token = "detail-token-xyz"
    details = {
        "/Users/x/secret.wav": "no",
        token: "t",
        "/tmp/other.wav": "two",
    }
    runner = RecordingRunner(
        fail=InputError(
            "failed /Users/x/secret.wav",
            details=details,
        )
    )
    with live_server(monkeypatch, tmp_path, runner=runner, token=token) as server:
        client = server.client()
        auth = {"Authorization": f"Bearer {token}"}
        created = client.request(
            "POST",
            "/v1/jobs",
            {"text": "x", "voice": "builtin:Vivian"},
            headers=auth,
        )
        assert created.status == 202
        job_id = created.json["job"]["id"]
        job = wait_job(client, job_id, headers=auth)
        assert job["status"] == "failed"

        response = client.request("GET", f"/v1/jobs/{job_id}", headers=auth)
        text = response.text
        for forbidden in ("/Users/x/secret.wav", "/tmp/other.wav", token):
            assert forbidden not in text
        error_details = job["error"]["details"]
        assert sorted(error_details.values()) == ["no", "t", "two"]
        assert {
            key for key in error_details if str(key).startswith("<redacted>")
        } == {"<redacted>", "<redacted>#2", "<redacted>#3"}

        result_path = (
            tmp_path / "data" / "serve" / "jobs" / job_id / "result.json"
        )
        result_text = result_path.read_text(encoding="utf-8")
        log = (
            tmp_path / "data" / "serve" / "jobs" / job_id / "run.log"
        ).read_text(encoding="utf-8")
        for blob in (result_text, log):
            for forbidden in ("/Users/x/secret.wav", "/tmp/other.wav", token):
                assert forbidden not in blob
