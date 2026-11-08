"""stdlib HTTP transport for ``qtts serve`` (FR-P2-01 §7–§9).

No third-party framework: ``ThreadingHTTPServer`` plus a bounded connection
semaphore, strict request framing, Host/token checks and the native/OpenAI
route tables.  Every outbound string is produced by :mod:`serve.sanitize`.
"""

from __future__ import annotations

import hmac
import json
import re
import socket
import sys
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Iterable, Optional
from urllib.parse import parse_qs, urlsplit

from .. import __version__
from ..capabilities import parse_voice_ref
from ..errors import InputError
from .jobs import API_STATUSES, JobStore, QueueFull, StoreClosed, iso, utc_now
from .sanitize import redact_text

_LOOPBACK = frozenset({"127.0.0.1", "::1", "localhost"})
_WILDCARD = frozenset({"0.0.0.0", "::"})
_SUBTITLE_FORMATS = frozenset({"srt", "vtt", "ass", "json"})
_FORMATS = frozenset({"wav", "flac"})
_RESPONSE_FORMATS = _FORMATS
_OPENAI_PATHS = frozenset({"/v1/audio/speech", "/v1/models"})
_ID_RE = re.compile(r"^/v1/jobs/(job_[0-9a-f]{32})$")
_SUBTITLE_RE = re.compile(
    r"^/v1/jobs/(job_[0-9a-f]{32})/subtitles/(srt|vtt|ass|json)$"
)
_AUDIO_RE = re.compile(r"^/v1/jobs/(job_[0-9a-f]{32})/audio$")
_CONTENT_TYPES = {
    "srt": "text/plain; charset=utf-8",
    "vtt": "text/vtt",
    "ass": "text/plain; charset=utf-8",
    "json": "application/json",
}
_AUDIO_TYPES = {"wav": "audio/wav", "flac": "audio/flac"}


class HTTPFailure(Exception):
    def __init__(
        self,
        status: int,
        error: str,
        message: str,
        *,
        details: Optional[dict[str, Any]] = None,
        headers: Optional[dict[str, str]] = None,
        openai: bool = False,
        kind: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.error = error
        self.message = message
        self.details = details
        self.headers = headers or {}
        self.openai = openai
        self.kind = kind or error


@dataclass
class ServeHTTPState:
    store: JobStore
    api: str
    backend: dict[str, Any]
    token: Optional[str]
    bind_host: str
    allowed_hosts: Iterable[str]
    max_connections: int
    max_body_bytes: int
    queue_size: int
    max_concurrency: int
    effective_max_concurrency: int
    started_at: float
    shutdown_event: Optional[threading.Event] = None

    def __post_init__(self) -> None:
        self.allowed_hosts = tuple(self.allowed_hosts)
        self._connection_lock = threading.Lock()
        self.connections = 0
        allowed = set(_LOOPBACK)
        bind_lower = self.bind_host.strip().lower()
        if bind_lower not in _WILDCARD:
            allowed.add(_strip_brackets(bind_lower))
        for host in self.allowed_hosts:
            value = _strip_brackets(str(host).strip().lower())
            if value:
                allowed.add(value)
        self._allowed_host_set = frozenset(allowed)
        self.is_loopback = bind_lower in _LOOPBACK
        self.token = self.token or None

    def allowed_hosts_public(self) -> list[str]:
        return sorted(self.allowed_hosts)

    def health(self) -> dict[str, Any]:
        jobs = self.store.health()
        with self._connection_lock:
            connections = self.connections
            limit = self.max_connections
        return {
            "ok": True,
            "version": __version__,
            "uptime_s": round(time.monotonic() - self.started_at, 3),
            "backend": {
                "name": self.backend.get("name"),
                "identity": self.backend.get("identity"),
                "models": list(self.backend.get("models") or []),
                "remote": bool(self.backend.get("remote")),
                "configured": bool(self.backend.get("configured", True)),
            },
            "api": self.api,
            "queue": {
                "running": jobs["running"],
                "queued": jobs["queued"],
                "max_concurrency": self.max_concurrency,
                "effective_max_concurrency": self.effective_max_concurrency,
                "queue_size": self.queue_size,
                "connections": connections,
                "max_connections": limit,
            },
            "jobs": jobs,
        }

    def connection_opened(self) -> None:
        with self._connection_lock:
            self.connections += 1

    def connection_closed(self) -> None:
        with self._connection_lock:
            if self.connections > 0:
                self.connections -= 1


def _strip_brackets(value: str) -> str:
    if value.startswith("[") and value.endswith("]"):
        return value[1:-1]
    return value


def _split_host_header(value: str) -> str:
    value = value.strip()
    if value.startswith("["):
        end = value.find("]")
        if end < 0:
            return value.lower()
        return value[1:end].lower()
    if value.count(":") == 1:
        return value.rsplit(":", 1)[0].lower()
    return value.lower()


def _is_loopback_host(host: str) -> bool:
    lowered = _strip_brackets(host.strip().lower())
    return lowered in _LOOPBACK


def _service_body(
    error: str, status: int, message: str, details: Optional[dict[str, Any]]
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "ok": False,
        "error": error,
        "http": status,
        "message": message,
    }
    if details:
        body["details"] = details
    return body


def _openai_body(
    message: str, kind: str, code: str
) -> dict[str, Any]:
    return {"error": {"message": message, "type": kind, "code": code}}


class _BoundedThreadingHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    block_on_close = False
    request_queue_size = 64

    def __init__(self, server_address, handler_class, *, max_connections: int):
        self._slots = threading.BoundedSemaphore(max_connections)
        super().__init__(server_address, handler_class)

    def process_request(self, request, client_address):  # type: ignore[override]
        if not self._slots.acquire(blocking=False):
            self._reject_busy(request)
            self.shutdown_request(request)
            return
        state = getattr(self, "state", None)
        if state is not None:
            state.connection_opened()
        super().process_request(request, client_address)

    def process_request_thread(self, request, client_address):  # type: ignore[override]
        try:
            super().process_request_thread(request, client_address)
        finally:
            state = getattr(self, "state", None)
            if state is not None:
                state.connection_closed()
            self._slots.release()

    def _reject_busy(self, request: socket.socket) -> None:
        body = json.dumps(
            _service_body(
                "server_busy", 503, "too many connections", None
            ),
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        response = (
            b"HTTP/1.1 503 Service Unavailable\r\n"
            b"Content-Type: application/json; charset=utf-8\r\n"
            b"Content-Length: " + str(len(body)).encode() + b"\r\n"
            b"Connection: close\r\n\r\n" + body
        )
        try:
            request.sendall(response)
        except OSError:
            pass


class ServeHTTPServer(_BoundedThreadingHTTPServer):
    state: Optional[ServeHTTPState] = None


def create_server(state: ServeHTTPState, host: str, port: int) -> ServeHTTPServer:
    if ":" in host:

        class Server(_BoundedThreadingHTTPServer):
            address_family = socket.AF_INET6

    else:

        class Server(_BoundedThreadingHTTPServer):  # type: ignore[no-redef]
            pass

    server = Server(
        (host, port), ServeRequestHandler, max_connections=state.max_connections
    )
    server.state = state
    return server  # type: ignore[return-value]


class ServeRequestHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "qtts-serve"
    sys_version = ""
    timeout = 60

    # -- entry points ------------------------------------------------------- #
    def do_GET(self) -> None:  # noqa: N802
        self._run("GET")

    def do_POST(self) -> None:  # noqa: N802
        self._run("POST")

    def do_DELETE(self) -> None:  # noqa: N802
        self._run("DELETE")

    def do_PUT(self) -> None:  # noqa: N802
        self._run("PUT")

    def do_PATCH(self) -> None:  # noqa: N802
        self._run("PATCH")

    def do_HEAD(self) -> None:  # noqa: N802
        self._run("HEAD")

    def do_OPTIONS(self) -> None:  # noqa: N802
        self._run("OPTIONS")

    def do_TRACE(self) -> None:  # noqa: N802
        self._run("TRACE")

    def do_CONNECT(self) -> None:  # noqa: N802
        self._run("CONNECT")

    @property
    def state(self) -> ServeHTTPState:
        return self.server.state  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
        return

    # -- dispatch ----------------------------------------------------------- #
    def _run(self, method: str) -> None:
        self._request_started = time.monotonic()
        self._job_id: Optional[str] = None
        self._response_status = 0
        parsed = urlsplit(self.path)
        self._openai_path = parsed.path in _OPENAI_PATHS
        try:
            self._check_host()
            self._check_auth(parsed.path)
            self._route(method, parsed)
        except HTTPFailure as failure:
            self._send_failure(failure)
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True
        except Exception as exc:  # noqa: BLE001 - never leak a traceback
            text = redact_text(
                f"{type(exc).__name__}: {exc}",
                (self.state.token,) if self.state.token else (),
            )
            print(f"qtts serve: internal error: {text}", file=sys.stderr, flush=True)
            try:
                self._send_json(
                    500,
                    _service_body("internal", 500, "internal server error"),
                )
            except OSError:
                pass
        finally:
            self._log_request()

    def _log_request(self) -> None:
        if not self._response_status:
            return
        path = urlsplit(self.path).path
        record = {
            "ts": iso(utc_now()),
            "method": self.command,
            "path": path,
            "status": self._response_status,
            "job_id": self._job_id,
            "ms": int(round((time.monotonic() - self._request_started) * 1000.0)),
        }
        try:
            print(
                json.dumps(record, ensure_ascii=False, separators=(",", ":")),
                file=sys.stderr,
                flush=True,
            )
        except Exception:
            pass

    # -- host / auth -------------------------------------------------------- #
    def _check_host(self) -> None:
        openai = getattr(self, "_openai_path", False)
        values = self.headers.get_all("Host") or []
        if not values:
            raise HTTPFailure(
                400,
                "invalid_request",
                "Host header required",
                openai=openai,
                kind="invalid_request",
            )
        if len(values) != 1:
            raise HTTPFailure(
                400,
                "invalid_request",
                "duplicate Host header",
                openai=openai,
                kind="invalid_request",
            )
        host = _split_host_header(values[0])
        if not host:
            raise HTTPFailure(
                400,
                "invalid_request",
                "Host header required",
                openai=openai,
                kind="invalid_request",
            )
        if host not in self.state._allowed_host_set:
            raise HTTPFailure(
                403,
                "forbidden",
                "host not allowed",
                openai=openai,
                kind="forbidden",
            )

    def _check_auth(self, path: str) -> None:
        token = self.state.token
        if not token:
            return
        if path == "/health" and self.state.is_loopback:
            return
        header = self.headers.get("Authorization")
        expected = f"Bearer {token}"
        ok = False
        if header is not None:
            try:
                ok = hmac.compare_digest(
                    header.encode("utf-8"), expected.encode("utf-8")
                )
            except Exception:
                ok = False
        if not ok:
            raise HTTPFailure(
                401,
                "unauthorized",
                "invalid or missing bearer token",
                headers={"WWW-Authenticate": "Bearer"},
                openai=getattr(self, "_openai_path", False),
                kind="unauthorized",
            )

    # -- routing ------------------------------------------------------------ #
    def _route(self, method: str, parsed) -> None:
        path = parsed.path
        state = self.state

        if path == "/health":
            self._allow(method, "GET", path)
            self._send_json(200, state.health())
            return

        native_enabled = state.api in ("native", "both")
        openai_enabled = state.api in ("openai", "both")

        if native_enabled and path == "/v1/jobs":
            if method == "POST":
                self._create_job()
                return
            if method == "GET":
                self._list_jobs(parsed.query)
                return
            self._method_not_allowed("GET, POST")

        match = _ID_RE.match(path)
        if native_enabled and match:
            job_id = match.group(1)
            if method == "GET":
                self._get_job(job_id)
                return
            if method == "DELETE":
                self._delete_job(job_id)
                return
            self._method_not_allowed("GET, DELETE")

        match = _AUDIO_RE.match(path)
        if native_enabled and match:
            self._allow(method, "GET", path)
            self._serve_audio(match.group(1))
            return

        match = _SUBTITLE_RE.match(path)
        if native_enabled and match:
            self._allow(method, "GET", path)
            self._serve_subtitle(match.group(1), match.group(2))
            return

        if openai_enabled and path == "/v1/audio/speech":
            self._allow(method, "POST", path)
            self._openai_speech()
            return
        if openai_enabled and path == "/v1/models":
            self._allow(method, "GET", path)
            self._openai_models()
            return

        raise HTTPFailure(404, "not_found", "not found")

    def _allow(self, method: str, allowed: str, path: str) -> None:
        if method not in {item.strip() for item in allowed.split(",")}:
            self._method_not_allowed(allowed)

    def _method_not_allowed(self, allow: str) -> None:
        openai = getattr(self, "_openai_path", False)
        raise HTTPFailure(
            405,
            "method_not_allowed",
            "method not allowed",
            headers={"Allow": allow},
            openai=openai,
            kind="method_not_allowed",
        )

    # -- request body ------------------------------------------------------- #
    def _read_json_body(self) -> dict[str, Any]:
        lengths = self.headers.get_all("Content-Length") or []
        if self.headers.get_all("Transfer-Encoding"):
            raise HTTPFailure(
                400, "invalid_request", "Transfer-Encoding is not supported"
            )
        if len(lengths) != 1:
            raise HTTPFailure(
                400, "invalid_request", "exactly one Content-Length is required"
            )
        raw_length = lengths[0].strip()
        if not raw_length.isdigit():
            raise HTTPFailure(
                400, "invalid_request", "invalid Content-Length"
            )
        length = int(raw_length)
        if length > self.state.max_body_bytes:
            raise HTTPFailure(
                413,
                "payload_too_large",
                "request body too large",
                details={"max_body_bytes": self.state.max_body_bytes},
            )
        content_type = self.headers.get("Content-Type")
        if not content_type:
            raise HTTPFailure(
                415, "unsupported_media_type", "application/json required"
            )
        media = content_type.split(";", 1)[0].strip().lower()
        if media != "application/json":
            raise HTTPFailure(
                415, "unsupported_media_type", "application/json required"
            )
        body = self.rfile.read(length)
        if len(body) != length:
            raise HTTPFailure(
                400, "invalid_request", "truncated request body"
            )
        try:
            text = body.decode("utf-8")
        except UnicodeDecodeError:
            raise HTTPFailure(400, "invalid_request", "body must be UTF-8") from None
        try:
            value = json.loads(
                text,
                object_pairs_hook=_no_duplicate_keys,
                parse_constant=_reject_constant,
            )
        except (ValueError, TypeError):
            raise HTTPFailure(400, "invalid_request", "invalid JSON") from None
        if not isinstance(value, dict):
            raise HTTPFailure(
                400, "invalid_request", "top-level JSON value must be an object"
            )
        return value

    # -- native: create/list/get/delete ------------------------------------ #
    def _create_job(self) -> None:
        body = self._read_json_body()
        request = _validate_job_request(body)
        try:
            job = self.state.store.submit(request)
        except StoreClosed:
            raise HTTPFailure(
                503,
                "service_unavailable",
                "server is shutting down",
            ) from None
        except QueueFull:
            raise HTTPFailure(
                429,
                "queue_full",
                "job queue is full",
                headers={"Retry-After": "1"},
            ) from None
        self._job_id = job.id
        self._send_json(
            202,
            {
                "ok": True,
                "job": {
                    "id": job.id,
                    "status": "queued",
                    "queue_position": job.initial_queue_position,
                    "created_at": iso(job.created_at),
                },
            },
        )

    def _list_jobs(self, query: str) -> None:
        values = parse_qs(query, keep_blank_values=True)
        unknown = set(values) - {"status", "limit"}
        if unknown:
            raise HTTPFailure(
                400,
                "invalid_request",
                f"unknown query parameter: {sorted(unknown)[0]}",
            )
        statuses: Optional[set[str]] = None
        if "status" in values:
            statuses = set()
            for raw_status in values["status"]:
                for chunk in raw_status.split(","):
                    name = chunk.strip()
                    if not name or name not in API_STATUSES:
                        raise HTTPFailure(
                            400, "invalid_request", f"invalid status: {name!r}"
                        )
                    statuses.add(name)
        limit = 100
        if "limit" in values:
            if len(values["limit"]) != 1:
                raise HTTPFailure(400, "invalid_request", "duplicate limit")
            raw = values["limit"][0]
            if not raw.isdigit():
                raise HTTPFailure(400, "invalid_request", "invalid limit")
            limit = int(raw)
            if limit < 1 or limit > 1000:
                raise HTTPFailure(
                    400, "invalid_request", "limit must be between 1 and 1000"
                )
        self._send_json(
            200, {"ok": True, "jobs": self.state.store.list(statuses, limit)}
        )

    def _get_job(self, job_id: str) -> None:
        snapshot = self.state.store.snapshot(job_id)
        if snapshot is None:
            raise HTTPFailure(404, "not_found", "job not found")
        self._send_json(200, {"ok": True, "job": snapshot})

    def _delete_job(self, job_id: str) -> None:
        outcome, job = self.state.store.cancel(job_id)
        if outcome == "not_found" or job is None:
            raise HTTPFailure(404, "not_found", "job not found")
        self._job_id = job.id
        if outcome == "conflict":
            raise HTTPFailure(
                409,
                "conflict",
                "job already finished",
                details={"status": job.public()["status"]},
            )
        if outcome == "cancel_requested":
            self._send_json(202, {"ok": True, "job": job.public()})
            return
        self._send_json(200, {"ok": True, "job": job.public()})

    # -- native: binary ----------------------------------------------------- #
    def _serve_audio(self, job_id: str) -> None:
        job = self.state.store.get(job_id)
        if job is None:
            raise HTTPFailure(404, "not_found", "job not found")
        snapshot = job.public()
        if snapshot["status"] not in ("done", "failed", "cancelled"):
            raise HTTPFailure(409, "conflict", "job is not finished")
        fmt = (job.request or {}).get("format", "wav")
        path = self.state.store.jobs_root / job.id / f"audio.{fmt}"
        if snapshot["status"] != "done" or not path.is_file():
            raise HTTPFailure(404, "not_found", "audio not found")
        self._send_bytes(
            200,
            path.read_bytes(),
            _AUDIO_TYPES.get(fmt, "application/octet-stream"),
            {"Cache-Control": "no-store"},
        )

    def _serve_subtitle(self, job_id: str, fmt: str) -> None:
        job = self.state.store.get(job_id)
        if job is None:
            raise HTTPFailure(404, "not_found", "job not found")
        path = self.state.store.jobs_root / job.id / f"audio.{fmt}"
        if job.public()["status"] != "done" or not path.is_file():
            raise HTTPFailure(404, "not_found", "subtitle not found")
        self._send_bytes(200, path.read_bytes(), _CONTENT_TYPES[fmt])

    # -- OpenAI ------------------------------------------------------------- #
    def _openai_speech(self) -> None:
        from .openai import OpenAIRequestError, map_speech_request

        body = self._read_json_body()
        try:
            request = map_speech_request(body)
        except OpenAIRequestError as exc:
            raise HTTPFailure(
                exc.status,
                exc.code,
                exc.message,
                details=exc.details,
                openai=True,
                kind=exc.code,
            ) from None
        try:
            job = self.state.store.submit(request, pin=True)
        except StoreClosed:
            raise HTTPFailure(
                503,
                "service_unavailable",
                "server is shutting down",
                openai=True,
                kind="service_unavailable",
            ) from None
        except QueueFull:
            raise HTTPFailure(
                429,
                "queue_full",
                "job queue is full",
                headers={"Retry-After": "1"},
                openai=True,
                kind="queue_full",
            ) from None
        self._job_id = job.id
        try:
            finished = self.state.store.wait(job.id)
            if finished is None:
                raise HTTPFailure(
                    503,
                    "internal",
                    "job disappeared",
                    openai=True,
                    kind="internal_error",
                )
            snapshot = finished.public()
            if snapshot["status"] != "done":
                status = _openai_status(snapshot)
                error = snapshot.get("error") or {}
                code = _openai_code(error.get("code"), snapshot["status"])
                if snapshot["status"] == "cancelled":
                    status = 503
                    code = "cancelled"
                raise HTTPFailure(
                    status,
                    code,
                    str(error.get("message") or "job failed"),
                    openai=True,
                    kind=str(error.get("error") or code),
                )
            fmt = (request or {}).get("format", "wav")
            path = self.state.store.jobs_root / finished.id / f"audio.{fmt}"
            if not path.is_file():
                raise HTTPFailure(
                    404,
                    "not_found",
                    "audio not found",
                    openai=True,
                    kind="not_found",
                )
            self._send_bytes(
                200,
                path.read_bytes(),
                _AUDIO_TYPES.get(fmt, "application/octet-stream"),
                {
                    "Content-Disposition": f'attachment; filename="speech.{fmt}"',
                    "Cache-Control": "no-store",
                },
            )
        finally:
            self.state.store.unpin(job.id)

    def _openai_models(self) -> None:
        models = list(self.state.backend.get("models") or [])
        self._send_json(
            200,
            {
                "object": "list",
                "data": [
                    {
                        "id": model,
                        "object": "model",
                        "created": 0,
                        "owned_by": "qtts",
                    }
                    for model in models
                ],
            },
        )

    # -- responses ---------------------------------------------------------- #
    def _send_failure(self, failure: HTTPFailure) -> None:
        if failure.openai:
            body = _openai_body(failure.message, failure.kind, failure.error)
            if failure.details:
                body["error"]["details"] = failure.details
        else:
            body = _service_body(
                failure.error, failure.status, failure.message, failure.details
            )
        self._send_json(failure.status, body, failure.headers)

    def _send_json(
        self,
        status: int,
        payload: dict[str, Any],
        headers: Optional[dict[str, str]] = None,
    ) -> None:
        data = json.dumps(
            payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        self._send_bytes(
            status,
            data,
            "application/json; charset=utf-8",
            headers,
        )

    def _send_bytes(
        self,
        status: int,
        data: bytes,
        content_type: str,
        headers: Optional[dict[str, str]] = None,
    ) -> None:
        self._response_status = int(status)
        try:
            self.send_response(int(status))
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Connection", "close")
            for key, value in (headers or {}).items():
                self.send_header(key, value)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(data)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            self.close_connection = True


def _no_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"non-finite number: {value}")


def _int_field(
    body: dict[str, Any],
    name: str,
    minimum: int,
    maximum: int,
    default: Optional[int],
) -> Optional[int]:
    if name not in body:
        return default
    value = body[name]
    if isinstance(value, bool) or not isinstance(value, int):
        raise HTTPFailure(400, "invalid_request", f"{name} must be an integer")
    if value < minimum or value > maximum:
        raise HTTPFailure(
            400,
            "invalid_request",
            f"{name} must be between {minimum} and {maximum}",
        )
    return value


def _require_string(
    body: dict[str, Any], name: str, *, maximum: Optional[int] = None
) -> str:
    value = body.get(name)
    if not isinstance(value, str) or not value:
        raise HTTPFailure(400, "invalid_request", f"{name} must be a non-empty string")
    if maximum is not None and len(value) > maximum:
        raise HTTPFailure(
            400, "invalid_request", f"{name} exceeds {maximum} characters"
        )
    return value


def _validate_job_request(body: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "text",
        "voice",
        "language",
        "instruct",
        "format",
        "subtitle",
        "max_new_tokens",
        "pause_ms",
        "seed",
        "metadata",
    }
    unknown = set(body) - allowed
    if unknown:
        raise HTTPFailure(
            400,
            "invalid_request",
            f"unknown field: {sorted(unknown)[0]}",
        )
    text = _require_string(body, "text", maximum=200000)
    voice = _require_string(body, "voice")
    try:
        parse_voice_ref(voice)
    except InputError as exc:
        raise HTTPFailure(
            400, "invalid_request", exc.message
        ) from None
    language = body.get("language", "auto")
    if not isinstance(language, str) or not language:
        raise HTTPFailure(400, "invalid_request", "language must be a non-empty string")
    if len(language) > 64:
        raise HTTPFailure(400, "invalid_request", "language exceeds 64 characters")
    instruct = body.get("instruct")
    if instruct is not None:
        if not isinstance(instruct, str):
            raise HTTPFailure(400, "invalid_request", "instruct must be a string or null")
        if len(instruct) > 20000:
            raise HTTPFailure(
                400, "invalid_request", "instruct exceeds 20000 characters"
            )
    fmt = body.get("format", "wav")
    if not isinstance(fmt, str) or fmt not in _FORMATS:
        raise HTTPFailure(
            400,
            "invalid_request",
            "format must be wav or flac",
            details={"supported": sorted(_FORMATS)},
        )
    subtitle_raw = body.get("subtitle", [])
    if not isinstance(subtitle_raw, list) or not all(
        isinstance(item, str) for item in subtitle_raw
    ):
        raise HTTPFailure(
            400, "invalid_request", "subtitle must be an array of strings"
        )
    subtitles = []
    for item in subtitle_raw:
        if item not in _SUBTITLE_FORMATS:
            raise HTTPFailure(
                400,
                "invalid_request",
                f"unsupported subtitle format: {item}",
                details={"supported": sorted(_SUBTITLE_FORMATS)},
            )
        if item not in subtitles:
            subtitles.append(item)
    if len(subtitles) > 4:
        raise HTTPFailure(
            400, "invalid_request", "subtitle supports at most 4 formats"
        )
    max_new_tokens = _int_field(body, "max_new_tokens", 1, 65536, None)
    pause_ms = _int_field(body, "pause_ms", 0, 60000, None)
    seed = _int_field(body, "seed", 0, 2**63 - 1, None)
    metadata = body.get("metadata")
    if metadata is not None:
        if not isinstance(metadata, dict):
            raise HTTPFailure(
                400, "invalid_request", "metadata must be an object or null"
            )
        if len(metadata) > 64:
            raise HTTPFailure(
                400, "invalid_request", "metadata has too many top-level keys"
            )
        canonical = json.dumps(
            metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        if len(canonical) > 4096:
            raise HTTPFailure(
                400, "invalid_request", "metadata exceeds 4096 bytes"
            )
    return {
        "text": text,
        "voice": voice,
        "language": language,
        "instruct": instruct,
        "format": fmt,
        "subtitle": subtitles,
        "max_new_tokens": max_new_tokens,
        "pause_ms": pause_ms,
        "seed": seed,
        "metadata": metadata,
    }


def _openai_status(snapshot: dict[str, Any]) -> int:
    error = snapshot.get("error") or {}
    code = error.get("code")
    return {
        2: 400,
        3: 503,
        4: 400,
        5: 422,
        6: 500,
    }.get(code, 500)


def _openai_code(code: Any, status: str) -> str:
    if status == "cancelled":
        return "cancelled"
    return {
        2: "invalid_request",
        3: "model_error",
        4: "invalid_request",
        5: "unprocessable",
        6: "internal_error",
    }.get(code, "internal_error")
