"""``qtts serve``: long-running local TTS HTTP service (FR-P2-01).

Importing this package stays lightweight: backends and ``commands.say`` are
imported lazily inside :func:`run`.
"""

from __future__ import annotations

import contextlib
import ipaddress
import json
import math
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from ..errors import InputError, ModelError, QttsError, UsageError
from ..utils import stdout_to_stderr
from .http import ServeHTTPState, create_server
from .jobs import JobStore, cleanup_orphan_jobs, cleanup_staging
from .runner import InProcessSayRunner

_LOOPBACK = frozenset({"127.0.0.1", "::1", "localhost"})
_WILDCARD = frozenset({"0.0.0.0", "::"})


def _flag(args: Any, name: str, default: Any) -> Any:
    value = getattr(args, name, None)
    return default if value is None else value


def _number(args: Any, name: str, default: Any, minimum: Any, *, integer: bool):
    value = _flag(args, name, default)
    if isinstance(value, bool):
        raise UsageError(f"--{name.replace('_', '-')} must be a number")
    if integer:
        if not isinstance(value, int):
            try:
                value = int(value)
            except (TypeError, ValueError):
                raise UsageError(
                    f"--{name.replace('_', '-')} must be an integer"
                ) from None
    else:
        if not isinstance(value, (int, float)):
            try:
                value = float(value)
            except (TypeError, ValueError):
                raise UsageError(
                    f"--{name.replace('_', '-')} must be a number"
                ) from None
        value = float(value)
        if not math.isfinite(value):
            raise UsageError(f"--{name.replace('_', '-')} must be finite")
    if value < minimum:
        raise UsageError(
            f"--{name.replace('_', '-')} must be >= {minimum}"
        )
    return value


def _validated_allowed_hosts(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    for raw in values:
        value = str(raw).strip()
        if not value or "://" in value or "/" in value:
            raise UsageError(f"invalid --allowed-host: {raw!r}")
        host = value
        if host.startswith("["):
            if not host.endswith("]"):
                raise UsageError(f"invalid --allowed-host: {raw!r}")
            host = host[1:-1]
            try:
                ipaddress.IPv6Address(host)
            except ValueError:
                raise UsageError(f"invalid --allowed-host: {raw!r}") from None
        elif ":" in host:
            try:
                ipaddress.ip_address(host)
            except ValueError:
                raise UsageError(
                    f"--allowed-host must not include a port: {raw!r}"
                ) from None
        result.append(host)
    return result


def _display_host(host: str) -> str:
    if ":" in host and not host.startswith("["):
        return f"[{host}]"
    return host


def _model_ids(spec, cfg) -> list[str]:
    if spec.remote:
        if cfg.backend_model:
            return [str(cfg.backend_model)]
        if spec.models:
            return [spec.models[0].model_id]
        return [spec.identity]
    return [model.model_id for model in spec.models]


def _backend_report(spec, cfg, configured: bool, *, startup: bool) -> dict[str, Any]:
    report = {
        "name": spec.name,
        "identity": spec.identity,
        "models": _model_ids(spec, cfg),
        "configured": configured,
    }
    if not startup:
        report["remote"] = bool(spec.remote)
    return report


def _startup_cleanup(root: Path, keep_jobs: int) -> None:
    # Snapshot before the worker starts: only residue present at startup may be
    # removed.  Newly submitted jobs write into staging concurrently.
    staging_entries = None
    staging_error: Optional[BaseException] = None
    try:
        staging_entries = list((Path(root) / "staging").iterdir())
    except Exception as exc:  # noqa: BLE001
        staging_error = exc

    def cleanup() -> None:
        if staging_error is not None:
            print(
                f"qtts serve: staging cleanup failed: {staging_error}",
                file=sys.stderr,
                flush=True,
            )
        else:
            try:
                cleanup_staging(root, staging_entries)
            except Exception as exc:  # noqa: BLE001
                print(
                    f"qtts serve: staging cleanup failed: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
        try:
            cleanup_orphan_jobs(root, keep_jobs)
        except Exception as exc:  # noqa: BLE001
            print(
                f"qtts serve: job cleanup failed: {exc}",
                file=sys.stderr,
                flush=True,
            )

    thread = threading.Thread(target=cleanup, name="qtts-serve-cleanup", daemon=True)
    thread.start()


def run(
    cfg,
    args,
    *,
    runner_factory: Optional[Callable[[], Any]] = None,
    guard_stdout: Optional[bool] = None,
    stop_event: Optional[threading.Event] = None,
    ready_callback: Optional[Callable[[Any, ServeHTTPState], None]] = None,
) -> dict[str, Any]:
    api = _flag(args, "api", "both")
    if api not in ("native", "openai", "both"):
        raise UsageError("--api must be native, openai or both")
    host = str(_flag(args, "host", "127.0.0.1")).strip()
    if not host:
        raise UsageError("--host must not be empty")
    port = _number(args, "port", 8000, 0, integer=True)
    if port > 65535:
        raise UsageError("--port must be <= 65535")
    max_concurrency = _number(args, "max_concurrency", 1, 1, integer=True)
    queue_size = _number(args, "queue_size", 32, 0, integer=True)
    max_connections = _number(args, "max_connections", 32, 1, integer=True)
    keep_jobs = _number(args, "keep_jobs", 100, 0, integer=True)
    max_body_bytes = _number(args, "max_body_bytes", 1048576, 1024, integer=True)
    shutdown_timeout = _number(args, "shutdown_timeout", 10.0, 0.0, integer=False)
    allowed_hosts = _validated_allowed_hosts(
        getattr(args, "allowed_host", None) or []
    )

    token = getattr(args, "token", None)
    if token is None:
        token = os.environ.get("QTTS_SERVE_TOKEN")
    if token is not None and token == "":
        raise UsageError("--token must not be empty")
    if host not in _LOOPBACK and not token:
        raise UsageError(
            "a serve token is required when binding to a non-loopback host"
        )
    if host in _WILDCARD and not allowed_hosts:
        raise UsageError(
            "--allowed-host is required when binding to a wildcard address"
        )

    from ..backends.registry import resolve_backend

    with stdout_to_stderr():
        spec = resolve_backend(cfg.backend, cfg, args)
        if spec.remote and cfg.offline:
            raise UsageError("--offline cannot be used with a remote backend")
        configured = True
        if spec.configured is not None:
            try:
                configured = bool(spec.configured())
            except QttsError:
                raise
            except Exception as exc:  # noqa: BLE001 - startup never leaks a traceback
                raise ModelError(
                    f"backend {spec.name!r} configured() failed: "
                    f"{type(exc).__name__}: {exc}",
                    details={"backend": spec.name},
                ) from exc
        if not configured and not bool(getattr(args, "dry_run", False)):
            raise ModelError(
                f"backend {spec.name!r} is not configured",
                details={"backend": spec.name},
            )

    effective = 1 if not spec.remote else int(max_concurrency)

    if bool(getattr(args, "dry_run", False)):
        return {
            "ok": True,
            "dry_run": True,
            "host": host,
            "port": port,
            "api": api,
            "backend": _backend_report(spec, cfg, configured, startup=False),
            "allowed_hosts": list(allowed_hosts),
            "max_concurrency": int(max_concurrency),
            "effective_max_concurrency": effective,
            "queue_size": int(queue_size),
            "max_connections": int(max_connections),
            "keep_jobs": int(keep_jobs),
            "max_body_bytes": int(max_body_bytes),
            "shutdown_timeout": float(shutdown_timeout),
        }

    root = Path(cfg.data_dir) / "serve"
    shared_engine_cache: dict[str, Any] = {}
    shared_aligner_cache: dict[str, Any] = {}

    if runner_factory is None:
        model = getattr(args, "model", None)

        def factory() -> InProcessSayRunner:
            if spec.remote:
                return InProcessSayRunner(
                    cfg,
                    model=model,
                    aligner_cache=shared_aligner_cache,
                )
            return InProcessSayRunner(
                cfg,
                model=model,
                engine_cache=shared_engine_cache,
                aligner_cache=shared_aligner_cache,
            )

        factory_fn: Callable[[], Any] = factory
    else:
        factory_fn = runner_factory

    secrets = [value for value in (token, os.environ.get("DASHSCOPE_API_KEY")) if value]
    host_dirs = [
        cfg.data_dir,
        cfg.cache_dir,
        cfg.model_dir,
        Path.cwd(),
        root / "staging",
        root / "jobs",
    ]
    host_dirs = [path for path in host_dirs if path is not None]

    store = JobStore(
        root,
        max_concurrency=int(max_concurrency),
        queue_size=int(queue_size),
        keep_jobs=int(keep_jobs),
        local_runtime=not spec.remote,
        runner_factory=factory_fn,
        secrets=secrets,
        host_dirs=host_dirs,
    )
    state = ServeHTTPState(
        store=store,
        api=api,
        backend=_backend_report(spec, cfg, configured, startup=False),
        token=token,
        bind_host=host,
        allowed_hosts=allowed_hosts,
        max_connections=int(max_connections),
        max_body_bytes=int(max_body_bytes),
        queue_size=int(queue_size),
        max_concurrency=int(max_concurrency),
        effective_max_concurrency=effective,
        started_at=time.monotonic(),
        shutdown_event=stop_event,
    )

    try:
        server = create_server(state, host, port)
    except OSError as exc:
        raise InputError(
            f"cannot bind {host}:{port}: {exc}"
        ) from exc

    _startup_cleanup(root, int(keep_jobs))

    actual_port = server.server_address[1]
    startup = {
        "ok": True,
        "event": "listening",
        "host": host,
        "port": actual_port,
        "api": api,
        "backend": _backend_report(spec, cfg, configured, startup=True),
        "allowed_hosts": list(allowed_hosts),
        "max_concurrency": int(max_concurrency),
        "effective_max_concurrency": effective,
        "queue_size": int(queue_size),
        "max_connections": int(max_connections),
    }
    if guard_stdout is None:
        guard_stdout = runner_factory is None

    event = stop_event or threading.Event()
    stop_requested = {"second": False}
    old_handlers: dict[int, Any] = {}

    def handle_signal(signum, frame) -> None:  # noqa: ARG001
        if stop_requested["second"]:
            os._exit(130)
        stop_requested["second"] = True
        event.set()

    def watch() -> None:
        event.wait()
        try:
            server.shutdown()
        except Exception:
            pass

    # Install the handlers and watcher before the listening line is visible:
    # a SIGINT delivered immediately after that line must always be graceful.
    if threading.current_thread() is threading.main_thread():
        for signum in (signal.SIGINT, signal.SIGTERM):
            try:
                old_handlers[signum] = signal.signal(signum, handle_signal)
            except (ValueError, OSError):
                pass
    watcher = threading.Thread(target=watch, name="qtts-serve-stop", daemon=True)
    watcher.start()

    if bool(getattr(cfg, "json", False)):
        sys.stdout.write(
            json.dumps(startup, ensure_ascii=False, separators=(",", ":")) + "\n"
        )
    else:
        sys.stdout.write(
            "qtts serve listening on "
            f"http://{_display_host(host)}:{actual_port} "
            f"(api={api}, backend={spec.name})\n"
        )
    sys.stdout.flush()

    stack = contextlib.ExitStack()
    try:
        if guard_stdout:
            stack.enter_context(stdout_to_stderr())
        store.start()
        if ready_callback is not None:
            ready_callback(server, state)
        server.serve_forever(poll_interval=0.1)
    finally:
        event.set()
        try:
            store.shutdown(float(shutdown_timeout))
        finally:
            try:
                server.server_close()
            finally:
                stack.close()
                for signum, handler in old_handlers.items():
                    try:
                        signal.signal(signum, handler)
                    except (ValueError, OSError):
                        pass
    return {"raw_stdout": True, "exit_code": 0}
