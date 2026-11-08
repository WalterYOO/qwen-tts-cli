"""In-memory job queue, worker threads and terminal publication (FR-P2-01).

All state transitions and the single terminal publication happen under one
``threading.Condition`` lock.  Workers are daemon threads (never a
ThreadPoolExecutor) so a backend stuck in ``generate()`` cannot keep the
interpreter alive after a forced shutdown.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from ..errors import CancelledError, QttsError
from .sanitize import public_error, public_payload, redact_text, redacted_traceback

TERMINAL = frozenset({"done", "failed", "cancelled"})
ACTIVE_QUEUE = frozenset({"queued", "waiting_lock"})
API_STATUSES = frozenset(
    {"queued", "running", "cancel_requested", "done", "failed", "cancelled"}
)


class QueueFull(Exception):
    pass


class StoreClosed(Exception):
    """Raised when submission races with server shutdown."""


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso(value: Optional[datetime]) -> Optional[str]:
    if value is None:
        return None
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def api_status(status: str) -> str:
    return "queued" if status == "waiting_lock" else status


@dataclass
class Job:
    id: str
    status: str
    created_at: datetime
    request: dict[str, Any]
    staging: Path
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    result: Optional[dict[str, Any]] = None
    error: Optional[dict[str, Any]] = None
    queue_position: Optional[int] = None
    initial_queue_position: Optional[int] = None
    queue_ms: Optional[float] = None
    run_ms: Optional[float] = None
    abandoned: bool = False
    pins: int = 0
    cancel_event: threading.Event = field(default_factory=threading.Event)

    def public(self, *, status: Optional[str] = None, finished_at: Any = None) -> dict[str, Any]:
        current = status or self.status
        shown = api_status(current)
        finished = self.finished_at if finished_at is None else finished_at
        result = self.result or {}
        payload = {
            "id": self.id,
            "status": shown,
            "created_at": iso(self.created_at),
            "started_at": iso(self.started_at),
            "finished_at": iso(finished),
            "request": self.request,
            "output": result.get("output") if shown == "done" else None,
            "subtitles": result.get("subtitles") if shown == "done" else {},
            "metrics": result.get("metrics") if shown == "done" else None,
            "error": self.error if shown in ("failed", "cancelled") else None,
            "timings": {
                "queue_ms": self.queue_ms,
                "run_ms": self.run_ms,
            },
            "queue_position": (
                self.queue_position if shown == "queued" else None
            ),
        }
        return payload


class JobStore:
    def __init__(
        self,
        root: str | Path,
        *,
        max_concurrency: int,
        queue_size: int,
        keep_jobs: int,
        local_runtime: bool,
        runner_factory: Callable[[], Any],
        secrets: Iterable[str] = (),
        host_dirs: Iterable[str | Path] = (),
    ) -> None:
        self.root = Path(root)
        self.staging_root = self.root / "staging"
        self.jobs_root = self.root / "jobs"
        self.max_concurrency = int(max_concurrency)
        self.effective_max_concurrency = 1 if local_runtime else int(max_concurrency)
        self.queue_size = int(queue_size)
        self.keep_jobs = int(keep_jobs)
        self.local_runtime = bool(local_runtime)
        self.runner_factory = runner_factory
        self.secrets = tuple(str(s) for s in secrets if str(s or ""))
        self.host_dirs = tuple(str(p) for p in host_dirs if str(p or ""))

        self._cond = threading.Condition(threading.Lock())
        self._jobs: dict[str, Job] = {}
        self._pending: list[str] = []
        self._idle_workers = 0
        self._stopping = False
        self._local_lock = threading.Lock()
        self._workers: list[threading.Thread] = []

        self.staging_root.mkdir(parents=True, exist_ok=True)
        self.jobs_root.mkdir(parents=True, exist_ok=True)

    # -- lifecycle ---------------------------------------------------------- #
    def start(self) -> None:
        for index in range(self.effective_max_concurrency):
            thread = threading.Thread(
                target=self._worker,
                name=f"qtts-serve-worker-{index}",
                daemon=True,
            )
            thread.start()
            self._workers.append(thread)
        # Make `queue_size == 0` handoff deterministic: workers are idle (and
        # counted as such) before the server accepts its first request.
        deadline = time.monotonic() + 2.0
        with self._cond:
            while (
                self._idle_workers < len(self._workers)
                and time.monotonic() < deadline
            ):
                self._cond.wait(0.05)

    def wake(self) -> None:
        with self._cond:
            self._cond.notify_all()

    # -- submission --------------------------------------------------------- #
    def submit(self, request: dict[str, Any], *, pin: bool = False) -> Job:
        with self._cond:
            if self._stopping:
                raise StoreClosed()
            # Jobs handed to already-idle workers do not occupy the waiting
            # queue.  ``queue_size == 0`` therefore accepts only that handoff.
            if len(self._pending) >= self.queue_size + self._idle_workers:
                raise QueueFull()
            job_id = "job_" + uuid.uuid4().hex
            staging = self.staging_root / job_id
            staging.mkdir(parents=True, exist_ok=False)
            job = Job(
                id=job_id,
                status="queued",
                created_at=utc_now(),
                request=request,
                staging=staging,
                pins=1 if pin else 0,
            )
            self._jobs[job_id] = job
            self._pending.append(job_id)
            self._renumber_locked()
            job.initial_queue_position = job.queue_position
            self._cond.notify_all()
            return job

    def pin(self, job_id: str) -> bool:
        with self._cond:
            job = self._jobs.get(job_id)
            if job is None:
                return False
            job.pins += 1
            return True

    def unpin(self, job_id: str) -> None:
        with self._cond:
            job = self._jobs.get(job_id)
            if job is None or job.pins <= 0:
                return
            job.pins -= 1
            if job.pins == 0:
                self._retain_locked()

    # -- read --------------------------------------------------------------- #
    def get(self, job_id: str) -> Optional[Job]:
        with self._cond:
            return self._jobs.get(job_id)

    def snapshot(self, job_id: str) -> Optional[dict[str, Any]]:
        with self._cond:
            job = self._jobs.get(job_id)
            return job.public() if job is not None else None

    def list(
        self, statuses: Optional[set[str]] = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        with self._cond:
            jobs = sorted(
                self._jobs.values(), key=lambda item: item.created_at, reverse=True
            )
            result = []
            for job in jobs:
                shown = api_status(job.status)
                if statuses is not None and shown not in statuses:
                    continue
                # Snapshot under the lock; Job fields are mutable.
                result.append(job.public())
                if len(result) >= limit:
                    break
            return result

    def wait(self, job_id: str, timeout: Optional[float] = None) -> Optional[Job]:
        deadline = None if timeout is None else time.monotonic() + timeout
        seen: Optional[Job] = None
        with self._cond:
            while True:
                job = self._jobs.get(job_id)
                if job is None:
                    # Retention may remove a finished job while a waiter is
                    # scheduled; hand back the terminal snapshot it last saw.
                    if seen is not None and seen.status in TERMINAL:
                        return seen
                    return None
                seen = job
                if job.status in TERMINAL:
                    return job
                if deadline is None:
                    self._cond.wait(0.1)
                else:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return job
                    self._cond.wait(min(remaining, 0.1))

    def health(self) -> dict[str, int]:
        with self._cond:
            counts = {
                "total": len(self._jobs),
                "queued": 0,
                "running": 0,
                "done": 0,
                "failed": 0,
                "cancelled": 0,
            }
            for job in self._jobs.values():
                # cancel_requested jobs still occupy a running slot.
                shown = "running" if job.status == "cancel_requested" else api_status(
                    job.status
                )
                if shown in ("queued", "running", "done", "failed", "cancelled"):
                    counts[shown] += 1
            return counts

    # -- cancellation ------------------------------------------------------- #
    def cancel(self, job_id: str) -> tuple[str, Optional[Job]]:
        """Return ``(outcome, job)``; outcome is ok/conflict/not_found."""
        with self._cond:
            job = self._jobs.get(job_id)
            if job is None:
                return "not_found", None
            if job.status in TERMINAL:
                return "conflict", job
            if job.status == "queued":
                try:
                    self._pending.remove(job_id)
                except ValueError:
                    pass
            job.cancel_event.set()
            if job.status in ACTIVE_QUEUE:
                self._prepare_cancel_locked(job)
                self._publish_locked(
                    job,
                    "cancelled",
                    job.staging,
                    run_log="cancelled before execution",
                )
                return "cancelled", job
            if job.status == "running":
                job.status = "cancel_requested"
            self._cond.notify_all()
            return "cancel_requested", job

    # -- publication -------------------------------------------------------- #
    def publish(
        self,
        job_id: str,
        status: str,
        staging: Optional[Path],
        *,
        result: Optional[dict[str, Any]] = None,
        error: Optional[dict[str, Any]] = None,
        run_log: Optional[str] = None,
    ) -> bool:
        with self._cond:
            job = self._jobs.get(job_id)
            if job is None or job.status in TERMINAL or job.abandoned:
                _discard(staging)
                return False
            if result is not None:
                job.result = result
            if error is not None:
                job.error = error
            published = self._publish_locked(job, status, staging, run_log=run_log)
            if not published:
                _discard(staging)
            return published

    def _publish_locked(
        self,
        job: Job,
        status: str,
        staging: Optional[Path],
        *,
        run_log: Optional[str] = None,
    ) -> bool:
        now = utc_now()
        if job.started_at is not None and job.run_ms is None:
            job.run_ms = round((now - job.started_at).total_seconds() * 1000.0, 3)
        job.finished_at = now
        public = job.public(status=status, finished_at=now)
        final_dir = self.jobs_root / job.id
        work_dir = staging
        temporary = None
        try:
            if work_dir is None:
                temporary = self.staging_root / f".{job.id}.forced-{uuid.uuid4().hex}"
                temporary.mkdir(parents=True, exist_ok=False)
                work_dir = temporary
            else:
                work_dir = Path(work_dir)
                if not work_dir.is_dir():
                    work_dir.mkdir(parents=True, exist_ok=True)
            (work_dir / "request.json").write_text(
                json.dumps(job.request, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            (work_dir / "result.json").write_text(
                json.dumps(
                    public, ensure_ascii=False, indent=2, allow_nan=False
                )
                + "\n",
                encoding="utf-8",
            )
            if run_log is not None:
                text = redact_text(run_log, self.secrets, self.host_dirs)
                (work_dir / "run.log").write_text(text + "\n", encoding="utf-8")
            # Exactly one publisher holds the lock; one atomic rename publishes.
            if final_dir.exists():
                shutil.rmtree(final_dir)
            final_dir.parent.mkdir(parents=True, exist_ok=True)
            os.replace(work_dir, final_dir)
        except OSError as exc:
            _discard(temporary or staging)
            job.status = "failed"
            job.finished_at = now
            job.error = {
                "ok": False,
                "error": "internal",
                "message": "internal server error",
            }
            job.abandoned = True
            _write_failure_artifacts(
                final_dir,
                job.request,
                job.public(status="failed", finished_at=now),
            )
            print(
                "qtts serve: cannot publish job "
                f"{job.id}: {redact_text(str(exc), self.secrets, self.host_dirs)}",
                file=sys.stderr,
                flush=True,
            )
            self._renumber_locked()
            self._retain_locked()
            self._cond.notify_all()
            return False
        job.status = status
        job.abandoned = staging is None and status == "cancelled"
        job.queue_position = None
        self._renumber_locked()
        self._retain_locked()
        self._cond.notify_all()
        return True

    # -- worker ------------------------------------------------------------- #
    def _needs_local_lock(self, job: Job) -> bool:
        if self.local_runtime:
            return True
        return bool(job.request.get("subtitle"))

    def _worker(self) -> None:
        runner = self.runner_factory()
        while True:
            with self._cond:
                while not self._pending and not self._stopping:
                    self._idle_workers += 1
                    try:
                        self._cond.wait()
                    finally:
                        self._idle_workers -= 1
                if self._stopping:
                    return
                job_id = self._pending.pop(0)
                job = self._jobs.get(job_id)
                if job is None or job.status != "queued":
                    self._renumber_locked()
                    continue
                needs_lock = self._needs_local_lock(job)
                job.status = "waiting_lock" if needs_lock else "running"
                if not needs_lock:
                    self._start_locked(job)
                self._renumber_locked()

            acquired = False
            if needs_lock:
                while True:
                    with self._cond:
                        if job.status in TERMINAL:
                            break
                    if job.cancel_event.is_set():
                        with self._cond:
                            if job.status not in TERMINAL:
                                self._prepare_cancel_locked(job)
                                self._publish_locked(
                                    job,
                                    "cancelled",
                                    job.staging,
                                    run_log="cancelled before execution",
                                )
                        break
                    if self._local_lock.acquire(timeout=0.1):
                        acquired = True
                        break
                if not acquired:
                    continue
                with self._cond:
                    if job.status in TERMINAL or job.cancel_event.is_set():
                        self._local_lock.release()
                        if job.status not in TERMINAL:
                            self._prepare_cancel_locked(job)
                            self._publish_locked(
                                job,
                                "cancelled",
                                job.staging,
                                run_log="cancelled before execution",
                            )
                        continue
                    self._start_locked(job)
                    self._renumber_locked()

            try:
                self._execute(job, runner)
            finally:
                if acquired:
                    self._local_lock.release()

    def _start_locked(self, job: Job) -> None:
        now = utc_now()
        job.status = "running"
        job.started_at = now
        job.queue_position = None
        job.queue_ms = round((now - job.created_at).total_seconds() * 1000.0, 3)

    def _execute(self, job: Job, runner) -> None:
        cancel_check = job.cancel_event.is_set
        try:
            payload = runner.submit(dict(job.request), job.staging, cancel_check)
            if not isinstance(payload, dict) or payload.get("ok") is False:
                raise RuntimeError("runner returned an unsuccessful payload")
            self._finish_done(job, payload)
        except CancelledError:
            self._prepare_cancel_locked(job)
            self.publish(
                job.id,
                "cancelled",
                job.staging,
                run_log="cancelled",
            )
        except QttsError as exc:
            error = public_error(
                exc, secrets=self.secrets, host_dirs=self.host_dirs
            )
            text = redact_text(exc.message, self.secrets, self.host_dirs)
            with self._cond:
                self._prepare_cancel_locked(job)
            self.publish(
                job.id,
                "failed",
                job.staging,
                error=error,
                run_log=text,
            )
        except Exception as exc:  # noqa: BLE001 - never leak to the caller
            text = redacted_traceback(exc, secrets=self.secrets)
            print(
                f"qtts serve: job {job.id} failed: {text}",
                file=sys.stderr,
                flush=True,
            )
            with self._cond:
                self._prepare_cancel_locked(job)
            self.publish(
                job.id,
                "failed",
                job.staging,
                error={
                    "ok": False,
                    "error": "internal",
                    "message": "internal server error",
                },
                run_log="internal server error\n" + text,
            )

    def _finish_done(self, job: Job, payload: dict[str, Any]) -> None:
        fmt = job.request.get("format", "wav")
        audio_path = job.staging / f"audio.{fmt}"
        if not audio_path.is_file():
            raise RuntimeError("runner produced no audio file")
        public = public_payload(
            payload,
            job.staging,
            secrets=self.secrets,
            host_dirs=self.host_dirs,
        )
        duration = public.get("duration_s")
        output = {
            "url": f"/v1/jobs/{job.id}/audio",
            "file": audio_path.name,
            "format": fmt,
            "bytes": audio_path.stat().st_size,
            "sha256": _sha256_file(audio_path),
            "duration_s": duration,
        }
        subtitles = {}
        for fmt_name, item in (public.get("subtitles") or {}).items():
            subtitles[fmt_name] = {
                "url": f"/v1/jobs/{job.id}/subtitles/{fmt_name}",
                "file": item["file"],
                "bytes": item["bytes"],
            }
        cache = public.get("cache") or {}
        timing = public.get("timing") or {}
        metrics = {
            "cache_hits": int(cache.get("hits", 0)),
            "segments": len(public.get("segments") or []),
            "load_ms": timing.get("model_load_ms"),
            "generate_ms": timing.get("generate_ms"),
            "total_ms": timing.get("total_ms"),
        }
        self.publish(
            job.id,
            "done",
            job.staging,
            result={
                "output": output,
                "subtitles": subtitles,
                "metrics": metrics,
                "payload": public,
            },
        )

    # -- shutdown ----------------------------------------------------------- #
    def shutdown(self, timeout: float) -> None:
        deadline = time.monotonic() + max(0.0, float(timeout))
        with self._cond:
            self._stopping = True
            pending = list(self._pending)
            self._pending.clear()
            for job_id in pending:
                job = self._jobs.get(job_id)
                if job is not None and job.status in ACTIVE_QUEUE:
                    job.cancel_event.set()
                    self._prepare_cancel_locked(job)
                    self._publish_locked(
                        job,
                        "cancelled",
                        job.staging,
                        run_log="cancelled during shutdown",
                    )
            for job in self._jobs.values():
                if job.status == "running":
                    job.cancel_event.set()
                    job.status = "cancel_requested"
                elif job.status == "waiting_lock":
                    job.cancel_event.set()
            self._renumber_locked()
            self._cond.notify_all()

        while time.monotonic() < deadline:
            with self._cond:
                active = [
                    job
                    for job in self._jobs.values()
                    if job.status not in TERMINAL
                ]
                if not active:
                    return
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._cond.wait(min(remaining, 0.1))

        with self._cond:
            stuck = [
                job for job in self._jobs.values() if job.status not in TERMINAL
            ]
            for job in stuck:
                self._prepare_cancel_locked(job)
        for job in stuck:
            self.publish(
                job.id,
                "cancelled",
                None,
                run_log="cancelled after shutdown timeout",
            )

    # -- internals ---------------------------------------------------------- #
    def _prepare_cancel_locked(self, job: Job) -> None:
        """Remove generated artefacts before a failed/cancelled publication."""
        try:
            for child in job.staging.iterdir():
                if child.name in ("request.json", "result.json", "run.log"):
                    continue
                if child.is_dir():
                    shutil.rmtree(child, ignore_errors=True)
                else:
                    child.unlink(missing_ok=True)
        except OSError:
            pass

    def _renumber_locked(self) -> None:
        active = [
            job for job in self._jobs.values() if job.status in ACTIVE_QUEUE
        ]
        active.sort(key=lambda item: (item.created_at, item.id))
        for index, job in enumerate(active):
            job.queue_position = index

    def _retain_locked(self) -> None:
        if self.keep_jobs <= 0:
            return
        # The actual jobs directory is authoritative: it may contain terminal
        # jobs from before this process started.
        try:
            entries = [
                entry for entry in self.jobs_root.iterdir() if entry.is_dir()
            ]
        except OSError:
            return
        candidates: list[tuple[float, Path]] = []
        for entry in entries:
            job = self._jobs.get(entry.name)
            if job is not None:
                if job.pins > 0 or job.status not in TERMINAL:
                    continue
                if job.finished_at is not None:
                    stamp = job.finished_at.timestamp()
                else:
                    try:
                        stamp = entry.stat().st_mtime
                    except OSError:
                        stamp = 0.0
            else:
                try:
                    stamp = entry.stat().st_mtime
                except OSError:
                    stamp = 0.0
            candidates.append((stamp, entry))
        if len(entries) <= self.keep_jobs:
            return
        candidates.sort(key=lambda item: (item[0], item[1].name))
        for _, entry in candidates[: len(entries) - self.keep_jobs]:
            shutil.rmtree(entry, ignore_errors=True)
            self._jobs.pop(entry.name, None)

    def scan_health(self) -> dict[str, int]:
        return self.health()


def _write_failure_artifacts(
    final_dir: Path,
    request: dict[str, Any],
    public: dict[str, Any],
) -> None:
    """Best-effort minimal terminal artifacts when atomic publish failed."""
    try:
        final_dir.mkdir(parents=True, exist_ok=True)
        (final_dir / "request.json").write_text(
            json.dumps(
                request, ensure_ascii=False, indent=2, allow_nan=False
            )
            + "\n",
            encoding="utf-8",
        )
        (final_dir / "result.json").write_text(
            json.dumps(
                public, ensure_ascii=False, indent=2, allow_nan=False
            )
            + "\n",
            encoding="utf-8",
        )
    except (OSError, TypeError, ValueError):
        return


def _discard(path: Optional[Path]) -> None:
    if path is None:
        return
    try:
        if Path(path).is_dir():
            shutil.rmtree(path, ignore_errors=True)
    except OSError:
        pass


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cleanup_staging(
    root: str | Path, entries: Optional[Iterable[Path]] = None
) -> None:
    """Clean staging entries, optionally restricted to a startup snapshot."""
    if entries is None:
        staging_root = Path(root) / "staging"
        try:
            entries = list(staging_root.iterdir())
        except OSError:
            return
    for entry in entries:
        try:
            if entry.is_dir():
                shutil.rmtree(entry, ignore_errors=True)
            else:
                entry.unlink(missing_ok=True)
        except OSError:
            continue


def cleanup_orphan_jobs(root: str | Path, keep_jobs: int) -> None:
    if keep_jobs <= 0:
        return
    jobs_root = Path(root) / "jobs"
    try:
        entries = [p for p in jobs_root.iterdir() if p.is_dir()]
    except OSError:
        return
    entries.sort(key=lambda p: p.stat().st_mtime)
    for entry in entries[: max(0, len(entries) - keep_jobs)]:
        shutil.rmtree(entry, ignore_errors=True)
