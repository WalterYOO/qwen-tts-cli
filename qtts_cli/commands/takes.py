"""`qtts takes` — inspect and promote multi-take assets without models."""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional

from ..errors import InputError, UsageError
from ..takes import (
    TakesStore,
    collision_key,
    validate_job,
    validate_job_id,
)
from ..voice_store import utc_now


def _store(cfg) -> TakesStore:
    return TakesStore(cfg.data_dir)


def _job(store: TakesStore, job_id: str) -> tuple[dict[str, Any], Path]:
    job = store.load_job(job_id)
    return job, store.job_dir(job_id)


def _check_files(store: TakesStore, job: dict[str, Any], job_dir: Path) -> None:
    store.verify_takes(job, job_dir=job_dir)


def _parse_ids(raw: Optional[str], job: dict[str, Any]) -> Optional[list[int]]:
    if raw is None:
        return None
    parts = [part.strip() for part in raw.split(",")]
    if not raw.strip() or any(not part for part in parts):
        raise UsageError("--ids must be a non-empty comma-separated list")
    ids: list[int] = []
    seen: set[int] = set()
    known = {int(take["id"]) for take in job["takes"]}
    for part in parts:
        try:
            take_id = int(part)
        except ValueError as exc:
            raise UsageError(f"invalid take id {part!r}") from exc
        if take_id in seen:
            raise UsageError(f"duplicate take id {take_id}")
        if take_id not in known:
            raise InputError(
                f"unknown take id {take_id}",
                details={"job": job.get("job_id"), "take": take_id},
            )
        seen.add(take_id)
        ids.append(take_id)
    return ids


# --------------------------------------------------------------------------- #
# list
# --------------------------------------------------------------------------- #
def list_jobs(cfg, args) -> dict[str, Any]:
    store = _store(cfg)
    job_id = validate_job_id(args.job_id)
    job, job_dir = _job(store, job_id)
    _check_files(store, job, job_dir)
    # `list --json` is the raw job.json document, not a wrapper.
    if cfg.json:
        return job
    return {"ok": True, "job": job, "job_path": str(job_dir)}


def format_list(payload: dict[str, Any]) -> str:
    job = payload["job"]
    selected = job.get("selected")
    lines = [
        f"{job['job_id']}: state={job['state']} format={job['format']} "
        f"takes={len(job['takes'])} selected={selected if selected is not None else '-'}"
    ]
    if job.get("selected_at"):
        lines.append(f"  selected_at: {job['selected_at']}")
    criteria = (job.get("request", {}).get("selection") or {}).get("criteria", [])
    if criteria:
        lines.append(f"  criteria: {','.join(criteria)}")
    for take in sorted(job["takes"], key=lambda item: int(item["id"])):
        metrics = take.get("metrics") or {}
        parts = [f"take {take['id']}"]
        if take.get("rank") is not None:
            parts.append(f"rank={take['rank']}")
        if take.get("seed") is not None:
            parts.append(f"seed={take['seed']}")
        for key in ("cer", "speaker_similarity", "lufs", "true_peak_dbtp"):
            if key in metrics:
                value = metrics[key]
                parts.append(f"{key}={value if value is not None else 'n/a'}")
        if int(take["id"]) == selected:
            parts.append("selected")
        lines.append("  " + " ".join(parts))
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# compare
# --------------------------------------------------------------------------- #
def compare(cfg, args) -> dict[str, Any]:
    store = _store(cfg)
    job_id = validate_job_id(args.job_id)
    job, job_dir = _job(store, job_id)
    _check_files(store, job, job_dir)
    ids = _parse_ids(args.ids, job)
    wanted = set(ids) if ids is not None else {int(t["id"]) for t in job["takes"]}
    takes = [
        take
        for take in sorted(job["takes"], key=lambda item: int(item["id"]))
        if int(take["id"]) in wanted
    ]
    return {
        "ok": True,
        "job_id": job["job_id"],
        "state": job["state"],
        "format": job["format"],
        "selected": job.get("selected"),
        "criteria": (job.get("request", {}).get("selection") or {}).get(
            "criteria", []
        ),
        "lufs_target": (job.get("request", {}).get("selection") or {}).get(
            "lufs_target"
        ),
        "takes": [
            {
                "id": take["id"],
                "seed": take.get("seed"),
                "rank": take.get("rank"),
                "file": take["file"],
                "audio_sha256": take.get("audio_sha256"),
                "duration_s": take.get("duration_s"),
                "metrics": take.get("metrics") or {},
                "unavailable": take.get("unavailable") or {},
            }
            for take in takes
        ],
    }


def format_compare(payload: dict[str, Any]) -> str:
    lines = [
        f"{payload['job_id']}: state={payload['state']} selected="
        f"{payload.get('selected') if payload.get('selected') is not None else '-'}"
    ]
    criteria = payload.get("criteria") or []
    if criteria:
        lines.append(f"  criteria: {','.join(criteria)}")
    for take in payload["takes"]:
        metrics = take.get("metrics") or {}
        values = ", ".join(
            f"{key}={value if value is not None else 'n/a'}"
            for key, value in metrics.items()
        )
        lines.append(
            f"  take {take['id']} rank={take.get('rank')} {values}".rstrip()
        )
        for key, reason in (take.get("unavailable") or {}).items():
            lines.append(f"    unavailable {key}: {reason}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# promote
# --------------------------------------------------------------------------- #
def _stage_copy(source: Path, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
    )
    try:
        with os.fdopen(fd, "wb") as out, open(source, "rb") as inp:
            shutil.copyfileobj(inp, out)
            out.flush()
            os.fsync(out.fileno())
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return Path(tmp)


def _copy_atomic(source: Path, target: Path) -> None:
    tmp = _stage_copy(source, target)
    try:
        os.replace(tmp, target)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _resolve_format(args, job: dict[str, Any]) -> tuple[str, Optional[Path]]:
    if args.format is not None and args.format != job["format"]:
        raise UsageError(
            f"--format {args.format} does not match the job format "
            f"{job['format']}; promote does not transcode",
            details={"format": args.format, "job_format": job["format"]},
        )
    if args.out == "-":
        return args.format or job["format"], None
    target = Path(args.out).expanduser()
    suffix = target.suffix.lower().lstrip(".")
    if suffix:
        if suffix not in ("wav", "flac"):
            raise UsageError(
                f"cannot infer audio format from {str(target)!r}: unknown suffix "
                f"'.{suffix}'; use .wav/.flac or pass --format",
                details={"path": str(target), "suffix": suffix},
            )
        if suffix != job["format"]:
            raise UsageError(
                f"cannot promote {job['format']} take to .{suffix}; "
                "promote does not transcode",
                details={"path": str(target), "job_format": job["format"]},
            )
        return suffix, target
    return args.format or job["format"], target


def promote(cfg, args) -> dict[str, Any]:
    if args.out == "-" and cfg.json:
        raise UsageError("--out - cannot be combined with --json")
    store = _store(cfg)
    job_id = validate_job_id(args.job_id)
    with store.lock(job_id):
        job, job_dir = _job(store, job_id)
        take = next(
            (item for item in job["takes"] if int(item["id"]) == int(args.id)),
            None,
        )
        if take is None:
            raise InputError(
                f"unknown take id {args.id}",
                details={"job": job_id, "take": args.id},
            )
        fmt, target = _resolve_format(args, job)
        source = job_dir / take["file"]
        source_key = collision_key(source)
        target_key = collision_key(target) if target is not None else None
        already_selected = (
            job["state"] == "complete" and job.get("selected") == int(take["id"])
        )
        # The only true no-op: already selected and writing onto the source.
        idempotent = already_selected and target_key == source_key

        forbidden = {
            collision_key(job_dir / "job.json"): "job.json",
            collision_key(job_dir / take["file"]): take["file"],
            collision_key(job_dir / take["recipe"]): take["recipe"],
        }
        for other in job["takes"]:
            forbidden[collision_key(job_dir / other["file"])] = other["file"]
            forbidden[collision_key(job_dir / other["recipe"])] = other["recipe"]
        if target_key is not None and target_key != source_key:
            if target_key in forbidden:
                raise UsageError(
                    f"--out may not overwrite takes asset {forbidden[target_key]!r}",
                    details={"path": str(target), "asset": forbidden[target_key]},
                )

        # Validate everything before writing any output or job state.
        store.verify_take(job, take, job_dir)
        previous_selected = job.get("selected")
        previous_state = job["state"]
        updated = dict(job)
        updated["state"] = "complete"
        updated["selected"] = int(take["id"])
        updated["selected_at"] = utc_now()
        try:
            validate_job(updated, expected_id=job_id)
        except Exception as exc:
            raise InputError(
                f"cannot promote take {take['id']} of job {job_id}: {exc}",
                details={"job": job_id, "take": take["id"]},
            ) from exc

        if idempotent:
            return {
                "ok": True,
                "stdout_used": False,
                "job_id": job_id,
                "take_id": int(take["id"]),
                "out": str(target),
                "format": fmt,
                "audio_sha256": take.get("audio_sha256"),
                "previous_selected": previous_selected,
                "previous_state": previous_state,
                "state": job["state"],
                "selected": job.get("selected"),
                "selected_at": job.get("selected_at"),
                "idempotent": True,
            }

        staged: Optional[Path] = None
        if target is not None and target_key != source_key:
            staged = _stage_copy(source, target)

        try:
            store.update_job(job_id, updated)
        except BaseException as exc:
            if staged is not None:
                try:
                    os.unlink(staged)
                except OSError:
                    pass
            raise InputError(
                f"cannot update takes job {job_id}: {exc}",
                details={"job": job_id, "path": str(job_dir / "job.json")},
            ) from exc

        if target is None:
            with open(source, "rb") as handle:
                shutil.copyfileobj(handle, sys.stdout.buffer)
            sys.stdout.buffer.flush()
        elif staged is not None:
            try:
                os.replace(staged, target)
            except BaseException as exc:
                try:
                    os.unlink(staged)
                except OSError:
                    pass
                try:
                    store.update_job(job_id, job)
                except BaseException as rollback_error:
                    raise InputError(
                        f"cannot write promoted take to {target} for job "
                        f"{job_id}; job rollback also failed: {rollback_error}",
                        details={
                            "job": job_id,
                            "path": str(target),
                            "rollback_error": str(rollback_error),
                        },
                    ) from exc
                raise InputError(
                    f"cannot write promoted take to {target} for job {job_id}: {exc}",
                    details={"job": job_id, "path": str(target)},
                ) from exc

        return {
            "ok": True,
            "stdout_used": target is None,
            "job_id": job_id,
            "take_id": int(take["id"]),
            "out": "-" if target is None else str(target),
            "format": fmt,
            "audio_sha256": take.get("audio_sha256"),
            "previous_selected": previous_selected,
            "previous_state": previous_state,
            "state": updated["state"],
            "selected": updated.get("selected"),
            "selected_at": updated.get("selected_at"),
            "idempotent": False,
        }


def format_promote(payload: dict[str, Any]) -> str:
    if payload.get("idempotent"):
        return (
            f"take {payload['take_id']} already selected; "
            f"source and destination are the same ({payload['out']})"
        )
    message = (
        f"promoted take {payload['take_id']} to {payload['out']} "
        f"({payload['format']}); selected={payload['selected']}"
    )
    if payload.get("previous_state") == "scoring_failed":
        message += "; job scoring_failed -> complete"
    return message


def format_result(payload: dict[str, Any]) -> str:
    """Dispatch the human formatter used by the `takes` subcommands."""
    if "job" in payload:
        return format_list(payload)
    if "state" in payload and "takes" in payload:
        return format_compare(payload)
    if "take_id" in payload:
        return format_promote(payload)
    return ""
