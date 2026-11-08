"""`qtts batch plan|run` (FR-P1-01)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..batch import (
    _status_record,
    prepare_jobs,
    run_batch,
    write_plan,
    write_status,
)
from ..batch import format_result as _format_batch
from ..backends.registry import resolve_backend
from ..lexicon import Lexicon


def plan(cfg, args) -> dict[str, Any]:
    out_dir = Path(args.out_dir).expanduser()
    lexicon = Lexicon.load(getattr(args, "lexicon", None))
    spec = resolve_backend(getattr(args, "backend", None), cfg, args)
    effective_backend_model = (
        getattr(args, "backend_model", None) or getattr(cfg, "backend_model", None)
    )
    jobs, payload, _plans = prepare_jobs(
        cfg,
        args.manifest,
        out_dir=out_dir,
        model_override=getattr(args, "model", None),
        lexicon=lexicon,
        tts_models=spec.models,
        backend_spec=spec,
        backend_model=effective_backend_model,
    )
    # Only a fully valid plan creates its --out-dir (no side effects on errors).
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        from ..errors import InputError

        raise InputError(f"cannot create --out-dir {out_dir}: {exc}") from exc
    plan_path = write_plan(out_dir, payload)
    state = {job["id"]: _status_record(job) for job in jobs}
    status_path = write_status(out_dir, jobs, state)
    return {
        "ok": True,
        "jobs": len(jobs),
        "plan": str(plan_path),
        "status": str(status_path),
        "models": sorted({job["model"] for job in jobs}),
        "voices": sorted({job["voice_ref"] for job in jobs}),
        "backend": {
            "name": spec.name,
            "remote": spec.remote,
            "identity": spec.identity,
        },
    }


def run(cfg, args) -> dict[str, Any]:
    lexicon = Lexicon.load(getattr(args, "lexicon", None))
    return run_batch(
        cfg,
        args.manifest,
        out_dir=args.out_dir,
        jobs_count=args.jobs,
        resume=args.resume,
        retry=args.retry,
        fail_fast=args.fail_fast,
        watch=args.watch,
        model_override=getattr(args, "model", None),
        lexicon=lexicon,
        backend=getattr(args, "backend", None),
        backend_model=(
            getattr(args, "backend_model", None)
            or getattr(cfg, "backend_model", None)
        ),
    )


def format_result(payload: dict[str, Any]) -> str:
    if "results" in payload:
        return _format_batch(payload)
    models = ", ".join(payload.get("models") or []) or "none"
    voices = ", ".join(payload.get("voices") or []) or "none"
    return (
        f"planned {payload['jobs']} job(s)\n"
        f"models: {models}\n"
        f"voices: {voices}\n"
        f"plan: {payload['plan']}\n"
        f"status: {payload['status']}"
    )
