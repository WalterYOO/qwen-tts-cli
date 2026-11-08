"""`qtts script lint|compile` (FR-P1-02)."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

from ..backends.registry import resolve_backend, run_preflight
from ..errors import InputError
from ..lexicon import Lexicon
from ..script import compile_script, load_script, script_plan_json
from ..utils import atomic_write_text
from ..voice_store import VoiceStore


def _compile(cfg, args, *, spec=None, data=None):
    # Parse the script before touching the model directory so syntax/schema
    # errors always win over a missing-model error (exit 2 before 3).
    if data is None:
        data = load_script(args.file)
    lexicon = Lexicon.load(getattr(args, "lexicon", None))
    return compile_script(
        cfg,
        args.file,
        models=spec.models if spec is not None else None,
        store=VoiceStore(cfg.data_dir),
        lexicon=lexicon,
        model_override=getattr(args, "model", None),
        script_data=data,
        backend_spec=spec,
        backend_identity=spec.identity if spec is not None else None,
        require_clone_prompt=(
            spec.require_clone_prompt if spec is not None else True
        ),
        speaker_policy=spec.speaker_policy() if spec is not None else "strict",
    )


def lint(cfg, args) -> dict[str, Any]:
    plan = _compile(cfg, args)
    return {
        "ok": True,
        "script": str(Path(args.file).expanduser()),
        "segments": len(plan.segments),
        "voices": sorted({seg.voice.ref for seg in plan.segments if seg.voice}),
        "models": sorted({seg.voice.model.model_id for seg in plan.segments if seg.voice}),
        "warnings": plan.warnings,
    }


def compile(cfg, args) -> dict[str, Any]:
    # Input errors keep priority over backend/model resolution.
    data = load_script(args.file)
    spec = resolve_backend(getattr(args, "backend", None), cfg, args)
    plan = _compile(cfg, args, spec=spec, data=data)
    run_preflight(plan, cfg, args, spec)
    payload = script_plan_json(plan)
    out = Path(args.out).expanduser()
    if str(out) == "-":
        sys.stdout.write(payload)
        sys.stdout.flush()
        return {"ok": True, "raw_stdout": True, "segments": len(plan.segments)}
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise InputError(f"cannot create {out.parent}: {exc}") from exc
    atomic_write_text(out, payload)
    return {
        "ok": True,
        "plan": str(out),
        "segments": len(plan.segments),
        "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    }


def format_result(payload: dict[str, Any]) -> str:
    if payload.get("raw_stdout"):
        return ""
    if "plan" in payload:
        return (
            f"compiled {payload['segments']} segment(s) -> {payload['plan']}\n"
            f"plan sha256: {payload['sha256']}"
        )
    voices = ", ".join(payload.get("voices") or []) or "none"
    models = ", ".join(payload.get("models") or []) or "none"
    lines = [
        f"ok: {payload['segments']} segment(s)",
        f"voices: {voices}",
        f"models: {models}",
    ]
    for warning in payload.get("warnings") or []:
        lines.append(f"warning: {warning}")
    return "\n".join(lines)
