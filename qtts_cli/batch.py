"""Batch manifest planning and execution (FR-P1-01).

All manifest lines are parsed, validated and compiled before the first model
load. ``plan.json`` is deterministic; ``status.jsonl`` is owned by one scheduler
thread and replaced atomically after every transition.
"""

from __future__ import annotations

import dataclasses
import json
import re
import sys
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

from .catalog import scan
from .errors import CapabilityError, InputError, ModelError, QttsError, UsageError
from .lexicon import Lexicon
from .plan import build_plan
from .script import DuplicateJSONKey, json_loads_strict
from .sidecar import recipe_path
from .subtitles import normalize_formats
from .utils import (
    atomic_write_json,
    atomic_write_text,
    read_json,
    sha256_file,
    sha256_json,
    sha256_text,
)
from .voice_store import VoiceStore

BATCH_SCHEMA_VERSION = 1
_STATUSES = {"pending", "running", "done", "failed", "skipped"}
_MANIFEST_KEYS = {
    "id",
    "text",
    "voice",
    "language",
    "instruct",
    "out",
    "format",
    "subtitle",
    "seed",
    "max_new_tokens",
    "pause_ms",
}
_SAFE_STEM = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_INFERENCE_LOCK = threading.Lock()


def _safe_stem(record_id: str) -> str:
    if _SAFE_STEM.match(record_id) and record_id not in (".", ".."):
        return record_id
    return f"qtts-{sha256_text(record_id)[:16]}"


def _line_error(line: int, code: str, message: str, **details: Any) -> InputError:
    return InputError(
        f"manifest line {line}: {message}",
        details={"line": line, "code": code, **details},
    )


def read_manifest(path: str | Path) -> tuple[list[dict[str, Any]], str]:
    file = Path(path).expanduser()
    if not file.is_file():
        raise InputError(f"manifest not found: {file}", details={"manifest": str(file)})
    try:
        raw_lines = file.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise InputError(f"cannot read manifest {file}: {exc}") from exc
    requests: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    used_stems: dict[str, str] = {}
    for line_number, raw_line in enumerate(raw_lines, 1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            request = json_loads_strict(line)
        except DuplicateJSONKey as exc:
            raise UsageError(
                f"manifest line {line_number}: invalid JSON: {exc}",
                details={"line": line_number, "code": "duplicate_key"},
            ) from exc
        except json.JSONDecodeError as exc:
            raise UsageError(
                f"manifest line {line_number}: invalid JSON: {exc.msg}",
                details={"line": line_number, "column": exc.colno, "code": "invalid_json"},
            ) from exc
        if not isinstance(request, dict):
            raise _line_error(line_number, "invalid_request", "each line must be an object")
        if "takes" in request or "select" in request:
            raise UsageError(
                f"manifest line {line_number}: multi-take selection is not "
                "supported in batch manifests; run `qtts say --takes N "
                "--select ...` instead",
                details={"line": line_number, "code": "unsupported_takes"},
            )
        unknown = set(request) - _MANIFEST_KEYS
        if unknown:
            raise _line_error(
                line_number,
                "unknown_field",
                f"unknown field(s): {', '.join(sorted(unknown))}",
            )
        record_id = request.get("id")
        if not isinstance(record_id, str) or not record_id:
            raise _line_error(line_number, "invalid_request", "id is required")
        if record_id in seen_ids:
            raise _line_error(
                line_number, "duplicate_id", f"duplicate id {record_id!r}", id=record_id
            )
        seen_ids.add(record_id)
        stem = _safe_stem(record_id)
        if stem.casefold() in used_stems:
            raise _line_error(
                line_number,
                "id_filename_conflict",
                f"output filename {stem!r} collides with {used_stems[stem.casefold()]!r}",
                id=record_id,
            )
        used_stems[stem.casefold()] = record_id
        text = request.get("text")
        if not isinstance(text, str) or not text.strip():
            raise _line_error(line_number, "invalid_request", "text is required")
        voice = request.get("voice")
        if not isinstance(voice, str) or not voice:
            raise _line_error(line_number, "invalid_request", "voice is required")
        language = request.get("language", "auto")
        if not isinstance(language, str) or not language:
            raise _line_error(line_number, "invalid_request", "language must be a string")
        instruct = request.get("instruct")
        if instruct is not None and not isinstance(instruct, str):
            raise _line_error(line_number, "invalid_request", "instruct must be string/null")
        seed = request.get("seed")
        if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
            raise _line_error(line_number, "invalid_request", "seed must be an integer")
        max_new_tokens = request.get("max_new_tokens")
        if max_new_tokens is not None and (
            not isinstance(max_new_tokens, int)
            or isinstance(max_new_tokens, bool)
            or max_new_tokens <= 0
        ):
            raise _line_error(
                line_number, "invalid_request", "max_new_tokens must be a positive integer"
            )
        pause_ms = request.get("pause_ms")
        if pause_ms is not None and (
            not isinstance(pause_ms, int) or isinstance(pause_ms, bool) or pause_ms < 0
        ):
            raise _line_error(line_number, "invalid_request", "pause_ms must be >= 0")
        subtitle = request.get("subtitle")
        if subtitle is not None and not isinstance(subtitle, (str, list, tuple)):
            raise _line_error(
                line_number, "invalid_request", "subtitle must be a string or list"
            )
        if isinstance(subtitle, (list, tuple)) and not all(
            isinstance(item, str) for item in subtitle
        ):
            raise _line_error(
                line_number, "invalid_request", "subtitle list items must be strings"
            )
        out_field = request.get("out")
        if out_field == "-":
            raise _line_error(line_number, "invalid_request", "out:'-' is not supported")
        if out_field is not None and not isinstance(out_field, str):
            raise _line_error(line_number, "invalid_request", "out must be a string")
        requested_format = request.get("format")
        if requested_format is not None and requested_format not in ("wav", "flac"):
            raise _line_error(line_number, "invalid_request", "format must be wav or flac")
        requests.append(
            {
                "id": record_id,
                "text": text,
                "voice": voice,
                "language": language,
                "instruct": instruct,
                "out": out_field,
                "format": requested_format,
                "subtitle": subtitle,
                "seed": seed,
                "max_new_tokens": max_new_tokens,
                "pause_ms": pause_ms,
                "line": line_number,
                "stem": stem,
            }
        )
    if not requests:
        raise UsageError("manifest contains no requests")
    return requests, sha256_file(file)


def _canonical_output(path: Path) -> str:
    try:
        return str(path.expanduser().resolve()).casefold()
    except (OSError, RuntimeError) as exc:
        raise InputError(f"cannot resolve output path {path}: {exc}") from exc


def prepare_jobs(
    cfg,
    manifest: str | Path,
    *,
    out_dir: str | Path,
    model_override: Optional[str] = None,
    lexicon: Optional[Lexicon] = None,
    models: Optional[list] = None,
    tts_models: Optional[list] = None,
    store: Optional[VoiceStore] = None,
    backend_spec=None,
    backend_model: Optional[str] = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], list]:
    """Parse, validate and compile every manifest line without loading weights."""
    requests, manifest_sha = read_manifest(manifest)
    # ``models`` keeps the historic local-model list (also used by tests);
    # TTS planning uses the backend's models, while the aligner always uses
    # what was scanned from the local model directory.
    local_models = models if models is not None else scan(cfg.model_dir)
    if tts_models is not None:
        model_list = tts_models
    else:
        model_list = local_models
        if not model_list:
            raise ModelError(
                "no local models found; pass --model-dir DIR or run `qtts models pull`",
                details={"model_dir": str(cfg.model_dir) if cfg.model_dir else None},
            )
    voice_store = store or VoiceStore(cfg.data_dir)
    backend_identity = (
        backend_spec.identity if backend_spec is not None else "transformers"
    )
    if backend_spec is not None:
        from .backends.registry import preflight_voice_ref

        for request in requests:
            preflight_voice_ref(request["voice"], backend_spec)
    out_root = Path(out_dir).expanduser()
    used_paths: dict[str, str] = {}
    jobs: list[dict[str, Any]] = []
    plans: list = []
    for request in requests:
        seed = request["seed"] if request["seed"] is not None else cfg.seed
        max_tokens = request["max_new_tokens"]
        if max_tokens is None:
            from .budget import DEFAULT_MAX_NEW_TOKENS

            max_tokens = DEFAULT_MAX_NEW_TOKENS
        pause_ms = request["pause_ms"] if request["pause_ms"] is not None else 300
        subtitle_formats = normalize_formats(request["subtitle"])
        if request["out"]:
            out_path = Path(request["out"]).expanduser()
            output_format = request["format"] or "wav"
        else:
            output_format = request["format"] or "wav"
            out_path = out_root / f"{request['stem']}.{output_format}"
        if out_path.suffix:
            suffix_format = out_path.suffix.lower().lstrip(".")
            if suffix_format not in ("wav", "flac"):
                raise _line_error(
                    request["line"],
                    "unknown_suffix",
                    f"unknown output suffix for {out_path}; use .wav or .flac",
                    id=request["id"],
                )
            if request["format"] and request["format"] != suffix_format:
                raise _line_error(
                    request["line"],
                    "format_mismatch",
                    f"format {request['format']!r} does not match {out_path.name}",
                    id=request["id"],
                )
            output_format = suffix_format
        canonical_out = _canonical_output(out_path)
        if canonical_out in used_paths:
            raise _line_error(
                request["line"],
                "output_path_conflict",
                f"output path {out_path} is already used by {used_paths[canonical_out]!r}",
                id=request["id"],
            )
        used_paths[canonical_out] = request["id"]
        job_cfg = dataclasses.replace(cfg, seed=int(seed), json=False)
        try:
            plan = build_plan(
                job_cfg,
                text=request["text"],
                voice_ref=request["voice"],
                language=request["language"],
                instruct=request["instruct"],
                max_new_tokens=int(max_tokens),
                pause_ms=int(pause_ms),
                output=str(out_path),
                fmt=output_format,
                subtitle_formats=subtitle_formats,
                models=model_list,
                store=voice_store,
                model_override=model_override,
                lexicon=lexicon,
                backend_identity=backend_identity,
                require_clone_prompt=(
                    backend_spec.require_clone_prompt
                    if backend_spec is not None
                    else True
                ),
                speaker_policy=(
                    backend_spec.speaker_policy()
                    if backend_spec is not None
                    else "strict"
                ),
            )
            if backend_spec is not None:
                from .backends.registry import run_preflight

                run_preflight(
                    plan,
                    job_cfg,
                    # Pure planning never needs credentials; the executing
                    # say.run() re-checks them before create/load.
                    SimpleNamespace(dry_run=True),
                    backend_spec,
                )
        except (InputError, CapabilityError) as exc:
            raise InputError(
                f"manifest line {request['line']} ({request['id']}): {exc.message}",
                details={
                    "line": request["line"],
                    "id": request["id"],
                    "voice": request["voice"],
                    **(exc.details or {}),
                },
            ) from exc
        if subtitle_formats:
            from .commands.say import _prepare_aligner

            _prepare_aligner(local_models, plan)
        voice = plan.voice
        request_payload = {
            "id": request["id"],
            "text": request["text"],
            "voice": request["voice"],
            "language": request["language"],
            "instruct": request["instruct"],
            "seed": int(seed),
            "max_new_tokens": int(max_tokens),
            "pause_ms": int(pause_ms),
            "format": output_format,
            "subtitle": subtitle_formats,
        }
        job_key = "sha256:" + sha256_json(
            {
                "schema_version": BATCH_SCHEMA_VERSION,
                "request": request_payload,
                "voice": voice.to_dict(),
                "model": {
                    "id": voice.model.model_id,
                    "revision": voice.model.revision,
                    "fingerprint": voice.model.fingerprint,
                },
                "lexicon_sha256": plan.lexicon_sha256,
                "output": canonical_out,
                "backend": backend_identity,
            }
        )
        jobs.append(
            {
                **request_payload,
                "output": str(out_path),
                "lexicon_sha256": plan.lexicon_sha256,
                "job_key": job_key,
                "model": voice.model.model_id,
                "voice_ref": voice.ref,
                "backend": (
                    backend_spec.name if backend_spec is not None else "transformers"
                ),
                "backend_model": backend_model,
                "backend_identity": backend_identity,
                "source_index": len(jobs),
                "estimated_duration_s": round(plan.estimated_duration_s, 3),
            }
        )
        plans.append((plan, job_cfg, out_path, output_format, subtitle_formats))
    plan_payload = {
        "schema_version": BATCH_SCHEMA_VERSION,
        "manifest_sha256": manifest_sha,
        "backend": (
            backend_spec.name if backend_spec is not None else "transformers"
        ),
        "backend_model": backend_model,
        "backend_identity": backend_identity,
        "jobs": [
            {key: value for key, value in job.items() if key != "source_index"}
            for job in jobs
        ],
    }
    return jobs, plan_payload, plans


def write_plan(out_dir: Path, plan_payload: dict[str, Any]) -> Path:
    path = out_dir / "plan.json"
    atomic_write_json(path, plan_payload)
    return path


def _status_record(job: dict[str, Any], status: str = "pending") -> dict[str, Any]:
    return {
        "id": job["id"],
        "status": status,
        "job_key": job["job_key"],
        "output": job["output"],
        "attempts": 0,
        "output_sha256": None,
        "last_error": None,
        "transient": None,
    }


def write_status(
    out_dir: Path, jobs: list[dict[str, Any]], state: dict[str, dict[str, Any]]
) -> Path:
    path = out_dir / "status.jsonl"
    lines = [
        json.dumps(state[job["id"]], ensure_ascii=False, sort_keys=True) for job in jobs
    ]
    atomic_write_text(path, "\n".join(lines) + "\n")
    return path


def _load_status(out_dir: Path) -> Optional[dict[str, dict[str, Any]]]:
    path = out_dir / "status.jsonl"
    if not path.is_file():
        return None
    result: dict[str, dict[str, Any]] = {}
    try:
        for raw in path.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            item = json.loads(raw)
            if isinstance(item, dict) and isinstance(item.get("id"), str):
                if "last_error" not in item:
                    item["last_error"] = item.pop("error", None)
                item.setdefault("transient", None)
                if item.get("status") not in _STATUSES:
                    item["status"] = "pending"
                result[item["id"]] = item
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return result


def _resume_state(job: dict[str, Any], existing: dict[str, Any]) -> str:
    status = existing.get("status")
    if status in (None, "pending", "running", "skipped"):
        return "pending"
    if status == "failed":
        return "failed"
    if status != "done":
        return "pending"
    if existing.get("job_key") != job["job_key"]:
        return "pending"
    output = Path(str(existing.get("output") or job["output"]))
    if not output.is_file():
        return "pending"
    expected = existing.get("output_sha256")
    if not expected or sha256_file(output) != expected:
        return "pending"
    recipe = recipe_path(output)
    if not recipe.is_file():
        return "pending"
    try:
        recipe_payload = read_json(recipe)
    except (OSError, ValueError):
        return "pending"
    if recipe_payload.get("job_key") != job["job_key"]:
        return "pending"
    return "done"


def _record_args(
    job: dict[str, Any],
    cfg,
    model_override,
    lexicon,
    backend: Optional[str] = None,
    backend_model: Optional[str] = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        backend=backend if backend is not None else job.get("backend"),
        backend_model=(
            backend_model if backend_model is not None else job.get("backend_model")
        ),
        text=job["text"],
        stdin=False,
        file=None,
        script=None,
        voice=job["voice"],
        language=job["language"],
        instruct=job["instruct"],
        out=job["output"],
        format=job["format"],
        subtitle=job["subtitle"] or None,
        subtitle_out_prefix=None,
        recipe_out=None,
        granularity=getattr(cfg, "granularity", "sentence"),
        max_new_tokens=job["max_new_tokens"],
        pause_ms=job["pause_ms"],
        model=model_override,
        lexicon=lexicon,
        dry_run=False,
        yes=True,
        max_cue_chars=getattr(cfg, "max_cue_chars", 42),
        max_line_width=getattr(cfg, "max_line_width", 42),
        max_cps=getattr(cfg, "max_cps", 20.0),
        min_cue_seconds=getattr(cfg, "min_cue_seconds", 0.8),
        max_cue_seconds=getattr(cfg, "max_cue_seconds", 7.0),
        max_merge_gap_ms=getattr(cfg, "max_merge_gap_ms", 300),
        no_subtitle_qc=getattr(cfg, "no_subtitle_qc", False),
    )


def _is_transient(exc: BaseException) -> bool:
    if isinstance(exc, ModelError):
        return True
    if isinstance(exc, QttsError):
        return False
    return isinstance(exc, (OSError, RuntimeError))


def _execute_job(
    job: dict[str, Any],
    plan_bundle,
    cfg,
    *,
    retry: int,
    model_override,
    lexicon,
    engine_cache: dict,
    aligner_cache: dict,
    backend: Optional[str] = None,
    backend_model: Optional[str] = None,
) -> dict[str, Any]:
    from .commands.say import run as say_run

    _plan, job_cfg, _out_path, _fmt, _subs = plan_bundle
    args = _record_args(
        job,
        cfg,
        model_override,
        lexicon,
        backend=backend,
        backend_model=backend_model,
    )
    attempts = 0
    last_error: Optional[dict[str, Any]] = None
    while attempts <= retry:
        attempts += 1
        try:
            with _INFERENCE_LOCK:
                say_run(
                    job_cfg,
                    args,
                    engine_cache=engine_cache,
                    aligner_cache=aligner_cache,
                )
            break
        except Exception as exc:
            # Only transient execution failures (model load/IO/runtime) may be
            # retried; a ValueError or other programming error fails at once.
            transient = _is_transient(exc)
            if isinstance(exc, QttsError):
                last_error = {"code": exc.code, "kind": exc.kind, "message": exc.message}
            else:
                last_error = {"code": None, "kind": "internal", "message": str(exc)}
            if not transient or attempts > retry:
                return {
                    "ok": False,
                    "attempts": attempts,
                    "error": last_error,
                    "transient": transient,
                }
    output = Path(job["output"])
    if not output.is_file():
        return {
            "ok": False,
            "attempts": attempts,
            "transient": False,
            "error": {
                "code": None,
                "kind": "internal",
                "message": f"executor reported success but {output} is missing",
            },
        }
    recipe = recipe_path(output)
    if recipe.is_file():
        try:
            recipe_payload = read_json(recipe)
        except (OSError, ValueError):
            recipe_payload = {}
        if isinstance(recipe_payload, dict):
            recipe_payload["job_key"] = job["job_key"]
            atomic_write_json(recipe, recipe_payload)
    return {"ok": True, "attempts": attempts, "output_sha256": sha256_file(output)}


def run_batch(
    cfg,
    manifest: str | Path,
    *,
    out_dir: str | Path,
    jobs_count: int = 1,
    resume: bool = False,
    retry: int = 0,
    fail_fast: bool = False,
    watch: bool = False,
    model_override: Optional[str] = None,
    lexicon: Optional[Lexicon] = None,
    backend: Optional[str] = None,
    backend_model: Optional[str] = None,
) -> dict[str, Any]:
    if jobs_count < 1:
        raise UsageError("--jobs must be >= 1")
    if retry < 0:
        raise UsageError("--retry must be >= 0")
    out_root = Path(out_dir).expanduser()
    from .backends.registry import resolve_backend

    backend_args = SimpleNamespace(
        backend=backend,
        backend_model=backend_model,
        model=model_override,
        dry_run=False,
    )
    spec = resolve_backend(backend, cfg, backend_args)
    effective_backend_model = backend_model or getattr(cfg, "backend_model", None)
    # A resume must not silently mix backends: check the existing plan before
    # any mkdir/write so a mismatch leaves the directory byte-identical.
    if out_root.is_dir():
        previous_plan_path = out_root / "plan.json"
        if previous_plan_path.is_file():
            try:
                previous_plan = read_json(previous_plan_path)
            except (OSError, ValueError):
                previous_plan = None
            if isinstance(previous_plan, dict):
                previous_identity = (
                    previous_plan.get("backend_identity") or "transformers"
                )
                if previous_identity != spec.identity:
                    raise UsageError(
                        f"batch plan at {previous_plan_path} was created for backend "
                        f"{previous_identity!r}, but this run uses {spec.identity!r}",
                        details={
                            "plan_backend": previous_identity,
                            "backend": spec.identity,
                        },
                    )
    # `batch plan` is metadata-only, but `batch run` must fail before any
    # scheduling/write when the remote backend has no credentials.
    if spec.configured is not None and not spec.configured():
        raise ModelError(
            f"backend {spec.name} is not configured; set its credentials or "
            "choose another backend",
            details={"backend": spec.name},
        )
    try:
        out_root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise InputError(f"cannot create --out-dir {out_root}: {exc}") from exc
    jobs, plan_payload, plan_bundles = prepare_jobs(
        cfg,
        manifest,
        out_dir=out_root,
        model_override=model_override,
        lexicon=lexicon,
        tts_models=spec.models,
        backend_spec=spec,
        backend_model=effective_backend_model,
    )
    plan_path = write_plan(out_root, plan_payload)
    existing = _load_status(out_root) if resume else None
    state: dict[str, dict[str, Any]] = {}
    for job in jobs:
        record = _status_record(job)
        if existing is not None and job["id"] in existing:
            previous = existing[job["id"]]
            status = _resume_state(job, previous)
            record["attempts"] = int(previous.get("attempts") or 0)
            record["output_sha256"] = previous.get("output_sha256")
            record["last_error"] = previous.get("last_error")
            record["transient"] = previous.get("transient")
            if status == "failed" and retry > 0 and previous.get("transient") is True:
                status = "pending"
            record["status"] = status
        state[job["id"]] = record
    status_path = write_status(out_root, jobs, state)

    def note(job_id: str, status: str) -> None:
        state[job_id]["status"] = status
        write_status(out_root, jobs, state)
        if watch:
            print(f"[batch] {job_id}: {status}", file=sys.stderr, flush=True)

    bundles = {job["id"]: bundle for job, bundle in zip(jobs, plan_bundles)}
    order = sorted(
        jobs,
        key=lambda job: (
            job["model"],
            job["voice_ref"],
            job["language"],
            job["estimated_duration_s"],
            job["source_index"],
        ),
    )
    engine_cache: dict = {}
    aligner_cache: dict = {}
    failed_any = False
    stop_scheduling = False
    pending_jobs = [job for job in order if state[job["id"]]["status"] == "pending"]

    futures: dict[Any, dict[str, Any]] = {}
    index = 0

    def launch(job: dict[str, Any]) -> None:
        note(job["id"], "running")

    with ThreadPoolExecutor(max_workers=jobs_count) as executor:
        while True:
            while (
                index < len(pending_jobs)
                and len(futures) < jobs_count
                and not stop_scheduling
            ):
                job = pending_jobs[index]
                index += 1
                launch(job)
                future = executor.submit(
                    _execute_job,
                    job,
                    bundles[job["id"]],
                    dataclasses.replace(cfg, json=False),
                    retry=retry,
                    model_override=model_override,
                    lexicon=lexicon,
                    engine_cache=engine_cache,
                    aligner_cache=aligner_cache,
                    backend=spec.name,
                    backend_model=effective_backend_model,
                )
                futures[future] = job
            if not futures:
                break
            done, _ = wait(list(futures), return_when=FIRST_COMPLETED)
            for future in done:
                job = futures.pop(future)
                try:
                    result = future.result()
                except Exception as exc:  # scheduler boundary
                    result = {
                        "ok": False,
                        "attempts": 1,
                        "transient": False,
                        "error": {"code": None, "kind": "internal", "message": str(exc)},
                    }
                record = state[job["id"]]
                record["attempts"] = result.get("attempts", record["attempts"])
                if result["ok"]:
                    record["status"] = "done"
                    record["output_sha256"] = result.get("output_sha256")
                    record["last_error"] = None
                    record["transient"] = None
                else:
                    record["status"] = "failed"
                    record["last_error"] = result.get("error")
                    record["transient"] = bool(result.get("transient"))
                    failed_any = True
                    if fail_fast:
                        stop_scheduling = True
                write_status(out_root, jobs, state)
                if watch:
                    print(
                        f"[batch] {job['id']}: {record['status']}",
                        file=sys.stderr,
                        flush=True,
                    )
    if stop_scheduling:
        for job in pending_jobs[index:]:
            if state[job["id"]]["status"] == "pending":
                state[job["id"]]["status"] = "skipped"
        write_status(out_root, jobs, state)

    counts = {status: 0 for status in _STATUSES}
    results = []
    for job in jobs:
        record = state[job["id"]]
        counts[record["status"]] = counts.get(record["status"], 0) + 1
        results.append(
            {
                "id": job["id"],
                "status": record["status"],
                "attempts": record["attempts"],
                "output": job["output"] if record["status"] == "done" else None,
                "last_error": record["last_error"],
                "transient": record["transient"],
            }
        )
    failed_any = failed_any or counts["failed"] > 0
    return {
        "ok": not failed_any,
        "exit_code": 0 if not failed_any else 6,
        "jobs": len(jobs),
        "done": counts["done"],
        "failed": counts["failed"],
        "skipped": counts["skipped"],
        "pending": counts["pending"],
        "running": counts["running"],
        "plan": str(plan_path),
        "status": str(status_path),
        "results": results,
    }


def format_result(payload: dict[str, Any]) -> str:
    lines = [
        f"batch: {payload['jobs']} job(s) "
        f"(done={payload['done']} failed={payload['failed']} "
        f"skipped={payload['skipped']} pending={payload['pending']})",
        f"plan: {payload.get('plan')}",
        f"status: {payload.get('status')}",
    ]
    for result in payload.get("results") or []:
        if result["status"] != "done":
            lines.append(
                f"  {result['id']}: {result['status']} "
                f"(attempts={result['attempts']})"
            )
    return "\n".join(lines)
