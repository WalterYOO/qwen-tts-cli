"""Project files: `qtts project init` and `qtts project lock` (FR-P1-09).

The lock is intentionally deterministic (sorted JSON, no timestamp) and uses
full sha256 hashes for every referenced model file: `catalog.fingerprint` is a
fast resume hint, not a strong identity (t10 pre-review).
"""

from __future__ import annotations

import dataclasses
import importlib.metadata
import json
import platform
import sys
from pathlib import Path
from typing import Any

from . import __version__
from .budget import DEFAULT_MAX_NEW_TOKENS
from .catalog import scan, select_aligner
from .errors import InputError, ModelError
from .plan import SAMPLING_DEFAULTS, resolve_dtype_name
from .subtitles import SubtitlePolicy, normalize_formats, policy_from_args
from .textnorm import NORMALIZATION_VERSION
from .utils import atomic_write_json, atomic_write_text, read_json, sha256_file
from .voice_store import VoiceStore, utc_now

PROJECT_FILENAME = "qtts.project.json"
LOCK_FILENAME = "qtts.lock.json"
PROJECT_SCHEMA_VERSION = 1
LOCK_SCHEMA_VERSION = 1


def _project_config(args, cfg) -> dict[str, Any]:
    policy = policy_from_args(args)
    formats = normalize_formats(getattr(args, "subtitle", None))
    voice = getattr(args, "voice", None) or None
    voices = [voice] if voice else []
    model_dir = getattr(args, "model_dir", None) or cfg.model_dir
    return {
        "schema_version": PROJECT_SCHEMA_VERSION,
        "qtts_version": __version__,
        "created_at": utc_now(),
        "model_dir": str(Path(model_dir).expanduser()) if model_dir else None,
        "voice": voice,
        "voices": voices,
        "language": getattr(args, "language", "auto") or "auto",
        "format": getattr(args, "format", "wav") or "wav",
        "seed": int(cfg.seed),
        "max_new_tokens": int(
            getattr(args, "max_new_tokens", DEFAULT_MAX_NEW_TOKENS) or DEFAULT_MAX_NEW_TOKENS
        ),
        "subtitle": {
            "formats": formats,
            "granularity": getattr(args, "granularity", "sentence") or "sentence",
            "policy": policy.to_dict(),
        },
    }


def init(cfg, args) -> dict[str, Any]:
    directory = Path(getattr(args, "directory", ".")).expanduser()
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise InputError(f"cannot create project directory {directory}: {exc}") from exc
    path = directory / PROJECT_FILENAME
    if path.exists() and not getattr(args, "force", False):
        raise InputError(
            f"project already exists: {path}; pass --force to overwrite",
            details={"project": str(path)},
        )
    config = _project_config(args, cfg)
    atomic_write_json(path, config)
    return {"ok": True, "project": str(path), "config": config}


def _resolve_path(value: str | Path, base: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _package_versions() -> dict[str, dict[str, Any]]:
    packages: dict[str, dict[str, Any]] = {}
    for distribution in ("qwen-tts", "qwen-asr", "transformers", "torch"):
        try:
            version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            packages[distribution] = {"available": False, "version": None}
        else:
            packages[distribution] = {"available": True, "version": version}
    return packages


def _model_record(model) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for path in sorted(model.path.rglob("*")):
        if not path.is_file():
            continue
        relative = str(path.relative_to(model.path))
        files.append(
            {
                "path": relative,
                "sha256": sha256_file(path),
                "size": path.stat().st_size,
            }
        )
    return {
        "id": model.model_id,
        "name": model.name,
        "kind": model.kind,
        "revision": model.revision,
        "fingerprint": model.fingerprint,
        "path": str(model.path),
        "files": files,
    }


def _voice_record(store: VoiceStore, resolution) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    voice_json_sha = None
    voice_json = None
    ref_audio_sha = None
    if resolution.voice_name:
        meta = store.require(resolution.voice_name)
        path = store.dir_for(resolution.voice_name) / "voice.json"
        if path.is_file():
            voice_json = str(path)
            voice_json_sha = sha256_file(path)
        source = meta.get("source") or {}
        declared_ref = isinstance(source, dict) and bool(source.get("ref_audio"))
        ref_path = store.dir_for(resolution.voice_name) / "ref.wav"
        if ref_path.is_file():
            # Identify the bytes actually on disk, independent of whether (or
            # how) voice.json declares them (t10 fix #3/#4).
            ref_audio_sha = sha256_file(ref_path)
        elif declared_ref:
            raise InputError(
                f"voice {resolution.voice_name} declares ref_audio but ref.wav is missing",
                details={"voice": resolution.voice_name, "path": str(ref_path)},
            )
    return {
        "ref": resolution.ref,
        "voice_name": resolution.voice_name,
        "kind": resolution.mode,
        "speaker": resolution.speaker,
        "icl": resolution.icl,
        "model": resolution.model.model_id,
        "voice_json": voice_json,
        "voice_json_sha256": voice_json_sha,
        "prompt_sha256": resolution.prompt_sha256,
        "ref_audio_sha256": ref_audio_sha,
        "consent": meta.get("consent"),
    }


def _load_project(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise InputError(f"project file not found: {path}", details={"project": str(path)})
    try:
        project = read_json(path)
    except (OSError, ValueError) as exc:
        raise InputError(f"cannot read project file {path}: {exc}") from exc
    if not isinstance(project, dict):
        raise InputError(f"project file must contain a JSON object: {path}")
    if project.get("schema_version") != PROJECT_SCHEMA_VERSION:
        raise InputError(
            f"unsupported project schema_version: {project.get('schema_version')!r}"
        )
    return project


def _voice_refs(project: dict[str, Any]) -> list[str]:
    refs: list[str] = []
    for candidate in [project.get("voice"), *(project.get("voices") or [])]:
        if isinstance(candidate, str) and candidate and candidate not in refs:
            refs.append(candidate)
    if not refs:
        raise InputError("project has no voice reference; set 'voice' or 'voices'")
    return refs


def _project_policy(project: dict[str, Any]) -> SubtitlePolicy:
    subtitle = project.get("subtitle") or {}
    raw = subtitle.get("policy") or {}
    defaults = SubtitlePolicy()
    return SubtitlePolicy(
        max_cue_chars=int(raw.get("max_cue_chars", defaults.max_cue_chars)),
        max_line_width=int(raw.get("max_line_width", defaults.max_line_width)),
        max_cps=float(raw.get("max_cps", defaults.max_cps)),
        min_cue_seconds=float(raw.get("min_cue_seconds", defaults.min_cue_seconds)),
        max_cue_seconds=float(raw.get("max_cue_seconds", defaults.max_cue_seconds)),
        max_merge_gap_ms=int(raw.get("max_merge_gap_ms", defaults.max_merge_gap_ms)),
        qc_enabled=bool(raw.get("qc_enabled", defaults.qc_enabled)),
    )


def lock(cfg, args) -> dict[str, Any]:
    from .capabilities import parse_voice_ref
    from .plan import resolve_voice

    project_path = Path(getattr(args, "project", None) or PROJECT_FILENAME).expanduser()
    project = _load_project(project_path)
    base = project_path.parent
    model_dir_value = project.get("model_dir") or cfg.model_dir
    if not model_dir_value:
        raise ModelError("project has no model_dir and no --model-dir was provided")
    model_dir = _resolve_path(model_dir_value, base)
    cfg_for_project = dataclasses.replace(cfg, model_dir=model_dir)

    models = scan(cfg_for_project.model_dir)
    if not models:
        raise ModelError(
            "no local models found for the project",
            details={"model_dir": str(cfg_for_project.model_dir)},
        )
    store = VoiceStore(cfg_for_project.data_dir)

    resolutions = []
    for ref in _voice_refs(project):
        resolutions.append(resolve_voice(models, store, parse_voice_ref(ref)))

    unique_models = {}
    for resolution in resolutions:
        unique_models[str(resolution.model.path)] = resolution.model
    model_records = [_model_record(model) for _, model in sorted(unique_models.items())]

    subtitle = project.get("subtitle") or {}
    formats = normalize_formats(subtitle.get("formats"))
    packages = _package_versions()
    aligner = None
    if formats:
        if not packages["qwen-asr"]["available"]:
            raise ModelError(
                "project requests subtitles but the qwen-asr package is not installed",
                details={"package": "qwen-asr", "hint": "install the 'align' extra"},
            )
        aligner_model = select_aligner(models)
        aligner = _model_record(aligner_model)

    policy = _project_policy(project)
    seed = int(project.get("seed", cfg_for_project.seed))
    max_new_tokens = int(
        project.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS) or DEFAULT_MAX_NEW_TOKENS
    )
    lock_data = {
        "schema_version": LOCK_SCHEMA_VERSION,
        "qtts_version": __version__,
        "runtime": {
            "python": sys.version,
            "python_version": platform.python_version(),
            "packages": packages,
        },
        "project": {
            "path": str(project_path),
            "sha256": sha256_file(project_path),
            "schema_version": project.get("schema_version"),
        },
        "models": model_records,
        "voices": [_voice_record(store, resolution) for resolution in resolutions],
        "aligner": aligner,
        "normalization": {"version": NORMALIZATION_VERSION},
        "sampling": {**SAMPLING_DEFAULTS, "max_new_tokens": max_new_tokens},
        "seed": seed,
        "subtitle_policy": policy.to_dict(),
        "device": cfg_for_project.device,
        "dtype": cfg_for_project.dtype,
        "effective_dtype": resolve_dtype_name(cfg_for_project),
    }

    out_value = getattr(args, "out", None)
    out_path = Path(out_value).expanduser() if out_value else project_path.with_name(LOCK_FILENAME)
    if not out_path.is_absolute():
        out_path = Path.cwd() / out_path
    atomic_write_text(
        out_path,
        json.dumps(lock_data, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return {"ok": True, "project": str(project_path), "lock": str(out_path), "lockfile": lock_data}


def format_result(payload: dict[str, Any]) -> str:
    if "lockfile" in payload:
        lock_data = payload["lockfile"]
        return (
            f"locked project {payload['project']} -> {payload['lock']} "
            f"({len(lock_data['models'])} model(s), {len(lock_data['voices'])} voice(s))"
        )
    if payload.get("config"):
        return f"wrote project {payload['project']} ({payload['config'].get('voice') or 'no voice'})"
    return str(payload)
