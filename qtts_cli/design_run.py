"""Deterministic design-run matrix, run directory and lock (no model imports)."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any, Optional

from .design_compile import CompiledDesign
from .errors import InputError, UsageError
from .utils import atomic_write_json, atomic_write_text, read_json, sha256_text

RUN_SCHEMA_VERSION = 1
MAX_CANDIDATES = 16
PRIMES_UNDER_100 = [
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
    53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
]
SAMPLING_PROFILES: dict[str, dict[str, Any]] = {
    "default": {
        "do_sample": True,
        "top_k": 50,
        "top_p": 1.0,
        "temperature": 0.9,
        "repetition_penalty": 1.05,
    },
    "stable": {
        "do_sample": True,
        "top_k": 50,
        "top_p": 1.0,
        "temperature": 0.7,
        "repetition_penalty": 1.05,
    },
    "expressive": {
        "do_sample": True,
        "top_k": 50,
        "top_p": 1.0,
        "temperature": 1.1,
        "repetition_penalty": 1.05,
    },
}
DEFAULT_PROFILES = ("default", "stable")


def _seeds_for(rng_seed: int, count: int = 2) -> list[int]:
    rng = random.Random(int(rng_seed))
    return sorted(rng.sample(PRIMES_UNDER_100, count))


def build_matrix(
    document: dict[str, Any],
    *,
    seeds: Optional[list[int]] = None,
    sampling: Optional[list[str]] = None,
    wide: bool = False,
    max_new_tokens: int = 2048,
    rng_seed: int = 0,
    allow_large: bool = False,
) -> dict[str, Any]:
    covers = [str(item) for item in (document.get("covers") or [])]
    if len(covers) > 2:
        raise UsageError(
            f"design has {len(covers)} covers; at most two are supported (use --wide for the second)"
        )
    if wide and len(covers) < 2:
        raise UsageError("--wide requires at least two covers")
    seed_values = list(seeds) if seeds else _seeds_for(rng_seed)
    if any(seed <= 0 for seed in seed_values):
        raise UsageError("seeds must be positive integers")
    if len(set(seed_values)) != len(seed_values):
        raise UsageError("--seeds contains duplicate values")
    profile_names = list(sampling) if sampling else list(DEFAULT_PROFILES)
    unknown = [name for name in profile_names if name not in SAMPLING_PROFILES]
    if unknown:
        raise UsageError(f"unknown sampling profile: {unknown[0]}")
    if len(set(profile_names)) != len(profile_names):
        raise UsageError("--sampling contains duplicate profiles")
    cover_indices = [0, 1] if wide else [0]
    candidates = [
        {
            "take_id": f"s{seed}-c{cover}-{profile}",
            "seed": seed,
            "cover_index": cover,
            "sampling": dict(SAMPLING_PROFILES[profile]),
            "sampling_profile": profile,
            "text": covers[cover] if cover < len(covers) else "",
        }
        for seed in seed_values
        for cover in cover_indices
        for profile in profile_names
    ]
    if len(candidates) > MAX_CANDIDATES and not allow_large:
        raise UsageError(
            f"design matrix has {len(candidates)} candidates; the limit is "
            f"{MAX_CANDIDATES} (pass --yes to allow a larger explicit matrix)"
        )
    return {
        "seeds": seed_values,
        "covers": covers,
        "sampling_profiles": profile_names,
        "sampling_order": list(profile_names),
        "wide": bool(wide),
        "max_new_tokens": int(max_new_tokens),
        "candidates": candidates,
    }


def _lock_handle(handle, path: Path) -> None:
    """Take a non-blocking exclusive lock on ``handle`` for this platform."""
    if sys.platform.startswith("win"):
        import msvcrt

        handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise UsageError(f"design run is locked: {path.parent}") from exc
    else:
        import fcntl

        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise UsageError(f"design run is locked: {path.parent}") from exc


def _unlock_handle(handle) -> None:
    if sys.platform.startswith("win"):
        import msvcrt

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class RunLock:
    """Advisory lock on ``RUN_DIR/.lock`` (fcntl on POSIX, msvcrt on Windows)."""

    def __init__(self, run_dir: str | Path) -> None:
        self.path = Path(run_dir).expanduser() / ".lock"
        self.handle = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = open(self.path, "a+")
        try:
            _lock_handle(self.handle, self.path)
        except BaseException:
            self.handle.close()
            self.handle = None
            raise
        return self

    def __exit__(self, *exc_info) -> None:
        if self.handle is not None:
            try:
                _unlock_handle(self.handle)
            finally:
                self.handle.close()
                self.handle = None


def matrix_signature(matrix: dict[str, Any]) -> str:
    payload = {
        key: matrix.get(key)
        for key in ("seeds", "sampling_profiles", "wide", "max_new_tokens", "covers")
    }
    return sha256_text(json.dumps(payload, sort_keys=True, ensure_ascii=False))


def _candidate_record(candidate: dict[str, Any], max_new_tokens: int) -> dict[str, Any]:
    return {
        "take_id": candidate["take_id"],
        "seed": candidate["seed"],
        "cover_index": candidate["cover_index"],
        "sampling": candidate["sampling"],
        "status": "pending",
        "wav": f"candidates/{candidate['take_id']}.wav",
        "recipe": f"candidates/{candidate['take_id']}.recipe.json",
        "sha256": None,
        "duration_s": None,
        "completion_status": None,
        "error": None,
        "max_new_tokens": int(max_new_tokens),
    }


def check_resume(
    previous: dict[str, Any],
    *,
    compiled: CompiledDesign,
    raw_text: str,
    matrix: dict[str, Any],
    model: Optional[dict[str, Any]] = None,
) -> None:
    """Raise ``UsageError`` unless ``previous`` matches the current inputs."""
    design = previous.get("design") or {}
    source = previous.get("source") or {}
    expected = {
        "name": (design.get("name"), compiled.structured.get("name")),
        "template": (design.get("template"), compiled.template),
        "template_sha256": (design.get("template_sha256"), compiled.template_sha256),
        "compiler_version": (design.get("compiler_version"), compiled.compiler_version),
        "rendered_sha256": (design.get("rendered_sha256"), compiled.rendered_sha256),
        "source_sha256": (source.get("sha256"), sha256_text(raw_text)),
    }
    changed = [key for key, (old, new) in expected.items() if old != new]
    if changed:
        raise UsageError(
            "resume mismatch: " + ", ".join(changed) + " changed"
        )
    if matrix_signature(previous.get("matrix") or {}) != matrix_signature(matrix):
        raise UsageError("resume mismatch: matrix (covers, seeds or sampling) changed")
    if model is not None:
        previous_model = previous.get("model") or {}
        if any(previous_model.get(key) != value for key, value in model.items()):
            raise UsageError("resume mismatch: model fingerprint changed")


def prepare_run(
    out: str | Path,
    *,
    document: dict[str, Any],
    raw_text: str,
    compiled: CompiledDesign,
    matrix: dict[str, Any],
    model: Optional[dict[str, Any]] = None,
    resume: bool = False,
    source_path: Optional[str] = None,
) -> dict[str, Any]:
    """Create or resume ``RUN_DIR`` and return the run manifest."""
    run_dir = Path(out).expanduser()
    manifest_path = run_dir / "run.json"
    nonempty = run_dir.exists() and any(
        child.name != ".lock" for child in run_dir.iterdir()
    )
    if nonempty and not resume:
        if manifest_path.is_file():
            raise UsageError(f"design run already exists: {run_dir}; pass --resume or choose another --out")
        raise UsageError(f"output directory is not empty: {run_dir}")
    if nonempty and resume and not manifest_path.is_file():
        raise UsageError(
            f"output directory is not empty and has no run.json: {run_dir}"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "candidates").mkdir(exist_ok=True)
    (run_dir / "report").mkdir(exist_ok=True)

    signature = matrix_signature(matrix)
    instruction_sha = compiled.rendered_sha256
    if resume and manifest_path.is_file():
        previous = read_json(manifest_path)
        check_resume(
            previous,
            compiled=compiled,
            raw_text=raw_text,
            matrix=matrix,
            model=model,
        )
        return previous

    manifest = {
        "schema_version": RUN_SCHEMA_VERSION,
        "kind": "design_run",
        "run_id": sha256_text(signature + instruction_sha)[:16],
        "created_at": compiled.structured.get("created_at") or _utc_now(),
        "updated_at": _utc_now(),
        "qtts_version": _version(),
        "source": {"path": source_path, "sha256": sha256_text(raw_text)},
        "design": {
            "name": compiled.structured.get("name"),
            "language": compiled.language,
            "template": compiled.template,
            "template_sha256": compiled.template_sha256,
            "schema_version": compiled.structured.get("schema_version", 1),
            "structured": _public_structured(compiled.structured),
            "rendered_instruction": compiled.rendered_instruction,
            "rendered_sha256": compiled.rendered_sha256,
            "compiler_version": compiled.compiler_version,
        },
        "model": model or {},
        "matrix": {
            "seeds": matrix["seeds"],
            "covers": matrix["covers"],
            "sampling_profiles": matrix["sampling_profiles"],
            "sampling_order": matrix["sampling_order"],
            "wide": matrix["wide"],
            "max_new_tokens": matrix["max_new_tokens"],
        },
        "candidates": [
            _candidate_record(candidate, matrix["max_new_tokens"])
            for candidate in matrix["candidates"]
        ],
    }
    atomic_write_json(run_dir / "design.compiled.json", compiled.to_dict())
    atomic_write_text(run_dir / "design.yaml", raw_text)
    atomic_write_json(manifest_path, manifest)
    return manifest


def save_manifest(run_dir: str | Path, manifest: dict[str, Any]) -> None:
    manifest["updated_at"] = _utc_now()
    atomic_write_json(Path(run_dir).expanduser() / "run.json", manifest)


def _public_structured(document: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in document.items()
        if key not in ("extras", "errors", "present_fields")
    }


def _version() -> str:
    from . import __version__

    return __version__


def _utc_now() -> str:
    from .voice_store import utc_now

    return utc_now()
