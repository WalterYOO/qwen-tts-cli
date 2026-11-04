"""Multi-take asset library (FR-P1-04).

This module is pure Python: it owns take numbering, canonical request/job-id
derivation, the on-disk job layout, lock/staging publication and selection
ranking. Importing it never imports torch, qwen_tts or qwen_asr, so dry-run
and the ``takes`` metadata commands stay model-free.
"""

from __future__ import annotations

import contextlib
import fcntl
import math
import os
import platform
import re
import secrets
import shutil
import tempfile
import unicodedata
from pathlib import Path
from typing import Any, Iterator, Optional

from .errors import InputError, UsageError
from .plan import GenerationPlan
from .utils import atomic_write_json, read_json, sha256_json, sha256_text

TAKES_MIN = 2
TAKES_MAX = 16
CRITERIA = ("cer", "speaker-sim", "lufs")
LUFS_TARGET = -16.0
JOB_SCHEMA_VERSION = 1
SPEAKER_SIM_VERSION = "speaker-sim.v1"
JOB_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
_TAKE_FILE_RE = re.compile(r"^take-(\d{2})\.(wav|flac)$")


# --------------------------------------------------------------------------- #
# CLI value parsing
# --------------------------------------------------------------------------- #
def parse_take_count(value: Any) -> int:
    if isinstance(value, bool):
        raise UsageError("--takes must be an integer")
    try:
        count = int(value)
    except (TypeError, ValueError) as exc:
        raise UsageError("--takes must be an integer") from exc
    if str(value).strip() != str(count):
        raise UsageError("--takes must be an integer")
    if count < TAKES_MIN or count > TAKES_MAX:
        raise UsageError(
            f"--takes must be between {TAKES_MIN} and {TAKES_MAX}",
            details={"takes": count, "min": TAKES_MIN, "max": TAKES_MAX},
        )
    return count


def parse_criteria(value: Any) -> list[str]:
    raw = "" if value is None else str(value)
    parts = [part.strip() for part in raw.split(",")]
    if not raw.strip() or any(not part for part in parts):
        raise UsageError("--select must be a non-empty comma-separated list")
    seen: set[str] = set()
    criteria: list[str] = []
    for part in parts:
        if part not in CRITERIA:
            raise UsageError(
                f"unknown selection criterion {part!r}",
                details={"criterion": part, "supported": list(CRITERIA)},
            )
        if part in seen:
            raise UsageError(f"duplicate selection criterion {part!r}")
        seen.add(part)
        criteria.append(part)
    return criteria


def takes_request(args) -> Optional[tuple[int, list[str]]]:
    """Validate ``--takes``/``--select`` as a pair, or return None."""
    takes = getattr(args, "takes", None)
    select = getattr(args, "select", None)
    if takes is None and select is None:
        return None
    if takes is None:
        raise UsageError("--select requires --takes")
    if select is None:
        raise UsageError("--takes requires --select")
    return parse_take_count(takes), parse_criteria(select)


# --------------------------------------------------------------------------- #
# Canonical request / job id
# --------------------------------------------------------------------------- #
def model_identity(model: Any) -> Optional[dict[str, Any]]:
    if model is None:
        return None
    return {
        "id": getattr(model, "model_id", None),
        "revision": getattr(model, "revision", None),
        "fingerprint": getattr(model, "fingerprint", None),
        "kind": getattr(model, "kind", None),
    }


def reference_identity(voice_store, name: str) -> dict[str, Any]:
    """Describe the speaker-sim reference without importing any runtime."""
    meta = voice_store.require(name)
    prompt_sha = voice_store.prompt_sha256(name)
    source = meta.get("source") or {}
    return {
        "name": name,
        "kind": meta.get("kind"),
        "prompt_sha256": prompt_sha,
        "ref_audio_sha256": source.get("ref_audio_sha256"),
        # Speaker scoring uses the prompt embedding when available, otherwise
        # it falls back to the reference wav.
        "embedding_source": "prompt" if prompt_sha is not None else "ref_wav",
        "scoring_version": SPEAKER_SIM_VERSION,
    }


def runtime_identity(cfg, plan: GenerationPlan) -> dict[str, Any]:
    """Predict the runtime without importing torch.

    ``dry_run_details()`` and the real run both call this, so the same request
    produces the same job_id. If loading resolves a different dtype, generation
    rekeys the plan/cache as it already does and the recipe records the actual
    runtime.
    """
    device_request = getattr(cfg, "device", "auto")
    if device_request == "auto":
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            device = "mps"
        elif shutil.which("nvidia-smi"):
            device = "cuda:0"
        else:
            device = "cpu"
    else:
        device = device_request
    requested_dtype = getattr(cfg, "dtype", "auto")
    if requested_dtype != "auto":
        dtype = plan.dtype_name
    elif device.startswith("cuda") or device == "mps":
        dtype = "bf16"
    else:
        dtype = "fp32"
    return {
        "device": device,
        "dtype": dtype,
        "device_request": device_request,
        "platform": platform.system(),
        "machine": platform.machine(),
    }


def canonical_request(
    plan: GenerationPlan,
    cfg,
    *,
    takes: int,
    criteria: list[str],
    scorer_models: dict[str, Any],
    lufs_target: float = LUFS_TARGET,
    fmt: Optional[str] = None,
    reference: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Build the request that defines job identity.

    Only generation inputs participate: output paths, recipes, QC, subtitles,
    cache hits/timings, timestamps and the absolute script path are excluded.
    """
    models: list[dict[str, Any]] = []
    seen_models: set[tuple[Any, ...]] = set()
    segments: list[dict[str, Any]] = []
    for seg in plan.segments:
        voice = seg.voice or plan.voice
        identity = model_identity(voice.model)
        key = (
            identity["id"] if identity else None,
            identity["revision"] if identity else None,
            identity["fingerprint"] if identity else None,
        )
        if key not in seen_models:
            seen_models.add(key)
            models.append(identity)
        segments.append(
            {
                "id": seg.id,
                "text_sha256": sha256_text(seg.text),
                "display_sha256": sha256_text(seg.display_text or seg.text),
                "language": seg.language,
                "instruct": seg.instruct,
                "pause_after_ms": int(seg.pause_after_ms),
                "voice": voice.identity,
                "seed": int(seg.seed if seg.seed is not None else plan.seed),
                "max_new_tokens": seg.max_new_tokens,
            }
        )
    seeds = [int(plan.seed) + delta for delta in range(takes)]
    return {
        "schema_version": JOB_SCHEMA_VERSION,
        "feature_version": "fr-p1-04.1",
        "normalization_version": plan.normalization_version,
        "text_sha256": sha256_text(plan.normalized_text or ""),
        "display_sha256": sha256_text(plan.display_text or plan.normalized_text or ""),
        "lexicon_sha256": plan.lexicon_sha256,
        "voice": plan.voice.identity,
        "language": plan.language,
        "instruct": plan.instruct,
        "sampling": dict(plan.sampling),
        "codec_rate": plan.codec_rate,
        "max_new_tokens": plan.sampling.get("max_new_tokens"),
        "leading_pause_ms": int(plan.leading_pause_ms),
        "trailing_pause_ms": int(plan.trailing_pause_ms),
        "format": fmt or plan.format,
        "runtime": runtime_identity(cfg, plan),
        "segments": segments,
        "models": models,
        "takes": int(takes),
        "seeds": seeds,
        "selection": {
            "criteria": list(criteria),
            "lufs_target": float(lufs_target),
            # QC-only references are downstream and do not define job identity.
            "reference": reference if "speaker-sim" in criteria else None,
        },
        "scorer_models": {
            "asr": model_identity(scorer_models.get("asr")),
            "base": model_identity(scorer_models.get("base")),
        },
    }


def request_id(request: dict[str, Any]) -> str:
    return "tk_" + sha256_json(request)[:16]


# --------------------------------------------------------------------------- #
# Path safety
# --------------------------------------------------------------------------- #
def collision_key(path: str | os.PathLike[str]) -> str:
    """Canonical key for macOS case/NFC-insensitive collision checks."""
    resolved = str(Path(path).expanduser().resolve(strict=False))
    key = os.path.normcase(resolved)
    if platform.system() == "Darwin":  # macOS default APFS is case-insensitive
        key = key.casefold()
    return unicodedata.normalize("NFC", key)


def validate_job_id(job_id: str) -> str:
    if not isinstance(job_id, str) or not JOB_ID_RE.fullmatch(job_id):
        raise UsageError(
            "JOB_ID must match [A-Za-z0-9._-]{1,64}",
            details={"job_id": str(job_id)},
        )
    if job_id in (".", ".."):
        raise UsageError("JOB_ID may not be '.' or '..'", details={"job_id": job_id})
    return job_id


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _strip_sha(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    return value.split(":", 1)[1] if value.startswith("sha256:") else value


class CorruptJob(Exception):
    """Raised internally when a job directory cannot be validated."""


class TakesStore:
    def __init__(self, data_dir: str | os.PathLike[str]) -> None:
        self.data_dir = Path(data_dir).expanduser()
        self.root = self.data_dir / "takes"
        self.locks = self.root / ".locks"

    # -- directories -------------------------------------------------------- #
    def ensure(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.locks.mkdir(parents=True, exist_ok=True)

    def _real_root(self) -> Path:
        return self.root.resolve()

    def _assert_under_root(self, path: Path, *, allow_missing: bool = True) -> None:
        try:
            parent = path.resolve(strict=not allow_missing).parent
        except OSError as exc:
            raise InputError(f"cannot resolve takes path {path}: {exc}") from exc
        if parent != self._real_root():
            raise UsageError(
                f"takes path escapes the takes root: {path}",
                details={"path": str(path), "root": str(self.root)},
            )

    def job_dir(self, job_id: str) -> Path:
        validate_job_id(job_id)
        candidate = self.root / job_id
        if candidate.is_symlink():
            raise UsageError(
                f"refusing symlinked takes job directory: {candidate}",
                details={"path": str(candidate)},
            )
        if candidate.exists():
            self._assert_under_root(candidate, allow_missing=False)
        return candidate

    def job_json(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "job.json"

    # -- locking ------------------------------------------------------------ #
    @contextlib.contextmanager
    def lock(self, job_id: str) -> Iterator[None]:
        validate_job_id(job_id)
        self.ensure()
        path = self.locks / f"{job_id}.lock"
        if path.is_symlink():
            raise InputError(f"refusing symlinked takes lock: {path}")
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            fd = os.open(str(path), flags, 0o600)
        except OSError as exc:
            raise InputError(
                f"cannot open takes lock {path}: {exc}", details={"path": str(path)}
            ) from exc
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            with contextlib.suppress(OSError):
                fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    # -- staging / publication --------------------------------------------- #
    def new_staging(self, job_id: str) -> Path:
        validate_job_id(job_id)
        self.ensure()
        path = Path(
            tempfile.mkdtemp(prefix=f".staging-{job_id}-", dir=str(self.root))
        )
        self._assert_under_root(path, allow_missing=False)
        return path

    def publish(self, staging: str | os.PathLike[str], job_id: str) -> Path:
        """Publish a fully-written staging directory when the target is absent."""
        target = self.job_dir(job_id)
        staging_path = Path(staging)
        self._assert_under_root(staging_path, allow_missing=False)
        if target.exists() or target.is_symlink():
            raise InputError(
                f"takes job already exists: {target}", details={"job": job_id}
            )
        _fsync_dir(staging_path)
        os.rename(staging_path, target)
        _fsync_dir(self.root)
        return target

    def quarantine(self, job_id: str) -> Path:
        """Move a damaged job directory aside under the takes root."""
        target = self.job_dir(job_id)
        if not target.exists():
            raise InputError(f"takes job not found: {target}", details={"job": job_id})
        backup = self.root / f".damaged-{job_id}-{secrets.token_hex(4)}"
        self._assert_under_root(target, allow_missing=False)
        os.rename(target, backup)
        _fsync_dir(self.root)
        return backup

    def restore(self, backup: str | os.PathLike[str], job_id: str) -> None:
        target = self.job_dir(job_id)
        backup_path = Path(backup)
        self._assert_under_root(backup_path, allow_missing=False)
        if target.exists() or target.is_symlink():
            raise InputError(
                f"cannot restore damaged job over {target}", details={"job": job_id}
            )
        os.rename(backup_path, target)
        _fsync_dir(self.root)

    def cleanup(self, path: str | os.PathLike[str]) -> None:
        import shutil

        candidate = Path(path)
        if not candidate.exists():
            return
        self._assert_under_root(candidate, allow_missing=False)
        shutil.rmtree(candidate)

    # -- job io ------------------------------------------------------------- #
    def read_job(self, job_id: str) -> dict[str, Any]:
        validate_job_id(job_id)
        target = self.job_json(job_id)
        if target.is_symlink():
            raise CorruptJob(f"job.json for {job_id} is a symlink")
        if not target.is_file():
            raise InputError(
                f"takes job not found: {self.root / job_id}",
                details={"job": job_id, "path": str(self.root / job_id)},
            )
        try:
            job = read_json(target)
        except (OSError, ValueError) as exc:
            raise CorruptJob(f"cannot read job.json for {job_id}: {exc}") from exc
        validate_job(job, expected_id=job_id)
        return job

    def load_job(self, job_id: str) -> dict[str, Any]:
        try:
            return self.read_job(job_id)
        except CorruptJob as exc:
            raise InputError(str(exc), details={"job": job_id}) from exc

    def write_job_json(self, directory: str | os.PathLike[str], job: dict[str, Any]) -> Path:
        validate_job(job, expected_id=job.get("job_id"))
        path = Path(directory) / "job.json"
        atomic_write_json(path, job)
        return path

    def update_job(self, job_id: str, job: dict[str, Any]) -> None:
        validate_job(job, expected_id=job_id)
        atomic_write_json(self.job_json(job_id), job)

    # -- integrity ---------------------------------------------------------- #
    def verify_takes(
        self,
        job: dict[str, Any],
        *,
        job_dir: Optional[Path] = None,
        take_ids: Optional[set[int]] = None,
        recipes: bool = True,
    ) -> None:
        directory = job_dir or self.job_dir(job["job_id"])
        for take in job["takes"]:
            if take_ids is not None and int(take["id"]) not in take_ids:
                continue
            self.verify_take(job, take, directory, recipes=recipes)

    def verify_take(
        self,
        job: dict[str, Any],
        take: dict[str, Any],
        directory: Path,
        *,
        recipes: bool = True,
    ) -> None:
        from .utils import sha256_file

        audio = directory / take["file"]
        recipe = directory / take["recipe"]
        if audio.is_symlink() or recipe.is_symlink():
            raise InputError(
                f"take {take['id']} asset is a symlink: refusing to trust it",
                details={"job": job.get("job_id"), "take": take.get("id")},
            )
        if not audio.is_file():
            raise InputError(
                f"take {take['id']} audio is missing: {audio}",
                details={"job": job.get("job_id"), "take": take.get("id")},
            )
        if not recipe.is_file():
            raise InputError(
                f"take {take['id']} recipe is missing: {recipe}",
                details={"job": job.get("job_id"), "take": take.get("id")},
            )
        expected = _strip_sha(take.get("audio_sha256"))
        actual = sha256_file(audio)
        if not expected or actual != expected:
            raise InputError(
                f"take {take['id']} audio hash mismatch",
                details={
                    "job": job.get("job_id"),
                    "take": take.get("id"),
                    "expected": expected,
                    "actual": actual,
                },
            )
        if recipes:
            self.verify_recipe(job, take, directory)

    def verify_recipe(
        self, job: dict[str, Any], take: dict[str, Any], directory: Path
    ) -> dict[str, Any]:
        """Parse and validate a take recipe; raise InputError (exit 4) on damage."""
        path = directory / take["recipe"]
        details = {"job": job.get("job_id"), "take": take.get("id"), "path": str(path)}
        if path.is_symlink() or not path.is_file():
            raise InputError(
                f"take {take['id']} recipe is missing: {path}", details=details
            )
        try:
            recipe = read_json(path)
        except (OSError, ValueError) as exc:
            raise InputError(
                f"take {take['id']} recipe is unreadable: {exc}", details=details
            ) from exc
        if not isinstance(recipe, dict):
            raise InputError(f"take {take['id']} recipe is not an object", details=details)
        take_id = recipe.get("take_id")
        if (
            isinstance(take_id, bool)
            or not isinstance(take_id, int)
            or take_id != take["id"]
            or recipe.get("job_id") != job.get("job_id")
        ):
            raise InputError("take recipe identity mismatch", details=details)
        audio = recipe.get("audio")
        if not isinstance(audio, dict) or audio.get("file") != take["file"]:
            raise InputError("take recipe audio file mismatch", details=details)
        expected = _strip_sha(take.get("audio_sha256"))
        actual = _strip_sha(audio.get("sha256"))
        if not expected or actual != expected:
            raise InputError("take recipe audio hash mismatch", details=details)
        return recipe


_BASENAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _basename(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _BASENAME_RE.fullmatch(value) or value in (".", ".."):
        raise CorruptJob(f"invalid {field}: {value!r}")
    return value


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_finite(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))


def _is_js_id(value: Any) -> bool:
    return isinstance(value, str) or _is_int(value)


def _validate_request_segments(request: dict[str, Any]) -> list[dict[str, Any]]:
    segments = request.get("segments")
    if not isinstance(segments, list) or not segments:
        raise CorruptJob("request segments must be a non-empty list")
    seen: set[str] = set()
    for segment in segments:
        if not isinstance(segment, dict):
            raise CorruptJob("each request segment must be an object")
        segment_id = segment.get("id")
        if not _is_js_id(segment_id):
            raise CorruptJob(f"invalid request segment id: {segment_id!r}")
        key = str(segment_id)
        if key in seen:
            raise CorruptJob(f"duplicate request segment id: {segment_id!r}")
        seen.add(key)
        for field in ("text_sha256", "display_sha256"):
            if not isinstance(segment.get(field), str):
                raise CorruptJob(f"request segment {segment_id} has no {field}")
        language = segment.get("language")
        if language is not None and not isinstance(language, str):
            raise CorruptJob(f"request segment {segment_id} has invalid language")
        pause = segment.get("pause_after_ms")
        if not _is_int(pause) or pause < 0:
            raise CorruptJob(f"request segment {segment_id} has invalid pause")
        seed = segment.get("seed")
        if not _is_int(seed) or not 0 <= seed <= 2**31 - 1:
            raise CorruptJob(f"request segment {segment_id} has invalid seed")
    return segments


def _validate_take_segments(
    take: dict[str, Any],
    request_segments: list[dict[str, Any]],
    job: dict[str, Any],
) -> None:
    take_id = int(take["id"])
    delta = take_id - 1
    segments = take["segments"]
    if len(segments) != len(request_segments):
        raise CorruptJob(f"take {take_id} segment count does not match the request")
    previous_end: Optional[float] = None
    for index, (segment, requested) in enumerate(zip(segments, request_segments)):
        if not isinstance(segment, dict):
            raise CorruptJob(f"take {take_id} segment {index} must be an object")
        if segment.get("id") != requested.get("id"):
            raise CorruptJob(f"take {take_id} segment {index} id mismatch")
        for field in ("text_sha256", "display_sha256", "language", "pause_after_ms"):
            if segment.get(field) != requested.get(field):
                raise CorruptJob(
                    f"take {take_id} segment {index} {field} does not match the request"
                )
        duration = segment.get("duration_s")
        start = segment.get("start_s")
        end = segment.get("end_s")
        if not (_is_finite(duration) and _is_finite(start) and _is_finite(end)):
            raise CorruptJob(f"take {take_id} segment {index} has invalid timing")
        duration_f = float(duration)
        start_f = float(start)
        end_f = float(end)
        if not 0.0 <= start_f < end_f:
            raise CorruptJob(f"take {take_id} segment {index} has invalid bounds")
        if end_f > float(take["duration_s"]) + 1e-3:
            raise CorruptJob(f"take {take_id} segment {index} exceeds the take duration")
        if abs((end_f - start_f) - duration_f) > 1e-3:
            raise CorruptJob(f"take {take_id} segment {index} duration is inconsistent")
        if previous_end is not None and start_f < previous_end - 1e-6:
            raise CorruptJob(f"take {take_id} segment {index} overlaps the previous segment")
        previous_end = end_f
        seed = segment.get("seed")
        expected_seed = int(requested["seed"]) + delta
        if (
            not _is_int(seed)
            or seed != expected_seed
            or not 0 <= seed <= 2**31 - 1
        ):
            raise CorruptJob(f"take {take_id} segment {index} has an invalid seed")


def validate_job(job: Any, *, expected_id: Optional[str] = None) -> None:
    if not isinstance(job, dict):
        raise CorruptJob("job.json must be an object")
    if job.get("schema_version") != JOB_SCHEMA_VERSION:
        raise CorruptJob(f"unsupported job schema_version: {job.get('schema_version')!r}")
    job_id = job.get("job_id")
    if not isinstance(job_id, str) or not JOB_ID_RE.fullmatch(job_id):
        raise CorruptJob(f"invalid job_id: {job_id!r}")
    if expected_id is not None and job_id != expected_id:
        raise CorruptJob(f"job_id mismatch: {job_id!r} != {expected_id!r}")
    if job.get("state") not in ("complete", "scoring_failed"):
        raise CorruptJob(f"invalid job state: {job.get('state')!r}")
    if job.get("format") not in ("wav", "flac"):
        raise CorruptJob(f"invalid job format: {job.get('format')!r}")
    sample_rate = job.get("sample_rate")
    if not _is_int(sample_rate) or sample_rate <= 0:
        raise CorruptJob(f"invalid job sample rate: {sample_rate!r}")
    request = job.get("request")
    if not isinstance(request, dict):
        raise CorruptJob("job request must be an object")
    take_count = request.get("takes")
    if not _is_int(take_count) or take_count < TAKES_MIN or take_count > TAKES_MAX:
        raise CorruptJob(f"invalid take count: {take_count!r}")
    seeds = request.get("seeds")
    if not isinstance(seeds, list) or len(seeds) != take_count:
        raise CorruptJob("request seeds do not match the take count")
    for seed in seeds:
        if not _is_int(seed) or not 0 <= seed <= 2**31 - 1:
            raise CorruptJob(f"invalid request seed: {seed!r}")
    request_segments = _validate_request_segments(request)
    takes = job.get("takes")
    if not isinstance(takes, list) or len(takes) != take_count:
        raise CorruptJob("takes list does not match the take count")
    seen: set[int] = set()
    for take in takes:
        if not isinstance(take, dict):
            raise CorruptJob("each take must be an object")
        take_id = take.get("id")
        if not _is_int(take_id) or take_id < 1:
            raise CorruptJob(f"invalid take id: {take_id!r}")
        if take_id in seen:
            raise CorruptJob(f"duplicate take id: {take_id}")
        seen.add(take_id)
        if take.get("index") != take_id - 1:
            raise CorruptJob(f"take {take_id} has an invalid index")
        if take.get("format") != job.get("format"):
            raise CorruptJob(f"take {take_id} format does not match the job")
        expected_file = f"take-{take_id:02d}.{job['format']}"
        filename = _basename(take.get("file"), "take file")
        if not _TAKE_FILE_RE.fullmatch(filename) or filename != expected_file:
            raise CorruptJob(f"invalid take file name: {filename!r}")
        expected_recipe = f"take-{take_id:02d}.recipe.json"
        recipe = _basename(take.get("recipe"), "take recipe")
        if not recipe.endswith(".recipe.json") or recipe != expected_recipe:
            raise CorruptJob(f"invalid take recipe name: {recipe!r}")
        if not _is_int(take.get("seed")) or take["seed"] != seeds[take_id - 1]:
            raise CorruptJob(f"take {take_id} seed does not match the request")
        if not _is_int(take.get("sample_rate")) or take["sample_rate"] != sample_rate:
            raise CorruptJob(f"take {take_id} sample_rate does not match the job")
        if not _is_finite(take.get("duration_s")) or float(take["duration_s"]) <= 0.0:
            raise CorruptJob(f"take {take_id} has an invalid duration")
        segment_seeds = take.get("segment_seeds")
        if not isinstance(segment_seeds, dict):
            raise CorruptJob(f"take {take_id} has no segment_seeds")
        expected_keys = {str(segment["id"]) for segment in request_segments}
        actual_keys = {str(key) for key in segment_seeds}
        if actual_keys != expected_keys or len(actual_keys) != len(segment_seeds):
            raise CorruptJob(f"take {take_id} segment_seeds keys do not match")
        for segment in request_segments:
            key = str(segment["id"])
            value = segment_seeds.get(key)
            expected = int(segment["seed"]) + take_id - 1
            if not _is_int(value) or value != expected or not 0 <= value <= 2**31 - 1:
                raise CorruptJob(f"take {take_id} segment seed {key} is invalid")
        segments = take.get("segments")
        if not isinstance(segments, list):
            raise CorruptJob(f"take {take_id} has no segments")
        _validate_take_segments(take, request_segments, job)
        metrics = take.get("metrics")
        if not isinstance(metrics, dict):
            raise CorruptJob(f"take {take_id} has no metrics")
        for value in metrics.values():
            if value is not None and not _is_finite(value):
                raise CorruptJob(f"take {take_id} has a non-numeric metric")
        unavailable = take.get("unavailable")
        if unavailable is not None and not isinstance(unavailable, dict):
            raise CorruptJob(f"take {take_id} unavailable must be an object")
    if seen != set(range(1, take_count + 1)):
        raise CorruptJob("take ids must be exactly 1..N")
    selected = job.get("selected")
    if isinstance(selected, bool):
        raise CorruptJob(f"complete job has invalid selected take: {selected!r}")
    if job.get("state") == "complete":
        if selected not in seen:
            raise CorruptJob(f"complete job has invalid selected take: {selected!r}")
        if not job.get("selected_at"):
            raise CorruptJob("complete job has no selected_at")
    elif selected is not None:
        raise CorruptJob("scoring_failed job must not select a take")


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #
def rank_takes(
    takes: list[dict[str, Any]],
    criteria: list[str],
    *,
    lufs_target: float = LUFS_TARGET,
) -> list[dict[str, Any]]:
    """Return takes ordered best-first using unrounded metric values."""

    def metric(criterion: str, take: dict[str, Any]) -> Optional[float]:
        key = {
            "cer": "cer",
            "speaker-sim": "speaker_similarity",
            "lufs": "lufs",
        }[criterion]
        value = (take.get("metrics") or {}).get(key)
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def sort_key(take: dict[str, Any]) -> tuple[Any, ...]:
        parts: list[float] = []
        for criterion in criteria:
            value = metric(criterion, take)
            if value is None:
                parts.append(float("inf"))
            elif criterion == "speaker-sim":
                parts.append(-value)
            elif criterion == "lufs":
                parts.append(abs(value - float(lufs_target)))
            else:
                parts.append(value)
        parts.append(float(take.get("id", 0)))
        return tuple(parts)

    return sorted(takes, key=sort_key)
