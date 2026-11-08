"""``qtts dub build`` public entry point (FR-P2-04).

``build(cfg, args, *, runner=None)`` never imports a model runtime at import
time.  The runner protocol is the same as audiobook's ``SegmentRunner``; the
default adapter calls ``commands.say.run`` lazily.
"""

from __future__ import annotations

import math
import re
import shutil
import signal
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .. import audio_io
from ..audiobook import project as _audiobook_project
from ..backends.registry import resolve_backend
from ..capabilities import parse_voice_ref
from ..errors import (
    CancelledError,
    InputError,
    ModelError,
    QualityError,
    UsageError,
)
from ..plan import SAMPLING_DEFAULTS
from ..utils import eprint, sha256_file
from ..voice_store import VoiceStore
from . import encode, manifest as manifest_mod, mix, synth
from . import timeline as timeline_mod

FORMATS = ("wav", "flac", "m4a", "mp3")
LOSSY = {"m4a", "mp3"}
LOSSLESS = {"wav", "flac"}
DEFAULT_SAMPLE_RATE = 24000
DEFAULT_LUFS = -18.0
DEFAULT_TRUE_PEAK = -1.5
DEFAULT_FPS = 25
DEFAULT_MIN_SPEED = 0.85
DEFAULT_MAX_SPEED = 1.25
DEFAULT_OVERFLOW = "warn"

_SPEAKER_NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


@dataclass
class SpeakerDefault:
    voice: Optional[str]
    instruct: Optional[str]


@dataclass
class Project:
    path: Path
    sha256: str
    voice: Optional[str]
    instruct: Optional[str]
    language: str
    sample_rate: Optional[int]
    lufs: Optional[float]
    true_peak: Optional[float]
    fps: Optional[int]
    speakers: dict[str, SpeakerDefault]


@dataclass
class Settings:
    fmt: str
    final: Path
    manifest: Path
    mux_path: Optional[Path]
    work_dir: Path
    stems_dir: Optional[Path]
    stems: dict[str, Path]
    timeline_json: Optional[Path]
    sample_rate: int
    lufs_enabled: bool
    target_lufs: float
    true_peak: float
    fps: int
    min_speed: float
    max_speed: float
    overflow: str
    force: bool
    dry_run: bool
    keep_work: bool
    no_cache: bool
    video: Optional[Path]
    bed_db: Optional[float]
    mux: bool
    model: Optional[str]
    seed: int
    language: str
    instruct: Optional[str]


# --------------------------------------------------------------------------- #
# Strict project loader (§3)
# --------------------------------------------------------------------------- #


def _invalid(message: str, path: Path) -> InputError:
    return _audiobook_project._invalid(message, path)


def _validate_ref(value: str, label: str, path: Path) -> str:
    try:
        parse_voice_ref(value)
    except InputError as exc:
        raise _invalid(f"invalid {label}: {value!r}", path) from exc
    return value


def load_project(path_value: str | Path) -> Project:
    path = Path(path_value).expanduser().resolve(strict=False)
    if not path.is_file():
        raise _invalid(f"project file not found: {path}", path)
    raw = _audiobook_project._read_text(
        path, _audiobook_project.MAX_PROJECT_BYTES, "project"
    )
    data = _audiobook_project._load_project_data(path, raw)
    if not isinstance(data, dict):
        raise _invalid("project must be a mapping", path)
    _audiobook_project._check_tree(data, path)
    _audiobook_project._unknown(
        data,
        {
            "schema_version",
            "voice",
            "instruct",
            "language",
            "sample_rate",
            "loudness",
            "timeline",
            "speakers",
        },
        "project",
        path,
    )

    schema = data.get("schema_version")
    if isinstance(schema, bool) or not isinstance(schema, int) or schema != 1:
        raise _invalid("schema_version must be the integer 1", path)

    voice = None
    if data.get("voice") is not None:
        voice = _validate_ref(
            _audiobook_project._string(data["voice"], "voice", path=path), "voice", path
        )
    instruct = None
    if data.get("instruct") is not None:
        instruct = _audiobook_project._string(
            data["instruct"], "instruct", path=path, allow_empty=True
        )
    language = _audiobook_project._string(
        data.get("language", "auto"), "language", path=path
    )

    sample_rate = None
    if "sample_rate" in data:
        sample_rate = int(
            _audiobook_project._number(
                data["sample_rate"],
                "sample_rate",
                path=path,
                minimum=12000,
                maximum=48000,
                integer=True,
            )
        )

    loudness = _audiobook_project._mapping(
        data.get("loudness", {}), "loudness", path=path
    )
    _audiobook_project._unknown(loudness, {"lufs", "true_peak"}, "loudness", path)
    lufs = None
    if "lufs" in loudness:
        lufs = float(
            _audiobook_project._number(
                loudness["lufs"], "loudness.lufs", path=path, minimum=-40, maximum=-5
            )
        )
    true_peak = None
    if "true_peak" in loudness:
        true_peak = float(
            _audiobook_project._number(
                loudness["true_peak"],
                "loudness.true_peak",
                path=path,
                minimum=-9,
                maximum=0,
            )
        )

    timeline = _audiobook_project._mapping(
        data.get("timeline", {}), "timeline", path=path
    )
    _audiobook_project._unknown(timeline, {"fps"}, "timeline", path)
    fps = None
    if "fps" in timeline:
        fps = int(
            _audiobook_project._number(
                timeline["fps"], "timeline.fps", path=path, minimum=1, maximum=120, integer=True
            )
        )

    speakers: dict[str, SpeakerDefault] = {}
    speakers_raw = _audiobook_project._mapping(
        data.get("speakers", {}), "speakers", path=path
    )
    for name, raw_entry in speakers_raw.items():
        if not isinstance(name, str) or not _SPEAKER_NAME.fullmatch(name):
            raise _invalid(f"invalid speaker name: {name!r}", path)
        entry = _audiobook_project._mapping(raw_entry, f"speakers.{name}", path=path)
        _audiobook_project._unknown(entry, {"voice", "instruct"}, f"speakers.{name}", path)
        entry_voice = None
        if entry.get("voice") is not None:
            entry_voice = _validate_ref(
                _audiobook_project._string(
                    entry["voice"], f"speakers.{name}.voice", path=path
                ),
                f"speakers.{name}.voice",
                path,
            )
        entry_instruct = None
        if entry.get("instruct") is not None:
            entry_instruct = _audiobook_project._string(
                entry["instruct"],
                f"speakers.{name}.instruct",
                path=path,
                allow_empty=True,
            )
        speakers[name] = SpeakerDefault(entry_voice, entry_instruct)

    try:
        digest = sha256_file(path)
    except OSError as exc:
        raise _invalid(f"cannot read project {path}: {exc}", path) from exc
    return Project(
        path=path,
        sha256=digest,
        voice=voice,
        instruct=instruct,
        language=language,
        sample_rate=sample_rate,
        lufs=lufs,
        true_peak=true_peak,
        fps=fps,
        speakers=speakers,
    )


# --------------------------------------------------------------------------- #
# CLI validation helpers
# --------------------------------------------------------------------------- #


def _usage_number(value: Any, name: str, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise UsageError(f"{name} must be a number")
    if not math.isfinite(float(value)) or float(value) < low or float(value) > high:
        raise UsageError(f"{name} must be between {low} and {high}")
    return float(value)


def _usage_int(value: Any, name: str, low: int, high: int) -> int:
    number = _usage_number(value, name, low, high)
    if float(number) != int(number):
        raise UsageError(f"{name} must be an integer")
    return int(number)


def parse_speaker_map(values: Optional[list[str]]) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw in values or []:
        if "=" not in raw:
            raise UsageError(f"--speaker-voice must be SPEAKER=VOICE_REF, got {raw!r}")
        speaker, _, voice = raw.partition("=")
        speaker = speaker.strip()
        voice = voice.strip()
        if not speaker or not voice:
            raise UsageError(f"--speaker-voice has an empty part: {raw!r}")
        if not _SPEAKER_NAME.fullmatch(speaker):
            raise UsageError(f"invalid speaker name in --speaker-voice: {speaker!r}")
        if speaker in result:
            raise UsageError(f"duplicate --speaker-voice for speaker {speaker!r}")
        try:
            parse_voice_ref(voice)
        except InputError as exc:
            raise UsageError(
                f"invalid voice reference in --speaker-voice for {speaker!r}: {exc.message}"
            ) from exc
        result[speaker] = voice
    return result


def _fps(args: Any, project: Optional[Project]) -> int:
    value = getattr(args, "fps", None)
    if value is not None:
        return _usage_int(value, "--fps", 1, 120)
    if project is not None and project.fps is not None:
        return int(project.fps)
    return DEFAULT_FPS


def _resolve_settings(
    cfg: Any,
    args: Any,
    project: Optional[Project],
) -> Settings:
    explicit = getattr(args, "format", None)
    if explicit is not None and explicit not in FORMATS:
        raise UsageError(f"unsupported format: {explicit!r}")

    raw_out = getattr(args, "out", None)
    if raw_out is None:
        fmt = explicit or "wav"
        final = manifest_mod.canonical(Path.cwd() / f"dub.{fmt}")
    else:
        raw = str(raw_out)
        if raw == "" or raw == "-":
            raise UsageError("--out must be a file path")
        target = Path(raw).expanduser()
        suffix = target.suffix.lower().lstrip(".")
        if suffix:
            if suffix not in FORMATS:
                raise UsageError(
                    f"unsupported output suffix '.{suffix}'; expected one of "
                    f"{', '.join(FORMATS)}"
                )
            if explicit is not None and explicit != suffix:
                raise UsageError(
                    f"--format {explicit} does not match --out suffix '.{suffix}'"
                )
            fmt = suffix
        else:
            fmt = explicit or "wav"
            target = Path(str(target) + "." + fmt)
        final = manifest_mod.canonical(target)

    stem = final.stem
    manifest_path = manifest_mod.canonical(final.with_name(stem + ".dub.json"))
    work_dir = manifest_mod.canonical(final.with_name(stem + ".dub-work"))

    video = getattr(args, "video", None)
    video_path = None
    if video is not None:
        raw_video = str(video)
        if raw_video == "" or raw_video == "-":
            raise UsageError("--video must be a file path")
        video_path = manifest_mod.canonical(raw_video)

    mux = bool(getattr(args, "mux", False))
    if mux and video_path is None:
        raise UsageError("--mux requires --video")

    bed_db = getattr(args, "bed_db", None)
    if bed_db is not None:
        if video_path is None:
            raise UsageError("--bed-db requires --video")
        bed_db = _usage_number(bed_db, "--bed-db", -40, 0)

    stems_raw = getattr(args, "stems_out", None)
    stems_dir = None
    if stems_raw is not None:
        raw_stems = str(stems_raw)
        if raw_stems == "" or raw_stems == "-":
            raise UsageError("--stems-out must be a directory path")
        stems_dir = manifest_mod.canonical(raw_stems)

    sample_rate_arg = getattr(args, "sample_rate", None)
    if sample_rate_arg is not None:
        sample_rate = _usage_int(sample_rate_arg, "--sample-rate", 12000, 48000)
    elif project is not None and project.sample_rate is not None:
        sample_rate = int(project.sample_rate)
    else:
        sample_rate = DEFAULT_SAMPLE_RATE

    no_loudness = bool(getattr(args, "no_loudness", False))
    lufs_arg = getattr(args, "lufs", None)
    if lufs_arg is not None and no_loudness:
        raise UsageError("--lufs cannot be combined with --no-loudness")
    if lufs_arg is not None:
        target_lufs = _usage_number(lufs_arg, "--lufs", -40, -5)
    elif project is not None and project.lufs is not None:
        target_lufs = float(project.lufs)
    else:
        target_lufs = DEFAULT_LUFS

    peak_arg = getattr(args, "true_peak", None)
    if peak_arg is not None:
        true_peak = _usage_number(peak_arg, "--true-peak", -9, 0)
    elif project is not None and project.true_peak is not None:
        true_peak = float(project.true_peak)
    else:
        true_peak = DEFAULT_TRUE_PEAK

    min_arg = getattr(args, "min_speed", None)
    max_arg = getattr(args, "max_speed", None)
    min_speed = (
        _usage_number(min_arg, "--min-speed", 0.5, 2.0)
        if min_arg is not None
        else DEFAULT_MIN_SPEED
    )
    max_speed = (
        _usage_number(max_arg, "--max-speed", 0.5, 2.0)
        if max_arg is not None
        else DEFAULT_MAX_SPEED
    )
    if min_speed >= max_speed:
        raise UsageError("--min-speed must be less than --max-speed")

    overflow = getattr(args, "overflow", None) or DEFAULT_OVERFLOW
    if overflow not in ("warn", "fail"):
        raise UsageError("--overflow must be 'warn' or 'fail'")

    language_arg = getattr(args, "language", None)
    if language_arg is not None:
        if not isinstance(language_arg, str) or not language_arg.strip():
            raise UsageError("--language must be a non-empty string")
        language = language_arg
    elif project is not None:
        language = project.language
    else:
        language = "auto"

    instruct_arg = getattr(args, "instruct", None)
    if instruct_arg is not None:
        if not isinstance(instruct_arg, str):
            raise UsageError("--instruct must be a string")
        global_instruct: Optional[str] = instruct_arg
    elif project is not None:
        global_instruct = project.instruct
    else:
        global_instruct = None

    # Pure CLI range/combination checks come first; a missing input file is an
    # input error only after all usage-level validation has passed.
    if video_path is not None and not video_path.is_file():
        raise InputError(
            f"video file not found: {video_path}", details={"video": str(video_path)}
        )

    voice_arg = getattr(args, "voice", None)
    if voice_arg is not None:
        parse_voice_ref(str(voice_arg))

    return Settings(
        fmt=fmt,
        final=final,
        manifest=manifest_path,
        mux_path=manifest_mod.canonical(final.with_name(stem + ".mp4")) if mux else None,
        work_dir=work_dir,
        stems_dir=stems_dir,
        stems={},
        timeline_json=None,
        sample_rate=sample_rate,
        lufs_enabled=not no_loudness,
        target_lufs=target_lufs,
        true_peak=true_peak,
        fps=_fps(args, project),
        min_speed=min_speed,
        max_speed=max_speed,
        overflow=overflow,
        force=bool(getattr(args, "force", False)),
        dry_run=bool(getattr(args, "dry_run", False)),
        keep_work=bool(getattr(args, "keep_work", False)),
        no_cache=bool(getattr(cfg, "no_cache", False)),
        video=video_path,
        bed_db=float(bed_db) if bed_db is not None else None,
        mux=mux,
        model=getattr(args, "model", None),
        seed=int(getattr(cfg, "seed", 1234)),
        language=language,
        instruct=global_instruct,
    )


@dataclass
class CuePlan:
    cue: Any
    ref: str
    instruct: Optional[str]
    resolution: Any


def _build_cue_plans(
    cues: list[Any],
    project: Optional[Project],
    *,
    cli_voice: Optional[str],
    speaker_map: dict[str, str],
    default_instruct: Optional[str],
) -> list[CuePlan]:
    default_ref = cli_voice if cli_voice is not None else (project.voice if project else None)
    plans: list[CuePlan] = []
    for cue in cues:
        speaker_instruct = None
        if cue.speaker is not None and project is not None:
            entry = project.speakers.get(cue.speaker)
            if entry is not None:
                speaker_instruct = entry.instruct
        instruct = (
            speaker_instruct
            if speaker_instruct is not None
            else default_instruct
        )
        if cue.speaker is None:
            ref = default_ref
        else:
            ref = speaker_map.get(cue.speaker)
            if ref is None and project is not None:
                entry = project.speakers.get(cue.speaker)
                if entry is not None:
                    ref = entry.voice
            if ref is None:
                ref = default_ref
        if ref is None:
            label = "anonymous cue" if cue.speaker is None else f"speaker {cue.speaker!r}"
            raise InputError(
                f"cue {cue.index} ({label}) has no voice; use --voice or a project mapping",
                details={"index": int(cue.index), "speaker": cue.speaker},
            )
        plans.append(CuePlan(cue=cue, ref=str(ref), instruct=instruct, resolution=None))
    return plans


# --------------------------------------------------------------------------- #
# Managed paths
# --------------------------------------------------------------------------- #


def _stems_for(cues: list[Any], settings: Settings) -> tuple[dict[str, Path], Optional[Path]]:
    # The name is reserved even when stems are not requested: it is also the
    # voices payload / manifest key for cues without a speaker.
    if any(cue.speaker == mix.ANONYMOUS for cue in cues):
        raise InputError(
            f"a timeline speaker named {mix.ANONYMOUS!r} is reserved",
            details={"speaker": mix.ANONYMOUS},
        )
    if settings.stems_dir is None:
        return {}, None
    speakers = {cue.speaker for cue in cues if cue.speaker is not None}
    keys = set(speakers)
    if any(cue.speaker is None for cue in cues):
        keys.add(mix.ANONYMOUS)
    stems = {key: settings.stems_dir / f"{key}.wav" for key in keys}
    return stems, settings.stems_dir / "timeline.json"


def _validate_managed(
    settings: Settings,
    parsed: timeline_mod.ParsedTimeline,
    project: Optional[Project],
) -> None:
    files: dict[str, Path] = {
        "final": settings.final,
        "manifest": settings.manifest,
    }
    if settings.mux_path is not None:
        files["mux"] = settings.mux_path
    for key, path in settings.stems.items():
        files[f"stem:{key}"] = path
    if settings.timeline_json is not None:
        files["timeline_json"] = settings.timeline_json
    dirs: dict[str, Path] = {"work": settings.work_dir}
    if settings.stems_dir is not None:
        dirs["stems"] = settings.stems_dir
    inputs: dict[str, Path] = {"timeline": parsed.path}
    if project is not None:
        inputs["project"] = project.path
    if settings.video is not None:
        inputs["video"] = settings.video
    manifest_mod.validate_paths(
        files=files,
        dirs=dirs,
        inputs=inputs,
        work_dir=settings.work_dir,
        stems_dir=settings.stems_dir,
    )
    manifest_mod.preflight_parents(files, dirs)
    manifest_mod.check_existing_targets(
        files,
        dirs,
        force=settings.force,
        dry_run=settings.dry_run,
        exclude_dirs={"work", "stems"},
    )


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #


def _dry_run_payload(
    *,
    parsed: timeline_mod.ParsedTimeline,
    project: Optional[Project],
    settings: Settings,
    plans: list[CuePlan],
    contexts: dict[str, Any],
    fingerprints: list[Optional[str]],
    voices: dict[str, dict[str, Any]],
    speakers: list[str],
    video_meta: Optional[dict[str, Any]],
    warnings: list[str],
    estimated_frames: int,
) -> dict[str, Any]:
    cue_payload = []
    for plan, fingerprint_value in zip(plans, fingerprints):
        cue = plan.cue
        cue_payload.append(
            {
                "index": int(cue.index),
                "start_s": cue.start_s,
                "end_s": cue.end_s,
                "speaker": cue.speaker,
                "voice": plan.ref,
                "instruct": plan.instruct,
                "text_chars": len(cue.text),
                "fingerprint": fingerprint_value,
            }
        )
    stems = None
    if settings.stems_dir is not None:
        stems = {
            "dir": str(settings.stems_dir),
            "files": {key: str(path) for key, path in sorted(settings.stems.items())},
            "timeline_json": str(settings.timeline_json),
        }
    return {
        "ok": True,
        "dry_run": True,
        "timeline": {
            "path": str(parsed.path),
            "format": parsed.format,
            "sha256": parsed.sha256,
            "cues": len(parsed.cues),
            "text_chars": parsed.text_chars,
        },
        "project": (
            {"path": str(project.path), "sha256": project.sha256}
            if project is not None
            else None
        ),
        "video": video_meta,
        "output": {
            "path": str(settings.final),
            "format": settings.fmt,
            "sample_rate": settings.sample_rate,
            "estimated_frames": int(estimated_frames),
        },
        "manifest": str(settings.manifest),
        "work_dir": str(settings.work_dir),
        "language": settings.language,
        "instruct": settings.instruct,
        "stems": stems,
        "mux": str(settings.mux_path) if settings.mux_path is not None else None,
        "bed": (
            {"path": str(settings.video), "volume_db": settings.bed_db}
            if settings.bed_db is not None
            else None
        ),
        "cues": cue_payload,
        "voices": voices,
        "speakers": speakers,
        "loudness": {
            "enabled": settings.lufs_enabled,
            "target_lufs": settings.target_lufs if settings.lufs_enabled else None,
            "true_peak_dbtp": settings.true_peak if settings.lufs_enabled else None,
        },
        "stretch": {
            "min_speed": settings.min_speed,
            "max_speed": settings.max_speed,
            "overflow": settings.overflow,
        },
        "warnings": list(warnings),
    }


def _synthesize_and_fit(
    *,
    settings: Settings,
    plans: list[CuePlan],
    fingerprints: list[Optional[str]],
    runner: Any,
    cache: synth.CueCache,
    work_dir: Path,
    video_frames: Optional[int],
    cancel_event: threading.Event,
    warnings: list[str],
    language: str,
) -> tuple[
    list[dict[str, Any]],
    list[int],
    dict[str, int],
    Optional[int],
]:
    """Render every cue to a work WAV, returning metadata and fail indexes."""
    stats = {"total": 0, "synthesized": 0, "reused": 0}
    metadata: list[dict[str, Any]] = []
    fail_indexes: list[int] = []
    src_rate: Optional[int] = None
    recipes_dir = work_dir / "recipes"
    rendered_dir = work_dir / "rendered"
    stretch_dir = work_dir / "stretch"

    def check_cancel() -> None:
        if cancel_event.is_set():
            raise CancelledError("cancelled")

    for plan, expected in zip(plans, fingerprints):
        check_cancel()
        cue = plan.cue
        geometry = mix.cue_geometry(
            cue.start_s, cue.end_s, settings.sample_rate, cue.index
        )
        wav_path, _ = cache.paths(cue.index)
        found = cache.lookup(cue.index, expected) if expected else None
        if found is not None:
            data, rate, _languages = found
            stats["reused"] += 1
        else:
            result = synth.synthesize_cue(
                runner,
                text=cue.text,
                out=wav_path,
                voice=plan.ref,
                language=language,
                instruct=plan.instruct,
                seed=settings.seed,
                cancel_check=cancel_event.is_set,
                recipe_dir=recipes_dir,
                index=cue.index,
            )
            stats["synthesized"] += 1
            data, rate = synth.validate_fresh_audio(wav_path)
            languages = sorted(result.get("languages_actual") or [])
            if expected:
                cache.write(
                    cue.index,
                    fingerprint_value=expected,
                    sample_rate=int(rate),
                    frames=int(data.shape[0]),
                    duration_s=round(data.shape[0] / float(rate), 3),
                    sha256=sha256_file(wav_path),
                    device_actual=result.get("device_actual"),
                    dtype_actual=result.get("dtype_actual"),
                    languages_actual=languages,
                )

        if src_rate is None:
            src_rate = int(rate)
            ratio = float(src_rate) / float(settings.sample_rate)
            if ratio < 0.5 or ratio > 2.0:
                raise InputError(
                    f"synthesis rate {src_rate} Hz cannot be resampled to "
                    f"{settings.sample_rate} Hz (ratio {ratio:.3f} outside 0.5-2.0)",
                    details={"source": src_rate, "target": settings.sample_rate},
                )
        elif int(rate) != src_rate:
            raise ModelError(
                f"inconsistent synthesis sample rate: cue {cue.index} reported "
                f"{rate} Hz, first cue reported {src_rate} Hz",
                details={"index": int(cue.index), "sample_rate": int(rate)},
            )

        fit = synth.fit_duration(
            source=data,
            source_path=wav_path,
            source_rate=int(rate),
            target_frames=geometry.target_frames,
            master_rate=settings.sample_rate,
            min_speed=settings.min_speed,
            max_speed=settings.max_speed,
            stretch_dir=stretch_dir,
            index=cue.index,
        )
        rendered_path = rendered_dir / f"{int(cue.index):04d}.wav"
        synth.write_rendered(rendered_path, fit.data, settings.sample_rate)

        overflow = bool(fit.overflow)
        if overflow:
            if settings.overflow == "fail":
                fail_indexes.append(int(cue.index))
            else:
                warnings.append(
                    f"cue {cue.index} overflows its slot by "
                    f"{(fit.rendered_frames - geometry.target_frames) / settings.sample_rate:.3f}s"
                )
        if video_frames is not None and (
            geometry.start_frame + fit.rendered_frames > video_frames
        ):
            overflow = True
            if settings.overflow == "fail":
                fail_indexes.append(int(cue.index))
            else:
                warnings.append(
                    f"cue {cue.index} ends beyond the video duration and is truncated"
                )
        metadata.append(
            {
                "index": int(cue.index),
                "start_frame": geometry.start_frame,
                "end_frame": geometry.end_frame,
                "target_frames": geometry.target_frames,
                "rendered_frames": int(fit.rendered_frames),
                "synth_duration_s": float(fit.synth_duration_s),
                "speed": float(fit.speed),
                "overflow": overflow,
                "voice": plan.ref,
                "instruct": plan.instruct,
                "rendered_path": rendered_path,
            }
        )
        stats["total"] += 1
        check_cancel()
    return metadata, sorted(set(fail_indexes)), stats, src_rate


def _media_preflight(
    settings: Settings, warnings: list[str]
) -> tuple[Optional[dict[str, Any]], Optional[int], Optional[str]]:
    video_meta = None
    video_frames = None
    video_sha = None
    if settings.video is not None:
        info = encode.probe_media(settings.video)
        if settings.mux and not info["has_video"]:
            raise InputError(
                f"--mux requires a video stream: {settings.video}",
                details={"video": str(settings.video)},
            )
        if settings.bed_db is not None and not info["has_audio"]:
            raise InputError(
                f"--bed-db requires an audio stream: {settings.video}",
                details={"video": str(settings.video)},
            )
        duration = float(info["duration_s"])
        video_frames = mix.half_up(duration * settings.sample_rate)
        try:
            video_sha = sha256_file(settings.video)
        except OSError as exc:
            raise InputError(
                f"cannot read video {settings.video}: {exc}",
                details={"video": str(settings.video)},
            ) from exc
        video_meta = {
            "path": str(settings.video),
            "sha256": video_sha,
            "duration_s": round(duration, 3),
        }
    if not settings.dry_run:
        if settings.fmt in LOSSY:
            encode.require_ffmpeg(settings.fmt)
            encode.require_ffprobe()
        if settings.mux:
            encode.require_ffmpeg("m4a")
            encode.require_ffprobe()
        if settings.bed_db is not None and shutil.which("ffmpeg") is None:
            raise ModelError("ffmpeg is required for --bed-db but was not found on PATH")
    return video_meta, video_frames, video_sha


def build(cfg: Any, args: Any, *, runner: Any = None) -> dict[str, Any]:
    project = (
        load_project(getattr(args, "project"))
        if getattr(args, "project", None) is not None
        else None
    )
    speaker_map = parse_speaker_map(getattr(args, "speaker_voice", None))
    fps = _fps(args, project)
    settings = _resolve_settings(cfg, args, project)
    parsed = timeline_mod.load_timeline(getattr(args, "timeline"), fps=fps)
    settings.stems, settings.timeline_json = _stems_for(parsed.cues, settings)
    _validate_managed(settings, parsed, project)

    language = settings.language
    plans = _build_cue_plans(
        parsed.cues,
        project,
        cli_voice=getattr(args, "voice", None),
        speaker_map=speaker_map,
        default_instruct=settings.instruct,
    )

    warnings: list[str] = []
    spec = resolve_backend(getattr(args, "backend", None), cfg, args)
    models = synth.load_models(cfg, spec)
    store = VoiceStore(cfg.data_dir)

    resolutions: dict[str, Any] = {}
    for plan in plans:
        resolution = resolutions.get(plan.ref)
        if resolution is None:
            resolution = synth.resolve_voice_ref(
                models=models,
                store=store,
                ref=plan.ref,
                spec=spec,
                model_override=settings.model,
            )
            resolutions[plan.ref] = resolution
            warnings.extend(resolution.warnings)
        plan.resolution = resolution

    remote = bool(getattr(spec, "remote", False))
    if settings.dry_run and not synth.runtime_is_resolved(cfg, spec):
        device, dtype = None, None
    else:
        device, dtype = synth.effective_runtime(cfg, spec)

    contexts: dict[str, Any] = {}
    if device is not None:
        for ref, resolution in resolutions.items():
            contexts[ref] = synth.build_context(
                cfg=cfg,
                spec=spec,
                store=store,
                resolution=resolution,
                device=device,
                dtype=dtype,
                seed=settings.seed,
            )
    fingerprints: list[Optional[str]] = []
    for plan in plans:
        context = contexts.get(plan.ref)
        fingerprints.append(
            synth.fingerprint(
                context,
                text=plan.cue.text,
                language=language,
                instruct=plan.instruct,
            )
            if context is not None
            else None
        )

    video_meta, video_frames, video_sha = _media_preflight(settings, warnings)
    if video_frames is not None:
        for cue in parsed.cues:
            end_frame = mix.half_up(float(cue.end_s) * settings.sample_rate)
            if end_frame > video_frames:
                raise InputError(
                    f"cue {cue.index} ends beyond the video duration",
                    details={
                        "index": int(cue.index),
                        "end_frame": int(end_frame),
                        "video_frames": int(video_frames),
                    },
                )
    estimated_frames = mix.estimate_frames(
        parsed.cues, settings.sample_rate, video_frames
    )
    mix.check_master_budget(estimated_frames)

    voices_payload: dict[str, dict[str, Any]] = {}
    for plan in plans:
        key = plan.cue.speaker if plan.cue.speaker is not None else mix.ANONYMOUS
        voices_payload[key] = {
            "ref": plan.ref,
            "identity": plan.resolution.identity,
            "instruct": plan.instruct,
        }
    speakers = sorted({cue.speaker for cue in parsed.cues if cue.speaker is not None})

    if settings.dry_run:
        return _dry_run_payload(
            parsed=parsed,
            project=project,
            settings=settings,
            plans=plans,
            contexts=contexts,
            fingerprints=fingerprints,
            voices=voices_payload,
            speakers=speakers,
            video_meta=video_meta,
            warnings=warnings,
            estimated_frames=estimated_frames,
        )

    # -- non-dry setup ------------------------------------------------------ #
    temp_paths: list[Path] = []
    lock_fd: Optional[int] = None
    previous_term = None
    main_thread = threading.current_thread() is threading.main_thread()
    cancel_event = threading.Event()

    def _on_sigterm(signum, frame):  # pragma: no cover - signal timing
        cancel_event.set()
        raise KeyboardInterrupt

    if main_thread:
        try:
            previous_term = signal.getsignal(signal.SIGTERM)
            signal.signal(signal.SIGTERM, _on_sigterm)
        except (ValueError, OSError):
            previous_term = None

    try:
        create_dirs = [settings.final.parent, settings.work_dir]
        if settings.stems_dir is not None:
            create_dirs.append(settings.stems_dir)
        for directory in dict.fromkeys(create_dirs):
            Path(directory).mkdir(parents=True, exist_ok=True)
        for name in ("cues", "recipes", "rendered", "stretch"):
            (settings.work_dir / name).mkdir(parents=True, exist_ok=True)
        lock_fd = manifest_mod.acquire_lock(settings.work_dir)

        runner_obj = runner
        if runner_obj is None:
            runner_obj = synth.SaySegmentRunner(
                cfg, args, settings.work_dir / "recipes"
            )
        cache = synth.CueCache(
            settings.work_dir / "cues",
            enabled=not settings.force and not settings.no_cache,
        )
        metadata, fail_indexes, stats, src_rate = _synthesize_and_fit(
            settings=settings,
            plans=plans,
            fingerprints=fingerprints,
            runner=runner_obj,
            cache=cache,
            work_dir=settings.work_dir,
            video_frames=video_frames,
            cancel_event=cancel_event,
            warnings=warnings,
            language=language,
        )
        if fail_indexes:
            raise QualityError(
                "cue(s) cannot fit without overflowing: "
                + ", ".join(str(index) for index in fail_indexes),
                details={"cues": fail_indexes},
            )
        if src_rate is None:
            raise ModelError("no cue was synthesized")

        if video_frames is not None:
            length_frames = int(video_frames)
        else:
            longest = max(
                (meta["start_frame"] + meta["rendered_frames"] for meta in metadata),
                default=0,
            )
            length_frames = max(int(estimated_frames), int(longest))
        mix.check_master_budget(length_frames)

        token = uuid.uuid4().hex
        speaker_by_index = {
            int(plan.cue.index): plan.cue.speaker for plan in plans
        }

        def place_rendered(target: np.ndarray, records: list[dict[str, Any]]) -> None:
            for meta in records:
                rendered, rendered_rate = audio_io.read_audio(
                    meta["rendered_path"], mono=True
                )
                if int(rendered_rate) != settings.sample_rate:
                    raise ModelError(
                        f"rendered cue {meta['index']} has rate {rendered_rate}, "
                        f"expected {settings.sample_rate}"
                    )
                if int(rendered.shape[0]) != int(meta["rendered_frames"]):
                    raise ModelError(
                        f"rendered cue {meta['index']} changed length "
                        f"({rendered.shape[0]} != {meta['rendered_frames']})"
                    )
                mix.place(target, rendered, meta["start_frame"])
                del rendered

        master = mix.allocate(length_frames)
        place_rendered(master, metadata)

        # Materialize one stem at a time.  Keeping every stem alive would scale
        # peak memory with the speaker count; the peak here is master + one
        # stem (+ one rendered cue being read back).
        stem_files: dict[str, dict[str, Any]] = {}
        stem_hashes: dict[str, str] = {}
        stem_temps: dict[str, Path] = {}
        if settings.stems_dir is not None:
            by_key: dict[str, list[dict[str, Any]]] = {
                key: [] for key in settings.stems
            }
            for meta in metadata:
                speaker = speaker_by_index[int(meta["index"])]
                key = speaker if speaker is not None else mix.ANONYMOUS
                by_key[key].append(meta)
            for key in sorted(settings.stems):
                stem_data = mix.allocate(length_frames)
                place_rendered(stem_data, by_key[key])
                stem_path = settings.stems[key]
                stem_tmp = settings.stems_dir / f"{key}.{token}.tmp"
                temp_paths.append(stem_tmp)
                encode.write_lossless(stem_tmp, stem_data, settings.sample_rate, "wav")
                del stem_data
                digest = encode.output_sha256(stem_tmp)
                stem_temps[key] = stem_tmp
                stem_hashes[stem_path.name] = digest
                stem_files[key] = {
                    "path": str(stem_path),
                    "size": encode.output_size(stem_tmp),
                    "sha256": digest,
                }

        if settings.bed_db is not None:
            bed = mix.decode_bed(settings.video, settings.sample_rate, length_frames)
            mix.apply_bed(master, bed, settings.bed_db)

        if length_frames > mix.MAX_VOICE_SECONDS * settings.sample_rate:
            warnings.append(
                f"voice master is longer than 2 hours "
                f"({length_frames / settings.sample_rate:.1f}s)"
            )

        loudness = mix.normalize(
            master,
            settings.sample_rate,
            enabled=settings.lufs_enabled,
            target_lufs=settings.target_lufs,
            true_peak=settings.true_peak,
        )
        warnings.extend(loudness.warnings)
        mix_data = loudness.mix
        clipped = mix.is_clipped(mix_data)
        expected_duration = length_frames / float(settings.sample_rate)

        final_tmp = settings.final.parent / f"{settings.final.name}.{token}.tmp"
        temp_paths.append(final_tmp)
        encoded_peak: Optional[dict[str, Any]] = None
        master_wav: Optional[Path] = None

        if settings.fmt in LOSSLESS:
            encode.write_lossless(
                final_tmp, mix_data, settings.sample_rate, settings.fmt
            )
        else:
            master_wav = settings.work_dir / f"master.{token}.wav"
            temp_paths.append(master_wav)
            encode.write_master_wav(master_wav, mix_data, settings.sample_rate)
            bitrate = 192 if settings.fmt == "mp3" else 128
            encode.encode_lossy(
                master_wav=master_wav,
                output=final_tmp,
                fmt=settings.fmt,
                bitrate_kbps=bitrate,
            )
            encode.verify_duration(final_tmp, expected_duration)
            encoded_peak = encode.decode_peak(final_tmp, settings.sample_rate)
            peak_value = (encoded_peak or {}).get("true_peak_dbtp")
            if peak_value is None:
                warnings.append("encoded true peak is unavailable")
                verified_from_encoded = False
            elif peak_value > settings.true_peak:
                warnings.append(
                    f"encoded true peak {peak_value:.2f} dBTP exceeds "
                    f"{settings.true_peak:.2f} dBTP"
                )
                verified_from_encoded = False
            else:
                verified_from_encoded = loudness.verified_ok
            loudness.verified_ok = verified_from_encoded

        stems_payload = None
        timeline_tmp: Optional[Path] = None
        if settings.stems_dir is not None:
            timeline_doc = mix.timeline_document(
                timeline_path=parsed.path,
                timeline_format=parsed.format,
                timeline_sha256=parsed.sha256,
                sample_rate=settings.sample_rate,
                cues=parsed.cues,
                metadata=metadata,
                files={key: settings.stems[key].name for key in sorted(settings.stems)},
                hashes=stem_hashes,
                warnings=warnings,
            )
            assert settings.timeline_json is not None
            timeline_tmp = settings.stems_dir / f"timeline.json.{token}.tmp"
            temp_paths.append(timeline_tmp)
            manifest_mod.write_json_temp(timeline_tmp, timeline_doc)
            stems_payload = {
                "dir": str(settings.stems_dir),
                "files": stem_files,
                "timeline_json": {
                    "path": str(settings.timeline_json),
                    "sha256": encode.output_sha256(timeline_tmp),
                },
            }

        mux_payload = None
        mux_tmp = None
        if settings.mux:
            assert settings.video is not None and settings.mux_path is not None
            if master_wav is None:
                master_wav = settings.work_dir / f"mux-master.{token}.wav"
                temp_paths.append(master_wav)
                encode.write_master_wav(master_wav, mix_data, settings.sample_rate)
            mux_tmp = settings.final.parent / f"{settings.mux_path.name}.{token}.mp4"
            temp_paths.append(mux_tmp)
            encode.mux(settings.video, master_wav, mux_tmp)
            assert video_meta is not None
            encode.verify_duration(mux_tmp, float(video_meta["duration_s"]))
            mux_payload = {
                "path": str(settings.mux_path),
                "size": encode.output_size(mux_tmp),
                "sha256": encode.output_sha256(mux_tmp),
            }

        output_payload = {
            "path": str(settings.final),
            "format": settings.fmt,
            "duration_s": round(expected_duration, 3),
            "size": encode.output_size(final_tmp),
            "sha256": encode.output_sha256(final_tmp),
        }
        manifest_doc = manifest_mod.build_document(
            timeline={
                "path": str(parsed.path),
                "sha256": parsed.sha256,
                "format": parsed.format,
            },
            project=(
                {"path": str(project.path), "sha256": project.sha256}
                if project is not None
                else None
            ),
            video=video_meta,
            request={
                "seed": settings.seed,
                "sampling": _sampling_defaults(),
                "backend": {
                    "name": spec.name,
                    "identity": spec.identity,
                    "remote": remote,
                },
                "device": "remote" if remote else device,
                "dtype": None if remote else dtype,
            },
            voices=voices_payload,
            cues={"count": len(parsed.cues), "text_chars": parsed.text_chars},
            speakers=speakers,
            sample_rate=settings.sample_rate,
            output=output_payload,
            mux=mux_payload,
            stems=stems_payload,
            bed=(
                {
                    "path": str(settings.video),
                    "sha256": video_sha,
                    "volume_db": settings.bed_db,
                }
                if settings.bed_db is not None
                else None
            ),
            loudness={
                "enabled": settings.lufs_enabled,
                "target_lufs": settings.target_lufs if settings.lufs_enabled else None,
                "true_peak_dbtp": (
                    settings.true_peak if loudness.limiter_ran else loudness.actual_true_peak_dbtp
                ),
                "verified_ok": loudness.verified_ok,
                "encoded_true_peak_dbtp": (encoded_peak or {}).get("true_peak_dbtp"),
            },
            stretch=mix.stretch_stats(metadata, sum(1 for m in metadata if m["overflow"])),
            warnings=warnings,
            clipped=clipped,
        )
        manifest_tmp = settings.final.parent / f"{settings.manifest.name}.{token}.tmp"
        temp_paths.append(manifest_tmp)
        manifest_mod.write_json_temp(manifest_tmp, manifest_doc)

        if cancel_event.is_set():
            raise CancelledError("cancelled before publication")

        later: list[tuple[Path, Path]] = []
        for key in sorted(stem_temps):
            later.append((stem_temps[key], settings.stems[key]))
        if settings.timeline_json is not None and timeline_tmp is not None:
            later.append((timeline_tmp, settings.timeline_json))
        if mux_tmp is not None and settings.mux_path is not None:
            later.append((mux_tmp, settings.mux_path))

        published = manifest_mod.publish(
            final_tmp=final_tmp,
            final=settings.final,
            later=later,
            manifest_tmp=manifest_tmp,
            manifest=settings.manifest,
            warnings=warnings,
            quiet=bool(getattr(cfg, "quiet", False)),
        )
        warnings.extend(published.warnings)
        temp_paths.clear()
        manifest_mod.release_lock(lock_fd)
        lock_fd = None
        _cleanup_work(settings.work_dir, settings.keep_work, warnings)
        if published.interrupted_after_publish:
            eprint("interrupted after publish")
            return {"raw_stdout": True, "exit_code": 130}
        if warnings and not getattr(cfg, "quiet", False):
            for warning in warnings:
                eprint(f"warning: {warning}")
        return {
            "ok": True,
            "timeline": {
                "path": str(parsed.path),
                "format": parsed.format,
                "sha256": parsed.sha256,
            },
            "project": (
                {"path": str(project.path), "sha256": project.sha256}
                if project is not None
                else None
            ),
            "output": output_payload,
            "manifest": str(settings.manifest),
            "cues": {"count": len(parsed.cues), "text_chars": parsed.text_chars},
            "segments": dict(stats),
            "mux": mux_payload,
            "stems": (
                {
                    "dir": str(settings.stems_dir),
                    "files": sorted(settings.stems),
                }
                if settings.stems_dir is not None
                else None
            ),
            "loudness": manifest_doc["loudness"],
            "stretch": manifest_doc["stretch"],
            "warnings": warnings,
            "clipped": clipped,
        }
    except (KeyboardInterrupt, CancelledError) as exc:
        if "runner_obj" in locals() and hasattr(runner_obj, "request_cancel"):
            try:
                runner_obj.request_cancel()
            except Exception:
                pass
        for path in temp_paths:
            encode.remove_quietly(path)
        manifest_mod.release_lock(lock_fd)
        lock_fd = None
        if isinstance(exc, CancelledError):
            raise KeyboardInterrupt from exc
        raise
    finally:
        for path in temp_paths:
            encode.remove_quietly(path)
        manifest_mod.release_lock(lock_fd)
        if main_thread and previous_term is not None:
            try:
                signal.signal(signal.SIGTERM, previous_term)
            except (ValueError, OSError):
                pass


def _cleanup_work(work_dir: Path, keep: bool, warnings: list[str]) -> None:
    if keep:
        return
    try:
        shutil.rmtree(work_dir)
    except FileNotFoundError:
        pass
    except OSError as exc:
        warnings.append(f"could not remove the work dir {work_dir}: {exc}")


def _sampling_defaults() -> dict[str, Any]:
    from ..budget import DEFAULT_MAX_NEW_TOKENS

    values = dict(SAMPLING_DEFAULTS)
    values["max_new_tokens"] = int(DEFAULT_MAX_NEW_TOKENS)
    return values


def run(cfg: Any, args: Any, *, runner: Any = None) -> dict[str, Any]:
    return build(cfg, args, runner=runner)


__all__ = ["build", "load_project", "run"]
