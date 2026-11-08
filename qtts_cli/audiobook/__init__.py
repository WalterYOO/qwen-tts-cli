"""Audiobook builder public entry point (FR-P2-03).

``run(cfg, args, *, runner=None)`` never imports a model runtime at import
time.  ``runner`` is the injectable :class:`SegmentRunner` protocol from the
contract; the default adapter calls ``commands.say.run`` lazily.
"""

from __future__ import annotations

import json
import math
import signal
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .. import audio_io, audio_qc, postprocess
from ..errors import CancelledError, InputError, ModelError, UsageError
from ..utils import atomic_write_text, eprint, sha256_file
from ..voice_store import VoiceStore
from . import audio, encode, manifest as manifest_mod
from .project import BgmSpec, Chapter, Project, load_project
from .segments import (
    SaySegmentRunner,
    SegmentCache,
    build_fingerprint_context,
    effective_runtime,
    fingerprint,
    load_models,
    normalize_dtype_name,
    resolve_chapter_voice,
    runtime_is_resolved,
)

FORMATS = ("wav", "flac", "m4a", "m4b", "mp3")
LOSSY = {"m4a", "m4b", "mp3"}
LOSSLESS = {"wav", "flac"}
DEFAULT_SAMPLE_RATE = 24000
DEFAULT_LUFS = -18.0
DEFAULT_TRUE_PEAK = -1.5


@dataclass
class Settings:
    fmt: str
    final: Path
    out_dir: Path
    manifest: Path
    lock: Path
    journal: Path
    sample_rate: int
    bitrate_kbps: Optional[int]
    paragraph_ms: int
    chapter_ms: int
    title: str
    author: Optional[str]
    album: str
    voice_override: Optional[str]
    model: Optional[str]
    bgm: Optional[BgmSpec]
    loudness_enabled: bool
    target_lufs: float
    true_peak: float
    force: bool
    keep_stems: bool
    dry_run: bool
    work_dirs: dict[str, Path]


def _range(value: Any, name: str, low: float, high: float) -> float:
    if value is None:
        raise UsageError(f"{name} is required")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise UsageError(f"{name} must be a number")
    if not math.isfinite(float(value)) or float(value) < low or float(value) > high:
        raise UsageError(f"{name} must be between {low} and {high}")
    return float(value)


def _int_range(value: Any, name: str, low: int, high: int) -> int:
    number = _range(value, name, low, high)
    if float(number) != int(number):
        raise UsageError(f"{name} must be an integer")
    return int(number)


def _resolve_settings(cfg, args, project: Project) -> Settings:
    explicit = getattr(args, "format", None)
    if explicit is not None and explicit not in FORMATS:
        raise UsageError(f"unsupported format: {explicit!r}")
    m4b = bool(getattr(args, "m4b", False))
    if m4b:
        if explicit is not None and explicit != "m4b":
            raise UsageError("--m4b conflicts with --format " + str(explicit))
        explicit = "m4b"

    if args.out is not None and args.out_dir is not None:
        raise UsageError("--out and --out-dir are mutually exclusive")

    out_dir = (
        manifest_mod.canonical(args.out_dir)
        if args.out_dir is not None
        else manifest_mod.canonical(project.path.parent / "build")
    )
    if args.out is not None:
        raw = str(args.out)
        if raw == "" or raw == "-":
            raise UsageError("--out must be a file path")
        target = Path(raw).expanduser()
        suffix = target.suffix.lower().lstrip(".")
        if suffix:
            if suffix not in FORMATS:
                raise UsageError(
                    f"unsupported output suffix '.{suffix}'; expected one of {', '.join(FORMATS)}"
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
    else:
        fmt = explicit or "wav"
        final = manifest_mod.canonical(out_dir / f"{project.path.stem}.{fmt}")

    if args.bitrate is not None:
        if fmt in LOSSLESS:
            raise UsageError("--bitrate is only valid for m4a/m4b/mp3")
        bitrate = _int_range(args.bitrate, "--bitrate", 32, 320)
    else:
        bitrate = 192 if fmt == "mp3" else (128 if fmt in LOSSY else None)

    if args.sample_rate is not None:
        sample_rate = _int_range(args.sample_rate, "--sample-rate", 8000, 48000)
    else:
        sample_rate = int(project.sample_rate or DEFAULT_SAMPLE_RATE)

    paragraph_ms = (
        _int_range(args.paragraph_pause_ms, "--paragraph-pause-ms", 0, 60000)
        if args.paragraph_pause_ms is not None
        else int(project.paragraph_ms)
    )
    chapter_ms = (
        _int_range(args.chapter_pause_ms, "--chapter-pause-ms", 0, 60000)
        if args.chapter_pause_ms is not None
        else int(project.chapter_ms)
    )

    no_loudness = bool(getattr(args, "no_loudness", False))
    lufs_arg = getattr(args, "lufs", None)
    if lufs_arg is not None and no_loudness:
        raise UsageError("--lufs cannot be combined with --no-loudness")
    lufs_source = lufs_arg if lufs_arg is not None else project.lufs
    if lufs_source is not None:
        target_lufs = _range(lufs_source, "--lufs", -40, -5)
    else:
        target_lufs = DEFAULT_LUFS
    peak_arg = args.true_peak
    peak_source = peak_arg if peak_arg is not None else project.true_peak
    if peak_source is not None:
        true_peak = _range(peak_source, "--true-peak", -9, 0)
    else:
        true_peak = DEFAULT_TRUE_PEAK

    voice_override = getattr(args, "voice", None)
    if voice_override is not None:
        from ..capabilities import parse_voice_ref

        try:
            parse_voice_ref(voice_override)
        except Exception as exc:
            raise UsageError(f"invalid --voice reference: {voice_override!r}") from exc

    bgm = _resolve_bgm(args, project)
    loudness_enabled = not no_loudness and (
        lufs_source is not None or peak_source is not None or fmt in LOSSY or bgm is not None
    )
    if peak_arg is not None and (
        not loudness_enabled
        or (lufs_source is None and fmt not in LOSSY and bgm is None)
    ):
        # A CLI ceiling alone does not turn normalization on for lossless
        # output, and --no-loudness always wins.
        raise UsageError("--true-peak requires loudness normalization")

    return Settings(
        fmt=fmt,
        final=final,
        out_dir=out_dir,
        manifest=manifest_mod.canonical(out_dir / f"{project.path.stem}.json"),
        lock=manifest_mod.canonical(out_dir / manifest_mod.LOCK_NAME),
        journal=manifest_mod.canonical(out_dir / manifest_mod.JOURNAL_NAME),
        sample_rate=sample_rate,
        bitrate_kbps=bitrate,
        paragraph_ms=paragraph_ms,
        chapter_ms=chapter_ms,
        title=getattr(args, "title", None) or project.title,
        author=(
            getattr(args, "author", None)
            if getattr(args, "author", None) is not None
            else project.author
        ),
        album=getattr(args, "album", None) or project.album,
        voice_override=voice_override,
        model=getattr(args, "model", None),
        bgm=bgm,
        loudness_enabled=loudness_enabled,
        target_lufs=target_lufs,
        true_peak=true_peak,
        force=bool(getattr(args, "force", False)),
        keep_stems=bool(getattr(args, "keep_stems", False)),
        dry_run=bool(getattr(args, "dry_run", False)),
        work_dirs={name: manifest_mod.canonical(Path(out_dir) / name) for name in manifest_mod.WORK_DIR_NAMES},
    )


def _resolve_bgm(args, project: Project) -> Optional[BgmSpec]:
    cli_path = getattr(args, "bgm", None)
    project_bgm = project.bgm
    if cli_path is not None:
        raw = str(cli_path)
        if raw == "":
            raise UsageError("--bgm must not be empty")
        path = manifest_mod.canonical(raw)
        if not path.is_file():
            raise InputError(f"BGM file not found: {path}", details={"bgm": str(path)})
        if path.stat().st_size > 512_000_000:
            raise InputError(
                f"BGM file is too large ({path.stat().st_size} bytes; limit 512000000)",
                details={"bgm": str(path)},
            )
        source = project_bgm
        return BgmSpec(
            path,
            _range(
                args.bgm_volume_db if args.bgm_volume_db is not None else (source.volume_db if source else -20.0),
                "--bgm-volume-db",
                -60,
                0,
            ),
            _range(
                args.ducking_db if args.ducking_db is not None else (source.ducking_db if source else -10.0),
                "--ducking-db",
                -60,
                0,
            ),
            _int_range(
                args.bgm_fade_ms if args.bgm_fade_ms is not None else (source.fade_ms if source else 500),
                "--bgm-fade-ms",
                0,
                60000,
            ),
        )
    if project_bgm is None:
        return None
    return BgmSpec(
        project_bgm.path,
        _range(
            args.bgm_volume_db if args.bgm_volume_db is not None else project_bgm.volume_db,
            "--bgm-volume-db",
            -60,
            0,
        ),
        _range(
            args.ducking_db if args.ducking_db is not None else project_bgm.ducking_db,
            "--ducking-db",
            -60,
            0,
        ),
        _int_range(
            args.bgm_fade_ms if args.bgm_fade_ms is not None else project_bgm.fade_ms,
            "--bgm-fade-ms",
            0,
            60000,
        ),
    )


def _chapter_voice(chapter: Chapter, settings: Settings) -> str:
    return settings.voice_override or chapter.voice


def _validate_managed(project: Project, settings: Settings) -> None:
    sources = []
    seen = set()
    for chapter in project.chapters:
        if chapter.source_path is None:
            continue
        key = manifest_mod.collision_key(chapter.source_path)
        if key not in seen:
            seen.add(key)
            sources.append(chapter.source_path)
    managed: dict[str, list[Path]] = {
        "project": [project.path],
        "sources": sources,
        "final": [settings.final],
        "manifest": [settings.manifest],
        "lock": [settings.lock],
        "journal": [settings.journal],
    }
    if settings.bgm is not None:
        managed["bgm"] = [settings.bgm.path]
    manifest_mod.validate_paths(managed, settings.work_dirs.values())


def _dry_run_payload(
    project: Project,
    settings: Settings,
    *,
    voices: dict[int, Any],
    warnings: list[str],
    bgm_meta: Optional[dict[str, Any]],
    fingerprints: dict[int, list[Optional[str]]],
    encoder: dict[str, Any],
) -> dict[str, Any]:
    chapters = []
    for chapter in project.chapters:
        voice = voices[chapter.index]
        chapters.append(
            {
                "index": chapter.index,
                "title": chapter.title,
                "voice": _chapter_voice(chapter, settings),
                "voice_identity": voice.identity,
                "language": chapter.language,
                "instruct": chapter.instruct,
                "paragraphs": len(chapter.paragraphs),
                "source": {
                    "path": str(chapter.source_path) if chapter.source_path else None,
                    "sha256": chapter.source_sha256,
                    "inline": chapter.inline,
                },
                "fingerprints": fingerprints.get(chapter.index, [None] * len(chapter.paragraphs)),
            }
        )
    return {
        "ok": True,
        "dry_run": True,
        "project": {"path": str(project.path), "sha256": project.sha256},
        "book": {
            "title": settings.title,
            "author": settings.author,
            "album": settings.album,
            "language": project.language,
        },
        "output": {
            "path": str(settings.final),
            "format": settings.fmt,
            "sample_rate": settings.sample_rate,
            "bitrate_kbps": settings.bitrate_kbps,
        },
        "manifest": str(settings.manifest),
        "work_dirs": {name: str(path) for name, path in settings.work_dirs.items()},
        "will_write": [
            str(settings.final),
            str(settings.manifest),
            str(settings.lock),
            str(settings.journal),
        ],
        "chapters": chapters,
        "paragraphs": project.paragraph_count,
        "bgm": bgm_meta,
        "loudness": {
            "enabled": settings.loudness_enabled,
            "target_lufs": settings.target_lufs if settings.loudness_enabled else None,
            "true_peak_dbtp": settings.true_peak if settings.loudness_enabled else None,
        },
        "encoder": encoder,
        "warnings": warnings,
    }


def _encoder_info(fmt: str, bitrate: Optional[int]) -> dict[str, Any]:
    if fmt in LOSSLESS:
        return {"name": "soundfile", "format": "WAV" if fmt == "wav" else "FLAC", "bitrate_kbps": None}
    return {"name": encode._FFMPEG_ENCODER[fmt], "format": fmt, "bitrate_kbps": bitrate}


def run(cfg, args, *, runner: Any = None) -> dict[str, Any]:
    project = load_project(getattr(args, "project"))
    settings = _resolve_settings(cfg, args, project)
    warnings: list[str] = []

    # -- managed paths and parents (read-only) ------------------------------ #
    _validate_managed(project, settings)
    manifest_mod.preflight_parents(
        final=settings.final,
        manifest=settings.manifest,
        lock=settings.lock,
        journal=settings.journal,
        out_dir=settings.out_dir,
    )
    if not settings.dry_run and not settings.force and (
        settings.final.exists() or settings.manifest.exists()
    ):
        existing = settings.final if settings.final.exists() else settings.manifest
        raise InputError(
            f"output already exists: {existing}; pass --force to overwrite",
            details={"path": str(existing)},
        )

    if settings.fmt in LOSSY:
        encode.require_ffmpeg(settings.fmt)

    from ..backends.registry import resolve_backend

    spec = resolve_backend(getattr(args, "backend", None), cfg, args)
    models = load_models(cfg, spec)
    store = VoiceStore(cfg.data_dir)

    voices: dict[int, Any] = {}
    resolutions: dict[str, Any] = {}
    for chapter in project.chapters:
        ref = _chapter_voice(chapter, settings)
        cached = resolutions.get((ref, settings.model or ""))
        if cached is None:
            cached = resolve_chapter_voice(
                models=models,
                store=store,
                ref=ref,
                spec=spec,
                model_override=settings.model,
            )
            resolutions[(ref, settings.model or "")] = cached
        voices[chapter.index] = cached
        warnings.extend(cached.warnings)

    runner_obj = runner if runner is not None else None
    cancel_event = threading.Event()

    # -- effective runtime and dry-run fingerprints ------------------------- #
    if settings.dry_run and not runtime_is_resolved(cfg, spec):
        # ``auto`` device needs torch to resolve; dry-run must not import it.
        device, dtype = None, None
    else:
        device, dtype = effective_runtime(cfg, spec)

    contexts: dict[int, Any] = {}
    fingerprints: dict[int, list[Optional[str]]] = {}
    if device is not None or getattr(spec, "remote", False):
        for chapter in project.chapters:
            resolution = voices[chapter.index]
            ctx = build_fingerprint_context(
                cfg=cfg,
                spec=spec,
                store=store,
                resolution=resolution,
                device=device if device is not None else "remote",
                dtype=dtype,
                seed=int(getattr(cfg, "seed", 1234)),
            )
            contexts[chapter.index] = ctx
            fingerprints[chapter.index] = [
                fingerprint(
                    ctx,
                    text=text,
                    language=chapter.language,
                    instruct=chapter.instruct,
                )
                for text in chapter.paragraphs
            ]

    bgm_meta: Optional[dict[str, Any]] = None
    if settings.bgm is not None:
        if settings.dry_run:
            probe = audio.probe_bgm(settings.bgm.path, settings.sample_rate)
            bgm_meta = {
                "path": str(settings.bgm.path),
                "bytes": int(settings.bgm.path.stat().st_size),
                "sample_rate": int(probe["sample_rate"]),
                "duration_s": float(probe["duration_s"]),
                "volume_db": settings.bgm.volume_db,
                "ducking_db": settings.bgm.ducking_db,
                "fade_ms": settings.bgm.fade_ms,
            }

    encoder = _encoder_info(settings.fmt, settings.bitrate_kbps)

    if settings.dry_run:
        # Only report a journal mismatch; never touch or recover it.
        recovery = manifest_mod.recover(
            settings.journal,
            expected_final=settings.final,
            expected_manifest=settings.manifest,
            dry_run=True,
        )
        warnings.extend(recovery.warnings)
        return _dry_run_payload(
            project,
            settings,
            voices=voices,
            warnings=warnings,
            bgm_meta=bgm_meta,
            fingerprints=fingerprints,
            encoder=encoder,
        )

    # -- non-dry setup ------------------------------------------------------ #
    temp_paths: list[Path] = []
    stem_paths: list[Path] = []
    lock = None
    previous_term = None
    main_thread = threading.current_thread() is threading.main_thread()

    def _on_sigterm(signum, frame):  # pragma: no cover - signal timing
        cancel_event.set()
        raise KeyboardInterrupt

    def _check_cancel() -> None:
        if cancel_event.is_set():
            raise CancelledError("cancelled")

    if main_thread:
        previous_term = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, _on_sigterm)

    try:
        manifest_mod.ensure_out_dir(settings.out_dir)
        lock = manifest_mod.acquire_lock(settings.lock)
        recovery = manifest_mod.recover(
            settings.journal,
            expected_final=settings.final,
            expected_manifest=settings.manifest,
        )
        warnings.extend(recovery.warnings)
        if settings.bgm is not None:
            bgm_data = audio.load_bgm(settings.bgm.path, settings.sample_rate)
            bgm_meta = {
                "path": str(settings.bgm.path),
                "bytes": int(bgm_data.bytes),
                "sha256": bgm_data.sha256,
                "sample_rate": int(bgm_data.source_sample_rate),
                "duration_s": round(bgm_data.duration_s, 3),
                "volume_db": settings.bgm.volume_db,
                "ducking_db": settings.bgm.ducking_db,
                "fade_ms": settings.bgm.fade_ms,
                "looped": None,
            }
        for directory in settings.work_dirs.values():
            directory.mkdir(parents=True, exist_ok=True)

        if runner_obj is None:
            runner_obj = SaySegmentRunner(cfg, args, settings.work_dirs["recipes"])

        no_reuse = settings.force or bool(getattr(cfg, "no_cache", False))
        cache = SegmentCache(settings.work_dirs["segments"], enabled=not no_reuse)
        stats = {"total": 0, "synthesized": 0, "reused": 0}
        chapter_segments: list[list[np.ndarray]] = []
        chapter_rates: list[list[int]] = []
        chapter_languages: dict[int, set[str]] = {
            chapter.index: set() for chapter in project.chapters
        }
        segment_fingerprints: list[dict[str, Any]] = []

        for chapter in project.chapters:
            segments: list[np.ndarray] = []
            rates: list[int] = []
            ctx = contexts.get(chapter.index)
            chapter_fps = fingerprints.get(chapter.index)
            for paragraph_index, text in enumerate(chapter.paragraphs, 1):
                _check_cancel()
                expected = None
                if ctx is not None:
                    expected = fingerprint(
                        ctx,
                        text=text,
                        language=chapter.language,
                        instruct=chapter.instruct,
                    )
                elif chapter_fps is not None:
                    expected = chapter_fps[paragraph_index - 1]
                found = cache.lookup(chapter.index, paragraph_index, expected) if expected else None
                if found is not None:
                    data, sr = found
                    stats["reused"] += 1
                    languages = _sidecar_languages(cache, chapter.index, paragraph_index)
                else:
                    wav_path, _ = cache.paths(chapter.index, paragraph_index)
                    result = _synthesize(
                        runner_obj,
                        text=text,
                        out=wav_path,
                        voice=_chapter_voice(chapter, settings),
                        language=chapter.language,
                        instruct=chapter.instruct,
                        seed=int(getattr(cfg, "seed", 1234)),
                        cancel_check=cancel_event.is_set,
                        recipe_dir=settings.work_dirs["recipes"],
                        chapter=chapter.index,
                        paragraph=paragraph_index,
                    )
                    stats["synthesized"] += 1
                    try:
                        data, sr = audio_io.read_audio(wav_path, mono=True)
                    except InputError as exc:
                        raise ModelError(
                            f"fresh synthesis output is unreadable: {wav_path}: {exc.message}",
                            details={"path": str(wav_path)},
                        ) from exc
                    if data.size == 0 or int(sr) <= 0 or not np.all(np.isfinite(data)):
                        raise ModelError(
                            f"fresh synthesis output is invalid: {wav_path}",
                            details={"path": str(wav_path)},
                        )
                    languages = sorted(result.get("languages_actual") or [])
                    actual_device = result.get("device_actual")
                    actual_dtype = result.get("dtype_actual")
                    if not getattr(spec, "remote", False) and device:
                        if actual_device and str(actual_device) != str(device):
                            warnings.append(
                                f"chapter {chapter.index} paragraph {paragraph_index}: "
                                f"say reported device {actual_device!r} but the fingerprint "
                                f"used {device!r}"
                            )
                        if dtype and actual_dtype and normalize_dtype_name(actual_dtype) != dtype:
                            warnings.append(
                                f"chapter {chapter.index} paragraph {paragraph_index}: "
                                f"say reported dtype {actual_dtype!r} but the fingerprint "
                                f"used {dtype!r}"
                            )
                    if expected:
                        cache.write_sidecar(
                            chapter.index,
                            paragraph_index,
                            fingerprint=expected,
                            sample_rate=int(sr),
                            frames=int(data.shape[0]),
                            duration_s=round(data.shape[0] / float(sr), 3),
                            sha256=sha256_file(wav_path),
                            device_actual=result.get("device_actual"),
                            dtype_actual=result.get("dtype_actual"),
                            languages_actual=languages,
                        )
                stats["total"] += 1
                segment_fingerprints.append(
                    {
                        "chapter": chapter.index,
                        "paragraph": paragraph_index,
                        "fingerprint": expected,
                        "languages_actual": sorted(set(languages)),
                    }
                )
                chapter_languages[chapter.index].update(languages)
                segments.append(data)
                rates.append(int(sr))
                _check_cancel()
            chapter_segments.append(segments)
            chapter_rates.append(rates)

        assembled = audio.assemble(
            chapter_segments,
            chapter_rates,
            settings.sample_rate,
            settings.paragraph_ms,
            settings.chapter_ms,
        )
        # The master now owns its samples; release the per-segment arrays so a
        # long book holds one speech buffer, not segments + chapter buffers.
        del chapter_segments, chapter_rates
        segments = rates = None
        master = assembled.full
        if settings.bgm is not None:
            bgm_meta["looped"] = bool(bgm_data.samples.size < master.shape[0])
            master = audio.mix_bgm(
                master,
                bgm_data.samples,
                settings.sample_rate,
                volume_db=settings.bgm.volume_db,
                ducking_db=settings.bgm.ducking_db,
                fade_ms=settings.bgm.fade_ms,
            )
        for chapter in project.chapters:
            short = assembled.end_frames[chapter.index - 1] - assembled.start_frames[chapter.index - 1]
            if short < 0.1 * settings.sample_rate:
                warnings.append(
                    f"chapter {chapter.index} is shorter than 0.1 s "
                    f"({short / settings.sample_rate:.3f} s)"
                )

        loudness_records: dict[int, Optional[dict[str, Any]]] = {}
        verified_ok: Optional[bool] = True if settings.loudness_enabled else None
        if settings.loudness_enabled:
            for chapter in project.chapters:
                start = assembled.start_frames[chapter.index - 1]
                stop = assembled.end_frames[chapter.index - 1]
                raw = np.array(master[start:stop], dtype=np.float32, copy=True)
                if raw.size == 0:
                    warnings.append(f"chapter {chapter.index} is empty")
                    loudness_records[chapter.index] = None
                    continue
                lufs, reason = audio_qc.integrated_loudness(raw, settings.sample_rate)
                if lufs is None:
                    warnings.append(
                        f"chapter {chapter.index} loudness unavailable ({reason}); left unchanged"
                    )
                    loudness_records[chapter.index] = None
                    continue
                before = postprocess.measure(raw, settings.sample_rate)
                processed, _, applied, _, _, dsp_warnings = postprocess.process(
                    raw,
                    settings.sample_rate,
                    {"lufs": settings.target_lufs, "true_peak": settings.true_peak},
                )
                warnings.extend(dsp_warnings)
                processed = processed[:, 0] if processed.ndim == 2 else processed
                after = postprocess.measure(processed, settings.sample_rate)
                peak = after.get("true_peak_dbtp")
                if peak is None:
                    verified_ok = None
                    warnings.append(f"chapter {chapter.index} true peak is unavailable")
                elif peak > settings.true_peak + postprocess.TRUE_PEAK_TOLERANCE_DB:
                    verified_ok = False
                    warnings.append(
                        f"chapter {chapter.index} true peak {peak:.2f} dBTP exceeds "
                        f"{settings.true_peak + postprocess.TRUE_PEAK_TOLERANCE_DB:.2f}"
                    )
                master[start:stop] = processed
                loudness_records[chapter.index] = {
                    "before": before,
                    "after": after,
                    "gain_db": float(applied.get("gain_db", 0.0)),
                    "target_lufs": settings.target_lufs,
                }

        clipped = bool(
            master.size and (float(np.max(master)) > 1.0 or float(np.min(master)) < -1.0)
        )

        token = uuid.uuid4().hex
        final_tmp = settings.final.parent / f"{settings.final.name}.{token}.tmp"
        manifest_tmp = settings.manifest.parent / f"{settings.manifest.name}.{token}.tmp"
        temp_paths.extend([final_tmp, manifest_tmp])
        master_wav = settings.work_dirs["chapters"] / f"master.{token}.wav"
        metadata_path = settings.work_dirs["chapters"] / f"meta.{token}.txt"
        stem_paths.extend([master_wav, metadata_path])

        expected_duration = master.shape[0] / float(settings.sample_rate)
        if settings.fmt in LOSSLESS:
            encode.write_lossless(final_tmp, master, settings.sample_rate, settings.fmt)
            encoded_peak = None
        else:
            encode.write_lossless(master_wav, master, settings.sample_rate, "wav")
            if settings.fmt == "m4b":
                windows = encode.chapter_windows(assembled.start_ms, audio.ms_for_frame(master.shape[0], settings.sample_rate))
                chapters_meta = [
                    {"title": chapter.title} for chapter in project.chapters
                ]
                encode.write_m4b_metadata(
                    metadata_path,
                    title=settings.title,
                    author=settings.author,
                    album=settings.album,
                    chapters=chapters_meta,
                    windows=windows,
                )
            encode.encode_lossy(
                master_wav=master_wav,
                output=final_tmp,
                fmt=settings.fmt,
                bitrate_kbps=int(settings.bitrate_kbps or 128),
                metadata_path=metadata_path if settings.fmt == "m4b" else None,
            )
            encode.verify_duration(final_tmp, expected_duration)
            encoded_peak = encode.decode_peak(final_tmp, settings.sample_rate)
            if settings.loudness_enabled:
                peak_value = (encoded_peak or {}).get("true_peak_dbtp")
                if peak_value is None:
                    verified_ok = None
                    warnings.append("encoded true peak is unavailable")
                elif peak_value > settings.true_peak + postprocess.TRUE_PEAK_TOLERANCE_DB:
                    verified_ok = False
                    warnings.append(
                        f"encoded true peak {peak_value:.2f} dBTP exceeds "
                        f"{settings.true_peak + postprocess.TRUE_PEAK_TOLERANCE_DB:.2f}"
                    )

        manifest_data = _build_manifest(
            cfg=cfg,
            args=args,
            project=project,
            settings=settings,
            spec=spec,
            voices=voices,
            assembled=assembled,
            master=master,
            final=settings.final,
            bgm_meta=bgm_meta,
            loudness_records=loudness_records,
            verified_ok=verified_ok,
            encoded_peak=encoded_peak,
            segment_fingerprints=segment_fingerprints,
            stats=stats,
            chapter_languages=chapter_languages,
            warnings=warnings,
            clipped=clipped,
            final_size=int(final_tmp.stat().st_size),
            runtime_device=device,
            runtime_dtype=dtype,
        )
        atomic_write_text(
            manifest_tmp,
            json.dumps(manifest_data, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        )

        if not settings.keep_stems:
            for path in stem_paths:
                encode.remove_quietly(path)
            stem_paths.clear()

        result = manifest_mod.publish(
            final_tmp=final_tmp,
            manifest_tmp=manifest_tmp,
            final=settings.final,
            manifest=settings.manifest,
            journal_path=settings.journal,
            work_dirs_to_remove=list(settings.work_dirs.values()),
            keep_work_dirs=settings.keep_stems,
        )
        warnings.extend(result.warnings)
        temp_paths.clear()
        if result.interrupted_after_publish:
            eprint("interrupted after publish")
            return {"raw_stdout": True, "exit_code": 130}
        if warnings and not getattr(cfg, "quiet", False):
            for warning in warnings:
                eprint(f"warning: {warning}")

        return {
            "ok": True,
            "project": {"path": str(project.path), "sha256": project.sha256},
            "output": {
                "path": str(settings.final),
                "format": settings.fmt,
                "sample_rate": settings.sample_rate,
                "duration_s": round(master.shape[0] / float(settings.sample_rate), 3),
                "size": int(settings.final.stat().st_size),
                "encoder": encoder["name"],
                "bitrate_kbps": settings.bitrate_kbps,
            },
            "manifest": str(settings.manifest),
            "chapters": len(project.chapters),
            "paragraphs": project.paragraph_count,
            "segments": dict(stats),
            "loudness": manifest_data["loudness"],
            "warnings": warnings,
            "clipped": clipped,
        }
    except (KeyboardInterrupt, CancelledError) as exc:
        if runner_obj is not None and hasattr(runner_obj, "request_cancel"):
            try:
                runner_obj.request_cancel()
            except Exception:
                pass
        for path in temp_paths:
            encode.remove_quietly(path)
        if not settings.keep_stems:
            for path in stem_paths:
                encode.remove_quietly(path)
        try:
            manifest_mod.recover(
                settings.journal,
                expected_final=settings.final,
                expected_manifest=settings.manifest,
            )
        except Exception:
            pass
        if isinstance(exc, CancelledError):
            raise KeyboardInterrupt from exc
        raise
    finally:
        for path in temp_paths:
            encode.remove_quietly(path)
        if not settings.keep_stems:
            for path in stem_paths:
                encode.remove_quietly(path)
        if lock is not None:
            lock.release()
        if main_thread and previous_term is not None:
            try:
                signal.signal(signal.SIGTERM, previous_term)
            except (ValueError, OSError):
                pass


def _sidecar_languages(cache: SegmentCache, chapter: int, paragraph: int) -> list[str]:
    _, sidecar_path = cache.paths(chapter, paragraph)
    try:
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
        values = payload.get("languages_actual")
        if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
            return []
        return sorted(set(values))
    except (OSError, ValueError, TypeError, AttributeError):
        return []


def _synthesize(
    runner: Any,
    *,
    text: str,
    out: Path,
    voice: str,
    language: str,
    instruct: Optional[str],
    seed: int,
    cancel_check,
    recipe_dir: Path,
    chapter: int,
    paragraph: int,
) -> dict[str, Any]:
    kwargs = dict(
        text=text,
        out=out,
        voice=voice,
        language=language,
        instruct=instruct,
        seed=seed,
        cancel_check=cancel_check,
    )
    if isinstance(runner, SaySegmentRunner):
        kwargs["recipe_out"] = recipe_dir / f"{chapter:04d}-{paragraph:04d}.json"
    return runner.synthesize(**kwargs)


def _build_manifest(
    *,
    cfg,
    args,
    project: Project,
    settings: Settings,
    spec,
    voices: dict[int, Any],
    assembled: audio.Assembled,
    master: np.ndarray,
    final: Path,
    bgm_meta: Optional[dict[str, Any]],
    loudness_records: dict[int, Optional[dict[str, Any]]],
    verified_ok: Optional[bool],
    encoded_peak: Optional[dict[str, Any]],
    segment_fingerprints: list[dict[str, Any]],
    stats: dict[str, int],
    chapter_languages: dict[int, set[str]],
    warnings: list[str],
    clipped: bool,
    final_size: int,
    runtime_device: Optional[str],
    runtime_dtype: Optional[str],
) -> dict[str, Any]:
    from .. import __version__

    remote = bool(getattr(spec, "remote", False))
    dtype = None if remote else runtime_dtype
    chapters = []
    for chapter in project.chapters:
        resolution = voices[chapter.index]
        index = chapter.index - 1
        chapters.append(
            {
                "index": chapter.index,
                "title": chapter.title,
                "voice": _chapter_voice(chapter, settings),
                "voice_identity": resolution.identity,
                "model_id": resolution.model.model_id,
                "model_revision": resolution.model.revision,
                "model_fingerprint": resolution.model.fingerprint,
                "language": chapter.language,
                "languages_actual": sorted(chapter_languages.get(chapter.index, set())),
                "instruct": chapter.instruct,
                "source": {
                    "path": str(chapter.source_path) if chapter.source_path else None,
                    "sha256": chapter.source_sha256,
                    "inline": chapter.inline,
                },
                "paragraphs": len(chapter.paragraphs),
                "start_ms": assembled.start_ms[index],
                "end_ms": assembled.end_ms[index],
                "duration_s": audio.duration_s(assembled.end_frames[index] - assembled.start_frames[index], settings.sample_rate),
                "loudness": loudness_records.get(chapter.index),
            }
        )
    if settings.loudness_enabled:
        loudness = {
            "enabled": True,
            "target_lufs": settings.target_lufs,
            "true_peak_dbtp": settings.true_peak,
            "verified_ok": verified_ok,
            "encoded_true_peak_dbtp": (encoded_peak or {}).get("true_peak_dbtp"),
        }
    else:
        loudness = {"enabled": False}
    return {
        "schema_version": 1,
        "qtts_version": __version__,
        "project": {"path": str(project.path), "sha256": project.sha256},
        "output": {
            "path": str(final),
            "format": settings.fmt,
            "sample_rate": settings.sample_rate,
            "channels": 1,
            "duration_s": round(master.shape[0] / float(settings.sample_rate), 3),
            "size": int(final_size),
            "encoder": _encoder_info(settings.fmt, settings.bitrate_kbps)["name"],
            "bitrate_kbps": settings.bitrate_kbps,
        },
        "book": {
            "title": settings.title,
            "author": settings.author,
            "album": settings.album,
            "language": project.language,
        },
        "synthesis": {
            "seed": int(getattr(cfg, "seed", 1234)),
            "sampling": {
                **_sampling_defaults(),
            },
            "dtype": None if remote else dtype,
            "backend": {
                "name": spec.name,
                "identity": spec.identity,
                "remote": remote,
            },
            "normalization_version": _normalization_version(),
        },
        "bgm": bgm_meta,
        "loudness": loudness,
        "chapters": chapters,
        "segment_fingerprints": segment_fingerprints,
        "segments": dict(stats),
        "warnings": list(warnings),
        "clipped": bool(clipped),
    }


def _sampling_defaults() -> dict[str, Any]:
    from ..plan import SAMPLING_DEFAULTS
    from ..budget import DEFAULT_MAX_NEW_TOKENS

    values = dict(SAMPLING_DEFAULTS)
    values["max_new_tokens"] = int(DEFAULT_MAX_NEW_TOKENS)
    return values


def _normalization_version() -> int:
    from ..textnorm import NORMALIZATION_VERSION

    return NORMALIZATION_VERSION
