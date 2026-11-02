"""`qtts say` — one command over builtin / clone / design generations."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

from .. import __version__
from ..align_engine import MAX_ALIGN_SECONDS, AlignerEngine
from ..audio_io import (
    concat,
    duration_s,
    read_audio,
    resolve_output_format,
    write_audio,
    write_audio_stdout,
)
from ..cache import AudioCache
from ..catalog import language_supported, scan, select_aligner
from ..errors import CapabilityError, InputError, ModelError, UsageError
from ..plan import (
    GenerationPlan,
    attach_cache_stats,
    build_plan,
    rekey_plan,
    resolve_dtype_name,
)
from ..sidecar import build_recipe, recipe_path
from ..subtitles import build_cues, coverage, full_text, monotonic, normalize_formats, render, shift_cues
from ..tts_engine import TtsEngine
from ..utils import atomic_write_json, atomic_write_text, emit_json, sha256_text
from ..voice_store import VoiceStore, utc_now


# --------------------------------------------------------------------------- #
# Input
# --------------------------------------------------------------------------- #
def read_input(args) -> tuple[str, str]:
    """Return (text, source description). Supports TEXT, '-', --file, --script."""
    explicit = [name for name in ("text", "file", "script") if getattr(args, name, None)]
    if len(explicit) > 1:
        raise UsageError("provide only one of TEXT, --file or --script")
    if getattr(args, "text", None):
        text = args.text
        if text == "-":
            return sys.stdin.read(), "stdin"
        return text, "argument"
    if getattr(args, "file", None):
        path = Path(args.file).expanduser()
        if not path.is_file():
            raise InputError(f"input file not found: {path}")
        return path.read_text(encoding="utf-8"), str(path)
    if getattr(args, "script", None):
        path = Path(args.script).expanduser()
        if not path.is_file():
            raise InputError(f"script file not found: {path}")
        if path.suffix.lower() in (".yaml", ".yml", ".json"):
            raise UsageError(
                "structured multi-role scripts are P1; P0 accepts plain text via --script or --file"
            )
        return path.read_text(encoding="utf-8"), str(path)
    if not sys.stdin.isatty():
        return sys.stdin.read(), "stdin"
    raise UsageError("no input text; pass TEXT, '-', --file FILE or --script FILE")


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #
def run(cfg, args) -> dict[str, Any]:
    text, source = read_input(args)
    if not text.strip():
        raise InputError("input text is empty")

    models = scan(cfg.model_dir)
    if not models:
        raise ModelError(
            "no local models found; pass --model-dir DIR or run `qtts models pull`",
            details={"model_dir": str(cfg.model_dir) if cfg.model_dir else None},
        )
    store = VoiceStore(cfg.data_dir)

    out_target: str = args.out if args.out is not None else "-"
    subtitle_formats = normalize_formats(args.subtitle)
    sub_prefix = _subtitle_prefix(args, out_target)
    recipe_arg = getattr(args, "recipe_out", None)

    # stdout has exactly one consumer (codex P0-2). Validate before any load.
    consumers: list[str] = []
    if out_target == "-":
        consumers.append("--out -")
    if cfg.json:
        consumers.append("--json")
    if sub_prefix == "-":
        consumers.append("--subtitle-out-prefix -")
    if recipe_arg == "-":
        consumers.append("--recipe-out -")
    if len(consumers) > 1:
        raise UsageError(
            "stdout can carry only one product; got " + " + ".join(consumers) + "; choose one"
        )
    if sub_prefix == "-" and len(subtitle_formats) != 1:
        raise UsageError("--subtitle-out-prefix - supports exactly one subtitle format")
    if out_target == "-" and subtitle_formats and sub_prefix is None:
        raise UsageError("--out - cannot host subtitle files; pass --subtitle-out-prefix")

    fmt = resolve_output_format(None if out_target == "-" else out_target, args.format)

    plan = build_plan(
        cfg,
        text=text,
        voice_ref=args.voice,
        language=args.language,
        instruct=args.instruct,
        max_new_tokens=args.max_new_tokens,
        pause_ms=args.pause_ms,
        output=None if out_target == "-" else out_target,
        fmt=fmt,
        subtitle_formats=subtitle_formats,
        granularity=args.granularity,
        models=models,
        store=store,
        model_override=args.model,
    )

    # Subtitle prerequisites are checked here, before TTS weights load (P0-1).
    aligner_model = _prepare_aligner(models, plan) if subtitle_formats else None

    cache = AudioCache(cfg.audio_cache_dir, enabled=not cfg.no_cache)
    attach_cache_stats(plan, cache, "wav")

    recipe_target = _recipe_target(recipe_arg, out_target, cfg, plan)
    stdout_used = bool(consumers)

    if args.dry_run:
        return {
            "ok": True,
            "dry_run": True,
            "models_loaded": False,
            "plan": plan.to_dict(),
            "will_write": _will_write(out_target, recipe_target, sub_prefix, subtitle_formats),
            "aligner": _aligner_dict(aligner_model),
            "dependencies": {
                "tts": _module_available("qwen_tts"),
                "align": _module_available("qwen_asr") if subtitle_formats else None,
            },
            "device": cfg.device,
            "dtype": "auto" if cfg.dtype == "auto" else cfg.dtype,
            "memory_estimate": _estimate_memory(plan, aligner_model is not None),
        }

    started = time.perf_counter()
    need_model = cfg.no_cache or any(
        not cache.exists(seg.cache_key, "wav") for seg in plan.segments
    )
    engine = TtsEngine(cfg, plan.voice.model)
    prompt_items = None
    if need_model:
        engine.load()
        # The runtime dtype may differ from the platform guess; rebuild cache
        # keys so a CPU fp32 run never reads bf16 artefacts (P1-5).
        if engine.dtype_name and engine.dtype_name != plan.dtype_name:
            rekey_plan(plan, dtype=engine.dtype_name, device=engine.device)
            attach_cache_stats(plan, cache, "wav")
        if plan.voice.mode == "clone":
            prompt_items = engine.build_clone_prompt_items(store, plan.voice.voice_name or "")

    generated_at = utc_now()
    segment_audio = []
    segment_meta = []
    first_sr: Optional[int] = None

    for seg in plan.segments:
        cached_path = None if cfg.no_cache else cache.lookup(seg.cache_key, "wav")
        if cached_path is not None:
            audio, sr = read_audio(cached_path)
            gen_info = {"cached": True, "generate_ms": 0.0}
        else:
            audio, sr, gen_info = engine.generate(
                text=seg.text,
                language=seg.language,
                mode=plan.voice.mode,
                speaker=plan.voice.speaker,
                instruct=seg.instruct if plan.voice.mode in ("builtin", "design") else None,
                prompt_items=prompt_items,
                sampling=plan.sampling,
                seed=int(cfg.seed),
            )
            gen_info["cached"] = False
            if not cfg.no_cache:
                _cache_segment(cache, seg.cache_key, audio, sr)
        if first_sr is None:
            first_sr = sr
        elif sr != first_sr:
            raise ModelError(f"segment sample rate changed: {first_sr} -> {sr}")
        segment_audio.append(audio)
        segment_meta.append(
            {
                "id": seg.id,
                "cache_key": seg.cache_key,
                "text_sha256": sha256_text(seg.text),
                "language": seg.language,
                "instruct": seg.instruct,
                "pause_after_ms": seg.pause_after_ms,
                "duration_ms": int(round(duration_s(audio, sr) * 1000)),
                "cached": bool(gen_info.get("cached")),
                "generate_ms": gen_info.get("generate_ms"),
            }
        )

    if first_sr is None:
        raise ModelError("no audio generated")
    sample_rate = first_sr
    full = concat(segment_audio, sample_rate, [s.pause_after_ms for s in plan.segments])
    total_s = duration_s(full, sample_rate)

    warnings = list(plan.warnings)
    subtitle_files: dict[str, str] = {}
    align_info: dict[str, Any] = {}
    coverages: list[float] = []

    if subtitle_formats:
        assert aligner_model is not None
        aligner = AlignerEngine(cfg, aligner_model)
        cues = []
        offset = 0.0
        for seg, audio, meta in zip(plan.segments, segment_audio, segment_meta):
            seg_duration = duration_s(audio, sample_rate)
            if seg_duration > MAX_ALIGN_SECONDS:
                raise InputError(
                    f"segment {seg.id} is {seg_duration:.1f}s; the forced aligner "
                    f"supports at most {MAX_ALIGN_SECONDS:.0f}s (codex P0-1)",
                    details={"segment": seg.id, "duration_s": round(seg_duration, 3)},
                )
            items, info = aligner.align(audio, sample_rate, seg.text, seg.language)
            align_info = info
            seg_coverage = coverage(seg.text, items)
            coverages.append(seg_coverage)
            meta["align_coverage"] = round(seg_coverage, 4)
            if seg_coverage < 0.3:
                raise InputError(
                    f"forced alignment matched only {seg_coverage:.0%} of segment {seg.id}; "
                    "subtitle timing would be unreliable",
                    details={"segment": seg.id, "coverage": round(seg_coverage, 4)},
                )
            if seg_coverage < 0.8:
                warnings.append(
                    f"segment {seg.id} alignment coverage is {seg_coverage:.0%}; "
                    "unmatched characters reuse neighbouring timings"
                )
            seg_cues = build_cues(
                seg.text,
                items,
                granularity=args.granularity,
                fallback_duration=seg_duration,
                audio_duration=seg_duration,
            )
            cues.extend(shift_cues(seg_cues, offset))
            offset += seg_duration + seg.pause_after_ms / 1000.0
        cues = monotonic(cues, audio_duration=total_s)
        if full_text(cues) != plan.normalized_text:
            warnings.append("subtitle text differs from the normalized source text; check alignment")
        subtitle_files = _write_subtitles(cues, subtitle_formats, sub_prefix, plan, out_target)

    if out_target == "-":
        write_audio_stdout(full, sample_rate, fmt)
    else:
        write_audio(out_target, full, sample_rate, fmt)

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    recipe = build_recipe(
        qtts_version=__version__,
        text=plan.normalized_text,
        source=source,
        voice_ref=plan.voice.ref,
        voice={
            "prompt_sha256": plan.voice.prompt_sha256,
            "ref_audio_sha256": plan.voice.ref_audio_sha256,
        },
        model={
            "id": plan.voice.model.model_id,
            "revision": plan.voice.model.revision,
            "fingerprint": plan.voice.model.fingerprint,
            "dtype": engine.dtype_name or resolve_dtype_name(cfg),
            "path": str(plan.voice.model.path),
        },
        sampling=plan.sampling,
        segments=segment_meta,
        created_at=generated_at,
        seed=plan.seed,
        normalization={"version": plan.normalization_version},
        device=engine.device,
        instruct=plan.instruct,
        language=plan.language,
        subtitle={"formats": subtitle_formats, "granularity": args.granularity, "files": subtitle_files}
        if subtitle_formats
        else None,
        timing={
            "total_ms": round(elapsed_ms, 1),
            "model_load_ms": round(engine.load_ms or 0.0, 1),
            "generate_ms": round(sum(m.get("generate_ms") or 0.0 for m in segment_meta), 1),
        },
    )
    if recipe_target == "-":
        emit_json(recipe)
    elif recipe_target is not None:
        atomic_write_json(recipe_target, recipe)

    payload: dict[str, Any] = {
        "ok": True,
        "stdout_used": stdout_used,
        "output": out_target,
        "format": fmt,
        "sample_rate": sample_rate,
        "duration_s": round(total_s, 3),
        "voice": plan.voice.to_dict(),
        "language": plan.language,
        "seed": plan.seed,
        "segments": segment_meta,
        "sampling": plan.sampling,
        "cache": cache.stats.to_dict(),
        "device": engine.device or "cache",
        "dtype": engine.dtype_name or resolve_dtype_name(cfg),
        "timing": recipe["timing"],
        "recipe": recipe_target if recipe_target else None,
        "warnings": warnings,
    }
    if subtitle_formats:
        payload["subtitles"] = subtitle_files
        payload["align"] = align_info
        payload["align_coverage"] = round(min(coverages), 4) if coverages else None
    return payload


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _prepare_aligner(models, plan: GenerationPlan):
    """Select and validate the aligner before TTS weights load (codex P0-1)."""
    model = select_aligner(models)
    for seg in plan.segments:
        if not language_supported(model, seg.language):
            raise CapabilityError(
                f"forced aligner {model.name} does not support language {seg.language!r}",
                details={"model": model.name, "language": seg.language, "supported": model.languages},
            )
        if seg.estimated_duration_s > MAX_ALIGN_SECONDS:
            raise InputError(
                f"segment {seg.id} is estimated at {seg.estimated_duration_s:.1f}s; the "
                f"forced aligner supports at most {MAX_ALIGN_SECONDS:.0f}s",
                details={"segment": seg.id, "estimated_duration_s": round(seg.estimated_duration_s, 3)},
            )
    return model


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _aligner_dict(model) -> Optional[dict[str, Any]]:
    if model is None:
        return None
    return {"model": model.model_id, "languages": model.languages}


def _recipe_target(recipe_arg, out_target: str, cfg, plan: GenerationPlan) -> Optional[str]:
    if recipe_arg:
        return "-" if recipe_arg == "-" else str(Path(recipe_arg).expanduser())
    if out_target != "-":
        return str(recipe_path(out_target))
    # No audio file to sit beside: keep the recipe in the data dir (P0-6).
    digest = sha256_text(plan.normalized_text)[:16]
    return str(Path(cfg.data_dir) / "recipes" / f"{digest}.recipe.json")


def _estimate_memory(plan: GenerationPlan, with_aligner: bool) -> dict[str, Any]:
    """Rough peak-memory estimate for the dry-run report (PRD FR-P0-07)."""
    size = plan.voice.model.size
    weights_gb = 3.4 if size == "1b7" else 1.4
    runtime_gb = 1.5
    aligner_gb = 1.2 if with_aligner else 0.0
    peak = weights_gb + runtime_gb + aligner_gb
    if peak >= 8.0:
        risk = "high"
    elif peak >= 4.5:
        risk = "medium"
    else:
        risk = "low"
    return {
        "models": [plan.voice.model.model_id] + (["Qwen3-ForcedAligner-0.6B"] if with_aligner else []),
        "weights_gb": round(weights_gb, 1),
        "estimated_peak_gb": round(peak, 1),
        "risk": risk,
        "note": "estimate only; measured peak RSS is reported by `qtts doctor --bench`",
    }


def _cache_segment(cache: AudioCache, key: str, audio, sr: int) -> None:
    handle = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    handle.close()
    tmp = Path(handle.name)
    try:
        write_audio(tmp, audio, sr, "wav")
        cache.store(key, tmp, "wav")
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass


def _subtitle_prefix(args, out_target: str) -> Optional[str]:
    if args.subtitle_out_prefix:
        return args.subtitle_out_prefix
    if out_target == "-":
        return None
    return str(Path(out_target).with_suffix(""))


def _will_write(
    out_target: str,
    recipe_target: Optional[str],
    sub_prefix: Optional[str],
    subtitle_formats: list[str],
) -> list[str]:
    paths: list[str] = []
    if out_target != "-":
        paths.append(out_target)
    if recipe_target not in (None, "-"):
        paths.append(str(recipe_target))
    if sub_prefix and sub_prefix != "-":
        for fmt in subtitle_formats:
            paths.append(f"{sub_prefix}.{fmt}")
    return paths


def _write_subtitles(cues, formats, prefix: Optional[str], plan: GenerationPlan, out_target: str) -> dict[str, str]:
    if prefix is None:
        raise UsageError("no subtitle output prefix available")
    files: dict[str, str] = {}
    if prefix == "-":
        if len(formats) != 1:
            raise UsageError("--subtitle-out-prefix - supports exactly one format")
        sys.stdout.write(render(cues, formats[0], language=plan.language, granularity=plan.granularity))
        sys.stdout.flush()
        files[formats[0]] = "-"
        return files
    for fmt in formats:
        path = Path(f"{prefix}.{fmt}")
        atomic_write_text(path, render(cues, fmt, language=plan.language, granularity=plan.granularity))
        files[fmt] = str(path)
    return files


def format_result(payload: dict[str, Any]) -> str:
    if payload.get("dry_run"):
        plan = payload["plan"]
        lines = [
            f"dry-run: {len(plan['segments'])} segment(s), voice={plan['voice']['ref']} "
            f"model={plan['voice']['model']}",
            f"estimated audio: {plan['estimated_duration_s']}s  cache hit rate: {plan['cache']['hit_rate']:.0%}",
        ]
        for seg in plan["segments"]:
            lines.append(
                f"  {seg['id']} [{seg['language']}] {seg['estimated_tokens']} tok ~{seg['estimated_duration_s']}s: "
                f"{seg['text'][:48]}{'...' if len(seg['text']) > 48 else ''}"
            )
        if plan.get("warnings"):
            lines.extend(f"warning: {w}" for w in plan["warnings"])
        memory = payload.get("memory_estimate")
        if memory:
            lines.append(
                f"memory: ~{memory['estimated_peak_gb']}GB peak ({memory['risk']} risk), "
                f"models={','.join(memory['models'])}"
            )
        if payload.get("will_write"):
            lines.append("will write: " + ", ".join(payload["will_write"]))
        lines.append("model loading skipped")
        return "\n".join(lines)

    lines = [
        f"wrote {payload['output']} ({payload['duration_s']}s, {payload['sample_rate']} Hz, {payload['format']})",
        f"voice={payload['voice']['ref']} seed={payload['seed']} device={payload['device']}/{payload['dtype']}",
        f"timing: total {payload['timing']['total_ms']}ms (load {payload['timing']['model_load_ms']}ms, "
        f"generate {payload['timing']['generate_ms']}ms)",
    ]
    if payload.get("recipe"):
        lines.append(f"recipe: {payload['recipe']}")
    for fmt, path in (payload.get("subtitles") or {}).items():
        lines.append(f"subtitle[{fmt}]: {path}")
    for warning in payload.get("warnings") or []:
        lines.append(f"warning: {warning}")
    return "\n".join(lines)
