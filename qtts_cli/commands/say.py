"""`qtts say` — one command over builtin / clone / design generations."""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import re
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

from .. import __version__
from ..align_engine import MAX_ALIGN_SECONDS, AlignerEngine
from ..audio_io import (
    concat,
    duration_s,
    read_audio,
    resolve_output_format,
    silence_ms,
    write_audio,
    write_audio_stdout,
)
from ..audio_qc import analyze as analyze_audio
from ..backends.base import (
    ensure_instance_capabilities,
    ensure_instance_protocol,
)
from ..backends.registry import (
    BUILTIN_NAMES,
    preflight_voice_ref,
    resolve_backend,
    run_preflight,
)
from ..backends.transformers import TransformersBackend
from ..tts_engine import TtsEngine
from ..cache import AudioCache
from ..catalog import language_supported, scan, select_aligner
from ..errors import (
    CancelledError,
    CapabilityError,
    InputError,
    ModelError,
    QttsError,
    UsageError,
)
from ..lexicon import Lexicon, MapBlock, spoken_items_to_display
from ..plan import (
    DEFAULT_PAUSE_MS,
    GenerationPlan,
    SAMPLING_DEFAULTS,
    attach_cache_stats,
    build_plan,
    rekey_plan,
    resolve_dtype_name,
)
from ..quality import evaluate as evaluate_quality
from ..quality import thresholds_from_args as qc_thresholds_from_args
from ..quality import validate_thresholds as validate_qc_thresholds
from ..budget import DEFAULT_MAX_NEW_TOKENS
from ..script import compile_script
from ..sidecar import build_recipe, recipe_path
from ..subtitle_qc import check_subtitles, raise_quality_error
from ..takes import takes_request
from ..subtitles import (
    build_cues,
    coverage,
    ensure_full_text,
    monotonic,
    normalize_formats,
    policy_from_args,
    render,
    shift_cues,
)
from ..utils import atomic_write_json, atomic_write_text, emit_json, sha256_text
from ..voice_store import VoiceStore, utc_now
from .qc import precheck_speaker_similarity as qc_precheck_speaker_similarity
from .qc import report_payload as qc_report_payload
from .qc import speaker_similarity as qc_speaker_similarity
from .qc import validate_reference_voice as validate_qc_reference_voice


# --------------------------------------------------------------------------- #
# Input
# --------------------------------------------------------------------------- #
def read_input(args) -> tuple[str, str]:
    """Return (text, source description). Supports TEXT, '-', --stdin, --file, --script."""
    explicit = [name for name in ("text", "file", "script") if getattr(args, name, None)]
    if getattr(args, "stdin", False):
        explicit.append("stdin")
    if len(explicit) > 1:
        raise UsageError("provide only one of TEXT, --stdin, --file or --script")
    if getattr(args, "stdin", False):
        return sys.stdin.read(), "stdin"
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
            # Structured scripts are compiled by ``run`` before this helper is
            # consulted; reaching here means a caller passed the wrong mode.
            return "", str(path)
        return path.read_text(encoding="utf-8"), str(path)
    if not sys.stdin.isatty():
        return sys.stdin.read(), "stdin"
    raise UsageError("no input text; pass TEXT, '-', --file FILE or --script FILE")


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #
def _check_cancel(cancel_check) -> None:
    if cancel_check is not None and cancel_check():
        raise CancelledError("cancelled")


def run(
    cfg,
    args,
    *,
    engine_cache: Optional[dict[str, Any]] = None,
    aligner_cache: Optional[dict[str, AlignerEngine]] = None,
    cancel_check=None,
) -> dict[str, Any]:
    takes_spec = takes_request(args)
    if takes_spec and getattr(args, "jsonl", False):
        raise UsageError("--takes/--select cannot be combined with --jsonl")
    script_path = getattr(args, "script", None)
    structured = bool(
        script_path and Path(script_path).suffix.lower() in (".yaml", ".yml", ".json")
    )
    if not structured and not getattr(args, "voice", None):
        raise UsageError("--voice is required (builtin:NAME | voice:NAME | design:DESCRIPTION)")
    if structured:
        conflicts = [
            name
            for name in ("text", "file")
            if getattr(args, name, None)
        ]
        if getattr(args, "stdin", False):
            conflicts.append("--stdin")
        if conflicts:
            raise UsageError(
                "--script cannot be combined with TEXT, --stdin or --file; "
                f"got {', '.join(conflicts)}"
            )
        text, source = "", str(Path(script_path).expanduser())
    else:
        text, source = read_input(args)
        if not text.strip():
            raise InputError("input text is empty")

    # CLI options are explicit defaults for structured scripts (see below);
    # missing attributes come from the argparse SUPPRESS sentinels.
    cli_language = getattr(args, "language", None)
    cli_instruct = getattr(args, "instruct", None)
    cli_max_new_tokens = getattr(args, "max_new_tokens", None)
    cli_pause_ms = getattr(args, "pause_ms", None)
    language_arg = "auto" if cli_language is None else cli_language
    max_new_tokens_arg = (
        DEFAULT_MAX_NEW_TOKENS if cli_max_new_tokens is None else cli_max_new_tokens
    )
    pause_ms_arg = DEFAULT_PAUSE_MS if cli_pause_ms is None else cli_pause_ms

    # Metadata-only backend resolution; nothing is created or loaded yet.
    spec = resolve_backend(getattr(args, "backend", None), cfg, args)
    if takes_spec and spec.name != "transformers":
        raise UsageError(
            "--takes/--select require the local transformers backend; "
            f"{spec.name} cannot provide takes or selection metrics"
        )

    policy = policy_from_args(args)
    # TTS models come from the backend; local scanning is only for the aligner,
    # speaker-similarity QC and takes, which always use local runtimes.
    tts_models = spec.models
    local_models = scan(cfg.model_dir)
    store = VoiceStore(cfg.data_dir)
    lexicon_arg = getattr(args, "lexicon", None)
    # Batch reuses this entry point and may pass an already-loaded Lexicon.
    lexicon = (
        lexicon_arg
        if isinstance(lexicon_arg, Lexicon)
        else Lexicon.load(lexicon_arg)
    )

    out_target: str = args.out if args.out is not None else "out.wav"
    subtitle_formats = normalize_formats(args.subtitle)
    sub_prefix = _subtitle_prefix(args, out_target)
    recipe_arg = getattr(args, "recipe_out", None)

    qc_enabled = bool(getattr(args, "qc", False))
    qc_reference_voice = getattr(args, "reference_voice", None)
    takes_speaker_sim = bool(takes_spec and "speaker-sim" in takes_spec[1])
    qc_thresholds = qc_thresholds_from_args(args) if qc_enabled else {}
    if qc_enabled:
        validate_qc_thresholds(qc_thresholds, has_text=True)
    elif qc_reference_voice and not takes_speaker_sim:
        raise UsageError(
            "--reference-voice requires --qc or --select speaker-sim"
        )
    if qc_reference_voice and spec.name != "transformers":
        raise UsageError(
            "--reference-voice requires the local transformers backend; "
            f"{spec.name} cannot compute speaker similarity"
        )

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

    if not structured:
        # Kind validation must win over model selection (exit 4, not 3), but
        # pure output-parameter errors above keep priority.
        preflight_voice_ref(args.voice, spec)

    if qc_reference_voice:
        # Profile and Base/runtime prechecks run after pure parameter/output
        # validation but before any generation or audio write (t10 P1-4/P2).
        validate_qc_reference_voice(VoiceStore(cfg.data_dir), qc_reference_voice)
        if (
            not getattr(args, "dry_run", False)
            and qc_enabled
            and not takes_speaker_sim
        ):
            qc_precheck_speaker_similarity(
                cfg, local_models, getattr(args, "model", None)
            )

    if structured:
        plan = compile_script(
            cfg,
            script_path,
            models=tts_models,
            store=store,
            lexicon=lexicon,
            model_override=args.model,
            voice_default=getattr(args, "voice", None),
            backend_spec=spec,
            backend_identity=spec.identity,
            require_clone_prompt=spec.require_clone_prompt,
            speaker_policy=spec.speaker_policy(),
            output=None if out_target == "-" else out_target,
            fmt=args.format,
            subtitle_formats=subtitle_formats,
            granularity=args.granularity,
            cli_language=cli_language,
            cli_instruct=cli_instruct,
            cli_max_new_tokens=cli_max_new_tokens,
            cli_pause_ms=cli_pause_ms,
        )
        if args.format:
            fmt = resolve_output_format(
                None if out_target == "-" else out_target, args.format
            )
        elif out_target != "-" and Path(out_target).suffix.lower() in (".wav", ".flac"):
            fmt = resolve_output_format(out_target, None)
        else:
            fmt = resolve_output_format(None, plan.format)
        plan.format = fmt
    else:
        fmt = resolve_output_format(None if out_target == "-" else out_target, args.format)
        plan = build_plan(
            cfg,
            text=text,
            voice_ref=args.voice,
            language=language_arg,
            instruct=cli_instruct,
            max_new_tokens=max_new_tokens_arg,
            pause_ms=pause_ms_arg,
            output=None if out_target == "-" else out_target,
            fmt=fmt,
            subtitle_formats=subtitle_formats,
            granularity=args.granularity,
            models=tts_models,
            store=store,
            model_override=args.model,
            lexicon=lexicon,
            backend_identity=spec.identity,
            require_clone_prompt=spec.require_clone_prompt,
            speaker_policy=spec.speaker_policy(),
        )

    # Backend-specific checks (API key, ref audio, offline) run after the plan
    # exists and before any create/load/aligner/cache preparation.
    run_preflight(plan, cfg, args, spec)
    # Only keys the backend actually forwards may reach generate(); an
    # explicitly requested unsupported knob is a capability error before load.
    supported_sampling = _validate_sampling_capabilities(plan, spec)
    _check_cancel(cancel_check)

    # Subtitle prerequisites are checked here, before TTS weights load (P0-1).
    aligner_model = _prepare_aligner(local_models, plan) if subtitle_formats else None

    cache = AudioCache(cfg.audio_cache_dir, enabled=not cfg.no_cache)
    attach_cache_stats(plan, cache, "wav")

    recipe_target = _recipe_target(recipe_arg, out_target, cfg, plan)
    stdout_used = bool(consumers)

    if args.dry_run:
        payload = {
            "ok": True,
            "dry_run": True,
            "models_loaded": False,
            "plan": plan.to_dict(),
            "will_write": _will_write(out_target, recipe_target, sub_prefix, subtitle_formats),
            "aligner": _aligner_dict(aligner_model),
            "backend": {
                "name": spec.name,
                "model": plan.voice.model.model_id,
                "remote": spec.remote,
                "identity": spec.identity,
                "configured": spec.configured() if spec.configured is not None else None,
            },
            "dependencies": {
                "tts": (
                    _module_available(spec.runtime_module)
                    if spec.runtime_module
                    else None
                ),
                "align": _module_available("qwen_asr") if subtitle_formats else None,
            },
            "device": "remote" if spec.remote else cfg.device,
            "dtype": None if spec.remote else ("auto" if cfg.dtype == "auto" else cfg.dtype),
            "memory_estimate": (
                {"remote": True, "estimated_peak_gb": None}
                if spec.remote
                else _estimate_memory(plan, aligner_model is not None)
            ),
            "qc": {
                "enabled": qc_enabled,
                "thresholds": qc_thresholds,
                "reference_voice": qc_reference_voice,
            },
        }
        if takes_spec:
            from ..takes_run import dry_run_details

            takes_count, criteria = takes_spec
            payload["takes"] = dry_run_details(
                cfg,
                args,
                plan=plan,
                models=local_models,
                voice_store=store,
                takes_count=takes_count,
                criteria=criteria,
                fmt=fmt,
                out_target=out_target,
                recipe_target=recipe_target,
                subtitle_formats=subtitle_formats,
                sub_prefix=sub_prefix,
            )
        return payload

    if takes_spec:
        from ..takes_run import run as run_takes

        takes_count, criteria = takes_spec
        return run_takes(
            cfg,
            args,
            plan=plan,
            models=local_models,
            voice_store=store,
            aligner_model=aligner_model,
            cache=cache,
            out_target=out_target,
            fmt=fmt,
            subtitle_formats=subtitle_formats,
            sub_prefix=sub_prefix,
            recipe_arg=recipe_arg,
            recipe_target=recipe_target,
            qc_enabled=qc_enabled,
            qc_reference_voice=qc_reference_voice,
            qc_thresholds=qc_thresholds,
            source=source,
            takes_count=takes_count,
            criteria=criteria,
            stdout_used=stdout_used,
        )

    started = time.perf_counter()
    need_model = cfg.no_cache or any(
        not cache.exists(seg.cache_key, "wav") for seg in plan.segments
    )
    _check_cancel(cancel_check)
    engine_slot: dict[str, Any] = (
        engine_cache if engine_cache is not None else {}
    )
    prompt_cache: dict[tuple[str, str], Any] = {}
    last_engine: Optional[Any] = None
    load_ms_total = 0.0

    def slot_key(voice) -> str:
        # Backend identity is part of the slot: a remote client and a local
        # engine for the same model must never share state.
        return f"{spec.name}:{voice.model.path}|{cfg.device}"

    def acquire(voice):
        """One live TTS model slot; release the previous model before loading."""
        nonlocal last_engine, load_ms_total
        key = slot_key(voice)
        for old_key, old_engine in list(engine_slot.items()):
            if old_key != key:
                _release_backend(old_engine, spec)
                engine_slot.pop(old_key, None)
                prompt_cache.clear()
        engine = engine_slot.get(key)
        if engine is None:
            engine = _create_backend(spec, cfg, voice.model)
            ensure_instance_protocol(engine, spec)
            ensure_instance_capabilities(engine, spec)
            engine_slot[key] = engine
        if getattr(engine, "model", None) is None:
            _invoke_backend(spec, "load()", engine.load)
            load_ms_total += float(engine.load_ms or 0.0)
        # The runtime dtype may differ from the platform guess; rebuild cache
        # keys so a CPU fp32 run never reads bf16 artefacts (P1-5).
        if engine.dtype_name and engine.dtype_name != plan.dtype_name:
            rekey_plan(plan, dtype=engine.dtype_name, device=engine.device)
            attach_cache_stats(plan, cache, "wav")
        last_engine = engine
        return engine

    def prompt_for(engine, voice):
        if voice.mode != "clone":
            return None
        key = (str(voice.model.path), json.dumps(voice.identity, sort_keys=True))
        if key not in prompt_cache:
            prompt_cache[key] = _invoke_backend(
                spec,
                "build_clone_prompt_items()",
                engine.build_clone_prompt_items,
                store,
                voice.voice_name or "",
            )
        return prompt_cache[key]

    if need_model:
        first_missing = next(
            (
                seg
                for seg in plan.segments
                if cfg.no_cache or not cache.exists(seg.cache_key, "wav")
            ),
            plan.segments[0],
        )
        acquire(first_missing.voice or plan.voice)

    generated_at = utc_now()
    segment_audio = []
    segment_meta = []
    first_sr: Optional[int] = None

    for seg in plan.segments:
        _check_cancel(cancel_check)
        voice = seg.voice or plan.voice
        cached_path = None if cfg.no_cache else cache.lookup(seg.cache_key, "wav")
        if cached_path is not None:
            audio, sr = read_audio(cached_path)
            gen_info = {"cached": True, "generate_ms": 0.0}
            last_engine = last_engine or engine_slot.get(slot_key(voice))
        else:
            engine = acquire(voice)
            sampling = {
                key: value
                for key, value in plan.sampling.items()
                if key in supported_sampling
            }
            if seg.max_new_tokens is not None and "max_new_tokens" in supported_sampling:
                sampling["max_new_tokens"] = int(seg.max_new_tokens)
            audio, sr, gen_info = _invoke_backend(
                spec,
                "generate()",
                engine.generate,
                text=seg.text,
                language=seg.language,
                mode=voice.mode,
                speaker=voice.speaker,
                instruct=seg.instruct if voice.mode in ("builtin", "design") else None,
                prompt_items=prompt_for(engine, voice),
                sampling=sampling,
                seed=int(seg.seed if seg.seed is not None else cfg.seed),
            )
            gen_info["cached"] = False
            # Record the exact keys the backend received (filtered sampling).
            gen_info["sampling_forwarded"] = dict(sampling)
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
                "source_segment_id": seg.source_segment_id,
                "cache_key": seg.cache_key,
                "text_sha256": sha256_text(seg.text),
                "display_sha256": sha256_text(seg.display_text or seg.text),
                "voice": (voice.to_dict() if seg.voice is not None else voice.ref),
                "language": seg.language,
                "instruct": seg.instruct,
                "seed": seg.seed,
                "max_new_tokens": seg.max_new_tokens,
                "pause_after_ms": seg.pause_after_ms,
                "duration_ms": int(round(duration_s(audio, sr) * 1000)),
                "cached": bool(gen_info.get("cached")),
                "generate_ms": gen_info.get("generate_ms"),
                "sampling_forwarded": gen_info.get("sampling_forwarded"),
            }
        )

    if first_sr is None:
        raise ModelError("no audio generated")
    sample_rate = first_sr
    core = concat(segment_audio, sample_rate, [s.pause_after_ms for s in plan.segments])
    parts = [core]
    if plan.leading_pause_ms:
        parts.insert(0, silence_ms(int(plan.leading_pause_ms), sample_rate))
    if plan.trailing_pause_ms:
        parts.append(silence_ms(int(plan.trailing_pause_ms), sample_rate))
    full = concat(parts, sample_rate) if len(parts) > 1 else core
    total_s = duration_s(full, sample_rate)

    warnings = list(plan.warnings)
    subtitle_files: dict[str, str] = {}
    align_info: dict[str, Any] = {}
    coverages: list[float] = []
    boundary_stats: dict[str, int] = {
        "sentences": 0,
        "splits": 0,
        "hard_cuts": 0,
        "merges": 0,
    }
    repair_stats: dict[str, int] = {}
    qc_report = None
    qc_failure = None

    if subtitle_formats:
        assert aligner_model is not None
        aligner_key = f"{aligner_model.path}|{cfg.device}|{plan.dtype_name}"
        if aligner_cache is not None:
            aligner = aligner_cache.get(aligner_key)
            if aligner is None:
                aligner = AlignerEngine(cfg, aligner_model)
                aligner_cache[aligner_key] = aligner
        else:
            aligner = AlignerEngine(cfg, aligner_model)
        cues = []
        offset = plan.leading_pause_ms / 1000.0
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
            display_text = seg.display_text or seg.text
            mapping = seg.spoken_to_display
            if mapping is None:
                mapping = [
                    MapBlock(i, i + 1, i, i + 1) for i in range(len(seg.text))
                ]
            display_items = spoken_items_to_display(
                display_text, seg.text, mapping, items
            )
            seg_cues = build_cues(
                display_text,
                display_items,
                granularity=args.granularity,
                fallback_duration=seg_duration,
                audio_duration=seg_duration,
                policy=policy,
                parent_ids=(f"seg:{seg.source_segment_id or seg.id}",),
                warnings=warnings,
                stats=boundary_stats,
                repair_stats=repair_stats,
            )
            cues.extend(shift_cues(seg_cues, offset))
            offset += seg_duration + seg.pause_after_ms / 1000.0
        cues = monotonic(cues, audio_duration=total_s, repair_stats=repair_stats)
        ensure_full_text(cues, plan.display_text or plan.normalized_text)
        qc_report = check_subtitles(
            cues,
            granularity=args.granularity,
            audio_duration=total_s,
            policy=policy,
            repairs=repair_stats,
        )
        for violation in qc_report.warnings:
            warnings.append(
                f"subtitle QC warning: {violation.message} (cue {violation.cue})"
            )
        if qc_report.ok:
            _check_cancel(cancel_check)
            subtitle_files = _write_subtitles(cues, subtitle_formats, sub_prefix, plan, out_target)
        else:
            # t10 contract: never write failing subtitles. Audio and recipe are
            # kept as an audit trail; the command still exits 5.
            qc_failure = qc_report

    _check_cancel(cancel_check)
    if out_target == "-":
        write_audio_stdout(full, sample_rate, fmt)
    else:
        write_audio(out_target, full, sample_rate, fmt)

    qc_result: Optional[dict[str, Any]] = None
    qc_violations: list[dict[str, Any]] = []
    if qc_enabled:
        # QC sees the final audio, including leading/trailing pauses.
        qc_metrics = analyze_audio(
            full,
            sample_rate,
            text=plan.normalized_text or plan.display_text,
            language=plan.language,
        )
        if qc_reference_voice:
            qc_store = VoiceStore(cfg.data_dir)
            validate_qc_reference_voice(qc_store, qc_reference_voice)
            # Only a loaded local transformers Base engine may be reused for
            # speaker embeddings; remote clients have no such capability.
            reusable = (
                last_engine
                if last_engine is not None
                and spec.name == "transformers"
                and getattr(last_engine, "model", None) is not None
                and last_engine.model_info.kind == "clone"
                else None
            )
            if reusable is None and last_engine is not None:
                _release_backend(last_engine, spec)
            qc_metrics["speaker_similarity"] = qc_speaker_similarity(
                cfg,
                local_models,
                qc_store,
                qc_reference_voice,
                full,
                sample_rate,
                getattr(args, "model", None),
                reusable_engine=reusable,
            )
            qc_metrics["unavailable"].pop("speaker_similarity", None)
        qc_violations = evaluate_quality(qc_metrics, qc_thresholds)
        qc_result = qc_report_payload(
            None if out_target == "-" else out_target,
            qc_metrics,
            qc_thresholds,
            qc_violations,
            qc_reference_voice,
        )

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    if spec.remote:
        # Remote backends have no local dtype/device; never report a platform guess.
        runtime_dtype = None
        runtime_device = "remote"
    else:
        runtime_dtype = (
            getattr(last_engine, "dtype_name", None) if last_engine else None
        ) or plan.dtype_name
        runtime_device = (
            getattr(last_engine, "device", None) if last_engine else None
        ) or cfg.device
    backend_payload = {
        "name": spec.name,
        "model": plan.voice.model.model_id,
        "remote": spec.remote,
        "identity": spec.identity,
    }
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
            "dtype": (
                None
                if spec.remote
                else (runtime_dtype or resolve_dtype_name(cfg))
            ),
            "path": str(plan.voice.model.path),
        },
        sampling=plan.sampling,
        segments=segment_meta,
        created_at=generated_at,
        seed=plan.seed,
        normalization={"version": plan.normalization_version},
        device=runtime_device,
        instruct=plan.instruct,
        language=plan.language,
        subtitle={
            "formats": subtitle_formats,
            "granularity": args.granularity,
            "files": subtitle_files,
            "policy": policy.to_dict(),
            "boundaries": dict(boundary_stats),
            "qc": qc_report.to_dict() if qc_report is not None else None,
        }
        if subtitle_formats
        else None,
        timing={
            "total_ms": round(elapsed_ms, 1),
            "model_load_ms": round(load_ms_total, 1),
            "generate_ms": round(sum(m.get("generate_ms") or 0.0 for m in segment_meta), 1),
        },
        backend=backend_payload,
    )
    voices = []
    models_used = []
    seen_voices: set[tuple[str, str]] = set()
    seen_models: set[str] = set()
    for seg in plan.segments:
        voice = seg.voice or plan.voice
        voice_key = (voice.ref, voice.model.model_id)
        if voice_key not in seen_voices:
            seen_voices.add(voice_key)
            voices.append(voice.to_dict())
        if voice.model.model_id not in seen_models:
            seen_models.add(voice.model.model_id)
            models_used.append(voice.model.to_dict())
    recipe["voices"] = voices
    recipe["models"] = models_used
    if plan.script_source:
        recipe["script_source"] = plan.script_source
    if plan.lexicon_sha256:
        recipe["lexicon_sha256"] = plan.lexicon_sha256
    if subtitle_formats and qc_report is not None:
        recipe["subtitle_qc"] = qc_report.to_dict()
    if qc_result is not None:
        recipe["qc"] = qc_result
    _check_cancel(cancel_check)
    if recipe_target == "-":
        emit_json(recipe)
    elif recipe_target is not None:
        atomic_write_json(recipe_target, recipe)
    if qc_failure is not None:
        raise_quality_error(qc_failure)

    payload: dict[str, Any] = {
        "ok": True,
        "stdout_used": stdout_used,
        "output": out_target,
        "format": fmt,
        "sample_rate": sample_rate,
        "duration_s": round(total_s, 3),
        "voice": plan.voice.to_dict(),
        "voices": voices,
        "models": models_used,
        "language": plan.language,
        "seed": plan.seed,
        "segments": segment_meta,
        "sampling": plan.sampling,
        "cache": cache.stats.to_dict(),
        "backend": backend_payload,
        "device": runtime_device or "cache",
        "dtype": (
            None if spec.remote else (runtime_dtype or resolve_dtype_name(cfg))
        ),
        "timing": recipe["timing"],
        "recipe": recipe_target if recipe_target else None,
        "warnings": warnings,
    }
    if subtitle_formats:
        payload["subtitles"] = subtitle_files
        payload["align"] = align_info
        payload["align_coverage"] = round(min(coverages), 4) if coverages else None
        payload["subtitle_qc"] = qc_report.to_dict() if qc_report is not None else None
        payload["subtitle_boundaries"] = dict(boundary_stats)
    if qc_result is not None:
        payload["qc"] = qc_result
        if qc_violations:
            payload["ok"] = False
            payload["exit_code"] = 5
    return payload


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _is_plugin_backend(spec) -> bool:
    return spec.name not in BUILTIN_NAMES


def _invoke_backend(spec, operation, callback, *args, **kwargs):
    """Run one backend call; wrap unexpected plugin exceptions as model errors.

    Built-in backends keep their existing error handling unchanged.
    """
    if not _is_plugin_backend(spec):
        return callback(*args, **kwargs)
    try:
        return callback(*args, **kwargs)
    except QttsError:
        raise
    except Exception as exc:
        raise ModelError(
            f"backend plugin {spec.name!r} {operation} failed: "
            f"{type(exc).__name__}: {exc}",
            details={"backend": spec.name, "operation": operation},
        ) from exc


def _release_backend(engine, spec) -> None:
    """Release a backend; plugin release failures never break a good run."""
    if not _is_plugin_backend(spec):
        engine.release()
        return
    try:
        engine.release()
    except Exception:
        return


def _create_backend(spec, cfg, model):
    """Create a backend instance, keeping ``TtsEngine`` patchable in tests."""
    if spec.name == "transformers":
        return TransformersBackend(cfg, model, engine=TtsEngine(cfg, model))
    try:
        return spec.create(cfg, model)
    except QttsError:
        raise
    except Exception as exc:
        raise ModelError(
            f"backend {spec.name} create() failed: "
            f"{type(exc).__name__}: {exc}",
            details={"backend": spec.name},
        ) from exc


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
    # Hash the whole generation request so identical text with a different
    # voice/seed/sampling cannot overwrite another run's recipe.
    request = {
        "text": plan.display_text or plan.normalized_text,
        "voice": plan.voice.identity,
        "voice_ref": plan.voice.ref,
        "model": plan.voice.model.model_id,
        "seed": plan.seed,
        "sampling": plan.sampling,
        "language": plan.language,
        "instruct": plan.instruct,
        "lexicon_sha256": plan.lexicon_sha256,
        "script_source": plan.script_source,
        "segments": [[seg.id, seg.text, seg.cache_key] for seg in plan.segments],
        "leading_pause_ms": plan.leading_pause_ms,
        "trailing_pause_ms": plan.trailing_pause_ms,
    }
    digest = sha256_text(
        json.dumps(request, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )[:32]
    return str(Path(cfg.data_dir) / "recipes" / f"{digest}.recipe.json")


def _explicit_sampling_keys(plan: GenerationPlan) -> set[str]:
    """Keys the caller actually asked for rather than plan defaults."""
    requested: set[str] = set()
    for key, value in plan.sampling.items():
        if key == "max_new_tokens":
            if int(value) != int(DEFAULT_MAX_NEW_TOKENS):
                requested.add(key)
        elif key not in SAMPLING_DEFAULTS or value != SAMPLING_DEFAULTS[key]:
            requested.add(key)
    for seg in plan.segments:
        if (
            seg.max_new_tokens is not None
            and int(seg.max_new_tokens) != int(DEFAULT_MAX_NEW_TOKENS)
        ):
            requested.add("max_new_tokens")
    return requested


def _validate_sampling_capabilities(
    plan: GenerationPlan, spec
) -> frozenset[str]:
    supported = frozenset(spec.capabilities.sampling)
    unsupported = sorted(_explicit_sampling_keys(plan) - supported)
    if unsupported:
        supported_text = ", ".join(sorted(supported)) or "none"
        raise CapabilityError(
            f"backend {spec.name} does not support sampling option(s): "
            f"{', '.join(unsupported)}; supported: {supported_text}",
            details={
                "backend": spec.name,
                "unsupported": unsupported,
                "supported": sorted(supported),
            },
        )
    return supported


def _estimate_memory(plan: GenerationPlan, with_aligner: bool) -> dict[str, Any]:
    """Rough peak-memory estimate for the dry-run report (PRD FR-P0-07)."""
    voices = [seg.voice or plan.voice for seg in plan.segments]
    model_infos = {}
    for voice in voices:
        model_infos[voice.model.model_id] = voice.model
    sizes = [model.size for model in model_infos.values()]
    weights_gb = 3.4 if "1b7" in sizes else 1.4
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
        "models": sorted(model_infos)
        + (["Qwen3-ForcedAligner-0.6B"] if with_aligner else []),
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


_SAFE_STEM = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def _safe_output_stem(record_id: str) -> str:
    """Use the id only when it is a safe ASCII path component; else hash it."""
    if _SAFE_STEM.match(record_id) and record_id not in (".", ".."):
        return record_id
    return f"qtts-{sha256_text(record_id)[:16]}"


def _jsonl_emit(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _jsonl_error(
    record_id: Any,
    code: str,
    message: str,
    details: Optional[dict[str, Any]] = None,
) -> None:
    error: dict[str, Any] = {"code": code, "message": message}
    if details:
        error["details"] = details
    _jsonl_emit({"id": record_id, "status": "error", "error": error})


def run_jsonl(cfg, args) -> dict[str, Any]:
    """`say --stdin --jsonl`: one request per line, one result per line."""
    if takes_request(args) is not None:
        raise UsageError("--takes/--select cannot be combined with --jsonl")
    if not getattr(args, "stdin", False):
        raise UsageError("--jsonl requires --stdin")
    if getattr(args, "json", False):
        raise UsageError("--jsonl cannot be combined with --json")
    if getattr(args, "out", None):
        raise UsageError("--jsonl uses --out-dir; --out is not allowed")
    if getattr(args, "recipe_out", None):
        raise UsageError("--jsonl cannot use --recipe-out")
    if getattr(args, "subtitle_out_prefix", None):
        raise UsageError("--jsonl cannot use --subtitle-out-prefix")
    if getattr(args, "dry_run", False):
        raise UsageError("--jsonl cannot use --dry-run")
    if getattr(args, "text", None) not in (None, "-"):
        raise UsageError("--jsonl reads requests from stdin; a positional TEXT is not allowed")
    if getattr(args, "file", None) or getattr(args, "script", None):
        raise UsageError("--jsonl reads requests from stdin; --file/--script are not allowed")

    # Resolve the backend once and fail fast on missing remote credentials
    # before consuming stdin.  No runtime import, create or load happens here.
    spec = resolve_backend(getattr(args, "backend", None), cfg, args)
    if spec.configured is not None and not spec.configured():
        raise ModelError(
            f"backend {spec.name} is not configured; set its credentials or "
            "choose another backend",
            details={"backend": spec.name},
        )

    out_dir_arg = getattr(args, "out_dir", None)
    if out_dir_arg == "-":
        raise UsageError("--out-dir cannot be '-'")
    out_dir = Path(out_dir_arg or ".").expanduser()
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise InputError(f"cannot create --out-dir {out_dir}: {exc}") from exc

    engine_cache: dict[str, Any] = {}
    aligner_cache: dict[str, AlignerEngine] = {}
    seen_ids: set[str] = set()
    used_stems: dict[str, str] = {}
    used_paths: dict[str, str] = {}
    total = 0
    ok_records = 0
    failed = False

    for line_number, raw_line in enumerate(sys.stdin, 1):
        line = raw_line.strip()
        if not line:
            continue
        total += 1
        try:
            request = json.loads(line)
        except json.JSONDecodeError as exc:
            failed = True
            _jsonl_error(None, "invalid_json", f"invalid JSON on line {line_number}: {exc.msg}",
                         {"line": line_number, "column": exc.colno})
            continue
        if not isinstance(request, dict):
            failed = True
            _jsonl_error(None, "invalid_request", "each JSONL line must be an object")
            continue

        record_id = request.get("id")
        if not isinstance(record_id, str) or not record_id:
            failed = True
            _jsonl_error(record_id if record_id is not None else None,
                         "invalid_request", "id is required and must be a non-empty string")
            continue
        if record_id in seen_ids:
            failed = True
            _jsonl_error(record_id, "duplicate_id", "id must be unique within the stream")
            continue
        seen_ids.add(record_id)

        stem = _safe_output_stem(record_id)
        stem_key = stem.casefold()
        if stem_key in used_stems:
            failed = True
            _jsonl_error(
                record_id,
                "id_filename_conflict",
                f"output filename {stem!r} collides with id {used_stems[stem_key]!r} "
                "(case-insensitive)",
            )
            continue
        used_stems[stem_key] = record_id

        text = request.get("text")
        if not isinstance(text, str) or not text.strip():
            failed = True
            _jsonl_error(record_id, "invalid_request", "text is required and must be non-empty")
            continue
        voice = request.get("voice") or getattr(args, "voice", None)
        if not isinstance(voice, str) or not voice:
            failed = True
            _jsonl_error(record_id, "invalid_request",
                         "voice is required (record.voice or --voice)")
            continue
        language = request.get("language", getattr(args, "language", "auto"))
        if not isinstance(language, str) or not language:
            failed = True
            _jsonl_error(record_id, "invalid_request", "language must be a non-empty string")
            continue
        instruct = request.get("instruct", getattr(args, "instruct", None))
        if instruct is not None and not isinstance(instruct, str):
            failed = True
            _jsonl_error(record_id, "invalid_request", "instruct must be a string or null")
            continue
        seed = request.get("seed", cfg.seed)
        if not isinstance(seed, int) or isinstance(seed, bool):
            failed = True
            _jsonl_error(record_id, "invalid_request", "seed must be an integer")
            continue
        subtitle = request.get("subtitle", getattr(args, "subtitle", None))
        if subtitle is not None and not isinstance(subtitle, (str, list, tuple)):
            failed = True
            _jsonl_error(record_id, "invalid_request",
                         "subtitle must be a string or a list of formats")
            continue
        out_field = request.get("out")
        if out_field == "-":
            failed = True
            _jsonl_error(record_id, "invalid_request", "per-record out:'-' is not allowed")
            continue
        if out_field is not None and not isinstance(out_field, str):
            failed = True
            _jsonl_error(record_id, "invalid_request", "out must be a string path")
            continue
        requested_format = request.get("format") or getattr(args, "format", None)
        if requested_format is not None and requested_format not in ("wav", "flac"):
            failed = True
            _jsonl_error(record_id, "invalid_request", "format must be wav or flac")
            continue
        if out_field:
            out_path = Path(out_field).expanduser()
            output_format = requested_format
        else:
            output_format = requested_format or "wav"
            out_path = out_dir / f"{stem}.{output_format}"

        # Canonicalize before generation: two ids (including relative aliases)
        # must never silently overwrite each other's output (t10 fix #10).
        try:
            canonical_out = str(out_path.expanduser().resolve()).casefold()
        except (OSError, RuntimeError) as exc:
            failed = True
            _jsonl_error(record_id, "invalid_request",
                         f"cannot resolve output path {out_path}: {exc}")
            continue
        if canonical_out in used_paths:
            failed = True
            _jsonl_error(
                record_id,
                "output_path_conflict",
                f"output path {out_path} is already used by id "
                f"{used_paths[canonical_out]!r}",
                {"path": str(out_path), "conflicts_with": used_paths[canonical_out]},
            )
            continue
        used_paths[canonical_out] = record_id

        record_args = SimpleNamespace(
            text=text,
            stdin=False,
            file=None,
            script=None,
            voice=voice,
            language=language,
            instruct=instruct,
            out=str(out_path),
            format=output_format,
            subtitle=subtitle,
            subtitle_out_prefix=None,
            recipe_out=None,
            granularity=getattr(args, "granularity", "sentence"),
            max_new_tokens=getattr(args, "max_new_tokens", 2048),
            pause_ms=getattr(args, "pause_ms", 300),
            model=getattr(args, "model", None),
            backend=getattr(args, "backend", None),
            backend_model=getattr(args, "backend_model", None),
            lexicon=getattr(args, "lexicon", None),
            dry_run=False,
            yes=True,
            max_cue_chars=getattr(args, "max_cue_chars", 42),
            max_line_width=getattr(args, "max_line_width", 42),
            max_cps=getattr(args, "max_cps", 20.0),
            min_cue_seconds=getattr(args, "min_cue_seconds", 0.8),
            max_cue_seconds=getattr(args, "max_cue_seconds", 7.0),
            max_merge_gap_ms=getattr(args, "max_merge_gap_ms", 300),
            no_subtitle_qc=getattr(args, "no_subtitle_qc", False),
        )
        record_cfg = dataclasses.replace(cfg, seed=int(seed), json=False)
        try:
            result = run(record_cfg, record_args, engine_cache=engine_cache,
                         aligner_cache=aligner_cache)
        except QttsError as exc:
            failed = True
            details = dict(exc.details)
            details["exit_code"] = exc.code
            _jsonl_error(record_id, exc.kind, exc.message, details)
            continue
        except Exception as exc:  # one bad request must not kill the stream
            failed = True
            _jsonl_error(record_id, "internal", str(exc))
            continue

        for warning in result.get("warnings") or []:
            print(f"warning: {warning}", file=sys.stderr)
            sys.stderr.flush()
        metrics = {
            "duration_s": result.get("duration_s"),
            "sample_rate": result.get("sample_rate"),
            "timing": result.get("timing"),
            "cache": result.get("cache"),
            "warnings": result.get("warnings") or [],
        }
        if result.get("subtitles"):
            metrics["subtitles"] = result["subtitles"]
            metrics["subtitle_qc"] = result.get("subtitle_qc")
            metrics["subtitle_boundaries"] = result.get("subtitle_boundaries")
        _jsonl_emit(
            {
                "id": record_id,
                "status": "ok",
                "output": result.get("output"),
                "recipe": result.get("recipe"),
                "metrics": metrics,
            }
        )
        ok_records += 1

    if total == 0:
        raise UsageError("--jsonl received no requests on stdin")
    return {
        "ok": not failed,
        "raw_stdout": True,
        "exit_code": 6 if failed else 0,
        "records": total,
        "ok_records": ok_records,
    }


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
            if memory.get("remote"):
                lines.append("memory: remote backend (not estimated locally)")
            else:
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
    takes = payload.get("takes")
    if takes:
        lines.append(
            f"takes: job={takes['job_id']} selected={takes['selected']} "
            f"criteria={','.join(takes.get('criteria') or [])} "
            f"{'reused' if takes.get('reused') else 'generated'}"
        )
    if payload.get("recipe"):
        lines.append(f"recipe: {payload['recipe']}")
    for fmt, path in (payload.get("subtitles") or {}).items():
        lines.append(f"subtitle[{fmt}]: {path}")
    for warning in payload.get("warnings") or []:
        lines.append(f"warning: {warning}")
    return "\n".join(lines)
