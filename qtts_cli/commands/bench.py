"""`qtts bench` — reproducible local TTS benchmarks (FR-P2-06).

The command layer owns argument validation, model/voice resolution, runtime
preflight, the single-resident TTS -> ASR -> Base lifecycle, gates and report
serialization.  Measurement math lives in :mod:`qtts_cli.benchmark`.
"""

from __future__ import annotations

import gc
import importlib.metadata
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any, Optional

from .. import __version__
from ..audio_io import read_audio
from ..benchmark import (
    SEGMENTER_VERSION,
    batch_cases,
    builtin_longform_text,
    case_plan_payload,
    corpus_payload,
    execute,
    longform_cases,
    score_cer,
    score_speaker,
    smoke_cases,
)
from ..budget import DEFAULT_MAX_NEW_TOKENS
from ..capabilities import VoiceRef, validate_capabilities
from ..catalog import (
    ModelInfo,
    language_supported,
    looks_like_explicit_path,
    scan,
    select,
    select_asr,
)
from ..commands.qc import base_model, validate_reference_voice
from ..errors import CapabilityError, InputError, ModelError, QttsError, UsageError
from ..plan import SAMPLING_DEFAULTS, resolve_voice
from ..qc import qc_language
from ..quality import embedding_array, write_report
from ..textnorm import NORMALIZATION_VERSION
from ..utils import peak_rss_bytes
from ..voice_store import VoiceStore

SUITES = ("smoke", "batch", "longform")
AGGREGATION = {
    "rtf": "median_per_run",
    "warm_first": "median_per_run",
    "wall_rtf": "total_wall_over_total_audio",
}
METRIC_KEYS = (
    "load_ms",
    "cold_first_segment_ms",
    "warm_first_segment_ms",
    "wall_s",
    "generate_s",
    "audio_s",
    "rtf",
    "wall_rtf",
    "segments_per_run",
    "segments_total",
    "segments_per_s",
    "process_peak_rss_bytes",
    "peak_rss_delta_bytes",
    "device_memory_mb",
    "device_memory_kind",
    "cer",
    "cer_distance",
    "cer_units",
    "speaker_similarity",
    "speaker_segments_scored",
    "speaker_segments_total",
)


# --------------------------------------------------------------------------- #
# Runtime seams (monkeypatched by tests; module import stays runtime-free)
# --------------------------------------------------------------------------- #
def _engine_factory(cfg, model: ModelInfo):
    from ..tts_engine import TtsEngine

    return TtsEngine(cfg, model)


def _asr_factory(cfg, model: ModelInfo):
    from ..asr_engine import AsrEngine

    return AsrEngine(cfg, model)


def _preflight_asr_runtime() -> None:
    from ..asr_engine import preflight_asr_runtime

    preflight_asr_runtime()


def _preflight_speaker(cfg, model: ModelInfo) -> None:
    engine = _engine_factory(cfg, model)
    probe = getattr(engine, "require_speaker_embedding", None)
    if probe is not None:
        probe()


def _clear_model_cache() -> None:
    from ..tts_engine import clear_model_cache

    clear_model_cache()


def _clear_asr_cache() -> None:
    from ..asr_engine import clear_asr_cache

    clear_asr_cache()


def _prepare_device_memory():
    """Reset CUDA peak stats and return a post-run memory reader when available."""
    try:
        import torch
    except Exception:
        return None
    try:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

            def read_cuda():
                return (
                    torch.cuda.max_memory_allocated() / (1024.0 * 1024.0),
                    "cuda_peak",
                )

            return read_cuda
    except Exception:
        pass
    try:
        if torch.backends.mps.is_available():

            def read_mps():
                return (
                    torch.mps.current_allocated_memory() / (1024.0 * 1024.0),
                    "mps_current",
                )

            return read_mps
    except Exception:
        pass
    return None


# --------------------------------------------------------------------------- #
# Validation and resolution
# --------------------------------------------------------------------------- #
def _validate_args(args) -> tuple[int, int, bool]:
    suite = args.suite
    if suite not in SUITES:
        raise UsageError(f"unknown suite: {suite!r}")

    text_explicit = getattr(args, "text", None) is not None
    file_explicit = getattr(args, "text_file", None) is not None
    if suite != "longform" and (text_explicit or file_explicit):
        raise UsageError("--text/--text-file are only valid with --suite longform")

    count_explicit = hasattr(args, "count")
    count = int(getattr(args, "count", 8))
    if count_explicit and suite != "batch":
        raise UsageError("--count is only valid with --suite batch")
    if not 1 <= count <= 64:
        raise UsageError("--count must be within [1, 64]")

    runs = int(getattr(args, "runs", 1))
    if not 1 <= runs <= 10:
        raise UsageError("--runs must be within [1, 10]")

    if getattr(args, "voice", None) is not None:
        _parse_voice(args.voice)

    cer = bool(args.cer)
    asr_model = getattr(args, "asr_model", None)
    if asr_model is not None and not cer:
        raise UsageError("--asr-model requires --cer")
    speaker_sim = bool(args.speaker_sim)
    reference_voice = getattr(args, "reference_voice", None)
    if reference_voice is not None and not speaker_sim:
        raise UsageError("--reference-voice requires --speaker-sim")
    if speaker_sim and not reference_voice:
        raise UsageError("--speaker-sim requires --reference-voice")
    if args.max_cer is not None:
        if not cer:
            raise UsageError("--max-cer requires --cer")
        if not math.isfinite(float(args.max_cer)) or float(args.max_cer) < 0.0:
            raise UsageError("--max-cer must be finite and >= 0")
    if args.min_speaker_similarity is not None:
        if not speaker_sim:
            raise UsageError("--min-speaker-similarity requires --speaker-sim")
        value = float(args.min_speaker_similarity)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise UsageError("--min-speaker-similarity must be within [0, 1]")

    report = getattr(args, "report", None)
    if report is not None:
        if str(report) == "-":
            raise UsageError("--report - is not supported by bench")
        if args.dry_run:
            raise UsageError("--dry-run and --report are mutually exclusive")
    return count, runs, count_explicit


def _parse_voice(raw: Optional[str]) -> Optional[VoiceRef]:
    if raw is None:
        return None
    prefix, separator, value = str(raw).partition(":")
    prefix = prefix.strip().lower()
    value = value.strip()
    if not separator or prefix not in ("builtin", "voice") or not value:
        raise UsageError(
            "--voice must be builtin:NAME or voice:NAME; design: is not supported by bench"
        )
    return VoiceRef(prefix, value)


def _strict_find_model(models: list[ModelInfo], spec: str) -> tuple[ModelInfo, str]:
    """Resolve --model without silently picking an ambiguous suffix match."""
    needle = str(spec).strip()
    path = Path(needle).expanduser()
    if path.is_dir():
        from ..catalog import find_model

        try:
            info = find_model(models, needle)
        except QttsError as exc:
            raise ModelError(
                f"cannot use --model {needle!r}: {exc.message}",
                details={"model": needle},
            ) from exc
        return info, str(info.path)
    if looks_like_explicit_path(needle):
        raise ModelError(
            f"model path not found: {needle}",
            details={"model": needle},
        )

    exact = [m for m in models if m.name == needle or m.model_id == needle]
    if len(exact) > 1:
        raise UsageError(
            f"model spec {needle!r} is ambiguous",
            details={"available": [m.name for m in exact]},
        )
    if len(exact) == 1:
        return exact[0], exact[0].name
    suffix = [
        m
        for m in models
        if m.name.endswith(needle) or m.model_id.endswith(needle)
    ]
    if len(suffix) > 1:
        raise UsageError(
            f"model spec {needle!r} is ambiguous",
            details={"available": [m.name for m in suffix]},
        )
    if len(suffix) == 1:
        return suffix[0], suffix[0].name
    raise ModelError(
        f"model not found: {needle}",
        details={"model": needle, "available": [m.name for m in models]},
    )


def _resolve_tts(
    models: list[ModelInfo], args
) -> tuple[ModelInfo, VoiceRef, Optional[str]]:
    parsed = _parse_voice(getattr(args, "voice", None))
    spec = getattr(args, "model", None)
    override: Optional[str] = None
    if spec is not None:
        model, override = _strict_find_model(models, spec)
        if model.kind not in ("builtin", "clone", "design"):
            raise ModelError(
                f"model {model.name} is not a TTS model (kind={model.kind})",
                details={"model": model.name, "kind": model.kind},
            )
    elif parsed is not None:
        try:
            model = select(models, parsed.mode)
        except QttsError as exc:
            raise ModelError(exc.message, details={"kind": parsed.mode}) from exc
    else:
        try:
            model = select(models, "builtin")
        except QttsError as exc:
            raise ModelError(exc.message, details={"kind": "builtin"}) from exc

    if parsed is None:
        if model.kind != "builtin":
            raise UsageError(
                "an explicit --model that is not a builtin model requires --voice"
            )
        if not model.speakers:
            raise InputError(
                f"model {model.name} has no built-in speakers; pass --voice voice:NAME"
            )
        parsed = VoiceRef("builtin", model.speakers[0])
    return model, parsed, override


def _validate_clone_assets(store: VoiceStore, name: str) -> dict[str, Any]:
    meta = store.require(name)
    if meta.get("kind") != "clone":
        raise InputError(
            f"voice {name} is not a clone profile (kind={meta.get('kind')})",
            details={"voice": name, "kind": meta.get("kind")},
        )
    if store.prompt_sha256(name) is None:
        raise InputError(
            f"voice {name} has no prompt.safetensors; re-add the voice",
            details={"voice": name},
        )
    return meta


def _resolve_voice_for_suite(
    models: list[ModelInfo],
    store: VoiceStore,
    ref: VoiceRef,
    args,
    override: Optional[str],
):
    voice = resolve_voice(models, store, ref, model_override=override)
    if voice.mode == "builtin":
        validate_capabilities(
            voice.model,
            mode="builtin",
            language=(
                None if args.language.lower() == "auto" else args.language
            ),
        )
    else:
        _validate_clone_assets(store, ref.value)
        validate_capabilities(
            voice.model,
            mode="clone",
            language=(
                None if args.language.lower() == "auto" else args.language
            ),
            icl=voice.icl,
            ref_text=voice.ref_text,
        )
    return voice


def _validate_case_capabilities(voice, cases) -> None:
    """Check every case's own language before any runtime/weights are touched.

    ``--language auto`` means the suite mixes Chinese and English cases, so the
    single ``language=None`` check in :func:`_resolve_voice_for_suite` is not
    sufficient.
    """
    for case in cases:
        validate_capabilities(
            voice.model,
            mode=voice.mode,
            language=case.language,
            icl=voice.icl,
            ref_text=voice.ref_text,
        )


def _validate_asr_case_languages(asr_model: ModelInfo, cases) -> None:
    """Reject unsupported ASR languages from model metadata before TTS loads."""
    for case in cases:
        language = qc_language(case.language, case.text)
        if not language_supported(asr_model, language):
            raise CapabilityError(
                f"ASR model {asr_model.name} does not support language {language!r}",
                details={
                    "language": language,
                    "case": case.id,
                    "model": asr_model.name,
                    "supported": asr_model.languages,
                },
            )


def _read_text_file(path: str) -> str:
    target = Path(path).expanduser()
    try:
        text = target.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise InputError(
            f"cannot read text file {target}: {exc}", details={"path": str(target)}
        ) from exc
    if not text.strip():
        raise InputError(
            f"text file is empty: {target}", details={"path": str(target)}
        )
    return text


def _reference_embedding(
    store: VoiceStore, name: str
) -> tuple[Any, str]:
    """Prompt embedding first; fall back to ref.wav (source kept in the report)."""
    if store.prompt_sha256(name) is not None:
        try:
            tensors = store.load_prompt_tensors(name)
        except Exception as exc:
            raise InputError(
                f"cannot read prompt for voice {name}: {exc}",
                details={"voice": name},
            ) from exc
        embedding = embedding_array(tensors.get("ref_spk_embedding"))
        if embedding is not None:
            return embedding, "prompt"
    if (store.dir_for(name) / "ref.wav").is_file():
        return None, "ref_wav"
    raise InputError(
        f"voice {name} has no reference embedding or ref.wav",
        details={"voice": name},
    )


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _request_label(raw: Optional[str], model: Optional[ModelInfo]) -> Optional[str]:
    """Keep user-facing names, but never leak an absolute model path."""
    if raw is None or model is None:
        return None
    try:
        if Path(raw).expanduser().is_dir():
            return model.model_id
    except OSError:
        pass
    return str(raw)


def _model_meta(model: Optional[ModelInfo]) -> Optional[dict[str, Any]]:
    if model is None:
        return None
    return {
        "name": model.name,
        "id": model.model_id,
        "revision": model.revision,
        "kind": model.kind,
        "fingerprint": model.fingerprint,
    }


def _empty_metrics() -> dict[str, Any]:
    return {key: None for key in METRIC_KEYS}


def _sanitize_numbers(mapping: dict[str, Any]) -> None:
    """Replace non-finite floats with ``None`` before gates and serialization.

    Canonical JSON is written with ``allow_nan=False``; a NaN/Inf metric would
    otherwise surface as a generic usage error instead of an unavailable value.
    """
    for key, value in list(mapping.items()):
        if isinstance(value, float) and not math.isfinite(value):
            mapping[key] = None


def _runtime_payload() -> Optional[dict[str, Any]]:
    torch_module = sys.modules.get("torch")
    torch_version = getattr(torch_module, "__version__", None)
    try:
        transformers_version = importlib.metadata.version("transformers")
    except Exception:
        transformers_version = None
    return {
        "python": platform.python_version(),
        "torch": torch_version,
        "transformers": transformers_version,
    }


def _request(
    args,
    voice_ref: str,
    *,
    text_source: str,
    count: int,
    model_label: Optional[str],
    asr_model_label: Optional[str],
) -> dict[str, Any]:
    return {
        "suite": args.suite,
        "model": model_label,
        "voice": voice_ref,
        "language": args.language,
        "runs": int(args.runs),
        "count": count if args.suite == "batch" else None,
        "text_source": text_source,
        "cer": bool(args.cer),
        "asr_model": asr_model_label,
        "speaker_sim": bool(args.speaker_sim),
        "reference_voice": getattr(args, "reference_voice", None),
        "max_cer": getattr(args, "max_cer", None),
        "min_speaker_similarity": getattr(args, "min_speaker_similarity", None),
    }


def _release_engine(engine) -> None:
    if engine is None:
        return
    try:
        engine.model = None
    except Exception:
        pass
    for name in ("prompt", "prompt_items", "_prompt"):
        if hasattr(engine, name):
            try:
                setattr(engine, name, None)
            except Exception:
                pass
    _clear_model_cache()
    del engine
    gc.collect()


def _release_asr(asr) -> None:
    if asr is None:
        return
    release = getattr(asr, "release", None)
    if release is not None:
        release()
    _clear_asr_cache()


def _payload(
    args,
    *,
    voice_ref: str,
    text_source: str,
    count: int,
    model_label: Optional[str],
    asr_model_label: Optional[str],
    models: dict[str, Any],
    cases: list[dict[str, Any]],
    runs: list[dict[str, Any]],
    metrics: dict[str, Any],
    sampling: dict[str, Any],
    seed: int,
    corpus: dict[str, Any],
    warnings: list[str],
    scoring_ms: Optional[float],
    runtime: Optional[dict[str, Any]],
    reference_embedding_source: Optional[str],
    requirements: Optional[dict[str, Any]] = None,
    gate_failures: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    failures = gate_failures or []
    payload: dict[str, Any] = {
        "schema_version": 1,
        "qtts_version": __version__,
        "normalization_version": NORMALIZATION_VERSION,
        "ok": not failures,
        "suite": args.suite,
        "dry_run": bool(args.dry_run),
        "request": _request(
            args,
            voice_ref,
            text_source=text_source,
            count=count,
            model_label=model_label,
            asr_model_label=asr_model_label,
        ),
        "models": models,
        "runtime": runtime,
        "sampling": sampling,
        "seed": int(seed),
        "corpus": corpus,
        "aggregation": dict(AGGREGATION),
        "scoring_ms": scoring_ms,
        "runs_count": len(runs),
        "metrics": metrics,
        "runs": runs,
        "cases": cases,
        "warnings": list(warnings),
        "reference_embedding_source": reference_embedding_source,
        "gate_failures": failures,
    }
    if requirements is not None:
        payload["requirements"] = requirements
    return payload


# --------------------------------------------------------------------------- #
# Command entry point
# --------------------------------------------------------------------------- #
def run(cfg, args) -> dict[str, Any]:
    # Process peak RSS baseline must precede any runtime import.
    baseline_rss = peak_rss_bytes()

    count, runs, _ = _validate_args(args)

    text_source = "builtin"
    longform_text: Optional[str] = None
    if args.suite == "longform":
        if getattr(args, "text", None) is not None:
            longform_text = args.text
            text_source = "text"
        elif getattr(args, "text_file", None) is not None:
            longform_text = _read_text_file(args.text_file)
            text_source = "text_file"
        else:
            longform_text = builtin_longform_text(args.language)
        if not str(longform_text).strip():
            raise InputError("benchmark text is empty")

    models = scan(cfg.model_dir)
    store = VoiceStore(cfg.data_dir)
    tts_model, parsed_voice, override = _resolve_tts(models, args)

    asr_model: Optional[ModelInfo] = None
    if args.cer:
        try:
            asr_model = select_asr(models, getattr(args, "asr_model", None))
        except QttsError:
            raise
    model_label = _request_label(getattr(args, "model", None), tts_model)
    asr_model_label = _request_label(getattr(args, "asr_model", None), asr_model)
    speaker_model: Optional[ModelInfo] = None
    if args.speaker_sim:
        # The TTS Base can score its own output; otherwise pick a Base
        # independently.  --model is the TTS override, not a Base override.
        speaker_model = (
            tts_model
            if tts_model.kind == "clone"
            else base_model(models, None)
        )
        validate_reference_voice(store, args.reference_voice)

    plan = None
    if args.suite == "longform":
        assert longform_text is not None
        cases, plan = longform_cases(
            cfg,
            text=longform_text,
            voice_ref=parsed_voice.raw,
            language=args.language,
            models=models,
            store=store,
            model_override=override,
        )
        voice = plan.voice
        if voice.model.model_id != tts_model.model_id:
            raise ModelError(
                "plan resolved a different TTS model than --model",
                details={"model": tts_model.model_id, "planned": voice.model.model_id},
            )
        if voice.mode == "clone":
            _validate_clone_assets(store, parsed_voice.value)
        sampling = dict(plan.sampling)
        seed = int(plan.seed)
        warnings = list(plan.warnings)
        normalization_version: Optional[int] = plan.normalization_version
        segmenter_version: Optional[int] = SEGMENTER_VERSION
    else:
        voice = _resolve_voice_for_suite(
            models, store, parsed_voice, args, override
        )
        if args.suite == "smoke":
            cases = smoke_cases(args.language)
        else:
            cases = batch_cases(args.language, count)
            if not cases:
                raise UsageError(
                    f"no batch cases for language {args.language!r}"
                )
        sampling = dict(SAMPLING_DEFAULTS)
        sampling["max_new_tokens"] = DEFAULT_MAX_NEW_TOKENS
        seed = int(cfg.seed)
        warnings = []
        normalization_version = None
        segmenter_version = None

    if not cases:
        raise UsageError(f"no cases selected for suite {args.suite!r}")

    # Capability checks are metadata-only and must finish before any engine
    # factory, runtime preflight or ``load()`` call.
    _validate_case_capabilities(voice, cases)
    if asr_model is not None:
        _validate_asr_case_languages(asr_model, cases)

    model_payload = {
        "tts": _model_meta(voice.model),
        "asr": _model_meta(asr_model),
        "speaker": _model_meta(speaker_model),
    }
    corpus = corpus_payload(
        args.suite,
        source=text_source,
        language=args.language,
        cases=cases,
        normalization_version=normalization_version,
        segmenter_version=segmenter_version,
    )

    if args.dry_run:
        requirements = {
            "tts_model": True,
            "asr_model": asr_model is not None,
            "speaker_model": speaker_model is not None,
            "asr_metadata": asr_model is not None,
            "speaker_metadata": speaker_model is not None,
            "runtime_checked": False,
        }
        reference_source = None
        if args.speaker_sim:
            reference_source = (
                "prompt"
                if store.prompt_sha256(args.reference_voice) is not None
                else "ref_wav"
            )
        max_tokens = int(sampling.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS))
        planned = [
            case_plan_payload(
                case, default_seed=seed, default_max_new_tokens=max_tokens
            )
            for case in cases
        ]
        payload = _payload(
            args,
            voice_ref=voice.ref,
            text_source=text_source,
            count=count,
            model_label=model_label,
            asr_model_label=asr_model_label,
            models=model_payload,
            cases=planned,
            runs=[],
            metrics=_empty_metrics(),
            sampling=sampling,
            seed=seed,
            corpus=corpus,
            warnings=warnings,
            scoring_ms=None,
            runtime=None,
            reference_embedding_source=reference_source,
            requirements=requirements,
        )
        payload["_canonical"] = True
        payload["exit_code"] = 0
        return payload

    # Non-dry preflight: runtime/class probes only, never weights.
    if asr_model is not None:
        _preflight_asr_runtime()
    if speaker_model is not None:
        _preflight_speaker(cfg, speaker_model)

    reference_embedding = None
    reference_audio = None
    reference_source: Optional[str] = None
    if speaker_model is not None:
        # Resolve the reference profile before loading TTS weights.  A prompt
        # embedding is a small tensor; ref.wav stays as raw samples for the
        # Base model that is loaded only after generation.
        reference_embedding, reference_source = _reference_embedding(
            store, args.reference_voice
        )
        if reference_embedding is None:
            reference_audio = read_audio(
                store.dir_for(args.reference_voice) / "ref.wav"
            )

    engine = None
    asr_engine = None
    base_engine = None
    execution = None
    scoring_started = 0.0
    scoring_ms = 0.0
    try:
        engine = _engine_factory(cfg, voice.model)
        prompt_items = None
        if voice.mode == "clone":
            try:
                prompt_items = engine.build_clone_prompt_items(
                    store, voice.voice_name
                )
            except QttsError:
                raise
            except Exception as exc:
                raise InputError(
                    f"cannot build clone prompt for voice {voice.voice_name}: {exc}",
                    details={"voice": voice.voice_name},
                ) from exc

        device_memory = _prepare_device_memory()
        execution = execute(
            engine,
            cases,
            runs,
            mode=voice.mode,
            speaker=voice.speaker,
            prompt_items=prompt_items,
            sampling=sampling,
            seed=seed,
            baseline_rss=baseline_rss,
            device_memory=device_memory,
        )
        prompt_items = None
        _release_engine(engine)
        engine = None

        if asr_model is not None:
            scoring_started = time.perf_counter()
            asr_engine = _asr_factory(cfg, asr_model)
            cer_stats = score_cer(asr_engine, execution.last_run_audio)
            execution.metrics.update(cer_stats)
            execution.runs[-1]["cer"] = cer_stats["cer"]
            scoring_ms += (time.perf_counter() - scoring_started) * 1000.0
            _release_asr(asr_engine)
            asr_engine = None

        if speaker_model is not None:
            scoring_started = time.perf_counter()
            base_engine = _engine_factory(cfg, speaker_model)
            similarity, scored, total = score_speaker(
                base_engine,
                reference_embedding,
                execution.last_run_audio,
                reference_audio=reference_audio,
            )
            execution.metrics["speaker_similarity"] = similarity
            execution.metrics["speaker_segments_scored"] = scored
            execution.metrics["speaker_segments_total"] = total
            execution.runs[-1]["speaker_similarity"] = similarity
            scoring_ms += (time.perf_counter() - scoring_started) * 1000.0
            _release_engine(base_engine)
            base_engine = None
    finally:
        if base_engine is not None:
            _release_engine(base_engine)
        if asr_engine is not None:
            _release_asr(asr_engine)
        if engine is not None:
            _release_engine(engine)

    # A backend may still return NaN/Inf despite the cosine guards; scrub the
    # whole payload surface so gates see ``null`` and canonical JSON stays valid.
    _sanitize_numbers(execution.metrics)
    for record in execution.runs:
        _sanitize_numbers(record)
    for record in execution.cases:
        _sanitize_numbers(record)

    gate_failures: list[dict[str, Any]] = []
    if args.max_cer is not None:
        value = execution.metrics["cer"]
        threshold = float(args.max_cer)
        if value is None:
            gate_failures.append(
                {
                    "gate": "max_cer",
                    "value": None,
                    "threshold": threshold,
                    "reason": "unavailable",
                }
            )
        elif float(value) > threshold:
            gate_failures.append(
                {
                    "gate": "max_cer",
                    "value": value,
                    "threshold": threshold,
                    "reason": "exceeded",
                }
            )
    if args.min_speaker_similarity is not None:
        value = execution.metrics["speaker_similarity"]
        threshold = float(args.min_speaker_similarity)
        if value is None:
            gate_failures.append(
                {
                    "gate": "min_speaker_similarity",
                    "value": None,
                    "threshold": threshold,
                    "reason": "unavailable",
                }
            )
        elif float(value) < threshold:
            gate_failures.append(
                {
                    "gate": "min_speaker_similarity",
                    "value": value,
                    "threshold": threshold,
                    "reason": "below",
                }
            )

    runtime_payload = _runtime_payload()
    payload = _payload(
        args,
        voice_ref=voice.ref,
        text_source=text_source,
        count=count,
        model_label=model_label,
        asr_model_label=asr_model_label,
        models=model_payload,
        cases=execution.cases,
        runs=execution.runs,
        metrics=execution.metrics,
        sampling=sampling,
        seed=seed,
        corpus=corpus,
        warnings=warnings,
        scoring_ms=round(scoring_ms, 1),
        runtime=runtime_payload,
        reference_embedding_source=reference_source,
        gate_failures=gate_failures,
    )
    exit_code = 5 if gate_failures else 0
    if getattr(args, "report", None) is not None:
        try:
            write_report(args.report, payload)
        except QttsError as exc:
            payload["ok"] = False
            payload["error"] = {
                "code": exc.code,
                "kind": exc.kind,
                "message": exc.message,
            }
            exit_code = exc.code
    payload["_canonical"] = True
    payload["exit_code"] = exit_code
    return payload


def format_result(payload: dict[str, Any]) -> str:
    request = payload.get("request") or {}
    metrics = payload.get("metrics") or {}
    lines: list[str] = []
    if payload.get("dry_run"):
        lines.append(
            f"bench dry-run: {payload.get('suite')} "
            f"({len(payload.get('cases') or [])} cases, no models loaded)"
        )
    else:
        lines.append(
            f"bench {payload.get('suite')}: {metrics.get('segments_total')} segments"
        )
        lines.append(
            f"  load {metrics.get('load_ms')} ms  cold {metrics.get('cold_first_segment_ms')} ms  "
            f"warm {metrics.get('warm_first_segment_ms')} ms"
        )
        lines.append(
            f"  wall {metrics.get('wall_s')} s  audio {metrics.get('audio_s')} s  "
            f"rtf {metrics.get('rtf')}  wall_rtf {metrics.get('wall_rtf')}"
        )
        if metrics.get("cer") is not None:
            lines.append(f"  cer {metrics.get('cer')} ({metrics.get('cer_units')} units)")
        if metrics.get("speaker_similarity") is not None:
            lines.append(
                f"  speaker_similarity {metrics.get('speaker_similarity')} "
                f"({metrics.get('speaker_segments_scored')}/"
                f"{metrics.get('speaker_segments_total')})"
            )
    for failure in payload.get("gate_failures") or []:
        lines.append(
            f"  gate {failure.get('gate')} {failure.get('reason')}: "
            f"value={failure.get('value')} threshold={failure.get('threshold')}"
        )
    if payload.get("error"):
        error = payload["error"]
        lines.append(f"report error {error.get('code')}: {error.get('message')}")
    lines.append(f"voice: {request.get('voice')}  runs: {request.get('runs')}")
    return "\n".join(lines)
