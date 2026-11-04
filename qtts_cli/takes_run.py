"""Sequential multi-take generation, scoring and publication (FR-P1-04).

The orchestrator keeps exactly one TTS engine slot, writes each take to a
staging directory and drops its audio before moving on. TTS is released before
ASR, ASR before Base, and Base before the forced aligner; subtitles and QC only
see the selected take.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from . import __version__
from .align_engine import AlignerEngine, clear_aligner_cache
from .asr_engine import (
    AsrEngine,
    canonical_asr_language,
    preflight_asr_runtime,
)
from .audio_io import concat, duration_s, read_audio, silence_ms, write_audio
from .audio_qc import analyze as analyze_audio, cosine_similarity
from .cache import AudioCache
from .catalog import language_supported, select_asr
from .errors import InputError, ModelError, QttsError, UsageError
from .lexicon import MapBlock, spoken_items_to_display
from .plan import GenerationPlan, attach_cache_stats, rekey_plan, reseed_plan
from .qc import levenshtein, normalize_for_qc, qc_language, text_units
from .quality import evaluate as evaluate_quality
from .sidecar import build_recipe
from .subtitle_qc import check_subtitles
from .subtitles import (
    build_cues,
    coverage,
    ensure_full_text,
    monotonic,
    policy_from_args,
    shift_cues,
)
from .takes import (
    LUFS_TARGET,
    CorruptJob,
    TakesStore,
    canonical_request,
    collision_key,
    model_identity,
    rank_takes,
    reference_identity,
    request_id,
)
from .tts_engine import TtsEngine, clear_model_cache
from .utils import atomic_write_json, sha256_file, sha256_json, sha256_text
from .voice_store import VoiceStore, utc_now
from .commands.qc import (
    base_model,
    profile_embedding,
    report_payload as qc_report_payload,
    require_speaker_embedding,
    validate_reference_voice,
)

_REQUEST_MARKER = "request_sha256"


# --------------------------------------------------------------------------- #
# Preflight
# --------------------------------------------------------------------------- #
class Scorers:
    def __init__(
        self,
        *,
        asr: Any = None,
        asr_languages: Optional[dict[str, Optional[str]]] = None,
        base: Any = None,
        reference_name: Optional[str] = None,
        reference: Optional[dict[str, Any]] = None,
    ) -> None:
        self.asr = asr
        self.asr_languages = asr_languages or {}
        self.base = base
        self.reference_name = reference_name
        self.reference = reference

    def identities(self) -> dict[str, Any]:
        return {"asr": self.asr, "base": self.base}


def _unique_voices(plan: GenerationPlan) -> list[Any]:
    voices: list[Any] = []
    seen: set[tuple[Any, ...]] = set()
    for seg in plan.segments:
        voice = seg.voice or plan.voice
        key = (
            voice.ref,
            voice.mode,
            voice.model.model_id,
            json.dumps(voice.identity, sort_keys=True, ensure_ascii=False),
        )
        if key not in seen:
            seen.add(key)
            voices.append(voice)
    return voices


def prepare_scorers(
    cfg,
    args,
    plan: GenerationPlan,
    models: list,
    voice_store: VoiceStore,
    criteria: list[str],
    *,
    dry_run: bool,
) -> Scorers:
    """Resolve scorer identities and preflight capabilities before any TTS."""
    asr = None
    asr_languages: dict[str, Optional[str]] = {}
    if "cer" in criteria:
        asr = select_asr(models, getattr(args, "asr_model", None))
        for seg in plan.segments:
            canonical = canonical_asr_language(seg.language)
            if canonical is not None and not language_supported(asr, canonical):
                raise UsageError(
                    f"ASR model {asr.name} does not support language {canonical!r}",
                    details={
                        "model": asr.name,
                        "language": canonical,
                        "supported": asr.languages,
                    },
                )
            asr_languages[seg.id] = canonical
        if not dry_run:
            preflight_asr_runtime()

    base = None
    reference_name: Optional[str] = None
    reference: Optional[dict[str, Any]] = None
    if "speaker-sim" in criteria:
        explicit = getattr(args, "reference_voice", None)
        if explicit:
            validate_reference_voice(voice_store, explicit)
            reference_name = explicit
        else:
            voices = _unique_voices(plan)
            if len(voices) != 1:
                raise UsageError(
                    "speaker-sim needs exactly one voice identity; "
                    "pass --reference-voice for multivoice scripts",
                    details={"voices": [v.ref for v in voices]},
                )
            voice = voices[0]
            if voice.mode != "clone":
                raise UsageError(
                    "speaker-sim needs a clone voice or --reference-voice; "
                    f"plan voice is {voice.mode}",
                    details={"voice": voice.ref, "mode": voice.mode},
                )
            reference_name = voice.voice_name
            validate_reference_voice(voice_store, reference_name)
        reference = reference_identity(voice_store, reference_name)
        base = base_model(models, getattr(args, "model", None))
        if not dry_run:
            require_speaker_embedding(TtsEngine(cfg, base))

    return Scorers(
        asr=asr,
        asr_languages=asr_languages,
        base=base,
        reference_name=reference_name,
        reference=reference,
    )


def _take_seed_vectors(plan: GenerationPlan, count: int) -> list[dict[str, Any]]:
    vectors: list[dict[str, Any]] = []
    for take_index in range(1, count + 1):
        take_plan = reseed_plan(plan, take_index)
        vectors.append(
            {
                "take": take_index,
                "seed": take_plan.seed,
                "segment_seeds": {seg.id: seg.seed for seg in take_plan.segments},
            }
        )
    return vectors


def dry_run_details(
    cfg,
    args,
    *,
    plan: GenerationPlan,
    models: list,
    voice_store: VoiceStore,
    takes_count: int,
    criteria: list[str],
    fmt: str,
    out_target: str,
    recipe_target: Optional[str],
    subtitle_formats: list[str],
    sub_prefix: Optional[str],
) -> dict[str, Any]:
    """Describe a takes run without creating directories or importing torch."""
    scorers = prepare_scorers(
        cfg, args, plan, models, voice_store, criteria, dry_run=True
    )
    lufs_target = LUFS_TARGET
    request = canonical_request(
        plan,
        cfg,
        takes=takes_count,
        criteria=criteria,
        scorer_models=scorers.identities(),
        fmt=fmt,
        lufs_target=lufs_target,
        reference=scorers.reference,
    )
    job_id = request_id(request)
    store = TakesStore(cfg.data_dir)
    job_dir = store.root / job_id
    take_paths = [
        str(job_dir / f"take-{index:02d}.{fmt}") for index in range(1, takes_count + 1)
    ]
    recipe_paths = [
        str(job_dir / f"take-{index:02d}.recipe.json")
        for index in range(1, takes_count + 1)
    ]
    dependencies: dict[str, Any] = {}
    if "cer" in criteria:
        dependencies["cer"] = {
            "available": True,
            "model": model_identity(scorers.asr),
            "languages": dict(scorers.asr_languages),
        }
    if "speaker-sim" in criteria:
        dependencies["speaker-sim"] = {
            "available": True,
            "model": model_identity(scorers.base),
            "reference": scorers.reference_name,
            "identity": scorers.reference,
        }
    if "lufs" in criteria:
        dependencies["lufs"] = {"available": True, "target": lufs_target}
    return {
        "job_id": job_id,
        "job_path": str(job_dir),
        "takes": takes_count,
        "criteria": list(criteria),
        "lufs_target": lufs_target,
        "seeds": _take_seed_vectors(plan, takes_count),
        "selection": {
            "criteria": list(criteria),
            "dependencies": dependencies,
            "scorer_models": {
                "asr": model_identity(scorers.asr),
                "base": model_identity(scorers.base),
            },
        },
        "output_paths": {
            "job": str(job_dir),
            "takes": take_paths,
            "recipes": recipe_paths,
            "out": None if out_target == "-" else out_target,
            "recipe": recipe_target if recipe_target not in (None, "-") else None,
            "subtitles": [f"{sub_prefix}.{sub}" for sub in subtitle_formats]
            if sub_prefix not in (None, "-")
            else [],
        },
        "memory": "takes are generated and released one at a time",
        "model_order": ["tts", "asr", "base", "aligner"],
    }


# --------------------------------------------------------------------------- #
# Take generation
# --------------------------------------------------------------------------- #
class TakeRuntime:
    def __init__(self, cfg, voice_store: VoiceStore) -> None:
        self.cfg = cfg
        self.voice_store = voice_store
        self.engine_slot: dict[str, TtsEngine] = {}
        self.prompt_cache: dict[tuple[str, str], Any] = {}
        self.last_engine: Optional[TtsEngine] = None
        self.dtype: Optional[str] = None
        self.device: Optional[str] = None
        self.load_ms = 0.0
        self.generate_ms = 0.0

    def acquire(self, voice) -> TtsEngine:
        key = f"{voice.model.path}|{self.cfg.device}"
        for old_key, old_engine in list(self.engine_slot.items()):
            if old_key != key:
                old_engine.model = None
                self.engine_slot.pop(old_key, None)
                clear_model_cache()
                self.prompt_cache.clear()
        engine = self.engine_slot.get(key)
        if engine is None:
            engine = TtsEngine(self.cfg, voice.model)
            self.engine_slot[key] = engine
        if engine.model is None:
            engine.load()
            self.load_ms += float(engine.load_ms or 0.0)
        self.last_engine = engine
        return engine

    def prompt_for(self, engine: TtsEngine, voice):
        if voice.mode != "clone":
            return None
        key = (str(voice.model.path), json.dumps(voice.identity, sort_keys=True))
        if key not in self.prompt_cache:
            self.prompt_cache[key] = engine.build_clone_prompt_items(
                self.voice_store, voice.voice_name or ""
            )
        return self.prompt_cache[key]

    def release(self) -> None:
        for engine in self.engine_slot.values():
            engine.model = None
        self.engine_slot.clear()
        self.prompt_cache.clear()
        clear_model_cache()


def _generate_take(
    cfg,
    base_plan: GenerationPlan,
    take_index: int,
    runtime: TakeRuntime,
    cache: AudioCache,
    staging: Path,
    fmt: str,
) -> tuple[GenerationPlan, list[dict[str, Any]], dict[str, Any], int]:
    plan = reseed_plan(base_plan, take_index)
    if runtime.dtype:
        rekey_plan(plan, dtype=runtime.dtype, device=runtime.device)
    attach_cache_stats(plan, cache, "wav")

    def missing(seg) -> bool:
        return cfg.no_cache or not cache.exists(seg.cache_key, "wav")

    if any(missing(seg) for seg in plan.segments):
        first_missing = next(
            (seg for seg in plan.segments if missing(seg)), plan.segments[0]
        )
        engine = runtime.acquire(first_missing.voice or plan.voice)
        if runtime.dtype is None:
            runtime.dtype = engine.dtype_name or plan.dtype_name
            runtime.device = engine.device
            if runtime.dtype != plan.dtype_name:
                rekey_plan(plan, dtype=runtime.dtype, device=runtime.device)
                attach_cache_stats(plan, cache, "wav")

    segment_audio: list[np.ndarray] = []
    segment_meta: list[dict[str, Any]] = []
    first_sr: Optional[int] = None
    offset = plan.leading_pause_ms / 1000.0

    for seg in plan.segments:
        voice = seg.voice or plan.voice
        cached_path = None if cfg.no_cache else cache.lookup(seg.cache_key, "wav")
        if cached_path is not None:
            audio, sr = read_audio(cached_path)
            gen_info = {"cached": True, "generate_ms": 0.0}
        else:
            engine = runtime.acquire(voice)
            sampling = dict(plan.sampling)
            if seg.max_new_tokens is not None:
                sampling["max_new_tokens"] = int(seg.max_new_tokens)
            audio, sr, gen_info = engine.generate(
                text=seg.text,
                language=seg.language,
                mode=voice.mode,
                speaker=voice.speaker,
                instruct=seg.instruct if voice.mode in ("builtin", "design") else None,
                prompt_items=runtime.prompt_for(engine, voice),
                sampling=sampling,
                seed=int(seg.seed if seg.seed is not None else plan.seed),
            )
            gen_info["cached"] = False
            runtime.generate_ms += float(gen_info.get("generate_ms") or 0.0)
            if not cfg.no_cache:
                _cache_segment(cache, seg.cache_key, audio, sr)
        if first_sr is None:
            first_sr = sr
        elif sr != first_sr:
            raise ModelError(f"segment sample rate changed: {first_sr} -> {sr}")
        seg_duration = duration_s(audio, sr)
        start = offset
        end = start + seg_duration
        segment_audio.append(audio)
        segment_meta.append(
            {
                "id": seg.id,
                "source_segment_id": seg.source_segment_id,
                "cache_key": seg.cache_key,
                "text": seg.text,
                "text_sha256": sha256_text(seg.text),
                "display_text": seg.display_text
                if seg.display_text is not None
                else seg.text,
                "display_sha256": sha256_text(seg.display_text or seg.text),
                "voice": voice.to_dict() if seg.voice is not None else voice.ref,
                "language": seg.language,
                "instruct": seg.instruct,
                "seed": int(seg.seed if seg.seed is not None else plan.seed),
                "max_new_tokens": seg.max_new_tokens,
                "pause_after_ms": int(seg.pause_after_ms),
                "duration_ms": int(round(seg_duration * 1000)),
                "duration_s": round(seg_duration, 6),
                "start_s": round(start, 6),
                "end_s": round(end, 6),
                "cached": bool(gen_info.get("cached")),
                "generate_ms": gen_info.get("generate_ms"),
            }
        )
        offset = end + seg.pause_after_ms / 1000.0

    if first_sr is None:
        raise ModelError("no audio generated")
    sample_rate = first_sr
    core = concat(
        segment_audio, sample_rate, [s.pause_after_ms for s in plan.segments]
    )
    parts = [core]
    if plan.leading_pause_ms:
        parts.insert(0, silence_ms(int(plan.leading_pause_ms), sample_rate))
    if plan.trailing_pause_ms:
        parts.append(silence_ms(int(plan.trailing_pause_ms), sample_rate))
    full = concat(parts, sample_rate) if len(parts) > 1 else core
    total_s = duration_s(full, sample_rate)

    filename = f"take-{take_index:02d}.{fmt}"
    recipe_name = f"take-{take_index:02d}.recipe.json"
    audio_path = staging / filename
    write_audio(audio_path, full, sample_rate, fmt)
    digest = sha256_file(audio_path)

    take = {
        "id": take_index,
        "index": take_index - 1,
        "seed": int(plan.seed),
        "segment_seeds": {seg.id: int(seg.seed) for seg in plan.segments},
        "file": filename,
        "format": fmt,
        "duration_s": round(total_s, 4),
        "sample_rate": int(sample_rate),
        "audio_sha256": f"sha256:{digest}",
        "recipe": recipe_name,
        "segments": [
            {
                "id": meta["id"],
                "text_sha256": meta["text_sha256"],
                "display_sha256": meta["display_sha256"],
                "language": meta["language"],
                "seed": meta["seed"],
                "pause_after_ms": meta["pause_after_ms"],
                "duration_s": meta["duration_s"],
                "start_s": meta["start_s"],
                "end_s": meta["end_s"],
            }
            for meta in segment_meta
        ],
        "metrics": {},
        "unavailable": {},
        "rank": None,
    }
    # Drop this take's arrays before the next take starts.
    del segment_audio, core, full, parts
    return plan, segment_meta, take, int(sample_rate)


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


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
class ScoringSession:
    def __init__(self) -> None:
        self.base_engine: Optional[TtsEngine] = None
        self.reference_name: Optional[str] = None
        self.reference_embedding: Any = None
        self.selected: Optional[int] = None

    def release(self) -> None:
        if self.base_engine is not None:
            self.base_engine.model = None
            self.base_engine = None
        clear_model_cache()


def _take_samples(path: Path) -> tuple[np.ndarray, int]:
    samples, sr = read_audio(path, mono=True)
    return samples, int(sr)


def _slice_for_segment(
    samples: np.ndarray,
    sr: int,
    meta: dict[str, Any],
    *,
    take_id: int,
) -> np.ndarray:
    duration = float(meta.get("duration_s") or 0.0)
    if duration > 300.0:
        raise InputError(
            f"take {take_id} segment {meta.get('id')} is {duration:.1f}s; "
            "scoring supports at most 300s per segment",
            details={"take": take_id, "segment": meta.get("id"), "duration_s": duration},
        )
    start = int(round(float(meta.get("start_s") or 0.0) * sr))
    end = int(round(float(meta.get("end_s") or 0.0) * sr))
    start = max(0, min(start, len(samples)))
    end = max(start, min(end, len(samples)))
    if end <= start:
        raise InputError(
            f"take {take_id} segment {meta.get('id')} has no audio samples",
            details={"take": take_id, "segment": meta.get("id")},
        )
    return np.ascontiguousarray(samples[start:end], dtype=np.float32)


def _score_lufs(take: dict[str, Any], staging: Path) -> None:
    samples, sr = _take_samples(staging / take["file"])
    metrics = analyze_audio(samples, sr)
    lufs = metrics.get("lufs_integrated")
    true_peak = metrics.get("true_peak_dbtp")
    take["metrics"]["lufs"] = lufs
    take["metrics"]["true_peak_dbtp"] = true_peak
    unavailable = metrics.get("unavailable") or {}
    if lufs is None:
        take["unavailable"]["lufs"] = unavailable.get("lufs", "unavailable")
    if true_peak is None:
        take["unavailable"]["true_peak_dbtp"] = unavailable.get("true_peak", "unavailable")


def _score_cer(
    cfg,
    plan: GenerationPlan,
    takes: list[dict[str, Any]],
    staging: Path,
    scorers: Scorers,
) -> None:
    engine = AsrEngine(cfg, scorers.asr)
    try:
        for take in takes:
            samples, sr = _take_samples(staging / take["file"])
            total_distance = 0
            total_units = 0
            details: list[dict[str, Any]] = []
            for seg, meta in zip(plan.segments, take["segments"]):
                chunk = _slice_for_segment(samples, sr, meta, take_id=take["id"])
                transcribed = engine.transcribe_samples(
                    chunk, sr, language=scorers.asr_languages.get(seg.id)
                )
                hypothesis = str(transcribed.get("text") or "")
                qc_lang = qc_language(seg.language, seg.text)
                reference = normalize_for_qc(seg.text, qc_lang)
                hypothesis_norm = normalize_for_qc(hypothesis, qc_lang)
                ref_units = text_units(reference, "cer")
                hyp_units = text_units(hypothesis_norm, "cer")
                distance = levenshtein(ref_units, hyp_units)
                total_distance += distance
                total_units += len(ref_units)
                details.append(
                    {
                        "id": seg.id,
                        "distance": distance,
                        "reference_units": len(ref_units),
                        "hypothesis_units": len(hyp_units),
                    }
                )
            if total_units <= 0:
                raise InputError(
                    f"take {take['id']} has no reference units for CER",
                    details={"take": take["id"]},
                )
            take["metrics"]["cer"] = float(total_distance) / float(total_units)
            take["metrics"]["cer_distance"] = total_distance
            take["metrics"]["cer_reference_units"] = total_units
            take["_cer_segments"] = details
    finally:
        engine.release()


def _speaker_target_embedding(
    engine: TtsEngine, voice_store: VoiceStore, name: str
) -> Any:
    reference = profile_embedding(voice_store, name)
    if reference is None:
        ref_samples, ref_sr = read_audio(voice_store.dir_for(name) / "ref.wav")
        reference = engine.extract_speaker_embedding(ref_samples, ref_sr)
    return reference


def _score_speaker(
    plan: GenerationPlan,
    takes: list[dict[str, Any]],
    staging: Path,
    engine: TtsEngine,
    voice_store: VoiceStore,
    name: str,
) -> Any:
    reference = _speaker_target_embedding(engine, voice_store, name)
    for take in takes:
        samples, sr = _take_samples(staging / take["file"])
        weighted = 0.0
        weight = 0.0
        minimum: Optional[float] = None
        details: list[dict[str, Any]] = []
        for seg, meta in zip(plan.segments, take["segments"]):
            chunk = _slice_for_segment(samples, sr, meta, take_id=take["id"])
            target = engine.extract_speaker_embedding(chunk, sr)
            similarity = cosine_similarity(reference, target)
            if similarity is None:
                raise InputError(
                    f"speaker embedding is unavailable for take {take['id']} "
                    f"segment {seg.id}",
                    details={"take": take["id"], "segment": seg.id},
                )
            value = float(similarity)
            seconds = max(float(meta.get("duration_s") or 0.0), 1e-6)
            weighted += value * seconds
            weight += seconds
            minimum = value if minimum is None else min(minimum, value)
            details.append(
                {"id": seg.id, "similarity": value, "duration_s": seconds}
            )
        if weight <= 0:
            raise InputError(f"take {take['id']} has no speaker audio")
        take["metrics"]["speaker_similarity"] = weighted / weight
        take["metrics"]["speaker_similarity_min"] = minimum
        take["_speaker_segments"] = details
    return reference


def score_takes(
    cfg,
    plan: GenerationPlan,
    takes: list[dict[str, Any]],
    staging: Path,
    criteria: list[str],
    scorers: Scorers,
    voice_store: VoiceStore,
    *,
    lufs_target: float = LUFS_TARGET,
) -> ScoringSession:
    """Score takes in residency order and return the selected session."""
    session = ScoringSession()
    try:
        for take in takes:
            _score_lufs(take, staging)
            if "lufs" in criteria and take["metrics"].get("lufs") is None:
                raise ModelError(
                    f"LUFS is unavailable for take {take['id']}; cannot rank by lufs",
                    details={
                        "take": take["id"],
                        "reason": take["unavailable"].get("lufs"),
                    },
                )

        if "cer" in criteria:
            _score_cer(cfg, plan, takes, staging, scorers)

        if "speaker-sim" in criteria:
            engine = TtsEngine(cfg, scorers.base)
            session.base_engine = engine
            require_speaker_embedding(engine)
            session.reference_name = scorers.reference_name
            session.reference_embedding = _score_speaker(
                plan, takes, staging, engine, voice_store, scorers.reference_name
            )

        for take in takes:
            metrics = take["metrics"]
            unavailable = take["unavailable"]
            if "cer" not in criteria:
                metrics["cer"] = None
                unavailable["cer"] = "not_selected"
            if "speaker-sim" not in criteria:
                metrics["speaker_similarity"] = None
                metrics["speaker_similarity_min"] = None
                unavailable["speaker_similarity"] = "not_selected"

        ranked = rank_takes(takes, criteria, lufs_target=lufs_target)
        for rank, take in enumerate(ranked, 1):
            take["rank"] = rank
            # Keep ranking on raw values; six decimals are display precision.
            for key, value in list(take["metrics"].items()):
                if isinstance(value, float):
                    take["metrics"][key] = round(value, 6)
        session.selected = int(ranked[0]["id"])
        return session
    except BaseException:
        session.release()
        raise


# --------------------------------------------------------------------------- #
# Job assembly / publication
# --------------------------------------------------------------------------- #
def _voices_and_models(
    plan: GenerationPlan,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    voices: list[dict[str, Any]] = []
    models: list[dict[str, Any]] = []
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
            models.append(
                {
                    "id": voice.model.model_id,
                    "revision": voice.model.revision,
                    "fingerprint": voice.model.fingerprint,
                    "path": str(voice.model.path),
                }
            )
    return voices, models


def _take_recipe(
    *,
    job_id: str,
    take_index: int,
    plan: GenerationPlan,
    base_plan: GenerationPlan,
    segment_meta: list[dict[str, Any]],
    take: dict[str, Any],
    criteria: list[str],
    created_at: str,
    runtime_dtype: Optional[str],
    runtime_device: Optional[str],
    source: str,
    lufs_target: float,
    reference: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    voices, models = _voices_and_models(plan)
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
            "dtype": runtime_dtype or plan.dtype_name,
            "path": str(plan.voice.model.path),
        },
        sampling=plan.sampling,
        segments=segment_meta,
        created_at=created_at,
        seed=int(base_plan.seed),
        normalization={"version": plan.normalization_version},
        device=runtime_device,
        instruct=plan.instruct,
        language=plan.language,
    )
    recipe["job_id"] = job_id
    recipe["take_id"] = take_index
    recipe["take_index"] = take_index - 1
    recipe["base_seed"] = int(base_plan.seed)
    recipe["take_seed"] = int(plan.seed)
    recipe["segment_seeds"] = {seg.id: int(seg.seed) for seg in plan.segments}
    recipe["lexicon_sha256"] = plan.lexicon_sha256
    recipe["seed_semantics"] = {
        "base_seed": int(base_plan.seed),
        "take_seed": int(plan.seed),
        "segment_seeds": {seg.id: int(seg.seed) for seg in plan.segments},
    }
    recipe["spoken_to_display"] = {
        seg.id: {
            "text_sha256": sha256_text(seg.text),
            "display_sha256": sha256_text(seg.display_text or seg.text),
        }
        for seg in plan.segments
    }
    recipe["cache_keys"] = {seg.id: seg.cache_key for seg in plan.segments}
    recipe["runtime"] = {
        "dtype": runtime_dtype or plan.dtype_name,
        "device": runtime_device or "cache",
    }
    recipe["audio"] = {
        "file": take["file"],
        "sha256": take["audio_sha256"],
        "duration_s": take["duration_s"],
        "sample_rate": take["sample_rate"],
        "format": take["format"],
    }
    recipe["voices"] = voices
    recipe["models"] = models
    recipe["selection"] = {
        "criteria": list(criteria),
        "lufs_target": float(lufs_target),
        "reference": reference,
        "metrics": dict(take["metrics"]),
        "unavailable": dict(take["unavailable"]),
    }
    for key in ("_cer_segments", "_speaker_segments"):
        value = take.get(key)
        if value is not None:
            recipe["selection"][key.strip("_")] = value
    return recipe


def _build_job(
    *,
    job_id: str,
    request: dict[str, Any],
    fmt: str,
    sample_rate: int,
    takes: list[dict[str, Any]],
    state: str,
    selected: Optional[int],
    selected_at: Optional[str],
    created_at: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "job_id": job_id,
        "state": state,
        "created_at": created_at,
        "qtts_version": __version__,
        "format": fmt,
        "sample_rate": int(sample_rate),
        "request": request,
        _REQUEST_MARKER: sha256_json(request),
        "takes": takes,
        "selected": selected,
        "selected_at": selected_at,
    }


def _publish_with_damage(
    store: TakesStore,
    job_id: str,
    staging: Path,
    *,
    damaged: Optional[Path],
) -> None:
    if damaged is None:
        store.publish(staging, job_id)
        return
    try:
        store.publish(staging, job_id)
    except BaseException:
        try:
            store.restore(damaged, job_id)
        except BaseException as restore_error:
            raise InputError(
                f"failed to publish takes job and could not restore the damaged "
                f"backup at {damaged}: {restore_error}",
                details={"job": job_id, "backup": str(damaged)},
            ) from restore_error
        raise


def _load_existing(
    store: TakesStore, job_id: str, request: dict[str, Any]
) -> tuple[Optional[dict[str, Any]], Optional[Path], bool]:
    """Return (job, job_dir, reused) or (None, damaged_backup, False)."""
    target = store.job_dir(job_id)
    if not target.exists():
        return None, None, False
    job_path = target / "job.json"
    if not job_path.is_file():
        return None, store.quarantine(job_id), False
    try:
        job = store.read_job(job_id)
    except (CorruptJob, InputError):
        return None, store.quarantine(job_id), False
    if (
        job["job_id"] != job_id
        or job.get(_REQUEST_MARKER) != sha256_json(request)
        or sha256_json(job.get("request")) != sha256_json(request)
    ):
        raise InputError(
            f"takes job {job_id} does not match the requested canonical request",
            details={"job": job_id, "path": str(target)},
        )
    store.verify_takes(job, job_dir=target)
    return job, target, True


# --------------------------------------------------------------------------- #
# Main entry
# --------------------------------------------------------------------------- #
def run(
    cfg,
    args,
    *,
    plan: GenerationPlan,
    models: list,
    voice_store: VoiceStore,
    aligner_model,
    cache: AudioCache,
    out_target: str,
    fmt: str,
    subtitle_formats: list[str],
    sub_prefix: Optional[str],
    recipe_arg: Optional[str],
    recipe_target: Optional[str],
    qc_enabled: bool,
    qc_reference_voice: Optional[str],
    qc_thresholds: dict[str, float],
    source: str,
    takes_count: int,
    criteria: list[str],
    stdout_used: bool,
) -> dict[str, Any]:
    started = time.perf_counter()
    scorers = prepare_scorers(
        cfg, args, plan, models, voice_store, criteria, dry_run=False
    )
    lufs_target = LUFS_TARGET
    request = canonical_request(
        plan,
        cfg,
        takes=takes_count,
        criteria=criteria,
        scorer_models=scorers.identities(),
        fmt=fmt,
        lufs_target=lufs_target,
        reference=scorers.reference,
    )
    job_id = request_id(request)
    # Validate every derived seed before touching the asset library.
    _take_seed_vectors(plan, takes_count)
    store = TakesStore(cfg.data_dir)
    runtime = TakeRuntime(cfg, voice_store)
    created_at = utc_now()
    reused = False
    session: Optional[ScoringSession] = None
    try:
        with store.lock(job_id):
            existing, target_or_backup, found = _load_existing(store, job_id, request)
            if found and existing is not None:
                job = existing
                job_dir = target_or_backup
                reused = True
                if job["state"] == "scoring_failed":
                    session = score_takes(
                        cfg,
                        plan,
                        job["takes"],
                        job_dir,
                        criteria,
                        scorers,
                        voice_store,
                        lufs_target=lufs_target,
                    )
                    for take in job["takes"]:
                        plan_t = reseed_plan(plan, int(take["id"]))
                        recipe = _take_recipe(
                            job_id=job_id,
                            take_index=int(take["id"]),
                            plan=plan_t,
                            base_plan=plan,
                            segment_meta=_segment_meta_from_take(plan_t, take),
                            take=take,
                            criteria=criteria,
                            created_at=job.get("created_at") or created_at,
                            runtime_dtype=(job["request"].get("runtime") or {}).get(
                                "dtype"
                            ),
                            runtime_device=(job["request"].get("runtime") or {}).get(
                                "device"
                            ),
                            source=source,
                            lufs_target=lufs_target,
                            reference=scorers.reference,
                        )
                        atomic_write_json(job_dir / take["recipe"], recipe)
                    job["state"] = "complete"
                    job["selected"] = session.selected
                    job["selected_at"] = utc_now()
                    store.update_job(job_id, job)
            else:
                damaged = target_or_backup if target_or_backup is not None else None
                staging = store.new_staging(job_id)
                try:
                    take_plans: dict[
                        int, tuple[GenerationPlan, list[dict[str, Any]]]
                    ] = {}
                    takes: list[dict[str, Any]] = []
                    sample_rate = 0
                    for take_index in range(1, takes_count + 1):
                        plan_t, meta, take, sample_rate = _generate_take(
                            cfg,
                            plan,
                            take_index,
                            runtime,
                            cache,
                            staging,
                            fmt,
                        )
                        take_plans[take_index] = (plan_t, meta)
                        takes.append(take)
                    runtime.release()

                    error: Optional[QttsError] = None
                    try:
                        session = score_takes(
                            cfg,
                            plan,
                            takes,
                            staging,
                            criteria,
                            scorers,
                            voice_store,
                            lufs_target=lufs_target,
                        )
                        state = "complete"
                        selected = session.selected
                        selected_at = utc_now()
                    except QttsError as exc:
                        state = "scoring_failed"
                        selected = None
                        selected_at = None
                        error = exc
                    except Exception as exc:  # unexpected scorer/runtime failure
                        state = "scoring_failed"
                        selected = None
                        selected_at = None
                        error = ModelError(f"takes scoring failed: {exc}")

                    for take in takes:
                        plan_t, meta = take_plans[int(take["id"])]
                        recipe = _take_recipe(
                            job_id=job_id,
                            take_index=int(take["id"]),
                            plan=plan_t,
                            base_plan=plan,
                            segment_meta=meta,
                            take=take,
                            criteria=criteria,
                            created_at=created_at,
                            runtime_dtype=runtime.dtype,
                            runtime_device=runtime.device,
                            source=source,
                            lufs_target=lufs_target,
                            reference=scorers.reference,
                        )
                        atomic_write_json(staging / take["recipe"], recipe)
                    job = _build_job(
                        job_id=job_id,
                        request=request,
                        fmt=fmt,
                        sample_rate=sample_rate,
                        takes=takes,
                        state=state,
                        selected=selected,
                        selected_at=selected_at,
                        created_at=created_at,
                    )
                    store.write_job_json(staging, job)
                    _publish_with_damage(store, job_id, staging, damaged=damaged)
                    if error is not None:
                        raise error
                except BaseException as exc:
                    store.cleanup(staging)
                    if damaged is not None and not store.job_dir(job_id).exists():
                        try:
                            store.restore(damaged, job_id)
                        except BaseException as restore_error:
                            raise InputError(
                                "take generation failed and the damaged job backup "
                                f"could not be restored; backup preserved at {damaged}",
                                details={
                                    "job": job_id,
                                    "backup": str(damaged),
                                    "original_error": str(exc),
                                    "restore_error": str(restore_error),
                                },
                            ) from exc
                    raise
    except BaseException:
        runtime.release()
        if session is not None:
            session.release()
        raise

    job_dir = store.job_dir(job_id)
    selected_take = next(
        take for take in job["takes"] if take["id"] == job["selected"]
    )
    # Validate the selected recipe before any downstream side effect.
    top_recipe = store.verify_recipe(job, selected_take, job_dir)
    return _downstream(
        cfg,
        args,
        plan=plan,
        models=models,
        voice_store=voice_store,
        job=job,
        job_dir=job_dir,
        selected_take=selected_take,
        top_recipe=top_recipe,
        aligner_model=aligner_model,
        out_target=out_target,
        fmt=fmt,
        subtitle_formats=subtitle_formats,
        sub_prefix=sub_prefix,
        recipe_target=recipe_target,
        qc_enabled=qc_enabled,
        qc_reference_voice=qc_reference_voice,
        qc_thresholds=qc_thresholds,
        stdout_used=stdout_used,
        session=session,
        reused=reused,
        started=started,
        runtime=runtime,
    )


def _segment_meta_from_take(
    plan: GenerationPlan, take: dict[str, Any]
) -> list[dict[str, Any]]:
    by_id = {meta["id"]: meta for meta in take["segments"]}
    out: list[dict[str, Any]] = []
    for seg in plan.segments:
        meta = by_id.get(seg.id, {})
        out.append(
            {
                "id": seg.id,
                "source_segment_id": seg.source_segment_id,
                "cache_key": seg.cache_key,
                "text": seg.text,
                "text_sha256": sha256_text(seg.text),
                "display_text": seg.display_text
                if seg.display_text is not None
                else seg.text,
                "display_sha256": sha256_text(seg.display_text or seg.text),
                "voice": seg.voice.to_dict() if seg.voice is not None else plan.voice.ref,
                "language": seg.language,
                "instruct": seg.instruct,
                "seed": int(seg.seed if seg.seed is not None else plan.seed),
                "max_new_tokens": seg.max_new_tokens,
                "pause_after_ms": int(seg.pause_after_ms),
                "duration_ms": int(round(float(meta.get("duration_s") or 0.0) * 1000)),
                "duration_s": meta.get("duration_s"),
                "start_s": meta.get("start_s"),
                "end_s": meta.get("end_s"),
            }
        )
    return out


# --------------------------------------------------------------------------- #
# Downstream selected take
# --------------------------------------------------------------------------- #
def _copy_atomic(source: Path, target: Path) -> None:
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
    )
    try:
        with os.fdopen(fd, "wb") as out, open(source, "rb") as inp:
            shutil.copyfileobj(inp, out)
            out.flush()
            os.fsync(out.fileno())
        os.replace(tmp, target)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _write_selected_bytes(source: Path) -> None:
    with open(source, "rb") as handle:
        shutil.copyfileobj(handle, sys.stdout.buffer)
    sys.stdout.buffer.flush()


def _subtitles_for_selected(
    cfg,
    args,
    plan: GenerationPlan,
    job_dir: Path,
    take: dict[str, Any],
    aligner_model,
    formats: list[str],
    sub_prefix: Optional[str],
    out_target: str,
) -> tuple[
    dict[str, str],
    dict[str, Any],
    list[float],
    Any,
    list[str],
    dict[str, int],
]:
    from .commands.say import _write_subtitles

    samples, sr = _take_samples(job_dir / take["file"])
    aligner = AlignerEngine(cfg, aligner_model)
    policy = policy_from_args(args)
    cues = []
    warnings = list(plan.warnings)
    coverages: list[float] = []
    align_info: dict[str, Any] = {}
    boundary_stats = {"sentences": 0, "splits": 0, "hard_cuts": 0, "merges": 0}
    repair_stats: dict[str, int] = {}
    offset = plan.leading_pause_ms / 1000.0
    take_segments = {meta["id"]: meta for meta in take["segments"]}
    try:
        for seg in plan.segments:
            meta = take_segments[seg.id]
            chunk = _slice_for_segment(samples, sr, meta, take_id=take["id"])
            seg_duration = duration_s(chunk, sr)
            items, info = aligner.align(chunk, sr, seg.text, seg.language)
            align_info = info
            seg_coverage = coverage(seg.text, items)
            coverages.append(seg_coverage)
            if seg_coverage < 0.3:
                raise InputError(
                    f"forced alignment matched only {seg_coverage:.0%} of segment "
                    f"{seg.id}; subtitle timing would be unreliable",
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
        total_s = duration_s(samples, sr)
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
        files = (
            _write_subtitles(cues, formats, sub_prefix, plan, out_target)
            if qc_report.ok
            else {}
        )
        return files, align_info, coverages, qc_report, warnings, boundary_stats
    finally:
        aligner.model = None
        clear_model_cache()
        clear_aligner_cache()


def _qc_selected(
    cfg,
    args,
    plan: GenerationPlan,
    models: list,
    voice_store: VoiceStore,
    job_dir: Path,
    take: dict[str, Any],
    thresholds: dict[str, float],
    reference_voice: Optional[str],
    session: Optional[ScoringSession],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    samples, sr = _take_samples(job_dir / take["file"])
    metrics = analyze_audio(
        samples,
        sr,
        text=plan.normalized_text or plan.display_text,
        language=plan.language,
    )
    local_engine = None
    if reference_voice:
        validate_reference_voice(voice_store, reference_voice)
        engine = None
        reference = None
        if (
            session is not None
            and session.base_engine is not None
            and session.reference_name == reference_voice
        ):
            engine = session.base_engine
            reference = session.reference_embedding
        if engine is None:
            engine = TtsEngine(cfg, base_model(models, getattr(args, "model", None)))
            require_speaker_embedding(engine)
            local_engine = engine
        try:
            if reference is None:
                reference = _speaker_target_embedding(
                    engine, voice_store, reference_voice
                )
            target = engine.extract_speaker_embedding(samples, sr)
            similarity = cosine_similarity(reference, target)
            if similarity is None:
                raise InputError(
                    "speaker embeddings are empty or have mismatched shapes",
                    details={"voice": reference_voice},
                )
            metrics["speaker_similarity"] = round(float(similarity), 4)
            metrics["unavailable"].pop("speaker_similarity", None)
        finally:
            if local_engine is not None:
                local_engine.model = None
                clear_model_cache()
    violations = evaluate_quality(metrics, thresholds)
    return (
        qc_report_payload(None, metrics, thresholds, violations, reference_voice),
        violations,
    )


def _downstream(
    cfg,
    args,
    *,
    plan: GenerationPlan,
    models: list,
    voice_store: VoiceStore,
    job: dict[str, Any],
    job_dir: Path,
    selected_take: dict[str, Any],
    top_recipe: dict[str, Any],
    aligner_model,
    out_target: str,
    fmt: str,
    subtitle_formats: list[str],
    sub_prefix: Optional[str],
    recipe_target: Optional[str],
    qc_enabled: bool,
    qc_reference_voice: Optional[str],
    qc_thresholds: dict[str, float],
    stdout_used: bool,
    session: Optional[ScoringSession],
    reused: bool,
    started: float,
    runtime: TakeRuntime,
) -> dict[str, Any]:
    selected_path = job_dir / selected_take["file"]
    protected = {collision_key(job_dir / "job.json"): "job.json"}
    for take in job["takes"]:
        protected[collision_key(job_dir / take["file"])] = take["file"]
        protected[collision_key(job_dir / take["recipe"])] = take["recipe"]
    targets: list[tuple[str, Path]] = []
    if out_target != "-":
        targets.append(("--out", Path(out_target)))
    if recipe_target not in (None, "-"):
        targets.append(("--recipe-out", Path(recipe_target)))
    if sub_prefix not in (None, "-"):
        for subtitle_format in subtitle_formats:
            targets.append(("--out-prefix", Path(f"{sub_prefix}.{subtitle_format}")))
    for option, target in targets:
        asset = protected.get(collision_key(target))
        if asset is not None:
            raise UsageError(
                f"{option} may not overwrite takes asset {asset!r}",
                details={"path": str(target), "asset": asset},
            )
    warnings = list(plan.warnings)
    subtitle_files: dict[str, str] = {}
    align_info: dict[str, Any] = {}
    coverages: list[float] = []
    qc_report = None
    qc_result: Optional[dict[str, Any]] = None
    qc_violations: list[dict[str, Any]] = []

    try:
        if qc_enabled:
            qc_result, qc_violations = _qc_selected(
                cfg,
                args,
                plan,
                models,
                voice_store,
                job_dir,
                selected_take,
                qc_thresholds,
                qc_reference_voice,
                session,
            )
        if session is not None:
            session.release()
        elif qc_reference_voice:
            clear_model_cache()

        boundary_stats = {"sentences": 0, "splits": 0, "hard_cuts": 0, "merges": 0}
        if subtitle_formats:
            assert aligner_model is not None
            (
                subtitle_files,
                align_info,
                coverages,
                qc_report,
                subtitle_warnings,
                boundary_stats,
            ) = _subtitles_for_selected(
                cfg,
                args,
                plan,
                job_dir,
                selected_take,
                aligner_model,
                subtitle_formats,
                sub_prefix,
                out_target,
            )
            warnings.extend(subtitle_warnings)

        if out_target == "-":
            _write_selected_bytes(selected_path)
        else:
            _copy_atomic(selected_path, Path(out_target))

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        top_recipe["takes"] = {
            "job_id": job["job_id"],
            "job_path": str(job_dir),
            "ids": [take["id"] for take in job["takes"]],
            "metrics": {
                take["id"]: take["metrics"] for take in job["takes"]
            },
            "unavailable": {
                take["id"]: take["unavailable"] for take in job["takes"]
            },
            "selected": job["selected"],
            "selected_at": job["selected_at"],
        }
        top_recipe["timing"] = {
            **(top_recipe.get("timing") or {}),
            "total_ms": round(elapsed_ms, 1),
            "model_load_ms": round(runtime.load_ms, 1),
            "generate_ms": round(runtime.generate_ms, 1),
        }
        if subtitle_formats:
            top_recipe["subtitle"] = {
                "formats": subtitle_formats,
                "granularity": args.granularity,
                "files": subtitle_files,
                "qc": qc_report.to_dict() if qc_report is not None else None,
                "boundaries": dict(boundary_stats),
            }
        if qc_result is not None:
            top_recipe["qc"] = qc_result
        if recipe_target == "-":
            from .utils import emit_json

            emit_json(top_recipe)
        elif recipe_target is not None:
            atomic_write_json(recipe_target, top_recipe)

        payload: dict[str, Any] = {
            "ok": not qc_violations,
            "stdout_used": stdout_used,
            "output": out_target,
            "format": fmt,
            "sample_rate": selected_take["sample_rate"],
            "duration_s": selected_take["duration_s"],
            "voice": plan.voice.to_dict(),
            "voices": top_recipe.get("voices") or [],
            "models": top_recipe.get("models") or [],
            "language": plan.language,
            "seed": int(plan.seed),
            "segments": selected_take["segments"],
            "sampling": plan.sampling,
            "cache": {
                "hit_rate": round(plan.cache_hit_rate, 4),
                **plan.cache_stats,
            },
            "device": (top_recipe.get("runtime") or {}).get("device") or cfg.device,
            "dtype": (top_recipe.get("runtime") or {}).get("dtype")
            or plan.dtype_name,
            "timing": top_recipe["timing"],
            "recipe": recipe_target if recipe_target else None,
            "warnings": warnings,
            "takes": {
                "job_id": job["job_id"],
                "job_path": str(job_dir),
                "count": len(job["takes"]),
                "state": job["state"],
                "criteria": (job["request"].get("selection") or {}).get(
                    "criteria", []
                ),
                "lufs_target": float(
                    (job["request"].get("selection") or {}).get(
                        "lufs_target", LUFS_TARGET
                    )
                ),
                "seeds": job["request"].get("seeds", []),
                "selected": job["selected"],
                "selected_at": job["selected_at"],
                "reused": reused,
                "items": [
                    {
                        "id": take["id"],
                        "seed": take["seed"],
                        "rank": take["rank"],
                        "file": take["file"],
                        "audio_sha256": take["audio_sha256"],
                        "duration_s": take["duration_s"],
                        "metrics": take["metrics"],
                        "unavailable": take["unavailable"],
                    }
                    for take in job["takes"]
                ],
            },
        }
        if subtitle_formats:
            payload["subtitles"] = subtitle_files
            payload["align"] = align_info
            payload["align_coverage"] = (
                round(min(coverages), 4) if coverages else None
            )
            payload["subtitle_qc"] = (
                qc_report.to_dict() if qc_report is not None else None
            )
            payload["subtitle_boundaries"] = dict(boundary_stats)
        if qc_result is not None:
            payload["qc"] = qc_result
            if qc_violations:
                payload["exit_code"] = 5
        if qc_report is not None and not qc_report.ok:
            payload["exit_code"] = 5
        return payload
    finally:
        if session is not None:
            session.release()
