"""Held-out generation, CER/speaker-similarity scoring and gates (§10).

The model runtimes are imported lazily inside the factory seams so that
``qtts_cli.finetune`` and every ``--dry-run`` stay free of torch/qwen imports.
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..audio_io import write_audio
from ..audio_qc import cosine_similarity
from ..benchmark import mean_cosine_similarity
from ..budget import DEFAULT_MAX_NEW_TOKENS
from ..capabilities import validate_capabilities
from ..errors import InputError, ModelError
from ..plan import SAMPLING_DEFAULTS
from ..qc import levenshtein, normalize_for_qc, qc_language, text_units
from ..utils import sha256_text
from .dataset import validate_ref_audio


def _engine_factory(cfg, model):
    from ..tts_engine import TtsEngine

    return TtsEngine(cfg, model)


def _asr_factory(cfg, model):
    from ..asr_engine import AsrEngine

    return AsrEngine(cfg, model)


def _base_factory(cfg, model):
    from ..tts_engine import TtsEngine

    return TtsEngine(cfg, model)


def _clear_tts_cache() -> None:
    from ..tts_engine import clear_model_cache

    clear_model_cache()


def _clear_asr_cache() -> None:
    from ..asr_engine import clear_asr_cache

    clear_asr_cache()


def _release_engine(engine) -> None:
    if engine is None:
        return
    try:
        engine.model = None
    except Exception:
        pass
    _clear_tts_cache()
    del engine
    gc.collect()


def _release_asr(asr) -> None:
    if asr is None:
        return
    release = getattr(asr, "release", None)
    if release is not None:
        release()
    _clear_asr_cache()


def resolve_language(item, eval_language: str, dataset_language: str) -> tuple[str, str]:
    """Return ``(raw_effective, resolved)`` for one eval item."""
    raw = item.language or eval_language or dataset_language or "Auto"
    return raw, qc_language(raw, item.text)


def _valid_audio(audio: Any, sr: Any) -> tuple[np.ndarray, int]:
    data = np.asarray(audio, dtype=np.float32).reshape(-1)
    if data.size == 0:
        raise ModelError("generation returned empty audio")
    if not np.all(np.isfinite(data)):
        raise ModelError("generation returned non-finite audio")
    try:
        rate = int(sr)
    except (TypeError, ValueError):
        raise ModelError(f"generation returned an invalid sample rate: {sr!r}")
    if rate <= 0:
        raise ModelError(f"generation returned a non-positive sample rate: {rate}")
    return data, rate


def _write_audio_file(path: Path, audio: np.ndarray, sr: int) -> None:
    try:
        write_audio(path, audio, sr, "wav")
    except OSError as exc:
        raise ModelError(
            f"cannot write eval audio {path}: {exc}", details={"path": str(path)}
        ) from exc
    except Exception as exc:
        raise ModelError(
            f"cannot encode eval audio {path}: {exc}", details={"path": str(path)}
        ) from exc


def evaluate(
    cfg,
    *,
    model,
    speaker: str,
    checkpoint_path: Path,
    voice_name: Optional[str],
    items,
    dataset,
    asr_model,
    base_model,
    max_cer: Optional[float],
    min_speaker_similarity: Optional[float],
    audio_dir: Optional[Path],
    eval_language: str,
) -> dict[str, Any]:
    if audio_dir is not None:
        if audio_dir.exists() and not audio_dir.is_dir():
            raise InputError(
                f"eval audio_dir is not a directory: {audio_dir}",
                details={"path": str(audio_dir)},
            )
        try:
            audio_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ModelError(
                f"cannot create eval audio directory {audio_dir}: {exc}",
                details={"path": str(audio_dir)},
            ) from exc

    records: list[dict[str, Any]] = []
    generated: list[tuple[dict[str, Any], np.ndarray, int]] = []
    engine = None
    try:
        resolved_languages: list[str] = []
        for item in items:
            _, language = resolve_language(item, eval_language, dataset.language)
            validate_capabilities(model, mode="builtin", language=language)
            resolved_languages.append(language)
        engine = _engine_factory(cfg, model)
        for index, item in enumerate(items, 1):
            language = resolved_languages[index - 1]
            audio, sr, _info = engine.generate(
                text=item.text,
                language=language,
                mode="builtin",
                speaker=speaker,
                instruct=None,
                sampling={
                    **SAMPLING_DEFAULTS,
                    "max_new_tokens": DEFAULT_MAX_NEW_TOKENS,
                },
                seed=int(cfg.seed),
            )
            data, rate = _valid_audio(audio, sr)
            audio_path: Optional[str] = None
            if audio_dir is not None:
                target = audio_dir / f"{index:04d}.wav"
                _write_audio_file(target, data, rate)
                audio_path = str(target)
            record = {
                "index": index,
                "text_sha256": sha256_text(item.text),
                "language": language,
                "duration_s": round(len(data) / float(rate), 3),
                "audio": audio_path,
            }
            records.append(record)
            generated.append((record, data, rate))
    finally:
        _release_engine(engine)

    metrics: dict[str, Any] = {}

    if asr_model is not None:
        asr = None
        distance = 0
        units = 0
        try:
            asr = _asr_factory(cfg, asr_model)
            for record, data, rate in generated:
                result = asr.transcribe_samples(
                    data, rate, language=record["language"]
                )
                if not isinstance(result, dict) or not isinstance(
                    result.get("text"), str
                ):
                    raise ModelError("ASR returned no transcription text")
                reference = text_units(
                    normalize_for_qc(
                        _the_text(record, items), record["language"]
                    ),
                    "cer",
                )
                hypothesis = text_units(
                    normalize_for_qc(result["text"], record["language"]), "cer"
                )
                item_distance = levenshtein(reference, hypothesis)
                item_units = len(reference)
                record["cer_distance"] = item_distance
                record["cer_units"] = item_units
                record["cer"] = (
                    round(item_distance / item_units, 6) if item_units else None
                )
                distance += item_distance
                units += item_units
        finally:
            _release_asr(asr)
        metrics["cer"] = round(distance / units, 6) if units else None
        metrics["cer_distance"] = distance
        metrics["cer_units"] = units

    if base_model is not None:
        base = None
        embeddings: list[Any] = []
        try:
            validate_ref_audio(dataset.ref_audio)
            base = _base_factory(cfg, base_model)
            from ..audio_io import read_audio

            reference_audio, reference_sr = read_audio(dataset.ref_audio)
            reference_embedding = base.extract_speaker_embedding(
                reference_audio, int(reference_sr)
            )
            for record, data, rate in generated:
                embedding = base.extract_speaker_embedding(data, rate)
                embeddings.append(embedding)
                value = cosine_similarity(reference_embedding, embedding)
                record["speaker_similarity"] = (
                    None if value is None else round(float(value), 4)
                )
            aggregate, scored = mean_cosine_similarity(
                reference_embedding, embeddings
            )
        finally:
            _release_engine(base)
        metrics["speaker_similarity"] = aggregate
        metrics["speaker_segments_scored"] = scored
        metrics["speaker_segments_total"] = len(generated)

    gate_failures: list[dict[str, Any]] = []
    if max_cer is not None:
        value = metrics.get("cer")
        if value is None:
            gate_failures.append(
                {
                    "gate": "max_cer",
                    "value": None,
                    "threshold": float(max_cer),
                    "reason": "unavailable",
                }
            )
        elif float(value) > float(max_cer):
            gate_failures.append(
                {
                    "gate": "max_cer",
                    "value": value,
                    "threshold": float(max_cer),
                    "reason": "exceeded",
                }
            )
    if min_speaker_similarity is not None:
        value = metrics.get("speaker_similarity")
        if value is None:
            gate_failures.append(
                {
                    "gate": "min_speaker_similarity",
                    "value": None,
                    "threshold": float(min_speaker_similarity),
                    "reason": "unavailable",
                }
            )
        elif float(value) < float(min_speaker_similarity):
            gate_failures.append(
                {
                    "gate": "min_speaker_similarity",
                    "value": value,
                    "threshold": float(min_speaker_similarity),
                    "reason": "below",
                }
            )

    return {
        "ok": True,
        "phase": "eval",
        "dry_run": False,
        "voice": voice_name,
        "checkpoint": str(checkpoint_path),
        "speaker": speaker,
        "items": records,
        "metrics": metrics,
        "thresholds": {
            "max_cer": max_cer,
            "min_speaker_similarity": min_speaker_similarity,
        },
        "gate_failures": gate_failures,
        "exit_code": 5 if gate_failures else 0,
        "warnings": [],
    }


def _the_text(record: dict[str, Any], items) -> str:
    return items[int(record["index"]) - 1].text
