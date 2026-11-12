"""`qtts qc` — audio quality metrics and optional explicit gates (FR-P1-07)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..audio_io import read_audio
from ..audio_qc import analyze, cosine_similarity
from ..catalog import ModelInfo, find_model, scan, select
from ..completeness import FAILURE_STATUSES
from ..errors import InputError, ModelError, QttsError, UsageError
from ..quality import (
    FULL_ONLY_THRESHOLD_NAMES,
    embedding_array,
    evaluate,
    thresholds_from_args,
    validate_thresholds,
    write_report,
)
from ..tts_engine import TtsEngine
from ..utils import sha256_file
from ..voice_store import VoiceStore


def _read_text(args) -> Optional[str]:
    text_arg = getattr(args, "text", None)
    file_arg = getattr(args, "text_file", None)
    if text_arg is not None and file_arg is not None:
        raise UsageError("--text and --text-file are mutually exclusive")
    if text_arg is not None:
        return text_arg
    if file_arg is None:
        return None
    path = Path(file_arg).expanduser()
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise InputError(
            f"cannot read text file {path}: {exc}", details={"path": str(path)}
        ) from exc


def _read_recipe(args) -> Optional[dict[str, Any]]:
    path_arg = getattr(args, "recipe", None)
    if not path_arg:
        return None
    path = Path(path_arg).expanduser()
    if not path.is_file():
        raise InputError(f"recipe file not found: {path}", details={"path": str(path)})
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise InputError(f"cannot read recipe {path}: {exc}", details={"path": str(path)}) from exc
    if not isinstance(payload, dict):
        raise InputError(f"recipe must be a JSON object: {path}", details={"path": str(path)})
    return payload


FULL_METRIC_KEYS = tuple(
    "f0_" + suffix
    for suffix in ("median_hz", "mean_hz", "std_hz", "p10_hz", "p90_hz")
) + ("voiced_ratio", "hnr_db", "spectral_centroid_hz", "ltas_slope_db_per_octave", "hf_lf_db")


def _read_audio(args) -> tuple[np.ndarray, int]:
    path = Path(args.audio).expanduser()
    if not path.is_file():
        raise InputError(f"audio file not found: {path}", details={"path": str(path)})
    samples, sr = read_audio(path, mono=False)
    if samples.ndim == 1:
        samples = samples[:, None]
    if samples.shape[0] == 0:
        raise InputError("cannot analyze empty audio", details={"path": str(path)})
    if samples.shape[1] > 2:
        raise InputError(
            f"audio has {samples.shape[1]} channels; qc supports mono/stereo only",
            details={"path": str(path), "channels": int(samples.shape[1])},
        )
    return samples, int(sr)


def profile_embedding(store: VoiceStore, name: str) -> Optional[np.ndarray]:
    """Return prompt.safetensors' speaker vector, or None to use ref.wav."""
    if store.prompt_sha256(name) is not None:
        try:
            tensors = store.load_prompt_tensors(name)
        except Exception as exc:  # corrupt/unreadable profile
            raise InputError(
                f"cannot read prompt for voice {name}: {exc}",
                details={"voice": name},
            ) from exc
        embedding = embedding_array(tensors.get("ref_spk_embedding"))
        if embedding is not None:
            return embedding
    if (store.dir_for(name) / "ref.wav").is_file():
        return None
    raise InputError(
        f"voice {name} has no reference embedding or ref.wav",
        details={"voice": name},
    )


def validate_reference_voice(store: VoiceStore, name: str) -> None:
    meta = store.require(name)
    if meta.get("kind") != "clone":
        raise InputError(
            f"--reference-voice requires a clone profile; {name} is {meta.get('kind')}",
            details={"voice": name, "kind": meta.get("kind")},
        )
    if store.prompt_sha256(name) is None and not (
        store.dir_for(name) / "ref.wav"
    ).is_file():
        raise InputError(
            f"voice {name} has neither prompt.safetensors nor ref.wav",
            details={"voice": name},
        )


def base_model(models: list[ModelInfo], override: Optional[str]) -> ModelInfo:
    """Resolve the speaker-embedding model, mapping selection failures to 3."""
    try:
        model = find_model(models, override) if override else select(models, "clone")
    except QttsError as exc:
        raise ModelError(
            f"cannot select a speaker-embedding model: {exc.message}",
            details={"model": override, "reason": exc.kind},
        ) from exc
    if model.kind != "clone":
        raise ModelError(
            f"model {model.name} cannot extract speaker embeddings (kind={model.kind})",
            details={"model": model.name, "kind": model.kind},
        )
    return model


def require_speaker_embedding(engine) -> None:
    """Run the engine's class-level capability probe (no weights loaded)."""
    probe = getattr(engine, "require_speaker_embedding", None)
    if probe is not None:
        probe()


def precheck_speaker_similarity(
    cfg, models: list[ModelInfo], override: Optional[str]
) -> ModelInfo:
    """Resolve the Base model and probe the runtime before any generation."""
    model = base_model(models, override)
    require_speaker_embedding(TtsEngine(cfg, model))
    return model


def speaker_similarity(
    cfg,
    models: list[ModelInfo],
    store: VoiceStore,
    name: str,
    samples: np.ndarray,
    sr: int,
    override: Optional[str],
    *,
    reusable_engine=None,
) -> float:
    reference = profile_embedding(store, name)
    engine = reusable_engine
    if engine is None or engine.model is None or engine.model_info.kind != "clone":
        engine = TtsEngine(cfg, base_model(models, override))
    # Class-level capability probe: no weights are loaded when unavailable.
    require_speaker_embedding(engine)
    if reference is None:
        ref_samples, ref_sr = read_audio(store.dir_for(name) / "ref.wav")
        reference = engine.extract_speaker_embedding(ref_samples, ref_sr)
    mono = samples.mean(axis=1) if samples.ndim > 1 else samples
    target = engine.extract_speaker_embedding(mono, sr)
    value = cosine_similarity(reference, target)
    if value is None:
        raise InputError(
            "speaker embeddings are empty or have mismatched shapes",
            details={"voice": name},
        )
    return round(float(value), 4)


def report_payload(
    audio: str | Path | None,
    metrics: dict[str, Any],
    thresholds: dict[str, float],
    violations: list[dict[str, Any]],
    reference_voice: Optional[str],
    metrics_mode: str = "basic",
    completion_checked: bool = False,
) -> dict[str, Any]:
    audio_path = Path(audio).expanduser() if audio is not None else None
    metric_payload = {
        key: value
        for key, value in metrics.items()
        if key
        not in (
            "duration_s",
            "sample_rate",
            "channels",
            "unavailable",
            "completion",
            "analysis",
        )
    }
    return {
        "schema_version": 1,
        "metrics_version": 1,
        "metrics_mode": metrics_mode,
        "ok": not violations,
        "audio": {
            "path": audio_path.name if audio_path else None,
            "duration_s": metrics["duration_s"],
            "sample_rate": metrics["sample_rate"],
            "channels": metrics["channels"],
            "sha256": sha256_file(audio_path) if audio_path else None,
        },
        "metrics": metric_payload,
        "completion": metrics.get("completion"),
        "completion_checked": completion_checked,
        "analysis": metrics.get("analysis"),
        "unavailable": metrics.get("unavailable") or {},
        "thresholds": thresholds,
        "violations": violations,
        "reference_voice": reference_voice,
    }


def _completion_violation(metrics: dict[str, Any]) -> Optional[dict[str, Any]]:
    completion = metrics.get("completion") or {}
    status = completion.get("status")
    if status not in FAILURE_STATUSES:
        return None
    return {
        "metric": "completion",
        "op": "status",
        "value": status,
        "threshold": "complete",
        "reason": ",".join(completion.get("reasons") or []) or None,
    }


def _require_full_violations(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    unavailable = metrics.get("unavailable") or {}
    violations: list[dict[str, Any]] = []
    for key in FULL_METRIC_KEYS:
        if metrics.get(key) is None:
            violations.append(
                {
                    "metric": key,
                    "op": "unavailable",
                    "value": None,
                    "threshold": 1.0,
                    "reason": unavailable.get(key, "unavailable"),
                }
            )
    return violations


def run(cfg, args) -> dict[str, Any]:
    thresholds = thresholds_from_args(args)
    text = _read_text(args)
    validate_thresholds(thresholds, has_text=bool(text and text.strip()))

    metrics_mode = (getattr(args, "metrics", None) or "basic").lower()
    require_full = bool(getattr(args, "require_full_metrics", False))
    if require_full and metrics_mode != "full":
        raise UsageError("--require-full-metrics requires --metrics full")
    if metrics_mode != "full":
        unsupported = sorted(set(thresholds) & FULL_ONLY_THRESHOLD_NAMES)
        if unsupported:
            flags = ", ".join("--" + name.replace("_", "-") for name in unsupported)
            raise UsageError(f"{flags} requires --metrics full")

    reference_voice = getattr(args, "reference_voice", None)
    store = VoiceStore(cfg.data_dir)
    if reference_voice:
        validate_reference_voice(store, reference_voice)

    recipe = _read_recipe(args)
    samples, sr = _read_audio(args)
    metrics = analyze(
        samples,
        sr,
        text=text,
        language=args.language,
        metrics=metrics_mode,
        recipe=recipe,
        hnr_backend=getattr(args, "hnr_backend", "praat"),
    )

    if reference_voice:
        models = scan(cfg.model_dir)
        metrics["speaker_similarity"] = speaker_similarity(
            cfg,
            models,
            store,
            reference_voice,
            samples,
            sr,
            getattr(args, "model", None),
        )
        metrics["unavailable"].pop("speaker_similarity", None)

    violations = evaluate(metrics, thresholds)
    if bool(getattr(args, "fail_on_incomplete", False)):
        completion_violation = _completion_violation(metrics)
        if completion_violation is not None:
            violations.append(completion_violation)
    if require_full:
        violations.extend(_require_full_violations(metrics))
    completion_checked = bool(
        recipe is not None
        or metrics_mode == "full"
        or getattr(args, "fail_on_incomplete", False)
    )
    payload = report_payload(
        args.audio,
        metrics,
        thresholds,
        violations,
        reference_voice,
        metrics_mode,
        completion_checked,
    )
    if getattr(args, "report", None):
        write_report(args.report, payload)
    payload["_canonical"] = True
    payload["exit_code"] = 5 if violations else 0
    return payload


def format_result(payload: dict[str, Any]) -> str:
    audio = payload.get("audio") or {}
    metrics = payload.get("metrics") or {}
    lines = [
        f"{audio.get('path')}: {audio.get('duration_s')}s "
        f"{audio.get('sample_rate')}Hz {audio.get('channels')}ch"
    ]
    for key in (
        "rms_dbfs",
        "peak_dbfs",
        "clipping_ratio",
        "dc_offset",
        "silence_ratio",
        "longest_silence_s",
        "lufs_integrated",
        "true_peak_dbtp",
        "speaker_similarity",
    ):
        value = metrics.get(key)
        lines.append(f"  {key}: {value if value is not None else 'n/a'}")
    speech = metrics.get("speech_rate")
    if speech:
        lines.append(
            f"  speech_rate: {speech.get('chars_per_s')} chars/s, "
            f"{speech.get('words_per_min')} wpm"
        )
    if payload.get("metrics_mode") == "full":
        for key in FULL_METRIC_KEYS:
            value = metrics.get(key)
            lines.append(f"  {key}: {value if value is not None else 'n/a'}")
        analysis = payload.get("analysis") or {}
        if analysis:
            lines.append(
                f"  analysis: f0={analysis.get('f0_backend')} "
                f"hnr={analysis.get('hnr_backend')}"
            )
    completion = payload.get("completion") or {}
    if completion and payload.get("completion_checked"):
        lines.append(
            f"  completion: {completion.get('status')} "
            f"({','.join(completion.get('reasons') or []) or 'no reasons'})"
        )
    for violation in payload.get("violations") or []:
        lines.append(
            f"  FAIL {violation['metric']} {violation['op']} "
            f"{violation.get('value')} (limit {violation.get('threshold')})"
        )
    lines.append("ok" if payload.get("ok") else "qc failed")
    return "\n".join(lines)
