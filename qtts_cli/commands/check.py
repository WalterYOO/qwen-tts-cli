"""`qtts check` — ASR-based CER/WER quality check (FR-P1-03)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..asr_engine import AsrEngine, canonical_asr_language
from ..audio_io import read_audio
from ..audio_qc import analyze
from ..catalog import scan, select_asr
from ..errors import InputError, UsageError
from ..qc import (
    error_blocks,
    error_ratio,
    levenshtein,
    normalize_for_qc,
    qc_language,
    resolve_metric,
    text_units,
)
from ..quality import write_report
from ..subtitles import parse_srt
from ..textnorm import NORMALIZATION_VERSION
from ..utils import read_json, sha256_file

MAX_CHECK_SECONDS = 300.0


def _provided_inputs(args) -> list[str]:
    return [
        name
        for name in ("text", "text_file", "srt")
        if getattr(args, name, None) is not None
    ]


def _read_reference(args) -> str:
    provided = _provided_inputs(args)
    if len(provided) != 1:
        raise UsageError(
            "check requires exactly one of --text, --text-file or --srt"
        )
    if args.text is not None:
        return args.text
    source = args.text_file if args.text_file is not None else args.srt
    path = Path(source).expanduser()
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise InputError(
            f"cannot read text file {path}: {exc}", details={"path": str(path)}
        ) from exc
    if args.srt is not None:
        cues = parse_srt(raw)
        if not cues:
            raise InputError(
                f"SRT file has no cues: {path}", details={"path": str(path)}
            )
        return " ".join(cue.text for cue in cues)
    return raw


def _read_audio(path: str) -> tuple[np.ndarray, int]:
    target = Path(path).expanduser()
    if not target.is_file():
        raise InputError(
            f"audio file not found: {target}", details={"path": str(target)}
        )
    samples, sr = read_audio(target, mono=False)
    if samples.ndim == 1:
        samples = samples[:, None]
    if samples.shape[0] == 0:
        raise InputError("cannot check empty audio", details={"path": str(target)})
    if not np.all(np.isfinite(samples)):
        raise InputError(
            "audio contains NaN or Inf samples", details={"path": str(target)}
        )
    duration = float(samples.shape[0]) / float(sr)
    if duration > MAX_CHECK_SECONDS:
        raise InputError(
            f"audio is {duration:.1f}s; `check` supports at most "
            f"{MAX_CHECK_SECONDS:.0f}s in Wave 3a (long-audio windowing is a later FR)",
            details={"duration_s": round(duration, 3), "limit_s": MAX_CHECK_SECONDS},
        )
    return samples, int(sr)


def _report(
    args,
    reference: str,
    hypothesis: str,
    *,
    metric: str,
    qc_lang: str,
    asr_model,
    asr_language: Optional[str],
    ratio: float,
    distance: int,
    errors: list[dict[str, Any]],
    duration: float,
    sample_rate: int,
    channels: int,
) -> dict[str, Any]:
    audio_path = Path(args.audio).expanduser()
    ref_units = len(text_units(reference, metric))
    hyp_units = len(text_units(hypothesis, metric))
    return {
        "schema_version": 1,
        "ok": ratio <= float(args.threshold),
        "audio": {
            "path": audio_path.name,
            "duration_s": round(duration, 4),
            "sample_rate": sample_rate,
            "channels": channels,
            "sha256": sha256_file(audio_path),
        },
        "metric": metric,
        "threshold": float(args.threshold),
        "ratio": ratio,
        "distance": distance,
        "reference_units": ref_units,
        "units": {
            "metric": metric,
            "reference": ref_units,
            "hypothesis": hyp_units,
        },
        "reference": reference,
        "hypothesis": hypothesis,
        "errors": errors,
        "comparison_language": qc_lang,
        "asr": {
            "model": asr_model.model_id,
            "revision": asr_model.revision,
            "language": asr_language,
        },
        "normalization": {
            "version": NORMALIZATION_VERSION,
            "language": qc_lang,
            "metric": metric,
        },
    }


def run(cfg, args) -> dict[str, Any]:
    # Language syntax and support are checked before anything else, so bad
    # input exits 2 even when the audio or ASR model is missing (t10 P1-1).
    asr_language = canonical_asr_language(args.language)
    threshold = float(args.threshold)
    if not math.isfinite(threshold) or threshold < 0:
        raise UsageError("--threshold must be finite and >= 0")

    reference_text = _read_reference(args)
    samples, sr = _read_audio(args.audio)

    models = scan(cfg.model_dir)
    model = select_asr(models, getattr(args, "asr_model", None))

    qc_lang = qc_language(args.language, reference_text)
    metric = resolve_metric(args.language, reference_text, args.metric)

    qc_payload = None
    qc_metrics_mode = getattr(args, "qc_metrics", None)
    qc_recipe = None
    recipe_path = getattr(args, "qc_recipe", None)
    if recipe_path and not qc_metrics_mode:
        raise UsageError("--qc-recipe requires --qc-metrics")
    if recipe_path:
        try:
            qc_recipe = read_json(Path(recipe_path).expanduser())
        except (OSError, ValueError) as exc:
            raise InputError(
                f"cannot read --qc-recipe {recipe_path}: {exc}",
                details={"path": str(recipe_path)},
            ) from exc
        if not isinstance(qc_recipe, dict):
            raise InputError("--qc-recipe must contain a JSON object")
    if qc_metrics_mode:
        measured = analyze(
            samples,
            sr,
            text=reference_text,
            language=qc_lang,
            metrics=qc_metrics_mode,
            recipe=qc_recipe,
            hnr_backend=getattr(args, "hnr_backend", "praat"),
        )
        qc_payload = {
            "metrics_mode": qc_metrics_mode,
            "metrics": {
                key: value
                for key, value in measured.items()
                if key not in ("unavailable", "completion", "analysis")
            },
            "unavailable": measured.get("unavailable") or {},
            "completion": measured.get("completion"),
            "analysis": measured.get("analysis"),
            "recipe": Path(recipe_path).expanduser().name if recipe_path else None,
        }

    engine = AsrEngine(cfg, model)
    transcription = engine.transcribe(
        args.audio, language=asr_language, timestamps=False
    )

    reference = normalize_for_qc(reference_text, qc_lang)
    hypothesis = normalize_for_qc(transcription["text"], qc_lang)
    separator = " " if metric == "wer" else ""
    ref_units = text_units(reference, metric)
    hyp_units = text_units(hypothesis, metric)
    distance = levenshtein(ref_units, hyp_units)
    ratio = error_ratio(distance, ref_units)
    errors = error_blocks(ref_units, hyp_units, separator=separator, language=qc_lang)

    payload = _report(
        args,
        reference,
        hypothesis,
        metric=metric,
        qc_lang=qc_lang,
        asr_model=model,
        asr_language=transcription["language"],
        ratio=ratio,
        distance=distance,
        errors=errors,
        duration=float(samples.shape[0]) / float(sr),
        sample_rate=sr,
        channels=int(samples.shape[1]),
    )
    if qc_payload is not None:
        payload["qc"] = qc_payload
    if getattr(args, "report", None):
        write_report(args.report, payload)
    payload["_canonical"] = True
    payload["exit_code"] = 0 if payload["ok"] else 5
    return payload


def format_result(payload: dict[str, Any]) -> str:
    ratio = float(payload["ratio"])
    threshold = float(payload["threshold"])
    lines = [
        f"{payload['metric']} {ratio:.4%} ({payload['distance']}/"
        f"{payload['reference_units']}) threshold {threshold:.4%}: "
        f"{'ok' if payload['ok'] else 'FAILED'}"
    ]
    asr = payload.get("asr") or {}
    lines.append(
        f"  asr={asr.get('model')} language={asr.get('language')} "
        f"audio={payload.get('audio', {}).get('path')}"
    )
    for block in payload.get("errors") or []:
        lines.append(
            f"  {block['op']}: ref={block['ref']!r} hyp={block['hyp']!r} "
            f"ref_span={block['ref_span']} hyp_span={block['hyp_span']}"
        )
    return "\n".join(lines)
