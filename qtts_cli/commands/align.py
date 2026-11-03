"""`qtts align` — known-text forced alignment to SRT/VTT/ASS/JSON (FR-P0-05)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

from .. import __version__
from ..align_engine import AlignerEngine
from ..audio_io import duration_s, read_audio
from ..catalog import scan, select_aligner
from ..errors import InputError, UsageError
from ..sidecar import SCHEMA_VERSION
from ..subtitle_qc import check_subtitles, raise_quality_error
from ..subtitles import (
    SrtCue,
    build_cues,
    build_cues_for_ranges,
    coverage,
    ensure_full_text,
    normalize_formats,
    parse_srt,
    policy_from_args,
    render,
)
from ..textnorm import normalize_text
from ..utils import atomic_write_json, atomic_write_text, emit_json, sha256_file, sha256_text
from ..voice_store import utc_now

# Below this fraction the aligner result is too sparse to schedule reliably.
MIN_COVERAGE = 0.3
WARN_COVERAGE = 0.8


def read_text_source(args) -> tuple[str, str, Optional[list[SrtCue]]]:
    """Return (text, source, srt_cues). SRT cue boundaries are preserved."""
    given = [name for name in ("text", "text_file", "srt", "script") if getattr(args, name, None)]
    if len(given) != 1:
        raise UsageError("provide exactly one of --text, --text-file, --srt or --script")
    if args.text:
        return args.text, "argument", None
    path = Path(getattr(args, args_key(given[0]))).expanduser()
    if not path.is_file():
        raise InputError(f"text file not found: {path}")
    raw = path.read_text(encoding="utf-8")
    if given[0] == "srt":
        cues = parse_srt(raw)
        if not cues:
            raise InputError(f"no subtitle cues found in {path}")
        # Keep cue text verbatim for range carving; normalization is applied
        # per cue below so boundaries survive (codex P1-3).
        return "\n".join(cue.text for cue in cues), str(path), cues
    if given[0] == "script" and path.suffix.lower() in (".yaml", ".yml", ".json"):
        raise UsageError("structured scripts are P1; pass plain text or an SRT file")
    return raw, str(path), None


def args_key(name: str) -> str:
    return {"text": "text", "text_file": "text_file", "srt": "srt", "script": "script"}[name]


def _cue_ranges(cue_texts: list[str]) -> tuple[str, list[tuple[int, int]]]:
    """Build the joined text and each cue's [start, end) character range."""
    ranges: list[tuple[int, int]] = []
    cursor = 0
    for text in cue_texts:
        ranges.append((cursor, cursor + len(text)))
        cursor += len(text) + 1  # the joining "\n"
    return "\n".join(cue_texts), ranges


def run(cfg, args) -> dict[str, Any]:
    raw_text, source, srt_cues = read_text_source(args)

    formats = normalize_formats(args.format) or ["srt"]
    policy = policy_from_args(args)
    boundary_stats: dict[str, int] = {
        "sentences": 0,
        "splits": 0,
        "hard_cuts": 0,
        "merges": 0,
    }
    repair_stats: dict[str, int] = {}
    prefix = args.out_prefix
    if prefix is None:
        prefix = str(Path(args.audio).with_suffix(""))
    recipe_arg = getattr(args, "recipe_out", None)

    # stdout single-consumer rule (codex P0-2), before any model load.
    if prefix == "-" and len(formats) != 1:
        raise UsageError("--out-prefix - supports exactly one format")
    consumers: list[str] = []
    if prefix == "-":
        consumers.append("--out-prefix -")
    if cfg.json:
        consumers.append("--json")
    if recipe_arg == "-":
        consumers.append("--recipe-out -")
    if len(consumers) > 1:
        raise UsageError(
            "stdout can carry only one product; got " + " + ".join(consumers) + "; choose one"
        )

    samples, sr = read_audio(args.audio)
    audio_s = duration_s(samples, sr)

    models = scan(cfg.model_dir)
    aligner_model = select_aligner(models)
    aligner = AlignerEngine(cfg, aligner_model)

    warnings: list[str] = []
    if srt_cues:
        normalized_cues = [normalize_text(cue.text, args.language) for cue in srt_cues]
        normalized, ranges = _cue_ranges(normalized_cues)
        if not normalized.strip():
            raise InputError("input text is empty after normalization")
        items, info = aligner.align(samples, sr, normalized, args.language)
        cues = build_cues_for_ranges(
            normalized, items, ranges, audio_duration=audio_s, repair_stats=repair_stats
        )
        for cue, original in zip(cues, srt_cues):
            cue.index = original.index
        # The "\n" join only feeds the aligner; cue content excludes it.
        expected_text = "".join(normalized_cues)
    else:
        normalized = normalize_text(raw_text, args.language)
        if not normalized.strip():
            raise InputError("input text is empty after normalization")
        items, info = aligner.align(samples, sr, normalized, args.language)
        cues = build_cues(
            normalized,
            items,
            granularity=args.granularity,
            fallback_duration=audio_s,
            audio_duration=audio_s,
            policy=policy,
            warnings=warnings,
            stats=boundary_stats,
            repair_stats=repair_stats,
        )
        expected_text = normalized

    matched_ratio = coverage(normalized, items)
    if matched_ratio < MIN_COVERAGE:
        raise InputError(
            f"forced alignment matched only {matched_ratio:.0%} of the text; "
            "subtitle timing would be unreliable",
            details={"coverage": round(matched_ratio, 4), "items": len(items)},
        )
    if matched_ratio < WARN_COVERAGE:
        warnings.append(
            f"alignment coverage is {matched_ratio:.0%}; unmatched characters reuse "
            "neighbouring timings"
        )

    qc_report = check_subtitles(
        cues,
        granularity=args.granularity,
        audio_duration=audio_s,
        policy=policy,
        repairs=repair_stats,
        # `--srt` boundaries are authoritative: report formatting findings as
        # warnings instead of rejecting user-supplied cues.
        formatting_strict=not bool(srt_cues),
    )
    # The text invariant is more fundamental than QC thresholds: a mismatch
    # must be exit 4 even when QC would also report a threshold error (fix #4
    # round 2).
    ensure_full_text(cues, expected_text)
    matched = True
    for violation in qc_report.warnings:
        warnings.append(f"subtitle QC warning: {violation.message} (cue {violation.cue})")
    if not qc_report.ok:
        raise_quality_error(qc_report)

    rendered = {
        fmt: render(cues, fmt, language=info.get("language"), granularity=args.granularity)
        for fmt in formats
    }

    files: dict[str, str] = {}
    if prefix == "-":
        sys.stdout.write(rendered[formats[0]])
        sys.stdout.flush()
        files[formats[0]] = "-"
    else:
        for fmt, text in rendered.items():
            path = Path(f"{prefix}.{fmt}")
            atomic_write_text(path, text)
            files[fmt] = str(path)

    timing = {"align_ms": info.get("align_ms"), "model_load_ms": round(aligner.load_ms or 0.0, 1)}

    recipe_target: Optional[str] = None
    if recipe_arg:
        recipe_target = "-" if recipe_arg == "-" else str(Path(recipe_arg).expanduser())
    recipe = {
        "schema_version": SCHEMA_VERSION,
        "qtts_version": __version__,
        "created_at": utc_now(),
        "generator": "qtts align",
        "audio": args.audio,
        "audio_sha256": sha256_file(args.audio),
        "input": {"text_sha256": sha256_text(normalized), "chars": len(normalized),
                  "source": source, "srt_preserved": bool(srt_cues)},
        "aligner": {"model": aligner_model.model_id, "revision": aligner_model.revision,
                    "fingerprint": aligner_model.fingerprint, "language": info.get("language")},
        "language": info.get("language"),
        "granularity": args.granularity,
        "subtitle_policy": policy.to_dict(),
        "subtitle_boundaries": dict(boundary_stats),
        "subtitle_qc": qc_report.to_dict(),
        "coverage": round(matched_ratio, 4),
        "cues": [cue.to_dict() for cue in cues],
        "timing": timing,
    }
    if recipe_target == "-":
        emit_json(recipe)
    elif recipe_target is not None:
        atomic_write_json(recipe_target, recipe)

    return {
        "ok": True,
        "stdout_used": bool(consumers),
        "audio": args.audio,
        "duration_s": round(audio_s, 3),
        "language": info.get("language"),
        "granularity": args.granularity,
        "items": len(items),
        "cues": len(cues),
        "coverage": round(matched_ratio, 4),
        "srt_preserved": bool(srt_cues),
        "files": files,
        "recipe": recipe_target if recipe_target else None,
        "text_matches_source": matched,
        "source": source,
        "timing": timing,
        "subtitle_qc": qc_report.to_dict(),
        "subtitle_boundaries": dict(boundary_stats),
        "warnings": warnings,
    }


def format_result(payload: dict[str, Any]) -> str:
    lines = [
        f"aligned {payload['audio']} ({payload['duration_s']}s, {payload['language']}, "
        f"{payload['items']} tokens -> {payload['cues']} cues, "
        f"coverage {payload['coverage']:.0%})"
    ]
    for fmt, path in payload["files"].items():
        lines.append(f"  {fmt}: {path}")
    if payload.get("recipe"):
        lines.append(f"  recipe: {payload['recipe']}")
    lines.append(f"  align {payload['timing']['align_ms']}ms, load {payload['timing']['model_load_ms']}ms")
    for warning in payload.get("warnings") or []:
        lines.append(f"warning: {warning}")
    return "\n".join(lines)
