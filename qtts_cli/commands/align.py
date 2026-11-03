"""`qtts align` — known-text forced alignment to SRT/VTT/ASS/JSON (FR-P0-05).

Audio longer than the aligner's five-minute limit is windowed at silences.
Plain text windows are anchored by plain ASR transcripts (no timestamps);
SRT input uses its cue boundaries as hard anchors and never runs ASR.
"""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import time
import unicodedata
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from .. import __version__
from ..align_engine import (
    MAX_ALIGN_SECONDS,
    AlignerEngine,
    clear_aligner_cache,
    resolve_align_language,
)
from ..asr_engine import AsrEngine, canonical_asr_language
from ..audio_io import duration_s, read_audio
from ..catalog import ModelInfo, scan, select_aligner, select_asr
from ..errors import InputError, UsageError
from ..lexicon import (
    Lexicon,
    MapBlock,
    PreparedText,
    display_boundary_for_spoken,
    project_text,
    protected_ranges,
    slice_prepared,
    spoken_items_to_display,
    spoken_items_to_display_positions,
)
from ..long_align import (
    MAX_WINDOW_S,
    MIN_WINDOW_S,
    AudioWindow,
    WindowAlignment,
    WindowItem,
    merge_window_items,
    monotonic_window_ranges,
    plan_windows,
    validate_srt_cues,
)
from ..sidecar import SCHEMA_VERSION
from ..subtitle_qc import check_subtitles, raise_quality_error
from ..subtitles import (
    SrtCue,
    AlignItem,
    build_cues,
    build_cues_for_ranges,
    coverage,
    ensure_full_text,
    normalize_formats,
    parse_srt,
    policy_from_args,
    render,
)
from ..utils import emit_json, sha256_file, sha256_text
from ..voice_store import utc_now

# Below this fraction the aligner result is too sparse to schedule reliably.
MIN_COVERAGE = 0.3
WARN_COVERAGE = 0.8
APPROXIMATE_WARNING = (
    "long-audio boundaries are approximate; subtitle boundaries are not guaranteed"
)
FIXED_WINDOW_WARNING = (
    "long-audio windows use fixed intervals; no qualifying silence found"
)


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


def _window_options(args) -> tuple[float, float]:
    """Validate long-audio window flags before any model is selected/loaded."""
    try:
        window_s = float(getattr(args, "window_seconds", 270.0))
        overlap_s = float(getattr(args, "window_overlap", 3.0))
    except (TypeError, ValueError) as exc:
        raise UsageError("--window-seconds/--window-overlap must be numbers") from exc
    if not math.isfinite(window_s) or not MIN_WINDOW_S <= window_s <= MAX_WINDOW_S:
        raise UsageError(
            f"--window-seconds must be between {MIN_WINDOW_S:g} and {MAX_WINDOW_S:g}",
            details={"window_seconds": window_s},
        )
    if not math.isfinite(overlap_s) or overlap_s < 2.0:
        raise UsageError(
            "--window-overlap must be at least 2 seconds",
            details={"window_overlap": overlap_s},
        )
    if MAX_WINDOW_S + overlap_s > MAX_ALIGN_SECONDS:
        raise UsageError(
            "--window-overlap is too large for the aligner's 300s limit",
            details={"window_overlap": overlap_s},
        )
    return window_s, overlap_s


def _prepare_text(args, raw_text: str, srt_cues, lexicon) -> tuple[
    PreparedText, str, str, Optional[list[tuple[int, int]]]
]:
    """Project text exactly once and return the global coordinate systems."""
    if srt_cues:
        projection = project_text(
            "\n".join(cue.text for cue in srt_cues),
            language=args.language,
            lexicon=lexicon,
        )
        normalized = projection.display_text
        if not normalized.strip():
            raise InputError("input text is empty after normalization")
        spans = projection.source_spans
        if spans is None or len(spans) != len(normalized):
            raise InputError("internal lexicon projection error: missing source spans")
        ranges: list[tuple[int, int]] = []
        raw_cursor = 0
        for cue in srt_cues:
            raw_start = raw_cursor
            raw_end = raw_start + len(cue.text)
            start = _display_boundary_for_source(spans, raw_start, cue=cue)
            end = _display_boundary_for_source(spans, raw_end, cue=cue)
            for boundary in (start, end):
                if any(
                    term_start < boundary < term_end
                    for term_start, term_end in protected_ranges(projection.mapping)
                ):
                    raise InputError(
                        "SRT cue boundary splits a lexicon term",
                        details={"cue_index": cue.index, "display_offset": boundary},
                    )
            if end <= start:
                raise InputError(
                    "SRT cue is empty after normalization",
                    details={"cue_index": cue.index},
                )
            ranges.append((start, end))
            raw_cursor = raw_end + 1  # skip the joining newline
        expected = "".join(normalized[start:end] for start, end in ranges)
        return projection, normalized, expected, ranges
    projection = project_text(raw_text, language=args.language, lexicon=lexicon)
    if not projection.display_text.strip():
        raise InputError("input text is empty after normalization")
    return projection, projection.display_text, projection.display_text, None


def _display_boundary_for_source(
    spans: Sequence[tuple[int, int]], offset: int, *, cue
) -> int:
    """Map a raw source offset to a display boundary, rejecting split spans."""
    for index, (start, end) in enumerate(spans):
        if start < offset < end:
            raise InputError(
                "SRT cue boundary splits a normalized token; move the cue boundary",
                details={"cue_index": cue.index, "source_offset": offset},
            )
    for index, (start, _end) in enumerate(spans):
        if start >= offset:
            return index
    return len(spans)


def _srt_window_ranges(
    windows: Sequence[AudioWindow],
    *,
    sr: int,
    cue_ranges: Sequence[tuple[int, int]],
    srt_cues: Sequence[SrtCue],
    display_chars: int,
) -> list[tuple[int, int]]:
    """Assign display cuts to core boundaries and reject cross-window cues."""
    boundary: dict[int, int] = {}
    for cue, (char_start, char_end) in zip(srt_cues, cue_ranges):
        for moment, char_index in ((cue.start, char_start), (cue.end, char_end)):
            sample = int(round(moment * sr))
            boundary[sample] = max(boundary.get(sample, char_index), char_index)
    ranges: list[tuple[int, int]] = []
    previous = 0
    for index, window in enumerate(windows):
        start = 0 if index == 0 else previous
        if index == len(windows) - 1:
            end = display_chars
        else:
            end = boundary.get(window.core_end_sample)
            if end is None:
                raise InputError(
                    "SRT window boundary does not map to a cue boundary",
                    details={"window": window.index, "sample": window.core_end_sample},
                )
        if end <= start:
            raise InputError(
                "SRT window has an empty text range",
                details={"window": window.index},
            )
        ranges.append((start, end))
        previous = end
    for cue_index, (char_start, char_end) in enumerate(cue_ranges):
        if not any(start <= char_start and char_end <= end for start, end in ranges):
            raise InputError(
                "SRT cue crosses an alignment window",
                details={"cue_index": srt_cues[cue_index].index},
            )
    return ranges


def _display_ranges_from_spoken(
    spoken_ranges: Sequence[tuple[int, int]],
    mapping: Sequence[MapBlock],
    display_chars: int,
) -> list[tuple[int, int]]:
    """Map global spoken ranges through the one PreparedText mapping."""
    ranges: list[tuple[int, int]] = []
    previous = 0
    for index, (spoken_start, spoken_end) in enumerate(spoken_ranges):
        if index == 0:
            start = 0
        else:
            # A cut inside a lexicon term snaps back to its display start; the
            # adjacent windows then share the whole term (overlap is allowed).
            start = display_boundary_for_spoken(
                mapping, spoken_start, edge="start"
            )
            start = min(start, previous)
        if index == len(spoken_ranges) - 1:
            end = display_chars
        else:
            end = display_boundary_for_spoken(mapping, spoken_end, edge="end")
            end = min(max(end, start + 1), display_chars)
        if end <= start:
            raise InputError(
                "window text range is empty after display mapping",
                details={"window": index},
            )
        ranges.append((start, end))
        previous = end
    return ranges


def _approximate_display_ranges(
    windows: Sequence[AudioWindow],
    projection: PreparedText,
    *,
    audio_s: float,
) -> list[tuple[int, int]]:
    """Map each window's actual audio span to a display range.

    Non-first windows start at ``audio_start_s`` (which includes left overlap),
    so adjacent display ranges may overlap.  A boundary inside a protected
    lexicon term snaps to the term start/end; the resulting ranges must stay
    non-empty, ordered, and leave at least one character for every later
    window.
    """
    total = len(projection.display_text)
    if total <= 0:
        return []
    protected = protected_ranges(projection.mapping)

    def containing(value: int) -> Optional[tuple[int, int]]:
        return next(
            (
                (term_start, term_end)
                for term_start, term_end in protected
                if term_start < value < term_end
            ),
            None,
        )

    def snap_start(value: int) -> int:
        term = containing(value)
        return value if term is None else term[0]

    def snap_end(value: int, start: int, max_end: int) -> int:
        term = containing(value)
        if term is None:
            return value
        term_start, term_end = term
        if term_end <= max_end:
            return term_end
        if term_start > start:
            return term_start
        raise InputError(
            "cannot approximate a window boundary without splitting a "
            "lexicon term",
            details={"term_start": term_start, "term_end": term_end},
        )

    ranges: list[tuple[int, int]] = []
    previous_start = 0
    previous_end = 0
    count = len(windows)
    for index, window in enumerate(windows):
        remaining = count - index - 1
        max_end = total - remaining
        if max_end <= 0:
            raise InputError(
                "approximate window text ranges cannot stay non-empty",
                details={"window": window.index},
            )
        start = (
            0
            if index == 0
            else int(round(total * window.audio_start_s / audio_s))
        )
        end = (
            total
            if index == count - 1
            else int(round(total * window.audio_end_s / audio_s))
        )
        start = max(previous_start, snap_start(start))
        # Never leave a hole before this window.  Snapping the previous end
        # back to a protected term start can otherwise drop unprotected text
        # between the two ranges; overlap is allowed, gaps are not.
        start = min(start, previous_end)
        # Every later window still needs at least one character.
        if start > max_end - 1:
            raise InputError(
                "approximate window text ranges cannot leave room for later windows",
                details={"window": window.index},
            )
        if containing(start) is not None:
            raise InputError(
                "cannot approximate a window boundary without splitting a "
                "lexicon term",
                details={"window": window.index},
            )
        end = max(previous_end, snap_end(end, start, max_end), start + 1)
        if containing(end) is not None:
            end = snap_end(end, start, max_end)
        if end > max_end:
            raise InputError(
                "approximate window text ranges cannot leave room for later windows",
                details={"window": window.index},
            )
        if end <= start:
            raise InputError(
                "approximate window text ranges cannot stay non-empty",
                details={"window": window.index},
            )
        ranges.append((start, end))
        previous_start = start
        previous_end = end
    if ranges:
        if ranges[0][0] != 0 or ranges[-1][1] != total:
            raise InputError(
                "approximate window text ranges do not cover the full reference",
                details={"reason": "incomplete_coverage"},
            )
        for previous, current in zip(ranges, ranges[1:]):
            if current[0] > previous[1]:
                raise InputError(
                    "approximate window text ranges leave a coverage hole",
                    details={"reason": "incomplete_coverage"},
                )
    return ranges


def _align_single(
    aligner: AlignerEngine,
    samples: np.ndarray,
    sr: int,
    projection: PreparedText,
    language: str,
) -> tuple[list[AlignItem], list[AlignItem], dict[str, Any]]:
    items, info = aligner.align(
        samples, sr, projection.spoken_text, language
    )
    display_items = spoken_items_to_display(
        projection.display_text,
        projection.spoken_text,
        projection.mapping,
        items,
    )
    return items, display_items, info


def _align_long_windows(
    aligner: AlignerEngine,
    samples: np.ndarray,
    sr: int,
    projection: PreparedText,
    windows: Sequence[AudioWindow],
    language: str,
    *,
    approximate: bool = False,
    tail_allowance_s: float = 0.0,
) -> tuple[list[AlignItem], dict[str, Any], list[int]]:
    """Run one aligner call per window and merge to global character times."""
    alignments: list[WindowAlignment] = []
    infos: list[dict[str, Any]] = []
    item_counts: list[int] = []
    raw_covered: dict[int, AlignItem] = {}
    for window in windows:
        if window.display_start is None or window.display_end is None:
            raise InputError(
                "internal windowing error: text range is missing",
                details={"window": window.index},
            )
        try:
            sliced = slice_prepared(
                projection, window.display_start, window.display_end
            )
        except InputError as exc:
            raise InputError(
                "window boundary splits a lexicon term",
                details={"window": window.index},
            ) from exc
        audio = samples[
            window.audio_start_sample:window.audio_end_sample
        ]
        if audio.size == 0:
            raise InputError(
                "window has empty audio", details={"window": window.index}
            )
        if sliced.spoken_text.strip():
            items, info = aligner.align(audio, sr, sliced.spoken_text, language)
        else:
            items = []
            info = {
                "align_ms": 0.0,
                "language": resolve_align_language(projection.spoken_text, language),
                "items": 0,
                "duration_s": window.audio_duration_s,
            }
        positioned = spoken_items_to_display_positions(
            sliced.display_text,
            sliced.spoken_text,
            sliced.mapping,
            items,
        )
        offset = window.audio_start_s
        global_items = [
            WindowItem(
                text=item.text,
                start_s=offset + float(item.start),
                end_s=offset + float(item.end),
                display_start=window.display_start + local_index,
                display_end=window.display_start + local_index + 1,
            )
            for local_index, item in positioned
        ]
        for item in global_items:
            char_index = item.display_start
            raw_covered.setdefault(
                char_index,
                AlignItem(
                    projection.display_text[char_index],
                    item.start_s,
                    item.end_s,
                ),
            )
        alignments.append(WindowAlignment(window=window, items=global_items))
        infos.append(info)
        item_counts.append(len(items))

    merge_stats: dict[str, Any] = {}
    timings = merge_window_items(
        projection.display_text,
        alignments,
        duration_s=len(samples) / float(sr),
        approximate=approximate,
        stats=merge_stats,
        tail_allowance_s=tail_allowance_s,
    )
    raw_items = [raw_covered[index] for index in sorted(raw_covered)]
    matched_ratio = coverage(projection.display_text, raw_items)
    flat_items = [AlignItem(timing.char, timing.start_s, timing.end_s) for timing in timings]
    total_ms = round(sum(float(info.get("align_ms") or 0.0) for info in infos), 1)
    resolved_language = resolve_align_language(projection.spoken_text, language)
    info = {
        "align_ms": total_ms,
        "language": resolved_language,
        "items": sum(item_counts),
        "duration_s": round(len(samples) / float(sr), 3),
        "coverage": matched_ratio,
        "monotonic_adjustments": int(merge_stats.get("monotonic_adjustments", 0)),
    }
    return flat_items, info, item_counts


def _stage_text(path: Path, text: str) -> tuple[str, Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(text.encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    return temporary, path


def _commit_staged(staged: Sequence[tuple[str, Path]]) -> None:
    """Commit all staged files, restoring pre-existing targets on failure."""

    def cleanup(paths: Sequence[Path]) -> None:
        for path in paths:
            try:
                if path.exists():
                    path.unlink()
            except OSError:
                pass

    try:
        for temporary, target in staged:
            parent = target.parent
            if not parent.is_dir():
                raise InputError(
                    f"output directory does not exist: {parent}",
                    details={"target": str(target)},
                )
            if target.is_dir():
                raise InputError(
                    f"output target is a directory: {target}",
                    details={"target": str(target)},
                )
            if not os.access(parent, os.W_OK):
                raise InputError(
                    f"output directory is not writable: {parent}",
                    details={"target": str(target)},
                )
            if target.exists() and not os.access(target, os.W_OK):
                raise InputError(
                    f"output target is not writable: {target}",
                    details={"target": str(target)},
                )
            if not Path(temporary).is_file():
                raise InputError(
                    "staged output disappeared before commit",
                    details={"target": str(target)},
                )
    except Exception:
        cleanup([Path(temporary) for temporary, _target in staged])
        raise

    backups: list[tuple[Path, Path]] = []
    committed: list[Path] = []
    try:
        for _temporary, target in staged:
            if target.exists():
                fd, backup_name = tempfile.mkstemp(
                    prefix=f".{target.name}.", suffix=".bak", dir=str(target.parent)
                )
                os.close(fd)
                backup = Path(backup_name)
                try:
                    os.replace(target, backup)
                except Exception:
                    try:
                        backup.unlink()
                    except OSError:
                        pass
                    raise
                backups.append((backup, target))
        for temporary, target in staged:
            os.replace(temporary, target)
            committed.append(target)
    except Exception as exc:
        for target in committed:
            try:
                if target.exists():
                    target.unlink()
            except OSError:
                pass
        restore_failures: list[tuple[Path, Path]] = []
        for backup, target in backups:
            if not backup.exists():
                continue
            try:
                os.replace(backup, target)
            except Exception:
                restore_failures.append((target, backup))
        cleanup([Path(temporary) for temporary, _target in staged])
        failed_backups = {backup for _target, backup in restore_failures}
        cleanup(
            [
                backup
                for backup, _target in backups
                if backup not in failed_backups
            ]
        )
        if restore_failures:
            preserved = ", ".join(
                f"{target} (backup: {backup})"
                for target, backup in restore_failures
            )
            raise InputError(
                "output commit failed and previous files could not be restored; "
                f"preserved backups: {preserved}",
                details={
                    "reason": "backup_restore_failed",
                    "restore_failures": [
                        {"target": str(target), "backup": str(backup)}
                        for target, backup in restore_failures
                    ],
                },
            ) from exc
        raise
    cleanup([backup for backup, _target in backups])


def _write_outputs(
    *,
    prefix: str,
    formats: Sequence[str],
    rendered: dict[str, str],
    recipe_target: Optional[str],
    recipe: dict[str, Any],
) -> dict[str, str]:
    """Serialize everything first, then commit subtitle/recipe files."""
    targets: list[tuple[str, Path]] = []
    if prefix != "-":
        targets.extend((fmt, Path(f"{prefix}.{fmt}")) for fmt in formats)
    if recipe_target not in (None, "-"):
        targets.append(("recipe", Path(recipe_target)))

    seen: dict[str, Path] = {}
    for label, target in targets:
        resolved_text = os.fspath(target.expanduser().resolve(strict=False))
        key = os.path.normcase(resolved_text)
        if sys.platform == "darwin":
            # normcase is a no-op on POSIX, but the default macOS filesystem
            # is case-insensitive.
            key = key.casefold()
        # macOS filesystem lookups are also normalization-insensitive, so the
        # same filename in NFC and NFD form must be treated as one target.
        key = unicodedata.normalize("NFC", key)
        previous = seen.get(key)
        if previous is not None:
            resolved = target.expanduser().resolve(strict=False)
            raise UsageError(
                "output targets must be distinct; "
                f"{previous} and {target} both resolve to {resolved}",
                details={
                    "targets": [
                        {"kind": "previous", "path": str(previous)},
                        {"kind": label, "path": str(target)},
                    ]
                },
            )
        seen[key] = target

    files: dict[str, str] = {}
    staged: list[tuple[str, Path]] = []
    try:
        for label, path in targets:
            if label == "recipe":
                text = json.dumps(recipe, ensure_ascii=False, indent=2) + "\n"
            else:
                text = rendered[label]
                files[label] = str(path)
            staged.append(_stage_text(path, text))
    except Exception:
        for temporary, _target in staged:
            try:
                if os.path.exists(temporary):
                    os.unlink(temporary)
            except OSError:
                pass
        raise

    if prefix == "-":
        # Everything has been computed and QC-checked before stdout is touched.
        if recipe_target not in (None, "-"):
            _commit_staged(staged)
        sys.stdout.write(rendered[formats[0]])
        sys.stdout.flush()
        files[formats[0]] = "-"
    else:
        _commit_staged(staged)
    if recipe_target == "-":
        emit_json(recipe)
    return files


def _window_payload(
    windows: Sequence[AudioWindow],
    *,
    transcripts: Optional[Sequence[Optional[str]]],
    item_counts: Optional[Sequence[int]],
) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for index, window in enumerate(windows):
        entry = {
            "index": window.index,
            "audio_start_s": round(window.audio_start_s, 3),
            "audio_end_s": round(window.audio_end_s, 3),
            "core_start_s": round(window.core_start_s, 3),
            "core_end_s": round(window.core_end_s, 3),
            "audio_start_sample": window.audio_start_sample,
            "audio_end_sample": window.audio_end_sample,
            "core_start_sample": window.core_start_sample,
            "core_end_sample": window.core_end_sample,
            "display_start": window.display_start,
            "display_end": window.display_end,
            "spoken_start": window.spoken_start,
            "spoken_end": window.spoken_end,
            "transcript_sha256": None,
            "items": None,
        }
        if transcripts is not None:
            transcript = transcripts[index]
            entry["transcript_sha256"] = (
                sha256_text(transcript) if transcript is not None else None
            )
        if item_counts is not None:
            entry["items"] = item_counts[index]
        payload.append(entry)
    return payload


def run(cfg, args) -> dict[str, Any]:
    window_s, overlap_s = _window_options(args)
    raw_text, source, srt_cues = read_text_source(args)
    approximate = bool(getattr(args, "approximate", False))
    if approximate and srt_cues:
        raise UsageError(
            "--approximate cannot be used with --srt; SRT cue boundaries are "
            "already hard anchors and do not need approximate windows"
        )
    lexicon = Lexicon.load(getattr(args, "lexicon", None))

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
    long_audio = audio_s > MAX_ALIGN_SECONDS
    warnings: list[str] = []

    if long_audio and srt_cues:
        validate_srt_cues(
            [(cue.start, cue.end) for cue in srt_cues], audio_s
        )

    # Parameter/input/timing validation happens before model selection/loading.
    projection, normalized, expected_text, cue_ranges = _prepare_text(
        args, raw_text, srt_cues, lexicon
    )

    models = scan(cfg.model_dir)
    windows: list[AudioWindow] = []
    transcript_texts: Optional[list[Optional[str]]] = None
    asr_model: Optional[ModelInfo] = None
    asr_ms = 0.0
    plan_kind: Optional[str] = None
    windowing: dict[str, Any] = {"enabled": False}

    # Both model choices happen before any weights are loaded.  Approximate
    # mode never selects ASR at all.
    aligner_model = select_aligner(models)
    if long_audio and not approximate and not srt_cues:
        canonical_asr_language(args.language)
        asr_model = select_asr(models, getattr(args, "asr_model", None))

    if long_audio:
        spoken_ranges: Optional[list[tuple[int, int]]] = None
        if srt_cues:
            cue_boundaries = []
            for cue, (char_start, char_end) in zip(srt_cues, cue_ranges or []):
                cue_boundaries.append((cue.start, char_start))
                cue_boundaries.append((cue.end, char_end))
            windows = plan_windows(
                samples,
                sr,
                window_s=window_s,
                overlap_s=overlap_s,
                cue_boundaries=cue_boundaries,
            )
            plan_kind = windows.kind
            display_ranges = _srt_window_ranges(
                windows,
                sr=sr,
                cue_ranges=cue_ranges or [],
                srt_cues=srt_cues,
                display_chars=len(normalized),
            )
            transcript_texts = None
        else:
            windows = plan_windows(
                samples,
                sr,
                window_s=window_s,
                overlap_s=overlap_s,
                approximate=approximate,
            )
            plan_kind = windows.kind
            if plan_kind == "fixed":
                warnings.append(FIXED_WINDOW_WARNING)
            if approximate:
                display_ranges = _approximate_display_ranges(
                    windows, projection, audio_s=audio_s
                )
                transcript_texts = None
            else:
                if asr_model is None:  # internal invariant
                    raise InputError("ASR model was not selected for long audio")
                clear_aligner_cache()
                asr = AsrEngine(cfg, asr_model)
                transcripts: list[str] = []
                started = time.perf_counter()
                try:
                    for window in windows:
                        audio = samples[
                            window.audio_start_sample:window.audio_end_sample
                        ]
                        result = asr.transcribe_samples(
                            audio, sr, language=args.language
                        )
                        transcripts.append(str(result.get("text") or ""))
                finally:
                    asr.release()
                    del asr
                asr_ms = (time.perf_counter() - started) * 1000.0
                spoken_ranges = monotonic_window_ranges(
                    projection.spoken_text,
                    transcripts,
                    language=args.language,
                    windows=windows,
                    duration_s=audio_s,
                )
                display_ranges = _display_ranges_from_spoken(
                    spoken_ranges, projection.mapping, len(normalized)
                )
                transcript_texts = list(transcripts)

        if approximate:
            warnings.append(APPROXIMATE_WARNING)
        if spoken_ranges is None:
            spoken_ranges = [(-1, -1)] * len(windows)
        windows = [
            window.with_text_ranges(
                display_start=display_ranges[index][0],
                display_end=display_ranges[index][1],
                spoken_start=(
                    None if spoken_ranges[index][0] < 0 else spoken_ranges[index][0]
                ),
                spoken_end=(
                    None if spoken_ranges[index][1] < 0 else spoken_ranges[index][1]
                ),
            )
            for index, window in enumerate(windows)
        ]

    if long_audio:
        # ASR must be fully released before the aligner claims accelerator memory.
        clear_aligner_cache()
    aligner = AlignerEngine(cfg, aligner_model)

    if long_audio:
        char_items, info, item_counts = _align_long_windows(
            aligner,
            samples,
            sr,
            projection,
            windows,
            args.language,
            approximate=approximate,
            tail_allowance_s=max(overlap_s, 1.0),
        )
        adjustments = int(info.get("monotonic_adjustments") or 0)
        if adjustments:
            warnings.append(
                f"approximate timing was monotonic-adjusted {adjustments} time(s)"
            )
        if srt_cues:
            cues = build_cues_for_ranges(
                normalized,
                char_items,
                cue_ranges or [],
                audio_duration=audio_s,
                repair_stats=repair_stats,
            )
            for cue, original in zip(cues, srt_cues):
                cue.index = original.index
        else:
            cues = build_cues(
                normalized,
                char_items,
                granularity=args.granularity,
                fallback_duration=audio_s,
                audio_duration=audio_s,
                policy=policy,
                warnings=warnings,
                stats=boundary_stats,
                repair_stats=repair_stats,
            )
        matched_ratio = float(info.get("coverage", 1.0))
        item_count = int(info.get("items") or 0)
        windowing = {
            "enabled": True,
            "plan": plan_kind or "silence",
            "approximate": approximate,
            "window_seconds": window_s,
            "overlap_seconds": overlap_s,
            "min_window_seconds": MIN_WINDOW_S,
            "max_window_seconds": MAX_WINDOW_S,
            "asr_model": (
                asr_model.model_id if asr_model is not None else None
            ),
            "asr_revision": asr_model.revision if asr_model is not None else None,
            "asr_fingerprint": (
                asr_model.fingerprint if asr_model is not None else None
            ),
            "matched_ratio": round(matched_ratio, 4),
            "align_items": item_count,
            "windows": _window_payload(
                windows,
                transcripts=transcript_texts,
                item_counts=item_counts,
            ),
        }
    else:
        # The <=300s path is intentionally unchanged: no ASR, same aligner call.
        items, display_items, info = _align_single(
            aligner, samples, sr, projection, args.language
        )
        if srt_cues:
            cues = build_cues_for_ranges(
                normalized,
                display_items,
                cue_ranges or [],
                audio_duration=audio_s,
                repair_stats=repair_stats,
            )
            for cue, original in zip(cues, srt_cues):
                cue.index = original.index
        else:
            cues = build_cues(
                normalized,
                display_items,
                granularity=args.granularity,
                fallback_duration=audio_s,
                audio_duration=audio_s,
                policy=policy,
                warnings=warnings,
                stats=boundary_stats,
                repair_stats=repair_stats,
            )
        matched_ratio = coverage(projection.spoken_text, items)
        item_count = len(items)

    if matched_ratio < MIN_COVERAGE:
        raise InputError(
            f"forced alignment matched only {matched_ratio:.0%} of the text; "
            "subtitle timing would be unreliable",
            details={"coverage": round(matched_ratio, 4), "items": item_count},
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
    # must be exit 4 even when QC would also report a threshold error.
    ensure_full_text(cues, expected_text)
    matched = True
    for violation in qc_report.warnings:
        warnings.append(f"subtitle QC warning: {violation.message} (cue {violation.cue})")
    if not qc_report.ok:
        raise_quality_error(qc_report)
    if long_audio:
        # Coverage and subtitle-QC warnings are appended after the windowing
        # payload is first constructed; take the only snapshot here, before
        # the recipe and JSON payload are serialized.
        windowing["warnings"] = list(warnings)

    rendered = {
        fmt: render(cues, fmt, language=info.get("language"), granularity=args.granularity)
        for fmt in formats
    }

    timing = {
        "align_ms": info.get("align_ms"),
        "model_load_ms": round(aligner.load_ms or 0.0, 1),
    }
    if long_audio:
        timing["asr_ms"] = round(asr_ms, 1)

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
        "windowing": windowing,
    }
    if lexicon is not None:
        recipe["lexicon_sha256"] = lexicon.sha256

    files = _write_outputs(
        prefix=prefix,
        formats=formats,
        rendered=rendered,
        recipe_target=recipe_target,
        recipe=recipe,
    )
    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)

    payload = {
        "ok": True,
        "stdout_used": bool(consumers),
        "audio": args.audio,
        "duration_s": round(audio_s, 3),
        "language": info.get("language"),
        "granularity": args.granularity,
        "items": item_count,
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
        "windowing": windowing,
    }
    return payload


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
    return "\n".join(lines)
