"""Offline design-run report: metrics, ranking, CSV/Markdown/HTML/reel."""

from __future__ import annotations

import csv
import html
import io
import shlex
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .audio_io import read_audio, write_audio
from .audio_qc import analyze
from .completeness import FAILURE_STATUSES, SEVERITY
from .errors import InputError
from .quality import canonical_report_text
from .utils import atomic_write_text, read_json, sha256_file

REPORT_SCHEMA_VERSION = 1
METRICS_VERSION = 1
DEFAULT_CLIP_RATIO = 0.001
DEFAULT_SILENCE_RATIO = 0.5
CSV_COLUMNS = [
    "take_id",
    "rank",
    "recommended",
    "status",
    "completion",
    "degraded",
    "duration_s",
    "sample_rate",
    "channels",
    "rms_dbfs",
    "peak_dbfs",
    "lufs_integrated",
    "clipping_ratio",
    "silence_ratio",
    "f0_" + "median" + "_hz",
    "voiced_ratio",
    "hnr_db",
    "spectral_centroid_hz",
    "hf_lf_db",
    "failure_count",
]


def run_path(root: Path, rel: Any) -> Optional[Path]:
    """Resolve ``rel`` inside ``root``; refuse traversal and symlink escapes."""
    if not rel:
        return None
    candidate = (root / str(rel)).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        return None
    return candidate


def _metric(metrics: dict[str, Any], key: str) -> Any:
    value = metrics.get(key)
    return value if value is not None else None


def _failure_count(metrics: dict[str, Any], completion: dict[str, Any]) -> int:
    """Count hard defects; missing (null) measurements never count as failures."""
    count = 0
    if completion.get("status") in FAILURE_STATUSES:
        count += 1
    clipping = metrics.get("clipping_ratio")
    if clipping is not None and clipping > DEFAULT_CLIP_RATIO:
        count += 1
    silence = metrics.get("silence_ratio")
    if silence is not None and silence > DEFAULT_SILENCE_RATIO:
        count += 1
    return count


def _rank_key(candidate: dict[str, Any]):
    metrics = candidate.get("metrics") or {}
    completion = candidate.get("completion") or {}
    severity = SEVERITY.get(completion.get("status"), 9)

    def asc(value: Any):
        return (value is None, value if value is not None else 0.0)

    return (
        candidate.get("failure_count", 0),
        severity,
        asc(metrics.get("clipping_ratio")),
        asc(metrics.get("silence_ratio")),
        # lower HNR is worse, so missing sorts last and lower values rank later
        (metrics.get("hnr_db") is None, -(metrics.get("hnr_db") or 0.0)),
        (metrics.get("voiced_ratio") is None, -(metrics.get("voiced_ratio") or 0.0)),
        candidate.get("take_id", ""),
    )


def _candidate_payload(
    run_dir: Path,
    record: dict[str, Any],
    *,
    metrics_mode: str,
) -> dict[str, Any]:
    take_id = record.get("take_id")
    wav_value = record.get("wav")
    wav_path = run_path(run_dir, wav_value)
    rel_path = None
    if wav_path is not None:
        try:
            rel_path = wav_path.relative_to(run_dir.resolve()).as_posix()
        except ValueError:
            rel_path = None
    payload: dict[str, Any] = {
        "take_id": take_id,
        "status": record.get("status"),
        "error": record.get("error"),
        "audio": {
            "path": rel_path,
            "sha256": None,
            "duration_s": record.get("duration_s"),
            "sample_rate": None,
            "channels": None,
        },
        "metrics": {},
        "completion": {"status": "failed", "reasons": ["audio_missing"]},
        "reasons": ["audio_missing"],
        "unavailable": {},
        "failure_count": 1,
        "degraded": True,
    }
    if wav_path is not None and wav_path.is_file():
        try:
            payload["audio"]["sha256"] = sha256_file(wav_path)
        except OSError:
            pass

    def fail(reason: str) -> dict[str, Any]:
        payload["status"] = "failed"
        payload["error"] = reason
        payload["completion"] = {"status": "failed", "reasons": [reason]}
        payload["reasons"] = [reason]
        payload["failure_count"] = 1
        payload["degraded"] = True
        return payload

    if str(record.get("status") or "").lower() == "failed":
        return fail(record.get("error") or "generation_failed")
    if wav_path is None or rel_path is None:
        return fail(record.get("error") or "audio_outside_run")
    if not wav_path.is_file():
        return fail(record.get("error") or "audio_missing")
    try:
        samples, sr = read_audio(wav_path, mono=False)
    except Exception as exc:  # corrupted or unsupported audio must not abort the run
        return fail(f"audio_unreadable: {exc}")
    if samples.ndim == 1:
        samples = samples[:, None]
    if samples.shape[1] > 2:
        return fail(f"unsupported_channels: {samples.shape[1]}")
    recipe = None
    recipe_path = run_path(run_dir, record.get("recipe"))
    if recipe_path is not None and recipe_path.is_file():
        try:
            recipe = read_json(recipe_path)
        except (OSError, ValueError):
            recipe = None
    try:
        metrics = analyze(
            samples,
            sr,
            text=record.get("text") or (recipe or {}).get("text"),
            language=(recipe or {}).get("language") or "auto",
            metrics=metrics_mode,
            recipe=recipe,
        )
    except Exception as exc:
        return fail(f"analysis_failed: {exc}")
    completion = metrics.get("completion") or {}
    failure = _failure_count(metrics, completion)
    degraded = completion.get("status") != "complete"
    audio = payload["audio"]
    audio.update(
        {
            "duration_s": metrics.get("duration_s"),
            "sample_rate": metrics.get("sample_rate"),
            "channels": metrics.get("channels"),
        }
    )
    payload.update(
        {
            "metrics": metrics,
            "completion": completion,
            "reasons": list(completion.get("reasons") or []),
            "unavailable": metrics.get("unavailable") or {},
            "failure_count": failure,
            "degraded": degraded,
        }
    )
    return payload


def _write_csv(path: Path, candidates: list[dict[str, Any]]) -> None:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(CSV_COLUMNS)
    for candidate in candidates:
        metrics = candidate.get("metrics") or {}
        completion = candidate.get("completion") or {}
        row = [
            candidate.get("take_id"),
            candidate.get("rank"),
            "yes" if candidate.get("recommended") else "no",
            candidate.get("status"),
            completion.get("status"),
            "yes" if candidate.get("degraded") else "no",
            _metric(metrics, "duration_s"),
            _metric(metrics, "sample_rate"),
            _metric(metrics, "channels"),
            _metric(metrics, "rms_dbfs"),
            _metric(metrics, "peak_dbfs"),
            _metric(metrics, "lufs_integrated"),
            _metric(metrics, "clipping_ratio"),
            _metric(metrics, "silence_ratio"),
            _metric(metrics, "f0_" + "median" + "_hz"),
            _metric(metrics, "voiced_ratio"),
            _metric(metrics, "hnr_db"),
            _metric(metrics, "spectral_centroid_hz"),
            _metric(metrics, "hf_lf_db"),
            candidate.get("failure_count", 0),
        ]
        writer.writerow(["" if value is None else value for value in row])
    atomic_write_text(path, buffer.getvalue())


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    design = report.get("design") or {}
    summary = report.get("summary") or {}
    lines = [
        f"# Design report: {design.get('name')}",
        "",
        f"- run: `{report.get('run_id')}`",
        f"- template: `{design.get('template')}`",
        f"- candidates: {summary.get('count', 0)} "
        f"(complete {summary.get('complete', 0)})",
        f"- best take: `{summary.get('best_take_id')}`",
        "",
        "| rank | take | status | completion | failures | recommended |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for candidate in report.get("candidates") or []:
        lines.append(
            f"| {candidate.get('rank')} | `{candidate.get('take_id')}` | "
            f"{candidate.get('status')} | {(candidate.get('completion') or {}).get('status')} | "
            f"{candidate.get('failure_count')} | "
            f"{'yes' if candidate.get('recommended') else ''} |"
        )
    if report.get("warnings"):
        lines.extend(["", "## Warnings"])
        lines.extend(f"- {warning}" for warning in report["warnings"])
    atomic_write_text(path, "\n".join(lines) + "\n")


def _write_html(path: Path, report: dict[str, Any], run_dir: Path) -> None:
    design = report.get("design") or {}
    rows = []
    for candidate in report.get("candidates") or []:
        audio = candidate.get("audio") or {}
        rel = audio.get("path") or ""
        rows.append(
            "<tr data-recommended=\"{rec}\" data-take=\"{take}\">"
            "<td class=\"rank\">{rank}</td><td>{take}</td>"
            "<td>{status}</td><td>{completion}</td><td>{failures}</td>"
            "<td><audio controls preload=\"none\" src=\"{src}\"></audio></td></tr>".format(
                rec="1" if candidate.get("recommended") else "0",
                take=html.escape(str(candidate.get("take_id"))),
                rank=candidate.get("rank"),
                status=html.escape(str(candidate.get("status"))),
                completion=html.escape(str((candidate.get("completion") or {}).get("status"))),
                failures=candidate.get("failure_count"),
                src=html.escape(str(Path(rel).as_posix())),
            )
        )
    freeze = shlex.quote(str(run_dir))
    warning_items = "".join(
        f"<li>{html.escape(str(warning))}</li>" for warning in (report.get("warnings") or [])
    )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Design audition: {html.escape(str(design.get('name')))}</title>
<style>
body{{font-family:system-ui,sans-serif;margin:2rem;background:#111;color:#eee}}
table{{border-collapse:collapse;width:100%}}td,th{{padding:.5rem;border-bottom:1px solid #333;text-align:left}}
tr[data-recommended="1"]{{background:#1d3320}}audio{{height:2rem}}
body.blind tr{{background:transparent!important}}body.blind td.rank{{visibility:hidden}}
</style></head><body>
<h1>Design audition: {html.escape(str(design.get('name')))}</h1>
<p>Run <code>{html.escape(str(report.get('run_id')))}</code></p>
<label><input type="checkbox" id="blind"> blind mode</label>
<table><thead><tr><th>rank</th><th>take</th><th>status</th><th>completion</th><th>failures</th><th>audio</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
<ul class="warnings">{warning_items}</ul>
<p>Freeze: <code>qtts design freeze {html.escape(freeze)} --take TAKE --name NAME --as design</code></p>
<script>
const blind=document.getElementById('blind');
const tbody=document.querySelector('tbody');
const original=Array.from(tbody.rows);
const apply=()=>{{
  document.body.classList.toggle('blind',blind.checked);
  const rows=blind.checked
    ? original.slice().sort((a,b)=>a.dataset.take.localeCompare(b.dataset.take))
    : original;
  rows.forEach(row=>tbody.appendChild(row));
}};
blind.addEventListener('change',apply);
</script></body></html>
"""
    atomic_write_text(path, document)


def _write_reel(path: Path, run_dir: Path, candidates: list[dict[str, Any]], gap: float) -> list[str]:
    warnings: list[str] = []
    parts = []
    sr = None
    channels = None
    for candidate in candidates:
        completion = (candidate.get("completion") or {}).get("status")
        if completion not in ("complete", "unknown"):
            continue
        audio = candidate.get("audio") or {}
        wav_path = run_path(run_dir, audio.get("path"))
        if wav_path is None or not wav_path.is_file():
            continue
        try:
            samples, sample_rate = read_audio(wav_path, mono=False)
        except Exception as exc:
            warnings.append(f"reel skipped {candidate.get('take_id')}: {exc}")
            continue
        if samples.ndim == 1:
            samples = samples[:, None]
        if sr is None:
            sr, channels = sample_rate, samples.shape[1]
        if sample_rate != sr or samples.shape[1] != channels:
            warnings.append(f"reel skipped {candidate.get('take_id')}: sample rate/channel mismatch")
            continue
        if parts and gap > 0:
            parts.append(np.zeros((int(round(gap * sr)), channels), dtype=np.float32))
        parts.append(samples)
    if not parts:
        warnings.append("reel has no complete/unknown candidates")
        return warnings
    write_audio(path, np.concatenate(parts, axis=0), int(sr or 24000), "wav")
    return warnings


def build_report(
    run_dir: str | Path,
    *,
    metrics: str = "basic",
    gap: float = 1.0,
    write_csv: bool = True,
    write_html: bool = True,
    write_reel: bool = True,
    require_full: bool = False,
    fail_on_degraded: bool = False,
) -> dict[str, Any]:
    root = Path(run_dir).expanduser()
    manifest_path = root / "run.json"
    if not manifest_path.is_file():
        raise InputError(f"not a design run directory: {root}", details={"path": str(root)})
    if not 0.0 <= float(gap) <= 30.0:
        raise InputError("--gap must be between 0 and 30 seconds")
    manifest = read_json(manifest_path)
    candidates = [
        _candidate_payload(root, record, metrics_mode=metrics)
        for record in (manifest.get("candidates") or [])
    ]
    candidates.sort(key=_rank_key)
    for rank, candidate in enumerate(candidates, start=1):
        candidate["rank"] = rank
        candidate["recommended"] = False

    complete = [item for item in candidates if (item.get("completion") or {}).get("status") == "complete"]
    if complete:
        complete[0]["recommended"] = True
    degraded = not complete
    warnings: list[str] = []
    if degraded:
        warnings.append("no complete candidate")

    mandatory = (
        "f0_" + "median" + "_hz",
        "f0_" + "mean" + "_hz",
        "f0_" + "std" + "_hz",
        "f0_" + "p10" + "_hz",
        "f0_" + "p90" + "_hz",
        "voiced_ratio",
        "hnr_db",
        "spectral_centroid_hz",
        "ltas_slope_db_per_octave",
        "hf_lf_db",
    )
    missing_full = any(
        candidate["metrics"].get(key) is None
        for candidate in candidates
        for key in mandatory
    )
    if metrics == "full" and missing_full:
        warnings.append("some full metrics are unavailable")

    measured = next(
        (item["metrics"].get("analysis") for item in candidates if item["metrics"].get("analysis")),
        {},
    )
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "metrics_version": METRICS_VERSION,
        "run_id": manifest.get("run_id"),
        "design": {
            "name": (manifest.get("design") or {}).get("name"),
            "template": (manifest.get("design") or {}).get("template"),
            "rendered_instruction": (manifest.get("design") or {}).get("rendered_instruction"),
        },
        "analysis": {
            "metrics": metrics,
            "f0_backend": measured.get("f0_backend"),
            "hnr_backend": measured.get("hnr_backend"),
            "versions": measured,
        },
        "candidates": candidates,
        "summary": {
            "count": len(candidates),
            "complete": len(complete),
            "degraded": degraded,
            "best_take_id": complete[0]["take_id"] if complete else None,
        },
        "warnings": warnings,
        "degraded": degraded,
    }

    report_dir = root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    if write_reel:
        try:
            report["warnings"].extend(
                _write_reel(report_dir / "all_candidates.wav", root, candidates, float(gap))
            )
        except Exception as exc:  # a broken reel must not lose the metrics report
            report["warnings"].append(f"reel failed: {exc}")
    atomic_write_text(report_dir / "report.json", canonical_report_text(report))
    if write_csv:
        _write_csv(report_dir / "metrics.csv", candidates)
    _write_markdown(report_dir / "report.md", report)
    if write_html:
        _write_html(report_dir / "audition.html", report, root)

    exit_code = 0
    if require_full and metrics == "full" and missing_full:
        exit_code = 5
    if fail_on_degraded and degraded:
        exit_code = 5
    report["exit_code"] = exit_code
    return report


def format_result(payload: dict[str, Any]) -> str:
    summary = payload.get("summary") or {}
    lines = [
        f"{payload.get('run_id')}: {summary.get('count', 0)} candidate(s), "
        f"complete={summary.get('complete', 0)} best={summary.get('best_take_id')}"
    ]
    for candidate in payload.get("candidates") or []:
        lines.append(
            f"  #{candidate.get('rank')} {candidate.get('take_id')} "
            f"completion={(candidate.get('completion') or {}).get('status')} "
            f"failures={candidate.get('failure_count')}"
            f"{' recommended' if candidate.get('recommended') else ''}"
        )
    for warning in payload.get("warnings") or []:
        lines.append(f"  warning: {warning}")
    return "\n".join(lines)
