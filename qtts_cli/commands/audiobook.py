"""`qtts audiobook build` CLI surface (FR-P2-03)."""

from __future__ import annotations

import argparse
from typing import Any

from .. import audiobook as audiobook_mod


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("project", help="audiobook project .yaml/.yml/.json")
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--out", help="output file (suffix selects the format)")
    target.add_argument(
        "--out-dir",
        help="output directory (default: build/ next to the project)",
    )
    parser.add_argument(
        "--format",
        choices=["wav", "flac", "m4a", "m4b", "mp3"],
        help="output format (default: wav)",
    )
    parser.add_argument("--m4b", action="store_true", help="alias for --format m4b")
    parser.add_argument("--bitrate", type=int, help="lossy bitrate in kbps, 32-320")
    parser.add_argument("--title", help="book title override")
    parser.add_argument("--author", help="book author override")
    parser.add_argument("--album", help="book album override")
    parser.add_argument("--voice", help="override every chapter voice")
    parser.add_argument("--model", help="TTS model id or path override")
    parser.add_argument("--bgm", help="background music file")
    parser.add_argument("--bgm-volume-db", type=float, help="BGM static gain, -60..0")
    parser.add_argument("--ducking-db", type=float, help="BGM duck depth, -60..0")
    parser.add_argument("--bgm-fade-ms", type=int, help="BGM fade in/out, 0-60000")
    loudness = parser.add_mutually_exclusive_group()
    loudness.add_argument("--lufs", type=float, help="per-chapter loudness target, -40..-5")
    loudness.add_argument(
        "--no-loudness", action="store_true", help="disable loudness normalization"
    )
    parser.add_argument("--true-peak", type=float, help="true-peak ceiling, -9..0 dBTP")
    parser.add_argument(
        "--paragraph-pause-ms", type=int, help="pause between paragraphs, 0-60000"
    )
    parser.add_argument(
        "--chapter-pause-ms", type=int, help="pause between chapters, 0-60000"
    )
    parser.add_argument("--sample-rate", type=int, help="master sample rate, 8000-48000")
    parser.add_argument("--keep-stems", action="store_true", help="keep segment/chapter stems")
    parser.add_argument("--force", action="store_true", help="overwrite final output/manifest")
    parser.add_argument("--dry-run", action="store_true", help="validate and print the plan")
    parser.set_defaults(func=run, formatter=format_result)


def run(cfg, args) -> dict[str, Any]:
    return audiobook_mod.run(cfg, args)


def format_result(payload: dict[str, Any]) -> str:
    if payload.get("dry_run"):
        lines = [
            f"dry run: {payload['output']['path']} [{payload['output']['format']}]",
            f"manifest: {payload['manifest']}",
            f"chapters: {len(payload.get('chapters') or [])}",
            f"paragraphs: {payload.get('paragraphs', 0)}",
        ]
        for chapter in payload.get("chapters") or []:
            lines.append(
                f"  {chapter['index']:02d} {chapter['title']} "
                f"({chapter['paragraphs']} paragraphs, {chapter['voice']})"
            )
        for warning in payload.get("warnings") or []:
            lines.append(f"warning: {warning}")
        return "\n".join(lines)
    output = payload.get("output") or {}
    lines = [
        f"wrote {output.get('path')} "
        f"({output.get('format')}, {output.get('duration_s')}s, "
        f"{output.get('size')} bytes)"
    ]
    segments = payload.get("segments") or {}
    lines.append(
        f"segments: {segments.get('total', 0)} total, "
        f"{segments.get('synthesized', 0)} synthesized, {segments.get('reused', 0)} reused"
    )
    lines.append(f"manifest: {payload.get('manifest')}")
    for warning in payload.get("warnings") or []:
        lines.append(f"warning: {warning}")
    return "\n".join(lines)
