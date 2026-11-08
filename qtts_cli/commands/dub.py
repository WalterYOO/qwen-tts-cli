"""``qtts dub build`` CLI surface (FR-P2-04)."""

from __future__ import annotations

import argparse
from typing import Any

from .. import dub as dub_mod


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("timeline", help="timeline .srt or .edl file")
    parser.add_argument("--out", help="output file (suffix selects the format)")
    parser.add_argument("--voice", help="default voice reference")
    parser.add_argument(
        "--speaker-voice",
        action="append",
        default=[],
        metavar="SPEAKER=VOICE_REF",
        help="per-speaker voice override; repeatable",
    )
    parser.add_argument("--project", help="dub project .yaml/.yml/.json")
    parser.add_argument(
        "--instruct",
        default=argparse.SUPPRESS,
        help="global style instruction (project per-speaker instruct still wins)",
    )
    parser.add_argument(
        "--language",
        default=argparse.SUPPRESS,
        help="synthesis language for every cue (default: project or auto)",
    )
    parser.add_argument("--video", help="video file for duration, mux and bed")
    parser.add_argument("--bed-db", type=float, help="video audio bed gain, -40..0 dB")
    parser.add_argument("--mux", action="store_true", help="write <out-stem>.mp4")
    parser.add_argument(
        "--format", choices=["wav", "flac", "m4a", "mp3"], help="output format (default: wav)"
    )
    parser.add_argument(
        "--min-speed",
        type=float,
        default=argparse.SUPPRESS,
        help="minimum stretch speed (default: 0.85)",
    )
    parser.add_argument(
        "--max-speed",
        type=float,
        default=argparse.SUPPRESS,
        help="maximum stretch speed (default: 1.25)",
    )
    parser.add_argument(
        "--overflow",
        choices=["warn", "fail"],
        default=argparse.SUPPRESS,
        help="overflow policy (default: warn)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=argparse.SUPPRESS,
        help="EDL timecode frame rate, 1-120 (default: 25)",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=argparse.SUPPRESS,
        help="master sample rate, 12000-48000 (default: 24000)",
    )
    loudness = parser.add_mutually_exclusive_group()
    loudness.add_argument(
        "--lufs",
        type=float,
        default=argparse.SUPPRESS,
        help="integrated loudness target, -40..-5 (default: -18)",
    )
    loudness.add_argument(
        "--no-loudness", action="store_true", help="disable loudness normalization"
    )
    parser.add_argument(
        "--true-peak",
        type=float,
        default=argparse.SUPPRESS,
        help="true-peak ceiling, -9..0 dBTP (default: -1.5)",
    )
    parser.add_argument("--stems-out", help="write per-speaker WAV stems to this dir")
    parser.add_argument("--model", help="TTS model id or path override")
    parser.add_argument("--keep-work", action="store_true", help="keep the work dir")
    parser.add_argument(
        "--force", action="store_true", help="overwrite existing outputs and refresh the cache"
    )
    parser.add_argument("--dry-run", action="store_true", help="validate and print the plan")
    parser.set_defaults(func=run, formatter=format_result)


def run(cfg, args) -> dict[str, Any]:
    return dub_mod.run(cfg, args)


def format_result(payload: dict[str, Any]) -> str:
    if payload.get("dry_run"):
        output = payload.get("output") or {}
        lines = [
            f"dry run: {output.get('path')} [{output.get('format')}]",
            f"manifest: {payload.get('manifest')}",
            f"cues: {len(payload.get('cues') or [])}",
            f"speakers: {', '.join(payload.get('speakers') or []) or '(none)'}",
        ]
        for cue in payload.get("cues") or []:
            speaker = cue.get("speaker") or "default"
            lines.append(
                f"  {cue['index']:04d} {cue['start_s']:.3f}-{cue['end_s']:.3f} "
                f"{speaker} ({cue['voice']})"
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
        f"cues: {segments.get('total', 0)} total, "
        f"{segments.get('synthesized', 0)} synthesized, {segments.get('reused', 0)} reused"
    )
    lines.append(f"manifest: {payload.get('manifest')}")
    if payload.get("mux"):
        lines.append(f"muxed: {payload['mux'].get('path')}")
    if payload.get("stems"):
        lines.append(f"stems: {payload['stems'].get('dir')}")
    for warning in payload.get("warnings") or []:
        lines.append(f"warning: {warning}")
    return "\n".join(lines)
