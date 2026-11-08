"""`qtts finetune` argument surface and human formatting (FR-P2-07 §1)."""

from __future__ import annotations

import argparse
from typing import Any

from .. import finetune


def add_arguments(parser: argparse.ArgumentParser) -> None:
    from ..cli import _add_common

    sub = parser.add_subparsers(dest="finetune_command", required=True)

    prepare = sub.add_parser("prepare", help="build the official training JSONL")
    _add_common(prepare, suppress=True)
    prepare.add_argument("--project", required=True, help="finetune project (.yaml/.yml/.json)")
    prepare.add_argument("--tokenizer-model-path", default=argparse.SUPPRESS)
    prepare.add_argument("--raw-jsonl", default=argparse.SUPPRESS)
    prepare.add_argument("--output-jsonl", default=argparse.SUPPRESS)
    prepare.add_argument("--force", action="store_true")
    prepare.add_argument("--dry-run", action="store_true")
    prepare.set_defaults(func=finetune.prepare, formatter=format_result)

    train = sub.add_parser("train", help="run the official finetuning script")
    _add_common(train, suppress=True)
    train.add_argument("--project", required=True)
    train.add_argument("--train-jsonl", default=argparse.SUPPRESS)
    train.add_argument("--init-model-path", default=argparse.SUPPRESS)
    train.add_argument("--output-dir", default=argparse.SUPPRESS)
    train.add_argument("--finetuning-dir", default=argparse.SUPPRESS)
    train.add_argument("--python", default=argparse.SUPPRESS)
    train.add_argument("--speaker-name", default=argparse.SUPPRESS)
    train.add_argument("--batch-size", type=int, default=argparse.SUPPRESS)
    train.add_argument("--lr", type=float, default=argparse.SUPPRESS)
    train.add_argument("--epochs", type=int, default=argparse.SUPPRESS)
    train.add_argument("--checkpoint", default=argparse.SUPPRESS)
    train.add_argument("--register", default=argparse.SUPPRESS)
    train.add_argument("--register-only", action="store_true")
    train.add_argument("--force", action="store_true")
    train.add_argument("--dry-run", action="store_true")
    train.set_defaults(func=finetune.train, formatter=format_result)

    evaluate = sub.add_parser("eval", help="generate held-out audio and score it")
    _add_common(evaluate, suppress=True)
    evaluate.add_argument("--project", required=True)
    evaluate.add_argument("--voice", default=argparse.SUPPRESS)
    evaluate.add_argument("--checkpoint", default=argparse.SUPPRESS)
    evaluate.add_argument("--asr-model", default=argparse.SUPPRESS)
    evaluate.add_argument("--base-model", default=argparse.SUPPRESS)
    evaluate.add_argument("--max-cer", type=float, default=argparse.SUPPRESS)
    evaluate.add_argument("--min-speaker-similarity", type=float, default=argparse.SUPPRESS)
    evaluate.add_argument("--audio-dir", default=argparse.SUPPRESS)
    evaluate.add_argument("--dry-run", action="store_true")
    evaluate.set_defaults(func=finetune.eval, formatter=format_result)


def run(cfg, args) -> dict[str, Any]:
    return args.func(cfg, args)


def format_result(payload: dict[str, Any]) -> str:
    phase = payload.get("phase")
    if phase == "prepare":
        prefix = "would prepare" if payload.get("dry_run") else "prepared"
        lines = [
            f"{prefix} {payload.get('lines')} item(s): {payload.get('output_jsonl')}",
            f"  raw: {payload.get('raw_jsonl')}",
            f"  device: {payload.get('device')}",
        ]
        if payload.get("dry_run"):
            lines.append(f"  temporary output: {payload.get('temporary_output')}")
        return "\n".join(lines)
    if phase == "train":
        if payload.get("dry_run"):
            selected = payload.get("selected") or {}
            return (
                f"would train {payload.get('output_dir')}; "
                f"checkpoints={len(payload.get('checkpoints') or [])} "
                f"selected={selected.get('path')}"
            )
        lines = [
            f"trained {payload.get('output_dir')} "
            f"(published={payload.get('published')}, registered={payload.get('registered')})"
        ]
        selected = payload.get("selected") or {}
        if selected:
            lines.append(
                f"  selected: epoch {selected.get('epoch')} {selected.get('path')}"
            )
        register = payload.get("register")
        if register:
            lines.append(
                f"  voice: {register.get('name')} -> {register.get('path')}"
            )
        for recovery in payload.get("recoveries") or []:
            lines.append(f"  recovered: {recovery}")
        for warning in payload.get("warnings") or []:
            lines.append(f"  warning: {warning}")
        return "\n".join(lines)
    if phase == "eval":
        if payload.get("dry_run"):
            return (
                f"would evaluate {payload.get('checkpoint')} with "
                f"{payload.get('items')} item(s); models not loaded"
            )
        metrics = payload.get("metrics") or {}
        lines = [
            f"evaluated {payload.get('checkpoint')} "
            f"({len(payload.get('items') or [])} item(s))"
        ]
        if "cer" in metrics:
            lines.append(
                f"  cer {metrics.get('cer')} "
                f"({metrics.get('cer_units')} units)"
            )
        if "speaker_similarity" in metrics:
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
        return "\n".join(lines)
    return str(payload)
