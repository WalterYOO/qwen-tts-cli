"""Shared helpers for audiobook tests (no network, no real models)."""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import soundfile as sf

from qtts_cli import audiobook
from qtts_cli.cli import build_parser
from qtts_cli.config import build_config


def write_project(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def inline_project(tmp_path: Path, *, chapters: Optional[list[dict[str, Any]]] = None, extra: str = "") -> Path:
    if chapters is None:
        chapters = [{"title": "Chapter 1", "text": "Hello world."}]
    lines = ["schema_version: 1", "voice: builtin:Vivian"]
    if extra:
        lines.append(extra)
    lines.append("chapters:")
    for chapter in chapters:
        lines.append(f"  - title: {json.dumps(chapter['title'])}")
        if "language" in chapter:
            lines.append(f"    language: {json.dumps(chapter['language'])}")
        if "voice" in chapter:
            lines.append(f"    voice: {json.dumps(chapter['voice'])}")
        if "instruct" in chapter:
            lines.append(f"    instruct: {json.dumps(chapter['instruct'])}")
        if "text" in chapter:
            lines.append("    text: |")
            for line in str(chapter["text"]).splitlines() or [""]:
                lines.append(f"      {line}")
        if "source" in chapter:
            lines.append(f"    source: {json.dumps(chapter['source'])}")
    return write_project(tmp_path / "book.yaml", "\n".join(lines) + "\n")


class FakeRunner:
    """Contract SegmentRunner that writes real WAV files."""

    def __init__(
        self,
        *,
        sample_rate: int = 24000,
        seconds: float = 0.5,
        languages: Optional[list[str]] = None,
        on_call: Optional[Callable[["FakeRunner", int], None]] = None,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.seconds = float(seconds)
        self.languages = languages
        self.calls: list[dict[str, Any]] = []
        self.on_call = on_call

    def synthesize(
        self,
        *,
        text: str,
        out: Path,
        voice: str,
        language: str,
        instruct: Optional[str],
        seed: int,
        cancel_check,
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "text": text,
                "out": Path(out),
                "voice": voice,
                "language": language,
                "instruct": instruct,
                "seed": seed,
            }
        )
        if self.on_call is not None:
            self.on_call(self, len(self.calls))
        frames = max(1, int(round(self.seconds * self.sample_rate)))
        tone = 0.2 * np.sin(2 * np.pi * 440 * np.arange(frames) / self.sample_rate)
        sf.write(str(out), tone.astype(np.float32), self.sample_rate, subtype="PCM_16")
        languages = self.languages or [language if language != "auto" else "English"]
        return {
            "sample_rate": self.sample_rate,
            "duration_s": self.seconds,
            "dtype_actual": "bf16",
            "device_actual": "cpu",
            "languages_actual": list(languages),
        }


def make_args(
    project: Path,
    model_dir: Path,
    *extra: str,
) -> Any:
    parser = build_parser()
    return parser.parse_args(
        ["audiobook", "build", str(project), "--model-dir", str(model_dir), *extra]
    )


def run_build(
    project: Path,
    model_dir: Path,
    runner: Any = None,
    *extra: str,
    data_dir: Optional[Path] = None,
):
    argv = ["audiobook", "build", str(project), "--model-dir", str(model_dir), *extra]
    if data_dir is not None:
        argv.extend(["--data-dir", str(data_dir)])
    args = build_parser().parse_args(argv)
    cfg = build_config(args)
    return audiobook.run(cfg, args, runner=runner)


@contextlib.contextmanager
def captured():
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        yield out, err


def manifest_of(project: Path) -> dict[str, Any]:
    return json.loads((project.parent / "build" / f"{project.stem}.json").read_text(encoding="utf-8"))


def read_audio(path: Path) -> tuple[np.ndarray, int]:
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return data[:, 0], int(sr)
