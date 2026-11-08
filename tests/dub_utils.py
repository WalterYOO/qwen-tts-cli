"""Shared helpers for dub tests (fake runner, timelines, CLI)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import soundfile as sf

from qtts_cli import dub
from qtts_cli.cli import build_parser
from qtts_cli.config import build_config


def _time(seconds: float) -> str:
    millis = int(round(float(seconds) * 1000))
    hours, remainder = divmod(millis, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, ms = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def write_project(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def srt_entry(index: int, start: float, end: float, text: str) -> str:
    return f"{index}\n{_time(start)} --> {_time(end)}\n{text}\n"


def write_srt(path: Path, entries: list[tuple[int, float, float, str]], *, newline: str = "\n",
              bom: bool = False) -> Path:
    body = "\n".join(srt_entry(index, start, end, text) for index, start, end, text in entries)
    if body:
        body += "\n"
    if newline != "\n":
        body = body.replace("\n", newline)
    if bom:
        body = "﻿" + body
    path.write_text(body, encoding="utf-8", newline="")
    return path


def edl_event(index: int, start: str, end: str, *, track: str = "V", kind: str = "C",
              src: Optional[tuple[str, str]] = None) -> str:
    srcin, srcout = src or (start, end)
    return (
        f"{index:03d}  AX       {track:<5} {kind:<9} "
        f"{srcin} {srcout} {start} {end}"
    )


class FakeRunner:
    """Synthesis protocol fake that writes real mono WAV files."""

    def __init__(
        self,
        *,
        sample_rate: int = 24000,
        seconds: float = 0.5,
        languages: Optional[list[str]] = None,
        sample_rates: Optional[list[int]] = None,
        on_call=None,
        silent: bool = False,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.seconds = float(seconds)
        self.languages = languages
        self.sample_rates = list(sample_rates or [])
        self.calls: list[dict[str, Any]] = []
        self.on_call = on_call
        self.silent = silent

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
        rate = (
            self.sample_rates[len(self.calls) - 1]
            if self.sample_rates
            else self.sample_rate
        )
        frames = max(1, int(round(self.seconds * rate)))
        if self.silent:
            tone = np.zeros(frames, dtype=np.float32)
        else:
            tone = 0.2 * np.sin(2 * np.pi * 440 * np.arange(frames) / rate)
        sf.write(str(out), tone.astype(np.float32), rate, subtype="PCM_16")
        languages = self.languages or [language if language != "auto" else "English"]
        return {
            "sample_rate": rate,
            "duration_s": self.seconds,
            "dtype_actual": "bf16",
            "device_actual": "cpu",
            "languages_actual": list(languages),
        }


def make_args(
    timeline: Path,
    model_dir: Path,
    *extra: str,
    data_dir: Optional[Path] = None,
) -> Any:
    argv = ["dub", "build", str(timeline), "--model-dir", str(model_dir), *extra]
    if data_dir is not None:
        argv.extend(["--data-dir", str(data_dir)])
    return build_parser().parse_args(argv)


def run_build(
    timeline: Path,
    model_dir: Path,
    runner: Any = None,
    *extra: str,
    data_dir: Optional[Path] = None,
):
    args = make_args(timeline, model_dir, *extra, data_dir=data_dir)
    cfg = build_config(args)
    return dub.run(cfg, args, runner=runner)


def manifest_of(out: Path) -> dict[str, Any]:
    return json.loads((out.parent / f"{out.stem}.dub.json").read_text(encoding="utf-8"))


def read_audio(path: Path) -> tuple[np.ndarray, int]:
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return data[:, 0], int(sr)


def tone(frames: int, sr: int, amplitude: float = 0.2) -> np.ndarray:
    return (amplitude * np.sin(2 * np.pi * 440 * np.arange(frames) / sr)).astype(np.float32)
