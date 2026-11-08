"""Backend runner used by ``qtts serve`` workers (FR-P2-01).

The runner is a thin adapter over :func:`qtts_cli.commands.say.run`; it never
implements a second TTS path.  ``commands.say`` is imported lazily so importing
``qtts_cli.serve`` stays free of torch/qwen_tts/dashscope/vllm.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Optional, Protocol


class SayRunner(Protocol):
    def submit(
        self,
        request: dict[str, Any],
        workdir: Path,
        cancel_check: Callable[[], bool],
    ) -> dict[str, Any]: ...


class InProcessSayRunner:
    """Run one job through ``commands.say.run`` in the calling worker thread."""

    def __init__(
        self,
        cfg,
        *,
        model: Optional[str] = None,
        engine_cache: Optional[dict[str, Any]] = None,
        aligner_cache: Optional[dict[str, Any]] = None,
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.engine_cache = engine_cache if engine_cache is not None else {}
        self.aligner_cache = aligner_cache if aligner_cache is not None else {}

    def _args(self, request: dict[str, Any], workdir: Path) -> SimpleNamespace:
        fmt = request["format"]
        out_path = workdir / f"audio.{fmt}"
        return SimpleNamespace(
            text=request["text"],
            stdin=False,
            jsonl=False,
            file=None,
            script=None,
            voice=request["voice"],
            language=request.get("language") or "auto",
            instruct=request.get("instruct"),
            out=str(out_path),
            out_dir=None,
            format=fmt,
            subtitle=list(request.get("subtitle") or []),
            subtitle_out_prefix=None,
            recipe_out=None,
            granularity="sentence",
            max_new_tokens=request.get("max_new_tokens"),
            pause_ms=request.get("pause_ms"),
            model=self.model,
            backend=getattr(self.cfg, "backend", None),
            backend_model=getattr(self.cfg, "backend_model", None),
            lexicon=None,
            qc=False,
            reference_voice=None,
            takes=None,
            select=None,
            asr_model=None,
            max_cue_chars=42,
            max_line_width=42,
            max_cps=20.0,
            min_cue_seconds=0.8,
            max_cue_seconds=7.0,
            max_merge_gap_ms=300,
            no_subtitle_qc=False,
            max_clipping_ratio=None,
            target_lufs=None,
            lufs_tolerance=None,
            max_true_peak_dbtp=None,
            max_dc_offset=None,
            max_silence_ratio=None,
            min_speech_rate=None,
            max_speech_rate=None,
            dry_run=False,
            yes=True,
        )

    def submit(
        self,
        request: dict[str, Any],
        workdir: Path,
        cancel_check: Callable[[], bool],
    ) -> dict[str, Any]:
        from ..commands.say import run as say_run

        seed = request.get("seed")
        cfg = dataclasses.replace(
            self.cfg,
            seed=int(self.cfg.seed if seed is None else seed),
            json=False,
        )
        return say_run(
            cfg,
            self._args(request, workdir),
            engine_cache=self.engine_cache,
            aligner_cache=self.aligner_cache,
            cancel_check=cancel_check,
        )
