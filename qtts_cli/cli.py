"""Command line entry point (`qtts`)."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

from . import __version__
from .budget import DEFAULT_MAX_NEW_TOKENS
from .config import build_config
from .errors import EXIT_INTERRUPT, QttsError, UsageError
from .quality import canonical_report_text
from .utils import emit_json, eprint


def _add_common(parser: argparse.ArgumentParser, *, suppress: bool = False) -> None:
    default = argparse.SUPPRESS if suppress else None
    flag_default = argparse.SUPPRESS if suppress else False
    parser.add_argument("--config", default=default, help="JSON config file")
    parser.add_argument("--data-dir", default=default, help="data dir (voices, cache)")
    parser.add_argument("--model-dir", default=default, help="local model directory")
    parser.add_argument("--cache-dir", default=default, help="cache directory")
    parser.add_argument("--device", default=default, choices=["auto", "cpu", "mps", "cuda", "cuda:0", "cuda:1"],
                        help="compute device (default: auto)")
    parser.add_argument("--dtype", default=default, choices=["auto", "bf16", "fp16", "fp32"],
                        help="model dtype (default: auto)")
    parser.add_argument("--seed", type=int, default=default, help="sampling seed (default: 1234)")
    parser.add_argument("--offline", action="store_true", default=flag_default,
                        help="forbid network access / downloads")
    parser.add_argument("--json", action="store_true", default=flag_default,
                        help="emit machine-readable JSON on stdout")
    parser.add_argument("--quiet", action="store_true", default=flag_default, help="suppress progress output")
    parser.add_argument("--no-cache", action="store_true", default=flag_default, help="disable the audio cache")
    parser.add_argument("--log-level", default=default,
                        choices=["debug", "info", "warning", "error"], help="log level")


class QttsArgumentParser(argparse.ArgumentParser):
    """argparse errors become ``UsageError`` so ``--json`` can emit one object."""

    def error(self, message: str) -> None:  # type: ignore[override]
        raise UsageError(message)


def _add_subtitle_policy(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--max-cue-chars", type=int, default=42,
                        help="max cue display width (CJK counts 2; default: 42)")
    parser.add_argument("--max-line-width", type=int, default=42,
                        help="max rendered line width (default: 42)")
    parser.add_argument("--max-cps", type=float, default=20.0,
                        help="max non-whitespace characters per second (default: 20)")
    parser.add_argument("--min-cue-seconds", type=float, default=0.8,
                        help="minimum display duration; short cues warn (default: 0.8)")
    parser.add_argument("--max-cue-seconds", type=float, default=7.0,
                        help="maximum display duration (default: 7.0)")
    parser.add_argument("--max-merge-gap-ms", type=int, default=300,
                        help="max silence between cues that may merge (default: 300)")
    parser.add_argument("--no-subtitle-qc", action="store_true",
                        help="disable subtitle threshold gate (structural repairs still run)")


def _add_backend_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--backend",
        default=None,
        help=(
            "TTS backend: transformers (default), dashscope, vllm-omni, or an "
            "installed plugin name (entry point group qtts.backends)"
        ),
    )
    parser.add_argument(
        "--backend-model",
        default=None,
        help="remote/plugin model id (not valid for transformers or vllm-omni)",
    )


def _add_qc_thresholds(parser: argparse.ArgumentParser) -> None:
    """QC gates are opt-in: every default is None (t10 H-E01)."""
    parser.add_argument("--max-clipping-ratio", type=float, default=None,
                        help="fail when clipping ratio exceeds this (recommended: 0.001)")
    parser.add_argument("--target-lufs", type=float, default=None,
                        help="target integrated loudness; pair with --lufs-tolerance")
    parser.add_argument("--lufs-tolerance", type=float, default=None,
                        help="allowed LU deviation from --target-lufs")
    parser.add_argument("--max-true-peak-dbtp", type=float, default=None,
                        help="fail when true peak exceeds this dBTP (recommended: -1.0)")
    parser.add_argument("--max-dc-offset", type=float, default=None,
                        help="fail when |dc offset| exceeds this (recommended: 0.01)")
    parser.add_argument("--max-silence-ratio", type=float, default=None,
                        help="fail when silence ratio exceeds this (recommended: 0.5)")
    parser.add_argument("--min-speech-rate", type=float, default=None,
                        help="fail below this many characters/second (requires text)")
    parser.add_argument("--max-speech-rate", type=float, default=None,
                        help="fail above this many characters/second (requires text)")


def build_parser() -> argparse.ArgumentParser:
    parser = QttsArgumentParser(
        prog="qtts",
        description="Local Qwen3-TTS CLI: say, voice assets, forced alignment, subtitles.",
    )
    parser.add_argument("--version", action="version", version=f"qtts {__version__}")
    _add_common(parser)
    sub = parser.add_subparsers(dest="command", required=True)

    # -- say ---------------------------------------------------------------- #
    from .commands import align as align_cmd
    from .commands import batch as batch_cmd
    from .commands import bench as bench_cmd
    from .commands import check as check_cmd
    from .commands import doctor as doctor_cmd
    from .commands import models as models_cmd
    from .commands import plugins as plugins_cmd
    from .commands import post as post_cmd
    from .commands import project as project_cmd
    from .commands import qc as qc_cmd
    from .commands import say as say_cmd
    from .commands import script as script_cmd
    from .commands import takes as takes_cmd
    from .commands import voice as voice_cmd

    say = sub.add_parser("say", help="synthesize speech (builtin/voice/design)")
    _add_common(say, suppress=True)
    say.add_argument("text", nargs="?", default=None, help="text, or '-' to read stdin")
    say.add_argument("--stdin", action="store_true",
                     help="read a single plain-text request from stdin")
    say.add_argument("--jsonl", action="store_true",
                     help="with --stdin: read JSONL requests and emit JSONL results")
    say.add_argument("--out-dir", help="JSONL default output directory (default: current dir)")
    say.add_argument("--file", help="read text from a file")
    say.add_argument("--script", help="read a plain-text or YAML/JSON multi-role script")
    say.add_argument("--voice",
                     help="builtin:Vivian | voice:NAME | design:DESCRIPTION")
    # SUPPRESS lets `say --script` distinguish an explicit CLI default from an
    # omitted option; missing values are resolved in commands.say.
    say.add_argument("--language", default=argparse.SUPPRESS,
                     help="auto|Chinese|English|... (auto is decided per segment)")
    say.add_argument("--instruct", default=argparse.SUPPRESS,
                     help="style instruction (1.7B CustomVoice / VoiceDesign)")
    say.add_argument("--out", default=None, help="output file (default: out.wav), or '-' for stdout")
    say.add_argument("--format", choices=["wav", "flac"], help="container (default: from --out)")
    say.add_argument("--subtitle", help="comma list of srt,vtt,ass,json")
    say.add_argument("--subtitle-out-prefix", help="prefix for subtitle files (or '-' for stdout)")
    say.add_argument("--recipe-out", help="write the reproducibility recipe here ('-' for stdout)")
    say.add_argument(
        "--granularity",
        choices=["sentence", "char", "word"],
        default="sentence",
        help="cue granularity (default: sentence)",
    )
    _add_subtitle_policy(say)
    say.add_argument("--max-new-tokens", type=int, default=argparse.SUPPRESS)
    say.add_argument("--pause-ms", type=int, default=argparse.SUPPRESS,
                     help="silence between segments")
    say.add_argument("--model", help="override the automatically selected model")
    say.add_argument("--lexicon", help="JSON pronunciation lexicon (schema_version 1)")
    say.add_argument("--qc", action="store_true",
                     help="measure the final assembled audio (no extra model without --reference-voice)")
    say.add_argument("--reference-voice",
                     help="clone profile used for speaker similarity (loads Base)")
    say.add_argument("--takes", type=int, default=argparse.SUPPRESS,
                     help="generate N takes (2-16); requires --select")
    say.add_argument("--select", default=argparse.SUPPRESS,
                     help="selection criteria priority, e.g. cer,speaker-sim,lufs")
    say.add_argument("--asr-model", default=argparse.SUPPRESS,
                     help="ASR model for --select cer (local path or repo id)")
    _add_qc_thresholds(say)
    _add_backend_flags(say)
    say.add_argument("--dry-run", action="store_true", help="print the plan without loading models")
    say.add_argument("--yes", action="store_true", help="non-interactive confirmation")
    say.set_defaults(func=say_cmd.run, formatter=say_cmd.format_result)

    # -- script ------------------------------------------------------------- #
    script = sub.add_parser("script", help="validate and compile multi-role scripts")
    script_sub = script.add_subparsers(dest="script_command", required=True)

    script_lint = script_sub.add_parser("lint", help="validate a script without writing a plan")
    _add_common(script_lint, suppress=True)
    script_lint.add_argument("file")
    script_lint.add_argument("--model", help="override the automatically selected model")
    script_lint.add_argument("--lexicon", help="JSON pronunciation lexicon")
    script_lint.set_defaults(func=script_cmd.lint, formatter=script_cmd.format_result)

    script_compile = script_sub.add_parser("compile", help="compile a deterministic plan")
    _add_common(script_compile, suppress=True)
    script_compile.add_argument("file")
    script_compile.add_argument("-o", "--out", required=True, help="plan JSON path ('-' for stdout)")
    script_compile.add_argument("--model", help="override the automatically selected model")
    script_compile.add_argument("--lexicon", help="JSON pronunciation lexicon")
    _add_backend_flags(script_compile)
    script_compile.set_defaults(func=script_cmd.compile, formatter=script_cmd.format_result)

    # -- batch -------------------------------------------------------------- #
    batch = sub.add_parser("batch", help="plan and run JSONL batch manifests")
    batch_sub = batch.add_subparsers(dest="batch_command", required=True)

    batch_plan = batch_sub.add_parser("plan", help="validate a manifest and write plan/status")
    _add_common(batch_plan, suppress=True)
    batch_plan.add_argument("manifest")
    batch_plan.add_argument("--out-dir", required=True)
    batch_plan.add_argument("--model", help="override the automatically selected model")
    batch_plan.add_argument("--lexicon", help="JSON pronunciation lexicon")
    _add_backend_flags(batch_plan)
    batch_plan.set_defaults(func=batch_cmd.plan, formatter=batch_cmd.format_result)

    batch_run = batch_sub.add_parser("run", help="execute a manifest")
    _add_common(batch_run, suppress=True)
    batch_run.add_argument("manifest")
    batch_run.add_argument("--out-dir", required=True)
    batch_run.add_argument("--jobs", type=int, default=1, help="scheduler concurrency (default: 1)")
    batch_run.add_argument("--resume", action="store_true", help="reuse verified done jobs")
    batch_run.add_argument("--retry", type=int, default=0, help="retry transient failures N times")
    batch_run.add_argument("--fail-fast", action="store_true",
                           help="stop scheduling after the first failure")
    batch_run.add_argument("--watch", action="store_true", help="progress on stderr")
    batch_run.add_argument("--model", help="override the automatically selected model")
    batch_run.add_argument("--lexicon", help="JSON pronunciation lexicon")
    _add_backend_flags(batch_run)
    batch_run.set_defaults(func=batch_cmd.run, formatter=batch_cmd.format_result)

    # -- voice -------------------------------------------------------------- #
    voice = sub.add_parser("voice", help="manage the voice asset library")
    voice_sub = voice.add_subparsers(dest="voice_command", required=True)

    voice_add = voice_sub.add_parser("add", help="add a clone or design voice")
    _add_common(voice_add, suppress=True)
    voice_add.add_argument("name")
    voice_add.add_argument("--kind", required=True, choices=["clone", "design"])
    voice_add.add_argument("--ref-audio", help="reference audio (clone)")
    voice_add.add_argument("--ref-text", help="reference transcript (clone, ICL mode)")
    voice_add.add_argument("--auto-ref-text", action="store_true",
                           help="transcribe --ref-audio with Qwen3-ASR for ICL")
    voice_add.add_argument("--asr-model",
                           help="ASR model path or id for --auto-ref-text")
    voice_add.add_argument("--mode", choices=["icl", "xvec"], default="icl")
    voice_add.add_argument("--description", help="natural-language voice description (design)")
    voice_add.add_argument("--sample-text", help="sample text used for design/materialization")
    voice_add.add_argument("--language", default=argparse.SUPPRESS,
                           help="reference language (default: Chinese profile; omitted leaves ASR auto)")
    voice_add.add_argument("--materialize-clone", action="store_true",
                           help="design -> generate reference -> freeze as clone prompt")
    voice_add.add_argument("--i-have-rights", action="store_true", help="confirm cloning consent")
    voice_add.add_argument("--license", help="license note stored in voice.json")
    voice_add.add_argument("--consent-note", help="free-form consent note")
    voice_add.add_argument("--tags", help="comma-separated tags")
    voice_add.add_argument("--model", help="override the automatically selected model")
    voice_add.add_argument("--yes", action="store_true")
    voice_add.set_defaults(func=voice_cmd.add, formatter=voice_cmd.format_result)

    voice_list = voice_sub.add_parser("list", help="list built-in and user voices")
    _add_common(voice_list, suppress=True)
    voice_list.set_defaults(func=voice_cmd.list_voices, formatter=voice_cmd.format_result)

    voice_inspect = voice_sub.add_parser("inspect", help="show voice details")
    _add_common(voice_inspect, suppress=True)
    voice_inspect.add_argument("name")
    voice_inspect.set_defaults(func=voice_cmd.inspect, formatter=voice_cmd.format_result)

    voice_preview = voice_sub.add_parser("preview", help="render an audition clip")
    _add_common(voice_preview, suppress=True)
    voice_preview.add_argument("name")
    voice_preview.add_argument("--text", help="preview text")
    voice_preview.add_argument("--out", help="output file")
    voice_preview.add_argument("--format", choices=["wav", "flac"])
    voice_preview.add_argument("--language", default="Chinese")
    voice_preview.set_defaults(func=voice_cmd.preview, formatter=voice_cmd.format_result)

    voice_rm = voice_sub.add_parser("rm", help="delete a user voice")
    _add_common(voice_rm, suppress=True)
    voice_rm.add_argument("name")
    voice_rm.add_argument("--yes", action="store_true")
    voice_rm.set_defaults(func=voice_cmd.rm, formatter=voice_cmd.format_result)

    voice_export = voice_sub.add_parser("export", help="export a voice as .qttsvoice")
    _add_common(voice_export, suppress=True)
    voice_export.add_argument("name")
    voice_export.add_argument("-o", "--out", required=True, help="archive path (.qttsvoice)")
    voice_export.add_argument("--force", action="store_true", help="overwrite an existing archive")
    voice_export.set_defaults(func=voice_cmd.export, formatter=voice_cmd.format_result)

    voice_import = voice_sub.add_parser("import", help="import a .qttsvoice archive")
    _add_common(voice_import, suppress=True)
    voice_import.add_argument("archive")
    voice_import.add_argument("--name", help="install under a different name")
    voice_import.add_argument("--force", action="store_true", help="replace an existing voice")
    voice_import.set_defaults(func=voice_cmd.import_, formatter=voice_cmd.format_result)

    # -- align -------------------------------------------------------------- #
    align = sub.add_parser("align", help="align known text to audio")
    _add_common(align, suppress=True)
    align.add_argument("audio")
    align.add_argument("--text")
    align.add_argument("--text-file")
    align.add_argument("--srt")
    align.add_argument("--script")
    align.add_argument("--language", default="auto")
    align.add_argument(
        "--granularity",
        choices=["sentence", "char", "word"],
        default="sentence",
        help="cue granularity (default: sentence)",
    )
    _add_subtitle_policy(align)
    align.add_argument("--format", default="srt", help="comma list of srt,vtt,ass,json")
    align.add_argument("--out-prefix", help="output prefix (default: audio stem)")
    align.add_argument("--lexicon", help="JSON pronunciation lexicon (spoken -> display)")
    align.add_argument("--recipe-out", help="write the alignment recipe here ('-' for stdout)")
    align.add_argument(
        "--window-seconds",
        type=float,
        default=270.0,
        help="long-audio target core window in seconds (240-285, default: 270)",
    )
    align.add_argument(
        "--window-overlap",
        type=float,
        default=3.0,
        help="long-audio audio context overlap in seconds (>=2, default: 3)",
    )
    align.add_argument(
        "--approximate",
        action="store_true",
        help=(
            "skip ASR for long audio; use audio-proportional text ranges and "
            "fixed-interval windows when silence planning fails (not with --srt)"
        ),
    )
    align.add_argument(
        "--asr-model",
        help="ASR model path or name for long-audio anchoring (default: best local)",
    )
    align.set_defaults(func=align_cmd.run, formatter=align_cmd.format_result)

    # -- project ------------------------------------------------------------ #
    project = sub.add_parser("project", help="create and lock reproducible project files")
    project_sub = project.add_subparsers(dest="project_command", required=True)

    project_init = project_sub.add_parser("init", help="write qtts.project.json")
    _add_common(project_init, suppress=True)
    project_init.add_argument("directory", nargs="?", default=".",
                              help="project directory (default: current directory)")
    project_init.add_argument("--force", action="store_true")
    project_init.add_argument("--voice", help="default voice reference")
    project_init.add_argument("--language", default="auto")
    project_init.add_argument("--format", choices=["wav", "flac"], default="wav")
    project_init.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    project_init.add_argument("--subtitle", help="default subtitle formats, e.g. srt,json")
    project_init.add_argument("--granularity", choices=["sentence", "char", "word"],
                              default="sentence")
    _add_subtitle_policy(project_init)
    project_init.set_defaults(func=project_cmd.init, formatter=project_cmd.format_result)

    project_lock = project_sub.add_parser("lock", help="write qtts.lock.json")
    _add_common(project_lock, suppress=True)
    project_lock.add_argument("--project", help="project file (default: ./qtts.project.json)")
    project_lock.add_argument("--out", help="lock file (default: beside the project)")
    project_lock.set_defaults(func=project_cmd.lock, formatter=project_cmd.format_result)

    # -- models ------------------------------------------------------------- #
    models = sub.add_parser("models", help="list, pull or prune local models")
    models_sub = models.add_subparsers(dest="models_command", required=True)
    models_list = models_sub.add_parser("list")
    _add_common(models_list, suppress=True)
    models_list.set_defaults(func=models_cmd.run, formatter=models_cmd.format_result)

    models_pull = models_sub.add_parser("pull")
    _add_common(models_pull, suppress=True)
    models_pull.add_argument("model_id")
    models_pull.add_argument("--source", choices=["hf", "modelscope"], default="hf")
    models_pull.add_argument("--revision")
    models_pull.set_defaults(func=models_cmd.run, formatter=models_cmd.format_result)

    models_prune = models_sub.add_parser("prune")
    _add_common(models_prune, suppress=True)
    models_prune.add_argument("--yes", action="store_true")
    models_prune.add_argument("--dry-run", action="store_true")
    models_prune.set_defaults(func=models_cmd.run, formatter=models_cmd.format_result)

    # -- doctor ------------------------------------------------------------- #
    doctor = sub.add_parser("doctor", help="diagnose the environment and models")
    _add_common(doctor, suppress=True)
    doctor.add_argument("--bench", action="store_true", help="run the local performance baseline")
    doctor.set_defaults(func=doctor_cmd.run, formatter=doctor_cmd.format_result)

    # -- check -------------------------------------------------------------- #
    check = sub.add_parser("check", help="ASR-based CER/WER quality check")
    _add_common(check, suppress=True)
    check.add_argument("audio")
    check.add_argument("--text")
    check.add_argument("--text-file")
    check.add_argument("--srt")
    check.add_argument("--language", default="auto")
    check.add_argument("--asr-model", help="ASR model path or id")
    check.add_argument("--metric", choices=["auto", "cer", "wer"], default="auto")
    check.add_argument("--threshold", type=float, default=0.03,
                       help="maximum error ratio (default: 0.03)")
    check.add_argument("--report", help="write a deterministic JSON report")
    check.set_defaults(func=check_cmd.run, formatter=check_cmd.format_result)

    # -- qc ----------------------------------------------------------------- #
    qc = sub.add_parser("qc", help="measure audio quality and apply explicit gates")
    _add_common(qc, suppress=True)
    qc.add_argument("audio")
    qc.add_argument("--text")
    qc.add_argument("--text-file")
    qc.add_argument("--language", default="auto")
    qc.add_argument("--reference-voice", help="clone profile for speaker similarity")
    qc.add_argument("--model", help="Base model override for speaker embeddings")
    _add_qc_thresholds(qc)
    qc.add_argument("--report", help="write a deterministic JSON report")
    qc.set_defaults(func=qc_cmd.run, formatter=qc_cmd.format_result)

    # -- post --------------------------------------------------------------- #
    post = sub.add_parser(
        "post", help="normalize loudness, limit true peak, resample and encode audio"
    )
    _add_common(post, suppress=True)
    post.add_argument("inputs", nargs="+", help="input .wav/.flac files")
    destination = post.add_mutually_exclusive_group()
    destination.add_argument("--out", help="single output file")
    destination.add_argument("--out-dir", help="output directory (requires --format)")
    post.add_argument("--lufs", type=float, help="target integrated loudness (LUFS)")
    post.add_argument(
        "--true-peak", type=float, help="true-peak ceiling in dBTP (default with --lufs: -1.0)"
    )
    post.add_argument("--sample-rate", type=int, help="target sample rate in Hz")
    post.add_argument(
        "--channels", choices=["keep", "mono", "stereo"], default="keep"
    )
    post.add_argument("--format", choices=["wav", "flac", "mp3", "m4a"])
    post.add_argument("--bitrate", default=argparse.SUPPRESS, help="mp3/m4a bitrate (default: 192k)")
    post.add_argument(
        "--bit-depth",
        type=int,
        choices=[16, 24],
        default=argparse.SUPPRESS,
        help="wav/flac bit depth (default: 24)",
    )
    post.add_argument("--trim-silence", action="store_true")
    post.add_argument(
        "--trim-threshold-dbfs", type=float, default=argparse.SUPPRESS
    )
    post.add_argument(
        "--trim-min-silence-s", type=float, default=argparse.SUPPRESS
    )
    post.add_argument(
        "--trim-padding-ms", type=int, default=argparse.SUPPRESS
    )
    post.add_argument(
        "--trim-mode",
        choices=["leading", "trailing", "both"],
        default=argparse.SUPPRESS,
    )
    post.add_argument(
        "--plugin",
        help="installed post-process plugin (entry point group qtts.postprocess)",
    )
    post.add_argument(
        "--plugin-param",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="plugin parameter; repeatable, value is JSON when parseable",
    )
    post.add_argument("--dry-run", action="store_true")
    post.add_argument("--report", help="write a deterministic JSON report")
    post.set_defaults(func=post_cmd.run, formatter=post_cmd.format_result)

    # -- bench -------------------------------------------------------------- #
    bench = sub.add_parser(
        "bench", help="run reproducible local TTS benchmarks (smoke/batch/longform)"
    )
    _add_common(bench, suppress=True)
    bench.add_argument("--suite", required=True, choices=["smoke", "batch", "longform"])
    bench.add_argument("--model", help="TTS model id or path override")
    bench.add_argument(
        "--voice", help="builtin:NAME or voice:NAME (design: is not supported)"
    )
    bench.add_argument(
        "--language", choices=["auto", "Chinese", "English"], default="auto"
    )
    text_group = bench.add_mutually_exclusive_group()
    text_group.add_argument("--text", help="longform text (suite longform only)")
    text_group.add_argument(
        "--text-file", help="longform UTF-8 text file (suite longform only)"
    )
    bench.add_argument(
        "--count",
        type=int,
        default=argparse.SUPPRESS,
        help="batch case count, 1-64 (default: 8)",
    )
    bench.add_argument("--runs", type=int, default=1, help="runs, 1-10 (default: 1)")
    bench.add_argument("--cer", action="store_true", help="score CER with a local ASR")
    bench.add_argument("--asr-model", help="ASR model id or path (requires --cer)")
    bench.add_argument(
        "--speaker-sim",
        action="store_true",
        help="score speaker similarity (requires --reference-voice)",
    )
    bench.add_argument(
        "--reference-voice", help="clone profile used as the similarity reference"
    )
    bench.add_argument("--max-cer", type=float, help="fail when CER exceeds this")
    bench.add_argument(
        "--min-speaker-similarity",
        type=float,
        help="fail when speaker similarity is below this",
    )
    bench.add_argument("--dry-run", action="store_true", help="plan without loading models")
    bench.add_argument("--report", help="write a deterministic JSON report")
    bench.set_defaults(func=bench_cmd.run, formatter=bench_cmd.format_result)

    # -- plugins ------------------------------------------------------------ #
    plugins = sub.add_parser("plugins", help="inspect installed TTS/plugin backends")
    plugins_sub = plugins.add_subparsers(dest="plugins_command", required=True)
    plugins_list = plugins_sub.add_parser(
        "list", help="list discovered qtts.backends and qtts.postprocess entry points"
    )
    _add_common(plugins_list, suppress=True)
    plugins_list.add_argument(
        "--strict", action="store_true",
        help="exit 3 when any discovered plugin is not usable",
    )
    plugins_list.set_defaults(
        func=plugins_cmd.list_plugins, formatter=plugins_cmd.format_result
    )

    # -- takes -------------------------------------------------------------- #
    takes = sub.add_parser("takes", help="inspect and promote multi-take assets")
    takes_sub = takes.add_subparsers(dest="takes_command", required=True)

    takes_list = takes_sub.add_parser("list", help="summarize a takes job")
    _add_common(takes_list, suppress=True)
    takes_list.add_argument("job_id")
    takes_list.set_defaults(func=takes_cmd.list_jobs, formatter=takes_cmd.format_result)

    takes_compare = takes_sub.add_parser("compare", help="compare take metrics")
    _add_common(takes_compare, suppress=True)
    takes_compare.add_argument("job_id")
    takes_compare.add_argument("--ids", help="comma-separated take ids (default: all)")
    takes_compare.set_defaults(func=takes_cmd.compare, formatter=takes_cmd.format_result)

    takes_promote = takes_sub.add_parser("promote", help="promote a take selection")
    _add_common(takes_promote, suppress=True)
    takes_promote.add_argument("job_id")
    takes_promote.add_argument("--id", type=int, required=True, help="take id")
    takes_promote.add_argument("--out", required=True, help="destination file or '-'")
    takes_promote.add_argument("--format", choices=["wav", "flac"],
                               help="must match the job format (no transcoding)")
    takes_promote.set_defaults(func=takes_cmd.promote, formatter=takes_cmd.format_result)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    # Detect --json before parsing: parse failures never produce `args`.
    want_json = "--json" in raw_argv
    parser = build_parser()
    try:
        args = parser.parse_args(raw_argv)
        want_json = want_json or bool(getattr(args, "json", False))
        cfg = build_config(args)
        if getattr(args, "jsonl", False):
            # JSONL owns stdout: per-record lines are emitted by the command.
            from .commands import say as say_cmd

            payload = say_cmd.run_jsonl(cfg, args)
        else:
            payload = args.func(cfg, args)
        canonical = bool(payload.pop("_canonical", False))
        # Canonical QC reports are byte-identical to --report files, so the
        # process exit code must not leak into the JSON document.
        exit_code = payload.get("exit_code")
        if canonical:
            payload.pop("exit_code", None)
        if payload.get("raw_stdout"):
            return int(exit_code or 0)
        if cfg.json:
            if canonical:
                sys.stdout.write(canonical_report_text(payload))
                sys.stdout.flush()
            else:
                emit_json(payload)
        else:
            text = args.formatter(payload)
            if text:
                # When a binary payload already went to stdout (--out - or
                # --out-prefix -), the human summary must not corrupt the pipe.
                if payload.get("stdout_used"):
                    print(text, file=sys.stderr)
                else:
                    print(text)
        if exit_code is not None:
            return int(exit_code)
        return 0 if payload.get("ok", True) else 1
    except QttsError as exc:
        if want_json:
            # Exactly one parseable JSON object on stderr; no extra text line.
            print(json.dumps(exc.to_dict(), ensure_ascii=False, indent=2), file=sys.stderr)
        else:
            eprint(f"error: {exc.message}")
        return exc.code
    except BrokenPipeError:  # `qtts say ... | head`
        return 0
    except KeyboardInterrupt:
        eprint("interrupted")
        return EXIT_INTERRUPT
    except (ValueError, TypeError) as exc:
        # A few surfaces (third-party argument parsing, malformed values) can
        # still raise plain builtins; never leak a traceback for bad input.
        if want_json:
            print(json.dumps({"ok": False, "error": "usage", "message": str(exc)},
                             ensure_ascii=False, indent=2), file=sys.stderr)
        else:
            eprint(f"error: {exc}")
        return 2
    except OSError as exc:
        if want_json:
            print(json.dumps({"ok": False, "error": "input", "message": str(exc)},
                             ensure_ascii=False, indent=2), file=sys.stderr)
        else:
            eprint(f"error: {exc}")
        return 4


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
