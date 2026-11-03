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
    from .commands import doctor as doctor_cmd
    from .commands import models as models_cmd
    from .commands import say as say_cmd
    from .commands import voice as voice_cmd

    say = sub.add_parser("say", help="synthesize speech (builtin/voice/design)")
    _add_common(say, suppress=True)
    say.add_argument("text", nargs="?", default=None, help="text, or '-' to read stdin")
    say.add_argument("--file", help="read text from a file")
    say.add_argument("--script", help="read a plain-text script from a file")
    say.add_argument("--voice", required=True,
                     help="builtin:Vivian | voice:NAME | design:DESCRIPTION")
    say.add_argument("--language", default="auto",
                     help="auto|Chinese|English|... (auto is decided per segment)")
    say.add_argument("--instruct", help="style instruction (1.7B CustomVoice / VoiceDesign)")
    say.add_argument("--out", default="out.wav", help="output file, or '-' for stdout")
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
    say.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    say.add_argument("--pause-ms", type=int, default=300, help="silence between segments")
    say.add_argument("--model", help="override the automatically selected model")
    say.add_argument("--dry-run", action="store_true", help="print the plan without loading models")
    say.add_argument("--yes", action="store_true", help="non-interactive confirmation")
    say.set_defaults(func=say_cmd.run, formatter=say_cmd.format_result)

    # -- voice -------------------------------------------------------------- #
    voice = sub.add_parser("voice", help="manage the voice asset library")
    voice_sub = voice.add_subparsers(dest="voice_command", required=True)

    voice_add = voice_sub.add_parser("add", help="add a clone or design voice")
    _add_common(voice_add, suppress=True)
    voice_add.add_argument("name")
    voice_add.add_argument("--kind", required=True, choices=["clone", "design"])
    voice_add.add_argument("--ref-audio", help="reference audio (clone)")
    voice_add.add_argument("--ref-text", help="reference transcript (clone, ICL mode)")
    voice_add.add_argument("--mode", choices=["icl", "xvec"], default="icl")
    voice_add.add_argument("--description", help="natural-language voice description (design)")
    voice_add.add_argument("--sample-text", help="sample text used for design/materialization")
    voice_add.add_argument("--language", default="Chinese")
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
    align.add_argument("--format", default="srt", help="comma list of srt,vtt,ass,json")
    align.add_argument("--out-prefix", help="output prefix (default: audio stem)")
    align.add_argument("--recipe-out", help="write the alignment recipe here ('-' for stdout)")
    align.set_defaults(func=align_cmd.run, formatter=align_cmd.format_result)

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
        payload = args.func(cfg, args)
        if cfg.json:
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
