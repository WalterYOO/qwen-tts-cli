"""Runtime completion engine behind the hidden ``qtts __complete`` command.

The engine walks the live argparse tree instead of maintaining a second copy
of the CLI grammar.  Dynamic values (voices/models) are read-only directory
metadata; completion never loads a model, touches the network, or writes a
file.  Any internal failure is silent by design: a broken TAB press must not
print a traceback into the user's terminal.
"""

from __future__ import annotations

import argparse
import os
import sys
import unicodedata
from pathlib import Path
from typing import Any, Optional

_MAX_CANDIDATES = 1000
_MAX_CANDIDATE_LEN = 4096

_STATIC_SPEAKERS = [
    "Vivian",
    "Serena",
    "Uncle_Fu",
    "Ryan",
    "Aiden",
    "Ono_Anna",
    "Sohee",
    "Eric",
    "Dylan",
]

_LANGUAGES = [
    "Auto",
    "Chinese",
    "English",
    "German",
    "Italian",
    "Portuguese",
    "Spanish",
    "Japanese",
    "Korean",
    "French",
    "Russian",
    "Cantonese",
    "Beijing Dialect",
    "Sichuan Dialect",
]

_BACKENDS = ["transformers", "dashscope", "vllm-omni"]

_FILE_DESTS = {
    "archive",
    "audio",
    "audio_dir",
    "base_model",
    "bgm",
    "cache_dir",
    "checkpoint",
    "config",
    "data_dir",
    "directory",
    "file",
    "init_model_path",
    "inputs",
    "jsonl",
    "lexicon",
    "manifest",
    "model_dir",
    "out",
    "out_dir",
    "out_prefix",
    "project",
    "rc_file",
    "recipe_out",
    "ref_audio",
    "report",
    "script",
    "srt",
    "subtitle_out_prefix",
    "text_file",
    "timeline",
    "tokenizer_model_path",
    "video",
}
_DIR_DESTS = {"audio_dir", "cache_dir", "data_dir", "dir", "directory", "model_dir", "out_dir"}
_VOICE_DESTS = {"voice", "reference_voice"}
_MODEL_DESTS = {"model", "model_id"}
_SHELL_VALUES = ["auto", "bash", "zsh", "fish"]

_ZERO_ACTIONS = (
    argparse._StoreTrueAction,
    argparse._StoreFalseAction,
    argparse._CountAction,
    argparse._HelpAction,
    argparse._VersionAction,
    argparse._StoreConstAction,
    argparse._AppendConstAction,
)


def _parser() -> argparse.ArgumentParser:
    # Imported lazily: cli imports commands.completion, which imports this
    # module during normal CLI startup.
    from ..cli import build_parser

    return build_parser()


def tokenize(line: str) -> list[str]:
    """A small, non-expanding POSIX-ish tokenizer for ``COMP_LINE``."""
    words: list[str] = []
    buf: list[str] = []
    quote: Optional[str] = None
    escaped = False
    last_was_escaped = False
    started = False
    for char in line:
        if escaped:
            buf.append(char)
            escaped = False
            last_was_escaped = True
            started = True
        else:
            last_was_escaped = False
        if last_was_escaped:
            continue
        if quote:
            if char == quote:
                quote = None
            else:
                buf.append(char)
        elif char == "\\":
            escaped = True
            started = True
        elif char in ("'", '"'):
            quote = char
            started = True
        elif char.isspace():
            if started or buf:
                words.append("".join(buf))
                buf = []
                started = False
        else:
            buf.append(char)
            started = True
    if escaped:
        buf.append("\\")
    if started or buf or quote:
        words.append("".join(buf))
    if (
        line
        and line[-1].isspace()
        and quote is None
        and not escaped
        and not last_was_escaped
    ):
        words.append("")
    return words


def _strip_program(words: list[str], cword: int) -> tuple[int, list[str]]:
    if not words:
        return max(cword, 0), words
    first = Path(words[0]).name.lower()
    if first in {"qtts", "qtts.exe"}:
        return max(cword - 1, 0), words[1:]
    if (
        len(words) >= 3
        and first.startswith("python")
        and words[1] == "-m"
        and words[2] in {"qtts_cli", "qtts_cli.__main__"}
    ):
        return max(cword - 3, 0), words[3:]
    return max(cword, 0), words


def _subparsers(parser: argparse.ArgumentParser) -> Optional[argparse._SubParsersAction]:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return action
    return None


def _find_option(parser: argparse.ArgumentParser, token: str) -> Optional[argparse.Action]:
    for action in parser._actions:
        if token in action.option_strings:
            return action
    return None


def _value_count(action: argparse.Action) -> Any:
    if isinstance(action, _ZERO_ACTIONS):
        return 0
    nargs = action.nargs
    if nargs == 0:
        return 0
    if nargs == "?":
        return "?"
    if nargs is None:
        return 1
    if nargs in ("*", "+"):
        return "many"
    if isinstance(nargs, int):
        return nargs
    return 1


def _context(
    parser: argparse.ArgumentParser, prior: list[str]
) -> tuple[argparse.ArgumentParser, Optional[argparse.Action], int, set[str], bool]:
    current = parser
    used: set[str] = set()
    positionals = 0
    expecting: Optional[argparse.Action] = None
    positional_only = False
    index = 0
    while index < len(prior):
        token = prior[index]
        if positional_only:
            positionals += 1
            index += 1
            continue
        if token == "--":
            positional_only = True
            index += 1
            continue
        if token.startswith("--") and "=" in token:
            option, _value = token.split("=", 1)
            action = _find_option(current, option)
            if action is not None:
                used.add(option)
                index += 1
                continue
        if token.startswith("-") and token != "-":
            action = _find_option(current, token)
            if action is not None:
                used.add(token)
                count = _value_count(action)
                if count == 0:
                    index += 1
                elif count == "?":
                    if index + 1 < len(prior) and not prior[index + 1].startswith("-"):
                        index += 2
                    else:
                        index += 1
                elif count == "many":
                    index += 1
                    while index < len(prior) and not prior[index].startswith("-"):
                        index += 1
                else:
                    if index + 1 < len(prior):
                        index += 2
                    else:
                        expecting = action
                        index += 1
                continue
        sub = _subparsers(current)
        if sub is not None and token in sub.choices:
            current = sub.choices[token]
            # Keep ``used``: global flags parsed before the subcommand should
            # also be hidden by the subparser's copy of the same options.
            positionals = 0
            expecting = None
            index += 1
            continue
        positionals += 1
        index += 1
    return current, expecting, positionals, used, positional_only


def _flag_value(words: list[str], flag: str) -> Optional[str]:
    for index, token in enumerate(words):
        if token == flag and index + 1 < len(words):
            return words[index + 1]
        if token.startswith(flag + "="):
            return token.split("=", 1)[1]
    return None


def _data_dir(words: list[str]) -> Path:
    from ..config import default_data_dir

    explicit = _flag_value(words, "--data-dir")
    if explicit:
        return Path(explicit).expanduser()
    env = os.environ.get("QTTS_DATA_DIR", "").strip()
    if env:
        return Path(env).expanduser()
    return default_data_dir()


def _model_dir(words: list[str]) -> Optional[Path]:
    from ..config import default_model_dir

    explicit = _flag_value(words, "--model-dir")
    if explicit:
        return Path(explicit).expanduser()
    return default_model_dir()


def _builtin_speakers(words: list[str]) -> list[str]:
    from ..catalog import scan

    model_dir = _model_dir(words)
    names: list[str] = []
    try:
        for model in scan(model_dir):
            if model.kind == "builtin":
                names.extend(model.speakers)
    except Exception:
        names = []
    if not names:
        names = list(_STATIC_SPEAKERS)
    seen: set[str] = set()
    out: list[str] = []
    for name in names:
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out


def _voice_names(words: list[str]) -> list[str]:
    from ..voice_store import VoiceStore

    try:
        store = VoiceStore(_data_dir(words))
        return sorted(
            {
                str(meta.get("name"))
                for meta in store.list()
                if isinstance(meta, dict) and meta.get("name")
            }
        )
    except Exception:
        return []


def _model_names(words: list[str], current: str) -> tuple[str, list[str]]:
    if "/" in current or current.startswith((".", "~")):
        return "file", []
    from ..catalog import scan

    try:
        names = [model.name for model in scan(_model_dir(words))]
    except Exception:
        names = []
    return "plain", [name for name in names if name.startswith(current)]


def _voice_candidates(words: list[str], current: str) -> list[str]:
    if ":" not in current:
        return [prefix for prefix in ("builtin:", "voice:", "design:") if prefix.startswith(current)]
    kind, value = current.split(":", 1)
    if kind == "builtin":
        return [
            "builtin:" + name
            for name in _builtin_speakers(words)
            if name.lower().startswith(value.lower())
        ]
    if kind == "voice":
        return ["voice:" + name for name in _voice_names(words) if name.startswith(value)]
    return []


def _kind_for_action(action: argparse.Action, prog: str = "") -> Optional[str]:
    dest = action.dest
    option_names = set(action.option_strings)
    if dest == "asr_model":
        # `finetune eval` accepts a checkpoint path here; `say`, `align` and
        # `check` use the same dest for a model repo id.
        return "file" if prog.endswith("finetune eval") else None
    if dest in _VOICE_DESTS or "--voice" in option_names or "--reference-voice" in option_names:
        return "voice"
    if dest == "shell":
        return "shell"
    if dest in _MODEL_DESTS:
        return "model"
    if dest in ("backend",) or "--backend" in option_names:
        return "backend"
    if dest == "language" or "--language" in option_names:
        return "language"
    if dest in _DIR_DESTS or dest.endswith("_dir"):
        return "dir"
    if dest in _FILE_DESTS or any(
        part in dest
        for part in (
            "file",
            "audio",
            "manifest",
            "script",
            "project",
            "jsonl",
            "lexicon",
            "report",
            "out",
            "rc",
            "video",
            "srt",
            "ref",
        )
    ):
        return "file"
    return None


def _value_candidates(
    action: argparse.Action, current: str, words: list[str], prog: str = ""
) -> tuple[str, list[str]]:
    if action.choices:
        return "plain", [
            str(choice) for choice in action.choices if str(choice).startswith(current)
        ]
    kind = _kind_for_action(action, prog)
    if kind == "voice":
        return "plain", _voice_candidates(words, current)
    if kind == "shell":
        return "plain", [name for name in _SHELL_VALUES if name.startswith(current)]
    if kind == "model":
        return _model_names(words, current)
    if kind == "backend":
        return "plain", [name for name in _BACKENDS if name.startswith(current)]
    if kind == "language":
        return "plain", [name for name in _LANGUAGES if name.startswith(current)]
    if kind in ("file", "dir"):
        return kind, []
    return "plain", []


def _option_candidates(
    parser: argparse.ArgumentParser, current: str, used: set[str]
) -> list[str]:
    groups: dict[int, argparse._MutuallyExclusiveGroup] = {}
    try:
        for group in parser._mutually_exclusive_groups:
            for action in group._group_actions:
                groups[id(action)] = group
    except AttributeError:
        groups = {}

    out: list[str] = []
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            continue
        if not action.option_strings:
            continue
        repeatable = isinstance(action, (argparse._AppendAction, argparse._CountAction))
        if used.intersection(action.option_strings) and not repeatable:
            continue
        group = groups.get(id(action))
        if group is not None:
            conflict = False
            for sibling in group._group_actions:
                if sibling is action:
                    continue
                if used.intersection(sibling.option_strings):
                    conflict = True
                    break
            if conflict:
                continue
        for option in action.option_strings:
            if option.startswith(current):
                out.append(option)
    return out


def _positional_actions(parser: argparse.ArgumentParser) -> list[argparse.Action]:
    return [
        action
        for action in parser._actions
        if not action.option_strings and not isinstance(action, argparse._SubParsersAction)
    ]


def _positional_kind(parser: argparse.ArgumentParser, index: int) -> Optional[str]:
    actions = _positional_actions(parser)
    if not actions:
        return None
    if index >= len(actions):
        last = actions[-1]
        if last.nargs not in ("*", "+"):
            return None
        action = last
    else:
        action = actions[index]
    if action.dest == "text":
        return None
    if action.dest == "name":
        subcommand = parser.prog.split()[-1]
        if subcommand in {"inspect", "preview", "rm", "export"}:
            return "voice"
    return _kind_for_action(action, parser.prog)


def _candidates(
    parser: argparse.ArgumentParser,
    prior: list[str],
    current: str,
    words: list[str],
    positional_only_flag: bool,
) -> tuple[str, list[str]]:
    current_parser, expecting, positionals, used, positional_only = _context(parser, prior)
    if expecting is not None:
        return _value_candidates(expecting, current, words, current_parser.prog)
    if not positional_only and current.startswith("-"):
        if current.startswith("--") and "=" in current:
            option, value = current.split("=", 1)
            action = _find_option(current_parser, option)
            if action is not None:
                kind, values = _value_candidates(
                    action, value, words, current_parser.prog
                )
                if kind == "plain":
                    return "plain", [f"{option}={item}" for item in values]
                # A file/dir hint cannot be applied to ``--opt=value``: the
                # shell would complete the whole token as a path.  Leave the
                # combined form alone rather than emitting a bogus hint.
                return "plain", []
            return "plain", []
        return "plain", _option_candidates(current_parser, current, used)

    sub = _subparsers(current_parser)
    if sub is not None and not positional_only:
        matches = [
            name
            for name in sub.choices
            if name != "__complete" and name.startswith(current)
        ]
        if matches:
            return "plain", matches

    kind = _positional_kind(current_parser, positionals)
    if kind == "voice":
        # A positional voice name is always a user-voice id (e.g. `voice rm
        # alice`); the `builtin:`/`voice:`/`design:` prefixes only apply to
        # the `--voice` option.
        return "plain", [
            name for name in _voice_names(words) if name.startswith(current)
        ]
    if kind == "shell":
        return "plain", [name for name in _SHELL_VALUES if name.startswith(current)]
    if kind == "model":
        return _model_names(words, current)
    if kind in ("file", "dir"):
        return kind, []
    if not current:
        if positional_only:
            return "plain", []
        return "plain", _option_candidates(current_parser, "", used)
    return "plain", []


def _clean(candidates: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        text = str(candidate)
        if not text or len(text) > _MAX_CANDIDATE_LEN:
            continue
        if any(
            unicodedata.category(char).startswith("C")
            or unicodedata.category(char) in ("Zl", "Zp")
            for char in text
        ):
            continue
        if text in seen:
            continue
        seen.add(text)
        out.append(text)
        if len(out) >= _MAX_CANDIDATES:
            break
    return out


def _parse_internal(
    argv: list[str],
) -> tuple[Optional[str], Optional[str], Optional[int], Optional[int], list[str]]:
    shell: Optional[str] = None
    line: Optional[str] = None
    point: Optional[int] = None
    cword: Optional[int] = None
    rest: list[str] = []
    index = 0
    while index < len(argv):
        token = argv[index]
        if token == "--":
            rest = argv[index + 1 :]
            break
        if token in {"--shell", "--line", "--point", "--cword"}:
            if index + 1 >= len(argv):
                break
            value = argv[index + 1]
            if token == "--shell":
                shell = value
            elif token == "--line":
                line = value
            elif token == "--point":
                try:
                    point = int(value)
                except ValueError:
                    point = None
            else:
                try:
                    cword = int(value)
                except ValueError:
                    cword = None
            index += 2
            continue
        index += 1
    return shell, line, point, cword, rest


def complete_main(argv: list[str]) -> int:
    """Entry point for ``qtts __complete``. Always exits 0."""
    kind = "plain"
    candidates: list[str] = []
    try:
        _shell, line, point, cword, rest = _parse_internal(argv)
        if line is not None:
            if point is not None:
                point = max(0, min(point, len(line)))
                prefix = line[:point]
            else:
                prefix = line
            words = tokenize(prefix)
            cword = len(words) - 1 if words else 0
        else:
            words = list(rest)
            if cword is None:
                cword = len(words) - 1 if words else 0
        cword, words = _strip_program(words, cword)
        if cword < 0:
            cword = 0
        current = words[cword] if cword < len(words) else ""
        prior = words[:cword]
        kind, candidates = _candidates(_parser(), prior, current, words, False)
    except Exception:
        if os.environ.get("QTTS_COMPLETION_DEBUG") == "1":
            import traceback

            traceback.print_exc()
        kind = "plain"
        candidates = []
    if kind not in {"plain", "file", "dir"}:
        kind = "plain"
    try:
        sys.stdout.write(f"#qtts-complete:v1 kind={kind}\n")
        for candidate in _clean(candidates):
            sys.stdout.write(candidate + "\n")
        sys.stdout.flush()
    except BrokenPipeError:
        return 0
    return 0
