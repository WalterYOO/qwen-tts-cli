"""The hidden `qtts __complete` engine."""

from __future__ import annotations

import json
from pathlib import Path

from qtts_cli.completion import engine


def _run(capsys, *args: str):
    rc = engine.complete_main(list(args))
    captured = capsys.readouterr()
    lines = captured.out.splitlines()
    assert rc == 0
    assert captured.err == ""
    assert lines[0] == "#qtts-complete:v1 kind=plain" or lines[0].startswith(
        "#qtts-complete:v1 kind="
    ), lines
    return lines[0], lines[1:]


def _line(capsys, line: str):
    return _run(capsys, "--shell", "bash", "--line", line, "--point", str(len(line)))


def test_tokenize_quotes_and_trailing_space():
    assert engine.tokenize("qtts say hello") == ["qtts", "say", "hello"]
    assert engine.tokenize("qtts say ") == ["qtts", "say", ""]
    assert engine.tokenize('qtts say "hello wor') == ["qtts", "say", "hello wor"]
    assert engine.tokenize("qtts say 'a b' ") == ["qtts", "say", "a b", ""]
    assert engine.tokenize(r"qtts say a\ b") == ["qtts", "say", "a b"]
    assert engine.tokenize(r"qtts say a\ ") == ["qtts", "say", "a "]
    assert engine.tokenize("qtts say '' ") == ["qtts", "say", "", ""]


def test_top_level_commands(capsys):
    header, candidates = _line(capsys, "qtts ")
    assert header.endswith("kind=plain")
    assert {"say", "voice", "align", "completion", "finetune"} <= set(candidates)
    assert "__complete" not in candidates


def test_prefix_matching(capsys):
    _, candidates = _line(capsys, "qtts sa")
    assert candidates == ["say"]


def test_nested_commands(capsys):
    _, candidates = _line(capsys, "qtts voice ")
    assert {"add", "list", "inspect", "preview", "rm", "export", "import"} <= set(
        candidates
    )


def test_choices(capsys):
    _, candidates = _line(capsys, "qtts say --device c")
    assert candidates == ["cpu", "cuda", "cuda:0", "cuda:1"]


def test_option_equals_returns_full_token(capsys):
    _, candidates = _line(capsys, "qtts say --device=c")
    assert candidates == ["--device=cpu", "--device=cuda", "--device=cuda:0", "--device=cuda:1"]


def test_voice_prefixes(capsys):
    _, candidates = _line(capsys, "qtts say --voice ")
    assert candidates == ["builtin:", "voice:", "design:"]
    _, candidates = _line(capsys, "qtts say --voice builtin:O")
    assert candidates == ["builtin:Ono_Anna"]


def test_voice_builtins_use_model_readme(capsys, model_dir: Path):
    line = f"qtts --model-dir {model_dir} say --voice builtin:"
    _, candidates = _line(capsys, line)
    assert "builtin:Vivian" in candidates
    assert "builtin:Ono_Anna" in candidates


def test_user_voice_names(capsys, tmp_path: Path):
    data = tmp_path / "data"
    voice_dir = data / "voices" / "alice"
    voice_dir.mkdir(parents=True)
    (voice_dir / "voice.json").write_text(json.dumps({"name": "alice"}), encoding="utf-8")
    line = f"qtts --data-dir {data} say --voice voice:"
    _, candidates = _line(capsys, line)
    assert candidates == ["voice:alice"]


def test_completion_shell_values(capsys):
    _, candidates = _line(capsys, "qtts completion ")
    assert {"auto", "bash", "zsh", "fish"} <= set(candidates)
    _, candidates = _line(capsys, "qtts completion z")
    assert candidates == ["zsh"]


def test_file_and_directory_kinds(capsys):
    for line in (
        "qtts align ",
        "qtts post ",
        "qtts dub build ",
        "qtts audiobook build --bgm ",
        "qtts finetune eval --checkpoint ",
        "qtts finetune eval --asr-model ",
        "qtts finetune eval --base-model ",
        "qtts finetune prepare --tokenizer-model-path ",
        "qtts finetune train --init-model-path ",
        "qtts finetune train --checkpoint ",
        "qtts completion --rc-file ",
    ):
        assert _line(capsys, line)[0].endswith("kind=file"), line
    assert _line(capsys, "qtts project init ")[0].endswith("kind=dir")


def test_shared_asr_model_dest_is_only_a_file_in_finetune_eval(capsys):
    for line in ("qtts say --asr-model ", "qtts align --asr-model ", "qtts check --asr-model "):
        assert _line(capsys, line)[0].endswith("kind=plain"), line


def test_option_equals_file_value_gets_no_bogus_file_hint(capsys):
    header, candidates = _line(capsys, "qtts say --out=foo")
    assert header.endswith("kind=plain")
    assert candidates == []


def test_model_names(capsys, model_dir: Path):
    line = f"qtts --model-dir {model_dir} models pull "
    _, candidates = _line(capsys, line)
    assert "Qwen3-TTS-12Hz-1.7B-CustomVoice" in candidates


def test_voice_name_positional(capsys, tmp_path: Path):
    data = tmp_path / "data"
    voice_dir = data / "voices" / "alice"
    voice_dir.mkdir(parents=True)
    (voice_dir / "voice.json").write_text(json.dumps({"name": "alice"}), encoding="utf-8")
    line = f"qtts --data-dir {data} voice rm "
    _, candidates = _line(capsys, line)
    assert candidates == ["alice"]


def test_option_terminator_stops_option_completion(capsys):
    _, candidates = _line(capsys, "qtts say -- --device")
    assert candidates == []
    _, candidates = _line(capsys, "qtts say -- ")
    assert candidates == []


def test_used_flags_are_hidden(capsys):
    _, candidates = _line(capsys, "qtts say --json --")
    assert "--json" not in candidates
    assert "--quiet" in candidates


def test_mutually_exclusive_options_are_hidden(capsys):
    _, candidates = _line(capsys, "qtts post --out x --")
    assert "--out-dir" not in candidates
    assert "--lufs" in candidates


def test_zsh_words_mode(capsys):
    header, candidates = _run(capsys, "--shell", "zsh", "--cword", "2", "--", "qtts", "voice", "")
    assert header.endswith("kind=plain")
    assert "add" in candidates


def test_fish_words_mode(capsys):
    header, candidates = _run(capsys, "--shell", "fish", "--cword", "1", "--", "qtts", "")
    assert header.endswith("kind=plain")
    assert "say" in candidates


def test_program_prefix_python_m(capsys):
    header, candidates = _run(
        capsys, "--shell", "zsh", "--cword", "3", "--", "python", "-m", "qtts_cli", ""
    )
    assert header.endswith("kind=plain")
    assert "say" in candidates


def test_clean_filters_control_characters_and_duplicates():
    values = ["a", "a", "b", "bad\nvalue", "bad\x1bvalue", ""]
    values += ["c1\x85", "c1\x9b", "line\u2028sep", "para\u2029sep"]
    assert engine._clean(values) == ["a", "b"]


def test_clean_caps_candidates():
    values = [f"value{i}" for i in range(1200)]
    assert len(engine._clean(values)) == 1000


def test_internal_failure_is_silent(monkeypatch, capsys):
    def boom():
        raise RuntimeError("boom")

    monkeypatch.setattr(engine, "_parser", boom)
    assert engine.complete_main(["--shell", "bash", "--line", "qtts ", "--point", "5"]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out.startswith("#qtts-complete:v1 kind=plain")


def test_debug_failure_writes_stderr(monkeypatch, capsys):
    def boom():
        raise RuntimeError("boom")

    monkeypatch.setattr(engine, "_parser", boom)
    monkeypatch.setenv("QTTS_COMPLETION_DEBUG", "1")
    assert engine.complete_main(["--shell", "bash", "--line", "qtts "]) == 0
    captured = capsys.readouterr()
    assert "RuntimeError: boom" in captured.err
