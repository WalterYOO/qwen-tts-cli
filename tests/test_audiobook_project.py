"""FR-P2-03 §12.1: strict project parsing, limits and BGM budgets."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from qtts_cli.audiobook import audio
from qtts_cli.audiobook.project import load_project
from qtts_cli.errors import InputError


def _project(tmp_path: Path, body: str, name: str = "book.yaml") -> Path:
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return path


BASE = """
schema_version: 1
voice: builtin:Vivian
chapters:
  - title: One
    text: Hello.
"""


def test_yaml_and_json_parse(tmp_path: Path):
    project = load_project(_project(tmp_path, BASE))
    assert project.title == "book"
    assert project.chapters[0].paragraphs == ["Hello."]
    json_path = _project(
        tmp_path,
        json.dumps(
            {
                "schema_version": 1,
                "voice": "builtin:Vivian",
                "chapters": [{"title": "One", "text": "Hello.\n\nWorld."}],
            }
        ),
        "book.json",
    )
    parsed = load_project(json_path)
    assert parsed.chapters[0].paragraphs == ["Hello.", "World."]


@pytest.mark.parametrize(
    "body",
    [
        "schema_version: 1\nvoice: builtin:Vivian\nunknown: 1\nchapters:\n  - title: A\n    text: x\n",
        "schema_version: 1\nvoice: builtin:Vivian\nchapters:\n  - title: A\n    text: x\n    extra: y\n",
        "schema_version: 1\nvoice: builtin:Vivian\nchapters:\n  - title: A\n    text: x\n    text: y\n",
        "schema_version: 1\nvoice: builtin:Vivian\nchapters:\n  - &a title: A\n    text: *a\n",
        "schema_version: 1\nvoice: builtin:Vivian\nchapters:\n  - title: A\n    text: x\n---\nschema_version: 1\n",
        "schema_version: true\nvoice: builtin:Vivian\nchapters:\n  - title: A\n    text: x\n",
        "schema_version: '1'\nvoice: builtin:Vivian\nchapters:\n  - title: A\n    text: x\n",
    ],
)
def test_invalid_projects_exit_4(tmp_path: Path, body: str):
    with pytest.raises(InputError):
        load_project(_project(tmp_path, body))


def test_json_duplicate_and_nonfinite_and_depth(tmp_path: Path):
    with pytest.raises(InputError):
        load_project(
            _project(
                tmp_path,
                '{"schema_version": 1, "schema_version": 1, "chapters": []}',
                "a.json",
            )
        )
    with pytest.raises(InputError):
        load_project(
            _project(
                tmp_path,
                '{"schema_version": NaN, "chapters": []}',
                "b.json",
            )
        )
    nested = "0"
    for _ in range(101):
        nested = '{"x": ' + nested + "}"
    with pytest.raises(InputError):
        load_project(_project(tmp_path, '{"schema_version": 1, "chapters": ' + nested + "}", "c.json"))
    many = json.dumps(
        {"schema_version": 1, "chapters": [{"title": "A", "text": "x"}], "junk": list(range(100001))}
    )
    with pytest.raises(InputError):
        load_project(_project(tmp_path, many, "d.json"))


def test_yaml_deep_recursion_and_event_cap(tmp_path: Path):
    deep = "schema_version: 1\nchapters: " + "[" * 500 + "]" * 500 + "\n"
    with pytest.raises(InputError):
        load_project(_project(tmp_path, deep))
    many = "schema_version: 1\njunk: [" + ",".join(["0"] * 50001) + "]\n"
    with pytest.raises(InputError):
        load_project(_project(tmp_path, many))


def test_chapter_and_paragraph_limits(tmp_path: Path):
    too_many = ["schema_version: 1", "voice: builtin:Vivian", "chapters:"]
    for index in range(501):
        too_many.append(f"  - title: C{index}")
        too_many.append("    text: x")
    with pytest.raises(InputError):
        load_project(_project(tmp_path, "\n".join(too_many)))

    huge = "x" * 200001
    with pytest.raises(InputError):
        load_project(
            _project(
                tmp_path,
                f"schema_version: 1\nvoice: builtin:Vivian\nchapters:\n  - title: A\n    text: {huge}\n",
            )
        )


def test_sources_absolute_relative_and_invalid(tmp_path: Path):
    (tmp_path / "text").mkdir()
    (tmp_path / "text" / "01.txt").write_text("From source.\n\nSecond.", encoding="utf-8")
    relative = _project(
        tmp_path,
        "schema_version: 1\nvoice: builtin:Vivian\nchapters:\n"
        "  - title: A\n    source: text/01.txt\n",
    )
    project = load_project(relative)
    assert project.chapters[0].paragraphs == ["From source.", "Second."]
    assert project.chapters[0].source_path == (tmp_path / "text" / "01.txt").resolve()

    absolute = _project(
        tmp_path,
        "schema_version: 1\nvoice: builtin:Vivian\nchapters:\n"
        f"  - title: A\n    source: {json.dumps(str((tmp_path / 'text' / '01.txt').resolve()))}\n",
        "absolute.yaml",
    )
    assert load_project(absolute).chapters[0].source_path is not None

    (tmp_path / "nul.txt").write_bytes(b"ok\x00bad")
    with pytest.raises(InputError):
        load_project(
            _project(
                tmp_path,
                "schema_version: 1\nvoice: builtin:Vivian\nchapters:\n  - title: A\n    source: nul.txt\n",
                "nul.yaml",
            )
        )
    (tmp_path / "bad.txt").write_bytes(b"\xff\xfe")
    with pytest.raises(InputError):
        load_project(
            _project(
                tmp_path,
                "schema_version: 1\nvoice: builtin:Vivian\nchapters:\n  - title: A\n    source: bad.txt\n",
                "bad.yaml",
            )
        )


def test_paragraph_split_and_empty_body(tmp_path: Path):
    project = load_project(
        _project(
            tmp_path,
            "schema_version: 1\nvoice: builtin:Vivian\nchapters:\n  - title: A\n    text: |\n"
            "      # Heading\n\n      Body one.\n\n\n      Body two.\n",
        )
    )
    assert project.chapters[0].paragraphs == ["# Heading", "Body one.", "Body two."]
    with pytest.raises(InputError):
        load_project(
            _project(
                tmp_path,
                "schema_version: 1\nvoice: builtin:Vivian\nchapters:\n  - title: A\n    text: '   '\n",
                "empty.yaml",
            )
        )


def test_language_and_instruct_precedence(tmp_path: Path):
    project = load_project(
        _project(
            tmp_path,
            "schema_version: 1\nlanguage: auto\ninstruct: calm\nvoice: builtin:Vivian\nchapters:\n"
            "  - title: A\n    text: x\n"
            "  - title: B\n    language: English\n    instruct: soft\n    text: y\n",
        )
    )
    assert project.chapters[0].language == "auto"
    assert project.chapters[0].instruct == "calm"
    assert project.chapters[1].language == "English"
    assert project.chapters[1].instruct == "soft"


def test_number_bool_rejection(tmp_path: Path):
    with pytest.raises(InputError):
        load_project(
            _project(
                tmp_path,
                "schema_version: 1\nvoice: builtin:Vivian\nsample_rate: true\n"
                "chapters:\n  - title: A\n    text: x\n",
            )
        )
    with pytest.raises(InputError):
        load_project(
            _project(
                tmp_path,
                "schema_version: 1\nvoice: builtin:Vivian\npause: {paragraph_ms: true}\n"
                "chapters:\n  - title: A\n    text: x\n",
            )
        )


def _bgm(path: Path, *, sr: int, seconds: float, channels: int = 1) -> Path:
    frames = int(round(sr * seconds))
    data = np.zeros((frames, channels), dtype=np.float32)
    sf.write(str(path), data, sr, subtype="PCM_16")
    return path


def test_bgm_memory_budgets(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(audio, "BGM_MAX_BUFFER_BYTES", 1000)

    ok = _bgm(tmp_path / "ok.wav", sr=8000, seconds=0.01)
    audio.probe_bgm(ok, 8000)

    # Passes at 8 kHz but exceeds the budget after resampling to 48 kHz.
    resample_blowup = _bgm(tmp_path / "resample.wav", sr=8000, seconds=0.02)
    with pytest.raises(InputError):
        audio.probe_bgm(resample_blowup, 48000)

    # Decoding the source buffer itself exceeds the budget (48 kHz stereo).
    source_blowup = _bgm(tmp_path / "stereo.wav", sr=48000, seconds=0.01, channels=2)
    with pytest.raises(InputError):
        audio.probe_bgm(source_blowup, 8000)


def test_bgm_duration_limit(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(audio, "BGM_MAX_SECONDS", 0.01)
    path = _bgm(tmp_path / "long.wav", sr=8000, seconds=0.02)
    with pytest.raises(InputError):
        audio.probe_bgm(path, 8000)
