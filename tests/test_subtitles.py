"""Subtitle cue merging and cross-format time consistency (FR-P0-05)."""

from __future__ import annotations

import json

import pytest

from qtts_cli.errors import QttsError
from qtts_cli.subtitles import (
    AlignItem,
    build_cues,
    full_text,
    monotonic,
    normalize_formats,
    render,
    shift_cues,
)


def test_char_cues_preserve_source_text():
    text = "你好，世界。"
    items = [
        AlignItem("你", 0.0, 0.2),
        AlignItem("好", 0.2, 0.4),
        AlignItem("世", 0.5, 0.7),
        AlignItem("界", 0.7, 1.0),
    ]
    cues = build_cues(text, items, granularity="char")
    assert full_text(cues) == text
    assert [c.text for c in cues] == ["你", "好，", "世", "界。"]


def test_word_cues_keep_latin_words_whole():
    text = "Hello world. Again!"
    items = [
        AlignItem("Hello", 0.0, 0.3),
        AlignItem("world", 0.3, 0.6),
        AlignItem("Again", 0.8, 1.1),
    ]
    cues = build_cues(text, items, granularity="char")
    assert [c.text for c in cues] == ["Hello ", "world. ", "Again!"]
    assert full_text(cues) == text


def test_multi_char_cjk_items_are_split_for_char_granularity():
    text = "你好世界"
    items = [AlignItem("你好", 0.0, 0.4), AlignItem("世界", 0.4, 0.8)]
    cues = build_cues(text, items, granularity="char")
    assert [c.text for c in cues] == ["你", "好", "世", "界"]
    assert full_text(cues) == text


def test_cues_are_monotonic():
    text = "abcdef"
    items = [AlignItem("a", 1.0, 1.5), AlignItem("b", 0.1, 0.2), AlignItem("c", 2.0, 2.5)]
    cues = monotonic(build_cues(text, items))
    for prev, cur in zip(cues, cues[1:]):
        assert cur.start >= prev.end - 1e-9


def test_shift_offsets_every_cue():
    cues = build_cues("你好。", [AlignItem("你", 0.0, 0.2), AlignItem("好", 0.2, 0.4)])
    shifted = shift_cues(cues, 2.5)
    assert shifted[0].start == pytest.approx(2.5)
    assert full_text(shifted) == "你好。"


def test_fallback_without_alignment_items():
    cues = build_cues("没有时间轴", [], granularity="char", fallback_duration=1.5)
    assert len(cues) == 1
    assert cues[0].text == "没有时间轴"
    assert cues[0].end == pytest.approx(1.5)


def test_renderers_agree_on_times():
    cues = build_cues("你好。", [AlignItem("你", 0.08, 0.24), AlignItem("好", 0.24, 1.002)])
    srt = render(cues, "srt")
    vtt = render(cues, "vtt")
    ass = render(cues, "ass")
    data = json.loads(render(cues, "json"))
    assert "00:00:00,080 --> 00:00:00,240" in srt
    assert "00:00:00.080 --> 00:00:00.240" in vtt
    assert "0:00:00.08,0:00:00.24" in ass
    assert data["cues"][0] == {"start": 0.08, "end": 0.24, "text": "你"}
    assert "".join(c["text"] for c in data["cues"]) == "你好。"


def test_json_render_has_schema_fields():
    data = json.loads(render(build_cues("x", []), "json", language="English"))
    assert data["schema_version"] == 1
    assert data["language"] == "English"
    assert data["granularity"] == "char"


def test_normalize_formats():
    assert normalize_formats("srt, vtt") == ["srt", "vtt"]
    assert normalize_formats(None) == []
    with pytest.raises(QttsError):
        normalize_formats("mp4")
