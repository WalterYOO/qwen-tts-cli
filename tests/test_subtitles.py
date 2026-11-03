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
    cues = monotonic(build_cues(text, items, granularity="char"))
    for prev, cur in zip(cues, cues[1:]):
        assert cur.start >= prev.end - 1e-9


def test_shift_offsets_every_cue():
    cues = build_cues(
        "你好。",
        [AlignItem("你", 0.0, 0.2), AlignItem("好", 0.2, 0.4)],
        granularity="char",
    )
    shifted = shift_cues(cues, 2.5)
    assert shifted[0].start == pytest.approx(2.5)
    assert full_text(shifted) == "你好。"


def test_fallback_without_alignment_items():
    cues = build_cues("没有时间轴", [], granularity="char", fallback_duration=1.5)
    assert len(cues) == 1
    assert cues[0].text == "没有时间轴"
    assert cues[0].end == pytest.approx(1.5)


def test_renderers_agree_on_times():
    cues = build_cues(
        "你好。",
        [AlignItem("你", 0.08, 0.24), AlignItem("好", 0.24, 1.002)],
        granularity="char",
    )
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
    assert data["granularity"] == "sentence"


def test_sentence_cues_match_chinese_sentences():
    text = "你好，世界。今天天气很好！我们出去走走吧？"
    items = [
        AlignItem("你", 0.0, 0.2),
        AlignItem("好", 0.2, 0.4),
        AlignItem("世", 0.4, 0.6),
        AlignItem("界", 0.6, 0.8),
        AlignItem("今", 0.9, 1.1),
        AlignItem("天", 1.1, 1.3),
        AlignItem("天", 1.3, 1.5),
        AlignItem("气", 1.5, 1.7),
        AlignItem("很", 1.7, 1.9),
        AlignItem("好", 1.9, 2.1),
        AlignItem("我", 2.2, 2.4),
        AlignItem("们", 2.4, 2.6),
        AlignItem("出", 2.6, 2.8),
        AlignItem("去", 2.8, 3.0),
        AlignItem("走", 3.0, 3.2),
        AlignItem("走", 3.2, 3.4),
        AlignItem("吧", 3.4, 3.6),
    ]
    cues = build_cues(text, items, granularity="sentence")
    assert [c.text for c in cues] == ["你好，世界。", "今天天气很好！", "我们出去走走吧？"]
    assert full_text(cues) == text
    assert cues[0].start == pytest.approx(0.0)
    assert cues[0].end == pytest.approx(0.8)
    assert cues[2].end == pytest.approx(3.6)


def test_sentence_cues_match_english_sentences():
    text = "Hello world. Again! Are you there?"
    items = [
        AlignItem("Hello", 0.0, 0.3),
        AlignItem("world", 0.3, 0.6),
        AlignItem("Again", 0.8, 1.1),
        AlignItem("Are", 1.3, 1.5),
        AlignItem("you", 1.5, 1.7),
        AlignItem("there", 1.7, 1.9),
    ]
    cues = build_cues(text, items, granularity="sentence")
    assert [c.text for c in cues] == ["Hello world. ", "Again! ", "Are you there?"]
    assert full_text(cues) == text


def test_sentence_granularity_rejects_unknown_values():
    with pytest.raises(QttsError):
        build_cues("你好。", [], granularity="line")


def test_normalize_formats():
    assert normalize_formats("srt, vtt") == ["srt", "vtt"]
    assert normalize_formats(None) == []
    with pytest.raises(QttsError):
        normalize_formats("mp4")
