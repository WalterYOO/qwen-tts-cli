"""Normalization, segmentation and budget estimation (FR-P0-04)."""

from __future__ import annotations

from pathlib import Path

import pytest

from qtts_cli.budget import codec_rate, estimate_tokens, fits, max_chars
from qtts_cli.catalog import scan, select
from qtts_cli.segmenter import segment_text, split_long
from qtts_cli.textnorm import detect_language, int_to_chinese, int_to_english, normalize_text


@pytest.mark.parametrize(
    "text",
    [
        "你好，世界。这是第二句。",
        "他说：「不要走。」然后就走了。",
        "Mr. Smith said 3.14 is pi. Really!",
        "第一段\n\n第二段！还有第三段？",
        "价格是¥123.45，日期2026-11-02。",
    ],
)
def test_segmentation_preserves_text(text):
    segments = segment_text(text)
    assert "".join(segments) == text
    assert all(seg for seg in segments)


def test_segmentation_does_not_split_quotes_or_decimals():
    text = "他说：「不要走。」然后就走了。"
    assert segment_text(text) == [text]
    assert segment_text("Pi is 3.14 today.") == ["Pi is 3.14 today."]
    assert segment_text("Dr. Smith arrived.") == ["Dr. Smith arrived."]


def test_segmentation_splits_english_sentences():
    assert segment_text("Hello world. This is fine!") == ["Hello world. ", "This is fine!"]


def test_split_long_respects_budget_and_preserves_text():
    text = "这是一个很长的句子，" * 40
    pieces = split_long(text, 50)
    assert all(len(piece) <= 50 for piece in pieces)
    assert "".join(pieces) == text


def test_split_long_never_cuts_a_number():
    text = "编号123456789012345678901234567890结束"
    pieces = split_long(text, 12)
    assert "".join(pieces) == text
    assert all(len(p) <= 12 for p in pieces)


def test_normalization_keeps_cjk_punctuation():
    assert normalize_text("你好，世界。", "Chinese") == "你好，世界。"


def test_normalization_fullwidth_digits_and_emoji():
    normalized = normalize_text("我有１２３个苹果😀。", "Chinese")
    assert "１２３" not in normalized
    assert "😀" not in normalized
    assert "一百二十三" in normalized


def test_normalization_dates_money_percent_units():
    assert normalize_text("2026-11-02", "Chinese") == "二零二六年十一月二日"
    assert "元" in normalize_text("¥123.45", "Chinese")
    assert "百分之五十" in normalize_text("50%", "Chinese")
    assert "五千克" in normalize_text("5kg", "Chinese")


def test_normalization_english_numbers():
    assert normalize_text("I have 42 apples.", "English") == "I have forty-two apples."
    assert "three point one four" in normalize_text("3.14", "English")


def test_normalization_drops_urls():
    assert "http" not in normalize_text("see https://example.com/x now", "English")


def test_int_converters():
    assert int_to_chinese(0) == "零"
    assert int_to_chinese(10) == "十"
    assert int_to_chinese(105) == "一百零五"
    assert int_to_chinese(1001) == "一千零一"
    assert int_to_chinese(2026) == "二千零二十六"
    assert int_to_english(0) == "zero"
    assert int_to_english(42) == "forty-two"
    assert int_to_english(1001) == "one thousand one"


def test_detect_language_per_segment():
    assert detect_language("你好") == "Chinese"
    assert detect_language("hello there") == "English"
    assert detect_language("12345") == "Auto"


def test_codec_rate_from_tokenizer_config(model_dir: Path):
    model = select(scan(model_dir), "builtin")
    assert codec_rate(model) == pytest.approx(12.5)
    assert estimate_tokens("你好，世界。", "Chinese") > 0
    assert fits("你好。", "Chinese", 2048)
    assert not fits("字" * 5000, "Chinese", 2048)
    assert max_chars("Chinese", 2048) >= 300
