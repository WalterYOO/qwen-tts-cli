from __future__ import annotations

import pytest

from qtts_cli.mapped_text import mapped_normalize, mapped_normalize_text
from qtts_cli.qc import normalize_for_qc
from qtts_cli.textnorm import normalize_text


SAMPLES = [
    "On 2026-11-02, visit https://x.test 😀 at 10:30, cost $5 and 3kg.",
    "日期2026-11-02，时间10:30，价格$5，重量3kg。",
    "Hello   world!\r\nLine2… 100%",
    "你好，世界！",
    "2026年",
    "ＡＢＣ　１２３",
    "www.example.com test@example.com",
    "",
]


@pytest.mark.parametrize("source", SAMPLES)
@pytest.mark.parametrize("language", ["auto", "English", "Chinese"])
def test_mapped_normalization_matches_qc_pipeline(source: str, language: str) -> None:
    mapped = mapped_normalize(source, language)
    assert mapped.text == normalize_for_qc(source, language)
    assert mapped.source == source
    assert len(mapped.chars) == len(mapped.text)
    for char in mapped.chars:
        assert 0 <= char.start < char.end <= len(source)


def test_number_expansion_keeps_whole_token_source_span() -> None:
    source = "2026年"
    mapped = mapped_normalize(source, "Chinese")
    assert mapped.text == normalize_for_qc(source, "Chinese")
    number = mapped.text.index("二零二六")
    for unit in mapped.chars[number:number + 4]:
        assert (unit.start, unit.end) == (0, 4)
    assert (mapped.chars[-1].start, mapped.chars[-1].end) == (4, 5)
    assert mapped.source_span(number, number + 4) == (0, 4)


def test_date_url_emoji_and_punctuation_provenance() -> None:
    source = "Visit https://x.test 😀 on 2026-11-02."
    mapped = mapped_normalize(source, "English")
    assert mapped.text == normalize_for_qc(source, "English")
    assert "http" not in mapped.text
    assert "😀" not in mapped.text

    visit = mapped.text.index("visit")
    assert mapped.source_span(visit, visit + 5) == (0, 5)

    date_start = source.index("2026")
    date_chars = [
        unit
        for unit in mapped.chars
        if date_start <= unit.start < date_start + 10
    ]
    assert date_chars
    assert all(
        date_start <= unit.start and unit.end <= date_start + 10
        for unit in date_chars
    )


def test_punctuation_dropped_but_boundaries_map_to_source() -> None:
    source = "你好，世界！"
    mapped = mapped_normalize(source, "Chinese")
    assert mapped.text == "你好世界"
    assert mapped.source_span(0, 2) == (0, 2)
    assert mapped.boundary(2) == 3
    assert mapped.source_span(2, 4) == (3, 5)
    assert mapped.end_boundary(4) == 5


def test_boundary_helpers_clamp_and_validate() -> None:
    mapped = mapped_normalize("abc", "English")
    assert mapped.boundary(0) == 0
    assert mapped.boundary(len(mapped.chars)) == 3
    assert mapped.end_boundary(0) == 0
    assert mapped.source_span(1, 1) == (1, 1)
    with pytest.raises(ValueError):
        mapped.source_span(2, 1)
    with pytest.raises(ValueError):
        mapped.source_span(0, len(mapped.chars) + 1)


def test_empty_text_has_no_units() -> None:
    mapped = mapped_normalize("", "Chinese")
    assert mapped.text == ""
    assert mapped.chars == ()
    assert mapped.language == "Chinese"


@pytest.mark.parametrize("source", SAMPLES)
@pytest.mark.parametrize("language", ["auto", "English", "Chinese"])
def test_display_normalization_matches_textnorm(source: str, language: str) -> None:
    mapped = mapped_normalize_text(source, language)
    assert mapped.text == normalize_text(source, language)
    assert len(mapped.chars) == len(mapped.text)
    for char in mapped.chars:
        assert 0 <= char.start < char.end <= len(source)
