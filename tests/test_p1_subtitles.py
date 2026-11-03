"""FR-P1-05a/05b: cue segmentation, balancing and quality gate."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from qtts_cli.audio_io import write_audio
from qtts_cli.cli import main
from qtts_cli.commands import align as align_cmd
from qtts_cli.commands import say as say_cmd
from qtts_cli.errors import InputError, QualityError
from qtts_cli.subtitle_qc import check_subtitles, raise_quality_error
from qtts_cli.subtitles import (
    AlignItem,
    Cue,
    SourceSpan,
    SubtitlePolicy,
    balance_spans,
    build_cues,
    build_cues_for_ranges,
    display_width,
    ensure_full_text,
    full_text,
    monotonic,
    plan_sentence_spans,
)


def _items(text: str, cps: float = 8.0) -> tuple[list[AlignItem], float]:
    items: list[AlignItem] = []
    now = 0.0
    step = 1.0 / cps
    for ch in text:
        if ch.isspace() or ch in "，。！？、；：,.!?;:":
            continue
        items.append(AlignItem(ch, now, now + step))
        now += step
    return items, now


def test_display_width_counts_wide_and_combining_characters():
    assert display_width("你好") == 4
    assert display_width("abc") == 3
    assert display_width("é") == 1
    assert display_width("你 a", ignore_whitespace=True) == 3


def test_long_sentence_splits_at_secondary_punctuation():
    text = "第一段文字很长啊，第二段文字也很长啊，第三段文字结束了。"
    items, duration = _items(text)
    stats: dict[str, int] = {}
    cues = build_cues(
        text,
        items,
        granularity="sentence",
        audio_duration=duration,
        policy=SubtitlePolicy(max_cue_chars=20, min_cue_seconds=0.0, max_cue_seconds=99),
        stats=stats,
    )
    assert full_text(cues) == text
    assert len(cues) > 1
    assert all(display_width(cue.text) <= 20 for cue in cues)
    assert stats["splits"] > 0


def test_no_hard_cut_inside_quoted_text():
    text = "开头“" + "甲乙" * 12 + "”结尾。"
    policy = SubtitlePolicy(max_cue_chars=10, min_cue_seconds=0.0)
    warnings: list[str] = []
    spans = plan_sentence_spans(text, policy, warnings=warnings)
    assert "".join(text[s.start : s.end] for s in spans) == text
    quote_start = text.index("“") + 1
    quote_end = text.index("”")
    for span in spans[:-1]:
        assert not (quote_start < span.end < quote_end)
    assert any("no safe split" in warning for warning in warnings)


def test_hard_cut_never_breaks_numbers_or_latin_words():
    for text in ("12345678901234567890。", "abcdefghijklmnopqrst。"):
        warnings: list[str] = []
        spans = plan_sentence_spans(
            text, SubtitlePolicy(max_cue_chars=6, min_cue_seconds=0.0), warnings=warnings
        )
        assert "".join(text[s.start : s.end] for s in spans) == text
        for span in spans[:-1]:
            cut = span.end
            assert not (text[cut - 1].isdigit() and text[cut].isdigit())
            assert not (text[cut - 1].isalpha() and text[cut].isalpha())
        assert any("no safe split" in warning for warning in warnings)


def test_short_spans_merge_only_within_the_same_parent_ids():
    text = "abcd"
    times = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4)]
    policy = SubtitlePolicy(min_cue_seconds=0.8, max_cue_seconds=7.0, max_cue_chars=42)
    same = [SourceSpan(0, 2, ("seg:s1", "sent:0")), SourceSpan(2, 4, ("seg:s1", "sent:0"))]
    merged = balance_spans(text, same, times, policy, stats={})
    assert len(merged) == 1
    assert (merged[0].start, merged[0].end) == (0, 4)

    crossed = [SourceSpan(0, 2, ("seg:s1", "sent:0")), SourceSpan(2, 4, ("seg:s1", "sent:1"))]
    assert len(balance_spans(text, crossed, times, policy)) == 2


def test_repeated_text_punctuation_and_whitespace_keep_exact_spans():
    text = "  重复 重复，重复。\n\n重复 重复！"
    warnings: list[str] = []
    spans = plan_sentence_spans(
        text,
        SubtitlePolicy(max_cue_chars=4, min_cue_seconds=0.0),
        warnings=warnings,
    )
    assert "".join(text[span.start : span.end] for span in spans) == text
    assert spans[0].start == 0
    assert spans[-1].end == len(text)

    items = [
        AlignItem(ch, i * 0.05, (i + 1) * 0.05)
        for i, ch in enumerate(text)
        if not ch.isspace()
    ]
    cues = build_cues(
        text,
        items,
        granularity="sentence",
        audio_duration=len(items) * 0.05,
        policy=SubtitlePolicy(max_cue_chars=4, min_cue_seconds=0.0),
    )
    assert full_text(cues) == text


def test_cps_is_rebalanced_after_timing_within_the_same_parent():
    text = "abcd"
    times = [(0.0, 0.1), (0.0, 0.1), (1.0, 1.1), (1.1, 1.2)]
    stats: dict[str, int] = {}
    spans = [SourceSpan(0, 2, ("seg:s1",)), SourceSpan(2, 4, ("seg:s1",))]
    merged = balance_spans(
        text,
        spans,
        times,
        SubtitlePolicy(max_cps=15.0, min_cue_seconds=0.0, max_merge_gap_ms=5000),
        stats=stats,
    )
    assert len(merged) == 1
    assert (merged[0].start, merged[0].end) == (0, 4)
    assert stats["cps_merges"] == 1

    crossed = [SourceSpan(0, 2, ("seg:s1",)), SourceSpan(2, 4, ("seg:s2",))]
    assert len(balance_spans(text, crossed, times, SubtitlePolicy(max_cps=15.0))) == 2


def test_srt_ranges_keep_boundaries_and_do_not_merge():
    text = "你好\n世界"
    items = [
        AlignItem("你", 0.0, 0.2),
        AlignItem("好", 0.2, 0.4),
        AlignItem("世", 0.5, 0.7),
        AlignItem("界", 0.7, 0.9),
    ]
    cues = build_cues_for_ranges(text, items, [(0, 2), (3, 5)], audio_duration=0.9)
    assert [cue.text for cue in cues] == ["你好", "世界"]
    assert cues[1].start >= cues[0].end


def test_short_sentence_is_a_warning_not_a_quality_failure():
    report = check_subtitles(
        [Cue(0.0, 0.1, "好。")],
        granularity="sentence",
        audio_duration=0.1,
        policy=SubtitlePolicy(),
    )
    assert report.ok is True
    assert any(v.code == "duration_too_short" and v.severity == "warning"
               for v in report.violations)
    assert not any(v.code == "cps_too_high" and v.severity == "error"
                   for v in report.violations)


def test_cps_equal_to_limit_is_not_a_violation():
    text = "一二三四五六七八九十一二三四五六七八九十"  # 20 chars, width 40
    report = check_subtitles(
        [Cue(0.0, 2.0, text)],
        granularity="sentence",
        audio_duration=2.0,
        policy=SubtitlePolicy(max_cps=20.0, max_line_width=42),
    )
    assert report.ok is True
    assert not any(v.code == "cps_too_high" for v in report.violations)


def test_cps_is_checked_even_when_the_cue_is_below_min_duration():
    report = check_subtitles(
        [Cue(0.0, 0.7, "一二三四五六七八九十一二三四五六七八九十")],
        granularity="sentence",
        audio_duration=0.7,
        policy=SubtitlePolicy(min_cue_seconds=0.8, max_cps=20.0),
    )
    assert report.ok is False
    assert any(v.code == "duration_too_short" and v.severity == "warning"
               for v in report.violations)
    assert any(v.code == "cps_too_high" and v.severity == "error"
               for v in report.violations)


def test_merge_does_not_cross_a_large_silence():
    text = "abcd"
    times = [(0.0, 0.1), (0.0, 0.1), (2.0, 2.1), (2.0, 2.1)]
    spans = [SourceSpan(0, 2, ("seg:s1",)), SourceSpan(2, 4, ("seg:s1",))]
    short = balance_spans(
        text,
        spans,
        times,
        SubtitlePolicy(min_cue_seconds=0.8, max_merge_gap_ms=300),
        stats={},
    )
    assert len(short) == 2

    cps = balance_spans(
        text,
        spans,
        times,
        SubtitlePolicy(min_cue_seconds=0.0, max_cps=15.0, max_merge_gap_ms=300),
        stats={},
    )
    assert len(cps) == 2

    allowed = balance_spans(
        text,
        spans,
        times,
        SubtitlePolicy(min_cue_seconds=0.0, max_cps=15.0, max_merge_gap_ms=5000),
        stats={},
    )
    assert len(allowed) == 1


def test_unknown_spans_are_never_merged():
    text = "abcd"
    times = [(0.0, 0.1), (0.0, 0.1), None, None]
    spans = [SourceSpan(0, 2, ("seg:s1",)), SourceSpan(2, 4, ("seg:s1",))]
    merged = balance_spans(
        text,
        spans,
        times,
        SubtitlePolicy(min_cue_seconds=0.0, max_cps=15.0, max_merge_gap_ms=5000),
        stats={},
    )
    assert len(merged) == 2


def test_split_width_is_capped_by_line_width():
    text = "一二三四五六七八九十"
    warnings: list[str] = []
    spans = plan_sentence_spans(
        text,
        SubtitlePolicy(max_cue_chars=10, max_line_width=4, min_cue_seconds=0.0),
        warnings=warnings,
    )
    assert "".join(text[s.start : s.end] for s in spans) == text
    assert all(display_width(text[s.start : s.end]) <= 4 for s in spans)

    items, duration = _items(text, cps=8.0)
    cues = build_cues(
        text,
        items,
        granularity="sentence",
        audio_duration=duration,
        policy=SubtitlePolicy(max_cue_chars=10, max_line_width=4,
                              min_cue_seconds=0.0),
    )
    assert full_text(cues) == text
    assert all(display_width(cue.text) <= 4 for cue in cues)


def test_duration_split_is_counted_in_stats():
    text = "甲乙，丙丁，戊己，庚辛。"
    items, duration = _items(text, cps=20.0)
    stats: dict[str, int] = {}
    build_cues(
        text,
        items,
        granularity="sentence",
        audio_duration=duration,
        policy=SubtitlePolicy(max_cue_seconds=0.2, min_cue_seconds=0.0,
                              max_cue_chars=42),
        stats=stats,
    )
    assert stats.get("splits", 0) >= 1


def test_join_invariant_failure_is_an_input_error():
    with pytest.raises(InputError, match="does not match"):
        ensure_full_text([Cue(0.0, 1.0, "你好")], "你好世界")


def test_align_text_mismatch_precedes_quality_error(
    monkeypatch, tmp_path: Path, model_dir: Path
):
    audio = tmp_path / "a.wav"
    write_audio(audio, np.zeros(24000, dtype=np.float32), 24000, "wav")
    monkeypatch.setattr(align_cmd, "AlignerEngine", _FakeAligner)
    monkeypatch.setattr(
        align_cmd,
        "build_cues",
        lambda *args, **kwargs: [
            Cue(0.0, 0.1, "一二三四五六七八九十一二三四五六七八九十")
        ],
    )
    prefix = tmp_path / "out"
    rc = main([
        "align", str(audio), "--text", "你好。", "--format", "srt",
        "--out-prefix", str(prefix), "--model-dir", str(model_dir),
        "--data-dir", str(tmp_path / "data"),
    ])
    assert rc == 4
    assert not (tmp_path / "out.srt").exists()


def test_mixed_structural_and_threshold_errors_prefer_exit_4():
    report = check_subtitles(
        [
            Cue(0.0, 1.0, "一二三四五六七八九十一二三四五六七八九十"),
            Cue(0.9, 1.9, "世界"),
        ],
        granularity="sentence",
        audio_duration=2.0,
        policy=SubtitlePolicy(max_cps=20.0),
    )
    codes = {v.code for v in report.errors}
    assert "cps_too_high" in codes
    assert "overlap" in codes
    with pytest.raises(InputError) as caught:
        raise_quality_error(report)
    assert caught.value.code == 4


def test_qc_detects_overlap_cps_and_line_width():
    overlap = check_subtitles(
        [Cue(0.0, 1.0, "你好"), Cue(0.5, 1.5, "世界")],
        granularity="sentence",
        audio_duration=2.0,
        policy=SubtitlePolicy(),
    )
    assert overlap.ok is False
    assert any(v.code == "overlap" for v in overlap.violations)

    fast_text = "一二三四五六七八九十一二三四五六七"
    fast = check_subtitles(
        [Cue(0.0, 1.0, fast_text)],
        granularity="sentence",
        audio_duration=1.0,
        policy=SubtitlePolicy(max_cps=20.0, max_line_width=42),
    )
    assert fast.ok is False
    assert any(v.code == "cps_too_high" for v in fast.violations)

    wide_text = "一二三四五六七八九十一二三四五六七八九十一二三四五六七八九十"
    wide = check_subtitles(
        [Cue(0.0, 5.0, wide_text)],
        granularity="sentence",
        audio_duration=5.0,
        policy=SubtitlePolicy(max_cps=20.0, max_line_width=42),
    )
    assert wide.ok is False
    assert any(v.code == "line_too_wide" for v in wide.violations)


def test_char_granularity_only_checks_structure():
    report = check_subtitles(
        [Cue(0.0, 0.01, "你")],
        granularity="char",
        audio_duration=0.01,
        policy=SubtitlePolicy(),
    )
    assert report.ok is True
    assert report.violations == []
    assert report.metrics["cue_count"] == 1


def test_no_qc_still_repairs_overlap_and_records_repairs():
    repairs: dict[str, int] = {}
    cues = monotonic(
        [Cue(0.0, 1.0, "你好"), Cue(0.5, 1.5, "世界")],
        audio_duration=2.0,
        repair_stats=repairs,
    )
    assert cues[1].start >= cues[0].end
    assert repairs["overlap"] >= 1
    report = check_subtitles(
        cues,
        granularity="sentence",
        audio_duration=2.0,
        policy=SubtitlePolicy(qc_enabled=False),
        repairs=repairs,
    )
    assert report.ok is True
    assert report.enabled is False
    assert report.to_dict()["repairs"]["overlap"] >= 1


def test_quality_error_has_numeric_code_and_qc_details():
    report = check_subtitles(
        [Cue(0.0, 1.0, "一二三四五六七八九十一二三四五六七八九十")],
        granularity="sentence",
        audio_duration=1.0,
        policy=SubtitlePolicy(max_cps=20.0),
    )
    with pytest.raises(QualityError) as caught:
        raise_quality_error(report)
    payload = caught.value.to_dict()
    assert payload["code"] == 5
    assert payload["error"] == "quality"
    assert payload["details"]["subtitle_qc"]["ok"] is False
    assert payload["details"]["subtitle_qc"]["violations"]


def test_unrepairable_structural_error_exits_4_even_without_qc():
    report = check_subtitles(
        [Cue(0.0, 0.0, "你好")],
        granularity="sentence",
        audio_duration=1.0,
        policy=SubtitlePolicy(qc_enabled=False),
    )
    assert report.enabled is False
    with pytest.raises(InputError) as caught:
        raise_quality_error(report)
    assert caught.value.code == 4
    assert "subtitle_qc" in caught.value.details


class _FakeEngine:
    def __init__(self, cfg, model):
        self.device = "cpu"
        self.dtype_name = "fp32"
        self.load_ms = 0.0
        self.model = None

    def load(self):
        self.model = object()
        return self

    def generate(self, **kwargs):
        return np.zeros(48000, dtype=np.float32), 24000, {"generate_ms": 1.0}


class _FakeAligner:
    def __init__(self, cfg, model):
        self.model = model

    def align(self, audio, sample_rate, text, language):
        items = []
        now = 0.0
        for ch in text:
            items.append(AlignItem(ch, now, now + 0.1))
            now += 0.1
        return items, {"align_ms": 1.0}


def test_exit_5_writes_audio_and_recipe_but_no_subtitles(
    monkeypatch, capsys, tmp_path: Path, model_dir: Path
):
    monkeypatch.setattr(say_cmd, "TtsEngine", _FakeEngine)
    monkeypatch.setattr(say_cmd, "AlignerEngine", _FakeAligner)
    out = tmp_path / "out.wav"
    rc = main([
        "say", "一二三四五六七八九十", "--voice", "builtin:Vivian",
        "--out", str(out), "--subtitle", "srt", "--max-cps", "1",
        "--model-dir", str(model_dir), "--data-dir", str(tmp_path / "data"),
        "--no-cache", "--json",
    ])
    assert rc == 5
    assert out.is_file()
    assert (tmp_path / "out.recipe.json").is_file()
    assert not (tmp_path / "out.srt").exists()
    _, err = capsys.readouterr()
    assert '"subtitle_qc"' in err
    assert '"cps_too_high"' in err
