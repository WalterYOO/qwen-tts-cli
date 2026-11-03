from __future__ import annotations

import numpy as np
import pytest

from qtts_cli.errors import InputError, UsageError
from qtts_cli.long_align import (
    AudioWindow,
    CharTiming,
    WindowAlignment,
    WindowItem,
    detect_silence_runs,
    merge_window_items,
    monotonic_window_ranges,
    plan_windows,
    validate_srt_cues,
)


def _audio(duration_s: float, silences=(), sr: int = 1000) -> np.ndarray:
    t = np.arange(int(duration_s * sr), dtype=np.float64) / float(sr)
    samples = (0.2 * np.sin(2.0 * np.pi * 220.0 * t)).astype(np.float32)
    for start_s, end_s in silences:
        samples[int(start_s * sr): int(end_s * sr)] = 0.0
    return samples


def _window(index: int, core_start_s: float, core_end_s: float, sr: int = 1000) -> AudioWindow:
    core_start = int(round(core_start_s * sr))
    core_end = int(round(core_end_s * sr))
    audio_start = max(0, core_start - 3 * sr)
    return AudioWindow(
        index=index,
        sr=sr,
        audio_start_sample=audio_start,
        audio_end_sample=core_end,
        core_start_sample=core_start,
        core_end_sample=core_end,
    )


def test_detect_silence_keeps_partial_tail_frame() -> None:
    runs = detect_silence_runs(np.zeros(1005, dtype=np.float32), 1000)
    assert runs == [(0, 1005)]


def test_detect_silence_threshold_is_inclusive_and_exact() -> None:
    assert detect_silence_runs(np.zeros(600, dtype=np.float32), 1000) == [(0, 600)]
    assert detect_silence_runs(np.zeros(599, dtype=np.float32), 1000) == []


def test_plan_windows_short_audio_single_window() -> None:
    windows = plan_windows(_audio(120.0), 1000)
    assert windows == [AudioWindow(0, 1000, 0, 120000, 0, 120000)]
    assert windows[0].display_start is None
    assert windows[0].spoken_start is None


def test_plan_windows_uses_dp_not_greedy_nearest() -> None:
    windows = plan_windows(
        _audio(600.0, silences=[
            (249.5, 250.5),
            (269.5, 270.5),
            (499.5, 500.5),
        ]),
        1000,
    )
    assert [(window.core_start_s, window.core_end_s) for window in windows] == [
        (0.0, 250.0),
        (250.0, 500.0),
        (500.0, 600.0),
    ]
    assert [window.audio_start_s for window in windows] == [0.0, 247.0, 497.0]
    assert [window.audio_end_s for window in windows] == [250.0, 500.0, 600.0]
    assert all(window.audio_duration_s <= 300.0 for window in windows)
    for previous, current in zip(windows, windows[1:]):
        assert previous.core_end_sample == current.core_start_sample
        assert previous.audio_end_s - current.audio_start_s == pytest.approx(3.0)


def test_plan_windows_respects_240_and_285_bounds() -> None:
    low = plan_windows(_audio(520.0, silences=[(240.5, 241.5)]), 1000)
    assert low[0].core_duration_s == pytest.approx(241.0)
    assert 240.0 <= low[0].core_duration_s <= 285.0

    high = plan_windows(_audio(530.0, silences=[(279.5, 280.5)]), 1000)
    assert high[0].core_duration_s == pytest.approx(280.0)
    assert 240.0 <= high[0].core_duration_s <= 285.0


def test_plan_windows_allows_short_final_core() -> None:
    windows = plan_windows(_audio(310.0, silences=[(269.5, 270.5)]), 1000)
    assert len(windows) == 2
    assert windows[0].core_duration_s == pytest.approx(270.0)
    assert windows[1].core_duration_s == pytest.approx(40.0)
    assert windows[1].audio_duration_s <= 300.0


def test_plan_windows_requires_valid_silence_or_approximate() -> None:
    with pytest.raises(InputError, match="--approximate"):
        plan_windows(_audio(400.0), 1000)


def test_plan_windows_srt_cues_are_hard_silent_anchors() -> None:
    windows = plan_windows(
        _audio(600.0, silences=[(254.0, 256.0), (499.0, 501.0)]),
        1000,
        cue_boundaries=[(255.0, 0), (500.0, 100)],
    )
    assert [(window.core_start_s, window.core_end_s) for window in windows] == [
        (0.0, 255.0),
        (255.0, 500.0),
        (500.0, 600.0),
    ]
    assert {window.core_end_s for window in windows[:-1]} <= {255.0, 500.0}


def test_plan_windows_srt_boundary_must_be_inside_silence() -> None:
    with pytest.raises(InputError, match="no valid SRT cue window plan"):
        plan_windows(
            _audio(600.0, silences=[(269.5, 270.5)]),
            1000,
            cue_boundaries=[(255.0, 0), (500.0, 100)],
        )


def test_srt_cue_boundaries_must_be_monotonic() -> None:
    with pytest.raises(InputError, match="not monotonic"):
        plan_windows(
            _audio(600.0),
            1000,
            cue_boundaries=[(300.0, 10), (200.0, 20)],
        )
    with pytest.raises(InputError, match="after the audio"):
        plan_windows(_audio(600.0), 1000, cue_boundaries=[(700.0, 0)])


def test_validate_srt_cues_rejects_invalid_cue_layout() -> None:
    validate_srt_cues([(0.0, 100.0), (100.0, 200.0)], 200.0)
    with pytest.raises(InputError, match="window limit"):
        validate_srt_cues([(0.0, 286.0)], 300.0)
    with pytest.raises(InputError, match="overlaps"):
        validate_srt_cues([(0.0, 100.0), (50.0, 150.0)], 200.0)
    with pytest.raises(InputError, match="after the audio"):
        validate_srt_cues([(0.0, 250.0)], 200.0)
    with pytest.raises(InputError, match="invalid start/end"):
        validate_srt_cues([(10.0, 5.0)], 200.0)
    with pytest.raises(InputError, match="no cues"):
        validate_srt_cues([], 200.0)


def test_plan_windows_invalid_parameters() -> None:
    samples = _audio(10.0)
    with pytest.raises(UsageError):
        plan_windows(samples, 1000, window_s=200.0)
    with pytest.raises(UsageError):
        plan_windows(samples, 1000, overlap_s=1.0)
    with pytest.raises(UsageError):
        plan_windows(samples, 1000, max_window_s=298.0)
    with pytest.raises(UsageError):
        plan_windows(samples, 0)
    with pytest.raises(InputError):
        plan_windows(np.zeros(0, dtype=np.float32), 1000)


def test_plan_windows_deterministic() -> None:
    samples = _audio(600.0, silences=[
        (249.5, 250.5),
        (269.5, 270.5),
        (499.5, 500.5),
    ])
    assert plan_windows(samples, 1000) == plan_windows(samples, 1000)


def test_monotonic_ranges_exact_windows() -> None:
    ranges = monotonic_window_ranges("abcdefghij", ["abcde", "fghij"])
    assert ranges == [(0, 5), (5, 10)]


def test_monotonic_ranges_tolerates_asr_edits() -> None:
    inserted = monotonic_window_ranges("abcdefghij", ["XXabcdeYY", "fghij"])
    deleted = monotonic_window_ranges("abcdefghij", ["abcghij"])
    substituted = monotonic_window_ranges("abcdefghij", ["abcXefghij"])
    assert inserted == [(0, 5), (5, 10)]
    assert deleted == [(0, 10)]
    assert substituted == [(0, 10)]


def test_monotonic_ranges_maps_punctuation_boundaries_to_source() -> None:
    ranges = monotonic_window_ranges("你好，世界！", ["你好", "世界"])
    assert ranges == [(0, 2), (2, 6)]


def test_monotonic_ranges_repeated_text_uses_order_as_context() -> None:
    ranges = monotonic_window_ranges("abcXabc", ["abc", "abc"])
    assert ranges == [(0, 3), (3, 7)]


def test_monotonic_ranges_rejects_unusable_inputs() -> None:
    with pytest.raises(InputError, match="no window transcripts"):
        monotonic_window_ranges("abcdef", [])
    with pytest.raises(InputError, match="empty after normalization"):
        monotonic_window_ranges("abcdef", [""])
    with pytest.raises(InputError, match="does not match"):
        monotonic_window_ranges("abcdef", ["zzz"])
    with pytest.raises(InputError, match="matches too little"):
        monotonic_window_ranges("abcdefghij", ["abcXXXXXX"])
    with pytest.raises(InputError, match="empty after normalization"):
        monotonic_window_ranges("", ["abc"])


def test_monotonic_ranges_requires_each_window_to_advance() -> None:
    with pytest.raises(InputError, match="does not match"):
        monotonic_window_ranges("abc", ["abc", "abc"])


def test_merge_prefers_core_candidate_over_overlap_candidate() -> None:
    # The first window's audio context extends to 10s even though its core
    # (ownership range) ends at 5s.
    window0 = AudioWindow(0, 1000, 0, 10000, 0, 5000)
    window1 = _window(1, 5.0, 10.0)
    result = merge_window_items(
        "abc",
        [
            WindowAlignment(
                window0,
                [WindowItem("abc", 6.0, 9.0, 0, 3)],
            ),
            WindowAlignment(
                window1,
                [WindowItem("abc", 6.5, 9.5, 0, 3)],
            ),
        ],
        duration_s=10.0,
    )
    assert result == [
        CharTiming(0, "a", 6.5, 9.5),
        CharTiming(1, "b", 6.5, 9.5),
        CharTiming(2, "c", 6.5, 9.5),
    ]


def test_merge_fills_punctuation_from_neighbours() -> None:
    window = _window(0, 0.0, 10.0)
    result = merge_window_items(
        "a, b",
        [
            WindowAlignment(
                window,
                [
                    WindowItem("a", 0.0, 1.0, 0, 1),
                    WindowItem("b", 2.0, 3.0, 3, 4),
                ],
            )
        ],
        duration_s=10.0,
    )
    assert [timing.char for timing in result] == list("a, b")
    assert result[1].start_s == result[2].start_s == 0.0
    assert result[3].start_s == 2.0


def test_merge_rejects_missing_non_punctuation_coverage() -> None:
    window = _window(0, 0.0, 10.0)
    with pytest.raises(InputError, match="no aligner item"):
        merge_window_items(
            "abc",
            [WindowAlignment(window, [WindowItem("a", 0.0, 1.0, 0, 1)])],
            duration_s=10.0,
        )


def test_merge_allows_item_crossing_core_boundary() -> None:
    window = _window(0, 10.0, 20.0)
    result = merge_window_items(
        "ab",
        [WindowAlignment(window, [WindowItem("ab", 9.0, 19.5, 0, 2)])],
        duration_s=30.0,
    )
    assert [(item.start_s, item.end_s) for item in result] == [
        (9.0, 19.5),
        (9.0, 19.5),
    ]


def test_merge_rejects_regressing_times_and_out_of_range_items() -> None:
    window = _window(0, 0.0, 10.0)
    with pytest.raises(InputError, match="monotonic"):
        merge_window_items(
            "ab",
            [
                WindowAlignment(
                    window,
                    [
                        WindowItem("a", 5.0, 6.0, 0, 1),
                        WindowItem("b", 1.0, 2.0, 1, 2),
                    ],
                )
            ],
            duration_s=10.0,
        )
    with pytest.raises(InputError, match="outside the audio"):
        merge_window_items(
            "a",
            [WindowAlignment(window, [WindowItem("a", 1.0, 11.0, 0, 1)])],
            duration_s=10.0,
        )
    with pytest.raises(InputError, match="outside the reference"):
        merge_window_items(
            "a",
            [WindowAlignment(window, [WindowItem("a", 1.0, 2.0, 0, 2)])],
            duration_s=10.0,
        )
    with pytest.raises(InputError, match="display range is empty"):
        merge_window_items(
            "a",
            [WindowAlignment(window, [WindowItem("", 1.0, 2.0, 0, 0)])],
            duration_s=10.0,
        )


def test_merge_rejects_item_beyond_its_own_window_audio() -> None:
    window0 = AudioWindow(0, 1000, 0, 5000, 0, 5000)
    window1 = AudioWindow(1, 1000, 2000, 10000, 5000, 10000)
    with pytest.raises(InputError, match="outside the audio") as exc_info:
        merge_window_items(
            "ab",
            [
                WindowAlignment(window0, [WindowItem("a", 4.5, 7.0, 0, 1)]),
                WindowAlignment(window1, [WindowItem("b", 7.0, 8.0, 1, 2)]),
            ],
            duration_s=10.0,
        )
    assert exc_info.value.details["reason"] == "item_outside_window_audio"


def test_merge_rejects_item_before_its_window_audio_start() -> None:
    window0 = AudioWindow(0, 1000, 2000, 5000, 2000, 5000)
    window1 = AudioWindow(1, 1000, 5000, 10000, 5000, 10000)
    with pytest.raises(InputError, match="outside the audio"):
        merge_window_items(
            "ab",
            [
                WindowAlignment(window0, [WindowItem("a", 1.0, 2.0, 0, 1)]),
                WindowAlignment(window1, [WindowItem("b", 5.0, 6.0, 1, 2)]),
            ],
            duration_s=10.0,
        )


def test_merge_empty_display_and_no_alignments() -> None:
    assert merge_window_items("", [], duration_s=10.0) == []
    with pytest.raises(InputError, match="no aligner item"):
        merge_window_items("a", [], duration_s=10.0)
    # Punctuation-only text with no items owns a zero-length timing.
    assert merge_window_items(",", [], duration_s=10.0) == [
        CharTiming(0, ",", 0.0, 0.0)
    ]


def test_plan_windows_approximate_falls_back_to_fixed_intervals() -> None:
    windows = plan_windows(_audio(330.0), 1000, approximate=True)
    assert windows.kind == "fixed"
    assert [window.core_duration_s for window in windows] == [270.0, 60.0]
    assert windows[0].audio_start_sample == 0
    assert windows[1].audio_start_sample == 267000
    assert all(window.core_duration_s <= 285.0 for window in windows)
    assert all(window.audio_duration_s <= 300.0 for window in windows)
    with pytest.raises(InputError, match="--approximate"):
        plan_windows(_audio(330.0), 1000)


def test_plan_windows_rejects_final_core_over_maximum() -> None:
    # The only silence at 241s would leave a 289s final core, which is longer
    # than max_window_s; the planner must reject it rather than emit it.
    with pytest.raises(InputError, match="no valid silence window plan"):
        plan_windows(_audio(530.0, silences=[(240.5, 241.5)]), 1000)
    # A second qualifying cut lets the DP find a valid 270/270/20 plan.
    windows = plan_windows(
        _audio(560.0, silences=[(269.5, 270.5), (539.5, 540.5)]), 1000
    )
    assert [window.core_duration_s for window in windows] == [270.0, 270.0, 20.0]


def test_plan_windows_fixed_fallback_keeps_final_core_within_maximum() -> None:
    windows = plan_windows(_audio(560.0), 1000, approximate=True)
    assert windows.kind == "fixed"
    assert [window.core_duration_s for window in windows] == [270.0, 270.0, 20.0]
    assert all(window.core_duration_s <= 285.0 for window in windows)


def test_merge_approximate_clamps_overlap_regression() -> None:
    window0 = _window(0, 0.0, 5.0)
    window1 = _window(1, 5.0, 10.0)
    windows = [
        WindowAlignment(window0, [WindowItem("a", 4.0, 5.0, 0, 1)]),
        WindowAlignment(window1, [WindowItem("b", 2.0, 2.5, 1, 2)]),
    ]
    with pytest.raises(InputError, match="monotonically"):
        merge_window_items("ab", windows, duration_s=10.0)
    stats: dict[str, int] = {}
    result = merge_window_items(
        "ab", windows, duration_s=10.0, approximate=True, stats=stats
    )
    assert [(item.start_s, item.end_s) for item in result] == [(4.0, 5.0), (5.0, 5.0)]
    assert stats["monotonic_adjustments"] == 1


def test_merge_dp_avoids_dead_end_candidate() -> None:
    window0 = _window(0, 0.0, 10.0)
    # A large left overlap is legal and lets the second window own early text.
    window1 = AudioWindow(1, 1000, 0, 20000, 10000, 20000)
    result = merge_window_items(
        "ab",
        [
            WindowAlignment(
                window0,
                [
                    WindowItem("a", 5.0, 6.0, 0, 1),
                    WindowItem("b", 1.0, 2.0, 1, 2),
                ],
            ),
            WindowAlignment(window1, [WindowItem("a", 0.0, 1.0, 0, 1)]),
        ],
        duration_s=20.0,
    )
    assert [(item.start_s, item.end_s) for item in result] == [(0.0, 1.0), (1.0, 2.0)]


def test_merge_tail_allowance_clips_only_last_window() -> None:
    window0 = _window(0, 0.0, 5.0)
    window1 = _window(1, 5.0, 10.0)
    result = merge_window_items(
        "a",
        [WindowAlignment(window1, [WindowItem("a", 9.5, 10.5, 0, 1)])],
        duration_s=10.0,
        tail_allowance_s=1.0,
    )
    assert [(item.start_s, item.end_s) for item in result] == [(9.5, 10.0)]
    with pytest.raises(InputError, match="outside the audio"):
        merge_window_items(
            "ab",
            [
                WindowAlignment(window0, [WindowItem("a", 4.5, 10.5, 0, 1)]),
                WindowAlignment(window1, [WindowItem("b", 6.0, 7.0, 1, 2)]),
            ],
            duration_s=10.0,
            tail_allowance_s=1.0,
        )
    with pytest.raises(InputError, match="outside the audio"):
        merge_window_items(
            "a",
            [WindowAlignment(window1, [WindowItem("a", 10.0, 10.5, 0, 1)])],
            duration_s=10.0,
            tail_allowance_s=1.0,
        )


def test_merge_allows_zero_duration_item_but_rejects_raw_reverse() -> None:
    window = _window(0, 0.0, 10.0)
    result = merge_window_items(
        "a", [WindowAlignment(window, [WindowItem("a", 2.0, 2.0, 0, 1)])],
        duration_s=10.0,
    )
    assert (result[0].start_s, result[0].end_s) == (2.0, 2.0)
    with pytest.raises(InputError, match="end precedes start"):
        merge_window_items(
            "a", [WindowAlignment(window, [WindowItem("a", 3.0, 2.0, 0, 1)])],
            duration_s=10.0,
        )


def test_monotonic_ranges_reject_sparse_transcript_tail() -> None:
    with pytest.raises(InputError, match="tail uncovered"):
        monotonic_window_ranges(
            "abcdefghijklmnopqrstuvwxyz",
            ["abc"],
            windows=[_window(0, 0.0, 10.0)],
            duration_s=10.0,
        )


def test_monotonic_ranges_short_reference_tail_is_not_exempt() -> None:
    # A 10-character reference used to get a hard-coded 8-character tolerance,
    # which let this 30% match pass.  The floor is now 2.
    with pytest.raises(InputError, match="tail uncovered"):
        monotonic_window_ranges(
            "abcdefghij",
            ["abc"],
            windows=[_window(0, 0.0, 10.0)],
            duration_s=10.0,
        )


def test_monotonic_ranges_tiny_final_suffix_still_passes() -> None:
    windows = [_window(0, 0.0, 10.0), _window(1, 10.0, 20.0)]
    assert monotonic_window_ranges(
        "ab", ["a", "b"], windows=windows, duration_s=20.0
    ) == [(0, 1), (1, 2)]
    assert monotonic_window_ranges(
        "abcd", ["ab", "cd"], windows=windows, duration_s=20.0
    ) == [(0, 2), (2, 4)]


def test_monotonic_ranges_reject_insufficient_reference_advance() -> None:
    with pytest.raises(InputError, match="advances too little"):
        monotonic_window_ranges(
            "abcdefghijklmnopqrstuvwxyz",
            ["abc", "defghijklmnopqrstuvwxyz"],
            windows=[_window(0, 0.0, 5.0), _window(1, 5.0, 10.0)],
            duration_s=10.0,
        )


def test_monotonic_ranges_normal_window_coverage_passes() -> None:
    ranges = monotonic_window_ranges(
        "abcdefghij",
        ["abcdefghij"],
        windows=[_window(0, 0.0, 10.0)],
        duration_s=10.0,
    )
    assert ranges == [(0, 10)]


def test_monotonic_ranges_six_repeats_are_not_truncated() -> None:
    ranges = monotonic_window_ranges("abc" * 6, ["abc"] * 6)
    assert ranges == [(index * 3, (index + 1) * 3) for index in range(6)]
