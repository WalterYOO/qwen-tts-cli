from __future__ import annotations

import json
import shutil
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pytest

from qtts_cli.audio_io import write_audio
from qtts_cli.cli import main
from qtts_cli.commands import align as align_cmd
from qtts_cli.errors import InputError, ModelError
from qtts_cli.lexicon import MapBlock, PreparedText
from qtts_cli.long_align import AudioWindow
from qtts_cli.subtitles import AlignItem


class FakeAligner:
    instances: list["FakeAligner"] = []
    sparse_fraction: float | None = None
    tail_extension: float = 0.0
    tail_extension_call: int | None = None
    align_calls: int = 0

    def __init__(self, cfg, model):
        assert not FakeAsr.loaded, "aligner was constructed while ASR was still resident"
        self.model_info = model
        self.load_ms = 2.0
        FakeAligner.instances.append(self)

    def align(self, samples, sr, text, language=None):
        call_index = FakeAligner.align_calls
        FakeAligner.align_calls += 1
        duration = len(samples) / float(sr)
        step = duration / max(1, len(text))
        items = [
            AlignItem(char, index * step, (index + 1) * step)
            for index, char in enumerate(text)
            if not char.isspace()
        ]
        if FakeAligner.sparse_fraction is not None:
            keep = max(1, int(round(len(items) * FakeAligner.sparse_fraction)))
            items = items[:keep]
        extend_tail = FakeAligner.tail_extension and (
            FakeAligner.tail_extension_call is None
            or FakeAligner.tail_extension_call == call_index
        )
        if extend_tail and items:
            last = items[-1]
            items[-1] = AlignItem(
                last.text, last.start, last.end + FakeAligner.tail_extension
            )
        return items, {
            "align_ms": 1.0,
            "language": language,
            "items": len(items),
            "duration_s": duration,
        }


class FakeAsr:
    instances: list["FakeAsr"] = []
    transcripts: list[str] = []
    calls: list[float] = []
    released: list["FakeAsr"] = []
    loaded = False

    def __init__(self, cfg, model):
        self.model_info = model
        self.released_here = False
        FakeAsr.instances.append(self)

    def transcribe_samples(self, samples, sr, *, language=None):
        FakeAsr.loaded = True
        duration = len(samples) / float(sr)
        FakeAsr.calls.append(duration)
        index = len(FakeAsr.calls) - 1
        if index >= len(FakeAsr.transcripts):
            raise AssertionError("unexpected ASR window")
        return {
            "text": FakeAsr.transcripts[index],
            "language": language,
            "items": None,
        }

    def release(self):
        FakeAsr.loaded = False
        self.released_here = True
        FakeAsr.released.append(self)


@pytest.fixture(autouse=True)
def _reset_fakes(monkeypatch):
    FakeAligner.instances = []
    FakeAligner.sparse_fraction = None
    FakeAligner.tail_extension = 0.0
    FakeAligner.tail_extension_call = None
    FakeAligner.align_calls = 0
    FakeAsr.instances = []
    FakeAsr.transcripts = []
    FakeAsr.calls = []
    FakeAsr.released = []
    FakeAsr.loaded = False
    monkeypatch.setattr(align_cmd, "AlignerEngine", FakeAligner)
    monkeypatch.setattr(align_cmd, "AsrEngine", FakeAsr)


def _add_asr(model_dir: Path) -> Path:
    root = model_dir / "Qwen3-ASR-0.6B"
    root.mkdir()
    (root / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_asr",
                "support_languages": ["chinese", "english"],
            }
        ),
        encoding="utf-8",
    )
    (root / "model.safetensors").write_bytes(b"fake")
    return root


def _long_audio(path: Path, duration: float = 310.0, silences=((269.5, 270.5),)) -> Path:
    sr = 1000
    t = np.arange(int(duration * sr), dtype=np.float64) / sr
    samples = (0.1 * np.sin(2.0 * np.pi * 220.0 * t)).astype(np.float32)
    for start, end in silences:
        samples[int(start * sr): int(end * sr)] = 0.0
    return write_audio(path, samples, sr, "wav")


def _run(tmp_path: Path, model_dir: Path, *args: str) -> int:
    return main(
        [
            *args,
            "--model-dir",
            str(model_dir),
            "--data-dir",
            str(tmp_path / "data"),
            "--cache-dir",
            str(tmp_path / "cache"),
        ]
    )


LONG_TEXT = "甲乙丙丁戊己庚辛壬癸子丑寅卯辰巳午未申酉"


def test_long_plain_asr_release_order_and_windowing(
    capsys, tmp_path: Path, model_dir: Path
):
    _add_asr(model_dir)
    audio = _long_audio(tmp_path / "long.wav")
    FakeAsr.transcripts = [LONG_TEXT[:10], LONG_TEXT[10:]]
    prefix = tmp_path / "aligned"
    rc = _run(
        tmp_path,
        model_dir,
        "align",
        str(audio),
        "--text",
        LONG_TEXT,
        "--language",
        "Chinese",
        "--format",
        "json",
        "--out-prefix",
        str(prefix),
        "--recipe-out",
        str(tmp_path / "aligned.recipe.json"),
        "--no-subtitle-qc",
        "--json",
    )
    assert rc == 0, capsys.readouterr().err
    payload = json.loads(capsys.readouterr().out)
    assert payload["windowing"]["enabled"] is True
    assert payload["windowing"]["approximate"] is False
    assert payload["windowing"]["asr_model"] == "Qwen3-ASR-0.6B"
    assert len(payload["windowing"]["windows"]) == 2
    assert payload["windowing"]["windows"][0]["transcript_sha256"]
    assert FakeAsr.calls == pytest.approx([270.0, 43.0])
    assert len(FakeAsr.instances) == 1
    assert FakeAsr.released and FakeAsr.released[0].released_here
    assert FakeAsr.loaded is False
    assert len(FakeAligner.instances) == 1
    data = json.loads(Path(f"{prefix}.json").read_text(encoding="utf-8"))
    assert "".join(cue["text"] for cue in data["cues"]) == LONG_TEXT
    recipe = json.loads((tmp_path / "aligned.recipe.json").read_text(encoding="utf-8"))
    assert recipe["windowing"] == payload["windowing"]
    assert payload["coverage"] == 1.0
    times = [(cue["start"], cue["end"]) for cue in data["cues"]]
    assert times[0][0] >= 0
    assert times[-1][1] <= payload["duration_s"] + 1e-6
    assert all(left[1] <= right[1] for left, right in zip(times, times[1:]))


def test_long_requires_asr_model_before_loading(
    capsys, tmp_path: Path, model_dir: Path
):
    audio = _long_audio(tmp_path / "long.wav")
    prefix = tmp_path / "missing"
    rc = _run(
        tmp_path,
        model_dir,
        "align",
        str(audio),
        "--text",
        LONG_TEXT,
        "--out-prefix",
        str(prefix),
        "--json",
    )
    assert rc == 3
    assert capsys.readouterr().out == ""
    assert not Path(f"{prefix}.srt").exists()
    assert FakeAligner.instances == []
    assert FakeAsr.instances == []


def test_long_approximate_skips_asr_and_warns(
    capsys, tmp_path: Path, model_dir: Path
):
    audio = _long_audio(tmp_path / "long.wav")
    prefix = tmp_path / "approx"
    rc = _run(
        tmp_path,
        model_dir,
        "align",
        str(audio),
        "--text",
        LONG_TEXT,
        "--language",
        "Chinese",
        "--approximate",
        "--format",
        "json",
        "--out-prefix",
        str(prefix),
        "--no-subtitle-qc",
        "--json",
    )
    assert rc == 0, capsys.readouterr().err
    payload = json.loads(capsys.readouterr().out)
    assert FakeAsr.instances == []
    assert payload["windowing"]["approximate"] is True
    assert payload["windowing"]["asr_model"] is None
    assert align_cmd.APPROXIMATE_WARNING in payload["warnings"]
    assert align_cmd.APPROXIMATE_WARNING in payload["windowing"]["warnings"]
    assert payload["windowing"]["warnings"] == payload["warnings"]
    data = json.loads(Path(f"{prefix}.json").read_text(encoding="utf-8"))
    assert "".join(cue["text"] for cue in data["cues"]) == LONG_TEXT


@pytest.mark.parametrize(
    "flags",
    [["--window-seconds", "239"], ["--window-seconds", "286"], ["--window-overlap", "1"]],
)
def test_window_parameters_fail_before_models(
    capsys, tmp_path: Path, model_dir: Path, flags
):
    audio = _long_audio(tmp_path / "long.wav")
    rc = _run(
        tmp_path,
        model_dir,
        "align",
        str(audio),
        "--text",
        LONG_TEXT,
        *flags,
        "--out-prefix",
        str(tmp_path / "bad"),
        "--json",
    )
    assert rc == 2
    assert capsys.readouterr().out == ""
    assert FakeAligner.instances == []


def test_long_invalid_asr_language_exits_2_before_models(
    capsys, tmp_path: Path, model_dir: Path
):
    _add_asr(model_dir)
    audio = _long_audio(tmp_path / "long.wav")
    rc = _run(
        tmp_path,
        model_dir,
        "align",
        str(audio),
        "--text",
        LONG_TEXT,
        "--language",
        "Klingon",
        "--out-prefix",
        str(tmp_path / "bad"),
        "--json",
    )
    assert rc == 2
    assert capsys.readouterr().out == ""
    assert FakeAsr.instances == []
    assert FakeAligner.instances == []


def test_short_align_ignores_asr_flags_and_keeps_bytes(
    capsys, tmp_path: Path, model_dir: Path
):
    audio = tmp_path / "short.wav"
    write_audio(audio, np.zeros(10000, dtype=np.float32), 1000, "wav")
    first = tmp_path / "first"
    second = tmp_path / "second"
    common = [
        "align",
        str(audio),
        "--text",
        "你好世界",
        "--language",
        "Chinese",
        "--format",
        "srt",
        "--no-subtitle-qc",
        "--json",
    ]
    assert _run(tmp_path, model_dir, *common, "--out-prefix", str(first)) == 0
    capsys.readouterr()
    assert _run(
        tmp_path,
        model_dir,
        *common,
        "--out-prefix",
        str(second),
        "--window-seconds",
        "250",
        "--window-overlap",
        "2",
        "--asr-model",
        "/does/not/exist",
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["windowing"] == {"enabled": False}
    assert FakeAsr.instances == []
    assert Path(f"{first}.srt").read_bytes() == Path(f"{second}.srt").read_bytes()


def test_long_srt_never_selects_asr_and_preserves_indices(
    capsys, tmp_path: Path, model_dir: Path
):
    audio = _long_audio(
        tmp_path / "srt.wav",
        duration=600.0,
        silences=((254.0, 256.0), (499.0, 501.0)),
    )
    texts = [
        "甲乙丙丁戊己庚辛壬癸",
        "子丑寅卯辰巳午未申酉",
        "金木水火土风雨雷电光",
    ]
    source = tmp_path / "source.srt"
    source.write_text(
        "7\n00:00:00,000 --> 00:04:15,000\n"
        + texts[0]
        + "\n\n"
        + "9\n00:04:15,000 --> 00:08:20,000\n"
        + texts[1]
        + "\n\n"
        + "11\n00:08:20,000 --> 00:10:00,000\n"
        + texts[2]
        + "\n",
        encoding="utf-8",
    )
    prefix = tmp_path / "srt_out"
    rc = _run(
        tmp_path,
        model_dir,
        "align",
        str(audio),
        "--srt",
        str(source),
        "--language",
        "Chinese",
        "--format",
        "srt",
        "--out-prefix",
        str(prefix),
        "--no-subtitle-qc",
        "--json",
    )
    assert rc == 0, capsys.readouterr().err
    payload = json.loads(capsys.readouterr().out)
    assert FakeAsr.instances == []
    assert payload["windowing"]["enabled"] is True
    assert payload["windowing"]["asr_model"] is None
    assert len(payload["windowing"]["windows"]) == 3
    output = Path(f"{prefix}.srt").read_text(encoding="utf-8")
    for index in ("7", "9", "11"):
        assert f"\n{index}\n" in f"\n{output}"
    assert all(text in output for text in texts)


def test_long_srt_cue_over_limit_exits_4_before_models(
    capsys, tmp_path: Path, model_dir: Path
):
    audio = _long_audio(
        tmp_path / "srt.wav", duration=400.0, silences=((285.0, 287.0),)
    )
    source = tmp_path / "source.srt"
    source.write_text(
        "1\n00:00:00,000 --> 00:04:46,000\n甲乙\n\n"
        "2\n00:04:46,000 --> 00:06:40,000\n丙丁\n",
        encoding="utf-8",
    )
    rc = _run(
        tmp_path,
        model_dir,
        "align",
        str(audio),
        "--srt",
        str(source),
        "--out-prefix",
        str(tmp_path / "bad"),
        "--json",
    )
    assert rc == 4
    error = json.loads(capsys.readouterr().err)
    assert "window limit" in error["message"]
    assert FakeAligner.instances == []


def test_long_srt_requires_silent_cue_anchor(
    capsys, tmp_path: Path, model_dir: Path
):
    audio = _long_audio(
        tmp_path / "srt.wav",
        duration=600.0,
        silences=((269.5, 270.5),),
    )
    source = tmp_path / "source.srt"
    source.write_text(
        "1\n00:00:00,000 --> 00:04:15,000\n甲乙\n\n"
        "2\n00:04:15,000 --> 00:08:20,000\n丙丁\n\n"
        "3\n00:08:20,000 --> 00:10:00,000\n戊己\n",
        encoding="utf-8",
    )
    rc = _run(
        tmp_path,
        model_dir,
        "align",
        str(audio),
        "--srt",
        str(source),
        "--out-prefix",
        str(tmp_path / "bad"),
        "--json",
    )
    assert rc == 4
    assert FakeAligner.instances == []


def test_long_lexicon_n_to_one_never_splits_term(
    capsys, tmp_path: Path, model_dir: Path
):
    _add_asr(model_dir)
    audio = _long_audio(tmp_path / "long.wav")
    lexicon = tmp_path / "lexicon.json"
    lexicon.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "entries": [{"term": "AI", "spoken": "人工智能"}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    text = "甲乙丙丁AI戊己庚辛"
    FakeAsr.transcripts = ["甲乙丙丁人工", "智能戊己庚辛"]
    prefix = tmp_path / "lex"
    rc = _run(
        tmp_path,
        model_dir,
        "align",
        str(audio),
        "--text",
        text,
        "--language",
        "Chinese",
        "--lexicon",
        str(lexicon),
        "--format",
        "json",
        "--out-prefix",
        str(prefix),
        "--no-subtitle-qc",
        "--json",
    )
    assert rc == 0, capsys.readouterr().err
    payload = json.loads(capsys.readouterr().out)
    windows = payload["windowing"]["windows"]
    assert windows[0]["display_end"] == 6  # through the whole "AI" term
    assert windows[1]["display_start"] == 4  # snap back to the term start
    data = json.loads(Path(f"{prefix}.json").read_text(encoding="utf-8"))
    cue_text = "".join(cue["text"] for cue in data["cues"])
    assert cue_text == text
    assert "人工智能" not in cue_text


def test_long_failure_writes_no_partial_outputs(
    capsys, tmp_path: Path, model_dir: Path
):
    _add_asr(model_dir)
    audio = _long_audio(tmp_path / "long.wav")
    FakeAsr.transcripts = [LONG_TEXT[:10], LONG_TEXT[10:]]

    class FailingAligner(FakeAligner):
        def align(self, samples, sr, text, language=None):
            if getattr(self, "calls", 0):
                raise ModelError("second window failed")
            self.calls = 1
            return super().align(samples, sr, text, language)

    align_cmd.AlignerEngine = FailingAligner
    prefix = tmp_path / "partial"
    recipe = tmp_path / "partial.recipe.json"
    rc = _run(
        tmp_path,
        model_dir,
        "align",
        str(audio),
        "--text",
        LONG_TEXT,
        "--language",
        "Chinese",
        "--out-prefix",
        str(prefix),
        "--recipe-out",
        str(recipe),
        "--json",
    )
    assert rc == 3
    assert not Path(f"{prefix}.srt").exists()
    assert not recipe.exists()
    assert not list(tmp_path.glob(".partial.*.tmp"))


def test_long_lexicon_projects_once(
    monkeypatch, capsys, tmp_path: Path, model_dir: Path
):
    calls: list[str] = []
    original = align_cmd.project_text

    def counting_project(text, **kwargs):
        calls.append(text)
        return original(text, **kwargs)

    monkeypatch.setattr(align_cmd, "project_text", counting_project)
    audio = _long_audio(tmp_path / "long.wav")
    lexicon = tmp_path / "lexicon.json"
    lexicon.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "entries": [{"term": "AI", "spoken": "人工智能"}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    rc = _run(
        tmp_path,
        model_dir,
        "align",
        str(audio),
        "--text",
        "甲乙AI戊己",
        "--language",
        "Chinese",
        "--lexicon",
        str(lexicon),
        "--approximate",
        "--format",
        "json",
        "--out-prefix",
        str(tmp_path / "once"),
        "--no-subtitle-qc",
        "--json",
    )
    assert rc == 0, capsys.readouterr().err
    assert len(calls) == 1


def test_approximate_warning_goes_to_stderr(
    capsys, tmp_path: Path, model_dir: Path
):
    audio = _long_audio(tmp_path / "long.wav")
    rc = _run(
        tmp_path,
        model_dir,
        "align",
        str(audio),
        "--text",
        LONG_TEXT,
        "--language",
        "Chinese",
        "--approximate",
        "--format",
        "srt",
        "--out-prefix",
        str(tmp_path / "warn"),
        "--no-subtitle-qc",
    )
    assert rc == 0, capsys.readouterr().err
    captured = capsys.readouterr()
    assert align_cmd.APPROXIMATE_WARNING in captured.err
    assert align_cmd.APPROXIMATE_WARNING not in captured.out
    assert "aligned" in captured.out


def test_approximate_ranges_use_audio_spans_and_snap_terms() -> None:
    text = "ABCDEFGHIJ"
    projection = PreparedText(
        display_text=text,
        spoken_text=text,
        mapping=[MapBlock(3, 7, 3, 7, term="DEFG")],
        source_spans=None,
    )
    windows = [
        AudioWindow(0, 1000, 0, 6000, 0, 5000),
        AudioWindow(1, 1000, 4000, 10000, 5000, 10000),
    ]
    assert align_cmd._approximate_display_ranges(
        windows, projection, audio_s=10.0
    ) == [(0, 7), (3, 10)]


def test_approximate_ranges_never_leave_a_coverage_hole() -> None:
    text = "ABCDEFGHIJK"
    projection = PreparedText(
        display_text=text,
        spoken_text=text,
        mapping=[MapBlock(3, 10, 3, 10, term="DEFGHIJ")],
        source_spans=None,
    )
    windows = [
        AudioWindow(0, 1000, 0, 6000, 0, 5000),
        AudioWindow(1, 1000, 8000, 10000, 5000, 10000),
        AudioWindow(2, 1000, 10000, 11000, 10000, 11000),
    ]
    ranges = align_cmd._approximate_display_ranges(
        windows, projection, audio_s=11.0
    )
    assert ranges == [(0, 3), (3, 10), (10, 11)]
    assert ranges[0][0] == 0
    assert ranges[-1][1] == len(text)
    assert all(
        current[0] <= previous[1]
        for previous, current in zip(ranges, ranges[1:])
    )
    assert any(start <= 3 and end >= 10 for start, end in ranges)


def test_long_approximate_falls_back_to_fixed_intervals(
    capsys, tmp_path: Path, model_dir: Path
):
    _add_asr(model_dir)
    audio = _long_audio(tmp_path / "continuous.wav", duration=330.0, silences=())
    prefix = tmp_path / "fixed"
    rc = _run(
        tmp_path,
        model_dir,
        "align",
        str(audio),
        "--text",
        LONG_TEXT,
        "--language",
        "Chinese",
        "--approximate",
        "--format",
        "json",
        "--out-prefix",
        str(prefix),
        "--no-subtitle-qc",
        "--json",
    )
    assert rc == 0, capsys.readouterr().err
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["windowing"]["plan"] == "fixed"
    assert payload["windowing"]["approximate"] is True
    assert align_cmd.FIXED_WINDOW_WARNING in payload["warnings"]
    assert align_cmd.FIXED_WINDOW_WARNING in payload["windowing"]["warnings"]
    assert align_cmd.FIXED_WINDOW_WARNING in captured.err
    assert FakeAsr.instances == []
    cues = json.loads(Path(f"{prefix}.json").read_text(encoding="utf-8"))["cues"]
    assert "".join(cue["text"] for cue in cues) == LONG_TEXT
    starts = [cue["start"] for cue in cues]
    ends = [cue["end"] for cue in cues]
    assert starts == sorted(starts)
    assert ends == sorted(ends)

    rc = _run(
        tmp_path,
        model_dir,
        "align",
        str(audio),
        "--text",
        LONG_TEXT,
        "--language",
        "Chinese",
        "--out-prefix",
        str(tmp_path / "strict"),
        "--json",
    )
    assert rc == 4
    assert "--approximate" in capsys.readouterr().err
    assert not Path(tmp_path / "strict.srt").exists()
    assert FakeAsr.instances == []


def test_long_approximate_repeated_text_overlapping_ranges(
    capsys, tmp_path: Path, model_dir: Path
):
    audio = _long_audio(tmp_path / "continuous.wav", duration=330.0, silences=())
    text = "你好世界。" * 20
    prefix = tmp_path / "repeat"
    rc = _run(
        tmp_path,
        model_dir,
        "align",
        str(audio),
        "--text",
        text,
        "--language",
        "Chinese",
        "--approximate",
        "--format",
        "json",
        "--out-prefix",
        str(prefix),
        "--no-subtitle-qc",
        "--json",
    )
    assert rc == 0, capsys.readouterr().err
    payload = json.loads(capsys.readouterr().out)
    cues = json.loads(Path(f"{prefix}.json").read_text(encoding="utf-8"))["cues"]
    assert "".join(cue["text"] for cue in cues) == text
    assert payload["windowing"]["plan"] == "fixed"
    assert 0.0 <= payload["coverage"] <= 1.0


def test_long_last_window_tail_is_clipped(
    capsys, tmp_path: Path, model_dir: Path
):
    FakeAligner.tail_extension = 1.59
    FakeAligner.tail_extension_call = 1
    audio = _long_audio(tmp_path / "long.wav")
    prefix = tmp_path / "tail"
    rc = _run(
        tmp_path,
        model_dir,
        "align",
        str(audio),
        "--text",
        LONG_TEXT,
        "--language",
        "Chinese",
        "--approximate",
        "--format",
        "json",
        "--out-prefix",
        str(prefix),
        "--no-subtitle-qc",
        "--json",
    )
    assert rc == 0, capsys.readouterr().err
    payload = json.loads(capsys.readouterr().out)
    cues = json.loads(Path(f"{prefix}.json").read_text(encoding="utf-8"))["cues"]
    assert cues[-1]["end"] <= payload["duration_s"] + 1e-6


def test_commit_failure_leaves_no_partial_outputs(
    capsys, tmp_path: Path, model_dir: Path
):
    audio = tmp_path / "short.wav"
    write_audio(audio, np.zeros(10000, dtype=np.float32), 1000, "wav")
    prefix = tmp_path / "out"
    recipe_dir = tmp_path / "recipe_dir"
    recipe_dir.mkdir()
    rc = _run(
        tmp_path,
        model_dir,
        "align",
        str(audio),
        "--text",
        "你好世界",
        "--language",
        "Chinese",
        "--format",
        "srt,json",
        "--out-prefix",
        str(prefix),
        "--recipe-out",
        str(recipe_dir),
        "--no-subtitle-qc",
        "--json",
    )
    assert rc != 0
    assert not Path(f"{prefix}.srt").exists()
    assert not Path(f"{prefix}.json").exists()

    old_srt = b"old subtitle"
    Path(f"{prefix}.srt").write_bytes(old_srt)
    Path(f"{prefix}.vtt").mkdir()
    rc = _run(
        tmp_path,
        model_dir,
        "align",
        str(audio),
        "--text",
        "你好世界",
        "--language",
        "Chinese",
        "--format",
        "srt,vtt",
        "--out-prefix",
        str(prefix),
        "--no-subtitle-qc",
        "--json",
    )
    assert rc != 0
    assert Path(f"{prefix}.srt").read_bytes() == old_srt


def test_commit_rolls_back_after_mid_commit_failure(
    monkeypatch, tmp_path: Path
):
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_bytes(b"first-old")
    second.write_bytes(b"second-old")
    staged = [
        align_cmd._stage_text(first, "first-new"),
        align_cmd._stage_text(second, "second-new"),
    ]
    original_replace = align_cmd.os.replace

    def flaky_replace(source, target):
        if Path(target) == second and Path(source).suffix == ".tmp":
            raise OSError("simulated replace failure")
        return original_replace(source, target)

    monkeypatch.setattr(align_cmd.os, "replace", flaky_replace)
    with pytest.raises(OSError):
        align_cmd._commit_staged(staged)
    assert first.read_bytes() == b"first-old"
    assert second.read_bytes() == b"second-old"
    assert not list(tmp_path.glob(".*.tmp"))
    assert not list(tmp_path.glob(".*.bak"))


def test_output_path_collision_exits_2(
    capsys, tmp_path: Path, model_dir: Path
):
    audio = tmp_path / "short.wav"
    write_audio(audio, np.zeros(10000, dtype=np.float32), 1000, "wav")
    prefix = tmp_path / "collision"
    rc = _run(
        tmp_path,
        model_dir,
        "align",
        str(audio),
        "--text",
        "你好世界",
        "--language",
        "Chinese",
        "--format",
        "json",
        "--out-prefix",
        str(prefix),
        "--recipe-out",
        f"{prefix}.json",
        "--no-subtitle-qc",
        "--json",
    )
    assert rc == 2
    captured = capsys.readouterr()
    assert "distinct" in captured.err
    assert captured.out == ""
    assert not Path(f"{prefix}.json").exists()


@pytest.mark.skipif(
    sys.platform != "darwin",
    reason="case-insensitive path folding is macOS-specific",
)
def test_output_path_case_collision_exits_2(
    capsys, tmp_path: Path, model_dir: Path
):
    audio = tmp_path / "short.wav"
    write_audio(audio, np.zeros(10000, dtype=np.float32), 1000, "wav")
    prefix = tmp_path / "casecollision"
    rc = _run(
        tmp_path,
        model_dir,
        "align",
        str(audio),
        "--text",
        "你好世界",
        "--language",
        "Chinese",
        "--format",
        "json",
        "--out-prefix",
        str(prefix),
        "--recipe-out",
        str(tmp_path / "CASECOLLISION.json"),
        "--no-subtitle-qc",
        "--json",
    )
    assert rc == 2
    captured = capsys.readouterr()
    assert "distinct" in captured.err
    assert captured.out == ""


def test_output_path_unicode_normalization_collision_exits_2(
    capsys, tmp_path: Path, model_dir: Path
):
    audio = tmp_path / "short.wav"
    write_audio(audio, np.zeros(10000, dtype=np.float32), 1000, "wav")
    nfc_prefix = "éout"
    nfd_prefix = unicodedata.normalize("NFD", nfc_prefix)
    assert nfd_prefix != nfc_prefix
    rc = _run(
        tmp_path,
        model_dir,
        "align",
        str(audio),
        "--text",
        "你好世界",
        "--language",
        "Chinese",
        "--format",
        "json",
        "--out-prefix",
        str(tmp_path / nfc_prefix),
        "--recipe-out",
        str(tmp_path / f"{nfd_prefix}.json"),
        "--no-subtitle-qc",
        "--json",
    )
    assert rc == 2
    captured = capsys.readouterr()
    assert "distinct" in captured.err
    assert captured.out == ""
    assert not list(tmp_path.glob("*.json"))


def test_commit_preserves_backup_when_restore_fails(
    monkeypatch, tmp_path: Path
):
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_bytes(b"first-old")
    second.write_bytes(b"second-old")
    staged = [
        align_cmd._stage_text(first, "first-new"),
        align_cmd._stage_text(second, "second-new"),
    ]
    original_replace = align_cmd.os.replace

    def flaky_replace(source, target):
        source_path = Path(source)
        target_path = Path(target)
        if target_path == second and source_path.suffix == ".tmp":
            raise OSError("simulated commit failure")
        if target_path == first and source_path.suffix == ".bak":
            raise OSError("simulated restore failure")
        return original_replace(source, target)

    monkeypatch.setattr(align_cmd.os, "replace", flaky_replace)
    with pytest.raises(InputError) as exc_info:
        align_cmd._commit_staged(staged)
    backups = list(tmp_path.glob(".*.bak"))
    assert len(backups) == 1
    assert backups[0].read_bytes() == b"first-old"
    assert str(backups[0]) in str(exc_info.value)
    assert not first.exists()
    assert second.read_bytes() == b"second-old"
    assert not list(tmp_path.glob(".*.tmp"))


def test_long_sparse_coverage_warns_but_succeeds(
    capsys, tmp_path: Path, model_dir: Path
):
    FakeAligner.sparse_fraction = 0.5
    audio = _long_audio(tmp_path / "continuous.wav", duration=330.0, silences=())
    text = "甲乙丙丁戊己庚辛壬癸" * 10
    prefix = tmp_path / "sparse"
    rc = _run(
        tmp_path,
        model_dir,
        "align",
        str(audio),
        "--text",
        text,
        "--language",
        "Chinese",
        "--approximate",
        "--format",
        "json",
        "--out-prefix",
        str(prefix),
        "--no-subtitle-qc",
        "--json",
    )
    assert rc == 0, capsys.readouterr().err
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert 0.3 <= payload["coverage"] <= 0.8
    assert payload["windowing"]["matched_ratio"] == payload["coverage"]
    coverage_warning = next(
        warning
        for warning in payload["warnings"]
        if "alignment coverage" in warning
    )
    assert coverage_warning in payload["windowing"]["warnings"]
    assert coverage_warning in captured.err


def test_long_sparse_coverage_below_minimum_exits_4(
    capsys, tmp_path: Path, model_dir: Path
):
    FakeAligner.sparse_fraction = 0.1
    audio = _long_audio(tmp_path / "continuous.wav", duration=330.0, silences=())
    text = "甲乙丙丁戊己庚辛壬癸" * 10
    prefix = tmp_path / "sparse"
    rc = _run(
        tmp_path,
        model_dir,
        "align",
        str(audio),
        "--text",
        text,
        "--language",
        "Chinese",
        "--approximate",
        "--format",
        "json",
        "--out-prefix",
        str(prefix),
        "--no-subtitle-qc",
        "--json",
    )
    assert rc == 4
    assert not Path(f"{prefix}.json").exists()


def test_srt_and_approximate_are_rejected_before_models(
    capsys, tmp_path: Path, model_dir: Path
):
    audio = tmp_path / "short.wav"
    write_audio(audio, np.zeros(10000, dtype=np.float32), 1000, "wav")
    source = tmp_path / "source.srt"
    source.write_text(
        "1\n00:00:00,000 --> 00:00:05,000\n你好\n", encoding="utf-8"
    )
    rc = _run(
        tmp_path,
        model_dir,
        "align",
        str(audio),
        "--srt",
        str(source),
        "--approximate",
        "--out-prefix",
        str(tmp_path / "bad"),
        "--json",
    )
    assert rc == 2
    assert "hard anchors" in capsys.readouterr().err
    assert FakeAligner.instances == []
    assert FakeAsr.instances == []


def test_missing_aligner_never_loads_asr(
    capsys, tmp_path: Path, model_dir: Path
):
    _add_asr(model_dir)
    shutil.rmtree(model_dir / "Qwen3-ForcedAligner-0.6B")
    audio = _long_audio(tmp_path / "long.wav")
    rc = _run(
        tmp_path,
        model_dir,
        "align",
        str(audio),
        "--text",
        LONG_TEXT,
        "--language",
        "Chinese",
        "--out-prefix",
        str(tmp_path / "bad"),
        "--json",
    )
    assert rc == 3
    assert FakeAsr.instances == []
    assert FakeAsr.calls == []


def test_long_approximate_is_deterministic(
    capsys, tmp_path: Path, model_dir: Path
):
    audio = _long_audio(tmp_path / "long.wav")

    def run_once(prefix: Path):
        rc = _run(
            tmp_path,
            model_dir,
            "align",
            str(audio),
            "--text",
            LONG_TEXT,
            "--language",
            "Chinese",
            "--approximate",
            "--format",
            "json",
            "--out-prefix",
            str(prefix),
            "--no-subtitle-qc",
            "--json",
        )
        assert rc == 0, capsys.readouterr().err
        payload = json.loads(capsys.readouterr().out)
        return payload, Path(f"{prefix}.json").read_bytes()

    first_payload, first_bytes = run_once(tmp_path / "one")
    second_payload, second_bytes = run_once(tmp_path / "two")
    assert first_bytes == second_bytes
    assert first_payload["windowing"]["windows"] == second_payload["windowing"]["windows"]
    assert first_payload["subtitle_qc"] == second_payload["subtitle_qc"]
