"""FR-P1-06: lexicon projection, director markers and subtitle mapping."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from qtts_cli.cli import main
from qtts_cli.commands import align as align_cmd
from qtts_cli.commands import say as say_cmd
from qtts_cli.errors import InputError, UsageError
from qtts_cli.lexicon import (
    Lexicon,
    LexiconEntry,
    _match_key,
    parse_director_markers,
    project_text,
    slice_prepared,
    spoken_items_to_display,
)
from qtts_cli.plan import prepare_text_pieces
from qtts_cli.subtitles import AlignItem


def _lexicon(tmp_path: Path, entries: list[tuple[str, str]]) -> Path:
    path = tmp_path / "lexicon.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "entries": [{"term": term, "spoken": spoken} for term, spoken in entries],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return path


def _direct(entries: list[tuple[str, str]]) -> Lexicon:
    return Lexicon(
        entries=[
            LexiconEntry(term=term, spoken=spoken, key=_match_key(term))
            for term, spoken in entries
        ],
        sha256="test",
    )


def _assert_complete(display: str, spoken: str, mapping) -> None:
    assert "".join(display[b.display_start : b.display_end] for b in mapping) == display
    assert "".join(spoken[b.spoken_start : b.spoken_end] for b in mapping) == spoken
    assert mapping[0].display_start == 0 and mapping[0].spoken_start == 0
    assert mapping[-1].display_end == len(display)
    assert mapping[-1].spoken_end == len(spoken)


def test_longest_match_case_and_fullwidth():
    lexicon = _direct([("A", "诶"), ("AI", "人工智能")])
    prepared = project_text("ＡＩ和AI", language="Chinese", lexicon=lexicon)
    assert prepared.display_text == "ＡＩ和AI"
    assert prepared.spoken_text == "人工智能和人工智能"
    terms = [b.term for b in prepared.mapping if b.term]
    assert terms == ["AI", "AI"]
    _assert_complete(prepared.display_text, prepared.spoken_text, prepared.mapping)


def test_duplicate_normalized_term_is_rejected(tmp_path: Path):
    path = _lexicon(tmp_path, [("AI", "人工智能")])
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["entries"].append({"term": "ａｉ", "spoken": "人工智能"})
    path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(InputError, match="duplicate lexicon term"):
        Lexicon.load(path)


def test_identity_when_no_replacement(tmp_path: Path):
    lexicon = Lexicon.load(_lexicon(tmp_path, [("AI", "人工智能")]))
    prepared = project_text("普通文本。", language="Chinese", lexicon=lexicon)
    assert prepared.display_text == prepared.spoken_text == "普通文本。"
    _assert_complete(prepared.display_text, prepared.spoken_text, prepared.mapping)


def test_length_change_mapping_and_slices():
    lexicon = _direct([("AI", "人工智能"), ("EE", "一")])
    prepared = project_text("AI和EE。", language="Chinese", lexicon=lexicon)
    assert prepared.display_text == "AI和EE。"
    assert prepared.spoken_text == "人工智能和一。"
    _assert_complete(prepared.display_text, prepared.spoken_text, prepared.mapping)
    first = slice_prepared(prepared, 0, 2)
    assert first.display_text == "AI"
    assert first.spoken_text == "人工智能"
    with pytest.raises(InputError, match="split inside lexicon term"):
        slice_prepared(prepared, 1, 3)


def test_cross_sentence_and_cross_quote_terms_are_rejected():
    with pytest.raises(InputError, match="sentence boundary"):
        project_text("好。世", language="Chinese", lexicon=_direct([("好。世", "x")]))
    with pytest.raises(InputError, match="quote boundary"):
        project_text('a"b', language="Chinese", lexicon=_direct([('a"b', "x")]))


def test_director_markers_pause_semantics():
    clean, markers = parse_director_markers(
        "[[pause:500]]你好。[[pause:0]][[pause:200]]世界。[[pause:300]]"
    )
    assert clean == "你好。世界。"
    assert [m["ms"] for m in markers] == [500, 0, 200, 300]

    display, pieces, warnings, leading, trailing = prepare_text_pieces(
        "[[pause:500]]你好。[[pause:0]][[pause:200]]世界。[[pause:300]]",
        language="Chinese",
        rate=12,
        max_new_tokens=2048,
    )
    assert display == "你好。世界。"
    assert not warnings
    assert leading == 500
    assert trailing == 300
    assert pieces[0]["pause_after_ms"] == 200
    assert pieces[-1]["pause_after_ms"] == 0
    assert "[[pause" not in "".join(p["display_text"] for p in pieces)


@pytest.mark.parametrize(
    "marker",
    ["[[speed:1.2]]", "[[rate:fast]]", "[[pitch:+2]]", "[[pause:-1]]", "[[bogus]]"],
)
def test_unsupported_or_invalid_markers_exit_2(marker: str):
    with pytest.raises(UsageError):
        parse_director_markers(f"你好。{marker}")


@pytest.mark.parametrize(
    "text",
    [
        "你好 [[pause:500",
        "你好 pause:500]]",
        "[[a[[pause:500]]",
    ],
)
def test_unclosed_director_markers_exit_2(text: str):
    with pytest.raises(UsageError):
        parse_director_markers(text)


class FakeEngine:
    loads = 0

    def __init__(self, cfg, model):
        self.cfg = cfg
        self.model_info = model
        self.device = "cpu"
        self.dtype_name = "fp32"
        self.load_ms = 0.0
        self.model = None

    def load(self):
        FakeEngine.loads += 1
        self.model = object()
        return self

    def generate(self, **kwargs):
        text = kwargs["text"]
        samples = max(1, len(text)) * 2400
        return np.zeros(samples, dtype=np.float32), 24000, {"generate_ms": 1.0}

    def build_clone_prompt_items(self, *args, **kwargs):
        return []


class FakeAligner:
    def __init__(self, cfg, model):
        self.load_ms = 0.0

    def align(self, audio, sample_rate, text, language):
        items = [
            AlignItem(ch, i * 0.1, (i + 1) * 0.1)
            for i, ch in enumerate(text)
            if not ch.isspace()
        ]
        return items, {"align_ms": 1.0}


def test_say_lexicon_subtitles_use_display_text(
    monkeypatch, capsys, tmp_path: Path, model_dir: Path
):
    lexicon = _lexicon(tmp_path, [("AI", "人工智能")])
    monkeypatch.setattr(say_cmd, "TtsEngine", FakeEngine)
    monkeypatch.setattr(say_cmd, "AlignerEngine", FakeAligner)
    out = tmp_path / "out.wav"
    rc = main(
        [
            "say",
            "AI发布。",
            "--voice",
            "builtin:Vivian",
            "--lexicon",
            str(lexicon),
            "--subtitle",
            "json",
            "--no-subtitle-qc",
            "--out",
            str(out),
            "--model-dir",
            str(model_dir),
            "--data-dir",
            str(tmp_path / "data"),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--json",
        ]
    )
    assert rc == 0, capsys.readouterr().err
    capsys.readouterr()
    subtitle = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
    joined = "".join(cue["text"] for cue in subtitle["cues"])
    assert joined == "AI发布。"
    assert "人工智能" not in joined
    recipe = json.loads((tmp_path / "out.recipe.json").read_text(encoding="utf-8"))
    assert recipe["lexicon_sha256"]


def test_marker_pauses_reach_audio_but_not_subtitles(
    monkeypatch, capsys, tmp_path: Path, model_dir: Path
):
    import soundfile as sf

    monkeypatch.setattr(say_cmd, "TtsEngine", FakeEngine)
    monkeypatch.setattr(say_cmd, "AlignerEngine", FakeAligner)
    out = tmp_path / "pauses.wav"
    rc = main(
        [
            "say",
            "[[pause:500]]你好。[[pause:250]]",
            "--voice",
            "builtin:Vivian",
            "--out",
            str(out),
            "--no-cache",
            "--model-dir",
            str(model_dir),
            "--data-dir",
            str(tmp_path / "data"),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--json",
        ]
    )
    assert rc == 0, capsys.readouterr().err
    capsys.readouterr()
    info = sf.info(str(out))
    assert info.duration == pytest.approx(1.05, abs=0.01)
    subtitle = tmp_path / "pauses.json"
    assert not subtitle.exists()

    dry = main(
        [
            "say",
            "[[pause:500]]你好。[[pause:250]]",
            "--voice",
            "builtin:Vivian",
            "--dry-run",
            "--model-dir",
            str(model_dir),
            "--data-dir",
            str(tmp_path / "data"),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--json",
        ]
    )
    assert dry == 0
    plan = json.loads(capsys.readouterr().out)["plan"]
    assert plan["leading_pause_ms"] == 500
    assert plan["trailing_pause_ms"] == 250
    assert all("[[pause" not in seg["display_text"] for seg in plan["segments"])


def test_align_lexicon_maps_external_audio_back_to_display(
    monkeypatch, capsys, tmp_path: Path, model_dir: Path
):
    import soundfile as sf

    lexicon = _lexicon(tmp_path, [("AI", "人工智能")])
    audio = tmp_path / "speech.wav"
    sf.write(audio, np.zeros(16800, dtype=np.float32), 24000)
    monkeypatch.setattr(align_cmd, "AlignerEngine", FakeAligner)
    prefix = tmp_path / "aligned"
    rc = main(
        [
            "align",
            str(audio),
            "--text",
            "AI发布。",
            "--language",
            "Chinese",
            "--lexicon",
            str(lexicon),
            "--format",
            "json",
            "--no-subtitle-qc",
            "--out-prefix",
            str(prefix),
            "--model-dir",
            str(model_dir),
            "--data-dir",
            str(tmp_path / "data"),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--json",
        ]
    )
    assert rc == 0, capsys.readouterr().err
    capsys.readouterr()
    payload = json.loads(Path(f"{prefix}.json").read_text(encoding="utf-8"))
    assert "".join(cue["text"] for cue in payload["cues"]) == "AI发布。"
    assert "人工智能" not in "".join(cue["text"] for cue in payload["cues"])


def test_align_srt_with_lexicon_measures_coverage_on_spoken_text(
    monkeypatch, capsys, tmp_path: Path, model_dir: Path
):
    import soundfile as sf

    lexicon = _lexicon(tmp_path, [("AI", "人工智能")])
    audio = tmp_path / "speech.wav"
    sf.write(audio, np.zeros(16800, dtype=np.float32), 24000)
    source_srt = tmp_path / "source.srt"
    source_srt.write_text(
        "1\n00:00:00,000 --> 00:00:01,000\nAI。\n\n"
        "2\n00:00:01,000 --> 00:00:02,000\n发布。\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(align_cmd, "AlignerEngine", FakeAligner)
    prefix = tmp_path / "srt_aligned"
    rc = main(
        [
            "align",
            str(audio),
            "--srt",
            str(source_srt),
            "--language",
            "Chinese",
            "--lexicon",
            str(lexicon),
            "--format",
            "json",
            "--no-subtitle-qc",
            "--out-prefix",
            str(prefix),
            "--model-dir",
            str(model_dir),
            "--data-dir",
            str(tmp_path / "data"),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--json",
        ]
    )
    assert rc == 0, capsys.readouterr().err
    result = json.loads(capsys.readouterr().out)
    assert result["coverage"] == pytest.approx(1.0)
    payload = json.loads(Path(f"{prefix}.json").read_text(encoding="utf-8"))
    assert "".join(cue["text"] for cue in payload["cues"]) == "AI。发布。"


def test_spoken_items_project_back_to_display():
    # 2 display chars -> 4 spoken chars, and 2 spoken chars -> 1 display char.
    from qtts_cli.lexicon import MapBlock

    mapping = [MapBlock(0, 2, 0, 4, "AI"), MapBlock(2, 4, 4, 5, "EE")]
    items = [
        AlignItem("人", 0.0, 0.1),
        AlignItem("工", 0.1, 0.2),
        AlignItem("智", 0.2, 0.3),
        AlignItem("能", 0.3, 0.4),
        AlignItem("一", 0.4, 0.5),
    ]
    projected = spoken_items_to_display("AIEE", "人工智能一", mapping, items)
    assert "".join(item.text for item in projected) == "AIEE"
    assert projected[0].start == pytest.approx(0.0)
    assert projected[-1].end == pytest.approx(0.5)


def test_spoken_items_skip_aligner_punctuation():
    # The aligner drops punctuation; mapping must still advance sequentially.
    from qtts_cli.lexicon import MapBlock

    text = "你好，世界。"
    mapping = [MapBlock(i, i + 1, i, i + 1) for i in range(len(text))]
    items = [
        AlignItem("你", 0.0, 0.1),
        AlignItem("好", 0.1, 0.2),
        AlignItem("世", 0.2, 0.3),
        AlignItem("界", 0.3, 0.4),
    ]
    projected = spoken_items_to_display(text, text, mapping, items)
    assert [item.text for item in projected] == ["你", "好", "世", "界"]
    assert projected[-1].end == pytest.approx(0.4)


def test_spoken_items_do_not_skip_content_characters():
    # The aligner omitted the second "好"; the trailing "好" item must not be
    # silently matched to a later duplicate (that would hide a bad alignment).
    from qtts_cli.lexicon import MapBlock

    text = "你好你好"
    mapping = [MapBlock(i, i + 1, i, i + 1) for i in range(len(text))]
    items = [
        AlignItem("你", 0.0, 0.1),
        AlignItem("好", 0.1, 0.2),
        AlignItem("好", 0.2, 0.3),
    ]
    projected = spoken_items_to_display(text, text, mapping, items)
    assert [item.text for item in projected] == ["你", "好"]
