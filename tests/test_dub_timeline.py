"""FR-P2-04 §10.1/§10.2: strict SRT and CMX3600-subset parsing."""

from __future__ import annotations

from pathlib import Path

import pytest

from dub_utils import write_srt
from qtts_cli.dub.timeline import (
    MAX_CUE_CHARS,
    MAX_CUES,
    MAX_TIMELINE_BYTES,
    load_timeline,
    parse_edl,
    parse_srt,
)
from qtts_cli.errors import InputError, UsageError


def _srt(*entries: tuple[int, float, float, str], newline: str = "\n", bom: bool = False):
    body = "\n".join(
        f"{index}\n{_stamp(start)} --> {_stamp(end)}\n{text}\n"
        for index, start, end, text in entries
    )
    if newline != "\n":
        body = body.replace("\n", newline)
    return ("﻿" if bom else "") + body


def _stamp(seconds: float) -> str:
    millis = int(round(seconds * 1000))
    hours, rest = divmod(millis, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    secs, ms = divmod(rest, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def test_srt_normal_crlf_bom(tmp_path: Path):
    text = _srt((1, 0.0, 1.25, "Hello"), (3, 2.0, 3.5, "World"), newline="\r\n", bom=True)
    cues = parse_srt(text, tmp_path / "t.srt")
    assert [cue.index for cue in cues] == [1, 3]
    assert cues[0].start_s == 0.0 and cues[0].end_s == 1.25
    assert cues[0].text == "Hello"
    assert cues[1].start_s == 2.0


def test_srt_index_rules(tmp_path: Path):
    path = tmp_path / "t.srt"
    with pytest.raises(InputError):
        parse_srt(_srt((0, 0, 1, "x")), path)
    with pytest.raises(InputError):
        parse_srt(_srt((1, 0, 1, "x"), (1, 1, 2, "y")), path)
    with pytest.raises(InputError):
        parse_srt(_srt((2, 0, 1, "x"), (1, 1, 2, "y")), path)
    with pytest.raises(InputError):
        parse_srt("one\n00:00:00,000 --> 00:00:01,000\nx\n", path)
    with pytest.raises(InputError):
        parse_srt("-1\n00:00:00,000 --> 00:00:01,000\nx\n", path)


def test_srt_time_rules(tmp_path: Path):
    path = tmp_path / "t.srt"
    bad = [
        "1\n00:60:00,000 --> 00:01:00,000\nx\n",
        "1\n00:00:60,000 --> 00:01:00,000\nx\n",
        "1\n00:00:00,00 --> 00:00:01,000\nx\n",
        "1\n00:00:00,500 --> 00:00:00,500\nx\n",
        "1\n00:00:00,500 --> 00:00:00,400\nx\n",
    ]
    for text in bad:
        with pytest.raises(InputError):
            parse_srt(text, path)


def test_srt_overlap_and_descending(tmp_path: Path):
    path = tmp_path / "t.srt"
    with pytest.raises(InputError):
        parse_srt(_srt((1, 0, 2, "a"), (2, 1.5, 3, "b")), path)
    with pytest.raises(InputError):
        parse_srt(_srt((1, 1, 2, "a"), (2, 0.5, 1.5, "b")), path)
    # Touching is allowed (half-open ranges).
    cues = parse_srt(_srt((1, 0, 1, "a"), (2, 1, 2, "b")), path)
    assert len(cues) == 2


def test_srt_empty_and_whitespace_text(tmp_path: Path):
    path = tmp_path / "t.srt"
    with pytest.raises(InputError):
        parse_srt("1\n00:00:00,000 --> 00:00:01,000\n\n", path)
    with pytest.raises(InputError):
        parse_srt("1\n00:00:00,000 --> 00:00:01,000\n   \n", path)
    with pytest.raises(InputError):
        parse_srt("", path)
    with pytest.raises(InputError):
        parse_srt("\n\n", path)


def test_srt_multiline_join_and_speaker(tmp_path: Path):
    path = tmp_path / "t.srt"
    cues = parse_srt(
        "1\n00:00:00,000 --> 00:00:01,000\n  Hello  \n world\n\n"
        "2\n00:00:01,000 --> 00:00:02,000\n alice:   Hi: there\n next\n",
        path,
    )
    assert cues[0].text == "Hello world"
    assert cues[0].speaker is None
    assert cues[1].speaker == "alice"
    assert cues[1].text == "Hi: there next"


def test_srt_whitespace_only_line_is_a_separator(tmp_path: Path):
    path = tmp_path / "t.srt"
    cues = parse_srt(
        "1\n00:00:00,000 --> 00:00:01,000\nHello\n   \t \n"
        "2\n00:00:01,000 --> 00:00:02,000\nWorld\n",
        path,
    )
    assert [cue.text for cue in cues] == ["Hello", "World"]


@pytest.mark.parametrize(
    "text",
    [
        # A dangling next index without a separator is index-like body text.
        "1\n00:00:00,000 --> 00:00:01,000\nHello\n2\n",
        # The next cue is truncated after its index.
        "1\n00:00:00,000 --> 00:00:01,000\nHello\n\n2\n",
        # A cue with only an index has no timing line.
        "1\n",
        # Timing line without a body.
        "1\n00:00:00,000 --> 00:00:01,000\n",
        # A numeric line in the body is ambiguous with a cue index.
        "1\n00:00:00,000 --> 00:00:01,000\nHello\n2024\n",
        # A timing line inside the body means the separator is missing.
        "1\n00:00:00,000 --> 00:00:01,000\nHello\n"
        "00:00:01,000 --> 00:00:02,000\nWorld\n",
    ],
)
def test_srt_strict_indivisible_blocks(tmp_path: Path, text: str):
    with pytest.raises(InputError):
        parse_srt(text, tmp_path / "t.srt")


def test_srt_speaker_without_remainder(tmp_path: Path):
    path = tmp_path / "t.srt"
    with pytest.raises(InputError):
        parse_srt("1\n00:00:00,000 --> 00:00:01,000\nalice:\n", path)
    with pytest.raises(InputError):
        parse_srt("1\n00:00:00,000 --> 00:00:01,000\nalice:   \n", path)
    # A speaker line followed by text is valid.
    cues = parse_srt("1\n00:00:00,000 --> 00:00:01,000\nalice:\nhello\n", path)
    assert cues[0].speaker == "alice"
    assert cues[0].text == "hello"


def test_srt_missing_blank_separator(tmp_path: Path):
    path = tmp_path / "t.srt"
    with pytest.raises(InputError):
        parse_srt(
            "1\n00:00:00,000 --> 00:00:01,000\nHello\n"
            "2\n00:00:01,000 --> 00:00:02,000\nWorld\n",
            path,
        )


def test_srt_limits(tmp_path: Path):
    path = tmp_path / "t.srt"
    with pytest.raises(InputError):
        parse_srt(_srt((1, 0, 1, "x" * (MAX_CUE_CHARS + 1))), path)
    total = []
    chunk = "x" * 190_000
    for index in range(1, 12):
        total.append((index, float(index), float(index) + 0.5, chunk))
    with pytest.raises(InputError):
        parse_srt(_srt(*total), path)
    too_many = [
        (index, float(index), float(index) + 0.5, "x")
        for index in range(1, MAX_CUES + 2)
    ]
    with pytest.raises(InputError):
        parse_srt(_srt(*too_many), path)


def test_load_timeline_suffix_and_bom(tmp_path: Path):
    path = tmp_path / "t.srt"
    write_srt(path, [(1, 0, 1, "Hello")], bom=True)
    parsed = load_timeline(path, fps=25)
    assert parsed.format == "srt"
    assert parsed.cues[0].text == "Hello"
    assert parsed.sha256
    with pytest.raises(UsageError):
        load_timeline(tmp_path / "t.txt", fps=25)
    missing = tmp_path / "missing.srt"
    with pytest.raises(InputError):
        load_timeline(missing, fps=25)


def test_load_timeline_size_limit(tmp_path: Path):
    path = tmp_path / "big.srt"
    path.write_bytes(b"x" * (MAX_TIMELINE_BYTES + 1))
    with pytest.raises(InputError):
        load_timeline(path, fps=25)


def test_load_timeline_rejects_nul(tmp_path: Path):
    path = tmp_path / "nul.srt"
    path.write_bytes(
        b"1\n00:00:00,000 --> 00:00:01,000\nhe\x00llo\n"
    )
    with pytest.raises(InputError):
        load_timeline(path, fps=25)


# --------------------------------------------------------------------------- #
# EDL
# --------------------------------------------------------------------------- #


def test_edl_valid_subset_and_text_attachment():
    text = """TITLE: Episode 1
FCM: NON-DROP FRAME
001  AX       V     C        00:00:00:00 00:00:03:12 00:00:00:00 00:00:03:12
* SPEAKER alice
* TEXT Hello there.
* TEXT How are you?
002  AX       V     C        00:00:03:12 00:00:05:00 00:00:03:12 00:00:05:00
* SPEAKER bob
* SPEAKER alice
* TEXT Last wins.
"""
    cues = parse_edl(text, Path("t.edl"), fps=25)
    assert [cue.index for cue in cues] == [1, 2]
    assert cues[0].start_s == 0.0
    assert cues[0].end_s == 3.48
    assert cues[0].text == "Hello there. How are you?"
    assert cues[0].speaker == "alice"
    assert cues[1].start_s == 3.48
    assert cues[1].speaker == "alice"
    assert cues[1].text == "Last wins."


def test_edl_comments_before_events_ignored():
    text = """* SPEAKER ignored
* TEXT ignored
* arbitrary
001 AX V C 00:00:00:00 00:00:01:00 00:00:00:00 00:00:01:00
* TEXT hi
* unknown ignored
"""
    cues = parse_edl(text, Path("t.edl"), fps=25)
    assert cues[0].speaker is None
    assert cues[0].text == "hi"


def test_edl_drop_frame_and_bad_header():
    base = "001 AX V C 00:00:00:00 00:00:01:00 00:00:00:00 00:00:01:00\n* TEXT hi\n"
    with pytest.raises(InputError):
        parse_edl("FCM: DROP FRAME\n" + base, Path("t.edl"), fps=25)
    with pytest.raises(InputError):
        parse_edl("FCM: SOMETHING\n" + base, Path("t.edl"), fps=25)
    with pytest.raises(InputError):
        parse_edl("TITLE: A\nTITLE: B\n" + base, Path("t.edl"), fps=25)
    with pytest.raises(InputError):
        parse_edl(base + "TITLE: late\n", Path("t.edl"), fps=25)


def test_edl_event_validation():
    path = Path("t.edl")
    with pytest.raises(InputError):
        parse_edl("001 AX A C 00:00:00:00 00:00:01:00 00:00:00:00 00:00:01:00\n* TEXT x\n", path, fps=25)
    with pytest.raises(InputError):
        parse_edl("001 AX V D 00:00:00:00 00:00:01:00 00:00:00:00 00:00:01:00\n* TEXT x\n", path, fps=25)
    with pytest.raises(InputError):
        parse_edl("001 AX V C 00:00:00:00 00:00:02:00 00:00:00:00 00:00:01:00\n* TEXT x\n", path, fps=25)
    with pytest.raises(InputError):
        parse_edl("001 AX V C 00:00:00:00 00:00:01:00 00:00:00:25 00:00:01:25\n* TEXT x\n", path, fps=25)
    with pytest.raises(InputError):
        parse_edl("001 AX V C 00:00:00:00 00:00:01:00 00:00:00:00 00:00:00:00\n* TEXT x\n", path, fps=25)
    with pytest.raises(InputError):
        parse_edl("001 AX V C 00:00:00:00 00:00:01:00 00:00:00:00 00:00:01:00\n", path, fps=25)
    with pytest.raises(InputError):
        parse_edl("garbage line\n", path, fps=25)
    with pytest.raises(InputError):
        parse_edl("", path, fps=25)


def test_edl_numbering_and_overlap():
    path = Path("t.edl")
    one = "001 AX V C 00:00:00:00 00:00:01:00 00:00:00:00 00:00:01:00\n* TEXT one\n"
    with pytest.raises(InputError):
        parse_edl(one + "001 AX V C 00:00:01:00 00:00:02:00 00:00:01:00 00:00:02:00\n* TEXT two\n", path, fps=25)
    with pytest.raises(InputError):
        parse_edl(one + "002 AX V C 00:00:01:00 00:00:02:00 00:00:00:12 00:00:01:12\n* TEXT two\n", path, fps=25)


def test_edl_recout_exclusive_and_fps_validation():
    text = (
        "001 AX V C 00:00:00:00 00:00:01:00 00:00:00:00 00:00:01:00\n* TEXT one\n"
        "002 AX V C 00:00:00:00 00:00:01:00 00:00:01:00 00:00:02:00\n* TEXT two\n"
    )
    cues = parse_edl(text, Path("t.edl"), fps=24)
    assert cues[0].end_s == 1.0
    assert cues[1].start_s == 1.0
    with pytest.raises(InputError):
        parse_edl("001 AX V C 00:00:00:00 00:00:01:00 00:00:00:00 00:00:01:00\n* TEXT x\n", Path("t.edl"), fps=0)
    with pytest.raises(InputError):
        parse_edl("001 AX V C 00:00:00:00 00:00:01:00 00:00:00:00 00:00:01:00\n* TEXT x\n", Path("t.edl"), fps=121)
