"""FR-P2-04 §10.9–§10.11: CLI, encoding, mux, publication and locking."""

from __future__ import annotations

import errno
import fcntl
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from dub_utils import FakeRunner, manifest_of, read_audio, run_build, write_srt
from qtts_cli import cli
from qtts_cli.dub import encode as dub_encode
from qtts_cli.dub import manifest as dub_manifest
from qtts_cli.errors import InputError, ModelError, UsageError

HAVE_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
ffmpeg_only = pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg/ffprobe not installed")


def _timeline(tmp_path: Path, *texts: str, seconds: float = 1.0) -> Path:
    entries = [
        (index, (index - 1) * seconds, index * seconds, text)
        for index, text in enumerate(texts, 1)
    ]
    return write_srt(tmp_path / "t.srt", entries)


def _run(timeline: Path, model_dir: Path, *extra: str, runner=None):
    return run_build(
        timeline,
        model_dir,
        runner or FakeRunner(seconds=0.5),
        "--out",
        str(timeline.parent / "out.wav"),
        "--voice",
        "builtin:Vivian",
        "--no-loudness",
        *extra,
    )


def test_output_defaults_and_suffix(tmp_path: Path, model_dir: Path, monkeypatch):
    timeline = _timeline(tmp_path, "hello")
    monkeypatch.chdir(tmp_path)
    payload = run_build(
        timeline,
        model_dir,
        FakeRunner(),
        "--voice",
        "builtin:Vivian",
        "--no-loudness",
    )
    assert Path(payload["output"]["path"]) == tmp_path / "dub.wav"
    assert (tmp_path / "dub.wav").is_file()
    assert (tmp_path / "dub.dub.json").is_file()

    (tmp_path / "dub.wav").unlink()
    (tmp_path / "dub.dub.json").unlink()
    payload = run_build(
        timeline,
        model_dir,
        FakeRunner(),
        "--format",
        "flac",
        "--voice",
        "builtin:Vivian",
        "--no-loudness",
    )
    assert Path(payload["output"]["path"]) == tmp_path / "dub.flac"
    assert (tmp_path / "dub.flac").is_file()

    no_suffix = tmp_path / "other"
    payload = run_build(
        timeline,
        model_dir,
        FakeRunner(),
        "--out",
        str(no_suffix),
        "--voice",
        "builtin:Vivian",
        "--no-loudness",
    )
    assert Path(payload["output"]["path"]) == tmp_path / "other.wav"

    payload = run_build(
        timeline,
        model_dir,
        FakeRunner(),
        "--out",
        str(tmp_path / "out.flac"),
        "--voice",
        "builtin:Vivian",
        "--no-loudness",
    )
    data, sr = read_audio(Path(payload["output"]["path"]))
    assert sr == 24000 and data.size


@pytest.mark.parametrize(
    "extra",
    [
        ("--out", "-"),
        ("--out", "out.ogg"),
        ("--format", "flac", "--out", "out.wav"),
        ("--sample-rate", "8000"),
        ("--fps", "0"),
        ("--fps", "121"),
        ("--min-speed", "1.2", "--max-speed", "1.1"),
        ("--min-speed", "0.4"),
        ("--max-speed", "2.1"),
        ("--bed-db", "-6"),
        ("--bed-db", "1"),
        ("--lufs", "-50"),
        ("--true-peak", "1"),
        ("--mux"),
    ],
)
def test_cli_usage_errors(tmp_path: Path, model_dir: Path, extra):
    timeline = _timeline(tmp_path, "hello")
    with pytest.raises(UsageError):
        run_build(
            timeline,
            model_dir,
            FakeRunner(),
            "--out",
            str(tmp_path / "out.wav"),
            "--voice",
            "builtin:Vivian",
            *extra,
        )


def test_fps_accepted_and_ignored_for_srt(tmp_path: Path, model_dir: Path):
    timeline = _timeline(tmp_path, "hello")
    payload = _run(timeline, model_dir, "--fps", "50")
    assert Path(payload["output"]["path"]).is_file()


def test_bad_timeline_suffix_is_usage_error(tmp_path: Path, model_dir: Path):
    path = tmp_path / "t.txt"
    path.write_text("hello", encoding="utf-8")
    with pytest.raises(UsageError):
        run_build(path, model_dir, FakeRunner(), "--out", str(tmp_path / "out.wav"))


def test_keep_work_and_default_removal(tmp_path: Path, model_dir: Path):
    timeline = _timeline(tmp_path, "hello")
    _run(timeline, model_dir, "--keep-work")
    assert (tmp_path / "out.dub-work").is_dir()
    (tmp_path / "out.wav").unlink()
    (tmp_path / "out.dub.json").unlink()
    _run(timeline, model_dir)
    assert not (tmp_path / "out.dub-work").exists()


def test_existing_targets_and_force(tmp_path: Path, model_dir: Path):
    timeline = _timeline(tmp_path, "hello")
    _run(timeline, model_dir)
    runner = FakeRunner()
    with pytest.raises(InputError):
        _run(timeline, model_dir, runner=runner)
    assert runner.calls == []
    _run(timeline, model_dir, "--force", runner=runner)
    assert runner.calls


def test_path_collisions_and_containment(tmp_path: Path, model_dir: Path):
    timeline = _timeline(tmp_path, "hello")
    cases = [
        ["--out", str(tmp_path / "same.wav"), "--stems-out", str(tmp_path / "same.wav")],
        ["--stems-out", str(tmp_path / "out.dub-work" / "stems")],
        ["--stems-out", str(tmp_path)],
    ]
    for extra in cases:
        with pytest.raises(InputError):
            run_build(
                timeline,
                model_dir,
                FakeRunner(),
                "--out",
                str(tmp_path / "out.wav"),
                "--voice",
                "builtin:Vivian",
                "--no-loudness",
                *extra,
            )
    work = tmp_path / "out.dub-work"
    work.mkdir()
    inside = work / "t.srt"
    write_srt(inside, [(1, 0, 1, "hello")])
    with pytest.raises(InputError):
        run_build(
            inside,
            model_dir,
            FakeRunner(),
            "--out",
            str(tmp_path / "out.wav"),
            "--voice",
            "builtin:Vivian",
            "--no-loudness",
        )


def test_collision_key_nfc_and_case(tmp_path: Path, model_dir: Path):
    timeline = _timeline(tmp_path, "hello")
    with pytest.raises(InputError):
        run_build(
            timeline,
            model_dir,
            FakeRunner(),
            "--out",
            str(tmp_path / "Case.wav"),
            "--voice",
            "builtin:Vivian",
            "--no-loudness",
            "--stems-out",
            str(tmp_path / "case.wav"),
        )
    with pytest.raises(InputError):
        run_build(
            timeline,
            model_dir,
            FakeRunner(),
            "--out",
            str(tmp_path / "é.wav"),
            "--voice",
            "builtin:Vivian",
            "--no-loudness",
            "--stems-out",
            str(tmp_path / "é.wav"),
        )


def test_bed_range_precedes_missing_video(tmp_path: Path, model_dir: Path):
    timeline = _timeline(tmp_path, "hello")
    with pytest.raises(UsageError):
        run_build(
            timeline,
            model_dir,
            FakeRunner(),
            "--out",
            str(tmp_path / "out.wav"),
            "--bed-db",
            "1",
            "--video",
            str(tmp_path / "missing.mp4"),
        )


def test_anonymous_speaker_collision(tmp_path: Path, model_dir: Path):
    timeline = _timeline(tmp_path, "__anonymous__: hi")
    with pytest.raises(InputError):
        run_build(
            timeline,
            model_dir,
            FakeRunner(),
            "--out",
            str(tmp_path / "out.wav"),
            "--voice",
            "builtin:Vivian",
            "--no-loudness",
            "--stems-out",
            str(tmp_path / "stems"),
        )


def test_anonymous_speaker_is_reserved_without_stems(tmp_path: Path, model_dir: Path):
    timeline = _timeline(tmp_path, "__anonymous__: hi")
    runner = FakeRunner()
    with pytest.raises(InputError):
        run_build(
            timeline,
            model_dir,
            runner,
            "--out",
            str(tmp_path / "out.wav"),
            "--voice",
            "builtin:Vivian",
            "--no-loudness",
        )
    assert runner.calls == []
    assert not (tmp_path / "out.wav").exists()


def test_existing_stems_dir_is_allowed_but_stem_files_are_not(
    tmp_path: Path, model_dir: Path
):
    timeline = _timeline(tmp_path, "plain", "alice: hi")
    stems = tmp_path / "stems"
    stems.mkdir()
    _run(timeline, model_dir, "--stems-out", str(stems))
    assert (stems / "alice.wav").is_file()

    (tmp_path / "out.wav").unlink()
    (tmp_path / "out.dub.json").unlink()
    shutil.rmtree(stems)
    stems.mkdir()
    (stems / "alice.wav").write_bytes(b"stale")
    runner = FakeRunner()
    with pytest.raises(InputError):
        _run(timeline, model_dir, "--stems-out", str(stems), runner=runner)
    assert runner.calls == []


def test_parent_preflight_before_runner(tmp_path: Path, model_dir: Path):
    timeline = _timeline(tmp_path, "hello")
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")
    runner = FakeRunner()
    with pytest.raises(InputError):
        run_build(
            timeline,
            model_dir,
            runner,
            "--out",
            str(blocker / "out.wav"),
            "--voice",
            "builtin:Vivian",
        )
    assert runner.calls == []


def test_second_live_build_lock(tmp_path: Path, model_dir: Path):
    timeline = _timeline(tmp_path, "hello")
    work = tmp_path / "out.dub-work"
    work.mkdir()
    lock_path = work / ".lock"
    descriptor = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(InputError):
            run_build(
                timeline,
                model_dir,
                FakeRunner(),
                "--out",
                str(tmp_path / "out.wav"),
                "--voice",
                "builtin:Vivian",
                "--no-loudness",
            )
    finally:
        os.close(descriptor)


def test_non_contention_lock_failure_is_model_error(tmp_path: Path, model_dir: Path, monkeypatch):
    timeline = _timeline(tmp_path, "hello")

    def boom(*args, **kwargs):
        raise OSError(errno.EIO, "lock broke")

    monkeypatch.setattr(fcntl, "flock", boom)
    with pytest.raises(ModelError):
        run_build(
            timeline,
            model_dir,
            FakeRunner(),
            "--out",
            str(tmp_path / "out.wav"),
            "--voice",
            "builtin:Vivian",
        )


@ffmpeg_only
def test_mux_with_generated_mp4(tmp_path: Path, model_dir: Path):
    timeline = _timeline(tmp_path, "hello")
    video = tmp_path / "video.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=black:s=32x32:d=1",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
         "-c:v", "mpeg4", "-c:a", "aac", "-shortest", str(video)],
        check=True,
    )
    payload = _run(timeline, model_dir, "--video", str(video), "--mux")
    assert Path(payload["output"]["path"]).is_file()
    assert Path(payload["mux"]["path"]).is_file()
    manifest = manifest_of(tmp_path / "out.wav")
    assert manifest["mux"]["path"] == str(tmp_path / "out.mp4")
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type", "-of", "json",
         str(tmp_path / "out.mp4")],
        capture_output=True, text=True, check=True,
    )
    assert "video" in probe.stdout


@ffmpeg_only
def test_mux_requires_video_stream(tmp_path: Path, model_dir: Path):
    timeline = _timeline(tmp_path, "hello")
    audio = tmp_path / "audio.wav"
    sf.write(str(audio), np.zeros(24000, dtype=np.float32), 24000, subtype="PCM_16")
    with pytest.raises(InputError):
        _run(timeline, model_dir, "--video", str(audio), "--mux")


@ffmpeg_only
def test_duration_sanity_failure(tmp_path: Path, model_dir: Path, monkeypatch):
    from qtts_cli.audiobook import encode as audiobook_encode

    timeline = _timeline(tmp_path, "hello")
    monkeypatch.setattr(audiobook_encode, "encoded_duration_s", lambda _: 99.0)
    with pytest.raises(ModelError):
        _run(timeline, model_dir, "--format", "m4a", "--out", str(tmp_path / "out.m4a"))
    assert not (tmp_path / "out.m4a").exists()


@ffmpeg_only
def test_lossy_formats(tmp_path: Path, model_dir: Path):
    timeline = _timeline(tmp_path, "hello")
    for fmt in ("m4a", "mp3"):
        out = tmp_path / f"out_{fmt}.{fmt}"
        payload = run_build(
            timeline,
            model_dir,
            FakeRunner(),
            "--out",
            str(out),
            "--voice",
            "builtin:Vivian",
            "--format",
            fmt,
        )
        assert out.is_file() and out.stat().st_size > 0
        assert payload["output"]["format"] == fmt
        assert payload["loudness"]["encoded_true_peak_dbtp"] is not None


@ffmpeg_only
def test_lossy_bitrates_and_faststart(tmp_path: Path, model_dir: Path, monkeypatch):
    from qtts_cli.audiobook import encode as audiobook_encode

    timeline = _timeline(tmp_path, "hello")
    commands: list[list[str]] = []
    real_run = audiobook_encode.subprocess.run

    def recording(command, *args, **kwargs):
        if isinstance(command, list):
            commands.append([str(part) for part in command])
        return real_run(command, *args, **kwargs)

    monkeypatch.setattr(audiobook_encode.subprocess, "run", recording)
    for fmt in ("m4a", "mp3"):
        commands.clear()
        out = tmp_path / f"rate_{fmt}.{fmt}"
        run_build(
            timeline,
            model_dir,
            FakeRunner(),
            "--out",
            str(out),
            "--voice",
            "builtin:Vivian",
            "--format",
            fmt,
        )
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=codec_name", "-of", "json", str(out)],
            capture_output=True, text=True, check=True,
        )
        stream = json.loads(probe.stdout)["streams"][0]
        encode_commands = [cmd for cmd in commands if "-b:a" in cmd]
        assert encode_commands
        command = encode_commands[-1]
        assert command[command.index("-b:a") + 1] == ("128k" if fmt == "m4a" else "192k")
        if fmt == "m4a":
            assert stream["codec_name"] == "aac"
            assert "+faststart" in command
            raw = out.read_bytes()
            assert raw.index(b"moov") < raw.index(b"mdat")
        else:
            assert stream["codec_name"] == "mp3"


def test_fixed_replace_order_and_atomic_temps(tmp_path: Path, model_dir: Path, monkeypatch):
    timeline = _timeline(tmp_path, "one", "alice: two")
    calls: list[tuple[Path, Path]] = []
    real_replace = dub_manifest.os.replace

    def recording(src, dst):
        calls.append((Path(src), Path(dst)))
        return real_replace(src, dst)

    monkeypatch.setattr(dub_manifest.os, "replace", recording)
    _run(timeline, model_dir, "--stems-out", str(tmp_path / "stems"))
    published = [
        (source, destination)
        for source, destination in calls
        if not any(part.endswith(".dub-work") for part in destination.parts)
    ]
    destinations = [destination.name for _, destination in published]
    assert destinations[0] == "out.wav"
    assert destinations[-1] == "out.dub.json"
    assert destinations[-2] == "timeline.json"
    assert set(destinations) == {
        "out.wav",
        "__anonymous__.wav",
        "alice.wav",
        "timeline.json",
        "out.dub.json",
    }
    for source, destination in published:
        assert source != destination
        assert source.name.endswith(".tmp")


def test_precommit_json_cancellation_prints_interrupted(tmp_path: Path, model_dir: Path, monkeypatch):
    import io
    from contextlib import redirect_stderr, redirect_stdout

    import qtts_cli.commands.dub as dub_cmd

    def stop(cfg, args):
        raise KeyboardInterrupt

    # The parser binds ``commands.dub.run`` at build time; that function calls
    # ``dub_mod.run`` at call time, which is the surface we can patch.
    monkeypatch.setattr(dub_cmd.dub_mod, "run", stop)
    timeline = _timeline(tmp_path, "hello")
    argv = ["dub", "build", str(timeline), "--model-dir", str(model_dir), "--json"]
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        rc = cli.main(argv)
    assert rc == 130
    assert out.getvalue() == ""
    assert err.getvalue() == "interrupted\n"


def _cancel_at_manifest(monkeypatch, manifest: Path) -> None:
    real_replace = dub_manifest.os.replace
    state = {"sent": False}

    def replace(src, dst):
        if Path(dst) == manifest and not state["sent"]:
            state["sent"] = True
            os.kill(os.getpid(), signal.SIGTERM)
            time.sleep(0.05)
        return real_replace(src, dst)

    monkeypatch.setattr(dub_manifest.os, "replace", replace)


def test_cancellation_before_publish_is_keyboard_interrupt(tmp_path: Path, model_dir: Path):
    timeline = _timeline(tmp_path, "one", "two")

    def stop(runner, call):
        if call == 2:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        _run(timeline, model_dir, runner=FakeRunner(on_call=stop))
    assert not (tmp_path / "out.wav").exists()
    assert (tmp_path / "out.dub-work" / "cues").exists()


def test_deferred_cancellation_after_commit(tmp_path: Path, model_dir: Path, monkeypatch, capsys):
    timeline = _timeline(tmp_path, "hello")
    _cancel_at_manifest(monkeypatch, tmp_path / "out.dub.json")
    payload = _run(timeline, model_dir, "--keep-work")
    captured = capsys.readouterr()
    assert payload == {"raw_stdout": True, "exit_code": 130}
    assert captured.err == "interrupted after publish\n"
    assert (tmp_path / "out.wav").is_file()
    assert (tmp_path / "out.dub.json").is_file()


def test_step1_replace_failure_keeps_old_outputs(tmp_path: Path, model_dir: Path, monkeypatch):
    timeline = _timeline(tmp_path, "hello")
    out = tmp_path / "out.wav"
    manifest = tmp_path / "out.dub.json"
    out.write_bytes(b"old-final")
    manifest.write_text("old-manifest", encoding="utf-8")
    real_replace = dub_manifest.os.replace
    state = {"count": 0}

    def replace(src, dst):
        state["count"] += 1
        if Path(dst) == out:
            raise OSError(errno.ENOSPC, "disk full")
        return real_replace(src, dst)

    monkeypatch.setattr(dub_manifest.os, "replace", replace)
    with pytest.raises(ModelError):
        _run(timeline, model_dir, "--force")
    assert state["count"] >= 1
    assert out.read_bytes() == b"old-final"
    assert manifest.read_text(encoding="utf-8") == "old-manifest"


def test_later_replace_failure_keeps_replaced_files(tmp_path: Path, model_dir: Path, monkeypatch):
    timeline = _timeline(tmp_path, "one", "alice: two")
    out = tmp_path / "out.wav"
    manifest = tmp_path / "out.dub.json"
    out.write_bytes(b"old-final")
    manifest.write_text("old-manifest", encoding="utf-8")
    real_replace = dub_manifest.os.replace
    state = {"count": 0}

    stems = tmp_path / "stems"

    def replace(src, dst):
        state["count"] += 1
        if Path(dst).parent == stems:
            raise OSError(errno.ENOSPC, "disk full")
        return real_replace(src, dst)

    monkeypatch.setattr(dub_manifest.os, "replace", replace)
    with pytest.raises(ModelError):
        _run(
            timeline,
            model_dir,
            "--force",
            "--stems-out",
            str(stems),
        )
    assert state["count"] >= 1
    assert out.read_bytes() != b"old-final"
    assert manifest.read_text(encoding="utf-8") == "old-manifest"


def test_later_failure_takes_precedence_over_recorded_signal(
    tmp_path: Path, model_dir: Path, monkeypatch, capsys
):
    timeline = _timeline(tmp_path, "one", "alice: two")
    out = tmp_path / "out.wav"
    stems = tmp_path / "stems"
    real_replace = dub_manifest.os.replace
    state = {"sent": False}

    def replace(src, dst):
        if Path(dst) == out and not state["sent"]:
            state["sent"] = True
            os.kill(os.getpid(), signal.SIGTERM)
            time.sleep(0.05)
        if Path(dst).parent == stems:
            raise OSError(errno.ENOSPC, "disk full")
        return real_replace(src, dst)

    monkeypatch.setattr(dub_manifest.os, "replace", replace)
    with pytest.raises(ModelError):
        _run(timeline, model_dir, "--stems-out", str(stems))
    captured = capsys.readouterr()
    assert "cancellation was recorded but publication did not complete" in captured.err
    assert out.is_file()
    assert not (tmp_path / "out.dub.json").exists()


def test_first_stretch_without_ffmpeg_keeps_cache(tmp_path: Path, model_dir: Path, monkeypatch):
    from qtts_cli.dub import synth

    timeline = _timeline(tmp_path, "hello")
    real_which = shutil.which
    monkeypatch.setattr(
        synth.shutil, "which", lambda name: None if name == "ffmpeg" else real_which(name)
    )
    runner = FakeRunner(seconds=0.5)
    with pytest.raises(ModelError):
        _run(timeline, model_dir, "--min-speed", "0.5", runner=runner)
    assert not (tmp_path / "out.wav").exists()
    assert (tmp_path / "out.dub-work" / "cues" / "0001.wav").is_file()


def test_lossy_preflight_before_synthesis(tmp_path: Path, model_dir: Path, monkeypatch):
    timeline = _timeline(tmp_path, "hello")
    monkeypatch.setattr(
        dub_encode,
        "require_ffmpeg",
        lambda fmt: (_ for _ in ()).throw(ModelError("no encoder")),
    )
    runner = FakeRunner()
    with pytest.raises(ModelError):
        _run(
            timeline,
            model_dir,
            "--format",
            "m4a",
            "--out",
            str(tmp_path / "out.m4a"),
            runner=runner,
        )
    assert runner.calls == []


def test_json_error_object(tmp_path: Path):
    import io
    from contextlib import redirect_stderr, redirect_stdout

    bad = tmp_path / "t.txt"
    bad.write_text("x", encoding="utf-8")
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        rc = cli.main(["dub", "build", str(bad), "--json"])
    assert rc == 2
    assert out.getvalue() == ""
    payload = json.loads(err.getvalue())
    assert payload["code"] == 2


def test_payloads_are_json_serializable(tmp_path: Path, model_dir: Path, capsys):
    timeline = _timeline(tmp_path, "hello")
    payload = _run(timeline, model_dir)
    json.dumps(payload)

    cli.main(
        [
            "dub",
            "build",
            str(timeline),
            "--model-dir",
            str(model_dir),
            "--out",
            str(tmp_path / "dry.wav"),
            "--voice",
            "builtin:Vivian",
            "--dry-run",
            "--json",
        ]
    )
    captured = capsys.readouterr()
    document = json.loads(captured.out)
    assert document["dry_run"] is True
    assert document["cues"][0]["fingerprint"] is None


def test_quiet_suppresses_warnings(tmp_path: Path, model_dir: Path, capsys):
    timeline = _timeline(tmp_path, "hello")
    payload = run_build(
        timeline,
        model_dir,
        FakeRunner(seconds=2.0),
        "--out",
        str(tmp_path / "out.wav"),
        "--voice",
        "builtin:Vivian",
        "--no-loudness",
        "--quiet",
    )
    captured = capsys.readouterr()
    assert payload["warnings"]
    assert "warning:" not in captured.err


def test_format_result_dry_and_normal():
    from qtts_cli.commands.dub import format_result

    dry = format_result(
        {
            "dry_run": True,
            "output": {"path": "out.wav", "format": "wav"},
            "manifest": "out.dub.json",
            "cues": [
                {
                    "index": 1,
                    "start_s": 0.0,
                    "end_s": 1.0,
                    "speaker": None,
                    "voice": "builtin:Vivian",
                }
            ],
            "speakers": [],
            "warnings": ["careful"],
        }
    )
    assert "dry run: out.wav" in dry and "warning: careful" in dry

    normal = format_result(
        {
            "output": {"path": "out.wav", "format": "wav", "duration_s": 1.0, "size": 10},
            "segments": {"total": 1, "synthesized": 1, "reused": 0},
            "manifest": "out.dub.json",
            "mux": {"path": "out.mp4"},
            "stems": {"dir": "stems"},
            "warnings": [],
        }
    )
    assert "wrote out.wav" in normal
    assert "muxed: out.mp4" in normal and "stems: stems" in normal


def test_import_isolation_subprocess():
    root = Path(__file__).resolve().parent.parent
    code = (
        "import sys\n"
        "import qtts_cli.dub\n"
        "import qtts_cli.commands.dub\n"
        "bad = [m for m in ('torch','qwen_tts','qwen_asr','dashscope','vllm_omni') if m in sys.modules]\n"
        "raise SystemExit(1 if bad else 0)\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root)
    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=root,
        env=env,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


class _CliRunner:
    """Default-runner stand-in used by the write-failure CLI tests."""

    def __init__(self, cfg, args, recipe_dir):
        self.recipe_dir = recipe_dir

    def synthesize(self, **kwargs):
        out = Path(kwargs["out"])
        out.parent.mkdir(parents=True, exist_ok=True)
        sf.write(
            str(out),
            np.full(12000, 0.2, dtype=np.float32),
            24000,
            subtype="PCM_16",
        )
        return {
            "sample_rate": 24000,
            "duration_s": 0.5,
            "dtype_actual": "bf16",
            "device_actual": "cpu",
            "languages_actual": ["English"],
        }

    def request_cancel(self):
        pass


def _cli_build_json(timeline: Path, model_dir: Path, *extra: str):
    import io
    from contextlib import redirect_stderr, redirect_stdout

    argv = [
        "dub",
        "build",
        str(timeline),
        "--model-dir",
        str(model_dir),
        "--out",
        str(timeline.parent / "out.wav"),
        "--voice",
        "builtin:Vivian",
        "--device",
        "cpu",
        "--dtype",
        "bf16",
        "--no-loudness",
        "--quiet",
        "--json",
        *extra,
    ]
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        rc = cli.main(argv)
    return rc, out.getvalue(), err.getvalue()


def _assert_model_json(rc, out, err):
    assert rc == 3
    assert out == ""
    assert "Traceback" not in err
    payload = json.loads(err)
    assert payload["code"] == 3
    assert payload["error"] == "model"


def test_final_write_soundfile_error_is_exit3_json(tmp_path: Path, model_dir: Path, monkeypatch):
    from qtts_cli.audiobook import encode as audiobook_encode
    from qtts_cli.dub import synth as dub_synth

    def boom(*args, **kwargs):
        raise sf.LibsndfileError(errno.EIO, "disk full")

    monkeypatch.setattr(dub_synth, "SaySegmentRunner", _CliRunner)
    monkeypatch.setattr(audiobook_encode, "write_lossless", boom)
    rc, out, err = _cli_build_json(_timeline(tmp_path, "hello"), model_dir)
    _assert_model_json(rc, out, err)
    assert not (tmp_path / "out.wav").exists()


def test_rendered_write_soundfile_error_is_exit3_json(
    tmp_path: Path, model_dir: Path, monkeypatch
):
    from qtts_cli.dub import synth as dub_synth

    real_write = sf.write

    def fail_rendered(path, *args, **kwargs):
        if Path(str(path)).parent.name == "rendered":
            raise sf.LibsndfileError(errno.EIO, "rendered write failed")
        return real_write(path, *args, **kwargs)

    monkeypatch.setattr(dub_synth, "SaySegmentRunner", _CliRunner)
    monkeypatch.setattr(sf, "write", fail_rendered)
    rc, out, err = _cli_build_json(_timeline(tmp_path, "hello"), model_dir)
    _assert_model_json(rc, out, err)
    assert not (tmp_path / "out.wav").exists()


def test_stem_write_soundfile_error_is_exit3_json(
    tmp_path: Path, model_dir: Path, monkeypatch
):
    from qtts_cli.audiobook import encode as audiobook_encode
    from qtts_cli.dub import synth as dub_synth

    stems = tmp_path / "stems"
    real_write = audiobook_encode.write_lossless

    def fail_stem(path, *args, **kwargs):
        if Path(path).parent == stems:
            raise sf.LibsndfileError(errno.EIO, "stem write failed")
        return real_write(path, *args, **kwargs)

    monkeypatch.setattr(dub_synth, "SaySegmentRunner", _CliRunner)
    monkeypatch.setattr(audiobook_encode, "write_lossless", fail_stem)
    rc, out, err = _cli_build_json(
        _timeline(tmp_path, "hello"), model_dir, "--stems-out", str(stems)
    )
    _assert_model_json(rc, out, err)
    assert not (tmp_path / "out.wav").exists()
    assert not list(stems.glob("*.wav"))


class _FailingWriter:
    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def write(self, _payload):
        raise OSError(errno.ENOSPC, "disk full")

    def flush(self):
        pass

    def fileno(self):
        return -1


def _fail_json_open(monkeypatch, predicate):
    import builtins

    real_open = builtins.open

    def opener(path, *args, **kwargs):
        if predicate(Path(path)):
            return _FailingWriter()
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(dub_manifest, "open", opener, raising=False)


def test_timeline_json_write_error_is_model_error(
    tmp_path: Path, model_dir: Path, monkeypatch
):
    _fail_json_open(monkeypatch, lambda path: path.name.startswith("timeline.json"))
    with pytest.raises(ModelError) as caught:
        _run(
            _timeline(tmp_path, "hello"),
            model_dir,
            "--stems-out",
            str(tmp_path / "stems"),
        )
    assert caught.value.code == 3
    assert not (tmp_path / "out.wav").exists()


def test_manifest_write_error_is_model_error(
    tmp_path: Path, model_dir: Path, monkeypatch
):
    _fail_json_open(monkeypatch, lambda path: ".dub.json" in path.name)
    with pytest.raises(ModelError) as caught:
        _run(_timeline(tmp_path, "hello"), model_dir)
    assert caught.value.code == 3
    assert not (tmp_path / "out.wav").exists()
