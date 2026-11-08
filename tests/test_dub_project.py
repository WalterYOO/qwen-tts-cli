"""FR-P2-04 §10.3/§10.4: project schema, voices and dry-run."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from dub_utils import (
    FakeRunner,
    run_build,
    write_project,
    write_srt,
)
from qtts_cli import dub
from qtts_cli.errors import InputError, UsageError


def _project(tmp_path: Path, body: str) -> Path:
    return write_project(tmp_path / "dub.yaml", body)


def test_project_loader_valid_schema(tmp_path: Path):
    path = _project(
        tmp_path,
        """schema_version: 1
voice: builtin:Vivian
instruct: calm
language: English
sample_rate: 16000
loudness: {lufs: -20, true_peak: -2}
timeline: {fps: 30}
speakers:
  alice: {voice: "voice:alice", instruct: bright}
""",
    )
    project = dub.load_project(path)
    assert project.voice == "builtin:Vivian"
    assert project.instruct == "calm"
    assert project.language == "English"
    assert project.sample_rate == 16000
    assert project.lufs == -20.0
    assert project.true_peak == -2.0
    assert project.fps == 30
    assert project.speakers["alice"].voice == "voice:alice"
    assert project.speakers["alice"].instruct == "bright"


@pytest.mark.parametrize(
    "body",
    [
        "schema_version: 1\nunknown: true\n",
        "schema_version: 2\n",
        "schema_version: true\n",
        "schema_version: 1\nspeakers:\n  bad name: {voice: builtin:Vivian}\n",
        "schema_version: 1\nsample_rate: 8000\n",
        "schema_version: 1\nsample_rate: 24000.5\n",
        "schema_version: 1\nloudness: {lufs: -1}\n",
        "schema_version: 1\ntimeline: {fps: 0}\n",
        "schema_version: 1\nvoice: not-a-ref\n",
        "schema_version: 1\nspeakers:\n  alice: {voice: nope}\n",
        "schema_version: 1\nspeakers:\n  alice: {unknown: x}\n",
        "schema_version: 1\nloudness: {unknown: 1}\n",
        "schema_version: 1\ntimeline: {unknown: 1}\n",
    ],
)
def test_project_loader_rejects_bad_schema(tmp_path: Path, body: str):
    with pytest.raises(InputError):
        dub.load_project(_project(tmp_path, body))


def test_project_loader_json_duplicate_and_nonfinite(tmp_path: Path):
    dup = tmp_path / "dub.json"
    dup.write_text('{"schema_version": 1, "schema_version": 1}', encoding="utf-8")
    with pytest.raises(InputError):
        dub.load_project(dup)
    nan = tmp_path / "nan.json"
    nan.write_text('{"schema_version": 1, "sample_rate": NaN}', encoding="utf-8")
    with pytest.raises(InputError):
        dub.load_project(nan)


def test_project_loader_bad_suffix(tmp_path: Path):
    path = tmp_path / "dub.txt"
    path.write_text("schema_version: 1\n", encoding="utf-8")
    with pytest.raises(InputError):
        dub.load_project(path)


def _timeline(tmp_path: Path, *texts: str) -> Path:
    entries = [
        (index, float(index - 1), float(index), text)
        for index, text in enumerate(texts, 1)
    ]
    return write_srt(tmp_path / "t.srt", entries)


def test_voice_precedence(tmp_path: Path, model_dir: Path):
    timeline = _timeline(tmp_path, "plain", "alice: hello", "bob: hi")
    project = _project(
        tmp_path,
        """schema_version: 1
voice: builtin:Vivian
speakers:
  alice: {voice: builtin:Serena, instruct: bright}
  bob: {instruct: dry}
""",
    )
    runner = FakeRunner()
    run_build(
        timeline,
        model_dir,
        runner,
        "--out",
        str(tmp_path / "out.wav"),
        "--project",
        str(project),
        "--no-loudness",
    )
    by_text = {call["text"]: call for call in runner.calls}
    assert by_text["plain"]["voice"] == "builtin:Vivian"
    assert by_text["hello"]["voice"] == "builtin:Serena"
    assert by_text["hello"]["instruct"] == "bright"
    assert by_text["hi"]["voice"] == "builtin:Vivian"
    assert by_text["hi"]["instruct"] == "dry"


def test_voice_cli_and_speaker_override(tmp_path: Path, model_dir: Path):
    timeline = _timeline(tmp_path, "plain", "alice: hello")
    project = _project(
        tmp_path,
        """schema_version: 1
voice: builtin:Vivian
speakers:
  alice: {voice: builtin:Serena, instruct: bright}
""",
    )
    runner = FakeRunner()
    run_build(
        timeline,
        model_dir,
        runner,
        "--out",
        str(tmp_path / "out.wav"),
        "--project",
        str(project),
        "--voice",
        "builtin:Ryan",
        "--speaker-voice",
        "alice=builtin:Aiden",
        "--no-loudness",
    )
    by_text = {call["text"]: call for call in runner.calls}
    assert by_text["plain"]["voice"] == "builtin:Ryan"
    assert by_text["hello"]["voice"] == "builtin:Aiden"
    assert by_text["hello"]["instruct"] == "bright"


def test_instruct_and_language_project_vs_cli(tmp_path: Path, model_dir: Path):
    timeline = _timeline(tmp_path, "hello")
    project = _project(
        tmp_path,
        """schema_version: 1
voice: builtin:Vivian
language: English
instruct: calm
""",
    )
    project_runner = FakeRunner()
    run_build(
        timeline,
        model_dir,
        project_runner,
        "--out",
        str(tmp_path / "project.wav"),
        "--project",
        str(project),
        "--no-loudness",
    )
    assert project_runner.calls[0]["language"] == "English"
    assert project_runner.calls[0]["instruct"] == "calm"

    cli_runner = FakeRunner()
    run_build(
        timeline,
        model_dir,
        cli_runner,
        "--out",
        str(tmp_path / "cli.wav"),
        "--project",
        str(project),
        "--language",
        "Chinese",
        "--instruct",
        "bright",
        "--no-loudness",
    )
    assert cli_runner.calls[0]["language"] == "Chinese"
    assert cli_runner.calls[0]["instruct"] == "bright"


def test_cli_instruct_is_global_but_speaker_project_wins(tmp_path: Path, model_dir: Path):
    timeline = _timeline(tmp_path, "alice: one", "bob: two")
    project = _project(
        tmp_path,
        """schema_version: 1
voice: builtin:Vivian
speakers:
  alice: {instruct: bright}
""",
    )
    runner = FakeRunner()
    run_build(
        timeline,
        model_dir,
        runner,
        "--out",
        str(tmp_path / "out.wav"),
        "--project",
        str(project),
        "--instruct",
        "loud",
        "--language",
        "Chinese",
        "--no-loudness",
    )
    by_text = {call["text"]: call for call in runner.calls}
    assert by_text["one"]["instruct"] == "bright"
    assert by_text["two"]["instruct"] == "loud"
    assert {call["language"] for call in runner.calls} == {"Chinese"}


def test_voice_all_mapped_needs_no_default(tmp_path: Path, model_dir: Path):
    timeline = _timeline(tmp_path, "alice: hello")
    runner = FakeRunner()
    run_build(
        timeline,
        model_dir,
        runner,
        "--out",
        str(tmp_path / "out.wav"),
        "--speaker-voice",
        "alice=builtin:Vivian",
        "--no-loudness",
    )
    assert runner.calls[0]["voice"] == "builtin:Vivian"


def test_voice_unknown_speaker_needs_default(tmp_path: Path, model_dir: Path):
    timeline = _timeline(tmp_path, "nobody: hello")
    with pytest.raises(InputError):
        run_build(
            timeline,
            model_dir,
            FakeRunner(),
            "--out",
            str(tmp_path / "out.wav"),
            "--no-loudness",
        )


def test_voice_case_sensitive(tmp_path: Path, model_dir: Path):
    timeline = _timeline(tmp_path, "Alice: hello")
    project = _project(
        tmp_path,
        """schema_version: 1
voice: builtin:Vivian
speakers:
  alice: {voice: builtin:Serena}
""",
    )
    runner = FakeRunner()
    run_build(
        timeline,
        model_dir,
        runner,
        "--out",
        str(tmp_path / "out.wav"),
        "--project",
        str(project),
        "--no-loudness",
    )
    assert runner.calls[0]["voice"] == "builtin:Vivian"


def test_invalid_ref_layering(tmp_path: Path, model_dir: Path):
    timeline = _timeline(tmp_path, "hello")
    with pytest.raises(UsageError):
        run_build(
            timeline,
            model_dir,
            FakeRunner(),
            "--out",
            str(tmp_path / "out.wav"),
            "--speaker-voice",
            "alice=nope",
        )
    with pytest.raises(InputError):
        run_build(
            timeline,
            model_dir,
            FakeRunner(),
            "--out",
            str(tmp_path / "out.wav"),
            "--voice",
            "nope",
        )
    bad = _project(tmp_path, "schema_version: 1\nvoice: nope\n")
    with pytest.raises(InputError):
        run_build(
            timeline,
            model_dir,
            FakeRunner(),
            "--out",
            str(tmp_path / "out.wav"),
            "--project",
            str(bad),
        )


def test_voice_preflight_before_synthesis(tmp_path: Path, model_dir: Path):
    timeline = _timeline(tmp_path, "hello")
    runner = FakeRunner()
    with pytest.raises(InputError):
        run_build(
            timeline,
            model_dir,
            runner,
            "--out",
            str(tmp_path / "out.wav"),
            "--voice",
            "voice:missing",
            "--no-loudness",
        )
    assert runner.calls == []


def test_dry_run_writes_nothing_and_uses_null_fingerprints(tmp_path: Path, model_dir: Path):
    timeline = _timeline(tmp_path, "hello")
    out = tmp_path / "nested" / "out.wav"
    runner = FakeRunner()
    payload = run_build(
        timeline,
        model_dir,
        runner,
        "--out",
        str(out),
        "--voice",
        "builtin:Vivian",
        "--stems-out",
        str(tmp_path / "stems"),
        "--dry-run",
    )
    assert payload["ok"] is True and payload["dry_run"] is True
    assert payload["cues"][0]["fingerprint"] is None
    assert payload["output"]["sample_rate"] == 24000
    assert payload["loudness"]["enabled"] is True
    assert runner.calls == []
    assert not out.parent.exists()
    assert not (tmp_path / "stems").exists()


def test_dry_run_explicit_device_computes_fingerprints(tmp_path: Path, model_dir: Path):
    timeline = _timeline(tmp_path, "hello")
    payload = run_build(
        timeline,
        model_dir,
        FakeRunner(),
        "--out",
        str(tmp_path / "out.wav"),
        "--voice",
        "builtin:Vivian",
        "--device",
        "cpu",
        "--dry-run",
    )
    assert payload["cues"][0]["fingerprint"].startswith("sha256:")


def test_dry_run_project_precedence(tmp_path: Path, model_dir: Path):
    timeline = _timeline(tmp_path, "hello")
    project = _project(
        tmp_path,
        """schema_version: 1
voice: builtin:Vivian
language: English
instruct: calm
sample_rate: 16000
loudness: {lufs: -22, true_peak: -3}
""",
    )
    payload = run_build(
        timeline,
        model_dir,
        FakeRunner(),
        "--out",
        str(tmp_path / "out.wav"),
        "--project",
        str(project),
        "--device",
        "cpu",
        "--dry-run",
    )
    assert payload["output"]["sample_rate"] == 16000
    assert payload["language"] == "English"
    assert payload["instruct"] == "calm"
    assert payload["cues"][0]["instruct"] == "calm"
    assert payload["loudness"] == {
        "enabled": True,
        "target_lufs": -22.0,
        "true_peak_dbtp": -3.0,
    }
    cli = run_build(
        timeline,
        model_dir,
        FakeRunner(),
        "--out",
        str(tmp_path / "out2.wav"),
        "--project",
        str(project),
        "--device",
        "cpu",
        "--sample-rate",
        "32000",
        "--lufs",
        "-18",
        "--true-peak",
        "-2",
        "--language",
        "Chinese",
        "--instruct",
        "bright",
        "--dry-run",
    )
    assert cli["output"]["sample_rate"] == 32000
    assert cli["language"] == "Chinese"
    assert cli["instruct"] == "bright"
    assert cli["cues"][0]["instruct"] == "bright"
    assert cli["loudness"]["target_lufs"] == -18.0
    assert cli["loudness"]["true_peak_dbtp"] == -2.0


def test_dry_run_no_torch_import(tmp_path: Path, model_dir: Path):
    timeline = _timeline(tmp_path, "hello")
    code = (
        "import sys\n"
        "from qtts_cli.cli import main\n"
        f"rc = main(['dub','build',{str(timeline)!r},'--model-dir',{str(model_dir)!r},"
        f"'--out',{str(tmp_path / 'out.wav')!r},'--voice','builtin:Vivian','--dry-run','--json'])\n"
        "bad = [m for m in ('torch','qwen_tts','qwen_asr','dashscope','vllm_omni') if m in sys.modules]\n"
        "print('HEAVY=' + ','.join(bad), file=sys.stderr)\n"
        "raise SystemExit(rc)\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parent.parent,
        env={"PYTHONPATH": str(Path(__file__).resolve().parent.parent)},
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "HEAVY=" in completed.stderr
    assert "HEAVY=\n" in completed.stderr + "\n"
    assert not (tmp_path / "out.wav").exists()


def test_dry_run_malformed_project_is_input_error(tmp_path: Path, model_dir: Path):
    timeline = _timeline(tmp_path, "hello")
    project = _project(tmp_path, "schema_version: 1\nunknown: true\n")
    with pytest.raises(InputError):
        run_build(
            timeline,
            model_dir,
            FakeRunner(),
            "--out",
            str(tmp_path / "out.wav"),
            "--project",
            str(project),
            "--dry-run",
        )
