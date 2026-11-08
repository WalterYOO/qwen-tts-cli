"""FR-P2-03 §12.4/§12.11/§12.12/§12.13/§12.14: CLI, paths, manifest, docs."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from audiobook_utils import FakeRunner, inline_project, run_build
from qtts_cli import audiobook
from qtts_cli.cli import main
from qtts_cli.errors import InputError
from qtts_cli.utils import sha256_text

ROOT = Path(__file__).resolve().parent.parent


class Bomb(FakeRunner):
    def synthesize(self, **kwargs):
        raise AssertionError("runner must not be called")


def _project(tmp_path: Path) -> Path:
    return inline_project(tmp_path, chapters=[{"title": "A", "text": "one"}])


def test_command_surface_usage_errors(tmp_path: Path):
    project = _project(tmp_path)
    assert main(["audiobook"]) == 2
    assert main(["audiobook", "build", str(project), "--out", "a.wav", "--out-dir", "b"]) == 2
    assert main(["audiobook", "build", str(project), "--m4b", "--format", "wav"]) == 2
    assert main(["audiobook", "build", str(project), "--out", "-"]) == 2
    assert main(["audiobook", "build", str(project), "--format", "mp3", "--bitrate", "31"]) == 2
    assert main(["audiobook", "build", str(project), "--format", "wav", "--bitrate", "128"]) == 2
    assert main(["audiobook", "build", str(project), "--format", "ogg"]) == 2
    assert main(["audiobook", "build", str(project), "--format", "wav", "--out", "x.flac"]) == 2
    assert main(["audiobook", "build", str(project), "--format", "wav", "--out", "x.ogg"]) == 2
    assert main(["audiobook", "build", str(project), "--format", "wav", "--lufs", "-50"]) == 2
    assert main(["audiobook", "build", str(project), "--format", "wav", "--true-peak", "-2"]) == 2
    assert main(["audiobook", "build", str(project), "--lufs", "-18", "--no-loudness"]) == 2
    assert (
        main(
            [
                "audiobook",
                "build",
                str(project),
                "--no-loudness",
                "--true-peak",
                "-2",
            ]
        )
        == 2
    )
    assert main(["audiobook", "build", str(project), "--sample-rate", "4000"]) == 2
    assert main(["audiobook", "build", str(project), "--chapter-pause-ms", "70000"]) == 2


def test_out_suffix_append_rule(tmp_path: Path, model_dir: Path):
    project = _project(tmp_path)
    payload = run_build(project, model_dir, Bomb(), "--dry-run", "--out", str(tmp_path / "plain"))
    assert payload["output"]["path"].endswith("plain.wav")
    assert payload["output"]["format"] == "wav"


def test_dry_run_writes_nothing_and_never_calls_runner(tmp_path: Path, model_dir: Path):
    project = _project(tmp_path)
    payload = run_build(project, model_dir, Bomb(), "--dry-run", "--json")
    assert payload["dry_run"] is True
    assert payload["will_write"]
    assert not (tmp_path / "build").exists()
    assert not (tmp_path / "build" / ".audiobook.lock").exists()


def test_json_success_and_error_streams(tmp_path: Path, model_dir: Path, capsys):
    project = _project(tmp_path)
    code = main(
        ["audiobook", "build", str(project), "--model-dir", str(model_dir), "--dry-run", "--json"]
    )
    captured = capsys.readouterr()
    assert code == 0
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert payload["dry_run"] is True

    missing = main(["audiobook", "build", str(tmp_path / "missing.yaml"), "--json"])
    captured = capsys.readouterr()
    assert missing == 4
    assert captured.out == ""
    error = json.loads(captured.err)
    assert error["ok"] is False and error["code"] == 4


def test_cli_cancellation_lines(tmp_path: Path, monkeypatch, capsys):
    project = _project(tmp_path)

    def interrupt(cfg, args, *, runner=None):
        raise KeyboardInterrupt

    monkeypatch.setattr(audiobook, "run", interrupt)
    assert main(["audiobook", "build", str(project)]) == 130
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "interrupted\n"

    monkeypatch.setattr(
        audiobook, "run", lambda cfg, args, *, runner=None: {"raw_stdout": True, "exit_code": 130}
    )
    assert main(["audiobook", "build", str(project), "--json"]) == 130
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_path_collisions(tmp_path: Path, model_dir: Path):
    journal_named = tmp_path / ".audiobook.publish.yaml"
    journal_named.write_text(
        "schema_version: 1\nvoice: builtin:Vivian\nchapters:\n  - title: A\n    text: x\n",
        encoding="utf-8",
    )
    with pytest.raises(InputError):
        run_build(journal_named, model_dir, Bomb())

    project = _project(tmp_path)
    segments = tmp_path / "build" / "segments"
    segments.mkdir(parents=True)
    (segments / "chapter.txt").write_text("inside work dir", encoding="utf-8")
    inside = tmp_path / "inside.yaml"
    inside.write_text(
        "schema_version: 1\nvoice: builtin:Vivian\nchapters:\n"
        "  - title: A\n    source: build/segments/chapter.txt\n",
        encoding="utf-8",
    )
    with pytest.raises(InputError):
        run_build(inside, model_dir, Bomb())

    with pytest.raises(InputError):
        run_build(project, model_dir, Bomb(), "--out", str(segments / "out.wav"))

    link = tmp_path / "bgm.wav"
    os.symlink(project, link)
    with pytest.raises(InputError):
        run_build(project, model_dir, Bomb(), "--bgm", str(link))


def test_manifest_languages_and_inline_sources(tmp_path: Path, model_dir: Path):
    project = inline_project(
        tmp_path,
        chapters=[
            {"title": "A", "text": "one\n\ntwo"},
            {"title": "B", "language": "English", "instruct": "soft", "text": "three"},
        ],
    )
    runner = FakeRunner(languages=["Chinese", "English"])
    run_build(project, model_dir, runner)
    manifest = json.loads((tmp_path / "build" / "book.json").read_text(encoding="utf-8"))
    assert manifest["chapters"][0]["language"] == "auto"
    assert manifest["chapters"][0]["instruct"] is None
    assert manifest["chapters"][1]["language"] == "English"
    assert manifest["chapters"][1]["instruct"] == "soft"
    assert manifest["chapters"][0]["languages_actual"] == ["Chinese", "English"]
    assert manifest["chapters"][0]["source"] == {
        "path": None,
        "sha256": sha256_text("one\n\ntwo"),
        "inline": True,
    }
    assert len(manifest["segment_fingerprints"]) == 3
    assert manifest["segment_fingerprints"][0]["chapter"] == 1
    assert manifest["segment_fingerprints"][0]["paragraph"] == 1
    assert manifest["segment_fingerprints"][0]["languages_actual"] == ["Chinese", "English"]
    assert manifest["chapters"][0]["model_id"]
    assert manifest["segments"] == {"total": 3, "synthesized": 3, "reused": 0}


def test_no_heavy_imports_in_fresh_processes(tmp_path: Path, model_dir: Path):
    project = _project(tmp_path)
    code = (
        "import sys\n"
        "import qtts_cli.audiobook, qtts_cli.commands.audiobook\n"
        "from qtts_cli.cli import main\n"
        f"rc = main(['audiobook','build',{str(project)!r},'--model-dir',{str(model_dir)!r},'--dry-run','--json'])\n"
        "bad = sorted(m for m in ('torch','qwen_tts','qwen_asr','dashscope','vllm_omni') if m in sys.modules)\n"
        "print(rc, bad)\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(ROOT),
        env={**os.environ, "PYTHONPATH": str(ROOT)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip().endswith("0 []")


LOUDNESS = "loudness:\n  lufs: -20\n  true_peak: -2"


def test_project_loudness_applies_and_cli_overrides(tmp_path: Path, model_dir: Path):
    project = inline_project(tmp_path, extra=LOUDNESS)
    planned = run_build(project, model_dir, Bomb(), "--dry-run", "--json")
    assert planned["loudness"] == {
        "enabled": True,
        "target_lufs": -20.0,
        "true_peak_dbtp": -2.0,
    }

    run_build(project, model_dir, FakeRunner())
    manifest = json.loads((tmp_path / "build" / "book.json").read_text(encoding="utf-8"))
    assert manifest["loudness"]["enabled"] is True
    assert manifest["loudness"]["target_lufs"] == -20.0
    assert manifest["loudness"]["true_peak_dbtp"] == -2.0
    assert manifest["chapters"][0]["loudness"]["target_lufs"] == -20.0

    (tmp_path / "build" / "book.wav").unlink()
    (tmp_path / "build" / "book.json").unlink()
    overridden = run_build(
        project,
        model_dir,
        Bomb(),
        "--dry-run",
        "--json",
        "--lufs",
        "-25",
        "--true-peak",
        "-3",
    )
    assert overridden["loudness"] == {
        "enabled": True,
        "target_lufs": -25.0,
        "true_peak_dbtp": -3.0,
    }


def test_project_true_peak_alone_enables_loudness(tmp_path: Path, model_dir: Path):
    project = inline_project(tmp_path, extra="loudness:\n  true_peak: -2")
    planned = run_build(project, model_dir, Bomb(), "--dry-run", "--json")
    assert planned["loudness"] == {
        "enabled": True,
        "target_lufs": -18.0,
        "true_peak_dbtp": -2.0,
    }


def test_dry_run_ignores_existing_output(tmp_path: Path, model_dir: Path):
    project = _project(tmp_path)
    build = tmp_path / "build"
    build.mkdir()
    final = build / "book.wav"
    manifest = build / "book.json"
    final.write_bytes(b"old-final")
    manifest.write_text("old-manifest", encoding="utf-8")
    before = (
        final.stat().st_mtime_ns,
        final.read_bytes(),
        manifest.stat().st_mtime_ns,
        manifest.read_text(encoding="utf-8"),
    )
    payload = run_build(project, model_dir, Bomb(), "--dry-run", "--json")
    assert payload["dry_run"] is True
    after = (
        final.stat().st_mtime_ns,
        final.read_bytes(),
        manifest.stat().st_mtime_ns,
        manifest.read_text(encoding="utf-8"),
    )
    assert before == after

    with pytest.raises(InputError):
        run_build(project, model_dir, Bomb())


def test_dry_run_fingerprints_with_explicit_device(tmp_path: Path, model_dir: Path):
    project = _project(tmp_path)
    explicit = run_build(
        project, model_dir, Bomb(), "--dry-run", "--json", "--device", "cpu"
    )
    fingerprints = explicit["chapters"][0]["fingerprints"]
    assert fingerprints and all(
        isinstance(value, str) and value.startswith("sha256:") for value in fingerprints
    )

    automatic = run_build(
        project, model_dir, Bomb(), "--dry-run", "--json", "--device", "auto"
    )
    assert automatic["chapters"][0]["fingerprints"] == [None]


def test_long_book_peak_memory_is_bounded(tmp_path: Path, model_dir: Path):
    import tracemalloc

    project = inline_project(
        tmp_path,
        chapters=[{"title": "A", "text": "one\n\ntwo\n\nthree\n\nfour"}],
    )
    runner = FakeRunner(seconds=30)
    tracemalloc.start()
    try:
        # ``--device cpu`` keeps torch (and its import-time allocations) out of
        # the measurement, so the peak reflects book buffers only.
        payload = run_build(
            project, model_dir, runner, "--keep-stems", "--device", "cpu"
        )
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert len(runner.calls) == 4
    assert payload["output"]["duration_s"] >= 120.0
    master_bytes = payload["output"]["duration_s"] * 24000 * 4
    assert peak <= 3 * master_bytes, (peak, master_bytes)


def test_docs_mention_audiobook():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert "qtts audiobook build" in readme
    assert "FR-P2-03" in changelog
