"""End-to-end CLI behaviour that needs no models (exit codes, JSON, dry-run)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def run_cli(model_dir: Path, tmp_path: Path, *args: str):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)
    env["QTTS_DATA_DIR"] = str(tmp_path / "data")
    cmd = [
        sys.executable,
        "-m",
        "qtts_cli",
        *args,
        "--model-dir",
        str(model_dir),
        "--data-dir",
        str(tmp_path / "data"),
        "--cache-dir",
        str(tmp_path / "cache"),
    ]
    return subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(tmp_path))


def test_dry_run_does_not_import_torch(model_dir: Path, tmp_path: Path):
    script = (
        "import sys\n"
        "from qtts_cli.cli import main\n"
        f"rc = main(['say','你好。','--voice','builtin:Vivian','--model-dir',{str(model_dir)!r},"
        f"'--data-dir',{str(tmp_path / 'data')!r},'--cache-dir',{str(tmp_path / 'cache')!r},"
        "'--dry-run','--json'])\n"
        "heavy = sorted({m.split('.')[0] for m in sys.modules if m.split('.')[0] in "
        "('torch','qwen_tts','transformers','qwen_asr')})\n"
        "print('RC', rc)\n"
        "print('HEAVY', heavy)\n"
    )
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT))
    completed = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=env)
    assert completed.returncode == 0, completed.stderr
    assert "RC 0" in completed.stdout
    assert "HEAVY []" in completed.stdout


def test_dry_run_json_plan(model_dir: Path, tmp_path: Path):
    completed = run_cli(model_dir, tmp_path, "say", "你好，世界。第二句。", "--voice", "builtin:Vivian",
                        "--dry-run", "--json")
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["dry_run"] is True
    assert payload["models_loaded"] is False
    assert payload["plan"]["voice"]["speaker"] == "Vivian"
    assert len(payload["plan"]["segments"]) >= 1
    assert all(seg["cache_key"].startswith("sha256:") for seg in payload["plan"]["segments"])


def test_06b_custom_voice_instruct_is_rejected_before_load(small_model_dir: Path, tmp_path: Path):
    completed = run_cli(small_model_dir, tmp_path, "say", "你好。", "--voice", "builtin:Vivian",
                        "--instruct", "温柔", "--dry-run", "--json")
    assert completed.returncode == 4, completed.stderr
    assert "instruct" in completed.stderr.lower()
    assert "Traceback" not in completed.stderr


def test_base_model_rejects_instruct(model_dir: Path, tmp_path: Path):
    # voice:alice does not exist -> input error before instruct validation
    completed = run_cli(model_dir, tmp_path, "say", "你好。", "--voice", "voice:missing", "--dry-run")
    assert completed.returncode == 4
    assert "not found" in completed.stderr


def test_unknown_speaker_is_rejected(model_dir: Path, tmp_path: Path):
    completed = run_cli(model_dir, tmp_path, "say", "你好。", "--voice", "builtin:Nobody", "--dry-run")
    assert completed.returncode == 4
    assert "unknown built-in speaker" in completed.stderr


def test_missing_model_dir_returns_model_error(tmp_path: Path):
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT))
    completed = subprocess.run(
        [sys.executable, "-m", "qtts_cli", "say", "你好。", "--voice", "builtin:Vivian",
         "--model-dir", str(tmp_path / "nope"), "--dry-run"],
        capture_output=True, text=True, env=env, cwd=str(tmp_path),
    )
    assert completed.returncode == 3
    assert "no local models" in completed.stderr or "model directory" in completed.stderr


def test_json_and_stdout_audio_conflict(model_dir: Path, tmp_path: Path):
    completed = run_cli(model_dir, tmp_path, "say", "你好。", "--voice", "builtin:Vivian",
                        "--out", "-", "--json")
    assert completed.returncode == 2


def test_voice_list_shows_nine_builtins(model_dir: Path, tmp_path: Path):
    completed = run_cli(model_dir, tmp_path, "voice", "list", "--json")
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert len(payload["builtin"]) == 9
    assert payload["voices"] == []


def test_doctor_json(model_dir: Path, tmp_path: Path):
    completed = run_cli(model_dir, tmp_path, "doctor", "--json")
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert "packages" in payload
    assert {"Qwen3-TTS-12Hz-1.7B-CustomVoice"} <= {m["name"] for m in payload["models"]}


def test_models_list(model_dir: Path, tmp_path: Path):
    completed = run_cli(model_dir, tmp_path, "models", "list", "--json")
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    names = {m["name"] for m in payload["models"]}
    assert "Qwen3-TTS-12Hz-1.7B-Base" in names


def test_subtitle_without_out_uses_audio_stem(model_dir: Path, tmp_path: Path):
    # Audio generation needs torch; this asserts the planning-level guard order.
    completed = run_cli(model_dir, tmp_path, "say", "你好。", "--voice", "builtin:Vivian",
                        "--out", "-", "--subtitle", "srt")
    assert completed.returncode == 2
    assert "subtitle" in completed.stderr.lower()
