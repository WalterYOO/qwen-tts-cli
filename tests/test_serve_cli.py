"""CLI validation, dry-run, startup line and import isolation (§1, §2, §11)."""

from __future__ import annotations

import json
import os
import queue
import re
import signal
import subprocess
import sys
import threading
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
MAIN = (
    "import sys, json\n"
    "from qtts_cli.cli import main\n"
    "rc = main(sys.argv[1:])\n"
    "print('IMPORTS=' + json.dumps([m for m in ('torch','qwen_tts',"
    "'dashscope','vllm_omni') if m in sys.modules]), file=sys.stderr)\n"
    "raise SystemExit(rc)\n"
)


def _env(**extra: str) -> dict[str, str]:
    env = dict(os.environ)
    env.pop("DASHSCOPE_API_KEY", None)
    env.pop("QTTS_SERVE_TOKEN", None)
    env["PYTHONPATH"] = str(REPO)
    env["PYTHONUNBUFFERED"] = "1"
    env.update(extra)
    return env


def _main(args: list[str], *, env_extra=None, timeout=60):
    return subprocess.run(
        [sys.executable, "-c", MAIN, *args],
        cwd=str(REPO),
        env=_env(**(env_extra or {})),
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _imports(stderr: str) -> list[str]:
    line = [line for line in stderr.splitlines() if line.startswith("IMPORTS=")][-1]
    return json.loads(line.split("=", 1)[1])


def _readline(proc: subprocess.Popen, timeout: float = 10.0) -> str:
    result: queue.Queue[str] = queue.Queue()

    def read() -> None:
        result.put(proc.stdout.readline())

    threading.Thread(target=read, daemon=True).start()
    return result.get(timeout=timeout)


def _serve_command(model_dir: Path, data_dir: Path, *extra: str) -> list[str]:
    return [
        "serve",
        "--port",
        "0",
        "--model-dir",
        str(model_dir),
        "--data-dir",
        str(data_dir),
        "--cache-dir",
        str(data_dir / "cache"),
        *extra,
    ]


def test_dry_run_json_reports_metadata_without_runtime_imports(model_dir: Path, tmp_path: Path):
    result = _main(
        [
            "serve",
            "--dry-run",
            "--json",
            "--model-dir",
            str(model_dir),
            "--data-dir",
            str(tmp_path / "data"),
        ]
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["dry_run"] is True
    assert payload["backend"]["name"] == "transformers"
    assert payload["backend"]["configured"] is True
    assert payload["backend"]["models"]
    assert payload["effective_max_concurrency"] == 1
    assert "event" not in payload
    assert _imports(result.stderr) == []


def test_dry_run_human_output(model_dir: Path, tmp_path: Path):
    result = _main(
        [
            "serve",
            "--dry-run",
            "--model-dir",
            str(model_dir),
            "--data-dir",
            str(tmp_path / "data"),
        ]
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("qtts serve (dry run)")
    assert "backend: transformers" in result.stdout
    assert "configured: true" in result.stdout
    assert "listening" not in result.stdout


@pytest.mark.parametrize(
    ("extra", "code"),
    [
        (["--port", "abc"], 2),
        (["--port", "-1"], 2),
        (["--max-concurrency", "0"], 2),
        (["--queue-size", "-1"], 2),
        (["--max-connections", "0"], 2),
        (["--keep-jobs", "-1"], 2),
        (["--max-body-bytes", "100"], 2),
        (["--shutdown-timeout", "-1"], 2),
        (["--shutdown-timeout", "nan"], 2),
        (["--shutdown-timeout", "inf"], 2),
        (["--token", ""], 2),
        (["--host", "0.0.0.0", "--allowed-host", "lan.test"], 2),
        (["--host", "0.0.0.0", "--token", "t"], 2),
        (["--allowed-host", "example.com:80"], 2),
        (["--allowed-host", "http://example.com"], 2),
        (["--backend", "dashscope", "--offline"], 2),
    ],
)
def test_cli_validation_exits_2(extra, code, model_dir: Path, tmp_path: Path):
    result = _main(
        [
            "serve",
            "--dry-run",
            "--model-dir",
            str(model_dir),
            "--data-dir",
            str(tmp_path / "data"),
            *extra,
        ],
    )
    assert result.returncode == code, (result.stdout, result.stderr)


def test_missing_remote_credentials_exit_3(tmp_path: Path):
    result = _main(
        [
            "serve",
            "--backend",
            "dashscope",
            "--port",
            "0",
            "--data-dir",
            str(tmp_path / "data"),
        ]
    )
    assert result.returncode == 3, result.stderr
    assert "not configured" in result.stderr


def test_import_qtts_serve_does_not_import_optional_runtimes():
    code = (
        "import sys, json\n"
        "import qtts_cli.serve\n"
        "import qtts_cli.commands.serve\n"
        "bad = [m for m in ('torch','qwen_tts','dashscope','vllm_omni') "
        "if m in sys.modules]\n"
        "print(json.dumps(bad))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(REPO),
        env=_env(),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == []


def test_json_startup_line_is_single_compact_and_cli_prints_nothing_more(
    model_dir: Path, tmp_path: Path
):
    proc = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "from qtts_cli.cli import main; raise SystemExit(main())",
            *_serve_command(model_dir, tmp_path / "data", "--json"),
        ],
        cwd=str(REPO),
        env=_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        line = _readline(proc)
        assert line.endswith("\n")
        payload = json.loads(line)
        assert payload["ok"] is True
        assert payload["event"] == "listening"
        assert payload["host"] == "127.0.0.1"
        assert isinstance(payload["port"], int) and payload["port"] > 0
        assert payload["api"] == "both"
        assert payload["backend"]["name"] == "transformers"
        assert payload["effective_max_concurrency"] == 1
        assert payload["max_concurrency"] == 1
        assert ", " not in line
        proc.send_signal(signal.SIGINT)
        out, err = proc.communicate(timeout=15)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)
    assert proc.returncode == 0, err
    assert out == ""
    assert "Traceback" not in err


def test_human_startup_line(model_dir: Path, tmp_path: Path):
    proc = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "from qtts_cli.cli import main; raise SystemExit(main())",
            *_serve_command(model_dir, tmp_path / "data"),
        ],
        cwd=str(REPO),
        env=_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        line = _readline(proc)
        assert re.fullmatch(
            r"qtts serve listening on http://127\.0\.0\.1:\d+ "
            r"\(api=both, backend=transformers\)\n",
            line,
        )
        proc.send_signal(signal.SIGINT)
        out, err = proc.communicate(timeout=15)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)
    assert proc.returncode == 0
    assert out == ""


def test_plugin_prints_during_resolve_do_not_pollute_stdout(tmp_path: Path):
    code = f"""
import threading
from pathlib import Path
from types import SimpleNamespace
import qtts_cli.backends.registry as registry
from qtts_cli import serve
from qtts_cli.config import Config

class Spec:
    name = "fake"
    identity = "fake"
    models = []
    remote = False
    configured = None

def noisy_resolve(*args, **kwargs):
    print("noise from plugin import")
    spec = Spec()
    spec.configured = lambda: (print("noise from models()"), True)[1]
    return spec

registry.resolve_backend = noisy_resolve
args = SimpleNamespace(
    api="both", host="127.0.0.1", port=0, allowed_host=[],
    max_concurrency=1, queue_size=8, max_connections=8, keep_jobs=10,
    max_body_bytes=65536, token=None, shutdown_timeout=1.0, dry_run=False,
    model=None,
)
stop = threading.Event()
threading.Timer(0.5, stop.set).start()
serve.run(Config(data_dir=Path({str(tmp_path / 'data')!r})), args, stop_event=stop)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(REPO),
        env=_env(),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    output = result.stdout.splitlines()
    assert len(output) == 1
    assert output[0].startswith("qtts serve listening on http://127.0.0.1:")
    assert "noise from plugin import" in result.stderr
    assert "noise from models()" in result.stderr
