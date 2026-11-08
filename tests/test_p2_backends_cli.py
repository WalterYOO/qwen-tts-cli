"""FR-P2-02 CLI integration: backends across say, jsonl, batch and script."""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile

from qtts_cli import plugins
from qtts_cli.backends import dashscope as ds
from qtts_cli.backends import vllm_omni as vo
from qtts_cli.backends.base import BackendCapabilities
from qtts_cli.catalog import ModelInfo
from qtts_cli.cli import main


def _wav_bytes(sr=24000, samples=480, channels=1):
    data = np.zeros((samples, channels), dtype=np.float32)
    data[:, 0] = np.linspace(-0.25, 0.25, samples)
    if channels == 2:
        data[:, 1] = data[:, 0]
    buffer = io.BytesIO()
    soundfile.write(buffer, data, sr, format="WAV")
    return buffer.getvalue()


def _run(monkeypatch, tmp_path, *args, stdin=None, model_dir=None):
    if stdin is not None:
        monkeypatch.setattr(sys, "stdin", io.StringIO(stdin))
    return main(
        [
            *args,
            "--model-dir",
            str(model_dir if model_dir is not None else tmp_path / "models"),
            "--data-dir",
            str(tmp_path / "data"),
            "--cache-dir",
            str(tmp_path / "cache"),
        ]
    )


class _DashScope:
    """Fake realtime session; counts clients and records connect calls."""

    def __init__(self, payload, *, sr=24000):
        self.payload = payload
        self.sr = sr
        self.clients = 0
        self.connect_kwargs = []
        self.order = []

    def factory(self, api_key, workspace, region):
        self.clients += 1
        self.order.append(("factory", api_key, workspace, region))
        outer = self

        class Session:
            def __enter__(self):
                outer.order.append("enter")
                return self

            def __exit__(self, *exc):
                outer.order.append("exit")

            def send_text(self, text):
                outer.order.append(("send_text", text))

            def flush(self):
                outer.order.append("flush")

            def stop(self):
                outer.order.append("stop")

            def audio_chunks(self):
                half = len(outer.payload) // 2
                return iter([outer.payload[:half], outer.payload[half:]])

        class TTS:
            def connect(self, **kwargs):
                outer.connect_kwargs.append(kwargs)
                outer.order.append("connect")
                return Session()

        return SimpleNamespace(audio=SimpleNamespace(tts=TTS()))


@pytest.fixture()
def dashscope(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "secret")
    monkeypatch.setenv("DASHSCOPE_WORKSPACE", "ws")
    monkeypatch.setenv("DASHSCOPE_REGION", "cn-shanghai")
    fake = _DashScope(_wav_bytes())
    monkeypatch.setattr(ds, "_client_factory", fake.factory)
    return fake


def test_say_dashscope_generates_and_reports_remote_fields(
    monkeypatch, tmp_path, dashscope, capsys
):
    out = tmp_path / "out.wav"
    rc = _run(
        monkeypatch,
        tmp_path,
        "say",
        "你好",
        "--backend",
        "dashscope",
        "--voice",
        "builtin:cherry",
        "--out",
        str(out),
        "--json",
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["backend"] == {
        "name": "dashscope",
        "model": "qwen3-tts-flash-realtime",
        "remote": True,
        "identity": "dashscope:qwen3-tts-flash-realtime:cn-shanghai:ws",
    }
    assert payload["device"] == "remote"
    assert payload["dtype"] is None
    assert payload["models"][0]["id"] == "qwen3-tts-flash-realtime"
    assert dashscope.connect_kwargs[0]["voice"] == "Cherry"
    assert dashscope.clients == 1
    recipe = json.loads(out.with_suffix(".recipe.json").read_text(encoding="utf-8"))
    assert recipe["backend"]["name"] == "dashscope"
    assert recipe["backend"]["identity"] == payload["backend"]["identity"]
    assert recipe["model"]["dtype"] is None
    assert recipe["device"] == "remote"


def test_say_dry_run_dashscope_needs_no_key_and_imports_no_runtime(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    rc = _run(
        monkeypatch,
        tmp_path,
        "say",
        "--dry-run",
        "--backend",
        "dashscope",
        "--voice",
        "builtin:Cherry",
        "你好",
        "--json",
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["backend"]["configured"] is False
    assert payload["device"] == "remote"
    assert payload["dtype"] is None
    assert payload["memory_estimate"] == {
        "remote": True,
        "estimated_peak_gb": None,
    }
    assert payload["dependencies"]["tts"] is False
    rc = _run(
        monkeypatch,
        tmp_path,
        "say",
        "--dry-run",
        "--backend",
        "dashscope",
        "--voice",
        "builtin:Cherry",
        "你好",
    )
    assert rc == 0
    assert "remote backend" in capsys.readouterr().out
    # Check the import guarantee in a clean interpreter: other tests in this
    # process may already have imported qwen_tts/torch for unrelated features.
    code = f"""
import sys
from qtts_cli.cli import main
rc = main([
    "say", "--dry-run", "--backend", "dashscope",
    "--voice", "builtin:Cherry", "你好", "--json",
    "--model-dir", {str(tmp_path / "models")!r},
    "--data-dir", {str(tmp_path / "data")!r},
])
assert rc == 0, rc
bad = {{"dashscope", "vllm_omni", "torch", "qwen_tts"}} & set(sys.modules)
assert not bad, bad
"""
    env = {key: value for key, value in os.environ.items() if key != "DASHSCOPE_API_KEY"}
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        env=env,
        cwd=str(Path(__file__).resolve().parents[1]),
    )
    assert result.returncode == 0, result.stderr.decode("utf-8", "replace")


@pytest.mark.parametrize(
    "extra",
    [
        ("--backend", "dashscope", "--model", "local"),
        ("--backend", "transformers", "--backend-model", "remote"),
        ("--backend", "dashscope", "--model", "local", "--backend-model", "remote"),
        ("--backend", "dashscope", "--backend-model", ""),
        ("--backend", "", "--voice", "builtin:Vivian"),
        ("--backend", "unknown-plugin"),
    ],
)
def test_backend_flag_conflicts_exit_two(
    monkeypatch, tmp_path, model_dir, extra, capsys
):
    monkeypatch.setattr(plugins, "_entry_points", lambda group: [])
    out = tmp_path / "out.wav"
    args = ["say", "你好", "--voice", "builtin:Vivian", *extra, "--out", str(out)]
    rc = _run(monkeypatch, tmp_path, *args, model_dir=model_dir)
    assert rc == 2
    assert not out.is_file()
    capsys.readouterr()


def test_config_file_backend_and_model_are_used(
    monkeypatch, tmp_path, capsys
):
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "backend": "dashscope",
                "backend_model": "qwen3-tts-flash-realtime",
            }
        ),
        encoding="utf-8",
    )
    rc = _run(
        monkeypatch,
        tmp_path,
        "say",
        "--dry-run",
        "--config",
        str(config),
        "--voice",
        "builtin:Cherry",
        "你好",
        "--json",
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["backend"]["name"] == "dashscope"
    assert payload["backend"]["model"] == "qwen3-tts-flash-realtime"


def test_dashscope_rejects_clone_design_and_instruct_before_any_client(
    monkeypatch, tmp_path, dashscope, capsys
):
    out = tmp_path / "out.wav"
    for voice in ("voice:alice", "design:温柔"):
        rc = _run(
            monkeypatch,
            tmp_path,
            "say",
            "你好",
            "--backend",
            "dashscope",
            "--voice",
            voice,
            "--out",
            str(out),
        )
        assert rc == 4
        assert not out.is_file()
        capsys.readouterr()
    rc = _run(
        monkeypatch,
        tmp_path,
        "say",
        "你好",
        "--backend",
        "dashscope",
        "--voice",
        "builtin:Cherry",
        "--instruct",
        "happy",
        "--out",
        str(out),
    )
    assert rc == 4
    assert dashscope.clients == 0
    assert dashscope.connect_kwargs == []
    capsys.readouterr()


def test_takes_and_reference_voice_require_transformers(
    monkeypatch, tmp_path, dashscope, capsys
):
    out = tmp_path / "out.wav"
    common = [
        "say",
        "你好",
        "--backend",
        "dashscope",
        "--voice",
        "builtin:Cherry",
        "--out",
        str(out),
    ]
    assert _run(
        monkeypatch, tmp_path, *common, "--takes", "2", "--select", "lufs"
    ) == 2
    assert not out.is_file()
    capsys.readouterr()
    assert _run(
        monkeypatch,
        tmp_path,
        *common,
        "--qc",
        "--reference-voice",
        "alice",
    ) == 2
    assert not out.is_file()
    assert dashscope.clients == 0
    capsys.readouterr()


def test_dashscope_missing_key_is_exit_three(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    out = tmp_path / "out.wav"
    rc = _run(
        monkeypatch,
        tmp_path,
        "say",
        "你好",
        "--backend",
        "dashscope",
        "--voice",
        "builtin:Cherry",
        "--out",
        str(out),
    )
    assert rc == 3
    assert not out.is_file()
    capsys.readouterr()


def test_dashscope_offline_is_usage_error(monkeypatch, tmp_path, dashscope, capsys):
    rc = _run(
        monkeypatch,
        tmp_path,
        "say",
        "--offline",
        "你好",
        "--backend",
        "dashscope",
        "--voice",
        "builtin:Cherry",
        "--dry-run",
        "--json",
    )
    assert rc == 2
    assert dashscope.clients == 0
    capsys.readouterr()


def test_dashscope_subtitles_use_local_aligner(
    monkeypatch, tmp_path, model_dir, capsys
):
    rc = _run(
        monkeypatch,
        tmp_path,
        "say",
        "--dry-run",
        "--backend",
        "dashscope",
        "--voice",
        "builtin:Cherry",
        "你好",
        "--subtitle",
        "srt",
        "--json",
        model_dir=model_dir,
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["aligner"]["model"] == "Qwen3-ForcedAligner-0.6B"

    empty = tmp_path / "empty_models"
    empty.mkdir()
    out = tmp_path / "out.wav"
    rc = _run(
        monkeypatch,
        tmp_path,
        "say",
        "--dry-run",
        "--backend",
        "dashscope",
        "--voice",
        "builtin:Cherry",
        "你好",
        "--subtitle",
        "srt",
        "--out",
        str(out),
        model_dir=empty,
    )
    assert rc == 3
    assert not out.is_file()
    capsys.readouterr()


def test_jsonl_dashscope_reuses_one_client(
    monkeypatch, tmp_path, dashscope, capsys
):
    lines = "\n".join(
        [
            json.dumps({"id": "a", "text": "你好"}),
            json.dumps({"id": "b", "text": "世界"}),
        ]
    ) + "\n"
    rc = _run(
        monkeypatch,
        tmp_path,
        "say",
        "--stdin",
        "--jsonl",
        "--backend",
        "dashscope",
        "--voice",
        "builtin:Cherry",
        "--out-dir",
        str(tmp_path / "out"),
        stdin=lines,
    )
    assert rc == 0
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [record["status"] for record in records] == ["ok", "ok"]
    assert dashscope.clients == 1
    assert len(dashscope.connect_kwargs) == 2


def test_batch_plan_records_backend_identity_and_mismatch_is_clean(
    monkeypatch, tmp_path, model_dir, dashscope, capsys
):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"id": "a", "text": "你好", "voice": "builtin:Cherry"}) + "\n",
        encoding="utf-8",
    )
    run_dir = tmp_path / "run"
    rc = _run(
        monkeypatch,
        tmp_path,
        "batch",
        "plan",
        str(manifest),
        "--out-dir",
        str(run_dir),
        "--backend",
        "dashscope",
        "--backend-model",
        "qwen3-tts-flash-realtime",
        "--json",
        model_dir=model_dir,
    )
    assert rc == 0
    capsys.readouterr()
    plan = json.loads((run_dir / "plan.json").read_text(encoding="utf-8"))
    assert plan["backend"] == "dashscope"
    assert plan["backend_model"] == "qwen3-tts-flash-realtime"
    assert plan["backend_identity"].startswith("dashscope:")
    job = plan["jobs"][0]
    assert job["backend"] == "dashscope"
    assert job["backend_model"] == "qwen3-tts-flash-realtime"
    assert job["backend_identity"] == plan["backend_identity"]

    plan_before = (run_dir / "plan.json").read_bytes()
    status_before = (run_dir / "status.jsonl").read_bytes()
    rc = _run(
        monkeypatch,
        tmp_path,
        "batch",
        "run",
        str(manifest),
        "--out-dir",
        str(run_dir),
        "--resume",
        "--backend",
        "transformers",
        "--json",
        model_dir=model_dir,
    )
    assert rc == 2
    assert (run_dir / "plan.json").read_bytes() == plan_before
    assert (run_dir / "status.jsonl").read_bytes() == status_before
    capsys.readouterr()


def test_batch_plan_does_not_need_key_but_run_does(
    monkeypatch, tmp_path, model_dir, capsys
):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"id": "a", "text": "你好", "voice": "builtin:Cherry"}) + "\n",
        encoding="utf-8",
    )
    run_dir = tmp_path / "run"
    rc = _run(
        monkeypatch,
        tmp_path,
        "batch",
        "plan",
        str(manifest),
        "--out-dir",
        str(run_dir),
        "--backend",
        "dashscope",
        "--json",
        model_dir=model_dir,
    )
    assert rc == 0
    capsys.readouterr()
    plan_before = (run_dir / "plan.json").read_bytes()
    status_before = (run_dir / "status.jsonl").read_bytes()

    rc = _run(
        monkeypatch,
        tmp_path,
        "batch",
        "run",
        str(manifest),
        "--out-dir",
        str(run_dir),
        "--backend",
        "dashscope",
        "--json",
        model_dir=model_dir,
    )
    assert rc == 3
    assert (run_dir / "plan.json").read_bytes() == plan_before
    assert (run_dir / "status.jsonl").read_bytes() == status_before
    capsys.readouterr()


@pytest.mark.parametrize("kind", ["unknown-backend", "bad-manifest"])
def test_batch_plan_errors_leave_no_out_dir(
    monkeypatch, tmp_path, model_dir, capsys, kind
):
    monkeypatch.setattr(plugins, "_entry_points", lambda group: [])
    manifest = tmp_path / "manifest.jsonl"
    if kind == "bad-manifest":
        manifest.write_text("{not json}\n", encoding="utf-8")
    else:
        manifest.write_text(
            json.dumps({"id": "a", "text": "你好", "voice": "builtin:Vivian"})
            + "\n",
            encoding="utf-8",
        )
    run_dir = tmp_path / "fresh-run"
    extra = ("--backend", "nope") if kind == "unknown-backend" else ()
    rc = _run(
        monkeypatch,
        tmp_path,
        "batch",
        "plan",
        str(manifest),
        "--out-dir",
        str(run_dir),
        *extra,
        "--json",
        model_dir=model_dir,
    )
    assert rc == 2
    assert not run_dir.exists()
    capsys.readouterr()


# --------------------------------------------------------------------------- #
# Plugin backend
# --------------------------------------------------------------------------- #
class _PluginEngine:
    loads = 0
    name = "demo"
    capabilities = BackendCapabilities(
        modes=frozenset({"builtin"}),
        instruct_modes=frozenset(),
        sampling=frozenset(),
        remote=True,
    )

    def __init__(self, cfg, model):
        self.model_info = model
        self.model = None
        self.device = "cpu"
        self.dtype_name = "fp32"
        self.load_ms = 0.0
        self.calls = []

    def load(self):
        _PluginEngine.loads += 1
        self.model = object()
        return self

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return np.zeros(2400, dtype=np.float32), 24000, {"generate_ms": 1.0}

    def build_clone_prompt_items(self, *args, **kwargs):
        return []

    def release(self):
        self.model = None


CAPS = BackendCapabilities(
    modes=frozenset({"builtin"}),
    instruct_modes=frozenset(),
    sampling=frozenset(),
    remote=True,
)

PLUGIN_MODEL = ModelInfo(
    path=Path("<plugin:demo>"),
    name="plugin-model",
    model_type="qwen3_tts",
    tts_model_type="custom_voice",
    kind="builtin",
    speakers=["Vivian"],
    languages=["Chinese", "English"],
)


class _DemoBackend:
    api_version = 1
    name = "demo"
    seen_backend_models = []
    engines = []

    @staticmethod
    def capabilities():
        return CAPS

    @staticmethod
    def models(cfg, backend_model=None):
        _DemoBackend.seen_backend_models.append(backend_model)
        return [PLUGIN_MODEL]

    @staticmethod
    def create(cfg, model):
        engine = _PluginEngine(cfg, model)
        _DemoBackend.engines.append(engine)
        return engine


def _install_backend_plugin(monkeypatch):
    class EntryPoint:
        name = "demo"
        dist = SimpleNamespace(name="demo-qtts", version="1.0.0")

        def load(self):
            return _DemoBackend

    monkeypatch.setattr(
        plugins,
        "_entry_points",
        lambda group: [EntryPoint()] if group == plugins.BACKEND_GROUP else [],
    )


def test_plugin_backend_end_to_end(monkeypatch, tmp_path, capsys):
    _DemoBackend.engines = []
    _DemoBackend.seen_backend_models = []
    _install_backend_plugin(monkeypatch)
    out = tmp_path / "out.wav"
    rc = _run(
        monkeypatch,
        tmp_path,
        "say",
        "你好",
        "--backend",
        "demo",
        "--backend-model",
        "custom",
        "--voice",
        "builtin:Vivian",
        "--out",
        str(out),
        "--json",
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["backend"]["identity"] == "plugin:demo@1.0.0:custom"
    assert payload["backend"]["remote"] is True
    assert payload["device"] == "remote"
    assert payload["dtype"] is None
    assert _DemoBackend.seen_backend_models == ["custom"]
    assert _DemoBackend.engines[0].calls[0]["mode"] == "builtin"
    # CAPS declares no sampling keys, so the plan defaults must be filtered.
    assert _DemoBackend.engines[0].calls[0]["sampling"] == {}
    recipe = json.loads(out.with_suffix(".recipe.json").read_text(encoding="utf-8"))
    assert recipe["backend"]["identity"] == "plugin:demo@1.0.0:custom"


def test_plugin_backend_rejects_local_model_flag(monkeypatch, tmp_path, capsys):
    _install_backend_plugin(monkeypatch)
    rc = _run(
        monkeypatch,
        tmp_path,
        "say",
        "你好",
        "--backend",
        "demo",
        "--model",
        "local",
        "--voice",
        "builtin:Vivian",
        "--dry-run",
        "--json",
    )
    assert rc == 2
    capsys.readouterr()


def _install_plugin_object(monkeypatch, obj, *, entry_point_name="demo"):
    class EntryPoint:
        name = entry_point_name
        dist = SimpleNamespace(name="demo-qtts", version="1.0.0")

        def load(self):
            return obj

    monkeypatch.setattr(
        plugins,
        "_entry_points",
        lambda group: [EntryPoint()] if group == plugins.BACKEND_GROUP else [],
    )


def _plugin_class(*, models=None, create=None, capabilities=None):
    values = {
        "api_version": 1,
        "name": "demo",
        "models": staticmethod(models or (lambda cfg, backend_model=None: [PLUGIN_MODEL])),
        "create": staticmethod(create or (lambda cfg, model: _PluginEngine(cfg, model))),
    }
    if capabilities is not None:
        values["capabilities"] = staticmethod(capabilities)
    return type("InlinePlugin", (), values)


def _say_plugin(tmp_path, *extra):
    return [
        "say",
        "你好",
        "--backend",
        "demo",
        "--voice",
        "builtin:Vivian",
        *extra,
    ]


@pytest.mark.parametrize(
    "plugin, expected",
    [
        (
            _plugin_class(models=lambda cfg, backend_model=None: [object()]),
            "ModelInfo",
        ),
        (
            _plugin_class(
                capabilities=lambda: (_ for _ in ()).throw(
                    RuntimeError("caps exploded")
                )
            ),
            "capabilities() failed",
        ),
    ],
)
def test_plugin_protocol_violations_are_single_json_exit_three(
    monkeypatch, tmp_path, capsys, plugin, expected
):
    _install_plugin_object(monkeypatch, plugin)
    rc = _run(
        monkeypatch,
        tmp_path,
        *_say_plugin(tmp_path, "--dry-run", "--json"),
    )
    assert rc == 3
    captured = capsys.readouterr()
    assert captured.out == ""
    payload = json.loads(captured.err)
    assert payload["ok"] is False
    assert payload["code"] == 3
    assert expected in payload["message"]
    assert "Traceback" not in captured.err


@pytest.mark.parametrize(
    "create",
    [
        lambda cfg, model: None,
        lambda cfg, model: SimpleNamespace(
            model_info=model, generate=lambda **kw: None, release=lambda: None
        ),
        lambda cfg, model: SimpleNamespace(
            model_info=model,
            load=lambda: None,
            generate=lambda **kw: None,
            release=lambda: None,
        ),
    ],
    ids=["none", "missing-load", "missing-capabilities"],
)
def test_plugin_create_protocol_violations_exit_three(
    monkeypatch, tmp_path, model_dir, capsys, create
):
    _install_plugin_object(monkeypatch, _plugin_class(create=create))
    out = tmp_path / "out.wav"
    rc = _run(
        monkeypatch,
        tmp_path,
        *_say_plugin(tmp_path, "--out", str(out), "--json"),
        model_dir=model_dir,
    )
    assert rc == 3
    assert not out.is_file()
    captured = capsys.readouterr()
    assert captured.out == ""
    payload = json.loads(captured.err)
    assert payload["code"] == 3
    assert "Traceback" not in captured.err


def _protocol_engine(
    model, *, omit=(), bad_model_info=False, load_error=False, generate_error=False
):
    class Engine:
        name = "demo"
        model = None
        device = "cpu"
        dtype_name = "fp32"
        load_ms = 0.0
        capabilities = CAPS

        def __init__(self):
            self.model_info = object() if bad_model_info else model

        def load(self):
            if load_error:
                raise RuntimeError("plugin load exploded")
            self.model = object()
            return self

        def generate(self, **kwargs):
            if generate_error:
                raise RuntimeError("plugin generate exploded")
            return np.zeros(2400, dtype=np.float32), 24000, {"generate_ms": 1.0}

        def build_clone_prompt_items(self, *args, **kwargs):
            return None

        def release(self):
            self.model = None

    for attribute in omit:
        delattr(Engine, attribute)
    return Engine()


@pytest.mark.parametrize(
    "factory_kwargs",
    [
        {"omit": ("load_ms",)},
        {"omit": ("build_clone_prompt_items",)},
        {"bad_model_info": True},
        {"load_error": True},
        {"generate_error": True},
    ],
    ids=[
        "missing-load-ms",
        "missing-clone-method",
        "bad-model-info",
        "load-raises",
        "generate-raises",
    ],
)
def test_plugin_protocol_and_call_failures_are_single_json_exit_three(
    monkeypatch, tmp_path, model_dir, capsys, factory_kwargs
):
    def create(cfg, model):
        return _protocol_engine(model, **factory_kwargs)

    _install_plugin_object(monkeypatch, _plugin_class(create=create))
    out = tmp_path / "out.wav"
    rc = _run(
        monkeypatch,
        tmp_path,
        *_say_plugin(tmp_path, "--out", str(out), "--json"),
        model_dir=model_dir,
    )
    assert rc == 3
    assert not out.is_file()
    captured = capsys.readouterr()
    assert captured.out == ""
    payload = json.loads(captured.err)
    assert payload["ok"] is False
    assert payload["code"] == 3
    assert "Traceback" not in captured.err


def test_reserved_plugin_name_never_shadows_the_builtin(
    monkeypatch, tmp_path, model_dir, capsys
):
    plugin = _plugin_class()
    plugin.name = "transformers"
    _install_plugin_object(monkeypatch, plugin, entry_point_name="transformers")
    rc = _run(
        monkeypatch,
        tmp_path,
        "say",
        "你好",
        "--backend",
        "transformers",
        "--voice",
        "builtin:Vivian",
        "--dry-run",
        "--json",
        model_dir=model_dir,
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["backend"]["name"] == "transformers"


# --------------------------------------------------------------------------- #
# Scripts and vLLM-Omni
# --------------------------------------------------------------------------- #
def _write_script(tmp_path: Path, voice: str) -> Path:
    path = tmp_path / "script.yaml"
    path.write_text(
        (
            "schema_version: 1\n"
            "defaults:\n"
            "  language: Chinese\n"
            "segments:\n"
            "  - id: one\n"
            f"    text: \"你好。\"\n"
            f"    voice: \"{voice}\"\n"
        ),
        encoding="utf-8",
    )
    return path


def _write_multi_voice_script(tmp_path: Path, clone="alice") -> Path:
    path = tmp_path / "multi.yaml"
    path.write_text(
        (
            "schema_version: 1\n"
            "defaults:\n"
            "  language: Chinese\n"
            "  voice: \"builtin:Vivian\"\n"
            "segments:\n"
            "  - id: narrator\n"
            "    text: \"你好。\"\n"
            "  - id: guest\n"
            "    text: \"世界。\"\n"
            f"    voice: \"voice:{clone}\"\n"
        ),
        encoding="utf-8",
    )
    return path


def _write_clone_voice(tmp_path: Path, name="alice", *, ref="ref.wav", mode="xvec"):
    voice_dir = tmp_path / "data" / "voices" / name
    voice_dir.mkdir(parents=True, exist_ok=True)
    source = {"type": "clone", "mode": mode, "ref_audio": ref}
    if mode == "icl":
        source["ref_text"] = "hello"
    (voice_dir / "voice.json").write_text(
        json.dumps(
            {"schema_version": 1, "name": name, "kind": "clone", "source": source}
        ),
        encoding="utf-8",
    )
    return voice_dir


def test_script_compile_vllm_accepts_ref_wav_only(monkeypatch, tmp_path, model_dir, capsys):
    directory = _write_clone_voice(tmp_path)
    (directory / "ref.wav").write_bytes(b"RIFF")
    script = _write_script(tmp_path, "voice:alice")
    out = tmp_path / "plan.json"
    rc = _run(
        monkeypatch,
        tmp_path,
        "script",
        "compile",
        str(script),
        "-o",
        str(out),
        "--backend",
        "vllm-omni",
        model_dir=model_dir,
    )
    assert rc == 0
    plan = json.loads(out.read_text(encoding="utf-8"))
    assert plan["segments"][0]["cache_key"]
    capsys.readouterr()


def test_script_compile_vllm_missing_or_escaping_ref_exits_four(
    monkeypatch, tmp_path, model_dir, capsys
):
    _write_clone_voice(tmp_path)
    script = _write_script(tmp_path, "voice:alice")
    out = tmp_path / "plan.json"
    rc = _run(
        monkeypatch,
        tmp_path,
        "script",
        "compile",
        str(script),
        "-o",
        str(out),
        "--backend",
        "vllm-omni",
        model_dir=model_dir,
    )
    assert rc == 4
    assert not out.is_file()

    (tmp_path / "data" / "voices" / "evil.wav").write_bytes(b"RIFF")
    directory = _write_clone_voice(tmp_path, "bob", ref="../evil.wav")
    assert directory.is_dir()
    script = _write_script(tmp_path, "voice:bob")
    rc = _run(
        monkeypatch,
        tmp_path,
        "script",
        "compile",
        str(script),
        "-o",
        str(out),
        "--backend",
        "vllm-omni",
        model_dir=model_dir,
    )
    assert rc == 4
    assert not out.is_file()
    capsys.readouterr()


def test_say_script_vllm_dry_run_and_missing_ref(
    monkeypatch, tmp_path, model_dir, capsys
):
    directory = _write_clone_voice(tmp_path)
    (directory / "ref.wav").write_bytes(b"RIFF")
    script = _write_script(tmp_path, "voice:alice")
    rc = _run(
        monkeypatch,
        tmp_path,
        "say",
        "--script",
        str(script),
        "--backend",
        "vllm-omni",
        "--dry-run",
        "--json",
        model_dir=model_dir,
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["backend"]["name"] == "vllm-omni"

    (directory / "ref.wav").unlink()
    rc = _run(
        monkeypatch,
        tmp_path,
        "say",
        "--script",
        str(script),
        "--backend",
        "vllm-omni",
        "--dry-run",
        "--json",
        model_dir=model_dir,
    )
    assert rc == 4
    capsys.readouterr()


@pytest.mark.parametrize("command", ["say", "compile"])
@pytest.mark.parametrize("case", ["missing", "escaping"])
def test_multi_voice_script_clone_preflight_before_load(
    monkeypatch, tmp_path, model_dir, capsys, command, case
):
    ref = "ref.wav"
    if case == "escaping":
        ref = "../evil.wav"
        (tmp_path / "data" / "voices").mkdir(parents=True, exist_ok=True)
        (tmp_path / "data" / "voices" / "evil.wav").write_bytes(b"RIFF")
    _write_clone_voice(tmp_path, ref=ref)
    script = _write_multi_voice_script(tmp_path)
    calls = []
    monkeypatch.setattr(
        vo, "_omni_factory", lambda *args, **kwargs: calls.append(args) or object()
    )
    out = tmp_path / ("out.wav" if command == "say" else "plan.json")
    if command == "say":
        args = [
            "say",
            "--script",
            str(script),
            "--backend",
            "vllm-omni",
            "--dry-run",
            "--out",
            str(out),
            "--json",
        ]
    else:
        args = [
            "script",
            "compile",
            str(script),
            "-o",
            str(out),
            "--backend",
            "vllm-omni",
            "--json",
        ]
    rc = _run(monkeypatch, tmp_path, *args, model_dir=model_dir)
    assert rc == 4
    assert calls == []
    assert not out.exists()
    capsys.readouterr()


def test_script_cache_key_contains_backend_identity(
    monkeypatch, tmp_path, model_dir, capsys
):
    script = _write_script(tmp_path, "builtin:Vivian")
    keys = []
    for backend in ("transformers", "vllm-omni"):
        out = tmp_path / f"{backend}.plan.json"
        rc = _run(
            monkeypatch,
            tmp_path,
            "script",
            "compile",
            str(script),
            "-o",
            str(out),
            "--backend",
            backend,
            model_dir=model_dir,
        )
        assert rc == 0
        plan = json.loads(out.read_text(encoding="utf-8"))
        keys.append(plan["segments"][0]["cache_key"])
    assert keys[0] != keys[1]
    capsys.readouterr()


def test_script_dashscope_passthrough_warns_and_clone_exits_four(
    monkeypatch, tmp_path, model_dir, dashscope, capsys
):
    script = _write_script(tmp_path, "builtin:NotKnown")
    out = tmp_path / "plan.json"
    rc = _run(
        monkeypatch,
        tmp_path,
        "script",
        "compile",
        str(script),
        "-o",
        str(out),
        "--backend",
        "dashscope",
        model_dir=model_dir,
    )
    assert rc == 0
    plan = json.loads(out.read_text(encoding="utf-8"))
    assert any("passing it through" in warning for warning in plan["warnings"])
    out.unlink()

    clone = _write_script(tmp_path, "voice:alice")
    rc = _run(
        monkeypatch,
        tmp_path,
        "script",
        "compile",
        str(clone),
        "-o",
        str(out),
        "--backend",
        "dashscope",
        model_dir=model_dir,
    )
    assert rc == 4
    assert not out.is_file()
    assert dashscope.clients == 0
    capsys.readouterr()
