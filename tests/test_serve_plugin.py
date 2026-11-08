"""Local plugin backend residency A -> B -> A through real ``say.run`` (§11.4)."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from qtts_cli import plugins, serve
from qtts_cli.backends.base import BackendCapabilities
from qtts_cli.catalog import ModelInfo
from qtts_cli.config import Config

from test_serve_helpers import Client, serve_args, wait_job

CAPS = BackendCapabilities(
    modes=frozenset({"builtin", "design"}),
    instruct_modes=frozenset({"design"}),
    sampling=frozenset(),
    remote=False,
)

MODEL_A = ModelInfo(
    path=Path("<plugin:A>"),
    name="plugin-A",
    model_type="qwen3_tts",
    tts_model_type="custom_voice",
    kind="builtin",
    speakers=["Vivian"],
    languages=["Chinese", "English"],
    supports_instruct=True,
)
MODEL_B = ModelInfo(
    path=Path("<plugin:B>"),
    name="plugin-B",
    model_type="qwen3_tts",
    tts_model_type="voice_design",
    kind="design",
    languages=["Chinese", "English"],
    supports_instruct=True,
)


class Tracker:
    live: set[str] = set()
    peak = 0
    generated: list[str] = []
    released: list[str] = []


class Engine:
    name = "demo"
    device = "cpu"
    dtype_name = "fp32"
    load_ms = 0.0
    capabilities = CAPS
    model = None

    def __init__(self, model):
        self.model_info = model

    def load(self):
        self.model = object()
        return self

    def generate(self, **kwargs):
        tracker = Tracker
        name = self.model_info.name
        tracker.live.add(name)
        tracker.peak = max(tracker.peak, len(tracker.live))
        tracker.generated.append(name)
        return np.zeros(2400, dtype=np.float32), 24000, {"generate_ms": 1.0}

    def build_clone_prompt_items(self, *args, **kwargs):
        return []

    def release(self):
        Tracker.live.discard(self.model_info.name)
        Tracker.released.append(self.model_info.name)
        self.model = None


class DemoPlugin:
    api_version = 1
    name = "demo"

    @staticmethod
    def capabilities():
        return CAPS

    @staticmethod
    def models(cfg, backend_model=None):
        return [MODEL_A, MODEL_B]

    @staticmethod
    def create(cfg, model):
        return Engine(model)


def test_local_plugin_never_keeps_two_models_and_releases(monkeypatch, tmp_path: Path):
    Tracker.live = set()
    Tracker.peak = 0
    Tracker.generated = []
    Tracker.released = []

    class EntryPoint:
        name = "demo"
        dist = SimpleNamespace(name="demo-qtts", version="1.0.0")

        def load(self):
            return DemoPlugin

    monkeypatch.setattr(
        plugins,
        "_entry_points",
        lambda group: [EntryPoint()] if group == plugins.BACKEND_GROUP else [],
    )

    cfg = Config(
        data_dir=tmp_path / "data",
        cache_dir=tmp_path / "cache",
        model_dir=tmp_path / "models",
        backend="demo",
        no_cache=True,
    )
    args = serve_args(tmp_path, max_concurrency=4)
    stop = threading.Event()
    ready: dict = {}
    errors: list[BaseException] = []

    def target() -> None:
        try:
            serve.run(
                cfg,
                args,
                guard_stdout=False,
                stop_event=stop,
                ready_callback=lambda server, state: ready.update(
                    server=server, state=state
                ),
            )
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    deadline = time.monotonic() + 15
    while "state" not in ready and time.monotonic() < deadline:
        if errors:
            raise errors[0]
        time.sleep(0.01)
    assert "state" in ready
    assert ready["state"].effective_max_concurrency == 1

    try:
        client = Client("127.0.0.1", ready["server"].server_address[1])
        for voice in ("builtin:Vivian", "design:calm", "builtin:Vivian"):
            created = client.request(
                "POST", "/v1/jobs", {"text": "你好", "voice": voice}
            )
            assert created.status == 202, created.text
            job = wait_job(client, created.json["job"]["id"])
            assert job["status"] == "done", job
        assert Tracker.generated == ["plugin-A", "plugin-B", "plugin-A"]
        assert Tracker.peak == 1
        assert Tracker.released[:2] == ["plugin-A", "plugin-B"]
    finally:
        stop.set()
        thread.join(15)
    assert not thread.is_alive()
    assert not errors
