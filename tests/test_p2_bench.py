"""FR-P2-06 benchmark tests: no real models, no torch/qwen imports."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from qtts_cli import benchmark
from qtts_cli.benchmark import BenchCase
from qtts_cli.catalog import ModelInfo
from qtts_cli.cli import main
from qtts_cli.commands import bench as bench_cmd
from qtts_cli.errors import ModelError
from qtts_cli.utils import peak_rss_bytes

REPO = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _model(name: str, kind: str, **kw) -> ModelInfo:
    values = {
        "path": Path("/nonexistent") / name,
        "name": name,
        "model_type": "qwen3_tts" if kind != "asr" else "qwen3_asr",
        "tts_model_type": {
            "builtin": "custom_voice",
            "clone": "base",
            "design": "voice_design",
        }.get(kind),
        "size": "1b7",
        "speakers": ["Vivian", "Serena"] if kind == "builtin" else [],
        "languages": ["Chinese", "English"],
        "supports_instruct": kind in ("builtin", "design"),
        "revision": "rev1",
        "fingerprint": "fp1",
        "kind": kind,
    }
    values.update(kw)
    return ModelInfo(**values)


def _models() -> list[ModelInfo]:
    return [
        _model("Qwen3-TTS-CustomVoice-1.7B", "builtin"),
        _model("Qwen3-TTS-Base-1.7B", "clone"),
        _model("Qwen3-TTS-VoiceDesign-1.7B", "design"),
        _model("Qwen3-ASR-0.6B", "asr"),
    ]


class FakeClock:
    def __init__(self, step: float = 0.1) -> None:
        self.step = step
        self.value = 0.0

    def __call__(self) -> float:
        current = self.value
        self.value += self.step
        return current


class FakeEngine:
    def __init__(
        self,
        *,
        log: list | None = None,
        durations: list[float] | None = None,
        fail: bool = False,
        embeddings: list | None = None,
        load_ms: float = 12.5,
    ) -> None:
        self.log = log if log is not None else []
        self.durations = list(durations or [])
        self.fail = fail
        self.embeddings = list(embeddings or [])
        self.load_ms = load_ms
        self.model = object()
        self.device = "cpu"
        self.dtype_name = "fp32"

    def load(self):
        self.log.append("load")
        return self

    def build_clone_prompt_items(self, store, name):
        self.log.append(("prompt", name))
        return ["prompt"]

    def require_speaker_embedding(self):
        self.log.append("speaker_probe")

    def extract_speaker_embedding(self, samples, sr):
        self.log.append("embedding")
        if not self.embeddings:
            return np.ones(4, dtype=np.float32)
        return self.embeddings.pop(0)

    def generate(self, *, text, language, mode, **kwargs):
        self.log.append(("generate", text, language))
        if self.fail:
            raise ModelError("generation blew up")
        duration = self.durations.pop(0) if self.durations else 0.5
        self.durations.append(duration)
        samples = np.zeros(max(1, int(24000 * duration)), dtype=np.float32)
        info = {
            "generate_ms": 50.0,
            "sample_rate": 24000,
            "duration_s": round(duration, 3),
            "mode": mode,
            "language": language,
        }
        return samples, 24000, info


class FakeAsr:
    def __init__(
        self,
        texts: list[str] | None = None,
        *,
        result=None,
        log: list | None = None,
        router=None,
    ) -> None:
        self.texts = list(texts or [])
        self.result = result
        self.router = router
        self.log = log if log is not None else []
        self.languages: list = []
        # (samples identity, sample count, language) per call, so tests can
        # prove each case's own audio and language reached the ASR.
        self.calls: list[tuple[int, int, object]] = []

    def transcribe_samples(self, samples, sr, *, language=None):
        self.calls.append((id(samples), len(samples), language))
        self.languages.append(language)
        self.log.append("transcribe")
        if self.result is not None:
            return self.result
        if self.router is not None:
            text = self.router(samples, sr, language=language)
        else:
            text = self.texts.pop(0) if self.texts else ""
        return {"text": text, "language": language, "items": []}

    def release(self):
        self.log.append("release")


class FakeStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.prompt_vectors = {
            "alice": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        }
        self.has_ref_wav = False

    def dir_for(self, name: str) -> Path:
        path = self.root / name
        if self.has_ref_wav:
            path.mkdir(parents=True, exist_ok=True)
            (path / "ref.wav").write_bytes(b"fake")
        return path

    def require(self, name: str):
        if name != "alice":
            from qtts_cli.errors import InputError

            raise InputError(f"voice not found: {name}")
        return {
            "name": name,
            "kind": "clone",
            "source": {"ref_text": "hello", "mode": "icl"},
        }

    def prompt_sha256(self, name: str):
        return "abc" if name in self.prompt_vectors else None

    def load_prompt_tensors(self, name: str):
        return {"ref_spk_embedding": self.prompt_vectors[name]}


def _install_fake_runtime(
    monkeypatch,
    *,
    models: list[ModelInfo] | None = None,
    engine: FakeEngine | None = None,
    asr: FakeAsr | None = None,
    store: FakeStore | None = None,
    log: list | None = None,
) -> dict:
    state = {
        "models": models if models is not None else _models(),
        "engine": engine,
        "asr": asr,
        "log": log if log is not None else [],
        "factories": [],
    }
    monkeypatch.setattr(bench_cmd, "scan", lambda _dir: state["models"])
    if store is not None:
        monkeypatch.setattr(bench_cmd, "VoiceStore", lambda _dir: store)

    def factory(cfg, model):
        state["factories"].append(model.kind)
        state["log"].append(("engine_factory", model.kind))
        if model.kind == "clone" and state.get("base_engine") is not None:
            return state["base_engine"]
        if state["engine"] is None:
            state["engine"] = FakeEngine(log=state["log"])
        return state["engine"]

    monkeypatch.setattr(bench_cmd, "_engine_factory", factory)
    def asr_factory(cfg, model):
        state["log"].append(("asr_factory", model.kind))
        asr = state["asr"] or FakeAsr()
        asr.log = state["log"]
        return asr

    monkeypatch.setattr(bench_cmd, "_asr_factory", asr_factory)
    monkeypatch.setattr(
        bench_cmd,
        "_preflight_asr_runtime",
        lambda: state["log"].append("preflight_asr"),
    )
    monkeypatch.setattr(
        bench_cmd,
        "_preflight_speaker",
        lambda cfg, model: state["log"].append(("preflight_speaker", model.kind)),
    )
    monkeypatch.setattr(
        bench_cmd, "_clear_model_cache", lambda: state["log"].append("clear_model")
    )
    monkeypatch.setattr(
        bench_cmd, "_clear_asr_cache", lambda: state["log"].append("clear_asr")
    )
    monkeypatch.setattr(bench_cmd, "_prepare_device_memory", lambda: None)
    return state


def _run(tmp_path: Path, capsys, argv: list[str]) -> tuple[int, str, str, dict | None]:
    args = [
        "--data-dir",
        str(tmp_path / "data"),
        "--model-dir",
        str(tmp_path / "models"),
        "bench",
        *argv,
    ]
    if "--json" not in args:
        args.append("--json")
    code = main(args)
    captured = capsys.readouterr()
    payload = None
    if captured.out.strip():
        payload = json.loads(captured.out)
    elif captured.err.strip():
        payload = json.loads(captured.err)
    return code, captured.out, captured.err, payload


# --------------------------------------------------------------------------- #
# 1. Metric math
# --------------------------------------------------------------------------- #
def test_metric_math_uses_external_clock_and_medians():
    log: list = []
    # A stale self-reported load_ms must never override the external clock.
    engine = FakeEngine(log=log, durations=[1.0, 2.0], load_ms=9999)
    cases = [
        BenchCase("a", "one", "English", suite_index=1),
        BenchCase("b", "two", "English", suite_index=2),
    ]
    rss_values = [100, 200, 300]

    def rss():
        return rss_values.pop(0)

    result = benchmark.execute(
        engine,
        cases,
        2,
        mode="builtin",
        speaker="Vivian",
        clock=FakeClock(0.1),
        peak_rss=rss,
        baseline_rss=50,
    )
    metrics = result.metrics
    # Each run: two 0.1 s external generate calls; audio is 1 + 2 s.
    assert metrics["wall_s"] == 1.4
    assert metrics["generate_s"] == 0.4
    assert metrics["audio_s"] == 6.0
    assert metrics["rtf"] == 0.067
    assert metrics["wall_rtf"] == 0.233
    assert metrics["load_ms"] == 100.0
    assert metrics["warm_first_segment_ms"] == 100.0
    assert metrics["cold_first_segment_ms"] == 200.0
    assert metrics["segments_per_run"] == 2
    assert metrics["segments_total"] == 4
    assert metrics["segments_per_s"] == 10.0
    assert metrics["process_peak_rss_bytes"] == 300
    assert metrics["peak_rss_delta_bytes"] == 250
    assert [run["rtf"] for run in result.runs] == [0.067, 0.067]
    assert result.runs[1]["wall_s"] == 0.5
    assert result.cases[0]["generate_ms"] == 50.0
    assert result.cases[0]["rtf"] == 0.1
    assert result.cases[1]["audio_duration_s"] == 2.0
    assert log.count("load") == 1


def test_three_runs_median_and_totals():
    engine = FakeEngine(durations=[1.0], load_ms=1.0)
    cases = [BenchCase("a", "x", "English")]
    clock_values = iter(
        [
            0.0, 0.1,
            0.2, 0.3, 0.4, 0.5,
            0.6, 0.7, 1.0, 1.1,
            1.2, 1.3, 1.8, 1.9,
            2.0,
        ]
    )
    result = benchmark.execute(
        engine,
        cases,
        3,
        mode="builtin",
        clock=lambda: next(clock_values),
        peak_rss=lambda: 1,
        baseline_rss=1,
    )
    metrics = result.metrics
    assert [run["rtf"] for run in result.runs] == [0.1, 0.3, 0.5]
    assert metrics["rtf"] == 0.3
    assert metrics["warm_first_segment_ms"] == 300.0
    assert metrics["generate_s"] == 0.9
    assert metrics["audio_s"] == 3.0
    assert metrics["wall_s"] == 2.0
    assert metrics["wall_rtf"] == 0.667
    assert metrics["segments_total"] == 3
    assert metrics["segments_per_s"] == 3.333


def test_median_uses_unrounded_per_run_values():
    engine = FakeEngine(durations=[1.0], load_ms=1.0)
    cases = [BenchCase("a", "x", "English")]
    # run generate walls: 0.3 s and 0.6 s over 1 s audio -> rtfs .3 / .6
    clock_values = iter(
        [0.0, 0.1, 0.2, 0.3, 0.6, 0.65, 0.7, 0.8, 1.4, 1.5, 1.6]
    )
    result = benchmark.execute(
        engine,
        cases,
        2,
        mode="builtin",
        clock=lambda: next(clock_values),
        peak_rss=lambda: 1,
        baseline_rss=1,
    )
    assert result.metrics["rtf"] == 0.45
    assert result.metrics["warm_first_segment_ms"] == 450.0


# --------------------------------------------------------------------------- #
# 2. Corpora and longform planning
# --------------------------------------------------------------------------- #
def test_smoke_and_batch_corpora():
    assert len(benchmark.smoke_cases("auto")) == 2
    chinese = benchmark.smoke_cases("Chinese")
    assert len(chinese) == 1 and chinese[0].language == "Chinese"
    english = benchmark.smoke_cases("English")
    assert len(english) == 1 and english[0].language == "English"

    batch = benchmark.batch_cases("Chinese", 5)
    assert [case.id for case in batch] == [f"batch-{i:03d}" for i in range(1, 6)]
    assert all(case.language == "Chinese" for case in batch)
    assert benchmark.batch_cases("English", 10)[8].id == "batch-009"
    assert benchmark.batch_cases("French", 3) == []


def test_longform_builtin_corpora_and_segment_plan(tmp_path):
    assert 1200 <= len(benchmark.LONGFORM_ZH) <= 2600
    assert "城市" in benchmark.LONGFORM_ZH
    assert "city" in benchmark.LONGFORM_EN.lower()

    cfg = SimpleNamespace(seed=1234, dtype="fp32", device="cpu")
    store = SimpleNamespace(require=lambda name: pytest.fail("builtin used store"))
    cases, plan = benchmark.longform_cases(
        cfg,
        text="第一句。第二句！",
        voice_ref="builtin:Vivian",
        language="auto",
        models=[_model("builtin", "builtin")],
        store=store,
    )
    from qtts_cli.textnorm import NORMALIZATION_VERSION

    assert len(cases) == 2
    assert [case.language for case in cases] == ["Chinese", "Chinese"]
    assert [case.id for case in cases] == ["longform-s001", "longform-s002"]
    assert plan.voice.model.kind == "builtin"
    assert plan.normalization_version == NORMALIZATION_VERSION


# --------------------------------------------------------------------------- #
# 3. Dry run
# --------------------------------------------------------------------------- #
def test_dry_run_plans_without_loading_or_writing(tmp_path, monkeypatch, capsys):
    state = _install_fake_runtime(monkeypatch)
    before = sorted(str(p) for p in tmp_path.rglob("*"))
    code, out, err, payload = _run(
        tmp_path, capsys, ["--suite", "smoke", "--runs", "3", "--dry-run"]
    )
    assert code == 0
    assert payload["dry_run"] is True
    assert payload["runs"] == [] and payload["runs_count"] == 0
    assert payload["metrics"] == {key: None for key in bench_cmd.METRIC_KEYS}
    assert payload["runtime"] is None
    assert payload["requirements"] == {
        "tts_model": True,
        "asr_model": False,
        "speaker_model": False,
        "asr_metadata": False,
        "speaker_metadata": False,
        "runtime_checked": False,
    }
    assert all(case["run"] is None for case in payload["cases"])
    assert all(case["audio_duration_s"] is None for case in payload["cases"])
    assert state["engine"] is None
    assert state["log"] == []
    after = sorted(str(p) for p in tmp_path.rglob("*"))
    assert before == after
    assert err == ""


def test_dry_run_subprocess_does_not_import_torch_or_qwen(tmp_path):
    script = f"""
import json, sys
from pathlib import Path
from qtts_cli.catalog import ModelInfo
from qtts_cli.commands import bench
bench.scan = lambda _dir: [ModelInfo(path=Path('/nonexistent'), name='m',
    model_type='qwen3_tts', tts_model_type='custom_voice', size='1b7',
    speakers=['Vivian'], languages=['Chinese','English'], supports_instruct=True,
    revision='r', fingerprint='f', kind='builtin')]
from qtts_cli.cli import main
code = main(['--data-dir', {str(tmp_path / 'data')!r},
             '--model-dir', {str(tmp_path / 'models')!r},
             'bench', '--suite', 'smoke', '--dry-run', '--json'])
bad = sorted(m for m in sys.modules if m.split('.')[0] in {{'torch','qwen_tts','qwen_asr'}})
sys.stderr.write(json.dumps({{'code': code, 'bad': bad}}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    meta = json.loads(completed.stderr)
    assert meta == {"code": 0, "bad": []}
    assert sorted(str(p) for p in tmp_path.rglob("*")) == []


# --------------------------------------------------------------------------- #
# 4. Parameter combinations
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "argv",
    [
        ["--suite", "smoke", "--text", "x"],
        ["--suite", "smoke", "--count", "2"],
        ["--suite", "batch", "--count", "0"],
        ["--suite", "smoke", "--runs", "0"],
        ["--suite", "smoke", "--runs", "11"],
        ["--suite", "smoke", "--asr-model", "x"],
        ["--suite", "smoke", "--speaker-sim"],
        ["--suite", "smoke", "--reference-voice", "alice"],
        ["--suite", "smoke", "--max-cer", "0.1"],
        ["--suite", "smoke", "--min-speaker-similarity", "0.5"],
        ["--suite", "smoke", "--voice", "design:hello"],
        ["--suite", "smoke", "--report", "-"],
        ["--suite", "smoke", "--dry-run", "--report", "x.json"],
        ["--suite", "longform", "--text", "a", "--text-file", "b.txt"],
    ],
)
def test_usage_errors_exit_two(tmp_path, monkeypatch, capsys, argv):
    _install_fake_runtime(monkeypatch)
    code, out, err, payload = _run(tmp_path, capsys, argv)
    assert code == 2
    assert out == ""
    assert err.strip()
    assert payload is not None and payload["code"] == 2


def test_missing_suite_exits_two(tmp_path, capsys):
    code = main(["--json", "--data-dir", str(tmp_path), "bench"])
    captured = capsys.readouterr()
    assert code == 2
    assert json.loads(captured.err)["code"] == 2


# --------------------------------------------------------------------------- #
# 5. Success path and canonical report
# --------------------------------------------------------------------------- #
def test_success_schema_and_json_equals_report(tmp_path, monkeypatch, capsys):
    engine = FakeEngine(durations=[0.5, 0.7])
    _install_fake_runtime(monkeypatch, engine=engine)
    report = tmp_path / "bench.json"
    code, out, err, payload = _run(
        tmp_path,
        capsys,
        ["--suite", "smoke", "--runs", "2", "--report", str(report)],
    )
    assert code == 0
    assert err == ""
    assert report.read_text(encoding="utf-8") == out
    assert payload["schema_version"] == 1
    assert payload["ok"] is True
    assert payload["suite"] == "smoke"
    assert payload["dry_run"] is False
    assert payload["runs_count"] == 2
    assert payload["request"]["voice"] == "builtin:Vivian"
    assert payload["request"]["count"] is None
    assert payload["models"]["tts"]["name"] == "Qwen3-TTS-CustomVoice-1.7B"
    assert "path" not in payload["models"]["tts"]
    assert payload["models"]["asr"] is None
    assert payload["models"]["speaker"] is None
    assert payload["corpus"]["id"] == "smoke.v1"
    assert payload["aggregation"]["rtf"] == "median_per_run"
    assert len(payload["cases"]) == 4
    assert {case["run"] for case in payload["cases"]} == {1, 2}
    assert all(len(case["text_sha256"]) == 64 for case in payload["cases"])
    assert payload["gate_failures"] == []
    assert payload["scoring_ms"] == 0.0
    assert payload["runtime"]["python"]
    assert "/nonexistent" not in out


def test_batch_count_cycles_and_language_filter(tmp_path, monkeypatch, capsys):
    engine = FakeEngine(durations=[0.2] * 32)
    _install_fake_runtime(monkeypatch, engine=engine)
    code, _out, _err, payload = _run(
        tmp_path,
        capsys,
        ["--suite", "batch", "--count", "5", "--language", "English", "--runs", "2"],
    )
    assert code == 0
    assert payload["request"]["count"] == 5
    assert len(payload["cases"]) == 10
    assert payload["cases"][0]["id"] == "batch-001"
    assert all(case["language"] == "English" for case in payload["cases"])
    assert payload["cases"][-1]["id"] == "batch-005"


def test_longform_text_and_file_sources(tmp_path, monkeypatch, capsys):
    engine = FakeEngine(durations=[0.3] * 20)
    _install_fake_runtime(monkeypatch, engine=engine)
    code, _out, _err, payload = _run(
        tmp_path,
        capsys,
        ["--suite", "longform", "--text", "第一句。第二句！", "--dry-run"],
    )
    assert code == 0
    assert payload["request"]["text_source"] == "text"
    assert [case["language"] for case in payload["cases"]] == ["Chinese", "Chinese"]
    assert payload["corpus"]["normalization_version"] is not None
    assert payload["corpus"]["segmenter_version"] == bench_cmd.SEGMENTER_VERSION

    path = tmp_path / "long.txt"
    path.write_text("Hello world. This is a second sentence.", encoding="utf-8")
    code, _out, _err, payload = _run(
        tmp_path,
        capsys,
        ["--suite", "longform", "--text-file", str(path), "--language", "English", "--dry-run"],
    )
    assert code == 0
    assert payload["request"]["text_source"] == "text_file"
    assert all(case["language"] == "English" for case in payload["cases"])


# --------------------------------------------------------------------------- #
# 6. Model / generation / text-file failures
# --------------------------------------------------------------------------- #
def test_no_models_exit_three(tmp_path, monkeypatch, capsys):
    _install_fake_runtime(monkeypatch, models=[])
    code, out, _err, payload = _run(tmp_path, capsys, ["--suite", "smoke"])
    assert code == 3
    assert out == ""
    assert payload["code"] == 3


def test_model_selection_exit_codes(tmp_path, monkeypatch, capsys):
    models = _models() + [
        _model("first-ambiguous", "builtin"),
        _model("second-ambiguous", "builtin"),
    ]
    _install_fake_runtime(monkeypatch, models=models)

    code, _out, _err, payload = _run(
        tmp_path, capsys, ["--suite", "smoke", "--model", "missing"]
    )
    assert code == 3 and payload["code"] == 3

    # An explicit path must never silently fall back to a same-basename model.
    code, _out, _err, payload = _run(
        tmp_path,
        capsys,
        ["--suite", "smoke", "--model", "/definitely/missing/Qwen3-TTS-CustomVoice-1.7B"],
    )
    assert code == 3 and payload["code"] == 3

    code, _out, _err, payload = _run(
        tmp_path, capsys, ["--suite", "smoke", "--model", "ambiguous"]
    )
    assert code == 2 and payload["code"] == 2

    code, _out, _err, payload = _run(
        tmp_path, capsys, ["--suite", "smoke", "--model", "Qwen3-TTS-Base-1.7B"]
    )
    assert code == 2 and payload["code"] == 2

    code, _out, _err, payload = _run(
        tmp_path, capsys, ["--suite", "smoke", "--voice", "builtin:Nobody"]
    )
    assert code == 4 and payload["code"] == 4


def test_asr_selection_and_non_tts_model_exit_codes(tmp_path, monkeypatch, capsys):
    models = _models() + [_model("Qwen3-ASR-1.7B", "asr")]
    _install_fake_runtime(monkeypatch, models=models)

    code, _out, _err, payload = _run(
        tmp_path, capsys, ["--suite", "smoke", "--cer"]
    )
    assert code == 2 and payload["code"] == 2

    code, _out, _err, payload = _run(
        tmp_path, capsys, ["--suite", "smoke", "--model", "Qwen3-ASR-0.6B"]
    )
    assert code == 3 and payload["code"] == 3

    code, _out, _err, payload = _run(
        tmp_path, capsys, ["--suite", "smoke", "--cer", "--asr-model", "missing"]
    )
    assert code == 3 and payload["code"] == 3

    code, _out, _err, payload = _run(
        tmp_path,
        capsys,
        [
            "--suite",
            "smoke",
            "--cer",
            "--asr-model",
            "/definitely/missing/Qwen3-ASR-0.6B",
        ],
    )
    assert code == 3 and payload["code"] == 3

    _install_fake_runtime(
        monkeypatch,
        models=[_model("builtin", "builtin"), _model("base", "clone")],
    )
    code, _out, _err, payload = _run(tmp_path, capsys, ["--suite", "smoke", "--cer"])
    assert code == 3 and payload["code"] == 3


def test_capability_and_reference_profile_errors_exit_four(tmp_path, monkeypatch, capsys):
    chinese_only = _model(
        "Qwen3-TTS-CustomVoice-1.7B", "builtin", languages=["Chinese"]
    )
    _install_fake_runtime(
        monkeypatch, models=[chinese_only, _model("Qwen3-TTS-Base-1.7B", "clone")]
    )
    code, _out, _err, payload = _run(
        tmp_path, capsys, ["--suite", "smoke", "--language", "English"]
    )
    assert code == 4 and payload["code"] == 4

    store = FakeStore(tmp_path / "voices")
    store.prompt_vectors = {}
    store.has_ref_wav = False
    _install_fake_runtime(
        monkeypatch, engine=FakeEngine(), store=store
    )
    code, _out, _err, payload = _run(
        tmp_path,
        capsys,
        ["--suite", "smoke", "--speaker-sim", "--reference-voice", "alice"],
    )
    assert code == 4 and payload["code"] == 4


def test_clone_profile_and_prompt_errors_exit_four(tmp_path, monkeypatch, capsys):
    engine = FakeEngine()
    _install_fake_runtime(monkeypatch, engine=engine)
    code, _out, _err, payload = _run(
        tmp_path, capsys, ["--suite", "smoke", "--voice", "voice:missing"]
    )
    assert code == 4 and payload["code"] == 4

    store = FakeStore(tmp_path / "voices")
    store.prompt_vectors = {}
    _install_fake_runtime(monkeypatch, engine=engine, store=store)
    code, _out, _err, payload = _run(
        tmp_path, capsys, ["--suite", "smoke", "--voice", "voice:alice"]
    )
    assert code == 4 and payload["code"] == 4


def test_generation_failure_exit_three(tmp_path, monkeypatch, capsys):
    _install_fake_runtime(monkeypatch, engine=FakeEngine(fail=True))
    code, out, _err, payload = _run(tmp_path, capsys, ["--suite", "smoke"])
    assert code == 3
    assert out == ""
    assert payload["code"] == 3


def test_text_file_errors_exit_four(tmp_path, monkeypatch, capsys):
    _install_fake_runtime(monkeypatch)
    missing = tmp_path / "missing.txt"
    code, _out, _err, payload = _run(
        tmp_path,
        capsys,
        ["--suite", "longform", "--text-file", str(missing), "--dry-run"],
    )
    assert code == 4 and payload["code"] == 4

    empty = tmp_path / "empty.txt"
    empty.write_text("   \n", encoding="utf-8")
    code, _out, _err, payload = _run(
        tmp_path,
        capsys,
        ["--suite", "longform", "--text-file", str(empty), "--dry-run"],
    )
    assert code == 4 and payload["code"] == 4


def test_preflight_order_before_tts_load(tmp_path, monkeypatch, capsys):
    log: list = []
    state = _install_fake_runtime(
        monkeypatch, engine=FakeEngine(log=log), asr=FakeAsr(texts=["x"]), log=log
    )

    def fail_preflight():
        log.append("preflight_asr")
        raise ModelError("asr runtime missing")

    monkeypatch.setattr(bench_cmd, "_preflight_asr_runtime", fail_preflight)
    code, _out, _err, payload = _run(
        tmp_path, capsys, ["--suite", "smoke", "--cer"]
    )
    assert code == 3 and payload["code"] == 3
    assert log == ["preflight_asr"]
    assert state["factories"] == []


@pytest.mark.parametrize(
    ("suite", "extra"),
    [("smoke", []), ("batch", ["--count", "16"])],
)
def test_auto_language_validates_each_case_before_load(
    tmp_path, monkeypatch, capsys, suite, extra
):
    chinese_only = _model(
        "Qwen3-TTS-CustomVoice-1.7B", "builtin", languages=["Chinese"]
    )
    log: list = []
    engine = FakeEngine(log=log)
    state = _install_fake_runtime(
        monkeypatch,
        models=[chinese_only, _model("Qwen3-TTS-Base-1.7B", "clone")],
        engine=engine,
        log=log,
    )
    code, out, _err, payload = _run(
        tmp_path, capsys, ["--suite", suite, "--language", "auto", *extra]
    )
    assert code == 4 and payload["code"] == 4
    assert out == ""
    assert state["factories"] == []
    assert "load" not in log
    assert not [item for item in log if isinstance(item, tuple) and item[0] == "generate"]
    assert "preflight_asr" not in log


def test_asr_language_preflight_runs_before_tts_load(tmp_path, monkeypatch, capsys):
    chinese_asr = _model("Qwen3-ASR-0.6B", "asr", languages=["Chinese"])
    log: list = []
    state = _install_fake_runtime(
        monkeypatch,
        models=[_model("Qwen3-TTS-CustomVoice-1.7B", "builtin"), chinese_asr],
        engine=FakeEngine(log=log),
        asr=FakeAsr(log=log),
        log=log,
    )
    code, out, _err, payload = _run(
        tmp_path, capsys, ["--suite", "smoke", "--language", "auto", "--cer"]
    )
    assert code == 4 and payload["code"] == 4
    assert out == ""
    assert state["factories"] == []
    assert "load" not in log
    assert log == []


def test_auto_case_audio_and_language_reach_asr_per_case(
    tmp_path, monkeypatch, capsys
):
    texts = {case[2]: case[1] for case in benchmark.SMOKE_CORPUS}
    asr = FakeAsr(router=lambda samples, sr, language: texts[language])
    _install_fake_runtime(
        monkeypatch, engine=FakeEngine(durations=[0.5, 0.5]), asr=asr
    )
    code, _out, _err, payload = _run(
        tmp_path, capsys, ["--suite", "smoke", "--language", "auto", "--cer"]
    )
    assert code == 0
    assert payload["metrics"]["cer"] == 0.0
    assert asr.languages == ["Chinese", "English"]
    assert len(asr.calls) == 2
    # Distinct generated arrays, one per case, each 0.5 s at 24 kHz.
    assert len({call[0] for call in asr.calls}) == 2
    assert [call[1] for call in asr.calls] == [12000, 12000]


def test_single_resident_release_order(tmp_path, monkeypatch, capsys):
    log: list = []
    store = FakeStore(tmp_path / "voices")
    state = _install_fake_runtime(
        monkeypatch,
        engine=FakeEngine(log=log, durations=[0.5, 0.5]),
        asr=FakeAsr(texts=["x", "y"], log=log),
        store=store,
        log=log,
    )
    state["base_engine"] = FakeEngine(
        log=log, embeddings=[np.array([1.0, 0, 0, 0])] * 2
    )
    code, _out, _err, payload = _run(
        tmp_path,
        capsys,
        [
            "--suite",
            "smoke",
            "--cer",
            "--speaker-sim",
            "--reference-voice",
            "alice",
        ],
    )
    assert code == 0
    assert payload["gate_failures"] == []

    def index_of(predicate):
        return next(i for i, item in enumerate(log) if predicate(item))

    preflight_asr = index_of(lambda item: item == "preflight_asr")
    preflight_speaker = index_of(
        lambda item: item == ("preflight_speaker", "clone")
    )
    builtin_factory = index_of(lambda item: item == ("engine_factory", "builtin"))
    load = index_of(lambda item: item == "load")
    generate = index_of(lambda item: isinstance(item, tuple) and item[0] == "generate")
    first_clear = index_of(lambda item: item == "clear_model")
    transcribe = index_of(lambda item: item == "transcribe")
    clear_asr = index_of(lambda item: item == "clear_asr")
    clone_factory = index_of(
        lambda item: item == ("engine_factory", "clone")
    )
    embedding = index_of(lambda item: item == "embedding")
    last_clear = max(i for i, item in enumerate(log) if item == "clear_model")

    assert (
        preflight_asr
        < preflight_speaker
        < builtin_factory
        < load
        < generate
        < first_clear
        < transcribe
        < clear_asr
        < clone_factory
        < embedding
    )
    assert embedding < last_clear
    assert ("asr_factory", "asr") in log


# --------------------------------------------------------------------------- #
# 7. Gates
# --------------------------------------------------------------------------- #
def test_cer_gate_pass_and_fail(tmp_path, monkeypatch, capsys):
    engine = FakeEngine(durations=[0.5])
    asr = FakeAsr(texts=["这是一次语音合成性能测试，用来测量本机生成一段完整句子的速度"])
    _install_fake_runtime(monkeypatch, engine=engine, asr=asr)
    code, _out, _err, payload = _run(
        tmp_path,
        capsys,
        ["--suite", "smoke", "--language", "Chinese", "--cer", "--max-cer", "0"],
    )
    assert code == 0
    assert payload["metrics"]["cer"] == 0.0
    assert payload["gate_failures"] == []

    asr2 = FakeAsr(texts=["完全错误"])
    _install_fake_runtime(monkeypatch, engine=FakeEngine(durations=[0.5]), asr=asr2)
    code, _out, _err, payload = _run(
        tmp_path,
        capsys,
        ["--suite", "smoke", "--language", "Chinese", "--cer", "--max-cer", "0"],
    )
    assert code == 5
    assert payload["ok"] is False
    assert payload["metrics"]["cer"] > 0
    assert payload["gate_failures"][0]["gate"] == "max_cer"
    assert payload["gate_failures"][0]["reason"] == "exceeded"


def test_null_gate_metric_exits_five(tmp_path, monkeypatch, capsys):
    _install_fake_runtime(monkeypatch, engine=FakeEngine(durations=[0.5]), asr=FakeAsr())
    monkeypatch.setattr(
        bench_cmd,
        "score_cer",
        lambda asr, items: {"cer": None, "cer_distance": 0, "cer_units": 0},
    )
    code, _out, _err, payload = _run(
        tmp_path, capsys, ["--suite", "smoke", "--cer", "--max-cer", "0.1"]
    )
    assert code == 5
    assert payload["gate_failures"][0]["reason"] == "unavailable"


def test_speaker_gate_pass_fail_and_zero_vector(tmp_path, monkeypatch, capsys):
    store = FakeStore(tmp_path / "voices")
    base = FakeEngine(embeddings=[np.array([1.0, 0, 0, 0])] * 2)
    state = _install_fake_runtime(
        monkeypatch, engine=FakeEngine(durations=[0.5, 0.5]), store=store
    )
    state["base_engine"] = base
    code, _out, _err, payload = _run(
        tmp_path,
        capsys,
        [
            "--suite",
            "smoke",
            "--speaker-sim",
            "--reference-voice",
            "alice",
            "--min-speaker-similarity",
            "0.9",
        ],
    )
    assert code == 0
    assert payload["metrics"]["speaker_similarity"] == 1.0
    assert payload["metrics"]["speaker_segments_scored"] == 2
    assert payload["metrics"]["speaker_segments_total"] == 2
    assert payload["reference_embedding_source"] == "prompt"

    state = _install_fake_runtime(
        monkeypatch, engine=FakeEngine(durations=[0.5, 0.5]), store=store
    )
    state["base_engine"] = FakeEngine(embeddings=[np.array([-1.0, 0, 0, 0])] * 2)
    code, _out, _err, payload = _run(
        tmp_path,
        capsys,
        [
            "--suite",
            "smoke",
            "--speaker-sim",
            "--reference-voice",
            "alice",
            "--min-speaker-similarity",
            "0.9",
        ],
    )
    assert code == 5
    assert payload["metrics"]["speaker_similarity"] == -1.0
    assert payload["gate_failures"][0]["reason"] == "below"

    state = _install_fake_runtime(
        monkeypatch, engine=FakeEngine(durations=[0.5, 0.5]), store=store
    )
    state["base_engine"] = FakeEngine(embeddings=[np.zeros(4)] * 2)
    code, _out, _err, payload = _run(
        tmp_path,
        capsys,
        ["--suite", "smoke", "--speaker-sim", "--reference-voice", "alice"],
    )
    assert code == 0
    assert payload["metrics"]["speaker_similarity"] is None
    assert payload["metrics"]["speaker_segments_scored"] == 0

    state = _install_fake_runtime(
        monkeypatch, engine=FakeEngine(durations=[0.5, 0.5]), store=store
    )
    state["base_engine"] = FakeEngine(embeddings=[np.zeros(4)] * 2)
    code, _out, _err, payload = _run(
        tmp_path,
        capsys,
        [
            "--suite",
            "smoke",
            "--speaker-sim",
            "--reference-voice",
            "alice",
            "--min-speaker-similarity",
            "0.1",
        ],
    )
    assert code == 5
    assert payload["gate_failures"][0]["reason"] == "unavailable"


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_target_embedding_is_null(tmp_path, monkeypatch, capsys, bad):
    store = FakeStore(tmp_path / "voices")
    state = _install_fake_runtime(
        monkeypatch, engine=FakeEngine(durations=[0.5, 0.5]), store=store
    )
    state["base_engine"] = FakeEngine(
        embeddings=[np.array([bad, 0.0, 0.0, 0.0])] * 2
    )
    code, _out, _err, payload = _run(
        tmp_path,
        capsys,
        ["--suite", "smoke", "--speaker-sim", "--reference-voice", "alice"],
    )
    assert code == 0
    assert payload["metrics"]["speaker_similarity"] is None
    assert payload["metrics"]["speaker_segments_scored"] == 0
    assert payload["gate_failures"] == []


def test_nonfinite_reference_embedding_is_null_with_gate(
    tmp_path, monkeypatch, capsys
):
    store = FakeStore(tmp_path / "voices")
    store.prompt_vectors["alice"] = np.array(
        [np.nan, 0.0, 0.0, 0.0], dtype=np.float32
    )
    state = _install_fake_runtime(
        monkeypatch, engine=FakeEngine(durations=[0.5, 0.5]), store=store
    )
    state["base_engine"] = FakeEngine(embeddings=[np.ones(4)] * 2)
    code, _out, _err, payload = _run(
        tmp_path,
        capsys,
        [
            "--suite",
            "smoke",
            "--speaker-sim",
            "--reference-voice",
            "alice",
            "--min-speaker-similarity",
            "0.1",
        ],
    )
    assert code == 5
    assert payload["metrics"]["speaker_similarity"] is None
    assert payload["gate_failures"][0]["reason"] == "unavailable"


# --------------------------------------------------------------------------- #
# 8. CER / speaker math
# --------------------------------------------------------------------------- #
def test_speaker_reference_falls_back_to_ref_wav(tmp_path, monkeypatch, capsys):
    store = FakeStore(tmp_path / "voices")
    store.prompt_vectors = {}
    store.has_ref_wav = True
    state = _install_fake_runtime(
        monkeypatch, engine=FakeEngine(durations=[0.5, 0.5]), store=store
    )
    state["base_engine"] = FakeEngine(
        embeddings=[np.array([1.0, 0.0, 0.0, 0.0])] * 3
    )
    calls: list = []

    def fake_read(path):
        calls.append(path)
        return np.ones(10, dtype=np.float32), 24000

    monkeypatch.setattr(bench_cmd, "read_audio", fake_read)
    code, _out, _err, payload = _run(
        tmp_path,
        capsys,
        ["--suite", "smoke", "--speaker-sim", "--reference-voice", "alice"],
    )
    assert code == 0
    assert payload["reference_embedding_source"] == "ref_wav"
    assert len(calls) == 1
    assert payload["metrics"]["speaker_similarity"] == 1.0


def test_cer_math_and_normalization():
    case = BenchCase("a", "abc.", "English")
    asr = FakeAsr(texts=["abd"])
    stats = benchmark.score_cer(asr, [(case, np.zeros(10), 24000)])
    assert stats == {"cer": 0.333333, "cer_distance": 1, "cer_units": 3}
    assert asr.languages == ["English"]

    case2 = BenchCase("b", "你好。", "auto")
    stats = benchmark.score_cer(
        FakeAsr(texts=[""]), [(case2, np.zeros(10), 24000)]
    )
    assert stats["cer"] == 1.0
    assert stats["cer_distance"] == 2
    assert stats["cer_units"] == 2

    class NoneAsr:
        def transcribe_samples(self, samples, sr, *, language=None):
            return None

    with pytest.raises(ModelError):
        benchmark.score_cer(NoneAsr(), [(case, np.zeros(10), 24000)])


def test_mean_cosine_and_speaker_null_rules():
    from qtts_cli.audio_qc import cosine_similarity

    reference = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    value, scored = benchmark.mean_cosine_similarity(
        reference, [reference, -reference]
    )
    assert value == 0.0 and scored == 2

    value, scored = benchmark.mean_cosine_similarity(
        reference, [reference, np.zeros(3)]
    )
    assert value is None and scored == 1

    value, scored = benchmark.mean_cosine_similarity(
        reference, [np.ones(4)]
    )
    assert value is None and scored == 0

    # Non-finite values are invalid embeddings, whether on either side.
    for bad in (np.nan, np.inf, -np.inf):
        assert cosine_similarity(reference, np.array([bad, 0.0, 0.0])) is None
        assert cosine_similarity(np.array([bad, 0.0, 0.0]), reference) is None
        value, scored = benchmark.mean_cosine_similarity(
            reference, [reference, np.array([bad, 0.0, 0.0])]
        )
        assert value is None and scored == 1
        value, scored = benchmark.mean_cosine_similarity(
            np.array([bad, 0.0, 0.0]), [reference]
        )
        assert value is None and scored == 0

    assert cosine_similarity(reference, None) is None
    assert cosine_similarity(reference, np.zeros(3)) is None
    assert cosine_similarity(reference, np.array([1.0, 0.0])) is None

    case = BenchCase("a", "x", "English")
    fake = FakeEngine(embeddings=[np.zeros(3)])
    similarity, scored, total = benchmark.score_speaker(
        fake, reference, [(case, np.zeros(10), 24000)]
    )
    assert similarity is None and scored == 0 and total == 1


def test_cer_null_when_reference_units_zero():
    case = BenchCase("a", "...", "English")
    stats = benchmark.score_cer(
        FakeAsr(texts=[""]), [(case, np.zeros(10), 24000)]
    )
    assert stats == {"cer": None, "cer_distance": 0, "cer_units": 0}


# --------------------------------------------------------------------------- #
# 9. Report failure
# --------------------------------------------------------------------------- #
def test_report_write_failure_still_emits_complete_payload(tmp_path, monkeypatch, capsys):
    engine = FakeEngine(durations=[0.5, 0.5])
    _install_fake_runtime(monkeypatch, engine=engine)
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")
    report = blocker / "report.json"
    code, out, err, payload = _run(
        tmp_path, capsys, ["--suite", "smoke", "--report", str(report)]
    )
    assert code == 4
    assert err == ""
    assert payload["error"]["code"] == 4
    assert payload["ok"] is False
    assert len(payload["cases"]) == 2
    assert payload["metrics"]["audio_s"] is not None
    assert not report.exists()


def test_report_failure_keeps_gate_results(tmp_path, monkeypatch, capsys):
    _install_fake_runtime(
        monkeypatch, engine=FakeEngine(durations=[0.5]), asr=FakeAsr(texts=["wrong"])
    )
    blocker = tmp_path / "blocker"
    blocker.write_text("x", encoding="utf-8")
    code, _out, _err, payload = _run(
        tmp_path,
        capsys,
        [
            "--suite",
            "smoke",
            "--language",
            "Chinese",
            "--cer",
            "--max-cer",
            "0",
            "--report",
            str(blocker / "r.json"),
        ],
    )
    assert code == 4
    assert payload["gate_failures"]
    assert payload["error"]["code"] == 4


# --------------------------------------------------------------------------- #
# 10. Peak RSS helper / doctor regression
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("system", "raw", "expected"),
    [("Darwin", 1234, 1234), ("Linux", 1234, 1234 * 1024)],
)
def test_peak_rss_platform_conversion(monkeypatch, system, raw, expected):
    from types import SimpleNamespace as NS

    from qtts_cli import doctor, utils

    monkeypatch.setattr(utils.platform, "system", lambda: system)
    monkeypatch.setattr(
        utils.resource, "getrusage", lambda _who: NS(ru_maxrss=raw)
    )
    assert peak_rss_bytes() == expected
    assert doctor._peak_rss_bytes() == expected


def test_doctor_bench_old_fields_unchanged(monkeypatch):
    from qtts_cli import doctor, tts_engine

    class FakeDoctorEngine:
        def __init__(self, cfg, model):
            self.cfg = cfg
            self.model = model
            self.model_info = model
            self.device = "cpu"
            self.dtype_name = "fp32"
            self.load_ms = None

        def load(self):
            self.load_ms = 5.0
            return self

        def generate(self, **kwargs):
            return np.zeros(2400, dtype=np.float32), 24000, {
                "duration_s": 0.1,
                "generate_ms": 10.0,
                "sample_rate": 24000,
            }

    monkeypatch.setattr(tts_engine, "TtsEngine", FakeDoctorEngine)
    monkeypatch.setattr(doctor, "_peak_rss_bytes", lambda: 10)
    report = doctor.run_bench(
        SimpleNamespace(seed=1234),
        models=[_model("builtin", "builtin")],
    )
    assert report["ok"] is True
    assert report["peak_rss_bytes"] == 10
    assert report["peak_rss_delta"] == "0B"
    assert "process_peak_rss_bytes" not in report
    assert len(report["cases"]) == 2


# --------------------------------------------------------------------------- #
# 11. No real models / no runtime import at module load
# --------------------------------------------------------------------------- #
def test_benchmark_module_source_has_no_runtime_import():
    source = Path(benchmark.__file__).read_text(encoding="utf-8")
    for forbidden in ("import torch", "import qwen_tts", "import qwen_asr"):
        assert forbidden not in source
