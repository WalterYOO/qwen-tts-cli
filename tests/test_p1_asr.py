"""FR-P1-03: ASR model discovery and the lazy transcription wrapper."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from qtts_cli import asr_engine
from qtts_cli.catalog import _load_one, scan, select_aligner, select_asr
from qtts_cli.config import Config
from qtts_cli.errors import InputError, ModelError, UsageError
from qtts_cli.models_ops import list_models


def _write_asr(
    root: Path,
    name: str,
    *,
    model_type: str = "qwen3_asr",
    languages: tuple[str, ...] = ("Chinese", "English"),
    weights: bool = True,
) -> Path:
    model = root / name
    model.mkdir(parents=True)
    (model / "config.json").write_text(
        json.dumps(
            {"model_type": model_type, "support_languages": list(languages)}
        ),
        encoding="utf-8",
    )
    if weights:
        (model / "model.safetensors").write_bytes(b"fake")
    return model


def test_scan_distinguishes_asr_and_aligner(tmp_path: Path):
    root = tmp_path / "models"
    root.mkdir()
    _write_asr(root, "Qwen3-ASR-0.6B")
    _write_asr(root, "Qwen3-ForcedAligner-0.6B")

    models = scan(root)
    kinds = {model.name: model.kind for model in models}
    assert kinds == {
        "Qwen3-ASR-0.6B": "asr",
        "Qwen3-ForcedAligner-0.6B": "align",
    }
    asr = next(model for model in models if model.kind == "asr")
    assert asr.size == "0b6"
    assert select_asr(models).name == "Qwen3-ASR-0.6B"
    assert select_aligner(models).name == "Qwen3-ForcedAligner-0.6B"

    listed = {entry["name"]: entry for entry in list_models(Config(model_dir=root))}
    assert listed["Qwen3-ASR-0.6B"]["kind"] == "asr"
    assert listed["Qwen3-ASR-0.6B"]["size_tier"] == "0b6"
    assert listed["Qwen3-ASR-0.6B"]["complete"] is True
    assert listed["Qwen3-ForcedAligner-0.6B"]["kind"] == "align"
    assert listed["Qwen3-ForcedAligner-0.6B"]["complete"] is True


def test_model_type_must_be_exactly_qwen3_asr(tmp_path: Path):
    root = tmp_path / "models"
    root.mkdir()
    _write_asr(root, "Mystery-Model", model_type="some_custom_asr_variant")
    assert scan(root) == []
    # The frozen directory pattern still works without a model_type.
    _write_asr(root, "Qwen3-ASR-0.6B", model_type="some_custom_asr_variant")
    assert [model.kind for model in scan(root)] == ["asr"]


def test_canonical_asr_language_is_case_insensitive():
    assert asr_engine.canonical_asr_language("chinese") == "Chinese"
    assert asr_engine.canonical_asr_language("ENGLISH") == "English"
    assert asr_engine.canonical_asr_language("auto") is None
    assert asr_engine.canonical_asr_language(None) is None
    with pytest.raises(UsageError) as bad:
        asr_engine.canonical_asr_language("Klingon")
    assert bad.value.code == 2


def test_scan_uses_directory_name_when_model_type_is_missing(tmp_path: Path):
    root = tmp_path / "models"
    root.mkdir()
    _write_asr(root, "Qwen3-ASR-0.6B", model_type="")
    assert scan(root)[0].kind == "asr"


def test_select_asr_path_id_name_and_failures(tmp_path: Path):
    root = tmp_path / "models"
    root.mkdir()
    small = _write_asr(root, "Qwen3-ASR-0.6B")
    _write_asr(root, "Qwen3-ASR-1.7B")
    aligner = _write_asr(root, "Qwen3-ForcedAligner-0.6B")
    models = scan(root)

    assert select_asr(models).name == "Qwen3-ASR-1.7B"  # highest version wins
    assert select_asr(models, str(small)).name == "Qwen3-ASR-0.6B"
    assert select_asr(models, "Qwen/Qwen3-ASR-0.6B").name == "Qwen3-ASR-0.6B"
    assert select_asr(models, "Qwen3-ASR-1.7B").name == "Qwen3-ASR-1.7B"
    with pytest.raises(ModelError) as missing:
        select_asr(models, "Qwen/Qwen3-ASR-9B")
    assert missing.value.code == 3
    with pytest.raises(ModelError) as aligner_spec:
        select_asr(models, str(aligner))
    assert aligner_spec.value.code == 3
    with pytest.raises(ModelError) as nonexistent:
        select_asr(models, str(tmp_path / "nope"))
    assert nonexistent.value.code == 3


def test_select_asr_no_models_and_ambiguous_equal_version(tmp_path: Path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ModelError) as missing:
        select_asr(scan(empty))
    assert missing.value.code == 3

    root = tmp_path / "models"
    root.mkdir()
    _write_asr(root, "Qwen3-ASR-0.6B")
    _write_asr(root, "Qwen3-ASR-0.6B-alt")
    with pytest.raises(UsageError) as ambiguous:
        select_asr(scan(root))
    assert ambiguous.value.code == 2


def test_normalize_transcription_shapes():
    result = SimpleNamespace(
        language="Chinese",
        text="你好",
        time_stamps=SimpleNamespace(
            items=[SimpleNamespace(text="你", start_time=0, end_time=0.5)]
        ),
    )
    normalized = asr_engine.normalize_transcription([result], timestamps=True)
    assert normalized == {
        "text": "你好",
        "language": "Chinese",
        "items": [{"text": "你", "start": 0.0, "end": 0.5}],
    }
    plain = asr_engine.normalize_transcription([result])
    assert plain["items"] is None
    assert asr_engine.normalize_transcription("")["text"] == ""
    assert asr_engine.normalize_transcription({"text": "hi"})["language"] is None

    with pytest.raises(ModelError):
        asr_engine.normalize_transcription([])
    with pytest.raises(ModelError):
        asr_engine.normalize_transcription([result, result])
    with pytest.raises(ModelError):
        asr_engine.normalize_transcription([SimpleNamespace(text=None)])


class _FakeTorch:
    bfloat16 = object()
    float16 = object()
    float32 = object()


class _FakeCfg:
    offline = False

    def resolve_device(self) -> str:
        return "cpu"

    def torch_dtype(self, device=None):
        return _FakeTorch.float32


class _FakeModel:
    loads = 0
    calls: list[dict] = []
    error: BaseException | None = None

    def __init__(self) -> None:
        self.forced_aligner = None

    @classmethod
    def from_pretrained(cls, path: str, **kwargs):
        cls.loads += 1
        cls.last_kwargs = kwargs
        return cls()

    def transcribe(self, path: str, **kwargs):
        self.__class__.calls.append({"path": path, **kwargs})
        if self.__class__.error is not None:
            raise self.__class__.error
        return [SimpleNamespace(language="Chinese", text="你好", time_stamps=None)]


@pytest.fixture()
def fake_runtime(monkeypatch):
    asr_engine.clear_asr_cache()
    _FakeModel.loads = 0
    _FakeModel.calls = []
    _FakeModel.error = None
    monkeypatch.setattr(asr_engine, "_runtime", lambda: (_FakeTorch, _FakeModel))
    yield _FakeModel
    asr_engine.clear_asr_cache()


def test_asr_engine_language_and_cache(tmp_path: Path, fake_runtime):
    model_dir = _write_asr(tmp_path, "Qwen3-ASR-0.6B")
    info = _load_one(model_dir)
    assert info is not None
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"fake")

    engine = asr_engine.AsrEngine(_FakeCfg(), info)
    result = engine.transcribe(audio, language="chinese")
    assert result["text"] == "你好"
    assert fake_runtime.calls[-1]["language"] == "Chinese"
    assert fake_runtime.calls[-1]["return_time_stamps"] is False

    # auto never reaches qwen-asr as the literal string "auto".
    engine.transcribe(audio)
    assert fake_runtime.calls[-1]["language"] is None

    asr_engine.AsrEngine(_FakeCfg(), info).load()
    assert fake_runtime.loads == 1  # process-level cache hit
    asr_engine.clear_asr_cache()
    asr_engine.AsrEngine(_FakeCfg(), info).load()
    assert fake_runtime.loads == 2

    with pytest.raises(UsageError) as bad_language:
        engine.transcribe(audio, language="Klingon")
    assert bad_language.value.code == 2
    with pytest.raises(InputError):
        engine.transcribe(tmp_path / "missing.wav")


def test_asr_cache_key_includes_aligner(tmp_path: Path, fake_runtime):
    root = tmp_path / "models"
    root.mkdir()
    asr_info = _load_one(_write_asr(root, "Qwen3-ASR-0.6B"))
    aligner_info = _load_one(_write_asr(root, "Qwen3-ForcedAligner-0.6B"))
    assert asr_info is not None and aligner_info is not None

    asr_engine.AsrEngine(_FakeCfg(), asr_info).load()
    asr_engine.AsrEngine(_FakeCfg(), asr_info).load()
    assert fake_runtime.loads == 1  # same (path, device, dtype, fingerprint)

    asr_engine.AsrEngine(_FakeCfg(), asr_info, aligner=aligner_info).load()
    assert fake_runtime.loads == 2  # aligner identity changes the cache bucket
    assert fake_runtime.last_kwargs["forced_aligner"] == str(aligner_info.path)


def test_asr_engine_wraps_errors_and_requires_aligner_for_timestamps(
    tmp_path: Path, fake_runtime
):
    model_dir = _write_asr(tmp_path, "Qwen3-ASR-0.6B")
    info = _load_one(model_dir)
    assert info is not None
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"fake")

    engine = asr_engine.AsrEngine(_FakeCfg(), info)
    with pytest.raises(ModelError) as no_aligner:
        engine.transcribe(audio, timestamps=True)
    assert no_aligner.value.code == 3

    fake_runtime.error = RuntimeError("boom")
    with pytest.raises(ModelError) as wrapped:
        engine.transcribe(audio)
    assert "boom" in wrapped.value.message
    assert wrapped.value.code == 3
