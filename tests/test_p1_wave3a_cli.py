"""Wave 3a CLI contracts: `check`, `qc`, `say --qc`, `voice add --auto-ref-text`."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from qtts_cli import asr_engine
from qtts_cli.cli import main
from qtts_cli.commands import check as check_cmd
from qtts_cli.commands import qc as qc_cmd
from qtts_cli.commands import say as say_cmd
from qtts_cli.commands import voice as voice_cmd
from qtts_cli.errors import ModelError
from qtts_cli.voice_store import VoiceStore


def _run(tmp_path: Path, model_dir: Path, *args: str) -> int:
    return main(
        [
            *args,
            "--model-dir",
            str(model_dir),
            "--data-dir",
            str(tmp_path / "data"),
            "--cache-dir",
            str(tmp_path / "cache"),
        ]
    )


def _write_wav(
    path: Path,
    *,
    seconds: float = 1.0,
    sr: int = 24000,
    amplitude: float = 0.2,
    channels: int = 1,
    constant: float | None = None,
) -> Path:
    count = int(round(sr * seconds))
    if constant is None:
        t = np.arange(count, dtype=np.float64) / float(sr)
        mono = amplitude * np.sin(2 * np.pi * 220.0 * t)
    else:
        mono = np.full(count, constant, dtype=np.float64)
    data = mono.astype(np.float32)
    if channels > 1:
        data = np.repeat(data[:, None], channels, axis=1)
    sf.write(path, data, sr)
    return path


def _add_asr_model(
    root: Path,
    name: str = "Qwen3-ASR-0.6B",
    *,
    languages: list[str] | None = None,
) -> Path:
    model = root / name
    model.mkdir()
    config: dict = {"model_type": "qwen3_asr"}
    if languages is not None:
        config["support_languages"] = list(languages)
    (model / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"fake")
    return model


class FakeAsrEngine:
    text = "你好世界"
    language = "Chinese"
    fail: BaseException | None = None
    calls: list[dict] = []

    def __init__(self, cfg, model):
        self.cfg = cfg
        self.model_info = model
        self.model = None

    def transcribe(self, path, *, language=None, timestamps=False):
        FakeAsrEngine.calls.append(
            {"path": str(path), "language": language, "timestamps": timestamps}
        )
        if FakeAsrEngine.fail is not None:
            raise FakeAsrEngine.fail
        return {
            "text": FakeAsrEngine.text,
            "language": FakeAsrEngine.language,
            "items": None,
        }


class FakeVoiceEngine:
    """Minimal TTS engine for `voice add` (imports torch only when called)."""

    def __init__(self, cfg, model):
        self.cfg = cfg
        self.model_info = model
        self.dtype_name = "fp32"
        self.model = None

    def load(self):
        self.model = object()
        return self

    def create_clone_prompt_items(
        self, path, *, ref_text=None, x_vector_only_mode=False
    ):
        import torch

        return [
            SimpleNamespace(
                ref_spk_embedding=torch.zeros((1, 4), dtype=torch.float32),
                ref_code=None,
                x_vector_only_mode=x_vector_only_mode,
                icl_mode=not x_vector_only_mode,
            )
        ]


@pytest.fixture()
def fake_asr(monkeypatch):
    FakeAsrEngine.text = "你好世界"
    FakeAsrEngine.language = "Chinese"
    FakeAsrEngine.fail = None
    FakeAsrEngine.calls = []
    monkeypatch.setattr(check_cmd, "AsrEngine", FakeAsrEngine)
    monkeypatch.setattr(voice_cmd, "AsrEngine", FakeAsrEngine)
    return FakeAsrEngine


# --------------------------------------------------------------------------- #
# check
# --------------------------------------------------------------------------- #
def test_check_is_canonical_deterministic_and_has_no_absolute_paths(
    fake_asr, capsys, tmp_path: Path, model_dir: Path
):
    _add_asr_model(model_dir)
    audio = _write_wav(tmp_path / "tone.wav")
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"

    rc = _run(
        tmp_path,
        model_dir,
        "check",
        str(audio),
        "--text",
        "你好，世界！",
        "--json",
        "--report",
        str(first),
    )
    assert rc == 0
    stdout = capsys.readouterr().out
    assert first.read_bytes() == stdout.encode("utf-8")

    rc = _run(
        tmp_path,
        model_dir,
        "check",
        str(audio),
        "--text",
        "你好，世界！",
        "--json",
        "--report",
        str(second),
    )
    assert rc == 0
    assert capsys.readouterr().out.encode("utf-8") == first.read_bytes()
    assert second.read_bytes() == first.read_bytes()

    payload = json.loads(first.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["ok"] is True
    assert payload["metric"] == "cer"
    assert payload["ratio"] == 0.0
    assert payload["distance"] == 0
    assert payload["errors"] == []
    assert payload["reference"] == "你好世界"
    assert payload["hypothesis"] == "你好世界"
    assert payload["audio"]["path"] == "tone.wav"
    assert "exit_code" not in payload
    assert "_canonical" not in payload
    assert str(tmp_path) not in stdout
    # Wave 3a freezes timestamps off; ASR auto language means ``None``.
    expected_call = {"path": str(audio), "language": None, "timestamps": False}
    assert fake_asr.calls == [expected_call, expected_call]


def test_check_threshold_boundary_is_inclusive(
    fake_asr, capsys, tmp_path: Path, model_dir: Path
):
    _add_asr_model(model_dir)
    audio = _write_wav(tmp_path / "tone.wav")
    fake_asr.text = "你号世界"  # one substitution: 1/4 = 0.25

    rc = _run(
        tmp_path, model_dir,
        "check", str(audio), "--text", "你好世界", "--threshold", "0.25", "--json",
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ratio"] == 0.25
    assert payload["ok"] is True
    assert payload["errors"][0]["op"] == "replace"
    assert payload["errors"][0]["start"] is None
    assert payload["errors"][0]["end"] is None

    report = tmp_path / "failed.json"
    rc = _run(
        tmp_path, model_dir,
        "check", str(audio), "--text", "你好世界", "--threshold", "0.24",
        "--json", "--report", str(report),
    )
    assert rc == 5
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert report.read_bytes() == json.dumps(
        payload, sort_keys=True, ensure_ascii=False, indent=2
    ).encode() + b"\n"


@pytest.mark.parametrize("value", ["nan", "inf", "-0.1"])
def test_check_rejects_invalid_threshold(
    fake_asr, capsys, tmp_path: Path, model_dir: Path, value: str
):
    _add_asr_model(model_dir)
    audio = _write_wav(tmp_path / "tone.wav")
    rc = _run(
        tmp_path, model_dir,
        "check", str(audio), "--text", "你好", "--threshold", value,
    )
    assert rc == 2
    assert "threshold" in capsys.readouterr().err


def test_check_requires_exactly_one_reference(
    fake_asr, capsys, tmp_path: Path, model_dir: Path
):
    _add_asr_model(model_dir)
    audio = _write_wav(tmp_path / "tone.wav")
    text_file = tmp_path / "ref.txt"
    text_file.write_text("你好", encoding="utf-8")

    assert _run(tmp_path, model_dir, "check", str(audio)) == 2
    assert _run(
        tmp_path, model_dir,
        "check", str(audio), "--text", "你好", "--text-file", str(text_file),
    ) == 2
    capsys.readouterr()


def test_check_reads_srt_and_rejects_empty_srt(
    fake_asr, capsys, tmp_path: Path, model_dir: Path
):
    _add_asr_model(model_dir)
    audio = _write_wav(tmp_path / "tone.wav")
    srt = tmp_path / "ref.srt"
    srt.write_text(
        "1\n00:00:00,000 --> 00:00:01,000\n你好世界\n", encoding="utf-8"
    )
    assert _run(
        tmp_path, model_dir, "check", str(audio), "--srt", str(srt), "--json"
    ) == 0
    assert json.loads(capsys.readouterr().out)["reference"] == "你好世界"

    empty = tmp_path / "empty.srt"
    empty.write_text("\n", encoding="utf-8")
    assert _run(
        tmp_path, model_dir, "check", str(audio), "--srt", str(empty)
    ) == 4
    capsys.readouterr()


def test_check_rejects_long_audio_before_model_selection(
    fake_asr, capsys, tmp_path: Path, model_dir: Path
):
    # No ASR model exists: reaching select_asr would exit 3.
    audio = _write_wav(tmp_path / "long.wav", seconds=301.0, sr=8000)
    rc = _run(tmp_path, model_dir, "check", str(audio), "--text", "你好")
    assert rc == 4
    assert "at most 300s" in capsys.readouterr().err
    assert fake_asr.calls == []


def test_check_missing_asr_model_exits_3(
    fake_asr, capsys, tmp_path: Path, model_dir: Path
):
    audio = _write_wav(tmp_path / "tone.wav")
    rc = _run(tmp_path, model_dir, "check", str(audio), "--text", "你好")
    assert rc == 3
    assert "ASR" in capsys.readouterr().err


def test_check_missing_audio_exits_4_before_model_load(
    fake_asr, capsys, tmp_path: Path, model_dir: Path
):
    rc = _run(
        tmp_path, model_dir,
        "check", str(tmp_path / "missing.wav"), "--text", "你好",
    )
    assert rc == 4
    assert fake_asr.calls == []
    capsys.readouterr()


def test_check_explicit_language_selects_wer(
    fake_asr, capsys, tmp_path: Path, model_dir: Path
):
    _add_asr_model(model_dir)
    audio = _write_wav(tmp_path / "tone.wav")
    fake_asr.text = "hello word"
    fake_asr.language = "English"
    rc = _run(
        tmp_path, model_dir,
        "check", str(audio), "--text", "hello world", "--language", "English",
        "--threshold", "0.5", "--json",
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["metric"] == "wer"
    assert payload["distance"] == 1
    assert payload["reference_units"] == 2
    assert fake_asr.calls[0]["language"] == "English"


def test_check_asr_runtime_error_exits_3(
    fake_asr, capsys, tmp_path: Path, model_dir: Path
):
    _add_asr_model(model_dir)
    audio = _write_wav(tmp_path / "tone.wav")
    fake_asr.fail = ModelError("backend exploded")
    rc = _run(tmp_path, model_dir, "check", str(audio), "--text", "你好")
    assert rc == 3
    assert "backend exploded" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# qc
# --------------------------------------------------------------------------- #
def test_qc_measure_only_is_canonical_and_loads_no_model(
    monkeypatch, capsys, tmp_path: Path, model_dir: Path
):
    def boom(*args, **kwargs):
        raise AssertionError("measure-only qc must not load a model")

    monkeypatch.setattr(qc_cmd, "TtsEngine", boom)
    audio = _write_wav(tmp_path / "tone.wav", seconds=2.0)
    report = tmp_path / "qc.json"
    rc = _run(
        tmp_path, model_dir, "qc", str(audio), "--json", "--report", str(report)
    )
    assert rc == 0
    stdout = capsys.readouterr().out
    assert report.read_bytes() == stdout.encode("utf-8")
    payload = json.loads(stdout)
    assert payload["ok"] is True
    assert payload["thresholds"] == {}
    assert payload["violations"] == []
    assert payload["audio"]["path"] == "tone.wav"
    assert payload["audio"]["channels"] == 1
    assert payload["metrics"]["lufs_integrated"] is not None
    assert payload["metrics"]["true_peak_dbtp"] is not None
    assert payload["metrics"]["silence_ratio"] is not None
    assert payload["unavailable"] == {"speech_rate": "no_text"}
    assert "exit_code" not in payload
    assert str(tmp_path) not in stdout


def test_qc_explicit_gate_fails_with_exit_5(
    capsys, tmp_path: Path, model_dir: Path
):
    audio = _write_wav(tmp_path / "dc.wav", constant=0.1)
    report = tmp_path / "dc.json"
    rc = _run(
        tmp_path, model_dir,
        "qc", str(audio), "--max-dc-offset", "0.01", "--json",
        "--report", str(report),
    )
    assert rc == 5
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["thresholds"] == {"max_dc_offset": 0.01}
    [violation] = payload["violations"]
    assert violation["metric"] == "dc_offset"
    assert violation["op"] == "abs>"
    assert violation["value"] == pytest.approx(0.1, abs=1e-3)
    assert violation["threshold"] == 0.01
    assert report.read_bytes() == json.dumps(
        payload, sort_keys=True, ensure_ascii=False, indent=2
    ).encode() + b"\n"


def test_qc_gated_unavailable_metric_is_a_violation(
    capsys, tmp_path: Path, model_dir: Path
):
    audio = _write_wav(tmp_path / "silence.wav", constant=0.0)
    rc = _run(
        tmp_path, model_dir,
        "qc", str(audio), "--target-lufs", "-23", "--lufs-tolerance", "1",
        "--json",
    )
    assert rc == 5
    payload = json.loads(capsys.readouterr().out)
    assert payload["metrics"]["lufs_integrated"] is None
    assert payload["unavailable"]["lufs"] == "digital_silence"
    assert payload["violations"][0]["metric"] == "lufs_integrated"
    assert payload["violations"][0]["op"] == "unavailable"
    assert payload["violations"][0]["reason"] == "digital_silence"


@pytest.mark.parametrize(
    "args",
    [
        ("--target-lufs", "-23"),
        ("--max-clipping-ratio", "nan"),
        ("--max-clipping-ratio", "1.5"),
        ("--max-dc-offset", "-0.1"),
        ("--min-speech-rate", "3"),
        ("--min-speech-rate", "5", "--max-speech-rate", "4", "--text", "你好"),
    ],
)
def test_qc_rejects_invalid_thresholds(
    capsys, tmp_path: Path, model_dir: Path, args: tuple[str, ...]
):
    audio = _write_wav(tmp_path / "tone.wav")
    rc = _run(tmp_path, model_dir, "qc", str(audio), *args)
    assert rc == 2
    capsys.readouterr()


def test_qc_reads_stereo_and_rejects_three_channels(
    capsys, tmp_path: Path, model_dir: Path
):
    stereo = _write_wav(tmp_path / "stereo.wav", channels=2)
    rc = _run(tmp_path, model_dir, "qc", str(stereo), "--json")
    assert rc == 0
    assert json.loads(capsys.readouterr().out)["audio"]["channels"] == 2

    three = _write_wav(tmp_path / "three.wav", channels=3)
    rc = _run(tmp_path, model_dir, "qc", str(three))
    assert rc == 4
    assert "mono/stereo" in capsys.readouterr().err


class FakeEmbeddingEngine:
    instances = 0

    def __init__(self, cfg, model):
        FakeEmbeddingEngine.instances += 1
        self.cfg = cfg
        self.model_info = model
        self.model = object()
        self.device = "cpu"
        self.dtype_name = "fp32"

    def load(self):
        return self

    def extract_speaker_embedding(self, samples, sr):
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)


def _write_clone_profile(tmp_path: Path, name: str = "alice") -> Path:
    directory = tmp_path / "data" / "voices" / name
    directory.mkdir(parents=True)
    (directory / "voice.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "name": name,
                "kind": "clone",
                "source": {"type": "clone", "mode": "icl", "ref_text": "你好"},
                "model": {"id": "Qwen3-TTS-12Hz-1.7B-Base", "revision": None},
                "language": "Chinese",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    _write_wav(directory / "ref.wav", seconds=2.0)
    return directory


def test_qc_reference_voice_reports_similarity(
    monkeypatch, capsys, tmp_path: Path, model_dir: Path
):
    _write_clone_profile(tmp_path)
    FakeEmbeddingEngine.instances = 0
    monkeypatch.setattr(qc_cmd, "TtsEngine", FakeEmbeddingEngine)
    audio = _write_wav(tmp_path / "tone.wav")
    rc = _run(
        tmp_path, model_dir,
        "qc", str(audio), "--reference-voice", "alice", "--json",
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["reference_voice"] == "alice"
    assert payload["metrics"]["speaker_similarity"] == 1.0
    assert "speaker_similarity" not in payload["unavailable"]
    assert FakeEmbeddingEngine.instances == 1


def test_qc_rejects_non_clone_reference_voice(
    capsys, tmp_path: Path, model_dir: Path
):
    directory = tmp_path / "data" / "voices" / "calm"
    directory.mkdir(parents=True)
    (directory / "voice.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "name": "calm",
                "kind": "design",
                "source": {"type": "design", "description": "calm"},
                "model": {"id": "design", "revision": None},
            }
        ),
        encoding="utf-8",
    )
    audio = _write_wav(tmp_path / "tone.wav")
    rc = _run(
        tmp_path, model_dir, "qc", str(audio), "--reference-voice", "calm"
    )
    assert rc == 4
    assert "requires a clone profile" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# say --qc
# --------------------------------------------------------------------------- #
class FakeSayEngine:
    loads = 0
    generated = 0

    def __init__(self, cfg, model):
        self.cfg = cfg
        self.model_info = model
        self.device = "cpu"
        self.dtype_name = "fp32"
        self.load_ms = 0.0
        self.model = None

    def load(self):
        FakeSayEngine.loads += 1
        self.model = object()
        return self

    def generate(self, **kwargs):
        FakeSayEngine.generated += 1
        t = np.arange(24000, dtype=np.float64) / 24000.0
        audio = (0.2 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
        return audio, 24000, {
            "generate_ms": 1.0,
            "sample_rate": 24000,
            "duration_s": 1.0,
            "mode": kwargs.get("mode"),
            "language": kwargs.get("language"),
        }

    def build_clone_prompt_items(self, *args, **kwargs):
        return []

    def extract_speaker_embedding(self, samples, sr):
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)


def test_say_qc_measure_only_adds_no_model(
    monkeypatch, capsys, tmp_path: Path, model_dir: Path
):
    FakeSayEngine.loads = 0
    monkeypatch.setattr(say_cmd, "TtsEngine", FakeSayEngine)

    def no_qc_model(*args, **kwargs):
        raise AssertionError("--qc without --reference-voice must not load Base")

    monkeypatch.setattr(qc_cmd, "TtsEngine", no_qc_model)
    out = tmp_path / "out.wav"
    rc = _run(
        tmp_path, model_dir,
        "say", "你好", "--voice", "builtin:Vivian", "--out", str(out),
        "--qc", "--json",
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["qc"]["ok"] is True
    assert payload["qc"]["thresholds"] == {}
    assert payload["qc"]["metrics"]["lufs_integrated"] is not None
    assert out.is_file()
    assert FakeSayEngine.loads == 1

    recipe = json.loads(out.with_suffix(".recipe.json").read_text(encoding="utf-8"))
    assert recipe["qc"]["ok"] is True
    assert recipe["qc"]["metrics"]["true_peak_dbtp"] is not None


def test_say_qc_violation_exits_5_with_complete_artifacts(
    monkeypatch, capsys, tmp_path: Path, model_dir: Path
):
    monkeypatch.setattr(say_cmd, "TtsEngine", FakeSayEngine)
    out = tmp_path / "out.wav"
    rc = _run(
        tmp_path, model_dir,
        "say", "你好", "--voice", "builtin:Vivian", "--out", str(out),
        "--qc", "--target-lufs", "0", "--lufs-tolerance", "1", "--json",
    )
    assert rc == 5
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["exit_code"] == 5
    assert payload["qc"]["violations"][0]["metric"] == "lufs_integrated"
    assert out.is_file()
    recipe = json.loads(out.with_suffix(".recipe.json").read_text(encoding="utf-8"))
    assert recipe["qc"]["violations"][0]["metric"] == "lufs_integrated"


def test_say_qc_dry_run_loads_no_models(
    monkeypatch, capsys, tmp_path: Path, model_dir: Path
):
    def boom(self):
        raise AssertionError("--dry-run must not load models")

    monkeypatch.setattr(say_cmd.TtsEngine, "load", boom)
    rc = _run(
        tmp_path, model_dir,
        "say", "你好", "--voice", "builtin:Vivian",
        "--qc", "--dry-run", "--json",
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["qc"]["enabled"] is True
    assert payload["qc"]["thresholds"] == {}
    assert payload["models_loaded"] is False


def test_say_reference_voice_requires_qc(
    capsys, tmp_path: Path, model_dir: Path
):
    _write_clone_profile(tmp_path)
    rc = _run(
        tmp_path, model_dir,
        "say", "你好", "--voice", "voice:alice", "--reference-voice", "alice",
    )
    assert rc == 2
    assert "requires --qc" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# voice add --auto-ref-text
# --------------------------------------------------------------------------- #
def test_auto_ref_text_conflicts_exit_2(
    capsys, tmp_path: Path, model_dir: Path
):
    ref = _write_wav(tmp_path / "ref.wav", seconds=3.0)
    cases = [
        ("--ref-text", "你好"),
        ("--mode", "xvec"),
    ]
    for extra in cases:
        rc = _run(
            tmp_path, model_dir,
            "voice", "add", "bob", "--kind", "clone", "--ref-audio", str(ref),
            "--auto-ref-text", "--i-have-rights", *extra,
        )
        assert rc == 2, extra
    assert not (tmp_path / "data" / "voices" / "bob").exists()

    rc = _run(
        tmp_path, model_dir,
        "voice", "add", "calm", "--kind", "design", "--description", "calm",
        "--auto-ref-text",
    )
    assert rc == 2
    assert "only valid with --kind clone" in capsys.readouterr().err


def test_auto_ref_text_requires_asr_model_and_leaves_no_profile(
    fake_asr, capsys, tmp_path: Path, model_dir: Path
):
    ref = _write_wav(tmp_path / "ref.wav", seconds=3.0)
    rc = _run(
        tmp_path, model_dir,
        "voice", "add", "bob", "--kind", "clone", "--ref-audio", str(ref),
        "--auto-ref-text", "--i-have-rights",
    )
    assert rc == 3
    assert "ASR" in capsys.readouterr().err
    voices = tmp_path / "data" / "voices"
    assert not (voices / "bob").exists()
    assert list(voices.glob(".bob.*")) == []


def test_auto_ref_text_asr_failure_leaves_no_profile(
    fake_asr, capsys, tmp_path: Path, model_dir: Path
):
    _add_asr_model(model_dir)
    ref = _write_wav(tmp_path / "ref.wav", seconds=3.0)
    fake_asr.fail = ModelError("asr exploded")
    rc = _run(
        tmp_path, model_dir,
        "voice", "add", "bob", "--kind", "clone", "--ref-audio", str(ref),
        "--auto-ref-text", "--i-have-rights",
    )
    assert rc == 3
    assert "asr exploded" in capsys.readouterr().err
    voices = tmp_path / "data" / "voices"
    assert not (voices / "bob").exists()
    assert list(voices.glob(".bob.*")) == []


def test_auto_ref_text_empty_transcript_exits_4(
    fake_asr, capsys, tmp_path: Path, model_dir: Path
):
    _add_asr_model(model_dir)
    ref = _write_wav(tmp_path / "ref.wav", seconds=3.0)
    fake_asr.text = "   "
    rc = _run(
        tmp_path, model_dir,
        "voice", "add", "bob", "--kind", "clone", "--ref-audio", str(ref),
        "--auto-ref-text", "--i-have-rights",
    )
    assert rc == 4
    assert "empty ref_text" in capsys.readouterr().err
    assert not (tmp_path / "data" / "voices" / "bob").exists()


def test_auto_ref_text_releases_asr_before_tts_and_records_metadata(
    fake_asr, monkeypatch, capsys, tmp_path: Path, model_dir: Path
):
    pytest.importorskip("torch")
    _add_asr_model(model_dir)
    ref = _write_wav(tmp_path / "ref.wav", seconds=3.0)
    fake_asr.text = "这是参考音频。"
    order: list[str] = []

    real_transcribe = FakeAsrEngine.transcribe

    def transcribe(self, path, *, language=None, timestamps=False):
        order.append("asr")
        return real_transcribe(
            self, path, language=language, timestamps=timestamps
        )

    monkeypatch.setattr(FakeAsrEngine, "transcribe", transcribe)

    def clear_cache():
        order.append("asr_release")

    monkeypatch.setattr(voice_cmd, "clear_asr_cache", clear_cache)

    class OrderedVoiceEngine(FakeVoiceEngine):
        def load(self):
            order.append("tts_load")
            return super().load()

        def create_clone_prompt_items(
            self, path, *, ref_text=None, x_vector_only_mode=False
        ):
            order.append(f"tts_prompt:{ref_text}")
            return super().create_clone_prompt_items(
                path, ref_text=ref_text, x_vector_only_mode=x_vector_only_mode
            )

    monkeypatch.setattr(voice_cmd, "TtsEngine", OrderedVoiceEngine)

    rc = _run(
        tmp_path, model_dir,
        "voice", "add", "bob", "--kind", "clone", "--ref-audio", str(ref),
        "--auto-ref-text", "--i-have-rights", "--json",
    )
    assert rc == 0, capsys.readouterr().err
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["ref_text_source"] == "asr"
    assert payload["asr_model"]["id"] == "Qwen3-ASR-0.6B"
    assert payload["voice"]["source"]["ref_text"] == "这是参考音频。"
    assert payload["voice"]["source"]["ref_text_source"] == "asr"
    assert order == ["asr", "asr_release", "tts_load", "tts_prompt:这是参考音频。"]

    meta = VoiceStore(tmp_path / "data").get("bob")
    assert meta is not None
    assert meta["source"]["asr_model"]["id"] == "Qwen3-ASR-0.6B"
    assert list((tmp_path / "data" / "voices").glob(".bob.*")) == []

    rc = _run(tmp_path, model_dir, "voice", "inspect", "bob", "--json")
    assert rc == 0
    inspected = json.loads(capsys.readouterr().out)
    assert inspected["voice"]["source"]["ref_text_source"] == "asr"


# --------------------------------------------------------------------------- #
# Wave 3a review regressions
# --------------------------------------------------------------------------- #
def test_check_invalid_language_exits_2_before_audio_or_model(
    fake_asr, capsys, tmp_path: Path, model_dir: Path
):
    rc = _run(
        tmp_path, model_dir,
        "check", str(tmp_path / "missing.wav"), "--text", "你好",
        "--language", "Klingon",
    )
    assert rc == 2
    assert "unsupported ASR language" in capsys.readouterr().err

    audio = _write_wav(tmp_path / "tone.wav")
    rc = _run(
        tmp_path, model_dir,
        "check", str(audio), "--text", "你好", "--language", "Klingon",
    )
    assert rc == 2
    assert fake_asr.calls == []


def test_check_canonicalizes_lowercase_language(
    fake_asr, capsys, tmp_path: Path, model_dir: Path
):
    _add_asr_model(model_dir)
    audio = _write_wav(tmp_path / "tone.wav")
    fake_asr.text = "2026年"
    rc = _run(
        tmp_path, model_dir,
        "check", str(audio), "--text", "2026年", "--language", "chinese",
        "--json",
    )
    assert rc == 0, capsys.readouterr().err
    payload = json.loads(capsys.readouterr().out)
    assert payload["asr"]["language"] == "Chinese"
    assert payload["reference"] == "二零二六年"
    assert fake_asr.calls[0]["language"] == "Chinese"

    fake_asr.text = "hello two thousand twentysix"
    fake_asr.language = "English"
    rc = _run(
        tmp_path, model_dir,
        "check", str(audio), "--text", "Hello 2026", "--language", "english",
        "--threshold", "0.0", "--json",
    )
    assert rc == 0, capsys.readouterr().err
    payload = json.loads(capsys.readouterr().out)
    assert payload["metric"] == "wer"
    assert payload["reference"] == "hello two thousand twentysix"
    assert payload["asr"]["language"] == "English"


def test_check_keeps_null_asr_language_and_records_comparison_language(
    fake_asr, capsys, tmp_path: Path, model_dir: Path
):
    _add_asr_model(model_dir)
    audio = _write_wav(tmp_path / "tone.wav")
    fake_asr.language = None
    rc = _run(
        tmp_path, model_dir,
        "check", str(audio), "--text", "你好世界", "--json",
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["asr"]["language"] is None
    assert payload["comparison_language"] == "Chinese"
    assert payload["normalization"]["language"] == "Chinese"


@pytest.mark.parametrize("source", ["text", "file"])
def test_qc_empty_text_with_speech_rate_gate_exits_2(
    capsys, tmp_path: Path, model_dir: Path, source: str
):
    audio = _write_wav(tmp_path / "tone.wav")
    if source == "text":
        extra = ("--text", "")
    else:
        empty = tmp_path / "empty.txt"
        empty.write_text("   \n", encoding="utf-8")
        extra = ("--text-file", str(empty))
    rc = _run(
        tmp_path, model_dir,
        "qc", str(audio), *extra, "--min-speech-rate", "1",
    )
    assert rc == 2
    assert "speech-rate" in capsys.readouterr().err


def test_qc_speaker_model_errors_are_exit_3(
    capsys, tmp_path: Path, model_dir: Path
):
    _write_clone_profile(tmp_path)
    audio = _write_wav(tmp_path / "tone.wav")

    rc = _run(
        tmp_path, model_dir,
        "qc", str(audio), "--reference-voice", "alice",
        "--model", str(tmp_path / "missing-model"),
    )
    assert rc == 3
    capsys.readouterr()

    rc = _run(
        tmp_path, model_dir,
        "qc", str(audio), "--reference-voice", "alice",
        "--model", "Qwen3-TTS-12Hz-1.7B-CustomVoice",
    )
    assert rc == 3
    assert "cannot extract speaker embeddings" in capsys.readouterr().err

    shutil.rmtree(model_dir / "Qwen3-TTS-12Hz-1.7B-Base")
    rc = _run(
        tmp_path, model_dir, "qc", str(audio), "--reference-voice", "alice"
    )
    assert rc == 3
    assert "speaker-embedding model" in capsys.readouterr().err


def test_qc_speaker_probe_failure_loads_no_embedding(
    monkeypatch, capsys, tmp_path: Path, model_dir: Path
):
    _write_clone_profile(tmp_path)

    class ProbeFailEngine(FakeEmbeddingEngine):
        extracted = 0

        def require_speaker_embedding(self):
            raise ModelError("spike runtime lacks speaker embeddings")

        def extract_speaker_embedding(self, samples, sr):
            ProbeFailEngine.extracted += 1
            return super().extract_speaker_embedding(samples, sr)

    ProbeFailEngine.extracted = 0
    monkeypatch.setattr(qc_cmd, "TtsEngine", ProbeFailEngine)
    audio = _write_wav(tmp_path / "tone.wav")
    rc = _run(
        tmp_path, model_dir, "qc", str(audio), "--reference-voice", "alice"
    )
    assert rc == 3
    assert "spike runtime" in capsys.readouterr().err
    assert ProbeFailEngine.extracted == 0


def test_say_reference_voice_is_validated_before_generation(
    monkeypatch, capsys, tmp_path: Path, model_dir: Path
):
    FakeSayEngine.loads = 0
    FakeSayEngine.generated = 0
    monkeypatch.setattr(say_cmd, "TtsEngine", FakeSayEngine)
    out = tmp_path / "out.wav"
    rc = _run(
        tmp_path, model_dir,
        "say", "你好", "--voice", "builtin:Vivian", "--out", str(out),
        "--qc", "--reference-voice", "missing",
    )
    assert rc == 4
    assert FakeSayEngine.generated == 0
    assert not out.exists()
    assert not out.with_suffix(".recipe.json").exists()
    capsys.readouterr()

    rc = _run(
        tmp_path, model_dir,
        "say", "你好", "--voice", "builtin:Vivian",
        "--qc", "--reference-voice", "missing", "--dry-run", "--json",
    )
    assert rc == 4
    assert FakeSayEngine.generated == 0


def test_auto_ref_text_asr_model_overrides_ambiguous_models(
    fake_asr, monkeypatch, capsys, tmp_path: Path, model_dir: Path
):
    pytest.importorskip("torch")
    _add_asr_model(model_dir, "Qwen3-ASR-0.6B")
    _add_asr_model(model_dir, "Qwen3-ASR-0.6B-alt")
    ref = _write_wav(tmp_path / "ref.wav", seconds=3.0)

    rc = _run(
        tmp_path, model_dir,
        "voice", "add", "bob", "--kind", "clone", "--ref-audio", str(ref),
        "--auto-ref-text", "--i-have-rights",
    )
    assert rc == 2
    assert "multiple ASR models" in capsys.readouterr().err

    monkeypatch.setattr(voice_cmd, "TtsEngine", FakeVoiceEngine)
    rc = _run(
        tmp_path, model_dir,
        "voice", "add", "bob", "--kind", "clone", "--ref-audio", str(ref),
        "--auto-ref-text", "--asr-model", "Qwen3-ASR-0.6B",
        "--i-have-rights", "--json",
    )
    assert rc == 0, capsys.readouterr().err
    payload = json.loads(capsys.readouterr().out)
    assert payload["asr_model"]["id"] == "Qwen3-ASR-0.6B"


def test_speaker_embedding_probe_runs_before_weight_load(monkeypatch, tmp_path):
    from qtts_cli import tts_engine

    module = ModuleType("fake_qwen_runtime_without_embedding")

    class FakeModel:
        loads = 0

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            cls.loads += 1
            raise AssertionError("weights must not load when capability is absent")

    FakeModel.__module__ = module.__name__
    module.FakeModel = FakeModel
    monkeypatch.setitem(sys.modules, module.__name__, module)

    class FakeTorch:
        bfloat16 = object()
        float16 = object()
        float32 = object()

    monkeypatch.setattr(
        tts_engine, "_runtime", lambda: (FakeTorch, FakeModel, None)
    )
    model = SimpleNamespace(
        path=tmp_path / "model", name="model", kind="clone", fingerprint="f"
    )
    engine = tts_engine.TtsEngine(SimpleNamespace(), model)
    with pytest.raises(ModelError):
        engine.extract_speaker_embedding(
            np.ones(16, dtype=np.float32), 24000
        )
    assert FakeModel.loads == 0


def test_speaker_embedding_class_probe_finds_conditional_generation():
    from qtts_cli.tts_engine import _class_supports_speaker_embedding

    module = ModuleType("fake_qwen_runtime_with_embedding")

    class InnerModel:
        @staticmethod
        def extract_speaker_embedding(audio, sr):
            return np.ones(3, dtype=np.float32)

    class WrapperModel:
        pass

    InnerModel.__module__ = module.__name__
    WrapperModel.__module__ = module.__name__
    module.InnerModel = InnerModel
    module.WrapperModel = WrapperModel
    sys.modules[module.__name__] = module
    try:
        assert _class_supports_speaker_embedding(WrapperModel) is True
    finally:
        sys.modules.pop(module.__name__, None)


# --------------------------------------------------------------------------- #
# Final t10 follow-ups
# --------------------------------------------------------------------------- #
class CapabilityFailEngine:
    def __init__(self, cfg, model):
        self.cfg = cfg
        self.model_info = model
        self.model = None
        self.dtype_name = "fp32"

    def require_speaker_embedding(self):
        raise ModelError("runtime lacks speaker embeddings")


def test_say_reference_voice_base_missing_fails_before_generation(
    monkeypatch, capsys, tmp_path: Path, model_dir: Path
):
    _write_clone_profile(tmp_path)
    shutil.rmtree(model_dir / "Qwen3-TTS-12Hz-1.7B-Base")
    FakeSayEngine.loads = 0
    FakeSayEngine.generated = 0
    monkeypatch.setattr(say_cmd, "TtsEngine", FakeSayEngine)
    out = tmp_path / "out.wav"

    rc = _run(
        tmp_path, model_dir,
        "say", "你好", "--voice", "builtin:Vivian", "--out", str(out),
        "--qc", "--reference-voice", "alice",
    )
    assert rc == 3
    assert "speaker-embedding model" in capsys.readouterr().err
    assert FakeSayEngine.generated == 0
    assert not out.exists()
    assert not out.with_suffix(".recipe.json").exists()


def test_say_reference_voice_capability_missing_fails_before_generation(
    monkeypatch, capsys, tmp_path: Path, model_dir: Path
):
    _write_clone_profile(tmp_path)
    FakeSayEngine.loads = 0
    FakeSayEngine.generated = 0
    monkeypatch.setattr(say_cmd, "TtsEngine", FakeSayEngine)
    monkeypatch.setattr(qc_cmd, "TtsEngine", CapabilityFailEngine)
    out = tmp_path / "out.wav"

    rc = _run(
        tmp_path, model_dir,
        "say", "你好", "--voice", "builtin:Vivian", "--out", str(out),
        "--qc", "--reference-voice", "alice",
    )
    assert rc == 3
    assert "runtime lacks speaker embeddings" in capsys.readouterr().err
    assert FakeSayEngine.generated == 0
    assert not out.exists()
    assert not out.with_suffix(".recipe.json").exists()


def test_say_output_parameter_errors_win_over_profile_precheck(
    monkeypatch, capsys, tmp_path: Path, model_dir: Path
):
    FakeSayEngine.generated = 0
    monkeypatch.setattr(say_cmd, "TtsEngine", FakeSayEngine)
    rc = _run(
        tmp_path, model_dir,
        "say", "你好", "--voice", "builtin:Vivian", "--out", "-",
        "--json", "--qc", "--reference-voice", "missing",
    )
    assert rc == 2
    assert "stdout can carry only one product" in capsys.readouterr().err
    assert FakeSayEngine.generated == 0


def test_voice_add_rejects_asr_model_without_auto_ref_text(
    capsys, tmp_path: Path, model_dir: Path
):
    ref = _write_wav(tmp_path / "ref.wav", seconds=3.0)
    rc = _run(
        tmp_path, model_dir,
        "voice", "add", "bob", "--kind", "clone", "--ref-audio", str(ref),
        "--ref-text", "你好", "--asr-model", "Qwen3-ASR-0.6B",
        "--i-have-rights",
    )
    assert rc == 2
    assert "requires --auto-ref-text" in capsys.readouterr().err

    rc = _run(
        tmp_path, model_dir,
        "voice", "add", "calm", "--kind", "design", "--description", "calm",
        "--asr-model", "Qwen3-ASR-0.6B",
    )
    assert rc == 2
    assert "requires --auto-ref-text" in capsys.readouterr().err


def test_check_language_supported_by_table_but_not_selected_model(
    monkeypatch, capsys, tmp_path: Path, model_dir: Path
):
    # English-only model; Chinese is in the frozen ASR table but unsupported.
    _add_asr_model(model_dir, languages=["English"])
    audio = _write_wav(tmp_path / "tone.wav")

    def must_not_load():
        raise AssertionError("language rejection must happen before runtime load")

    monkeypatch.setattr(asr_engine, "_runtime", must_not_load)
    rc = _run(
        tmp_path, model_dir,
        "check", str(audio), "--text", "你好", "--language", "Chinese",
    )
    assert rc == 2
    assert "does not support language" in capsys.readouterr().err


def _add_klingon_base(root: Path) -> Path:
    model = root / "Qwen3-TTS-12Hz-1.7B-Base-Klingon"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_tts",
                "tts_model_type": "base",
                "tts_model_size": "1b7",
                "tokenizer_type": "12hz",
                "talker_config": {
                    "codec_language_id": {"chinese": 2055, "klingon": 9999}
                },
            }
        ),
        encoding="utf-8",
    )
    (model / "model.safetensors").write_bytes(b"fake")
    return model


def test_auto_ref_text_invalid_language_cleans_staging(
    capsys, tmp_path: Path, model_dir: Path
):
    _add_asr_model(model_dir)
    klingon = _add_klingon_base(model_dir)
    ref = _write_wav(tmp_path / "ref.wav", seconds=3.0)
    rc = _run(
        tmp_path, model_dir,
        "voice", "add", "bob", "--kind", "clone", "--ref-audio", str(ref),
        "--auto-ref-text", "--language", "Klingon",
        "--model", klingon.name, "--i-have-rights",
    )
    assert rc == 2
    assert "unsupported ASR language" in capsys.readouterr().err
    voices = tmp_path / "data" / "voices"
    assert not (voices / "bob").exists()
    assert list(voices.glob(".bob.*")) == []
