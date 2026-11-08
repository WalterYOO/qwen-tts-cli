"""FR-P2-07 §10/§12.10: generation, CER/similarity, gates and eval CLI."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from qtts_cli import catalog
from qtts_cli.cli import main
from qtts_cli.errors import InputError, ModelError
from qtts_cli.finetune import evaluate as evaluate_mod
from qtts_cli.qc import levenshtein, normalize_for_qc, text_units
from qtts_cli.voice_store import VoiceStore
from finetune_utils import (
    make_project,
    write_checkpoint,
    write_dataset,
    write_tone,
    write_tts_model,
)


def make_model(tmp_path: Path) -> tuple[Path, "catalog.ModelInfo"]:
    checkpoint = write_checkpoint(tmp_path / "out", 0, "alice")
    return checkpoint, catalog.load_model(checkpoint)


def items(*texts: str, languages=None):
    languages = languages or [None] * len(texts)
    return [
        SimpleNamespace(text=text, language=language)
        for text, language in zip(texts, languages)
    ]


def dataset(ref: Path, language: str = "Auto"):
    return SimpleNamespace(ref_audio=ref, language=language)


def run_evaluate(
    monkeypatch,
    tmp_path: Path,
    *,
    texts=("hello",),
    languages=None,
    asr_texts=None,
    base_vectors=None,
    ref: Path | None = None,
    audio_dir: Path | None = None,
    eval_language: str = "Auto",
    dataset_language: str = "Auto",
    max_cer=None,
    min_similarity=None,
    engine=None,
):
    checkpoint, model = make_model(tmp_path)
    engine = engine or FakeEngine()
    asr = None if asr_texts is None else FakeAsr(asr_texts)
    base = None if base_vectors is None else FakeBase(base_vectors)
    calls = {"engine": 0, "asr": 0, "base": 0, "tts_cache": 0, "asr_cache": 0}

    def engine_factory(cfg, model):
        calls["engine"] += 1
        return engine

    def asr_factory(cfg, model):
        calls["asr"] += 1
        return asr

    def base_factory(cfg, model):
        calls["base"] += 1
        return base

    monkeypatch.setattr(evaluate_mod, "_engine_factory", engine_factory)
    monkeypatch.setattr(evaluate_mod, "_asr_factory", asr_factory)
    monkeypatch.setattr(evaluate_mod, "_base_factory", base_factory)
    monkeypatch.setattr(evaluate_mod, "_clear_tts_cache", lambda: calls.__setitem__("tts_cache", calls["tts_cache"] + 1))
    monkeypatch.setattr(evaluate_mod, "_clear_asr_cache", lambda: calls.__setitem__("asr_cache", calls["asr_cache"] + 1))
    cfg = SimpleNamespace(seed=7)
    ref = ref or write_tone(tmp_path / "ref.wav", seconds=2.5)
    payload = evaluate_mod.evaluate(
        cfg,
        model=model,
        speaker="alice",
        checkpoint_path=checkpoint,
        voice_name=None,
        items=items(*texts, languages=languages),
        dataset=dataset(ref, dataset_language),
        asr_model=asr if asr is not None else None,
        base_model=base if base is not None else None,
        max_cer=max_cer,
        min_speaker_similarity=min_similarity,
        audio_dir=audio_dir,
        eval_language=eval_language,
    )
    payload["_calls"] = calls
    payload["_engine"] = engine
    payload["_asr"] = asr
    payload["_base"] = base
    payload["_checkpoint"] = checkpoint
    return payload


class FakeEngine:
    def __init__(self, audio=None, sr=24000):
        self.calls: list[dict] = []
        self.model = None
        self._audio = audio
        self._sr = sr

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        audio = self._audio
        if audio is None:
            audio = np.ones(2400, dtype=np.float32)
        return np.asarray(audio, dtype=np.float32), self._sr, {"generate_ms": 1.0}


class FakeAsr:
    def __init__(self, texts):
        self.texts = list(texts)
        self.calls: list[str] = []
        self.released = False

    def transcribe_samples(self, audio, sr, *, language):
        self.calls.append(language)
        return {"text": self.texts.pop(0)}

    def release(self):
        self.released = True


class FakeBase:
    def __init__(self, vectors):
        self.vectors = list(vectors)
        self.calls = 0

    def extract_speaker_embedding(self, audio, sr):
        self.calls += 1
        if self.calls == 1:
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)
        return np.asarray(self.vectors.pop(0), dtype=np.float32)


def test_cer_exact_math_and_per_item_payload(tmp_path: Path, monkeypatch):
    payload = run_evaluate(
        monkeypatch,
        tmp_path,
        texts=("hello world", "abc"),
        languages=("English", "English"),
        asr_texts=("hello word", "abd"),
        audio_dir=tmp_path / "audio",
    )
    items_payload = payload["items"]
    assert len(items_payload) == 2
    expected = []
    for index, (reference, hypothesis) in enumerate(
        [("hello world", "hello word"), ("abc", "abd")], 1
    ):
        ref_units = text_units(normalize_for_qc(reference, "English"), "cer")
        hyp_units = text_units(normalize_for_qc(hypothesis, "English"), "cer")
        distance = levenshtein(ref_units, hyp_units)
        expected.append((distance, len(ref_units)))
        record = items_payload[index - 1]
        assert record["index"] == index
        assert record["cer_distance"] == distance
        assert record["cer_units"] == len(ref_units)
        assert record["cer"] == round(distance / len(ref_units), 6)
        assert record["audio"] == str(tmp_path / "audio" / f"{index:04d}.wav")
        assert record["language"] == "English"
        assert record["duration_s"] == 0.1
        assert len(record["text_sha256"]) == 64
    total_distance = sum(distance for distance, _ in expected)
    total_units = sum(units for _, units in expected)
    assert payload["metrics"]["cer_distance"] == total_distance
    assert payload["metrics"]["cer_units"] == total_units
    assert payload["metrics"]["cer"] == round(total_distance / total_units, 6)
    assert payload["_calls"]["tts_cache"] == 1
    assert payload["_calls"]["asr_cache"] == 1


def test_similarity_mean_and_null_propagation(tmp_path: Path, monkeypatch):
    payload = run_evaluate(
        monkeypatch,
        tmp_path,
        texts=("a", "b", "c"),
        base_vectors=(
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
        ),
    )
    values = [record["speaker_similarity"] for record in payload["items"]]
    assert values[0] == 1.0
    assert values[1] is None
    assert values[2] == 0.0
    assert payload["metrics"]["speaker_similarity"] is None
    assert payload["metrics"]["speaker_segments_scored"] == 2
    assert payload["metrics"]["speaker_segments_total"] == 3


def test_similarity_all_valid(tmp_path: Path, monkeypatch):
    payload = run_evaluate(
        monkeypatch,
        tmp_path,
        texts=("a", "b"),
        base_vectors=(np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])),
    )
    assert payload["metrics"]["speaker_similarity"] == 1.0
    assert payload["metrics"]["speaker_segments_scored"] == 2
    assert payload["metrics"]["speaker_segments_total"] == 2


def test_metric_key_sets_absent_vs_null(tmp_path: Path, monkeypatch):
    payload = run_evaluate(monkeypatch, tmp_path)
    assert payload["metrics"] == {}
    assert set(payload["items"][0]) == {
        "index",
        "text_sha256",
        "language",
        "duration_s",
        "audio",
    }

    payload = run_evaluate(
        monkeypatch,
        tmp_path / "cer",
        texts=("!!!",),
        languages=("English",),
        asr_texts=("",),
    )
    assert set(payload["metrics"]) == {"cer", "cer_distance", "cer_units"}
    assert payload["metrics"] == {"cer": None, "cer_distance": 0, "cer_units": 0}
    assert payload["items"][0]["cer"] is None
    assert payload["items"][0]["cer_units"] == 0

    payload = run_evaluate(
        monkeypatch,
        tmp_path / "sim",
        texts=("a",),
        base_vectors=(np.array([1.0, 0.0, 0.0]),),
    )
    assert set(payload["metrics"]) == {
        "speaker_similarity",
        "speaker_segments_scored",
        "speaker_segments_total",
    }
    assert "cer" not in payload["metrics"]


def test_audio_names_pad_and_grow_past_9999(tmp_path: Path, monkeypatch):
    payload = run_evaluate(
        monkeypatch,
        tmp_path,
        texts=tuple(f"t{index}" for index in range(10000)),
        audio_dir=tmp_path / "audio",
    )
    assert (tmp_path / "audio" / "0001.wav").is_file()
    assert (tmp_path / "audio" / "9999.wav").is_file()
    assert (tmp_path / "audio" / "10000.wav").is_file()
    assert payload["items"][9999]["audio"].endswith("10000.wav")


def test_audio_atomically_replaced(tmp_path: Path, monkeypatch):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    target = audio_dir / "0001.wav"
    target.write_bytes(b"old")
    run_evaluate(monkeypatch, tmp_path, audio_dir=audio_dir)
    assert target.read_bytes() != b"old"
    assert not list(audio_dir.glob("*.tmp"))


def test_language_precedence_and_generation_kwargs(tmp_path: Path, monkeypatch):
    engine = FakeEngine()
    payload = run_evaluate(
        monkeypatch,
        tmp_path,
        texts=("hello", "你好", "bonjour"),
        languages=("English", None, None),
        eval_language="Chinese",
        dataset_language="French",
        engine=engine,
    )
    assert [item["language"] for item in payload["items"]] == [
        "English",
        "Chinese",
        "Chinese",
    ]
    assert [call["language"] for call in engine.calls] == ["English", "Chinese", "Chinese"]
    assert all(call["mode"] == "builtin" for call in engine.calls)
    assert all(call["speaker"] == "alice" for call in engine.calls)
    assert all(call["instruct"] is None for call in engine.calls)
    assert all(call["seed"] == 7 for call in engine.calls)


def test_release_in_finally_on_error(tmp_path: Path, monkeypatch):
    engine = FakeEngine()

    def fail(**kwargs):
        engine.calls.append(kwargs)
        raise ModelError("generation failed")

    engine.generate = fail
    with pytest.raises(ModelError):
        run_evaluate(monkeypatch, tmp_path, engine=engine)
    assert engine.model is None


def test_gate_exceeded_below_and_unavailable(tmp_path: Path, monkeypatch):
    payload = run_evaluate(
        monkeypatch,
        tmp_path,
        texts=("hello",),
        languages=("English",),
        asr_texts=("xxxxx",),
        max_cer=0.1,
    )
    assert payload["exit_code"] == 5
    assert payload["ok"] is True
    assert payload["gate_failures"][0]["gate"] == "max_cer"
    assert payload["gate_failures"][0]["reason"] == "exceeded"

    payload = run_evaluate(
        monkeypatch,
        tmp_path / "below",
        texts=("!!!",),
        asr_texts=("",),
        max_cer=0.1,
    )
    assert payload["gate_failures"][0]["reason"] == "unavailable"
    assert payload["gate_failures"][0]["value"] is None

    payload = run_evaluate(
        monkeypatch,
        tmp_path / "sim",
        texts=("a",),
        base_vectors=(np.array([0.0, 1.0, 0.0]),),
        min_similarity=0.9,
    )
    assert payload["exit_code"] == 5
    assert payload["gate_failures"][0]["gate"] == "min_speaker_similarity"
    assert payload["gate_failures"][0]["reason"] == "below"

    payload = run_evaluate(
        monkeypatch,
        tmp_path / "null",
        texts=("a",),
        base_vectors=(np.array([0.0, 0.0, 0.0]),),
        min_similarity=0.9,
    )
    assert payload["gate_failures"][0]["reason"] == "unavailable"


def test_no_thresholds_is_zero_and_warnings_present(tmp_path: Path, monkeypatch):
    payload = run_evaluate(monkeypatch, tmp_path)
    assert payload["exit_code"] == 0
    assert payload["gate_failures"] == []
    assert payload["warnings"] == []


def test_cer_only_ignores_missing_ref_but_similarity_requires_it(tmp_path: Path, monkeypatch):
    missing = tmp_path / "missing.wav"
    payload = run_evaluate(
        monkeypatch,
        tmp_path,
        texts=("hello",),
        asr_texts=("hello",),
        ref=missing,
    )
    assert payload["metrics"]["cer"] == 0.0

    with pytest.raises(InputError) as exc:
        run_evaluate(
            monkeypatch,
            tmp_path / "sim",
            texts=("hello",),
            base_vectors=(np.array([1.0, 0.0, 0.0]),),
            ref=missing,
        )
    assert exc.value.code == 4


# --------------------------------------------------------------------------- #
# CLI/orchestration
# --------------------------------------------------------------------------- #
def make_cli_env(tmp_path: Path, eval_body: str, *, train: bool = True):
    models = tmp_path / "models"
    models.mkdir()
    base = write_tts_model(models, "Qwen3-TTS-12Hz-1.7B-Base", "base")
    _, audio_items, ref = write_dataset(tmp_path / "audio")
    extra = "eval:\n" + eval_body
    project = make_project(
        tmp_path,
        data_dir=tmp_path / "data",
        init_model=base,
        official=make_official(tmp_path / "official"),
        output_dir=tmp_path / "out",
        ref=ref,
        items=audio_items,
        extra=extra,
    )
    checkpoint = write_checkpoint(tmp_path / "out", 0, "alice")
    return project, models, checkpoint, ref, base


def make_official(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "prepare_data.py").write_text("# fake", encoding="utf-8")
    (root / "sft_12hz.py").write_text("# fake", encoding="utf-8")
    return root


def run_eval_cli(tmp_path: Path, models: Path, *args: str) -> int:
    return main(
        [
            "finetune",
            "eval",
            *args,
            "--model-dir",
            str(models),
            "--data-dir",
            str(tmp_path / "data"),
            "--cache-dir",
            str(tmp_path / "cache"),
        ]
    )


def make_profile(tmp_path: Path, checkpoint: Path, *, name: str = "alice") -> VoiceStore:
    from qtts_cli.utils import sha256_file, sha256_text

    weights_sha = sha256_file(checkpoint / "model.safetensors")
    config_sha = sha256_file(checkpoint / "config.json")
    checkpoint_sha = sha256_text(f"{weights_sha}:{config_sha}:{name}")
    store = VoiceStore(tmp_path / "data")
    store.create(
        name=name,
        kind="finetune",
        source={
            "type": "finetune",
            "speaker": name,
            "weights_sha256": weights_sha,
            "weights_size": (checkpoint / "model.safetensors").stat().st_size,
            "config_sha256": config_sha,
            "checkpoint_sha256": checkpoint_sha,
            "output_dir": str(checkpoint.parent),
        },
        model={
            "id": checkpoint.name,
            "revision": None,
            "fingerprint": checkpoint_sha[:16],
            "path": str(checkpoint),
        },
    )
    return store


def test_voice_overrides_project_checkpoint(tmp_path: Path, capsys):
    project, models, epoch0, ref, _ = make_cli_env(
        tmp_path,
        "  checkpoint: 0\n"
        "  items:\n"
        "    - text: hello\n"
        "      language: English\n",
    )
    other = write_checkpoint(tmp_path / "elsewhere", 1, "alice")
    make_profile(tmp_path, other)
    code = run_eval_cli(
        tmp_path, models, "--project", str(project), "--voice", "alice",
        "--dry-run", "--json",
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["voice"] == "alice"
    assert payload["checkpoint"] == str(other)
    assert payload["speaker"] == "alice"
    assert payload["models_loaded"] is False


def test_checkpoint_and_voice_is_2(tmp_path: Path):
    project, models, epoch0, _, _ = make_cli_env(
        tmp_path, "  items:\n    - text: hello\n"
    )
    make_profile(tmp_path, epoch0)
    assert (
        run_eval_cli(
            tmp_path,
            models,
            "--project",
            str(project),
            "--voice",
            "alice",
            "--checkpoint",
            "0",
            "--dry-run",
            "--json",
        )
        == 2
    )


def test_missing_or_non_finetune_voice_is_2(tmp_path: Path):
    project, models, epoch0, _, _ = make_cli_env(
        tmp_path, "  items:\n    - text: hello\n"
    )
    store = VoiceStore(tmp_path / "data")
    store.create(
        name="cloned",
        kind="clone",
        source={},
        model={"id": "x", "revision": None, "fingerprint": "0" * 16, "path": str(epoch0)},
    )
    assert (
        run_eval_cli(
            tmp_path, models, "--project", str(project), "--voice", "absent",
            "--dry-run", "--json",
        )
        == 2
    )
    assert (
        run_eval_cli(
            tmp_path, models, "--project", str(project), "--voice", "cloned",
            "--dry-run", "--json",
        )
        == 2
    )


def test_changed_or_missing_profile_checkpoint_is_4(tmp_path: Path):
    project, models, epoch0, _, _ = make_cli_env(
        tmp_path, "  items:\n    - text: hello\n"
    )
    make_profile(tmp_path, epoch0)
    weights = epoch0 / "model.safetensors"
    weights.write_bytes(weights.read_bytes() + b"x")
    assert (
        run_eval_cli(
            tmp_path, models, "--project", str(project), "--voice", "alice",
            "--dry-run", "--json",
        )
        == 4
    )


def test_voice_uses_profile_speaker_and_ignores_train(tmp_path: Path):
    models = tmp_path / "models"
    models.mkdir()
    write_tts_model(models, "Qwen3-TTS-12Hz-1.7B-Base", "base")
    _, audio_items, ref = write_dataset(tmp_path / "audio")
    checkpoint = write_checkpoint(tmp_path / "out", 0, "alice")
    make_profile(tmp_path, checkpoint)
    project = tmp_path / "only.yaml"
    project.write_text(
        "\n".join(
            [
                "schema_version: 1",
                "dataset:",
                f"  ref_audio: {ref}",
                "  items:",
                f"    - audio: {audio_items[0]}",
                '      text: "hello"',
                "eval:",
                "  items:",
                "    - text: hello",
            ]
        ),
        encoding="utf-8",
    )
    # No train block: the profile supplies the speaker.
    code = main(
        [
            "finetune",
            "eval",
            "--project",
            str(project),
            "--voice",
            "alice",
            "--dry-run",
            "--json",
            "--model-dir",
            str(models),
            "--data-dir",
            str(tmp_path / "data"),
        ]
    )
    assert code == 0


def test_missing_output_dir_is_4(tmp_path: Path):
    project, models, _, _, _ = make_cli_env(
        tmp_path, "  items:\n    - text: hello\n"
    )
    import shutil

    shutil.rmtree(tmp_path / "out")
    assert (
        run_eval_cli(
            tmp_path, models, "--project", str(project), "--dry-run", "--json"
        )
        == 4
    )


def test_speaker_absent_is_4(tmp_path: Path):
    project, models, _, _, _ = make_cli_env(
        tmp_path, "  items:\n    - text: hello\n"
    )
    text = Path(project).read_text(encoding="utf-8")
    Path(project).write_text(
        text.replace("speaker_name: alice", "speaker_name: nobody"), encoding="utf-8"
    )
    assert (
        run_eval_cli(
            tmp_path, models, "--project", str(project), "--dry-run", "--json"
        )
        == 4
    )


def test_max_cer_without_asr_is_2(tmp_path: Path):
    project, models, _, _, _ = make_cli_env(
        tmp_path, "  items:\n    - text: hello\n"
    )
    assert (
        run_eval_cli(
            tmp_path, models, "--project", str(project), "--max-cer", "0.1",
            "--dry-run", "--json",
        )
        == 2
    )


def test_similarity_without_base_spec_auto_selects_base(tmp_path: Path, capsys):
    # §10.1: no explicit base_model, but similarity requested -> select(clone).
    project, models, _, _, _ = make_cli_env(
        tmp_path, "  items:\n    - text: hello\n"
    )
    assert (
        run_eval_cli(
            tmp_path, models, "--project", str(project),
            "--min-speaker-similarity", "0.5", "--dry-run", "--json",
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["base_model"] == "Qwen3-TTS-12Hz-1.7B-Base"


def test_similarity_without_usable_base_is_4(tmp_path: Path):
    project, models, _, _, _ = make_cli_env(
        tmp_path, "  items:\n    - text: hello\n"
    )
    import shutil

    for child in models.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
    assert (
        run_eval_cli(
            tmp_path, models, "--project", str(project),
            "--min-speaker-similarity", "0.5", "--dry-run", "--json",
        )
        == 4
    )


def test_dry_run_audio_dir_file_is_4(tmp_path: Path):
    project, models, _, _, _ = make_cli_env(
        tmp_path, "  items:\n    - text: hello\n"
    )
    audio_dir = tmp_path / "audio-dir"
    audio_dir.write_text("not a directory", encoding="utf-8")
    assert (
        run_eval_cli(
            tmp_path, models, "--project", str(project), "--audio-dir",
            str(audio_dir), "--dry-run", "--json",
        )
        == 4
    )
    assert not (tmp_path / "out" / "0001.wav").exists()


def test_dry_run_ref_audio_missing_for_similarity_is_4(tmp_path: Path):
    project, models, _, _, _ = make_cli_env(
        tmp_path, "  items:\n    - text: hello\n"
    )
    text = Path(project).read_text(encoding="utf-8")
    missing = tmp_path / "gone.wav"
    Path(project).write_text(
        text.replace(f"  ref_audio: {tmp_path / 'audio' / 'ref.wav'}",
                     f"  ref_audio: {missing}"),
        encoding="utf-8",
    )
    assert (
        run_eval_cli(
            tmp_path, models, "--project", str(project),
            "--min-speaker-similarity", "0.5", "--dry-run", "--json",
        )
        == 4
    )
    assert not missing.exists()


def test_asr_resolution_is_4_and_load_is_3(tmp_path: Path, monkeypatch):
    project, models, _, _, _ = make_cli_env(
        tmp_path, "  items:\n    - text: hello\n"
    )
    assert (
        run_eval_cli(
            tmp_path, models, "--project", str(project), "--asr-model", "missing",
            "--dry-run", "--json",
        )
        == 4
    )

    asr_dir = models / "Qwen3-ASR-0.6B"
    asr_dir.mkdir()
    (asr_dir / "config.json").write_text(
        json.dumps({"model_type": "qwen3_asr", "support_languages": ["english"]}),
        encoding="utf-8",
    )

    def boom(cfg, model):
        raise ModelError("asr load failed")

    monkeypatch.setattr(evaluate_mod, "_asr_factory", boom)
    monkeypatch.setattr(evaluate_mod, "_engine_factory", lambda cfg, model: FakeEngine())
    monkeypatch.setattr(evaluate_mod, "_clear_tts_cache", lambda: None)
    assert (
        run_eval_cli(
            tmp_path, models, "--project", str(project), "--asr-model",
            "Qwen3-ASR-0.6B", "--json",
        )
        == 3
    )


def test_base_resolution_is_4_and_load_is_3(tmp_path: Path, monkeypatch):
    project, models, _, _, _ = make_cli_env(
        tmp_path, "  items:\n    - text: hello\n"
    )
    assert (
        run_eval_cli(
            tmp_path, models, "--project", str(project), "--base-model", "missing",
            "--dry-run", "--json",
        )
        == 4
    )

    def boom(cfg, model):
        raise ModelError("base load failed")

    monkeypatch.setattr(evaluate_mod, "_base_factory", boom)
    monkeypatch.setattr(evaluate_mod, "_engine_factory", lambda cfg, model: FakeEngine())
    monkeypatch.setattr(evaluate_mod, "_clear_tts_cache", lambda: None)
    assert (
        run_eval_cli(
            tmp_path, models, "--project", str(project), "--base-model",
            "Qwen3-TTS-12Hz-1.7B-Base", "--json",
        )
        == 3
    )


def test_unreadable_ref_for_similarity_is_4(tmp_path: Path, monkeypatch):
    with pytest.raises(InputError) as exc:
        run_evaluate(
            monkeypatch,
            tmp_path,
            texts=("hello",),
            base_vectors=(np.array([1.0, 0.0, 0.0]),),
            ref=tmp_path,
        )
    assert exc.value.code == 4


def test_cli_gate_failure_exits_5(tmp_path: Path, monkeypatch, capsys):
    project, models, _, _, _ = make_cli_env(
        tmp_path,
        "  asr_model: Qwen3-ASR-0.6B\n"
        "  thresholds:\n"
        "    max_cer: 0.1\n"
        "  items:\n"
        "    - text: hello\n",
    )
    asr_dir = models / "Qwen3-ASR-0.6B"
    asr_dir.mkdir()
    (asr_dir / "config.json").write_text(
        json.dumps({"model_type": "qwen3_asr", "support_languages": ["english"]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(evaluate_mod, "_engine_factory", lambda cfg, model: FakeEngine())
    monkeypatch.setattr(
        evaluate_mod, "_asr_factory", lambda cfg, model: FakeAsr(["completely wrong"])
    )
    monkeypatch.setattr(evaluate_mod, "_clear_tts_cache", lambda: None)
    monkeypatch.setattr(evaluate_mod, "_clear_asr_cache", lambda: None)
    assert (
        run_eval_cli(tmp_path, models, "--project", str(project), "--json") == 5
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["exit_code"] == 5
    assert payload["gate_failures"][0]["reason"] == "exceeded"


def test_dry_run_does_not_generate_or_load(tmp_path: Path, monkeypatch, capsys):
    project, models, _, _, _ = make_cli_env(
        tmp_path,
        "  audio_dir: eval-audio\n"
        "  items:\n"
        "    - text: hello\n",
    )

    def boom(*args, **kwargs):
        raise AssertionError("dry-run must not create engines")

    monkeypatch.setattr(evaluate_mod, "_engine_factory", boom)
    assert (
        run_eval_cli(
            tmp_path, models, "--project", str(project), "--dry-run", "--json"
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["models_loaded"] is False
    assert payload["items"] == 1
    assert payload["will_write"] == {"audio_dir": str(tmp_path / "eval-audio")}
    assert not (tmp_path / "eval-audio").exists()


def test_eval_module_import_is_torch_free():
    code = (
        "import sys\n"
        "from qtts_cli.finetune import evaluate\n"
        "bad = [m for m in ('torch','qwen_tts','qwen_asr','dashscope',"
        "'vllm_omni','accelerate','librosa','safetensors') if m in sys.modules]\n"
        "print(','.join(bad))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "PYTHONPATH": "."},
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == ""
