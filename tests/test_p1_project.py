"""FR-P1-09: project init and deterministic lock files."""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from safetensors.numpy import save_file

from qtts_cli import project
from qtts_cli.config import Config
from qtts_cli.errors import InputError, ModelError
from qtts_cli.textnorm import NORMALIZATION_VERSION
from qtts_cli.utils import sha256_file
from qtts_cli.voice_store import VoiceStore

WAV = b"RIFF\x00\x00\x00\x00WAVE"


def _init_args(directory: Path, **overrides):
    data = dict(
        directory=str(directory),
        force=False,
        voice="builtin:Vivian",
        language="Chinese",
        format="wav",
        max_new_tokens=2048,
        subtitle=None,
        granularity="sentence",
        max_cue_chars=42,
        max_line_width=42,
        max_cps=20.0,
        min_cue_seconds=0.8,
        max_cue_seconds=7.0,
        no_subtitle_qc=False,
        model_dir=None,
    )
    data.update(overrides)
    return SimpleNamespace(**data)


def _lock_args(path: Path, out: Path | None = None):
    return SimpleNamespace(project=str(path), out=str(out) if out else None)


def _cfg(tmp_path: Path, model_dir: Path, seed: int = 1234) -> Config:
    return Config(data_dir=tmp_path / "data", model_dir=model_dir, seed=seed)


def _write_project(directory: Path, **overrides) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    config = {
        "schema_version": 1,
        "qtts_version": "test",
        "created_at": "2026-01-01T00:00:00Z",
        "model_dir": str(overrides.pop("model_dir", "")),
        "voice": "builtin:Vivian",
        "voices": [],
        "language": "Chinese",
        "format": "wav",
        "seed": 99,
        "max_new_tokens": 512,
        "subtitle": {
            "formats": [],
            "granularity": "sentence",
            "policy": {
                "max_cue_chars": 42,
                "max_line_width": 42,
                "max_cps": 20.0,
                "min_cue_seconds": 0.8,
                "max_cue_seconds": 7.0,
                "qc_enabled": True,
            },
        },
    }
    config.update(overrides)
    path = directory / project.PROJECT_FILENAME
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def test_init_writes_project_and_refuses_overwrite(tmp_path: Path, model_dir: Path):
    cfg = _cfg(tmp_path, model_dir, seed=7)
    path = tmp_path / "proj"
    first = project.init(cfg, _init_args(path, subtitle="srt"))
    data = json.loads(Path(first["project"]).read_text(encoding="utf-8"))
    assert data["schema_version"] == 1
    assert data["model_dir"] == str(model_dir)
    assert data["voice"] == "builtin:Vivian"
    assert data["seed"] == 7
    assert data["subtitle"]["formats"] == ["srt"]
    assert data["subtitle"]["policy"]["max_cps"] == 20.0
    assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$", data["created_at"])

    with pytest.raises(InputError, match="already exists"):
        project.init(cfg, _init_args(path))
    project.init(cfg, _init_args(path, force=True, voice="builtin:Ryan"))
    data = json.loads(Path(first["project"]).read_text(encoding="utf-8"))
    assert data["voice"] == "builtin:Ryan"


def test_lock_records_full_sha256_and_no_timestamp(tmp_path: Path, model_dir: Path):
    cfg = _cfg(tmp_path, model_dir, seed=1234)
    args = project.init(cfg, _init_args(
        tmp_path / "proj", subtitle="srt", max_new_tokens=512, max_cue_chars=30
    ))
    lock_path = tmp_path / "proj" / project.LOCK_FILENAME
    result = project.lock(cfg, _lock_args(Path(args["project"])))
    assert Path(result["lock"]) == lock_path
    raw = lock_path.read_text(encoding="utf-8")
    data = json.loads(raw)
    assert "created_at" not in data
    assert list(data) == sorted(data)
    assert data["seed"] == 1234
    assert data["sampling"]["max_new_tokens"] == 512
    assert data["normalization"]["version"] == NORMALIZATION_VERSION
    assert data["subtitle_policy"]["max_cue_chars"] == 30
    assert data["dtype"] == "auto"
    assert data["effective_dtype"] in ("bf16", "fp32")
    assert set(data["runtime"]["packages"]) == {"qwen-tts", "qwen-asr", "transformers", "torch"}
    assert data["models"]
    for model in data["models"]:
        model_path = Path(model["path"])
        assert model["files"]
        for entry in model["files"]:
            file_path = model_path / entry["path"]
            assert entry["sha256"] == sha256_file(file_path)
            assert entry["size"] == file_path.stat().st_size
    assert data["aligner"] is not None
    assert data["voices"][0]["ref"] == "builtin:Vivian"


def test_lock_resolves_relative_model_dir_from_project(tmp_path: Path, model_dir: Path):
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    relative = Path("..") / "models"
    path = _write_project(project_dir, model_dir=str(relative))
    cfg = Config(data_dir=tmp_path / "data", model_dir=None, seed=1)
    result = project.lock(cfg, _lock_args(path))
    assert Path(result["lockfile"]["models"][0]["path"]).parent == model_dir


def test_lock_missing_voice_fails_before_writing(tmp_path: Path, model_dir: Path):
    path = _write_project(tmp_path / "proj", model_dir=str(model_dir), voice="voice:missing")
    cfg = _cfg(tmp_path, model_dir)
    with pytest.raises(InputError, match="voice not found"):
        project.lock(cfg, _lock_args(path))
    assert not (tmp_path / "proj" / project.LOCK_FILENAME).exists()


def test_lock_missing_aligner_for_subtitle_fails(tmp_path: Path, small_model_dir: Path):
    cfg = _cfg(tmp_path, small_model_dir)
    path = _write_project(
        tmp_path / "proj", model_dir=str(small_model_dir),
        subtitle={
            "formats": ["srt"],
            "granularity": "sentence",
            "policy": {},
        },
    )
    with pytest.raises(ModelError, match="forced aligner"):
        project.lock(cfg, _lock_args(path))
    assert not (tmp_path / "proj" / project.LOCK_FILENAME).exists()


def test_lock_records_user_voice_hashes_and_all_voices(tmp_path: Path, model_dir: Path):
    cfg = _cfg(tmp_path, model_dir)
    store = VoiceStore(cfg.data_dir)
    staging = store.staging_dir("alice")
    store.create(
        name="alice",
        kind="clone",
        source={
            "type": "clone",
            "mode": "xvec",
            "ref_audio": "ref.wav",
            "ref_audio_sha256": "a" * 64,
        },
        model={"id": "m", "revision": None},
        consent={"rights_confirmed": True, "license": "self-owned", "note": ""},
        staging=staging,
    )
    save_file({"ref_spk_embedding": np.zeros((1, 4), dtype="float32")},
              str(staging / "prompt.safetensors"))
    (staging / "ref.wav").write_bytes(WAV)
    store.commit_staging(staging, "alice")

    path = _write_project(
        tmp_path / "proj",
        model_dir=str(model_dir),
        voice="voice:alice",
        voices=["builtin:Vivian"],
    )
    data = project.lock(cfg, _lock_args(path))["lockfile"]
    assert len(data["voices"]) == 2
    alice = data["voices"][0]
    assert alice["voice_name"] == "alice"
    assert alice["voice_json_sha256"] == sha256_file(store.dir_for("alice") / "voice.json")
    assert alice["prompt_sha256"] == sha256_file(store.dir_for("alice") / "prompt.safetensors")
    assert alice["ref_audio_sha256"] == sha256_file(store.dir_for("alice") / "ref.wav")
    assert alice["consent"]["license"] == "self-owned"
    assert data["voices"][1]["ref"] == "builtin:Vivian"


def test_lock_fails_if_declared_ref_audio_is_missing(tmp_path: Path, model_dir: Path):
    cfg = _cfg(tmp_path, model_dir)
    store = VoiceStore(cfg.data_dir)
    staging = store.staging_dir("alice")
    store.create(
        name="alice",
        kind="clone",
        source={
            "type": "clone",
            "mode": "xvec",
            "ref_audio": "ref.wav",
            "ref_audio_sha256": "a" * 64,
        },
        model={"id": "m", "revision": None},
        staging=staging,
    )
    save_file({"ref_spk_embedding": np.zeros((1, 4), dtype="float32")},
              str(staging / "prompt.safetensors"))
    store.commit_staging(staging, "alice")
    path = _write_project(
        tmp_path / "proj", model_dir=str(model_dir), voice="voice:alice"
    )
    with pytest.raises(InputError, match="ref.wav is missing"):
        project.lock(cfg, _lock_args(path))
    assert not (tmp_path / "proj" / project.LOCK_FILENAME).exists()


def test_lock_hashes_ref_wav_even_when_not_declared(tmp_path: Path, model_dir: Path):
    cfg = _cfg(tmp_path, model_dir)
    store = VoiceStore(cfg.data_dir)
    staging = store.staging_dir("alice")
    store.create(
        name="alice",
        kind="clone",
        source={"type": "clone", "mode": "xvec"},
        model={"id": "m", "revision": None},
        staging=staging,
    )
    save_file({"ref_spk_embedding": np.zeros((1, 4), dtype="float32")},
              str(staging / "prompt.safetensors"))
    (staging / "ref.wav").write_bytes(WAV)
    store.commit_staging(staging, "alice")
    path = _write_project(
        tmp_path / "proj", model_dir=str(model_dir), voice="voice:alice"
    )
    data = project.lock(cfg, _lock_args(path))["lockfile"]
    assert data["voices"][0]["ref_audio_sha256"] == sha256_file(
        store.dir_for("alice") / "ref.wav"
    )


def test_lock_requires_qwen_asr_package_even_when_aligner_exists(
    monkeypatch, tmp_path: Path, model_dir: Path
):
    original = project._package_versions

    def packages_without_asr():
        packages = original()
        packages["qwen-asr"] = {"available": False, "version": None}
        return packages

    monkeypatch.setattr(project, "_package_versions", packages_without_asr)
    path = _write_project(tmp_path / "proj", model_dir=str(model_dir),
                          subtitle={"formats": ["srt"], "granularity": "sentence",
                                    "policy": {}})
    with pytest.raises(ModelError, match="qwen-asr package is not installed"):
        project.lock(_cfg(tmp_path, model_dir), _lock_args(path))
    assert not (tmp_path / "proj" / project.LOCK_FILENAME).exists()
