"""Shared fixtures: synthetic model directories that need no torch."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

SPEAKERS = {
    "serena": 3066,
    "vivian": 3065,
    "uncle_fu": 3010,
    "ryan": 3061,
    "aiden": 2861,
    "ono_anna": 2873,
    "sohee": 2864,
    "eric": 2875,
    "dylan": 2878,
}
LANGUAGES = {
    "chinese": 2055,
    "english": 2050,
    "german": 2053,
    "italian": 2070,
    "portuguese": 2071,
    "spanish": 2054,
    "japanese": 2058,
    "korean": 2064,
    "french": 2061,
    "russian": 2069,
}


def _write_tts_model(
    root: Path,
    name: str,
    tts_model_type: str,
    size: str,
    *,
    speakers: dict[str, int] | None = None,
    readme: bool = True,
) -> Path:
    model = root / name
    model.mkdir(parents=True)
    talker = {"codec_language_id": LANGUAGES}
    if speakers:
        talker["spk_id"] = speakers
    config = {
        "model_type": "qwen3_tts",
        "tts_model_type": tts_model_type,
        "tts_model_size": size,
        "tokenizer_type": "qwen3_tts_tokenizer_12hz",
        "talker_config": talker,
    }
    (model / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"fake")
    tokenizer = model / "speech_tokenizer"
    tokenizer.mkdir()
    (tokenizer / "config.json").write_text(
        json.dumps({"output_sample_rate": 24000, "encode_downsample_rate": 1920}), encoding="utf-8"
    )
    (tokenizer / "model.safetensors").write_bytes(b"fake")
    if readme and speakers:
        rows = ["| Speaker | Voice Description | Native language |", "| --- | --- | --- |"]
        for key in speakers:
            rows.append(f"| {key.title()} | synthetic | Chinese |")
        (model / "README.md").write_text("\n".join(rows), encoding="utf-8")
    return model


@pytest.fixture()
def model_dir(tmp_path: Path) -> Path:
    root = tmp_path / "models"
    root.mkdir()
    _write_tts_model(root, "Qwen3-TTS-12Hz-1.7B-CustomVoice", "custom_voice", "1b7", speakers=SPEAKERS)
    _write_tts_model(root, "Qwen3-TTS-12Hz-1.7B-Base", "base", "1b7")
    _write_tts_model(root, "Qwen3-TTS-12Hz-1.7B-VoiceDesign", "voice_design", "1b7")
    aligner = root / "Qwen3-ForcedAligner-0.6B"
    aligner.mkdir()
    (aligner / "config.json").write_text(
        json.dumps({"model_type": "qwen3_asr", "support_languages": ["chinese", "english"]}),
        encoding="utf-8",
    )
    (aligner / "model.safetensors").write_bytes(b"fake")
    return root


@pytest.fixture()
def small_model_dir(tmp_path: Path) -> Path:
    """CustomVoice 0.6B (no instruct) only."""
    root = tmp_path / "models_small"
    root.mkdir()
    _write_tts_model(root, "Qwen3-TTS-12Hz-0.6B-CustomVoice", "custom_voice", "0b6", speakers=SPEAKERS)
    return root
