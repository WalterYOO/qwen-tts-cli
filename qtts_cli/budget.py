"""Token / duration budget estimation (FR-P0-04).

The 12Hz speech tokenizer emits ``output_sample_rate / encode_downsample_rate``
frames per second (24000 / 1920 = 12.5). That is derived from the model's own
tokenizer config, never hard-coded.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .catalog import ModelInfo

DEFAULT_MAX_NEW_TOKENS = 2048
DEFAULT_CODEC_RATE = 12.5

# Conservative speaking rates (characters per second of audio). Chinese TTS is
# typically 4.5-5.5 chars/s; English narration ~14-16 chars/s.
CHARS_PER_SECOND = {
    "Chinese": 4.5,
    "English": 15.0,
    "Japanese": 5.0,
    "Korean": 5.5,
    "Cantonese": 5.0,
}
FALLBACK_CHARS_PER_SECOND = 8.0

# Fraction of max_new_tokens considered safe before segmenting further.
BUDGET_MARGIN = 0.9


def codec_rate(model: Optional[ModelInfo]) -> float:
    """Frames (codec tokens) per second, read from speech_tokenizer/config.json."""
    if model is None:
        return DEFAULT_CODEC_RATE
    config_path = Path(model.path) / "speech_tokenizer" / "config.json"
    try:
        with open(config_path, "r", encoding="utf-8") as fh:
            cfg = json.load(fh)
        sample_rate = float(cfg.get("output_sample_rate") or cfg.get("input_sample_rate") or 0)
        downsample = float(cfg.get("encode_downsample_rate") or 0)
        if sample_rate > 0 and downsample > 0:
            return sample_rate / downsample
    except (OSError, ValueError, json.JSONDecodeError):
        pass
    return DEFAULT_CODEC_RATE


def chars_per_second(language: str) -> float:
    return CHARS_PER_SECOND.get(language, FALLBACK_CHARS_PER_SECOND)


def estimate_duration_s(text: str, language: str = "Auto") -> float:
    visible = sum(1 for ch in text if not ch.isspace())
    return visible / chars_per_second(language)


def estimate_tokens(text: str, language: str = "Auto", rate: float = DEFAULT_CODEC_RATE) -> int:
    return max(1, int(round(estimate_duration_s(text, language) * rate)))


def usable_tokens(max_new_tokens: int, margin: float = BUDGET_MARGIN) -> int:
    return max(1, int(max_new_tokens * margin))


def max_chars(language: str, max_new_tokens: int, rate: float = DEFAULT_CODEC_RATE) -> int:
    """Largest character budget that fits in max_new_tokens."""
    seconds = usable_tokens(max_new_tokens) / rate
    return max(1, int(seconds * chars_per_second(language)))


def fits(text: str, language: str, max_new_tokens: int, rate: float = DEFAULT_CODEC_RATE) -> bool:
    return estimate_tokens(text, language, rate) <= usable_tokens(max_new_tokens)
