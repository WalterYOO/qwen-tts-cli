"""Shared fixtures/helpers for FR-P2-07 finetune tests (no torch, no network)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import soundfile as sf

from qtts_cli.cli import build_parser
from qtts_cli.config import build_config

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


def write_tone(
    path: Path,
    *,
    seconds: float = 1.0,
    sample_rate: int = 24000,
    channels: int = 1,
    fmt: str = "WAV",
) -> Path:
    frames = max(1, int(round(seconds * sample_rate)))
    tone = 0.2 * np.sin(2 * np.pi * 440 * np.arange(frames) / sample_rate)
    data = tone.astype(np.float32)
    if channels > 1:
        data = np.stack([data] * channels, axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), data, sample_rate, format=fmt, subtype="PCM_16")
    return path


def write_dataset(root: Path, *, seconds: list[float] | None = None) -> tuple[Path, list[Path], Path]:
    root.mkdir(parents=True, exist_ok=True)
    durations = seconds or [1.0, 1.5]
    items = [
        write_tone(root / f"utt_{index:04d}.wav", seconds=duration)
        for index, duration in enumerate(durations, 1)
    ]
    ref = write_tone(root / "ref.wav", seconds=2.5)
    return root, items, ref


def write_tts_model(
    root: Path,
    name: str,
    tts_model_type: str,
    *,
    speakers: Optional[dict[str, int]] = None,
    size: str = "1b7",
    weights: bytes = b"fake-weights",
    tokenizer: bool = True,
) -> Path:
    model = root / name
    model.mkdir(parents=True, exist_ok=True)
    talker: dict[str, Any] = {"codec_language_id": LANGUAGES}
    if speakers is not None:
        talker["spk_id"] = speakers
    config = {
        "model_type": "qwen3_tts",
        "tts_model_type": tts_model_type,
        "tts_model_size": size,
        "talker_config": talker,
    }
    (model / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (model / "model.safetensors").write_bytes(weights)
    if tokenizer:
        tokenizer_dir = model / "speech_tokenizer"
        tokenizer_dir.mkdir(exist_ok=True)
        (tokenizer_dir / "config.json").write_text(
            json.dumps({"output_sample_rate": 24000}), encoding="utf-8"
        )
        (tokenizer_dir / "model.safetensors").write_bytes(b"fake-tokenizer")
    return model


def write_checkpoint(
    root: Path,
    epoch: int,
    speaker: str = "alice",
    *,
    name: Optional[str] = None,
    tts_model_type: str = "custom_voice",
    model_type: str = "qwen3_tts",
    speaker_id: int = 3000,
    weights: bytes = b"checkpoint-weights",
    tokenizer: bool = True,
    extra_config: Optional[dict[str, Any]] = None,
) -> Path:
    target = root / (name or f"checkpoint-epoch-{epoch}")
    target.mkdir(parents=True, exist_ok=True)
    config = {
        "model_type": model_type,
        "tts_model_type": tts_model_type,
        "tts_model_size": "0b6",
        "talker_config": {
            "spk_id": {speaker: speaker_id},
            "codec_language_id": LANGUAGES,
        },
    }
    if extra_config:
        config.update(extra_config)
    (target / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (target / "model.safetensors").write_bytes(weights)
    if tokenizer:
        tokenizer_dir = target / "speech_tokenizer"
        tokenizer_dir.mkdir(exist_ok=True)
        (tokenizer_dir / "config.json").write_text(
            json.dumps({"output_sample_rate": 24000}), encoding="utf-8"
        )
        (tokenizer_dir / "model.safetensors").write_bytes(b"fake-tokenizer")
    return target


def raw_records(project_path: Path, items: list[Path], ref: Path, texts: list[str]) -> list[dict[str, str]]:
    return [
        {"audio": str(path), "text": text, "ref_audio": str(ref)}
        for path, text in zip(items, texts)
    ]


def prepared_line(record: dict[str, str], rows: int = 1) -> dict[str, Any]:
    return {
        "audio": record["audio"],
        "text": record["text"],
        "ref_audio": record["ref_audio"],
        "audio_codes": [[0] * 16 for _ in range(rows)],
    }


class FakeExternal:
    """Records run_external calls; callbacks create the expected artifact."""

    def __init__(self, on_call=None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.on_call = on_call

    def __call__(self, argv, *, cwd, env, quiet=False):
        self.calls.append(
            {"argv": list(argv), "cwd": Path(cwd), "env": dict(env), "quiet": quiet}
        )
        if self.on_call is not None:
            self.on_call(self, self.calls[-1])
        return 0


def prepared_writer(rows: int = 1, mutate=None):
    """Return a callback that writes a valid prepared JSONL from the raw input."""

    def callback(fake, call):
        argv = call["argv"]
        raw_path = Path(argv[argv.index("--input_jsonl") + 1])
        output = Path(argv[-1])
        records = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines()]
        lines = [prepared_line(record, rows) for record in records]
        if mutate is not None:
            lines = mutate(lines)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            "".join(json.dumps(line, ensure_ascii=False) + "\n" for line in lines),
            encoding="utf-8",
        )

    return callback


def checkpoint_writer(
    epochs: int = 2,
    speaker: str = "alice",
    *,
    weights: Optional[list[bytes]] = None,
    extra_config: Optional[dict[str, Any]] = None,
):
    def callback(fake, call):
        argv = call["argv"]
        output = Path(argv[argv.index("--output_model_path") + 1])
        speaker_name = argv[argv.index("--speaker_name") + 1]
        for epoch in range(epochs):
            payload = weights[epoch] if weights and epoch < len(weights) else f"weights-{epoch}".encode()
            write_checkpoint(
                output,
                epoch,
                speaker_name,
                weights=payload,
                extra_config=extra_config,
            )

    return callback


def write_project(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


def make_project(
    tmp_path: Path,
    *,
    data_dir: Path,
    init_model: Path,
    official: Path,
    output_dir: Path,
    ref: Path,
    items: list[Path],
    texts: list[str] | None = None,
    extra: str = "",
) -> Path:
    texts = texts or [f"第 {index} 句测试文本。" for index in range(1, len(items) + 1)]
    lines = [
        "schema_version: 1",
        "dataset:",
        f"  ref_audio: {ref}",
        "  items:",
    ]
    for item, text in zip(items, texts):
        lines.extend([f"    - audio: {item}", f'      text: "{text}"'])
    lines.extend(
        [
            "train:",
            f"  finetuning_dir: {official}",
            f"  init_model_path: {init_model}",
            f"  output_dir: {output_dir}",
            "  speaker_name: alice",
            "",
        ]
    )
    body = "\n".join(lines) + (extra or "")
    return write_project(tmp_path / "project.yaml", body)


def make_official(root: Path, *, prepare: bool = True, train: bool = True) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    if prepare:
        (root / "prepare_data.py").write_text("# fake\n", encoding="utf-8")
    if train:
        (root / "sft_12hz.py").write_text("# fake\n", encoding="utf-8")
    return root


def parse_args(argv: list[str]):
    return build_parser().parse_args(argv)


def config_for(args, *, data_dir: Optional[Path] = None, model_dir: Optional[Path] = None):
    if data_dir is not None:
        args.data_dir = str(data_dir)
    if model_dir is not None:
        args.model_dir = str(model_dir)
    return build_config(args)


def cli_json(capsys, argv: list[str]):
    from qtts_cli.cli import main

    code = main(argv)
    out = capsys.readouterr()
    payload = json.loads(out.out) if out.out.strip() else None
    error = json.loads(out.err) if out.err.strip().startswith("{") else None
    return code, payload, error, out
