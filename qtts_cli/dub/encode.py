"""Lossless/lossy encoding, media probing and muxing for dub (FR-P2-04 §7)."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..audiobook import encode as audiobook_encode
from ..errors import InputError, ModelError
from ..utils import sha256_file


def require_ffprobe() -> str:
    probe = shutil.which("ffprobe")
    if probe is None:
        raise ModelError("ffprobe is required but was not found on PATH")
    return probe


def require_ffmpeg(fmt: str) -> str:
    """Preflight ffmpeg and the encoder for a lossy format."""
    return audiobook_encode.require_ffmpeg(fmt)


def probe_media(path: Path) -> dict[str, Any]:
    """Return ``duration_s`` and stream presence via ffprobe (exit 4 on bogus)."""
    probe = require_ffprobe()
    completed = subprocess.run(
        [
            probe,
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=codec_type",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise InputError(
            f"cannot probe video {path}: {completed.stderr.strip()}",
            details={"video": str(path)},
        )
    try:
        payload = json.loads(completed.stdout or "{}")
        duration = float(payload["format"]["duration"])
        kinds = {
            str(stream.get("codec_type"))
            for stream in payload.get("streams") or []
            if isinstance(stream, dict)
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise InputError(
            f"cannot read the duration of {path}", details={"video": str(path)}
        ) from exc
    if not np.isfinite(duration) or duration <= 0:
        raise InputError(
            f"video has no usable duration: {path}", details={"video": str(path)}
        )
    return {
        "duration_s": float(duration),
        "has_video": "video" in kinds,
        "has_audio": "audio" in kinds,
    }


def _write_error(path: Path, exc: BaseException) -> ModelError:
    return ModelError(
        f"cannot write audio output {path}: {exc}", details={"path": str(path)}
    )


def write_lossless(path: Path, master: np.ndarray, sr: int, fmt: str) -> None:
    # ``soundfile.LibsndfileError`` / ``SoundFileError`` derive from
    # RuntimeError (not OSError or QttsError); map both to the contract's
    # write-failure exit code (3) without swallowing cancellation or memory
    # errors.
    try:
        audiobook_encode.write_lossless(path, master, int(sr), fmt)
    except (RuntimeError, OSError) as exc:
        raise _write_error(Path(path), exc) from exc
    except Exception as exc:
        # ``soundfile.SoundFileError`` itself is not a RuntimeError in every
        # soundfile release; only remap that specific class.
        import soundfile

        if isinstance(exc, soundfile.SoundFileError):
            raise _write_error(Path(path), exc) from exc
        raise


def write_master_wav(path: Path, master: np.ndarray, sr: int) -> None:
    write_lossless(path, master, int(sr), "wav")


def output_sha256(path: Path) -> str:
    try:
        return sha256_file(Path(path))
    except OSError as exc:
        raise ModelError(
            f"cannot read back output {path}: {exc}", details={"path": str(path)}
        ) from exc


def output_size(path: Path) -> int:
    try:
        return int(Path(path).stat().st_size)
    except OSError as exc:
        raise ModelError(
            f"cannot stat output {path}: {exc}", details={"path": str(path)}
        ) from exc


def encode_lossy(*, master_wav: Path, output: Path, fmt: str, bitrate_kbps: int) -> None:
    audiobook_encode.encode_lossy(
        master_wav=master_wav,
        output=output,
        fmt=fmt,
        bitrate_kbps=int(bitrate_kbps),
    )


def verify_duration(path: Path, expected_s: float) -> float:
    return audiobook_encode.verify_duration(path, float(expected_s))


def decode_peak(path: Path, sr: int) -> Optional[dict[str, Optional[float]]]:
    return audiobook_encode.decode_peak(path, int(sr))


def mux(video: Path, master_wav: Path, output: Path) -> None:
    executable = require_ffmpeg("m4a")  # preflights ffmpeg + the AAC encoder
    command = [
        executable,
        "-v",
        "error",
        "-y",
        "-i",
        str(video),
        "-i",
        str(master_wav),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-shortest",
        str(output),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise ModelError(
            f"ffmpeg failed to mux {video}: {completed.stderr.strip()}",
            details={"video": str(video)},
        )
    if not Path(output).is_file() or Path(output).stat().st_size == 0:
        raise ModelError(f"ffmpeg produced no muxed output for {video}")


def encoder_name(fmt: str) -> str:
    return "soundfile" if fmt in ("wav", "flac") else audiobook_encode._FFMPEG_ENCODER[fmt]


def uuid_path(target: Path, suffix: str = ".tmp") -> Path:
    return audiobook_encode.uuid_path(Path(target), suffix)


def remove_quietly(path: Path) -> bool:
    return audiobook_encode.remove_quietly(Path(path))
