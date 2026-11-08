"""Dataset audio validation, raw JSONL writing and prepared JSONL checks."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from ..audio_io import audio_info as _audio_info
from ..errors import InputError, ModelError, QttsError
from ..utils import canonical_json, sha256_file, sha256_text

MAX_PREPARED_LINES = 100_000
MAX_CODE_ROWS = 10_000
CODE_COLUMNS = 16
MAX_CODE = 2**31 - 1


def audio_info(path: str | Path) -> dict[str, Any]:
    """Test seam: soundfile metadata without decoding."""
    return _audio_info(path)


def _duration(info: dict[str, Any]) -> float:
    frames = info.get("frames")
    rate = info.get("sample_rate")
    try:
        frames_i = int(frames)
        rate_i = int(rate)
    except (TypeError, ValueError):
        return 0.0
    return float(frames_i) / float(rate_i) if rate_i > 0 else 0.0


def _audio_error(path: Path, check: str, message: str) -> InputError:
    return InputError(
        f"dataset audio {path} {message}",
        details={"path": str(path), "check": check},
    )


def validate_audio(
    path: Path,
    *,
    minimum_s: float,
    maximum_s: float,
) -> dict[str, Any]:
    try:
        info = audio_info(path)
    except QttsError:
        raise
    except Exception as exc:  # defensive: a seam may raise a plain error
        raise InputError(
            f"cannot inspect audio file {path}: {exc}", details={"path": str(path)}
        ) from exc
    container = str(info.get("format") or "").upper()
    if container not in ("WAV", "FLAC"):
        raise _audio_error(path, "container", f"must be WAV or FLAC, got {container!r}")
    if int(info.get("sample_rate") or 0) != 24_000:
        raise _audio_error(
            path, "sample_rate", f"must be 24000 Hz, got {info.get('sample_rate')}"
        )
    if int(info.get("channels") or 0) != 1:
        raise _audio_error(path, "channels", f"must be mono, got {info.get('channels')} channel(s)")
    frames = int(info.get("frames") or 0)
    if frames <= 0:
        raise _audio_error(path, "frames", "must contain at least one frame")
    duration = _duration(info)
    if duration < minimum_s or duration > maximum_s:
        raise _audio_error(
            path,
            "duration",
            f"duration {duration:.4f}s must be within {minimum_s:g}..{maximum_s:g}s",
        )
    return info


def validate_dataset_audio(dataset) -> dict[str, Any]:
    """Validate every training item and the reference clip (§3.1)."""
    total = 0.0
    for item in dataset.items:
        info = validate_audio(item.audio, minimum_s=0.5, maximum_s=30.0)
        total += _duration(info)
    validate_audio(dataset.ref_audio, minimum_s=2.0, maximum_s=30.0)
    return {"items": len(dataset.items), "duration_s": round(total, 3)}


def validate_ref_audio(ref_audio: Path) -> dict[str, Any]:
    return validate_audio(ref_audio, minimum_s=2.0, maximum_s=30.0)


def raw_records(dataset) -> list[dict[str, str]]:
    return [
        {
            "audio": str(item.audio),
            "text": item.text,
            "ref_audio": str(dataset.ref_audio),
        }
        for item in dataset.items
    ]


def raw_text(dataset) -> str:
    lines = [
        json.dumps(record, ensure_ascii=False) for record in raw_records(dataset)
    ]
    return "\n".join(lines) + "\n"


def write_raw_jsonl(path: Path, dataset) -> str:
    """Write `<path>.tmp` then atomically replace `<path>`; return its sha256."""
    payload = raw_text(dataset).encode("utf-8")
    tmp = Path(f"{path}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except OSError as exc:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise ModelError(
            f"cannot write raw JSONL {path}: {exc}", details={"path": str(path)}
        ) from exc
    return sha256_file(path)


def items_sha256(dataset) -> str:
    return sha256_text(
        canonical_json([[str(item.audio), item.text] for item in dataset.items])
    )


def _read_sha_or_none(path: Optional[Path]) -> Optional[str]:
    if path is None:
        return None
    try:
        if not path.is_file():
            return None
        return sha256_file(path)
    except OSError:
        return None


def raw_prepared_hashes(
    raw_jsonl: Optional[Path], output_jsonl: Optional[Path]
) -> tuple[Optional[str], Optional[str]]:
    return _read_sha_or_none(raw_jsonl), _read_sha_or_none(output_jsonl)


def _json_error(path: Path, line: int, message: str, cls) -> Exception:
    return cls(
        f"invalid prepared JSONL {path} line {line}: {message}",
        details={"path": str(path), "line": line},
    )


def _validate_codes(codes: Any, path: Path, line: int, cls) -> None:
    if not isinstance(codes, list) or not codes:
        raise _json_error(path, line, "audio_codes must be a non-empty list", cls)
    if len(codes) > MAX_CODE_ROWS:
        raise _json_error(
            path, line, f"audio_codes has more than {MAX_CODE_ROWS} rows", cls
        )
    for row_index, row in enumerate(codes):
        if not isinstance(row, list) or len(row) != CODE_COLUMNS:
            raise _json_error(
                path,
                line,
                f"audio_codes[{row_index}] must be a list of {CODE_COLUMNS} integers",
                cls,
            )
        for value in row:
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                or value > MAX_CODE
            ):
                raise _json_error(
                    path,
                    line,
                    f"audio_codes[{row_index}] contains an integer outside 0..{MAX_CODE}",
                    cls,
                )


def validate_prepared_jsonl(
    path: Path,
    expected: Optional[Sequence[dict[str, str]]] = None,
    *,
    error_cls=ModelError,
) -> int:
    """Stream one prepared JSONL file; return its line count.

    ``expected`` (our own raw records) is compared line by line.  External
    train files pass ``expected=None`` and are shape-checked only.
    """
    if not path.is_file():
        raise error_cls(
            f"prepared JSONL not found: {path}", details={"path": str(path)}
        )
    count = 0
    try:
        with open(path, "r", encoding="utf-8", newline="") as handle:
            last_ended = True
            for count, line in enumerate(handle, 1):
                if count > MAX_PREPARED_LINES:
                    raise _json_error(
                        path,
                        count,
                        f"more than {MAX_PREPARED_LINES} lines",
                        error_cls,
                    )
                last_ended = line.endswith("\n")
                if not line.strip():
                    raise _json_error(path, count, "blank line", error_cls)
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise _json_error(path, count, f"invalid JSON: {exc}", error_cls) from exc
                if not isinstance(record, dict):
                    raise _json_error(path, count, "line must be a JSON object", error_cls)
                for key in ("audio", "text", "ref_audio"):
                    value = record.get(key)
                    if not isinstance(value, str):
                        raise _json_error(
                            path, count, f"{key} must be a string", error_cls
                        )
                if "audio_codes" not in record:
                    raise _json_error(path, count, "audio_codes is required", error_cls)
                _validate_codes(record["audio_codes"], path, count, error_cls)
                if expected is not None:
                    index = count - 1
                    if index >= len(expected):
                        raise _json_error(
                            path, count, "more lines than the raw JSONL", error_cls
                        )
                    for key in ("audio", "text", "ref_audio"):
                        if record[key] != expected[index][key]:
                            raise _json_error(
                                path,
                                count,
                                f"{key} does not match the raw JSONL",
                                error_cls,
                            )
            if not last_ended:
                raise _json_error(
                    path, max(1, count), "file must end with a newline", error_cls
                )
    except UnicodeDecodeError as exc:
        raise error_cls(
            f"prepared JSONL {path} is not valid UTF-8: {exc}",
            details={"path": str(path)},
        ) from exc
    except OSError as exc:
        raise error_cls(
            f"cannot read prepared JSONL {path}: {exc}", details={"path": str(path)}
        ) from exc
    if count == 0:
        raise _json_error(path, 1, "file is empty", error_cls)
    if expected is not None and count != len(expected):
        raise _json_error(
            path,
            count,
            f"line count {count} does not match the raw JSONL ({len(expected)})",
            error_cls,
        )
    return count


def prepared_jsonl_nonempty(path: Path) -> None:
    if not path.is_file():
        raise ModelError(
            f"prepared JSONL not found: {path}", details={"path": str(path)}
        )
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise ModelError(
            f"cannot stat prepared JSONL {path}: {exc}", details={"path": str(path)}
        ) from exc
    if size <= 0:
        raise ModelError(
            f"prepared JSONL is empty: {path}", details={"path": str(path)}
        )


def finite_duration(seconds: Any) -> float:
    try:
        value = float(seconds)
    except (TypeError, ValueError):
        return 0.0
    return value if math.isfinite(value) else 0.0


def text_sha256(text: str) -> str:
    return sha256_text(text)


def iter_audio_paths(dataset) -> Iterable[Path]:
    yield dataset.ref_audio
    for item in dataset.items:
        yield item.audio
