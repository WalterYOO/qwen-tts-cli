"""`qtts post` — deterministic audio post-processing (FR-P2-05).

The command layer owns argument/path preflight, decoding, per-file dispatch,
encoding verification, report serialization and human formatting. All DSP
lives in :mod:`qtts_cli.postprocess`.
"""

from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
import tempfile
import unicodedata
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..audio_io import read_audio
from ..errors import (
    EXIT_BATCH,
    InputError,
    ModelError,
    QttsError,
    UsageError,
)
from ..postprocess import (
    LIMITER_ALGORITHM,
    POST_DSP_VERSION,
    TRUE_PEAK_TOLERANCE_DB,
    apply_gain,
    measure,
    process,
)
from ..quality import write_report
from ..takes import collision_key
from ..utils import atomic_write_bytes, sha256_bytes, sha256_file

FORMATS = ("wav", "flac", "mp3", "m4a")
LOSSLESS_FORMATS = ("wav", "flac")
LOSSY_FORMATS = ("mp3", "m4a")
_BITRATE_RE = re.compile(r"^\d+(?:\.\d+)?[kKmM]?$")
_SUBTYPES = {16: "PCM_16", 24: "PCM_24"}
_CODECS = {"mp3": "libmp3lame", "m4a": "aac"}
_ENCODERS = {"mp3": "libmp3lame", "m4a": "aac"}

_DEFAULT_TRIM_THRESHOLD_DBFS = -50.0
_DEFAULT_TRIM_MIN_SILENCE_S = 0.2
_DEFAULT_TRIM_PADDING_MS = 50
_DEFAULT_TRIM_MODE = "both"


def _sf():
    import soundfile

    return soundfile


def _input_suffix(path: Path) -> Optional[str]:
    suffix = path.suffix.lower().lstrip(".")
    return suffix or None


def _input_format(path: Path) -> Optional[str]:
    suffix = _input_suffix(path)
    return suffix if suffix in FORMATS else None


def _input_stem(path: Path) -> str:
    name = path.name
    lowered = name.lower()
    for suffix in (".wav", ".flac"):
        if lowered.endswith(suffix):
            return name[: -len(suffix)]
    return ""


def _collision_key(path: str | os.PathLike[str]) -> str:
    """Reuse the takes helper, then apply case folding on every platform."""
    return collision_key(path).casefold()


def _basename_key(name: str) -> str:
    return unicodedata.normalize("NFC", os.path.normcase(name)).casefold()


def _validate_inputs(args) -> list[Path]:
    sources: list[Path] = []
    for raw in [str(value) for value in args.inputs]:
        source = Path(raw).expanduser()
        fmt = _input_format(source)
        if fmt not in LOSSLESS_FORMATS:
            raise UsageError(
                f"input must be .wav or .flac: {raw!r}",
                details={"path": raw, "supported": list(LOSSLESS_FORMATS)},
            )
        if source.exists() and source.is_dir():
            raise UsageError(f"input is a directory: {raw!r}", details={"path": raw})
        if not _input_stem(source):
            raise UsageError(
                f"input has an empty stem: {raw!r}", details={"path": raw}
            )
        sources.append(source)
    return sources


def _target_for(source: Path, args) -> tuple[Path, str]:
    if args.out is not None:
        if str(args.out) == "":
            raise UsageError("--out must not be empty")
        target = Path(args.out).expanduser()
        suffix = target.suffix.lower().lstrip(".")
        if suffix:
            if suffix not in FORMATS:
                raise UsageError(
                    f"unsupported output suffix '.{suffix}' for --out {str(target)!r}",
                    details={"path": str(target), "supported": list(FORMATS)},
                )
            if args.format and args.format != suffix:
                raise UsageError(
                    f"--format {args.format} does not match --out suffix "
                    f"'.{suffix}' for {str(target)!r}"
                )
            return target, suffix
        if not args.format:
            raise UsageError(
                f"--out {str(target)!r} has no suffix; pass --format"
            )
        return Path(str(target) + "." + args.format), args.format

    if args.out_dir is None or str(args.out_dir) == "":
        raise UsageError("--out-dir must not be empty")
    if not args.format:
        raise UsageError("--out-dir requires an explicit --format")
    stem = _input_stem(source)
    if not stem:
        raise UsageError(f"input has an empty stem: {str(source)!r}")
    return Path(args.out_dir).expanduser() / f"{stem}.{args.format}", args.format


def _preflight_paths(
    sources: list[Path], targets: list[Path], report: Optional[str]
) -> None:
    seen: dict[str, tuple[str, str]] = {}
    entries: list[tuple[str, str]] = []
    entries.extend(("input", str(path)) for path in sources)
    entries.extend(("output", str(path)) for path in targets)
    if report is not None:
        entries.append(("report", str(Path(report).expanduser())))
    for kind, path in entries:
        key = _collision_key(path)
        if key in seen:
            other_kind, other_path = seen[key]
            raise UsageError(
                f"path collision: {kind} {path!r} is the same as "
                f"{other_kind} {other_path!r}",
                details={"path": path, "collides_with": other_path},
            )
        seen[key] = (kind, path)

    basenames: dict[str, str] = {}
    for target in targets:
        key = _basename_key(target.name)
        if key in basenames:
            raise UsageError(
                f"output stem conflict: {str(target)!r} and "
                f"{basenames[key]!r} have the same normalized name",
                details={"path": str(target), "conflicts_with": basenames[key]},
            )
        basenames[key] = str(target)

    for target in targets:
        if target.exists() and target.is_dir():
            raise UsageError(
                f"output path is a directory: {str(target)!r}",
                details={"path": str(target)},
            )


def _validate_target_parents(targets: list[Path]) -> None:
    """Require every existing ancestor of a target to be a directory."""
    for target in targets:
        ancestor = target.parent
        while True:
            if ancestor.is_symlink() or ancestor.exists():
                if not ancestor.is_dir():
                    raise UsageError(
                        f"output parent is not a directory: {str(ancestor)!r}",
                        details={"path": str(ancestor)},
                    )
                break
            if ancestor.parent == ancestor:
                break
            ancestor = ancestor.parent


def _option_values(args) -> dict[str, Any]:
    lufs = args.lufs
    true_peak = args.true_peak
    sample_rate = args.sample_rate
    for label, value in (
        ("--lufs", lufs),
        ("--true-peak", true_peak),
        ("--trim-threshold-dbfs", getattr(args, "trim_threshold_dbfs", None)),
        ("--trim-min-silence-s", getattr(args, "trim_min_silence_s", None)),
    ):
        if value is not None and not math.isfinite(float(value)):
            raise UsageError(f"{label} must be finite")
    if lufs is not None and not (-70.0 <= float(lufs) <= 0.0):
        raise UsageError("--lufs must be within [-70, 0]")
    if true_peak is not None and float(true_peak) > 0.0:
        raise UsageError("--true-peak must be <= 0")
    if sample_rate is not None and not (8000 <= int(sample_rate) <= 192000):
        raise UsageError("--sample-rate must be within [8000, 192000]")

    trim = bool(args.trim_silence)
    trim_explicit = any(
        hasattr(args, name)
        for name in (
            "trim_threshold_dbfs",
            "trim_min_silence_s",
            "trim_padding_ms",
            "trim_mode",
        )
    )
    if trim_explicit and not trim:
        raise UsageError(
            "--trim-threshold-dbfs/--trim-min-silence-s/--trim-padding-ms/"
            "--trim-mode require --trim-silence"
        )
    threshold = float(
        getattr(args, "trim_threshold_dbfs", _DEFAULT_TRIM_THRESHOLD_DBFS)
    )
    min_silence = float(
        getattr(args, "trim_min_silence_s", _DEFAULT_TRIM_MIN_SILENCE_S)
    )
    padding = int(getattr(args, "trim_padding_ms", _DEFAULT_TRIM_PADDING_MS))
    mode = str(getattr(args, "trim_mode", _DEFAULT_TRIM_MODE))
    if threshold > 0.0:
        raise UsageError("--trim-threshold-dbfs must be <= 0")
    if min_silence < 0.0:
        raise UsageError("--trim-min-silence-s must be >= 0")
    if padding < 0:
        raise UsageError("--trim-padding-ms must be >= 0")

    return {
        "lufs": None if lufs is None else float(lufs),
        "true_peak": None if true_peak is None else float(true_peak),
        "sample_rate": None if sample_rate is None else int(sample_rate),
        "channels": str(args.channels),
        "trim": trim,
        "trim_threshold_dbfs": threshold,
        "trim_min_silence_s": min_silence,
        "trim_padding_ms": padding,
        "trim_mode": mode,
    }


def _validate_format_options(args, formats: set[str]) -> tuple[int, str]:
    lossy = bool(formats & set(LOSSY_FORMATS))
    lossless = bool(formats & set(LOSSLESS_FORMATS))

    bit_depth_explicit = hasattr(args, "bit_depth")
    bit_depth = int(getattr(args, "bit_depth", 24))
    if bit_depth_explicit and lossy:
        raise UsageError("--bit-depth is only valid for wav/flac output")
    if bit_depth not in _SUBTYPES:
        raise UsageError("--bit-depth must be 16 or 24")

    bitrate_explicit = hasattr(args, "bitrate")
    bitrate = str(getattr(args, "bitrate", "192k"))
    if bitrate_explicit and lossless:
        raise UsageError("--bitrate is only valid for mp3/m4a output")
    if lossy and not _BITRATE_RE.fullmatch(bitrate.strip()):
        raise UsageError(
            f"invalid --bitrate {bitrate!r}; expected e.g. 192k",
            details={"bitrate": bitrate},
        )
    return bit_depth, bitrate


def _preflight_ffmpeg(formats: set[str]) -> Optional[str]:
    needed = sorted(formats & set(LOSSY_FORMATS))
    if not needed:
        return None
    executable = shutil.which("ffmpeg")
    if executable is None:
        raise ModelError(
            "ffmpeg is required for mp3/m4a output but was not found on PATH"
        )
    try:
        completed = subprocess.run(
            [executable, "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise ModelError(f"cannot execute ffmpeg: {exc}") from exc
    if completed.returncode != 0:
        raise ModelError(
            "ffmpeg -encoders failed",
            details={"returncode": completed.returncode},
        )
    reported = f"{completed.stdout}\n{completed.stderr}".lower()
    for fmt in needed:
        if _ENCODERS[fmt] not in reported:
            raise ModelError(
                f"ffmpeg does not provide the {_ENCODERS[fmt]} encoder "
                f"required for {fmt}"
            )
    return executable


def _codec(fmt: str, bit_depth: int) -> str:
    if fmt == "wav":
        return "pcm_s24le" if bit_depth == 24 else "pcm_s16le"
    if fmt == "flac":
        return "flac"
    return _CODECS[fmt]


def _entry_shape(
    raw: str,
    target: Path,
    input_format: str,
    output_format: str,
    bit_depth: int,
) -> dict[str, Any]:
    return {
        "input": raw,
        "output": str(target),
        "ok": False,
        "error": None,
        "format": output_format,
        "input_format": input_format,
        "output_format": output_format,
        "codec": _codec(output_format, bit_depth),
        "encoder": "soundfile" if output_format in LOSSLESS_FORMATS else "ffmpeg",
        "dsp_version": POST_DSP_VERSION,
        "sample_rate": None,
        "channels": None,
        "input_duration_s": None,
        "output_duration_s": None,
        "input_sha256": None,
        "output_sha256": None,
        "before": None,
        "after_dsp": None,
        "after_encoded": None,
        "applied": None,
        "warnings": [],
    }


def _failure_entry(
    raw: str,
    target: Path,
    input_format: str,
    output_format: str,
    bit_depth: int,
    error: QttsError,
) -> dict[str, Any]:
    entry = _entry_shape(raw, target, input_format, output_format, bit_depth)
    entry["error"] = {
        "code": error.code,
        "kind": error.kind,
        "message": error.message,
    }
    return entry


def _run_ffmpeg(executable: str, command: list[str], *, action: str, target: Path) -> None:
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError as exc:
        raise ModelError(f"cannot execute ffmpeg while {action}: {exc}") from exc
    if completed.returncode != 0:
        lines = (completed.stderr or "").strip().splitlines()
        raise ModelError(
            f"ffmpeg failed while {action} for {str(target)!r}",
            details={
                "returncode": completed.returncode,
                "stderr": lines[-1] if lines else "",
            },
        )


def _encode_once(
    samples: np.ndarray,
    sr: int,
    fmt: str,
    *,
    bit_depth: int,
    bitrate: str,
    ffmpeg: Optional[str],
    target: Path,
) -> tuple[bytes, np.ndarray, int]:
    """Encode into the target directory and decode the result for measurement."""
    parent = target.parent
    try:
        parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise InputError(
            f"cannot create output directory {str(parent)!r}: {exc}",
            details={"path": str(parent)},
        ) from exc

    with tempfile.TemporaryDirectory(
        prefix=f".{target.name}.", dir=str(parent)
    ) as work:
        temp = Path(work) / target.name
        source = Path(work) / "source.wav"
        decoded = Path(work) / "decoded.wav"
        try:
            if fmt in LOSSLESS_FORMATS:
                _sf().write(
                    temp,
                    samples,
                    int(sr),
                    format="WAV" if fmt == "wav" else "FLAC",
                    subtype=_SUBTYPES[bit_depth],
                )
                payload = temp.read_bytes()
                decoded_samples, decoded_sr = read_audio(temp, mono=False)
                return payload, decoded_samples, decoded_sr

            assert ffmpeg is not None
            _sf().write(source, samples, int(sr), format="WAV", subtype="PCM_24")
            _run_ffmpeg(
                ffmpeg,
                [
                    ffmpeg,
                    "-nostdin",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    str(source),
                    "-c:a",
                    _CODECS[fmt],
                    "-b:a",
                    str(bitrate),
                    str(temp),
                ],
                action="encoding",
                target=target,
            )
            _run_ffmpeg(
                ffmpeg,
                [
                    ffmpeg,
                    "-nostdin",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    str(temp),
                    "-map",
                    "0:a:0",
                    "-c:a",
                    "pcm_s24le",
                    str(decoded),
                ],
                action="decoding for verification",
                target=target,
            )
            payload = temp.read_bytes()
            decoded_samples, decoded_sr = read_audio(decoded, mono=False)
            return payload, decoded_samples, decoded_sr
        except QttsError:
            raise
        except OSError as exc:
            raise InputError(
                f"cannot encode output {str(target)!r}: {exc}",
                details={"path": str(target)},
            ) from exc
        except Exception as exc:
            raise InputError(
                f"cannot encode output {str(target)!r}: {exc}",
                details={"path": str(target)},
            ) from exc


def _encode_verified(
    samples: np.ndarray,
    sr: int,
    fmt: str,
    *,
    bit_depth: int,
    bitrate: str,
    ffmpeg: Optional[str],
    target: Path,
    ceiling: Optional[float],
) -> tuple[bytes, dict[str, Optional[float]], float, int, float]:
    """Encode, decode and retry with static attenuation until the scope passes."""
    margin = 0.0
    retries = 0
    while True:
        candidate = samples if margin == 0.0 else apply_gain(samples, -margin)
        payload, decoded, decoded_sr = _encode_once(
            candidate,
            sr,
            fmt,
            bit_depth=bit_depth,
            bitrate=bitrate,
            ffmpeg=ffmpeg,
            target=target,
        )
        after_encoded = measure(decoded, decoded_sr)
        duration = float(decoded.shape[0]) / float(decoded_sr)
        if ceiling is None:
            return payload, after_encoded, duration, retries, margin
        peak = after_encoded.get("true_peak_dbtp")
        if peak is None:
            raise InputError(
                "true peak is unavailable after encoding; cannot verify the ceiling",
                details={"path": str(target)},
            )
        if peak <= float(ceiling) + TRUE_PEAK_TOLERANCE_DB:
            return payload, after_encoded, duration, retries, margin
        if retries >= 2:
            raise InputError(
                f"encoded true peak {peak} dBTP exceeds ceiling "
                f"{float(ceiling)} dBTP after {retries} retries",
                details={"path": str(target), "true_peak_dbtp": peak},
            )
        margin += max(0.0, float(peak) - float(ceiling)) + 0.2
        retries += 1


def _warnings(
    lufs_target: Optional[float], measured: Optional[dict[str, Optional[float]]]
) -> list[str]:
    if lufs_target is None:
        return []
    value = None if measured is None else measured.get("lufs_integrated")
    if value is None or abs(float(value) - float(lufs_target)) > 0.1:
        return ["loudness_target_missed"]
    return []


def _process_one(
    raw: str,
    source: Path,
    target: Path,
    fmt: str,
    args,
    values: dict[str, Any],
    bit_depth: int,
    bitrate: str,
    ffmpeg: Optional[str],
) -> dict[str, Any]:
    input_format = _input_format(source) or ""
    entry = _entry_shape(raw, target, input_format, fmt, bit_depth)

    samples, input_sr = read_audio(source, mono=False)
    if samples.shape[1] > 2:
        raise InputError(
            f"audio has {samples.shape[1]} channels; post supports at most 2",
            details={"path": raw, "channels": int(samples.shape[1])},
        )
    entry["input_sha256"] = sha256_file(source)
    entry["input_duration_s"] = round(
        float(samples.shape[0]) / float(input_sr), 6
    )

    output, output_sr, applied, before, after_dsp, warnings = process(
        samples, input_sr, values
    )
    predicted_duration = float(output.shape[0]) / float(output_sr)
    entry["before"] = before
    entry["after_dsp"] = after_dsp
    entry["sample_rate"] = int(output_sr)
    entry["channels"] = int(output.shape[1])
    entry["output_duration_s"] = round(predicted_duration, 6)
    ceilings = applied.get("true_peak_ceiling_dbtp")

    applied["encode_retries"] = 0
    applied["encoder_margin_db"] = 0.0
    applied.setdefault("safety_attenuation_db", 0.0)
    applied["limiter_algorithm"] = LIMITER_ALGORITHM

    if args.dry_run:
        entry["applied"] = applied
        entry["warnings"] = list(warnings) + _warnings(
            values.get("lufs"), after_dsp
        )
        entry["ok"] = True
        return entry

    payload, after_encoded, duration, retries, margin = _encode_verified(
        output,
        output_sr,
        fmt,
        bit_depth=bit_depth,
        bitrate=bitrate,
        ffmpeg=ffmpeg,
        target=target,
        ceiling=None if ceilings is None else float(ceilings),
    )
    output_sha256 = sha256_bytes(payload)
    try:
        atomic_write_bytes(target, payload)
    except OSError as exc:
        raise InputError(
            f"cannot write output {str(target)!r}: {exc}",
            details={"path": str(target)},
        ) from exc

    applied["encode_retries"] = retries
    applied["encoder_margin_db"] = round(float(margin), 3)
    entry["applied"] = applied
    entry["after_encoded"] = after_encoded
    entry["output_duration_s"] = round(float(duration), 6)
    entry["output_sha256"] = output_sha256
    entry["warnings"] = list(warnings) + _warnings(
        values.get("lufs"), after_encoded
    )
    entry["ok"] = True
    return entry


def _request(
    args, values: dict[str, Any], formats: set[str], bit_depth: int, bitrate: str
) -> dict[str, Any]:
    lossy = bool(formats & set(LOSSY_FORMATS))
    lossless = bool(formats & set(LOSSLESS_FORMATS))
    effective_format = args.format
    if effective_format is None and len(formats) == 1:
        effective_format = next(iter(formats))
    return {
        "lufs": values["lufs"],
        "true_peak": values["true_peak"],
        "true_peak_ceiling_scope": "encoded",
        "sample_rate": values["sample_rate"],
        "channels": values["channels"],
        "format": effective_format,
        "bitrate": bitrate if lossy else None,
        "bit_depth": bit_depth if lossless else None,
        "trim_silence": values["trim"],
        "trim_threshold_dbfs": values["trim_threshold_dbfs"],
        "trim_min_silence_s": values["trim_min_silence_s"],
        "trim_padding_ms": values["trim_padding_ms"],
        "trim_mode": values["trim_mode"],
        "output_dir": (
            str(Path(args.out_dir).expanduser())
            if args.out_dir is not None
            else None
        ),
    }


def run(cfg, args) -> dict[str, Any]:
    sources = _validate_inputs(args)
    has_out = args.out is not None
    has_dir = args.out_dir is not None
    if has_out == has_dir:
        raise UsageError("provide exactly one of --out or --out-dir")
    if has_out and len(sources) != 1:
        raise UsageError("--out accepts exactly one input; use --out-dir for many")
    if args.out is not None and str(args.out) == "-":
        raise UsageError("--out - is not supported by post")
    if args.out_dir is not None and str(args.out_dir) == "-":
        raise UsageError("--out-dir - is not supported by post")
    if args.report is not None:
        if str(args.report) == "-":
            raise UsageError("--report - is not supported by post")
        if args.dry_run:
            raise UsageError("--dry-run and --report are mutually exclusive")

    resolved = [(source, *_target_for(source, args)) for source in sources]
    targets = [item[1] for item in resolved]
    formats = {item[2] for item in resolved}
    _preflight_paths(sources, targets, args.report)
    _validate_target_parents(targets)

    values = _option_values(args)
    bit_depth, bitrate = _validate_format_options(args, formats)
    ffmpeg = _preflight_ffmpeg(formats)

    files: list[dict[str, Any]] = []
    codes: list[int] = []
    for raw, source, target, fmt in (
        (str(item[0]), item[0], item[1], item[2]) for item in resolved
    ):
        try:
            files.append(
                _process_one(
                    raw,
                    source,
                    target,
                    fmt,
                    args,
                    values,
                    bit_depth,
                    bitrate,
                    ffmpeg,
                )
            )
            codes.append(0)
        except QttsError as exc:
            files.append(
                _failure_entry(
                    raw,
                    target,
                    _input_format(source) or "",
                    fmt,
                    bit_depth,
                    exc,
                )
            )
            codes.append(exc.code)
        except OSError as exc:
            error = InputError(
                f"cannot process {raw!r}: {exc}", details={"path": raw}
            )
            files.append(
                _failure_entry(
                    raw, target, _input_format(source) or "", fmt, bit_depth, error
                )
            )
            codes.append(error.code)
        except Exception as exc:  # keep per-file reporting deterministic
            error = InputError(
                f"post-processing failed for {raw!r}: {exc}", details={"path": raw}
            )
            files.append(
                _failure_entry(
                    raw, target, _input_format(source) or "", fmt, bit_depth, error
                )
            )
            codes.append(error.code)

    ok = all(entry["ok"] for entry in files)
    if ok:
        exit_code = 0
    elif all(not entry["ok"] for entry in files):
        exit_code = codes[0]
    else:
        exit_code = EXIT_BATCH

    payload: dict[str, Any] = {
        "schema_version": 1,
        "ok": ok,
        "dry_run": bool(args.dry_run),
        "request": _request(args, values, formats, bit_depth, bitrate),
        "files": files,
    }
    if args.report is not None:
        try:
            write_report(args.report, payload)
        except QttsError as exc:
            payload["ok"] = False
            payload["error"] = {
                "code": exc.code,
                "kind": exc.kind,
                "message": exc.message,
            }
            exit_code = exc.code
    payload["_canonical"] = True
    payload["exit_code"] = exit_code
    return payload


def format_result(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    if payload.get("dry_run"):
        lines.append("post dry-run (no files written)")
    for entry in payload.get("files") or []:
        if entry.get("ok"):
            measured = entry.get("after_encoded") or entry.get("after_dsp") or {}
            lufs = measured.get("lufs_integrated")
            peak = measured.get("true_peak_dbtp")
            suffix = ""
            if lufs is not None:
                suffix += f", {lufs} LUFS"
            if peak is not None:
                suffix += f", {peak} dBTP"
            lines.append(
                f"{entry['input']} -> {entry['output']} "
                f"({entry['output_format']}, {entry['sample_rate']}Hz, "
                f"{entry['channels']}ch, {entry['output_duration_s']}s{suffix})"
            )
        else:
            error = entry.get("error") or {}
            lines.append(
                f"{entry['input']} -> {entry['output']}: "
                f"error {error.get('code')}: {error.get('message')}"
            )
    if payload.get("error"):
        error = payload["error"]
        lines.append(f"report error {error.get('code')}: {error.get('message')}")
    return "\n".join(lines)
