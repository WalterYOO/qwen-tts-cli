"""Unit and CLI tests for FR-P2-05 `qtts post` (no model runtime)."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from qtts_cli.audio_qc import integrated_loudness, true_peak_dbtp
from qtts_cli.cli import main
from qtts_cli.commands import post as post_cmd
from qtts_cli.errors import InputError
from qtts_cli.postprocess import (
    LIMITER_ALGORITHM,
    POST_DSP_VERSION,
    convert_channels,
    limit_true_peak,
    process,
    trim_silence,
)

SR = 24000
FFMPEG = shutil.which("ffmpeg")

ENTRY_KEYS = {
    "input",
    "output",
    "ok",
    "error",
    "format",
    "input_format",
    "output_format",
    "codec",
    "encoder",
    "dsp_version",
    "sample_rate",
    "channels",
    "input_duration_s",
    "output_duration_s",
    "input_sha256",
    "output_sha256",
    "before",
    "after_dsp",
    "after_encoded",
    "applied",
    "warnings",
}


def _tone(
    duration: float = 1.0,
    *,
    sr: int = SR,
    amplitude: float = 0.1,
    frequency: float = 440.0,
) -> np.ndarray:
    t = np.arange(int(round(duration * sr)), dtype=np.float64) / float(sr)
    return (amplitude * np.sin(2.0 * math.pi * frequency * t)).astype(np.float32)


def _write(
    path: Path,
    samples: np.ndarray,
    sr: int = SR,
    *,
    subtype: str = "PCM_24",
) -> Path:
    data = np.asarray(samples, dtype=np.float32)
    if data.ndim == 1:
        data = data[:, None]
    fmt = "FLAC" if path.suffix.lower() == ".flac" else "WAV"
    sf.write(str(path), data, sr, format=fmt, subtype=subtype)
    return path


def _read(path: Path) -> tuple[np.ndarray, int]:
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return data, int(sr)


def _run(capsys, *args: str) -> tuple[int, str, str]:
    code = main(list(args))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def _json(out: str) -> dict:
    return json.loads(out)


def _ffmpeg_decode(path: Path, destination: Path) -> tuple[np.ndarray, int]:
    completed = subprocess.run(
        [
            FFMPEG,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(path),
            "-map",
            "0:a:0",
            "-c:a",
            "pcm_s24le",
            str(destination),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return _read(destination)


# --------------------------------------------------------------------------- #
# 1. Silence trimming
# --------------------------------------------------------------------------- #
def test_trim_silence_edges_padding_internal_pause_and_stereo():
    silence = np.zeros(int(0.4 * SR), dtype=np.float32)
    speech = _tone(0.6).astype(np.float32)
    signal = np.concatenate([silence, speech, np.zeros(int(0.3 * SR), np.float32)])
    trimmed, info = trim_silence(signal, SR, -50.0, 0.2, 50, "both")
    assert info["start_s"] == pytest.approx(0.35, abs=0.005)
    assert info["end_s"] == pytest.approx(0.25, abs=0.005)
    assert 0.0 < info["removed_s"] < 0.7

    internal = np.concatenate([speech, np.zeros(int(0.3 * SR), np.float32), speech])
    kept, info = trim_silence(internal, SR, -50.0, 0.2, 50, "both")
    assert kept.shape[0] == internal.shape[0]
    assert info["removed_s"] == 0.0

    # One active channel prevents a "silent" frame from being trimmed.
    left = np.concatenate([np.zeros(int(0.4 * SR), np.float32), speech])
    right = _tone(1.0)
    stereo = np.stack([left, right], axis=1)
    asymmetrical, _ = trim_silence(stereo, SR, -50.0, 0.2, 0, "both")
    assert asymmetrical.shape[0] == stereo.shape[0]

    tail, info = trim_silence(
        np.concatenate([speech, np.zeros(int(0.010 * SR), np.float32)]),
        SR,
        -50.0,
        0.005,
        0,
        "trailing",
    )
    assert info["end_s"] == pytest.approx(0.010, abs=0.001)
    assert tail.shape[0] == speech.shape[0]

    with pytest.raises(InputError) as exc:
        trim_silence(np.zeros(SR, dtype=np.float32), SR, -50.0, 0.2, 50, "both")
    assert exc.value.code == 4


@pytest.mark.parametrize("tail_ms", [5, 15])
def test_trim_tail_5_and_15ms_and_active_partial_frame(tail_ms):
    speech = _tone(0.1)
    tail_samples = int(round(tail_ms / 1000.0 * SR))
    signal = np.concatenate(
        [speech, np.zeros(tail_samples, dtype=np.float32)]
    )
    trimmed, info = trim_silence(signal, SR, -50.0, 0.001, 0, "trailing")
    assert info["end_s"] == pytest.approx(tail_ms / 1000.0, abs=1.0 / SR)
    assert trimmed.shape[0] == speech.shape[0]

    # A partial final frame that still contains speech must not be trimmed.
    active = _tone(0.105)
    kept, info = trim_silence(active, SR, -50.0, 0.001, 0, "trailing")
    assert info["removed_s"] == 0.0
    assert kept.shape[0] == active.shape[0]


def test_all_silent_trim_fails_without_output(tmp_path, capsys):
    source = _write(tmp_path / "silent.wav", np.zeros(SR, dtype=np.float32))
    output = tmp_path / "out.wav"
    code, _, _ = _run(
        capsys,
        "post",
        str(source),
        "--out",
        str(output),
        "--trim-silence",
        "--json",
    )
    assert code == 4
    assert not output.exists()


def test_digital_silence_at_minus_240_threshold_is_empty(tmp_path, capsys):
    source = _write(tmp_path / "silent.wav", np.zeros(SR, dtype=np.float32))
    output = tmp_path / "out.wav"
    code, out, err = _run(
        capsys,
        "post",
        str(source),
        "--out",
        str(output),
        "--trim-silence",
        "--trim-threshold-dbfs",
        "-240",
        "--json",
    )
    assert code == 4
    assert err == ""
    entry = _json(out)["files"][0]
    assert set(entry) == ENTRY_KEYS
    assert entry["error"]["code"] == 4
    assert not output.exists()


# --------------------------------------------------------------------------- #
# 2-5. Loudness, true peak, resampling, channels
# --------------------------------------------------------------------------- #
def test_loudness_normalization_is_within_half_lu(tmp_path, capsys):
    source = _write(tmp_path / "quiet.wav", _tone(1.0, amplitude=10 ** (-30 / 20)))
    output = tmp_path / "loud.wav"
    code, out, err = _run(
        capsys,
        "post",
        str(source),
        "--out",
        str(output),
        "--lufs",
        "-20",
        "--json",
    )
    assert code == 0, err
    payload = _json(out)
    entry = payload["files"][0]
    measured, _ = integrated_loudness(*_read(output))
    assert measured == pytest.approx(-20.0, abs=0.5)
    assert entry["after_encoded"]["lufs_integrated"] == measured
    assert payload["request"]["true_peak_ceiling_scope"] == "encoded"


def test_loudness_target_missed_warning_is_stable(tmp_path, capsys):
    source = _write(tmp_path / "in.wav", _tone(1.0, amplitude=0.1))
    output = tmp_path / "out.wav"
    code, out, err = _run(
        capsys,
        "post",
        str(source),
        "--out",
        str(output),
        "--lufs",
        "-20",
        "--true-peak",
        "-20",
        "--json",
    )
    assert code == 0, err
    entry = _json(out)["files"][0]
    assert entry["warnings"] == ["loudness_target_missed"]
    measured = entry["after_encoded"]["lufs_integrated"]
    assert abs(measured - (-20.0)) > 0.1


def test_true_peak_ceiling_and_intersample_overshoot(tmp_path, capsys):
    alternating = np.empty(SR, dtype=np.float32)
    alternating[0::2] = 0.99
    alternating[1::2] = -0.99
    source = _write(tmp_path / "hot.wav", alternating)
    output = tmp_path / "limited.wav"
    code, out, err = _run(
        capsys,
        "post",
        str(source),
        "--out",
        str(output),
        "--true-peak",
        "-3",
        "--json",
    )
    assert code == 0, err
    decoded, sr = _read(output)
    measured, reason = true_peak_dbtp(decoded, sr)
    assert reason is None
    assert measured <= -2.9
    # No sample clipping: the limiter attenuates instead of clipping to 1.0.
    assert float(np.max(np.abs(decoded))) < 1.0
    entry = _json(out)["files"][0]
    assert entry["after_encoded"]["true_peak_dbtp"] == measured
    assert entry["applied"]["limiter_algorithm"] == LIMITER_ALGORITHM
    assert entry["dsp_version"] == POST_DSP_VERSION


def test_resample_24k_to_48k_preserves_duration(tmp_path, capsys):
    source = _write(tmp_path / "in.wav", _tone(1.0, sr=24000))
    output = tmp_path / "out.wav"
    code, out, err = _run(
        capsys,
        "post",
        str(source),
        "--out",
        str(output),
        "--sample-rate",
        "48000",
        "--json",
    )
    assert code == 0, err
    decoded, sr = _read(output)
    assert sr == 48000
    assert abs(decoded.shape[0] / sr - 1.0) < 1.0 / 48000
    assert _json(out)["files"][0]["applied"]["resample"] == {
        "from": 24000,
        "to": 48000,
    }


def test_channel_conversion_mono_stereo_and_mean():
    mono = np.array([[1.0], [2.0], [-1.0]], dtype=np.float32)
    stereo, info = convert_channels(mono, "stereo")
    assert info == {"from": 1, "to": 2}
    assert np.allclose(stereo[:, 0], stereo[:, 1])
    assert np.allclose(stereo[:, 0], mono[:, 0])

    source = np.array([[1.0, 3.0], [-2.0, 4.0]], dtype=np.float32)
    downmixed, info = convert_channels(source, "mono")
    assert info == {"from": 2, "to": 1}
    assert np.allclose(downmixed[:, 0], [2.0, 1.0])


# --------------------------------------------------------------------------- #
# 6. Formats and ffmpeg preflight
# --------------------------------------------------------------------------- #
def test_wav_to_flac_and_bit_depths(tmp_path, capsys):
    source = _write(tmp_path / "in.wav", _tone(0.5))
    flac = tmp_path / "out.flac"
    code, _, err = _run(capsys, "post", str(source), "--out", str(flac))
    assert code == 0, err
    assert sf.info(str(flac)).subtype == "PCM_24"
    _read(flac)

    wav16 = tmp_path / "out16.wav"
    code, _, err = _run(
        capsys, "post", str(source), "--out", str(wav16), "--bit-depth", "16"
    )
    assert code == 0, err
    assert sf.info(str(wav16)).subtype == "PCM_16"


@pytest.mark.parametrize("value", ["nan", "inf"])
@pytest.mark.parametrize("mode", ["json", "plain", "report"])
def test_nonfinite_trim_min_silence_is_global_usage_error(
    value, mode, tmp_path, capsys
):
    source = _write(tmp_path / "in.wav", _tone(0.3))
    output = tmp_path / "existing.wav"
    output.write_bytes(b"OLD-BYTES")
    report = tmp_path / "report.json"
    args = [
        "post",
        str(source),
        "--out",
        str(output),
        "--trim-silence",
        "--trim-min-silence-s",
        value,
    ]
    if mode == "json":
        args.append("--json")
    elif mode == "report":
        args.extend(["--report", str(report)])
    code, out, err = _run(capsys, *args)
    assert code == 2
    assert out == ""
    assert err
    assert output.read_bytes() == b"OLD-BYTES"
    assert not report.exists()
    assert not any(path.suffix == ".tmp" for path in tmp_path.iterdir())


def test_ffmpeg_missing_fails_before_any_directory_or_output(tmp_path, monkeypatch, capsys):
    source = _write(tmp_path / "in.wav", _tone(0.5))
    out_dir = tmp_path / "new-output"
    real_which = shutil.which
    monkeypatch.setattr(
        post_cmd.shutil,
        "which",
        lambda name: None if name == "ffmpeg" else real_which(name),
    )
    code, _, _ = _run(
        capsys,
        "post",
        str(source),
        "--out-dir",
        str(out_dir),
        "--format",
        "mp3",
        "--json",
    )
    assert code == 3
    assert not out_dir.exists()


def test_missing_encoder_fails_global_preflight(tmp_path, monkeypatch, capsys):
    source = _write(tmp_path / "in.wav", _tone(0.5))
    out_dir = tmp_path / "out"

    class Result:
        returncode = 0
        stdout = "Encoders:\n"
        stderr = ""

    def fake_run(*_args, **_kwargs):
        return Result()

    monkeypatch.setattr(post_cmd.subprocess, "run", fake_run)
    code, _, _ = _run(
        capsys,
        "post",
        str(source),
        "--out-dir",
        str(out_dir),
        "--format",
        "m4a",
        "--json",
    )
    assert code == 3
    assert not out_dir.exists()


@pytest.mark.skipif(FFMPEG is None, reason="ffmpeg is required")
@pytest.mark.parametrize("fmt", ["mp3", "m4a"])
def test_lossy_formats_are_encoded_and_true_peak_is_verified(fmt, tmp_path, capsys):
    alternating = np.empty(int(0.5 * 48000), dtype=np.float32)
    alternating[0::2] = 0.99
    alternating[1::2] = -0.99
    source = _write(tmp_path / "hot.wav", alternating, 48000)
    output = tmp_path / f"out.{fmt}"
    code, out, err = _run(
        capsys,
        "post",
        str(source),
        "--out",
        str(output),
        "--format",
        fmt,
        "--true-peak",
        "-1",
        "--json",
    )
    assert code == 0, err
    assert output.is_file()
    decoded, sr = _ffmpeg_decode(output, tmp_path / "decoded.wav")
    measured, reason = true_peak_dbtp(decoded, sr)
    assert reason is None
    assert measured <= -0.9
    entry = _json(out)["files"][0]
    assert entry["after_encoded"]["true_peak_dbtp"] == measured
    assert entry["output_format"] == fmt


@pytest.mark.parametrize(
    "args",
    [
        ("--out", "out.mp3", "--format", "wav"),
        ("--out", "out.ogg", "--format", "wav"),
        ("--out", "out.ogg"),
        ("--out", "out.wav", "--format", "flac"),
    ],
)
def test_output_format_conflicts_are_usage_errors(args, tmp_path, capsys):
    source = _write(tmp_path / "in.wav", _tone(0.2))
    code, _, _ = _run(capsys, "post", str(source), *args, "--json")
    assert code == 2


def test_mp3_input_unknown_input_and_empty_stem_are_usage_errors(tmp_path, capsys):
    wav = _write(tmp_path / "in.wav", _tone(0.2))
    mp3 = tmp_path / "in.mp3"
    mp3.write_bytes(b"not audio")
    code, _, _ = _run(capsys, "post", str(mp3), "--out", str(tmp_path / "o.wav"))
    assert code == 2
    unknown = tmp_path / "in.bin"
    unknown.write_bytes(b"not audio")
    code, _, _ = _run(capsys, "post", str(unknown), "--out", str(tmp_path / "o.wav"))
    assert code == 2
    hidden = tmp_path / ".wav"
    hidden.write_bytes(b"")
    code, _, _ = _run(capsys, "post", str(hidden), "--out", str(tmp_path / "o.wav"))
    assert code == 2
    assert wav.exists()


def test_format_option_scope_rules(tmp_path, capsys):
    source = _write(tmp_path / "in.wav", _tone(0.2))
    code, _, _ = _run(
        capsys,
        "post",
        str(source),
        "--out",
        str(tmp_path / "out.mp3"),
        "--bit-depth",
        "16",
    )
    assert code == 2
    code, _, _ = _run(
        capsys,
        "post",
        str(source),
        "--out",
        str(tmp_path / "out.wav"),
        "--bitrate",
        "128k",
    )
    assert code == 2
    code, _, _ = _run(
        capsys,
        "post",
        str(source),
        "--out",
        str(tmp_path / "out.wav"),
        "--trim-padding-ms",
        "10",
    )
    assert code == 2
    code, _, _ = _run(
        capsys,
        "post",
        str(source),
        "--out",
        str(tmp_path / "out.wav"),
        "--lufs",
        "1",
    )
    assert code == 2


# --------------------------------------------------------------------------- #
# 7. Atomicity
# --------------------------------------------------------------------------- #
def test_failure_preserves_existing_output_and_leaves_no_temp(tmp_path, capsys):
    source = _write(tmp_path / "short.wav", _tone(0.1))
    output = tmp_path / "existing.wav"
    output.write_bytes(b"OLD-BYTES")
    code, _, _ = _run(
        capsys, "post", str(source), "--out", str(output), "--lufs", "-20", "--json"
    )
    assert code == 4
    assert output.read_bytes() == b"OLD-BYTES"
    assert list(tmp_path.glob("*.tmp")) == []


def test_encoding_failure_preserves_existing_output(tmp_path, monkeypatch, capsys):
    source = _write(tmp_path / "in.wav", _tone(0.5))
    output = tmp_path / "existing.wav"
    output.write_bytes(b"OLD-BYTES")

    def boom(*_args, **_kwargs):
        raise InputError("encoder exploded")

    monkeypatch.setattr(post_cmd, "_encode_once", boom)
    code, _, _ = _run(capsys, "post", str(source), "--out", str(output), "--json")
    assert code == 4
    assert output.read_bytes() == b"OLD-BYTES"
    assert list(tmp_path.glob("*.tmp")) == []


def test_encoded_retry_exhaustion_preserves_existing_output(
    tmp_path, monkeypatch, capsys
):
    source = _write(tmp_path / "in.wav", _tone(0.5))
    output = tmp_path / "existing.wav"
    output.write_bytes(b"OLD-BYTES")

    def always_hot(samples, sr, fmt, **_kwargs):
        decoded = np.full_like(samples, 0.9, dtype=np.float32)
        return b"NEW-BYTES", decoded, sr

    monkeypatch.setattr(post_cmd, "_encode_once", always_hot)
    code, out, err = _run(
        capsys,
        "post",
        str(source),
        "--out",
        str(output),
        "--true-peak",
        "-3",
        "--json",
    )
    assert code == 4
    assert err == ""
    payload = _json(out)
    assert set(payload["files"][0]) == ENTRY_KEYS
    assert payload["files"][0]["ok"] is False
    assert payload["files"][0]["error"]["code"] == 4
    assert output.read_bytes() == b"OLD-BYTES"
    assert not any(path.suffix == ".tmp" for path in tmp_path.iterdir())


def test_success_atomically_replaces_existing_output(tmp_path, capsys):
    source = _write(tmp_path / "in.wav", _tone(0.5))
    output = tmp_path / "existing.wav"
    output.write_bytes(b"OLD-BYTES")
    code, _, err = _run(capsys, "post", str(source), "--out", str(output))
    assert code == 0, err
    assert output.read_bytes() != b"OLD-BYTES"
    _read(output)


def test_encoded_retry_path(monkeypatch, tmp_path):
    calls: list[float] = []

    def fake_encode(samples, sr, fmt, **_kwargs):
        calls.append(float(np.max(np.abs(samples))))
        decoded = samples * 4.0 if len(calls) == 1 else samples
        return b"payload", decoded.astype(np.float32), sr

    monkeypatch.setattr(post_cmd, "_encode_once", fake_encode)
    after_dsp = np.full((4800, 1), 0.8, dtype=np.float32)
    payload, metrics, duration, retries, margin = post_cmd._encode_verified(
        after_dsp,
        48000,
        "wav",
        bit_depth=24,
        bitrate="192k",
        ffmpeg=None,
        target=tmp_path / "out.wav",
        ceiling=-3.0,
    )
    assert payload == b"payload"
    assert retries == 1
    assert margin > 0.0
    assert len(calls) == 2
    assert calls[1] < calls[0]
    assert metrics["true_peak_dbtp"] <= -2.9
    assert duration == pytest.approx(0.1)


# --------------------------------------------------------------------------- #
# 8. Path guards
# --------------------------------------------------------------------------- #
def test_input_output_and_report_collisions(tmp_path, capsys):
    source = _write(tmp_path / "in.wav", _tone(0.3))
    output = tmp_path / "out.wav"
    for args in (
        (str(source), "--out", str(source)),
        (str(source), "--out", str(output), "--report", str(source)),
        (str(source), "--out", str(output), "--report", str(output)),
    ):
        code, _, _ = _run(capsys, "post", *args, "--json")
        assert code == 2
    assert not output.exists()


def test_symlink_case_nfc_and_relative_collisions(tmp_path, monkeypatch, capsys):
    source = _write(tmp_path / "source.wav", _tone(0.3))
    link = tmp_path / "link.wav"
    link.symlink_to(source)
    code, _, _ = _run(capsys, "post", str(source), "--out", str(link), "--json")
    assert code == 2

    upper = tmp_path / "SOURCE.WAV"
    code, _, _ = _run(capsys, "post", str(source), "--out", str(upper), "--json")
    assert code == 2

    composed = tmp_path / "café.wav"
    decomposed = tmp_path / "café.wav"
    _write(composed, _tone(0.3))
    code, _, _ = _run(
        capsys, "post", str(composed), "--out", str(decomposed), "--json"
    )
    assert code == 2

    monkeypatch.chdir(tmp_path)
    code, _, _ = _run(capsys, "post", "source.wav", "--out", "source.wav", "--json")
    assert code == 2


def test_destination_resolution_and_stdout_guards(tmp_path, capsys):
    upper = tmp_path / "UPPER.WAV"
    _write(upper, _tone(0.3))
    bare = tmp_path / "no-extension"
    code, _, err = _run(
        capsys, "post", str(upper), "--out", str(bare), "--format", "wav"
    )
    assert code == 0, err
    assert (tmp_path / "no-extension.wav").is_file()

    code, _, _ = _run(capsys, "post", str(upper), "--out", str(tmp_path / "bare"))
    assert code == 2
    code, _, _ = _run(
        capsys, "post", str(upper), "--out-dir", str(tmp_path / "dir")
    )
    assert code == 2
    code, _, _ = _run(capsys, "post", str(upper), "--out", "-", "--format", "wav")
    assert code == 2
    code, _, _ = _run(
        capsys, "post", str(upper), "--out-dir", "-", "--format", "wav"
    )
    assert code == 2
    code, _, _ = _run(
        capsys,
        "post",
        str(upper),
        "--out",
        str(tmp_path / "out.wav"),
        "--report",
        "-",
    )
    assert code == 2


def test_missing_destination_and_stem_conflict(tmp_path, capsys):
    source = _write(tmp_path / "in.wav", _tone(0.3))
    code, _, _ = _run(capsys, "post", str(source), "--json")
    assert code == 2

    first = tmp_path / "one" / "same.wav"
    second = tmp_path / "two" / "same.wav"
    first.parent.mkdir()
    second.parent.mkdir()
    _write(first, _tone(0.2))
    _write(second, _tone(0.2))
    out_dir = tmp_path / "out"
    code, _, _ = _run(
        capsys,
        "post",
        str(first),
        str(second),
        "--out-dir",
        str(out_dir),
        "--format",
        "wav",
        "--json",
    )
    assert code == 2
    assert not out_dir.exists()


# --------------------------------------------------------------------------- #
# 9. Multi-input aggregation
# --------------------------------------------------------------------------- #
def test_partial_failure_exits_six_and_keeps_success(tmp_path, capsys):
    good = _write(tmp_path / "good.wav", _tone(0.5))
    missing = tmp_path / "missing.wav"
    out_dir = tmp_path / "out"
    code, out, err = _run(
        capsys,
        "post",
        str(good),
        str(missing),
        "--out-dir",
        str(out_dir),
        "--format",
        "wav",
        "--json",
    )
    assert code == 6, err
    payload = _json(out)
    assert err == ""
    assert payload["ok"] is False
    assert [item["ok"] for item in payload["files"]] == [True, False]
    assert set(payload["files"][0]) == ENTRY_KEYS
    assert set(payload["files"][1]) == ENTRY_KEYS
    assert set(payload["files"][0]) == set(payload["files"][1])
    assert payload["files"][1]["error"]["code"] == 4
    assert (out_dir / "good.wav").is_file()
    assert not (out_dir / "missing.wav").exists()


def test_all_failures_return_first_input_error(tmp_path, capsys):
    first = tmp_path / "first.wav"
    second = tmp_path / "second.wav"
    out_dir = tmp_path / "out"
    code, out, _ = _run(
        capsys,
        "post",
        str(first),
        str(second),
        "--out-dir",
        str(out_dir),
        "--format",
        "wav",
        "--json",
    )
    assert code == 4
    payload = _json(out)
    assert [item["input"] for item in payload["files"]] == [
        str(first),
        str(second),
    ]
    assert all(item["error"]["code"] == 4 for item in payload["files"])


# --------------------------------------------------------------------------- #
# 10. JSON/report identity and dry-run
# --------------------------------------------------------------------------- #
def test_json_and_report_are_byte_identical(tmp_path, capsys):
    source = _write(tmp_path / "in.wav", _tone(0.5))
    output = tmp_path / "out.flac"
    report = tmp_path / "report.json"
    code, out, err = _run(
        capsys,
        "post",
        str(source),
        "--out",
        str(output),
        "--report",
        str(report),
        "--json",
    )
    assert code == 0, err
    assert out.encode("utf-8") == report.read_bytes()
    payload = _json(out)
    assert "exit_code" not in payload
    assert payload["request"]["true_peak_ceiling_scope"] == "encoded"
    entry = payload["files"][0]
    assert entry["input_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert entry["output_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()


def test_dry_run_writes_nothing_and_does_not_create_directories(tmp_path, capsys):
    source = _write(tmp_path / "in.wav", _tone(0.5))
    out_dir = tmp_path / "not-created"
    code, out, err = _run(
        capsys,
        "post",
        str(source),
        "--out-dir",
        str(out_dir),
        "--format",
        "wav",
        "--lufs",
        "-20",
        "--dry-run",
        "--json",
    )
    assert code == 0, err
    assert not out_dir.exists()
    entry = _json(out)["files"][0]
    assert entry["after_dsp"] is not None
    assert entry["after_encoded"] is None
    assert entry["output_sha256"] is None
    assert entry["output"] == str(out_dir / "in.wav")
    assert entry["ok"] is True


@pytest.mark.parametrize("dry_run", [True, False])
def test_target_parent_file_is_global_usage_error(dry_run, tmp_path, capsys):
    source = _write(tmp_path / "in.wav", _tone(0.3))
    parent_file = tmp_path / "parent-file"
    parent_file.write_bytes(b"PARENT")

    for args in (
        ("--out-dir", str(parent_file), "--format", "wav"),
        ("--out", str(parent_file / "out.wav")),
    ):
        extra = ("--dry-run",) if dry_run else ()
        code, out, err = _run(capsys, "post", str(source), *args, *extra, "--json")
        assert code == 2
        assert out == ""
        assert json.loads(err)["code"] == 2
        assert parent_file.read_bytes() == b"PARENT"
        assert not (parent_file / "out.wav").exists()
        assert not (parent_file / "in.wav").exists()


def test_request_output_dir_expands_tilde(tmp_path, monkeypatch, capsys):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    source = _write(tmp_path / "in.wav", _tone(0.3))
    code, out, err = _run(
        capsys,
        "post",
        str(source),
        "--out-dir",
        "~/build",
        "--format",
        "wav",
        "--dry-run",
        "--json",
    )
    assert code == 0, err
    payload = _json(out)
    assert payload["request"]["output_dir"] == str(home / "build")
    assert payload["files"][0]["output"] == str(home / "build" / "in.wav")
    assert not (home / "build").exists()


def test_dry_run_with_report_is_usage_error(tmp_path, capsys):
    source = _write(tmp_path / "in.wav", _tone(0.2))
    code, _, _ = _run(
        capsys,
        "post",
        str(source),
        "--out",
        str(tmp_path / "out.wav"),
        "--dry-run",
        "--report",
        str(tmp_path / "report.json"),
    )
    assert code == 2
    assert not (tmp_path / "report.json").exists()


def test_report_write_failure_keeps_audio_and_returns_four(tmp_path, capsys):
    source = _write(tmp_path / "in.wav", _tone(0.5))
    output = tmp_path / "out.wav"
    bad_report = tmp_path / "report-dir"
    bad_report.mkdir()
    code, out, _ = _run(
        capsys,
        "post",
        str(source),
        "--out",
        str(output),
        "--report",
        str(bad_report),
        "--json",
    )
    assert code == 4
    assert output.is_file()
    payload = _json(out)
    assert payload["ok"] is False
    assert payload["error"]["code"] == 4
    assert payload["files"][0]["ok"] is True


# --------------------------------------------------------------------------- #
# 11. Model-free import
# --------------------------------------------------------------------------- #
def test_post_import_does_not_load_model_runtimes():
    repo = Path(__file__).resolve().parents[1]
    code = (
        "import sys; import qtts_cli.commands.post; "
        "bad=[m for m in ('torch','qwen_tts','qwen_asr') if m in sys.modules]; "
        "assert not bad, bad"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code], cwd=str(repo), capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stderr


# --------------------------------------------------------------------------- #
# Contract addendum §M edge cases
# --------------------------------------------------------------------------- #
def test_nan_inf_and_multichannel_inputs_fail_per_file(tmp_path, capsys):
    nan_path = tmp_path / "nan.wav"
    _write(nan_path, np.array([0.1, np.nan, 0.2], dtype=np.float32), subtype="FLOAT")
    inf_path = tmp_path / "inf.wav"
    _write(inf_path, np.array([0.1, np.inf, 0.2], dtype=np.float32), subtype="FLOAT")
    three = _write(tmp_path / "three.wav", np.zeros((SR, 3), dtype=np.float32))

    for source in (nan_path, inf_path):
        output = tmp_path / f"{source.stem}-out.wav"
        code, _, _ = _run(capsys, "post", str(source), "--out", str(output), "--json")
        assert code == 4
        assert not output.exists()
    output = tmp_path / "three-out.wav"
    code, _, _ = _run(capsys, "post", str(three), "--out", str(output), "--json")
    assert code == 4
    assert not output.exists()


def test_short_audio_and_silence_rules(tmp_path, capsys):
    short = _write(tmp_path / "short.wav", _tone(0.35))
    output = tmp_path / "short-out.wav"
    code, _, _ = _run(
        capsys, "post", str(short), "--out", str(output), "--lufs", "-20", "--json"
    )
    assert code == 4
    assert not output.exists()

    # No LUFS/true-peak request: silence is allowed to pass through.
    silent = _write(tmp_path / "silent.wav", np.zeros(SR, dtype=np.float32))
    passthrough = tmp_path / "silent-out.wav"
    code, _, err = _run(capsys, "post", str(silent), "--out", str(passthrough))
    assert code == 0, err
    assert passthrough.is_file()


def test_limit_true_peak_never_clips_and_meets_safety_tolerance():
    alternating = np.empty(4800, dtype=np.float32)
    alternating[0::2] = 0.995
    alternating[1::2] = -0.995
    limited, reduction, achieved, safety = limit_true_peak(alternating, 48000, -2.0)
    assert reduction >= 0.0
    assert safety >= 0.0
    assert achieved <= -1.9
    assert float(np.max(np.abs(limited))) < 1.0


def test_cpu_dsp_process_reports_before_after_dsp_and_applied():
    signal = _tone(1.0, amplitude=0.5)
    output, sr, applied, before, after_dsp, warnings = process(
        signal,
        SR,
        {
            "lufs": -20.0,
            "true_peak": None,
            "sample_rate": 48000,
            "channels": "stereo",
            "trim": False,
            "trim_threshold_dbfs": -50.0,
            "trim_min_silence_s": 0.2,
            "trim_padding_ms": 50,
            "trim_mode": "both",
        },
    )
    assert sr == 48000
    assert output.shape[1] == 2
    assert before["lufs_integrated"] is not None
    assert after_dsp["lufs_integrated"] is not None
    assert applied["gain_db"] < 0
    assert applied["true_peak_ceiling_dbtp"] == -1.0
    assert applied["limiter_algorithm"] == LIMITER_ALGORITHM
    assert warnings == []


def test_global_json_error_emits_single_stderr_object(tmp_path, capsys):
    source = _write(tmp_path / "in.wav", _tone(0.2))
    code, out, err = _run(
        capsys,
        "post",
        str(source),
        "--out",
        str(source),
        "--json",
    )
    assert code == 2
    assert out == ""
    payload = json.loads(err)
    assert payload["ok"] is False
    assert payload["code"] == 2
    assert payload["error"] == "usage"
