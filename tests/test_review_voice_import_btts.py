"""Independent review tests for FR-P2-09 (btts voice import).

These complement ``test_voice_import_btts.py`` with adversarial bundle cases
and a commit-time rollback scenario. They do not modify the implementation.
"""

from __future__ import annotations

import io
import json
import os
import stat
import struct
import warnings
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from qtts_cli import voice_btts
from qtts_cli.commands import voice as voice_cmd
from qtts_cli.errors import InputError, ModelError
from qtts_cli.voice_store import VoiceStore

from test_voice_import_btts import (
    _bundle,
    _clone_voice,
    _design_voice,
    _error,
    _invoke,
    _leftovers,
    _voice_meta,
)


@pytest.fixture()
def review_tts(monkeypatch):
    calls: list[dict] = []

    class FakeEngine:
        instances = 0
        fail_on_call = 0

        def __init__(self, cfg, model):
            type(self).instances += 1
            self.dtype_name = "fp32"

        def load(self):
            return self

        def create_clone_prompt_items(
            self, path, *, ref_text=None, x_vector_only_mode=False
        ):
            calls.append(
                {
                    "path": str(path),
                    "ref_text": ref_text,
                    "xvec": x_vector_only_mode,
                }
            )
            if type(self).fail_on_call and len(calls) >= type(self).fail_on_call:
                raise ModelError("fake prompt failure")
            return [
                SimpleNamespace(
                    ref_spk_embedding=None,
                    ref_code=None,
                    x_vector_only_mode=x_vector_only_mode,
                    icl_mode=not x_vector_only_mode,
                )
            ]

    def fake_save(
        self,
        name,
        *,
        ref_spk_embedding,
        ref_code=None,
        x_vector_only_mode=False,
        icl_mode=True,
        directory=None,
    ):
        target = Path(directory) / "prompt.safetensors"
        target.write_bytes(b"fake-prompt")
        return target

    monkeypatch.setattr(voice_cmd, "TtsEngine", FakeEngine)
    monkeypatch.setattr(voice_cmd, "clear_model_cache", lambda: None)
    monkeypatch.setattr(VoiceStore, "save_prompt_tensors", fake_save)
    return SimpleNamespace(calls=calls, cls=FakeEngine)


def _flaky_replace(real, store: VoiceStore):
    def replace(src, dst, *args, **kwargs):
        src_path = Path(src)
        dst_path = Path(dst)
        if (
            src_path.parent == store.root
            and src_path.name.startswith(".bob.")
            and dst_path == store.dir_for("bob")
        ):
            raise OSError("simulated publish failure")
        return real(src, dst, *args, **kwargs)

    return replace


def test_commit_failure_restores_replaced_voice_and_new_voice(
    monkeypatch, capsys, tmp_path, model_dir, review_tts
):
    store = VoiceStore(tmp_path / "data")
    staging = store.staging_dir("alice")
    store.create(
        name="alice",
        kind="design",
        source={"type": "design", "description": "old"},
        model={"id": "old-model", "revision": None},
        staging=staging,
    )
    store.commit_staging(staging, "alice")

    bundle = _bundle(tmp_path, [_clone_voice("alice"), _clone_voice("bob")])
    monkeypatch.setattr(
        voice_cmd.os, "replace", _flaky_replace(os.replace, store)
    )
    rc, _, err = _invoke(
        capsys,
        tmp_path,
        model_dir,
        "voice",
        "import-btts",
        str(bundle),
        "--all",
        "--i-have-rights",
        "--force",
        "--json",
    )
    assert rc == 4
    assert "simulated publish failure" in _error(capsys, err)["message"]
    assert _voice_meta(tmp_path, "alice")["model"]["id"] == "old-model"
    assert not (tmp_path / "data" / "voices" / "bob").exists()
    assert _leftovers(tmp_path, "alice") == []
    assert _leftovers(tmp_path, "bob") == []


@pytest.mark.parametrize(
    ("member", "needle"),
    [
        ("/etc/passwd", "absolute"),
        ("C:/evil", "absolute or drive"),
        ("bundle\\evil", "absolute or drive"),
    ],
)
def test_raw_zip_unsafe_members_are_rejected(
    capsys, tmp_path, model_dir, review_tts, member, needle
):
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr(member, b"x")
    rc, _, err = _invoke(
        capsys,
        tmp_path,
        model_dir,
        "voice",
        "import-btts",
        str(archive),
        "--all",
        "--i-have-rights",
        "--json",
    )
    assert rc == 4
    assert needle in _error(capsys, err)["message"]


def test_zip_symlink_member_is_rejected(capsys, tmp_path, model_dir, review_tts):
    archive = tmp_path / "symlink.zip"
    info = zipfile.ZipInfo("bundle/alice/ref.txt")
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("bundle/", b"")
        handle.writestr(info, b"../../etc/passwd")
    rc, _, err = _invoke(
        capsys,
        tmp_path,
        model_dir,
        "voice",
        "import-btts",
        str(archive),
        "--all",
        "--i-have-rights",
        "--json",
    )
    assert rc == 4
    assert "non-regular" in _error(capsys, err)["message"]


def test_zip_compression_ratio_bomb_is_rejected(
    capsys, tmp_path, model_dir, review_tts
):
    archive = tmp_path / "bomb.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as handle:
        handle.writestr("manifest.json", b"{}")
        handle.writestr("bundle/alice/profile.json", b"0" * (2 * 1024 * 1024))
    rc, _, err = _invoke(
        capsys,
        tmp_path,
        model_dir,
        "voice",
        "import-btts",
        str(archive),
        "--all",
        "--i-have-rights",
        "--json",
    )
    assert rc == 4
    assert "compression ratio" in _error(capsys, err)["message"]


@pytest.mark.parametrize(
    ("field", "value", "needle"),
    [
        ("kind", "design", "kind"),
        ("language", "en", "language"),
        ("profile_sha256", "sha256:" + "0" * 64, "profile_sha256"),
    ],
)
def test_manifest_voice_metadata_mismatch_is_rejected(
    capsys, tmp_path, model_dir, review_tts, field, value, needle
):
    bundle = _bundle(tmp_path, [_clone_voice()])
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["voices"][0][field] = value
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    rc, _, err = _invoke(
        capsys,
        tmp_path,
        model_dir,
        "voice",
        "import-btts",
        str(bundle),
        "--voice",
        "alice",
        "--i-have-rights",
        "--json",
    )
    assert rc == 4
    assert needle in _error(capsys, err)["message"]


def test_manifest_extra_file_entry_is_rejected(
    capsys, tmp_path, model_dir, review_tts
):
    bundle = _bundle(tmp_path, [_clone_voice()])
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["root_files"].append(
        {"path": "EVIL.txt", "sha256": "sha256:" + "0" * 64, "bytes": 1}
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    rc, _, err = _invoke(
        capsys,
        tmp_path,
        model_dir,
        "voice",
        "import-btts",
        str(bundle),
        "--voice",
        "alice",
        "--i-have-rights",
        "--json",
    )
    assert rc == 4
    assert "manifest" in _error(capsys, err)["message"]


def test_empty_asr_transcript_leaves_no_staging(
    monkeypatch, capsys, tmp_path, model_dir, review_tts
):
    class EmptyAsr:
        def __init__(self, cfg, model):
            pass

        def transcribe(self, path, *, language=None, timestamps=False):
            return {"text": "   ", "language": language}

    monkeypatch.setattr(voice_cmd, "AsrEngine", EmptyAsr)
    monkeypatch.setattr(
        voice_cmd,
        "select_asr",
        lambda models, spec=None: SimpleNamespace(
            model_id="fake-asr",
            revision=None,
            fingerprint="fake",
            path=tmp_path / "asr",
        ),
    )
    monkeypatch.setattr(voice_cmd, "clear_asr_cache", lambda: None)
    bundle = _bundle(tmp_path, [_design_voice()])
    rc, _, err = _invoke(
        capsys,
        tmp_path,
        model_dir,
        "voice",
        "import-btts",
        str(bundle),
        "--voice",
        "narrator",
        "--i-have-rights",
        "--auto-ref-text",
        "--json",
    )
    assert rc == 4
    assert "empty ref_text" in _error(capsys, err)["message"]
    assert not (tmp_path / "data" / "voices" / "narrator").exists()
    assert _leftovers(tmp_path, "narrator") == []
    assert review_tts.calls == []


def test_ref_text_with_multiple_design_voices_is_usage_error(
    capsys, tmp_path, model_dir, review_tts
):
    bundle = _bundle(
        tmp_path,
        [_design_voice("one"), _design_voice("two")],
    )
    rc, _, err = _invoke(
        capsys,
        tmp_path,
        model_dir,
        "voice",
        "import-btts",
        str(bundle),
        "--all",
        "--i-have-rights",
        "--ref-text",
        "text",
        "--json",
    )
    assert rc == 2
    assert "single design voice" in _error(capsys, err)["message"]
    assert review_tts.calls == []


def test_clone_xvec_skips_transcript_and_records_mode(
    capsys, tmp_path, model_dir, review_tts
):
    bundle = _bundle(tmp_path, [_clone_voice()])
    rc, payload, err = _invoke(
        capsys,
        tmp_path,
        model_dir,
        "voice",
        "import-btts",
        str(bundle),
        "--voice",
        "alice",
        "--i-have-rights",
        "--mode",
        "xvec",
        "--json",
    )
    assert rc == 0, err
    assert payload["voices"][0]["ref_text_source"] is None
    assert review_tts.calls[0]["ref_text"] is None
    assert review_tts.calls[0]["xvec"] is True
    meta = _voice_meta(tmp_path, "alice")
    assert meta["source"]["mode"] == "xvec"
    assert meta["source"]["ref_text"] is None
    assert meta["source"]["btts"]["reference"]["transcript_confirmed"] is True


def test_design_without_manifest_auto_ref_text_is_explicitly_unverified(
    monkeypatch, capsys, tmp_path, model_dir, review_tts
):
    class FakeAsr:
        def __init__(self, cfg, model):
            pass

        def transcribe(self, path, *, language=None, timestamps=False):
            return {"text": "转写结果", "language": language}

    monkeypatch.setattr(voice_cmd, "AsrEngine", FakeAsr)
    monkeypatch.setattr(
        voice_cmd,
        "select_asr",
        lambda models, spec=None: SimpleNamespace(
            model_id="fake-asr",
            revision=None,
            fingerprint="fake",
            path=tmp_path / "asr",
        ),
    )
    monkeypatch.setattr(voice_cmd, "clear_asr_cache", lambda: None)
    bundle = _bundle(tmp_path, [_design_voice("narrator")], include_manifest=False)
    rc, payload, err = _invoke(
        capsys,
        tmp_path,
        model_dir,
        "voice",
        "import-btts",
        str(bundle),
        "--voice",
        "narrator",
        "--i-have-rights",
        "--auto-ref-text",
        "--json",
    )
    assert rc == 0, err
    assert payload["manifest"] == {
        "present": False,
        "verified": False,
        "sha256": None,
    }
    meta = _voice_meta(tmp_path, "narrator")
    assert meta["source"]["btts"]["manifest"]["present"] is False


def _wav_mode_bundle(tmp_path: Path, *, zip_mode: bool) -> Path:
    """A real ``btts voice export --mode wav`` layout: root ``*.wav`` only."""
    files = {
        "manifest.json": json.dumps(
            {
                "schema_version": 1,
                "kind": "btts-voice-export",
                "mode": "wav",
                "root_files": [],
                "voices": [],
            }
        ).encode("utf-8"),
        "LICENSE.txt": b"root license\n",
        "README.txt": b"readme\n",
        "alice.wav": b"RIFF....WAVEfmt ",
    }
    if zip_mode:
        archive = tmp_path / "wav-export.zip"
        with zipfile.ZipFile(archive, "w") as handle:
            handle.writestr("wav-export/", b"")
            for rel, data in files.items():
                handle.writestr(f"wav-export/{rel}", data)
        return archive
    root = tmp_path / "wav-export"
    root.mkdir()
    for rel, data in files.items():
        (root / rel).write_bytes(data)
    return root


@pytest.mark.parametrize("zip_mode", [False, True])
def test_real_wav_mode_bundle_gets_full_mode_hint(
    capsys, tmp_path, model_dir, review_tts, zip_mode
):
    bundle = _wav_mode_bundle(tmp_path, zip_mode=zip_mode)
    rc, _, err = _invoke(
        capsys,
        tmp_path,
        model_dir,
        "voice",
        "import-btts",
        str(bundle),
        "--all",
        "--i-have-rights",
        "--json",
    )
    assert rc == 4
    payload = _error(capsys, err)
    assert "--mode full" in payload["message"]
    assert payload["details"]["file"] == "manifest.json"
    assert payload["details"]["mode"] == "wav"
    assert review_tts.calls == []


def _malformed_zip(tmp_path: Path, mutate) -> Path:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as handle:
        handle.writestr("manifest.json", b"{}")
        handle.writestr("alice/profile.json", b"{}")
    data = bytearray(buffer.getvalue())
    eocd = data.rfind(b"PK\x05\x06")
    assert eocd >= 0
    (central_offset,) = struct.unpack_from("<I", data, eocd + 16)
    mutate(data, central_offset)
    archive = tmp_path / "malformed.zip"
    archive.write_bytes(bytes(data))
    return archive


def test_malformed_zip_unsupported_extract_version_is_input_error(
    capsys, tmp_path, model_dir, review_tts
):
    archive = _malformed_zip(
        tmp_path, lambda data, off: struct.pack_into("<H", data, off + 6, 999)
    )
    rc, _, err = _invoke(
        capsys,
        tmp_path,
        model_dir,
        "voice",
        "import-btts",
        str(archive),
        "--all",
        "--i-have-rights",
        "--json",
    )
    assert rc == 4
    payload = _error(capsys, err)
    assert payload["error"] == "input"
    assert "invalid btts zip bundle" in payload["message"]
    assert payload["details"]["bundle"] == str(archive)
    assert "file" in payload["details"]


def test_malformed_zip_bad_utf8_filename_is_input_error(
    capsys, tmp_path, model_dir, review_tts
):
    def mutate(data: bytearray, off: int) -> None:
        struct.pack_into("<H", data, off + 8, 0x800)
        data[off + 46] = 0x92

    archive = _malformed_zip(tmp_path, mutate)
    rc, _, err = _invoke(
        capsys,
        tmp_path,
        model_dir,
        "voice",
        "import-btts",
        str(archive),
        "--all",
        "--i-have-rights",
        "--json",
    )
    assert rc == 4
    payload = _error(capsys, err)
    assert payload["error"] == "input"
    assert "invalid btts zip bundle" in payload["message"]
    assert payload["details"]["bundle"] == str(archive)


def test_malformed_zip_broken_local_header_is_input_error(
    capsys, tmp_path, model_dir, review_tts
):
    archive = _malformed_zip(
        tmp_path,
        lambda data, off: struct.pack_into("<I", data, off + 42, 0xFFFFFF00),
    )
    rc, _, err = _invoke(
        capsys,
        tmp_path,
        model_dir,
        "voice",
        "import-btts",
        str(archive),
        "--all",
        "--i-have-rights",
        "--json",
    )
    assert rc == 4
    payload = _error(capsys, err)
    assert payload["error"] == "input"
    assert "invalid btts zip bundle" in payload["message"]
    assert payload["details"]["bundle"] == str(archive)
    assert payload["details"]["file"]


def test_open_rel_wraps_zip_oserror_with_bundle_and_file(monkeypatch, tmp_path):
    archive = tmp_path / "bundle.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("manifest.json", b"{}")

    def broken_open(self, name, *args, **kwargs):
        raise OSError(22, "Invalid argument")

    monkeypatch.setattr(zipfile.ZipFile, "open", broken_open)
    with pytest.raises(InputError) as excinfo:
        voice_btts._read_rel_path(archive, "zip", "", "manifest.json", 1024)
    assert excinfo.value.details["bundle"] == str(archive)
    assert excinfo.value.details["file"] == "manifest.json"


def test_oversized_manifest_is_rejected(capsys, tmp_path, model_dir, review_tts):
    bundle = _bundle(tmp_path, [_clone_voice()])
    (bundle / "manifest.json").write_bytes(
        b"{" + b" " * (voice_btts.MAX_PROFILE_BYTES + 1) + b"}"
    )
    rc, _, err = _invoke(
        capsys,
        tmp_path,
        model_dir,
        "voice",
        "import-btts",
        str(bundle),
        "--voice",
        "alice",
        "--i-have-rights",
        "--json",
    )
    assert rc == 4
    payload = _error(capsys, err)
    assert "manifest.json is too large" in payload["message"]
    assert payload["details"]["file"] == "manifest.json"
    assert not (tmp_path / "data" / "voices" / "alice").exists()


def test_bundle_file_count_limit_is_reported(capsys, tmp_path, model_dir, review_tts):
    archive = tmp_path / "big.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("manifest.json", b"{}")
        for index in range(voice_btts.MAX_FILES):
            handle.writestr(f"v{index:03d}/f.txt", b"x")
    rc, _, err = _invoke(
        capsys,
        tmp_path,
        model_dir,
        "voice",
        "import-btts",
        str(archive),
        "--all",
        "--i-have-rights",
        "--json",
    )
    assert rc == 4
    payload = _error(capsys, err)
    assert "too many files" in payload["message"]
    assert str(voice_btts.MAX_FILES) in payload["message"]


def test_asr_model_error_exits_three_and_leaves_no_staging(
    monkeypatch, capsys, tmp_path, model_dir, review_tts
):
    class FailingAsr:
        def __init__(self, cfg, model):
            pass

        def transcribe(self, path, *, language=None, timestamps=False):
            raise ModelError("fake ASR failure")

    monkeypatch.setattr(voice_cmd, "AsrEngine", FailingAsr)
    monkeypatch.setattr(
        voice_cmd,
        "select_asr",
        lambda models, spec=None: SimpleNamespace(
            model_id="fake-asr",
            revision=None,
            fingerprint="fake",
            path=tmp_path / "asr",
        ),
    )
    monkeypatch.setattr(voice_cmd, "clear_asr_cache", lambda: None)
    bundle = _bundle(tmp_path, [_design_voice()])
    rc, _, err = _invoke(
        capsys,
        tmp_path,
        model_dir,
        "voice",
        "import-btts",
        str(bundle),
        "--voice",
        "narrator",
        "--i-have-rights",
        "--auto-ref-text",
        "--json",
    )
    assert rc == 3
    assert "fake ASR failure" in _error(capsys, err)["message"]
    assert not (tmp_path / "data" / "voices" / "narrator").exists()
    assert _leftovers(tmp_path, "narrator") == []
    assert review_tts.calls == []


_ZIP_PARSE_EXCEPTIONS = [
    NotImplementedError("zip file version 23.1"),
    UnicodeDecodeError("utf-8", b"\x92", 0, 1, "invalid start byte"),
    OSError(22, "Invalid argument"),
    ValueError("bad central directory"),
]


@pytest.mark.parametrize(
    "error",
    _ZIP_PARSE_EXCEPTIONS,
    ids=lambda exc: type(exc).__name__,
)
def test_zip_scan_wraps_all_parse_errors(monkeypatch, tmp_path, error):
    archive = tmp_path / "bundle.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("manifest.json", b"{}")

    class BrokenZipFile:
        def __init__(self, *args, **kwargs):
            raise error

    monkeypatch.setattr(zipfile, "ZipFile", BrokenZipFile)
    with pytest.raises(InputError) as excinfo:
        voice_btts.load_bundle(archive)
    assert excinfo.value.details["bundle"] == str(archive)
    assert "file" in excinfo.value.details
    assert "invalid btts zip bundle" in excinfo.value.message


@pytest.mark.parametrize(
    "error",
    _ZIP_PARSE_EXCEPTIONS,
    ids=lambda exc: type(exc).__name__,
)
def test_open_rel_wraps_all_parse_errors(monkeypatch, tmp_path, error):
    archive = tmp_path / "bundle.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("manifest.json", b"{}")

    def broken_open(self, name, *args, **kwargs):
        raise error

    monkeypatch.setattr(zipfile.ZipFile, "open", broken_open)
    with pytest.raises(InputError) as excinfo:
        voice_btts._read_rel_path(archive, "zip", "", "manifest.json", 1024)
    assert excinfo.value.details["bundle"] == str(archive)
    assert excinfo.value.details["file"] == "manifest.json"
    assert "invalid btts zip bundle" in excinfo.value.message


def _overlapping_zip(tmp_path: Path) -> Path:
    """Two central-directory entries sharing one local header (overlap)."""
    archive = tmp_path / "overlap.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("manifest.json", b"{}")
        handle.writestr("README.txt", b"readme")
        handle.writestr("README2.txt", b"readme2")
    data = bytearray(archive.read_bytes())
    eocd = data.rfind(b"PK\x05\x06")
    assert eocd >= 0
    (central_offset,) = struct.unpack_from("<I", data, eocd + 16)

    def next_entry(offset: int) -> int:
        name_len, extra_len, comment_len = struct.unpack_from(
            "<HHH", data, offset + 28
        )
        return offset + 46 + name_len + extra_len + comment_len

    readme = next_entry(central_offset)
    readme2 = next_entry(readme)
    (readme_offset,) = struct.unpack_from("<I", data, readme + 42)
    struct.pack_into("<I", data, readme2 + 42, readme_offset)
    archive.write_bytes(bytes(data))
    return archive


def test_overlapping_zip_entries_become_one_json_error(
    capsys, tmp_path, model_dir, review_tts
):
    archive = _overlapping_zip(tmp_path)
    # Simulate PYTHONWARNINGS=error::UserWarning: the warning must be turned
    # into InputError, never escape as a traceback or a bare stderr line.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        rc, _, err = _invoke(
            capsys,
            tmp_path,
            model_dir,
            "voice",
            "import-btts",
            str(archive),
            "--all",
            "--i-have-rights",
            "--json",
        )
    assert rc == 4
    payload = _error(capsys, err)
    assert payload["error"] == "input"
    assert "Overlapped" in payload["message"]
    assert payload["details"]["bundle"] == str(archive)
    assert payload["details"]["file"] == "README.txt"
