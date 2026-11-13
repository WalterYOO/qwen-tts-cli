"""FR-P2-09: import voices from btts export bundles.

The TTS engine and prompt serialization are faked; these tests cover bundle
security, selection/consent rules, provenance, atomicity and the clone/design
conversion paths without loading model weights or touching the network.
"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile

from qtts_cli.cli import main
from qtts_cli.commands import voice as voice_cmd
from qtts_cli.errors import ModelError
from qtts_cli.voice_archive import export_voice, import_voice
from qtts_cli.voice_store import VoiceStore

TEXT = "你好，欢迎使用语音克隆。"
ASR_TEXT = "这是设计音色试听文本。"
USER_TEXT = "这是自定义参考文本内容。"


def _wav_bytes(seconds: float = 4.0, sr: int = 24000) -> bytes:
    t = np.arange(int(seconds * sr), dtype=np.float32) / sr
    data = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    buffer = io.BytesIO()
    soundfile.write(buffer, data, sr, format="WAV")
    return buffer.getvalue()


WAV = _wav_bytes()


def _sha(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _clone_profile(name: str, language: str = "zh") -> dict:
    return {
        "schema_version": 1,
        "voice_id": f"v_{name}",
        "name": name,
        "kind": "clone",
        "language": language,
        "identity": {"structured": {}, "rendered_instruction": ""},
        "design": None,
        "reference": {
            "audio": "ref.wav",
            "audio_sha256": _sha(WAV),
            "transcript": "ref.txt",
            "transcript_sha256": _sha(TEXT.encode("utf-8")),
            "transcript_confirmed": True,
            "duration_s": 4.0,
            "sample_rate": 24000,
        },
        "defaults": {"cfg_scale": 1.0, "seed": 19},
        "provenance": {
            "source": "reference_audio",
            "model_id": "BreezeBlue/breeze-tts-2",
            "api_protocol": "local-mps/v1",
            "cli_version": "0.2.5",
        },
        "license": {"model": "breezeblue-research-non-commercial"},
        "consent": {"note": "btts profile consent"},
        "tags": ["narration", "test"],
    }


def _design_profile(name: str, language: str = "zh") -> dict:
    return {
        "schema_version": 1,
        "voice_id": f"v_{name}",
        "name": name,
        "kind": "design",
        "language": language,
        "identity": {
            "structured": {},
            "rendered_instruction": "温柔女声",
            "template": "zh-late-night-radio/v1",
            "compiler_version": "1",
        },
        "design": {
            "seed": 19,
            "cfg_scale": 1.0,
            "instruction": "温柔女声",
            "max_new_tokens": 300,
        },
        "reference": None,
        "defaults": {"cfg_scale": 1.0, "seed": 19},
        "provenance": {
            "source": "designed",
            "model_id": "BreezeBlue/breeze-tts-2",
            "api_protocol": "local-mps/v1",
            "cli_version": "0.2.5",
        },
        "license": {"model": "breezeblue-research-non-commercial"},
        "consent": {"note": "btts design consent"},
        "tags": ["design"],
    }


def _clone_voice(name: str = "alice", language: str = "zh") -> tuple[str, dict, dict]:
    files = {
        "LICENSE.txt": b"voice license\n",
        "preview.wav": WAV,
        "ref.wav": WAV,
        "ref.txt": (TEXT + "\n").encode("utf-8"),
    }
    return name, _clone_profile(name, language), files


def _design_voice(
    name: str = "narrator", language: str = "zh", *, design_yaml: bool = True
) -> tuple[str, dict, dict]:
    files = {"LICENSE.txt": b"voice license\n", "preview.wav": WAV}
    if design_yaml:
        files["design.yaml"] = b"schema_version: 1\nname: narrator\n"
    return name, _design_profile(name, language), files


def _bundle(
    tmp_path: Path,
    voices: list[tuple[str, dict, dict]],
    *,
    zip_mode: bool = False,
    include_manifest: bool = True,
    manifest_mode: str = "full",
    name: str = "bundle",
) -> Path:
    root_files = {"LICENSE.txt": b"root license\n", "README.txt": b"readme\n"}
    payload = dict(root_files)
    voice_entries = []
    for voice_name, profile, files in voices:
        profile_bytes = (
            json.dumps(profile, ensure_ascii=False, indent=2) + "\n"
        ).encode("utf-8")
        full = {"profile.json": profile_bytes, **files}
        for filename, data in full.items():
            payload[f"{voice_name}/{filename}"] = data
        voice_entries.append(
            {
                "name": voice_name,
                "voice_id": profile["voice_id"],
                "kind": profile["kind"],
                "language": profile["language"],
                "tags": profile.get("tags", []),
                "profile_sha256": _sha(profile_bytes),
                "files": [
                    {
                        "path": f"{voice_name}/{filename}",
                        "sha256": _sha(data),
                        "bytes": len(data),
                    }
                    for filename, data in sorted(full.items())
                ],
            }
        )
    manifest = {
        "schema_version": 1,
        "kind": "btts-voice-export",
        "exported_at": "2026-01-01T00:00:00Z",
        "cli_version": "0.2.5",
        "mode": manifest_mode,
        "license": {"model": "breezeblue-research-non-commercial"},
        "root_files": [
            {"path": filename, "sha256": _sha(data), "bytes": len(data)}
            for filename, data in sorted(root_files.items())
        ],
        "voices": voice_entries,
        "warnings": [],
    }

    if zip_mode:
        archive = tmp_path / f"{name}.zip"
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as handle:
            handle.writestr(f"{name}/", b"")
            for rel, data in payload.items():
                handle.writestr(f"{name}/{rel}", data)
            if include_manifest:
                handle.writestr(
                    f"{name}/manifest.json",
                    json.dumps(manifest, ensure_ascii=False).encode("utf-8"),
                )
        return archive

    directory = tmp_path / name
    directory.mkdir()
    for rel, data in payload.items():
        target = directory / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    if include_manifest:
        (directory / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
        )
    return directory


@pytest.fixture()
def fake_tts(monkeypatch):
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


@pytest.fixture()
def fake_asr(monkeypatch, tmp_path):
    calls: list[dict] = []

    class FakeAsr:
        def __init__(self, cfg, model):
            pass

        def transcribe(self, path, *, language=None, timestamps=False):
            calls.append({"path": str(path), "language": language})
            return {"text": ASR_TEXT, "language": language, "items": None}

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
    return calls


def _invoke(capsys, tmp_path, model_dir, *args):
    rc = main(
        [
            *args,
            "--model-dir",
            str(model_dir),
            "--data-dir",
            str(tmp_path / "data"),
            "--cache-dir",
            str(tmp_path / "cache"),
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out) if captured.out.strip() else None
    return rc, payload, captured.err


def _error(capsys, err: str) -> dict:
    assert err.strip(), "expected a JSON error on stderr"
    return json.loads(err)


def _voice_meta(tmp_path: Path, name: str) -> dict:
    path = tmp_path / "data" / "voices" / name / "voice.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _leftovers(tmp_path: Path, name: str) -> list[Path]:
    root = tmp_path / "data" / "voices"
    if not root.is_dir():
        return []
    return list(root.glob(f".{name}.*"))


def test_clone_directory_bundle_success(
    monkeypatch, capsys, tmp_path, model_dir, fake_tts
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
        "--json",
    )
    assert rc == 0, err
    assert payload["kind"] == "btts-import"
    assert payload["manifest"]["present"] is True
    assert payload["manifest"]["verified"] is True
    assert payload["voices"][0]["ref_text_source"] == "btts"

    voice_dir = tmp_path / "data" / "voices" / "alice"
    assert (voice_dir / "prompt.safetensors").is_file()
    assert (voice_dir / "ref.wav").read_bytes() == WAV
    assert (voice_dir / "preview.wav").read_bytes() == WAV
    meta = _voice_meta(tmp_path, "alice")
    assert meta["kind"] == "clone"
    assert meta["source"]["origin"] == "btts"
    assert meta["source"]["ref_text"] == TEXT
    assert meta["source"]["ref_text_source"] == "btts"
    assert meta["source"]["btts"]["name"] == "alice"
    assert meta["source"]["btts"]["voice_id"] == "v_alice"
    assert meta["source"]["btts"]["license"]["model"] == "breezeblue-research-non-commercial"
    assert meta["source"]["btts"]["profile_sha256"]
    assert meta["consent"]["rights_confirmed"] is True
    assert meta["consent"]["license"] == "btts-import"
    assert meta["model"]["id"].endswith("Base")
    assert fake_tts.calls[0]["ref_text"] == TEXT
    assert fake_tts.calls[0]["xvec"] is False


def test_clone_zip_bundle_success(
    monkeypatch, capsys, tmp_path, model_dir, fake_tts
):
    bundle = _bundle(tmp_path, [_clone_voice()], zip_mode=True)
    rc, payload, err = _invoke(
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
    assert rc == 0, err
    assert payload["bundle_format"] == "zip"
    assert payload["voices"][0]["name"] == "alice"
    assert (tmp_path / "data" / "voices" / "alice" / "voice.json").is_file()


def test_directory_without_manifest_is_allowed_but_reported(
    monkeypatch, capsys, tmp_path, model_dir, fake_tts
):
    bundle = _bundle(tmp_path, [_clone_voice()], include_manifest=False)
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
        "--json",
    )
    assert rc == 0, err
    assert payload["manifest"]["present"] is False
    assert payload["manifest"]["verified"] is False
    assert any("manifest.json is missing" in warning for warning in payload["warnings"])


def test_zip_without_manifest_is_rejected(
    monkeypatch, capsys, tmp_path, model_dir, fake_tts
):
    bundle = _bundle(tmp_path, [_clone_voice()], zip_mode=True, include_manifest=False)
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
    assert "manifest.json" in _error(capsys, err)["message"]


def test_manifest_hash_mismatch_is_rejected(
    monkeypatch, capsys, tmp_path, model_dir, fake_tts
):
    bundle = _bundle(tmp_path, [_clone_voice()])
    transcript = bundle / "alice" / "ref.txt"
    data = bytearray(transcript.read_bytes())
    data[0] ^= 0xFF  # same size, different bytes: a hash mismatch, not a size mismatch
    transcript.write_bytes(bytes(data))
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
    error = _error(capsys, err)
    assert "sha256" in error["message"]
    assert error["details"]["bundle"] == str(bundle)
    assert error["details"]["file"].endswith("ref.txt")
    assert not (tmp_path / "data" / "voices" / "alice").exists()


def test_wav_mode_bundle_is_rejected(
    monkeypatch, capsys, tmp_path, model_dir, fake_tts
):
    bundle = _bundle(tmp_path, [_clone_voice()], manifest_mode="wav")
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
    assert "--mode full" in _error(capsys, err)["message"]


def test_symlink_in_directory_bundle_is_rejected(
    monkeypatch, capsys, tmp_path, model_dir, fake_tts
):
    bundle = _bundle(tmp_path, [_clone_voice()])
    target = bundle / "alice" / "ref.txt"
    target.unlink()
    target.symlink_to(bundle / "LICENSE.txt")
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
    assert "symlink" in _error(capsys, err)["message"]


def test_zip_path_traversal_is_rejected(
    monkeypatch, capsys, tmp_path, model_dir, fake_tts
):
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("bundle/", b"")
        handle.writestr("bundle/manifest.json", b"{}")
        handle.writestr("bundle/../evil", b"x")
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
    assert "escapes" in _error(capsys, err)["message"]
    assert not (tmp_path / "evil").exists()


def test_zip_casefold_collision_is_rejected(
    monkeypatch, capsys, tmp_path, model_dir, fake_tts
):
    bundle = _bundle(tmp_path, [_clone_voice("alice"), _clone_voice("Alice")], zip_mode=True)
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
    assert "case-insensitive" in _error(capsys, err)["message"]


def test_missing_rights_is_rejected_before_model_load(
    monkeypatch, capsys, tmp_path, model_dir, fake_tts
):
    bundle = _bundle(tmp_path, [_clone_voice()])
    rc, _, err = _invoke(
        capsys,
        tmp_path,
        model_dir,
        "voice",
        "import-btts",
        str(bundle),
        "--voice",
        "alice",
        "--json",
    )
    assert rc == 4
    assert "i-have-rights" in _error(capsys, err)["message"]
    assert fake_tts.cls.instances == 0


def test_design_default_imports_recipe_without_model_or_asr(
    monkeypatch, capsys, tmp_path, model_dir, fake_tts
):
    bundle = _bundle(tmp_path, [_design_voice()])
    monkeypatch.setattr(
        voice_cmd, "scan", lambda _dir: pytest.fail("model scan must not run")
    )
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
        "--json",
    )
    assert rc == 0, err
    entry = payload["voices"][0]
    assert entry["kind"] == "design"
    assert entry["design_mode"] == "recipe"
    assert entry["ref_text_source"] is None
    assert payload["design_mode"] == "auto"
    assert any("acoustic-equivalence-not-possible" in item for item in payload["warnings"])

    voice_dir = tmp_path / "data" / "voices" / "narrator"
    assert (voice_dir / "voice.json").is_file()
    assert (voice_dir / "preview.wav").read_bytes() == WAV
    assert not (voice_dir / "ref.wav").exists()
    meta = _voice_meta(tmp_path, "narrator")
    assert meta["kind"] == "design"
    assert meta["source"]["origin"] == "btts"
    assert meta["source"]["description"] == "温柔女声"
    assert meta["source"]["materialized"] is False
    assert meta["source"]["design"]["template_id"] == "zh-late-night-radio/v1"
    assert meta["source"]["design"]["template"] == "qtts/zh-late-night-radio/v1"
    assert meta["source"]["design"]["rendered_instruction"] == "温柔女声"
    assert meta["source"]["design"]["compiler_family"] == "btts"
    assert meta["source"]["design"]["compiler_version"] == "1"
    assert meta["source"]["design"]["seed"] == 19
    assert meta["source"]["design"]["cfg_scale"] == 1.0
    assert meta["source"]["design"]["max_new_tokens"] == 300
    assert meta["source"]["design"]["yaml"].startswith("schema_version: 1")
    assert meta["source"]["design"]["yaml_sha256"]
    assert meta["source"]["preview"]["origin"] == "btts"
    assert meta["source"]["btts"]["bundle"] == "bundle"
    assert meta["source"]["btts"]["manifest"]["verified"] is True
    assert meta["source"]["btts"]["license"]["model"] == (
        "breezeblue-research-non-commercial"
    )
    assert meta["default_instruct"] == "温柔女声"
    assert meta["model"]["id"] is None
    assert "btts" in meta["tags"]
    assert meta["consent"]["license"] == "btts-import"
    assert "btts profile consent is provenance only" in meta["consent"]["note"]
    assert fake_tts.calls == []
    assert fake_tts.cls.instances == 0
    assert _leftovers(tmp_path, "narrator") == []


def test_design_clone_without_ref_text_is_an_input_error(
    monkeypatch, capsys, tmp_path, model_dir, fake_tts
):
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
        "--design-mode",
        "clone",
        "--json",
    )
    assert rc == 4
    error = _error(capsys, err)
    assert "--auto-ref-text" in error["message"]
    assert error["details"]["voice"] == "narrator"
    assert not (tmp_path / "data" / "voices" / "narrator").exists()
    assert _leftovers(tmp_path, "narrator") == []
    assert fake_tts.calls == []


def test_design_auto_ref_text_uses_asr(
    monkeypatch, capsys, tmp_path, model_dir, fake_tts, fake_asr
):
    bundle = _bundle(tmp_path, [_design_voice()])
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
    assert payload["voices"][0]["ref_text_source"] == "asr"
    assert fake_asr[0]["language"] == "Chinese"
    assert fake_tts.calls[0]["ref_text"] == ASR_TEXT
    meta = _voice_meta(tmp_path, "narrator")
    assert meta["kind"] == "clone"
    assert meta["source"]["ref_text"] == ASR_TEXT
    assert meta["source"]["ref_text_source"] == "asr"
    assert meta["source"]["asr_model"]["id"] == "fake-asr"
    assert meta["source"]["btts"]["kind"] == "design"
    assert meta["source"]["btts"]["design"]["instruction"] == "温柔女声"
    assert meta["default_instruct"] is None


def test_design_explicit_ref_text(
    monkeypatch, capsys, tmp_path, model_dir, fake_tts
):
    bundle = _bundle(tmp_path, [_design_voice()])
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
        "--ref-text",
        USER_TEXT,
        "--json",
    )
    assert rc == 0, err
    assert payload["voices"][0]["ref_text_source"] == "user"
    assert fake_tts.calls[0]["ref_text"] == USER_TEXT
    meta = _voice_meta(tmp_path, "narrator")
    assert meta["source"]["ref_text_source"] == "user"
    assert "asr_model" not in meta["source"]


def test_design_xvec_skips_transcript(
    monkeypatch, capsys, tmp_path, model_dir, fake_tts
):
    bundle = _bundle(tmp_path, [_design_voice()])
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
        "--mode",
        "xvec",
        "--json",
    )
    assert rc == 0, err
    assert payload["voices"][0]["ref_text_source"] is None
    assert fake_tts.calls[0]["ref_text"] is None
    assert fake_tts.calls[0]["xvec"] is True
    meta = _voice_meta(tmp_path, "narrator")
    assert meta["source"]["mode"] == "xvec"
    assert meta["source"]["ref_text"] is None


def test_all_imports_clone_and_design(
    monkeypatch, capsys, tmp_path, model_dir, fake_tts, fake_asr
):
    bundle = _bundle(tmp_path, [_clone_voice(), _design_voice()])
    rc, payload, err = _invoke(
        capsys,
        tmp_path,
        model_dir,
        "voice",
        "import-btts",
        str(bundle),
        "--all",
        "--i-have-rights",
        "--auto-ref-text",
        "--json",
    )
    assert rc == 0, err
    by_name = {entry["name"]: entry for entry in payload["voices"]}
    assert by_name["alice"]["ref_text_source"] == "btts"
    assert by_name["narrator"]["ref_text_source"] == "asr"
    assert (tmp_path / "data" / "voices" / "alice" / "voice.json").is_file()
    assert (tmp_path / "data" / "voices" / "narrator" / "voice.json").is_file()
    assert len(fake_tts.calls) == 2
    assert len(fake_asr) == 1


def test_existing_voice_requires_force_and_force_replaces(
    monkeypatch, capsys, tmp_path, model_dir, fake_tts
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

    bundle = _bundle(tmp_path, [_clone_voice()])
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
    assert "already exists" in _error(capsys, err)["message"]
    assert _voice_meta(tmp_path, "alice")["model"]["id"] == "old-model"
    assert _leftovers(tmp_path, "alice") == []

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
        "--force",
        "--json",
    )
    assert rc == 0, err
    assert payload["voices"][0]["replaced"] is True
    assert _voice_meta(tmp_path, "alice")["model"]["id"].endswith("Base")
    assert _leftovers(tmp_path, "alice") == []


def _simulate_case_sensitive_fs(monkeypatch, target: Path) -> None:
    """Make ``target.exists()`` report False, as on a case-sensitive FS."""
    real_exists = Path.exists

    def fake_exists(self):
        if self == target:
            return False
        return real_exists(self)

    monkeypatch.setattr(Path, "exists", fake_exists)


def test_existing_store_casefold_conflict_is_detected_before_model_load(
    monkeypatch, capsys, tmp_path, model_dir, fake_tts
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

    _simulate_case_sensitive_fs(monkeypatch, store.dir_for("Alice"))
    monkeypatch.setattr(
        voice_cmd, "scan", lambda _dir: pytest.fail("model scan must not run")
    )
    bundle = _bundle(tmp_path, [_clone_voice()])
    rc, _, err = _invoke(
        capsys,
        tmp_path,
        model_dir,
        "voice",
        "import-btts",
        str(bundle),
        "--voice",
        "alice",
        "--name",
        "Alice",
        "--i-have-rights",
        "--json",
    )
    assert rc == 4
    assert "already exists" in _error(capsys, err)["message"]
    assert fake_tts.calls == []
    assert _voice_meta(tmp_path, "alice")["model"]["id"] == "old-model"


def test_force_replaces_casefold_existing_voice(
    monkeypatch, capsys, tmp_path, model_dir, fake_tts
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

    _simulate_case_sensitive_fs(monkeypatch, store.dir_for("Alice"))
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
        "--name",
        "Alice",
        "--i-have-rights",
        "--force",
        "--json",
    )
    assert rc == 0, err
    assert payload["voices"][0]["replaced"] is True
    meta = _voice_meta(tmp_path, "Alice")
    assert meta["name"] == "Alice"
    assert meta["model"]["id"].endswith("Base")
    assert _leftovers(tmp_path, "Alice") == []


def test_name_override_and_multi_voice_name_rejected(
    monkeypatch, capsys, tmp_path, model_dir, fake_tts
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
        "--name",
        "bob",
        "--i-have-rights",
        "--json",
    )
    assert rc == 0, err
    assert payload["voices"][0]["name"] == "bob"
    assert (tmp_path / "data" / "voices" / "bob" / "voice.json").is_file()
    assert not (tmp_path / "data" / "voices" / "alice").exists()

    rc, _, err = _invoke(
        capsys,
        tmp_path,
        model_dir,
        "voice",
        "import-btts",
        str(bundle),
        "--all",
        "--name",
        "bob",
        "--i-have-rights",
        "--json",
    )
    assert rc == 2
    assert "--name" in _error(capsys, err)["message"]

    rc, _, err = _invoke(
        capsys,
        tmp_path,
        model_dir,
        "voice",
        "import-btts",
        str(bundle),
        "--voice",
        "alice",
        "--voice",
        "alice",
        "--i-have-rights",
        "--json",
    )
    assert rc == 2
    assert "unique" in _error(capsys, err)["message"]


def test_failure_during_second_voice_leaves_no_partial_profiles(
    monkeypatch, capsys, tmp_path, model_dir, fake_tts
):
    bundle = _bundle(tmp_path, [_clone_voice("alice"), _clone_voice("bob")])
    fake_tts.cls.fail_on_call = 2
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
    assert rc == 3
    assert "fake prompt failure" in _error(capsys, err)["message"]
    assert not (tmp_path / "data" / "voices" / "alice").exists()
    assert not (tmp_path / "data" / "voices" / "bob").exists()
    assert _leftovers(tmp_path, "alice") == []
    assert _leftovers(tmp_path, "bob") == []


def test_unknown_voice_selection_is_rejected(
    monkeypatch, capsys, tmp_path, model_dir, fake_tts
):
    bundle = _bundle(tmp_path, [_clone_voice()])
    rc, _, err = _invoke(
        capsys,
        tmp_path,
        model_dir,
        "voice",
        "import-btts",
        str(bundle),
        "--voice",
        "missing",
        "--i-have-rights",
        "--json",
    )
    assert rc == 4
    error = _error(capsys, err)
    assert error["details"]["unknown"] == ["missing"]
    assert error["details"]["available"] == ["alice"]


@pytest.mark.parametrize(
    "extra",
    [
        ["--auto-ref-text", "--mode", "xvec"],
        ["--asr-model", "fake"],
        ["--ref-text", USER_TEXT, "--mode", "xvec"],
    ],
)
def test_invalid_flag_combinations_are_usage_errors(
    monkeypatch, capsys, tmp_path, model_dir, fake_tts, extra
):
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
        "--json",
        *extra,
    )
    assert rc == 2
    assert _error(capsys, err)["error"] == "usage"


def test_ref_text_on_clone_is_a_usage_error(
    monkeypatch, capsys, tmp_path, model_dir, fake_tts
):
    bundle = _bundle(tmp_path, [_clone_voice()])
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
        "--ref-text",
        USER_TEXT,
        "--json",
    )
    assert rc == 2
    assert "design voice" in _error(capsys, err)["message"]


def test_auto_ref_text_with_only_clone_is_a_usage_error(
    monkeypatch, capsys, tmp_path, model_dir, fake_tts
):
    bundle = _bundle(tmp_path, [_clone_voice()])
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
        "--auto-ref-text",
        "--json",
    )
    assert rc == 2
    assert "design voice" in _error(capsys, err)["message"]


@pytest.mark.parametrize(
    "extra",
    [
        ["--design-mode", "recipe", "--mode", "xvec"],
        ["--design-mode", "recipe", "--auto-ref-text"],
        ["--design-mode", "recipe", "--ref-text", USER_TEXT],
        ["--design-mode", "recipe", "--asr-model", "fake"],
        ["--design-mode", "recipe", "--model", "fake"],
        ["--design-mode", "auto", "--mode", "xvec"],
        ["--design-mode", "auto", "--auto-ref-text"],
        ["--design-mode", "auto", "--ref-text", USER_TEXT],
        ["--design-mode", "auto", "--asr-model", "fake"],
        ["--design-mode", "auto", "--model", "fake"],
    ],
)
def test_recipe_and_auto_reject_clone_only_options(
    monkeypatch, capsys, tmp_path, model_dir, fake_tts, extra
):
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
        "--json",
        *extra,
    )
    assert rc == 2
    error = _error(capsys, err)
    assert error["error"] == "usage"
    assert "clone-only" in error["message"]
    assert fake_tts.calls == []


def test_explicit_recipe_imports_offline(monkeypatch, capsys, tmp_path, model_dir, fake_tts):
    bundle = _bundle(tmp_path, [_design_voice()])
    monkeypatch.setattr(voice_cmd, "scan", lambda _dir: pytest.fail("model scan must not run"))
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
        "--design-mode",
        "recipe",
        "--json",
    )
    assert rc == 0, err
    assert payload["design_mode"] == "recipe"
    assert payload["voices"][0]["design_mode"] == "recipe"
    assert fake_tts.calls == []


def _mutated_design_voice(**updates):
    name, profile, files = _design_voice()
    for path, value in updates.items():
        node = profile
        keys = path.split(".")
        for key in keys[:-1]:
            node = node[key]
        node[keys[-1]] = value
    return name, profile, files


def test_recipe_instruction_falls_back_to_identity_rendered(
    monkeypatch, capsys, tmp_path, model_dir, fake_tts
):
    bundle = _bundle(
        tmp_path,
        [_mutated_design_voice(
            **{
                "design.instruction": "   ",
                "identity.rendered_instruction": "备选描述",
            }
        )],
    )
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
        "--json",
    )
    assert rc == 0, err
    meta = _voice_meta(tmp_path, "narrator")
    assert meta["source"]["description"] == "备选描述"
    assert meta["source"]["design"]["rendered_instruction"] == "备选描述"
    assert meta["default_instruct"] == "备选描述"


def test_recipe_without_any_instruction_is_an_input_error(
    monkeypatch, capsys, tmp_path, model_dir, fake_tts
):
    bundle = _bundle(
        tmp_path,
        [_mutated_design_voice(
            **{
                "design.instruction": "",
                "identity.rendered_instruction": "",
            }
        )],
    )
    monkeypatch.setattr(voice_cmd, "scan", lambda _dir: pytest.fail("model scan must not run"))
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
        "--json",
    )
    assert rc == 4
    assert "no instruction" in _error(capsys, err)["message"]
    assert not (tmp_path / "data" / "voices" / "narrator").exists()
    assert _leftovers(tmp_path, "narrator") == []


def test_recipe_instruction_mismatch_is_preserved_with_warning(
    monkeypatch, capsys, tmp_path, model_dir, fake_tts
):
    bundle = _bundle(
        tmp_path,
        [_mutated_design_voice(
            **{
                "design.instruction": "主描述",
                "identity.rendered_instruction": "旧描述",
            }
        )],
    )
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
        "--json",
    )
    assert rc == 0, err
    meta = _voice_meta(tmp_path, "narrator")
    assert meta["source"]["description"] == "主描述"
    assert meta["source"]["design"]["identity_rendered_instruction"] == "旧描述"
    assert "instruction-mismatch" in payload["voices"][0]["warnings"]


def test_unknown_btts_template_keeps_raw_id_and_warns(
    monkeypatch, capsys, tmp_path, model_dir, fake_tts
):
    bundle = _bundle(
        tmp_path,
        [_mutated_design_voice(**{"identity.template": "btts/unknown/v1"})],
    )
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
        "--json",
    )
    assert rc == 0, err
    entry = payload["voices"][0]
    assert entry["template"] is None
    assert entry["btts_template"] == "btts/unknown/v1"
    assert entry["template_mapped"] is False
    assert any("unsupported-btts-template" in item for item in payload["warnings"])
    meta = _voice_meta(tmp_path, "narrator")
    assert meta["source"]["design"]["template"] is None
    assert meta["source"]["design"]["template_id"] == "btts/unknown/v1"
    assert meta["source"]["btts"]["identity"]["template"] == "btts/unknown/v1"


def test_recipe_archive_round_trip_keeps_embedded_design_yaml(
    monkeypatch, capsys, tmp_path, model_dir, fake_tts
):
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
        "--json",
    )
    assert rc == 0, err
    store = VoiceStore(tmp_path / "data")
    original = store.get("narrator")
    archive = tmp_path / "narrator.qttsvoice"
    export_voice(store, "narrator", archive)
    other = VoiceStore(tmp_path / "other-data")
    import_voice(other, archive)
    restored = other.get("narrator")
    assert restored["source"]["design"] == original["source"]["design"]
    assert restored["source"]["btts"] == original["source"]["btts"]
    assert (other.dir_for("narrator") / "preview.wav").read_bytes() == (
        store.dir_for("narrator") / "preview.wav"
    ).read_bytes()
    assert restored["source"]["design"]["yaml"].startswith("schema_version: 1")


def test_btts_provenance_truncates_large_fields_with_sha(
    monkeypatch, capsys, tmp_path, model_dir, fake_tts
):
    long_value = "x" * 5000
    bundle = _bundle(
        tmp_path,
        [_mutated_design_voice(
            **{
                "identity.rendered_instruction": long_value,
                "defaults.direction": "y" * 600,
            }
        )],
    )
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
        "--json",
    )
    assert rc == 0, err
    meta = _voice_meta(tmp_path, "narrator")
    truncations = meta["source"]["btts"]["truncations"]
    assert truncations["identity.rendered_instruction"]["truncated"] is True
    assert truncations["identity.rendered_instruction"]["length"] == len(long_value)
    assert truncations["identity.rendered_instruction"]["sha256"]
    assert truncations["defaults.direction"]["sha256"]
    assert len(meta["source"]["btts"]["identity"]["rendered_instruction"]) == 4096
    assert len(meta["source"]["btts"]["defaults"]["direction"]) == 512


def test_dry_run_writes_nothing_and_skips_model_and_asr(
    monkeypatch, capsys, tmp_path, model_dir, fake_tts
):
    bundle = _bundle(tmp_path, [_clone_voice(), _design_voice()])
    monkeypatch.setattr(voice_cmd, "scan", lambda _dir: pytest.fail("model scan must not run"))
    monkeypatch.setattr(
        voice_cmd, "AsrEngine", lambda *args, **kwargs: pytest.fail("ASR must not run")
    )
    rc, payload, err = _invoke(
        capsys,
        tmp_path,
        model_dir,
        "voice",
        "import-btts",
        str(bundle),
        "--all",
        "--i-have-rights",
        "--auto-ref-text",
        "--dry-run",
        "--json",
    )
    assert rc == 0, err
    assert payload["dry_run"] is True
    assert {entry["design_mode"] for entry in payload["voices"]} == {"clone"}
    assert not (tmp_path / "data" / "voices").exists()
    assert fake_tts.calls == []
    assert any("asr-skipped-in-dry-run" in item for item in payload["warnings"])


def test_auto_mode_dry_run_plans_recipe_and_clone(
    monkeypatch, capsys, tmp_path, model_dir, fake_tts
):
    bundle = _bundle(tmp_path, [_clone_voice("alice"), _design_voice("narrator")])
    monkeypatch.setattr(voice_cmd, "scan", lambda _dir: pytest.fail("model scan must not run"))
    rc, payload, err = _invoke(
        capsys,
        tmp_path,
        model_dir,
        "voice",
        "import-btts",
        str(bundle),
        "--all",
        "--i-have-rights",
        "--dry-run",
        "--json",
    )
    assert rc == 0, err
    modes = {entry["name"]: entry["design_mode"] for entry in payload["voices"]}
    assert modes == {"alice": "clone", "narrator": "recipe"}
    recipe = next(entry for entry in payload["voices"] if entry["name"] == "narrator")
    assert recipe["template"] == "qtts/zh-late-night-radio/v1"
    assert not (tmp_path / "data" / "voices").exists()


def test_mixed_recipe_and_clone_import_rolls_back_together(
    monkeypatch, capsys, tmp_path, model_dir, fake_tts
):
    bundle = _bundle(tmp_path, [_clone_voice("bob"), _design_voice("narrator")])
    store = VoiceStore(tmp_path / "data")
    import os

    real_replace = os.replace

    def flaky(src, dst, *args, **kwargs):
        if str(dst) == str(store.dir_for("bob")):
            raise OSError("simulated publish failure")
        return real_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(voice_cmd.os, "replace", flaky)
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
    assert "simulated publish failure" in _error(capsys, err)["message"]
    assert not store.dir_for("bob").exists()
    assert not store.dir_for("narrator").exists()
    assert _leftovers(tmp_path, "bob") == []
    assert _leftovers(tmp_path, "narrator") == []


def test_text_formatter_uses_recipe_and_clone_labels(
    monkeypatch, capsys, tmp_path, model_dir, fake_tts
):
    bundle = _bundle(tmp_path, [_clone_voice("alice"), _design_voice("narrator")])
    rc = main(
        [
            "voice",
            "import-btts",
            str(bundle),
            "--all",
            "--i-have-rights",
            "--model-dir",
            str(model_dir),
            "--data-dir",
            str(tmp_path / "data"),
            "--cache-dir",
            str(tmp_path / "cache"),
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "narrator: design <- btts design" in out
    assert "alice: clone <- btts clone" in out
    assert "recipe" in out
    assert "qtts/zh-late-night-radio/v1" in out


def test_voice_preview_replaces_btts_preview_provenance(
    monkeypatch, capsys, tmp_path, model_dir, fake_tts
):
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
        "--json",
    )
    assert rc == 0, err
    before = _voice_meta(tmp_path, "narrator")

    class FakePreviewEngine:
        dtype_name = "fp32"
        device = "cpu"

        def __init__(self, cfg, model):
            pass

        def load(self):
            return self

        def generate(self, **kwargs):
            return np.zeros(2400, dtype=np.float32), 24000, {"generate_ms": 1.0}

    monkeypatch.setattr(voice_cmd, "TtsEngine", FakePreviewEngine)
    rc, _, err = _invoke(
        capsys,
        tmp_path,
        model_dir,
        "voice",
        "preview",
        "narrator",
        "--json",
    )
    assert rc == 0, err
    after = _voice_meta(tmp_path, "narrator")
    preview_path = tmp_path / "data" / "voices" / "narrator" / "preview.wav"
    digest = hashlib.sha256(preview_path.read_bytes()).hexdigest()
    assert after["source"]["preview"]["origin"] == "qtts"
    assert after["source"]["preview"]["file"] == "preview.wav"
    assert after["source"]["preview"]["sha256"] == digest
    assert after["source"]["preview"]["updated_at"]
    assert after["source"]["btts"]["preview_sha256"] == (
        before["source"]["btts"]["preview_sha256"]
    )
