"""FR-P2-02: DashScope / vLLM-Omni adapters and backend identity."""

from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile

from qtts_cli.catalog import ModelInfo, scan
from qtts_cli.errors import CapabilityError, InputError, ModelError, UsageError
from qtts_cli.plan import VoiceResolution, build_plan
from qtts_cli.voice_store import VoiceStore
from qtts_cli.backends import dashscope as ds
from qtts_cli.backends import vllm_omni as vo
from qtts_cli.backends.base import (
    BackendCapabilities,
    BackendSpec,
    ensure_instance_capabilities,
    ensure_instance_protocol,
)


# --------------------------------------------------------------------------- #
# DashScope
# --------------------------------------------------------------------------- #
def _wav_bytes(sr=24000, samples=480, channels=1):
    data = np.zeros((samples, channels), dtype=np.float32)
    data[:, 0] = np.linspace(-0.25, 0.25, samples)
    if channels == 2:
        data[:, 1] = data[:, 0]
    buffer = io.BytesIO()
    soundfile.write(buffer, data, sr, format="WAV")
    return buffer.getvalue()


@pytest.fixture()
def dashscope_env(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "secret-key")
    monkeypatch.setenv("DASHSCOPE_WORKSPACE", "ws-1")
    monkeypatch.setenv("DASHSCOPE_REGION", "cn-shanghai")


def test_dashscope_voice_list_matches_official_48():
    assert len(ds.DASHSCOPE_VOICES) == 48
    assert len(set(ds.DASHSCOPE_VOICES)) == 48
    assert {
        "Momo",
        "Eldric Sage",
        "Ono Anna",
        "Radio Gol",
        "Kiki",
    } <= set(ds.DASHSCOPE_VOICES)
    for stale in ("Pancake", "Gethsemane", "Warrior", "Anvil", "Alex"):
        assert stale not in ds.DASHSCOPE_VOICES


def test_dashscope_session_order_decode_and_factory_args(
    monkeypatch, dashscope_env
):
    calls = []

    class Session:
        def __enter__(self):
            calls.append("enter")
            return self

        def __exit__(self, *exc):
            calls.append("exit")

        def send_text(self, text):
            calls.append(("send_text", text))

        def flush(self):
            calls.append("flush")

        def stop(self):
            calls.append("stop")

        def audio_chunks(self):
            calls.append("chunks")
            payload = _wav_bytes(channels=2)
            half = len(payload) // 2
            return iter([payload[:half], payload[half:]])

    class TTS:
        def connect(self, **kwargs):
            calls.append(("connect", kwargs))
            return Session()

    class Client:
        audio = SimpleNamespace(tts=TTS())

    def factory(api_key, workspace, region):
        calls.append(("factory", api_key, workspace, region))
        return Client()

    monkeypatch.setattr(ds, "_client_factory", factory)
    backend = ds.DashScopeBackend(
        SimpleNamespace(), ds.synthetic_model("qwen3-tts-flash-realtime")
    )
    audio, sr, info = backend.generate(
        text="hello", language="English", mode="builtin", speaker="cherry"
    )

    assert calls[0] == ("factory", "secret-key", "ws-1", "cn-shanghai")
    assert calls[1][0] == "connect"
    connect = calls[1][1]
    assert connect["model"] == "qwen3-tts-flash-realtime"
    assert connect["voice"] == "Cherry"  # canonical spelling
    assert connect["format"] == "mp3"
    assert connect["sample_rate"] == ds.SAMPLE_RATE
    assert calls[2:] == [
        "enter",
        ("send_text", "hello"),
        "flush",
        "stop",
        "chunks",
        "exit",
    ]
    assert sr == 24000
    assert audio.ndim == 1
    assert audio.dtype == np.float32
    assert audio.size == 480
    assert np.all(np.isfinite(audio))
    assert info["backend"] == "dashscope"
    assert info["generate_ms"] > 0


def test_dashscope_decodes_real_mp3_bytes(tmp_path):
    import shutil
    import subprocess

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("ffmpeg is required to build an mp3 fixture")
    source = tmp_path / "in.wav"
    mono = np.zeros((2400, 1), dtype=np.float32)
    mono[:, 0] = 0.25 * np.sin(
        2 * np.pi * 220.0 * np.arange(2400) / 24000.0
    )
    stereo = np.repeat(mono, 2, axis=1)
    soundfile.write(source, stereo, 24000)
    target = tmp_path / "out.mp3"
    result = subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error", "-i", str(source), str(target)],
        capture_output=True,
    )
    if result.returncode != 0:
        pytest.skip(f"ffmpeg could not encode mp3: {result.stderr.decode()}")
    audio, sr = ds._decode_mp3(target.read_bytes())
    assert sr == 24000
    assert audio.ndim == 1
    assert audio.dtype == np.float32
    assert audio.size > 0
    assert np.all(np.isfinite(audio))


def test_dashscope_rejects_non_bytes_and_empty_payloads(monkeypatch, dashscope_env):
    def run(chunks):
        class Session:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return None

            def send_text(self, text):
                return None

            def flush(self):
                return None

            def stop(self):
                return None

            def audio_chunks(self):
                return iter(chunks)

        class Client:
            audio = SimpleNamespace(tts=SimpleNamespace(connect=lambda **kw: Session()))

        monkeypatch.setattr(ds, "_client_factory", lambda *a: Client())
        backend = ds.DashScopeBackend(SimpleNamespace(), ds.synthetic_model("m"))
        return backend.generate(
            text="hi", language="English", mode="builtin", speaker="Cherry"
        )

    with pytest.raises(ModelError) as excinfo:
        run(["not-bytes"])
    assert excinfo.value.code == 3
    with pytest.raises(ModelError) as excinfo:
        run([])
    assert excinfo.value.code == 3
    with pytest.raises(ModelError) as excinfo:
        run([b"not audio"])
    assert excinfo.value.code == 3


def _plan(voice, *, instruct=None):
    return SimpleNamespace(voice=voice, segments=[], instruct=instruct)


def _builtin_voice():
    return VoiceResolution(
        ref="builtin:Cherry", mode="builtin", model=ds.synthetic_model("m")
    )


def test_dashscope_preflight_key_offline_and_instruct(monkeypatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    cfg = SimpleNamespace(offline=False)
    dry = SimpleNamespace(dry_run=True)
    real = SimpleNamespace(dry_run=False)

    with pytest.raises(ModelError) as excinfo:
        ds.preflight(_plan(_builtin_voice()), cfg, real)
    assert excinfo.value.code == 3

    ds.preflight(_plan(_builtin_voice()), cfg, dry)  # dry-run needs no key

    with pytest.raises(UsageError) as excinfo:
        ds.preflight(
            _plan(_builtin_voice()),
            SimpleNamespace(offline=True),
            dry,
        )
    assert excinfo.value.code == 2

    with pytest.raises(CapabilityError) as excinfo:
        ds.preflight(
            _plan(_builtin_voice(), instruct="happy"), cfg, dry
        )
    assert excinfo.value.code == 4


def test_dashscope_preflight_rejects_clone_and_design():
    cfg = SimpleNamespace(offline=False)
    args = SimpleNamespace(dry_run=True)
    for mode, extra in (("clone", {"voice_name": "alice"}), ("design", {"description": "warm"})):
        voice = VoiceResolution(
            ref=f"{mode}:x", mode=mode, model=ds.synthetic_model("m"), **extra
        )
        with pytest.raises(CapabilityError) as excinfo:
            ds.preflight(_plan(voice), cfg, args)
        assert excinfo.value.code == 4


def test_registry_voice_preflight_rejects_non_builtin():
    from qtts_cli.backends.registry import preflight_voice_ref, resolve_backend

    args = SimpleNamespace(backend="dashscope", backend_model=None, model=None)
    spec = resolve_backend("dashscope", SimpleNamespace(), args)
    assert spec.identity.startswith("dashscope:qwen3-tts-flash-realtime:")
    for raw in ("voice:alice", "design:warm"):
        with pytest.raises(CapabilityError) as excinfo:
            preflight_voice_ref(raw, spec)
        assert excinfo.value.code == 4


# --------------------------------------------------------------------------- #
# vLLM-Omni
# --------------------------------------------------------------------------- #
def _base_model():
    return ModelInfo(
        path=Path("/tmp/definitely-not-a-model"),
        name="Qwen3-TTS-12Hz-1.7B-Base",
        model_type="qwen3_tts",
        tts_model_type="base",
        size="1b7",
    )


class _Stage:
    def __init__(self, audio, sr):
        self.outputs = [
            SimpleNamespace(multimodal_output={"audio": audio, "sr": sr})
        ]


class _Omni:
    def __init__(self, stages):
        self.stages = stages
        self.inputs = None

    def generate(self, inputs):
        self.inputs = inputs
        return self.stages


def _generate(mode, **kwargs):
    backend = vo.VllmOmniBackend(None, _base_model())
    omni = _Omni([_Stage(np.ones(8, dtype=np.float32), 24000)])
    backend.model = omni
    audio, sr, info = backend.generate(
        text="hello", language="English", mode=mode, sampling={"max_new_tokens": 32}, **kwargs
    )
    return backend, omni, audio, sr, info


def test_vllm_builtin_additional_information_is_field_by_field():
    _backend, omni, audio, sr, info = _generate(
        "builtin", speaker="Vivian", instruct="happy"
    )
    assert omni.inputs[0]["prompt_token_ids"] == [0] * vo.PROMPT_TOKEN_LENGTH
    assert omni.inputs[0]["additional_information"] == {
        "task_type": ["CustomVoice"],
        "text": ["hello"],
        "language": ["English"],
        "max_new_tokens": [32],
        "full_utterance_decode": [True],
        "speaker": ["Vivian"],
        "instruct": ["happy"],
    }
    assert sr == 24000
    assert audio.shape == (8,)
    assert info["backend"] == "vllm-omni"


def test_vllm_design_additional_information():
    _backend, omni, _audio, _sr, _info = _generate("design", instruct="warm")
    assert omni.inputs[0]["additional_information"] == {
        "task_type": ["VoiceDesign"],
        "text": ["hello"],
        "language": ["English"],
        "max_new_tokens": [32],
        "full_utterance_decode": [True],
        "instruct": ["warm"],
        "non_streaming_mode": [True],
    }


@pytest.mark.parametrize("xvec", [False, True])
def test_vllm_clone_additional_information_and_absolute_ref(tmp_path, xvec):
    ref = tmp_path / "ref.wav"
    prompt = {
        "ref_audio": str(ref.resolve()),
        "ref_text": "hi",
        "x_vector_only_mode": xvec,
    }
    _backend, omni, _audio, _sr, _info = _generate(
        "clone", prompt_items=prompt
    )
    assert omni.inputs[0]["additional_information"] == {
        "task_type": ["Base"],
        "text": ["hello"],
        "language": ["English"],
        "max_new_tokens": [32],
        "full_utterance_decode": [True],
        "ref_audio": [str(ref.resolve())],
        "ref_text": ["hi"],
        "x_vector_only_mode": [xvec],
    }
    assert Path(omni.inputs[0]["additional_information"]["ref_audio"][0]).is_absolute()


def test_vllm_output_chunks_are_concatenated_and_mono():
    backend = vo.VllmOmniBackend(None, _base_model())
    omni = _Omni(
        [
            _Stage(np.array([[1.0], [2.0]], dtype=np.float32), [24000]),
            _Stage([np.array([3.0], dtype=np.float32)], [24000]),
        ]
    )
    backend.model = omni
    audio, sr, _info = backend.generate(
        text="h", language="English", mode="builtin", speaker="Vivian"
    )
    assert sr == 24000
    assert audio.tolist() == [1.0, 2.0, 3.0]


def test_vllm_rejects_inconsistent_rates_and_non_finite_audio():
    backend = vo.VllmOmniBackend(None, _base_model())
    backend.model = _Omni(
        [
            _Stage(np.ones(4, dtype=np.float32), 24000),
            _Stage(np.ones(4, dtype=np.float32), 16000),
        ]
    )
    with pytest.raises(ModelError) as excinfo:
        backend.generate(
            text="h", language="English", mode="builtin", speaker="Vivian"
        )
    assert excinfo.value.code == 3

    backend.model = _Omni(
        [_Stage(np.array([np.nan, 0.0], dtype=np.float32), 24000)]
    )
    with pytest.raises(ModelError) as excinfo:
        backend.generate(
            text="h", language="English", mode="builtin", speaker="Vivian"
        )
    assert excinfo.value.code == 3


def _make_clone_voice(store: VoiceStore, name: str, *, ref="ref.wav", mode="xvec", ref_text=None):
    directory = store.dir_for(name)
    directory.mkdir(parents=True, exist_ok=True)
    source = {"type": "clone", "mode": mode, "ref_audio": ref}
    if ref_text is not None:
        source["ref_text"] = ref_text
    (directory / "voice.json").write_text(
        json.dumps(
            {"schema_version": 1, "name": name, "kind": "clone", "source": source}
        ),
        encoding="utf-8",
    )
    return directory


def _clone_plan(store: VoiceStore, name: str, *, icl=False, ref_text=None):
    return SimpleNamespace(
        voice=VoiceResolution(
            ref=f"voice:{name}",
            mode="clone",
            model=_base_model(),
            voice_name=name,
            icl=icl,
            ref_text=ref_text,
        ),
        segments=[],
        instruct=None,
    )


def test_vllm_preflight_accepts_ref_wav_only(tmp_path):
    store = VoiceStore(tmp_path / "data")
    directory = _make_clone_voice(store, "alice")
    (directory / "ref.wav").write_bytes(b"RIFF")
    vo.preflight(
        _clone_plan(store, "alice"),
        SimpleNamespace(data_dir=tmp_path / "data"),
        None,
    )


def test_vllm_preflight_rejects_missing_and_escaping_refs(tmp_path):
    store = VoiceStore(tmp_path / "data")
    _make_clone_voice(store, "alice")
    cfg = SimpleNamespace(data_dir=tmp_path / "data")
    with pytest.raises(InputError) as excinfo:
        vo.preflight(_clone_plan(store, "alice"), cfg, None)
    assert excinfo.value.code == 4

    escaping = _make_clone_voice(store, "bob", ref="../evil.wav")
    (tmp_path / "data" / "voices" / "evil.wav").write_bytes(b"RIFF")
    with pytest.raises(InputError) as excinfo:
        vo.preflight(_clone_plan(store, "bob"), cfg, None)
    assert excinfo.value.code == 4
    assert escaping.is_dir()


def test_vllm_preflight_requires_ref_text_for_icl(tmp_path):
    store = VoiceStore(tmp_path / "data")
    directory = _make_clone_voice(store, "alice", mode="icl")
    (directory / "ref.wav").write_bytes(b"RIFF")
    cfg = SimpleNamespace(data_dir=tmp_path / "data")
    with pytest.raises(InputError) as excinfo:
        vo.preflight(_clone_plan(store, "alice", icl=True), cfg, None)
    assert excinfo.value.code == 4
    vo.preflight(
        _clone_plan(store, "alice", icl=True, ref_text="hello"),
        cfg,
        None,
    )


# --------------------------------------------------------------------------- #
# Identity / capability invariants
# --------------------------------------------------------------------------- #
def test_backend_identity_changes_the_cache_key(tmp_path, model_dir):
    cfg = SimpleNamespace(seed=1234, dtype="fp32", device="auto")
    store = VoiceStore(tmp_path / "data")
    models = scan(model_dir)
    first = build_plan(
        cfg,
        text="你好。",
        voice_ref="builtin:Vivian",
        models=models,
        store=store,
        backend_identity="transformers",
    )
    second = build_plan(
        cfg,
        text="你好。",
        voice_ref="builtin:Vivian",
        models=models,
        store=store,
        backend_identity="plugin:fake@1.0.0:-",
    )
    assert first.segments[0].cache_key != second.segments[0].cache_key
    assert first.voice.identity["backend"] == "transformers"
    assert second.voice.identity["backend"] == "plugin:fake@1.0.0:-"


def _sampling_spec(supported):
    return BackendSpec(
        name="demo",
        identity="demo",
        models=[_base_model()],
        capabilities=BackendCapabilities(
            modes=frozenset({"builtin"}),
            instruct_modes=frozenset(),
            sampling=frozenset(supported),
        ),
        create=lambda cfg, model: None,
        remote=True,
    )


def test_sampling_is_filtered_to_declared_capabilities():
    from qtts_cli.commands.say import _validate_sampling_capabilities
    from qtts_cli.budget import DEFAULT_MAX_NEW_TOKENS
    from qtts_cli.plan import SAMPLING_DEFAULTS

    plan = SimpleNamespace(
        sampling={**SAMPLING_DEFAULTS, "max_new_tokens": DEFAULT_MAX_NEW_TOKENS},
        segments=[SimpleNamespace(max_new_tokens=DEFAULT_MAX_NEW_TOKENS)],
    )
    assert _validate_sampling_capabilities(
        plan, _sampling_spec({"max_new_tokens"})
    ) == frozenset({"max_new_tokens"})
    assert _validate_sampling_capabilities(plan, _sampling_spec(set())) == frozenset()

    explicit = SimpleNamespace(
        sampling={**plan.sampling, "temperature": 0.5},
        segments=plan.segments,
    )
    with pytest.raises(CapabilityError) as excinfo:
        _validate_sampling_capabilities(explicit, _sampling_spec(set()))
    assert excinfo.value.code == 4
    assert "temperature" in excinfo.value.message
    assert "supported: none" in excinfo.value.message


def test_canonical_or_passthrough_warns_for_unknown_backend_voice(tmp_path, model_dir):
    cfg = SimpleNamespace(seed=1234, dtype="fp32", device="auto")
    store = VoiceStore(tmp_path / "data")
    models = scan(model_dir)
    with pytest.raises(InputError):
        build_plan(
            cfg,
            text="hi",
            voice_ref="builtin:NotKnown",
            language="English",
            models=models,
            store=store,
        )
    plan = build_plan(
        cfg,
        text="hi",
        voice_ref="builtin:NotKnown",
        language="English",
        models=models,
        store=store,
        backend_identity="dashscope:m:cn-beijing:-",
        speaker_policy="canonical_or_passthrough",
    )
    assert plan.voice.speaker == "NotKnown"
    assert any("passing it through" in warning for warning in plan.warnings)


def test_dashscope_sdk_surface_when_installed():
    """§J13: assert the real SDK symbols without any network access."""
    pytest.importorskip("dashscope")
    import importlib
    import pkgutil

    import dashscope

    client = dashscope.Client(api_key="test-key")
    tts = client.audio.tts
    assert callable(tts.connect)

    module = importlib.import_module(tts.connect.__module__)
    session = getattr(module, "SynthesisSession", None)
    if session is None:
        audio_pkg = importlib.import_module("dashscope.audio")
        for info in pkgutil.walk_packages(
            audio_pkg.__path__, audio_pkg.__name__ + "."
        ):
            try:
                candidate = importlib.import_module(info.name)
            except Exception:
                continue
            session = getattr(candidate, "SynthesisSession", None)
            if session is not None:
                break
    assert session is not None, "dashscope SDK has no SynthesisSession class"
    missing = [
        name
        for name in ("send_text", "flush", "stop", "audio_chunks")
        if not callable(getattr(session, name, None))
    ]
    assert not missing, f"SynthesisSession lacks callable(s): {missing}"


def test_instance_capabilities_must_exist_and_have_the_right_type():
    spec = BackendSpec(
        name="demo",
        identity="demo",
        models=[_base_model()],
        capabilities=BackendCapabilities(
            modes=frozenset({"builtin"}),
            instruct_modes=frozenset(),
            sampling=frozenset(),
        ),
        create=lambda cfg, model: None,
        remote=False,
    )

    class Missing:
        pass

    class Wrong:
        capabilities = object()

    class Raises:
        @staticmethod
        def capabilities():
            raise RuntimeError("caps failed")

    for instance in (Missing(), Wrong(), Raises()):
        with pytest.raises(ModelError) as excinfo:
            ensure_instance_capabilities(instance, spec)
        assert excinfo.value.code == 3


def test_instance_protocol_requires_load_generate_release_and_model_info():
    spec = BackendSpec(
        name="demo",
        identity="demo",
        models=[_base_model()],
        capabilities=BackendCapabilities(
            modes=frozenset({"builtin"}),
            instruct_modes=frozenset(),
            sampling=frozenset(),
        ),
        create=lambda cfg, model: None,
        remote=False,
    )

    class Valid:
        name = "demo"
        model_info = _base_model()
        model = None
        device = None
        dtype_name = None
        load_ms = 0.0
        capabilities = spec.capabilities

        def load(self):
            return self

        def generate(self, **kwargs):
            return None

        def build_clone_prompt_items(self, *args, **kwargs):
            return None

        def release(self):
            return None

    ensure_instance_protocol(Valid(), spec)
    for instance in (
        None,
        SimpleNamespace(model_info=object(), generate=lambda **kw: None, release=lambda: None),
        SimpleNamespace(load=lambda: None, generate=lambda **kw: None, release=lambda: None),
        SimpleNamespace(
            name="demo",
            model_info=object(),
            model=None,
            device=None,
            dtype_name=None,
            load_ms=0.0,
            capabilities=spec.capabilities,
            load=lambda: None,
            generate=lambda **kw: None,
            build_clone_prompt_items=lambda *a, **k: None,
            release=lambda: None,
        ),
    ):
        with pytest.raises(ModelError) as excinfo:
            ensure_instance_protocol(instance, spec)
        assert excinfo.value.code == 3


def test_instance_capabilities_may_not_be_weaker_than_the_spec():
    model = _base_model()
    spec = BackendSpec(
        name="demo",
        identity="demo",
        models=[model],
        capabilities=BackendCapabilities(
            modes=frozenset({"builtin", "clone"}),
            instruct_modes=frozenset(),
            sampling=frozenset(),
        ),
        create=lambda cfg, model: None,
        remote=False,
    )

    class Weaker:
        capabilities = BackendCapabilities(
            modes=frozenset({"builtin"}), instruct_modes=frozenset(), sampling=frozenset()
        )

    class CallableWeaker:
        @staticmethod
        def capabilities():
            return BackendCapabilities(
                modes=frozenset({"builtin"}),
                instruct_modes=frozenset(),
                sampling=frozenset(),
            )

    with pytest.raises(ModelError) as excinfo:
        ensure_instance_capabilities(Weaker(), spec)
    assert excinfo.value.code == 3
    with pytest.raises(ModelError) as excinfo:
        ensure_instance_capabilities(CallableWeaker(), spec)
    assert excinfo.value.code == 3
