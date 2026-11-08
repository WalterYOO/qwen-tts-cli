"""DashScope realtime TTS backend (FR-P2-02, builtin voices only).

``dashscope`` is imported only inside the lazily-executed client factory, so
module import, ``--dry-run`` and ``plugins list`` stay runtime-free.  The
realtime session API was verified against ``dashscope==2.0.3``:
``client.audio.tts.connect(...)`` returns a context-managed session with
``send_text`` / ``flush`` / ``stop`` / ``audio_chunks``.
"""

from __future__ import annotations

import io
import os
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..catalog import ModelInfo
from ..errors import CapabilityError, InputError, ModelError, QttsError, UsageError
from .base import BackendCapabilities

DEFAULT_MODEL_ID = "qwen3-tts-flash-realtime"
DEFAULT_REGION = "cn-beijing"
SAMPLE_RATE = 24000

# Documented voice list for qwen3-tts-flash-realtime.  Matching is
# case-insensitive and returns this canonical spelling; unknown ids pass
# through so pre-enrolled voices keep working.
# https://help.aliyun.com/zh/model-studio/qwen-tts
DASHSCOPE_VOICES: tuple[str, ...] = (
    "Cherry",
    "Serena",
    "Ethan",
    "Chelsie",
    "Momo",
    "Vivian",
    "Moon",
    "Maia",
    "Kai",
    "Nofish",
    "Bella",
    "Jennifer",
    "Ryan",
    "Katerina",
    "Aiden",
    "Eldric Sage",
    "Mia",
    "Mochi",
    "Bellona",
    "Vincent",
    "Bunny",
    "Neil",
    "Elias",
    "Arthur",
    "Nini",
    "Seren",
    "Pip",
    "Stella",
    "Bodega",
    "Sonrisa",
    "Alek",
    "Dolce",
    "Sohee",
    "Ono Anna",
    "Lenn",
    "Emilien",
    "Andre",
    "Radio Gol",
    "Jada",
    "Dylan",
    "Li",
    "Marcus",
    "Roy",
    "Peter",
    "Sunny",
    "Eric",
    "Rocky",
    "Kiki",
)

DASHSCOPE_LANGUAGES: tuple[str, ...] = (
    "Chinese",
    "English",
    "Japanese",
    "Korean",
    "German",
    "French",
    "Russian",
    "Portuguese",
    "Spanish",
    "Italian",
)

CAPABILITIES = BackendCapabilities(
    modes=frozenset({"builtin"}),
    instruct_modes=frozenset(),
    sampling=frozenset(),
    languages=DASHSCOPE_LANGUAGES,
    remote=True,
    notes=("builtin voices only", "no local weights"),
)


def _default_client_factory(api_key: str, workspace: Optional[str], region: str):
    import dashscope

    kwargs: dict[str, Any] = {"api_key": api_key, "region": region}
    if workspace:
        kwargs["workspace"] = workspace
    return dashscope.Client(**kwargs)


# Tests replace this seam; nothing at module import touches the SDK.
_client_factory = _default_client_factory


def _decode_mp3(payload: bytes) -> tuple[np.ndarray, int]:
    """Decode the session payload and return mono contiguous float32 samples."""
    try:
        import soundfile

        data, sr = soundfile.read(
            io.BytesIO(payload), dtype="float32", always_2d=True
        )
    except Exception as exc:
        raise ModelError(
            f"dashscope audio could not be decoded: {exc}",
            details={"bytes": len(payload)},
        ) from exc
    array = np.asarray(data, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ModelError(
            "dashscope returned empty audio",
            details={"shape": list(array.shape), "bytes": len(payload)},
        )
    mono = array.mean(axis=1, dtype=np.float64).astype(np.float32)
    if not np.all(np.isfinite(mono)):
        raise ModelError("dashscope returned non-finite audio")
    if float(sr) <= 0.0 or not float(sr).is_integer():
        raise ModelError(f"dashscope returned invalid sample rate: {sr!r}")
    return np.ascontiguousarray(mono, dtype=np.float32), int(sr)


def synthetic_model(model_id: str) -> ModelInfo:
    return ModelInfo(
        path=Path(f"<dashscope:{model_id}>"),
        name=model_id,
        model_type="qwen3_tts",
        tts_model_type="custom_voice",
        kind="builtin",
        speakers=list(DASHSCOPE_VOICES),
        languages=list(DASHSCOPE_LANGUAGES),
        supports_instruct=False,
        fingerprint=f"dashscope:{model_id}",
    )


def backend_identity(model_id: str, region: Optional[str], workspace: Optional[str]) -> str:
    region = (region or DEFAULT_REGION).strip() or DEFAULT_REGION
    return f"dashscope:{model_id}:{region}:{workspace or '-'}"


class DashScopeBackend:
    name = "dashscope"
    capabilities = CAPABILITIES

    def __init__(self, cfg, model: ModelInfo) -> None:
        self.cfg = cfg
        self.model_info = model
        self.model: Any = None
        self.client: Any = None
        self.device: Optional[str] = None
        self.dtype_name: Optional[str] = None
        self.load_ms: Optional[float] = None

    def canonical_voice(self, speaker: Optional[str]) -> Optional[str]:
        if speaker is None:
            return None
        for voice in DASHSCOPE_VOICES:
            if voice.lower() == str(speaker).lower():
                return voice
        return str(speaker)

    def load(self) -> "DashScopeBackend":
        if self.client is not None:
            return self
        api_key = os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            raise ModelError(
                "DASHSCOPE_API_KEY is required for the dashscope backend",
                details={"env": "DASHSCOPE_API_KEY"},
            )
        workspace = os.environ.get("DASHSCOPE_WORKSPACE") or None
        region = os.environ.get("DASHSCOPE_REGION") or DEFAULT_REGION
        started = time.perf_counter()
        try:
            client = _client_factory(api_key, workspace, region)
        except QttsError:
            raise
        except Exception as exc:
            raise ModelError(f"cannot create dashscope client: {exc}") from exc
        self.load_ms = (time.perf_counter() - started) * 1000.0
        self.client = client
        self.model = client
        return self

    def build_clone_prompt_items(self, store, voice_name: str) -> Any:
        raise CapabilityError(
            "dashscope backend supports builtin voices only; enrolled "
            "clone/design/finetune voices must be referenced as builtin:VOICE_ID",
            details={"backend": self.name, "voice": voice_name},
        )

    def generate(
        self,
        *,
        text: str,
        language: str,
        mode: str,
        speaker: Optional[str] = None,
        instruct: Optional[str] = None,
        prompt_items: Any = None,
        sampling: Optional[dict[str, Any]] = None,
        seed: int = 1234,
    ) -> tuple[np.ndarray, int, dict[str, Any]]:
        if mode != "builtin":
            raise CapabilityError(
                "dashscope backend supports builtin voices only; enrolled "
                "clone/design/finetune voices must be referenced as builtin:VOICE_ID",
                details={"backend": self.name, "mode": mode},
            )
        if instruct:
            raise CapabilityError(
                "dashscope backend does not forward instruction",
                details={"backend": self.name},
            )
        if not speaker:
            raise InputError("dashscope generation requires a builtin speaker")
        self.load()
        voice = self.canonical_voice(speaker)
        started = time.perf_counter()
        try:
            with self.client.audio.tts.connect(
                model=self.model_info.name,
                voice=voice,
                format="mp3",
                sample_rate=SAMPLE_RATE,
            ) as session:
                session.send_text(text)
                session.flush()
                session.stop()
                chunks = list(session.audio_chunks())
        except QttsError:
            raise
        except Exception as exc:
            raise ModelError(f"dashscope generation failed: {exc}") from exc
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        parts = []
        for index, chunk in enumerate(chunks):
            if not isinstance(chunk, (bytes, bytearray)):
                raise ModelError(
                    "dashscope returned a non-bytes audio chunk",
                    details={"index": index, "type": type(chunk).__name__},
                )
            parts.append(bytes(chunk))
        payload = b"".join(parts)
        if not payload:
            raise ModelError("dashscope returned an empty audio payload")
        audio, sr = _decode_mp3(payload)
        info = {
            "generate_ms": round(elapsed_ms, 1),
            "sample_rate": sr,
            "duration_s": round(len(audio) / float(sr), 3),
            "mode": mode,
            "language": language,
            "backend": self.name,
        }
        return audio, sr, info

    def release(self) -> None:
        self.client = None
        self.model = None


def preflight_voice_ref(raw_ref: str) -> None:
    from ..capabilities import parse_voice_ref

    parsed = parse_voice_ref(raw_ref)
    if parsed.kind != "builtin":
        raise CapabilityError(
            "dashscope backend supports builtin voices only; enrolled "
            "clone/design/finetune voices must be referenced as builtin:VOICE_ID",
            details={"backend": "dashscope", "voice": raw_ref},
        )


def preflight(plan, cfg, args) -> None:
    """Metadata-only checks; ``--dry-run`` never requires the API key."""
    voices = [plan.voice, *[seg.voice for seg in plan.segments if seg.voice is not None]]
    seen: set[int] = set()
    for voice in voices:
        if id(voice) in seen:
            continue
        seen.add(id(voice))
        if voice.mode != "builtin":
            raise CapabilityError(
                "dashscope backend supports builtin voices only; enrolled "
                "clone/design/finetune voices must be referenced as builtin:VOICE_ID",
                details={"backend": "dashscope", "mode": voice.mode},
            )
    has_instruct = bool(plan.instruct) or any(
        bool(seg.instruct) for seg in plan.segments
    )
    if has_instruct:
        raise CapabilityError(
            "dashscope backend does not forward instruction",
            details={"backend": "dashscope"},
        )
    if getattr(cfg, "offline", False):
        raise UsageError(
            "dashscope backend cannot be used with --offline",
            details={"backend": "dashscope"},
        )
    if not getattr(args, "dry_run", False) and not os.environ.get("DASHSCOPE_API_KEY"):
        raise ModelError(
            "DASHSCOPE_API_KEY is required for the dashscope backend",
            details={"env": "DASHSCOPE_API_KEY"},
        )
