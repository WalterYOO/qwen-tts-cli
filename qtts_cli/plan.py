"""GenerationPlan IR: compile text + voice + options before any model loads (PRD §8.2)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from . import __version__
from .budget import DEFAULT_MAX_NEW_TOKENS, codec_rate, estimate_duration_s, estimate_tokens, max_chars
from .cache import AudioCache, cache_key
from .capabilities import VoiceRef, parse_voice_ref, validate_capabilities
from .catalog import ModelInfo, find_model, require_speaker, select
from .errors import InputError
from .segmenter import segment_text, split_long
from .textnorm import NORMALIZATION_VERSION, detect_language, normalize_text
from .voice_store import VoiceStore

DEFAULT_PAUSE_MS = 300

SAMPLING_DEFAULTS: dict[str, Any] = {
    "do_sample": True,
    "top_k": 50,
    "top_p": 1.0,
    "temperature": 0.9,
    "repetition_penalty": 1.05,
}


def resolve_dtype_name(cfg) -> str:
    if cfg.dtype != "auto":
        return cfg.dtype
    device = cfg.device
    if device == "auto":
        # Match Config.resolve_device without importing torch.
        import platform

        if platform.system() == "Darwin" and platform.machine() == "arm64":
            return "bf16"
        return "fp32"
    return "bf16" if device.startswith("cuda") or device == "mps" else "fp32"


@dataclass
class VoiceResolution:
    ref: str
    mode: str  # builtin | clone | design
    model: ModelInfo
    speaker: Optional[str] = None
    voice_name: Optional[str] = None
    description: Optional[str] = None
    ref_text: Optional[str] = None
    icl: bool = False
    prompt_sha256: Optional[str] = None
    ref_audio_sha256: Optional[str] = None
    identity: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ref": self.ref,
            "mode": self.mode,
            "model": self.model.model_id,
            "speaker": self.speaker,
            "voice_name": self.voice_name,
            "description": self.description,
            "icl": self.icl,
            "prompt_sha256": self.prompt_sha256,
            "ref_audio_sha256": self.ref_audio_sha256,
        }


@dataclass
class SegmentPlan:
    id: str
    text: str
    language: str
    instruct: Optional[str]
    pause_after_ms: int
    estimated_tokens: int
    estimated_duration_s: float
    cache_key: str
    source_index: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "language": self.language,
            "instruct": self.instruct,
            "pause_after_ms": self.pause_after_ms,
            "estimated_tokens": self.estimated_tokens,
            "estimated_duration_s": round(self.estimated_duration_s, 3),
            "cache_key": self.cache_key,
        }


@dataclass
class GenerationPlan:
    text: str
    normalized_text: str
    voice: VoiceResolution
    segments: list[SegmentPlan]
    sampling: dict[str, Any]
    seed: int
    language: Optional[str]
    instruct: Optional[str]
    output: Optional[str]
    format: str
    codec_rate: float
    warnings: list[str] = field(default_factory=list)
    subtitle_formats: list[str] = field(default_factory=list)
    granularity: str = "char"
    cache_hit_rate: float = 0.0
    cache_stats: dict[str, Any] = field(default_factory=dict)
    dtype_name: str = "auto"
    device_name: Optional[str] = None
    normalization_version: int = NORMALIZATION_VERSION

    @property
    def estimated_duration_s(self) -> float:
        pauses = sum(s.pause_after_ms for s in self.segments[:-1]) / 1000.0
        return sum(s.estimated_duration_s for s in self.segments) + pauses

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "qtts_version": __version__,
            "text_chars": len(self.normalized_text),
            "voice": self.voice.to_dict(),
            "language": self.language,
            "instruct": self.instruct,
            "seed": self.seed,
            "sampling": self.sampling,
            "codec_rate": round(self.codec_rate, 4),
            "estimated_duration_s": round(self.estimated_duration_s, 3),
            "segments": [s.to_dict() for s in self.segments],
            "output": {"path": self.output, "format": self.format},
            "subtitle": {"formats": self.subtitle_formats, "granularity": self.granularity},
            "cache": {"hit_rate": round(self.cache_hit_rate, 4), **self.cache_stats},
            "warnings": self.warnings,
        }


def resolve_voice(
    models: Iterable[ModelInfo],
    store: VoiceStore,
    voice_ref: VoiceRef,
    *,
    model_override: Optional[str] = None,
) -> VoiceResolution:
    model_list = list(models)
    override = find_model(model_list, model_override) if model_override else None

    if voice_ref.kind == "builtin":
        model = override or select(model_list, "builtin")
        speaker = require_speaker(model, voice_ref.value)
        return VoiceResolution(
            ref=voice_ref.raw,
            mode="builtin",
            model=model,
            speaker=speaker,
            identity={"type": "builtin", "speaker": speaker},
        )

    if voice_ref.kind == "design":
        model = override or select(model_list, "design")
        return VoiceResolution(
            ref=voice_ref.raw,
            mode="design",
            model=model,
            description=voice_ref.value,
            identity={"type": "design", "description": voice_ref.value},
        )

    # voice:<name>
    meta = store.require(voice_ref.value)
    kind = str(meta.get("kind"))
    if kind == "clone":
        model = override or select(model_list, "clone")
        source = meta.get("source") or {}
        ref_text = source.get("ref_text")
        icl = str(source.get("mode") or "icl").lower() == "icl"
        prompt_sha = store.prompt_sha256(voice_ref.value)
        # Fail before loading Base weights if the frozen prompt is missing
        # (codex P0-9).
        if prompt_sha is None:
            raise InputError(
                f"voice {voice_ref.value} has no prompt.safetensors; re-add the voice "
                "or run voice add --materialize-clone",
                details={"voice": voice_ref.value, "path": str(store.dir_for(voice_ref.value))},
            )
        return VoiceResolution(
            ref=voice_ref.raw,
            mode="clone",
            model=model,
            voice_name=voice_ref.value,
            ref_text=ref_text,
            icl=icl,
            prompt_sha256=prompt_sha,
            ref_audio_sha256=source.get("ref_audio_sha256"),
            identity={
                "type": "voice",
                "name": voice_ref.value,
                "prompt_sha256": prompt_sha,
                "mode": "icl" if icl else "xvec",
            },
        )
    if kind == "design":
        model = override or select(model_list, "design")
        source = meta.get("source") or {}
        description = source.get("description")
        if not description:
            raise InputError(f"voice {voice_ref.value} has no design description")
        return VoiceResolution(
            ref=voice_ref.raw,
            mode="design",
            model=model,
            voice_name=voice_ref.value,
            description=description,
            identity={"type": "design", "voice": voice_ref.value, "description": description},
        )
    raise InputError(f"voice {voice_ref.value} has unsupported kind: {kind!r}")


def build_plan(
    cfg,
    *,
    text: str,
    voice_ref: str,
    language: str = "auto",
    instruct: Optional[str] = None,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    pause_ms: int = DEFAULT_PAUSE_MS,
    output: Optional[str] = None,
    fmt: Optional[str] = None,
    subtitle_formats: Optional[list[str]] = None,
    granularity: str = "char",
    models: Iterable[ModelInfo],
    store: VoiceStore,
    model_override: Optional[str] = None,
) -> GenerationPlan:
    if max_new_tokens <= 0:
        raise InputError("--max-new-tokens must be a positive integer")
    if pause_ms < 0:
        raise InputError("--pause-ms must be >= 0")

    parsed = parse_voice_ref(voice_ref)
    voice = resolve_voice(models, store, parsed, model_override=model_override)

    effective_instruct = instruct
    if voice.mode == "design" and not effective_instruct:
        effective_instruct = voice.description

    # Capability gate BEFORE any model load (PRD §4.2).
    validate_capabilities(
        voice.model,
        mode=voice.mode,
        instruct=effective_instruct,
        language=None if language.lower() == "auto" else language,
        icl=voice.icl,
        ref_text=voice.ref_text,
        description=voice.description,
    )

    normalized = normalize_text(text, language)
    if not normalized.strip():
        raise InputError("input text is empty after normalization")

    rate = codec_rate(voice.model)
    dtype_name = resolve_dtype_name(cfg)
    fingerprint = voice.model.fingerprint or None
    sampling = dict(SAMPLING_DEFAULTS)
    sampling["max_new_tokens"] = int(max_new_tokens)

    warnings: list[str] = []
    raw_segments = segment_text(normalized)
    expanded: list[tuple[str, int]] = []
    for idx, seg in enumerate(raw_segments):
        seg_language = detect_language(seg) if language.lower() == "auto" else language
        if seg_language == "Auto":
            seg_language = "Chinese" if any("一" <= ch <= "鿿" for ch in seg) else "English"
        limit = max_chars(seg_language, max_new_tokens, rate)
        pieces = split_long(seg, limit) if len(seg) > limit else [seg]
        if len(pieces) > 1:
            warnings.append(
                f"segment {idx + 1} exceeded the token budget and was split into {len(pieces)} pieces"
            )
        for piece in pieces:
            expanded.append((piece, idx))

    segments: list[SegmentPlan] = []
    for i, (seg_text, source_index) in enumerate(expanded, 1):
        seg_language = detect_language(seg_text) if language.lower() == "auto" else language
        if seg_language == "Auto":
            seg_language = "Chinese" if any("一" <= ch <= "鿿" for ch in seg_text) else "English"
        # Validate the *resolved* per-segment language, including under
        # --language auto (codex P0-9).
        validate_capabilities(
            voice.model,
            mode=voice.mode,
            language=seg_language,
            description=voice.description,
            icl=voice.icl,
            ref_text=voice.ref_text,
        )
        is_last = i == len(expanded)
        doc_instruct = effective_instruct if voice.mode in ("builtin", "design") else None
        if voice.mode == "clone":
            doc_instruct = None  # Base ignores instruct by design
        key = cache_key(
            text_normalized=seg_text,
            voice_identity=voice.identity,
            model_id=voice.model.model_id,
            model_revision=voice.model.revision,
            dtype=dtype_name,
            language=seg_language,
            instruct=doc_instruct,
            sampling=sampling,
            seed=int(cfg.seed),
            model_fingerprint=fingerprint,
        )
        segments.append(
            SegmentPlan(
                id=f"s{i:03d}",
                text=seg_text,
                language=seg_language,
                instruct=doc_instruct,
                pause_after_ms=0 if is_last else int(pause_ms),
                estimated_tokens=estimate_tokens(seg_text, seg_language, rate),
                estimated_duration_s=estimate_duration_s(seg_text, seg_language),
                cache_key=key,
                source_index=source_index,
            )
        )

    return GenerationPlan(
        text=text,
        normalized_text=normalized,
        voice=voice,
        segments=segments,
        sampling=sampling,
        seed=int(cfg.seed),
        language=language,
        instruct=effective_instruct,
        output=output,
        format=fmt or "wav",
        codec_rate=rate,
        warnings=warnings,
        subtitle_formats=list(subtitle_formats or []),
        granularity=granularity,
        dtype_name=dtype_name,
        normalization_version=NORMALIZATION_VERSION,
    )


def rekey_plan(plan: GenerationPlan, *, dtype: str, device: Optional[str] = None) -> GenerationPlan:
    """Recompute cache keys once the runtime dtype is known (codex P1-5)."""
    plan.device_name = device
    if dtype == plan.dtype_name:
        return plan
    plan.dtype_name = dtype
    for seg in plan.segments:
        seg.cache_key = cache_key(
            text_normalized=seg.text,
            voice_identity=plan.voice.identity,
            model_id=plan.voice.model.model_id,
            model_revision=plan.voice.model.revision,
            model_fingerprint=plan.voice.model.fingerprint or None,
            dtype=dtype,
            language=seg.language,
            instruct=seg.instruct,
            sampling=plan.sampling,
            seed=plan.seed,
        )
    return plan


def attach_cache_stats(plan: GenerationPlan, cache: AudioCache, fmt: str = "wav") -> GenerationPlan:
    stats = cache.probe([s.cache_key for s in plan.segments], fmt)
    plan.cache_hit_rate = stats.hit_rate
    plan.cache_stats = stats.to_dict()
    return plan
