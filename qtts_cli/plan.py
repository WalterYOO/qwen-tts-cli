"""GenerationPlan IR: compile text + voice + options before any model loads (PRD §8.2)."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from . import __version__
from .budget import DEFAULT_MAX_NEW_TOKENS, codec_rate, estimate_duration_s, estimate_tokens, max_chars
from .cache import AudioCache, cache_key
from .capabilities import VoiceRef, parse_voice_ref, validate_capabilities
from .catalog import ModelInfo, find_model, require_speaker, select
from .errors import InputError, UsageError
from .lexicon import (
    Lexicon,
    MapBlock,
    parse_director_markers,
    project_text,
    protected_ranges,
    slice_prepared,
)
from .segmenter import segment_text, split_long
from .textnorm import NORMALIZATION_VERSION, detect_language
from .utils import sha256_text
from .voice_store import VoiceStore

DEFAULT_PAUSE_MS = 300
MAX_SEED = 2**31 - 1

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
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ref": self.ref,
            "mode": self.mode,
            "model": self.model.model_id,
            "model_revision": self.model.revision,
            "model_fingerprint": self.model.fingerprint,
            "speaker": self.speaker,
            "voice_name": self.voice_name,
            "description": self.description,
            "ref_text": self.ref_text,
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
    source_segment_id: Optional[str] = None
    voice: Optional["VoiceResolution"] = None
    display_text: Optional[str] = None
    spoken_to_display: Optional[list[MapBlock]] = None
    role: Optional[str] = None
    seed: Optional[int] = None
    max_new_tokens: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "id": self.id,
            "text": self.text,
            "display_text": self.display_text if self.display_text is not None else self.text,
            "language": self.language,
            "instruct": self.instruct,
            "pause_after_ms": self.pause_after_ms,
            "estimated_tokens": self.estimated_tokens,
            "estimated_duration_s": round(self.estimated_duration_s, 3),
            "cache_key": self.cache_key,
        }
        if self.role is not None:
            payload["role"] = self.role
        if self.source_segment_id is not None:
            payload["source_segment_id"] = self.source_segment_id
        if self.seed is not None:
            payload["seed"] = self.seed
        if self.max_new_tokens is not None:
            payload["max_new_tokens"] = self.max_new_tokens
        if self.voice is not None:
            payload["voice"] = self.voice.to_dict()
            payload["model"] = self.voice.model.model_id
        if self.spoken_to_display is not None:
            payload["spoken_to_display"] = [block.to_dict() for block in self.spoken_to_display]
        return payload


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
    granularity: str = "sentence"
    cache_hit_rate: float = 0.0
    cache_stats: dict[str, Any] = field(default_factory=dict)
    dtype_name: str = "auto"
    device_name: Optional[str] = None
    normalization_version: int = NORMALIZATION_VERSION
    display_text: Optional[str] = None
    lexicon_sha256: Optional[str] = None
    script_source: Optional[str] = None
    leading_pause_ms: int = 0
    trailing_pause_ms: int = 0

    @property
    def estimated_duration_s(self) -> float:
        pauses = sum(s.pause_after_ms for s in self.segments[:-1]) / 1000.0
        return (
            sum(s.estimated_duration_s for s in self.segments)
            + pauses
            + (self.leading_pause_ms + self.trailing_pause_ms) / 1000.0
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "qtts_version": __version__,
            "text_chars": len(self.normalized_text),
            "display_text": self.display_text if self.display_text is not None else self.normalized_text,
            "lexicon_sha256": self.lexicon_sha256,
            "script_source": self.script_source,
            "voice": self.voice.to_dict(),
            "language": self.language,
            "instruct": self.instruct,
            "seed": self.seed,
            "sampling": self.sampling,
            "codec_rate": round(self.codec_rate, 4),
            "estimated_duration_s": round(self.estimated_duration_s, 3),
            "leading_pause_ms": self.leading_pause_ms,
            "trailing_pause_ms": self.trailing_pause_ms,
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
    backend_identity: Optional[str] = None,
    require_clone_prompt: bool = True,
    speaker_policy: str = "strict",
) -> VoiceResolution:
    if speaker_policy not in ("strict", "canonical_or_passthrough"):
        raise UsageError(f"unknown speaker policy: {speaker_policy!r}")
    model_list = list(models)
    override = find_model(model_list, model_override) if model_override else None

    if voice_ref.kind == "builtin":
        model = override or select(model_list, "builtin")
        warnings: list[str] = []
        if speaker_policy == "strict":
            speaker = require_speaker(model, voice_ref.value)
        else:
            speaker = model.canonical_speaker(voice_ref.value)
            if speaker is None:
                speaker = voice_ref.value
                warnings.append(
                    f"backend voice {speaker!r} is not in the known list; "
                    "passing it through"
                )
        identity: dict[str, Any] = {"type": "builtin", "speaker": speaker}
        if backend_identity:
            identity["backend"] = backend_identity
        return VoiceResolution(
            ref=voice_ref.raw,
            mode="builtin",
            model=model,
            speaker=speaker,
            identity=identity,
            warnings=warnings,
        )

    if voice_ref.kind == "design":
        model = override or select(model_list, "design")
        identity = {"type": "design", "description": voice_ref.value}
        if backend_identity:
            identity["backend"] = backend_identity
        return VoiceResolution(
            ref=voice_ref.raw,
            mode="design",
            model=model,
            description=voice_ref.value,
            identity=identity,
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
        # (codex P0-9).  Backends that only need ref.wav (vllm-omni) opt out.
        if prompt_sha is None and require_clone_prompt:
            raise InputError(
                f"voice {voice_ref.value} has no prompt.safetensors; re-add the voice "
                "or run voice add --materialize-clone",
                details={"voice": voice_ref.value, "path": str(store.dir_for(voice_ref.value))},
            )
        identity: dict[str, Any] = {
            "type": "voice",
            "name": voice_ref.value,
            "prompt_sha256": prompt_sha,
            "mode": "icl" if icl else "xvec",
        }
        # ICL prompt items embed ref_text; editing it must invalidate cached
        # audio even when prompt.safetensors is unchanged.
        if icl:
            identity["ref_text_sha256"] = sha256_text(ref_text or "")
        if backend_identity:
            identity["backend"] = backend_identity
        return VoiceResolution(
            ref=voice_ref.raw,
            mode="clone",
            model=model,
            voice_name=voice_ref.value,
            ref_text=ref_text,
            icl=icl,
            prompt_sha256=prompt_sha,
            ref_audio_sha256=source.get("ref_audio_sha256"),
            identity=identity,
        )
    if kind == "design":
        model = override or select(model_list, "design")
        source = meta.get("source") or {}
        description = source.get("description")
        if not description:
            raise InputError(f"voice {voice_ref.value} has no design description")
        resolved_identity = {
            "type": "design",
            "voice": voice_ref.value,
            "description": description,
        }
        if backend_identity:
            resolved_identity["backend"] = backend_identity
        return VoiceResolution(
            ref=voice_ref.raw,
            mode="design",
            model=model,
            voice_name=voice_ref.value,
            description=description,
            identity=resolved_identity,
        )
    raise InputError(f"voice {voice_ref.value} has unsupported kind: {kind!r}")


def _detect_segment_language(text: str, language: str) -> str:
    seg_language = detect_language(text) if language.lower() == "auto" else language
    if seg_language == "Auto":
        seg_language = "Chinese" if any("一" <= ch <= "鿿" for ch in text) else "English"
    return seg_language


def _split_ranges(
    text: str, limit: int, protected: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    """Split display text by budget without cutting through a lexicon term."""
    if len(text) <= limit or limit <= 0:
        return [(0, len(text))]

    def adjust(cut: int, start: int) -> int:
        for term_start, term_end in protected:
            if term_start < cut < term_end:
                return term_start if term_start > start else term_end
        return cut

    ranges: list[tuple[int, int]] = []
    cursor = 0
    while len(text) - cursor > limit:
        candidate = split_long(text[cursor:], limit)[0]
        cut = adjust(cursor + len(candidate), cursor)
        if cut <= cursor:
            cut = min(cursor + limit, len(text))
        ranges.append((cursor, cut))
        cursor = cut
    if cursor < len(text):
        ranges.append((cursor, len(text)))
    return ranges


def _split_piece(
    prepared,
    *,
    language: str,
    max_new_tokens: int,
    rate: float,
    warnings: list[str],
) -> list[dict[str, Any]]:
    """Split one sentence projection into budget-sized pieces."""
    display_text = prepared.display_text
    if not display_text:
        return []
    display_protected = protected_ranges(prepared.mapping)
    limit = max_chars(
        _detect_segment_language(prepared.spoken_text or display_text, language),
        max_new_tokens,
        rate,
    )
    if len(prepared.spoken_text) <= limit:
        return [
            {
                "display_text": display_text,
                "spoken_text": prepared.spoken_text,
                "mapping": prepared.mapping,
                "language": _detect_segment_language(
                    prepared.spoken_text or display_text, language
                ),
                "pause_after_ms": 0,
                "piece_index": 0,
                "piece_count": 1,
            }
        ]
    ratio = limit / max(1, len(prepared.spoken_text))
    display_limit = max(1, int(len(display_text) * ratio))
    ranges = _split_ranges(display_text, display_limit, display_protected)
    if len(ranges) <= 1:
        warnings.append(
            "a segment exceeded the token budget but cannot be split without "
            "cutting a lexicon term"
        )
        ranges = [(0, len(display_text))]
    else:
        warnings.append(
            "a segment exceeded the token budget and was split into "
            f"{len(ranges)} pieces"
        )
    pieces: list[dict[str, Any]] = []
    for index, (start, end) in enumerate(ranges):
        item = slice_prepared(prepared, start, end)
        pieces.append(
            {
                "display_text": item.display_text,
                "spoken_text": item.spoken_text,
                "mapping": item.mapping,
                "language": _detect_segment_language(
                    item.spoken_text or item.display_text, language
                ),
                "pause_after_ms": 0,
                "piece_index": index,
                "piece_count": len(ranges),
            }
        )
    return pieces


def prepare_text_pieces(
    text: str,
    *,
    language: str,
    rate: float,
    max_new_tokens: int,
    pause_ms: int = DEFAULT_PAUSE_MS,
    lexicon: Optional[Lexicon] = None,
) -> tuple[str, list[dict[str, Any]], list[str], int, int]:
    """Project raw text, split by sentence/budget, and resolve pause markers.

    Order is raw -> markers -> lexicon display projection -> spoken projection
    -> sentences/budget.  Returns ``(display_text, pieces, warnings,
    leading_pause_ms, trailing_pause_ms)``.
    """
    clean, markers = parse_director_markers(text)
    tokens: list[tuple[str, Optional[int]]] = []
    cursor = 0
    for marker in markers:
        offset = max(cursor, min(int(marker["offset"]), len(clean)))
        tokens.append((clean[cursor:offset], None))
        tokens.append(("", int(marker["ms"])))
        cursor = offset
    tokens.append((clean[cursor:], None))

    warnings: list[str] = []
    pieces: list[dict[str, Any]] = []
    pending_pause = 0
    leading_pause = 0
    trailing_pause = 0
    sentence_ordinal = 0
    for chunk_text, marker_pause in tokens:
        if marker_pause is not None:
            pending_pause += marker_pause
            continue
        if not chunk_text.strip():
            continue
        prepared = project_text(chunk_text, language=language, lexicon=lexicon)
        chunk_pieces: list[dict[str, Any]] = []
        for sentence in segment_text(prepared.display_text):
            if not sentence:
                continue
            sentence_start = sum(len(p["display_text"]) for p in chunk_pieces)
            sentence_projection = slice_prepared(
                prepared, sentence_start, sentence_start + len(sentence)
            )
            split = _split_piece(
                sentence_projection,
                language=language,
                max_new_tokens=max_new_tokens,
                rate=rate,
                warnings=warnings,
            )
            for piece in split:
                piece["source_ordinal"] = sentence_ordinal
                chunk_pieces.append(piece)
            sentence_ordinal += 1
        if not chunk_pieces:
            continue
        if not pieces:
            leading_pause += pending_pause
        else:
            pieces[-1]["pause_after_ms"] = pending_pause
        pending_pause = 0
        for index, piece in enumerate(chunk_pieces):
            piece["pause_after_ms"] = pause_ms if index < len(chunk_pieces) - 1 else 0
        pieces.extend(chunk_pieces)
    if pieces:
        trailing_pause = pending_pause
        pieces[-1]["pause_after_ms"] = 0
        for piece in pieces:
            piece["source_ordinal"] = piece.get("source_ordinal", 0)
    display_text = "".join(piece["display_text"] for piece in pieces)
    return display_text, pieces, warnings, leading_pause, trailing_pause


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
    granularity: str = "sentence",
    models: Iterable[ModelInfo],
    store: VoiceStore,
    model_override: Optional[str] = None,
    lexicon: Optional[Lexicon] = None,
    backend_identity: Optional[str] = None,
    require_clone_prompt: bool = True,
    speaker_policy: str = "strict",
) -> GenerationPlan:
    if max_new_tokens <= 0:
        raise InputError("--max-new-tokens must be a positive integer")
    if pause_ms < 0:
        raise InputError("--pause-ms must be >= 0")

    parsed = parse_voice_ref(voice_ref)
    voice = resolve_voice(
        models,
        store,
        parsed,
        model_override=model_override,
        backend_identity=backend_identity,
        require_clone_prompt=require_clone_prompt,
        speaker_policy=speaker_policy,
    )

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

    if not text.strip():
        raise InputError("input text is empty")

    rate = codec_rate(voice.model)
    dtype_name = resolve_dtype_name(cfg)
    fingerprint = voice.model.fingerprint or None
    sampling = dict(SAMPLING_DEFAULTS)
    sampling["max_new_tokens"] = int(max_new_tokens)

    display_text, pieces, piece_warnings, leading_pause, trailing_pause = prepare_text_pieces(
        text,
        language=language,
        rate=rate,
        max_new_tokens=max_new_tokens,
        pause_ms=pause_ms,
        lexicon=lexicon,
    )
    warnings = list(voice.warnings) + list(piece_warnings)
    if not display_text.strip():
        raise InputError("input text is empty after normalization/director markers")

    segments: list[SegmentPlan] = []
    sentence_counts: dict[int, int] = {}
    for piece in pieces:
        sentence_counts[piece["source_ordinal"]] = sentence_counts.get(
            piece["source_ordinal"], 0
        ) + 1
    sentence_ids = {
        ordinal: f"s{index:03d}"
        for index, ordinal in enumerate(sorted(sentence_counts), 1)
    }
    for i, piece in enumerate(pieces):
        seg_language = piece["language"]
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
        doc_instruct = effective_instruct if voice.mode in ("builtin", "design") else None
        if voice.mode == "clone":
            doc_instruct = None  # Base ignores instruct by design
        key = cache_key(
            text_normalized=piece["spoken_text"],
            voice_identity=voice.identity,
            model_id=voice.model.model_id,
            model_revision=voice.model.revision,
            dtype=dtype_name,
            language=seg_language,
            instruct=doc_instruct,
            sampling=sampling,
            seed=int(cfg.seed),
            model_fingerprint=fingerprint,
            lexicon_sha256=lexicon.sha256 if lexicon else None,
        )
        base_id = sentence_ids[piece["source_ordinal"]]
        if piece["piece_count"] > 1:
            segment_id = f"{base_id}.p{piece['piece_index'] + 1}"
        else:
            segment_id = base_id
        segments.append(
            SegmentPlan(
                id=segment_id,
                text=piece["spoken_text"],
                language=seg_language,
                instruct=doc_instruct,
                pause_after_ms=piece["pause_after_ms"],
                estimated_tokens=estimate_tokens(piece["spoken_text"], seg_language, rate),
                estimated_duration_s=estimate_duration_s(piece["spoken_text"], seg_language),
                cache_key=key,
                source_index=i,
                source_segment_id=base_id,
                voice=voice,
                display_text=piece["display_text"],
                spoken_to_display=piece["mapping"],
                seed=int(cfg.seed),
                max_new_tokens=int(max_new_tokens),
            )
        )

    return GenerationPlan(
        text=text,
        normalized_text=display_text,
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
        display_text=display_text,
        lexicon_sha256=lexicon.sha256 if lexicon else None,
        leading_pause_ms=leading_pause,
        trailing_pause_ms=trailing_pause,
    )


def rekey_plan(plan: GenerationPlan, *, dtype: str, device: Optional[str] = None) -> GenerationPlan:
    """Recompute cache keys once the runtime dtype is known (codex P1-5)."""
    plan.device_name = device
    if dtype == plan.dtype_name:
        return plan
    plan.dtype_name = dtype
    for seg in plan.segments:
        voice = seg.voice or plan.voice
        seg.cache_key = cache_key(
            text_normalized=seg.text,
            voice_identity=voice.identity,
            model_id=voice.model.model_id,
            model_revision=voice.model.revision,
            model_fingerprint=voice.model.fingerprint or None,
            dtype=dtype,
            language=seg.language,
            instruct=seg.instruct,
            sampling=(
                {**plan.sampling, "max_new_tokens": seg.max_new_tokens}
                if seg.max_new_tokens is not None
                else plan.sampling
            ),
            seed=int(seg.seed if seg.seed is not None else plan.seed),
            lexicon_sha256=plan.lexicon_sha256,
        )
    return plan


def validate_seed(seed: Any, *, label: str = "seed") -> int:
    """Validate one effective seed without wrapping or truncating it (FR-P1-04)."""
    try:
        value = int(seed)
    except (TypeError, ValueError) as exc:
        raise UsageError(f"{label} must be an integer") from exc
    if isinstance(seed, bool) or value != seed:
        raise UsageError(f"{label} must be an integer")
    if value < 0 or value > MAX_SEED:
        raise UsageError(
            f"{label} {value} is outside the supported range [0, {MAX_SEED}]",
            details={"seed": value, "max": MAX_SEED},
        )
    return value


def reseed_plan(plan: GenerationPlan, take_index: int) -> GenerationPlan:
    """Return an independent copy of ``plan`` for one take.

    Take 1 is an equivalent copy that preserves every parsed seed (including
    per-segment script seeds). Take ``t`` adds ``t - 1`` to the plan seed and
    to every segment's effective seed, then rekeys all segment cache keys.
    The base plan is never mutated and cache keys never contain the job id or
    take index (FR-P1-04).
    """
    if isinstance(take_index, bool) or not isinstance(take_index, int) or take_index < 1:
        raise UsageError("take_index must be a positive integer")
    delta = take_index - 1
    clone = copy.deepcopy(plan)
    clone.seed = validate_seed(int(plan.seed) + delta, label="take seed")
    for index, seg in enumerate(clone.segments):
        base = int(plan.segments[index].seed if plan.segments[index].seed is not None else plan.seed)
        effective = validate_seed(base + delta, label=f"segment {seg.id} seed")
        seg.seed = effective
        voice = seg.voice or clone.voice
        seg.cache_key = cache_key(
            text_normalized=seg.text,
            voice_identity=voice.identity,
            model_id=voice.model.model_id,
            model_revision=voice.model.revision,
            model_fingerprint=voice.model.fingerprint or None,
            dtype=clone.dtype_name,
            language=seg.language,
            instruct=seg.instruct,
            sampling=(
                {**clone.sampling, "max_new_tokens": seg.max_new_tokens}
                if seg.max_new_tokens is not None
                else clone.sampling
            ),
            seed=effective,
            lexicon_sha256=clone.lexicon_sha256,
        )
    return clone


def attach_cache_stats(plan: GenerationPlan, cache: AudioCache, fmt: str = "wav") -> GenerationPlan:
    stats = cache.probe([s.cache_key for s in plan.segments], fmt)
    plan.cache_hit_rate = stats.hit_rate
    plan.cache_stats = stats.to_dict()
    return plan
