"""Multi-role script compiler (FR-P1-02).

A script is validated and compiled into the same :class:`GenerationPlan` used by
single-voice ``say``. Every segment carries its own resolved voice/model, so
execution can switch models without losing the plan-level default voice.

Validation is deliberately two-phase: all syntax/schema checks run before the
model directory is scanned, so a bad script exits 2 even when no models exist.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Optional

try:  # PyYAML is a declared dependency; a broken env must not traceback (exit 3).
    import yaml
except ImportError:  # pragma: no cover - exercised by the dependency test
    yaml = None  # type: ignore[assignment]

from .budget import DEFAULT_MAX_NEW_TOKENS, codec_rate, estimate_duration_s, estimate_tokens
from .cache import cache_key
from .capabilities import CapabilityError, parse_voice_ref, validate_capabilities
from .catalog import ModelInfo, scan
from .errors import InputError, ModelError, UsageError
from .lexicon import Lexicon, parse_director_markers
from .plan import (
    DEFAULT_PAUSE_MS,
    SAMPLING_DEFAULTS,
    GenerationPlan,
    SegmentPlan,
    VoiceResolution,
    prepare_text_pieces,
    resolve_dtype_name,
    resolve_voice,
)
from .voice_store import VoiceStore

SCRIPT_SCHEMA_VERSION = 1
MAX_SCRIPT_BYTES = 2_000_000
MAX_SCRIPT_NODES = 50_000
_TOP_KEYS = {"schema_version", "defaults", "segments"}
_DEFAULT_KEYS = {
    "voice", "language", "format", "seed", "max_new_tokens", "pause_ms", "instruct",
}
_SEGMENT_KEYS = {
    "id", "text", "role", "voice", "language", "instruct", "pause_after_ms",
    "seed", "max_new_tokens",
}


class DuplicateJSONKey(ValueError):
    """Raised by :func:`json_loads_strict` for duplicate object keys."""


def json_loads_strict(raw: str) -> Any:
    """``json.loads`` that rejects duplicate keys instead of keeping the last."""

    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in pairs:
            if key in out:
                raise DuplicateJSONKey(f"duplicate key {key!r}")
            out[key] = value
        return out

    return json.loads(raw, object_pairs_hook=pairs_hook)


def _json_load_strict(raw: str, file: Path) -> Any:
    try:
        return json_loads_strict(raw)
    except DuplicateJSONKey as exc:
        raise UsageError(f"invalid JSON in {file}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise UsageError(
            f"invalid JSON in {file}: line {exc.lineno} column {exc.colno}: {exc.msg}"
        ) from exc


def _yaml_load_strict(raw: str, file: Path) -> Any:
    if yaml is None:
        raise ModelError(
            "PyYAML is required for YAML scripts; install qtts with its dependencies "
            "or use a .json script",
            details={"dependency": "PyYAML"},
        )

    class StrictLoader(yaml.SafeLoader):  # type: ignore[misc,name-defined]
        def construct_mapping(self, node, deep=False):  # type: ignore[no-untyped-def]
            self.flatten_mapping(node)
            mapping: dict[Any, Any] = {}
            for key_node, value_node in node.value:
                key = self.construct_object(key_node, deep=deep)
                try:
                    duplicate = key in mapping
                except TypeError as exc:
                    raise UsageError(
                        f"invalid YAML in {file}: unhashable mapping key"
                    ) from exc
                if duplicate:
                    raise UsageError(
                        f"invalid YAML in {file}: duplicate mapping key {key!r}"
                    )
                mapping[key] = self.construct_object(value_node, deep=deep)
            return mapping

    try:
        event_count = 0
        for event in yaml.parse(raw):
            event_count += 1
            if event_count > MAX_SCRIPT_NODES:
                raise UsageError(f"script {file} has too many YAML nodes")
            # AliasEvent carries an anchor too; reject every anchor outright.
            if getattr(event, "anchor", None) is not None:
                raise UsageError(
                    f"invalid YAML in {file}: aliases/anchors are not allowed"
                )
        documents = list(yaml.load_all(raw, Loader=StrictLoader))
    except UsageError:
        raise
    except yaml.YAMLError as exc:
        raise UsageError(f"invalid YAML in {file}: {exc}") from exc
    if len(documents) != 1:
        raise UsageError(
            f"invalid YAML in {file}: exactly one document is required, "
            f"found {len(documents)}"
        )
    return documents[0]


def load_script(path: str | Path) -> dict[str, Any]:
    file = Path(path).expanduser()
    if not file.is_file():
        raise InputError(f"script file not found: {file}", details={"script": str(file)})
    try:
        size = file.stat().st_size
    except OSError as exc:
        raise InputError(f"cannot stat script {file}: {exc}") from exc
    if size > MAX_SCRIPT_BYTES:
        raise UsageError(
            f"script {file} is too large ({size} bytes; limit {MAX_SCRIPT_BYTES})"
        )
    suffix = file.suffix.lower()
    try:
        raw = file.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise InputError(f"cannot read script {file}: {exc}") from exc
    if suffix in (".yaml", ".yml"):
        data = _yaml_load_strict(raw, file)
    elif suffix == ".json":
        data = _json_load_strict(raw, file)
    else:
        raise UsageError(f"script must be .yaml/.yml/.json: {file}")
    if not isinstance(data, dict):
        raise UsageError("script must be an object with schema_version and segments")
    unknown = set(data) - _TOP_KEYS
    if unknown:
        raise UsageError(f"unknown script field(s): {', '.join(sorted(unknown))}")
    if data.get("schema_version") != SCRIPT_SCHEMA_VERSION:
        raise UsageError(
            f"unsupported script schema_version: {data.get('schema_version')!r}"
        )
    return data


def _segment_error(index: int, segment_id: Optional[str], message: str) -> UsageError:
    return UsageError(
        f"script segment {segment_id or index}: {message}",
        details={"segment": segment_id, "segment_index": index, "field": None},
    )


def _int_field(
    value: Any, field: str, index: int, segment_id: str, *, minimum: int
) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise UsageError(
            f"script segment {segment_id}: {field} must be an integer",
            details={"segment": segment_id, "segment_index": index, "field": field},
        )
    if value < minimum:
        raise UsageError(
            f"script segment {segment_id}: {field} must be >= {minimum}",
            details={"segment": segment_id, "segment_index": index, "field": field},
        )
    return value


def _next_generated_id(used: set[str], counter: list[int]) -> str:
    while True:
        candidate = f"s{counter[0]:03d}"
        counter[0] += 1
        if candidate not in used:
            used.add(candidate)
            return candidate


def _validate_defaults(
    data: dict[str, Any],
    cfg,
    *,
    cli_language: Optional[str],
    cli_instruct: Optional[str],
    cli_max_new_tokens: Optional[int],
    cli_pause_ms: Optional[int],
    voice_default: Optional[str],
) -> dict[str, Any]:
    """Validate the ``defaults`` block and merge explicit CLI overrides."""
    defaults = data.get("defaults") or {}
    if not isinstance(defaults, dict):
        raise UsageError("script.defaults must be an object")
    unknown_defaults = set(defaults) - _DEFAULT_KEYS
    if unknown_defaults:
        raise UsageError(
            f"unknown script.defaults field(s): {', '.join(sorted(unknown_defaults))}"
        )

    language = (
        cli_language if cli_language is not None else defaults.get("language", "auto")
    )
    if not isinstance(language, str) or not language:
        raise UsageError("script language must be a non-empty string")

    pause_value = (
        cli_pause_ms
        if cli_pause_ms is not None
        else defaults.get("pause_ms", DEFAULT_PAUSE_MS)
    )
    tokens_value = (
        cli_max_new_tokens
        if cli_max_new_tokens is not None
        else defaults.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS)
    )
    seed_value = defaults.get("seed", cfg.seed)
    pause = _int_field(pause_value, "defaults.pause_ms", 0, "defaults", minimum=0)
    tokens = _int_field(
        tokens_value, "defaults.max_new_tokens", 0, "defaults", minimum=1
    )
    seed = _int_field(seed_value, "defaults.seed", 0, "defaults", minimum=0)

    instruct = cli_instruct if cli_instruct is not None else defaults.get("instruct")
    if instruct is not None and not isinstance(instruct, str):
        raise UsageError("script.defaults.instruct must be a string or null")

    format_default = defaults.get("format", "wav")
    if format_default not in ("wav", "flac"):
        raise UsageError("script.defaults.format must be wav or flac")
    script_voice = defaults.get("voice", voice_default)
    if script_voice is not None and not isinstance(script_voice, str):
        raise UsageError("script.defaults.voice must be a string")
    return {
        "language": language,
        "pause": pause,
        "tokens": tokens,
        "seed": seed,
        "instruct": instruct,
        "format": format_default,
        "voice": script_voice,
    }


def _validate_segments(
    data: dict[str, Any], defaults: dict[str, Any]
) -> list[dict[str, Any]]:
    """Validate every segment's shape before any model lookup. Exit 2 on errors."""
    segments_raw = data.get("segments")
    if not isinstance(segments_raw, list) or not segments_raw:
        raise UsageError("script.segments must be a non-empty list")

    specs: list[dict[str, Any]] = []
    explicit_ids: set[str] = set()
    for index, item in enumerate(segments_raw):
        if not isinstance(item, dict):
            raise UsageError(f"script.segments[{index}] must be an object")
        unknown = set(item) - _SEGMENT_KEYS
        raw_id = item.get("id")
        label = raw_id if isinstance(raw_id, str) else f"#{index + 1}"
        if unknown:
            raise UsageError(
                f"script segment {label}: unknown field(s): {', '.join(sorted(unknown))}",
                details={"segment": label, "segment_index": index, "field": None},
            )
        if raw_id is not None:
            if not isinstance(raw_id, str) or not raw_id:
                raise UsageError(
                    f"script.segments[{index}].id must be a non-empty string"
                )
            if raw_id in explicit_ids:
                raise UsageError(
                    f"duplicate script segment id: {raw_id!r}",
                    details={"segment": raw_id, "segment_index": index, "field": "id"},
                )
            explicit_ids.add(raw_id)

        text = item.get("text")
        if not isinstance(text, str) or not text.strip():
            raise _segment_error(index, raw_id, "text must be a non-empty string")
        # Catch malformed director markers before scanning models.
        parse_director_markers(text)

        voice_ref = item.get("voice", defaults["voice"])
        if not isinstance(voice_ref, str) or not voice_ref:
            raise _segment_error(index, raw_id, "voice is required (segment or defaults)")
        language = item.get("language", defaults["language"])
        if not isinstance(language, str) or not language:
            raise _segment_error(index, raw_id, "language must be a non-empty string")
        instruct = item.get("instruct", defaults["instruct"])
        if instruct is not None and not isinstance(instruct, str):
            raise _segment_error(index, raw_id, "instruct must be a string or null")
        pause_after = _int_field(
            item.get("pause_after_ms", defaults["pause"]),
            "pause_after_ms", index, raw_id or label, minimum=0,
        )
        seed = _int_field(
            item.get("seed", defaults["seed"]), "seed", index, raw_id or label, minimum=0
        )
        max_tokens = _int_field(
            item.get("max_new_tokens", defaults["tokens"]),
            "max_new_tokens", index, raw_id or label, minimum=1,
        )
        role = item.get("role")
        if role is not None and not isinstance(role, str):
            raise _segment_error(index, raw_id, "role must be a string or null")
        specs.append(
            {
                "index": index,
                "raw_id": raw_id,
                "text": text,
                "voice_ref": voice_ref,
                "language": language,
                "instruct": instruct,
                "pause_after": pause_after,
                "seed": seed,
                "max_tokens": max_tokens,
                "role": role,
            }
        )

    used_ids = set(explicit_ids)
    counter = [1]
    for spec in specs:
        spec["seg_id"] = spec["raw_id"] or _next_generated_id(used_ids, counter)
    return specs


def _load_models(cfg, models: Optional[Iterable[ModelInfo]]) -> list[ModelInfo]:
    if models is not None:
        return list(models)
    model_list = scan(cfg.model_dir)
    if not model_list:
        raise ModelError(
            "no local models found; pass --model-dir DIR or run `qtts models pull`",
            details={"model_dir": str(cfg.model_dir) if cfg.model_dir else None},
        )
    return model_list


def compile_script(
    cfg,
    script_path: str | Path,
    *,
    models: Optional[Iterable[ModelInfo]] = None,
    store: VoiceStore,
    lexicon: Optional[Lexicon] = None,
    model_override: Optional[str] = None,
    voice_default: Optional[str] = None,
    output: Optional[str] = None,
    fmt: Optional[str] = None,
    subtitle_formats: Optional[list[str]] = None,
    granularity: str = "sentence",
    cli_language: Optional[str] = None,
    cli_instruct: Optional[str] = None,
    cli_max_new_tokens: Optional[int] = None,
    cli_pause_ms: Optional[int] = None,
    script_data: Optional[dict[str, Any]] = None,
) -> GenerationPlan:
    """Validate a script and compile it without loading model weights."""
    data = script_data if script_data is not None else load_script(script_path)
    defaults = _validate_defaults(
        data,
        cfg,
        cli_language=cli_language,
        cli_instruct=cli_instruct,
        cli_max_new_tokens=cli_max_new_tokens,
        cli_pause_ms=cli_pause_ms,
        voice_default=voice_default,
    )
    specs = _validate_segments(data, defaults)
    # Model lookup happens only after every syntax/schema error has been ruled out.
    model_list = _load_models(cfg, models)
    dtype_name = resolve_dtype_name(cfg)

    all_segments: list[SegmentPlan] = []
    warnings: list[str] = []
    default_voice_resolution: Optional[VoiceResolution] = None
    natural_models: set[str] = set()
    final_ids: set[str] = set()
    plan_leading = 0
    plan_trailing = 0

    for spec in specs:
        index = spec["index"]
        seg_id = spec["seg_id"]
        text = spec["text"]
        voice_ref = spec["voice_ref"]
        language = spec["language"]
        instruct = spec["instruct"]
        pause_after = spec["pause_after"]
        seed = spec["seed"]
        max_tokens = spec["max_tokens"]
        role = spec["role"]

        try:
            parsed = parse_voice_ref(voice_ref)
            natural = resolve_voice(model_list, store, parsed)
            natural_models.add(natural.model.model_id)
            if model_override and len(natural_models) > 1:
                raise UsageError(
                    "--model cannot override a multi-model script; remove segment-level "
                    "voices or drop --model",
                    details={"models": sorted(natural_models)},
                )
            voice = (
                resolve_voice(model_list, store, parsed, model_override=model_override)
                if model_override
                else natural
            )
            effective_instruct = instruct
            if voice.mode == "design" and not effective_instruct:
                effective_instruct = voice.description
            if voice.mode == "clone" and effective_instruct:
                raise CapabilityError(
                    "clone voices do not support instruct; remove instruct from this segment"
                )
            validate_capabilities(
                voice.model,
                mode=voice.mode,
                instruct=effective_instruct,
                language=None if language.lower() == "auto" else language,
                icl=voice.icl,
                ref_text=voice.ref_text,
                description=voice.description,
            )
        except (InputError, CapabilityError) as exc:
            raise InputError(
                f"script segment {seg_id}: {exc.message}",
                details={
                    "segment": seg_id,
                    "segment_index": index,
                    "voice": voice_ref,
                    **(exc.details or {}),
                },
            ) from exc

        if default_voice_resolution is None:
            default_voice_resolution = voice
        display_text, pieces, segment_warnings, leading, trailing = prepare_text_pieces(
            text,
            language=language,
            rate=codec_rate(voice.model),
            max_new_tokens=int(max_tokens),
            pause_ms=int(pause_after),
            lexicon=lexicon,
        )
        if not display_text.strip():
            raise _segment_error(
                index, seg_id, "text is empty after removing director markers"
            )
        warnings.extend(f"segment {seg_id}: {warning}" for warning in segment_warnings)

        # A middle segment's leading pause belongs after the previous segment.
        if leading:
            if all_segments:
                all_segments[-1].pause_after_ms += int(leading)
            else:
                plan_leading += int(leading)

        doc_instruct = effective_instruct if voice.mode in ("builtin", "design") else None
        sampling = {**SAMPLING_DEFAULTS, "max_new_tokens": int(max_tokens)}
        multiple = len(pieces) > 1
        for piece_index, piece in enumerate(pieces):
            piece_id = f"{seg_id}.p{piece_index + 1}" if multiple else seg_id
            if piece_id in final_ids:
                raise UsageError(
                    f"duplicate script segment id after splitting: {piece_id!r}",
                    details={"segment": seg_id, "id": piece_id, "segment_index": index},
                )
            final_ids.add(piece_id)
            seg_language = piece["language"]
            try:
                validate_capabilities(
                    voice.model,
                    mode=voice.mode,
                    language=seg_language,
                    description=voice.description,
                    icl=voice.icl,
                    ref_text=voice.ref_text,
                )
            except CapabilityError as exc:
                raise InputError(
                    f"script segment {seg_id}: {exc.message}",
                    details={
                        "segment": seg_id,
                        "segment_index": index,
                        "language": seg_language,
                        **(exc.details or {}),
                    },
                ) from exc
            key = cache_key(
                text_normalized=piece["spoken_text"],
                voice_identity=voice.identity,
                model_id=voice.model.model_id,
                model_revision=voice.model.revision,
                dtype=dtype_name,
                language=seg_language,
                instruct=doc_instruct,
                sampling=sampling,
                seed=int(seed),
                model_fingerprint=voice.model.fingerprint or None,
                lexicon_sha256=lexicon.sha256 if lexicon else None,
            )
            pause = piece["pause_after_ms"]
            if piece_index == len(pieces) - 1:
                pause = int(pause_after)
            all_segments.append(
                SegmentPlan(
                    id=piece_id,
                    text=piece["spoken_text"],
                    language=seg_language,
                    instruct=doc_instruct,
                    pause_after_ms=pause,
                    estimated_tokens=estimate_tokens(
                        piece["spoken_text"], seg_language, codec_rate(voice.model)
                    ),
                    estimated_duration_s=estimate_duration_s(piece["spoken_text"], seg_language),
                    cache_key=key,
                    source_index=index,
                    source_segment_id=seg_id,
                    voice=voice,
                    display_text=piece["display_text"],
                    spoken_to_display=piece["mapping"],
                    role=role,
                    seed=int(seed),
                    max_new_tokens=int(max_tokens),
                )
            )

        # A middle segment's trailing pause belongs before the next segment.
        if trailing:
            if index == len(specs) - 1:
                plan_trailing += int(trailing)
            else:
                all_segments[-1].pause_after_ms += int(trailing)

    if model_override and len(natural_models) > 1:
        raise UsageError(
            "--model cannot override a multi-model script; remove segment-level "
            "voices or drop --model",
            details={"models": sorted(natural_models)},
        )
    assert default_voice_resolution is not None
    assert all_segments
    all_segments[-1].pause_after_ms = 0
    display_all = "".join(seg.display_text or seg.text for seg in all_segments)
    return GenerationPlan(
        text=display_all,
        normalized_text=display_all,
        voice=default_voice_resolution,
        segments=all_segments,
        sampling={**SAMPLING_DEFAULTS, "max_new_tokens": int(defaults["tokens"])},
        seed=int(defaults["seed"]),
        language=defaults["language"],
        instruct=defaults["instruct"],
        output=output,
        format=fmt or defaults["format"],
        codec_rate=codec_rate(default_voice_resolution.model),
        warnings=warnings,
        subtitle_formats=list(subtitle_formats or []),
        granularity=granularity,
        dtype_name=dtype_name,
        display_text=display_all,
        lexicon_sha256=lexicon.sha256 if lexicon else None,
        script_source=str(Path(script_path).expanduser()),
        leading_pause_ms=plan_leading,
        trailing_pause_ms=plan_trailing,
    )


def script_plan_json(plan: GenerationPlan) -> str:
    """Byte-stable plan JSON: no timestamps, runtime cache state or absolute paths."""
    payload = plan.to_dict()
    payload.pop("cache", None)
    payload.pop("script_source", None)
    output = payload.get("output")
    if isinstance(output, dict) and output.get("path"):
        path = Path(str(output["path"]))
        if path.is_absolute():
            output["path"] = None
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
