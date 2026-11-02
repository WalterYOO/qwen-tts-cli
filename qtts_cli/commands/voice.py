"""`qtts voice` — clone/design voice asset library (FR-P0-02/03)."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Optional

from .. import __version__
from ..admit import check_ref_audio
from ..audio_io import duration_s, resolve_output_format, write_audio
from ..capabilities import validate_capabilities
from ..catalog import ModelInfo, find_model, scan, select
from ..errors import InputError, ModelError, UsageError
from ..plan import SAMPLING_DEFAULTS, resolve_dtype_name
from ..sidecar import build_recipe, recipe_path, write_recipe
from ..textnorm import NORMALIZATION_VERSION
from ..tts_engine import TtsEngine, clear_model_cache
from ..utils import confirm, is_tty, sha256_file
from ..voice_store import VoiceStore, utc_now, validate_name

DEFAULT_PREVIEW_TEXT = "你好，这是音色试听。Hello, this is a voice preview."
DEFAULT_DESIGN_SAMPLE = "你好，很高兴认识你。"


# --------------------------------------------------------------------------- #
# add
# --------------------------------------------------------------------------- #
def add(cfg, args) -> dict[str, Any]:
    if args.kind == "clone":
        return _add_clone(cfg, args)
    if args.kind == "design":
        return _add_design(cfg, args)
    raise UsageError("--kind must be clone or design")


def _require_consent(args) -> dict[str, Any]:
    rights = bool(getattr(args, "i_have_rights", False))
    if not rights and is_tty():
        rights = confirm(
            "Confirm you have the rights to clone this voice (consent will be recorded)."
        )
    if not rights:
        raise InputError(
            "cloning requires explicit consent; pass --i-have-rights to record it",
            details={"consent": {"rights_confirmed": False}},
        )
    return {
        "rights_confirmed": True,
        "license": args.license or "self-owned",
        "note": args.consent_note or "",
    }


def _add_clone(cfg, args) -> dict[str, Any]:
    name = validate_name(args.name)
    store = VoiceStore(cfg.data_dir)
    if store.exists(name):
        raise InputError(f"voice already exists: {name}")
    if not args.ref_audio:
        raise UsageError("voice add --kind clone requires --ref-audio FILE")
    mode = (args.mode or "icl").lower()
    if mode not in ("icl", "xvec"):
        raise UsageError("--mode must be icl or xvec")
    if mode == "icl" and not (args.ref_text or "").strip():
        raise InputError("--mode icl requires --ref-text; use --mode xvec to skip it")

    report = check_ref_audio(args.ref_audio, ref_text=args.ref_text, language=args.language)
    if not report.ok:
        raise InputError(
            "reference audio rejected by admission checks",
            details=report.to_dict(),
        )
    consent = _require_consent(args)

    models = scan(cfg.model_dir)
    model = find_model(models, args.model) if args.model else select(models, "clone")
    # Capability gate before the model is loaded (codex P0-8).
    validate_capabilities(
        model,
        mode="clone",
        language=args.language,
        icl=(mode == "icl"),
        ref_text=args.ref_text if mode == "icl" else None,
    )
    engine = TtsEngine(cfg, model)
    engine.load()
    items = engine.create_clone_prompt_items(
        args.ref_audio,
        ref_text=args.ref_text if mode == "icl" else None,
        x_vector_only_mode=(mode == "xvec"),
    )
    if not items:
        raise ModelError("model returned an empty clone prompt")
    item = items[0]

    # Build the whole profile in a staging dir and publish it atomically so a
    # failure never leaves a half-written voice (codex P1-2).
    staging = store.staging_dir(name)
    try:
        meta = store.create(
            name=name,
            kind="clone",
            source={
                "type": "clone",
                "ref_audio_sha256": sha256_file(args.ref_audio),
                "ref_text": args.ref_text if mode == "icl" else None,
                "mode": mode,
                "ref_audio": "ref.wav",
            },
            model={"id": model.model_id, "revision": model.revision,
                   "fingerprint": model.fingerprint, "path": str(model.path)},
            tokenizer={"type": "12hz", "codec_rate_hz": 12.5},
            dtype=engine.dtype_name or resolve_dtype_name(cfg),
            language=args.language,
            default_instruct=None,
            tags=_split_tags(args.tags),
            consent=consent,
            qtts_version=__version__,
            staging=staging,
        )
        store.copy_ref_audio(name, args.ref_audio, directory=staging)
        store.save_prompt_tensors(
            name,
            ref_spk_embedding=item.ref_spk_embedding,
            ref_code=item.ref_code,
            x_vector_only_mode=bool(item.x_vector_only_mode),
            icl_mode=bool(item.icl_mode),
            directory=staging,
        )
        store.commit_staging(staging, name)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return {
        "ok": True,
        "voice": meta,
        "admission": report.to_dict(),
        "model": model.model_id,
    }


def _add_design(cfg, args) -> dict[str, Any]:
    name = validate_name(args.name)
    store = VoiceStore(cfg.data_dir)
    if store.exists(name):
        raise InputError(f"voice already exists: {name}")
    description = (args.description or "").strip()
    if not description:
        raise UsageError("voice add --kind design requires --description TEXT")
    sample_text = (args.sample_text or DEFAULT_DESIGN_SAMPLE).strip()
    language = args.language or "Chinese"

    models = scan(cfg.model_dir)
    design_model = find_model(models, args.model) if args.model else select(models, "design")
    # Validate the design model kind/description/language before loading (P0-8).
    validate_capabilities(
        design_model,
        mode="design",
        description=description,
        instruct=description,
        language=language,
    )
    base_model = None
    if args.materialize_clone:
        base_model = select(models, "clone")
        validate_capabilities(
            base_model,
            mode="clone",
            language=language,
            icl=True,
            ref_text=sample_text,
        )

    staging = store.staging_dir(name)
    try:
        meta = store.create(
            name=name,
            kind="design",
            source={
                "type": "design",
                "description": description,
                "sample_text": sample_text,
                "language": language,
                "ref_audio": None,
                "materialized": False,
            },
            model={
                "id": design_model.model_id,
                "revision": design_model.revision,
                "fingerprint": design_model.fingerprint,
                "path": str(design_model.path),
            },
            dtype=resolve_dtype_name(cfg),
            language=language,
            default_instruct=description,
            tags=_split_tags(args.tags),
            consent={
                "rights_confirmed": True,
                "license": args.license or "ai-generated-design",
                "note": args.consent_note or "",
            },
            qtts_version=__version__,
            staging=staging,
        )

        result: dict[str, Any] = {"ok": True, "voice": meta, "materialized": False}
        if args.materialize_clone:
            assert base_model is not None
            result.update(
                _materialize_design(
                    cfg, store, staging, meta, description, sample_text, language,
                    design_model, base_model,
                )
            )
        store.commit_staging(staging, name)
        return result
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _materialize_design(
    cfg,
    store: VoiceStore,
    staging: Path,
    meta: dict[str, Any],
    description: str,
    sample_text: str,
    language: str,
    design_model,
    base_model,
) -> dict[str, Any]:
    design_engine = TtsEngine(cfg, design_model)
    design_engine.load()
    audio, sr, gen_info = design_engine.generate(
        text=sample_text,
        language=language,
        mode="design",
        instruct=description,
        sampling={**SAMPLING_DEFAULTS, "max_new_tokens": 2048},
        seed=int(cfg.seed),
    )
    ref_path = Path(staging) / "ref.wav"
    write_audio(ref_path, audio, sr, "wav")

    # Release the design model before loading Base; two 1.7B models must not
    # coexist on MPS (codex P1-2).
    design_engine = None
    clear_model_cache()

    base_engine = TtsEngine(cfg, base_model)
    base_engine.load()
    items = base_engine.create_clone_prompt_items(ref_path, ref_text=sample_text, x_vector_only_mode=False)
    if not items:
        raise ModelError("clone model returned an empty prompt")
    item = items[0]
    store.save_prompt_tensors(
        meta.get("name", ""),
        ref_spk_embedding=item.ref_spk_embedding,
        ref_code=item.ref_code,
        x_vector_only_mode=False,
        icl_mode=True,
        directory=staging,
    )
    store.set_preview(meta.get("name", ""), ref_path, directory=staging)

    source = dict(meta.get("source") or {})
    source.update(
        {
            "type": "design",
            "description": description,
            "sample_text": sample_text,
            "materialized": True,
            "ref_audio": "ref.wav",
            "ref_audio_sha256": sha256_file(ref_path),
            "ref_text": sample_text,
            "mode": "icl",
        }
    )
    meta = dict(meta)
    meta.update(
        {
            "kind": "clone",
            "source": source,
            "model": {"id": base_model.model_id, "revision": base_model.revision,
                      "fingerprint": base_model.fingerprint, "path": str(base_model.path)},
            "dtype": base_engine.dtype_name,
        }
    )
    store.save_meta(meta, directory=staging)
    return {
        "materialized": True,
        "voice": meta,
        "design_preview": {
            "duration_s": round(duration_s(audio, sr), 3),
            "generate_ms": gen_info.get("generate_ms"),
            "sample_rate": sr,
        },
    }


# --------------------------------------------------------------------------- #
# list / inspect
# --------------------------------------------------------------------------- #
def list_voices(cfg, args) -> dict[str, Any]:
    models = scan(cfg.model_dir)
    builtin: list[dict[str, Any]] = []
    seen: set[str] = set()
    for model in models:
        if model.kind != "builtin":
            continue
        for speaker in model.speakers:
            key = speaker.lower()
            if key in seen:
                continue
            seen.add(key)
            builtin.append(
                {
                    "name": speaker,
                    "ref": f"builtin:{speaker}",
                    "model": model.name,
                    "languages": model.languages,
                    "supports_instruct": model.supports_instruct,
                }
            )
    store = VoiceStore(cfg.data_dir)
    voices = []
    for meta in store.list():
        voices.append(
            {
                "name": meta.get("name"),
                "kind": meta.get("kind"),
                "language": meta.get("language"),
                "created_at": meta.get("created_at"),
                "model": (meta.get("model") or {}).get("id"),
                "materialized": bool((meta.get("source") or {}).get("materialized")),
                "has_prompt": store.prompt_sha256(str(meta.get("name"))) is not None,
            }
        )
    return {"ok": True, "builtin": builtin, "voices": voices}


def inspect(cfg, args) -> dict[str, Any]:
    name = args.name
    models = scan(cfg.model_dir)
    for model in models:
        canonical = model.canonical_speaker(name)
        if canonical:
            return {
                "ok": True,
                "kind": "builtin",
                "speaker": canonical,
                "ref": f"builtin:{canonical}",
                "model": model.to_dict(),
            }
    store = VoiceStore(cfg.data_dir)
    meta = store.require(name)
    preview = store.dir_for(name) / "preview.wav"
    return {
        "ok": True,
        "kind": meta.get("kind"),
        "voice": meta,
        "prompt_sha256": store.prompt_sha256(name),
        "preview": str(preview) if preview.is_file() else None,
    }


# --------------------------------------------------------------------------- #
# preview / rm
# --------------------------------------------------------------------------- #
def preview(cfg, args) -> dict[str, Any]:
    text = args.text or DEFAULT_PREVIEW_TEXT
    models = scan(cfg.model_dir)
    store = VoiceStore(cfg.data_dir)
    name = args.name

    builtin_model: Optional[ModelInfo] = None
    speaker = None
    for model in models:
        canonical = model.canonical_speaker(name)
        if canonical:
            builtin_model, speaker = model, canonical
            break

    language = args.language or "Chinese"
    prompt_items = None
    meta: dict[str, Any] = {}
    if builtin_model is not None:
        target_model = builtin_model
        mode = "builtin"
        description = None
        instr = None
        ref = f"builtin:{speaker}"
        voice_name = None
    else:
        meta = store.require(name)
        kind = str(meta.get("kind"))
        if kind == "clone":
            target_model = select(models, "clone")
            mode = "clone"
            description = None
            instr = None
            ref = f"voice:{name}"
            voice_name = name
        elif kind == "design":
            target_model = select(models, "design")
            mode = "design"
            description = (meta.get("source") or {}).get("description")
            instr = description
            ref = f"voice:{name}"
            voice_name = None
        else:
            raise InputError(f"voice {name} has unsupported kind: {kind!r}")

    # Same capability gate as `say`, before any weight load (codex P0-8).
    source = (meta.get("source") or {}) if meta else {}
    clone_mode = str(source.get("mode") or "icl").lower()
    if mode == "clone":
        # Check the frozen prompt and ICL metadata before engine.load(), not
        # after (codex P0-9 follow-up).
        if store.prompt_sha256(voice_name or name) is None:
            raise InputError(
                f"voice {name} has no prompt.safetensors; re-add the voice or run "
                "voice add --materialize-clone",
                details={"voice": name, "path": str(store.dir_for(voice_name or name))},
            )
        if clone_mode == "icl" and not str(source.get("ref_text") or "").strip():
            raise InputError(
                f"voice {name} is an ICL clone but has no ref_text; re-add it with --ref-text",
                details={"voice": name, "mode": "icl"},
            )
    validate_capabilities(
        target_model,
        mode=mode,
        language=language,
        instruct=instr,
        description=description,
        icl=(mode == "clone" and clone_mode == "icl"),
        ref_text=source.get("ref_text") if mode == "clone" else None,
    )

    engine = TtsEngine(cfg, target_model)
    if mode == "clone":
        engine.load()
        prompt_items = engine.build_clone_prompt_items(store, voice_name or name)
    audio, sr, info = engine.generate(
        text=text,
        language=language,
        mode=mode,
        speaker=speaker,
        instruct=instr,
        prompt_items=prompt_items,
        sampling={**SAMPLING_DEFAULTS, "max_new_tokens": 2048},
        seed=int(cfg.seed),
    )

    if args.out:
        out = Path(args.out)
    elif builtin_model is not None:
        out = Path(f"{speaker}.preview.wav")
    else:
        out = store.dir_for(name) / "preview.wav"
    fmt = resolve_output_format(out, args.format)
    write_audio(out, audio, sr, fmt)
    if builtin_model is None and out.name == "preview.wav" and out.parent == store.dir_for(name):
        store.set_preview(name, out)

    recipe = build_recipe(
        qtts_version=__version__,
        text=text,
        source="voice preview",
        voice_ref=ref,
        voice={"prompt_sha256": store.prompt_sha256(name) if builtin_model is None else None,
               "ref_audio_sha256": source.get("ref_audio_sha256")},
        model={"id": target_model.model_id, "revision": target_model.revision,
               "fingerprint": target_model.fingerprint,
               "dtype": engine.dtype_name, "path": str(target_model.path)},
        sampling={**SAMPLING_DEFAULTS, "max_new_tokens": 2048},
        segments=[{"id": "s001", "cache_key": None, "language": language,
                   "instruct": instr, "pause_after_ms": 0,
                   "duration_ms": int(round(duration_s(audio, sr) * 1000))}],
        created_at=utc_now(),
        seed=int(cfg.seed),
        normalization={"version": NORMALIZATION_VERSION},
        device=engine.device,
        language=language,
        timing={"total_ms": info.get("generate_ms"), "generate_ms": info.get("generate_ms")},
    )
    write_recipe(out, recipe)

    return {
        "ok": True,
        "voice": ref,
        "output": str(out),
        "duration_s": round(duration_s(audio, sr), 3),
        "sample_rate": sr,
        "generate_ms": info.get("generate_ms"),
        "recipe": str(recipe_path(out)),
    }


def rm(cfg, args) -> dict[str, Any]:
    store = VoiceStore(cfg.data_dir)
    if not store.exists(args.name):
        raise InputError(f"voice not found: {args.name}")
    if not args.yes:
        if not is_tty():
            raise InputError("voice rm requires --yes in non-interactive mode")
        if not confirm(f"Delete voice {args.name}?"):
            # Cancellation is not a failure (PRD exit codes; codex P1-7).
            return {"ok": True, "cancelled": True, "voice": args.name}
    store.remove(args.name)
    return {"ok": True, "removed": args.name}


# --------------------------------------------------------------------------- #
# helpers / formatting
# --------------------------------------------------------------------------- #
def _split_tags(value: Optional[str]) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def format_result(payload: dict[str, Any]) -> str:
    if "builtin" in payload and "voices" in payload:
        lines = ["built-in voices:"]
        for entry in payload["builtin"]:
            lines.append(f"  {entry['name']:10s} {entry['model']}  {','.join(entry['languages'][:4])}")
        lines.append("user voices:")
        if not payload["voices"]:
            lines.append("  (none)")
        for entry in payload["voices"]:
            lines.append(
                f"  {entry['name']:10s} kind={entry['kind']} lang={entry['language']} "
                f"prompt={'yes' if entry['has_prompt'] else 'no'}"
            )
        return "\n".join(lines)
    if payload.get("voice") and isinstance(payload.get("voice"), dict) and payload["voice"].get("name"):
        meta = payload["voice"]
        lines = [
            f"{meta['name']}: kind={meta.get('kind')} language={meta.get('language')} "
            f"created={meta.get('created_at')}",
            f"  consent={meta.get('consent')}",
        ]
        if payload.get("admission"):
            warnings = payload["admission"].get("warnings") or []
            lines.append(f"  admission: ok ({len(warnings)} warning(s))")
            for warn in warnings:
                lines.append(f"    warn {warn['name']}={warn['value']}: {warn.get('advice', '')}")
        if payload.get("materialized"):
            preview = payload.get("design_preview") or {}
            lines.append(f"  materialized clone; preview {preview.get('duration_s')}s")
        return "\n".join(lines)
    if payload.get("output"):
        return f"wrote {payload['output']} ({payload['duration_s']}s)"
    if payload.get("removed"):
        return f"removed voice {payload['removed']}"
    if payload.get("cancelled"):
        return "cancelled"
    if payload.get("speaker"):
        return f"builtin:{payload['speaker']} ({payload['model']['id']})"
    return str(payload)
