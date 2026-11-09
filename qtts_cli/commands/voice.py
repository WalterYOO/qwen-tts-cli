"""`qtts voice` — clone/design voice asset library (FR-P0-02/03)."""

from __future__ import annotations

import contextlib
import json
import math
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Optional

from .. import __version__
from ..admit import check_ref_audio
from ..asr_engine import AsrEngine, clear_asr_cache
from ..audio_io import duration_s, resolve_output_format, write_audio
from ..capabilities import VoiceRef, validate_capabilities
from ..catalog import ModelInfo, find_model, scan, select, select_asr
from ..errors import InputError, ModelError, UsageError
from ..plan import SAMPLING_DEFAULTS, resolve_dtype_name, resolve_voice
from ..sidecar import build_recipe, recipe_path, write_recipe
from ..textnorm import NORMALIZATION_VERSION
from ..tts_engine import TtsEngine, clear_model_cache
from ..utils import atomic_write_bytes, confirm, is_tty, sha256_file
from ..voice_archive import export_voice, import_voice
from ..voice_btts import BttsBundle, BundleVoice, load_bundle, read_bundle_file
from ..voice_store import VoiceStore, utc_now, validate_name

DEFAULT_PREVIEW_TEXT = "你好，这是音色试听。Hello, this is a voice preview."
DEFAULT_DESIGN_SAMPLE = "你好，很高兴认识你。"


# --------------------------------------------------------------------------- #
# add
# --------------------------------------------------------------------------- #
def add(cfg, args) -> dict[str, Any]:
    if getattr(args, "auto_ref_text", False) and args.kind != "clone":
        raise UsageError("--auto-ref-text is only valid with --kind clone")
    if getattr(args, "asr_model", None) and not getattr(args, "auto_ref_text", False):
        raise UsageError("--asr-model requires --auto-ref-text")
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

    auto_ref_text = bool(getattr(args, "auto_ref_text", False))
    explicit_language = getattr(args, "language", None)
    language = explicit_language or "Chinese"
    ref_text = getattr(args, "ref_text", None)
    if auto_ref_text:
        if ref_text is not None:
            raise UsageError("--auto-ref-text cannot be combined with --ref-text")
        if mode != "icl":
            raise UsageError("--auto-ref-text requires --mode icl")
    elif mode == "icl" and not (ref_text or "").strip():
        raise InputError(
            "--mode icl requires --ref-text or --auto-ref-text; "
            "use --mode xvec to skip it"
        )

    report = check_ref_audio(
        args.ref_audio,
        ref_text=None if auto_ref_text else ref_text,
        language=language,
    )
    if not report.ok:
        raise InputError(
            "reference audio rejected by admission checks",
            details=report.to_dict(),
        )
    consent = _require_consent(args)

    models = scan(cfg.model_dir)
    model = find_model(models, args.model) if args.model else select(models, "clone")
    # Capability gate before the model is loaded (codex P0-8). With auto
    # ref_text the ICL text is not known yet; it is validated after ASR.
    validate_capabilities(
        model,
        mode="clone",
        language=language,
        icl=(mode == "icl" and not auto_ref_text),
        ref_text=None if auto_ref_text else ref_text,
    )

    asr_model: Optional[ModelInfo] = None
    staging = store.staging_dir(name)
    try:
        # Stage the reference first: ASR and the clone prompt must read the
        # same bytes, and no half-written profile may survive a failure.
        store.copy_ref_audio(name, args.ref_audio, directory=staging)
        staged_ref = Path(staging) / "ref.wav"

        if auto_ref_text:
            asr_model = select_asr(models, getattr(args, "asr_model", None))
            asr = AsrEngine(cfg, asr_model)
            transcription = asr.transcribe(
                staged_ref,
                language=explicit_language,
                timestamps=False,
            )
            ref_text = str(transcription.get("text") or "").strip()
            if not ref_text:
                raise InputError(
                    "ASR produced an empty ref_text for the reference audio",
                    details={"model": asr_model.model_id},
                )
            # Free the 1.8G ASR model before loading Base (MPS co-residency).
            asr = None
            clear_asr_cache()

        engine = TtsEngine(cfg, model)
        engine.load()

        source: dict[str, Any] = {
            "type": "clone",
            "ref_text": ref_text if mode == "icl" else None,
            "ref_text_source": (
                "asr" if auto_ref_text else ("user" if ref_text else None)
            ),
            "mode": mode,
        }
        meta = _freeze_clone_prompt(
            cfg,
            store,
            name=name,
            staged_ref=staged_ref,
            ref_text=ref_text,
            mode=mode,
            language=language,
            model=model,
            engine=engine,
            source=source,
            consent=consent,
            staging=staging,
            tags=_split_tags(args.tags),
            asr_model=asr_model,
        )
        store.commit_staging(staging, name)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    stored = meta.get("source") or {}
    return {
        "ok": True,
        "voice": meta,
        "admission": report.to_dict(),
        "model": model.model_id,
        "ref_text_source": stored.get("ref_text_source"),
        "asr_model": stored.get("asr_model"),
    }


def _freeze_clone_prompt(
    cfg,
    store: VoiceStore,
    *,
    name: str,
    staged_ref: Path,
    ref_text: Optional[str],
    mode: str,
    language: str,
    model,
    engine,
    source: dict[str, Any],
    consent: dict[str, Any],
    staging: Path,
    tags: Optional[list[str]],
    preview_src: Optional[Path] = None,
    asr_model: Optional[ModelInfo] = None,
    allow_existing: bool = False,
) -> dict[str, Any]:
    """Build and persist one clone profile from a staged ``ref.wav``.

    Shared by ``voice add --kind clone`` and ``voice import-btts`` so the
    import path exercises the same prompt-creation and atomic-staging code.
    """
    items = engine.create_clone_prompt_items(
        staged_ref,
        ref_text=ref_text if mode == "icl" else None,
        x_vector_only_mode=(mode == "xvec"),
    )
    if not items:
        raise ModelError("model returned an empty clone prompt")
    item = items[0]

    source = dict(source)
    source["ref_audio"] = "ref.wav"
    source["ref_audio_sha256"] = sha256_file(staged_ref)
    if asr_model is not None:
        source["asr_model"] = {
            "id": asr_model.model_id,
            "revision": asr_model.revision,
            "fingerprint": asr_model.fingerprint,
        }

    meta = store.create(
        name=name,
        kind="clone",
        source=source,
        model={"id": model.model_id, "revision": model.revision,
               "fingerprint": model.fingerprint, "path": str(model.path)},
        tokenizer={"type": "12hz", "codec_rate_hz": 12.5},
        dtype=engine.dtype_name or resolve_dtype_name(cfg),
        language=language,
        default_instruct=None,
        tags=list(tags or []),
        consent=consent,
        qtts_version=__version__,
        staging=staging,
        allow_existing=allow_existing,
    )
    store.save_prompt_tensors(
        name,
        ref_spk_embedding=item.ref_spk_embedding,
        ref_code=item.ref_code,
        x_vector_only_mode=bool(item.x_vector_only_mode),
        icl_mode=bool(item.icl_mode),
        directory=staging,
    )
    if preview_src is not None:
        store.set_preview(name, preview_src, directory=staging)
    return meta


def _add_design(cfg, args) -> dict[str, Any]:
    name = validate_name(args.name)
    store = VoiceStore(cfg.data_dir)
    if store.exists(name):
        raise InputError(f"voice already exists: {name}")
    description = (args.description or "").strip()
    if not description:
        raise UsageError("voice add --kind design requires --description TEXT")
    sample_text = (args.sample_text or DEFAULT_DESIGN_SAMPLE).strip()
    language = getattr(args, "language", None) or "Chinese"

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
            # The design description is no longer an instruct once this is a
            # clone profile; leaving it would silently reapply the prompt.
            "default_instruct": None,
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
    payload = {
        "ok": True,
        "kind": meta.get("kind"),
        "voice": meta,
        "prompt_sha256": store.prompt_sha256(name),
        "preview": str(preview) if preview.is_file() else None,
    }
    if meta.get("kind") == "finetune":
        source = meta.get("source") or {}
        model = meta.get("model") or {}
        path = model.get("path")
        payload["checkpoint"] = {
            "path": path,
            "exists": bool(path) and Path(str(path)).is_dir(),
            "speaker": source.get("speaker"),
            "fingerprint": model.get("fingerprint"),
        }
    return payload


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
        elif kind == "finetune":
            resolution = resolve_voice(models, store, VoiceRef("voice", name))
            target_model = resolution.model
            speaker = resolution.speaker
            mode = "builtin"
            description = None
            instr = None
            ref = f"voice:{name}"
            voice_name = name
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


def export(cfg, args) -> dict[str, Any]:
    store = VoiceStore(cfg.data_dir)
    return export_voice(store, args.name, args.out, force=bool(args.force))


def import_(cfg, args) -> dict[str, Any]:
    store = VoiceStore(cfg.data_dir)
    return import_voice(store, args.archive, name=args.name, force=bool(args.force))


# --------------------------------------------------------------------------- #
# import-btts
# --------------------------------------------------------------------------- #
def import_btts(cfg, args) -> dict[str, Any]:
    requested = list(getattr(args, "voice", None) or [])
    if args.all and requested:
        raise UsageError("--all cannot be combined with --voice")
    if not args.all and not requested:
        raise UsageError("select voices with --voice NAME (repeatable) or --all")
    if len(set(requested)) != len(requested):
        raise UsageError("--voice names must be unique")
    if args.name:
        if args.all:
            raise UsageError(
                "--name requires exactly one selected voice; it cannot be used with --all"
            )
        if len(requested) != 1:
            raise UsageError("--name requires exactly one selected voice")

    mode = (args.mode or "icl").lower()
    auto = bool(getattr(args, "auto_ref_text", False))
    explicit_text = args.ref_text
    if auto and explicit_text is not None:
        raise UsageError("--auto-ref-text cannot be combined with --ref-text")
    if auto and mode != "icl":
        raise UsageError("--auto-ref-text requires --mode icl")
    if getattr(args, "asr_model", None) and not auto:
        raise UsageError("--asr-model requires --auto-ref-text")
    if explicit_text is not None and mode != "icl":
        raise UsageError("--ref-text requires --mode icl")
    if explicit_text is not None and not explicit_text.strip():
        raise UsageError("--ref-text must not be empty")
    if not getattr(args, "i_have_rights", False):
        raise InputError(
            "importing btts voices requires explicit consent; pass --i-have-rights "
            "to record it",
            details={"consent": {"rights_confirmed": False}},
        )

    bundle = load_bundle(args.bundle)
    selected = _select_btts_voices(bundle, requested, args.all)
    if args.name and len(selected) != 1:
        raise UsageError("--name requires exactly one selected voice")
    if explicit_text is not None and (
        len(selected) != 1 or selected[0].kind != "design"
    ):
        raise UsageError("--ref-text is only valid for a single design voice")
    if auto and not any(voice.kind == "design" for voice in selected):
        raise UsageError(
            "--auto-ref-text requires at least one design voice; clone voices "
            "always use the transcript from the bundle"
        )

    store = VoiceStore(cfg.data_dir)
    existing = store.existing_names()
    targets: list[tuple[BundleVoice, str]] = []
    seen_targets: dict[str, str] = {}
    replace_paths: dict[str, Path] = {}
    conflicts: list[str] = []
    for voice in selected:
        target = validate_name(args.name or voice.name)
        folded = target.casefold()
        other = seen_targets.get(folded)
        if other is not None:
            raise InputError(
                f"two selected voices would install as {target!r}",
                details={"bundle": str(bundle.path), "voices": sorted({other, target})},
            )
        seen_targets[folded] = target
        on_disk = existing.get(folded)
        if on_disk is not None:
            if on_disk != target:
                # A case-only rename must replace the existing directory, not
                # create a second profile beside it on case-sensitive systems.
                replace_paths[target] = store.root / on_disk
            if not args.force:
                conflicts.append(target)
        targets.append((voice, target))
    if conflicts and not args.force:
        raise InputError(
            "voice already exists: "
            + ", ".join(sorted(conflicts))
            + "; pass --force to replace",
            details={"bundle": str(bundle.path), "voices": sorted(conflicts)},
        )

    models = scan(cfg.model_dir)
    model = (
        find_model(models, args.model)
        if getattr(args, "model", None)
        else select(models, "clone")
    )

    plans = [
        _btts_plan(voice, target, bundle=bundle, mode=mode,
                   auto=auto, explicit_text=explicit_text)
        for voice, target in targets
    ]
    for plan in plans:
        validate_capabilities(
            model,
            mode="clone",
            language=plan["voice"].language,
            icl=(mode == "icl" and plan["ref_text"] is not None),
            ref_text=plan["ref_text"],
        )

    asr_model = select_asr(models, getattr(args, "asr_model", None)) if auto else None
    asr = None
    staged: list[tuple[str, Path]] = []
    staging_by_target: dict[str, Path] = {}
    replaced: set[str] = set()
    try:
        for plan in plans:
            voice = plan["voice"]
            target = plan["target"]
            staging = store.staging_dir(target)
            staged.append((target, staging))
            staging_by_target[target] = staging

            ref_rel = (
                f"{voice.name}/{voice.ref_audio}"
                if voice.kind == "clone"
                else voice.preview
            )
            atomic_write_bytes(staging / "ref.wav", read_bundle_file(bundle, ref_rel))
            atomic_write_bytes(
                staging / "preview.wav",
                read_bundle_file(bundle, voice.preview),
            )
            staged_ref = Path(staging) / "ref.wav"

            if auto and voice.kind == "design":
                if asr is None:
                    assert asr_model is not None
                    asr = AsrEngine(cfg, asr_model)
                transcription = asr.transcribe(
                    staged_ref, language=voice.language, timestamps=False
                )
                text = str(transcription.get("text") or "").strip()
                if not text:
                    raise InputError(
                        "ASR produced an empty ref_text for the design preview",
                        details={
                            "bundle": str(bundle.path),
                            "voice": voice.name,
                            "file": voice.preview,
                            "model": asr_model.model_id if asr_model else None,
                        },
                    )
                plan["ref_text"] = text
                plan["ref_text_source"] = "asr"

            report = check_ref_audio(
                staged_ref,
                ref_text=plan["ref_text"] if mode == "icl" else None,
                language=voice.language,
            )
            if not report.ok:
                details = report.to_dict()
                details.update(
                    {"bundle": str(bundle.path), "voice": voice.name, "file": ref_rel}
                )
                raise InputError(
                    "reference audio rejected by admission checks", details=details
                )
            plan["admission"] = report
            validate_capabilities(
                model,
                mode="clone",
                language=voice.language,
                icl=(mode == "icl"),
                ref_text=plan["ref_text"] if mode == "icl" else None,
            )

        # Release ASR memory before the Base model is loaded (MPS co-residency).
        asr = None
        if auto:
            clear_asr_cache()

        engine = TtsEngine(cfg, model)
        engine.load()
        for plan in plans:
            voice = plan["voice"]
            target = plan["target"]
            staging = staging_by_target[target]
            source = {
                "type": "clone",
                "origin": "btts",
                "mode": mode,
                "ref_text": plan["ref_text"] if mode == "icl" else None,
                "ref_text_source": (
                    plan["ref_text_source"] if mode == "icl" else None
                ),
                "btts": _btts_provenance(voice, bundle),
            }
            meta = _freeze_clone_prompt(
                cfg,
                store,
                name=target,
                staged_ref=staging / "ref.wav",
                ref_text=plan["ref_text"],
                mode=mode,
                language=voice.language,
                model=model,
                engine=engine,
                source=source,
                consent=_btts_consent(voice, bundle),
                staging=staging,
                tags=voice.tags,
                preview_src=staging / "preview.wav",
                asr_model=(
                    asr_model if plan["ref_text_source"] == "asr" else None
                ),
                allow_existing=bool(args.force),
            )
            plan["meta"] = meta

        replaced = _commit_imported(
            store,
            staged,
            force=bool(args.force),
            replace=replace_paths,
        )
        staged = []
    finally:
        for _, staging in staged:
            shutil.rmtree(staging, ignore_errors=True)
        if auto:
            clear_asr_cache()

    results = []
    for plan in plans:
        voice = plan["voice"]
        target = plan["target"]
        results.append(
            {
                "name": target,
                "btts_name": voice.name,
                "kind": "clone",
                "btts_kind": voice.kind,
                "language": voice.language,
                "btts_language": voice.btts_language,
                "profile_sha256": voice.profile_sha256,
                "ref_text_source": (
                    plan["ref_text_source"] if mode == "icl" else None
                ),
                "replaced": target in replaced,
                "output": str(store.dir_for(target)),
                "files": sorted(voice.files),
            }
        )
    return {
        "ok": True,
        "kind": "btts-import",
        "bundle": str(bundle.path),
        "bundle_format": bundle.format,
        "manifest": {
            "present": bundle.manifest_present,
            "verified": bundle.manifest_verified,
            "sha256": bundle.manifest_sha256,
        },
        "voices": results,
        "replaced": bool(replaced),
        "warnings": list(bundle.warnings),
    }


def _select_btts_voices(
    bundle: BttsBundle, requested: list[str], all_voices: bool
) -> list[BundleVoice]:
    if all_voices:
        return [bundle.voices[name] for name in sorted(bundle.voices)]
    unknown = [name for name in requested if name not in bundle.voices]
    if unknown:
        raise InputError(
            "voice not found in btts bundle: " + ", ".join(unknown),
            details={
                "bundle": str(bundle.path),
                "unknown": unknown,
                "available": sorted(bundle.voices),
            },
        )
    return [bundle.voices[name] for name in requested]


def _btts_plan(
    voice: BundleVoice,
    target: str,
    *,
    bundle: BttsBundle,
    mode: str,
    auto: bool,
    explicit_text: Optional[str],
) -> dict[str, Any]:
    ref_text: Optional[str] = None
    ref_text_source: Optional[str] = None
    if voice.kind == "clone":
        if mode == "icl":
            ref_text = voice.ref_text
            ref_text_source = "btts"
    elif mode == "icl":
        if explicit_text is not None:
            ref_text = explicit_text
            ref_text_source = "user"
        elif auto:
            ref_text_source = "asr"
        else:
            raise InputError(
                f"design voice {voice.name} has no reference transcript; pass "
                "--auto-ref-text (Qwen3-ASR on preview.wav), --ref-text TEXT, "
                "or --mode xvec",
                details={
                    "bundle": str(bundle.path),
                    "voice": voice.name,
                    "file": voice.preview,
                },
            )
    return {
        "voice": voice,
        "target": target,
        "ref_text": ref_text,
        "ref_text_source": ref_text_source,
    }


def _commit_imported(
    store: VoiceStore,
    staged: list[tuple[str, Path]],
    *,
    force: bool,
    replace: Optional[dict[str, Path]] = None,
) -> set[str]:
    """Publish every staged profile, rolling the whole set back on failure."""
    replace = replace or {}
    committed: list[tuple[Path, Optional[Path], Path]] = []
    replaced: set[str] = set()
    try:
        for name, staging in staged:
            target = store.dir_for(name)
            restore = replace.get(name, target)
            backup: Optional[Path] = None
            if restore.exists():
                if not force:
                    raise InputError(
                        f"voice already exists: {name}; pass --force to replace",
                        details={"voice": name},
                    )
                backup = target.with_name(
                    f".{target.name}.backup.{uuid.uuid4().hex[:8]}"
                )
                os.replace(restore, backup)
            try:
                os.replace(staging, target)
            except BaseException:
                if backup is not None:
                    with contextlib.suppress(OSError):
                        os.replace(backup, restore)
                raise
            committed.append((target, backup, restore))
            if backup is not None:
                replaced.add(name)
    except BaseException:
        for target, backup, restore in reversed(committed):
            with contextlib.suppress(OSError):
                shutil.rmtree(target, ignore_errors=True)
            if backup is not None:
                with contextlib.suppress(OSError):
                    os.replace(backup, restore)
        raise
    for _, backup, _ in committed:
        if backup is not None:
            shutil.rmtree(backup, ignore_errors=True)
    return replaced


def _btts_consent(voice: BundleVoice, bundle: BttsBundle) -> dict[str, Any]:
    license_block = voice.profile.get("license")
    license_model = (
        license_block.get("model") if isinstance(license_block, dict) else None
    )
    license_value = (
        f"btts:{license_model}"
        if isinstance(license_model, str) and license_model
        else "btts-import"
    )
    return {
        "rights_confirmed": True,
        "license": license_value,
        "note": (
            f"imported from btts bundle {bundle.path.name}; the btts profile "
            "consent is provenance only and was not used as qtts consent"
        ),
    }


def _btts_provenance(voice: BundleVoice, bundle: BttsBundle) -> dict[str, Any]:
    profile = voice.profile
    reference = (
        profile.get("reference")
        if isinstance(profile.get("reference"), dict)
        else None
    )
    design = profile.get("design") if isinstance(profile.get("design"), dict) else None
    provenance = (
        profile.get("provenance")
        if isinstance(profile.get("provenance"), dict)
        else {}
    )
    identity = profile.get("identity") if isinstance(profile.get("identity"), dict) else {}
    license_block = profile.get("license") if isinstance(profile.get("license"), dict) else {}

    out: dict[str, Any] = {
        "bundle": str(bundle.path),
        "bundle_format": bundle.format,
        "manifest": {
            "present": bundle.manifest_present,
            "verified": bundle.manifest_verified,
            "sha256": bundle.manifest_sha256,
        },
        "name": voice.name,
        "voice_id": voice.voice_id,
        "kind": voice.kind,
        "language": voice.btts_language,
        "profile_sha256": voice.profile_sha256,
        "profile_path": voice.profile_rel,
        "tags": list(voice.tags),
        "license": {"model": _bounded_str(license_block.get("model"), 256)},
        "model": {
            "id": _bounded_str(provenance.get("model_id"), 256),
            "digest": _bounded_str(provenance.get("model_digest"), 256),
        },
        "provenance": {
            "source": _bounded_str(provenance.get("source"), 64),
            "source_take_id": _bounded_str(provenance.get("source_take_id"), 256),
            "source_session_id": _bounded_str(
                provenance.get("source_session_id"), 256
            ),
            "api_protocol": _bounded_str(provenance.get("api_protocol"), 128),
            "cli_version": _bounded_str(provenance.get("cli_version"), 64),
        },
        "identity": {
            "rendered_instruction": _bounded_str(
                identity.get("rendered_instruction"), 4096
            ),
            "template": _bounded_str(identity.get("template"), 256),
            "compiler_version": _bounded_str(identity.get("compiler_version"), 64),
        },
        "files": {
            filename: ref.sha256 for filename, ref in sorted(voice.files.items())
        },
    }
    if reference is not None:
        out["reference"] = {
            "audio": _bounded_str(reference.get("audio"), 128),
            "audio_sha256": _bounded_str(reference.get("audio_sha256"), 128),
            "transcript": _bounded_str(reference.get("transcript"), 128),
            "transcript_sha256": _bounded_str(
                reference.get("transcript_sha256"), 128
            ),
            "transcript_confirmed": bool(reference.get("transcript_confirmed")),
            "duration_s": _bounded_number(reference.get("duration_s")),
            "sample_rate": _bounded_int(reference.get("sample_rate")),
        }
    if design is not None:
        out["design"] = {
            "instruction": _bounded_str(design.get("instruction"), 4096),
            "seed": _bounded_int(design.get("seed")),
            "cfg_scale": _bounded_number(design.get("cfg_scale")),
            "max_new_tokens": _bounded_int(design.get("max_new_tokens")),
            "design_yaml": "design.yaml" if voice.design_yaml else None,
            "design_yaml_sha256": (
                voice.files["design.yaml"].sha256 if voice.design_yaml else None
            ),
        }
    consent = profile.get("consent")
    if isinstance(consent, dict):
        out["consent"] = _bounded_json(consent)
    elif consent is not None:
        out["consent"] = {"present": True}
    return out


def _bounded_str(value: Any, limit: int) -> Optional[str]:
    if not isinstance(value, str):
        return None
    value = value.strip()
    if not value:
        return None
    return value[:limit]


def _bounded_json(value: Any, limit: int = 4096) -> Any:
    try:
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        return {"present": True}
    if len(encoded) > limit:
        return {"present": True, "truncated": True}
    return value


def _bounded_number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def _bounded_int(value: Any) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def rm(cfg, args) -> dict[str, Any]:
    store = VoiceStore(cfg.data_dir)
    meta = store.get(args.name)
    if meta is None:
        raise InputError(f"voice not found: {args.name}")
    if not args.yes:
        if not is_tty():
            raise InputError("voice rm requires --yes in non-interactive mode")
        if not confirm(f"Delete voice {args.name}?"):
            # Cancellation is not a failure (PRD exit codes; codex P1-7).
            return {"ok": True, "cancelled": True, "voice": args.name}
    store.remove(args.name)
    payload: dict[str, Any] = {"ok": True, "removed": args.name}
    if meta.get("kind") == "finetune":
        payload["checkpoint"] = (meta.get("model") or {}).get("path")
    return payload


# --------------------------------------------------------------------------- #
# helpers / formatting
# --------------------------------------------------------------------------- #
def _split_tags(value: Optional[str]) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def format_result(payload: dict[str, Any]) -> str:
    if payload.get("archive") and "replaced" in payload:
        meta = payload.get("voice") or {}
        action = "replaced" if payload.get("replaced") else "imported"
        return (
            f"{action} voice {meta.get('name')} from {payload['archive']} -> "
            f"{payload['output']} ({', '.join(payload.get('files') or [])})"
        )
    if payload.get("archive"):
        meta = payload.get("voice") or {}
        return (
            f"exported voice {meta.get('name')} -> {payload['archive']} "
            f"({len(payload.get('files') or [])} files, sha256 {payload.get('archive_sha256')})"
        )
    if payload.get("kind") == "btts-import":
        manifest = payload.get("manifest") or {}
        if manifest.get("verified"):
            manifest_text = "manifest verified"
        elif manifest.get("present"):
            manifest_text = "manifest not verified"
        else:
            manifest_text = "manifest missing"
        lines = [
            f"imported {len(payload.get('voices') or [])} voice(s) from "
            f"{payload.get('bundle')} ({manifest_text})"
        ]
        for entry in payload.get("voices") or []:
            source = entry.get("ref_text_source")
            source_text = f" ref_text_source={source}" if source else ""
            replaced = " replaced" if entry.get("replaced") else ""
            lines.append(
                f"  {entry['name']}: clone <- btts {entry['btts_kind']} "
                f"({entry['btts_language']} -> {entry['language']})"
                f"{source_text}{replaced}"
            )
        for warning in payload.get("warnings") or []:
            lines.append(f"  warning: {warning}")
        return "\n".join(lines)
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
        source = meta.get("source") or {}
        if source.get("ref_text_source"):
            lines.append(f"  ref_text_source={source['ref_text_source']}")
        if source.get("asr_model"):
            asr = source["asr_model"]
            lines.append(
                f"  asr_model={asr.get('id')} revision={asr.get('revision')}"
            )
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
        if payload.get("checkpoint"):
            return (
                f"removed voice {payload['removed']}; checkpoint left at "
                f"{payload['checkpoint']}"
            )
        return f"removed voice {payload['removed']}"
    if payload.get("cancelled"):
        return "cancelled"
    if payload.get("speaker"):
        return f"builtin:{payload['speaker']} ({payload['model']['id']})"
    return str(payload)
