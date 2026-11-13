"""`qtts voice` — clone/design voice asset library (FR-P0-02/03)."""

from __future__ import annotations

import contextlib
import json
import math
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any, Optional

import yaml

from .. import __version__
from ..admit import check_ref_audio
from ..asr_engine import AsrEngine, clear_asr_cache
from ..audio_io import duration_s, resolve_output_format, write_audio
from ..capabilities import VoiceRef, validate_capabilities
from ..catalog import ModelInfo, find_model, scan, select, select_asr
from ..design_compile import template_sha256
from ..design_yaml import (
    DESIGN_SCHEMA_VERSION,
    MAX_DESIGN_BYTES,
    normalize_document,
    resolve_template_id,
)
from ..errors import InputError, ModelError, UsageError
from ..plan import SAMPLING_DEFAULTS, resolve_dtype_name, resolve_voice
from ..sidecar import build_recipe, recipe_path, write_recipe
from ..textnorm import NORMALIZATION_VERSION
from ..tts_engine import TtsEngine, clear_model_cache
from ..utils import (
    atomic_write_bytes,
    confirm,
    eprint,
    is_tty,
    sha256_bytes,
    sha256_file,
    sha256_json,
    sha256_text,
)
from ..voice_archive import (
    export_bundle,
    export_voice,
    import_bundle as import_voice_bundle,
    import_voice,
)
from ..voice_btts import (
    BttsBundle,
    BundleVoice,
    design_block,
    design_compiler_version,
    design_defaults,
    design_identity,
    design_instruction,
    design_structured,
    design_template,
    load_bundle,
    read_bundle_file,
)
from ..voice_import_journal import (
    commit_import as commit_btts_import,
    has_pending_journal,
    recover_import as recover_btts_import,
)
from ..voice_store import VoiceStore, utc_now, validate_name

DEFAULT_PREVIEW_TEXT = "你好，这是音色试听。Hello, this is a voice preview."
DEFAULT_DESIGN_SAMPLE = "你好，很高兴认识你。"


# --------------------------------------------------------------------------- #
# add
# --------------------------------------------------------------------------- #
def add(cfg, args) -> dict[str, Any]:
    if getattr(args, "auto_ref_text", False) and args.kind != "clone":
        raise UsageError("--auto-ref-text is only valid with --kind clone")
    if getattr(args, "design_yaml", None) and args.kind != "design":
        raise UsageError("--design-yaml is only valid with --kind design")
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
    actual = store.existing_names().get(name.casefold())
    if actual is not None:
        if actual != name:
            raise InputError(
                f"voice name collides case-insensitively with existing {actual!r}: "
                f"{name!r}"
            )
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
    actual = store.existing_names().get(name.casefold())
    if actual is not None:
        if actual != name:
            raise InputError(
                f"voice name collides case-insensitively with existing {actual!r}: "
                f"{name!r}"
            )
        raise InputError(f"voice already exists: {name}")
    description = (args.description or "").strip()
    design_block = None
    design_yaml = getattr(args, "design_yaml", None)
    if design_yaml:
        from ..design_compile import compile_design
        from ..design_lint import lint_design
        from ..design_yaml import load_design

        document, raw_text, yaml_sha = load_design(design_yaml)
        compiled = compile_design(document)
        report = lint_design(document, compiled)
        if not report["ok"]:
            raise UsageError(
                "design YAML has blocking lint issues",
                details={"issues": [
                    item for item in report["issues"] if item["level"] == "block"
                ]},
            )
        if description and description != compiled.rendered_instruction.strip():
            raise UsageError("use either --description or --design-yaml, not both")
        description = compiled.rendered_instruction.strip()
        design_block = {
            "schema_version": document.get("schema_version", 1),
            "template": compiled.template,
            "template_sha256": compiled.template_sha256,
            "structured": document,
            "rendered_instruction": compiled.rendered_instruction,
            "rendered_sha256": compiled.rendered_sha256,
            "compiler_version": compiled.compiler_version,
            "yaml_sha256": yaml_sha,
            "yaml": raw_text,
        }
    if not description:
        raise UsageError("voice add --kind design requires --description TEXT or --design-yaml FILE")
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
                **({"design": design_block} if design_block else {}),
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
_LANGUAGE_ALIASES = {
    "zh": "chinese",
    "cn": "chinese",
    "yue": "cantonese",
    "en": "english",
    "ja": "japanese",
    "jp": "japanese",
    "ko": "korean",
    "de": "german",
    "fr": "french",
    "es": "spanish",
    "it": "italian",
    "pt": "portuguese",
    "ru": "russian",
}


def _language_key(value: Any) -> Optional[str]:
    if value is None:
        return None
    key = str(value).strip().casefold()
    if not key:
        return None
    return _LANGUAGE_ALIASES.get(key, key)


def _name_matches(name: str, query: str, mode: str) -> bool:
    candidate = name.casefold()
    needle = query.casefold()
    if mode == "exact":
        return candidate == needle
    if mode == "prefix":
        return candidate.startswith(needle)
    return needle in candidate


def _voice_record(store: VoiceStore, name: str, meta: dict[str, Any]) -> dict[str, Any]:
    source = meta.get("source") or {}
    design = source.get("design") if isinstance(source.get("design"), dict) else {}
    return {
        "name": name,
        "kind": meta.get("kind"),
        "language": meta.get("language"),
        "created_at": meta.get("created_at"),
        "model": (meta.get("model") or {}).get("id"),
        "materialized": bool(source.get("materialized")),
        "has_prompt": store.prompt_sha256(name) is not None,
        "tags": [str(tag) for tag in (meta.get("tags") or [])],
        "source_kind": source.get("kind") or source.get("type"),
        "design_template": design.get("template") if design else None,
    }


def list_voices(cfg, args) -> dict[str, Any]:
    query = getattr(args, "query", None)
    match = getattr(args, "match", None) or "fuzzy"
    names_only = bool(getattr(args, "names_only", False))
    if names_only and getattr(args, "json", False):
        raise UsageError("--names-only cannot be combined with --json")
    if query is not None:
        query = query.strip()
        if query.startswith("@"):
            query = query[1:].strip()
        if not query:
            raise UsageError("empty voice query")
    kinds = {str(value).casefold() for value in (getattr(args, "kind", None) or [])}
    unknown_kinds = kinds - {"clone", "design", "finetune"}
    if unknown_kinds:
        raise UsageError(f"unknown voice kind: {sorted(unknown_kinds)[0]}")
    tags = {
        tag.casefold()
        for value in (getattr(args, "tag", None) or [])
        for tag in [str(value).strip()] if tag
    }
    language = _language_key(getattr(args, "language", None))
    include_builtin = bool(getattr(args, "include_builtin", False))
    structural = bool(kinds or tags)
    filtered = bool(query or structural or language or names_only)

    models = scan(cfg.model_dir)
    builtin: list[dict[str, Any]] = []
    seen: set[str] = set()
    allow_builtin = not names_only and not structural and (
        include_builtin or language is None
    )
    if allow_builtin:
        for model in models:
            if model.kind != "builtin":
                continue
            model_languages = {_language_key(item) for item in (model.languages or [])}
            if language is not None and language not in model_languages:
                continue
            for speaker in model.speakers:
                key = speaker.casefold()
                if key in seen:
                    continue
                if query is not None and not _name_matches(speaker, query, match):
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
    scan_result = store.scan_entries()
    warnings = list(scan_result["warnings"])
    voices: list[dict[str, Any]] = []
    for entry in scan_result["voices"]:
        if entry["status"] not in ("ok", "mismatch"):
            continue
        meta = entry["meta"] or {}
        name = entry["name"]
        if query is not None and not _name_matches(name, query, match):
            continue
        if kinds and str(meta.get("kind", "")).casefold() not in kinds:
            continue
        entry_tags = {str(tag).casefold() for tag in (meta.get("tags") or [])}
        if tags and not tags.issubset(entry_tags):
            continue
        if language is not None and _language_key(meta.get("language")) != language:
            continue
        voices.append(_voice_record(store, name, meta))

    if names_only:
        names = [entry["name"] for entry in voices]
        return {
            "ok": True,
            "kind": "voice_list",
            "names_only": True,
            "names": names,
            "count": len(names),
            "warnings": warnings,
        }
    return {
        "ok": True,
        "schema_version": 1,
        "kind": "voice_list",
        "query": query,
        "match": match,
        "filters": {
            "tags": sorted(tags),
            "kinds": sorted(kinds),
            "language": language,
            "include_builtin": include_builtin,
        },
        "filtered": filtered,
        "count": len(builtin) + len(voices),
        "builtin": builtin,
        "voices": voices,
        "warnings": warnings,
    }


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
    directory = store.resolve_dir(name)
    if directory is None or not directory.exists():
        raise InputError(f"voice not found: {name}", details={"voice": name})
    warnings: list[str] = []
    if directory.is_symlink():
        warnings.append("voice directory is a symlink")
    meta_path = directory / "voice.json"
    if meta_path.is_symlink():
        warnings.append("voice.json is a symlink")
    if not meta_path.is_file():
        raise InputError(
            f"voice {directory.name} has no voice.json", details={"voice": directory.name}
        )
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise InputError(
            f"corrupt voice metadata: {meta_path}: {exc}",
            details={"voice": directory.name},
        ) from exc
    if not isinstance(meta, dict):
        raise InputError(
            f"invalid voice metadata: {meta_path}", details={"voice": directory.name}
        )
    stored = meta.get("name")
    if stored != directory.name:
        raise InputError(
            f"voice name mismatch: directory {directory.name!r} contains {stored!r}; "
            f"run: qtts voice rename {directory.name} {stored}",
            details={"voice": directory.name, "name": stored},
        )
    if meta.get("kind") == "clone":
        store.validate_reference(meta, directory)

    files = []
    for child in sorted(directory.iterdir()):
        if child.name.startswith(".") or not (child.is_file() or child.is_symlink()):
            continue
        files.append(
            {
                "name": child.name,
                "size": child.stat().st_size if child.is_file() else None,
                "symlink": child.is_symlink(),
            }
        )
        if child.is_symlink():
            warnings.append(f"profile file is a symlink: {child.name}")

    source = meta.get("source") or {}
    design = source.get("design") if isinstance(source.get("design"), dict) else None
    design_yaml = directory / "design.yaml"
    if design_yaml.is_file():
        try:
            design_text = design_yaml.read_text(encoding="utf-8")[:65536]
        except (OSError, UnicodeDecodeError):
            design_text = None
    else:
        design_text = None

    preview = directory / "preview.wav"
    preview_block = source.get("preview") if isinstance(source.get("preview"), dict) else {}
    if design_text is None and isinstance(design, dict):
        embedded = design.get("yaml")
        design_text = embedded[:65536] if isinstance(embedded, str) else None
    payload = {
        "ok": True,
        "kind": meta.get("kind"),
        "voice": meta,
        "directory": directory.name,
        "files": files,
        "prompt_sha256": store.prompt_sha256(directory.name),
        "preview": str(preview) if preview.is_file() else None,
        "preview_origin": preview_block.get("origin"),
        "design": design,
        "design_yaml": design_text,
        "warnings": warnings,
    }
    if meta.get("kind") == "finetune":
        model = meta.get("model") or {}
        path = model.get("path")
        payload["checkpoint"] = {
            "path": path,
            "exists": bool(path) and Path(str(path)).is_dir(),
            "speaker": source.get("speaker"),
            "fingerprint": model.get("fingerprint"),
        }
    return payload


def show(cfg, args) -> dict[str, Any]:
    """``voice show`` is the canonical name; ``inspect`` stays as an alias."""
    return inspect(cfg, args)


def rename(cfg, args) -> dict[str, Any]:
    store = VoiceStore(cfg.data_dir)
    result = store.rename(args.old, args.new, dry_run=bool(args.dry_run))
    return {"ok": True, **result}


def tag(cfg, args) -> dict[str, Any]:
    store = VoiceStore(cfg.data_dir)
    list_only = not (
        args.add or args.remove or args.set or args.clear
    )
    result = store.set_tags(
        args.name,
        add=args.add or (),
        remove=args.remove or (),
        replace=args.set,
        clear=bool(args.clear),
        dry_run=bool(args.dry_run),
    )
    return {"ok": True, "list_only": list_only, **result}


def delete(cfg, args) -> dict[str, Any]:
    store = VoiceStore(cfg.data_dir)
    if not args.dry_run and not args.yes:
        if not is_tty():
            raise UsageError("voice delete requires --yes in non-interactive mode")
        if not confirm(f"Delete voice {args.name}?"):
            return {"ok": True, "cancelled": True, "voice": args.name}
    try:
        meta = store.get(args.name)
    except InputError:
        meta = None
    result = store.delete(args.name, dry_run=bool(args.dry_run))
    payload: dict[str, Any] = {"ok": True, **result}
    if not args.dry_run:
        payload["removed"] = args.name
        if meta and meta.get("kind") == "finetune":
            payload["checkpoint"] = (meta.get("model") or {}).get("path")
    return payload


# --------------------------------------------------------------------------- #
# preview / rm
# --------------------------------------------------------------------------- #
def preview(cfg, args) -> dict[str, Any]:
    text = args.text or DEFAULT_PREVIEW_TEXT
    models = scan(cfg.model_dir)
    store = VoiceStore(cfg.data_dir)
    name = args.name
    no_save = bool(getattr(args, "no_save", False))
    if no_save and not args.out:
        raise UsageError("--no-save requires --out")
    if getattr(args, "format", None) == "flac" and not args.out:
        raise UsageError("--format flac requires --out ending in .flac")
    if no_save and args.out:
        profile = store.resolve_dir(name)
        if profile is not None:
            target = Path(args.out).expanduser().resolve()
            if target == (profile / "preview.wav").resolve():
                raise UsageError("--no-save cannot write the profile preview.wav")
    if args.out:
        # Validate suffix/format agreement before any model loads.
        resolve_output_format(Path(args.out), getattr(args, "format", None))

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
    if (
        builtin_model is None
        and out.name == "preview.wav"
        and out.parent == store.dir_for(name)
        and out.is_file()
        and not no_save
    ):
        eprint(f"overwriting preview: {out}")
    write_audio(out, audio, sr, fmt)
    if (
        not no_save
        and builtin_model is None
        and out.name == "preview.wav"
        and out.parent == store.dir_for(name)
    ):
        store.set_preview(name, out)
        meta_for_preview = store.get(name)
        if meta_for_preview is not None:
            source_for_preview = dict(meta_for_preview.get("source") or {})
            source_for_preview["preview"] = {
                "origin": "qtts",
                "file": "preview.wav",
                "sha256": sha256_file(out),
                "updated_at": utc_now(),
            }
            meta_for_preview["source"] = source_for_preview
            store.save_meta(meta_for_preview)

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
    if args.all:
        if args.name:
            raise UsageError("--all cannot be combined with a voice NAME")
        if args.out:
            raise UsageError("--all uses --out-dir, not --out")
        if not args.out_dir:
            raise UsageError("--all requires --out-dir DIRECTORY")
        return export_bundle(
            store,
            args.out_dir,
            force=bool(args.force),
            dry_run=bool(getattr(args, "dry_run", False)),
        )
    if args.out_dir:
        raise UsageError("--out-dir requires --all")
    if not args.name:
        raise UsageError("voice export NAME -o FILE, or --all --out-dir DIR")
    if not args.out:
        raise UsageError("voice export NAME requires -o/--out FILE")
    return export_voice(
        store,
        args.name,
        args.out,
        force=bool(args.force),
        dry_run=bool(getattr(args, "dry_run", False)),
    )


def import_(cfg, args) -> dict[str, Any]:
    store = VoiceStore(cfg.data_dir)
    return import_voice(store, args.archive, name=args.name, force=bool(args.force))


def import_bundle(cfg, args) -> dict[str, Any]:
    store = VoiceStore(cfg.data_dir)
    return import_voice_bundle(
        store,
        args.bundle_dir,
        force=bool(args.force),
        dry_run=bool(getattr(args, "dry_run", False)),
    )


# --------------------------------------------------------------------------- #
# import-btts
# --------------------------------------------------------------------------- #
def _btts_design_mode(args) -> str:
    explicit = str(getattr(args, "design_mode", "") or "").strip().lower()
    clone_flags = bool(
        getattr(args, "auto_ref_text", False)
        or getattr(args, "ref_text", None) is not None
        or getattr(args, "asr_model", None)
        or getattr(args, "mode", None) is not None
        or getattr(args, "model", None)
    )
    if explicit:
        if explicit not in ("auto", "recipe", "clone"):
            raise UsageError("--design-mode must be auto, recipe or clone")
        if explicit in ("auto", "recipe") and clone_flags:
            raise UsageError(
                f"--design-mode {explicit} cannot be combined with clone-only "
                "options; use --design-mode clone"
            )
        return explicit
    return "clone" if clone_flags else "auto"


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

    design_mode = _btts_design_mode(args)
    clone_mode = (args.mode or "icl").lower()
    auto = bool(getattr(args, "auto_ref_text", False)) if design_mode == "clone" else False
    explicit_text = args.ref_text if design_mode == "clone" else None
    if design_mode == "clone":
        if auto and explicit_text is not None:
            raise UsageError("--auto-ref-text cannot be combined with --ref-text")
        if auto and clone_mode != "icl":
            raise UsageError("--auto-ref-text requires --mode icl")
        if getattr(args, "asr_model", None) and not auto:
            raise UsageError("--asr-model requires --auto-ref-text")
        if explicit_text is not None and clone_mode != "icl":
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
    if has_pending_journal(store):
        if getattr(args, "dry_run", False):
            recover_btts_import(store, dry_run=True)
        else:
            recover_btts_import(store)

    existing = store.existing_names()
    targets: list[tuple[BundleVoice, str]] = []
    seen_targets: dict[str, str] = {}
    replace_names: dict[str, str] = {}
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
            replace_names[target] = on_disk
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

    plans = [
        (
            _btts_recipe_plan(voice, target, bundle=bundle)
            if voice.kind == "design" and design_mode in ("auto", "recipe")
            else _btts_plan(
                voice,
                target,
                bundle=bundle,
                mode=clone_mode or "icl",
                auto=auto,
                explicit_text=explicit_text,
            )
        )
        for voice, target in targets
    ]
    clone_plans = [plan for plan in plans if plan["design_mode"] == "clone"]
    recipe_plans = [plan for plan in plans if plan["design_mode"] == "recipe"]

    models = scan(cfg.model_dir) if clone_plans and not getattr(args, "dry_run", False) else []
    model = None
    if clone_plans and not getattr(args, "dry_run", False):
        model = (
            find_model(models, args.model)
            if getattr(args, "model", None)
            else select(models, "clone")
        )
        for plan in clone_plans:
            validate_capabilities(
                model,
                mode="clone",
                language=plan["voice"].language,
                icl=(clone_mode == "icl" and plan["ref_text"] is not None),
                ref_text=plan["ref_text"],
            )

    if getattr(args, "dry_run", False):
        warnings = list(bundle.warnings)
        for plan in plans:
            plan["warnings"] = list(plan.get("warnings") or [])
            warnings.extend(plan["warnings"])
            if plan["design_mode"] == "recipe":
                warnings.append(
                    "acoustic-equivalence-not-possible: btts uses Breeze TTS 2; "
                    "qtts uses Qwen3-TTS VoiceDesign"
                )
        with tempfile.TemporaryDirectory(prefix="qtts-btts-dry-") as tmp:
            for plan in clone_plans:
                if clone_mode == "icl" and plan["ref_text"] is None:
                    plan["warnings"].append("asr-skipped-in-dry-run")
                    if "asr-skipped-in-dry-run" not in warnings:
                        warnings.append("asr-skipped-in-dry-run")
                    continue
                voice = plan["voice"]
                ref_rel = (
                    f"{voice.name}/{voice.ref_audio}"
                    if voice.kind == "clone"
                    else voice.preview
                )
                staged_ref = Path(tmp) / f"{plan['target']}.wav"
                atomic_write_bytes(staged_ref, read_bundle_file(bundle, ref_rel))
                report = check_ref_audio(
                    staged_ref,
                    ref_text=plan["ref_text"] if clone_mode == "icl" else None,
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
        return {
            "ok": True,
            "kind": "btts-import",
            "dry_run": True,
            "bundle": str(bundle.path),
            "bundle_format": bundle.format,
            "design_mode": design_mode,
            "manifest": {
                "present": bundle.manifest_present,
                "verified": bundle.manifest_verified,
                "sha256": bundle.manifest_sha256,
            },
            "voices": [
                _btts_result(plan, replaced=False, store=store)
                for plan in plans
            ],
            "replaced": False,
            "warnings": warnings,
        }

    asr_model = (
        select_asr(models, getattr(args, "asr_model", None))
        if auto and any(plan["voice"].kind == "design" for plan in clone_plans)
        else None
    )
    asr = None
    staged: list[tuple[str, Path]] = []
    staging_by_target: dict[str, Path] = {}
    replaced: set[str] = set()
    try:
        for plan in recipe_plans:
            voice = plan["voice"]
            target = plan["target"]
            staging = store.staging_dir(target)
            staged.append((target, staging))
            staging_by_target[target] = staging
            plan["meta"] = _stage_btts_recipe(
                cfg,
                store,
                plan,
                bundle=bundle,
                force=bool(args.force),
                staging=staging,
            )

        for plan in clone_plans:
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
                ref_text=plan["ref_text"] if clone_mode == "icl" else None,
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
            assert model is not None
            validate_capabilities(
                model,
                mode="clone",
                language=voice.language,
                icl=(clone_mode == "icl"),
                ref_text=plan["ref_text"] if clone_mode == "icl" else None,
            )

        asr = None
        if auto:
            clear_asr_cache()

        if clone_plans:
            assert model is not None
            engine = TtsEngine(cfg, model)
            engine.load()
            for plan in clone_plans:
                voice = plan["voice"]
                target = plan["target"]
                staging = staging_by_target[target]
                source = {
                    "type": "clone",
                    "origin": "btts",
                    "mode": clone_mode,
                    "ref_text": plan["ref_text"] if clone_mode == "icl" else None,
                    "ref_text_source": (
                        plan["ref_text_source"] if clone_mode == "icl" else None
                    ),
                    "btts": _btts_provenance(voice, bundle),
                }
                plan["meta"] = _freeze_clone_prompt(
                    cfg,
                    store,
                    name=target,
                    staged_ref=staging / "ref.wav",
                    ref_text=plan["ref_text"],
                    mode=clone_mode or "icl",
                    language=voice.language,
                    model=model,
                    engine=engine,
                    source=source,
                    consent=_btts_consent(voice, bundle),
                    staging=staging,
                    tags=_btts_tags(voice),
                    preview_src=staging / "preview.wav",
                    asr_model=(
                        asr_model if plan["ref_text_source"] == "asr" else None
                    ),
                    allow_existing=bool(args.force),
                )

        replaced = commit_btts_import(
            store,
            [
                (target, staging, replace_names.get(target))
                for target, staging in staged
            ],
        )
        staged = []
    finally:
        # If commit_import left a journal, recovery may still need the staging
        # directory to distinguish "replace not attempted" from "installed".
        if not has_pending_journal(store):
            for _, staging in staged:
                shutil.rmtree(staging, ignore_errors=True)
        if auto:
            clear_asr_cache()

    warnings = list(bundle.warnings)
    for plan in plans:
        warnings.extend(plan.get("warnings") or [])
        if plan["design_mode"] == "recipe":
            warnings.append(
                "acoustic-equivalence-not-possible: btts uses Breeze TTS 2; "
                "qtts uses Qwen3-TTS VoiceDesign"
            )
    return {
        "ok": True,
        "kind": "btts-import",
        "dry_run": False,
        "bundle": str(bundle.path),
        "bundle_format": bundle.format,
        "design_mode": design_mode,
        "manifest": {
            "present": bundle.manifest_present,
            "verified": bundle.manifest_verified,
            "sha256": bundle.manifest_sha256,
        },
        "voices": [
            _btts_result(plan, replaced=plan["target"] in replaced, store=store)
            for plan in plans
        ],
        "replaced": bool(replaced),
        "warnings": warnings,
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
        "design_mode": "clone",
        "ref_text": ref_text,
        "ref_text_source": ref_text_source,
        "warnings": [],
    }


def _btts_recipe_plan(
    voice: BundleVoice, target: str, *, bundle: BttsBundle
) -> dict[str, Any]:
    instruction = design_instruction(voice)
    if not instruction:
        raise InputError(
            f"design voice {voice.name} has no instruction; cannot import a recipe",
            details={"bundle": str(bundle.path), "voice": voice.name},
        )
    return {
        "voice": voice,
        "target": target,
        "design_mode": "recipe",
        "instruction": instruction,
        "ref_text": None,
        "ref_text_source": None,
        "warnings": [],
    }


def _read_btts_design_yaml(
    voice: BundleVoice, bundle: BttsBundle
) -> tuple[Optional[str], Optional[dict[str, Any]], list[str]]:
    if not voice.design_yaml:
        return None, None, []
    warnings: list[str] = []
    try:
        raw = read_bundle_file(bundle, voice.design_yaml)
    except InputError as exc:
        return None, None, [f"invalid-design-yaml: {exc}"]
    if len(raw) > MAX_DESIGN_BYTES:
        return None, None, [
            f"invalid-design-yaml: design.yaml is larger than {MAX_DESIGN_BYTES} bytes"
        ]
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        return None, None, [f"invalid-design-yaml: {exc}"]
    try:
        document = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        return None, None, [f"invalid-design-yaml: {exc}"]
    if not isinstance(document, dict):
        return text, None, ["invalid-design-yaml: design.yaml must be a mapping"]
    try:
        normalized = normalize_document(document)
    except Exception as exc:  # normalize is defensive; keep the recipe usable
        return text, None, [f"invalid-design-yaml: {exc}"]
    if normalized.get("errors"):
        warnings.append(
            "invalid-design-yaml: " + "; ".join(str(item) for item in normalized["errors"])
        )
    return text, normalized, warnings


def _btts_tags(voice: BundleVoice) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for tag in [*voice.tags, "btts"]:
        key = tag.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(tag)
    return out


def _stage_btts_recipe(
    cfg,
    store: VoiceStore,
    plan: dict[str, Any],
    *,
    bundle: BttsBundle,
    force: bool,
    staging: Path,
) -> dict[str, Any]:
    voice = plan["voice"]
    instruction = plan["instruction"]
    yaml_text, normalized, warnings = _read_btts_design_yaml(voice, bundle)
    plan["warnings"].extend(warnings)

    structured = design_structured(voice)
    covers: list[str] = []
    if normalized:
        covers = [str(item) for item in (normalized.get("covers") or []) if str(item).strip()]
        if not structured:
            structured = {
                key: value
                for key, value in normalized.items()
                if key not in ("extras", "errors", "present_fields")
            }
        if normalized.get("name") and normalized.get("name") != voice.name:
            plan["warnings"].append(
                f"design-yaml-mismatch: name {normalized.get('name')!r} != {voice.name!r}"
            )
        normalized_language = str(normalized.get("language") or "").strip().lower()
        if normalized_language and normalized_language != voice.btts_language.strip().lower():
            plan["warnings"].append(
                "design-yaml-mismatch: language "
                f"{normalized_language!r} != {voice.btts_language!r}"
            )
    if not covers and isinstance(structured.get("covers"), list):
        covers = [str(item) for item in structured["covers"] if str(item).strip()]
    sample_text = covers[0] if covers else DEFAULT_DESIGN_SAMPLE

    raw_template = design_template(voice)
    if not raw_template and normalized:
        raw_template = normalized.get("template")
    template = resolve_template_id(raw_template)
    if raw_template and template is None:
        plan["warnings"].append(f"unsupported-btts-template: {raw_template}")
    template_hash = template_sha256(template) if template else None

    identity = design_identity(voice)
    rendered_by_identity = identity.get("rendered_instruction")
    design = design_block(voice)
    defaults = design_defaults(voice)
    compiler_version = design_compiler_version(voice)
    preview = read_bundle_file(bundle, voice.preview)
    preview_sha = sha256_bytes(preview)
    btts = _btts_provenance(voice, bundle)

    design_source: dict[str, Any] = {
        "schema_version": DESIGN_SCHEMA_VERSION,
        "template": template,
        "template_id": raw_template,
        "template_sha256": template_hash,
        "structured": structured,
        "rendered_instruction": instruction,
        "rendered_sha256": sha256_text(instruction),
        "compiler_family": "btts" if compiler_version else None,
        "compiler_version": compiler_version,
        "yaml": yaml_text,
        "yaml_sha256": sha256_text(yaml_text) if yaml_text is not None else None,
        "seed": _bounded_int(design.get("seed")),
        "cfg_scale": _bounded_number(design.get("cfg_scale")),
        "max_new_tokens": _bounded_int(design.get("max_new_tokens")),
    }
    if (
        isinstance(rendered_by_identity, str)
        and rendered_by_identity.strip()
        and rendered_by_identity.strip() != instruction
    ):
        design_source["identity_rendered_instruction"] = rendered_by_identity.strip()
        plan["warnings"].append("instruction-mismatch")

    source = {
        "type": "design",
        "origin": "btts",
        "description": instruction,
        "sample_text": sample_text,
        "language": voice.language,
        "materialized": False,
        "preview": {"origin": "btts", "file": "preview.wav", "sha256": preview_sha},
        "design": design_source,
        "btts": btts,
    }
    model_id = ((btts.get("model") or {}) if isinstance(btts.get("model"), dict) else {}).get("id")
    meta = store.create(
        name=plan["target"],
        kind="design",
        source=source,
        model={
            "id": None,
            "source": "btts",
            "btts_model": model_id,
            "note": "qtts VoiceDesign model is selected at say/preview time",
        },
        dtype=None,
        language=voice.language,
        default_instruct=instruction,
        tags=_btts_tags(voice),
        consent=_btts_consent(voice, bundle),
        qtts_version=__version__,
        staging=staging,
        allow_existing=force,
    )
    atomic_write_bytes(staging / "preview.wav", preview)
    return meta


def _btts_result(
    plan: dict[str, Any], *, replaced: bool, store: VoiceStore
) -> dict[str, Any]:
    voice = plan["voice"]
    meta = plan.get("meta") or {}
    source = meta.get("source") if isinstance(meta.get("source"), dict) else {}
    design = source.get("design") if isinstance(source.get("design"), dict) else {}
    instruction = (
        plan.get("instruction")
        or source.get("description")
        or design_instruction(voice)
        or ""
    )
    raw_template = design.get("template_id") or design_template(voice)
    canonical_template = design.get("template") or resolve_template_id(raw_template)
    return {
        "name": plan["target"],
        "btts_name": voice.name,
        "kind": "design" if plan["design_mode"] == "recipe" else "clone",
        "btts_kind": voice.kind,
        "design_mode": plan["design_mode"],
        "template": canonical_template,
        "btts_template": raw_template,
        "template_mapped": bool(canonical_template),
        "instruction_sha256": sha256_text(instruction) if instruction else None,
        "language": voice.language,
        "btts_language": voice.btts_language,
        "profile_sha256": voice.profile_sha256,
        "ref_text_source": (
            plan["ref_text_source"]
            if plan["design_mode"] == "clone" and plan.get("ref_text_source")
            else None
        ),
        "preview_origin": "btts",
        "metadata_compatible": True,
        "acoustic_equivalence": False,
        "seed_replayed": False,
        "cfg_replayed": False,
        "replaced": replaced,
        "output": str(store.dir_for(plan["target"])),
        "files": sorted(voice.files),
        "warnings": list(plan.get("warnings") or []),
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
    note = (
        f"imported from btts bundle {bundle.path.name}; the btts profile "
        "consent is provenance only and was not used as qtts consent"
    )
    if isinstance(license_model, str) and license_model:
        note += f"; btts model license: {license_model}"
    return {
        "rights_confirmed": True,
        "license": "btts-import",
        "note": note,
    }


def _btts_provenance(voice: BundleVoice, bundle: BttsBundle) -> dict[str, Any]:
    profile = voice.profile
    reference = (
        profile.get("reference")
        if isinstance(profile.get("reference"), dict)
        else None
    )
    design = design_block(voice)
    provenance = (
        profile.get("provenance")
        if isinstance(profile.get("provenance"), dict)
        else {}
    )
    identity = design_identity(voice)
    license_block = profile.get("license") if isinstance(profile.get("license"), dict) else {}
    defaults = design_defaults(voice)
    consent = profile.get("consent")
    truncations: dict[str, dict[str, Any]] = {}

    def text(value: Any, limit: int, field: str) -> Optional[str]:
        if isinstance(value, str):
            stripped = value.strip()
            if len(stripped) > limit:
                truncations[field] = {
                    "truncated": True,
                    "limit": limit,
                    "length": len(stripped),
                    "sha256": sha256_text(stripped),
                }
        return _bounded_str(value, limit)

    def data(value: Any, limit: int, field: str) -> Any:
        try:
            encoded = json.dumps(value, ensure_ascii=False, sort_keys=True)
        except (TypeError, ValueError):
            encoded = None
        if encoded is not None and len(encoded) > limit:
            truncations[field] = {
                "truncated": True,
                "limit": limit,
                "length": len(encoded),
                "sha256": sha256_text(encoded),
            }
        return _bounded_json(value, limit)

    out: dict[str, Any] = {
        "bundle": bundle.path.name,
        "bundle_format": bundle.format,
        "bundle_root_prefix": bundle.root_prefix,
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
        "license": data(license_block, 4096, "license"),
        "model": {
            "id": text(provenance.get("model_id"), 256, "model.id"),
            "digest": text(provenance.get("model_digest"), 256, "model.digest"),
        },
        "provenance": {
            "source": text(provenance.get("source"), 64, "provenance.source"),
            "source_take_id": text(
                provenance.get("source_take_id"), 256, "provenance.source_take_id"
            ),
            "source_session_id": text(
                provenance.get("source_session_id"), 256, "provenance.source_session_id"
            ),
            "api_protocol": text(
                provenance.get("api_protocol"), 128, "provenance.api_protocol"
            ),
            "cli_version": text(
                provenance.get("cli_version"), 64, "provenance.cli_version"
            ),
            "model_id": text(
                provenance.get("model_id"), 256, "provenance.model_id"
            ),
            "model_digest": text(
                provenance.get("model_digest"), 256, "provenance.model_digest"
            ),
        },
        "identity": {
            "structured": data(
                identity.get("structured"), 16384, "identity.structured"
            ),
            "rendered_instruction": text(
                identity.get("rendered_instruction"),
                4096,
                "identity.rendered_instruction",
            ),
            "template": text(identity.get("template"), 256, "identity.template"),
            "compiler_family": (
                "btts" if design_compiler_version(voice) else None
            ),
            "compiler_version": text(
                identity.get("compiler_version"), 64, "identity.compiler_version"
            ),
        },
        "design": {
            "instruction": text(
                design.get("instruction"), 4096, "design.instruction"
            ),
            "seed": _bounded_int(design.get("seed")),
            "cfg_scale": _bounded_number(design.get("cfg_scale")),
            "max_new_tokens": _bounded_int(design.get("max_new_tokens")),
            "design_yaml": "design.yaml" if voice.design_yaml else None,
            "design_yaml_sha256": (
                voice.files["design.yaml"].sha256 if voice.design_yaml else None
            ),
        },
        "defaults": {
            "seed": _bounded_int(defaults.get("seed")),
            "cfg_scale": _bounded_number(defaults.get("cfg_scale")),
            "direction": text(defaults.get("direction"), 512, "defaults.direction"),
        },
        "files": {
            filename: ref.sha256 for filename, ref in sorted(voice.files.items())
        },
        "preview_sha256": (
            voice.files["preview.wav"].sha256
            if "preview.wav" in voice.files
            else None
        ),
    }
    if reference is not None:
        out["reference"] = {
            "audio": text(reference.get("audio"), 128, "reference.audio"),
            "audio_sha256": text(
                reference.get("audio_sha256"), 128, "reference.audio_sha256"
            ),
            "transcript": text(
                reference.get("transcript"), 128, "reference.transcript"
            ),
            "transcript_sha256": text(
                reference.get("transcript_sha256"),
                128,
                "reference.transcript_sha256",
            ),
            "transcript_confirmed": bool(reference.get("transcript_confirmed")),
            "duration_s": _bounded_number(reference.get("duration_s")),
            "sample_rate": _bounded_int(reference.get("sample_rate")),
        }
    if isinstance(consent, dict):
        out["consent"] = data(consent, 4096, "consent")
    elif consent is not None:
        out["consent"] = {"present": True, "truncated": True}
        truncations["consent"] = {
            "truncated": True,
            "limit": 4096,
            "length": None,
            "sha256": None,
        }
    if truncations:
        out["truncations"] = truncations
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
    if payload.get("names_only"):
        names = payload.get("names") or []
        return "\n".join(names) if names else "no matching voices."
    if "bundle_dir" in payload:
        verb = "would import" if payload.get("dry_run") else "imported"
        lines = [
            f"{verb} {len(payload.get('voices') or [])} voice(s) from "
            f"{payload['bundle_dir']}"
        ]
        for entry in payload.get("voices") or []:
            action = "replaced" if entry.get("replaced") else "imported"
            lines.append(f"  {entry['name']}: {action}")
        for warning in payload.get("warnings") or []:
            lines.append(f"  warning: {warning}")
        return "\n".join(lines)
    if payload.get("bundle") and "voices" in payload and "out_dir" in payload:
        verb = "would export" if payload.get("dry_run") else "exported"
        lines = [
            f"{verb} {len(payload['voices'])} voice(s) -> {payload['out_dir']} "
            f"({payload['bundle']})"
        ]
        for entry in payload["voices"]:
            lines.append(f"  {entry['name']} ({entry['kind']}): {entry['archive']}")
        for warning in payload.get("warnings") or []:
            lines.append(f"  warning: {warning}")
        return "\n".join(lines)
    if "old" in payload and "new" in payload:
        verb = "would rename" if payload.get("dry_run") else "renamed"
        return f"{verb} voice {payload['old']} -> {payload['new']}"
    if "before" in payload and "after" in payload:
        if payload.get("list_only"):
            return f"{payload['voice']}: {', '.join(payload['before']) or '(no tags)'}"
        if not payload.get("changed"):
            return f"{payload['voice']}: tags unchanged ({', '.join(payload['after']) or 'none'})"
        verb = "would set" if payload.get("dry_run") else "set"
        return (
            f"{verb} tags for {payload['voice']}: "
            f"{', '.join(payload['after']) or '(none)'}"
        )
    if payload.get("archive") and payload.get("dry_run"):
        meta = payload.get("voice") or {}
        return (
            f"would export voice {meta.get('name')} -> {payload['archive']} "
            f"({len(payload.get('files') or [])} files)"
        )
    if payload.get("dry_run") and payload.get("path") and "voice" in payload:
        return f"would delete voice {payload['voice']} ({payload['path']})"
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
        verb = "would import" if payload.get("dry_run") else "imported"
        lines = [
            f"{verb} {len(payload.get('voices') or [])} voice(s) from "
            f"{payload.get('bundle')} ({manifest_text})"
        ]
        for entry in payload.get("voices") or []:
            source = entry.get("ref_text_source")
            source_text = f" ref_text_source={source}" if source else ""
            replaced = " replaced" if entry.get("replaced") else ""
            mode = entry.get("design_mode") or entry.get("kind") or "clone"
            detail = f", template {entry['template']}" if entry.get("template") else ""
            lines.append(
                f"  {entry['name']}: {entry.get('kind')} <- btts "
                f"{entry['btts_kind']} ({entry['btts_language']} -> "
                f"{entry['language']}, {mode}{detail})"
                f"{source_text}{replaced}"
            )
        for warning in payload.get("warnings") or []:
            lines.append(f"  warning: {warning}")
        return "\n".join(lines)
    if "builtin" in payload and "voices" in payload:
        lines: list[str] = []
        if payload.get("filtered"):
            for entry in payload["builtin"]:
                lines.append(
                    f"  {entry['name']:10s} builtin  {','.join(entry['languages'][:4])}"
                )
            for entry in payload["voices"]:
                lines.append(
                    f"  {entry['name']:10s} kind={entry['kind']} lang={entry['language']} "
                    f"tags={','.join(entry.get('tags') or []) or '-'}"
                )
            if not lines:
                lines.append("no matching voices.")
        else:
            lines.append("built-in voices:")
            for entry in payload["builtin"]:
                lines.append(
                    f"  {entry['name']:10s} {entry['model']}  "
                    f"{','.join(entry['languages'][:4])}"
                )
            lines.append("user voices:")
            if not payload["voices"]:
                lines.append("  (none)")
            for entry in payload["voices"]:
                lines.append(
                    f"  {entry['name']:10s} kind={entry['kind']} lang={entry['language']} "
                    f"prompt={'yes' if entry['has_prompt'] else 'no'}"
                )
        for warning in payload.get("warnings") or []:
            lines.append(f"  warning: {warning}")
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
        files = payload.get("files") or []
        if files:
            names = ", ".join(entry["name"] for entry in files)
            lines.append(f"  files: {names}")
        if payload.get("prompt_sha256"):
            lines.append(f"  prompt_sha256: {payload['prompt_sha256']}")
        if payload.get("preview"):
            lines.append(f"  preview: {payload['preview']}")
        if payload.get("design") is not None:
            lines.append("  design: embedded")
        if payload.get("design_yaml"):
            lines.append(f"  design.yaml: {len(payload['design_yaml'])} bytes")
        for warning in payload.get("warnings") or []:
            lines.append(f"  warning: {warning}")
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
