"""`qtts design` — structured voice-design workflow (offline commands first)."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from .. import __version__
from ..audio_io import write_audio
from ..completeness import assess
from ..design_compile import CompiledDesign, compile_design
from ..design_lint import lint_design
from ..design_yaml import (
    SUPPORTED_LANGUAGES,
    TEMPLATES,
    dump_design,
    example_document,
    load_design,
    template_for_language,
    template_language,
)
from ..errors import UsageError
from ..utils import atomic_write_json, atomic_write_text, read_json, sha256_file, sha256_text
from ..voice_store import validate_name


def init(cfg, args) -> dict[str, Any]:
    path = Path(args.file).expanduser()
    language = str(args.language).strip().lower() if args.language else None
    template = args.template
    if template is not None:
        if template not in TEMPLATES:
            raise UsageError(f"unknown design template: {template!r}")
        template_language_value = template_language(template)
        if language and template_language_value != language:
            raise UsageError(
                f"--template {template} is {template_language_value}, not {language}"
            )
        language = template_language_value or language or "zh"
    else:
        language = language or "zh"
        template = template_for_language(language)
        if template is None:
            supported = ", ".join(sorted(SUPPORTED_LANGUAGES))
            raise UsageError(
                f"unsupported design language: {language!r}; use {supported}"
            )
    name = args.name or path.stem
    validate_name(name)
    if path.exists() and not args.force:
        raise UsageError(f"design file already exists: {path}; pass --force to overwrite")

    document = example_document(name, language=language, template=template)
    text = dump_design(document)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, text)
    return {
        "ok": True,
        "file": str(path),
        "name": name,
        "language": language,
        "template": template,
        "written": True,
    }


def _compile_or_placeholder(document: dict[str, Any]) -> CompiledDesign:
    if document.get("template") in TEMPLATES:
        return compile_design(document)
    rendered = str(
        document.get("instruction_override")
        or document.get("description")
        or ""
    ).strip()
    return CompiledDesign(
        structured=document,
        rendered_instruction=rendered,
        rendered_sha256=sha256_text(rendered),
        template=str(document.get("template") or ""),
        template_sha256="",
        compiler_version="1",
        language=str(document.get("language") or "zh"),
        warnings=[],
    )


def lint(cfg, args) -> dict[str, Any]:
    document, raw_text, _ = load_design(args.file)
    compiled = _compile_or_placeholder(document)
    report = lint_design(document, compiled)
    report["source"] = {
        "path": str(Path(args.file).expanduser()),
        "sha256": sha256_text(raw_text),
        "bytes": len(raw_text.encode("utf-8")),
    }
    report["show_rendered"] = bool(getattr(args, "show_rendered", False))
    strict = bool(getattr(args, "strict", False))
    has_blocks = not report["ok"]
    has_warnings = report["summary"]["warnings"] > 0
    report["exit_code"] = 1 if has_blocks or (strict and has_warnings) else 0
    return report


def _parse_seeds(value: Any) -> list[int] | None:
    if not value:
        return None
    seeds = []
    for part in str(value).split(","):
        part = part.strip()
        if not part:
            raise UsageError("--seeds must be a comma-separated list of positive integers")
        try:
            seed = int(part)
        except ValueError as exc:
            raise UsageError(f"invalid seed: {part!r}") from exc
        if seed <= 0:
            raise UsageError("seeds must be positive integers")
        seeds.append(seed)
    return seeds


def explore(cfg, args) -> dict[str, Any]:
    from ..design_run import build_matrix, check_resume

    document, raw_text, _ = load_design(args.file)
    compiled = _compile_or_placeholder(document)
    lint_report = lint_design(document, compiled)
    if not lint_report["ok"]:
        raise UsageError(
            "design lint has blocking issues; run `qtts design lint` and fix them",
            details={"issues": [
                item for item in lint_report["issues"] if item["level"] == "block"
            ]},
        )
    if args.max_new_tokens <= 0:
        raise UsageError("--max-new-tokens must be positive")

    out = Path(args.out).expanduser()
    matrix = build_matrix(
        document,
        seeds=_parse_seeds(args.seeds),
        sampling=args.sampling,
        wide=bool(args.wide),
        max_new_tokens=int(args.max_new_tokens),
        rng_seed=int(
            args.seed if args.seed is not None else (getattr(cfg, "seed", 0) or 0)
        ),
        allow_large=bool(args.yes),
    )

    manifest_path = out / "run.json"
    nonempty = out.exists() and any(
        child.name != ".lock" for child in out.iterdir()
    )
    if nonempty and not args.resume:
        if manifest_path.is_file():
            raise UsageError(
                f"design run already exists: {out}; pass --resume or choose another --out"
            )
        raise UsageError(f"output directory is not empty: {out}")
    if nonempty and args.resume and not manifest_path.is_file():
        raise UsageError(
            f"output directory is not empty and has no run.json: {out}"
        )
    if args.resume and manifest_path.is_file():
        check_resume(
            read_json(manifest_path),
            compiled=compiled,
            raw_text=raw_text,
            matrix=matrix,
        )

    preview = {
        "ok": True,
        "kind": "design_explore",
        "dry_run": bool(args.dry_run),
        "file": str(Path(args.file).expanduser()),
        "out": str(out),
        "name": document.get("name"),
        "template": compiled.template,
        "language": compiled.language,
        "rendered_instruction": compiled.rendered_instruction,
        "lint": {
            "grade": lint_report["grade"],
            "total": lint_report["total"],
            "blocks": lint_report["summary"]["blocks"],
            "warnings": lint_report["summary"]["warnings"],
        },
        "matrix": matrix,
        "candidate_count": len(matrix["candidates"]),
    }
    if args.dry_run:
        return preview

    from ..capabilities import validate_capabilities
    from ..catalog import find_model, scan, select
    from ..design_run import RunLock, prepare_run, save_manifest
    from ..tts_engine import TtsEngine

    language_name = _language_name(compiled.language)
    rendered = compiled.rendered_instruction.strip()
    models = scan(cfg.model_dir)
    model_override = getattr(args, "model", None)
    model = find_model(models, model_override) if model_override else select(models, "design")
    validate_capabilities(
        model,
        mode="design",
        description=rendered,
        language=language_name,
    )
    model_payload = {
        "id": model.model_id,
        "revision": model.revision,
        "fingerprint": model.fingerprint,
        "path": str(model.path),
    }

    with RunLock(out):
        manifest = prepare_run(
            out,
            document=document,
            raw_text=raw_text,
            compiled=compiled,
            matrix=matrix,
            model=model_payload,
            resume=bool(args.resume),
            source_path=str(Path(args.file).expanduser()),
        )
        matrix_by_id = {item["take_id"]: item for item in matrix["candidates"]}
        warnings: list[str] = []
        engine = TtsEngine(cfg, model)
        engine.load()
        manifest.setdefault("model", {})["dtype"] = engine.dtype_name
        interrupted = False
        for record in manifest.get("candidates") or []:
            take_id = record.get("take_id")
            candidate = matrix_by_id.get(take_id)
            if candidate is None:
                continue
            if not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", str(take_id or "")):
                record.update(status="failed", error="invalid take_id",
                              completion_status="failed")
                save_manifest(out, manifest)
                continue
            if record.get("pruned"):
                continue
            wav_rel = f"candidates/{take_id}.wav"
            recipe_rel = f"candidates/{take_id}.recipe.json"
            record["wav"] = wav_rel
            record["recipe"] = recipe_rel
            if record.get("status") == "ok":
                wav = out / wav_rel
                if (
                    wav.is_file()
                    and record.get("sha256")
                    and record.get("sha256") == sha256_file(wav)
                ):
                    continue
                warnings.append(f"{take_id}: audio changed since generation; regenerating")
            elif record.get("status") == "failed" and not args.retry_failed:
                continue
            try:
                audio, sr, info = engine.generate(
                    text=candidate["text"],
                    language=language_name,
                    mode="design",
                    instruct=rendered,
                    sampling={
                        **candidate["sampling"],
                        "max_new_tokens": matrix["max_new_tokens"],
                    },
                    seed=int(candidate["seed"]),
                )
            except KeyboardInterrupt:
                interrupted = True
                break
            except Exception as exc:
                record.update(status="failed", error=f"{type(exc).__name__}: {exc}",
                              completion_status="failed")
                save_manifest(out, manifest)
                continue

            wav_path = out / wav_rel
            wav_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = wav_path.with_name(wav_path.name + ".tmp")
            write_audio(tmp_path, audio, sr, "wav")
            os.replace(tmp_path, wav_path)
            duration = round(float(len(audio)) / float(sr), 3)
            recipe = {
                "schema_version": 1,
                "kind": "design_candidate",
                "text": candidate["text"],
                "text_sha256": sha256_text(candidate["text"]),
                "language": language_name,
                "instruct": rendered,
                "instruct_sha256": compiled.rendered_sha256,
                "seed": int(candidate["seed"]),
                "sampling": candidate["sampling"],
                "max_new_tokens": matrix["max_new_tokens"],
                "codec_rate": 12.5,
                "model": model_payload,
                "duration_s": duration,
                "sample_rate": int(sr),
                "finish_reason": None,
                "generated_tokens": None,
            }
            completion = assess(
                duration_s=duration,
                text=candidate["text"],
                language=language_name,
                recipe=recipe,
            )
            atomic_write_json(out / recipe_rel, recipe)
            record.update(
                status="ok",
                sha256=sha256_file(wav_path),
                duration_s=duration,
                completion_status=completion["status"],
                error=None,
            )
            save_manifest(out, manifest)
        save_manifest(out, manifest)

    manifest["interrupted"] = interrupted
    manifest["warnings"] = warnings
    manifest["out_dir"] = str(out)
    if interrupted:
        manifest["exit_code"] = 130
    return manifest


_LANGUAGE_NAMES = {
    "zh": "Chinese",
    "cn": "Chinese",
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
    "de": "German",
    "fr": "French",
    "ru": "Russian",
    "pt": "Portuguese",
    "es": "Spanish",
    "it": "Italian",
}


def _language_name(value: Any) -> str:
    from ..catalog import display_language

    key = str(value or "zh").strip().lower()
    return _LANGUAGE_NAMES.get(key) or display_language(key) or "Chinese"


def _design_block(manifest: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    from ..errors import InputError

    design = manifest.get("design") or {}
    yaml_path = run_dir / "design.yaml"
    try:
        yaml_text = yaml_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise InputError(
            f"run is missing design.yaml: {yaml_path}", details={"path": str(yaml_path)}
        ) from exc
    yaml_sha = sha256_text(yaml_text)
    expected_sha = (manifest.get("source") or {}).get("sha256")
    if expected_sha and yaml_sha != expected_sha:
        raise InputError(
            "design.yaml changed since the run was created",
            details={"expected": expected_sha, "actual": yaml_sha},
        )
    rendered = str(design.get("rendered_instruction") or "").strip()
    if not rendered:
        raise InputError("run has no rendered design instruction")
    rendered_sha = design.get("rendered_sha256")
    if rendered_sha and sha256_text(rendered) != rendered_sha:
        raise InputError("rendered instruction does not match its recorded sha256")
    structured = design.get("structured")
    if not isinstance(structured, dict):
        raise InputError("run design.structured is not an object")
    if not design.get("template"):
        raise InputError("run design has no template")
    return {
        "schema_version": design.get("schema_version", 1),
        "template": design.get("template"),
        "template_sha256": design.get("template_sha256"),
        "structured": structured,
        "rendered_instruction": rendered,
        "rendered_sha256": rendered_sha or sha256_text(rendered),
        "compiler_version": design.get("compiler_version"),
        "yaml_sha256": yaml_sha,
        "yaml": yaml_text,
    }


def _commit_staged(store, staging: Path, name: str, *, force: bool) -> None:
    import shutil
    import uuid

    target = store.dir_for(name)
    backup: Path | None = None
    if target.is_symlink():
        raise UsageError(f"refusing to replace symlinked voice directory: {target}")
    if target.exists():
        if not force:
            raise UsageError(f"voice already exists: {name}; pass --force to replace it")
        backup = target.parent / f".backup-{name}-{uuid.uuid4().hex[:8]}"
        target.rename(backup)
    try:
        store.commit_staging(staging, name)
    except BaseException:
        if backup is not None and backup.exists() and not target.exists():
            backup.rename(target)
        raise
    else:
        if backup is not None:
            shutil.rmtree(backup, ignore_errors=True)


def freeze(cfg, args) -> dict[str, Any]:
    """Freeze one audition take into a design or clone voice profile."""
    import shutil

    from ..admit import check_samples
    from ..audio_io import duration_s, read_audio
    from ..completeness import FAILURE_STATUSES, STATUS_UNKNOWN, assess
    from ..design_report import run_path
    from ..errors import InputError, QualityError
    from ..utils import sha256_file
    from ..voice_store import VoiceStore, validate_name

    run_dir = Path(args.run_dir).expanduser()
    manifest_path = run_dir / "run.json"
    if not manifest_path.is_file():
        raise InputError(f"not a design run directory: {run_dir}", details={"path": str(run_dir)})
    manifest = read_json(manifest_path)
    candidates = manifest.get("candidates") or []
    record = next((item for item in candidates if item.get("take_id") == args.take), None)
    if record is None:
        raise UsageError(f"take not found in run: {args.take}")
    if str(record.get("status") or "").lower() == "failed":
        raise QualityError(f"take {args.take} failed to generate; refusing to freeze it")

    wav_path = run_path(run_dir, record.get("wav"))
    if wav_path is None or not wav_path.is_file():
        raise InputError(f"take audio is missing: {args.take}", details={"take": args.take})
    actual_sha = sha256_file(wav_path)
    expected_sha = record.get("sha256")
    if not isinstance(expected_sha, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha):
        raise InputError(
            f"take record has no valid sha256: {args.take}",
            details={"take": args.take},
        )
    if expected_sha.lower() != actual_sha.lower():
        raise InputError(
            f"take audio changed since generation: {args.take}",
            details={"take": args.take, "expected": expected_sha, "actual": actual_sha},
        )

    recipe = None
    recipe_path = run_path(run_dir, record.get("recipe"))
    if recipe_path is not None and recipe_path.is_file():
        try:
            recipe = read_json(recipe_path)
        except (OSError, ValueError):
            recipe = None
    language_name = _language_name((recipe or {}).get("language") or (manifest.get("design") or {}).get("language"))
    cover_text = str((recipe or {}).get("text") or record.get("text") or "").strip()
    warnings: list[str] = []

    name = validate_name(args.name)
    store = VoiceStore(cfg.data_dir)
    actual = store.existing_names().get(name.casefold())
    if actual is not None and actual != name:
        raise UsageError(
            f"voice name collides case-insensitively with existing {actual!r}: "
            f"{name!r}"
        )
    target = store.dir_for(name)
    if target.is_symlink():
        raise UsageError(f"refusing to replace symlinked voice directory: {target}")
    if target.exists() and not args.force:
        raise UsageError(f"voice already exists: {name}; pass --force to replace it")
    design = _design_block(manifest, run_dir)
    rendered = str(design.get("rendered_instruction") or "").strip()
    tags = [part.strip() for part in (args.tag or []) for part in part.split(",") if part.strip()]
    staging = store.staging_dir(name)
    try:
        if args.as_kind == "design":
            if not rendered:
                raise InputError("run has no rendered design instruction; cannot freeze --as design")
            meta = store.create(
                name=name,
                kind="design",
                source={
                    "type": "design",
                    "description": rendered,
                    "sample_text": cover_text,
                    "language": language_name,
                    "origin": "design-freeze",
                    "take_id": args.take,
                    "take_sha256": actual_sha,
                    "design": design,
                },
                model=manifest.get("model") or {},
                dtype=None,
                language=language_name,
                default_instruct=rendered,
                tags=tags,
                consent={
                    "rights_confirmed": True,
                    "license": args.license or "ai-generated-design",
                    "note": args.consent_note or "",
                },
                qtts_version=__version__,
                staging=staging,
                allow_existing=bool(args.force),
            )
            _commit_staged(store, staging, name, force=bool(args.force))
            return {
                "ok": True,
                "as": "design",
                "take_id": args.take,
                "voice": meta,
                "warnings": warnings,
                "label": "design freeze stores a regenerable recipe, not a frozen acoustic take",
            }

        # --as clone
        from ..catalog import find_model, scan, select
        from ..capabilities import validate_capabilities
        from ..commands.voice import _freeze_clone_prompt, _require_consent
        from ..tts_engine import TtsEngine

        samples, sr = read_audio(wav_path)
        seconds = duration_s(samples, sr)
        completion_status = record.get("completion_status")
        if completion_status is None:
            completion_status = assess(
                duration_s=seconds,
                text=cover_text or None,
                language=language_name,
                recipe=recipe,
            )["status"]
        if completion_status in FAILURE_STATUSES:
            raise QualityError(
                f"take {args.take} is {completion_status}; refusing to freeze it as a clone"
            )
        if completion_status == STATUS_UNKNOWN and not args.allow_incomplete:
            raise QualityError(
                f"take {args.take} completion is unknown; pass --allow-incomplete to override"
            )
        if seconds < 1.0:
            raise QualityError(f"reference is {seconds:.2f}s; at least 1s is required")
        if seconds < 6.0:
            warnings.append(f"reference is {seconds:.2f}s; 6-30s is recommended")
        if seconds > 30.0 and not args.allow_long_ref:
            raise UsageError(
                f"reference is {seconds:.2f}s; pass --allow-long-ref to freeze clips longer than 30s"
            )
        ref_text = str(args.ref_text or cover_text).strip()
        ref_text_source = "user" if args.ref_text else "cover"
        if not ref_text:
            raise UsageError("cannot determine the reference transcript; pass --ref-text TEXT")
        if not args.confirm_transcript:
            raise UsageError(
                "refusing to record an unconfirmed transcript; pass --confirm-transcript",
                details={"ref_text": ref_text, "ref_text_source": ref_text_source},
            )

        consent = _require_consent(args)
        models = scan(cfg.model_dir)
        model = find_model(models, args.model) if args.model else select(models, "clone")
        validate_capabilities(
            model,
            mode="clone",
            language=language_name,
            icl=True,
            ref_text=ref_text,
        )
        admission = check_samples(
            samples,
            sr,
            ref_text=ref_text,
            language=language_name,
        )
        blocking = [check for check in admission.errors if check.name != "duration_s"]
        if blocking:
            raise InputError(
                "reference audio rejected by admission checks",
                details=admission.to_dict(),
            )
        warnings.extend(
            f"admission: {check.name}: {check.advice}"
            for check in admission.warnings
            if check.advice
        )
        store.copy_ref_audio(name, wav_path, directory=staging)
        engine = TtsEngine(cfg, model)
        engine.load()
        source = {
            "type": "clone",
            "ref_text": ref_text,
            "ref_text_source": ref_text_source,
            "mode": "icl",
            "origin": "design-freeze",
            "take_id": args.take,
            "take_sha256": actual_sha,
            "design": design,
        }
        meta = _freeze_clone_prompt(
            cfg,
            store,
            name=name,
            staged_ref=staging / "ref.wav",
            ref_text=ref_text,
            mode="icl",
            language=language_name,
            model=model,
            engine=engine,
            source=source,
            consent=consent,
            staging=staging,
            tags=tags,
            preview_src=wav_path,
            allow_existing=bool(args.force),
        )
        _commit_staged(store, staging, name, force=bool(args.force))
        return {
            "ok": True,
            "as": "clone",
            "take_id": args.take,
            "voice": meta,
            "warnings": warnings,
            "ref_text_source": ref_text_source,
            "admission": admission.to_dict(),
        }
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def report(cfg, args) -> dict[str, Any]:
    from ..design_report import build_report

    if args.require_full_metrics and args.metrics != "full":
        raise UsageError("--require-full-metrics requires --metrics full")
    return build_report(
        args.run_dir,
        metrics=args.metrics,
        gap=float(args.gap),
        write_csv=bool(args.csv),
        write_html=bool(args.html),
        write_reel=bool(args.reel),
        require_full=bool(args.require_full_metrics),
        fail_on_degraded=bool(args.fail_on_degraded),
    )


def prune(cfg, args) -> dict[str, Any]:
    from ..design_prune import prune_run

    return prune_run(args.run_dir, keep=int(args.keep), apply=bool(args.yes))


def format_result(payload: dict[str, Any]) -> str:
    if "kept" in payload and "pruned" in payload and "run_dir" in payload:
        from ..design_prune import format_result as format_prune

        return format_prune(payload)
    if payload.get("kind") == "design_run":
        candidates = payload.get("candidates") or []
        ok = sum(1 for item in candidates if item.get("status") == "ok")
        failed = sum(1 for item in candidates if item.get("status") == "failed")
        lines = [
            f"run {payload.get('run_id')}: {ok}/{len(candidates)} candidate(s) ready"
            + (f", {failed} failed" if failed else "")
            + (" (interrupted)" if payload.get("interrupted") else "")
        ]
        for warning in payload.get("warnings") or []:
            lines.append(f"  warning: {warning}")
        lines.append(f"  next: qtts design report {payload.get('out_dir') or '<RUN_DIR>'}")
        return "\n".join(lines)
    if payload.get("kind") == "design_explore":
        matrix = payload.get("matrix") or {}
        prefix = "dry-run" if payload.get("dry_run") else "run"
        lines = [
            f"{prefix}: {payload.get('candidate_count', 0)} candidate(s) "
            f"for {payload.get('name')} -> {payload.get('out')}",
            f"  template={payload.get('template')} language={payload.get('language')} "
            f"lint={payload.get('lint', {}).get('grade')}",
        ]
        for candidate in matrix.get("candidates") or []:
            lines.append(
                f"  {candidate['take_id']} seed={candidate['seed']} "
                f"cover={candidate['cover_index']} "
                f"sampling={candidate['sampling_profile']} text={candidate['text']!r}"
            )
        return "\n".join(lines)
    if payload.get("as") in ("design", "clone") and "voice" in payload:
        voice = payload.get("voice") or {}
        lines = [
            f"froze take {payload.get('take_id')} as {payload.get('as')} voice "
            f"{voice.get('name')}"
        ]
        if payload.get("label"):
            lines.append(f"  note: {payload['label']}")
        for warning in payload.get("warnings") or []:
            lines.append(f"  warning: {warning}")
        return "\n".join(lines)
    if payload.get("run_id") and "candidates" in payload and "summary" in payload:
        from ..design_report import format_result as format_report

        return format_report(payload)
    if payload.get("kind") == "design_lint":
        lines = [
            f"{payload['name']}: {payload['grade']} ({payload['total']}/100) "
            f"template={payload['template']} language={payload['language']}"
        ]
        scores = payload.get("scores") or {}
        lines.append(
            "  " + " ".join(f"{key}={value}" for key, value in scores.items())
        )
        summary = payload.get("summary") or {}
        lines.append(
            f"  blocks={summary.get('blocks', 0)} "
            f"warnings={summary.get('warnings', 0)} infos={summary.get('infos', 0)}"
        )
        for issue in payload.get("issues") or []:
            lines.append(
                f"  {issue['level'].upper():5s} {issue['code']}: {issue['message']}"
            )
        if payload.get("show_rendered"):
            lines.append("rendered instruction:")
            lines.append(payload["instruction"]["rendered"] or "(empty)")
        return "\n".join(lines)
    if payload.get("written"):
        return (
            f"wrote {payload['file']} (name={payload['name']} "
            f"language={payload['language']} template={payload['template']})"
        )
    return str(payload)
