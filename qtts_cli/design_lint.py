"""Deterministic offline lint for design YAML (no model, no network)."""

from __future__ import annotations

import re
from typing import Any

from .design_compile import CompiledDesign
from .design_yaml import SUPPORTED_LANGUAGES, resolve_template_id
from .errors import InputError
from .voice_store import validate_name

_INLINE_EVENT = re.compile(r"\[[^\]\n]{1,40}\]")
_DIALOGUE = re.compile(r"(?m)^\s*[A-Za-z一-鿿]{1,8}\s*[:：]")
_CJK = re.compile(r"[㐀-鿿]")
_VAGUE = (
    "nice",
    "natural sounding",
    "normal",
    "好听",
    "自然",
    "普通",
    "不错",
)
_INSTRUCTION_MAX = 1000
_INSTRUCTION_LONG = 300
_COVER_WARN = 120
_COVER_MAX = 200


def _issue(level: str, code: str, message: str, field: str | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"level": level, "code": code, "message": message}
    if field:
        payload["field"] = field
    return payload


def _score(value: int) -> int:
    return max(0, min(100, int(value)))


def lint_design(document: dict[str, Any], compiled: CompiledDesign) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    present = set(document.get("present_fields") or [])
    language = compiled.language
    instruction = compiled.rendered_instruction or ""
    override = str(document.get("instruction_override") or "").strip()
    description = str(document.get("description") or "").strip()

    for message in document.get("errors") or []:
        code = "invalid-schema" if "schema_version" in message else "invalid-field"
        issues.append(_issue("block", code, message))

    if resolve_template_id(document.get("template")) is None:
        issues.append(
            _issue("block", "invalid-template", f"unknown template: {document.get('template')!r}", "template")
        )

    if not instruction.strip():
        issues.append(_issue("block", "empty-instruction", "compiled instruction is empty"))
    elif len(instruction) > _INSTRUCTION_MAX:
        issues.append(
            _issue("block", "instruction-too-long",
                   f"instruction is {len(instruction)} characters (max {_INSTRUCTION_MAX})")
        )
    elif len(instruction) > _INSTRUCTION_LONG:
        issues.append(
            _issue("warn", "long-instruction",
                   f"instruction is {len(instruction)} characters (recommended <= {_INSTRUCTION_LONG})")
        )

    if _INLINE_EVENT.search(instruction):
        issues.append(
            _issue("block", "inline-event",
                   "instruction contains bracketed stage directions such as [laughs]")
        )
    if _DIALOGUE.search(instruction):
        issues.append(
            _issue("block", "dialogue-in-instruction",
                   "instruction looks like dialogue; keep spoken lines in covers")
        )

    if language not in SUPPORTED_LANGUAGES:
        level = "warn" if (override or description) else "block"
        issues.append(
            _issue(level, "unsupported-language",
                   f"unsupported design language: {language!r}; provide instruction_override")
        )
    else:
        has_cjk = bool(_CJK.search(instruction))
        if language == "zh" and instruction.strip() and not has_cjk:
            issues.append(
                _issue("block", "language-mismatch",
                       "language is zh but the instruction has no Chinese characters")
            )
        if language == "en" and has_cjk:
            issues.append(
                _issue("block", "language-mismatch",
                       "language is en but the instruction contains Chinese characters")
            )

    covers = document.get("covers") or []
    if "covers" not in present:
        issues.append(_issue("block", "no-anchors", "covers are required", "covers"))
    elif not covers:
        issues.append(_issue("warn", "no-covers", "covers is empty", "covers"))
    if len(covers) > 2:
        issues.append(
            _issue("warn", "too-many-covers",
                   f"design has {len(covers)} covers; only the first two are used",
                   "covers")
        )
    for index, cover in enumerate(covers):
        text = str(cover)
        if not text.strip() or len(text) > _COVER_MAX:
            issues.append(
                _issue("block", "invalid-cover",
                       f"cover #{index + 1} must be non-empty and <= {_COVER_MAX} characters",
                       "covers")
            )
        elif len(text) > _COVER_WARN:
            issues.append(
                _issue("warn", "cover-length",
                       f"cover #{index + 1} is {len(text)} characters (recommended <= {_COVER_WARN})",
                       "covers")
            )

    lower = instruction.casefold()
    for term in _VAGUE:
        if term in lower:
            issues.append(
                _issue("warn", "vague-term", f"vague term: {term!r}")
            )
            break
    if any(term in lower for term in ("fast", "偏快", "快")) and any(
        term in lower for term in ("slow", "偏慢", "慢")
    ):
        issues.append(_issue("warn", "conflict", "instruction mixes fast and slow delivery"))

    name = document.get("name")
    try:
        validate_name(str(name or ""))
    except InputError:
        issues.append(
            _issue("warn", "name-format", f"name {name!r} does not match the voice name rules", "name")
        )

    if not document.get("avoid"):
        issues.append(_issue("warn", "no-avoid", "avoid list is empty", "avoid"))
    for key in document.get("extras") or {}:
        issues.append(_issue("info", "unknown-field", f"unknown field: {key}", str(key)))
    for warning in compiled.warnings:
        code = warning.get("code", "compiler-warning")
        level = "warn" if code != "override-ignores-structure" else "info"
        issues.append(
            _issue(level, code, str(warning.get("value") or code), warning.get("field"))
        )

    blocks = [item for item in issues if item["level"] == "block"]
    warnings = [item for item in issues if item["level"] == "warn"]
    infos = [item for item in issues if item["level"] == "info"]

    identity = document.get("identity") or {}
    delivery = document.get("delivery_baseline") or {}
    scene = document.get("scene") or {}
    specificity = (
        40
        + (15 if identity.get("timbre") else 0)
        + (10 if identity.get("pitch") else 0)
        + (10 if identity.get("persona") else 0)
        + (10 if identity.get("articulation") else 0)
        + (15 if covers else 0)
    )
    executability = (
        50
        + (20 if document.get("avoid") else 0)
        + (15 if covers else 0)
        + (15 if scene else 0)
    )
    consistency = 100
    if any(item["code"] in ("conflict", "both-overrides") for item in issues):
        consistency -= 15
    if any(item["code"] in ("inline-event", "dialogue-in-instruction") for item in issues):
        consistency -= 40
    language_match = 100
    if any(item["code"] == "language-mismatch" for item in issues):
        language_match = 30
    elif any(item["code"] == "unsupported-language" for item in issues):
        language_match = 70
    testability = 20 if not covers else (70 if len(covers) == 1 else 100)
    if any(item["code"] == "cover-length" for item in issues):
        testability -= 15
    scores = {
        "specificity": _score(specificity),
        "executability": _score(executability),
        "consistency": _score(consistency),
        "language_match": _score(language_match),
        "testability": _score(testability),
    }
    total = int(round(sum(scores.values()) / len(scores)))
    if blocks:
        # A blocked design is never "usable", whatever the component scores say.
        grade = "red"
    else:
        grade = "green" if total >= 85 else ("yellow" if total >= 70 else "red")

    return {
        "ok": not blocks,
        "schema_version": 1,
        "kind": "design_lint",
        "name": document.get("name"),
        "template": document.get("template"),
        "language": language,
        "instruction": {
            "rendered": instruction,
            "sha256": compiled.rendered_sha256,
            "length": len(instruction),
        },
        "scores": scores,
        "total": total,
        "grade": grade,
        "issues": issues,
        "summary": {
            "blocks": len(blocks),
            "warnings": len(warnings),
            "infos": len(infos),
        },
    }
