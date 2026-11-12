"""Structured voice-design YAML (btts-style fields, qtts compiler input).

Reads tolerate unknown fields (kept in ``extras``); writes are strict. YAML
anchors/aliases, duplicate keys, oversized files and non-UTF-8 input are
rejected before any compiler sees them.
"""

from __future__ import annotations

import copy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import yaml

from .errors import InputError
from .utils import sha256_text

DESIGN_SCHEMA_VERSION = 1
MAX_DESIGN_BYTES = 256 * 1024
DEFAULT_LANGUAGE = "zh"

TEMPLATES: dict[str, dict[str, Any]] = {
    "qtts/zh-late-night-radio/v1": {
        "language": "zh",
        "identity": {"pace": "medium_slow", "energy": "medium_low"},
        "covers": ["晚上好，欢迎收听今天的节目。"],
    },
    "qtts/en-podcast-host/v1": {
        "language": "en",
        "identity": {"pace": "medium", "energy": "medium"},
        "covers": ["Welcome back to the show."],
    },
    "qtts/ja-podcast-host/v1": {
        "language": "ja",
        "identity": {"pace": "medium", "energy": "medium"},
        "covers": ["また番組に戻ってきました。"],
    },
    "qtts/ko-podcast-host/v1": {
        "language": "ko",
        "identity": {"pace": "medium", "energy": "medium"},
        "covers": ["다시 프로그램으로 돌아왔습니다."],
    },
    "qtts/de-podcast-host/v1": {
        "language": "de",
        "identity": {"pace": "medium", "energy": "medium"},
        "covers": ["Willkommen zurück in der Sendung."],
    },
    "qtts/fr-podcast-host/v1": {
        "language": "fr",
        "identity": {"pace": "medium", "energy": "medium"},
        "covers": ["Bienvenue de nouveau dans l'émission."],
    },
    "qtts/ru-podcast-host/v1": {
        "language": "ru",
        "identity": {"pace": "medium", "energy": "medium"},
        "covers": ["С возвращением в нашу передачу."],
    },
    "qtts/pt-podcast-host/v1": {
        "language": "pt",
        "identity": {"pace": "medium", "energy": "medium"},
        "covers": ["Bem-vindo de volta ao programa."],
    },
    "qtts/es-podcast-host/v1": {
        "language": "es",
        "identity": {"pace": "medium", "energy": "medium"},
        "covers": ["Bienvenidos de nuevo al programa."],
    },
    "qtts/it-podcast-host/v1": {
        "language": "it",
        "identity": {"pace": "medium", "energy": "medium"},
        "covers": ["Bentornati al programma."],
    },
}
DEFAULT_TEMPLATES = {
    "zh": "qtts/zh-late-night-radio/v1",
    "en": "qtts/en-podcast-host/v1",
    "ja": "qtts/ja-podcast-host/v1",
    "ko": "qtts/ko-podcast-host/v1",
    "de": "qtts/de-podcast-host/v1",
    "fr": "qtts/fr-podcast-host/v1",
    "ru": "qtts/ru-podcast-host/v1",
    "pt": "qtts/pt-podcast-host/v1",
    "es": "qtts/es-podcast-host/v1",
    "it": "qtts/it-podcast-host/v1",
}
SUPPORTED_LANGUAGES = frozenset(DEFAULT_TEMPLATES)
EXAMPLE_COVERS = {
    "zh": ["晚上好，欢迎收听今天的节目。", "外面下着雨，我们慢慢聊。"],
    "en": ["Welcome back to the show.", "Let's take this slowly."],
    "ja": ["また番組に戻ってきました。", "今日はゆっくり話しましょう。"],
    "ko": ["다시 프로그램으로 돌아왔습니다.", "오늘은 천천히 이야기해 봐요."],
    "de": ["Willkommen zurück in der Sendung.", "Lass uns heute langsam sprechen."],
    "fr": ["Bienvenue de nouveau dans l'émission.", "Parlons-en doucement aujourd'hui."],
    "ru": ["С возвращением в нашу передачу.", "Давайте сегодня поговорим неспешно."],
    "pt": ["Bem-vindo de volta ao programa.", "Vamos conversar com calma hoje."],
    "es": ["Bienvenidos de nuevo al programa.", "Hablemos despacio hoy."],
    "it": ["Bentornati al programma.", "Parliamo con calma oggi."],
}

KNOWN_TOP_LEVEL = {
    "schema_version",
    "name",
    "language",
    "template",
    "identity",
    "delivery_baseline",
    "scene",
    "avoid",
    "covers",
    "description",
    "instruction_override",
}


class _StrictLoader(yaml.SafeLoader):
    """SafeLoader that rejects anchors/aliases and duplicate mapping keys."""

    def compose_node(self, parent, index):
        if self.check_event(yaml.AliasEvent):
            raise yaml.constructor.ConstructorError(
                None, None, "YAML aliases are not allowed", self.peek_event().start_mark
            )
        node = super().compose_node(parent, index)
        if getattr(node, "anchor", None):
            raise yaml.constructor.ConstructorError(
                None,
                None,
                "YAML anchors are not allowed",
                getattr(node, "start_mark", None),
            )
        return node

    def construct_mapping(self, node, deep=False):
        if isinstance(node, yaml.MappingNode):
            seen: set[Any] = set()
            for key_node, _ in node.value:
                key = self.construct_object(key_node, deep=deep)
                try:
                    duplicate = key in seen
                except TypeError:
                    duplicate = False
                if duplicate:
                    raise yaml.constructor.ConstructorError(
                        "while constructing a mapping",
                        node.start_mark,
                        f"duplicate key: {key!r}",
                        key_node.start_mark,
                    )
                try:
                    seen.add(key)
                except TypeError:
                    pass
        return super().construct_mapping(node, deep=deep)


def template_for_language(language: str) -> Optional[str]:
    return DEFAULT_TEMPLATES.get((language or "").strip().lower())


def template_language(template: str) -> Optional[str]:
    info = TEMPLATES.get(template)
    return info.get("language") if info else None


def _structure(value: Any, field: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a mapping")
    return dict(value)


def _string_list(value: Any, field: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")
    out = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError(f"{field} entries must be strings")
        out.append(item)
    return out


def load_design(path: str | Path) -> tuple[dict[str, Any], str, str]:
    """Return ``(structured, raw_text, sha256)`` or raise ``InputError`` (4)."""
    target = Path(path).expanduser()
    try:
        with target.open("rb") as handle:
            raw_bytes = handle.read(MAX_DESIGN_BYTES + 1)
    except OSError as exc:
        raise InputError(f"cannot read design file {target}: {exc}", details={"path": str(target)}) from exc
    if len(raw_bytes) > MAX_DESIGN_BYTES:
        raise InputError(
            f"design file is larger than {MAX_DESIGN_BYTES} bytes: {target}",
            details={"path": str(target)},
        )
    try:
        raw_text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise InputError(
            f"design file is not valid UTF-8: {target}", details={"path": str(target)}
        ) from exc
    try:
        document = yaml.load(raw_text, Loader=_StrictLoader)
    except yaml.YAMLError as exc:
        raise InputError(
            f"cannot parse design YAML {target}: {exc}", details={"path": str(target)}
        ) from exc
    if not isinstance(document, dict):
        raise InputError(
            f"design YAML must be a mapping: {target}", details={"path": str(target)}
        )
    structured = normalize_document(document)
    return structured, raw_text, sha256_text(raw_text)


def normalize_document(document: dict[str, Any]) -> dict[str, Any]:
    """Fill defaults and coerce known fields; malformed values become errors."""
    data = copy.deepcopy(document)
    errors: list[str] = []
    schema_version = data.get("schema_version", DESIGN_SCHEMA_VERSION)
    if schema_version != DESIGN_SCHEMA_VERSION:
        errors.append(f"unsupported schema_version: {schema_version!r}")

    name = data.get("name")
    language = str(data.get("language") or DEFAULT_LANGUAGE).strip().lower()
    template = data.get("template")
    identity = delivery = scene = {}
    avoid: list[str] = []
    covers: list[str] = []
    for field, target, parser in (
        ("identity", "identity", _structure),
        ("delivery_baseline", "delivery", _structure),
        ("scene", "scene", _structure),
        ("avoid", "avoid", _string_list),
        ("covers", "covers", _string_list),
    ):
        try:
            parsed = parser(data.get(field), field)
        except ValueError as exc:
            errors.append(str(exc))
            parsed = {} if target in ("identity", "delivery", "scene") else []
        if target == "identity":
            identity = parsed
        elif target == "delivery":
            delivery = parsed
        elif target == "scene":
            scene = parsed
        elif target == "avoid":
            avoid = parsed
        else:
            covers = parsed

    extras = {key: value for key, value in data.items() if key not in KNOWN_TOP_LEVEL}
    return {
        "schema_version": schema_version,
        "name": name,
        "language": language,
        "template": template,
        "identity": identity,
        "delivery_baseline": delivery,
        "scene": scene,
        "avoid": avoid,
        "covers": covers,
        "description": data.get("description"),
        "instruction_override": data.get("instruction_override"),
        "extras": extras,
        "errors": errors,
        "present_fields": sorted(key for key in KNOWN_TOP_LEVEL if key in data),
    }


def example_document(name: str, *, language: str = "zh", template: Optional[str] = None) -> dict[str, Any]:
    template = template or template_for_language(language)
    if template is None:
        raise InputError(f"unsupported design language: {language!r}")
    info = TEMPLATES[template]
    language = info["language"]
    if language == "zh":
        return {
            "schema_version": DESIGN_SCHEMA_VERSION,
            "name": name,
            "language": "zh",
            "template": template,
            "identity": {
                "age": "30s",
                "gender_presentation": "female",
                "timbre": ["warm", "clean", "slightly_husky"],
                "pitch": "low_mid",
                "articulation": "clear, relaxed",
                "persona": "late_night_host",
            },
            "delivery_baseline": {
                "pace": "medium_slow",
                "energy": "medium_low",
                "tone": ["calm", "intimate"],
            },
            "scene": {"proximity": "close_mic", "room": "dry_studio"},
            "avoid": ["childlike", "shouting"],
            "covers": list(EXAMPLE_COVERS["zh"]),
            "description": None,
            "instruction_override": None,
        }
    return {
        "schema_version": DESIGN_SCHEMA_VERSION,
        "name": name,
        "language": language,
        "template": template,
        "identity": {
            "age": "30s",
            "gender_presentation": "female",
            "timbre": ["warm", "clear"],
            "pitch": "low_mid",
            "articulation": "clear, relaxed",
            "persona": "podcast_host",
        },
        "delivery_baseline": {"pace": "medium", "energy": "medium", "tone": ["calm"]},
        "scene": {"proximity": "close_mic", "room": "dry_studio"},
        "avoid": ["childlike", "shouting"],
        "covers": list(EXAMPLE_COVERS.get(language, info["covers"])),
        "description": None,
        "instruction_override": None,
    }


def dump_design(document: dict[str, Any]) -> str:
    payload = {
        key: value
        for key, value in document.items()
        if key not in ("extras", "errors", "present_fields")
    }
    payload.update(document.get("extras") or {})
    return yaml.safe_dump(
        payload,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    )


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
