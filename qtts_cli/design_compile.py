"""Compile structured design YAML into one natural-language instruction.

Pure Python; no torch and no model imports. Unknown vocabulary is rendered by
replacing underscores with spaces and reported as a warning instead of failing
silently or inventing a meaning.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

from .design_yaml import TEMPLATES, resolve_template_id
from .utils import sha256_text

COMPILER_VERSION = "1"

_ZH = {
    "gender_presentation": {"female": "女性", "male": "男性", "neutral": "中性"},
    "timbre": {
        "warm": "温暖",
        "clean": "干净",
        "clear": "清澈",
        "slightly_husky": "略带沙哑",
        "bright": "明亮",
        "dark": "偏暗",
        "breathy": "带气声",
        "nasal": "带鼻音",
        "smooth": "顺滑",
        "crisp": "清脆",
        "soft": "柔和",
        "metallic": "有金属感",
        "deep": "低沉",
        "husky": "沙哑",
    },
    "pitch": {
        "low": "偏低",
        "low_mid": "中低",
        "mid": "中等",
        "high_mid": "中高",
        "high": "偏高",
    },
    "articulation": {
        "clear": "清晰",
        "relaxed": "松弛",
        "precise": "准确",
        "lazy": "慵懒",
        "crisp": "利落",
        "soft": "柔和",
    },
    "persona": {
        "late_night_host": "深夜电台主持人",
        "documentary_narrator": "纪录片旁白",
        "anime_character": "动画角色",
        "podcast_host": "播客主持人",
        "narrator": "旁白",
        "audiobook_narrator": "有声书主播",
        "game_character": "游戏角色",
        "news_anchor": "新闻主播",
    },
    "pace": {
        "slow": "慢",
        "medium_slow": "偏慢",
        "medium": "适中",
        "medium_fast": "偏快",
        "fast": "快",
    },
    "energy": {
        "low": "低",
        "medium_low": "中低",
        "medium": "中等",
        "medium_high": "中高",
        "high": "高",
    },
    "tone": {
        "calm": "平静",
        "intimate": "亲密",
        "warm": "温暖",
        "cool": "冷静",
        "playful": "俏皮",
        "serious": "严肃",
        "gentle": "温柔",
        "wry": "略带调侃",
        "lively": "活泼",
    },
    "proximity": {
        "close_mic": "近场麦克风",
        "mid_distance": "中等距离",
        "far_field": "远场麦克风",
        "far_mic": "远场麦克风",
    },
    "room": {
        "dry_studio": "干声录音棚",
        "treated_room": "吸音房间",
        "natural_room": "自然房间",
        "outdoor": "户外",
    },
    "avoid": {
        "childlike": "童声",
        "shouting": "喊叫",
        "robotic": "机械感",
        "overly_bright": "过亮",
        "monotone": "单调",
        "overly_dramatic": "过度戏剧化",
    },
}
_EN = {
    "gender_presentation": {"female": "female", "male": "male", "neutral": "neutral"},
    "timbre": {
        "warm": "warm",
        "clean": "clean",
        "clear": "clear",
        "slightly_husky": "slightly husky",
        "bright": "bright",
        "dark": "dark",
        "breathy": "breathy",
        "nasal": "nasal",
        "smooth": "smooth",
        "crisp": "crisp",
        "soft": "soft",
        "metallic": "metallic",
        "deep": "deep",
        "husky": "husky",
    },
    "pitch": {
        "low": "low",
        "low_mid": "low-mid",
        "mid": "mid",
        "high_mid": "high-mid",
        "high": "high",
    },
    "articulation": {
        "clear": "clear",
        "relaxed": "relaxed",
        "precise": "precise",
        "lazy": "lazy",
        "crisp": "crisp",
        "soft": "soft",
    },
    "persona": {
        "late_night_host": "late-night radio host",
        "documentary_narrator": "documentary narrator",
        "anime_character": "animation character",
        "podcast_host": "podcast host",
        "narrator": "narrator",
        "audiobook_narrator": "audiobook narrator",
        "game_character": "game character",
        "news_anchor": "news anchor",
    },
    "pace": {
        "slow": "slow",
        "medium_slow": "slightly slow",
        "medium": "medium",
        "medium_fast": "medium-fast",
        "fast": "fast",
    },
    "energy": {
        "low": "low",
        "medium_low": "medium-low",
        "medium": "medium",
        "medium_high": "medium-high",
        "high": "high",
    },
    "tone": {
        "calm": "calm",
        "intimate": "intimate",
        "warm": "warm",
        "cool": "cool",
        "playful": "playful",
        "serious": "serious",
        "gentle": "gentle",
        "wry": "wry",
        "lively": "lively",
    },
    "proximity": {
        "close_mic": "close-mic",
        "mid_distance": "mid-distance",
        "far_field": "far-field",
        "far_mic": "distant mic",
    },
    "room": {
        "dry_studio": "dry studio",
        "treated_room": "treated room",
        "natural_room": "natural room",
        "outdoor": "outdoors",
    },
    "avoid": {
        "childlike": "childlike",
        "shouting": "shouting",
        "robotic": "robotic",
        "overly_bright": "overly bright",
        "monotone": "monotone",
        "overly_dramatic": "overly dramatic",
    },
}
_AGE_ZH = {
    "teen": "少年",
    "child": "儿童",
    "20s": "二十多岁",
    "30s": "三十多岁",
    "40s": "四十多岁",
    "50s": "五十多岁",
    "60s": "六十多岁",
    "senior": "老年",
}
_AGE_EN = {
    "teen": "teenage",
    "child": "child",
    "20s": "in their twenties",
    "30s": "in their thirties",
    "40s": "in their forties",
    "50s": "in their fifties",
    "60s": "in their sixties",
    "senior": "elderly",
}


@dataclass
class CompiledDesign:
    structured: dict[str, Any]
    rendered_instruction: str
    rendered_sha256: str
    template: str
    template_sha256: str
    compiler_version: str
    language: str
    warnings: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "structured": self.structured,
            "rendered_instruction": self.rendered_instruction,
            "rendered_sha256": self.rendered_sha256,
            "template": self.template,
            "template_sha256": self.template_sha256,
            "compiler_version": self.compiler_version,
            "language": self.language,
            "warnings": self.warnings,
        }


def template_sha256(template: str) -> str:
    canonical = resolve_template_id(template)
    if canonical is None:
        raise ValueError(f"invalid template: {template!r}")
    return sha256_text(
        json.dumps(TEMPLATES[canonical], sort_keys=True, ensure_ascii=False)
    )


def _translate(
    value: Any, field_name: str, language: str, warnings: list[dict[str, Any]]
) -> str:
    text = str(value).strip()
    if not text:
        return ""
    table = _ZH if language == "zh" else _EN
    mapping = table.get(field_name, {})
    key = text.casefold().replace(" ", "_")
    if key in mapping:
        return mapping[key]
    warnings.append(
        {"code": "unknown-vocab", "field": field_name, "value": text}
    )
    return text.replace("_", " ")


def _items(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _translate_items(
    value: Any, field_name: str, language: str, warnings: list[dict[str, Any]]
) -> list[str]:
    return [
        translated
        for translated in (
            _translate(item, field_name, language, warnings) for item in _items(value)
        )
        if translated
    ]


def _age(value: Any, language: str) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    table = _AGE_ZH if language == "zh" else _AGE_EN
    return table.get(text.casefold(), text)


def _render_zh(identity, delivery, scene, avoid, warnings) -> str:
    sentences: list[str] = []
    gender = _translate(identity.get("gender_presentation", ""), "gender_presentation", "zh", warnings)
    age = _age(identity.get("age"), "zh")
    head = "声音"
    if age or gender:
        head = f"一位{age}{gender}声音" if (age or gender) else head
    parts = [head]
    timbre = _translate_items(identity.get("timbre"), "timbre", "zh", warnings)
    if timbre:
        parts.append("音色" + "、".join(timbre))
    pitch = _translate(identity.get("pitch", ""), "pitch", "zh", warnings)
    if pitch:
        parts.append("音高" + pitch)
    articulation = _translate_items(identity.get("articulation"), "articulation", "zh", warnings)
    if articulation:
        parts.append("吐字" + "、".join(articulation))
    persona = _translate(identity.get("persona", ""), "persona", "zh", warnings)
    if persona:
        parts.append("身份是" + persona)
    sentences.append("，".join(parts) + "。")

    pace = _translate(delivery.get("pace", ""), "pace", "zh", warnings)
    energy = _translate(delivery.get("energy", ""), "energy", "zh", warnings)
    tone = _translate_items(delivery.get("tone"), "tone", "zh", warnings)
    delivery_parts = []
    if pace:
        delivery_parts.append("语速" + pace)
    if energy:
        delivery_parts.append("能量" + energy)
    if tone:
        delivery_parts.append("语气" + "、".join(tone))
    if delivery_parts:
        sentences.append("说话时" + "，".join(delivery_parts) + "。")

    proximity = _translate(scene.get("proximity", ""), "proximity", "zh", warnings)
    room = _translate(scene.get("room", ""), "room", "zh", warnings)
    scene_parts = [part for part in (proximity, room) if part]
    if scene_parts:
        sentences.append("录音场景为" + "、".join(scene_parts) + "。")

    avoid_items = _translate_items(avoid, "avoid", "zh", warnings)
    if avoid_items:
        sentences.append("避免" + "、".join(avoid_items) + "。")
    return "".join(sentences)


def _render_en(identity, delivery, scene, avoid, warnings) -> str:
    sentences: list[str] = []
    gender = _translate(identity.get("gender_presentation", ""), "gender_presentation", "en", warnings)
    age = _age(identity.get("age"), "en")
    timbre = _translate_items(identity.get("timbre"), "timbre", "en", warnings)
    pitch = _translate(identity.get("pitch", ""), "pitch", "en", warnings)
    articulation = _translate_items(identity.get("articulation"), "articulation", "en", warnings)
    persona = _translate(identity.get("persona", ""), "persona", "en", warnings)

    age_gender = f"{age or ''} {gender}".strip()
    head = f"A {age_gender} voice" if age_gender else "A voice"
    descriptors = []
    if timbre:
        descriptors.append(f"{', '.join(timbre)} timbre")
    if pitch:
        descriptors.append(f"{pitch} pitch")
    if articulation:
        descriptors.append(f"{', '.join(articulation)} articulation")
    if descriptors:
        head += " with " + ", ".join(descriptors)
    if persona:
        head += f" (persona: {persona})"
    sentences.append(head + ".")

    pace = _translate(delivery.get("pace", ""), "pace", "en", warnings)
    energy = _translate(delivery.get("energy", ""), "energy", "en", warnings)
    tone = _translate_items(delivery.get("tone"), "tone", "en", warnings)
    delivery_parts = []
    if pace:
        delivery_parts.append(f"{pace} pace")
    if energy:
        delivery_parts.append(f"{energy} energy")
    if tone:
        delivery_parts.append(f"{', '.join(tone)} tone")
    if delivery_parts:
        sentences.append("Speak with " + ", ".join(delivery_parts) + ".")

    proximity = _translate(scene.get("proximity", ""), "proximity", "en", warnings)
    room = _translate(scene.get("room", ""), "room", "en", warnings)
    scene_parts = [part for part in (proximity, room) if part]
    if scene_parts:
        sentences.append("Recorded " + ", ".join(scene_parts) + ".")

    avoid_items = _translate_items(avoid, "avoid", "en", warnings)
    if avoid_items:
        sentences.append("Avoid " + ", ".join(avoid_items) + ".")
    return " ".join(sentences)


def compile_design(document: dict[str, Any]) -> CompiledDesign:
    """Render ``document`` (output of :func:`design_yaml.normalize_document`)."""
    template = resolve_template_id(document.get("template"))
    if template is None:
        raise ValueError(f"invalid template: {document.get('template')!r}")
    language = str(document.get("language") or "zh").strip().lower()
    warnings: list[dict[str, Any]] = []
    override = str(document.get("instruction_override") or "").strip()
    description = str(document.get("description") or "").strip()
    if override:
        rendered = override
        if description:
            warnings.append({"code": "both-overrides", "field": "instruction_override"})
        warnings.append(
            {"code": "override-ignores-structure", "field": "instruction_override"}
        )
    elif description:
        rendered = description
    else:
        identity = document.get("identity") or {}
        delivery = document.get("delivery_baseline") or {}
        scene = document.get("scene") or {}
        avoid = document.get("avoid") or []
        if language == "zh":
            rendered = _render_zh(identity, delivery, scene, avoid, warnings)
        else:
            # Qwen3-TTS accepts English design instructions for every supported
            # language; keep one renderer so non-zh templates are not empty.
            rendered = _render_en(identity, delivery, scene, avoid, warnings)
    return CompiledDesign(
        structured=document,
        rendered_instruction=rendered,
        rendered_sha256=sha256_text(rendered),
        template=template,
        template_sha256=template_sha256(template),
        compiler_version=COMPILER_VERSION,
        language=language,
        warnings=warnings,
    )
