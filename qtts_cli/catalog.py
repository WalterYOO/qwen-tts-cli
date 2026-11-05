"""Model discovery and capability introspection from local model directories.

Everything here reads only JSON/README files. Importing this module must never
import torch or qwen_tts: `--dry-run` and capability validation depend on it.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from .errors import InputError, ModelError, UsageError

MODEL_KIND_BY_TTS_TYPE = {
    "custom_voice": "builtin",
    "base": "clone",
    "voice_design": "design",
}

_LANGUAGE_DISPLAY = {
    "auto": "Auto",
    "chinese": "Chinese",
    "english": "English",
    "german": "German",
    "italian": "Italian",
    "portuguese": "Portuguese",
    "spanish": "Spanish",
    "japanese": "Japanese",
    "korean": "Korean",
    "french": "French",
    "russian": "Russian",
    "cantonese": "Cantonese",
    "beijing_dialect": "Beijing Dialect",
    "sichuan_dialect": "Sichuan Dialect",
}

_README_ROW = re.compile(
    r"^\|\s*([A-Za-z][A-Za-z0-9_ ]*?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*$"
)
_README_HEADER_WORDS = {"speaker", "model", "language", "name"}


def display_language(name: str) -> str:
    key = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    if key in _LANGUAGE_DISPLAY:
        return _LANGUAGE_DISPLAY[key]
    return " ".join(part.capitalize() for part in key.split("_"))


def _read_json(path: Path) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def _speakers_from_readme(model_dir: Path) -> dict[str, str]:
    """Map lowercase speaker id -> display name from the README table."""
    found: dict[str, str] = {}
    readme = model_dir / "README.md"
    if not readme.is_file():
        return found
    try:
        lines = readme.read_text(encoding="utf-8").splitlines()
    except OSError:
        return found
    for line in lines:
        match = _README_ROW.match(line)
        if not match:
            continue
        name, description, native = (part.strip() for part in match.groups())
        if name.lower() in _README_HEADER_WORDS:
            continue
        # Drop documentation rows: the third column is not a language there.
        if "documentation" in native.lower() or "documentation" in description.lower():
            continue
        found[name.lower()] = name
    return found


@dataclass
class ModelInfo:
    path: Path
    name: str
    model_type: str  # config.model_type, e.g. qwen3_tts / qwen3_asr
    tts_model_type: Optional[str] = None  # custom_voice / base / voice_design
    size: Optional[str] = None  # 1b7 / 0b6
    speakers: list[str] = field(default_factory=list)
    speaker_ids: dict[str, int] = field(default_factory=dict)
    languages: list[str] = field(default_factory=list)
    supports_instruct: bool = False
    tokenizer_type: Optional[str] = None
    revision: Optional[str] = None
    fingerprint: str = ""
    kind: str = "unknown"  # builtin / clone / design / align / asr

    @property
    def model_id(self) -> str:
        return self.name

    def speaker_id(self, name: str) -> Optional[int]:
        return self.speaker_ids.get(name.lower())

    def canonical_speaker(self, name: str) -> Optional[str]:
        key = name.lower()
        for display in self.speakers:
            if display.lower() == key:
                return display
        return None

    def to_dict(self) -> dict:
        return {
            "id": self.model_id,
            "path": str(self.path),
            "kind": self.kind,
            "model_type": self.model_type,
            "tts_model_type": self.tts_model_type,
            "size": self.size,
            "speakers": self.speakers,
            "languages": self.languages,
            "supports_instruct": self.supports_instruct,
            "tokenizer_type": self.tokenizer_type,
            "revision": self.revision,
            "fingerprint": self.fingerprint,
        }


def _size_from_name(name: str) -> Optional[str]:
    """Parse ``Qwen3-ASR-0.6B`` -> ``0b6`` for stable tier ordering."""
    match = re.search(r"(\d+(?:\.\d+)?)\s*B", name, re.IGNORECASE)
    if not match:
        return None
    value = match.group(1)
    if "." in value:
        major, minor = value.split(".", 1)
        return f"{major}b{minor}"
    return f"{int(value)}b0"


def _load_one(model_dir: Path) -> Optional[ModelInfo]:
    config = _read_json(model_dir / "config.json")
    if config is None:
        return None
    model_type = str(config.get("model_type", ""))
    info = ModelInfo(
        path=model_dir,
        name=model_dir.name,
        model_type=model_type,
        tokenizer_type=config.get("tokenizer_type"),
        revision=config.get("revision") or _read_revision(model_dir),
        fingerprint=model_fingerprint(model_dir),
    )

    name_lower = model_dir.name.lower()
    # Qwen3-ASR and Qwen3-ForcedAligner share model_type "qwen3_asr"; the
    # directory name is the only reliable discriminator. Check the aligner
    # first so an aligner is never misclassified as ASR (t10 H-A01).
    if "aligner" in name_lower:
        info.kind = "align"
        langs = config.get("support_languages") or []
        info.languages = [display_language(x) for x in langs] if langs else []
        return info
    # Only the known ASR model_type or the frozen directory pattern; matching
    # any "asr" substring would misclassify third-party models (t10 P2-4).
    if model_type.lower() == "qwen3_asr" or re.match(r"qwen3[-_]asr", name_lower):
        info.kind = "asr"
        info.size = _size_from_name(model_dir.name)
        langs = config.get("support_languages") or []
        info.languages = [display_language(x) for x in langs] if langs else []
        return info

    if model_type != "qwen3_tts":
        return None

    info.tts_model_type = config.get("tts_model_type")
    info.size = config.get("tts_model_size")
    info.kind = MODEL_KIND_BY_TTS_TYPE.get(info.tts_model_type or "", "unknown")

    talker = config.get("talker_config") or {}
    speaker_ids = talker.get("spk_id") or {}
    if isinstance(speaker_ids, dict):
        info.speaker_ids = {str(k).lower(): int(v) for k, v in speaker_ids.items()}
        display = _speakers_from_readme(model_dir)
        info.speakers = [display.get(key, key) for key in info.speaker_ids]

    lang_map = talker.get("codec_language_id") or {}
    languages = ["Auto"]
    if isinstance(lang_map, dict):
        for key in lang_map:
            name = display_language(str(key))
            if name not in languages:
                languages.append(name)
    info.languages = languages

    # PRD §3.1: Base has no instruct; 0.6B CustomVoice silently drops it, so the
    # CLI must reject it instead. VoiceDesign's instruct is the voice description.
    if info.tts_model_type == "voice_design":
        info.supports_instruct = True
    elif info.tts_model_type == "custom_voice":
        info.supports_instruct = str(info.size) == "1b7"
    else:
        info.supports_instruct = False
    return info


_FINGERPRINT_FILES = (
    "config.json",
    "revision.txt",
    ".revision",
    "speech_tokenizer/config.json",
    "model.safetensors",
    "speech_tokenizer/model.safetensors",
)


def model_fingerprint(model_dir: str | Path) -> str:
    """Cheap content fingerprint for local models without a revision.

    Small files are hashed in full; large weights contribute name + size so the
    scan stays fast. Replaced weights change the fingerprint (codex P1-5).
    """
    root = Path(model_dir)
    digest = hashlib.sha256()
    for rel in _FINGERPRINT_FILES:
        path = root / rel
        try:
            if not path.is_file():
                continue
            stat = path.stat()
        except OSError:
            continue
        digest.update(f"{rel}:{stat.st_size}".encode("utf-8"))
        if stat.st_size <= 1 << 20:
            try:
                digest.update(path.read_bytes())
            except OSError:
                pass
    return digest.hexdigest()[:16]


def _read_revision(model_dir: Path) -> Optional[str]:
    for name in ("revision.txt", ".revision"):
        candidate = model_dir / name
        if candidate.is_file():
            try:
                return candidate.read_text(encoding="utf-8").strip() or None
            except OSError:
                return None
    return None


def scan(model_dir: Path | str | None) -> list[ModelInfo]:
    """Enumerate usable local model directories. Never raises for missing dir."""
    if model_dir is None:
        return []
    root = Path(model_dir).expanduser()
    if not root.is_dir():
        return []
    out: list[ModelInfo] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        info = _load_one(child)
        if info is not None:
            out.append(info)
    return out


def require_model_dir(config) -> Path:
    model_dir = getattr(config, "model_dir", None)
    if model_dir is None or not Path(model_dir).is_dir():
        raise ModelError(
            "no model directory configured; pass --model-dir DIR or set QTTS_MODEL_DIR",
            details={"model_dir": str(model_dir) if model_dir else None},
        )
    return Path(model_dir)


def select(models: Iterable[ModelInfo], kind: str) -> ModelInfo:
    """Pick the best model for a generation kind (1.7B preferred over 0.6B)."""
    candidates = [m for m in models if m.kind == kind]
    if not candidates:
        raise ModelError(
            f"no local model available for mode '{kind}'",
            details={"kind": kind, "available": [m.name for m in models]},
        )
    candidates.sort(key=lambda m: (0 if m.size == "1b7" else 1, m.name))
    return candidates[0]


def select_aligner(models: Iterable[ModelInfo]) -> ModelInfo:
    for model in models:
        if model.kind == "align":
            return model
    raise ModelError(
        "no forced aligner model found; expected a *ForcedAligner* directory",
        details={"available": [m.name for m in models]},
    )


def _asr_version(model: ModelInfo) -> float:
    """Best-effort version number for "highest version" ASR selection."""
    match = re.fullmatch(r"(\d+)b(\d+)", str(model.size or "").lower())
    if match:
        return float(f"{match.group(1)}.{match.group(2)}")
    match = re.search(r"(\d+(?:\.\d+)?)\s*B", model.name, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return 0.0


def looks_like_explicit_path(spec: str) -> bool:
    """True when ``spec`` denotes a filesystem path, not a name/repo id.

    Absolute paths, ``~``, ``./``/``../`` and Windows separators are always
    paths.  A plain ``org/name`` repo id retains its historical suffix
    matching; deeper slash specs (``models/Qwen3-ASR-0.6B``) are paths.
    """
    value = str(spec).strip()
    if not value:
        return False
    if value.startswith("~") or "\\" in value:
        return True
    if Path(value).is_absolute():
        return True
    if value in (".", "..") or value.startswith("./") or value.startswith("../"):
        return True
    if "/" in value:
        parts = [part for part in value.split("/") if part]
        return len(parts) != 2
    return False


def select_asr(models: Iterable[ModelInfo], spec: Optional[str] = None) -> ModelInfo:
    """Pick an ASR model by local path, repo id/directory name, or best local.

    ``spec`` may be a directory, ``Qwen/Qwen3-ASR-0.6B``, or a unique name
    suffix. With no spec the highest local version wins; equal versions are
    ambiguous and require an explicit choice.
    """
    models = list(models)
    if spec is not None and str(spec).strip():
        needle = str(spec).strip()
        path = Path(needle).expanduser()
        if path.is_dir():
            info = _load_one(path)
            if info is None:
                raise ModelError(
                    f"not a supported model directory: {path}",
                    details={"path": str(path)},
                )
            if info.kind != "asr":
                raise ModelError(
                    f"model {path.name} is not an ASR model (kind={info.kind})",
                    details={"path": str(path), "kind": info.kind},
                )
            return info
        if looks_like_explicit_path(needle):
            raise ModelError(
                f"ASR model path not found: {needle}",
                details={"path": str(path)},
            )
        candidates = [m for m in models if m.kind == "asr"]
        basename = needle.rstrip("/").split("/")[-1]
        for predicate in (
            lambda m: m.name == needle or m.model_id == needle,
            lambda m: m.name == basename,
            lambda m: m.name.endswith(needle),
            lambda m: m.name.endswith(basename),
        ):
            hits = [m for m in candidates if predicate(m)]
            if len(hits) == 1:
                return hits[0]
            if len(hits) > 1:
                raise UsageError(
                    f"ASR model spec {needle!r} is ambiguous",
                    details={"available": [m.name for m in hits]},
                )
        raise ModelError(
            f"ASR model not found: {needle}",
            details={"available": [m.name for m in candidates]},
        )

    candidates = [m for m in models if m.kind == "asr"]
    if not candidates:
        raise ModelError(
            "no local ASR model found; pull Qwen/Qwen3-ASR-0.6B or pass --asr-model PATH",
            details={"available": [m.name for m in models]},
        )
    ranked = sorted(candidates, key=lambda m: (-_asr_version(m), m.name))
    best = ranked[0]
    if len(ranked) > 1 and _asr_version(ranked[1]) == _asr_version(best):
        raise UsageError(
            "multiple ASR models with the same version; pass --asr-model",
            details={"available": [m.name for m in ranked]},
        )
    return best


def find_model(models: Iterable[ModelInfo], spec: str) -> ModelInfo:
    """Resolve an explicit --model value: exact name, name suffix, or path."""
    needle = str(spec).strip()
    path = Path(needle).expanduser()
    if path.is_dir():
        info = _load_one(path)
        if info is None:
            raise InputError(f"not a supported model directory: {path}")
        return info
    for model in models:
        if model.name == needle or model.model_id == needle:
            return model
    for model in models:
        if model.name.endswith(needle):
            return model
    raise InputError(
        f"model not found: {needle}",
        details={"available": [m.name for m in models]},
    )


def require_speaker(model: ModelInfo, name: str) -> str:
    if not model.speakers:
        raise InputError(f"model {model.name} has no built-in speakers")
    canonical = model.canonical_speaker(name)
    if canonical is None:
        raise InputError(
            f"unknown built-in speaker: {name}",
            details={"speaker": name, "supported": model.speakers, "model": model.name},
        )
    return canonical


def language_supported(model: ModelInfo, language: str) -> bool:
    if not language or language.lower() == "auto":
        return True
    if not model.languages:
        return True
    return any(lang.lower() == language.lower() for lang in model.languages)
