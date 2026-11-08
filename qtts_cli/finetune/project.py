"""Strict `qtts finetune` project loader (FR-P2-07 §2).

Reuses the audiobook loader mechanics (duplicate JSON keys, non-finite
numbers, YAML nesting, size/depth/node limits) but has its own schema and
its own path policy: every path is resolved relative to the project file
directory.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from ..audiobook.project import (
    MAX_PROJECT_BYTES,
    _check_tree,
    _load_project_data,
    _read_text,
    _unknown,
)
from ..errors import InputError
from ..voice_store import validate_name

SCHEMA_VERSION = 1
MAX_ITEMS = 100_000
MAX_TEXT_CHARS = 2_000
MAX_LANGUAGE_CHARS = 32
DEVICE_CHOICES = ("auto", "cpu", "mps", "cuda", "cuda:0", "cuda:1")
SPEAKER_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,63}$")

_TOP_KEYS = {"schema_version", "dataset", "prepare", "train", "register", "eval"}
_DATASET_KEYS = {"ref_audio", "language", "items"}
_ITEM_KEYS = {"audio", "text"}
_PREPARE_KEYS = {"tokenizer_model_path", "device", "raw_jsonl", "output_jsonl"}
_TRAIN_KEYS = {
    "finetuning_dir",
    "python",
    "init_model_path",
    "output_dir",
    "train_jsonl",
    "speaker_name",
    "batch_size",
    "lr",
    "num_epochs",
    "checkpoint",
}
_REGISTER_KEYS = {"name", "consent"}
_CONSENT_KEYS = {"rights_confirmed", "license", "note"}
_EVAL_KEYS = {
    "items",
    "language",
    "asr_model",
    "base_model",
    "thresholds",
    "audio_dir",
    "checkpoint",
}
_EVAL_ITEM_KEYS = {"text", "language"}
_THRESHOLD_KEYS = {"max_cer", "min_speaker_similarity"}


@dataclass(frozen=True)
class DatasetItem:
    audio: Path
    text: str


@dataclass(frozen=True)
class EvalItem:
    text: str
    language: Optional[str] = None


@dataclass
class DatasetSpec:
    ref_audio: Path
    language: str
    items: list[DatasetItem]


@dataclass
class PrepareSpec:
    tokenizer_model_path: Optional[Path] = None
    device: Optional[str] = None
    raw_jsonl: Optional[Path] = None
    output_jsonl: Optional[Path] = None


@dataclass
class TrainSpec:
    finetuning_dir: Optional[Path] = None
    python: Optional[str] = None
    init_model_path: Optional[Path] = None
    output_dir: Optional[Path] = None
    train_jsonl: Optional[Path] = None
    speaker_name: Optional[str] = None
    batch_size: Optional[int] = None
    lr: Optional[float] = None
    num_epochs: Optional[int] = None
    checkpoint: Any = None


@dataclass
class RegisterSpec:
    name: Optional[str] = None
    consent: Optional[dict[str, Any]] = None


@dataclass
class EvalSpec:
    items: list[EvalItem] = field(default_factory=list)
    language: str = "Auto"
    asr_model: Optional[str] = None
    base_model: Optional[str] = None
    max_cer: Optional[float] = None
    min_speaker_similarity: Optional[float] = None
    audio_dir: Optional[Path] = None
    checkpoint: Any = None


@dataclass
class Project:
    path: Path
    project_dir: Path
    dataset: DatasetSpec
    prepare: PrepareSpec = field(default_factory=PrepareSpec)
    train: TrainSpec = field(default_factory=TrainSpec)
    register: RegisterSpec = field(default_factory=RegisterSpec)
    eval: Optional[EvalSpec] = None


# --------------------------------------------------------------------------- #
# Scalar helpers
# --------------------------------------------------------------------------- #
def _invalid(message: str, path: Path) -> InputError:
    return InputError(message, details={"project": str(path)})


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _mapping(value: Any, name: str, path: Path) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise _invalid(f"{name} must be a mapping", path)
    return value


def _string(
    value: Any,
    name: str,
    path: Path,
    *,
    allow_empty: bool = False,
    limit: Optional[int] = None,
) -> str:
    if not isinstance(value, str):
        raise _invalid(f"{name} must be a string", path)
    text = value.strip()
    if not text and not allow_empty:
        raise _invalid(f"{name} must not be empty", path)
    if limit is not None and len(text) > limit:
        raise _invalid(f"{name} must be at most {limit} characters", path)
    return text


def _number(
    value: Any,
    name: str,
    path: Path,
    *,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
    integer: bool = False,
    exclusive_minimum: bool = False,
) -> float | int:
    if not _is_number(value):
        raise _invalid(f"{name} must be a number", path)
    if not math.isfinite(float(value)):
        raise _invalid(f"{name} must be finite", path)
    if integer and float(value) != int(value):
        raise _invalid(f"{name} must be an integer", path)
    number = float(value)
    if minimum is not None:
        if exclusive_minimum and not number > minimum:
            raise _invalid(f"{name} must be > {minimum}", path)
        if not exclusive_minimum and number < minimum:
            raise _invalid(f"{name} must be >= {minimum}", path)
    if maximum is not None and number > maximum:
        raise _invalid(f"{name} must be <= {maximum}", path)
    return int(value) if integer else float(value)


def reject_nul(value: str, name: str, path: Path) -> str:
    if "\x00" in value:
        raise _invalid(f"{name} contains a NUL byte", path)
    return value


def resolve_path(value: str, base: Path, name: str, path: Path) -> Path:
    """Resolve a user path relative to the project dir (strict=False)."""
    text = reject_nul(str(value), name, path)
    candidate = Path(text).expanduser()
    if not candidate.is_absolute():
        candidate = base / candidate
    return candidate.resolve(strict=False)


def _path_value(value: Any, name: str, base: Path, path: Path) -> Path:
    return resolve_path(_string(value, name, path), base, name, path)


def _checkpoint_value(value: Any, name: str, base: Path, path: Path) -> Any:
    if isinstance(value, bool):
        raise _invalid(f"{name} must be 'last', an epoch number or a path", path)
    if isinstance(value, int):
        if value < 0:
            raise _invalid(f"{name} must be >= 0", path)
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise _invalid(f"{name} must not be empty", path)
        if text == "last":
            return "last"
        # Every other non-empty string is a checkpoint directory path and is
        # resolved like every other path in the project.
        return resolve_path(text, base, name, path)
    raise _invalid(f"{name} must be 'last', an epoch number or a path", path)


# --------------------------------------------------------------------------- #
# Block parsers
# --------------------------------------------------------------------------- #
def _parse_dataset(value: Any, base: Path, path: Path) -> DatasetSpec:
    data = _mapping(value, "dataset", path)
    _unknown(data, _DATASET_KEYS, "dataset", path)
    if "ref_audio" not in data:
        raise _invalid("dataset.ref_audio is required", path)
    ref_audio = _path_value(data["ref_audio"], "dataset.ref_audio", base, path)
    language = "Auto"
    if "language" in data:
        language = _string(
            data["language"], "dataset.language", path, limit=MAX_LANGUAGE_CHARS
        )
    if "items" not in data:
        raise _invalid("dataset.items is required", path)
    raw_items = data["items"]
    if not isinstance(raw_items, list) or not raw_items:
        raise _invalid("dataset.items must be a non-empty list", path)
    if len(raw_items) > MAX_ITEMS:
        raise _invalid(f"dataset.items has too many entries (limit {MAX_ITEMS})", path)
    items: list[DatasetItem] = []
    seen: dict[Path, int] = {}
    for index, raw in enumerate(raw_items):
        label = f"dataset.items[{index}]"
        item = _mapping(raw, label, path)
        _unknown(item, _ITEM_KEYS, label, path)
        if "audio" not in item or "text" not in item:
            raise _invalid(f"{label} requires audio and text", path)
        audio = _path_value(item["audio"], f"{label}.audio", base, path)
        text = _string(item["text"], f"{label}.text", path, limit=MAX_TEXT_CHARS)
        if audio in seen:
            raise _invalid(
                f"duplicate dataset audio {audio} (items {seen[audio]} and {index})",
                path,
            )
        seen[audio] = index
        items.append(DatasetItem(audio=audio, text=text))
    return DatasetSpec(ref_audio=ref_audio, language=language, items=items)


def _parse_prepare(value: Any, base: Path, path: Path) -> PrepareSpec:
    data = _mapping(value, "prepare", path)
    _unknown(data, _PREPARE_KEYS, "prepare", path)
    spec = PrepareSpec()
    if "tokenizer_model_path" in data:
        spec.tokenizer_model_path = _path_value(
            data["tokenizer_model_path"], "prepare.tokenizer_model_path", base, path
        )
    if "device" in data:
        device = _string(data["device"], "prepare.device", path)
        if device not in DEVICE_CHOICES:
            raise _invalid(
                "prepare.device must be one of " + ", ".join(DEVICE_CHOICES), path
            )
        spec.device = device
    if "raw_jsonl" in data:
        spec.raw_jsonl = _path_value(data["raw_jsonl"], "prepare.raw_jsonl", base, path)
    if "output_jsonl" in data:
        spec.output_jsonl = _path_value(
            data["output_jsonl"], "prepare.output_jsonl", base, path
        )
    return spec


def _parse_train(value: Any, base: Path, path: Path) -> TrainSpec:
    data = _mapping(value, "train", path)
    _unknown(data, _TRAIN_KEYS, "train", path)
    spec = TrainSpec()
    for key in ("finetuning_dir", "init_model_path", "output_dir", "train_jsonl"):
        if key in data:
            setattr(spec, key, _path_value(data[key], f"train.{key}", base, path))
    if "python" in data:
        spec.python = _string(data["python"], "train.python", path)
    if "speaker_name" in data:
        if not isinstance(data["speaker_name"], str):
            raise _invalid("train.speaker_name must be a string", path)
        speaker = data["speaker_name"].strip().lower()
        if not SPEAKER_RE.match(speaker):
            raise _invalid(
                "train.speaker_name must match ^[a-z0-9][a-z0-9_.-]{0,63}$", path
            )
        spec.speaker_name = speaker
    if "batch_size" in data:
        spec.batch_size = _number(
            data["batch_size"],
            "train.batch_size",
            path,
            minimum=1,
            maximum=1024,
            integer=True,
        )
    if "lr" in data:
        spec.lr = float(
            _number(
                data["lr"],
                "train.lr",
                path,
                minimum=0.0,
                maximum=1.0,
                exclusive_minimum=True,
            )
        )
    if "num_epochs" in data:
        spec.num_epochs = _number(
            data["num_epochs"],
            "train.num_epochs",
            path,
            minimum=1,
            maximum=1000,
            integer=True,
        )
    if "checkpoint" in data:
        spec.checkpoint = _checkpoint_value(
            data["checkpoint"], "train.checkpoint", base, path
        )
    return spec


def _parse_consent(value: Any, path: Path) -> dict[str, Any]:
    data = _mapping(value, "register.consent", path)
    _unknown(data, _CONSENT_KEYS, "register.consent", path)
    rights = data.get("rights_confirmed")
    if not isinstance(rights, bool):
        raise _invalid("register.consent.rights_confirmed must be a boolean", path)
    license_value = data.get("license")
    note = data.get("note", "")
    if license_value is not None and not isinstance(license_value, str):
        raise _invalid("register.consent.license must be a string", path)
    if not isinstance(note, str):
        raise _invalid("register.consent.note must be a string", path)
    return {"rights_confirmed": rights, "license": license_value, "note": note}


def _parse_register(value: Any, path: Path) -> RegisterSpec:
    data = _mapping(value, "register", path)
    _unknown(data, _REGISTER_KEYS, "register", path)
    spec = RegisterSpec()
    if "name" in data:
        name = _string(data["name"], "register.name", path)
        validate_name(name)
        spec.name = name
    if "consent" in data:
        spec.consent = _parse_consent(data["consent"], path)
    return spec


def _parse_eval(value: Any, base: Path, path: Path) -> EvalSpec:
    data = _mapping(value, "eval", path)
    _unknown(data, _EVAL_KEYS, "eval", path)
    spec = EvalSpec()
    if "language" in data:
        spec.language = _string(data["language"], "eval.language", path)
    raw_items = data.get("items")
    if not isinstance(raw_items, list) or not raw_items:
        raise _invalid("eval.items must be a non-empty list", path)
    if len(raw_items) > MAX_ITEMS:
        raise _invalid(f"eval.items has too many entries (limit {MAX_ITEMS})", path)
    for index, raw in enumerate(raw_items):
        label = f"eval.items[{index}]"
        item = _mapping(raw, label, path)
        _unknown(item, _EVAL_ITEM_KEYS, label, path)
        if "text" not in item:
            raise _invalid(f"{label}.text is required", path)
        text = _string(item["text"], f"{label}.text", path, limit=MAX_TEXT_CHARS)
        language = None
        if "language" in item:
            language = _string(item["language"], f"{label}.language", path)
        spec.items.append(EvalItem(text=text, language=language))
    for key in ("asr_model", "base_model"):
        if key in data:
            setattr(spec, key, _string(data[key], f"eval.{key}", path))
    if "thresholds" in data:
        thresholds = _mapping(data["thresholds"], "eval.thresholds", path)
        _unknown(thresholds, _THRESHOLD_KEYS, "eval.thresholds", path)
        if "max_cer" in thresholds:
            spec.max_cer = float(
                _number(thresholds["max_cer"], "eval.thresholds.max_cer", path, minimum=0)
            )
        if "min_speaker_similarity" in thresholds:
            spec.min_speaker_similarity = float(
                _number(
                    thresholds["min_speaker_similarity"],
                    "eval.thresholds.min_speaker_similarity",
                    path,
                    minimum=0,
                    maximum=1,
                )
            )
    if "audio_dir" in data:
        spec.audio_dir = _path_value(data["audio_dir"], "eval.audio_dir", base, path)
    if "checkpoint" in data:
        spec.checkpoint = _checkpoint_value(
            data["checkpoint"], "eval.checkpoint", base, path
        )
    return spec


# --------------------------------------------------------------------------- #
# Containment / distinctness
# --------------------------------------------------------------------------- #
def _overlaps(a: Path, b: Path) -> bool:
    return a == b or a in b.parents or b in a.parents


def output_containment_conflict(
    output_dir: Path, others: Iterable[Optional[Path]]
) -> Optional[Path]:
    for other in others:
        if other is not None and _overlaps(output_dir, other):
            return other
    return None


def validate_output_containment(
    output_dir: Path,
    *,
    project_file: Path,
    official_dir: Optional[Path],
    init_model: Optional[Path],
    jsonl_paths: Iterable[Optional[Path]],
    input_paths: Iterable[Optional[Path]],
    data_dir: Optional[Path],
    profile_dir: Optional[Path],
) -> None:
    others: list[Optional[Path]] = [
        project_file,
        official_dir,
        init_model,
        data_dir,
        profile_dir,
        *jsonl_paths,
        *input_paths,
    ]
    conflict = output_containment_conflict(output_dir, others)
    if conflict is not None:
        raise InputError(
            f"train.output_dir {output_dir} overlaps an input or store path: {conflict}",
            details={"output_dir": str(output_dir), "conflict": str(conflict)},
        )


def ensure_pairwise_distinct(paths: Iterable[tuple[str, Optional[Path]]]) -> None:
    entries = [(label, path) for label, path in paths if path is not None]
    for index, (label_a, path_a) in enumerate(entries):
        for label_b, path_b in entries[index + 1 :]:
            if path_a == path_b:
                raise InputError(
                    f"paths must be distinct: {label_a} and {label_b} both resolve to {path_a}",
                    details={
                        "path": str(path_a),
                        "first": label_a,
                        "second": label_b,
                    },
                )


def check_pairwise_distinct(paths: Iterable[tuple[str, Optional[Path]]]) -> None:
    ensure_pairwise_distinct(paths)


# --------------------------------------------------------------------------- #
# Loader
# --------------------------------------------------------------------------- #
def default_raw_jsonl(project: Project) -> Path:
    return project.project_dir / f"{project.path.stem}.raw.jsonl"


def default_output_jsonl(project: Project) -> Path:
    return project.project_dir / f"{project.path.stem}.train.jsonl"


def load_project(
    path_value: str | Path,
    *,
    data_dir: str | Path | None = None,
    register_name: str | None = None,
) -> Project:
    raw_path = Path(path_value).expanduser()
    suffix = raw_path.suffix.lower()
    if suffix not in (".yaml", ".yml", ".json"):
        raise InputError(
            f"finetune project must be .yaml/.yml/.json: {raw_path}",
            details={"project": str(raw_path)},
        )
    path = raw_path.resolve(strict=False)
    if not path.is_file():
        raise InputError(
            f"finetune project not found: {path}", details={"project": str(path)}
        )
    raw = _read_text(path, MAX_PROJECT_BYTES, "finetune project")
    try:
        data = _load_project_data(path, raw)
    except InputError:
        raise
    _check_tree(data, path)
    if not isinstance(data, dict):
        raise _invalid("finetune project must be a mapping", path)
    _unknown(data, _TOP_KEYS, "project", path)
    version = data.get("schema_version")
    if isinstance(version, bool) or not isinstance(version, int) or version != SCHEMA_VERSION:
        raise _invalid("schema_version must be the integer 1", path)
    if "dataset" not in data:
        raise _invalid("dataset is required", path)
    base = path.parent
    dataset = _parse_dataset(data["dataset"], base, path)
    prepare = (
        _parse_prepare(data["prepare"], base, path)
        if "prepare" in data
        else PrepareSpec()
    )
    train = _parse_train(data["train"], base, path) if "train" in data else TrainSpec()
    register = (
        _parse_register(data["register"], path)
        if "register" in data
        else RegisterSpec()
    )
    eval_spec = _parse_eval(data["eval"], base, path) if "eval" in data else None

    project = Project(
        path=path,
        project_dir=base,
        dataset=dataset,
        prepare=prepare,
        train=train,
        register=register,
        eval=eval_spec,
    )

    if train.output_dir is not None:
        profile = None
        name = register_name if register_name is not None else register.name
        if data_dir is not None and name:
            profile = Path(data_dir).expanduser() / "voices" / name
        validate_output_containment(
            train.output_dir,
            project_file=path,
            official_dir=train.finetuning_dir,
            init_model=train.init_model_path,
            jsonl_paths=(
                prepare.raw_jsonl,
                prepare.output_jsonl,
                train.train_jsonl,
            ),
            input_paths=(
                dataset.ref_audio,
                *(item.audio for item in dataset.items),
            ),
            data_dir=(
                Path(data_dir).expanduser().resolve(strict=False)
                if data_dir is not None
                else None
            ),
            profile_dir=profile.resolve(strict=False) if profile is not None else None,
        )
    return project
