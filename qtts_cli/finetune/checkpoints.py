"""Checkpoint discovery, validation, selection and atomic publication (§6)."""

from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .. import catalog
from ..errors import InputError, ModelError, QttsError
from .runner import DeferringHandlers

EPOCH_RE = re.compile(r"^checkpoint-epoch-(0|[1-9][0-9]*)$")


@dataclass
class Checkpoint:
    epoch: int
    path: Path
    fingerprint: str
    model: Any = None

    def to_dict(self, *, selected: bool = False) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "path": str(self.path),
            "fingerprint": self.fingerprint,
            "selected": selected,
        }


def parse_epoch(name: str) -> Optional[int]:
    match = EPOCH_RE.match(name)
    return int(match.group(1)) if match else None


def discover(output_dir: Path) -> list[tuple[int, Path]]:
    if not output_dir.is_dir():
        return []
    found: list[tuple[int, Path]] = []
    try:
        children = list(output_dir.iterdir())
    except OSError as exc:
        raise ModelError(
            f"cannot list output directory {output_dir}: {exc}",
            details={"path": str(output_dir)},
        ) from exc
    for child in children:
        if not child.is_dir() or child.is_symlink():
            continue
        epoch = parse_epoch(child.name)
        if epoch is not None:
            found.append((epoch, child))
        elif child.name.startswith("checkpoint-epoch-"):
            raise _malformed(child, "directory name is not checkpoint-epoch-<N>")
    found.sort(key=lambda item: item[0])
    return found


def _malformed(path: Path, message: str) -> ModelError:
    return ModelError(
        f"malformed checkpoint {path}: {message}", details={"path": str(path)}
    )


def validate_checkpoint(
    path: Path, speaker_name: str, *, require_speaker: bool = True
) -> Checkpoint:
    epoch = parse_epoch(path.name)
    if epoch is None:
        raise _malformed(path, "directory name is not checkpoint-epoch-<N>")
    if not path.is_dir():
        raise _malformed(path, "not a directory")
    config_path = path / "config.json"
    try:
        with open(config_path, "r", encoding="utf-8") as handle:
            config = json.load(handle)
    except OSError as exc:
        raise _malformed(path, f"cannot read config.json: {exc}") from exc
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise _malformed(path, f"invalid config.json: {exc}") from exc
    if not isinstance(config, dict):
        raise _malformed(path, "config.json must be an object")
    weights = path / "model.safetensors"
    try:
        if not weights.is_file():
            raise _malformed(path, "model.safetensors is missing")
        if weights.stat().st_size <= 0:
            raise _malformed(path, "model.safetensors is empty")
    except OSError as exc:
        raise _malformed(path, f"cannot stat model.safetensors: {exc}") from exc
    if config.get("model_type") != "qwen3_tts":
        raise _malformed(path, 'config.model_type must be "qwen3_tts"')
    if config.get("tts_model_type") != "custom_voice":
        raise _malformed(path, 'config.tts_model_type must be "custom_voice"')
    talker = config.get("talker_config")
    if not isinstance(talker, dict):
        raise _malformed(path, "config.talker_config must be an object")
    speaker_ids = talker.get("spk_id")
    if not isinstance(speaker_ids, dict):
        raise _malformed(path, "config.talker_config.spk_id must be an object")
    if require_speaker and speaker_ids.get(speaker_name) != 3000:
        raise _malformed(
            path,
            f"config.talker_config.spk_id[{speaker_name!r}] must be 3000",
        )
    if not (path / "speech_tokenizer" / "config.json").is_file():
        raise _malformed(path, "speech_tokenizer/config.json is missing")
    tokenizer_weights = sorted((path / "speech_tokenizer").glob("*.safetensors"))
    if not any(candidate.is_file() for candidate in tokenizer_weights):
        raise _malformed(path, "speech_tokenizer/*.safetensors is missing")
    try:
        info = catalog.load_model(path)
    except QttsError as exc:
        raise _malformed(path, f"model is not loadable: {exc.message}") from exc
    if info.kind != "builtin":
        raise _malformed(path, f"model kind is {info.kind!r}, expected 'builtin'")
    if require_speaker and info.canonical_speaker(speaker_name) is None:
        raise _malformed(path, f"speaker {speaker_name!r} is not in the model")
    return Checkpoint(epoch=epoch, path=path, fingerprint=info.fingerprint, model=info)


def validate_output(
    output_dir: Path, speaker_name: str, *, require_speaker: bool = True
) -> list[Checkpoint]:
    found = discover(output_dir)
    if not found:
        raise ModelError(
            "training produced no checkpoint-epoch-*",
            details={"output_dir": str(output_dir)},
        )
    return [
        validate_checkpoint(path, speaker_name, require_speaker=require_speaker)
        for _, path in found
    ]


def _require_output(output_dir: Optional[Path]) -> Path:
    if output_dir is None:
        raise InputError("train.output_dir is required for checkpoint selection")
    if not output_dir.is_dir():
        raise InputError(
            f"output directory not found: {output_dir}",
            details={"path": str(output_dir)},
        )
    return output_dir


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def select_checkpoint(
    selector: Any,
    *,
    output_dir: Optional[Path],
    speaker_name: str,
    require_inside: Optional[Path] = None,
    require_speaker: bool = True,
) -> Checkpoint:
    if selector is None or (isinstance(selector, str) and selector == "last"):
        root = _require_output(output_dir)
        checkpoints = validate_output(root, speaker_name, require_speaker=require_speaker)
        return checkpoints[-1]
    if isinstance(selector, bool):
        raise InputError("invalid checkpoint selector")
    if isinstance(selector, int):
        root = _require_output(output_dir)
        if selector < 0:
            raise InputError("checkpoint epoch must be non-negative")
        path = root / f"checkpoint-epoch-{selector}"
        if not path.is_dir():
            raise InputError(
                f"checkpoint epoch {selector} not found in {root}",
                details={"path": str(path), "epoch": selector},
            )
        return validate_checkpoint(path, speaker_name, require_speaker=require_speaker)
    if isinstance(selector, (str, Path)) and str(selector).strip():
        path = Path(selector).expanduser().resolve(strict=False)
        if not path.is_dir():
            raise InputError(
                f"checkpoint directory not found: {path}", details={"path": str(path)}
            )
        if require_inside is not None and not _inside(path, require_inside):
            raise InputError(
                f"checkpoint {path} is outside the published output {require_inside}",
                details={"path": str(path), "output_dir": str(require_inside)},
            )
        try:
            return validate_checkpoint(
                path, speaker_name, require_speaker=require_speaker
            )
        except ModelError as exc:
            raise InputError(
                f"checkpoint {path} is invalid: {exc.message}",
                details={"path": str(path)},
            ) from exc
    raise InputError(
        "checkpoint must be 'last', a non-negative epoch number or a directory",
        details={"checkpoint": str(selector)},
    )


def profile_checkpoint(checkpoint: Checkpoint, speaker_name: str) -> dict[str, Any]:
    return {
        "epoch": checkpoint.epoch,
        "path": str(checkpoint.path),
        "speaker": speaker_name,
        "fingerprint": checkpoint.fingerprint,
    }


# --------------------------------------------------------------------------- #
# Publication
# --------------------------------------------------------------------------- #
def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def remove_stale_old(output_dir: Path) -> None:
    old = output_dir.with_name(f"{output_dir.name}.old")
    if old.exists() or old.is_symlink():
        try:
            _remove_path(old)
        except OSError as exc:
            raise ModelError(
                f"cannot remove stale backup {old}: {exc}", details={"path": str(old)}
            ) from exc


def publish_output_dir(partial: Path, final: Path) -> list[str]:
    """Atomically publish ``partial`` as ``final`` with rollback (§5.3)."""
    warnings: list[str] = []
    old = final.with_name(f"{final.name}.old")
    deferring = DeferringHandlers()
    moved_old = False
    with deferring:
        try:
            if final.exists() or final.is_symlink():
                if old.exists() or old.is_symlink():
                    _remove_path(old)
                os.replace(final, old)
                moved_old = True
            os.replace(partial, final)  # commit point
        except BaseException as exc:
            if moved_old:
                try:
                    os.replace(old, final)
                except OSError as restore_exc:
                    raise ModelError(
                        "output publication failed and rollback also failed; "
                        f"manual recovery required: {old} -> {final}, "
                        f"partial kept at {partial}: {restore_exc}",
                        details={
                            "old": str(old),
                            "partial": str(partial),
                            "final": str(final),
                            "published": False,
                        },
                    ) from restore_exc
            if isinstance(exc, OSError):
                raise ModelError(
                    f"cannot publish output directory {final}: {exc}",
                    details={"partial": str(partial), "final": str(final)},
                ) from exc
            raise
        if moved_old:
            try:
                _remove_path(old)
            except OSError as exc:
                warnings.append(f"could not remove backup {old}: {exc}")
    deferring.raise_if_deferred(published=True, registered=False)
    return warnings
