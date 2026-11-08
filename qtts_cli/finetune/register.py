"""Finetune profile construction and atomic replacement (§5.3/§6.3)."""

from __future__ import annotations

import hashlib
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Optional

from .. import __version__
from ..errors import ModelError
from ..utils import sha256_text
from ..voice_store import VoiceStore, utc_now
from .checkpoints import Checkpoint
from .runner import DeferringHandlers


def sha256_stream(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    try:
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
                size += len(chunk)
    except OSError as exc:
        raise ModelError(
            f"cannot hash {path}: {exc}", details={"path": str(path)}
        ) from exc
    return digest.hexdigest(), size


def build_checkpoint_hashes(
    checkpoint: Checkpoint, speaker_name: str
) -> dict[str, Any]:
    weights_sha, weights_size = sha256_stream(checkpoint.path / "model.safetensors")
    config_sha, _ = sha256_stream(checkpoint.path / "config.json")
    checkpoint_sha = sha256_text(f"{weights_sha}:{config_sha}:{speaker_name}")
    return {
        "weights_sha256": weights_sha,
        "weights_size": weights_size,
        "config_sha256": config_sha,
        "checkpoint_sha256": checkpoint_sha,
    }


def build_voice_meta(
    *,
    store: VoiceStore,
    name: str,
    checkpoint: Checkpoint,
    speaker_name: str,
    dataset,
    dataset_sha: str,
    ref_sha: str,
    raw_jsonl_sha: Optional[str],
    prepared_jsonl_sha: Optional[str],
    output_dir: Optional[Path],
    init_model: Any,
    batch_size: int,
    lr: float,
    num_epochs: int,
    consent: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    hashes = build_checkpoint_hashes(checkpoint, speaker_name)
    source_output = (
        str(output_dir) if output_dir is not None else str(checkpoint.path.parent)
    )
    init_payload = (
        None
        if init_model is None
        else {
            "id": init_model.model_id,
            "path": str(init_model.path),
            "fingerprint": init_model.fingerprint,
        }
    )
    meta = {
        "schema_version": 1,
        "name": name,
        "kind": "finetune",
        "created_at": utc_now(),
        "source": {
            "type": "finetune",
            "speaker": speaker_name,
            "checkpoint": checkpoint.path.name,
            "weights_sha256": hashes["weights_sha256"],
            "weights_size": hashes["weights_size"],
            "config_sha256": hashes["config_sha256"],
            "checkpoint_sha256": hashes["checkpoint_sha256"],
            "output_dir": source_output,
            "init_model": init_payload,
            "dataset": {
                "items": len(dataset.items),
                "items_sha256": dataset_sha,
                "ref_audio_sha256": ref_sha,
                "raw_jsonl_sha256": raw_jsonl_sha,
                "prepared_jsonl_sha256": prepared_jsonl_sha,
            },
            "train": {
                "batch_size": int(batch_size),
                "lr": float(lr),
                "num_epochs": int(num_epochs),
                "epoch": checkpoint.epoch,
            },
        },
        "model": {
            "id": checkpoint.path.name,
            "revision": None,
            "fingerprint": hashes["checkpoint_sha256"][:16],
            "path": str(checkpoint.path),
        },
        "tokenizer": {"type": "12hz", "codec_rate_hz": 12.5},
        "dtype": None,
        "language": dataset.language,
        "default_instruct": None,
        "tags": [],
        "consent": consent,
        "qtts_version": __version__,
    }
    return meta, hashes


def profile_backups(store: VoiceStore, name: str) -> list[Path]:
    if not store.root.is_dir():
        return []
    return sorted(
        child
        for child in store.root.iterdir()
        if child.name.startswith(f".{name}.old-") and child.is_dir()
    )


def restore_profile_backup(store: VoiceStore, name: str, backup: Path) -> None:
    target = store.dir_for(name)
    try:
        os.replace(backup, target)
    except OSError as exc:
        raise ModelError(
            f"cannot restore profile backup {backup} -> {target}: {exc}",
            details={"backup": str(backup), "profile": str(target)},
        ) from exc


def remove_profile_backups(store: VoiceStore, name: str) -> None:
    for backup in profile_backups(store, name):
        try:
            shutil.rmtree(backup)
        except OSError as exc:
            raise ModelError(
                f"cannot remove profile backup {backup}: {exc}",
                details={"path": str(backup)},
            ) from exc


def write_staging_profile(
    store: VoiceStore, name: str, meta: dict[str, Any]
) -> Path:
    """Build the complete staging profile; caller owns its removal on error."""
    staging = store.staging_dir(name)
    try:
        store.save_meta(meta, directory=staging)
    except BaseException as exc:
        shutil.rmtree(staging, ignore_errors=True)
        if isinstance(exc, OSError):
            raise ModelError(
                f"cannot write profile staging directory {staging}: {exc}",
                details={"path": str(staging), "profile": name},
            ) from exc
        raise
    return staging


def publish_profile(
    store: VoiceStore,
    staging: Path,
    name: str,
    *,
    published: bool,
) -> list[str]:
    """Commit a staging profile with old-profile rollback (§5.3)."""
    warnings: list[str] = []
    target = store.dir_for(name)
    backup = target.with_name(f".{target.name}.old-{uuid.uuid4().hex[:8]}")
    deferring = DeferringHandlers()
    moved = False
    with deferring:
        try:
            if target.exists() or target.is_symlink():
                os.replace(target, backup)
                moved = True
            os.replace(staging, target)  # commit point
        except BaseException as exc:
            shutil.rmtree(staging, ignore_errors=True)
            if moved:
                try:
                    os.replace(backup, target)
                except OSError as restore_exc:
                    raise ModelError(
                        "profile registration failed and rollback also failed; "
                        f"manual recovery required: {backup} -> {target}: {restore_exc}",
                        details={
                            "backup": str(backup),
                            "staging": str(staging),
                            "profile": str(target),
                            "published": published,
                            "registered": False,
                        },
                    ) from restore_exc
            if isinstance(exc, OSError):
                raise ModelError(
                    f"cannot publish profile {target}: {exc}",
                    details={
                        "path": str(target),
                        "published": published,
                        "registered": False,
                    },
                ) from exc
            raise
        if moved:
            try:
                shutil.rmtree(backup)
            except OSError as exc:
                warnings.append(f"could not remove old profile {backup}: {exc}")
    deferring.raise_if_deferred(published=published, registered=True)
    return warnings
