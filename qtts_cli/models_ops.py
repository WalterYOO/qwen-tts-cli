"""`models list/pull/prune` (FR-P0-08)."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Optional

from .catalog import _load_one
from .errors import InputError, ModelError
from .utils import atomic_write_text, confirm, human_bytes, is_tty


def _dir_size(path: Path) -> int:
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file():
                total += item.stat().st_size
        except OSError:
            continue
    return total


def list_models(cfg) -> list[dict[str, Any]]:
    model_dir = Path(cfg.model_dir) if cfg.model_dir else None
    if model_dir is None or not model_dir.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for child in sorted(model_dir.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        info = _load_one(child)
        entry: dict[str, Any] = {
            "name": child.name,
            "path": str(child),
            "size_bytes": _dir_size(child),
            "size": human_bytes(_dir_size(child)),
        }
        if info is not None:
            entry.update(
                {
                    "kind": info.kind,
                    "tts_model_type": info.tts_model_type,
                    "size_tier": info.size,
                    "speakers": len(info.speakers),
                    "languages": info.languages,
                    "revision": info.revision,
                }
            )
            entry["complete"] = _complete(child, info.kind)
        else:
            entry["kind"] = "unknown"
            entry["complete"] = False
        out.append(entry)
    return out


def _complete(model_dir: Path, kind: str) -> bool:
    if not (model_dir / "config.json").is_file():
        return False
    if kind == "align":
        return bool(list(model_dir.glob("*.safetensors")))
    return bool(list(model_dir.glob("*.safetensors"))) and (
        model_dir / "speech_tokenizer" / "model.safetensors"
    ).is_file()


def pull(
    cfg,
    model_id: str,
    *,
    source: str = "hf",
    revision: Optional[str] = None,
) -> dict[str, Any]:
    if cfg.offline:
        raise InputError("--offline is set; refusing to download models")
    model_dir = Path(cfg.model_dir) if cfg.model_dir else None
    if model_dir is None:
        raise ModelError("no model directory configured; pass --model-dir DIR")
    model_dir.mkdir(parents=True, exist_ok=True)
    target = model_dir / model_id.rstrip("/").split("/")[-1]
    source = (source or "hf").lower()
    if source not in ("hf", "modelscope"):
        raise InputError(f"unsupported source: {source!r}; expected hf or modelscope")

    if source == "hf":
        try:
            from huggingface_hub import snapshot_download
        except Exception as exc:  # pragma: no cover
            raise ModelError("huggingface_hub is not installed; pip install 'qwen-tts-cli[hf]'") from exc
        kwargs: dict[str, Any] = {"repo_id": model_id, "local_dir": str(target)}
        if revision:
            kwargs["revision"] = revision
        try:
            snapshot_download(**kwargs)
        except Exception as exc:
            raise ModelError(f"failed to download {model_id} from Hugging Face: {exc}") from exc
    else:
        try:
            from modelscope import snapshot_download as ms_download
        except Exception:
            try:
                from modelscope.hub.snapshot_download import snapshot_download as ms_download
            except Exception as exc:  # pragma: no cover
                raise ModelError("modelscope is not installed; pip install 'qwen-tts-cli[modelscope]'") from exc
        try:
            try:
                ms_download(model_id, revision=revision, local_dir=str(target))
            except TypeError:
                ms_download(model_id, revision=revision, cache_dir=str(model_dir))
        except Exception as exc:
            raise ModelError(f"failed to download {model_id} from ModelScope: {exc}") from exc

    if revision:
        atomic_write_text(target / "revision.txt", revision + "\n")
    if not (target / "config.json").is_file():
        raise ModelError(
            f"download finished but {target / 'config.json'} is missing; the model id may be wrong",
            details={"target": str(target)},
        )
    info = _load_one(target)
    return {
        "ok": True,
        "model": model_id,
        "source": source,
        "revision": revision,
        "path": str(target),
        "kind": info.kind if info else "unknown",
        "size": human_bytes(_dir_size(target)),
    }


def prune(cfg, *, yes: bool = False, dry_run: bool = False) -> dict[str, Any]:
    model_dir = Path(cfg.model_dir) if cfg.model_dir else None
    if model_dir is None or not model_dir.is_dir():
        return {"ok": True, "removed": [], "candidates": []}
    candidates: list[Path] = []
    for child in sorted(model_dir.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        info = _load_one(child)
        if info is None or not _complete(child, info.kind):
            candidates.append(child)
    result = {
        "ok": True,
        "dry_run": dry_run,
        "candidates": [{"name": p.name, "path": str(p), "size": human_bytes(_dir_size(p))} for p in candidates],
        "removed": [],
    }
    if not candidates or dry_run:
        return result
    if not yes and not is_tty():
        raise InputError("prune would delete incomplete models; pass --yes to confirm")
    if not yes and not confirm(f"Delete {len(candidates)} incomplete model director{'y' if len(candidates) == 1 else 'ies'}?"):
        result["cancelled"] = True
        return result
    for path in candidates:
        shutil.rmtree(path)
        result["removed"].append(path.name)
    return result
