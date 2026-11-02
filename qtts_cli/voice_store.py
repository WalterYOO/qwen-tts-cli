"""Voice asset store (FR-P0-02, PRD §8.1).

Layout::

    <data-dir>/voices/<name>/
      voice.json          metadata (no binary payloads)
      prompt.safetensors  clone prompt tensors (never pickle)
      ref.wav             reference audio
      preview.wav         fixed audition line
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .errors import InputError
from .utils import atomic_write_json, read_json, sha256_file

SCHEMA_VERSION = 1
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_RESERVED = {"builtin", "design", "voice"}


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def validate_name(name: str) -> str:
    if not name or not _NAME_RE.match(name):
        raise InputError(
            f"invalid voice name: {name!r}; use letters, digits, '_', '-', '.' (max 64, must not start with '.')",
            details={"name": name},
        )
    if name.lower() in _RESERVED:
        raise InputError(f"voice name {name!r} is reserved")
    return name


class VoiceStore:
    def __init__(self, data_dir: str | Path) -> None:
        self.root = Path(data_dir).expanduser() / "voices"

    # -- paths -------------------------------------------------------------- #
    def dir_for(self, name: str) -> Path:
        return self.root / validate_name(name)

    def exists(self, name: str) -> bool:
        return (self.dir_for(name) / "voice.json").is_file()

    def require(self, name: str) -> dict[str, Any]:
        meta = self.get(name)
        if meta is None:
            raise InputError(
                f"voice not found: {name}",
                details={"voice": name, "store": str(self.root)},
            )
        return meta

    # -- read --------------------------------------------------------------- #
    def get(self, name: str) -> Optional[dict[str, Any]]:
        path = self.dir_for(name) / "voice.json"
        if not path.is_file():
            return None
        try:
            meta = read_json(path)
        except (OSError, ValueError) as exc:
            raise InputError(f"corrupt voice metadata: {path}: {exc}") from exc
        if not isinstance(meta, dict) or meta.get("name") != name:
            raise InputError(f"invalid voice metadata: {path}")
        return meta

    def list(self) -> list[dict[str, Any]]:
        if not self.root.is_dir():
            return []
        out: list[dict[str, Any]] = []
        for child in sorted(self.root.iterdir()):
            if not child.is_dir():
                continue
            try:
                meta = self.get(child.name)
            except InputError:
                continue
            if meta is not None:
                out.append(meta)
        return out

    # -- write -------------------------------------------------------------- #
    def staging_dir(self, name: str) -> Path:
        """A temp directory beside the final profile for an atomic commit (P1-2)."""
        name = validate_name(name)
        self.root.mkdir(parents=True, exist_ok=True)
        return Path(tempfile.mkdtemp(prefix=f".{name}.", dir=str(self.root)))

    def commit_staging(self, staging: str | Path, name: str) -> Path:
        """Atomically publish a fully written staging directory."""
        name = validate_name(name)
        target = self.dir_for(name)
        if target.exists():
            raise InputError(f"voice already exists: {name}")
        os.replace(Path(staging), target)
        return target

    def save_meta(self, meta: dict[str, Any], directory: Optional[Path] = None) -> Path:
        name = validate_name(str(meta.get("name", "")))
        base = Path(directory) if directory is not None else self.dir_for(name)
        path = base / "voice.json"
        atomic_write_json(path, meta)
        return path

    def create(
        self,
        *,
        name: str,
        kind: str,
        source: dict[str, Any],
        model: dict[str, Any],
        tokenizer: Optional[dict[str, Any]] = None,
        dtype: Optional[str] = None,
        language: Optional[str] = None,
        default_instruct: Optional[str] = None,
        tags: Optional[list[str]] = None,
        consent: Optional[dict[str, Any]] = None,
        qtts_version: str = "0.1.0",
        staging: Optional[Path] = None,
    ) -> dict[str, Any]:
        name = validate_name(name)
        if self.exists(name):
            raise InputError(
                f"voice already exists: {name}; remove it first or choose another name",
                details={"voice": name},
            )
        meta = {
            "schema_version": SCHEMA_VERSION,
            "name": name,
            "kind": kind,
            "created_at": utc_now(),
            "source": source,
            "model": model,
            "tokenizer": tokenizer or {"type": "12hz", "codec_rate_hz": 12.5},
            "dtype": dtype,
            "language": language,
            "default_instruct": default_instruct,
            "tags": list(tags or []),
            "consent": consent or {"rights_confirmed": False, "license": None, "note": ""},
            "qtts_version": qtts_version,
        }
        base = Path(staging) if staging is not None else self.dir_for(name)
        base.mkdir(parents=True, exist_ok=True)
        self.save_meta(meta, directory=base)
        return meta

    def copy_ref_audio(self, name: str, src: str | Path, directory: Optional[Path] = None) -> Path:
        base = Path(directory) if directory is not None else self.dir_for(name)
        target = base / "ref.wav"
        _atomic_copy(Path(src).expanduser(), target)
        return target

    def save_prompt_tensors(
        self,
        name: str,
        *,
        ref_spk_embedding: Any,
        ref_code: Any = None,
        x_vector_only_mode: bool = False,
        icl_mode: bool = True,
        directory: Optional[Path] = None,
    ) -> Path:
        """Persist clone prompt tensors with safetensors (no pickle)."""
        import torch
        from safetensors.torch import save_file

        tensors: dict[str, Any] = {
            "ref_spk_embedding": _to_cpu_contiguous(ref_spk_embedding),
            "flags": torch.tensor(
                [1.0 if x_vector_only_mode else 0.0, 1.0 if icl_mode else 0.0], dtype=torch.float32
            ),
        }
        if ref_code is not None:
            tensors["ref_code"] = _to_cpu_contiguous(ref_code)
        base = Path(directory) if directory is not None else self.dir_for(name)
        base.mkdir(parents=True, exist_ok=True)
        path = base / "prompt.safetensors"
        # safetensors writes in place; stage a unique temp then replace so a
        # concurrent reader never sees a partial file (codex P1-8).
        fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(base))
        os.close(fd)
        try:
            save_file(tensors, tmp)
            with open(tmp, "rb") as fh:
                os.fsync(fh.fileno())
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        return path

    def load_prompt_tensors(self, name: str) -> dict[str, Any]:
        from safetensors.torch import load_file

        path = self.dir_for(name) / "prompt.safetensors"
        if not path.is_file():
            raise InputError(
                f"voice {name} has no prompt.safetensors; re-add the voice or run --materialize-clone",
                details={"voice": name, "path": str(path)},
            )
        tensors = load_file(str(path))
        flags = tensors.pop("flags", None)
        if flags is not None:
            tensors["x_vector_only_mode"] = bool(flags[0].item())
            tensors["icl_mode"] = bool(flags[1].item())
        else:
            tensors["x_vector_only_mode"] = False
            tensors["icl_mode"] = True
        return tensors

    def set_preview(self, name: str, src: str | Path, directory: Optional[Path] = None) -> Path:
        base = Path(directory) if directory is not None else self.dir_for(name)
        target = base / "preview.wav"
        _atomic_copy(Path(src).expanduser(), target)
        return target

    def prompt_sha256(self, name: str) -> Optional[str]:
        path = self.dir_for(name) / "prompt.safetensors"
        return sha256_file(path) if path.is_file() else None

    def remove(self, name: str) -> None:
        path = self.dir_for(validate_name(name))
        if not path.is_dir():
            raise InputError(f"voice not found: {name}", details={"voice": name})
        shutil.rmtree(path)


def _atomic_copy(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{dest.name}.", suffix=".tmp", dir=str(dest.parent))
    try:
        with os.fdopen(fd, "wb") as out, open(src, "rb") as inp:
            shutil.copyfileobj(inp, out)
            out.flush()
            os.fsync(out.fileno())
        os.replace(tmp, dest)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _to_cpu_contiguous(tensor: Any) -> Any:
    try:
        return tensor.detach().to("cpu").contiguous()
    except AttributeError:  # already a plain value
        return tensor
