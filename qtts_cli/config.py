"""Configuration resolution: directories, device/dtype defaults, JSON config file."""

from __future__ import annotations

import json
import os
import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .errors import InputError, UsageError
from .utils import read_json


def default_data_dir() -> Path:
    """platformdirs-compatible user data dir without the dependency."""
    system = platform.system()
    if system == "Darwin":
        base = Path.home() / "Library" / "Application Support"
    elif system == "Windows":
        base = Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming")
    else:
        base = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
    return base / "qtts"


def _first_existing(*candidates: Path) -> Optional[Path]:
    for cand in candidates:
        try:
            if cand.is_dir():
                return cand
        except OSError:
            continue
    return None


def default_model_dir() -> Optional[Path]:
    env = os.environ.get("QTTS_MODEL_DIR") or os.environ.get("QWEN_TTS_MODEL_DIR")
    if env:
        return Path(env).expanduser()
    cwd = Path.cwd()
    return _first_existing(
        cwd / "models",
        cwd / ".." / "Qwen3-TTS" / "models",
        Path.home() / "workspace" / "QwenLM" / "Qwen3-TTS" / "models",
    )


@dataclass
class Config:
    data_dir: Path = field(default_factory=default_data_dir)
    model_dir: Optional[Path] = None
    cache_dir: Optional[Path] = None
    device: str = "auto"
    dtype: str = "auto"
    offline: bool = False
    seed: int = 1234
    json: bool = False
    quiet: bool = False
    no_cache: bool = False
    log_level: str = "warning"
    config_file: Optional[Path] = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir).expanduser()
        if self.model_dir is not None:
            self.model_dir = Path(self.model_dir).expanduser()
        if self.cache_dir is None:
            self.cache_dir = self.data_dir / "cache"
        else:
            self.cache_dir = Path(self.cache_dir).expanduser()

    # -- paths -------------------------------------------------------------- #
    @property
    def voices_dir(self) -> Path:
        return self.data_dir / "voices"

    @property
    def audio_cache_dir(self) -> Path:
        return self.cache_dir / "audio"  # type: ignore[operator]

    # -- resolution --------------------------------------------------------- #
    def resolve_device(self) -> str:
        """auto -> mps > cuda > cpu (PRD §9: explicit, logged, never silent)."""
        if self.device != "auto":
            return self.device
        try:
            import torch  # local import: --dry-run must not import torch
        except Exception:  # pragma: no cover - no runtime installed
            return "cpu"
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda:0"
        return "cpu"

    def torch_dtype(self, device: Optional[str] = None) -> Any:
        import torch  # local import by design

        wanted = self.dtype
        dev = device or self.device
        if wanted == "auto":
            if dev.startswith("cuda") or dev == "mps":
                return torch.bfloat16
            return torch.float32
        return {
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
            "fp32": torch.float32,
            "float32": torch.float32,
        }[wanted]


def load_config_file(path: str | os.PathLike[str] | None) -> dict[str, Any]:
    """Optional JSON config. Unknown keys are ignored."""
    if path is None:
        return {}
    p = Path(path).expanduser()
    if not p.is_file():
        raise InputError(f"config file not found: {p}", details={"config": str(p)})
    try:
        data = read_json(p)
    except json.JSONDecodeError as exc:
        raise UsageError(f"invalid JSON in config file {p}: {exc}", details={"config": str(p)}) from exc
    except OSError as exc:
        raise InputError(f"cannot read config file {p}: {exc}", details={"config": str(p)}) from exc
    if not isinstance(data, dict):
        raise UsageError(f"config file {p} must contain a JSON object", details={"config": str(p)})
    known = {
        "data_dir",
        "model_dir",
        "cache_dir",
        "device",
        "dtype",
        "offline",
        "seed",
        "no_cache",
    }
    return {key: value for key, value in data.items() if key in known}


def build_config(args: Any) -> Config:
    """Merge config file < explicit CLI flags (flags win)."""
    file_cfg = load_config_file(getattr(args, "config", None))

    def pick(name: str, default: Any = None) -> Any:
        val = getattr(args, name, None)
        if val is not None:
            return val
        if name in file_cfg:
            return file_cfg[name]
        return default

    model_dir = pick("model_dir")
    if model_dir is None:
        model_dir = default_model_dir()

    try:
        seed = int(pick("seed", 1234))
    except (TypeError, ValueError) as exc:
        raise UsageError(f"--seed must be an integer, got {pick('seed')!r}") from exc

    return Config(
        data_dir=pick("data_dir") or default_data_dir(),
        model_dir=model_dir,
        cache_dir=pick("cache_dir"),
        device=pick("device", "auto") or "auto",
        dtype=pick("dtype", "auto") or "auto",
        offline=bool(pick("offline", False)),
        seed=seed,
        json=bool(getattr(args, "json", False)),
        quiet=bool(getattr(args, "quiet", False)),
        no_cache=bool(pick("no_cache", False)),
        log_level=getattr(args, "log_level", None) or "warning",
        config_file=getattr(args, "config", None),
        extra=file_cfg,
    )
