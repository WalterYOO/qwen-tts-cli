"""Audio cache. Cache keys and recipes are deliberately separate (PRD §8.3)."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .utils import atomic_write_json, read_json, sha256_json


def cache_key(
    *,
    text_normalized: str,
    voice_identity: dict[str, Any],
    model_id: str,
    model_revision: Optional[str],
    dtype: str,
    language: str,
    instruct: Optional[str],
    sampling: dict[str, Any],
    seed: int,
    model_fingerprint: Optional[str] = None,
    lexicon_sha256: Optional[str] = None,
) -> str:
    """sha256 over the canonical recipe inputs. qtts version is excluded by design."""
    return "sha256:" + sha256_json(
        {
            "text_normalized": text_normalized,
            "lexicon_sha256": lexicon_sha256,
            "voice": voice_identity,
            "model_id": model_id,
            "model_revision": model_revision,
            # Local models often have no revision; a content fingerprint keeps
            # replaced weights from silently reusing stale audio (codex P1-5).
            "model_fingerprint": model_fingerprint,
            "dtype": dtype,
            "language": language,
            "instruct": instruct,
            "sampling": sampling,
            "seed": seed,
        }
    )


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0

    @property
    def total(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return 0.0 if self.total == 0 else self.hits / self.total

    def to_dict(self) -> dict[str, Any]:
        return {"hits": self.hits, "misses": self.misses, "hit_rate": round(self.hit_rate, 4)}


@dataclass
class AudioCache:
    root: Path
    enabled: bool = True
    stats: CacheStats = field(default_factory=CacheStats)

    def __post_init__(self) -> None:
        self.root = Path(self.root)

    @staticmethod
    def _digest(key: str) -> str:
        return key.split(":", 1)[1] if key.startswith("sha256:") else key

    def _dir(self, key: str) -> Path:
        digest = self._digest(key)
        return self.root / digest[:2]

    def path_for(self, key: str, fmt: str) -> Path:
        return self._dir(key) / f"{self._digest(key)}.{fmt}"

    def exists(self, key: str, fmt: str = "wav") -> bool:
        """True only for an enabled cache with a non-empty entry."""
        if not self.enabled:
            return False
        path = self.path_for(key, fmt)
        try:
            return path.is_file() and path.stat().st_size > 0
        except OSError:
            return False

    def lookup(self, key: str, fmt: str = "wav") -> Optional[Path]:
        if self.exists(key, fmt):
            self.stats.hits += 1
            return self.path_for(key, fmt)
        self.stats.misses += 1
        return None

    def store(self, key: str, src: Path, fmt: str = "wav", meta: Optional[dict[str, Any]] = None) -> Path:
        dest = self.path_for(key, fmt)
        dest.parent.mkdir(parents=True, exist_ok=True)
        # Unique temp name: concurrent writers for the same key must not
        # overwrite each other's partial file (codex P1-8).
        fd, tmp = tempfile.mkstemp(prefix=f".{dest.name}.", suffix=".tmp", dir=str(dest.parent))
        try:
            with os.fdopen(fd, "wb") as out, open(src, "rb") as inp:
                while True:
                    chunk = inp.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
                out.flush()
                os.fsync(out.fileno())
            os.replace(tmp, dest)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        if meta is not None:
            atomic_write_json(dest.with_suffix(dest.suffix + ".json"), meta)
        return dest

    def meta_for(self, key: str, fmt: str = "wav") -> Optional[dict[str, Any]]:
        path = self.path_for(key, fmt).with_suffix(f".{fmt}.json")
        if not path.is_file():
            return None
        try:
            return read_json(path)
        except (OSError, ValueError):
            return None

    def probe(self, keys: list[str], fmt: str = "wav") -> CacheStats:
        """Dry-run hit-rate probe that does not mutate counters."""
        hits = sum(1 for key in keys if self.exists(key, fmt))
        return CacheStats(hits=hits, misses=len(keys) - hits)
