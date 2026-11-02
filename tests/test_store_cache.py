"""Voice store metadata round-trips and cache-key semantics (PRD §8.1/§8.3)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from qtts_cli.cache import AudioCache, cache_key
from qtts_cli.errors import InputError
from qtts_cli.voice_store import VoiceStore, utc_now, validate_name

ISO_Z = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


def _meta(**overrides):
    base = dict(
        name="alice",
        kind="clone",
        source={"type": "clone", "ref_audio_sha256": "abc", "ref_text": "你好", "mode": "icl"},
        model={"id": "Qwen/Qwen3-TTS-12Hz-1.7B-Base", "revision": "main"},
        dtype="bf16",
        language="Chinese",
    )
    base.update(overrides)
    return base


def test_voice_store_roundtrip(tmp_path: Path):
    store = VoiceStore(tmp_path)
    meta = store.create(**_meta())
    assert meta["schema_version"] == 1
    assert meta["qtts_version"]
    assert ISO_Z.match(meta["created_at"])
    assert store.exists("alice")
    loaded = store.get("alice")
    assert loaded == meta
    assert [m["name"] for m in store.list()] == ["alice"]


def test_voice_store_rejects_duplicate(tmp_path: Path):
    store = VoiceStore(tmp_path)
    store.create(**_meta())
    with pytest.raises(InputError, match="already exists"):
        store.create(**_meta())


def test_voice_store_remove(tmp_path: Path):
    store = VoiceStore(tmp_path)
    store.create(**_meta())
    store.remove("alice")
    assert not store.exists("alice")
    with pytest.raises(InputError):
        store.require("alice")


@pytest.mark.parametrize("bad", ["../evil", "", ".", "builtin", "a/b"])
def test_voice_name_validation(bad):
    with pytest.raises(InputError):
        validate_name(bad)


def test_voice_store_has_no_prompt_by_default(tmp_path: Path):
    store = VoiceStore(tmp_path)
    store.create(**_meta())
    assert store.prompt_sha256("alice") is None


def test_utc_now_format():
    assert ISO_Z.match(utc_now())


def test_cache_key_is_deterministic_and_seed_sensitive():
    kwargs = dict(
        text_normalized="你好",
        voice_identity={"type": "builtin", "speaker": "Vivian"},
        model_id="Qwen3-TTS-12Hz-1.7B-CustomVoice",
        model_revision=None,
        dtype="bf16",
        language="Chinese",
        instruct=None,
        sampling={"temperature": 0.9, "top_k": 50},
        seed=1234,
    )
    first = cache_key(**kwargs)
    assert first == cache_key(**kwargs)
    assert first.startswith("sha256:")
    assert cache_key(**{**kwargs, "seed": 99}) != first
    assert cache_key(**{**kwargs, "text_normalized": "再见"}) != first


def test_cache_key_ignores_qtts_version():
    """Version is recorded in the recipe, never in the cache key (PRD §8.3)."""
    kwargs = dict(
        text_normalized="hi",
        voice_identity={"type": "builtin", "speaker": "Ryan"},
        model_id="m",
        model_revision=None,
        dtype="bf16",
        language="English",
        instruct=None,
        sampling={},
        seed=1,
    )
    assert cache_key(**kwargs) == cache_key(**kwargs)


def test_cache_store_lookup_probe(tmp_path: Path):
    cache = AudioCache(tmp_path / "cache")
    src = tmp_path / "a.wav"
    src.write_bytes(b"RIFFfake")
    key = "sha256:" + "a" * 64
    assert cache.lookup(key, "wav") is None
    cached = cache.store(key, src, "wav", meta={"hello": "world"})
    assert cached.is_file()
    assert cache.lookup(key, "wav") == cached
    assert cache.meta_for(key, "wav") == {"hello": "world"}
    stats = cache.probe([key, "sha256:" + "b" * 64])
    assert stats.hits == 1 and stats.misses == 1


def test_cache_disabled_always_misses(tmp_path: Path):
    cache = AudioCache(tmp_path / "cache", enabled=False)
    assert cache.lookup("sha256:" + "a" * 64) is None
