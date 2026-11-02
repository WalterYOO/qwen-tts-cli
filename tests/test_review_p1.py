"""Regression tests for codex P1 review findings 1-10."""

from __future__ import annotations

import os
import shutil
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from qtts_cli import catalog, cli, tts_engine
from qtts_cli.cache import AudioCache, cache_key
from qtts_cli.catalog import ModelInfo, model_fingerprint
from qtts_cli.commands import voice as voice_cmd
from qtts_cli.config import Config
from qtts_cli.errors import InputError
from qtts_cli.plan import attach_cache_stats, build_plan, rekey_plan
from qtts_cli.segmenter import split_long
from qtts_cli.textnorm import normalize_text
from qtts_cli.voice_store import VoiceStore


def _meta(**overrides):
    base = dict(
        name="alice",
        kind="clone",
        source={"type": "clone", "ref_audio_sha256": "abc", "ref_text": "你好", "mode": "icl"},
        model={"id": "m", "revision": None},
        dtype="bf16",
        language="Chinese",
    )
    base.update(overrides)
    return base


# --------------------------------------------------------------------------- #
# P1-5: cache fingerprint + runtime dtype
# --------------------------------------------------------------------------- #
def test_cache_key_is_fingerprint_sensitive():
    kwargs = dict(
        text_normalized="你好",
        voice_identity={"type": "builtin", "speaker": "Vivian"},
        model_id="m",
        model_revision=None,
        dtype="bf16",
        language="Chinese",
        instruct=None,
        sampling={},
        seed=1,
    )
    first = cache_key(**kwargs)
    assert first != cache_key(**{**kwargs, "model_fingerprint": "abc"})
    assert cache_key(**{**kwargs, "model_fingerprint": "abc"}) != cache_key(
        **{**kwargs, "model_fingerprint": "def"}
    )


def test_model_fingerprint_changes_when_weights_change(tmp_path: Path):
    model = tmp_path / "m"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"a" * 8)
    first = model_fingerprint(model)
    (model / "model.safetensors").write_bytes(b"a" * 64)
    assert model_fingerprint(model) != first


def test_rekey_plan_uses_resolved_dtype(model_dir: Path, tmp_path: Path):
    # "mps" makes the planned dtype deterministically bf16 without importing torch.
    cfg = Config(data_dir=tmp_path / "data", model_dir=model_dir, device="mps", dtype="auto", seed=3)
    plan = build_plan(
        cfg, text="你好。", voice_ref="builtin:Vivian",
        models=catalog.scan(model_dir), store=VoiceStore(cfg.data_dir),
    )
    before = [seg.cache_key for seg in plan.segments]
    rekey_plan(plan, dtype="fp32", device="cpu")
    after = [seg.cache_key for seg in plan.segments]
    assert plan.dtype_name == "fp32"
    assert before != after
    assert all(key.startswith("sha256:") for key in after)


# --------------------------------------------------------------------------- #
# P1-6: probe honours enabled and non-empty files
# --------------------------------------------------------------------------- #
def test_probe_ignores_disabled_and_empty_cache(tmp_path: Path):
    cache = AudioCache(tmp_path / "cache")
    src = tmp_path / "a.wav"
    src.write_bytes(b"RIFF")
    key = "sha256:" + "a" * 64
    empty = cache.path_for(key, "wav")
    empty.parent.mkdir(parents=True, exist_ok=True)
    empty.write_bytes(b"")
    assert cache.probe([key]).hits == 0
    cache.store(key, src)
    assert cache.probe([key]).hits == 1
    disabled = AudioCache(tmp_path / "cache", enabled=False)
    assert disabled.probe([key]).hits == 0
    assert disabled.lookup(key) is None


# --------------------------------------------------------------------------- #
# P1-8: concurrent atomic cache writes leave no temp files
# --------------------------------------------------------------------------- #
def test_concurrent_cache_store_is_atomic(tmp_path: Path):
    cache = AudioCache(tmp_path / "cache")
    key = "sha256:" + "b" * 64
    payloads = [bytes([i]) * (1024 * (i + 1)) for i in range(6)]
    sources = []
    for idx, data in enumerate(payloads):
        path = tmp_path / f"src{idx}.wav"
        path.write_bytes(data)
        sources.append(path)

    def worker(path: Path) -> None:
        cache.store(key, path)

    threads = [threading.Thread(target=worker, args=(path,)) for path in sources]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    final = cache.path_for(key, "wav")
    assert final.is_file()
    assert final.read_bytes() in payloads
    assert list(cache.root.rglob("*.tmp")) == []


def test_voice_staging_commit_is_atomic(tmp_path: Path):
    store = VoiceStore(tmp_path)
    staging = store.staging_dir("alice")
    store.create(**_meta(), staging=staging)
    store.commit_staging(staging, "alice")
    assert store.exists("alice")
    second = store.staging_dir("alice")
    with pytest.raises(InputError, match="already exists"):
        store.create(**_meta(), staging=second)
    with pytest.raises(InputError, match="already exists"):
        store.commit_staging(second, "alice")
    shutil.rmtree(second, ignore_errors=True)


def test_no_cache_never_reports_hits(model_dir: Path, tmp_path: Path):
    cfg = Config(data_dir=tmp_path / "data", model_dir=model_dir, device="cpu", dtype="fp32", seed=1)
    plan = build_plan(
        cfg, text="你好。世界。", voice_ref="builtin:Vivian",
        models=catalog.scan(model_dir), store=VoiceStore(cfg.data_dir),
    )
    cache = AudioCache(tmp_path / "cache")
    src = tmp_path / "seg.wav"
    src.write_bytes(b"RIFF")
    for seg in plan.segments:
        cache.store(seg.cache_key, src)
    attach_cache_stats(plan, cache)
    assert plan.cache_stats["hits"] == len(plan.segments)
    attach_cache_stats(plan, AudioCache(tmp_path / "cache", enabled=False))
    assert plan.cache_stats["hits"] == 0


# --------------------------------------------------------------------------- #
# P1-2: failed materialization leaves no half-profile
# --------------------------------------------------------------------------- #
def test_materialize_failure_rolls_back_profile(monkeypatch, model_dir: Path, tmp_path: Path):
    cfg = Config(data_dir=tmp_path / "data", model_dir=model_dir, device="cpu", dtype="fp32", seed=1)
    args = SimpleNamespace(
        name="designed",
        description="温柔女声",
        sample_text="你好。",
        language="Chinese",
        model=None,
        materialize_clone=True,
        tags=None,
        license=None,
        consent_note=None,
    )

    def boom(*args, **kwargs):
        raise RuntimeError("materialize failed")

    monkeypatch.setattr(voice_cmd, "_materialize_design", boom)
    with pytest.raises(RuntimeError, match="materialize failed"):
        voice_cmd._add_design(cfg, args)
    store = VoiceStore(cfg.data_dir)
    assert not store.exists("designed")
    assert list(store.root.glob(".designed.*")) == []


# --------------------------------------------------------------------------- #
# P1-1: --offline reaches from_pretrained
# --------------------------------------------------------------------------- #
class _FakeTorch:
    bfloat16 = "bf16"
    float16 = "fp16"
    float32 = "fp32"


def _fake_cfg(offline: bool):
    return SimpleNamespace(
        offline=offline,
        device="cpu",
        dtype="fp32",
        resolve_device=lambda: "cpu",
        torch_dtype=lambda device: "fp32",
    )


def test_offline_passes_local_files_only(monkeypatch, tmp_path: Path):
    captured: dict = {}

    class FakeModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            captured.update(kwargs)
            return object()

    monkeypatch.setattr(tts_engine, "_runtime", lambda: (_FakeTorch, FakeModel, object))
    tts_engine._MODEL_CACHE.clear()
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    try:
        engine = tts_engine.TtsEngine(_fake_cfg(True), ModelInfo(path=tmp_path / "m", name="m", model_type="qwen3_tts"))
        engine.load()
        assert captured["local_files_only"] is True
        assert os.environ.get("HF_HUB_OFFLINE") == "1"
    finally:
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)
        tts_engine._MODEL_CACHE.clear()


def test_online_does_not_force_local_files_only(monkeypatch, tmp_path: Path):
    captured: dict = {}

    class FakeModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            captured.update(kwargs)
            return object()

    monkeypatch.setattr(tts_engine, "_runtime", lambda: (_FakeTorch, FakeModel, object))
    tts_engine._MODEL_CACHE.clear()
    try:
        engine = tts_engine.TtsEngine(_fake_cfg(False), ModelInfo(path=tmp_path / "m", name="m", model_type="qwen3_tts"))
        engine.load()
        assert "local_files_only" not in captured
    finally:
        tts_engine._MODEL_CACHE.clear()


# --------------------------------------------------------------------------- #
# P1-10: CJK-adjacent dates and quote-aware long splits
# --------------------------------------------------------------------------- #
def test_date_adjacent_to_cjk_is_normalized():
    out = normalize_text("日期2026-11-02发布", "Chinese")
    assert "二零二六年十一月二日" in out
    assert "2026-11-02" not in out


def test_split_long_does_not_cut_inside_quote():
    text = "a" * 30 + "“" + "b" * 30 + "”" + "c" * 30
    pieces = split_long(text, 40)
    assert "".join(pieces) == text
    assert pieces[0] == text[:30]


# --------------------------------------------------------------------------- #
# P0-3 / P1-7: stdout routing and cancellation exit semantics
# --------------------------------------------------------------------------- #
def test_human_summary_goes_to_stderr_when_stdout_used(monkeypatch, capsys, model_dir: Path, tmp_path: Path):
    from qtts_cli.commands import say as say_cmd

    monkeypatch.setattr(say_cmd, "run", lambda cfg, args: {"ok": True, "stdout_used": True})
    monkeypatch.setattr(say_cmd, "format_result", lambda payload: "SUMMARY")
    rc = cli.main([
        "say", "你好", "--voice", "builtin:Vivian", "--out", "-",
        "--model-dir", str(model_dir), "--data-dir", str(tmp_path / "data"),
    ])
    out, err = capsys.readouterr()
    assert rc == 0
    assert out == ""
    assert "SUMMARY" in err


def test_cancelled_voice_removal_is_success_payload():
    assert voice_cmd.format_result({"ok": True, "cancelled": True, "voice": "alice"}) == "cancelled"
