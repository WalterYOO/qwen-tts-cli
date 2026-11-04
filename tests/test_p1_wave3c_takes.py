"""FR-P1-04: multi-take generation, selection, asset library and CLI."""

from __future__ import annotations

import copy
import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from qtts_cli.audio_io import write_audio
from qtts_cli.catalog import scan
from qtts_cli.cli import main
from qtts_cli.config import Config
from qtts_cli.errors import InputError, ModelError, UsageError
from qtts_cli.plan import build_plan, reseed_plan
from qtts_cli.takes import TakesStore, canonical_request, request_id
from qtts_cli.takes_run import _publish_with_damage
from qtts_cli.voice_store import VoiceStore


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
class FakeEngine:
    instances = 0
    loads = 0
    seeds: list[int] = []
    embedding_calls = 0
    amplitudes: dict[int, float] = {}

    def __init__(self, cfg, model):
        FakeEngine.instances += 1
        self.cfg = cfg
        self.model_info = model
        self.device = "cpu"
        self.dtype_name = "fp32"
        self.load_ms = 0.0
        self.model = None

    def load(self):
        FakeEngine.loads += 1
        self.model = object()
        return self

    def generate(self, **kwargs):
        seed = int(kwargs["seed"])
        FakeEngine.seeds.append(seed)
        amplitude = FakeEngine.amplitudes.get(seed, 0.1)
        t = np.arange(24000, dtype=np.float64) / 24000.0
        audio = (amplitude * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
        return audio, 24000, {"generate_ms": 1.0}

    def build_clone_prompt_items(self, *args, **kwargs):
        return []

    def require_speaker_embedding(self):
        return None

    def extract_speaker_embedding(self, samples, sr):
        FakeEngine.embedding_calls += 1
        mean = float(np.mean(np.abs(np.asarray(samples, dtype=np.float32))))
        return np.array([1.0, mean, 0.0], dtype=np.float32)


class FakeAligner:
    calls = 0

    def __init__(self, cfg, model):
        self.cfg = cfg
        self.model_info = model
        self.model = None

    def align(self, audio, sample_rate, text, language):
        FakeAligner.calls += 1
        from qtts_cli.subtitles import AlignItem

        items = [
            AlignItem(ch, i * 0.1, (i + 1) * 0.1)
            for i, ch in enumerate(text)
            if not ch.isspace()
        ]
        return items, {"align_ms": 1.0}


class FakeAsrEngine:
    texts: list[str] = []
    calls: list[object] = []
    released = 0
    error: BaseException | None = None

    def __init__(self, cfg, model):
        self.cfg = cfg
        self.model_info = model

    def transcribe_samples(self, samples, sr, *, language=None):
        FakeAsrEngine.calls.append(language)
        if FakeAsrEngine.error is not None:
            raise FakeAsrEngine.error
        if not FakeAsrEngine.texts:
            raise AssertionError("FakeAsrEngine ran out of transcripts")
        return {"text": FakeAsrEngine.texts.pop(0), "language": language}

    def release(self):
        FakeAsrEngine.released += 1


@pytest.fixture()
def fake_engines(monkeypatch):
    FakeEngine.instances = 0
    FakeEngine.loads = 0
    FakeEngine.seeds = []
    FakeEngine.embedding_calls = 0
    FakeEngine.amplitudes = {}
    FakeAsrEngine.texts = []
    FakeAsrEngine.calls = []
    FakeAsrEngine.released = 0
    FakeAsrEngine.error = None
    from qtts_cli import takes_run

    monkeypatch.setattr(takes_run, "TtsEngine", FakeEngine)
    monkeypatch.setattr(takes_run, "AsrEngine", FakeAsrEngine)
    monkeypatch.setattr(takes_run, "preflight_asr_runtime", lambda: None)
    return FakeEngine, FakeAsrEngine


@pytest.fixture()
def asr_model_dir(model_dir: Path) -> Path:
    model = model_dir / "Qwen3-ASR-0.6B"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_asr",
                "support_languages": ["chinese", "english"],
            }
        ),
        encoding="utf-8",
    )
    (model / "model.safetensors").write_bytes(b"fake")
    return model_dir


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _run(tmp_path: Path, model_dir: Path, *args: str) -> int:
    return main(
        [
            *args,
            "--model-dir",
            str(model_dir),
            "--data-dir",
            str(tmp_path / "data"),
            "--cache-dir",
            str(tmp_path / "cache"),
        ]
    )


def _say(
    tmp_path: Path,
    model_dir: Path,
    *extra: str,
    text: str = "你好。",
    voice: str = "builtin:Vivian",
    out: str | None = None,
) -> int:
    args = [
        "say",
        text,
        "--voice",
        voice,
        "--out",
        out or str(tmp_path / "out.wav"),
        *extra,
    ]
    return _run(tmp_path, model_dir, *args)


def _json(capsys) -> dict:
    return json.loads(capsys.readouterr().out)


def _job(tmp_path: Path, job_id: str) -> dict:
    return json.loads(
        (tmp_path / "data" / "takes" / job_id / "job.json").read_text(
            encoding="utf-8"
        )
    )


def _write_wav(path: Path, *, seconds: float = 1.0, amplitude: float = 0.2) -> Path:
    count = int(round(24000 * seconds))
    t = np.arange(count, dtype=np.float64) / 24000.0
    audio = (amplitude * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
    write_audio(path, audio, 24000, "wav")
    return path


def _write_clone(tmp_path: Path, name: str = "alice") -> None:
    directory = tmp_path / "data" / "voices" / name
    directory.mkdir(parents=True)
    (directory / "voice.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "name": name,
                "kind": "clone",
                "source": {"type": "clone", "mode": "icl", "ref_text": "你好"},
                "model": {"id": "Qwen3-TTS-12Hz-1.7B-Base", "revision": None},
                "language": "Chinese",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    _write_wav(directory / "ref.wav")


def _plan(tmp_path: Path, model_dir: Path, text: str = "你好。"):
    cfg = Config(
        data_dir=tmp_path / "data",
        model_dir=model_dir,
        device="cpu",
        dtype="fp32",
        seed=1234,
    )
    return build_plan(
        cfg,
        text=text,
        voice_ref="builtin:Vivian",
        language="Chinese",
        models=scan(model_dir),
        store=VoiceStore(cfg.data_dir),
        fmt="wav",
    )


# --------------------------------------------------------------------------- #
# Seeds / plan copying
# --------------------------------------------------------------------------- #
def test_reseed_plan_is_pure_and_advances_every_seed(tmp_path, model_dir):
    plan = _plan(tmp_path, model_dir, "你好。世界。")
    assert len(plan.segments) == 2
    base_keys = [seg.cache_key for seg in plan.segments]
    base_seeds = [seg.seed for seg in plan.segments]

    take1 = reseed_plan(plan, 1)
    assert take1.seed == plan.seed
    assert [seg.seed for seg in take1.segments] == base_seeds
    assert [seg.cache_key for seg in take1.segments] == base_keys

    take2 = reseed_plan(plan, 2)
    assert take2.seed == plan.seed + 1
    assert [seg.seed for seg in take2.segments] == [seed + 1 for seed in base_seeds]
    assert [seg.cache_key for seg in take2.segments] != base_keys
    assert [seg.cache_key for seg in plan.segments] == base_keys
    assert [seg.seed for seg in plan.segments] == base_seeds
    assert take1.segments is not plan.segments


def test_reseed_overflow_exits_2(tmp_path, model_dir):
    plan = _plan(tmp_path, model_dir)
    plan.seed = 2**31 - 1
    with pytest.raises(UsageError) as info:
        reseed_plan(plan, 2)
    assert info.value.code == 2


# --------------------------------------------------------------------------- #
# Canonical request / job id
# --------------------------------------------------------------------------- #
def test_job_id_uses_only_generation_inputs(tmp_path, model_dir):
    plan = _plan(tmp_path, model_dir)
    cfg = Config(data_dir=tmp_path / "data", model_dir=model_dir, seed=1234)
    request = canonical_request(
        plan, cfg, takes=2, criteria=["lufs"], scorer_models={}, fmt="wav"
    )
    job_id = request_id(request)
    assert job_id.startswith("tk_") and len(job_id) == 19

    ignored = copy.deepcopy(plan)
    ignored.output = "/tmp/other.wav"
    ignored.subtitle_formats = ["srt"]
    ignored.script_source = "/absolute/script.yaml"
    ignored.cache_stats = {"hits": 9}
    ignored.cache_hit_rate = 0.9
    assert (
        request_id(
            canonical_request(
                ignored, cfg, takes=2, criteria=["lufs"], scorer_models={}, fmt="wav"
            )
        )
        == job_id
    )

    for mutate in (
        lambda p: setattr(p, "seed", p.seed + 1),
        lambda p: p.sampling.__setitem__("temperature", 0.5),
        lambda p: setattr(p.segments[0], "language", "English"),
        lambda p: setattr(p, "lexicon_sha256", "abc"),
    ):
        changed = copy.deepcopy(plan)
        mutate(changed)
        assert (
            request_id(
                canonical_request(
                    changed,
                    cfg,
                    takes=2,
                    criteria=["lufs"],
                    scorer_models={},
                    fmt="wav",
                )
            )
            != job_id
        )


# --------------------------------------------------------------------------- #
# CLI validation / dry-run
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "extra",
    [
        ("--takes", "2"),
        ("--select", "lufs"),
        ("--takes", "1", "--select", "lufs"),
        ("--takes", "17", "--select", "lufs"),
        ("--takes", "2", "--select", "lufs,lufs"),
        ("--takes", "2", "--select", "bogus"),
        ("--takes", "2", "--select", "lufs", "--reference-voice", "alice"),
    ],
)
def test_takes_cli_validation_exits_2(capsys, tmp_path, model_dir, extra):
    rc = _say(tmp_path, model_dir, *extra)
    captured = capsys.readouterr()
    assert rc == 2
    assert captured.out == ""


def test_dry_run_job_id_matches_real_run(fake_engines, capsys, tmp_path, model_dir):
    assert (
        _say(
            tmp_path,
            model_dir,
            "--takes",
            "2",
            "--select",
            "lufs",
            "--dry-run",
            "--json",
        )
        == 0
    )
    dry = _json(capsys)
    assert (
        _say(
            tmp_path,
            model_dir,
            "--takes",
            "2",
            "--select",
            "lufs",
            "--no-cache",
            "--json",
        )
        == 0
    )
    real = _json(capsys)
    assert dry["takes"]["job_id"] == real["takes"]["job_id"]


def test_takes_dry_run_has_no_side_effects(
    fake_engines, monkeypatch, capsys, tmp_path, model_dir
):
    def boom(self):
        raise AssertionError("dry-run must not load TTS")

    monkeypatch.setattr(FakeEngine, "load", boom)
    rc = _say(
        tmp_path,
        model_dir,
        "--takes",
        "2",
        "--select",
        "lufs",
        "--dry-run",
        "--json",
    )
    assert rc == 0
    payload = _json(capsys)
    takes = payload["takes"]
    assert takes["takes"] == 2
    assert [item["seed"] for item in takes["seeds"]] == [1234, 1235]
    assert takes["selection"]["dependencies"]["lufs"]["target"] == -16.0
    assert not (tmp_path / "data" / "takes").exists()
    assert payload["models_loaded"] is False


def test_takes_jsonl_rejected_before_model_load(
    fake_engines, monkeypatch, capsys, tmp_path, model_dir
):
    def boom(self):
        raise AssertionError("JSONL conflict must be rejected before models load")

    monkeypatch.setattr(FakeEngine, "load", boom)
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    rc = _run(
        tmp_path,
        model_dir,
        "say",
        "--stdin",
        "--jsonl",
        "--voice",
        "builtin:Vivian",
        "--takes",
        "2",
        "--select",
        "lufs",
    )
    captured = capsys.readouterr()
    assert rc == 2
    assert captured.out == ""


def test_batch_manifest_rejects_takes(tmp_path, model_dir, capsys):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"id": "a", "text": "你好", "takes": 2, "select": "lufs"}) + "\n",
        encoding="utf-8",
    )
    rc = _run(
        tmp_path,
        model_dir,
        "batch",
        "plan",
        str(manifest),
        "--out-dir",
        str(tmp_path / "batch"),
    )
    assert rc == 2
    assert capsys.readouterr().out == ""


# --------------------------------------------------------------------------- #
# Generation + lufs selection + asset library
# --------------------------------------------------------------------------- #
def test_takes_lufs_publishes_job_and_reuses_it(
    fake_engines, capsys, tmp_path, model_dir
):
    FakeEngine.amplitudes = {1234: 0.05, 1235: 0.3}
    rc = _say(tmp_path, model_dir, "--takes", "2", "--select", "lufs", "--json")
    assert rc == 0
    payload = _json(capsys)
    takes = payload["takes"]
    job_id = takes["job_id"]
    assert takes["state"] == "complete"
    assert takes["selected"] in (1, 2)
    assert len(takes["items"]) == 2
    for item in takes["items"]:
        assert item["metrics"]["lufs"] is not None
        assert item["metrics"]["true_peak_dbtp"] is not None
        assert item["metrics"]["cer"] is None
        assert item["unavailable"]["cer"] == "not_selected"

    job_dir = tmp_path / "data" / "takes" / job_id
    job = _job(tmp_path, job_id)
    assert job["state"] == "complete"
    assert job["selected"] == takes["selected"]
    assert {take["id"] for take in job["takes"]} == {1, 2}
    for take in job["takes"]:
        assert (job_dir / take["file"]).is_file()
        assert (job_dir / take["recipe"]).is_file()
        recipe = json.loads((job_dir / take["recipe"]).read_text(encoding="utf-8"))
        assert recipe["job_id"] == job_id
        assert recipe["take_id"] == take["id"]
        assert recipe["base_seed"] == 1234
        assert recipe["audio"]["sha256"] == take["audio_sha256"]
        assert recipe["model"]["fingerprint"]
        assert recipe["lexicon_sha256"] == job["request"]["lexicon_sha256"]
        assert recipe["seed_semantics"]["base_seed"] == 1234
        assert recipe["seed_semantics"]["take_seed"] == take["seed"]
        assert recipe["spoken_to_display"]
    assert set(job["request"]["runtime"]) >= {
        "device",
        "dtype",
        "device_request",
        "platform",
        "machine",
    }
    selected_take = next(t for t in job["takes"] if t["id"] == job["selected"])
    assert (tmp_path / "out.wav").read_bytes() == (
        job_dir / selected_take["file"]
    ).read_bytes()

    top_recipe = json.loads((tmp_path / "out.recipe.json").read_text(encoding="utf-8"))
    assert top_recipe["job_id"] == job_id
    assert top_recipe["takes"]["selected"] == job["selected"]
    assert sorted(top_recipe["takes"]["ids"]) == [1, 2]

    generated = len(FakeEngine.seeds)
    rc = _say(tmp_path, model_dir, "--takes", "2", "--select", "lufs", "--json")
    assert rc == 0
    payload = _json(capsys)
    assert payload["takes"]["reused"] is True
    assert len(FakeEngine.seeds) == generated


def test_reseed_seed_vectors_are_used_for_each_take(
    fake_engines, capsys, tmp_path, model_dir
):
    FakeEngine.amplitudes = {1234: 0.2, 1235: 0.2}
    assert _say(tmp_path, model_dir, "--takes", "2", "--select", "lufs", "--json") == 0
    _json(capsys)
    assert FakeEngine.seeds == [1234, 1235]


def test_takes_out_dash_is_binary_only(
    fake_engines, capsysbinary, tmp_path, model_dir
):
    FakeEngine.amplitudes = {1234: 0.1, 1235: 0.3}
    rc = _say(
        tmp_path,
        model_dir,
        "--takes",
        "2",
        "--select",
        "lufs",
        "--out",
        "-",
    )
    captured = capsysbinary.readouterr()
    assert rc == 0
    assert captured.out.startswith(b"RIFF")
    job_id = next((tmp_path / "data" / "takes").glob("tk_*")).name
    job = _job(tmp_path, job_id)
    selected = next(t for t in job["takes"] if t["id"] == job["selected"])
    assert captured.out == (
        tmp_path / "data" / "takes" / job_id / selected["file"]
    ).read_bytes()


# --------------------------------------------------------------------------- #
# CER scoring
# --------------------------------------------------------------------------- #
def test_cer_is_micro_averaged_and_uses_spoken_text(
    fake_engines, capsys, tmp_path, asr_model_dir
):
    FakeAsrEngine.texts = ["你好", "世界和平", "你好", "世界"]
    rc = _say(
        tmp_path,
        asr_model_dir,
        "--takes",
        "2",
        "--select",
        "cer",
        "--json",
        text="你好。世界和平。",
    )
    assert rc == 0
    payload = _json(capsys)
    items = {item["id"]: item for item in payload["takes"]["items"]}
    assert items[1]["metrics"]["cer"] == 0.0
    # Segment 2 lost two of six reference units; micro-average is 2/6.
    assert items[2]["metrics"]["cer"] == pytest.approx(2 / 6, abs=1e-6)
    assert payload["takes"]["selected"] == 1
    assert FakeAsrEngine.calls == ["Chinese"] * 4
    assert FakeAsrEngine.released == 1


def test_cer_scoring_failure_keeps_scoring_failed_job_and_retries(
    fake_engines, capsys, tmp_path, asr_model_dir
):
    FakeAsrEngine.error = ModelError("asr exploded")
    rc = _say(tmp_path, asr_model_dir, "--takes", "2", "--select", "cer")
    assert rc == 3
    job_id = next((tmp_path / "data" / "takes").glob("tk_*")).name
    job = _job(tmp_path, job_id)
    assert job["state"] == "scoring_failed"
    assert job["selected"] is None
    assert len(job["takes"]) == 2
    generated = len(FakeEngine.seeds)

    FakeAsrEngine.error = None
    FakeAsrEngine.texts = ["你好", "你好"]
    rc = _say(tmp_path, asr_model_dir, "--takes", "2", "--select", "cer", "--json")
    assert rc == 0
    payload = _json(capsys)
    assert payload["takes"]["state"] == "complete"
    job = _job(tmp_path, job_id)
    assert job["state"] == "complete"
    assert job["selected"] is not None
    assert len(FakeEngine.seeds) == generated  # retry did not regenerate audio


def test_cer_structural_failure_exits_4(fake_engines, tmp_path, asr_model_dir):
    FakeAsrEngine.error = InputError("bad audio")
    rc = _say(tmp_path, asr_model_dir, "--takes", "2", "--select", "cer")
    assert rc == 4
    job_id = next((tmp_path / "data" / "takes").glob("tk_*")).name
    assert _job(tmp_path, job_id)["state"] == "scoring_failed"


# --------------------------------------------------------------------------- #
# Speaker similarity
# --------------------------------------------------------------------------- #
def test_speaker_sim_uses_explicit_reference_and_segment_windows(
    fake_engines, capsys, tmp_path, model_dir
):
    _write_clone(tmp_path)
    FakeEngine.amplitudes = {1234: 0.5, 1235: 0.2}
    rc = _say(
        tmp_path,
        model_dir,
        "--takes",
        "2",
        "--select",
        "speaker-sim",
        "--reference-voice",
        "alice",
        "--json",
        text="你好。世界。",
    )
    assert rc == 0
    payload = _json(capsys)
    items = {item["id"]: item for item in payload["takes"]["items"]}
    assert (
        items[2]["metrics"]["speaker_similarity"]
        > items[1]["metrics"]["speaker_similarity"]
    )
    assert items[2]["metrics"]["speaker_similarity_min"] is not None
    assert payload["takes"]["selected"] == 2
    # One reference embedding plus two segments for each of two takes.
    assert FakeEngine.embedding_calls == 5
    # One TTS slot, one Base scorer, and one no-load capability precheck.
    assert FakeEngine.instances == 3
    recipe = json.loads(
        (
            tmp_path
            / "data"
            / "takes"
            / payload["takes"]["job_id"]
            / "take-02.recipe.json"
        ).read_text(encoding="utf-8")
    )
    assert len(recipe["selection"]["speaker_segments"]) == 2


def test_speaker_sim_builtin_without_reference_exits_2(
    fake_engines, capsys, tmp_path, model_dir
):
    rc = _say(tmp_path, model_dir, "--takes", "2", "--select", "speaker-sim")
    assert rc == 2
    assert "clone voice or --reference-voice" in capsys.readouterr().err


def test_speaker_sim_multivoice_without_reference_exits_2(
    fake_engines, capsys, tmp_path, model_dir
):
    script = tmp_path / "script.yaml"
    script.write_text(
        """
schema_version: 1
defaults:
  language: Chinese
segments:
  - id: a
    text: "你好。"
    voice: "builtin:Serena"
  - id: b
    text: "世界。"
    voice: "builtin:Vivian"
""",
        encoding="utf-8",
    )
    rc = _run(
        tmp_path,
        model_dir,
        "say",
        "--script",
        str(script),
        "--takes",
        "2",
        "--select",
        "speaker-sim",
    )
    assert rc == 2


def test_script_segment_seeds_are_preserved(fake_engines, capsys, tmp_path, model_dir):
    script = tmp_path / "seeds.yaml"
    script.write_text(
        """
schema_version: 1
defaults:
  language: Chinese
  seed: 100
segments:
  - id: a
    text: "你好。"
    voice: "builtin:Vivian"
    seed: 200
  - id: b
    text: "世界。"
    voice: "builtin:Vivian"
    seed: 300
""",
        encoding="utf-8",
    )
    rc = _run(
        tmp_path,
        model_dir,
        "say",
        "--script",
        str(script),
        "--takes",
        "2",
        "--select",
        "lufs",
        "--no-cache",
        "--json",
    )
    assert rc == 0
    payload = _json(capsys)
    assert FakeEngine.seeds == [200, 300, 201, 301]
    job = _job(tmp_path, payload["takes"]["job_id"])
    assert job["request"]["seeds"] == [100, 101]
    assert job["takes"][1]["segment_seeds"] == {"a": 201, "b": 301}


def test_subtitles_and_qc_use_only_the_selected_take(
    fake_engines, monkeypatch, capsys, tmp_path, model_dir
):
    from qtts_cli import takes_run

    FakeAligner.calls = 0
    aligner_clears: list[int] = []
    monkeypatch.setattr(takes_run, "AlignerEngine", FakeAligner)
    monkeypatch.setattr(
        takes_run, "clear_aligner_cache", lambda: aligner_clears.append(1)
    )
    FakeEngine.amplitudes = {1234: 0.3, 1235: 0.05}
    rc = _say(
        tmp_path,
        model_dir,
        "--takes",
        "2",
        "--select",
        "lufs",
        "--subtitle",
        "srt",
        "--json",
    )
    assert rc == 0
    payload = _json(capsys)
    assert payload["takes"]["selected"] in (1, 2)
    assert FakeAligner.calls == 1  # one selected take, one segment
    assert aligner_clears == [1]
    assert (tmp_path / "out.srt").is_file()


def test_qc_failure_keeps_completed_job(fake_engines, capsys, tmp_path, model_dir):
    FakeEngine.amplitudes = {1234: 0.1, 1235: 0.1}
    rc = _say(
        tmp_path,
        model_dir,
        "--takes",
        "2",
        "--select",
        "lufs",
        "--qc",
        "--target-lufs",
        "0",
        "--lufs-tolerance",
        "1",
        "--json",
    )
    assert rc == 5
    payload = _json(capsys)
    job_id = payload["takes"]["job_id"]
    assert _job(tmp_path, job_id)["state"] == "complete"
    assert (tmp_path / "out.wav").is_file()


def test_generation_failure_leaves_no_job(
    fake_engines, monkeypatch, tmp_path, model_dir
):
    original = FakeEngine.generate

    def flaky(self, **kwargs):
        if int(kwargs["seed"]) == 1235:
            raise ModelError("second take failed")
        return original(self, **kwargs)

    monkeypatch.setattr(FakeEngine, "generate", flaky)
    rc = _say(tmp_path, model_dir, "--takes", "2", "--select", "lufs", "--no-cache")
    assert rc == 3
    takes_root = tmp_path / "data" / "takes"
    assert not list(takes_root.glob("tk_*"))
    assert not list(takes_root.glob(".staging-*"))


def test_speaker_sim_dry_run_loads_no_torch(
    fake_engines, monkeypatch, capsys, tmp_path, model_dir
):
    _write_clone(tmp_path)

    def boom(self):
        raise AssertionError("dry-run must not load a Base model")

    monkeypatch.setattr(FakeEngine, "load", boom)
    rc = _say(
        tmp_path,
        model_dir,
        "--takes",
        "2",
        "--select",
        "speaker-sim",
        "--reference-voice",
        "alice",
        "--dry-run",
        "--json",
    )
    assert rc == 0
    payload = _json(capsys)
    dependency = payload["takes"]["selection"]["dependencies"]["speaker-sim"]
    assert dependency["reference"] == "alice"
    assert dependency["identity"]["name"] == "alice"
    assert dependency["identity"]["embedding_source"] == "ref_wav"
    assert not (tmp_path / "data" / "takes").exists()


# --------------------------------------------------------------------------- #
# Damage / canonical mismatch / publication
# --------------------------------------------------------------------------- #
def test_damaged_job_is_quarantined_and_rebuilt(
    fake_engines, capsys, tmp_path, model_dir
):
    FakeEngine.amplitudes = {1234: 0.1, 1235: 0.1}
    assert (
        _say(
            tmp_path,
            model_dir,
            "--takes",
            "2",
            "--select",
            "lufs",
            "--no-cache",
            "--json",
        )
        == 0
    )
    job_id = _json(capsys)["takes"]["job_id"]
    job_dir = tmp_path / "data" / "takes" / job_id
    (job_dir / "job.json").write_text("{not json", encoding="utf-8")
    generated = len(FakeEngine.seeds)

    rc = _say(
        tmp_path,
        model_dir,
        "--takes",
        "2",
        "--select",
        "lufs",
        "--no-cache",
        "--json",
    )
    assert rc == 0
    assert len(FakeEngine.seeds) == generated + 2
    assert _job(tmp_path, job_id)["state"] == "complete"
    assert list((tmp_path / "data" / "takes").glob(".damaged-*"))


def test_canonical_mismatch_exits_4_without_overwrite(
    fake_engines, capsys, tmp_path, model_dir
):
    FakeEngine.amplitudes = {1234: 0.1, 1235: 0.1}
    assert _say(tmp_path, model_dir, "--takes", "2", "--select", "lufs", "--json") == 0
    job_id = _json(capsys)["takes"]["job_id"]
    job_dir = tmp_path / "data" / "takes" / job_id
    job = _job(tmp_path, job_id)
    job["request"]["sampling"]["temperature"] = 0.123
    (job_dir / "job.json").write_text(json.dumps(job), encoding="utf-8")
    generated = len(FakeEngine.seeds)

    rc = _say(tmp_path, model_dir, "--takes", "2", "--select", "lufs")
    assert rc == 4
    assert len(FakeEngine.seeds) == generated
    assert _job(tmp_path, job_id)["request"]["sampling"]["temperature"] == 0.123


def test_publish_failure_restores_damaged_job(tmp_path, monkeypatch):
    store = TakesStore(tmp_path / "data")
    store.ensure()
    job_id = "tk_publishfailure"
    target = store.root / job_id
    target.mkdir()
    (target / "job.json").write_text("old", encoding="utf-8")
    damaged = store.quarantine(job_id)
    staging = store.new_staging(job_id)
    (staging / "job.json").write_text("new", encoding="utf-8")

    def explode(*args, **kwargs):
        raise OSError("rename failed")

    monkeypatch.setattr(store, "publish", explode)
    with pytest.raises(OSError):
        _publish_with_damage(store, job_id, staging, damaged=damaged)
    assert (target / "job.json").read_text(encoding="utf-8") == "old"


# --------------------------------------------------------------------------- #
# takes list / compare / promote
# --------------------------------------------------------------------------- #
def _prepare_job(fake_engines, capsys, tmp_path, model_dir) -> tuple[str, Path]:
    FakeEngine.amplitudes = {1234: 0.1, 1235: 0.3}
    assert _say(tmp_path, model_dir, "--takes", "2", "--select", "lufs", "--json") == 0
    job_id = _json(capsys)["takes"]["job_id"]
    return job_id, tmp_path / "data" / "takes" / job_id


def test_takes_list_compare_and_promote_roundtrip(
    fake_engines, capsys, tmp_path, model_dir
):
    job_id, job_dir = _prepare_job(fake_engines, capsys, tmp_path, model_dir)

    assert _run(tmp_path, model_dir, "takes", "list", job_id, "--json") == 0
    listed = _json(capsys)
    assert listed["job_id"] == job_id
    assert listed["state"] == "complete"

    assert (
        _run(tmp_path, model_dir, "takes", "compare", job_id, "--ids", "1,2", "--json")
        == 0
    )
    compared = _json(capsys)
    assert [take["id"] for take in compared["takes"]] == [1, 2]
    assert compared["takes"][0]["metrics"]["lufs"] is not None

    out = tmp_path / "promoted.wav"
    assert (
        _run(
            tmp_path,
            model_dir,
            "takes",
            "promote",
            job_id,
            "--id",
            "2",
            "--out",
            str(out),
            "--json",
        )
        == 0
    )
    payload = _json(capsys)
    assert payload["selected"] == 2
    assert out.read_bytes() == (job_dir / "take-02.wav").read_bytes()
    assert _job(tmp_path, job_id)["selected"] == 2

    selected_at = _job(tmp_path, job_id)["selected_at"]
    rc = _run(
        tmp_path,
        model_dir,
        "takes",
        "promote",
        job_id,
        "--id",
        "2",
        "--out",
        str(job_dir / "take-02.wav"),
        "--json",
    )
    assert rc == 0
    assert _json(capsys)["idempotent"] is True
    assert _job(tmp_path, job_id)["selected_at"] == selected_at


def test_promote_rejects_assets_and_transcoding(
    fake_engines, capsys, tmp_path, model_dir
):
    job_id, job_dir = _prepare_job(fake_engines, capsys, tmp_path, model_dir)
    for target in (
        job_dir / "job.json",
        job_dir / "take-01.wav",
        job_dir / "take-01.recipe.json",
    ):
        assert (
            _run(
                tmp_path,
                model_dir,
                "takes",
                "promote",
                job_id,
                "--id",
                "2",
                "--out",
                str(target),
            )
            == 2
        )
        capsys.readouterr()
    assert (
        _run(
            tmp_path,
            model_dir,
            "takes",
            "promote",
            job_id,
            "--id",
            "2",
            "--out",
            str(tmp_path / "x.mp3"),
        )
        == 2
    )
    capsys.readouterr()
    assert (
        _run(
            tmp_path,
            model_dir,
            "takes",
            "promote",
            job_id,
            "--id",
            "2",
            "--out",
            str(tmp_path / "x.wav"),
            "--format",
            "flac",
        )
        == 2
    )
    capsys.readouterr()
    assert (
        _run(
            tmp_path,
            model_dir,
            "takes",
            "promote",
            job_id,
            "--id",
            "2",
            "--out",
            "-",
            "--json",
        )
        == 2
    )
    capsys.readouterr()


def test_takes_path_safety(fake_engines, capsys, tmp_path, model_dir):
    job_id, _ = _prepare_job(fake_engines, capsys, tmp_path, model_dir)
    assert _run(tmp_path, model_dir, "takes", "list", "../escape") == 2
    capsys.readouterr()
    assert _run(tmp_path, model_dir, "takes", "list", "/tmp/absolute") == 2
    capsys.readouterr()
    assert _run(tmp_path, model_dir, "takes", "list", "tk_missing") == 4
    capsys.readouterr()
    assert _run(tmp_path, model_dir, "takes", "compare", job_id, "--ids", "9") == 4
    capsys.readouterr()
    assert _run(tmp_path, model_dir, "takes", "compare", job_id, "--ids", "1,1") == 2
    capsys.readouterr()

    link = tmp_path / "data" / "takes" / "tk_symlink"
    link.symlink_to(tmp_path, target_is_directory=True)
    assert _run(tmp_path, model_dir, "takes", "list", "tk_symlink") == 2
    capsys.readouterr()


def test_promote_out_dash_writes_only_bytes(
    fake_engines, capsysbinary, tmp_path, model_dir
):
    job_id, job_dir = _prepare_job(fake_engines, capsysbinary, tmp_path, model_dir)
    rc = _run(
        tmp_path, model_dir, "takes", "promote", job_id, "--id", "1", "--out", "-"
    )
    captured = capsysbinary.readouterr()
    assert rc == 0
    assert captured.out == (job_dir / "take-01.wav").read_bytes()
    assert _job(tmp_path, job_id)["selected"] == 1


# --------------------------------------------------------------------------- #
# Review fixes
# --------------------------------------------------------------------------- #
def _write_job_dir(job_dir: Path, job: dict) -> None:
    (job_dir / "job.json").write_text(json.dumps(job), encoding="utf-8")


def _scoring_failed_job(
    fake_engines, capsys, tmp_path, asr_model_dir
) -> tuple[str, Path]:
    FakeAsrEngine.error = ModelError("asr exploded")
    assert (
        _say(tmp_path, asr_model_dir, "--takes", "2", "--select", "cer", "--no-cache")
        == 3
    )
    capsys.readouterr()
    job_id = next((tmp_path / "data" / "takes").glob("tk_*")).name
    return job_id, tmp_path / "data" / "takes" / job_id


def test_reference_voice_changes_job_id_and_does_not_reuse_complete_job(
    fake_engines, capsys, tmp_path, model_dir
):
    _write_clone(tmp_path, "alice")
    _write_clone(tmp_path, "bob")
    FakeEngine.amplitudes = {1234: 0.5, 1235: 0.2}

    assert (
        _say(
            tmp_path,
            model_dir,
            "--takes",
            "2",
            "--select",
            "speaker-sim",
            "--reference-voice",
            "alice",
            "--no-cache",
            "--json",
        )
        == 0
    )
    alice_payload = _json(capsys)
    alice_id = alice_payload["takes"]["job_id"]
    alice_dir = tmp_path / "data" / "takes" / alice_id
    alice_job = (alice_dir / "job.json").read_bytes()
    alice_recipe = json.loads(
        (alice_dir / "take-01.recipe.json").read_text(encoding="utf-8")
    )
    assert alice_job
    assert alice_recipe["selection"]["reference"]["name"] == "alice"
    assert (
        _job(tmp_path, alice_id)["request"]["selection"]["reference"]["name"]
        == "alice"
    )

    assert (
        _say(
            tmp_path,
            model_dir,
            "--takes",
            "2",
            "--select",
            "speaker-sim",
            "--reference-voice",
            "bob",
            "--no-cache",
            "--json",
        )
        == 0
    )
    bob_payload = _json(capsys)
    bob_id = bob_payload["takes"]["job_id"]
    bob_dir = tmp_path / "data" / "takes" / bob_id
    assert bob_id != alice_id
    assert alice_dir.is_dir() and bob_dir.is_dir()
    assert bob_payload["takes"]["reused"] is False
    assert len(FakeEngine.seeds) == 4
    assert (
        _job(tmp_path, bob_id)["request"]["selection"]["reference"]["name"] == "bob"
    )
    bob_recipe = json.loads(
        (bob_dir / "take-01.recipe.json").read_text(encoding="utf-8")
    )
    assert bob_recipe["selection"]["reference"]["name"] == "bob"
    # Running bob must not rewrite alice's completed job.
    assert (alice_dir / "job.json").read_bytes() == alice_job


def test_promote_scoring_failed_transitions_and_has_no_side_effect_on_failure(
    fake_engines, monkeypatch, capsysbinary, tmp_path, asr_model_dir
):
    job_id, job_dir = _scoring_failed_job(
        fake_engines, capsysbinary, tmp_path, asr_model_dir
    )
    out = tmp_path / "promoted.wav"
    assert (
        _run(
            tmp_path,
            asr_model_dir,
            "takes",
            "promote",
            job_id,
            "--id",
            "1",
            "--out",
            str(out),
        )
        == 0
    )
    capsysbinary.readouterr()
    assert out.read_bytes() == (job_dir / "take-01.wav").read_bytes()
    job = _job(tmp_path, job_id)
    assert job["state"] == "complete"
    assert job["selected"] == 1
    assert job["selected_at"]

    # Failure to persist job.json must not leave the copied file behind.
    job2_id, job2_dir = _scoring_failed_job(
        fake_engines, capsysbinary, tmp_path / "second", asr_model_dir
    )
    target = tmp_path / "second" / "promoted.wav"

    def explode(*args, **kwargs):
        raise OSError("disk full")

    with monkeypatch.context() as patch:
        patch.setattr(TakesStore, "update_job", explode)
        assert (
            _run(
                tmp_path / "second",
                asr_model_dir,
                "takes",
                "promote",
                job2_id,
                "--id",
                "1",
                "--out",
                str(target),
            )
            == 4
        )
        assert not target.exists()
        assert _job(tmp_path / "second", job2_id)["state"] == "scoring_failed"
        assert (
            _run(
                tmp_path / "second",
                asr_model_dir,
                "takes",
                "promote",
                job2_id,
                "--id",
                "1",
                "--out",
                "-",
            )
            == 4
        )
        assert capsysbinary.readouterr().out == b""

    # source == target on a scoring_failed job: select in place, no copy.
    promoted = job2_dir / "take-01.wav"
    before = promoted.read_bytes()
    assert (
        _run(
            tmp_path / "second",
            asr_model_dir,
            "takes",
            "promote",
            job2_id,
            "--id",
            "1",
            "--out",
            str(promoted),
        )
        == 0
    )
    capsysbinary.readouterr()
    assert promoted.read_bytes() == before
    assert _job(tmp_path / "second", job2_id)["state"] == "complete"
    snapshot = (job2_dir / "job.json").read_bytes()
    assert (
        _run(
            tmp_path / "second",
            asr_model_dir,
            "takes",
            "promote",
            job2_id,
            "--id",
            "1",
            "--out",
            str(promoted),
            "--json",
        )
        == 0
    )
    assert _json(capsysbinary)["idempotent"] is True
    assert (job2_dir / "job.json").read_bytes() == snapshot


def test_damaged_job_restored_when_generation_fails(
    fake_engines, monkeypatch, capsys, tmp_path, model_dir
):
    FakeEngine.amplitudes = {1234: 0.1, 1235: 0.1}
    assert _say(tmp_path, model_dir, "--takes", "2", "--select", "lufs", "--json") == 0
    job_id = _json(capsys)["takes"]["job_id"]
    job_dir = tmp_path / "data" / "takes" / job_id
    (job_dir / "job.json").write_text("{not json", encoding="utf-8")

    original = FakeEngine.generate

    def flaky(self, **kwargs):
        if int(kwargs["seed"]) == 1235:
            raise ModelError("second take failed")
        return original(self, **kwargs)

    monkeypatch.setattr(FakeEngine, "generate", flaky)
    rc = _say(
        tmp_path, model_dir, "--takes", "2", "--select", "lufs", "--no-cache"
    )
    assert rc == 3
    assert (job_dir / "job.json").read_text(encoding="utf-8") == "{not json"
    takes_root = tmp_path / "data" / "takes"
    assert not list(takes_root.glob(".staging-*"))
    assert not list(takes_root.glob(".damaged-*"))


def test_cer_missing_qwen_asr_preflight_exits_3_before_tts(
    fake_engines, monkeypatch, capsys, tmp_path, asr_model_dir
):
    from qtts_cli import takes_run

    def boom():
        raise ModelError("qwen-asr missing")

    monkeypatch.setattr(takes_run, "preflight_asr_runtime", boom)
    assert (
        _say(
            tmp_path,
            asr_model_dir,
            "--takes",
            "2",
            "--select",
            "cer",
            "--no-cache",
        )
        == 3
    )
    assert FakeEngine.seeds == []
    assert not list((tmp_path / "data" / "takes").glob("tk_*"))
    capsys.readouterr()

    calls: list[int] = []

    def record():
        calls.append(1)

    monkeypatch.setattr(takes_run, "preflight_asr_runtime", record)
    assert (
        _say(
            tmp_path,
            asr_model_dir,
            "--takes",
            "2",
            "--select",
            "cer",
            "--dry-run",
            "--json",
        )
        == 0
    )
    assert calls == []
    capsys.readouterr()


def test_takes_list_compare_reject_audio_hash_mismatch(
    fake_engines, capsys, tmp_path, model_dir
):
    job_id, job_dir = _prepare_job(fake_engines, capsys, tmp_path, model_dir)
    audio = job_dir / "take-01.wav"
    data = bytearray(audio.read_bytes())
    data[0] ^= 0xFF
    audio.write_bytes(bytes(data))

    assert _run(tmp_path, model_dir, "takes", "list", job_id, "--json") == 4
    assert capsys.readouterr().out == ""
    assert _run(tmp_path, model_dir, "takes", "compare", job_id, "--json") == 4
    assert capsys.readouterr().out == ""


def test_takes_list_compare_reject_corrupt_recipe(
    fake_engines, capsys, tmp_path, model_dir
):
    job_id, job_dir = _prepare_job(fake_engines, capsys, tmp_path, model_dir)
    (job_dir / "take-01.recipe.json").write_text("{oops", encoding="utf-8")
    assert _run(tmp_path, model_dir, "takes", "list", job_id) == 4
    assert capsys.readouterr().out == ""
    assert _run(tmp_path, model_dir, "takes", "compare", job_id) == 4
    assert capsys.readouterr().out == ""


def test_reused_job_with_corrupt_selected_recipe_exits_4_before_output(
    fake_engines, capsysbinary, tmp_path, model_dir
):
    FakeEngine.amplitudes = {1234: 0.1, 1235: 0.3}
    assert (
        _say(tmp_path, model_dir, "--takes", "2", "--select", "lufs", "--json") == 0
    )
    job_id = _json(capsysbinary)["takes"]["job_id"]
    job_dir = tmp_path / "data" / "takes" / job_id
    job = _job(tmp_path, job_id)
    selected = next(take for take in job["takes"] if take["id"] == job["selected"])
    (job_dir / selected["recipe"]).write_text("{oops", encoding="utf-8")
    before = (job_dir / "job.json").read_bytes()

    rc = _say(
        tmp_path,
        model_dir,
        "--takes",
        "2",
        "--select",
        "lufs",
        "--out",
        "-",
        "--no-cache",
    )
    assert rc == 4
    assert capsysbinary.readouterr().out == b""
    assert (job_dir / "job.json").read_bytes() == before


def test_job_segment_schema_mismatch_exits_4(
    fake_engines, capsys, tmp_path, model_dir
):
    job_id, job_dir = _prepare_job(fake_engines, capsys, tmp_path, model_dir)
    base = _job(tmp_path, job_id)
    variants = []

    missing = copy.deepcopy(base)
    missing["takes"][0]["segments"].pop()
    variants.append(missing)

    wrong_id = copy.deepcopy(base)
    wrong_id["takes"][0]["segments"][0]["id"] = "bogus"
    variants.append(wrong_id)

    wrong_seeds = copy.deepcopy(base)
    wrong_seeds["takes"][0]["segment_seeds"] = {"bogus": 1234}
    variants.append(wrong_seeds)

    for variant in variants:
        _write_job_dir(job_dir, variant)
        assert _run(tmp_path, model_dir, "takes", "list", job_id) == 4
        assert capsys.readouterr().out == ""


def test_out_of_range_segment_timing_exits_4(
    fake_engines, capsys, tmp_path, model_dir
):
    FakeEngine.amplitudes = {1234: 0.1, 1235: 0.2}
    assert (
        _say(
            tmp_path,
            model_dir,
            "--takes",
            "2",
            "--select",
            "lufs",
            "--json",
            text="你好。世界。",
        )
        == 0
    )
    job_id = _json(capsys)["takes"]["job_id"]
    job_dir = tmp_path / "data" / "takes" / job_id
    base = _job(tmp_path, job_id)
    variants = []

    beyond = copy.deepcopy(base)
    segment = beyond["takes"][0]["segments"][0]
    segment["end_s"] = beyond["takes"][0]["duration_s"] + 10
    variants.append(beyond)

    negative = copy.deepcopy(base)
    segment = negative["takes"][0]["segments"][0]
    segment["start_s"] = -1.0
    segment["end_s"] = -1.0 + segment["duration_s"]
    variants.append(negative)

    overlap = copy.deepcopy(base)
    segments = overlap["takes"][0]["segments"]
    segments[1]["start_s"] = 0.0
    segments[1]["end_s"] = segments[1]["duration_s"]
    variants.append(overlap)

    for variant in variants:
        _write_job_dir(job_dir, variant)
        assert _run(tmp_path, model_dir, "takes", "list", job_id) == 4
        assert capsys.readouterr().out == ""


def test_selected_bool_rejected(fake_engines, capsys, tmp_path, model_dir):
    job_id, job_dir = _prepare_job(fake_engines, capsys, tmp_path, model_dir)
    job = _job(tmp_path, job_id)
    job["selected"] = True
    _write_job_dir(job_dir, job)
    assert _run(tmp_path, model_dir, "takes", "list", job_id) == 4
    assert capsys.readouterr().out == ""


def test_takes_downstream_rejects_overwriting_job_assets(
    fake_engines, monkeypatch, capsys, tmp_path, model_dir
):
    from qtts_cli import takes_run

    job_id, job_dir = _prepare_job(fake_engines, capsys, tmp_path, model_dir)
    protected = [job_dir / "job.json", job_dir / "take-01.wav", job_dir / "take-01.recipe.json"]
    snapshot = {path: path.read_bytes() for path in protected}

    for target in protected:
        assert (
            _say(
                tmp_path,
                model_dir,
                "--takes",
                "2",
                "--select",
                "lufs",
                "--out",
                str(target),
            )
            == 2
        )
        assert capsys.readouterr().out == ""

    assert (
        _say(
            tmp_path,
            model_dir,
            "--takes",
            "2",
            "--select",
            "lufs",
            "--recipe-out",
            str(job_dir / "job.json"),
        )
        == 2
    )
    capsys.readouterr()

    assert (
        _say(
            tmp_path,
            model_dir,
            "--takes",
            "2",
            "--select",
            "lufs",
            "--out",
            str(job_dir / "custom.wav"),
        )
        == 0
    )
    assert (job_dir / "custom.wav").is_file()
    capsys.readouterr()

    FakeAligner.calls = 0
    monkeypatch.setattr(takes_run, "AlignerEngine", FakeAligner)
    assert (
        _say(
            tmp_path,
            model_dir,
            "--takes",
            "2",
            "--select",
            "lufs",
            "--subtitle",
            "srt",
            "--subtitle-out-prefix",
            str(job_dir / "take-01"),
        )
        == 0
    )
    assert (job_dir / "take-01.srt").is_file()
    capsys.readouterr()

    for path, before in snapshot.items():
        assert path.read_bytes() == before
