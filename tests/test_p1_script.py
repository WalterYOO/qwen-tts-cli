"""FR-P1-02: multi-role script lint/compile and `say --script`."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from qtts_cli import script as script_mod
from qtts_cli.cli import main
from qtts_cli.commands import say as say_cmd
from qtts_cli.errors import ModelError


def _write_script(tmp_path: Path, body: str, suffix: str = ".yaml") -> Path:
    path = tmp_path / f"script{suffix}"
    path.write_text(body, encoding="utf-8")
    return path


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


def _two_role_yaml() -> str:
    return """
schema_version: 1
defaults:
  language: Chinese
  format: wav
segments:
  - id: narrator
    text: "你好。"
    voice: "builtin:Vivian"
    pause_after_ms: 100
  - id: guest
    text: "世界。"
    voice: "design:温柔"
"""


class FakeEngine:
    loads = 0
    instances = []

    def __init__(self, cfg, model):
        self.cfg = cfg
        self.model_info = model
        self.device = "cpu"
        self.dtype_name = "fp32"
        self.load_ms = 0.0
        self.model = None
        FakeEngine.instances.append(self)

    def load(self):
        FakeEngine.loads += 1
        self.model = object()
        return self

    def generate(self, **kwargs):
        return np.zeros(4800, dtype=np.float32), 24000, {"generate_ms": 1.0}

    def build_clone_prompt_items(self, *args, **kwargs):
        return []


def test_compile_is_deterministic_and_loads_no_models(
    monkeypatch, tmp_path: Path, model_dir: Path
):
    script = _write_script(tmp_path, _two_role_yaml())

    def boom(self):
        raise AssertionError("compile must not load models")

    monkeypatch.setattr(say_cmd.TtsEngine, "load", boom)
    first = tmp_path / "a.plan.json"
    second = tmp_path / "b.plan.json"
    assert _run(tmp_path, model_dir, "script", "compile", str(script), "-o", str(first)) == 0
    assert _run(tmp_path, model_dir, "script", "compile", str(script), "-o", str(second)) == 0
    assert first.read_bytes() == second.read_bytes()
    plan = json.loads(first.read_text(encoding="utf-8"))
    assert [seg["id"] for seg in plan["segments"]] == ["narrator", "guest"]
    assert {seg["model"] for seg in plan["segments"]} == {
        "Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    }
    assert plan["segments"][0]["source_segment_id"] == "narrator"
    assert plan["segments"][1]["display_text"] == "世界。"
    assert "cache" not in plan
    assert "script_source" not in plan


def test_json_script_generates_ids_and_inherits_defaults(
    capsys, tmp_path: Path, model_dir: Path
):
    script = _write_script(
        tmp_path,
        json.dumps(
            {
                "schema_version": 1,
                "defaults": {
                    "voice": "builtin:Vivian",
                    "language": "Chinese",
                    "pause_ms": 120,
                },
                "segments": [{"text": "第一句。"}, {"text": "第二句。"}],
            },
            ensure_ascii=False,
        ),
        suffix=".json",
    )
    out = tmp_path / "plan.json"
    assert _run(tmp_path, model_dir, "script", "compile", str(script), "-o", str(out)) == 0
    capsys.readouterr()
    plan = json.loads(out.read_text(encoding="utf-8"))
    assert [seg["id"] for seg in plan["segments"]] == ["s001", "s002"]
    assert [seg["source_segment_id"] for seg in plan["segments"]] == ["s001", "s002"]
    assert plan["segments"][0]["pause_after_ms"] == 120
    assert plan["segments"][1]["language"] == "Chinese"


def test_lint_reports_segment_id_for_unknown_voice(
    capsys, tmp_path: Path, model_dir: Path
):
    script = _write_script(
        tmp_path,
        """
schema_version: 1
segments:
  - id: narrator
    text: "你好。"
    voice: "builtin:Nobody"
""",
    )
    rc = _run(tmp_path, model_dir, "script", "lint", str(script), "--json")
    assert rc == 4
    payload = json.loads(capsys.readouterr().err)
    assert payload["details"]["segment"] == "narrator"


def test_duplicate_ids_and_unknown_fields_exit_2(
    tmp_path: Path, model_dir: Path
):
    duplicate = _write_script(
        tmp_path,
        """
schema_version: 1
segments:
  - id: a
    text: "你好。"
    voice: "builtin:Vivian"
  - id: a
    text: "世界。"
    voice: "builtin:Vivian"
""",
    )
    assert _run(tmp_path, model_dir, "script", "lint", str(duplicate)) == 2
    unknown = _write_script(
        tmp_path,
        """
schema_version: 1
segments:
  - id: a
    text: "你好。"
    voice: "builtin:Vivian"
    speed: 2
""",
    )
    assert _run(tmp_path, model_dir, "script", "lint", str(unknown)) == 2


def test_segment_schema_errors_exit_2_before_model_scan(tmp_path: Path):
    empty_models = tmp_path / "empty_models"
    empty_models.mkdir()
    missing_text = _write_script(
        tmp_path,
        """
schema_version: 1
segments:
  - id: a
    voice: "builtin:Vivian"
""",
    )
    assert _run(tmp_path, empty_models, "script", "lint", str(missing_text)) == 2

    bad_type = _write_script(
        tmp_path,
        """
schema_version: 1
segments:
  - id: a
    text: "你好。"
    voice: "builtin:Vivian"
    pause_after_ms: "soon"
""",
    )
    assert _run(tmp_path, empty_models, "script", "lint", str(bad_type)) == 2

    bad_yaml = _write_script(tmp_path, "schema_version: 1\nsegments: [\n")
    assert _run(tmp_path, empty_models, "script", "lint", str(bad_yaml)) == 2


def test_missing_icl_ref_text_and_unsupported_language_are_lint_errors(
    tmp_path: Path, model_dir: Path
):
    voice_dir = tmp_path / "data" / "voices" / "alice"
    voice_dir.mkdir(parents=True)
    (voice_dir / "voice.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "name": "alice",
                "kind": "clone",
                "source": {"type": "clone", "mode": "icl"},
                "model": {"id": "m", "revision": None},
            }
        ),
        encoding="utf-8",
    )
    (voice_dir / "prompt.safetensors").write_bytes(b"fake")
    missing_text = _write_script(
        tmp_path,
        """
schema_version: 1
segments:
  - id: alice
    text: "你好。"
    voice: "voice:alice"
""",
    )
    assert _run(tmp_path, model_dir, "script", "lint", str(missing_text)) == 4
    language = _write_script(
        tmp_path,
        """
schema_version: 1
segments:
  - id: klingon
    text: "Hello."
    voice: "builtin:Vivian"
    language: Klingon
""",
    )
    assert _run(tmp_path, model_dir, "script", "lint", str(language)) == 4


def test_model_override_conflicts_with_multi_model_script(
    tmp_path: Path, model_dir: Path
):
    script = _write_script(tmp_path, _two_role_yaml())
    rc = _run(
        tmp_path,
        model_dir,
        "script",
        "lint",
        str(script),
        "--model",
        "Qwen3-TTS-12Hz-1.7B-CustomVoice",
    )
    assert rc == 2


def test_yaml_strictness(tmp_path: Path):
    duplicate = _write_script(
        tmp_path,
        """
schema_version: 1
segments: []
segments: []
""",
    )
    with pytest.raises(Exception) as caught:
        script_mod.load_script(duplicate)
    assert getattr(caught.value, "code", None) == 2

    multi = _write_script(tmp_path, "schema_version: 1\n---\nschema_version: 1\n")
    with pytest.raises(Exception) as caught:
        script_mod.load_script(multi)
    assert getattr(caught.value, "code", None) == 2

    alias = _write_script(
        tmp_path,
        """
schema_version: 1
defaults: &d
  language: Chinese
segments: []
other: *d
""",
    )
    with pytest.raises(Exception) as caught:
        script_mod.load_script(alias)
    assert getattr(caught.value, "code", None) == 2

    anchor = _write_script(
        tmp_path,
        """
schema_version: 1
defaults: &d
  language: Chinese
segments: []
""",
    )
    with pytest.raises(Exception) as caught:
        script_mod.load_script(anchor)
    assert getattr(caught.value, "code", None) == 2


def test_top_level_project_is_rejected(tmp_path: Path, model_dir: Path):
    script = _write_script(
        tmp_path,
        """
schema_version: 1
project: demo
segments:
  - text: "你好。"
    voice: "builtin:Vivian"
""",
    )
    assert _run(tmp_path, model_dir, "script", "lint", str(script)) == 2


def test_missing_pyyaml_is_model_error(monkeypatch, tmp_path: Path):
    script = _write_script(tmp_path, "schema_version: 1\nsegments: []\n")
    monkeypatch.setattr(script_mod, "yaml", None)
    with pytest.raises(ModelError):
        script_mod.load_script(script)


def test_say_script_switches_models_with_one_slot(
    monkeypatch, capsys, tmp_path: Path, model_dir: Path
):
    FakeEngine.loads = 0
    FakeEngine.instances = []
    monkeypatch.setattr(say_cmd, "TtsEngine", FakeEngine)
    script = _write_script(tmp_path, _two_role_yaml())
    out = tmp_path / "out.wav"
    rc = _run(
        tmp_path,
        model_dir,
        "say",
        "--script",
        str(script),
        "--out",
        str(out),
        "--json",
    )
    assert rc == 0, capsys.readouterr().err
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert out.is_file()
    assert len(payload["segments"]) == 2
    assert len(payload["models"]) == 2
    assert payload["duration_s"] > 0
    assert FakeEngine.loads == 2
    live = sum(1 for engine in FakeEngine.instances if engine.model is not None)
    assert live == 1


def test_say_script_dry_run_matches_compiled_segments(
    monkeypatch, capsys, tmp_path: Path, model_dir: Path
):
    monkeypatch.setattr(say_cmd, "TtsEngine", FakeEngine)
    script = _write_script(tmp_path, _two_role_yaml())
    plan_path = tmp_path / "plan.json"
    assert _run(tmp_path, model_dir, "script", "compile", str(script), "-o", str(plan_path)) == 0
    capsys.readouterr()
    rc = _run(
        tmp_path,
        model_dir,
        "say",
        "--script",
        str(script),
        "--dry-run",
        "--json",
    )
    assert rc == 0
    dry = json.loads(capsys.readouterr().out)
    compiled = json.loads(plan_path.read_text(encoding="utf-8"))
    assert dry["plan"]["segments"] == compiled["segments"]
    assert dry["models_loaded"] is False


def test_multisentence_segment_pieces_have_unique_ids(
    capsys, tmp_path: Path, model_dir: Path
):
    script = _write_script(
        tmp_path,
        """
schema_version: 1
segments:
  - text: "你好。再见。"
    voice: "builtin:Vivian"
  - id: tail
    text: "第三句。"
    voice: "builtin:Vivian"
""",
    )
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    assert _run(tmp_path, model_dir, "script", "compile", str(script), "-o", str(first)) == 0
    capsys.readouterr()
    assert _run(tmp_path, model_dir, "script", "compile", str(script), "-o", str(second)) == 0
    capsys.readouterr()
    assert first.read_bytes() == second.read_bytes()
    plan = json.loads(first.read_text(encoding="utf-8"))
    assert [seg["id"] for seg in plan["segments"]] == ["s001.p1", "s001.p2", "tail"]
    assert [seg["source_segment_id"] for seg in plan["segments"]] == [
        "s001",
        "s001",
        "tail",
    ]
    assert "".join(seg["display_text"] for seg in plan["segments"]) == plan["display_text"]


def test_marker_split_pieces_have_unique_ids(capsys, tmp_path: Path, model_dir: Path):
    script = _write_script(
        tmp_path,
        """
schema_version: 1
segments:
  - text: "[[pause:200]]你好。[[pause:300]]再见。[[pause:250]]"
    voice: "builtin:Vivian"
""",
    )
    out = tmp_path / "plan.json"
    assert _run(tmp_path, model_dir, "script", "compile", str(script), "-o", str(out)) == 0
    capsys.readouterr()
    plan = json.loads(out.read_text(encoding="utf-8"))
    assert [seg["id"] for seg in plan["segments"]] == ["s001.p1", "s001.p2"]
    assert [seg["source_segment_id"] for seg in plan["segments"]] == ["s001", "s001"]
    assert plan["leading_pause_ms"] == 200
    assert plan["trailing_pause_ms"] == 250
    assert plan["segments"][0]["pause_after_ms"] == 300
    assert plan["segments"][1]["pause_after_ms"] == 0


def test_say_script_rejects_conflicting_input_sources(tmp_path: Path, model_dir: Path):
    script = _write_script(tmp_path, _two_role_yaml())
    source = tmp_path / "source.txt"
    source.write_text("ignored", encoding="utf-8")
    out = str(tmp_path / "out.wav")
    assert _run(
        tmp_path,
        model_dir,
        "say",
        "ignored",
        "--script",
        str(script),
        "--out",
        out,
    ) == 2
    assert _run(
        tmp_path,
        model_dir,
        "say",
        "--stdin",
        "--script",
        str(script),
        "--out",
        out,
    ) == 2
    assert _run(
        tmp_path,
        model_dir,
        "say",
        "--file",
        str(source),
        "--script",
        str(script),
        "--out",
        out,
    ) == 2
    assert not Path(out).exists()


def test_say_script_cli_options_override_defaults(
    capsys, tmp_path: Path, model_dir: Path
):
    script = _write_script(
        tmp_path,
        """
schema_version: 1
defaults:
  language: Chinese
  pause_ms: 300
segments:
  - text: "Hello."
    voice: "builtin:Vivian"
  - text: "World."
    voice: "builtin:Vivian"
""",
    )
    rc = _run(
        tmp_path,
        model_dir,
        "say",
        "--script",
        str(script),
        "--dry-run",
        "--json",
        "--language",
        "English",
        "--instruct",
        "calm",
        "--max-new-tokens",
        "512",
        "--pause-ms",
        "150",
    )
    assert rc == 0, capsys.readouterr().err
    plan = json.loads(capsys.readouterr().out)["plan"]
    assert plan["language"] == "English"
    assert plan["instruct"] == "calm"
    assert plan["sampling"]["max_new_tokens"] == 512
    assert all(seg["language"] == "English" for seg in plan["segments"])
    assert plan["segments"][0]["pause_after_ms"] == 150


def test_say_script_cli_overrides_reach_recipe(
    monkeypatch, capsys, tmp_path: Path, model_dir: Path
):
    monkeypatch.setattr(say_cmd, "TtsEngine", FakeEngine)
    script = _write_script(
        tmp_path,
        """
schema_version: 1
defaults:
  language: Chinese
  pause_ms: 300
segments:
  - text: "Hello."
    voice: "builtin:Vivian"
  - text: "World."
    voice: "builtin:Vivian"
""",
    )
    out = tmp_path / "overrides.wav"
    rc = _run(
        tmp_path,
        model_dir,
        "say",
        "--script",
        str(script),
        "--out",
        str(out),
        "--json",
        "--language",
        "English",
        "--instruct",
        "calm",
        "--max-new-tokens",
        "512",
        "--pause-ms",
        "150",
    )
    assert rc == 0, capsys.readouterr().err
    capsys.readouterr()
    recipe = json.loads(out.with_suffix(".recipe.json").read_text(encoding="utf-8"))
    assert recipe["language"] == "English"
    assert recipe["instruct"] == "calm"
    assert recipe["sampling"]["max_new_tokens"] == 512
    first, second = recipe["segments"]
    assert first["language"] == "English"
    assert first["instruct"] == "calm"
    assert first["max_new_tokens"] == 512
    assert first["pause_after_ms"] == 150
    assert second["pause_after_ms"] == 0


def test_script_leading_trailing_pauses_reach_plan_and_audio(
    monkeypatch, capsys, tmp_path: Path, model_dir: Path
):
    import soundfile as sf

    monkeypatch.setattr(say_cmd, "TtsEngine", FakeEngine)
    script = _write_script(
        tmp_path,
        """
schema_version: 1
segments:
  - text: "[[pause:500]]你好。[[pause:250]]"
    voice: "builtin:Vivian"
""",
    )
    plan_path = tmp_path / "plan.json"
    assert _run(tmp_path, model_dir, "script", "compile", str(script), "-o", str(plan_path)) == 0
    capsys.readouterr()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert plan["leading_pause_ms"] == 500
    assert plan["trailing_pause_ms"] == 250
    assert [seg["id"] for seg in plan["segments"]] == ["s001"]

    out = tmp_path / "pauses.wav"
    rc = _run(
        tmp_path,
        model_dir,
        "say",
        "--script",
        str(script),
        "--out",
        str(out),
        "--json",
    )
    assert rc == 0, capsys.readouterr().err
    info = sf.info(str(out))
    assert info.duration == pytest.approx(0.95, abs=0.01)


def test_script_middle_pauses_merge_into_neighbour_segments(
    capsys, tmp_path: Path, model_dir: Path
):
    script = _write_script(
        tmp_path,
        """
schema_version: 1
defaults:
  pause_ms: 300
segments:
  - id: a
    text: "第一句。"
    voice: "builtin:Vivian"
  - id: b
    text: "[[pause:200]]第二句。[[pause:300]]"
    voice: "builtin:Vivian"
  - id: c
    text: "第三句。"
    voice: "builtin:Vivian"
""",
    )
    out = tmp_path / "plan.json"
    assert _run(tmp_path, model_dir, "script", "compile", str(script), "-o", str(out)) == 0
    capsys.readouterr()
    plan = json.loads(out.read_text(encoding="utf-8"))
    pauses = [seg["pause_after_ms"] for seg in plan["segments"]]
    assert pauses == [500, 600, 0]
    assert plan["leading_pause_ms"] == 0
    assert plan["trailing_pause_ms"] == 0


def _write_icl_voice(tmp_path: Path, *, ref_text: str, mode: str = "icl") -> None:
    voice_dir = tmp_path / "data" / "voices" / "alice"
    voice_dir.mkdir(parents=True, exist_ok=True)
    (voice_dir / "voice.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "name": "alice",
                "kind": "clone",
                "source": {"type": "clone", "mode": mode, "ref_text": ref_text},
                "model": {"id": "m", "revision": None},
            }
        ),
        encoding="utf-8",
    )
    (voice_dir / "prompt.safetensors").write_bytes(b"fake")


def _dry_run_cache_key(tmp_path: Path, model_dir: Path, capsys) -> str:
    rc = _run(
        tmp_path,
        model_dir,
        "say",
        "你好。",
        "--voice",
        "voice:alice",
        "--language",
        "Chinese",
        "--dry-run",
        "--json",
    )
    assert rc == 0, capsys.readouterr().err
    return json.loads(capsys.readouterr().out)["plan"]["segments"][0]["cache_key"]


def test_icl_ref_text_change_invalidates_audio_cache(
    capsys, tmp_path: Path, model_dir: Path
):
    _write_icl_voice(tmp_path, ref_text="旧文本")
    first = _dry_run_cache_key(tmp_path, model_dir, capsys)
    _write_icl_voice(tmp_path, ref_text="新文本")
    second = _dry_run_cache_key(tmp_path, model_dir, capsys)
    assert first != second

    _write_icl_voice(tmp_path, ref_text="旧文本", mode="xvec")
    xvec_first = _dry_run_cache_key(tmp_path, model_dir, capsys)
    _write_icl_voice(tmp_path, ref_text="新文本", mode="xvec")
    xvec_second = _dry_run_cache_key(tmp_path, model_dir, capsys)
    assert xvec_first == xvec_second


def test_stdout_recipe_name_covers_full_request(capsys, tmp_path: Path, model_dir: Path):
    def recipe_line(voice: str) -> str:
        rc = _run(
            tmp_path,
            model_dir,
            "say",
            "你好。",
            "--voice",
            voice,
            "--out",
            "-",
            "--dry-run",
        )
        assert rc == 0, capsys.readouterr().err
        output = capsys.readouterr().out
        return next(line for line in output.splitlines() if line.startswith("will write:"))

    vivian = recipe_line("builtin:Vivian")
    ryan = recipe_line("builtin:Ryan")
    assert vivian != ryan
    assert recipe_line("builtin:Vivian") == vivian
