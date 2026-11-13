"""Offline design layer: YAML safety, compiler, lint, run matrix, report."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from qtts_cli.cli import main
from qtts_cli.design_compile import compile_design, template_sha256
from qtts_cli.design_lint import lint_design
from qtts_cli.design_report import CSV_COLUMNS, build_report
from qtts_cli.design_run import (
    MAX_CANDIDATES,
    SAMPLING_PROFILES,
    RunLock,
    build_matrix,
    matrix_signature,
    prepare_run,
)
from qtts_cli.design_yaml import (
    MAX_DESIGN_BYTES,
    TEMPLATE_STARTERS,
    dump_design,
    example_document,
    load_design,
    normalize_document,
    resolve_template_id,
)
from qtts_cli.errors import InputError, UsageError
from qtts_cli.utils import sha256_file, sha256_text

ZH_TEMPLATE = "qtts/zh-late-night-radio/v1"
EN_TEMPLATE = "qtts/en-podcast-host/v1"
SR = 24000


def _document(language: str = "zh", **overrides):
    doc = example_document("demo-voice", language=language)
    doc.update(overrides)
    return normalize_document(doc)


def _compiled(language: str = "zh", **overrides):
    return compile_design(_document(language, **overrides))


def _write(path: Path, text: str | bytes) -> Path:
    path.write_bytes(text if isinstance(text, bytes) else text.encode("utf-8"))
    return path


def _tone(seconds: float = 2.0, frequency: float = 220.0) -> np.ndarray:
    t = np.arange(int(round(SR * seconds)), dtype=np.float64) / float(SR)
    return (0.3 * np.sin(2.0 * math.pi * frequency * t)).astype(np.float32)


# --------------------------------------------------------------------------- #
# YAML safety
# --------------------------------------------------------------------------- #
def test_yaml_rejects_anchors_and_aliases(tmp_path):
    anchored = "base: &b\n  x: 1\nother: *b\n"
    with pytest.raises(InputError) as exc:
        load_design(_write(tmp_path / "anchors.yaml", anchored))
    assert exc.value.code == 4


def test_yaml_rejects_duplicate_keys(tmp_path):
    with pytest.raises(InputError, match="duplicate key"):
        load_design(_write(tmp_path / "dup.yaml", "name: a\nname: b\n"))


def test_yaml_rejects_oversize_file(tmp_path):
    raw = b"name: demo\n#" + b"x" * (MAX_DESIGN_BYTES + 1)
    with pytest.raises(InputError, match="larger than"):
        load_design(_write(tmp_path / "big.yaml", raw))


def test_yaml_rejects_non_utf8(tmp_path):
    with pytest.raises(InputError, match="UTF-8"):
        load_design(_write(tmp_path / "bin.yaml", b"\xff\xfe\x00\x01"))


def test_yaml_requires_mapping(tmp_path):
    with pytest.raises(InputError, match="mapping"):
        load_design(_write(tmp_path / "list.yaml", "- a\n- b\n"))


def test_normalize_collects_malformed_fields():
    document = normalize_document({"template": ZH_TEMPLATE, "identity": "warm"})
    assert any("identity must be a mapping" in item for item in document["errors"])
    report = lint_design(document, compile_design(document))
    assert not report["ok"]
    assert any(item["code"] == "invalid-field" for item in report["issues"])


def test_example_document_round_trips(tmp_path):
    document = _document()
    path = _write(tmp_path / "design.yaml", dump_design(document))
    loaded, raw, sha = load_design(path)
    assert loaded["name"] == "demo-voice"
    assert loaded["errors"] == []
    assert raw == path.read_text(encoding="utf-8")
    assert len(sha) == 64


# --------------------------------------------------------------------------- #
# Compiler
# --------------------------------------------------------------------------- #
def test_compile_zh_uses_structured_vocabulary():
    compiled = _compiled()
    text = compiled.rendered_instruction
    assert "深夜电台主持人" in text
    assert "音色温暖、干净、略带沙哑" in text
    assert "语速偏慢" in text
    assert "避免童声、喊叫" in text
    assert compiled.language == "zh"
    assert compiled.warnings == []


def test_compile_en_uses_structured_vocabulary():
    compiled = _compiled("en")
    text = compiled.rendered_instruction
    assert text.startswith("A in their thirties female voice")
    assert "warm, clear timbre" in text
    assert "podcast host" in text
    assert "Avoid childlike, shouting." in text


def test_instruction_override_wins_and_reports_warnings():
    compiled = _compiled(
        description="ignored description",
        instruction_override="完全由我决定的描述。",
    )
    assert compiled.rendered_instruction == "完全由我决定的描述。"
    codes = {item["code"] for item in compiled.warnings}
    assert {"both-overrides", "override-ignores-structure"} <= codes


def test_description_alone_is_used():
    compiled = _compiled(description="温暖的声音。")
    assert compiled.rendered_instruction == "温暖的声音。"
    assert compiled.warnings == []


def test_unknown_vocabulary_is_preserved_with_warning():
    document = _document()
    document["identity"]["timbre"] = ["silky"]
    compiled = compile_design(document)
    assert "silky" in compiled.rendered_instruction
    assert any(item["code"] == "unknown-vocab" for item in compiled.warnings)


def test_rendered_sha_is_stable():
    first = _compiled()
    second = _compiled()
    assert first.rendered_sha256 == second.rendered_sha256
    assert len(first.rendered_sha256) == 64


# --------------------------------------------------------------------------- #
# Lint
# --------------------------------------------------------------------------- #
def test_lint_valid_example_is_green():
    report = lint_design(_document(), _compiled())
    assert report["ok"] is True
    assert report["grade"] == "green"
    assert report["summary"] == {"blocks": 0, "warnings": 0, "infos": 0}
    assert report["total"] >= 85


def test_lint_blocks_unknown_template_and_empty_instruction():
    document = normalize_document(
        {
            "name": "demo-voice",
            "language": "zh",
            "template": "qtts/nope/v1",
            "covers": ["你好。"],
        }
    )
    from qtts_cli.commands.design import _compile_or_placeholder

    report = lint_design(document, _compile_or_placeholder(document))
    codes = {item["code"] for item in report["issues"]}
    assert {"invalid-template", "empty-instruction"} <= codes
    assert report["ok"] is False
    assert report["grade"] == "red"


def test_lint_blocks_missing_covers_and_inline_events():
    missing = _document()
    missing["present_fields"].remove("covers")
    missing["covers"] = []
    report = lint_design(missing, compile_design(missing))
    assert any(item["code"] == "no-anchors" for item in report["issues"])

    inline = _compiled(instruction_override="你好，[笑]今天过得怎么样？")
    report = lint_design(_document(instruction_override="你好，[笑]今天过得怎么样？"), inline)
    assert any(item["code"] == "inline-event" for item in report["issues"])


def test_lint_blocks_dialogue_and_language_mismatch():
    dialogue = _compiled(description="主持人：大家好。")
    report = lint_design(_document(description="主持人：大家好。"), dialogue)
    assert any(item["code"] == "dialogue-in-instruction" for item in report["issues"])

    mismatch = _compiled("en", instruction_override="这是中文描述。")
    report = lint_design(_document("en", instruction_override="这是中文描述。"), mismatch)
    assert any(item["code"] == "language-mismatch" for item in report["issues"])


def test_lint_warns_on_vague_terms_conflicts_and_long_covers():
    vague = _compiled(instruction_override="听起来自然就好。")
    report = lint_design(_document(instruction_override="听起来自然就好。"), vague)
    assert any(item["code"] == "vague-term" for item in report["issues"])

    conflict = _compiled(instruction_override="语速快一点又慢一点。")
    report = lint_design(_document(instruction_override="语速快一点又慢一点。"), conflict)
    assert any(item["code"] == "conflict" for item in report["issues"])

    cover = "啊" * 150
    document = _document(covers=[cover])
    report = lint_design(document, compile_design(document))
    assert any(item["code"] == "cover-length" for item in report["issues"])
    assert report["summary"]["warnings"] >= 1

    document = _document(covers=["一", "二", "三"])
    report = lint_design(document, compile_design(document))
    assert any(item["code"] == "too-many-covers" for item in report["issues"])


def test_lint_blocks_instruction_too_long():
    override = "好" * 1001
    compiled = _compiled(instruction_override=override)
    report = lint_design(_document(instruction_override=override), compiled)
    assert any(item["code"] == "instruction-too-long" for item in report["issues"])


def test_lint_reports_unknown_fields_and_name_format():
    document = _document(mood="calm")
    report = lint_design(document, compile_design(document))
    assert any(item["code"] == "unknown-field" and item["level"] == "info" for item in report["issues"])

    document = _document(name="bad/name")
    report = lint_design(document, compile_design(document))
    assert any(item["code"] == "name-format" for item in report["issues"])


@pytest.mark.parametrize("language", ["ja", "ko", "de", "fr", "ru", "pt", "es", "it"])
def test_multilanguage_templates_compile_and_lint(language):
    document = normalize_document(example_document("demo-voice", language=language))
    assert document["language"] == language
    assert document["covers"]
    compiled = compile_design(document)
    assert compiled.rendered_instruction.strip()
    report = lint_design(document, compiled)
    assert not [issue for issue in report["issues"] if issue["level"] == "block"]


def test_design_language_display_names_are_canonical():
    from qtts_cli.commands.design import _language_name

    assert _language_name("ja") == "Japanese"
    assert _language_name("Japanese") == "Japanese"
    assert _language_name("japanese") == "Japanese"
    assert _language_name("zh") == "Chinese"


def test_design_init_multilanguage(tmp_path):
    path = tmp_path / "ja-voice.yaml"
    assert main(["design", "init", str(path), "--language", "ja"]) == 0
    document, _, _ = load_design(path)
    assert document["language"] == "ja"
    assert document["template"] == "qtts/ja-podcast-host/v1"
    assert "また番組" in document["covers"][0]


def test_lint_unsupported_language_needs_override():
    document = normalize_document(
        {"name": "demo-voice", "language": "xx", "template": ZH_TEMPLATE, "covers": ["hello"]}
    )
    report = lint_design(document, compile_design(document))
    assert any(
        item["code"] == "unsupported-language" and item["level"] == "block"
        for item in report["issues"]
    )

    document["instruction_override"] = "Une voix calme."
    compiled = compile_design(document)
    report = lint_design(document, compiled)
    assert any(
        item["code"] == "unsupported-language" and item["level"] == "warn"
        for item in report["issues"]
    )


# --------------------------------------------------------------------------- #
# CLI: design init / lint
# --------------------------------------------------------------------------- #
def test_design_init_and_lint_exit_codes(tmp_path, capsys):
    path = tmp_path / "demo-voice.yaml"
    assert main(["design", "init", str(path)]) == 0
    assert path.is_file()
    assert "wrote" in capsys.readouterr().out

    assert main(["design", "init", str(path)]) == 2
    assert main(["design", "init", str(path), "--force"]) == 0
    assert main(["design", "lint", str(path)]) == 0
    out = capsys.readouterr().out
    assert "green" in out


def test_design_init_infers_language_from_template(tmp_path):
    path = tmp_path / "podcast.yaml"
    assert main(["design", "init", str(path), "--template", EN_TEMPLATE, "--name", "podcast"]) == 0
    document, _, _ = load_design(path)
    assert document["language"] == "en"
    assert document["template"] == EN_TEMPLATE
    assert main(
        ["design", "init", str(tmp_path / "conflict.yaml"), "--template", EN_TEMPLATE,
         "--language", "zh"]
    ) == 2


@pytest.mark.parametrize(
    "raw",
    [
        "zh-late-night-radio/v1",
        "btts/zh-late-night-radio/v1",
        "qtts/zh-late-night-radio/v1",
    ],
)
def test_template_aliases_share_one_identity(raw):
    canonical = "qtts/zh-late-night-radio/v1"
    assert resolve_template_id(raw) == canonical
    assert template_sha256(raw) == template_sha256(canonical)


def test_normalize_canonicalizes_btts_template_alias():
    document = normalize_document(
        {
            "name": "demo-voice",
            "language": "zh",
            "template": "btts/zh-documentary-narrator/v1",
            "covers": ["这片土地已经沉睡了千年。"],
        }
    )
    assert document["template"] == "qtts/zh-documentary-narrator/v1"


def test_unknown_template_is_not_guessed():
    assert resolve_template_id("btts/does-not-exist/v1") is None


@pytest.mark.parametrize(
    "template",
    [
        "qtts/zh-documentary-narrator/v1",
        "qtts/zh-anime-character/v1",
        "qtts/en-audiobook-narrator/v1",
        "qtts/en-game-character/v1",
    ],
)
def test_design_init_new_templates_use_their_own_starters(tmp_path, template):
    path = tmp_path / "new-template.yaml"
    assert main(["design", "init", str(path), "--template", template, "--name", "demo"]) == 0
    document, _, _ = load_design(path)
    assert document["template"] == template
    starter = TEMPLATE_STARTERS[template]
    assert document["identity"]["persona"] == starter["identity"]["persona"]
    assert document["covers"] == starter["covers"]


def test_design_init_accepts_btts_alias_and_writes_canonical(tmp_path, capsys):
    path = tmp_path / "alias.yaml"
    assert main(
        [
            "design",
            "init",
            str(path),
            "--template",
            "btts/zh-documentary-narrator/v1",
            "--name",
            "demo",
        ]
    ) == 0
    document, _, _ = load_design(path)
    assert document["template"] == "qtts/zh-documentary-narrator/v1"
    assert document["identity"]["persona"] == "documentary_narrator"


_BTTS_VOCAB = {
    "zh": {
        "age": ["teen", "child", "20s", "30s", "40s", "50s", "60s", "senior"],
        "gender_presentation": ["female", "male", "neutral"],
        "timbre": [
            "warm", "clean", "slightly_husky", "bright", "dark", "breathy",
            "nasal", "smooth", "crisp", "soft", "metallic", "deep",
        ],
        "pitch": ["low", "low_mid", "mid", "high_mid", "high"],
        "articulation": ["clear", "relaxed", "precise", "lazy", "crisp", "soft"],
        "persona": [
            "late_night_host", "documentary_narrator", "anime_character",
            "podcast_host", "audiobook_narrator", "game_character",
        ],
        "pace": ["slow", "medium_slow", "medium", "medium_fast", "fast"],
        "energy": ["low", "medium_low", "medium", "medium_high", "high"],
        "tone": [
            "calm", "intimate", "warm", "cool", "playful", "serious",
            "gentle", "wry",
        ],
        "proximity": ["close_mic", "mid_distance", "far_field", "far_mic"],
        "room": ["dry_studio", "treated_room", "natural_room"],
        "avoid": ["childlike", "shouting", "robotic", "overly_bright", "monotone"],
    },
    "en": {
        "age": ["teen", "child", "20s", "30s", "40s", "50s", "60s", "senior"],
        "gender_presentation": ["female", "male", "neutral"],
        "timbre": [
            "warm", "clean", "slightly_husky", "bright", "dark", "breathy",
            "nasal", "smooth", "crisp", "soft", "metallic", "deep",
        ],
        "pitch": ["low", "low_mid", "mid", "high_mid", "high"],
        "articulation": ["clear", "relaxed", "precise", "lazy", "crisp", "soft"],
        "persona": [
            "late_night_host", "documentary_narrator", "anime_character",
            "podcast_host", "audiobook_narrator", "game_character",
        ],
        "pace": ["slow", "medium_slow", "medium", "medium_fast", "fast"],
        "energy": ["low", "medium_low", "medium", "medium_high", "high"],
        "tone": [
            "calm", "intimate", "warm", "cool", "playful", "serious",
            "gentle", "wry",
        ],
        "proximity": ["close_mic", "mid_distance", "far_field", "far_mic"],
        "room": ["dry_studio", "treated_room", "natural_room"],
        "avoid": ["childlike", "shouting", "robotic", "overly_bright", "monotone"],
    },
}
_LIST_FIELDS = {"timbre", "articulation", "tone", "avoid"}


@pytest.mark.parametrize("language", ["zh", "en"])
def test_btts_vocab_tokens_have_no_unknown_vocab(language):
    vocab = _BTTS_VOCAB[language]
    for field, tokens in vocab.items():
        for token in tokens:
            document = example_document("demo-voice", language=language)
            value = [token] if field in _LIST_FIELDS else token
            if field in {"pace", "energy", "tone"}:
                document["delivery_baseline"][field] = value
            elif field in {"proximity", "room"}:
                document["scene"][field] = value
            elif field == "avoid":
                document["avoid"] = [token]
            else:
                document["identity"][field] = value
            compiled = compile_design(normalize_document(document))
            unknown = [
                warning for warning in compiled.warnings
                if warning["code"] == "unknown-vocab"
            ]
            assert not unknown, (language, field, token, unknown)


def test_design_lint_block_and_parse_error_exit_codes(tmp_path):
    blocked = tmp_path / "blocked.yaml"
    _write(
        blocked,
        dump_design(
            {
                "schema_version": 1,
                "name": "demo-voice",
                "language": "zh",
                "template": "qtts/nope/v1",
                "covers": ["你好。"],
                "description": "测试声音。",
            }
        ),
    )
    assert main(["design", "lint", str(blocked)]) == 1

    broken = _write(tmp_path / "broken.yaml", "name: [unterminated\n")
    assert main(["design", "lint", str(broken)]) == 4


def test_design_lint_strict_warnings_exit_one(tmp_path):
    path = _write(
        tmp_path / "warn.yaml",
        dump_design(_document(covers=["啊" * 150])),
    )
    assert main(["design", "lint", str(path)]) == 0
    assert main(["design", "lint", str(path), "--strict"]) == 1


# --------------------------------------------------------------------------- #
# Matrix
# --------------------------------------------------------------------------- #
def test_build_matrix_is_deterministic_and_bounded():
    document = _document()
    first = build_matrix(document)
    second = build_matrix(document)
    assert first == second
    assert len(first["candidates"]) == 4
    assert first["seeds"] == sorted(first["seeds"])
    assert {item["take_id"] for item in first["candidates"]} == {
        f"s{seed}-c0-{profile}"
        for seed in first["seeds"]
        for profile in ("default", "stable")
    }
    other = build_matrix(document, rng_seed=7)
    assert other == build_matrix(document, rng_seed=7)
    assert other["seeds"] != first["seeds"]


def test_build_matrix_wide_and_explicit_options():
    document = _document()
    wide = build_matrix(document, wide=True)
    assert len(wide["candidates"]) == 8
    assert {item["cover_index"] for item in wide["candidates"]} == {0, 1}
    narrow = build_matrix(document, sampling=["expressive"], seeds=[3])
    assert len(narrow["candidates"]) == 1
    assert narrow["candidates"][0]["sampling_profile"] == "expressive"


def test_build_matrix_rejects_bad_options():
    document = _document()
    with pytest.raises(UsageError, match="two covers"):
        build_matrix(_document(covers=["你好。"]), wide=True, seeds=[3])
    with pytest.raises(UsageError, match="at most two"):
        build_matrix(_document(covers=["一", "二", "三"]), seeds=[3])
    with pytest.raises(UsageError, match="unknown sampling"):
        build_matrix(document, sampling=["wild"])
    with pytest.raises(UsageError, match="positive"):
        build_matrix(document, seeds=[0])
    with pytest.raises(UsageError, match="duplicate values"):
        build_matrix(document, seeds=[3, 3])
    with pytest.raises(UsageError, match="duplicate profiles"):
        build_matrix(document, sampling=["default", "default"])
    with pytest.raises(UsageError, match=str(MAX_CANDIDATES)):
        build_matrix(document, seeds=[2, 3, 5, 7, 11], wide=True)


def test_build_matrix_copies_sampling_maps():
    matrix = build_matrix(_document(), seeds=[3], sampling=["default"])
    matrix["candidates"][0]["sampling"]["temperature"] = 0.1
    assert SAMPLING_PROFILES["default"]["temperature"] == 0.9


# --------------------------------------------------------------------------- #
# Run directory + lock
# --------------------------------------------------------------------------- #
def _prepare(tmp_path, name="run"):
    document = _document()
    return prepare_run(
        tmp_path / name,
        document=document,
        raw_text=dump_design(document),
        compiled=compile_design(document),
        matrix=build_matrix(document),
        source_path="design.yaml",
    )


def test_prepare_run_writes_manifest_and_resumes(tmp_path):
    manifest = _prepare(tmp_path)
    run_dir = tmp_path / "run"
    for name in ("run.json", "design.yaml", "design.compiled.json"):
        assert (run_dir / name).is_file()
    assert (run_dir / "candidates").is_dir()
    assert (run_dir / "report").is_dir()
    assert manifest["kind"] == "design_run"
    assert all(item["status"] == "pending" for item in manifest["candidates"])
    assert manifest["source"]["path"] == "design.yaml"
    assert len(manifest["source"]["sha256"]) == 64

    with pytest.raises(UsageError, match="already exists"):
        _prepare(tmp_path)
    document = _document()
    resumed = prepare_run(
        run_dir,
        document=document,
        raw_text=dump_design(document),
        compiled=compile_design(document),
        matrix=build_matrix(document),
        resume=True,
    )
    assert resumed["run_id"] == manifest["run_id"]


def test_prepare_run_resume_with_template_alias_uses_canonical(tmp_path):
    document = example_document("demo-voice", language="zh")
    document["template"] = "btts/zh-late-night-radio/v1"
    raw_text = dump_design(document)
    manifest = prepare_run(
        tmp_path / "run",
        document=document,
        raw_text=raw_text,
        compiled=compile_design(document),
        matrix=build_matrix(document),
    )
    assert manifest["design"]["template"] == ZH_TEMPLATE

    resumed = prepare_run(
        tmp_path / "run",
        document=document,
        raw_text=raw_text,
        compiled=compile_design(document),
        matrix=build_matrix(document),
        resume=True,
    )
    assert resumed["run_id"] == manifest["run_id"]
    assert resumed["design"]["template"] == ZH_TEMPLATE


def test_prepare_run_resume_mismatch(tmp_path):
    manifest = _prepare(tmp_path)
    changed = _document(instruction_override="完全不同的描述。")
    with pytest.raises(UsageError, match="resume mismatch"):
        prepare_run(
            tmp_path / "run",
            document=changed,
            raw_text=dump_design(changed),
            compiled=compile_design(changed),
            matrix=build_matrix(changed),
            resume=True,
        )
    assert manifest["run_id"]


def test_prepare_run_resume_rejects_name_and_source_changes(tmp_path):
    _prepare(tmp_path)
    renamed = _document(name="other-voice")
    with pytest.raises(UsageError, match="resume mismatch"):
        prepare_run(
            tmp_path / "run",
            document=renamed,
            raw_text=dump_design(renamed),
            compiled=compile_design(renamed),
            matrix=build_matrix(renamed),
            resume=True,
        )

    foreign = tmp_path / "foreign"
    foreign.mkdir(exist_ok=True)
    (foreign / "notes.txt").write_text("x")
    document = _document()
    with pytest.raises(UsageError, match="no run.json"):
        prepare_run(
            foreign,
            document=document,
            raw_text=dump_design(document),
            compiled=compile_design(document),
            matrix=build_matrix(document),
            resume=True,
        )


def test_prepare_run_refuses_nonempty_foreign_directory(tmp_path):
    target = tmp_path / "busy"
    target.mkdir()
    (target / "notes.txt").write_text("x")
    from qtts_cli.design_compile import compile_design as compile_again

    document = _document()
    with pytest.raises(UsageError, match="not empty"):
        prepare_run(
            target,
            document=document,
            raw_text=dump_design(document),
            compiled=compile_again(document),
            matrix=build_matrix(document),
        )


def test_run_lock_is_exclusive(tmp_path):
    with RunLock(tmp_path):
        with pytest.raises(UsageError, match="locked"):
            with RunLock(tmp_path):
                pass
    with RunLock(tmp_path):
        pass


def test_run_lock_uses_msvcrt_on_windows(tmp_path, monkeypatch):
    calls = []

    class FakeMsvcrt:
        LK_NBLCK = 2
        LK_UNLCK = 0
        fail = False

        @classmethod
        def locking(cls, fd, mode, count):
            calls.append((fd, mode, count))
            if cls.fail:
                raise OSError("busy")

    monkeypatch.setitem(sys.modules, "msvcrt", FakeMsvcrt)
    monkeypatch.setattr(sys, "platform", "win32")
    with RunLock(tmp_path):
        pass
    assert [(mode, count) for _, mode, count in calls] == [(2, 1), (0, 1)]

    calls.clear()
    FakeMsvcrt.fail = True
    with pytest.raises(UsageError, match="locked"):
        with RunLock(tmp_path):
            pass
    assert len(calls) == 1
    assert calls[0][1:] == (2, 1)


def test_matrix_signature_tracks_options():
    document = _document()
    one = matrix_signature(build_matrix(document, seeds=[3]))
    two = matrix_signature(build_matrix(document, seeds=[5]))
    assert one != two


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def _run_with_candidates(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    (run_dir / "candidates").mkdir(parents=True)
    sf.write(run_dir / "candidates" / "good.wav", _tone(), SR)
    sf.write(run_dir / "candidates" / "silent.wav", np.zeros(SR * 2, dtype=np.float32), SR)
    (run_dir / "candidates" / "good.recipe.json").write_text(
        json.dumps({"finish_reason": "stop", "text": "Hello there."})
    )
    (run_dir / "candidates" / "silent.recipe.json").write_text(
        json.dumps({"finish_reason": "stop", "text": "Hello there."})
    )
    yaml_text = "name: demo-voice\n"
    (run_dir / "design.yaml").write_text(yaml_text)
    manifest = {
        "schema_version": 1,
        "kind": "design_run",
        "run_id": "abcdef0123456789",
        "source": {"path": "design.yaml", "sha256": sha256_text(yaml_text)},
        "design": {
            "name": "demo-voice",
            "language": "zh",
            "template": ZH_TEMPLATE,
            "template_sha256": "t" * 64,
            "compiler_version": "1",
            "structured": {"name": "demo-voice"},
            "rendered_instruction": "测试。",
            "rendered_sha256": sha256_text("测试。"),
        },
        "model": {},
        "matrix": {"seeds": [2], "covers": ["你好。"], "sampling_profiles": ["default"]},
        "candidates": [
            {
                "take_id": "s2-c0-default",
                "seed": 2,
                "cover_index": 0,
                "status": "ok",
                "wav": "candidates/good.wav",
                "recipe": "candidates/good.recipe.json",
                "sha256": sha256_file(run_dir / "candidates" / "good.wav"),
            },
            {
                "take_id": "s3-c0-default",
                "seed": 3,
                "cover_index": 0,
                "status": "ok",
                "wav": "candidates/silent.wav",
                "recipe": "candidates/silent.recipe.json",
                "sha256": sha256_file(run_dir / "candidates" / "silent.wav"),
            },
            {
                "take_id": "x<script>",
                "seed": 5,
                "cover_index": 0,
                "status": "failed",
                "wav": "candidates/missing.wav",
                "recipe": None,
            },
        ],
    }
    (run_dir / "run.json").write_text(json.dumps(manifest))
    return run_dir


def test_report_ranks_complete_candidates_first(tmp_path):
    run_dir = _run_with_candidates(tmp_path)
    report = build_report(run_dir, write_reel=False)
    order = [item["take_id"] for item in report["candidates"]]
    assert order == ["s2-c0-default", "s3-c0-default", "x<script>"]
    assert report["summary"]["best_take_id"] == "s2-c0-default"
    assert [item["take_id"] for item in report["candidates"] if item["recommended"]] == [
        "s2-c0-default"
    ]
    assert report["candidates"][1]["failure_count"] == 1
    assert report["candidates"][2]["failure_count"] == 1
    assert report["candidates"][2]["reasons"] == ["generation_failed"]
    assert report["candidates"][2]["degraded"] is True
    assert report["exit_code"] == 0


def test_report_writes_csv_markdown_and_html(tmp_path):
    run_dir = _run_with_candidates(tmp_path)
    build_report(run_dir, write_reel=False)
    report_dir = run_dir / "report"
    for name in ("report.json", "metrics.csv", "report.md", "audition.html"):
        assert (report_dir / name).is_file()

    rows = (report_dir / "metrics.csv").read_text().strip().splitlines()
    assert rows[0].split(",") == CSV_COLUMNS
    assert len(rows) == 4
    assert "s2-c0-default" in rows[1]

    page = (report_dir / "audition.html").read_text()
    assert "x&lt;script&gt;" in page
    assert "x<script>" not in page
    assert "blind" in page
    assert "qtts design freeze" in page

    markdown = (report_dir / "report.md").read_text()
    assert "best take: `s2-c0-default`" in markdown


def _rewrite_manifest(run_dir: Path, mutate) -> None:
    path = run_dir / "run.json"
    manifest = json.loads(path.read_text())
    mutate(manifest)
    path.write_text(json.dumps(manifest))


def test_report_keeps_going_on_corrupt_wav(tmp_path):
    run_dir = _run_with_candidates(tmp_path)
    (run_dir / "candidates" / "good.wav").write_bytes(b"not a wav")
    report = build_report(run_dir, write_reel=False)
    item = next(c for c in report["candidates"] if c["take_id"] == "s2-c0-default")
    assert item["status"] == "failed"
    assert item["error"].startswith("audio_unreadable")
    assert item["failure_count"] == 1
    assert (run_dir / "report" / "report.json").is_file()
    assert report["summary"]["count"] == 3


def test_report_never_recommends_failed_manifest_status(tmp_path):
    run_dir = _run_with_candidates(tmp_path)

    def mutate(manifest):
        manifest["candidates"][0]["status"] = "failed"
        manifest["candidates"][0]["error"] = "oom"

    _rewrite_manifest(run_dir, mutate)
    report = build_report(run_dir, write_reel=False)
    failed = next(c for c in report["candidates"] if c["take_id"] == "s2-c0-default")
    assert failed["status"] == "failed"
    assert failed["recommended"] is False
    assert failed["completion"]["status"] == "failed"
    assert report["summary"]["best_take_id"] != "s2-c0-default"


def test_report_rejects_paths_outside_run_dir(tmp_path):
    run_dir = _run_with_candidates(tmp_path)
    outside = tmp_path / "outside.wav"
    sf.write(outside, _tone(), SR)

    def mutate(manifest):
        manifest["candidates"][0]["wav"] = "../../outside.wav"

    _rewrite_manifest(run_dir, mutate)
    report = build_report(run_dir, write_reel=False)
    item = next(c for c in report["candidates"] if c["take_id"] == "s2-c0-default")
    assert item["status"] == "failed"
    assert item["audio"]["path"] is None
    assert "audio_outside_run" in item["reasons"]
    page = (run_dir / "report" / "audition.html").read_text()
    assert "outside.wav" not in page


def test_report_reel_warning_is_written_everywhere(tmp_path):
    run_dir = _run_with_candidates(tmp_path)
    sf.write(run_dir / "candidates" / "silent.wav", np.zeros(16000 * 2, dtype=np.float32), 16000)
    build_report(run_dir, write_reel=True, gap=0.2)
    report = json.loads((run_dir / "report" / "report.json").read_text())
    assert any("reel skipped" in warning for warning in report["warnings"])
    assert "reel skipped" in (run_dir / "report" / "report.md").read_text()
    assert "reel skipped" in (run_dir / "report" / "audition.html").read_text()


def test_report_html_blind_mode_and_escaping(tmp_path):
    run_dir = _run_with_candidates(tmp_path)
    build_report(run_dir, write_reel=False)
    page = (run_dir / "report" / "audition.html").read_text()
    assert "body.blind" in page
    assert "data-take=" in page
    assert "x&lt;script&gt;" in page


def test_report_is_deterministic(tmp_path):
    run_dir = _run_with_candidates(tmp_path)
    build_report(run_dir, write_reel=False)
    first = (run_dir / "report" / "report.json").read_bytes()
    build_report(run_dir, write_reel=False)
    second = (run_dir / "report" / "report.json").read_bytes()
    assert first == second


def test_report_writes_reel(tmp_path):
    run_dir = _run_with_candidates(tmp_path)
    report = build_report(run_dir, write_reel=True, gap=0.5)
    reel = run_dir / "report" / "all_candidates.wav"
    assert reel.is_file()
    data, sr = sf.read(reel)
    assert sr == SR
    # good + silent + one 0.5 s gap
    assert data.shape[0] == pytest.approx(4.5 * SR, rel=0.01)
    assert report["summary"]["complete"] == 2


def test_report_fail_on_degraded_and_validation(tmp_path):
    run_dir = _run_with_candidates(tmp_path)
    report = build_report(run_dir, write_reel=False, fail_on_degraded=True)
    assert report["degraded"] is False
    assert report["exit_code"] == 0

    degraded_dir = _run_with_candidates(tmp_path / "degraded")
    manifest = json.loads((degraded_dir / "run.json").read_text())
    manifest["candidates"] = [manifest["candidates"][2]]
    (degraded_dir / "run.json").write_text(json.dumps(manifest))
    report = build_report(degraded_dir, write_reel=False, fail_on_degraded=True)
    assert report["degraded"] is True
    assert report["exit_code"] == 5

    with pytest.raises(InputError, match="not a design run"):
        build_report(tmp_path / "missing")
    with pytest.raises(InputError, match="gap"):
        build_report(run_dir, gap=99, write_reel=False)


def test_report_require_full_metrics_exits_five(tmp_path):
    run_dir = tmp_path / "short"
    (run_dir / "candidates").mkdir(parents=True)
    sf.write(run_dir / "candidates" / "short.wav", _tone(0.2), SR)
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "run_id": "short-run",
                "design": {"name": "demo", "template": ZH_TEMPLATE},
                "candidates": [
                    {
                        "take_id": "s2-c0-default",
                        "status": "ok",
                        "wav": "candidates/short.wav",
                        "recipe": None,
                    }
                ],
            }
        )
    )
    report = build_report(
        run_dir, metrics="full", write_reel=False, require_full=True
    )
    assert report["exit_code"] == 5
    assert "some full metrics are unavailable" in report["warnings"]


# --------------------------------------------------------------------------- #
# CLI: design explore / report
# --------------------------------------------------------------------------- #
def test_design_explore_dry_run(tmp_path, capsys):
    design = _write(tmp_path / "demo-voice.yaml", dump_design(_document()))
    out = tmp_path / "run"
    code = main(
        ["design", "explore", str(design), "--out", str(out), "--dry-run",
         "--seeds", "3,5", "--sampling", "stable"]
    )
    assert code == 0
    assert not out.exists()
    text = capsys.readouterr().out
    assert "dry-run: 2 candidate(s)" in text
    assert "s3-c0-stable" in text
    assert "s5-c0-stable" in text


def test_design_explore_validation_exit_codes(tmp_path):
    design = _write(tmp_path / "demo-voice.yaml", dump_design(_document()))
    out = tmp_path / "run"
    assert main(
        ["design", "explore", str(design), "--out", str(out), "--dry-run",
         "--seeds", "bad"]
    ) == 2
    blocked = _write(
        tmp_path / "blocked.yaml",
        dump_design({**_document(), "template": "qtts/nope/v1"}),
    )
    assert main(["design", "explore", str(blocked), "--out", str(out), "--dry-run"]) == 2
    (out / "foreign").parent.mkdir(parents=True, exist_ok=True)
    (out / "foreign.txt").write_text("x")
    assert main(["design", "explore", str(design), "--out", str(out), "--dry-run"]) == 2


class _FakeModel:
    model_id = "fake-design"
    revision = "test"
    fingerprint = "fingerprint"
    name = "fake-design"
    kind = "design"
    path = Path("/nonexistent/fake-design")


class _FakeEngine:
    calls = 0

    def __init__(self, cfg, model):
        self.dtype_name = "fp32"
        self.device = "cpu"

    def load(self):
        return self

    def generate(self, **kwargs):
        type(self).calls += 1
        return _tone(0.8), SR, {"generate_ms": 1.0}


def _patch_fake_model(monkeypatch):
    model = _FakeModel()
    monkeypatch.setattr("qtts_cli.catalog.scan", lambda model_dir: [model])
    monkeypatch.setattr("qtts_cli.catalog.select", lambda models, kind: models[0])
    monkeypatch.setattr("qtts_cli.catalog.find_model", lambda models, name: models[0])
    monkeypatch.setattr("qtts_cli.capabilities.validate_capabilities", lambda *a, **k: None)
    monkeypatch.setattr("qtts_cli.tts_engine.TtsEngine", _FakeEngine)
    _FakeEngine.calls = 0
    return model


def test_design_explore_generates_and_resumes(tmp_path, monkeypatch):
    _patch_fake_model(monkeypatch)
    design = _write(tmp_path / "demo-voice.yaml", dump_design(_document()))
    out = tmp_path / "run"
    args = ["design", "explore", str(design), "--out", str(out),
            "--seeds", "3", "--sampling", "stable", "--max-new-tokens", "128"]
    assert main(args) == 0
    manifest = json.loads((out / "run.json").read_text())
    assert len(manifest["candidates"]) == 1
    record = manifest["candidates"][0]
    assert record["status"] == "ok"
    assert record["completion_status"] in ("unknown", "complete")
    assert (out / record["wav"]).is_file()
    recipe = json.loads((out / record["recipe"]).read_text())
    assert recipe["seed"] == 3
    assert recipe["instruct_sha256"] == compile_design(_document()).rendered_sha256
    assert recipe["max_new_tokens"] == 128
    assert _FakeEngine.calls == 1

    assert main(args + ["--resume"]) == 0
    assert _FakeEngine.calls == 1
    resumed = json.loads((out / "run.json").read_text())
    assert resumed["run_id"] == manifest["run_id"]

    # --retry-failed never touches candidates that already succeeded.
    assert main(args + ["--resume", "--retry-failed"]) == 0
    assert _FakeEngine.calls == 1

    def fail_first(current):
        current["candidates"][0].update(status="failed", error="boom")

    _rewrite_manifest(out, fail_first)
    assert main(args + ["--resume"]) == 0
    assert _FakeEngine.calls == 1
    assert json.loads((out / "run.json").read_text())["candidates"][0]["status"] == "failed"
    assert main(args + ["--resume", "--retry-failed"]) == 0
    assert _FakeEngine.calls == 2
    record = json.loads((out / "run.json").read_text())["candidates"][0]
    assert record["status"] == "ok"
    assert record["wav"] == "candidates/s3-c0-stable.wav"


def test_design_explore_resume_ignores_manifest_paths(tmp_path, monkeypatch):
    _patch_fake_model(monkeypatch)
    design = _write(tmp_path / "demo-voice.yaml", dump_design(_document()))
    out = tmp_path / "run"
    args = ["design", "explore", str(design), "--out", str(out),
            "--seeds", "3", "--sampling", "stable"]
    assert main(args) == 0
    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"keep me")

    def escape(current):
        current["candidates"][0].update(
            wav="../../outside.wav",
            recipe="../../outside.recipe.json",
            sha256=None,
        )

    _rewrite_manifest(out, escape)
    assert main(args + ["--resume"]) == 0
    assert outside.read_bytes() == b"keep me"
    assert not (tmp_path / "outside.recipe.json").exists()
    record = json.loads((out / "run.json").read_text())["candidates"][0]
    assert record["wav"] == "candidates/s3-c0-stable.wav"
    assert record["recipe"] == "candidates/s3-c0-stable.recipe.json"
    assert (out / record["wav"]).is_file()


def test_design_prune_keeps_top_n_and_reports_pruned_state(tmp_path, monkeypatch):
    _patch_fake_model(monkeypatch)
    design = _write(tmp_path / "demo-voice.yaml", dump_design(_document()))
    out = tmp_path / "run"
    args = ["design", "explore", str(design), "--out", str(out),
            "--seeds", "3,5", "--sampling", "stable"]
    assert main(args) == 0
    for take in ("s3-c0-stable", "s5-c0-stable"):
        assert (out / "candidates" / f"{take}.wav").is_file()

    # Default is a preview and writes nothing.
    assert main(["design", "prune", str(out), "--keep", "1"]) == 0
    assert (out / "candidates" / "s5-c0-stable.wav").is_file()

    assert main(["design", "prune", str(out), "--keep", "1", "--yes"]) == 0
    manifest = json.loads((out / "run.json").read_text())
    by_id = {record["take_id"]: record for record in manifest["candidates"]}
    assert by_id["s3-c0-stable"].get("pruned") is not True
    assert by_id["s5-c0-stable"]["pruned"] is True
    assert by_id["s5-c0-stable"]["wav"] is None
    assert by_id["s5-c0-stable"]["recipe"] is None
    assert not (out / "candidates" / "s5-c0-stable.wav").exists()
    assert not (out / "candidates" / "s5-c0-stable.recipe.json").exists()
    assert (out / "candidates" / "s3-c0-stable.wav").is_file()
    assert not (out / ".prune").exists()

    report = build_report(out, write_reel=False)
    assert report["summary"]["pruned"] == 1
    assert report["summary"]["analyzed_count"] == 1
    pruned = next(item for item in report["candidates"] if item["take_id"] == "s5-c0-stable")
    assert pruned["status"] == "pruned"
    assert pruned["rank"] is None
    assert pruned["recommended"] is False
    assert pruned["failure_count"] == 0
    assert [item["take_id"] for item in report["candidates"]] == [
        "s3-c0-stable",
        "s5-c0-stable",
    ]
    assert report["candidates"][0]["rank"] == 1

    calls = _FakeEngine.calls
    assert main(args + ["--resume"]) == 0
    assert _FakeEngine.calls == calls


def test_design_prune_rejects_symlinked_candidate(tmp_path, monkeypatch):
    _patch_fake_model(monkeypatch)
    design = _write(tmp_path / "demo-voice.yaml", dump_design(_document()))
    out = tmp_path / "run"
    args = ["design", "explore", str(design), "--out", str(out),
            "--seeds", "3,5", "--sampling", "stable"]
    assert main(args) == 0
    target = out / "candidates" / "s3-c0-stable.wav"
    link = out / "candidates" / "s5-c0-stable.wav"
    link.unlink()
    link.symlink_to(target)
    assert main(["design", "prune", str(out), "--keep", "1", "--yes"]) == 4
    assert link.is_symlink()
    assert target.is_file()


def test_design_prune_recovers_interrupted_staging(tmp_path, monkeypatch):
    _patch_fake_model(monkeypatch)
    design = _write(tmp_path / "demo-voice.yaml", dump_design(_document()))
    out = tmp_path / "run"
    args = ["design", "explore", str(design), "--out", str(out),
            "--seeds", "3,5", "--sampling", "stable"]
    assert main(args) == 0
    # Simulate a crash after moving a file but before run.json was rewritten.
    staged = out / ".prune" / "deadbeef"
    (staged / "candidates").mkdir(parents=True)
    src = out / "candidates" / "s5-c0-stable.wav"
    dst = staged / "candidates" / "s5-c0-stable.wav"
    import os as _os
    _os.replace(src, dst)
    assert not src.exists()

    assert main(["design", "prune", str(out), "--keep", "2", "--yes"]) == 0
    assert src.is_file()
    assert not staged.exists()


def test_design_prune_apply_honors_run_lock(tmp_path, monkeypatch):
    from qtts_cli.design_run import RunLock

    _patch_fake_model(monkeypatch)
    design = _write(tmp_path / "demo-voice.yaml", dump_design(_document()))
    out = tmp_path / "run"
    assert main([
        "design", "explore", str(design), "--out", str(out),
        "--seeds", "3,5", "--sampling", "stable",
    ]) == 0

    with RunLock(out):
        assert main(["design", "prune", str(out), "--keep", "1", "--yes"]) == 2
        assert main(["design", "prune", str(out), "--keep", "1"]) == 0
    assert (out / "candidates" / "s5-c0-stable.wav").is_file()
    assert (out / "candidates" / "s3-c0-stable.wav").is_file()


def test_design_prune_dry_run_does_not_recover(tmp_path, monkeypatch):
    _patch_fake_model(monkeypatch)
    design = _write(tmp_path / "demo-voice.yaml", dump_design(_document()))
    out = tmp_path / "run"
    args = ["design", "explore", str(design), "--out", str(out),
            "--seeds", "3,5", "--sampling", "stable"]
    assert main(args) == 0
    staged = out / ".prune" / "deadbeef"
    (staged / "candidates").mkdir(parents=True)
    src = out / "candidates" / "s5-c0-stable.wav"
    dst = staged / "candidates" / "s5-c0-stable.wav"
    src.rename(dst)

    assert main(["design", "prune", str(out), "--keep", "2"]) == 0
    assert dst.is_file()
    assert not src.exists()
    assert staged.is_dir()


def test_design_prune_recovery_ignores_paths_outside_candidates(tmp_path):
    from qtts_cli.design_prune import _recover

    run_dir = _run_with_candidates(tmp_path)
    staged = run_dir / ".prune" / "deadbeef"
    (staged / "report").mkdir(parents=True)
    (staged / "report" / "pwn.txt").write_text("x")
    (staged / "candidates").mkdir()
    (staged / "candidates" / "silent.wav").symlink_to(
        run_dir / "candidates" / "good.wav"
    )

    _recover(run_dir)
    assert not (run_dir / "report" / "pwn.txt").exists()
    assert not staged.exists()
    assert (run_dir / "candidates" / "good.wav").is_file()


def test_design_prune_recovery_rejects_symlinked_candidates(tmp_path):
    from qtts_cli.design_prune import _recover

    run_dir = _run_with_candidates(tmp_path)
    staged = run_dir / ".prune" / "deadbeef"
    (staged / "candidates").mkdir(parents=True)
    (staged / "candidates" / "good.wav").write_bytes(b"staged")
    outside = tmp_path / "outside-candidates"
    outside.mkdir()
    (run_dir / "candidates").rename(outside / "candidates")
    (run_dir / "candidates").symlink_to(outside / "candidates")

    with pytest.raises(InputError, match="candidates"):
        _recover(run_dir)
    assert (staged / "candidates" / "good.wav").read_bytes() == b"staged"


def test_design_prune_rejects_symlinked_directories(tmp_path, monkeypatch):
    _patch_fake_model(monkeypatch)
    design = _write(tmp_path / "demo-voice.yaml", dump_design(_document()))
    out = tmp_path / "run"
    args = ["design", "explore", str(design), "--out", str(out),
            "--seeds", "3,5", "--sampling", "stable"]
    assert main(args) == 0

    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep.txt").write_text("keep")
    (out / ".prune").symlink_to(outside)
    assert main(["design", "prune", str(out), "--keep", "1", "--yes"]) == 4
    assert (outside / "keep.txt").is_file()
    (out / ".prune").unlink()

    moved = tmp_path / "moved-candidates"
    (out / "candidates").rename(moved)
    (out / "candidates").symlink_to(moved)
    assert main(["design", "prune", str(out), "--keep", "1", "--yes"]) == 4
    assert (moved / "s5-c0-stable.wav").is_file()


def test_report_rejects_symlinked_report_directory(tmp_path):
    run_dir = _run_with_candidates(tmp_path)
    outside = tmp_path / "outside-report"
    outside.mkdir()
    (run_dir / "report").symlink_to(outside)
    with pytest.raises(InputError, match="symlink"):
        build_report(run_dir, write_reel=False)
    assert not (outside / "report.json").exists()


def test_design_freeze_as_design(tmp_path, capsys):
    run_dir = _run_with_candidates(tmp_path)
    data_dir = tmp_path / "data"
    base = ["design", "freeze", str(run_dir), "--take", "s2-c0-default",
            "--name", "frozen", "--as", "design",
            "--data-dir", str(data_dir)]
    assert main(base) == 0
    profile = data_dir / "voices" / "frozen" / "voice.json"
    meta = json.loads(profile.read_text())
    assert meta["kind"] == "design"
    assert meta["source"]["description"] == "测试。"
    assert meta["source"]["design"]["rendered_instruction"] == "测试。"
    assert meta["source"]["origin"] == "design-freeze"
    assert meta["source"]["take_id"] == "s2-c0-default"
    assert "froze take s2-c0-default" in capsys.readouterr().out

    assert main(base) == 2
    assert main(base + ["--force"]) == 0


def test_design_freeze_rejects_casefold_collision(tmp_path, monkeypatch, capsys):
    from qtts_cli.voice_store import VoiceStore

    run_dir = _run_with_candidates(tmp_path)
    monkeypatch.setattr(
        VoiceStore, "existing_names", lambda self: {"frozen": "Frozen"}
    )
    assert main([
        "design", "freeze", str(run_dir), "--take", "s2-c0-default",
        "--name", "frozen", "--as", "design",
        "--data-dir", str(tmp_path / "data"),
    ]) == 2
    assert "case-insensitively" in capsys.readouterr().err


def test_design_freeze_rejects_failed_and_unverifiable_takes(tmp_path):
    run_dir = _run_with_candidates(tmp_path)
    base = ["design", "freeze", str(run_dir), "--name", "frozen",
            "--as", "design",
            "--data-dir", str(tmp_path / "data")]

    assert main(base + ["--take", "x<script>"]) == 5

    _rewrite_manifest(
        run_dir,
        lambda manifest: manifest["candidates"][0].update(sha256=None),
    )
    assert main(base + ["--take", "s2-c0-default"]) == 4


def test_design_freeze_rejects_tampered_design_yaml(tmp_path):
    run_dir = _run_with_candidates(tmp_path)
    (run_dir / "design.yaml").write_text("name: tampered\n")
    assert main(
        ["design", "freeze", str(run_dir), "--take", "s2-c0-default",
         "--name", "frozen", "--as", "design",
         "--data-dir", str(tmp_path / "data")]
    ) == 4
    (run_dir / "design.yaml").unlink()
    assert main(
        ["design", "freeze", str(run_dir), "--take", "s2-c0-default",
         "--name", "frozen", "--as", "design",
         "--data-dir", str(tmp_path / "data")]
    ) == 4


def test_say_design_dry_run_and_conflicts(tmp_path, capsys):
    design = _write(tmp_path / "demo-voice.yaml", dump_design(_document()))
    assert main(["say", "你好", "--design", str(design), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "深夜电台主持人" in out
    assert main(["say", "你好", "--design", str(design), "--voice", "builtin:Vivian",
                 "--dry-run"]) == 2
    assert main(["say", "你好", "--design", str(design), "--instruct", "quiet",
                 "--dry-run"]) == 2
    assert main(["say", "--stdin", "--jsonl", "--design", str(design),
                 "--out-dir", str(tmp_path / "jsonl")]) == 2
    assert main(["say", "--script", str(tmp_path / "script.yaml"),
                 "--design", str(design), "--dry-run"]) == 2


def test_design_report_cli(tmp_path):
    run_dir = _run_with_candidates(tmp_path)
    assert main(["design", "report", str(run_dir)]) == 0
    for name in ("report.json", "metrics.csv", "report.md", "audition.html",
                 "all_candidates.wav"):
        assert (run_dir / "report" / name).is_file()

    run_dir = _run_with_candidates(tmp_path / "noreel")
    assert main(["design", "report", str(run_dir), "--no-reel", "--no-html"]) == 0
    assert not (run_dir / "report" / "all_candidates.wav").exists()
    assert not (run_dir / "report" / "audition.html").exists()
    assert (run_dir / "report" / "metrics.csv").is_file()
