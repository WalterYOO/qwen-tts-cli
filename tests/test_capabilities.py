"""Capability matrix and voice-reference validation (PRD §3.1, §11.4/10)."""

from __future__ import annotations

import pytest

from qtts_cli.capabilities import parse_voice_ref, validate_capabilities
from qtts_cli.catalog import require_speaker, scan, select
from qtts_cli.errors import CapabilityError, InputError


def _model(model_dir, kind):
    return select(scan(model_dir), kind)


def test_parse_voice_refs():
    assert parse_voice_ref("builtin:Vivian").mode == "builtin"
    assert parse_voice_ref("voice:alice").mode == "clone"
    assert parse_voice_ref("design:温暖的女声").mode == "design"


@pytest.mark.parametrize("bad", ["", "Vivian", "clone:alice", "builtin:", "voice:../etc"])
def test_parse_voice_ref_rejects(bad):
    with pytest.raises(InputError):
        parse_voice_ref(bad)


def test_builtin_speakers_are_enumerated_from_model(model_dir):
    model = _model(model_dir, "builtin")
    assert len(model.speakers) == 9
    assert "Vivian" in model.speakers
    assert require_speaker(model, "vivian") == "Vivian"
    with pytest.raises(InputError):
        require_speaker(model, "nobody")


def test_06b_custom_voice_rejects_instruct(small_model_dir):
    model = _model(small_model_dir, "builtin")
    assert model.supports_instruct is False
    with pytest.raises(CapabilityError, match="does not support --instruct"):
        validate_capabilities(model, mode="builtin", instruct="温柔")


def test_17b_custom_voice_accepts_instruct(model_dir):
    model = _model(model_dir, "builtin")
    validate_capabilities(model, mode="builtin", instruct="温柔")


def test_base_rejects_instruct(model_dir):
    model = _model(model_dir, "clone")
    with pytest.raises(CapabilityError):
        validate_capabilities(model, mode="clone", instruct="x", ref_text="hello", icl=True)


def test_icl_requires_ref_text(model_dir):
    model = _model(model_dir, "clone")
    with pytest.raises(CapabilityError, match="ref-text"):
        validate_capabilities(model, mode="clone", icl=True, ref_text="")


def test_xvec_does_not_require_ref_text(model_dir):
    model = _model(model_dir, "clone")
    validate_capabilities(model, mode="clone", icl=False, ref_text=None)


def test_design_requires_description(model_dir):
    model = _model(model_dir, "design")
    with pytest.raises(CapabilityError):
        validate_capabilities(model, mode="design", description="")


def test_unsupported_language(model_dir):
    model = _model(model_dir, "builtin")
    with pytest.raises(CapabilityError, match="language"):
        validate_capabilities(model, mode="builtin", language="Klingon")


def test_mode_model_mismatch(model_dir):
    builtin = _model(model_dir, "builtin")
    with pytest.raises(CapabilityError):
        validate_capabilities(builtin, mode="clone")
