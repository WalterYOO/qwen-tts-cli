"""FR-P2-08: plugin discovery, validation and post-process ordering."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from qtts_cli import plugins
from qtts_cli.backends.base import BackendCapabilities
from qtts_cli.cli import main
from qtts_cli.errors import InputError, ModelError, UsageError
from qtts_cli.postprocess import process


def _entry_point(name, obj, *, distribution="demo-dist", version="1.0.0"):
    class EntryPoint:
        def __init__(self):
            self.name = name
            self.dist = SimpleNamespace(name=distribution, version=version)

        def load(self):
            if isinstance(obj, BaseException):
                raise obj
            return obj

    return EntryPoint()


def _patch_entry_points(monkeypatch, groups):
    calls = []

    def fake(group):
        calls.append(group)
        return list(groups.get(group, []))

    monkeypatch.setattr(plugins, "_entry_points", fake)
    return calls


def _caps(**overrides):
    values = {
        "modes": frozenset({"builtin"}),
        "instruct_modes": frozenset(),
        "sampling": frozenset(),
        "remote": True,
    }
    values.update(overrides)
    return BackendCapabilities(**values)


class BackendPlugin:
    api_version = 1
    name = "demo"

    def __init__(self, *, calls=None):
        self.calls = calls if calls is not None else []

    def capabilities(self):
        self.calls.append(("capabilities",))
        return _caps()

    def models(self, cfg, backend_model=None):
        self.calls.append(("models", backend_model))
        raise AssertionError("plugins list must not call models()")

    def create(self, cfg, model):
        raise AssertionError("plugins list must not call create()")


class PostPlugin:
    api_version = 1
    name = "post"

    def process(self, samples, sr, params):
        return samples, sr


def _run(tmp_path: Path, *args: str) -> int:
    return main(
        [
            *args,
            "--model-dir",
            str(tmp_path / "models"),
            "--data-dir",
            str(tmp_path / "data"),
            "--cache-dir",
            str(tmp_path / "cache"),
        ]
    )


def test_no_entry_points_lists_nothing_and_exits_zero(monkeypatch, tmp_path, capsys):
    _patch_entry_points(monkeypatch, {})
    rc = _run(tmp_path, "plugins", "list", "--json")
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["plugins"] == []
    assert payload["plugin_api_version"] == plugins.PLUGIN_API_VERSION


def test_entry_point_group_failure_is_a_warning_not_a_crash(
    monkeypatch, tmp_path, capsys
):
    def boom(group):
        raise RuntimeError("metadata exploded")

    monkeypatch.setattr(plugins, "_entry_points", boom)
    rc = _run(tmp_path, "plugins", "list", "--json")
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["plugins"] == []
    assert any("metadata exploded" in warning for warning in payload["warnings"])


def test_broken_api_mismatch_invalid_and_duplicate_are_listed(
    monkeypatch, tmp_path, capsys
):
    broken_obj = RuntimeError("import failed")
    unsupported = BackendPlugin()
    unsupported.name = "oldapi"
    unsupported.api_version = 99
    bool_api = PostPlugin()
    bool_api.name = "boolapi"
    bool_api.api_version = True
    upper = PostPlugin()
    upper.name = "Upper"
    duplicate_a = PostPlugin()
    duplicate_a.name = "same"
    duplicate_b = PostPlugin()
    duplicate_b.name = "same"
    no_process = SimpleNamespace(api_version=1, name="noproc")
    good = PostPlugin()

    _patch_entry_points(
        monkeypatch,
        {
            plugins.BACKEND_GROUP: [
                _entry_point("broken", broken_obj),
                _entry_point("old", unsupported),
            ],
            plugins.POSTPROCESS_GROUP: [
                _entry_point("bool", bool_api),
                _entry_point("upper", upper),
                _entry_point("same-a", duplicate_a, distribution="a"),
                _entry_point("same-b", duplicate_b, distribution="b"),
                _entry_point("noproc", no_process),
                _entry_point("post", good),
            ],
        },
    )
    rc = _run(tmp_path, "plugins", "list", "--json")
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    by_name = {item["name"]: item for item in payload["plugins"]}
    assert by_name["broken"]["status"] == "broken"
    assert by_name["oldapi"]["status"] == "unsupported_api"
    assert by_name["boolapi"]["status"] == "invalid"
    assert by_name["Upper"]["status"] == "invalid"
    assert by_name["same"]["status"] == "duplicate_name"
    assert by_name["noproc"]["status"] == "invalid"
    assert by_name["post"]["status"] == "ok"
    assert payload["ok"] is True


def test_strict_exits_three_without_crashing(monkeypatch, tmp_path, capsys):
    bad = PostPlugin()
    bad.name = "bad"
    bad.api_version = 2
    _patch_entry_points(
        monkeypatch, {plugins.POSTPROCESS_GROUP: [_entry_point("bad", bad)]}
    )
    rc = _run(tmp_path, "plugins", "list", "--strict", "--json")
    assert rc == 3
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["plugins"][0]["status"] == "unsupported_api"


def test_plugins_list_never_calls_models_or_create(monkeypatch, tmp_path, capsys):
    plugin = BackendPlugin(calls=[])
    _patch_entry_points(
        monkeypatch, {plugins.BACKEND_GROUP: [_entry_point("demo", plugin)]}
    )
    rc = _run(tmp_path, "plugins", "list", "--json")
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["plugins"][0]["status"] == "ok"
    assert plugin.calls == []  # no models(), create() or capabilities()


def test_plugin_name_is_required_and_builtins_are_reserved(
    monkeypatch, tmp_path, capsys
):
    nameless = SimpleNamespace(
        api_version=1,
        models=lambda cfg, backend_model=None: [],
        create=lambda cfg, model: None,
    )
    reserved = BackendPlugin()
    reserved.name = "transformers"
    _patch_entry_points(
        monkeypatch,
        {
            plugins.BACKEND_GROUP: [
                _entry_point("nameless", nameless),
                _entry_point("reserved", reserved),
            ]
        },
    )
    rc = _run(tmp_path, "plugins", "list", "--json")
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    by_name = {item["name"]: item for item in payload["plugins"]}
    assert by_name["nameless"]["status"] == "invalid"
    assert "declare a non-empty string 'name'" in by_name["nameless"]["reason"]
    assert by_name["transformers"]["status"] == "invalid"
    assert "reserved built-in backend name" in by_name["transformers"]["reason"]


def test_unknown_and_unusable_backends_use_usage_and_model_codes(
    monkeypatch, tmp_path, capsys
):
    bad = BackendPlugin()
    bad.api_version = 9
    _patch_entry_points(
        monkeypatch, {plugins.BACKEND_GROUP: [_entry_point("demo", bad)]}
    )
    assert main(
        [
            "say",
            "--backend",
            "missing",
            "--voice",
            "builtin:Vivian",
            "hi",
            "--model-dir",
            str(tmp_path / "models"),
            "--data-dir",
            str(tmp_path / "data"),
            "--cache-dir",
            str(tmp_path / "cache"),
        ]
    ) == 2
    capsys.readouterr()
    assert main(
        [
            "say",
            "--backend",
            "demo",
            "--voice",
            "builtin:Vivian",
            "hi",
            "--model-dir",
            str(tmp_path / "models"),
            "--data-dir",
            str(tmp_path / "data"),
            "--cache-dir",
            str(tmp_path / "cache"),
        ]
    ) == 3
    capsys.readouterr()


# --------------------------------------------------------------------------- #
# post-process hook
# --------------------------------------------------------------------------- #
def _tone(sr=16000, seconds=1.0):
    t = np.arange(int(sr * seconds), dtype=np.float64) / sr
    return (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)


def test_post_plugin_runs_after_resample_and_before_loudness_gain():
    calls = []

    class Recorder:
        name = "recorder"
        api_version = 1

        def process(self, samples, sr, params):
            calls.append((sr, samples.copy(), dict(params)))
            return samples, sr

    options = {
        "trim": False,
        "channels": "keep",
        "sample_rate": 8000,
        "lufs": -30.0,
    }
    output, out_sr, applied, _before, _after, _warnings = process(
        _tone(), 16000, options, plugin=Recorder()
    )
    assert out_sr == 8000
    assert len(calls) == 1
    hook_sr, hook_samples, _params = calls[0]
    assert hook_sr == 8000
    # The plugin sees the resampled, pre-gain signal; loudness gain is applied
    # afterwards, so the final peak differs from the hook input.
    assert np.max(np.abs(hook_samples)) > np.max(np.abs(output))
    assert applied["postprocess_plugin"]["name"] == "recorder"


@pytest.mark.parametrize(
    "result",
    [
        (np.zeros((0, 1), dtype=np.float32), 8000),
        (np.array([[0.0, 0.0]], dtype=np.float32), 8000),
        (np.zeros((10, 1, 1), dtype=np.float32), 8000),
        (np.array([np.nan], dtype=np.float32), 8000),
        (np.zeros(800, dtype=np.float32), 16000),
        (np.zeros(100000, dtype=np.float32), 8000),
        ("not audio", 8000),
    ],
)
def test_post_plugin_invalid_results_exit_four_without_output(result):
    class Bad:
        name = "bad"
        api_version = 1

        def process(self, samples, sr, params):
            return result

    with pytest.raises(InputError) as excinfo:
        process(
            _tone(sr=8000, seconds=1.0),
            8000,
            {"channels": "keep"},
            plugin=Bad(),
        )
    assert excinfo.value.code == 4


def test_post_plugin_exception_is_a_model_error():
    class Boom:
        name = "boom"
        api_version = 1

        def process(self, samples, sr, params):
            raise RuntimeError("dsp failed")

    with pytest.raises(ModelError) as excinfo:
        process(_tone(), 16000, {"channels": "keep"}, plugin=Boom())
    assert excinfo.value.code == 3


def test_post_plugin_cli_failure_writes_nothing(monkeypatch, tmp_path, capsys):
    import soundfile

    class Bad:
        api_version = 1
        name = "badpost"

        def process(self, samples, sr, params):
            return samples * np.nan, sr

    _patch_entry_points(
        monkeypatch, {plugins.POSTPROCESS_GROUP: [_entry_point("badpost", Bad())]}
    )
    source = tmp_path / "in.wav"
    soundfile.write(source, _tone(), 16000)
    out = tmp_path / "out.wav"
    rc = _run(
        tmp_path,
        "post",
        str(source),
        "--out",
        str(out),
        "--plugin",
        "badpost",
        "--json",
    )
    assert rc == 4
    assert not out.is_file()
    capsys.readouterr()


def test_post_plugin_argument_errors(monkeypatch, tmp_path, capsys):
    import soundfile

    _patch_entry_points(monkeypatch, {})
    source = tmp_path / "in.wav"
    soundfile.write(source, _tone(), 16000)
    common = [
        "post",
        str(source),
        "--out",
        str(tmp_path / "out.wav"),
    ]
    assert _run(tmp_path, *common, "--plugin", "missing") == 2
    capsys.readouterr()
    assert _run(tmp_path, *common, "--plugin-param", "a=1") == 2
    capsys.readouterr()
    assert (
        _run(
            tmp_path,
            *common,
            "--plugin",
            "missing",
            "--plugin-param",
            "a=1",
            "--plugin-param",
            "a=2",
        )
        == 2
    )
    capsys.readouterr()


@pytest.mark.parametrize(
    "param",
    ['x={"v":1e999}', "x=[1e999]", 'x={"v":[1e999]}'],
)
@pytest.mark.parametrize("mode", ["json", "report"])
def test_nested_non_finite_plugin_params_exit_two_before_writing(
    monkeypatch, tmp_path, capsys, param, mode
):
    import soundfile

    _patch_entry_points(monkeypatch, {})
    source = tmp_path / "in.wav"
    soundfile.write(source, _tone(), 16000)
    out = tmp_path / "out.wav"
    report = tmp_path / "report.json"
    extra = ["--json"] if mode == "json" else ["--report", str(report)]
    rc = _run(
        tmp_path,
        "post",
        str(source),
        "--out",
        str(out),
        "--plugin",
        "demo",
        "--plugin-param",
        param,
        *extra,
    )
    assert rc == 2
    assert not out.is_file()
    assert not report.is_file()
    capsys.readouterr()


def test_plugin_param_values_parse_json_and_keep_strings():
    from qtts_cli.commands.post import _plugin_params

    args = SimpleNamespace(
        plugin_param=[
            "count=3",
            "enabled=true",
            "name=speed",
            "obj={\"a\": 1}",
        ]
    )
    assert _plugin_params(args) == {
        "count": 3,
        "enabled": True,
        "name": "speed",
        "obj": {"a": 1},
    }

    bad = SimpleNamespace(plugin_param=["value=NaN"])
    with pytest.raises(UsageError):
        _plugin_params(bad)
