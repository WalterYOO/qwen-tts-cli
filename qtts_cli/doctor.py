"""Environment / model diagnostics and the local performance baseline (FR-P0-08)."""

from __future__ import annotations

import importlib.util
import platform
import resource
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Optional

from . import __version__
from .catalog import ModelInfo, scan
from .utils import human_bytes

PINNED_TRANSFORMERS = {"qwen_tts": "4.57.3", "qwen_asr": "4.57.6"}
BENCH_SENTENCES = (
    ("这是一次性能基线测试，用来测量本机生成速度。", "Chinese"),
    ("This is a performance baseline test for the local speech model.", "English"),
)


def _package_version(name: str) -> Optional[str]:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return None


def _module_present(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _torch_report() -> dict[str, Any]:
    info: dict[str, Any] = {"installed": _module_present("torch")}
    if not info["installed"]:
        return info
    try:
        import torch

        info.update(
            {
                "version": torch.__version__,
                "mps": bool(torch.backends.mps.is_available()),
                "cuda": bool(torch.cuda.is_available()),
                 "cuda_devices": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            }
        )
    except Exception as exc:  # pragma: no cover
        info["error"] = f"{type(exc).__name__}: {exc}"
    return info


def _disk_report(paths: list[Path]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path in paths:
        probe = path
        while not probe.exists() and probe.parent != probe:
            probe = probe.parent
        try:
            usage = shutil.disk_usage(probe)
            out.append(
                {
                    "path": str(path),
                    "total": human_bytes(usage.total),
                    "free": human_bytes(usage.free),
                }
            )
        except OSError as exc:
            out.append({"path": str(path), "error": str(exc)})
    return out


def _model_weights_ok(model_dir: Path) -> tuple[bool, Optional[str]]:
    config = model_dir / "config.json"
    if not config.is_file():
        return False, "missing config.json"
    weights = list(model_dir.glob("*.safetensors")) + list(model_dir.glob("*.bin"))
    if not weights:
        return False, "missing model weights (*.safetensors)"
    tokenizer = model_dir / "speech_tokenizer"
    if config.is_file() and tokenizer.is_dir() and not list(tokenizer.glob("*.safetensors")):
        return False, "missing speech_tokenizer weights"
    return True, None


def _version_warnings() -> list[str]:
    warnings: list[str] = []
    tf = _package_version("transformers")
    qtts = _package_version("qwen-tts")
    qasr = _package_version("qwen-asr")
    if qtts and tf and tf != PINNED_TRANSFORMERS["qwen_tts"]:
        warnings.append(
            f"qwen-tts {qtts} pins transformers {PINNED_TRANSFORMERS['qwen_tts']}, found {tf} "
            "(known metadata conflict; verified working with 4.57.6)"
        )
    if qasr and tf and tf != PINNED_TRANSFORMERS["qwen_asr"]:
        warnings.append(
            f"qwen-asr {qasr} pins transformers {PINNED_TRANSFORMERS['qwen_asr']}, found {tf}"
        )
    return warnings


def gather(cfg) -> dict[str, Any]:
    models = scan(cfg.model_dir)
    model_reports: list[dict[str, Any]] = []
    incomplete: list[str] = []
    for model in models:
        ok, problem = _model_weights_ok(Path(model.path))
        entry = model.to_dict()
        entry["name"] = model.name
        entry["weights_ok"] = ok
        if problem:
            entry["problem"] = problem
            incomplete.append(model.name)
        model_reports.append(entry)

    report: dict[str, Any] = {
        "ok": not incomplete,
        "qtts_version": __version__,
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
            "platform": f"{platform.system()} {platform.machine()}",
        },
        "packages": {
            name: _package_version(name)
            for name in (
                "qwen-tts",
                "qwen-asr",
                "transformers",
                "torch",
                "numpy",
                "soundfile",
                "scipy",
            )
        },
        "torch": _torch_report(),
        "tools": {
            name: shutil.which(name)
            for name in ("ffmpeg", "ffprobe", "sox", "flac")
        },
        "disk": _disk_report([cfg.data_dir, Path(cfg.model_dir) if cfg.model_dir else cfg.data_dir]),
        "model_dir": str(cfg.model_dir) if cfg.model_dir else None,
        "models": model_reports,
        "incomplete_models": incomplete,
        "capabilities": [
            {
                "model": m.name,
                "kind": m.kind,
                "size": m.size,
                "speakers": m.speakers,
                "languages": m.languages,
                "supports_instruct": m.supports_instruct,
            }
            for m in models
        ],
        "device": cfg.device,
        "offline": cfg.offline,
        "warnings": _version_warnings(),
        "advice": [],
    }

    if not report["torch"]["installed"]:
        report["advice"].append("install torch and qwen-tts (pip install -e '.[tts]') to generate audio")
    if not _module_present("qwen_tts"):
        report["advice"].append("qwen-tts is missing; `say`/`voice` cannot run")
    if not _module_present("qwen_asr"):
        report["advice"].append("qwen-asr is missing; `align`/`--subtitle`/`check` cannot run")
    if not cfg.model_dir or not Path(cfg.model_dir).is_dir():
        report["advice"].append("no model directory; pass --model-dir or set QTTS_MODEL_DIR")
    if report["tools"].get("ffmpeg") is None:
        report["advice"].append("ffmpeg not found (optional in P0; required for P1+ conversions)")
    if report["tools"].get("sox") is None:
        report["advice"].append("SoX not found (harmless for wav/flac output)")
    if incomplete:
        report["advice"].append("run `qtts models list` and re-pull incomplete models")
    return report


def _peak_rss_bytes() -> int:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux reports KiB.
    return int(usage if platform.system() == "Darwin" else usage * 1024)


def run_bench(cfg, models: Optional[list[ModelInfo]] = None) -> dict[str, Any]:
    """Load the local TTS (+ aligner) model and record load time, RTF, peak RSS."""
    from .tts_engine import TtsEngine
    from .catalog import select, select_aligner
    from .errors import QttsError

    models = models if models is not None else scan(cfg.model_dir)
    result: dict[str, Any] = {"ok": False, "cases": []}
    if not models:
        result["error"] = "no models found"
        return result
    started = time.perf_counter()
    peak_before = _peak_rss_bytes()
    try:
        model = select(models, "builtin")
    except QttsError:
        model = models[0]
    engine = TtsEngine(cfg, model)
    try:
        engine.load()
    except QttsError as exc:
        result["error"] = exc.message
        return result
    load_ms = engine.load_ms or 0.0
    result["model"] = model.name
    result["device"] = engine.device
    result["dtype"] = engine.dtype_name
    result["load_ms"] = round(load_ms, 1)

    generated: dict[str, Any] = {}
    sampling = {
        "do_sample": True,
        "top_k": 50,
        "top_p": 1.0,
        "temperature": 0.9,
        "repetition_penalty": 1.05,
        "max_new_tokens": 512,
    }
    for text, language in BENCH_SENTENCES:
        speaker = model.speakers[0] if model.speakers else None
        try:
            audio, sr, info = engine.generate(
                text=text,
                language=language,
                mode="builtin",
                speaker=speaker,
                instruct=None,
                sampling=sampling,
                seed=cfg.seed,
            )
        except QttsError as exc:
            result["cases"].append({"language": language, "error": exc.message})
            continue
        duration = info["duration_s"]
        rtf = (info["generate_ms"] / 1000.0) / duration if duration else None
        case = {
            "language": language,
            "text": text,
            "audio_duration_s": duration,
            "generate_ms": info["generate_ms"],
            "rtf": round(rtf, 3) if rtf else None,
            "sample_rate": info["sample_rate"],
        }
        result["cases"].append(case)
        generated[language] = (audio, sr)

    # Same-process TTS -> align smoke (PRD §11.13).
    if "Chinese" in generated:
        try:
            aligner_info = select_aligner(models)
            from .align_engine import AlignerEngine

            aligner = AlignerEngine(cfg, aligner_info)
            audio, sr = generated["Chinese"]
            items, info = aligner.align(audio, sr, BENCH_SENTENCES[0][0], "Chinese")
            result["align"] = {
                "model": aligner_info.name,
                "load_ms": round(aligner.load_ms or 0.0, 1),
                "align_ms": info["align_ms"],
                "items": info["items"],
            }
        except QttsError as exc:
            result["align"] = {"error": exc.message}
        except Exception as exc:  # pragma: no cover
            result["align"] = {"error": f"{type(exc).__name__}: {exc}"}

    result["peak_rss_bytes"] = _peak_rss_bytes()
    result["peak_rss_delta"] = human_bytes(max(0, _peak_rss_bytes() - peak_before))
    result["total_ms"] = round((time.perf_counter() - started) * 1000.0, 1)
    result["ok"] = bool(result["cases"]) and all("error" not in c for c in result["cases"])
    return result


def format_report(report: dict[str, Any]) -> str:
    lines: list[str] = []
    add = lines.append
    add(f"qtts {report['qtts_version']}  |  Python {report['python']['version']}  |  {report['python']['platform']}")
    add(f"python: {report['python']['executable']}")
    add("")
    add("packages:")
    for name, version in report["packages"].items():
        add(f"  {name:12s} {version or '-'}")
    torch_info = report["torch"]
    if torch_info.get("installed"):
        add(
            f"torch device: chosen={report['device']} mps={torch_info.get('mps')} "
            f"cuda={torch_info.get('cuda')} ({torch_info.get('cuda_devices', 0)} devices)"
        )
    else:
        add("torch: NOT INSTALLED")
    add("")
    add("tools:")
    for name, path in report["tools"].items():
        add(f"  {name:8s} {path or 'missing'}")
    add("")
    add(f"model dir: {report['model_dir'] or '(not configured)'}")
    for model in report["models"]:
        status = "ok" if model.get("weights_ok") else f"INCOMPLETE ({model.get('problem')})"
        extra = ""
        if model.get("kind") == "builtin" and model.get("speakers"):
            extra = f" speakers={len(model['speakers'])}"
        add(f"  {model.get('id', model.get('name'))}: {model['kind']} [{status}]{extra}")
    if not report["models"]:
        add("  (no models found)")
    add("")
    for warning in report["warnings"]:
        add(f"warning: {warning}")
    for advice in report["advice"]:
        add(f"advice: {advice}")
    return "\n".join(lines)


def format_bench(result: dict[str, Any]) -> str:
    lines: list[str] = []
    if "error" in result:
        lines.append(f"bench failed: {result['error']}")
        return "\n".join(lines)
    lines.append(
        f"model={result.get('model')} device={result.get('device')} dtype={result.get('dtype')} "
        f"load={result.get('load_ms')}ms"
    )
    for case in result.get("cases", []):
        if "error" in case:
            lines.append(f"  {case['language']}: ERROR {case['error']}")
            continue
        lines.append(
            f"  {case['language']}: audio={case['audio_duration_s']}s "
            f"generate={case['generate_ms']}ms rtf={case['rtf']} sr={case['sample_rate']}"
        )
    if "align" in result:
        align = result["align"]
        if "error" in align:
            lines.append(f"  align: ERROR {align['error']}")
        else:
            lines.append(
                f"  align: model={align['model']} load={align['load_ms']}ms "
                f"align={align['align_ms']}ms items={align['items']}"
            )
    lines.append(f"  peak RSS: {result.get('peak_rss_delta')} (max {human_bytes(result.get('peak_rss_bytes', 0))})")
    return "\n".join(lines)
