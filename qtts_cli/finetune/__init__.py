"""`qtts finetune` orchestration: prepare, train and eval (FR-P2-07).

Only metadata, JSON/JSONL and filesystem code runs in-process here.  The
official training scripts are invoked as external subprocesses; evaluation
loads the local runtimes lazily inside :mod:`qtts_cli.finetune.evaluate`.
"""

from __future__ import annotations

import math
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Optional

from .. import catalog
from ..catalog import ModelInfo, find_model, scan, select, select_asr
from ..errors import InputError, ModelError, QttsError, UsageError
from ..utils import sha256_file
from ..voice_store import VoiceStore, validate_name
from . import checkpoints as ckpt
from . import dataset as ds
from . import register as registration
from . import runner as runner_mod
from .project import (
    Project,
    SPEAKER_RE,
    default_raw_jsonl,
    ensure_pairwise_distinct,
    load_project,
    resolve_path,
    validate_output_containment,
)
from .runner import (
    CommitInterrupt,
    build_env,
    hold_lock_spec,
    hold_locks,
    project_lock_path,
    report_commit_interrupt,
    report_interrupt_state,
)

DEFAULT_BATCH_SIZE = 2
DEFAULT_LR = 2e-5
DEFAULT_EPOCHS = 3


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _arg(args, name: str):
    return getattr(args, name, None)


def _cli_path(args, name: str, project: Project, label: Optional[str] = None) -> Optional[Path]:
    value = _arg(args, name)
    if value is None:
        return None
    return resolve_path(str(value), project.project_dir, label or f"--{name}", project.path)


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise InputError(f"{label} not found: {path}", details={"path": str(path)})


def _remove_file(path: Path, label: str) -> None:
    try:
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        elif path.exists() or path.is_symlink():
            path.unlink()
    except OSError as exc:
        raise ModelError(
            f"cannot remove {label} {path}: {exc}", details={"path": str(path)}
        ) from exc


def _resolve_python(args, project: Project) -> str:
    raw = _arg(args, "python")
    if raw is None:
        raw = project.train.python
    if raw is None:
        return sys.executable
    text = str(raw)
    if os.sep in text or (os.altsep and os.altsep in text):
        return str(resolve_path(text, project.project_dir, "train.python", project.path))
    found = shutil.which(text)
    if not found:
        raise InputError(
            f"python executable not found: {text}", details={"python": text}
        )
    return found


def _resolve_official(args, project: Project) -> Path:
    cli = _cli_path(args, "finetuning_dir", project)
    official = cli or project.train.finetuning_dir
    if official is None:
        raise InputError("train.finetuning_dir is required", details={"field": "finetuning_dir"})
    return official


def _resolve_init(args, project: Project) -> Optional[Path]:
    cli = _cli_path(args, "init_model_path", project)
    return cli or project.train.init_model_path


def _load_base_model(path: Path) -> ModelInfo:
    try:
        info = catalog.load_model(path)
    except InputError:
        raise
    if info.kind != "clone":
        raise InputError(
            f"init model is not a Base model: {path} (kind={info.kind})",
            details={"path": str(path), "kind": info.kind},
        )
    return info


def _resolve_tokenizer(args, project: Project, init_model: Optional[Path]) -> Path:
    cli = _cli_path(args, "tokenizer_model_path", project)
    tokenizer = cli or project.prepare.tokenizer_model_path
    if tokenizer is None:
        if init_model is None:
            raise InputError(
                "tokenizer model path is required: pass --tokenizer-model-path or configure train.init_model_path",
                details={"field": "prepare.tokenizer_model_path"},
            )
        tokenizer = init_model / "speech_tokenizer"
        if not (tokenizer / "config.json").is_file():
            raise InputError(
                f"init model has no speech_tokenizer/config.json: {init_model}",
                details={"path": str(tokenizer)},
            )
        _load_base_model(init_model)
    if not tokenizer.is_dir() or not (tokenizer / "config.json").is_file():
        raise InputError(
            f"tokenizer model directory must contain config.json: {tokenizer}",
            details={"path": str(tokenizer)},
        )
    return tokenizer


def _prepare_paths(args, project: Project) -> tuple[Path, Path]:
    raw = _cli_path(args, "raw_jsonl", project)
    if raw is None:
        raw = project.prepare.raw_jsonl
    if raw is None:
        raw = default_raw_jsonl(project)
    output = _cli_path(args, "output_jsonl", project)
    if output is None:
        output = project.prepare.output_jsonl
    if output is None:
        output = project.project_dir / f"{project.path.stem}.train.jsonl"
    return raw, output


def _prepared_path(project: Project) -> Path:
    value = project.prepare.output_jsonl
    if value is None:
        value = project.project_dir / f"{project.path.stem}.train.jsonl"
    return value


def _raw_path(project: Project) -> Path:
    value = project.prepare.raw_jsonl
    return value if value is not None else default_raw_jsonl(project)


def _prepare_device(args, project: Project) -> str:
    cli = _arg(args, "device")
    if cli is not None:
        return str(cli)
    if project.prepare.device is not None:
        return project.prepare.device
    return "auto"


def _speaker_name(args, project: Project, *, required: bool = True) -> Optional[str]:
    cli = _arg(args, "speaker_name")
    if cli is not None:
        value = str(cli).strip().lower()
        if not SPEAKER_RE.match(value):
            raise UsageError(
                "--speaker-name must match ^[a-z0-9][a-z0-9_.-]{0,63}$"
            )
        return value
    if project.train.speaker_name is not None:
        return project.train.speaker_name
    if required:
        raise InputError("train.speaker_name is required", details={"field": "speaker_name"})
    return None


def _int_option(args, name: str, project_value, *, default: int, low: int, high: int) -> int:
    cli = _arg(args, name)
    if cli is not None:
        value = int(cli)
        if not low <= value <= high:
            raise UsageError(f"--{name.replace('_', '-')} must be within [{low}, {high}]")
        return value
    if project_value is not None:
        return int(project_value)
    return default


def _lr_option(args, project: Project) -> float:
    cli = _arg(args, "lr")
    if cli is not None:
        value = float(cli)
        if not (value > 0.0 and value <= 1.0):
            raise UsageError("--lr must be > 0 and <= 1")
        return value
    if project.train.lr is not None:
        return float(project.train.lr)
    return DEFAULT_LR


def _checkpoint_option(args, project_value):
    cli = _arg(args, "checkpoint")
    if cli is None:
        return project_value if project_value is not None else "last"
    text = str(cli).strip()
    if text == "last":
        return "last"
    try:
        number = int(text, 10)
    except ValueError:
        return resolve_path(text, Path.cwd(), "--checkpoint", Path.cwd())
    if number < 0:
        raise UsageError("--checkpoint epoch must be non-negative")
    return number


def _checkpoint_option_for_project(args, project: Project, project_value):
    cli = _arg(args, "checkpoint")
    if cli is None:
        return project_value if project_value is not None else "last"
    text = str(cli).strip()
    if text == "last":
        return "last"
    try:
        number = int(text, 10)
    except ValueError:
        return resolve_path(text, project.project_dir, "--checkpoint", project.path)
    if number < 0:
        raise UsageError("--checkpoint epoch must be non-negative")
    return number


def _registration(args, project: Project) -> tuple[Optional[str], Optional[dict[str, Any]]]:
    if hasattr(args, "register"):
        raw = getattr(args, "register")
        if str(raw).strip() == "":
            raise UsageError("--register requires a non-empty name")
        name = str(raw)
    else:
        name = project.register.name
    if name is None:
        return None, None
    validate_name(name)
    consent = project.register.consent
    if not consent or consent.get("rights_confirmed") is not True:
        raise InputError(
            "registration requires register.consent.rights_confirmed: true in the project",
            details={"voice": name},
        )
    return name, dict(consent)


def _profile_dir(cfg, name: str) -> Path:
    return VoiceStore(cfg.data_dir).dir_for(name).resolve(strict=False)


def _wrap_recovery_error(exc: QttsError, recoveries: list[str], **extra) -> QttsError:
    details = dict(exc.details)
    details.update(extra)
    if recoveries:
        details["recoveries"] = list(recoveries)
    return type(exc)(exc.message, details=details)


def _recovery_error(
    message: str,
    recoveries: list[str],
    *,
    output_dir: Optional[Path] = None,
    profile: Optional[Path] = None,
) -> InputError:
    details: dict[str, Any] = {"recovery_required": True, "recoveries": list(recoveries)}
    if output_dir is not None:
        details["output_dir"] = str(output_dir)
    if profile is not None:
        details["profile"] = str(profile)
    details["instructions"] = (
        "a previous run was interrupted; rerun with --force to replace the restored artifact"
    )
    return InputError(message, details=details)


def _target_state(path: Path) -> str:
    if not path.exists() and not path.is_symlink():
        return "absent"
    return "dir" if path.is_dir() and not path.is_symlink() else "other"


# --------------------------------------------------------------------------- #
# prepare
# --------------------------------------------------------------------------- #
def prepare(cfg, args) -> dict[str, Any]:
    project = load_project(args.project)
    device = _prepare_device(args, project)
    init_model = _resolve_init(args, project)
    tokenizer = _resolve_tokenizer(args, project, init_model)
    official = _resolve_official(args, project)
    _require_file(official / "prepare_data.py", "official prepare_data.py")
    python = _resolve_python(args, project)
    raw, output = _prepare_paths(args, project)
    raw_tmp = Path(f"{raw}.tmp")
    output_tmp = Path(f"{output}.tmp")
    output_lock = Path(f"{output}.lock")

    input_paths: list[tuple[str, Optional[Path]]] = [
        ("project", project.path),
        ("dataset.ref_audio", project.dataset.ref_audio),
        *(('dataset.items[%d].audio' % index, item.audio) for index, item in enumerate(project.dataset.items)),
        ("raw_jsonl", raw),
        ("raw_jsonl.tmp", raw_tmp),
        ("output_jsonl", output),
        ("output_jsonl.tmp", output_tmp),
        ("output_jsonl.lock", output_lock),
    ]
    ensure_pairwise_distinct(input_paths)

    dataset_info = ds.validate_dataset_audio(project.dataset)

    if output.exists() and not output.is_file():
        raise InputError(
            f"output JSONL exists but is not a regular file: {output}",
            details={"path": str(output)},
        )
    if output.exists() and not args.force:
        raise InputError(
            f"output JSONL already exists: {output}; pass --force to overwrite",
            details={"path": str(output)},
        )
    if raw.exists() and not raw.is_file():
        raise InputError(
            f"raw JSONL exists but is not a regular file: {raw}",
            details={"path": str(raw)},
        )
    for label, parent in (("output", output.parent), ("raw", raw.parent)):
        if parent.exists() and not parent.is_dir():
            raise InputError(
                f"{label} parent is not a directory: {parent}",
                details={"path": str(parent)},
            )

    final_command = [
        python,
        str(official / "prepare_data.py"),
        "--device",
        device,
        "--tokenizer_model_path",
        str(tokenizer),
        "--input_jsonl",
        str(raw),
        "--output_jsonl",
        str(output),
    ]
    real_command = final_command[:-1] + [str(output_tmp)]

    if args.dry_run:
        return {
            "ok": True,
            "phase": "prepare",
            "dry_run": True,
            "project": str(project.path),
            "dataset": dataset_info,
            "raw_jsonl": str(raw),
            "output_jsonl": str(output),
            "raw_sha256": None,
            "output_sha256": None,
            "lines": len(project.dataset.items),
            "tokenizer_model_path": str(tokenizer),
            "init_model_path": str(init_model) if init_model is not None else None,
            "device": device,
            "command": final_command,
            "will_write": {"raw_jsonl": str(raw), "output_jsonl": str(output)},
            "temporary_output": str(output_tmp),
        }

    with hold_locks([project_lock_path(project.path), output_lock]):
        try:
            output.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ModelError(
                f"cannot create output directory {output.parent}: {exc}",
                details={"path": str(output.parent)},
            ) from exc
        raw_sha = ds.write_raw_jsonl(raw, project.dataset)
        _remove_file(output_tmp, "stale temporary output")
        env = build_env(bool(cfg.offline))
        try:
            runner_mod.run_external(
                real_command,
                cwd=official,
                env=env,
                quiet=bool(cfg.quiet),
            )
        except KeyboardInterrupt:
            _remove_file(output_tmp, "temporary output")
            raise
        except ModelError:
            _remove_file(output_tmp, "temporary output")
            raise
        try:
            ds.prepared_jsonl_nonempty(output_tmp)
            lines = ds.validate_prepared_jsonl(
                output_tmp, ds.raw_records(project.dataset)
            )
        except ModelError:
            _remove_file(output_tmp, "temporary output")
            raise
        try:
            with open(output_tmp, "rb") as handle:
                os.fsync(handle.fileno())
            os.replace(output_tmp, output)
        except OSError as exc:
            raise ModelError(
                f"cannot publish prepared JSONL {output}: {exc}",
                details={"temporary_output": str(output_tmp)},
            ) from exc

    try:
        output_sha = sha256_file(output)
    except OSError as exc:
        raise ModelError(
            f"cannot hash prepared JSONL {output}: {exc}", details={"path": str(output)}
        ) from exc
    return {
        "ok": True,
        "phase": "prepare",
        "dry_run": False,
        "project": str(project.path),
        "dataset": dataset_info,
        "raw_jsonl": str(raw),
        "output_jsonl": str(output),
        "raw_sha256": raw_sha,
        "output_sha256": output_sha,
        "lines": lines,
        "tokenizer_model_path": str(tokenizer),
        "init_model_path": str(init_model) if init_model is not None else None,
        "device": device,
        "command": real_command,
    }


# --------------------------------------------------------------------------- #
# train
# --------------------------------------------------------------------------- #
def _prepare_output_recovery(
    output_dir: Path, force: bool, *, dry_run: bool
) -> tuple[list[str], bool]:
    """Inspect final/.old/.partial; return pending recoveries and a dry-run flag."""
    old = Path(f"{output_dir}.old")
    partial = Path(f"{output_dir}.partial")
    state = _target_state(output_dir)
    if state == "other":
        raise InputError(
            f"output_dir exists but is not a directory: {output_dir}",
            details={"path": str(output_dir)},
        )
    if state == "dir" and not force:
        raise InputError(
            f"output_dir already exists: {output_dir}; pass --force to replace",
            details={"path": str(output_dir)},
        )
    recoveries: list[str] = []
    if old.exists() and state == "absent":
        if dry_run:
            raise _recovery_error(
                "a previous publication was interrupted; rerun without --dry-run "
                "to restore it (then pass --force to train again)",
                ["restored_old"],
                output_dir=output_dir,
            )
        recoveries.append("restored_old")
    elif partial.exists() and not force:
        # Only a stale partial with no publication to recover: refuse without
        # --force. When `.old` exists the interrupted publication takes
        # priority; the stale `.partial` is handled inside the locks.
        raise InputError(
            f"a failed training run left {partial}; pass --force to discard it",
            details={"path": str(partial)},
        )
    return recoveries, False


def _profile_preflight(
    store: VoiceStore,
    name: str,
    force: bool,
    *,
    dry_run: bool,
) -> bool:
    target = store.dir_for(name)
    backups = registration.profile_backups(store, name)
    if target.exists():
        if not force:
            raise InputError(
                f"voice already exists: {name}; pass --force to replace it",
                details={"voice": name, "path": str(target)},
            )
        return False
    if len(backups) > 1:
        raise InputError(
            f"multiple profile backups for {name}; remove them manually",
            details={"voice": name, "backups": [str(path) for path in backups]},
        )
    if len(backups) == 1:
        if dry_run:
            raise _recovery_error(
                "an interrupted registration left a profile backup; rerun without "
                "--dry-run to restore it (then pass --force to register again)",
                ["restored_profile"],
                profile=target,
            )
        return True
    return False


def _train_paths(
    cfg,
    project: Project,
    *,
    official: Optional[Path],
    init_model: Optional[Path],
    output_dir: Optional[Path],
    prepared: Path,
    train_jsonl: Optional[Path],
    profile: Optional[Path],
) -> None:
    entries: list[tuple[str, Optional[Path]]] = [
        ("project", project.path),
        ("official_dir", official),
        ("init_model", init_model),
        ("prepared_jsonl", prepared),
        ("prepared_jsonl.tmp", Path(f"{prepared}.tmp")),
        ("train_jsonl", train_jsonl if train_jsonl != prepared else None),
        ("data_dir", Path(cfg.data_dir).expanduser().resolve(strict=False)),
        ("profile", profile),
        ("dataset.ref_audio", project.dataset.ref_audio),
        *(
            (f"dataset.items[{index}].audio", item.audio)
            for index, item in enumerate(project.dataset.items)
        ),
    ]
    raw = _raw_path(project)
    entries.extend([("raw_jsonl", raw), ("raw_jsonl.tmp", Path(f"{raw}.tmp"))])
    if output_dir is not None:
        entries.extend(
            [
                ("output_dir", output_dir),
                ("output_dir.partial", Path(f"{output_dir}.partial")),
                ("output_dir.old", Path(f"{output_dir}.old")),
                ("output_dir.lock", Path(f"{output_dir}.lock")),
            ]
        )
    ensure_pairwise_distinct(entries)


def _dry_train_scan(
    output_dir: Optional[Path],
    speaker: str,
    selector,
    *,
    register_only: bool,
) -> tuple[list[ckpt.Checkpoint], Any]:
    if register_only:
        selected = ckpt.select_checkpoint(
            selector, output_dir=output_dir, speaker_name=speaker
        )
        return [selected], selected
    if output_dir is not None and output_dir.is_dir():
        found = ckpt.validate_output(output_dir, speaker)
        selected = ckpt.select_checkpoint(
            selector,
            output_dir=output_dir,
            speaker_name=speaker,
            require_inside=output_dir,
        )
        return found, selected
    return [], None


def _payload_checkpoint(
    checkpoint: ckpt.Checkpoint, *, selected: bool, hashes: Optional[dict] = None
) -> dict[str, Any]:
    payload = checkpoint.to_dict(selected=selected)
    if hashes:
        payload["fingerprint"] = hashes["checkpoint_sha256"][:16]
    return payload


def train(cfg, args) -> dict[str, Any]:
    project = load_project(args.project)
    register_name, consent = _registration(args, project)
    register_only = bool(_arg(args, "register_only"))
    if register_only and register_name is None:
        raise UsageError("--register-only requires --register NAME or register.name")

    speaker = _speaker_name(args, project)
    assert speaker is not None
    batch_size = _int_option(
        args, "batch_size", project.train.batch_size, default=DEFAULT_BATCH_SIZE, low=1, high=1024
    )
    epochs = _int_option(
        args, "epochs", project.train.num_epochs, default=DEFAULT_EPOCHS, low=1, high=1000
    )
    lr = _lr_option(args, project)
    selector = _checkpoint_option_for_project(args, project, project.train.checkpoint)
    force = bool(_arg(args, "force"))
    dry_run = bool(_arg(args, "dry_run"))
    init_path = _resolve_init(args, project)
    output_dir = _cli_path(args, "output_dir", project) or project.train.output_dir
    prepared = _prepared_path(project)
    train_jsonl = _cli_path(args, "train_jsonl", project)
    if train_jsonl is None:
        train_jsonl = project.train.train_jsonl or prepared
    official: Optional[Path] = None
    python: Optional[str] = None
    init_model: Optional[ModelInfo] = None
    store = VoiceStore(cfg.data_dir)
    profile_dir = _profile_dir(cfg, register_name) if register_name is not None else None

    if not register_only:
        official = _resolve_official(args, project)
        _require_file(official / "sft_12hz.py", "official sft_12hz.py")
        if init_path is None:
            raise InputError(
                "train.init_model_path is required", details={"field": "init_model_path"}
            )
        if output_dir is None:
            raise InputError(
                "train.output_dir is required", details={"field": "output_dir"}
            )
        ds.validate_dataset_audio(project.dataset)
        init_model = _load_base_model(init_path)
        python = _resolve_python(args, project)
        if not train_jsonl.is_file():
            raise InputError(
                f"train JSONL not found: {train_jsonl}",
                details={"path": str(train_jsonl)},
            )
        ds.validate_prepared_jsonl(train_jsonl, error_cls=InputError)
        validate_output_containment(
            output_dir,
            project_file=project.path,
            official_dir=official,
            init_model=init_path,
            jsonl_paths=(prepared, train_jsonl, _raw_path(project)),
            input_paths=(
                project.dataset.ref_audio,
                *(item.audio for item in project.dataset.items),
            ),
            data_dir=Path(cfg.data_dir).expanduser().resolve(strict=False),
            profile_dir=profile_dir,
        )
    else:
        if init_path is not None:
            init_model = _load_base_model(init_path)
        if output_dir is not None:
            validate_output_containment(
                output_dir,
                project_file=project.path,
                official_dir=None,
                init_model=init_path,
                jsonl_paths=(prepared, train_jsonl, _raw_path(project)),
                input_paths=(
                    project.dataset.ref_audio,
                    *(item.audio for item in project.dataset.items),
                ),
                data_dir=Path(cfg.data_dir).expanduser().resolve(strict=False),
                profile_dir=profile_dir,
            )
        if output_dir is not None and not output_dir.is_dir():
            if not isinstance(selector, (str, Path)) or selector == "last":
                raise InputError(
                    f"output directory not found: {output_dir}",
                    details={"path": str(output_dir)},
                )

    _train_paths(
        cfg,
        project,
        official=official,
        init_model=init_path,
        output_dir=output_dir,
        prepared=prepared,
        train_jsonl=train_jsonl if not register_only else None,
        profile=profile_dir,
    )

    pending_output: list[str] = []
    if not register_only:
        pending_output, _ = _prepare_output_recovery(output_dir, force, dry_run=dry_run)
    if register_name is not None:
        _profile_preflight(store, register_name, force, dry_run=dry_run)

    if dry_run:
        checkpoints, selected = _dry_train_scan(
            output_dir, speaker, selector, register_only=register_only
        )
        if register_only and selected is not None and output_dir is None:
            checkpoints = [selected]
        command = []
        if not register_only:
            assert output_dir is not None and official is not None and init_path is not None
            assert python is not None
            command = [
                python,
                str(official / "sft_12hz.py"),
                "--init_model_path",
                str(init_path),
                "--output_model_path",
                str(Path(f"{output_dir}.partial")),
                "--train_jsonl",
                str(train_jsonl),
                "--batch_size",
                str(batch_size),
                "--lr",
                repr(float(lr)),
                "--num_epochs",
                str(epochs),
                "--speaker_name",
                speaker,
            ]
        return {
            "ok": True,
            "phase": "train",
            "dry_run": True,
            "models_loaded": False,
            "output_dir": str(output_dir) if output_dir is not None else None,
            "partial_kept": False,
            "recoveries": [],
            "published": False,
            "registered": False,
            "checkpoints": [
                cp.to_dict(selected=selected is not None and cp.path == selected.path)
                for cp in checkpoints
            ],
            "selected": (
                {
                    "epoch": selected.epoch,
                    "path": str(selected.path),
                    "speaker": speaker,
                    "fingerprint": selected.fingerprint,
                }
                if selected is not None
                else None
            ),
            "register": (
                {"name": register_name, "registered": False, "path": str(profile_dir)}
                if register_name is not None
                else None
            ),
            "command": command,
            "warnings": [],
            "will_create_partial": bool(
                output_dir is not None and not output_dir.is_dir()
            ),
        }

    if output_dir is not None:
        artifact_lock = Path(f"{output_dir}.lock")
    else:
        assert register_only
        selected_pre = ckpt.select_checkpoint(
            selector, output_dir=None, speaker_name=speaker
        )
        artifact_lock = Path(f"{selected_pre.path.parent}.lock")

    recoveries: list[str] = []
    with hold_locks([project_lock_path(project.path), artifact_lock]):
        if not register_only:
            assert output_dir is not None
            old = Path(f"{output_dir}.old")
            partial = Path(f"{output_dir}.partial")
            try:
                if not output_dir.exists() and old.exists():
                    os.replace(old, output_dir)
                    recoveries.append("restored_old")
                    if not force:
                        raise InputError(
                            "recovered a previous publication; rerun with --force to replace it",
                            details={"output_dir": str(output_dir)},
                        )
                elif output_dir.exists():
                    ckpt.remove_stale_old(output_dir)
                if partial.exists():
                    _remove_file(partial, "stale partial output")
            except InputError as exc:
                raise _wrap_recovery_error(
                    exc, recoveries, output_dir=str(output_dir)
                ) from exc
            except OSError as exc:
                raise ModelError(
                    f"I/O error while recovering output state: {exc}",
                    details={
                        "recoveries": recoveries,
                        "output_dir": str(output_dir),
                    },
                ) from exc

        if register_name is not None:
            backups = registration.profile_backups(store, register_name)
            target = store.dir_for(register_name)
            try:
                if not target.exists() and backups:
                    registration.restore_profile_backup(
                        store, register_name, backups[0]
                    )
                    recoveries.append("restored_profile")
                    if not force:
                        raise InputError(
                            "recovered an interrupted registration; rerun with --force to replace it",
                            details={"voice": register_name},
                        )
                elif target.exists() and backups:
                    registration.remove_profile_backups(store, register_name)
            except InputError as exc:
                raise _wrap_recovery_error(
                    exc, recoveries, profile=str(target)
                ) from exc

        if register_only:
            selected = ckpt.select_checkpoint(
                selector, output_dir=output_dir, speaker_name=speaker
            )
            checkpoints = [selected]
            command: list[str] = []
            published = False
        else:
            assert output_dir is not None and official is not None and init_path is not None
            assert python is not None
            partial = Path(f"{output_dir}.partial")
            try:
                output_dir.parent.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise ModelError(
                    f"cannot create output parent {output_dir.parent}: {exc}",
                    details={"path": str(output_dir.parent)},
                ) from exc
            command = [
                python,
                str(official / "sft_12hz.py"),
                "--init_model_path",
                str(init_path),
                "--output_model_path",
                str(partial),
                "--train_jsonl",
                str(train_jsonl),
                "--batch_size",
                str(batch_size),
                "--lr",
                repr(float(lr)),
                "--num_epochs",
                str(epochs),
                "--speaker_name",
                speaker,
            ]
            try:
                runner_mod.run_external(
                    command,
                    cwd=official,
                    env=build_env(bool(cfg.offline)),
                    quiet=bool(cfg.quiet),
                )
            except KeyboardInterrupt:
                report_interrupt_state(
                    published=False, registered=False, partial=partial
                )
                raise
            except ModelError as exc:
                details = dict(exc.details)
                details.update(
                    {
                        "partial": str(partial),
                        "partial_kept": True,
                        "output_dir": str(output_dir),
                        "recoveries": recoveries,
                    }
                )
                raise ModelError(exc.message, details=details) from exc
            try:
                ckpt.validate_output(partial, speaker)
            except ModelError as exc:
                details = dict(exc.details)
                details.update({"partial": str(partial), "partial_kept": True})
                raise ModelError(exc.message, details=details) from exc
            warnings: list[str] = []
            try:
                warnings.extend(ckpt.publish_output_dir(partial, output_dir))
            except CommitInterrupt as exc:
                exc.partial = None
                report_commit_interrupt(exc)
                raise
            except KeyboardInterrupt:
                report_interrupt_state(
                    published=False, registered=False, partial=partial
                )
                raise
            try:
                checkpoints = ckpt.validate_output(output_dir, speaker)
            except ModelError as exc:
                details = dict(exc.details)
                details["published"] = True
                raise ModelError(exc.message, details=details) from exc
            try:
                selected = ckpt.select_checkpoint(
                    selector,
                    output_dir=output_dir,
                    speaker_name=speaker,
                    require_inside=output_dir,
                )
            except InputError as exc:
                raise _wrap_recovery_error(
                    exc, recoveries, published=True, checkpoint_path=str(output_dir)
                ) from exc
            published = True

    registered = False
    profile_path = str(profile_dir) if profile_dir is not None else None
    if register_only:
        warnings = []
    selected_hashes: Optional[dict[str, Any]] = None
    if register_name is not None:
        assert profile_dir is not None
        try:
            dataset_sha = ds.items_sha256(project.dataset)
            ref_sha = sha256_file(project.dataset.ref_audio)
            raw_sha, prepared_sha = ds.raw_prepared_hashes(
                _raw_path(project), prepared
            )
            meta, family_hashes = registration.build_voice_meta(
                store=store,
                name=register_name,
                checkpoint=selected,
                speaker_name=speaker,
                dataset=project.dataset,
                dataset_sha=dataset_sha,
                ref_sha=ref_sha,
                raw_jsonl_sha=raw_sha,
                prepared_jsonl_sha=prepared_sha,
                output_dir=output_dir,
                init_model=init_model,
                batch_size=batch_size,
                lr=lr,
                num_epochs=epochs,
                consent=consent or {},
            )
            staging = registration.write_staging_profile(store, register_name, meta)
        except KeyboardInterrupt:
            report_interrupt_state(
                published=published, registered=False, partial=None
            )
            raise
        except QttsError as exc:
            raise _wrap_recovery_error(
                exc,
                recoveries,
                published=published,
                registered=False,
                checkpoint=str(selected.path),
            ) from exc
        except OSError as exc:
            raise ModelError(
                f"cannot build finetune profile: {exc}",
                details={
                    "published": published,
                    "registered": False,
                    "checkpoint": str(selected.path),
                },
            ) from exc
        try:
            warnings.extend(
                registration.publish_profile(
                    store, staging, register_name, published=published
                )
            )
        except CommitInterrupt as exc:
            exc.partial = None
            report_commit_interrupt(exc)
            raise
        except KeyboardInterrupt:
            report_interrupt_state(
                published=published, registered=False, partial=None
            )
            raise
        except QttsError as exc:
            raise _wrap_recovery_error(
                exc,
                recoveries,
                published=published,
                registered=False,
                checkpoint=str(selected.path),
            ) from exc
        registered = True
        selected_hashes = family_hashes

    if selected_hashes is None:
        # §5.4 always reports the selected checkpoint's weight hash, even when
        # no profile is registered. The registration path already computed it
        # above and must not hash the weights twice.
        selected_hashes = registration.build_checkpoint_hashes(selected, speaker)

    selected_payload: dict[str, Any] = {
        "epoch": selected.epoch,
        "path": str(selected.path),
        "speaker": speaker,
        "fingerprint": selected_hashes["checkpoint_sha256"][:16],
        "weights_sha256": selected_hashes["weights_sha256"],
    }

    return {
        "ok": True,
        "phase": "train",
        "dry_run": False,
        "output_dir": str(output_dir) if output_dir is not None else None,
        "partial_kept": False,
        "recoveries": recoveries,
        "published": published,
        "registered": registered,
        "checkpoints": [
            _payload_checkpoint(
                cp,
                selected=cp.path == selected.path,
                hashes=selected_hashes if cp.path == selected.path else None,
            )
            for cp in checkpoints
        ],
        "selected": selected_payload,
        "register": (
            {
                "name": register_name,
                "registered": registered,
                "path": profile_path,
            }
            if register_name is not None
            else None
        ),
        "command": command,
        "warnings": warnings,
    }


# --------------------------------------------------------------------------- #
# eval
# --------------------------------------------------------------------------- #
def _resolve_eval_models(
    cfg, project, args, *, similarity_requested: bool = False
):
    asr_spec = _arg(args, "asr_model")
    if asr_spec is None:
        asr_spec = project.eval.asr_model if project.eval is not None else None
    base_spec = _arg(args, "base_model")
    if base_spec is None:
        base_spec = project.eval.base_model if project.eval is not None else None
    models = scan(cfg.model_dir)
    asr_model = None
    if asr_spec:
        try:
            asr_model = select_asr(models, str(asr_spec))
        except QttsError as exc:
            raise InputError(
                f"cannot resolve ASR model {asr_spec!r}: {exc.message}",
                details={"asr_model": str(asr_spec)},
            ) from exc
    base_model = None
    if base_spec:
        try:
            base_model = find_model(models, str(base_spec))
        except QttsError as exc:
            raise InputError(
                f"cannot resolve Base model {base_spec!r}: {exc.message}",
                details={"base_model": str(base_spec)},
            ) from exc
        if base_model.kind != "clone":
            raise InputError(
                f"base model is not a Base model: {base_model.name} (kind={base_model.kind})",
                details={"base_model": base_model.name, "kind": base_model.kind},
            )
    elif similarity_requested:
        # §10.1: no explicit spec but similarity requested — auto-select the
        # best local Base model. Resolution failure is an input error (4).
        try:
            base_model = select(models, "clone")
        except QttsError as exc:
            raise InputError(
                f"cannot auto-select a Base model: {exc.message}",
                details={"kind": "clone"},
            ) from exc
    return asr_model, base_model


def _threshold(args, project, name: str):
    cli = _arg(args, name)
    if cli is not None:
        value = float(cli)
        if not math.isfinite(value):
            raise UsageError(f"--{name.replace('_', '-')} must be finite")
        if name == "max_cer" and value < 0:
            raise UsageError("--max-cer must be >= 0")
        if name == "min_speaker_similarity" and not 0.0 <= value <= 1.0:
            raise UsageError("--min-speaker-similarity must be within [0, 1]")
        return value
    if project.eval is None:
        return None
    return getattr(project.eval, name)


def eval(cfg, args) -> dict[str, Any]:  # noqa: A001 - command name
    project = load_project(args.project)
    if project.eval is None:
        raise InputError("the project has no eval block", details={"project": str(project.path)})
    spec = project.eval
    store = VoiceStore(cfg.data_dir)
    voice_name = _arg(args, "voice")
    explicit_checkpoint = _arg(args, "checkpoint")

    if voice_name is not None:
        if explicit_checkpoint is not None:
            raise UsageError("--checkpoint cannot be combined with --voice")
        meta = store.get(str(voice_name))
        if meta is None:
            raise UsageError(f"voice not found: {voice_name}")
        if meta.get("kind") != "finetune":
            raise UsageError(
                f"voice {voice_name} is not a finetune profile (kind={meta.get('kind')})"
            )
        from ..capabilities import VoiceRef
        from ..plan import resolve_voice

        resolution = resolve_voice([], store, VoiceRef("voice", str(voice_name)))
        model = resolution.model
        speaker = resolution.speaker
        assert speaker is not None
        checkpoint_path = Path(model.path)
        payload_voice = str(voice_name)
    else:
        speaker_name = _speaker_name(args, project)
        assert speaker_name is not None
        selector = _checkpoint_option_for_project(args, project, spec.checkpoint)
        output_dir = project.train.output_dir
        selected = ckpt.select_checkpoint(
            selector,
            output_dir=output_dir,
            speaker_name=speaker_name,
            require_speaker=False,
        )
        model = selected.model
        canonical = model.canonical_speaker(speaker_name)
        if canonical is None:
            raise InputError(
                f"speaker {speaker_name!r} is not in the checkpoint",
                details={"speaker": speaker_name, "path": str(selected.path)},
            )
        speaker = canonical
        checkpoint_path = selected.path
        payload_voice = None

    max_cer = _threshold(args, project, "max_cer")
    min_similarity = _threshold(args, project, "min_speaker_similarity")
    asr_model, base_model = _resolve_eval_models(
        cfg, project, args, similarity_requested=min_similarity is not None
    )
    if max_cer is not None and asr_model is None:
        raise UsageError("--max-cer requires CER-capable asr_model")
    if min_similarity is not None and base_model is None:
        raise UsageError("--min-speaker-similarity requires a Base model")

    audio_dir = _cli_path(args, "audio_dir", project)
    if audio_dir is None:
        audio_dir = spec.audio_dir
    if audio_dir is not None and audio_dir.exists() and not audio_dir.is_dir():
        raise InputError(
            f"eval audio_dir is not a directory: {audio_dir}",
            details={"path": str(audio_dir)},
        )
    if base_model is not None:
        # Similarity will read the reference audio; fail before returning a
        # dry-run preview or creating any directory/model.
        ds.validate_ref_audio(project.dataset.ref_audio)

    if args.dry_run:
        return {
            "ok": True,
            "phase": "eval",
            "dry_run": True,
            "models_loaded": False,
            "voice": payload_voice,
            "checkpoint": str(checkpoint_path),
            "speaker": speaker,
            "asr_model": asr_model.model_id if asr_model is not None else None,
            "base_model": base_model.model_id if base_model is not None else None,
            "thresholds": {
                "max_cer": max_cer,
                "min_speaker_similarity": min_similarity,
            },
            "audio_dir": str(audio_dir) if audio_dir is not None else None,
            "items": len(spec.items),
            "will_write": {"audio_dir": str(audio_dir) if audio_dir is not None else None},
        }

    if voice_name is not None:
        source = (store.require(str(voice_name)).get("source") or {})
        source_output = source.get("output_dir")
        if source_output:
            artifact_lock = Path(f"{Path(str(source_output))}.lock")
        else:
            artifact_lock = Path(f"{checkpoint_path.parent}.lock")
    elif isinstance(selector, int) or selector == "last":
        if project.train.output_dir is None:
            raise InputError("train.output_dir is required for eval checkpoint selection")
        artifact_lock = Path(f"{project.train.output_dir}.lock")
    else:
        artifact_lock = Path(f"{checkpoint_path.parent}.lock")

    from . import evaluate as evaluate_mod

    with hold_lock_spec(
        [(project_lock_path(project.path), False), (artifact_lock, True)]
    ):
        return evaluate_mod.evaluate(
            cfg,
            model=model,
            speaker=speaker,
            checkpoint_path=checkpoint_path,
            voice_name=payload_voice,
            items=spec.items,
            dataset=project.dataset,
            asr_model=asr_model,
            base_model=base_model,
            max_cer=max_cer,
            min_speaker_similarity=min_similarity,
            audio_dir=audio_dir,
            eval_language=spec.language,
        )
