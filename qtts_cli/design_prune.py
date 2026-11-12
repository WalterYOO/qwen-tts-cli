"""Prune non-kept design candidates without touching ``run.json`` first.

The two-phase sequence is: move candidate files into ``.prune/<uuid>/``, then
atomically rewrite ``run.json`` with ``pruned: true`` and null paths, then drop
the staging directory.  A crash before the manifest rewrite is recovered by
moving files back; a crash after it simply deletes the staging copy.
"""

from __future__ import annotations

import os
import re
import shutil
import uuid
from pathlib import Path
from typing import Any

from .design_report import analyze_run
from .design_run import RunLock, save_manifest
from .errors import InputError, UsageError
from .utils import read_json

_PRUNE_ROOT = ".prune"
_TAKE_ID = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
_CANDIDATE_SUFFIXES = (".wav", ".recipe.json")
_CANDIDATE_NAME = re.compile(
    r"^[A-Za-z0-9_.-]{1,64}(?:\.wav|\.recipe\.json)$"
)


def _candidates_dir(root: Path) -> Path:
    """Resolve ``root/candidates`` and require it to stay inside ``root``."""
    candidates_path = root / "candidates"
    if candidates_path.is_symlink() or not candidates_path.is_dir():
        raise InputError(
            f"candidates directory is missing or a symlink; refusing recovery: "
            f"{candidates_path}",
            details={"path": str(candidates_path)},
        )
    resolved = candidates_path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise InputError(
            f"candidates directory escapes the run directory: {candidates_path}",
            details={"path": str(candidates_path), "resolved": str(resolved)},
        ) from exc
    return candidates_path


def _restore_file(root: Path, source: Path, staging: Path) -> None:
    """Move one staged file back to ``root/candidates/`` safely."""
    rel = source.relative_to(staging)
    if len(rel.parts) != 2 or rel.parts[0] != "candidates":
        return
    if not _CANDIDATE_NAME.fullmatch(rel.parts[1]):
        return
    candidates_root = _candidates_dir(root).resolve()
    destination = candidates_root / rel.parts[1]
    if destination.parent != candidates_root:
        return
    if destination.is_symlink() or destination.exists():
        return
    os.replace(source, destination)


def _take_id_from_rel(rel: Path) -> str:
    name = rel.name
    for suffix in _CANDIDATE_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return rel.stem


def _recover(root: Path) -> None:
    base = root / _PRUNE_ROOT
    if base.is_symlink():
        raise InputError(
            f"prune staging path is a symlink; refusing to touch it: {base}",
            details={"path": str(base)},
        )
    if not base.is_dir():
        return
    manifest_path = root / "run.json"
    records: dict[str, dict[str, Any]] = {}
    if manifest_path.is_file():
        try:
            manifest = read_json(manifest_path)
        except (OSError, ValueError):
            manifest = {}
        records = {
            str(record.get("take_id")): record
            for record in (manifest.get("candidates") or [])
            if isinstance(record, dict)
        }
    for staging in sorted(base.iterdir()):
        if staging.is_symlink():
            raise InputError(
                f"prune staging entry is a symlink; refusing to touch it: {staging}",
                details={"path": str(staging)},
            )
        if not staging.is_dir():
            continue
        for path in sorted(staging.rglob("*")):
            if not path.is_file() or path.is_symlink():
                continue
            try:
                path.resolve().relative_to(staging.resolve())
            except ValueError:
                continue
            rel = path.relative_to(staging)
            take_id = _take_id_from_rel(rel)
            record = records.get(take_id) or {}
            if record.get("pruned"):
                continue
            _restore_file(root, path, staging)
        shutil.rmtree(staging, ignore_errors=True)
    try:
        base.rmdir()
    except OSError:
        pass


def _candidate_files(root: Path, take_id: Any) -> list[Path]:
    text = str(take_id or "")
    if not _TAKE_ID.fullmatch(text):
        raise InputError(
            f"invalid take_id in run.json: {take_id!r}",
            details={"take_id": take_id},
        )
    files: list[Path] = []
    candidates_dir = _candidates_dir(root).resolve()
    for suffix in _CANDIDATE_SUFFIXES:
        path = root / "candidates" / f"{text}{suffix}"
        if path.is_symlink():
            raise InputError(
                f"refusing to prune a symlink: {path}", details={"path": str(path)}
            )
        if path.exists() and not path.is_file():
            raise InputError(
                f"refusing to prune a non-file: {path}", details={"path": str(path)}
            )
        if not path.exists():
            continue
        if path.resolve().parent != candidates_dir:
            raise InputError(
                f"candidate path escapes the run directory: {path}",
                details={"path": str(path)},
            )
        files.append(path)
    return files


def prune_run(
    run_dir: str | Path,
    *,
    keep: int,
    apply: bool = False,
) -> dict[str, Any]:
    root = Path(run_dir).expanduser()
    if not isinstance(keep, int) or isinstance(keep, bool) or keep < 0:
        raise UsageError("--keep must be a non-negative integer")
    if not (root / "run.json").is_file():
        raise InputError(
            f"not a design run directory: {root}", details={"path": str(root)}
        )
    if not apply:
        return _prune_run(root, keep=keep, apply=False)
    # Pruning rewrites files shared with explore --resume; use the same lock.
    with RunLock(root):
        return _prune_run(root, keep=keep, apply=True)


def _prune_run(
    root: Path,
    *,
    keep: int,
    apply: bool,
) -> dict[str, Any]:
    if not isinstance(keep, int) or isinstance(keep, bool) or keep < 0:
        raise UsageError("--keep must be a non-negative integer")
    manifest_path = root / "run.json"
    if not manifest_path.is_file():
        raise InputError(
            f"not a design run directory: {root}", details={"path": str(root)}
        )
    for name in ("candidates", _PRUNE_ROOT):
        path = root / name
        if path.is_symlink():
            raise InputError(
                f"refusing to prune through a symlink: {path}",
                details={"path": str(path)},
            )
    if apply:
        # Recovery moves files, so a dry run must never trigger it.
        _recover(root)
    manifest, candidates = analyze_run(root, metrics="basic")
    analyzed = [item for item in candidates if item.get("status") != "pruned"]
    kept = {item.get("take_id") for item in analyzed[:keep]}
    selected = [
        record
        for record in (manifest.get("candidates") or [])
        if record.get("take_id") not in kept and not record.get("pruned")
    ]

    plan: list[dict[str, Any]] = []
    total_bytes = 0
    for record in selected:
        files = _candidate_files(root, record.get("take_id"))
        size = sum(path.stat().st_size for path in files)
        total_bytes += size
        plan.append(
            {
                "take_id": record.get("take_id"),
                "files": [path.relative_to(root).as_posix() for path in files],
                "bytes": size,
            }
        )

    payload: dict[str, Any] = {
        "ok": True,
        "run_dir": str(root),
        "keep": keep,
        "dry_run": not apply,
        "kept": [item.get("take_id") for item in analyzed[:keep]],
        "pruned": plan,
        "freed_bytes": total_bytes,
        "warnings": [],
    }
    if (root / "report").exists():
        payload["warnings"].append(
            "report/ is now stale; rerun `qtts design report` to refresh it"
        )
    if not apply or not plan:
        payload["dry_run"] = not apply
        return payload

    staging = root / _PRUNE_ROOT / uuid.uuid4().hex
    staging.mkdir(parents=True, exist_ok=False)
    manifest_saved = False
    try:
        for record in selected:
            for path in _candidate_files(root, record.get("take_id")):
                rel = path.relative_to(root)
                destination = staging / rel
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.replace(path.resolve(), destination)
        for record in selected:
            record["pruned"] = True
            record["pruned_sha256"] = record.get("sha256")
            record["wav"] = None
            record["recipe"] = None
        save_manifest(root, manifest)
        manifest_saved = True
    except BaseException:
        if not manifest_saved:
            # Scan the staging tree instead of an in-memory list so a move
            # that landed just before the interruption is still restored.
            for source in sorted(staging.rglob("*")):
                if not source.is_file() or source.is_symlink():
                    continue
                try:
                    source.resolve().relative_to(staging.resolve())
                except ValueError:
                    continue
                _restore_file(root, source, staging)
        shutil.rmtree(staging, ignore_errors=True)
        raise
    shutil.rmtree(staging, ignore_errors=True)
    try:
        (root / _PRUNE_ROOT).rmdir()
    except OSError:
        pass
    payload["dry_run"] = False
    return payload


def format_result(payload: dict[str, Any]) -> str:
    verb = "would prune" if payload.get("dry_run") else "pruned"
    plan = payload.get("pruned") or []
    lines = [
        f"{verb} {len(plan)} candidate(s); kept "
        f"{len(payload.get('kept') or [])} "
        f"({payload.get('freed_bytes', 0)} bytes)"
    ]
    for item in plan:
        lines.append(f"  {item['take_id']}: {', '.join(item['files']) or '(no files)'}")
    for warning in payload.get("warnings") or []:
        lines.append(f"  warning: {warning}")
    return "\n".join(lines)
