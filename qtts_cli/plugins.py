"""Plugin SDK discovery for backend and post-process plugins (FR-P2-08).

Plugins are discovered only through installed entry points.  Installing a
plugin is a trust decision: discovery imports the entry-point module (its
top-level code runs) but never calls ``models``/``create``/``process`` and
never auto-installs anything.

This module imports only :mod:`importlib.metadata`; it never imports
``torch``, ``dashscope`` or ``vllm_omni``.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import re
from dataclasses import dataclass
from typing import Any, Iterable, Optional

from .backends.base import BackendCapabilities
from .errors import ModelError, UsageError

PLUGIN_API_VERSION = 1
BACKEND_GROUP = "qtts.backends"
POSTPROCESS_GROUP = "qtts.postprocess"
_NAME_RE = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")

STATUS_OK = "ok"
STATUS_UNSUPPORTED_API = "unsupported_api"
STATUS_BROKEN = "broken"
STATUS_INVALID = "invalid"
STATUS_DUPLICATE = "duplicate_name"

DEFAULT_BACKEND_CAPABILITIES = BackendCapabilities(
    modes=frozenset({"builtin"}),
    instruct_modes=frozenset(),
    sampling=frozenset(),
    remote=True,
)


@dataclass
class PluginRecord:
    kind: str  # backend | postprocess
    name: str
    distribution: Optional[str]
    version: Optional[str]
    status: str
    reason: Optional[str]
    obj: Any = None

    @property
    def usable(self) -> bool:
        return self.status == STATUS_OK

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "name": self.name,
            "distribution": self.distribution,
            "version": self.version,
            "status": self.status,
            "reason": self.reason,
        }

    def applied_payload(self, params: dict[str, Any]) -> dict[str, Any]:
        return {
            "name": self.name,
            "params": params,
            "api_version": int(getattr(self.obj, "api_version", PLUGIN_API_VERSION)),
            "distribution": self.distribution,
            "version": self.version,
        }


def _default_entry_points(group: str) -> Iterable[Any]:
    return importlib.metadata.entry_points(group=group)


# Tests replace this seam instead of installing distributions.
_entry_points = _default_entry_points


def _entry_points_for(group: str) -> tuple[list[Any], list[str]]:
    try:
        return list(_entry_points(group)), []
    except Exception as exc:
        return [], [f"cannot list {group} entry points: {type(exc).__name__}: {exc}"]


def _distribution(entry_point: Any) -> tuple[Optional[str], Optional[str]]:
    dist = getattr(entry_point, "dist", None)
    if dist is None:
        return getattr(entry_point, "name", None), None
    distribution = None
    version = None
    try:
        distribution = dist.name
    except Exception:
        pass
    try:
        version = dist.version
    except Exception:
        pass
    return distribution or getattr(entry_point, "name", None), version


def _shape_ok(kind: str, obj: Any) -> Optional[str]:
    if kind == "backend":
        if not callable(getattr(obj, "models", None)):
            return "missing callable 'models(cfg, backend_model=None)'"
        if not callable(getattr(obj, "create", None)):
            return "missing callable 'create(cfg, model)'"
        caps = getattr(obj, "capabilities", None)
        if caps is not None and not callable(caps):
            # Discovery must not execute plugin code beyond importing the
            # entry-point module; the return type is validated on first use in
            # plugin_default_capabilities().
            return "'capabilities' must be callable"
        preflight = getattr(obj, "preflight", None)
        if preflight is not None and not callable(preflight):
            return "'preflight' must be callable"
        return None
    if not callable(getattr(obj, "process", None)):
        return "missing callable 'process(samples, sr, params)'"
    return None


def _builtin_backend_names() -> tuple[str, ...]:
    # Imported lazily so this module stays import-light and cycle-free.
    from .backends.registry import BUILTIN_NAMES

    return BUILTIN_NAMES


def _discover_group(kind: str, group: str) -> tuple[list[PluginRecord], list[str]]:
    entry_points, warnings = _entry_points_for(group)
    records: list[PluginRecord] = []
    for entry_point in entry_points:
        ep_name = str(getattr(entry_point, "name", "") or "")
        distribution, version = _distribution(entry_point)
        try:
            obj = entry_point.load()
        except Exception as exc:
            records.append(
                PluginRecord(
                    kind=kind,
                    name=ep_name,
                    distribution=distribution,
                    version=version,
                    status=STATUS_BROKEN,
                    reason=f"{type(exc).__name__}: {exc}",
                )
            )
            continue

        declared = getattr(obj, "name", None)
        name = declared if isinstance(declared, str) and declared else ep_name
        api_version = getattr(obj, "api_version", None)
        reason = None
        if not isinstance(declared, str) or not declared:
            # The entry-point name is only a label for diagnostics; routing
            # must use a name the plugin object itself declares.
            status = STATUS_INVALID
            reason = "plugin must declare a non-empty string 'name'"
        elif kind == "backend" and declared in _builtin_backend_names():
            status = STATUS_INVALID
            reason = f"reserved built-in backend name {declared!r}"
        elif isinstance(api_version, bool) or not isinstance(api_version, int):
            status = STATUS_INVALID
            reason = "api_version must be an integer"
        elif api_version != PLUGIN_API_VERSION:
            status = STATUS_UNSUPPORTED_API
            reason = (
                f"unsupported plugin API version {api_version}; "
                f"qtts requires {PLUGIN_API_VERSION}"
            )
        elif name != name.lower() or not _NAME_RE.match(name):
            status = STATUS_INVALID
            reason = (
                "plugin name must be lowercase and match "
                "'^[a-z][a-z0-9_-]{0,31}$'"
            )
        else:
            shape_error = _shape_ok(kind, obj)
            if shape_error is not None:
                status = STATUS_INVALID
                reason = shape_error
            else:
                status = STATUS_OK
        records.append(
            PluginRecord(
                kind=kind,
                name=name,
                distribution=distribution,
                version=version,
                status=status,
                reason=reason,
                obj=obj,
            )
        )
    return records, warnings


def discover_plugins() -> tuple[list[PluginRecord], list[str]]:
    """Discover every plugin; never raises for a broken plugin or group."""
    backend_records, backend_warnings = _discover_group("backend", BACKEND_GROUP)
    post_records, post_warnings = _discover_group("postprocess", POSTPROCESS_GROUP)
    records = backend_records + post_records
    warnings = backend_warnings + post_warnings

    groups: dict[tuple[str, str], list[PluginRecord]] = {}
    for record in records:
        groups.setdefault((record.kind, record.name), []).append(record)
    for claimants in groups.values():
        if len(claimants) > 1:
            for record in claimants:
                record.status = STATUS_DUPLICATE
                record.reason = (
                    "duplicate plugin name from multiple entry points: "
                    + ", ".join(
                        sorted(
                            f"{item.distribution or '?'}=={item.version or '?'}"
                            for item in claimants
                        )
                    )
                )
    records.sort(key=lambda item: (item.kind, item.name, item.distribution or ""))
    return records, warnings


def _matches(records: list[PluginRecord], kind: str, name: str) -> list[PluginRecord]:
    return [record for record in records if record.kind == kind and record.name == name]


def resolve_backend_plugin(
    name: str, records: Optional[list[PluginRecord]] = None
) -> PluginRecord:
    if records is None:
        records, _ = discover_plugins()
    matches = _matches(records, "backend", name)
    if not matches:
        usable = sorted(
            record.name
            for record in records
            if record.kind == "backend" and record.usable
        )
        raise UsageError(
            f"unknown backend {name!r}; built-ins are transformers, dashscope, "
            f"vllm-omni; installed plugins: {', '.join(usable) or 'none'}",
            details={"backend": name, "plugins": usable},
        )
    usable = [record for record in matches if record.usable]
    if not usable:
        record = matches[0]
        raise ModelError(
            f"backend plugin {name!r} is not usable ({record.status}): {record.reason}",
            details={
                "backend": name,
                "status": record.status,
                "plugin": record.distribution,
                "reason": record.reason,
            },
        )
    return usable[0]


def resolve_postprocess_plugin(
    name: str, records: Optional[list[PluginRecord]] = None
) -> PluginRecord:
    if records is None:
        records, _ = discover_plugins()
    matches = _matches(records, "postprocess", name)
    if not matches:
        usable = sorted(
            record.name
            for record in records
            if record.kind == "postprocess" and record.usable
        )
        raise UsageError(
            f"unknown post-process plugin {name!r}; installed plugins: "
            f"{', '.join(usable) or 'none'}",
            details={"plugin": name, "plugins": usable},
        )
    usable = [record for record in matches if record.usable]
    if not usable:
        record = matches[0]
        raise ModelError(
            f"post-process plugin {name!r} is not usable ({record.status}): {record.reason}",
            details={
                "plugin": name,
                "status": record.status,
                "distribution": record.distribution,
                "reason": record.reason,
            },
        )
    return usable[0]


def plugin_default_capabilities(record: PluginRecord) -> BackendCapabilities:
    caps = getattr(record.obj, "capabilities", None)
    if caps is None:
        return DEFAULT_BACKEND_CAPABILITIES
    if not callable(caps):
        raise ModelError(
            f"backend plugin {record.name!r} 'capabilities' must be callable",
            details={"backend": record.name},
        )
    try:
        resolved = caps()
    except Exception as exc:
        raise ModelError(
            f"backend plugin {record.name!r} capabilities() failed: "
            f"{type(exc).__name__}: {exc}",
            details={"backend": record.name},
        ) from exc
    if not isinstance(resolved, BackendCapabilities):
        raise ModelError(
            f"backend plugin {record.name!r} capabilities() must return "
            "BackendCapabilities",
            details={
                "backend": record.name,
                "returned": type(resolved).__name__,
            },
        )
    return resolved
