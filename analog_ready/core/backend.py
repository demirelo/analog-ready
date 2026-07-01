"""The backend registry. Default backends (mzi_pure, aimc_simple) are pure-PyTorch and always
available. Optional backends (mzi_torchonn, aimc_aihwkit) import their heavy dependency LAZILY —
a missing dep surfaces as available=False+reason, never an exception, and is never imported by
`import analog_ready`. Discovery is a static internal table (the package runs from source, so we
do not depend on installed entry points)."""
from __future__ import annotations

import importlib
import importlib.util
from dataclasses import dataclass


@dataclass(frozen=True)
class BackendStatus:
    name: str
    available: bool
    reason: str | None = None


class NoiseQuantBackend:
    """Interface a backend implements. Subclasses provide real bodies."""

    name = "base"

    def apply_noise(self, x, **kw):
        raise NotImplementedError

    def quantize_weights(self, w, **kw):
        raise NotImplementedError

    def forward_matmul(self, x, w, **kw):
        raise NotImplementedError


# name -> (module path, class name, optional-dep or None). Loading is lazy: the module is imported
# only when status()/get() runs, so `import analog_ready` never pulls torchonn/aihwkit.
_REGISTRY: dict[str, tuple[str, str, str | None]] = {
    "mzi_pure": ("analog_ready.backends.mzi_pure", "MZIPureBackend", None),
    "aimc_simple": ("analog_ready.backends.aimc_simple", "AIMCSimpleBackend", None),
    "mzi_torchonn": ("analog_ready.backends.mzi_torchonn", "MZITorchonnBackend", "torchonn"),
    "aimc_aihwkit": ("analog_ready.backends.aimc_aihwkit", "AIMCAihwkitBackend", "aihwkit"),
}


def _load(name: str):
    """Return (cls, None) on success or (None, reason) if the backend or its dep is unavailable."""
    module_path, cls_name, dep = _REGISTRY[name]
    if dep is not None:
        if importlib.util.find_spec(dep) is None:
            return None, f"optional dependency {dep!r} is not installed"
    try:
        module = importlib.import_module(module_path)
        return getattr(module, cls_name), None
    except Exception as exc:  # import-time failure
        if dep is None:  # a default backend MUST always load — surface this, don't mask as graceful
            return None, f"HARD ERROR (default backend failed to import): {type(exc).__name__}: {exc}"
        return None, f"optional dependency {dep!r} present but failed to import: {type(exc).__name__}: {exc}"


def status() -> list[BackendStatus]:
    rows = []
    for name in _REGISTRY:
        cls, reason = _load(name)
        rows.append(BackendStatus(name=name, available=cls is not None, reason=reason))
    return rows


def get(name: str):
    """The backend class if available, else None."""
    if name not in _REGISTRY:
        return None
    cls, _ = _load(name)
    return cls


def available() -> list[str]:
    return [r.name for r in status() if r.available]
