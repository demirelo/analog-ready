"""analog-ready — Analog Robustness CI for AI accelerators (AIMC-first; photonic MZI demo included).

The top-level import is deliberately light: it pulls in NO optional/heavy backends (torchonn,
aihwkit, brevitas, torchao). Optional backends are discovered lazily by the registry and report
availability without being imported here.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
