"""W11 · F1/F2 — packaging/tooling fixes from the audit.

F1: `make test-*` must invoke `python3 -m pytest`, not bare `pytest` (bare resolves to whatever
`pytest` is on PATH — a 3.8 interpreter locally, below the project's >=3.9 floor).
F2: numpy must be capped <2 so a torch<2.9 install doesn't emit the NumPy-2.x ABI warning block on
import (looks like a crash in a demo, even though the command exits 0).
"""
import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent


def test_makefile_pytest_targets_use_python_m_pytest():
    lines = (_ROOT / "Makefile").read_text().splitlines()
    offenders = [ln for ln in lines
                 if "pytest" in ln and not ln.lstrip().startswith("#")
                 and "python3 -m pytest" not in ln and "python -m pytest" not in ln]
    assert not offenders, f"bare `pytest` invocation(s) in Makefile: {offenders}"


def test_numpy_dependency_is_capped_below_2():
    pj = (_ROOT / "pyproject.toml").read_text()
    assert re.search(r"numpy>=[\d.]+,\s*<2", pj), "numpy dependency must be bounded '<2'"
