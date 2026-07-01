"""Core import must succeed with ZERO optional deps, and must NOT eagerly import optional libs.
The backend registry must report optional backends as unavailable+reason without raising."""
import sys


def test_import_does_not_pull_optional_libs():
    import analog_ready  # noqa: F401
    assert "torchonn" not in sys.modules, "core import must not eagerly import torchonn"
    assert "aihwkit" not in sys.modules, "core import must not eagerly import aihwkit"


def test_registry_status_lists_default_and_optional():
    from analog_ready.core.backend import status, BackendStatus

    rows = status()
    assert rows and all(isinstance(r, BackendStatus) for r in rows)
    by_name = {r.name: r for r in rows}

    # pure-PyTorch defaults are always available
    assert by_name["mzi_pure"].available is True
    assert by_name["aimc_simple"].available is True

    # at least one optional backend is present but unavailable, with a non-empty reason
    optional = [r for r in rows if r.name in ("mzi_torchonn", "aimc_aihwkit")]
    assert optional, "optional backends should be registered (just unavailable)"
    unavailable = [r for r in optional if not r.available]
    assert unavailable, "optional backends must report available=False without their dep"
    assert all(r.reason for r in unavailable), "an unavailable backend must explain why"
