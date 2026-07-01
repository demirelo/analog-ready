"""The registry degrades gracefully: a missing optional dependency surfaces as available=False
with a reason, never an exception, and is discovered lazily (no eager optional import)."""


def test_status_never_raises():
    from analog_ready.core.backend import status

    rows = status()  # must not raise even though torchonn/aihwkit are absent
    assert isinstance(rows, list)


def test_missing_optional_reports_reason_mentioning_module():
    from analog_ready.core.backend import status

    by_name = {r.name: r for r in status()}
    tor = by_name.get("mzi_torchonn")
    assert tor is not None and tor.available is False
    assert "torchonn" in tor.reason.lower()


def test_get_available_returns_backend_unavailable_returns_none():
    from analog_ready.core.backend import get

    pure = get("mzi_pure")
    assert pure is not None  # a class or instance implementing the backend interface

    # an unavailable backend resolves to None rather than raising
    assert get("mzi_torchonn") is None


def test_backend_interface_shape():
    from analog_ready.core.backend import get

    b = get("aimc_simple")
    inst = b() if isinstance(b, type) else b
    for method in ("apply_noise", "quantize_weights", "forward_matmul"):
        assert callable(getattr(inst, method))
