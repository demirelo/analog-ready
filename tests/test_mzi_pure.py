"""The pure-PyTorch Clements/Reck MZI engine builds a genuine N x N unitary from 2x2 MZI blocks,
including non-power-of-2 N. (UU^H == I)."""
import pytest
import torch


@pytest.mark.parametrize("n", [2, 3, 4, 5, 7])
def test_mzi_unitary_is_unitary_for_arbitrary_n(n):
    from analog_ready.backends.mzi_pure import mzi_unitary

    torch.manual_seed(n)
    u = mzi_unitary(n)  # default: random phases
    assert u.shape == (n, n)
    eye = torch.eye(n, dtype=u.dtype)
    prod = u @ u.conj().transpose(-1, -2)
    assert torch.allclose(prod, eye, atol=1e-5), f"UU^H not identity for N={n}"


def test_mzi_backend_is_registered_and_available():
    from analog_ready.core.backend import get

    assert get("mzi_pure") is not None


@pytest.mark.mzi_torchonn
def test_pure_matches_torchonn_within_1e4():
    """Only runs when the optional torchonn dep is installed (Image A); skipped in the core gate."""
    torchonn = pytest.importorskip("torchonn")  # noqa: F841
    from analog_ready.backends.mzi_pure import mzi_unitary
    from analog_ready.backends.mzi_torchonn import reference_unitary

    n = 4
    torch.manual_seed(0)
    params = torch.rand(n * (n - 1) // 2, 2)
    u_pure = mzi_unitary(n, params)
    u_ref = reference_unitary(n, params)
    assert torch.allclose(u_pure, u_ref, atol=1e-4)
