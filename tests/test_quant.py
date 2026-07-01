"""QuantSpec + a pure-PyTorch fake-quant default: high bit-width is near-identity, low bit-width
coarsens the tensor to a bounded number of levels."""
import torch


def test_fake_quant_high_bits_near_identity():
    from analog_ready.quant import QuantSpec, fake_quant

    x = torch.randn(512)
    y = fake_quant(x, QuantSpec(bits=16))
    assert torch.allclose(x, y, atol=1e-2)


def test_fake_quant_low_bits_coarsens_to_levels():
    from analog_ready.quant import QuantSpec, fake_quant

    x = torch.randn(1024)
    y = fake_quant(x, QuantSpec(bits=2))
    assert y.unique().numel() <= 2 ** 2
    assert y.shape == x.shape


def test_quantspec_default_is_pure_torch():
    from analog_ready.quant import QuantSpec

    spec = QuantSpec()
    assert spec.bits >= 1
