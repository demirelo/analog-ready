"""W7 · F1 — faithful analog weight-programming noise (immutable acceptance oracle).

Replaces the W1 arbitrary additive-output sigma with the standard analog-inference decomposition:
programming noise that scales with the weight magnitude (proportional / conductance-dependent) plus
a magnitude-independent read/thermal floor. Deterministic, non-mutating, identity at zero.
"""
import statistics

import torch

from analog_ready.device_noise import program_noise


def test_zero_sigmas_are_the_identity():
    w = torch.randn(8, 8)
    out = program_noise(w, prop_sigma=0.0, read_sigma=0.0, seed=0)
    assert torch.equal(out, w)


def test_deterministic_given_seed():
    w = torch.randn(4, 4)
    a = program_noise(w, prop_sigma=0.1, read_sigma=0.05, seed=3)
    b = program_noise(w, prop_sigma=0.1, read_sigma=0.05, seed=3)
    assert torch.equal(a, b)


def test_does_not_mutate_the_input_weight():
    w = torch.randn(4, 4)
    w0 = w.clone()
    program_noise(w, prop_sigma=0.3, read_sigma=0.1, seed=1)
    assert torch.equal(w, w0)


def test_programming_noise_scales_with_weight_magnitude():
    # prop noise only: a large-|w| entry must be perturbed far more than a small-|w| entry.
    w = torch.tensor([[10.0, 0.01]])
    diffs_big, diffs_small = [], []
    for s in range(200):
        out = program_noise(w, prop_sigma=0.2, read_sigma=0.0, seed=s)
        diffs_big.append(float(out[0, 0] - w[0, 0]))
        diffs_small.append(float(out[0, 1] - w[0, 1]))
    assert statistics.pstdev(diffs_big) > 10.0 * statistics.pstdev(diffs_small)


def test_read_noise_is_magnitude_independent():
    # read floor only: perturbation scale is the same regardless of |w|.
    w = torch.tensor([[10.0, 0.01]])
    diffs_big, diffs_small = [], []
    for s in range(200):
        out = program_noise(w, prop_sigma=0.0, read_sigma=0.1, seed=s)
        diffs_big.append(float(out[0, 0] - w[0, 0]))
        diffs_small.append(float(out[0, 1] - w[0, 1]))
    assert abs(statistics.pstdev(diffs_big) - statistics.pstdev(diffs_small)) < 0.05


def test_does_not_clobber_global_rng():
    # Prepare the weight OUTSIDE the timed window: `torch.randn(...)` as a call argument is evaluated
    # before program_noise is entered, so passing it inline would itself consume global RNG and no
    # callee could restore it. The real contract is that program_noise's OWN draws don't advance the
    # caller's global RNG.
    w = torch.randn(4, 4)
    torch.manual_seed(12345)
    before = torch.randn(3)
    torch.manual_seed(12345)
    program_noise(w, prop_sigma=0.2, read_sigma=0.1, seed=7)
    after = torch.randn(3)
    assert torch.equal(before, after)
