"""W8 · F1 — conductance drift over time (immutable acceptance oracle).

PCM/ReRAM conductances decay as a power law G(t) = G0 * (t/t0)^(-nu). A UNIFORM decay (same nu for
every weight) is just a global scale — cosine-invisible / drift-compensatable — so the real residual
degradation comes from nu VARIABILITY across devices: per-weight nu_i = nu + sigma_nu * eps_i.
Deterministic, non-mutating, identity at t == t0.
"""
import torch

from analog_ready.drift import drift_weight


def test_no_time_elapsed_is_identity():
    w = torch.randn(8, 8)
    out = drift_weight(w, nu=0.06, t_s=1.0, t0=1.0, sigma_nu=0.05, seed=0)
    assert torch.allclose(out, w)   # t_s == t0 -> factor 1 for every weight


def test_zero_nu_and_zero_spread_is_identity():
    w = torch.randn(8, 8)
    out = drift_weight(w, nu=0.0, t_s=1e6, t0=1.0, sigma_nu=0.0, seed=0)
    assert torch.allclose(out, w)


def test_uniform_drift_is_a_power_law_decay():
    w = torch.full((4, 4), 2.0)
    out = drift_weight(w, nu=0.1, t_s=100.0, t0=1.0, sigma_nu=0.0, seed=0)
    factor = (100.0 / 1.0) ** (-0.1)
    assert torch.allclose(out, w * factor, atol=1e-5)   # sigma_nu=0 -> every weight same factor


def test_drift_reduces_conductance_magnitude_over_time():
    w = torch.randn(6, 6)
    out = drift_weight(w, nu=0.06, t_s=1e5, t0=1.0, sigma_nu=0.0, seed=0)
    assert float(out.abs().sum()) < float(w.abs().sum())   # conductance decays for t > t0


def test_nu_spread_makes_drift_nonuniform():
    w = torch.full((4, 4), 3.0)   # identical weights -> any spread in output is from nu variability
    out = drift_weight(w, nu=0.06, t_s=1e4, t0=1.0, sigma_nu=0.05, seed=0)
    assert float(out.std()) > 1e-4


def test_deterministic_and_does_not_mutate_input():
    w = torch.randn(4, 4)
    w0 = w.clone()
    a = drift_weight(w, nu=0.06, t_s=1e4, t0=1.0, sigma_nu=0.05, seed=1)
    b = drift_weight(w, nu=0.06, t_s=1e4, t0=1.0, sigma_nu=0.05, seed=1)
    assert torch.equal(a, b)
    assert torch.equal(w, w0)


def test_does_not_clobber_global_rng():
    w = torch.randn(4, 4)   # built outside the timed window on purpose
    torch.manual_seed(999)
    before = torch.randn(3)
    torch.manual_seed(999)
    drift_weight(w, nu=0.06, t_s=1e4, t0=1.0, sigma_nu=0.05, seed=2)
    after = torch.randn(3)
    assert torch.equal(before, after)
