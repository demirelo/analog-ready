"""W15 — correctness-audit fixes. Acceptance oracle.

Two independent audits (an 8-lens multi-agent workflow + a Codex xhigh full-repo pass, both
verifying by execution and against primary sources) confirmed a set of correctness defects. This
oracle pins the fixes — and, just as deliberately, pins the CORRECT behaviors that a plausible-but-
wrong "fix" would break (dilation), so the audit's own false positive can never be applied later.
"""
import torch
import torch.nn as nn

from analog_ready.features import Workload, extract_op_features


# --------------------------------------------------- grouped conv (the real cost-math bug)
def test_grouped_conv_features_match_pytorch_weight_count():
    # Depthwise: PyTorch weight is (out_ch, in_ch/groups, kh, kw) = (16, 1, 3, 3) = 144 weights.
    # Pre-fix, K ignored `groups`, overcounting weights 16x and MACs 16x — a wrong break-even
    # verdict for MobileNet-style models (reuse 16x under, ENOB_req log2(144/9) ~ 4 bits over).
    dw = nn.Conv2d(16, 16, kernel_size=3, groups=16)
    f = extract_op_features(nn.Sequential(dw), Workload(batch=1, conv_positions=196))[0]
    assert f.K == 1 * 3 * 3, "K must be (in_ch/groups)*kh*kw"
    assert f.weight_numel == dw.weight.numel() == 144
    assert f.macs == 196 * 9 * 16

    grouped = nn.Conv2d(8, 16, kernel_size=3, groups=4)     # in/groups = 2 channels per group
    g = extract_op_features(nn.Sequential(grouped), Workload(batch=1, conv_positions=10))[0]
    assert g.K == 2 * 3 * 3
    assert g.weight_numel == grouped.weight.numel()


def test_dilated_conv_features_are_unchanged_by_dilation():
    # REGRESSION GUARD against the audit's own false positive: dilation spreads the taps but adds
    # NO weights and NO MACs — a dilated k=3 conv still contracts over in_ch*3*3 elements. The
    # "effective kernel" (k + (d-1)(k-1)) is receptive field, not contraction depth.
    dl = nn.Conv2d(3, 16, kernel_size=3, dilation=2)
    f = extract_op_features(nn.Sequential(dl), Workload(batch=1, conv_positions=196))[0]
    assert f.K == 3 * 3 * 3                       # NOT 3*5*5
    assert f.weight_numel == dl.weight.numel() == 432


# --------------------------------------------------- datasets RNG hygiene (claim now true)
def test_trained_classifier_preserves_the_global_rng():
    from analog_ready.datasets import trained_classifier
    s0 = torch.random.get_rng_state()
    m1, X1, y1 = trained_classifier(seed=0)
    assert torch.equal(torch.random.get_rng_state(), s0), \
        "trained_classifier must not advance the caller's global RNG (its docstring says so)"
    # and it stays byte-deterministic across calls
    m2, X2, y2 = trained_classifier(seed=0)
    assert torch.equal(X1, X2) and torch.equal(y1, y2)
    for k in m1.state_dict():
        assert torch.equal(m1.state_dict()[k], m2.state_dict()[k])


# --------------------------------------------------- ADC level-count honesty
def test_adc_level_count_is_2_to_b_minus_1_and_documented():
    from analog_ready import adc as adc_mod
    from analog_ready.adc import adc_quantize
    x = torch.linspace(-1, 1, 20001).unsqueeze(0)
    for b in (2, 3, 4):
        for fs in (None, 1.0):
            assert len(torch.unique(adc_quantize(x, bits=b, full_scale=fs))) == 2 ** b - 1
    # the docstring must state the true level count, not imply a full 2^b-code ADC
    assert "2**b - 1" in (adc_mod.__doc__ or "") or "2^b - 1" in (adc_mod.__doc__ or "")


# --------------------------------------------------- drift differential-pair disclosure
def test_drift_docstring_discloses_the_differential_pair_approximation():
    from analog_ready import drift as drift_mod
    doc = (drift_mod.__doc__ or "").lower()
    assert "differential" in doc, \
        "signed-weight drift understates drift of near-zero weights vs a G+/G- pair; disclose it"


# --------------------------------------------------- photonic profile internal consistency
def test_photonic_mac_energy_matches_its_cited_source():
    from analog_ready.profiles import load_profile
    v = load_profile("photonic_clements_mzi").fields["mac_energy_pj"].value
    assert abs(v - 0.033) < 1e-9, "profile comment cites TFLN ~33 fJ/MAC; the value must match it"


# --------------------------------------------------- cost-model prose honesty
def test_cost_model_docstring_does_not_attribute_our_ratio_to_lightcode():
    from analog_ready import cost_model as cm
    assert "reports up to ~10x" not in (cm.__doc__ or ""), \
        "the ~10x ratio is derived from coefficients at small arrays, not a LightCode-reported figure"
