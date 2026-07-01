"""THE SPINE (must not slip): the break-even envelope (X converter-energy, Y reuse, Z max ENOB_req)
and the sensitivity sweep that FLIPS an op's verdict — the buyer-facing 'where analog stops winning'.
A report without a flip is just a noise-injection benchmark."""
import torch.nn as nn


def _feat(k=1024, n=1024, tokens=512):
    from analog_ready.features import extract_op_features, Workload

    return extract_op_features(nn.Linear(k, n), Workload(tokens=tokens, batch=1))[0]


def test_break_even_reports_xyz_thresholds_and_actuals():
    from analog_ready.profiles import load_profile
    from analog_ready.breakeven import break_even

    be = break_even(_feat(), load_profile("aimc_sram_8bit"))
    # thresholds (the envelope) ...
    assert be.y_reuse_threshold is not None and be.z_enob_avail is not None
    # ... and this op's actual position against them
    assert be.reuse is not None and be.enob_req is not None
    assert isinstance(be.favorable, bool)
    # per-constraint verdicts are exposed, not just the aggregate
    assert isinstance(be.precision_ok, bool) and isinstance(be.reuse_ok, bool)
    assert isinstance(be.converter_ok, bool)


def test_enob_sweep_flips_the_verdict():
    from analog_ready.profiles import load_profile
    from analog_ready.breakeven import sweep

    # ENOB_req = input_bits(4) + 0.5*log2(K=1024) = 4 + 5 = 9. Sweep enob_avail across 9.
    pts = sweep(_feat(), load_profile("aimc_sram_8bit"),
                param="enob_avail", values=[6.0, 8.0, 9.0, 10.0, 12.0])
    precision = [p.precision_ok for p in pts]
    favorable = [p.favorable for p in pts]
    # the precision constraint must flip false -> true as the ADC gets better
    assert precision[0] is False and precision[-1] is True
    # and the overall verdict actually changes across the swept range (the DoD)
    assert any(favorable) and not all(favorable)


def test_low_reuse_op_fails_the_reuse_constraint():
    from analog_ready.profiles import load_profile
    from analog_ready.breakeven import break_even

    # a one-shot projection (M=1) barely amortizes the analog weight programming cost
    be = break_even(_feat(tokens=1), load_profile("aimc_sram_8bit"))
    assert be.reuse == 1
    assert be.reuse_ok is False
