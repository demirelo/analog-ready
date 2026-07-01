"""The cost model splits energy into compute + ADC/DAC conversion + DRAM movement (LightCode
framing). Every coefficient carries {value, unit, source, uncertainty}, and for realistic
coefficients converters+memory dominate the analog compute energy."""
import torch.nn as nn


def _feat(k=512, n=512, tokens=64):
    from analog_ready.features import extract_op_features, Workload

    return extract_op_features(nn.Linear(k, n), Workload(tokens=tokens, batch=1))[0]


def test_op_cost_breakdown_sums_and_is_auditable():
    from analog_ready.profiles import load_profile
    from analog_ready.cost_model import estimate_op

    c = estimate_op(_feat(), load_profile("aimc_pcm_4bit"))
    assert c.compute_pj > 0 and c.conversion_pj > 0 and c.dram_pj > 0
    assert abs(c.total_pj - (c.compute_pj + c.conversion_pj + c.dram_pj)) < 1e-6
    # every coefficient the estimate used is traceable to a value/unit/source/uncertainty
    assert c.coefficients
    for coeff in c.coefficients.values():
        assert coeff.unit and coeff.source and coeff.uncertainty


def test_converters_and_memory_dominate_compute():
    from analog_ready.profiles import load_profile
    from analog_ready.cost_model import estimate_op

    c = estimate_op(_feat(), load_profile("aimc_pcm_4bit"))
    # the honest story: the analog MAC is cheap, but moving/converting the data is not
    assert (c.conversion_pj + c.dram_pj) > c.compute_pj


def test_estimate_aggregates_total_over_ops():
    from analog_ready.profiles import load_profile
    from analog_ready.cost_model import estimate
    from analog_ready.features import extract_op_features, Workload

    feats = extract_op_features(nn.Sequential(nn.Linear(256, 256), nn.Linear(256, 64)),
                                Workload(tokens=32, batch=1))
    rep = estimate(feats, load_profile("aimc_pcm_4bit"))
    assert len(rep.ops) == 2
    assert abs(rep.total_pj - sum(o.total_pj for o in rep.ops)) < 1e-6
