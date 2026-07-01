"""W2 cost-model spine — per-op feature extraction (FLOPs / shape / contraction depth) from a
model. This is the model-intrinsic input the cost model and score consume; pure shape math, no
profile and no forward pass required."""
import torch.nn as nn


def _mlp():
    return nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 32))


def test_extract_linear_features():
    from analog_ready.features import extract_op_features, Workload

    feats = extract_op_features(_mlp(), Workload(tokens=10, batch=2))
    lin = [f for f in feats if f.op_type == "linear"]
    assert len(lin) == 2  # two Linear; ReLU is not an eligible op
    f0 = lin[0]
    assert (f0.K, f0.N) == (64, 128)        # in_features, out_features
    assert f0.M == 20                        # tokens * batch
    assert f0.macs == 20 * 64 * 128
    assert f0.flops == 2 * f0.macs
    assert f0.weight_numel == 64 * 128
    assert f0.contraction_dim == 64          # the dot-product depth the ENOB bound uses


def test_conv_features_unfold_the_contraction():
    from analog_ready.features import extract_op_features, Workload

    feats = extract_op_features(nn.Sequential(nn.Conv2d(3, 16, kernel_size=3)),
                                Workload(batch=1, conv_positions=196))
    c = [f for f in feats if f.op_type == "conv2d"][0]
    assert c.K == 3 * 3 * 3       # in_ch * kh * kw  (im2col contraction)
    assert c.N == 16              # out channels
    assert c.M == 196             # output positions
    assert c.macs == 196 * 27 * 16


def test_weight_reuse_is_macs_over_weights():
    from analog_ready.features import extract_op_features, Workload

    f = extract_op_features(_mlp(), Workload(tokens=100, batch=1))[0]
    # reuse = how many MACs amortize one weight load = M for a Linear
    assert f.weight_reuse == f.macs / f.weight_numel == 100


def test_bare_top_level_module_is_extracted():
    from analog_ready.features import extract_op_features, Workload

    feats = extract_op_features(nn.Linear(8, 4), Workload(tokens=1, batch=1))
    assert len(feats) == 1 and feats[0].op_type == "linear"
