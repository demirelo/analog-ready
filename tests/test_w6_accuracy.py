"""W6 · F1+F2 — task-accuracy harness + an offline-trained fixture. The product's headline is a real
accuracy number, not a proxy: top-1 on real labels, clean vs under-noise. The fixture is a tiny
model that reaches REAL high accuracy on a deterministic separable task (no downloads), so the gate
is meaningful offline. New-module imports are inside test bodies so collection stays clean."""
from __future__ import annotations

import torch
import torch.nn as nn


def test_top1_accuracy_is_argmax_correct():
    from analog_ready.accuracy import top1_accuracy

    class Logits(nn.Module):
        def forward(self, x):
            return x

    m = Logits()
    x = torch.tensor([[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 1.0]])  # argmax 0,1,2
    assert top1_accuracy(m, x, torch.tensor([0, 1, 2])) == 1.0
    assert abs(top1_accuracy(m, x, torch.tensor([0, 0, 0])) - 1.0 / 3.0) < 1e-6


def test_trained_classifier_is_really_accurate_and_deterministic():
    from analog_ready.accuracy import top1_accuracy
    from analog_ready.datasets import trained_classifier
    m, X, y = trained_classifier(seed=0)
    acc = top1_accuracy(m, X, y)
    assert acc >= 0.9                                    # a REAL accuracy, not chance
    m2, X2, y2 = trained_classifier(seed=0)
    assert torch.equal(X, X2) and torch.equal(y, y2)
    assert top1_accuracy(m2, X2, y2) == acc              # deterministic


def test_accuracy_under_noise_degrades_and_is_deterministic():
    from analog_ready.accuracy import accuracy_under_noise, top1_accuracy
    from analog_ready.datasets import trained_classifier
    m, X, y = trained_classifier(seed=0)
    clean = top1_accuracy(m, X, y)
    # sigma=0 reproduces clean exactly (the wrapper is a no-op at sigma=0)
    assert accuracy_under_noise(m, X, y, sigma=0.0, draws=2, seed=0) == clean
    # a huge sigma drives predictions toward chance, so accuracy must fall below clean regardless of
    # the fixture's output scale; and the result is reproducible.
    noised = accuracy_under_noise(m, X, y, sigma=100.0, draws=4, seed=0)
    assert noised < clean
    assert accuracy_under_noise(m, X, y, sigma=100.0, draws=4, seed=0) == noised


def test_accuracy_report_shape_and_drop():
    from analog_ready.accuracy import accuracy_report
    from analog_ready.datasets import trained_classifier
    m, X, y = trained_classifier(seed=0)
    r = accuracy_report(m, X, y, sigma=100.0, draws=4, seed=0)
    assert {"clean", "under_noise", "drop", "sigma"}.issubset(r)
    assert abs(r["drop"] - (r["clean"] - r["under_noise"])) < 1e-9
    assert r["drop"] > 0.0


def test_accuracy_does_not_mutate_training_mode():
    from analog_ready.accuracy import accuracy_under_noise, top1_accuracy
    from analog_ready.datasets import trained_classifier
    m, X, y = trained_classifier(seed=0)
    m.train()
    top1_accuracy(m, X, y)
    accuracy_under_noise(m, X, y, sigma=1.0, draws=2, seed=0)
    assert m.training is True
