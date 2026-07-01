"""W6 · F2 — a tiny, fully offline, DETERMINISTIC "trained" fixture. `trained_classifier(seed)`
builds a well-separated synthetic blob-classification task (no downloads) and trains a tiny MLP on
it to convergence with a fixed seed, so the fixture reaches REAL >=0.9 top-1 accuracy — not a
proxy — and (because the blobs are well separated but the model's margins are finite) its accuracy
falls under large injected noise. Everything here uses local torch.Generator objects, never the
global RNG, so repeated calls with the same seed are byte-identical."""
from __future__ import annotations

import torch
import torch.nn as nn


N_CLASSES = 4
N_FEATURES = 6
N_PER_CLASS = 40
BLOB_SPACING = 8.0   # centers spaced far apart relative to blob std => well-separated task
BLOB_STD = 0.5


def _make_blobs(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """N_CLASSES Gaussian blobs in N_FEATURES dims, centers placed on a simplex-like spread so every
    pair of classes is far apart relative to the blob std. Deterministic via a local Generator."""
    g = torch.Generator()
    g.manual_seed(seed)
    centers = torch.zeros(N_CLASSES, N_FEATURES)
    for c in range(N_CLASSES):
        centers[c, c % N_FEATURES] = BLOB_SPACING * (1 + c // N_FEATURES)
        centers[c, (c + 1) % N_FEATURES] = -BLOB_SPACING * 0.5 * (c % 2)

    xs, ys = [], []
    for c in range(N_CLASSES):
        pts = centers[c] + BLOB_STD * torch.randn(N_PER_CLASS, N_FEATURES, generator=g)
        xs.append(pts)
        ys.append(torch.full((N_PER_CLASS,), c, dtype=torch.long))
    X = torch.cat(xs, dim=0)
    y = torch.cat(ys, dim=0)

    # deterministic shuffle so classes are interleaved (order doesn't matter for accuracy, but
    # keeps the fixture from looking artificially block-sorted).
    perm = torch.randperm(X.shape[0], generator=g)
    return X[perm], y[perm]


class _TinyMLP(nn.Module):
    """Small enough that GaussianNoise at a large sigma reliably swamps its O(1)-scaled logits, but
    expressive enough to separate the blobs perfectly once trained."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(N_FEATURES, 16)
        self.fc2 = nn.Linear(16, N_CLASSES)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        return self.fc2(h)


def trained_classifier(*, seed: int = 0) -> tuple[nn.Module, torch.Tensor, torch.Tensor]:
    """(model, X, y): a tiny MLP trained to convergence on a deterministic, well-separated synthetic
    blob-classification task. top1_accuracy(model, X, y) >= 0.9 by construction (the task is
    trivially separable and training runs to convergence). Fully deterministic: identical X, y, and
    trained weights (hence identical accuracy) across calls with the same seed. Output logits are
    O(1)-scaled (no manual rescaling), so injected noise at a large sigma actually changes
    predictions."""
    X, y = _make_blobs(seed)

    g = torch.Generator()
    g.manual_seed(seed + 1)
    # nn.Linear's DEFAULT __init__ draws from the global RNG (kaiming init) even though every
    # parameter is overwritten below — fork_rng isolates that draw so this function truly never
    # advances the caller's global RNG (the docstring's claim, previously false).
    with torch.random.fork_rng():
        model = _TinyMLP()
    # deterministic parameter init via the local generator (never the global RNG)
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.empty(p.shape).normal_(mean=0.0, std=0.5, generator=g))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for _ in range(300):
        optimizer.zero_grad()
        logits = model(X)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        if loss.item() < 1e-4:
            break
    model.eval()
    return model, X, y
