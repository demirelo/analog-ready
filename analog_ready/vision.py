"""W6 · F3 — the optional real-vision path. Real ResNet-18 (torchvision pretrained weights) +
a folder of the user's OWN labelled images (ImageFolder), so accuracy is measured on a real model
and real (locally-held) images — your images/labels never leave the machine.

NETWORK DISCLOSURE (honest provenance): `load_resnet18()` calls torchvision with pretrained ImageNet
weights, which DOWNLOADS ~45 MB from PyTorch's CDN on first use, cached under ~/.cache/torch/hub. So
`eval --model resnet18` is the one verb that touches the network; an air-gapped run must pre-populate
that cache. `doctor` reports this exception. torchvision is an OPTIONAL extra
(`pip install analog-ready[vision]`); a missing torchvision must surface as a clear, catchable
ImportError-style message, never an uncaught traceback from deep inside this module."""
from __future__ import annotations

import torch.nn as nn


def _require_torchvision():
    try:
        import torchvision
    except ImportError as e:
        raise ImportError(
            "vision support requires the optional 'torchvision' dependency; "
            "install it with `pip install analog-ready[vision]` (or `pip install torchvision`)."
        ) from e
    return torchvision


def load_resnet18() -> nn.Module:
    """Real torchvision ResNet-18 (ImageNet-pretrained weights). Raises a clear ImportError if
    torchvision is not installed."""
    torchvision = _require_torchvision()
    try:
        weights = torchvision.models.ResNet18_Weights.DEFAULT
    except AttributeError:
        # older torchvision: no Weights enum, pretrained=True is the only path
        return torchvision.models.resnet18(pretrained=True)
    return torchvision.models.resnet18(weights=weights)


def load_labeled_folder(data_dir: str):
    """Load a torchvision ImageFolder at `data_dir` (one subdirectory per class) through a standard
    resize/normalize transform, returning (inputs, labels): a single batched 4-D tensor and a 1-D
    label tensor. Raises a clear ImportError if torchvision is absent; raises whatever ImageFolder
    raises (e.g. FileNotFoundError) for a missing/empty/unreadable directory — callers (the CLI)
    are responsible for turning that into a clean non-zero exit rather than a raw traceback."""
    torchvision = _require_torchvision()
    import torch
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = torchvision.datasets.ImageFolder(root=data_dir, transform=transform)
    if len(dataset) == 0:
        raise ValueError(f"no labelled images found under {data_dir!r}")
    images, labels = zip(*[dataset[i] for i in range(len(dataset))])
    inputs = torch.stack(images, dim=0)
    label_tensor = torch.tensor(labels, dtype=torch.long)
    return inputs, label_tensor
