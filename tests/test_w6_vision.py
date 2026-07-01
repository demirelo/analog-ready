"""W6 · F3 — the optional real-vision path. Real ResNet-18 (torchvision weights) + a folder of the
user's OWN labelled images (data stays local). torchvision is an OPTIONAL extra, so these tests
importorskip it exactly like the torchonn backend — the capability ships; the core gate stays
network-free."""
from __future__ import annotations

import pytest
import torch


@pytest.mark.vision
def test_load_resnet18_real_weights_and_accuracy_path():
    pytest.importorskip("torchvision")
    from analog_ready.accuracy import top1_accuracy
    from analog_ready.vision import load_resnet18
    model = load_resnet18()
    assert isinstance(model, torch.nn.Module)
    # exercise the REAL-model accuracy path on a tiny random labelled batch (1000 ImageNet classes)
    x = torch.randn(2, 3, 64, 64)
    y = torch.randint(0, 1000, (2,))
    acc = top1_accuracy(model, x, y)
    assert 0.0 <= acc <= 1.0


@pytest.mark.vision
def test_load_labeled_folder_reads_an_image_folder(tmp_path):
    pytest.importorskip("torchvision")
    from PIL import Image
    from analog_ready.vision import load_labeled_folder
    # build a tiny ImageFolder: two classes, one image each
    for cls in ("a", "b"):
        d = tmp_path / cls
        d.mkdir()
        Image.new("RGB", (64, 64), color=(0, 0, 0)).save(d / "img.png")
    inputs, labels = load_labeled_folder(str(tmp_path))
    assert inputs.ndim == 4 and inputs.shape[0] == labels.shape[0] == 2
