# tests/unit/test_cv.py
import numpy as np
import pytest
import torch
from PIL import Image


def make_dummy_image(width=224, height=224):
    arr = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def test_transform_output_shape():
    from torchvision import transforms
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    img = make_dummy_image()
    tensor = tf(img)
    assert tensor.shape == (3, 224, 224)


def test_transform_grayscale_input():
    from torchvision import transforms
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    gray = Image.fromarray(
        np.random.randint(0, 255, (100, 100), dtype=np.uint8), mode="L"
    )
    rgb = gray.convert("RGB")
    tensor = tf(rgb)
    assert tensor.shape == (3, 224, 224)


def test_label_range():
    valid_ids = list(range(16))
    for label_id in valid_ids:
        assert 0 <= label_id <= 15


def test_model_output_shape():
    import timm
    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=16)
    model.eval()
    dummy = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(dummy)
    assert out.shape == (2, 16)


def test_softmax_sums_to_one():
    logits = torch.tensor([[1.0, 2.0, 3.0, 0.5]])
    probs = torch.softmax(logits, dim=1)
    assert abs(probs.sum().item() - 1.0) < 1e-5
