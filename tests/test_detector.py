import torch

from omnidet.config import ModelConfig
from omnidet.models.detector import OmniDet


def test_omnidet_forward_shapes() -> None:
    cfg = ModelConfig(input_size=256, accuracy_input_size=320, num_queries=50)
    model = OmniDet(cfg)

    outputs = model(torch.randn(2, 3, 256, 256))

    assert outputs["pred_logits"].shape == (2, 50, cfg.num_classes + 1)
    assert outputs["pred_boxes"].shape == (2, 50, 4)
    assert torch.all(outputs["pred_boxes"] >= 0)
    assert torch.all(outputs["pred_boxes"] <= 1)


def test_omnidet_supports_ssm_lite_neck_variant() -> None:
    cfg = ModelConfig(input_size=128, num_queries=20, neck_channels=64, neck_variant="ssm_lite")
    model = OmniDet(cfg)

    outputs = model(torch.randn(1, 3, 128, 128))

    assert outputs["pred_logits"].shape == (1, 20, cfg.num_classes + 1)
    assert outputs["pred_boxes"].shape == (1, 20, 4)
