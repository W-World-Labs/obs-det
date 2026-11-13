import sys
from types import SimpleNamespace

import torch
from torch import nn

from omnidet.config import ModelConfig
from omnidet.models.backbone import OmniBackbone, build_backbone
from omnidet.models.pretrained_backbone import TimmFeatureBackbone
from omnidet.models.neck import EdgeMambaFPN


def test_backbone_returns_pyramid_features() -> None:
    cfg = ModelConfig(input_size=256)
    model = build_backbone(cfg)

    features = model(torch.randn(1, 3, 256, 256))

    assert features["p2"].shape == (1, cfg.p2_channels, 64, 64)
    assert features["p3"].shape == (1, cfg.p3_channels, 32, 32)
    assert features["p4"].shape == (1, cfg.p4_channels, 16, 16)
    assert features["p5"].shape == (1, cfg.p5_channels, 8, 8)


def test_timm_feature_backbone_projects_feature_pyramid(monkeypatch) -> None:
    calls: dict[str, object] = {}

    class FakeFeatureInfo:
        def channels(self) -> list[int]:
            return [5, 7, 11, 13]

        def reduction(self) -> list[int]:
            return [4, 8, 16, 32]

    class FakeTimmModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.feature_info = FakeFeatureInfo()
            self.weight = nn.Parameter(torch.ones(1))

        def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
            batch, _, height, width = x.shape
            return [
                x.new_ones(batch, 5, height // 4, width // 4),
                x.new_ones(batch, 7, height // 8, width // 8),
                x.new_ones(batch, 11, height // 16, width // 16),
                x.new_ones(batch, 13, height // 32, width // 32),
            ]

    def create_model(**kwargs):
        calls.update(kwargs)
        return FakeTimmModel()

    monkeypatch.setitem(sys.modules, "timm", SimpleNamespace(create_model=create_model))
    cfg = ModelConfig(
        input_size=128,
        backbone_variant="timm",
        backbone_name="hf-hub:omnidet/ecdet-s-backbone",
        backbone_pretrained=True,
        freeze_backbone=True,
        backbone_out_indices=(0, 1, 2, 3),
        p2_channels=6,
        p3_channels=8,
        p4_channels=10,
        p5_channels=12,
    )

    model = TimmFeatureBackbone(cfg)
    features = model(torch.randn(2, 3, 128, 128))

    assert calls == {
        "model_name": cfg.backbone_name,
        "pretrained": True,
        "features_only": True,
        "out_indices": cfg.backbone_out_indices,
    }
    assert features["p2"].shape == (2, cfg.p2_channels, 32, 32)
    assert features["p3"].shape == (2, cfg.p3_channels, 16, 16)
    assert features["p4"].shape == (2, cfg.p4_channels, 8, 8)
    assert features["p5"].shape == (2, cfg.p5_channels, 4, 4)
    assert all(not p.requires_grad for p in model.feature_extractor.parameters())


def test_edge_mamba_fpn_outputs_uniform_channels() -> None:
    cfg = ModelConfig(input_size=256, neck_channels=96)
    backbone = OmniBackbone(cfg)
    neck = EdgeMambaFPN(cfg)

    outputs = neck(backbone(torch.randn(1, 3, 256, 256)))

    assert set(outputs) == {"p2", "p3", "p4", "p5"}
    assert outputs["p2"].shape == (1, cfg.neck_channels, 64, 64)
    assert outputs["p5"].shape == (1, cfg.neck_channels, 8, 8)
