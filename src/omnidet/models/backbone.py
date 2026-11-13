import torch
from torch import nn

from omnidet.config import ModelConfig
from omnidet.models.blocks import AntiAliasStem, ConvBNAct, GatedConvBlock
from omnidet.models.pretrained_backbone import TimmFeatureBackbone


class OmniBackbone(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.stem = AntiAliasStem(cfg.p2_channels)
        self.p2 = nn.Sequential(
            GatedConvBlock(cfg.p2_channels),
            GatedConvBlock(cfg.p2_channels),
        )
        self.p3 = nn.Sequential(
            ConvBNAct(cfg.p2_channels, cfg.p3_channels, kernel_size=3, stride=2),
            GatedConvBlock(cfg.p3_channels),
        )
        self.p4 = nn.Sequential(
            ConvBNAct(cfg.p3_channels, cfg.p4_channels, kernel_size=3, stride=2),
            GatedConvBlock(cfg.p4_channels),
        )
        self.p5 = nn.Sequential(
            ConvBNAct(cfg.p4_channels, cfg.p5_channels, kernel_size=3, stride=2),
            GatedConvBlock(cfg.p5_channels),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        p2 = self.p2(self.stem(x))
        p3 = self.p3(p2)
        p4 = self.p4(p3)
        p5 = self.p5(p4)
        return {"p2": p2, "p3": p3, "p4": p4, "p5": p5}


def build_backbone(cfg: ModelConfig) -> nn.Module:
    if cfg.backbone_variant == "omnidet":
        return OmniBackbone(cfg)
    if cfg.backbone_variant == "timm":
        return TimmFeatureBackbone(cfg)
    raise ValueError("Unknown backbone_variant. Choices: omnidet, timm")
