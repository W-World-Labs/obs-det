import torch
from torch import nn
import torch.nn.functional as F

from omnidet.config import ModelConfig
from omnidet.models.blocks import ConvBNAct, GatedConvBlock
from omnidet.models.features import FeatureSpec


class LowRankContextBlock(nn.Module):
    """Exportable low-rank context mixer used as an SSM-lite research path."""

    def __init__(self, channels: int, rank: int = 16) -> None:
        super().__init__()
        rank = min(rank, channels)
        self.reduce = nn.Conv2d(channels, rank, kernel_size=1)
        self.dwconv = nn.Conv2d(rank, rank, kernel_size=7, padding=3, groups=rank)
        self.expand = nn.Conv2d(rank, channels, kernel_size=1)
        self.gate = nn.Conv2d(channels, channels, kernel_size=1)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        context = self.expand(self.act(self.dwconv(self.reduce(x))))
        return x + context * torch.sigmoid(self.gate(x))


class EdgeMambaFPN(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        channels = cfg.neck_channels
        self.variant = cfg.neck_variant
        # The neck normalizes the backbone pyramid into this contract; every
        # task head binds to the FeatureSpec rather than to a specific backbone.
        self.feature_spec = FeatureSpec.pyramid(channels=channels)
        self.lateral = nn.ModuleDict(
            {
                "p2": ConvBNAct(cfg.p2_channels, channels, kernel_size=1),
                "p3": ConvBNAct(cfg.p3_channels, channels, kernel_size=1),
                "p4": ConvBNAct(cfg.p4_channels, channels, kernel_size=1),
                "p5": ConvBNAct(cfg.p5_channels, channels, kernel_size=1),
            }
        )
        block_type = LowRankContextBlock if cfg.neck_variant == "ssm_lite" else GatedConvBlock
        self.fuse = nn.ModuleDict(
            {
                "p2": block_type(channels),
                "p3": block_type(channels),
                "p4": block_type(channels),
                "p5": block_type(channels),
            }
        )
        self.edge_path = nn.Sequential(
            ConvBNAct(channels, channels, kernel_size=3, groups=channels),
            ConvBNAct(channels, channels, kernel_size=1),
        )

    def forward(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        p5 = self.lateral["p5"](features["p5"])
        p4 = self.lateral["p4"](features["p4"]) + F.interpolate(
            p5, size=features["p4"].shape[-2:], mode="nearest"
        )
        p3 = self.lateral["p3"](features["p3"]) + F.interpolate(
            p4, size=features["p3"].shape[-2:], mode="nearest"
        )
        p2 = self.lateral["p2"](features["p2"]) + F.interpolate(
            p3, size=features["p2"].shape[-2:], mode="nearest"
        )
        p2 = p2 + self.edge_path(p2)
        return {
            "p2": self.fuse["p2"](p2),
            "p3": self.fuse["p3"](p3),
            "p4": self.fuse["p4"](p4),
            "p5": self.fuse["p5"](p5),
        }
