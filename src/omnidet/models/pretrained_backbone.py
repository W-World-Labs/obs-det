from __future__ import annotations

import torch
from torch import nn

from omnidet.config import ModelConfig
from omnidet.models.blocks import ConvBNAct
from omnidet.models.features import DEFAULT_PYRAMID_STRIDES


# The backbone adapter must reproduce the canonical pyramid strides so the neck
# and every task head share one feature contract.
PYRAMID_REDUCTIONS = DEFAULT_PYRAMID_STRIDES


class TimmFeatureBackbone(nn.Module):
    """Feature adapter for timm; upstream checkpoints retain their own licenses."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        try:
            import timm
        except ImportError as exc:  # pragma: no cover - exercised only without optional dep
            raise ImportError(
                "backbone_variant='timm' requires timm. Install with `pip install timm`."
            ) from exc

        self.feature_extractor = timm.create_model(
            model_name=cfg.backbone_name,
            pretrained=cfg.backbone_pretrained,
            features_only=True,
            out_indices=cfg.backbone_out_indices,
        )
        self.frozen = bool(cfg.freeze_backbone)
        if self.frozen:
            self.feature_extractor.requires_grad_(False)
            # eval() so BatchNorm running stats stop updating in train mode.
            self.feature_extractor.eval()

        channels = list(self.feature_extractor.feature_info.channels())
        reductions = [int(r) for r in self.feature_extractor.feature_info.reduction()]
        index_by_reduction = {reduction: idx for idx, reduction in enumerate(reductions)}
        missing = [
            f"{level}/{reduction}x"
            for level, reduction in PYRAMID_REDUCTIONS.items()
            if reduction not in index_by_reduction
        ]
        if missing:
            raise ValueError(
                f"Backbone '{cfg.backbone_name}' does not expose required feature levels: "
                + ", ".join(missing)
            )

        out_channels = {
            "p2": cfg.p2_channels,
            "p3": cfg.p3_channels,
            "p4": cfg.p4_channels,
            "p5": cfg.p5_channels,
        }
        self.level_indices = {
            level: index_by_reduction[reduction]
            for level, reduction in PYRAMID_REDUCTIONS.items()
        }
        self.projections = nn.ModuleDict(
            {
                level: ConvBNAct(channels[self.level_indices[level]], out_channels[level], kernel_size=1)
                for level in PYRAMID_REDUCTIONS
            }
        )

    def train(self, mode: bool = True) -> "TimmFeatureBackbone":
        # Always keep a frozen backbone in eval() — BN/Dropout shouldn't update.
        super().train(mode)
        if self.frozen:
            self.feature_extractor.eval()
        return self

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # When frozen, run the backbone under no_grad so its intermediate
        # activations don't sit in autograd's saved-tensors graph — a
        # ResNet50 at 960² batch 16 saves ~6-8 GB this way. The projection
        # layers (trainable) take the no-grad output as input, which works
        # because their own weights still have requires_grad=True.
        if self.frozen:
            with torch.no_grad():
                raw_features = self.feature_extractor(x)
        else:
            raw_features = self.feature_extractor(x)
        if isinstance(raw_features, dict):
            raw_features = list(raw_features.values())
        return {
            level: self.projections[level](raw_features[index])
            for level, index in self.level_indices.items()
        }
