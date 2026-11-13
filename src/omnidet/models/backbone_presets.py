from __future__ import annotations

from dataclasses import dataclass, replace

from omnidet.config import ModelConfig


@dataclass(frozen=True)
class BackbonePreset:
    name: str
    backbone_variant: str
    backbone_name: str
    pretrained: bool
    freeze_by_default: bool
    out_indices: tuple[int, ...]
    reductions: tuple[int, ...]
    requires_pyramid: bool = True


BACKBONE_PRESETS: dict[str, BackbonePreset] = {
    "fastvit_t8_dist": BackbonePreset(
        name="fastvit_t8_dist",
        backbone_variant="timm",
        backbone_name="fastvit_t8.apple_dist_in1k",
        pretrained=True,
        freeze_by_default=True,
        out_indices=(0, 1, 2, 3),
        reductions=(4, 8, 16, 32),
    ),
    "repvit_m1_dist": BackbonePreset(
        name="repvit_m1_dist",
        backbone_variant="timm",
        backbone_name="repvit_m1.dist_in1k",
        pretrained=True,
        freeze_by_default=True,
        out_indices=(0, 1, 2, 3),
        reductions=(4, 8, 16, 32),
    ),
    "efficientvit_b0": BackbonePreset(
        name="efficientvit_b0",
        backbone_variant="timm",
        backbone_name="efficientvit_b0.r224_in1k",
        pretrained=True,
        freeze_by_default=True,
        out_indices=(0, 1, 2, 3),
        reductions=(4, 8, 16, 32),
    ),
    "convnextv2_atto_fcmae": BackbonePreset(
        name="convnextv2_atto_fcmae",
        backbone_variant="timm",
        backbone_name="convnextv2_atto.fcmae_ft_in1k",
        pretrained=True,
        freeze_by_default=True,
        out_indices=(0, 1, 2, 3),
        reductions=(4, 8, 16, 32),
    ),
    "edgenext_xx_small": BackbonePreset(
        name="edgenext_xx_small",
        backbone_variant="timm",
        backbone_name="edgenext_xx_small.in1k",
        pretrained=True,
        freeze_by_default=True,
        out_indices=(0, 1, 2, 3),
        reductions=(4, 8, 16, 32),
    ),
    # Classic CNN backbone with a native P2-P5 pyramid — useful as a
    # high-capacity sanity check against the clean-room small backbone.
    # timm code is Apache-2.0; checkpoint/data rights remain separate.
    "resnet50": BackbonePreset(
        name="resnet50",
        backbone_variant="timm",
        backbone_name="resnet50.a1_in1k",
        pretrained=True,
        freeze_by_default=True,
        out_indices=(1, 2, 3, 4),
        reductions=(4, 8, 16, 32),
    ),
    # ConvNeXt v2 with FCMAE (Fully-Convolutional Masked Autoencoder)
    # pretraining — the design doc's MAE-style backbone option that still
    # emits a P2-P5 pyramid (unlike vanilla ViT MAE which gives a single 16x
    # token map). FCMAE checkpoint model cards declare CC-BY-NC-4.0.
    "convnextv2_tiny_fcmae": BackbonePreset(
        name="convnextv2_tiny_fcmae",
        backbone_variant="timm",
        backbone_name="convnextv2_tiny.fcmae_ft_in1k",
        pretrained=True,
        freeze_by_default=True,
        out_indices=(0, 1, 2, 3),
        reductions=(4, 8, 16, 32),
    ),
    # Base variant (88.7M params, 3.1x Tiny). Better feature capacity for
    # tiny-aerial-object detection; expected faster convergence per literature.
    # Memory cost is ~3x backbone activations vs Tiny -> requires smaller
    # batch on a 32GB GPU when combined with unfrozen + P2 memory.
    "convnextv2_base_fcmae": BackbonePreset(
        name="convnextv2_base_fcmae",
        backbone_variant="timm",
        backbone_name="convnextv2_base.fcmae_ft_in1k",
        pretrained=True,
        freeze_by_default=False,  # default unfrozen — frozen has been proven
                                  # actively harmful for aerial domain shift.
        out_indices=(0, 1, 2, 3),
        reductions=(4, 8, 16, 32),
    ),
    # Large variant (197.7M params). Heaviest option that still trains on a
    # single 32GB GPU when frozen, or with very small batch when unfrozen.
    "convnextv2_large_fcmae": BackbonePreset(
        name="convnextv2_large_fcmae",
        backbone_variant="timm",
        backbone_name="convnextv2_large.fcmae_ft_in1k",
        pretrained=True,
        freeze_by_default=False,
        out_indices=(0, 1, 2, 3),
        reductions=(4, 8, 16, 32),
    ),
}


def get_backbone_preset(name: str) -> BackbonePreset:
    try:
        return BACKBONE_PRESETS[name]
    except KeyError as exc:
        choices = ", ".join(sorted(BACKBONE_PRESETS))
        raise ValueError(f"Unknown backbone preset '{name}'. Choices: {choices}") from exc


def apply_backbone_preset(cfg: ModelConfig, name: str) -> ModelConfig:
    preset = get_backbone_preset(name)
    return replace(
        cfg,
        backbone_variant=preset.backbone_variant,
        backbone_name=preset.backbone_name,
        backbone_pretrained=preset.pretrained,
        freeze_backbone=preset.freeze_by_default,
        backbone_out_indices=preset.out_indices,
    )
