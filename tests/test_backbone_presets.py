from omnidet.config import ModelConfig
from omnidet.models.backbone_presets import apply_backbone_preset, get_backbone_preset


def test_fastvit_preset_enables_pretrained_timm_backbone() -> None:
    cfg = apply_backbone_preset(ModelConfig(), "fastvit_t8_dist")

    assert cfg.backbone_variant == "timm"
    assert cfg.backbone_name == "fastvit_t8.apple_dist_in1k"
    assert cfg.backbone_pretrained is True
    assert cfg.freeze_backbone is True
    assert cfg.backbone_out_indices == (0, 1, 2, 3)


def test_backbone_preset_reports_pyramid_support() -> None:
    preset = get_backbone_preset("convnextv2_atto_fcmae")

    assert preset.requires_pyramid is True
    assert preset.reductions == (4, 8, 16, 32)
