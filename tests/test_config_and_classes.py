from omnidet.config import ModelConfig, TrainConfig
from omnidet.data.classes import OMNIDET_CLASSES, visdrone_category_to_omnidet


def test_default_model_config_targets_rtx5090() -> None:
    cfg = ModelConfig()
    assert cfg.input_size == 960
    assert cfg.accuracy_input_size == 1280
    assert cfg.validation_device == "local:rtx5090"
    assert cfg.num_classes == len(OMNIDET_CLASSES)
    assert cfg.backbone_variant == "omnidet"
    assert cfg.backbone_name == "omnidet_cleanroom_s"
    assert cfg.backbone_pretrained is False


def test_model_config_supports_pretrained_backbone_route() -> None:
    cfg = ModelConfig(
        backbone_variant="timm",
        backbone_name="hf-hub:example/ecdet-s-backbone",
        backbone_pretrained=True,
        freeze_backbone=True,
        backbone_out_indices=(0, 1, 2, 3),
    )

    assert cfg.backbone_variant == "timm"
    assert cfg.backbone_name == "hf-hub:example/ecdet-s-backbone"
    assert cfg.backbone_pretrained is True
    assert cfg.freeze_backbone is True
    assert cfg.backbone_out_indices == (0, 1, 2, 3)


def test_training_config_uses_small_object_bias() -> None:
    cfg = TrainConfig()
    assert cfg.small_box_weight > 1.0
    assert cfg.min_small_box_pixels == 32


def test_visdrone_mapping_keeps_person_vehicle_classes() -> None:
    assert visdrone_category_to_omnidet(1) == "pedestrian"
    assert visdrone_category_to_omnidet(4) == "car"
    assert visdrone_category_to_omnidet(9) == "bus"
    assert visdrone_category_to_omnidet(0) is None
    assert visdrone_category_to_omnidet(11) is None
