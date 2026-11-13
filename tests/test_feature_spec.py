import pytest
import torch

from omnidet.config import ModelConfig
from omnidet.models.detector import OmniDet
from omnidet.models.features import DEFAULT_PYRAMID_STRIDES, FeatureSpec
from omnidet.models.neck import EdgeMambaFPN


def test_pyramid_builds_canonical_strides() -> None:
    spec = FeatureSpec.pyramid(channels=128)

    assert spec.levels == ("p2", "p3", "p4", "p5")
    assert spec.strides == (4, 8, 16, 32)
    assert spec.channels == 128
    assert spec.stride_by_level == DEFAULT_PYRAMID_STRIDES
    assert spec.finest_level == "p2"


def test_pyramid_rejects_unknown_level() -> None:
    with pytest.raises(ValueError):
        FeatureSpec.pyramid(channels=64, levels=("p2", "p9"))


def test_invalid_specs_are_rejected() -> None:
    with pytest.raises(ValueError):
        FeatureSpec(levels=("p2", "p3"), strides=(4,), channels=64)
    with pytest.raises(ValueError):
        FeatureSpec(levels=("p2",), strides=(4,), channels=0)


def test_validate_accepts_matching_features() -> None:
    spec = FeatureSpec.pyramid(channels=16)
    features = {level: torch.randn(1, 16, 8, 8) for level in spec.levels}

    spec.validate(features)  # should not raise


def test_validate_rejects_contract_violations() -> None:
    spec = FeatureSpec.pyramid(channels=16)

    missing = {"p2": torch.randn(1, 16, 8, 8)}
    with pytest.raises(ValueError):
        spec.validate(missing)

    wrong_channels = {level: torch.randn(1, 8, 8, 8) for level in spec.levels}
    with pytest.raises(ValueError):
        spec.validate(wrong_channels)


def test_neck_exposes_feature_spec_and_head_consumes_it() -> None:
    cfg = ModelConfig(input_size=128, neck_channels=48)
    neck = EdgeMambaFPN(cfg)

    assert neck.feature_spec.channels == cfg.neck_channels

    model = OmniDet(cfg)
    assert model.head.feature_spec == neck.feature_spec

    outputs = model(torch.randn(1, 3, cfg.input_size, cfg.input_size))
    assert outputs["pred_logits"].shape[0] == 1
    assert outputs["pred_boxes"].shape[-1] == 4
