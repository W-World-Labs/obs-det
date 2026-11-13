from omnidet.hardware import HARDWARE_PROFILES, model_config_for_profile, train_config_for_profile


def test_rtx5090_profile_targets_32gb_training() -> None:
    profile = HARDWARE_PROFILES["rtx5090_32g"]

    assert profile.vram_gb == 32
    assert profile.default_input_size == 960
    assert profile.precision == "bf16"


def test_rtxpro6000_profile_uses_larger_small_object_capacity() -> None:
    model_cfg = model_config_for_profile("rtxpro6000_96g")
    train_cfg = train_config_for_profile("rtxpro6000_96g")

    assert model_cfg.input_size == 1280
    assert model_cfg.accuracy_input_size == 1536
    assert model_cfg.neck_channels > model_config_for_profile("rtx5090_32g").neck_channels
    assert model_cfg.num_queries > model_config_for_profile("rtx5090_32g").num_queries
    assert train_cfg.batch_size > train_config_for_profile("rtx5090_32g").batch_size
