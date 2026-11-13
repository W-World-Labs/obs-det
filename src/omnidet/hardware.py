from dataclasses import dataclass

from omnidet.config import ModelConfig, TrainConfig


@dataclass(frozen=True)
class HardwareProfile:
    name: str
    gpu_name: str
    vram_gb: int
    default_input_size: int
    accuracy_input_size: int
    batch_size: int
    p2_channels: int
    p3_channels: int
    p4_channels: int
    p5_channels: int
    neck_channels: int
    num_queries: int
    backbone_variant: str
    backbone_name: str
    backbone_pretrained: bool
    freeze_backbone: bool
    backbone_out_indices: tuple[int, ...]
    precision: str
    compile_model: bool
    channels_last: bool
    tf32: bool
    fused_adamw: bool


HARDWARE_PROFILES: dict[str, HardwareProfile] = {
    "rtx5090_32g": HardwareProfile(
        name="rtx5090_32g",
        gpu_name="NVIDIA GeForce RTX 5090",
        vram_gb=32,
        default_input_size=960,
        accuracy_input_size=1280,
        batch_size=8,
        p2_channels=64,
        p3_channels=96,
        p4_channels=160,
        p5_channels=256,
        neck_channels=128,
        num_queries=300,
        backbone_variant="omnidet",
        backbone_name="omnidet_cleanroom_s",
        backbone_pretrained=False,
        freeze_backbone=False,
        backbone_out_indices=(0, 1, 2, 3),
        precision="bf16",
        compile_model=True,
        channels_last=True,
        tf32=True,
        fused_adamw=True,
    ),
    "rtxpro6000_96g": HardwareProfile(
        name="rtxpro6000_96g",
        gpu_name="NVIDIA RTX PRO 6000 Blackwell",
        vram_gb=96,
        default_input_size=1280,
        accuracy_input_size=1536,
        batch_size=24,
        p2_channels=96,
        p3_channels=144,
        p4_channels=224,
        p5_channels=384,
        neck_channels=192,
        num_queries=600,
        backbone_variant="omnidet",
        backbone_name="omnidet_cleanroom_m",
        backbone_pretrained=False,
        freeze_backbone=False,
        backbone_out_indices=(0, 1, 2, 3),
        precision="bf16",
        compile_model=True,
        channels_last=True,
        tf32=True,
        fused_adamw=True,
    ),
}


def get_hardware_profile(name: str) -> HardwareProfile:
    try:
        return HARDWARE_PROFILES[name]
    except KeyError as exc:
        choices = ", ".join(sorted(HARDWARE_PROFILES))
        raise ValueError(f"Unknown hardware profile '{name}'. Choices: {choices}") from exc


def model_config_for_profile(name: str) -> ModelConfig:
    profile = get_hardware_profile(name)
    return ModelConfig(
        input_size=profile.default_input_size,
        accuracy_input_size=profile.accuracy_input_size,
        validation_device=f"{profile.name}:{profile.gpu_name}",
        p2_channels=profile.p2_channels,
        p3_channels=profile.p3_channels,
        p4_channels=profile.p4_channels,
        p5_channels=profile.p5_channels,
        neck_channels=profile.neck_channels,
        num_queries=profile.num_queries,
        backbone_variant=profile.backbone_variant,
        backbone_name=profile.backbone_name,
        backbone_pretrained=profile.backbone_pretrained,
        freeze_backbone=profile.freeze_backbone,
        backbone_out_indices=profile.backbone_out_indices,
    )


def train_config_for_profile(name: str) -> TrainConfig:
    profile = get_hardware_profile(name)
    return TrainConfig(
        batch_size=profile.batch_size,
        image_size=profile.default_input_size,
    )
