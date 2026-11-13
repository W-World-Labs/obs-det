from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelConfig:
    input_size: int = 960
    accuracy_input_size: int = 1280
    validation_device: str = "local:rtx5090"
    backbone_variant: str = "omnidet"
    backbone_name: str = "omnidet_cleanroom_s"
    backbone_pretrained: bool = False
    freeze_backbone: bool = False
    backbone_out_indices: tuple[int, ...] = (0, 1, 2, 3)
    p2_channels: int = 64
    p3_channels: int = 96
    p4_channels: int = 160
    p5_channels: int = 256
    neck_channels: int = 128
    num_queries: int = 300
    num_classes: int = 8
    active_tasks: tuple[str, ...] = ("detection",)
    task_loss_weights: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    num_keypoints: int = 17
    seg_num_classes: int | None = None
    seg_head_channels: int = 64
    mesh_enabled: bool = False
    mesh_output_dim: int = 0
    mesh_num_vertices: int = 478
    mesh_vertex_dim: int = 3
    reid_embedding_dim: int = 128
    quality_head_channels: int = 64
    depth_head_channels: int = 64
    neck_variant: str = "conv_gated"
    # When True, P2 (stride-4) tokens are added to the decoder cross-attention
    # memory. Recent UAV / small-object DETR papers (Drone-DETR 2024,
    # UAV-DETR 2025, Dome-DETR 2025) report this as essential for tiny aerial
    # objects; the cost stays bounded because decoder cross-attention is
    # O(num_queries · num_memory_tokens), not O(n²) as in encoder self-attn.
    use_p2_memory: bool = False


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 8
    epochs: int = 150
    base_lr: float = 2e-4
    weight_decay: float = 1e-4
    small_box_weight: float = 1.5
    min_small_box_pixels: int = 32
    image_size: int = 960
    class_names: tuple[str, ...] = field(
        default=(
            "person",
            "pedestrian",
            "car",
            "van",
            "truck",
            "bus",
            "motorcycle",
            "bicycle",
        )
    )
