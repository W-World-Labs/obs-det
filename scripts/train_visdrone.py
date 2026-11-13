import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

# Use /dev/shm-backed tensor sharing so DataLoader workers don't hit the fd
# limit at high --num-workers ("received 0 items of ancdata").
torch.multiprocessing.set_sharing_strategy("file_system")

from omnidet.config import ModelConfig, TrainConfig
from omnidet.data.classes import OMNIDET_CLASSES
from omnidet.data.dataset import COCODetectionDataset, VisDroneDetectionDataset, collate_detections
from omnidet.hardware import get_hardware_profile, model_config_for_profile
from omnidet.losses.criterion import DetectionCriterion, MultiTaskCriterion
from omnidet.models.backbone_presets import BACKBONE_PRESETS, apply_backbone_preset
from omnidet.models.detector import OmniDet
from omnidet.training.acceleration import (
    build_param_groups,
    configure_cuda_kernels,
    make_adamw,
    precision_to_dtype,
    prepare_model,
)
from omnidet.training.engine import eval_one_epoch, train_one_step


SUPPORTED_TASKS = ("detection", "segmentation", "pose", "mesh", "reid", "depth", "quality")


def parse_out_indices(value: str) -> tuple[int, ...]:
    indices = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not indices:
        raise argparse.ArgumentTypeError("backbone out indices cannot be empty")
    return indices


def parse_task_list(value: str) -> tuple[str, ...]:
    tasks = tuple(task.strip().lower() for task in value.split(",") if task.strip())
    if not tasks:
        raise argparse.ArgumentTypeError("tasks cannot be empty")
    supported = set(SUPPORTED_TASKS)
    if "all" in tasks:
        if len(tasks) != 1:
            raise argparse.ArgumentTypeError("tasks=all 不能与其他任务混用")
        return tuple(sorted(supported))
    unknown = [task for task in tasks if task not in supported]
    if unknown:
        raise argparse.ArgumentTypeError(f"unsupported tasks: {', '.join(sorted(unknown))}")
    return tasks


def parse_float_list(value: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("task-loss-weights cannot be empty")
    return values


def ensure_task_weights(weights: tuple[float, ...], *, task_count: int = len(SUPPORTED_TASKS)) -> tuple[float, ...]:
    if len(weights) >= task_count:
        return weights[:task_count]
    return weights + (1.0,) * (task_count - len(weights))


def parse_model_spec(path: Path) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix in {".yml", ".yaml"}:
        try:
            import yaml
        except ModuleNotFoundError as exc:
            raise argparse.ArgumentTypeError(
                f"无法解析 YAML 模型规范: 未安装 pyyaml，支持的替代方案是 .json。请先运行: pip install pyyaml"
            ) from exc
        spec = yaml.safe_load(raw)
    else:
        spec = json.loads(raw)
    if not isinstance(spec, dict):
        raise argparse.ArgumentTypeError("模型规范必须是字典")
    return spec


def normalize_task_spec(spec: dict[str, Any]) -> tuple[str, ...]:
    tasks = spec.get("tasks") or spec.get("active_tasks")
    if tasks is None:
        return ("detection",)
    if isinstance(tasks, str):
        tasks = tasks.split(",")
    if not isinstance(tasks, (list, tuple)):
        raise argparse.ArgumentTypeError("模型规范中的 tasks 必须是逗号分隔字符串或列表")
    normalized = tuple(task.strip().lower() for task in tasks if str(task).strip())
    if "all" in normalized:
        if len(normalized) != 1:
            raise argparse.ArgumentTypeError("modelspec tasks=all 不能与其他任务混用")
        return SUPPORTED_TASKS
    unknown = [task for task in normalized if task not in SUPPORTED_TASKS]
    if unknown:
        raise argparse.ArgumentTypeError(f"模型规范中的未识别任务: {', '.join(sorted(unknown))}")
    return normalized


def normalize_weight_spec(spec: dict[str, Any]) -> tuple[float, ...] | None:
    raw = spec.get("task_loss_weights") or spec.get("task-loss-weights")
    if raw is None:
        return None
    if isinstance(raw, str):
        raw = tuple(float(item.strip()) for item in raw.split(",") if item.strip())
    elif isinstance(raw, (list, tuple)):
        raw = tuple(float(item) for item in raw)
    else:
        raise argparse.ArgumentTypeError("模型规范中的 task-loss-weights 类型不合法")
    if not raw:
        raise argparse.ArgumentTypeError("模型规范中的 task-loss-weights 不能为空")
    return ensure_task_weights(raw)


def normalize_model_spec(spec: dict[str, Any]) -> dict[str, Any]:
    normalized = {key: value for key, value in spec.items()}
    tasks = normalize_task_spec(spec)
    normalized["active_tasks"] = tasks
    weight_spec = normalize_weight_spec(spec)
    if weight_spec is not None:
        normalized["task_loss_weights"] = weight_spec
    return normalized


def parse_model_spec_value(cfg_value: Any, *, key: str) -> Any:
    if key in {"active_tasks", "task_loss_weights"}:
        return cfg_value
    if key in {"mesh_enabled"}:
        return bool(cfg_value)
    if isinstance(cfg_value, bool):
        return bool(cfg_value)
    if isinstance(cfg_value, int):
        return int(cfg_value)
    if isinstance(cfg_value, float):
        return float(cfg_value)
    return cfg_value


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    mode: str,
    *,
    total_steps: int,
    base_lr: float,
    min_lr: float,
    warmup_steps: int,
    warmup_start_factor: float,
    step_size: int,
    gamma: float,
    onecycle_pct_start: float,
    onecycle_div_factor: float,
    onecycle_final_div_factor: float,
) -> tuple[Any, bool]:
    if mode == "none":
        return None, False
    if total_steps <= 0:
        return None, False

    if mode == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=max(1, step_size), gamma=gamma
        )
        return scheduler, False

    if mode == "cosine":
        if warmup_steps > 0 and warmup_steps < total_steps:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=warmup_start_factor,
                end_factor=1.0,
                total_iters=warmup_steps,
            )
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, total_steps - warmup_steps),
                eta_min=min_lr,
            )
            return (
                torch.optim.lr_scheduler.SequentialLR(
                    optimizer,
                    [warmup_scheduler, cosine_scheduler],
                    milestones=[warmup_steps],
                ),
                True,
            )
        return (
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=total_steps,
                eta_min=min_lr,
            ),
            True,
        )

    if mode == "onecycle":
        return (
            torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=base_lr,
                epochs=1,
                steps_per_epoch=max(1, total_steps),
                pct_start=onecycle_pct_start,
                div_factor=onecycle_div_factor,
                final_div_factor=onecycle_final_div_factor,
            ),
            True,
        )

    raise ValueError(f"Unsupported lr scheduler: {mode}")


def build_dataset(
    dataset_name: str,
    images_path: Path,
    annotations_path: Path,
    image_size: int,
    *,
    active_tasks: tuple[str, ...] = ("detection",),
    include_segmentation: bool = False,
    include_keypoints: bool = False,
    include_mesh: bool = False,
    include_reid: bool = False,
    include_depth: bool = False,
    include_quality: bool = False,
    num_keypoints: int = 17,
    reid_embedding_dim: int = 128,
    mesh_num_vertices: int = 478,
    mesh_vertex_dim: int = 3,
    augment: bool = False,
):
    if dataset_name == "visdrone":
        dataset = VisDroneDetectionDataset(
            images_path, annotations_path, image_size=image_size, augment=augment
        )
        return dataset, len(OMNIDET_CLASSES)
    if dataset_name == "coco":
        dataset = COCODetectionDataset(
            images_path,
            annotations_path,
            image_size=image_size,
            include_segmentation=include_segmentation or "segmentation" in active_tasks,
            include_keypoints=include_keypoints or "pose" in active_tasks,
            include_mesh=include_mesh or "mesh" in active_tasks,
            include_reid=include_reid or "reid" in active_tasks,
            include_depth=include_depth or "depth" in active_tasks,
            include_quality=include_quality or "quality" in active_tasks,
            num_keypoints=num_keypoints,
            reid_embedding_dim=max(1, reid_embedding_dim),
            mesh_num_vertices=mesh_num_vertices,
            mesh_vertex_dim=mesh_vertex_dim,
        )
        return dataset, len(dataset.category_ids)
    raise ValueError(f"Unsupported dataset '{dataset_name}'")


def _load_yaml_or_json(path: Path) -> dict[str, Any]:
    """Load a YAML or JSON config file into a flat dict."""
    raw = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yml", ".yaml"}:
        try:
            import yaml
        except ModuleNotFoundError as exc:
            raise argparse.ArgumentTypeError(
                "YAML config requires pyyaml: pip install pyyaml (or use .json)"
            ) from exc
        data = yaml.safe_load(raw)
    else:
        data = json.loads(raw)
    if not isinstance(data, dict):
        raise argparse.ArgumentTypeError(f"config root must be a mapping, got {type(data).__name__}")
    return data


_CONFIG_PATH_KEYS = (
    "images", "annotations", "val_images", "val_annotations", "out", "log_csv", "model_spec",
)


def _normalize_config_value(key: str, value: Any) -> Any:
    """Coerce a config value to the type argparse would have produced.

    The flat config-file dict goes through ``parser.set_defaults`` so that any
    CLI flag the user *also* passes overrides what's in the file. That means
    we must hand argparse values in the same shape its ``type=`` converters
    would have produced.
    """
    if value is None:
        return value
    if key == "tasks":
        if isinstance(value, str):
            return parse_task_list(value)
        if isinstance(value, (list, tuple)):
            return parse_task_list(",".join(str(item) for item in value))
        raise argparse.ArgumentTypeError(f"'tasks' must be a list or comma string, got {type(value).__name__}")
    if key == "task_loss_weights":
        if isinstance(value, str):
            return parse_float_list(value)
        if isinstance(value, (list, tuple)):
            return tuple(float(item) for item in value)
    if key == "backbone_out_indices":
        if isinstance(value, str):
            return parse_out_indices(value)
        if isinstance(value, (list, tuple)):
            return tuple(int(item) for item in value)
    if key in _CONFIG_PATH_KEYS and isinstance(value, str):
        return Path(value)
    return value


def _apply_config_defaults(parser: argparse.ArgumentParser, config_path: Path) -> None:
    """Apply a YAML/JSON config as parser defaults so CLI flags override it."""
    data = _load_yaml_or_json(config_path)
    valid_dests = {action.dest for action in parser._actions if action.dest != "help"}
    defaults: dict[str, Any] = {}
    unknown: list[str] = []
    for key, value in data.items():
        # Accept both hyphen and underscore keys; argparse uses underscored dest.
        norm_key = key.replace("-", "_")
        if norm_key not in valid_dests:
            unknown.append(key)
            continue
        defaults[norm_key] = _normalize_config_value(norm_key, value)
    if unknown:
        print(f"[config] ignored unknown keys: {unknown}", flush=True)
    if defaults:
        # Required positional args become optional once defaulted from config.
        for action in parser._actions:
            if action.dest in defaults and action.required:
                action.required = False
        parser.set_defaults(**defaults)


def main() -> None:
    # First-pass to pick up --config so we can apply its defaults BEFORE the
    # main argparse runs. CLI flags then override config values naturally via
    # argparse's normal precedence.
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path)
    pre_args, _ = pre_parser.parse_known_args()

    parser = argparse.ArgumentParser(
        description="Train OmniDet on VisDrone/COCO. Configurable via --config <yaml> "
                    "(any CLI flag still overrides the config)."
    )
    parser.add_argument(
        "--config", type=Path,
        help="YAML/JSON config file: any argparse dest can be set here (e.g. backbone_preset, "
             "epochs, batch_size, augment, freeze_backbone). CLI flags override the config.",
    )
    parser.add_argument("--model-spec", type=Path, help="JSON/YAML 模型规范（YOLO风格）路径")
    parser.add_argument("--dataset", choices=["visdrone", "coco"], default="visdrone")
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--val-images", type=Path)
    parser.add_argument("--val-annotations", type=Path)
    parser.add_argument("--val-interval", type=int, default=1)
    parser.add_argument(
        "--hardware-profile",
        choices=["rtx5090_32g", "rtxpro6000_96g"],
        default="rtx5090_32g",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, default=Path("checkpoints/omnidet_v0.pt"))
    parser.add_argument(
        "--log-csv",
        type=Path,
        help="Optional CSV path for per-epoch training logs",
    )
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--backbone-preset", choices=sorted(BACKBONE_PRESETS))
    parser.add_argument("--backbone-variant", choices=["omnidet", "timm"])
    parser.add_argument("--backbone-name")
    parser.add_argument("--backbone-out-indices", type=parse_out_indices)
    parser.add_argument("--pretrained-backbone", dest="backbone_pretrained", action="store_true")
    parser.add_argument("--no-pretrained-backbone", dest="backbone_pretrained", action="store_false")
    parser.set_defaults(backbone_pretrained=None)
    parser.add_argument("--freeze-backbone", dest="freeze_backbone", action="store_true")
    parser.add_argument("--unfreeze-backbone", dest="freeze_backbone", action="store_false")
    parser.set_defaults(freeze_backbone=None)
    parser.add_argument(
        "--use-p2-memory",
        dest="use_p2_memory",
        action="store_true",
        help="Include P2 (stride-4) tokens in the decoder cross-attention memory. "
        "Recommended for tiny-object datasets (VisDrone, AI-TOD); costs +57.6k tokens "
        "at 960x960 input but cross-attn stays bounded by num_queries.",
    )
    parser.add_argument(
        "--no-p2-memory",
        dest="use_p2_memory",
        action="store_false",
        help="Decoder memory uses P3+P4+P5 only.",
    )
    parser.set_defaults(use_p2_memory=None)
    parser.add_argument(
        "--backbone-lr-scale",
        type=float,
        default=0.1,
        help="Multiplier on --lr applied to backbone parameters. Literature "
        "(RT-DETR/D-FINE/RF-DETR) converges on 0.05-0.1 when fine-tuning a "
        "pretrained backbone on a domain-shifted dataset (e.g. ImageNet -> "
        "VisDrone). Ignored when --freeze-backbone is set.",
    )
    parser.add_argument(
        "--small-box-weight",
        type=float,
        default=1.0,
        help="Linearly upweight L1+GIoU+NWD loss for tiny targets up to this "
        "multiplier when the box side is below --min-small-box-pixels. "
        "Default 1.0 disables the reweighting.",
    )
    parser.add_argument(
        "--min-small-box-pixels",
        type=int,
        default=32,
        help="Box-side threshold (in original image pixels) above which the "
        "small-box weight collapses back to 1.0.",
    )
    parser.add_argument(
        "--nwd-weight",
        type=float,
        default=0.0,
        help="Weight of the Normalized Wasserstein Distance term added to the "
        "box regression loss. NWD treats boxes as 2D Gaussians and stays "
        "well-behaved on sub-pixel boxes where IoU/GIoU collapse "
        "(arXiv 2110.13389). Recommended 0.5-1.0 for VisDrone/AI-TOD.",
    )
    parser.add_argument(
        "--nwd-constant",
        type=float,
        default=0.05,
        help="Per-dataset constant in NWD = exp(-sqrt(W^2)/C). Smaller C => "
        "more sensitivity at tiny scales. Default 0.05 is tuned for "
        "normalised cxcywh VisDrone-scale boxes (1-3% wide).",
    )
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--channels-last", action="store_true", default=True)
    parser.add_argument("--no-channels-last", dest="channels_last", action="store_false")
    parser.add_argument("--tf32", action="store_true", default=True)
    parser.add_argument("--no-tf32", dest="tf32", action="store_false")
    parser.add_argument("--cudnn-benchmark", action="store_true", default=True)
    parser.add_argument("--no-cudnn-benchmark", dest="cudnn_benchmark", action="store_false")
    parser.add_argument("--fused-adamw", action="store_true", default=True)
    parser.add_argument("--no-fused-adamw", dest="fused_adamw", action="store_false")
    parser.add_argument(
        "--lr-scheduler",
        choices=["none", "cosine", "step", "onecycle"],
        default="none",
        help="Learning-rate scheduler mode",
    )
    parser.add_argument("--lr-min", type=float, default=1e-6)
    parser.add_argument("--lr-warmup-steps", type=int, default=500)
    parser.add_argument("--lr-warmup-start-factor", type=float, default=0.1)
    parser.add_argument("--lr-step-size", type=int, default=30)
    parser.add_argument("--lr-gamma", type=float, default=0.1)
    parser.add_argument("--onecycle-pct-start", type=float, default=0.3)
    parser.add_argument("--onecycle-div-factor", type=float, default=25.0)
    parser.add_argument("--onecycle-final-div-factor", type=float, default=10000.0)
    parser.add_argument("--tasks", type=parse_task_list, default=None)
    parser.add_argument("--task-loss-weights", type=parse_float_list, default=None)
    parser.add_argument("--num-keypoints", type=int, default=17)
    parser.add_argument("--seg-num-classes", type=int)
    parser.add_argument("--seg-head-channels", type=int, default=64)
    parser.add_argument("--coco-include-segmentation", action="store_true")
    parser.add_argument("--coco-include-keypoints", action="store_true")
    parser.add_argument("--coco-include-mesh", action="store_true")
    parser.add_argument("--coco-include-reid", action="store_true")
    parser.add_argument("--coco-include-depth", action="store_true")
    parser.add_argument("--coco-include-quality", action="store_true")
    parser.add_argument("--mesh-enabled", action="store_true")
    parser.add_argument("--mesh-output-dim", type=int, default=0)
    parser.add_argument("--mesh-num-vertices", type=int, default=478)
    parser.add_argument("--mesh-vertex-dim", type=int, default=3)
    parser.add_argument("--reid-embedding-dim", type=int, default=128)
    parser.add_argument("--depth-head-channels", type=int, default=64)
    parser.add_argument("--quality-head-channels", type=int, default=64)
    parser.add_argument(
        "--mesh-distill-from-teachers",
        action="store_true",
        default=True,
        help="Enable mesh teacher distillation when mesh_teacher fields are present",
    )
    parser.add_argument(
        "--no-mesh-distill-from-teachers",
        dest="mesh_distill_from_teachers",
        action="store_false",
        help="Disable mesh teacher distillation",
    )
    parser.add_argument("--grad-clip", type=float, default=0.0)
    parser.add_argument(
        "--augment",
        action="store_true",
        help="enable train-time hflip + photometric augmentation (val always unaugmented)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="stop after this many optimizer steps (0 = no limit); for smoke training runs",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="DataLoader worker processes; raise to keep the GPU fed (host core count is the ceiling)",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=4,
        help="batches prefetched per worker (only used when --num-workers > 0)",
    )
    if pre_args.config is not None:
        _apply_config_defaults(parser, pre_args.config)
        print(f"[config] loaded defaults from {pre_args.config}", flush=True)
    args = parser.parse_args()

    profile = get_hardware_profile(args.hardware_profile)
    profile_model_cfg = model_config_for_profile(profile.name)
    model_spec = normalize_model_spec(parse_model_spec(args.model_spec)) if args.model_spec else None
    image_size = args.image_size or profile.default_input_size
    batch_size = args.batch_size or profile.batch_size
    precision = args.precision or profile.precision
    train_cfg = TrainConfig(batch_size=batch_size, epochs=args.epochs, image_size=image_size)
    model_cfg = ModelConfig(
        input_size=image_size,
        accuracy_input_size=profile_model_cfg.accuracy_input_size,
        validation_device=profile_model_cfg.validation_device,
        p2_channels=profile_model_cfg.p2_channels,
        p3_channels=profile_model_cfg.p3_channels,
        p4_channels=profile_model_cfg.p4_channels,
        p5_channels=profile_model_cfg.p5_channels,
        neck_channels=profile_model_cfg.neck_channels,
        num_queries=profile_model_cfg.num_queries,
        num_classes=profile_model_cfg.num_classes,
        neck_variant=profile_model_cfg.neck_variant,
        backbone_variant=args.backbone_variant or profile_model_cfg.backbone_variant,
        backbone_name=args.backbone_name or profile_model_cfg.backbone_name,
        backbone_pretrained=(
            profile_model_cfg.backbone_pretrained
            if args.backbone_pretrained is None
            else args.backbone_pretrained
        ),
        freeze_backbone=(
            profile_model_cfg.freeze_backbone
            if args.freeze_backbone is None
            else args.freeze_backbone
        ),
        backbone_out_indices=args.backbone_out_indices or profile_model_cfg.backbone_out_indices,
        use_p2_memory=(
            False if args.use_p2_memory is None else args.use_p2_memory
        ),
    )
    if args.backbone_preset:
        model_cfg = apply_backbone_preset(model_cfg, args.backbone_preset)
        if args.backbone_variant:
            model_cfg = replace(model_cfg, backbone_variant=args.backbone_variant)
        if args.backbone_name:
            model_cfg = replace(model_cfg, backbone_name=args.backbone_name)
        if args.backbone_pretrained is not None:
            model_cfg = replace(model_cfg, backbone_pretrained=args.backbone_pretrained)
        if args.freeze_backbone is not None:
            model_cfg = replace(model_cfg, freeze_backbone=args.freeze_backbone)
        if args.backbone_out_indices:
            model_cfg = replace(model_cfg, backbone_out_indices=args.backbone_out_indices)
    if model_spec is not None:
        spec_overrides: dict[str, Any] = {}
        for key, value in model_spec.items():
            if hasattr(model_cfg, key):
                spec_overrides[key] = parse_model_spec_value(value, key=key)
        if spec_overrides:
            model_cfg = replace(model_cfg, **spec_overrides)
    active_tasks = args.tasks if args.tasks is not None else model_cfg.active_tasks
    task_loss_weights = (
        ensure_task_weights(args.task_loss_weights, task_count=len(active_tasks))
        if args.task_loss_weights is not None
        else ensure_task_weights(model_cfg.task_loss_weights, task_count=len(active_tasks))
    )
    model_cfg = replace(
        model_cfg,
        num_classes=profile_model_cfg.num_classes,
        active_tasks=active_tasks,
        task_loss_weights=task_loss_weights,
        num_keypoints=args.num_keypoints,
        seg_num_classes=args.seg_num_classes,
        seg_head_channels=args.seg_head_channels,
        reid_embedding_dim=args.reid_embedding_dim,
        depth_head_channels=args.depth_head_channels,
        quality_head_channels=args.quality_head_channels,
        mesh_enabled=args.mesh_enabled or "mesh" in active_tasks,
        mesh_output_dim=args.mesh_output_dim,
        mesh_num_vertices=args.mesh_num_vertices,
        mesh_vertex_dim=args.mesh_vertex_dim,
    )
    print(
        f"hardware_profile={profile.name} gpu={profile.gpu_name} "
        f"vram_gb={profile.vram_gb} image_size={image_size} batch_size={batch_size} "
        f"dataset={args.dataset} "
        f"backbone_variant={model_cfg.backbone_variant} backbone_name={model_cfg.backbone_name} "
        f"pretrained_backbone={model_cfg.backbone_pretrained} "
        f"freeze_backbone={model_cfg.freeze_backbone}"
    )
    dataset, num_classes = build_dataset(
        args.dataset,
        args.images,
        args.annotations,
        image_size=train_cfg.image_size,
        active_tasks=active_tasks,
        include_segmentation=args.coco_include_segmentation,
        include_keypoints=args.coco_include_keypoints,
        include_mesh=args.coco_include_mesh,
        include_reid=args.coco_include_reid,
        include_depth=args.coco_include_depth,
        include_quality=args.coco_include_quality,
        num_keypoints=args.num_keypoints,
        reid_embedding_dim=args.reid_embedding_dim,
        mesh_num_vertices=args.mesh_num_vertices,
        mesh_vertex_dim=args.mesh_vertex_dim,
        augment=args.augment,
    )
    model_cfg = replace(model_cfg, num_classes=num_classes)
    print(f"dataset_size={len(dataset)} num_classes={num_classes}")
    loader_extra: dict[str, Any] = {}
    if args.num_workers > 0:
        loader_extra["persistent_workers"] = True
        loader_extra["prefetch_factor"] = args.prefetch_factor
    loader = DataLoader(
        dataset,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_detections,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        **loader_extra,
    )
    val_loader = None
    if args.val_images and args.val_annotations:
        val_dataset, _ = build_dataset(
            args.dataset,
            args.val_images,
            args.val_annotations,
            image_size=train_cfg.image_size,
            active_tasks=active_tasks,
            include_segmentation=args.coco_include_segmentation,
            include_keypoints=args.coco_include_keypoints,
            include_mesh=args.coco_include_mesh,
            include_reid=args.coco_include_reid,
            include_depth=args.coco_include_depth,
            include_quality=args.coco_include_quality,
            num_keypoints=args.num_keypoints,
            reid_embedding_dim=args.reid_embedding_dim,
            mesh_num_vertices=args.mesh_num_vertices,
            mesh_vertex_dim=args.mesh_vertex_dim,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=train_cfg.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_detections,
            pin_memory=torch.cuda.is_available(),
            **loader_extra,
        )
        print(f"val_dataset_size={len(val_dataset)}")
    accel_state = configure_cuda_kernels(
        enable_tf32=args.tf32,
        cudnn_benchmark=args.cudnn_benchmark,
    )
    print(f"acceleration={accel_state}")
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model = prepare_model(
        OmniDet(model_cfg),
        device=device,
        channels_last=args.channels_last,
        compile_model=args.compile,
    )
    if active_tasks == ("detection",):
        criterion = DetectionCriterion(
            num_classes=model_cfg.num_classes,
            small_box_weight=args.small_box_weight,
            min_small_box_pixels=args.min_small_box_pixels,
            image_size=image_size,
            nwd_weight=args.nwd_weight,
            nwd_constant=args.nwd_constant,
        ).to(device)
    else:
        criterion = MultiTaskCriterion(
            num_classes=model_cfg.num_classes,
            active_tasks=active_tasks,
            task_loss_weights=task_loss_weights,
            seg_num_classes=args.seg_num_classes,
            num_keypoints=args.num_keypoints,
            reid_embedding_dim=args.reid_embedding_dim,
            mesh_num_vertices=args.mesh_num_vertices,
            mesh_vertex_dim=args.mesh_vertex_dim,
            mesh_distill_from_teachers=args.mesh_distill_from_teachers,
            small_box_weight=args.small_box_weight,
            min_small_box_pixels=args.min_small_box_pixels,
            image_size=image_size,
            nwd_weight=args.nwd_weight,
            nwd_constant=args.nwd_constant,
        ).to(device)
    # Differential LR for backbone vs head/neck. Backbone gets a small slice of
    # the base LR (default 0.1x) so a pretrained backbone can adapt to the new
    # domain without forgetting its low-level features, while the randomly-
    # initialised head trains at full speed.
    param_groups = build_param_groups(
        model,
        base_lr=args.lr,
        backbone_lr_scale=args.backbone_lr_scale,
    )
    optimizer = make_adamw(
        param_groups,
        lr=args.lr,  # per-group lr in param_groups overrides this default
        weight_decay=train_cfg.weight_decay,
        fused=args.fused_adamw and device.type == "cuda",
    )
    for group in optimizer.param_groups:
        print(
            f"param_group name={group.get('name', '?')} "
            f"lr={group['lr']:.2e} num_params={sum(p.numel() for p in group['params']) / 1e6:.2f}M",
            flush=True,
        )
    amp_dtype = precision_to_dtype(precision)
    grad_scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16" and device.type == "cuda")

    loader_steps = max(len(loader), 1)
    total_steps = loader_steps * args.epochs
    lr_scheduler, step_lr_per_batch = build_lr_scheduler(
        optimizer,
        args.lr_scheduler,
        total_steps=total_steps,
        base_lr=args.lr,
        min_lr=args.lr_min,
        warmup_steps=max(0, args.lr_warmup_steps),
        warmup_start_factor=args.lr_warmup_start_factor,
        step_size=args.lr_step_size,
        gamma=args.lr_gamma,
        onecycle_pct_start=args.onecycle_pct_start,
        onecycle_div_factor=args.onecycle_div_factor,
        onecycle_final_div_factor=args.onecycle_final_div_factor,
    )

    step_counter = 0
    best_val_loss = float("inf")
    best_epoch = -1

    fields = [
        "epoch",
        "train_loss_total",
        "train_loss_cls",
        "train_loss_bbox",
        "val_loss_total",
        "val_loss_cls",
        "val_loss_bbox",
        "val_mAP",
        "val_mAP50",
        "lr",
        "best_val_loss",
        "best_epoch",
    ]
    csv_writer = None
    log_file = None
    if args.log_csv:
        args.log_csv.parent.mkdir(parents=True, exist_ok=True)
        log_file = args.log_csv.open("w", newline="", encoding="utf-8")
        csv_writer = csv.DictWriter(log_file, fieldnames=fields)
        csv_writer.writeheader()

    try:
        for epoch in range(args.epochs):
            train_loss_total = 0.0
            train_loss_cls = 0.0
            train_loss_bbox = 0.0
            train_step_count = 0

            for step, (images, targets) in enumerate(loader):
                losses = train_one_step(
                    model,
                    criterion,
                    optimizer,
                    images,
                    targets,
                    device,
                    amp_dtype=amp_dtype,
                    grad_scaler=grad_scaler,
                    channels_last=args.channels_last,
                    grad_clip_norm=args.grad_clip if args.grad_clip > 0 else None,
                )
                train_loss_total += losses["loss_total"]
                train_loss_cls += losses["loss_cls"]
                train_loss_bbox += losses["loss_bbox"]
                train_step_count += 1
                step_counter += 1

                if step % 10 == 0:
                    current_lr = optimizer.param_groups[0]["lr"]
                    print(
                        f"epoch={epoch} step={step} step_counter={step_counter} "
                        f"lr={current_lr:.6f} "
                        f"loss_total={losses['loss_total']:.4f} "
                        f"loss_cls={losses['loss_cls']:.4f} "
                        f"loss_bbox={losses['loss_bbox']:.4f}",
                        flush=True,
                    )

                if lr_scheduler is not None and step_lr_per_batch:
                    lr_scheduler.step()

                if args.max_steps and step_counter >= args.max_steps:
                    print(f"reached max_steps={args.max_steps}, stopping early", flush=True)
                    break

            if train_step_count:
                train_loss_total /= train_step_count
                train_loss_cls /= train_step_count
                train_loss_bbox /= train_step_count

            val_losses = None
            val_mAP = None
            val_mAP50 = None

            if val_loader is not None and (epoch + 1) % args.val_interval == 0:
                val_losses = eval_one_epoch(
                    model, criterion, val_loader, device,
                    amp_dtype=amp_dtype, channels_last=args.channels_last,
                )
                print(
                    f"epoch={epoch} val "
                    f"loss_total={val_losses['loss_total']:.4f} "
                    f"loss_cls={val_losses['loss_cls']:.4f} "
                    f"loss_bbox={val_losses['loss_bbox']:.4f}",
                    flush=True,
                )
                if args.dataset == "coco":
                    from omnidet.evaluation.coco_eval import collect_coco_predictions, run_coco_map

                    preds = collect_coco_predictions(
                        model, val_loader, device,
                        amp_dtype=amp_dtype, channels_last=args.channels_last,
                    )
                    metrics = run_coco_map(args.val_annotations, preds)
                    if metrics:
                        val_mAP = metrics["mAP"]
                        val_mAP50 = metrics["mAP50"]
                        print(
                            f"epoch={epoch} mAP={val_mAP:.4f} mAP50={val_mAP50:.4f}",
                            flush=True,
                        )

                current_val_loss = val_losses["loss_total"]
                if current_val_loss < best_val_loss:
                    best_val_loss = current_val_loss
                    best_epoch = epoch
                    best_path = args.out.with_name(f"{args.out.stem}_best{args.out.suffix}")
                    best_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {"model": model.state_dict(), "model_cfg": model_cfg, "epoch": epoch},
                        best_path,
                    )
                    print(
                        f"best_val improved epoch={best_epoch} "
                        f"val_loss_total={best_val_loss:.4f} saved={best_path}",
                        flush=True,
                    )

            if lr_scheduler is not None and not step_lr_per_batch:
                lr_scheduler.step()

            epoch_lr = optimizer.param_groups[0]["lr"]
            print(
                f"epoch={epoch} train "
                f"loss_total={train_loss_total:.4f} "
                f"loss_cls={train_loss_cls:.4f} "
                f"loss_bbox={train_loss_bbox:.4f} "
                f"lr={epoch_lr:.6f} "
                f"best_val_loss={best_val_loss if best_val_loss != float('inf') else 'nan'} "
                f"best_epoch={best_epoch}",
                flush=True,
            )
            if csv_writer is not None:
                csv_writer.writerow(
                    {
                        "epoch": epoch,
                        "train_loss_total": f"{train_loss_total:.6f}",
                        "train_loss_cls": f"{train_loss_cls:.6f}",
                        "train_loss_bbox": f"{train_loss_bbox:.6f}",
                        "val_loss_total": "" if val_losses is None else f"{val_losses['loss_total']:.6f}",
                        "val_loss_cls": "" if val_losses is None else f"{val_losses['loss_cls']:.6f}",
                        "val_loss_bbox": "" if val_losses is None else f"{val_losses['loss_bbox']:.6f}",
                        "val_mAP": "" if val_mAP is None else f"{val_mAP:.6f}",
                        "val_mAP50": "" if val_mAP50 is None else f"{val_mAP50:.6f}",
                        "lr": f"{epoch_lr:.8f}",
                        "best_val_loss": "" if best_val_loss == float("inf") else f"{best_val_loss:.6f}",
                        "best_epoch": f"{best_epoch}",
                    }
                )
                log_file.flush()

            if args.max_steps and step_counter >= args.max_steps:
                break
    finally:
        if log_file is not None:
            log_file.close()

    print(f"train_done best_val_loss={best_val_loss if best_val_loss != float('inf') else None}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "model_cfg": model_cfg}, args.out)
    print(f"saved={args.out}", flush=True)


if __name__ == "__main__":
    main()
