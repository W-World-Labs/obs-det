from collections.abc import Iterable

import torch
from torch import nn


def precision_to_dtype(precision: str) -> torch.dtype | None:
    match precision:
        case "fp32":
            return None
        case "fp16":
            return torch.float16
        case "bf16":
            return torch.bfloat16
        case _:
            raise ValueError(f"Unsupported precision: {precision}")


def configure_cuda_kernels(enable_tf32: bool, cudnn_benchmark: bool) -> dict[str, bool]:
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        torch.backends.cuda.matmul.allow_tf32 = enable_tf32
        torch.backends.cudnn.allow_tf32 = enable_tf32
        torch.backends.cudnn.benchmark = cudnn_benchmark
        torch.set_float32_matmul_precision("high" if enable_tf32 else "highest")
    return {
        "cuda_available": cuda_available,
        "tf32": bool(enable_tf32 and cuda_available),
        "cudnn_benchmark": bool(cudnn_benchmark and cuda_available),
    }


def apply_memory_format(x: torch.Tensor, channels_last: bool) -> torch.Tensor:
    if channels_last and x.ndim == 4:
        return x.contiguous(memory_format=torch.channels_last)
    return x.contiguous()


def images_to_device(
    images: torch.Tensor,
    device: torch.device | str,
    channels_last: bool,
    *,
    non_blocking: bool = True,
) -> torch.Tensor:
    """Move an image batch to ``device`` as float CHW in [0, 1].

    Accepts the fast-path contract: uint8 HWC ``(B, H, W, 3)`` from the
    DataLoader, where the float conversion and HWC->CHW transpose run on the
    GPU instead of inside a CPU-side worker. Also accepts already-float CHW
    tensors and passes them through, so synthetic-data callers (smoke tests)
    keep working unchanged.
    """
    images = images.to(device, non_blocking=non_blocking)
    if images.dtype == torch.uint8:
        images = images.permute(0, 3, 1, 2).float().div_(255.0)
    return apply_memory_format(images, channels_last)


def prepare_model(
    model: nn.Module,
    device: torch.device | str,
    channels_last: bool,
    compile_model: bool,
) -> nn.Module:
    model = model.to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    if compile_model:
        model = torch.compile(model)
    return model


def make_adamw(
    parameters: Iterable[torch.nn.Parameter],
    lr: float,
    weight_decay: float,
    fused: bool,
) -> torch.optim.AdamW:
    try:
        return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay, fused=fused)
    except RuntimeError:
        return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay, fused=False)
    except TypeError:
        return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)


def build_param_groups(
    model: nn.Module,
    base_lr: float,
    backbone_lr_scale: float,
) -> list[dict]:
    """Split optimizer params into a low-LR pretrained-backbone group and a head/neck group.

    Literature (RT-DETR / D-FINE / RF-DETR / "Rethinking Feature Backbone
    Fine-tuning for RS") converges on backbone LR = 0.05-0.1 × head LR when
    fine-tuning a pretrained backbone on a domain-shifted dataset. Frozen
    parameters (requires_grad=False) are dropped so the optimizer doesn't
    track them — this matters for the fused AdamW kernel which assumes every
    entry is live.

    Only the *pretrained* parts (the timm feature extractor itself) get the
    reduced LR. The 1×1 channel-adapter projections inside the backbone module
    are randomly initialised and stay at full LR alongside neck + head — they
    need to learn from scratch and shouldn't be slowed down.
    """
    # torch.compile wraps the module so named_parameters() yields
    # "_orig_mod.backbone.feature_extractor.*" instead of
    # "backbone.feature_extractor.*". Strip the optional prefix so the filter
    # works whether build_param_groups runs pre- or post-compile.
    backbone_marker = "backbone.feature_extractor."
    backbone, rest = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # ``backbone.feature_extractor.*`` is the pretrained timm trunk; every
        # other ``backbone.*`` parameter (projections, etc.) was randomly
        # initialised and trains with the head.
        if backbone_marker in name:
            backbone.append(param)
        else:
            rest.append(param)
    groups: list[dict] = []
    if backbone:
        groups.append(
            {"params": backbone, "lr": base_lr * backbone_lr_scale, "name": "backbone"}
        )
    if rest:
        groups.append({"params": rest, "lr": base_lr, "name": "head"})
    if not groups:
        # Falling back to a single empty group would break AdamW; surface the
        # configuration error instead.
        raise ValueError("No trainable parameters found in the model.")
    return groups
