from collections.abc import Mapping

import torch
from torch import nn
from torch.utils.data import DataLoader

from omnidet.training.acceleration import apply_memory_format, images_to_device


def move_targets_to_device(
    targets: list[dict[str, object]], device: torch.device | str
) -> list[dict[str, object]]:
    def _to_device(value: object) -> object:
        if torch.is_tensor(value):
            return value.to(device)
        if isinstance(value, list):
            return [_to_device(entry) for entry in value]
        if isinstance(value, tuple):
            return tuple(_to_device(entry) for entry in value)
        if isinstance(value, dict):
            return {key: _to_device(item) for key, item in value.items()}
        return value

    return [{key: _to_device(value) for key, value in target.items()} for target in targets]


def train_one_step(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    images: torch.Tensor,
    targets: list[dict[str, object]],
    device: torch.device | str,
    amp_dtype: torch.dtype | None = None,
    grad_scaler: torch.amp.GradScaler | None = None,
    channels_last: bool = False,
    grad_clip_norm: float | None = None,
) -> dict[str, float]:
    model.train()
    torch_device = torch.device(device)
    images = images_to_device(images, torch_device, channels_last)
    targets = move_targets_to_device(targets, device)
    optimizer.zero_grad(set_to_none=True)
    use_amp = amp_dtype is not None and torch_device.type == "cuda"
    with torch.autocast(device_type=torch_device.type, dtype=amp_dtype, enabled=use_amp):
        # Pass targets so the head can seed DN queries when supported.
        # ``OmniDet.forward`` accepts targets=None for backward compat.
        try:
            outputs = model(images, targets=targets)
        except TypeError:
            # Compiled / wrapped models that don't accept ``targets`` kwarg.
            outputs = model(images)
        losses: Mapping[str, torch.Tensor] = criterion(outputs, targets)
    if grad_scaler is not None and grad_scaler.is_enabled():
        grad_scaler.scale(losses["loss_total"]).backward()
        grad_scaler.unscale_(optimizer)
        if grad_clip_norm is not None and grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        grad_scaler.step(optimizer)
        grad_scaler.update()
    else:
        losses["loss_total"].backward()
        if grad_clip_norm is not None and grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        optimizer.step()
    # Return detached GPU tensors instead of forcing a .cpu() sync every step.
    # Callers that need a Python number can call .item() at log time (every Nth
    # step), turning 4N+ syncs/sec into a handful and letting the GPU stay busy.
    return {name: value.detach() for name, value in losses.items()}


@torch.no_grad()
def eval_one_epoch(
    model: nn.Module,
    criterion: nn.Module,
    loader: DataLoader,
    device: torch.device | str,
    amp_dtype: torch.dtype | None = None,
    channels_last: bool = False,
) -> dict[str, float]:
    model.eval()
    torch_device = torch.device(device)
    totals: dict[str, float] = {}
    for images, targets in loader:
        images = images_to_device(images, torch_device, channels_last)
        targets = move_targets_to_device(targets, torch_device)
        use_amp = amp_dtype is not None and torch_device.type == "cuda"
        with torch.autocast(device_type=torch_device.type, dtype=amp_dtype, enabled=use_amp):
            outputs = model(images)
            losses: Mapping[str, torch.Tensor] = criterion(outputs, targets)
        for key, value in losses.items():
            totals[key] = totals.get(key, 0.0) + value.detach()
    n = max(len(loader), 1)
    return {key: total / n for key, total in totals.items()}
