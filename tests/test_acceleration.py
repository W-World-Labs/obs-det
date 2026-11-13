import torch

from omnidet.training.acceleration import (
    apply_memory_format,
    configure_cuda_kernels,
    images_to_device,
    make_adamw,
    precision_to_dtype,
)


def test_precision_to_dtype_maps_training_precisions() -> None:
    assert precision_to_dtype("fp32") is None
    assert precision_to_dtype("fp16") is torch.float16
    assert precision_to_dtype("bf16") is torch.bfloat16


def test_apply_memory_format_supports_channels_last() -> None:
    tensor = torch.randn(1, 3, 16, 16)

    converted = apply_memory_format(tensor, channels_last=True)

    assert converted.is_contiguous(memory_format=torch.channels_last)


def test_make_adamw_falls_back_when_fused_is_unavailable() -> None:
    model = torch.nn.Linear(4, 2)

    optimizer = make_adamw(model.parameters(), lr=1e-4, weight_decay=1e-4, fused=False)

    assert isinstance(optimizer, torch.optim.AdamW)


def test_configure_cuda_kernels_is_safe_without_cuda() -> None:
    state = configure_cuda_kernels(enable_tf32=True, cudnn_benchmark=True)

    assert "cuda_available" in state


def test_images_to_device_converts_uint8_hwc_to_float_chw() -> None:
    images = torch.full((2, 8, 8, 3), 255, dtype=torch.uint8)

    out = images_to_device(images, device="cpu", channels_last=False)

    assert out.shape == (2, 3, 8, 8)
    assert out.dtype == torch.float32
    assert torch.allclose(out, torch.ones_like(out))


def test_images_to_device_passes_float_chw_through() -> None:
    images = torch.rand((2, 3, 8, 8), dtype=torch.float32)

    out = images_to_device(images, device="cpu", channels_last=False)

    assert out.shape == (2, 3, 8, 8)
    assert out.dtype == torch.float32
    assert torch.allclose(out, images)
