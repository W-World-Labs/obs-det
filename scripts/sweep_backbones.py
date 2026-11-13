"""Sweep candidate backbones through the profiling gate before training.

For each backbone preset (plus the clean-room baseline) this builds an OmniDet
model on random weights and reports parameter counts and per-component latency.
Backbones that blow the RTX 5090 latency budget, or that cannot expose the
required P2/P3/P4/P5 pyramid, are rejected here - before any training compute.

Pretrained weights are NOT downloaded: latency and parameter counts do not
depend on the weights, only on the architecture.
"""

import argparse
import csv
import json
import time
from dataclasses import replace
from pathlib import Path

import torch

from omnidet.hardware import get_hardware_profile, model_config_for_profile
from omnidet.models.backbone_presets import BACKBONE_PRESETS, apply_backbone_preset
from omnidet.models.detector import OmniDet
from omnidet.training.acceleration import apply_memory_format, precision_to_dtype, prepare_model


def count_params(module: torch.nn.Module) -> int:
    return sum(param.numel() for param in module.parameters())


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is not available")
    return torch.device(name)


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def time_callable(fn, warmup: int, iters: int, device: torch.device) -> float:
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        sync(device)
        start = time.perf_counter()
        for _ in range(iters):
            fn()
        sync(device)
        elapsed = time.perf_counter() - start
    return elapsed * 1000.0 / max(1, iters)


def profile_one(
    cfg,
    device: torch.device,
    amp_dtype,
    *,
    batch_size: int,
    channels_last: bool,
    warmup: int,
    iters: int,
) -> dict[str, object]:
    model = OmniDet(cfg)
    params = {
        "total": count_params(model),
        "backbone": count_params(model.backbone),
        "neck": count_params(model.neck),
        "head": count_params(model.head),
    }
    model = prepare_model(model, device=device, channels_last=channels_last, compile_model=False)
    model.eval()
    images = apply_memory_format(
        torch.randn(batch_size, 3, cfg.input_size, cfg.input_size, device=device), channels_last
    )
    use_amp = amp_dtype is not None and device.type == "cuda"

    def autocast():
        return torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp)

    with torch.inference_mode(), autocast():
        feats = model.backbone(images)
        neck_out = model.neck(feats)

    def run_backbone():
        with autocast():
            model.backbone(images)

    def run_neck():
        with autocast():
            model.neck(feats)

    def run_head():
        with autocast():
            model.head(neck_out)

    def run_full():
        with autocast():
            model(images)

    backbone_ms = time_callable(run_backbone, warmup, iters, device)
    neck_ms = time_callable(run_neck, warmup, iters, device)
    head_ms = time_callable(run_head, warmup, iters, device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    full_ms = time_callable(run_full, warmup, iters, device)
    memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if device.type == "cuda" else 0.0
    trunk_share = 100.0 * (backbone_ms + neck_ms) / full_ms if full_ms > 0 else 0.0
    return {
        "params": params,
        "backbone_ms": backbone_ms,
        "neck_ms": neck_ms,
        "head_ms": head_ms,
        "full_ms": full_ms,
        "fps": 1000.0 / full_ms if full_ms > 0 else 0.0,
        "trunk_share": trunk_share,
        "memory_mb": memory_mb,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--hardware-profile", choices=["rtx5090_32g", "rtxpro6000_96g"], default="rtx5090_32g"
    )
    parser.add_argument(
        "--presets",
        nargs="*",
        choices=sorted(BACKBONE_PRESETS),
        help="subset of backbone presets to sweep (default: all)",
    )
    parser.add_argument("--no-cleanroom", dest="include_cleanroom", action="store_false")
    parser.set_defaults(include_cleanroom=True)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], help="default: hardware profile")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--image-size", type=int, help="override profile input size")
    parser.add_argument("--no-channels-last", dest="channels_last", action="store_false")
    parser.set_defaults(channels_last=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--csv-out", type=Path)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    device = resolve_device(args.device)
    profile = get_hardware_profile(args.hardware_profile)
    precision = args.precision or profile.precision
    amp_dtype = precision_to_dtype(precision)
    base_cfg = model_config_for_profile(args.hardware_profile)
    if args.image_size:
        base_cfg = replace(base_cfg, input_size=args.image_size)

    candidates: list[tuple[str, object]] = []
    if args.include_cleanroom:
        candidates.append((f"cleanroom:{base_cfg.backbone_name}", base_cfg))
    for preset_name in args.presets or sorted(BACKBONE_PRESETS):
        cfg = apply_backbone_preset(base_cfg, preset_name)
        # latency/params do not depend on trained weights - skip the download.
        cfg = replace(cfg, backbone_pretrained=False)
        candidates.append((preset_name, cfg))

    print("=" * 96)
    print(
        f"backbone sweep  profile={profile.name}  device={device}  precision={precision}  "
        f"image_size={base_cfg.input_size}  batch_size={args.batch_size}"
    )
    print("=" * 96)

    rows: list[tuple[str, object, dict | None, str]] = []
    for name, cfg in candidates:
        try:
            result = profile_one(
                cfg,
                device,
                amp_dtype,
                batch_size=args.batch_size,
                channels_last=args.channels_last,
                warmup=args.warmup,
                iters=args.iters,
            )
            status = "ok"
        except Exception as exc:  # noqa: BLE001 - a rejected backbone is information, not a crash
            result = None
            status = f"{type(exc).__name__}: {exc}"
        rows.append((name, cfg, result, status))

    header = (
        f"{'backbone':<28} {'params(M)':>10} {'bb(ms)':>8} {'neck(ms)':>9} "
        f"{'head(ms)':>9} {'full(ms)':>9} {'fps':>8} {'trunk%':>8} {'mem(MB)':>9}"
    )
    print(header)
    print("-" * len(header))
    for name, _cfg, result, status in rows:
        if result is None:
            print(f"{name:<28} REJECTED  {status}")
            continue
        params = result["params"]
        print(
            f"{name:<28} {params['total'] / 1e6:>10.2f} {result['backbone_ms']:>8.2f} "
            f"{result['neck_ms']:>9.2f} {result['head_ms']:>9.2f} {result['full_ms']:>9.2f} "
            f"{result['fps']:>8.1f} {result['trunk_share']:>7.1f}% {result['memory_mb']:>9.1f}"
        )

    print(
        "\nnote: backbone/neck/head are timed in isolation; kernel fusion and overlap "
        "in the full forward can make the parts sum to more than 100% of 'full'."
    )
    if device.type != "cuda":
        print("(CPU run - latency numbers are not representative of the RTX 5090 target)")

    if args.csv_out:
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        with args.csv_out.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "backbone", "status", "params_total", "params_backbone", "params_neck",
                    "params_head", "backbone_ms", "neck_ms", "head_ms", "full_ms", "fps",
                    "trunk_share_pct", "memory_mb",
                ]
            )
            for name, _cfg, result, status in rows:
                if result is None:
                    writer.writerow([name, status, "", "", "", "", "", "", "", "", "", "", ""])
                    continue
                params = result["params"]
                writer.writerow(
                    [
                        name, "ok", params["total"], params["backbone"], params["neck"],
                        params["head"], f"{result['backbone_ms']:.4f}", f"{result['neck_ms']:.4f}",
                        f"{result['head_ms']:.4f}", f"{result['full_ms']:.4f}", f"{result['fps']:.2f}",
                        f"{result['trunk_share']:.2f}", f"{result['memory_mb']:.1f}",
                    ]
                )
        print(f"\nwrote {args.csv_out}")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "hardware_profile": profile.name,
            "device": str(device),
            "precision": precision,
            "image_size": base_cfg.input_size,
            "batch_size": args.batch_size,
            "results": [
                {"backbone": name, "status": status, "metrics": result}
                for name, _cfg, result, status in rows
            ],
        }
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {args.json_out}")

    rejected = [name for name, _cfg, result, _status in rows if result is None]
    if rejected:
        print(f"\nrejected: {', '.join(rejected)}")
    print("\nsweep done - shortlist the survivors and compare APs on tiny-VisDrone")


if __name__ == "__main__":
    main()
