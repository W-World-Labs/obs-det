"""Random-model smoke gate for the OmniDet R&D flywheel.

Runs the cheap checks that must pass before a full training run is worth its
compute, all on an untrained model:

  1. build        - construct model + criterion, report per-component params
  2. numerics     - forward/criterion/backward, check shapes, NaN/Inf, grad flow
  3. profile      - per-component latency so the shared-trunk premise is verified
  4. overfit      - tiny fixed batch, confirm loss_total actually goes down
  5. onnx         - v0 deployment graph (detection head only) exports cleanly

Exit code is non-zero if any gate fails, so this can act as a CI gate.
"""

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import torch

from omnidet.export.onnx import export_onnx
from omnidet.hardware import get_hardware_profile, model_config_for_profile
from omnidet.losses.criterion import DetectionCriterion, MultiTaskCriterion
from omnidet.models.detector import OmniDet
from omnidet.training.acceleration import (
    apply_memory_format,
    make_adamw,
    precision_to_dtype,
    prepare_model,
)
from omnidet.training.engine import train_one_step

SUPPORTED_TASKS = ("detection", "segmentation", "pose", "mesh", "reid", "depth", "quality")

TASK_OUTPUT_KEYS = {
    "detection": ("pred_logits", "pred_boxes"),
    "segmentation": ("pred_seg",),
    "pose": ("pred_keypoints",),
    "mesh": ("pred_mesh",),
    "reid": ("pred_reid",),
    "depth": ("pred_depth",),
    "quality": ("pred_quality",),
}

TASK_LOSS_KEYS = {
    "detection": "loss_det_total",
    "segmentation": "loss_seg_total",
    "pose": "loss_pose_total",
    "mesh": "loss_mesh_total",
    "reid": "loss_reid_total",
    "depth": "loss_depth_total",
    "quality": "loss_quality_total",
}

GATE_NAMES = ("numerics", "profile", "overfit", "onnx")


def parse_tasks(value: str) -> tuple[str, ...]:
    tasks = tuple(task.strip().lower() for task in value.split(",") if task.strip())
    if "all" in tasks:
        return SUPPORTED_TASKS
    if not tasks:
        raise argparse.ArgumentTypeError("tasks cannot be empty")
    unknown = [task for task in tasks if task not in SUPPORTED_TASKS]
    if unknown:
        raise argparse.ArgumentTypeError(f"unsupported tasks: {', '.join(sorted(unknown))}")
    return tasks


def count_params(module: torch.nn.Module) -> int:
    return sum(param.numel() for param in module.parameters())


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is not available")
    return torch.device(name)


def build_model_config(args: argparse.Namespace, active_tasks: tuple[str, ...]) -> "ModelConfig":
    profile_cfg = model_config_for_profile(args.hardware_profile)
    image_size = args.image_size or profile_cfg.input_size
    task_loss_weights = tuple(1.0 for _ in active_tasks)
    return replace(
        profile_cfg,
        input_size=image_size,
        num_classes=args.num_classes,
        active_tasks=active_tasks,
        task_loss_weights=task_loss_weights,
        num_keypoints=args.num_keypoints,
        seg_num_classes=args.seg_num_classes,
        reid_embedding_dim=args.reid_embedding_dim,
        mesh_enabled=args.mesh_enabled or "mesh" in active_tasks,
        mesh_num_vertices=args.mesh_num_vertices,
        mesh_vertex_dim=args.mesh_vertex_dim,
    )


def build_criterion(cfg: "ModelConfig", device: torch.device) -> torch.nn.Module:
    if tuple(cfg.active_tasks) == ("detection",):
        return DetectionCriterion(num_classes=cfg.num_classes).to(device)
    return MultiTaskCriterion(
        num_classes=cfg.num_classes,
        active_tasks=cfg.active_tasks,
        task_loss_weights=cfg.task_loss_weights,
        seg_num_classes=cfg.seg_num_classes,
        num_keypoints=cfg.num_keypoints,
        reid_embedding_dim=cfg.reid_embedding_dim,
        mesh_num_vertices=cfg.mesh_num_vertices,
        mesh_vertex_dim=cfg.mesh_vertex_dim,
    ).to(device)


def make_synthetic_targets(
    batch_size: int, cfg: "ModelConfig", device: torch.device, *, num_objects: int, seed: int
) -> list[dict[str, object]]:
    """Fixed synthetic targets matching the dataset contract for every active task.

    Targets are deterministic so the overfit gate has a stable objective to drive
    loss_total toward zero.
    """
    generator = torch.Generator().manual_seed(seed)
    seg_classes = cfg.seg_num_classes if cfg.seg_num_classes is not None else cfg.num_classes + 1
    targets: list[dict[str, object]] = []
    for index in range(batch_size):
        labels = torch.randint(0, max(1, cfg.num_classes), (num_objects,), generator=generator)
        centers = 0.2 + 0.6 * torch.rand((num_objects, 2), generator=generator)
        sizes = 0.05 + 0.15 * torch.rand((num_objects, 2), generator=generator)
        target: dict[str, object] = {
            "labels": labels.to(device),
            "boxes": torch.cat([centers, sizes], dim=1).to(device),
            "image_id": torch.tensor(index, dtype=torch.long, device=device),
            "orig_size": torch.tensor([cfg.input_size, cfg.input_size], dtype=torch.long, device=device),
        }
        if "segmentation" in cfg.active_tasks:
            target["seg_mask"] = torch.randint(
                0, max(1, seg_classes), (cfg.input_size, cfg.input_size), generator=generator
            ).to(device)
        if "pose" in cfg.active_tasks:
            target["keypoints"] = torch.rand(
                (num_objects, cfg.num_keypoints, 3), generator=generator
            ).to(device)
        if "mesh" in cfg.active_tasks:
            target["mesh"] = torch.rand(
                (cfg.mesh_num_vertices, cfg.mesh_vertex_dim), generator=generator
            ).to(device)
            target["mesh_teachers"] = []
        if "reid" in cfg.active_tasks:
            target["reid_embedding"] = torch.rand(
                (num_objects, cfg.reid_embedding_dim), generator=generator
            ).to(device)
        if "depth" in cfg.active_tasks:
            target["depth"] = torch.rand(
                (cfg.input_size, cfg.input_size), generator=generator
            ).to(device)
        if "quality" in cfg.active_tasks:
            target["quality"] = torch.rand((num_objects,), generator=generator).to(device)
        targets.append(target)
    return targets


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


def gate_numerics(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    cfg: "ModelConfig",
    device: torch.device,
    amp_dtype,
    *,
    batch_size: int,
    channels_last: bool,
    num_objects: int,
) -> tuple[bool, str]:
    errors: list[str] = []
    model.train()
    images = apply_memory_format(
        torch.randn(batch_size, 3, cfg.input_size, cfg.input_size, device=device), channels_last
    )
    targets = make_synthetic_targets(batch_size, cfg, device, num_objects=num_objects, seed=0)
    use_amp = amp_dtype is not None and device.type == "cuda"
    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
        outputs = model(images)
        losses = criterion(outputs, targets)

    for task in cfg.active_tasks:
        for key in TASK_OUTPUT_KEYS[task]:
            if key not in outputs:
                errors.append(f"missing output '{key}' for task '{task}'")
            elif not torch.isfinite(outputs[key]).all():
                errors.append(f"output '{key}' has NaN/Inf")

    if "loss_total" not in losses:
        errors.append("criterion did not return loss_total")
    elif not torch.isfinite(losses["loss_total"]).all():
        errors.append("loss_total is NaN/Inf")

    if isinstance(criterion, MultiTaskCriterion):
        for task in cfg.active_tasks:
            loss_key = TASK_LOSS_KEYS[task]
            if loss_key not in losses:
                errors.append(f"missing loss '{loss_key}' for task '{task}'")
            elif not torch.isfinite(losses[loss_key]).all():
                errors.append(f"loss '{loss_key}' is NaN/Inf")

    grad_detail = ""
    if "loss_total" in losses and torch.isfinite(losses["loss_total"]).all():
        model.zero_grad(set_to_none=True)
        losses["loss_total"].backward()
        grad_norm_sum = 0.0
        params_with_grad = 0
        nan_grad = False
        for param in model.parameters():
            if param.grad is None:
                continue
            if not torch.isfinite(param.grad).all():
                nan_grad = True
                continue
            norm = float(param.grad.norm())
            grad_norm_sum += norm
            if norm > 0.0:
                params_with_grad += 1
        if nan_grad:
            errors.append("NaN/Inf in gradients")
        if params_with_grad == 0:
            errors.append("no parameter received a non-zero gradient")
        grad_detail = f", grad_norm_sum={grad_norm_sum:.3e}, params_with_grad={params_with_grad}"
        model.zero_grad(set_to_none=True)

    if errors:
        return False, "; ".join(errors)
    loss_total = float(losses["loss_total"])
    return True, f"outputs+losses finite, loss_total={loss_total:.4f}{grad_detail}"


def gate_profile(
    model: torch.nn.Module,
    cfg: "ModelConfig",
    device: torch.device,
    amp_dtype,
    *,
    batch_size: int,
    channels_last: bool,
    warmup: int,
    iters: int,
) -> tuple[bool, str]:
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

    trunk_ms = backbone_ms + neck_ms
    trunk_share = 100.0 * trunk_ms / full_ms if full_ms > 0 else 0.0
    fps = 1000.0 / full_ms if full_ms > 0 else 0.0
    print(
        f"    backbone={backbone_ms:.3f}ms  neck={neck_ms:.3f}ms  head={head_ms:.3f}ms  "
        f"full={full_ms:.3f}ms  fps={fps:.1f}"
    )
    print(f"    shared trunk (backbone+neck) = {trunk_share:.1f}% of full forward")
    if device.type == "cuda":
        memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        print(f"    max_memory={memory_mb:.1f}MB  batch_size={batch_size}")
    else:
        print("    (CPU run - latency numbers are not representative of the RTX 5090 target)")
    return True, (
        f"full={full_ms:.3f}ms fps={fps:.1f} trunk_share={trunk_share:.1f}% "
        f"(backbone={backbone_ms:.3f} neck={neck_ms:.3f} head={head_ms:.3f})"
    )


def gate_overfit(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    cfg: "ModelConfig",
    device: torch.device,
    amp_dtype,
    *,
    batch_size: int,
    channels_last: bool,
    precision: str,
    steps: int,
    lr: float,
    target_loss: float,
    num_objects: int,
) -> tuple[bool, str]:
    torch.manual_seed(1234)
    images = torch.randn(batch_size, 3, cfg.input_size, cfg.input_size, device=device)
    targets = make_synthetic_targets(batch_size, cfg, device, num_objects=num_objects, seed=1234)
    optimizer = make_adamw(
        model.parameters(), lr=lr, weight_decay=0.0, fused=device.type == "cuda"
    )
    grad_scaler = torch.amp.GradScaler(
        "cuda", enabled=precision == "fp16" and device.type == "cuda"
    )
    first_loss = None
    last_loss = None
    for step in range(steps):
        losses = train_one_step(
            model,
            criterion,
            optimizer,
            images,
            targets,
            device,
            amp_dtype=amp_dtype,
            grad_scaler=grad_scaler,
            channels_last=channels_last,
        )
        last_loss = losses["loss_total"]
        if first_loss is None:
            first_loss = last_loss
        if step == 0 or (step + 1) % max(1, steps // 5) == 0:
            print(f"    step {step + 1:4d}/{steps}  loss_total={last_loss:.5f}")

    if first_loss is None or last_loss is None:
        return False, "no steps ran"
    if not (last_loss == last_loss):  # NaN check
        return False, f"loss_total became NaN (start={first_loss:.5f})"
    ratio = last_loss / first_loss if first_loss > 0 else float("inf")
    learnable = last_loss < first_loss * 0.5 or last_loss < target_loss
    detail = (
        f"loss_total {first_loss:.5f} -> {last_loss:.5f} (ratio={ratio:.3f}) over {steps} steps"
    )
    if not learnable:
        return False, "loss did not drop enough - " + detail
    return True, detail


def gate_onnx(cfg: "ModelConfig", out_path: Path) -> tuple[bool, str]:
    if "detection" not in cfg.active_tasks:
        return True, "skipped: detection not active, v0 ONNX graph is detection-only"
    export_cfg = replace(cfg, active_tasks=("detection",), mesh_enabled=False)
    export_onnx(out_path, export_cfg)
    if not out_path.exists():
        return False, f"export_onnx returned but {out_path} is missing"
    size_kb = out_path.stat().st_size / 1024.0
    try:
        import onnx

        onnx.checker.check_model(str(out_path))
        return True, f"exported {out_path} ({size_kb:.0f}KB), onnx.checker passed"
    except ImportError:
        return True, f"exported {out_path} ({size_kb:.0f}KB), onnx not installed - checker skipped"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--hardware-profile",
        choices=["rtx5090_32g", "rtxpro6000_96g"],
        default="rtx5090_32g",
    )
    parser.add_argument("--tasks", type=parse_tasks, default=("detection",), help="comma list or 'all'")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], help="default: hardware profile")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--image-size", type=int, help="override profile input size (use 320-512 for a fast run)")
    parser.add_argument("--num-classes", type=int, default=8)
    parser.add_argument("--num-objects", type=int, default=4, help="synthetic objects per image")
    parser.add_argument("--no-channels-last", dest="channels_last", action="store_false")
    parser.set_defaults(channels_last=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--overfit-steps", type=int, default=150)
    parser.add_argument("--overfit-lr", type=float, default=1e-3)
    parser.add_argument("--overfit-target-loss", type=float, default=0.05)
    parser.add_argument("--onnx-out", type=Path, default=Path("artifacts/smoke_omnidet.onnx"))
    parser.add_argument(
        "--skip",
        nargs="*",
        choices=GATE_NAMES,
        default=[],
        help="gates to skip",
    )
    parser.add_argument("--num-keypoints", type=int, default=17)
    parser.add_argument("--seg-num-classes", type=int)
    parser.add_argument("--reid-embedding-dim", type=int, default=128)
    parser.add_argument("--mesh-enabled", action="store_true")
    parser.add_argument("--mesh-num-vertices", type=int, default=478)
    parser.add_argument("--mesh-vertex-dim", type=int, default=3)
    parser.add_argument("--json-out", type=Path, help="optional path to dump gate results as JSON")
    args = parser.parse_args()

    active_tasks = tuple(args.tasks)
    device = resolve_device(args.device)
    profile = get_hardware_profile(args.hardware_profile)
    precision = args.precision or profile.precision
    amp_dtype = precision_to_dtype(precision)
    cfg = build_model_config(args, active_tasks)

    print("=" * 72)
    print("OmniDet smoke check (random model)")
    print("=" * 72)
    print(
        f"hardware_profile={profile.name} device={device} precision={precision} "
        f"image_size={cfg.input_size} batch_size={args.batch_size}"
    )
    print(f"active_tasks={active_tasks} num_classes={cfg.num_classes}")

    # Gate 0: build + per-component parameter report.
    model = OmniDet(cfg)
    component_params = {
        "backbone": count_params(model.backbone),
        "neck": count_params(model.neck),
        "head": count_params(model.head),
    }
    head_params = {}
    for attr in (
        "detection_head",
        "segmentation_head",
        "pose_head",
        "mesh_head",
        "reid_head",
        "depth_head",
        "quality_head",
    ):
        submodule = getattr(model.head, attr, None)
        if submodule is not None:
            head_params[attr] = count_params(submodule)
    total_params = count_params(model)
    feature_spec = model.neck.feature_spec
    print(f"\n[build] total params = {total_params / 1e6:.2f}M")
    print(
        f"    feature contract: levels={feature_spec.levels} "
        f"strides={feature_spec.strides} channels={feature_spec.channels}"
    )
    for name, value in component_params.items():
        print(f"    {name:<10} {value / 1e6:7.3f}M")
    for name, value in head_params.items():
        print(f"      - {name:<18} {value / 1e6:7.3f}M")

    model = prepare_model(model, device=device, channels_last=args.channels_last, compile_model=False)
    criterion = build_criterion(cfg, device)

    results: dict[str, dict[str, object]] = {}

    def record(name: str, passed: bool, detail: str, skipped: bool = False) -> None:
        status = "SKIP" if skipped else ("PASS" if passed else "FAIL")
        results[name] = {"status": status, "detail": detail}

    gates = [
        (
            "numerics",
            lambda: gate_numerics(
                model, criterion, cfg, device, amp_dtype,
                batch_size=args.batch_size, channels_last=args.channels_last,
                num_objects=args.num_objects,
            ),
        ),
        (
            "profile",
            lambda: gate_profile(
                model, cfg, device, amp_dtype,
                batch_size=args.batch_size, channels_last=args.channels_last,
                warmup=args.warmup, iters=args.iters,
            ),
        ),
        (
            "overfit",
            lambda: gate_overfit(
                model, criterion, cfg, device, amp_dtype,
                batch_size=args.batch_size, channels_last=args.channels_last,
                precision=precision, steps=args.overfit_steps, lr=args.overfit_lr,
                target_loss=args.overfit_target_loss, num_objects=args.num_objects,
            ),
        ),
        ("onnx", lambda: gate_onnx(cfg, args.onnx_out)),
    ]

    for name, runner in gates:
        if name in args.skip:
            print(f"\n[{name}] skipped")
            record(name, True, "skipped via --skip", skipped=True)
            continue
        print(f"\n[{name}] running...")
        try:
            passed, detail = runner()
        except Exception as exc:  # noqa: BLE001 - smoke gate must report, not crash
            passed, detail = False, f"{type(exc).__name__}: {exc}"
        record(name, passed, detail)
        print(f"[{name}] {results[name]['status']}: {detail}")

    print("\n" + "=" * 72)
    print("summary")
    print("=" * 72)
    for name, _ in gates:
        entry = results[name]
        print(f"  {name:<10} {entry['status']:<5} {entry['detail']}")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(
                {
                    "hardware_profile": profile.name,
                    "device": str(device),
                    "precision": precision,
                    "active_tasks": list(active_tasks),
                    "total_params": total_params,
                    "component_params": component_params,
                    "head_params": head_params,
                    "gates": results,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nwrote {args.json_out}")

    failed = [name for name, entry in results.items() if entry["status"] == "FAIL"]
    if failed:
        print(f"\nFAILED gates: {', '.join(failed)} - do not start a full training run")
        raise SystemExit(1)
    print("\nall gates passed - the flywheel is clear to train")


if __name__ == "__main__":
    main()
