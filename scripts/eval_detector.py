"""Generic mAP evaluation for any detector that produces COCO-format predictions.

Two modes:

  1. From an OmniDet checkpoint: load it, run inference on the val set, score.
  2. From an existing COCO-format predictions JSON (e.g. dumped by a YOLO
     baseline): skip inference, score directly.

Both paths use pycocotools - the same evaluator Ultralytics uses internally -
so the reported numbers are directly comparable to any external baseline scored
on the same ground-truth JSON.
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from omnidet.config import ModelConfig
from omnidet.data.dataset import COCODetectionDataset, collate_detections
from omnidet.evaluation.coco_eval import collect_coco_predictions, run_coco_map
from omnidet.models.detector import OmniDet
from omnidet.training.acceleration import precision_to_dtype, prepare_model


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is not available")
    return torch.device(name)


def predictions_from_checkpoint(args: argparse.Namespace) -> list[dict]:
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg: ModelConfig = checkpoint["model_cfg"]
    if args.image_size:
        cfg = replace(cfg, input_size=args.image_size)

    dataset = COCODetectionDataset(args.images, args.annotations, image_size=cfg.input_size)
    loader_kwargs: dict = {}
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_detections,
        pin_memory=torch.cuda.is_available(),
        **loader_kwargs,
    )

    model = OmniDet(cfg)
    state = checkpoint["model"]
    # Strip a torch.compile prefix if the checkpoint was saved from a compiled model.
    if any(k.startswith("_orig_mod.") for k in state.keys()):
        state = {k.replace("_orig_mod.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state)
    model = prepare_model(model, device=device, channels_last=args.channels_last, compile_model=False)
    amp_dtype = precision_to_dtype(args.precision)

    print(f"running inference: {len(dataset)} images, input_size={cfg.input_size}, batch={args.batch_size}")
    return collect_coco_predictions(
        model,
        loader,
        device,
        amp_dtype=amp_dtype,
        channels_last=args.channels_last,
        score_thresh=args.score_thresh,
    )


def print_metrics_table(metrics: dict[str, float]) -> None:
    print()
    print("COCO mAP table (same metrics Ultralytics YOLO reports):")
    print(f"  mAP @[0.50:0.95]      = {metrics['mAP']:.4f}")
    print(f"  AP50                  = {metrics['mAP50']:.4f}")
    print(f"  AP75                  = {metrics['mAP75']:.4f}")
    print(f"  AP_small  (<32^2)     = {metrics['AP_small']:.4f}")
    print(f"  AP_medium             = {metrics['AP_medium']:.4f}")
    print(f"  AP_large              = {metrics['AP_large']:.4f}")
    print(f"  AR_max_1              = {metrics['AR_1']:.4f}")
    print(f"  AR_max_10             = {metrics['AR_10']:.4f}")
    print(f"  AR_max_100            = {metrics['AR_100']:.4f}")
    print(f"  AR_small  (<32^2)     = {metrics['AR_small']:.4f}")
    print(f"  AR_medium             = {metrics['AR_medium']:.4f}")
    print(f"  AR_large              = {metrics['AR_large']:.4f}")
    print()
    print("Design-doc small-object recall (custom area bins, AR @maxDets=100):")
    print(f"  AR @ area<16^2        = {metrics['AR_lt16']:.4f}")
    print(f"  AR @ area<32^2        = {metrics['AR_lt32']:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--annotations", type=Path, required=True, help="COCO json ground truth")
    parser.add_argument("--images", type=Path, help="images dir (required when using --checkpoint)")
    parser.add_argument("--checkpoint", type=Path, help="OmniDet checkpoint to evaluate")
    parser.add_argument(
        "--predictions",
        type=Path,
        help="existing COCO predictions json (skip inference; e.g. YOLO baseline output)",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--image-size", type=int, help="override checkpoint input_size")
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--no-channels-last", dest="channels_last", action="store_false")
    parser.set_defaults(channels_last=True)
    parser.add_argument("--score-thresh", type=float, default=0.05)
    parser.add_argument(
        "--predictions-out",
        type=Path,
        help="dump inference predictions to this path (COCO json) for re-use / external comparison",
    )
    parser.add_argument("--json-out", type=Path, help="write metrics dict to JSON")
    parser.add_argument("--tag", default="omnidet", help="label printed in the header (for comparison tables)")
    args = parser.parse_args()

    if not args.checkpoint and not args.predictions:
        parser.error("provide either --checkpoint (run inference) or --predictions (score existing json)")
    if args.checkpoint and not args.images:
        parser.error("--checkpoint requires --images")

    if args.predictions:
        predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
        print(f"loaded {len(predictions)} predictions from {args.predictions}")
    else:
        predictions = predictions_from_checkpoint(args)
        print(f"got {len(predictions)} raw predictions")
        if args.predictions_out:
            args.predictions_out.parent.mkdir(parents=True, exist_ok=True)
            args.predictions_out.write_text(json.dumps(predictions))
            print(f"wrote predictions to {args.predictions_out}")

    print(f"\nscoring [{args.tag}] against {args.annotations}")
    metrics = run_coco_map(args.annotations, predictions)
    if not metrics:
        raise SystemExit("pycocotools not available — cannot score")
    print_metrics_table(metrics)

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        payload = {"tag": args.tag, "annotations": str(args.annotations), "metrics": metrics}
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
