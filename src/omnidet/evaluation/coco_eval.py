from __future__ import annotations

import contextlib
import io
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from omnidet.training.acceleration import images_to_device


def collect_coco_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device | str,
    amp_dtype: torch.dtype | None = None,
    channels_last: bool = False,
    score_thresh: float = 0.05,
) -> list[dict]:
    torch_device = torch.device(device)
    category_ids: tuple[int, ...] = loader.dataset.category_ids
    results: list[dict] = []

    model.eval()
    with torch.no_grad():
        for images, targets in loader:
            images = images_to_device(images, torch_device, channels_last)
            use_amp = amp_dtype is not None and torch_device.type == "cuda"
            with torch.autocast(device_type=torch_device.type, dtype=amp_dtype, enabled=use_amp):
                outputs = model(images)

            pred_logits = outputs["pred_logits"]  # (B, Q, C+1)
            pred_boxes = outputs["pred_boxes"]    # (B, Q, 4) normalized cxcywh
            scores_all = pred_logits.softmax(-1)[..., :-1]  # (B, Q, C)

            for i, target in enumerate(targets):
                image_id = int(target["image_id"])
                orig_size = target.get("orig_size")
                if orig_size is not None:
                    orig_w, orig_h = float(orig_size[0]), float(orig_size[1])
                else:
                    orig_w = orig_h = float(images.shape[-1])

                q_scores, q_labels = scores_all[i].max(-1)  # (Q,)
                keep = q_scores > score_thresh
                if not keep.any():
                    continue

                q_scores = q_scores[keep].cpu()
                q_labels = q_labels[keep].cpu()
                boxes = pred_boxes[i][keep].cpu()  # (K, 4)

                cx, cy, bw, bh = boxes.unbind(-1)
                x1 = (cx - bw / 2) * orig_w
                y1 = (cy - bh / 2) * orig_h
                abs_w = bw * orig_w
                abs_h = bh * orig_h

                for score, label, xi, yi, wi, hi in zip(
                    q_scores.tolist(),
                    q_labels.tolist(),
                    x1.tolist(), y1.tolist(), abs_w.tolist(), abs_h.tolist(),
                ):
                    if label >= len(category_ids):
                        continue
                    results.append({
                        "image_id": image_id,
                        "category_id": int(category_ids[label]),
                        "bbox": [xi, yi, wi, hi],
                        "score": float(score),
                    })

    return results


# COCO eval's stats vector layout (matches pycocotools and Ultralytics):
#   0: AP @ IoU=0.50:0.95 all      | 1: AP50 all      | 2: AP75 all
#   3: AP small (<32^2)            | 4: AP medium     | 5: AP large
#   6: AR maxDets=1                | 7: AR maxDets=10 | 8: AR maxDets=100
#   9: AR small                    | 10: AR medium    | 11: AR large
_COCO_STAT_KEYS = (
    "mAP", "mAP50", "mAP75",
    "AP_small", "AP_medium", "AP_large",
    "AR_1", "AR_10", "AR_100",
    "AR_small", "AR_medium", "AR_large",
)


def _empty_metrics() -> dict[str, float]:
    return {key: 0.0 for key in _COCO_STAT_KEYS} | {"AR_lt16": 0.0, "AR_lt32": 0.0}


def run_coco_map(annotations_path: Path, predictions: list[dict]) -> dict[str, float]:
    """Score COCO-format predictions against COCO-format ground truth.

    Returns the full pycocotools 12-element AP/AR table (the same metrics
    Ultralytics YOLO reports), plus the design-doc-required small-object
    recall at the <16x16 and <32x32 thresholds derived from a second pass
    with custom area ranges.
    """
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError:
        print("pycocotools not installed — skipping mAP", flush=True)
        return {}

    coco_gt = COCO(str(annotations_path))
    if not predictions:
        return _empty_metrics()

    coco_dt = coco_gt.loadRes(predictions)
    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    with contextlib.redirect_stdout(io.StringIO()):
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    metrics = {key: float(evaluator.stats[i]) for i, key in enumerate(_COCO_STAT_KEYS)}

    # Second pass with custom area ranges for the design-doc <16^2 and <32^2
    # small-object recall bins. pycocotools reports AR for the last areaRng
    # entries, so we put 'all' first and the custom bins after.
    custom = COCOeval(coco_gt, coco_dt, "bbox")
    custom.params.areaRng = [[0.0, 1e10], [0.0, 16 ** 2], [0.0, 32 ** 2]]
    custom.params.areaRngLbl = ["all", "lt16", "lt32"]
    with contextlib.redirect_stdout(io.StringIO()):
        custom.evaluate()
        custom.accumulate()
    # AR at maxDets=100 is the recall axis: precision/recall shape is
    # (T, K, A, M); take area indices 1,2 (lt16, lt32) and average over T, K.
    recall = custom.eval["recall"]  # (T, K, A, M)
    metrics["AR_lt16"] = float(recall[:, :, 1, -1][recall[:, :, 1, -1] > -1].mean()) if recall.size else 0.0
    metrics["AR_lt32"] = float(recall[:, :, 2, -1][recall[:, :, 2, -1] > -1].mean()) if recall.size else 0.0
    return metrics
