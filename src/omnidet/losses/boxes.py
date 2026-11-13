import torch


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack((cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h), dim=-1)


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    sizes = (boxes[:, 2:] - boxes[:, :2]).clamp(min=0)
    return sizes[:, 0] * sizes[:, 1]


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    left_top = torch.maximum(boxes1[:, None, :2], boxes2[:, :2])
    right_bottom = torch.minimum(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (right_bottom - left_top).clamp(min=0)
    intersection = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - intersection
    return intersection / union.clamp(min=1e-6), union


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    iou, union = box_iou(boxes1, boxes2)
    left_top = torch.minimum(boxes1[:, None, :2], boxes2[:, :2])
    right_bottom = torch.maximum(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (right_bottom - left_top).clamp(min=0)
    enclosing_area = wh[:, :, 0] * wh[:, :, 1]
    return iou - (enclosing_area - union) / enclosing_area.clamp(min=1e-6)


def normalized_wasserstein_distance(
    boxes1_cxcywh: torch.Tensor,
    boxes2_cxcywh: torch.Tensor,
    *,
    constant: float = 0.05,
) -> torch.Tensor:
    """Pairwise NWD similarity between cxcywh boxes (arXiv 2110.13389).

    Treats each box as a 2D Gaussian (mean = centre, std = half-width / half-
    height) and uses the closed-form Wasserstein-2 distance:

        W² = (μ₁ - μ₂)² + (σ₁ - σ₂)²       (diagonal covariance case)
        NWD = exp(-√W² / C)

    Unlike IoU, NWD stays well-behaved when boxes are sub-pixel small, because
    the metric is continuous in box shape and never collapses to zero union.
    The ``constant`` is an aerial-imagery dataset hyperparameter — for
    normalized [0,1] coordinates on VisDrone-scale objects, C ≈ 0.05 keeps the
    similarity sensitive across the typical 1-3% box-size range.

    Returns a (N, M) similarity tensor in (0, 1] — 1 means identical Gaussian.
    """
    cx1, cy1, w1, h1 = boxes1_cxcywh.unbind(-1)
    cx2, cy2, w2, h2 = boxes2_cxcywh.unbind(-1)
    # Outer-product broadcast: (N, M)
    d_cx = cx1[:, None] - cx2[None, :]
    d_cy = cy1[:, None] - cy2[None, :]
    d_w = 0.5 * (w1[:, None] - w2[None, :])
    d_h = 0.5 * (h1[:, None] - h2[None, :])
    w2_squared = d_cx**2 + d_cy**2 + d_w**2 + d_h**2
    return torch.exp(-torch.sqrt(w2_squared.clamp(min=1e-12)) / constant)
