import torch
from scipy.optimize import linear_sum_assignment

from omnidet.losses.boxes import (
    box_cxcywh_to_xyxy,
    generalized_box_iou,
    normalized_wasserstein_distance,
)


class HungarianMatcher:
    def __init__(
        self,
        class_cost: float = 2.0,
        bbox_cost: float = 5.0,
        giou_cost: float = 2.0,
        nwd_cost: float = 0.0,
        nwd_constant: float = 0.05,
    ) -> None:
        self.class_cost = class_cost
        self.bbox_cost = bbox_cost
        self.giou_cost = giou_cost
        # NWD cost helps the matcher when IoU/GIoU is degenerate on tiny boxes
        # (arXiv 2110.13389). Disabled by default (weight=0); enable via
        # train_visdrone.py --nwd-weight + the criterion's nwd_weight forwarded
        # by build_criterion (the matcher and the loss use the same weight so
        # the cost surface and the gradient signal stay aligned).
        self.nwd_cost = nwd_cost
        self.nwd_constant = nwd_constant

    @torch.no_grad()
    def __call__(
        self, outputs: dict[str, torch.Tensor], targets: list[dict[str, torch.Tensor]]
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        probabilities = outputs["pred_logits"].softmax(-1)
        pred_boxes = outputs["pred_boxes"]
        matches: list[tuple[torch.Tensor, torch.Tensor]] = []
        for batch_index, target in enumerate(targets):
            target_labels = target["labels"].to(pred_boxes.device)
            target_boxes = target["boxes"].to(pred_boxes.device)
            if target_labels.numel() == 0:
                matches.append(
                    (
                        torch.empty(0, dtype=torch.long, device=pred_boxes.device),
                        torch.empty(0, dtype=torch.long, device=pred_boxes.device),
                    )
                )
                continue
            class_cost = -probabilities[batch_index][:, target_labels]
            bbox_cost = torch.cdist(pred_boxes[batch_index], target_boxes, p=1)
            giou_cost = -generalized_box_iou(
                box_cxcywh_to_xyxy(pred_boxes[batch_index]), box_cxcywh_to_xyxy(target_boxes)
            )
            total_cost = (
                self.class_cost * class_cost
                + self.bbox_cost * bbox_cost
                + self.giou_cost * giou_cost
            )
            if self.nwd_cost > 0.0:
                # -NWD because higher NWD = better match = lower cost.
                nwd_cost = -normalized_wasserstein_distance(
                    pred_boxes[batch_index], target_boxes, constant=self.nwd_constant
                )
                total_cost = total_cost + self.nwd_cost * nwd_cost
            pred_indices, target_indices = linear_sum_assignment(total_cost.cpu().numpy())
            matches.append(
                (
                    torch.as_tensor(pred_indices, dtype=torch.long, device=pred_boxes.device),
                    torch.as_tensor(target_indices, dtype=torch.long, device=pred_boxes.device),
                )
            )
        return matches
