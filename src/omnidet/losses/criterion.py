import math

import torch
from torch import nn
import torch.nn.functional as F

from omnidet.losses.boxes import (
    box_cxcywh_to_xyxy,
    generalized_box_iou,
    normalized_wasserstein_distance,
)
from omnidet.losses.matcher import HungarianMatcher


def _small_object_weight(
    target_boxes_cxcywh: torch.Tensor,
    small_box_weight: float,
    min_small_box_pixels: int,
    image_size: int,
) -> torch.Tensor:
    """Per-box loss multiplier that upweights tiny targets.

    The weight scales from 1.0 for "large enough" boxes (>= ``min_small_box_pixels``
    on the shorter side) up to ``small_box_weight`` for the smallest boxes. The
    function uses sqrt(area) so a 16-pixel box gets the same weight regardless of
    aspect ratio, and clamps so the multiplier never blows up on degenerate
    boxes. Coordinates are normalised cxcywh in [0, 1].
    """
    if small_box_weight <= 1.0 or target_boxes_cxcywh.numel() == 0:
        return torch.ones(target_boxes_cxcywh.shape[0], device=target_boxes_cxcywh.device)
    side_pixels = torch.sqrt(
        target_boxes_cxcywh[:, 2] * target_boxes_cxcywh[:, 3]
    ).clamp(min=1e-6) * float(image_size)
    # Linear interpolation: 0 px -> small_box_weight, min_small_box_pixels px -> 1.0
    fraction = (side_pixels / max(float(min_small_box_pixels), 1e-6)).clamp(min=0.0, max=1.0)
    return small_box_weight - (small_box_weight - 1.0) * fraction


class DetectionCriterion(nn.Module):
    def __init__(
        self,
        num_classes: int,
        eos_coef: float = 0.1,
        bbox_weight: float = 5.0,
        giou_weight: float = 2.0,
        dn_weight: float = 1.0,
        # --- Small-object hooks (Task #3, #4) ---
        small_box_weight: float = 1.0,
        min_small_box_pixels: int = 32,
        image_size: int = 960,
        # NWD blending; weight=0 disables. Constant tuned for normalised
        # VisDrone-scale boxes (objects typically 1-3% wide).
        nwd_weight: float = 0.0,
        nwd_constant: float = 0.05,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        # Matcher and box-regression loss share nwd_weight so the matching cost
        # surface is aligned with the gradient signal the model sees afterwards.
        self.matcher = HungarianMatcher(
            nwd_cost=float(nwd_weight),
            nwd_constant=float(nwd_constant),
        )
        self.bbox_weight = bbox_weight
        self.giou_weight = giou_weight
        self.dn_weight = dn_weight
        self.small_box_weight = float(small_box_weight)
        self.min_small_box_pixels = int(min_small_box_pixels)
        self.image_size = int(image_size)
        self.nwd_weight = float(nwd_weight)
        self.nwd_constant = float(nwd_constant)
        weight = torch.ones(num_classes + 1)
        weight[-1] = eos_coef
        self.register_buffer("empty_weight", weight)

    def _dn_loss(
        self, outputs: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Direct supervised loss on the prepended denoising queries.

        DN pairs are predefined (each DN query came from a specific GT), so no
        Hungarian matching is needed. The padding slots are masked out via
        ``dn_valid_mask``.
        """
        dn_logits = outputs["pred_logits_dn"]
        dn_boxes = outputs["pred_boxes_dn"]
        dn_target_boxes = outputs["dn_target_boxes"]
        dn_target_labels = outputs["dn_target_labels"]
        dn_valid = outputs["dn_valid_mask"]

        zero = dn_logits.sum() * 0.0
        if not dn_valid.any():
            return zero, zero, zero

        # Classification on all DN slots (including padding -> "no object" =
        # num_classes), using the eos weight to dampen the padding contribution.
        loss_cls = F.cross_entropy(
            dn_logits.transpose(1, 2),
            dn_target_labels,
            weight=self.empty_weight,
        )

        valid_flat = dn_valid.reshape(-1)
        boxes_flat = dn_boxes.reshape(-1, 4)
        tgt_boxes_flat = dn_target_boxes.reshape(-1, 4)
        sel_pred = boxes_flat[valid_flat]
        sel_tgt = tgt_boxes_flat[valid_flat]
        if sel_pred.numel() == 0:
            return loss_cls, zero, zero
        weight = _small_object_weight(
            sel_tgt, self.small_box_weight, self.min_small_box_pixels, self.image_size
        )
        # Per-box L1 then weighted average. ``F.l1_loss(reduction='none')``
        # returns (N, 4); average across the 4 box coords first to mirror the
        # original mean(), then weight per-instance.
        per_box_l1 = F.l1_loss(sel_pred, sel_tgt, reduction="none").mean(dim=-1)
        loss_bbox = (per_box_l1 * weight).sum() / weight.sum().clamp(min=1e-6)
        giou = generalized_box_iou(box_cxcywh_to_xyxy(sel_pred), box_cxcywh_to_xyxy(sel_tgt))
        per_box_giou = 1.0 - torch.diag(giou)
        loss_giou = (per_box_giou * weight).sum() / weight.sum().clamp(min=1e-6)
        if self.nwd_weight > 0.0:
            nwd = normalized_wasserstein_distance(sel_pred, sel_tgt, constant=self.nwd_constant)
            per_box_nwd = 1.0 - torch.diag(nwd)
            loss_nwd = (per_box_nwd * weight).sum() / weight.sum().clamp(min=1e-6)
            loss_giou = loss_giou + self.nwd_weight * loss_nwd
        return loss_cls, loss_bbox, loss_giou

    def forward(
        self, outputs: dict[str, torch.Tensor], targets: list[dict[str, torch.Tensor]]
    ) -> dict[str, torch.Tensor]:
        matches = self.matcher(outputs, targets)
        batch_size, num_queries, _ = outputs["pred_logits"].shape
        target_classes = torch.full(
            (batch_size, num_queries),
            self.num_classes,
            dtype=torch.long,
            device=outputs["pred_logits"].device,
        )
        pred_boxes: list[torch.Tensor] = []
        target_boxes: list[torch.Tensor] = []
        for batch_index, (pred_idx, target_idx) in enumerate(matches):
            if pred_idx.numel() == 0:
                continue
            labels = targets[batch_index]["labels"].to(outputs["pred_logits"].device)
            boxes = targets[batch_index]["boxes"].to(outputs["pred_boxes"].device)
            target_classes[batch_index, pred_idx] = labels[target_idx]
            pred_boxes.append(outputs["pred_boxes"][batch_index, pred_idx])
            target_boxes.append(boxes[target_idx])

        loss_cls = F.cross_entropy(
            outputs["pred_logits"].transpose(1, 2),
            target_classes,
            weight=self.empty_weight,
        )
        if pred_boxes:
            matched_pred_boxes = torch.cat(pred_boxes, dim=0)
            matched_target_boxes = torch.cat(target_boxes, dim=0)
            weight = _small_object_weight(
                matched_target_boxes,
                self.small_box_weight,
                self.min_small_box_pixels,
                self.image_size,
            )
            per_box_l1 = F.l1_loss(matched_pred_boxes, matched_target_boxes, reduction="none").mean(dim=-1)
            loss_bbox = (per_box_l1 * weight).sum() / weight.sum().clamp(min=1e-6)
            giou = generalized_box_iou(
                box_cxcywh_to_xyxy(matched_pred_boxes), box_cxcywh_to_xyxy(matched_target_boxes)
            )
            per_box_giou = 1.0 - torch.diag(giou)
            loss_giou = (per_box_giou * weight).sum() / weight.sum().clamp(min=1e-6)
            if self.nwd_weight > 0.0:
                nwd = normalized_wasserstein_distance(
                    matched_pred_boxes, matched_target_boxes, constant=self.nwd_constant
                )
                per_box_nwd = 1.0 - torch.diag(nwd)
                loss_nwd = (per_box_nwd * weight).sum() / weight.sum().clamp(min=1e-6)
                loss_giou = loss_giou + self.nwd_weight * loss_nwd
        else:
            loss_bbox = outputs["pred_boxes"].sum() * 0.0
            loss_giou = outputs["pred_boxes"].sum() * 0.0
        loss_total = loss_cls + self.bbox_weight * loss_bbox + self.giou_weight * loss_giou

        result = {
            "loss_cls": loss_cls,
            "loss_bbox": loss_bbox,
            "loss_giou": loss_giou,
        }
        # DN auxiliary loss when DN queries were used in this forward pass.
        if "pred_logits_dn" in outputs:
            dn_cls, dn_bbox, dn_giou = self._dn_loss(outputs)
            dn_total = dn_cls + self.bbox_weight * dn_bbox + self.giou_weight * dn_giou
            loss_total = loss_total + self.dn_weight * dn_total
            result["loss_dn_cls"] = dn_cls
            result["loss_dn_bbox"] = dn_bbox
            result["loss_dn_giou"] = dn_giou
            result["loss_dn_total"] = dn_total

        result["loss_total"] = loss_total
        return result


class MultiTaskCriterion(nn.Module):
    def __init__(
        self,
        num_classes: int,
        active_tasks: tuple[str, ...] = ("detection",),
        task_loss_weights: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        *,
        seg_num_classes: int | None = None,
        num_keypoints: int = 17,
        mesh_num_vertices: int = 478,
        mesh_vertex_dim: int = 3,
        reid_embedding_dim: int = 128,
        depth_target_size: int | None = None,
        mesh_distill_from_teachers: bool = True,
        eos_coef: float = 0.1,
        bbox_weight: float = 5.0,
        giou_weight: float = 2.0,
        small_box_weight: float = 1.0,
        min_small_box_pixels: int = 32,
        image_size: int = 960,
        nwd_weight: float = 0.0,
        nwd_constant: float = 0.05,
    ) -> None:
        super().__init__()
        self.active_tasks = tuple(active_tasks)
        self.task_weights = {
            name: 1.0 for name in ("detection", "segmentation", "pose", "mesh", "reid", "depth", "quality")
        }
        for index, task_name in enumerate(active_tasks):
            if index < len(task_loss_weights):
                self.task_weights[task_name] = task_loss_weights[index]
        self.num_classes = num_classes
        self.seg_num_classes = seg_num_classes
        self.num_keypoints = num_keypoints
        self.mesh_num_vertices = max(1, mesh_num_vertices)
        self.mesh_vertex_dim = max(1, mesh_vertex_dim)
        self.reid_embedding_dim = max(1, reid_embedding_dim)
        self.depth_target_size = depth_target_size
        self.mesh_distill_from_teachers = mesh_distill_from_teachers
        self.detection_criterion = DetectionCriterion(
            num_classes=num_classes,
            eos_coef=eos_coef,
            bbox_weight=bbox_weight,
            giou_weight=giou_weight,
            small_box_weight=small_box_weight,
            min_small_box_pixels=min_small_box_pixels,
            image_size=image_size,
            nwd_weight=nwd_weight,
            nwd_constant=nwd_constant,
        )
        self.matcher = HungarianMatcher()

    @staticmethod
    def _zero(reference: torch.Tensor) -> torch.Tensor:
        return reference.sum() * 0.0

    def _normalize_keypoints(self, keypoints: torch.Tensor) -> torch.Tensor:
        if keypoints.numel() == 0:
            return keypoints.reshape(0, self.num_keypoints, 3).to(dtype=torch.float32)
        if keypoints.dim() == 2:
            keypoints = keypoints.reshape(-1, 3)
        if keypoints.dim() == 1:
            return torch.zeros((0, self.num_keypoints, 3), dtype=torch.float32, device=keypoints.device)
        if keypoints.shape[1] % 3 != 0:
            return torch.zeros((0, self.num_keypoints, 3), dtype=torch.float32, device=keypoints.device)

        expanded = keypoints.reshape(keypoints.shape[0], -1, 3)
        if expanded.shape[1] == self.num_keypoints:
            return expanded.to(dtype=torch.float32)
        padded = torch.zeros(
            (expanded.shape[0], self.num_keypoints, 3),
            device=expanded.device,
            dtype=expanded.dtype,
        )
        keep = min(expanded.shape[1], self.num_keypoints)
        padded[:, :keep, :] = expanded[:, :keep, :].to(dtype=torch.float32)
        return padded.to(dtype=torch.float32)

    def _segmentation_loss(self, pred_seg: torch.Tensor, targets: list[dict[str, torch.Tensor]]) -> torch.Tensor:
        losses: list[torch.Tensor] = []
        num_classes = self.seg_num_classes if self.seg_num_classes is not None else pred_seg.shape[1]
        for batch_index, target in enumerate(targets):
            mask = target.get("seg_mask")
            if mask is None or mask.numel() == 0:
                continue
            mask = mask.to(device=pred_seg.device, dtype=torch.long)
            if mask.dim() != 2:
                continue
            if mask.shape != pred_seg.shape[-2:]:
                mask = F.interpolate(
                    mask[None, None].to(dtype=torch.float32),
                    size=pred_seg.shape[-2:],
                    mode="nearest",
                )[0, 0].to(dtype=torch.long)
            if mask.max().item() >= num_classes:
                mask = torch.clamp(mask, 0, num_classes - 1)
            losses.append(F.cross_entropy(pred_seg[batch_index][None], mask[None]))
        if not losses:
            return self._zero(pred_seg[0])
        return torch.stack(losses).mean()

    def _pose_loss(self, outputs: dict[str, torch.Tensor], targets: list[dict[str, torch.Tensor]]) -> torch.Tensor:
        pred_kpts = outputs["pred_keypoints"]
        matches = self.matcher(outputs, targets)
        pose_losses: list[torch.Tensor] = []
        for batch_index, (pred_idx, target_idx) in enumerate(matches):
            if pred_idx.numel() == 0:
                continue
            gt_raw = targets[batch_index].get("keypoints", torch.empty((0, self.num_keypoints, 3)))
            gt_kpts = self._normalize_keypoints(gt_raw.to(device=pred_kpts.device))
            if gt_kpts.shape[0] == 0 or pred_kpts.shape[2] != gt_kpts.shape[1]:
                continue
            selected_pred = pred_kpts[batch_index, pred_idx]
            selected_gt = gt_kpts[target_idx]
            visible = selected_gt[:, :, 2] > 0
            if visible.any():
                pose_losses.append(F.l1_loss(selected_pred[:, :, :2][visible], selected_gt[:, :, :2][visible]))
            else:
                pose_losses.append(self._zero(selected_pred))
        if not pose_losses:
            return self._zero(pred_kpts)
        return torch.stack(pose_losses).mean()

    @staticmethod
    def _mesh_to_tensor(
        raw: torch.Tensor | list[float] | list[list[float]] | None,
        num_vertices: int,
        vertex_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if raw is None:
            return torch.zeros((0, num_vertices, vertex_dim), device=device, dtype=dtype)
        if not isinstance(raw, torch.Tensor):
            raw = torch.as_tensor(raw, dtype=dtype, device=device)
        if raw.numel() == 0:
            return torch.zeros((0, num_vertices, vertex_dim), device=device, dtype=dtype)

        if raw.dim() == 1:
            if raw.numel() % vertex_dim != 0:
                return torch.zeros((0, num_vertices, vertex_dim), device=device, dtype=dtype)
            raw = raw.reshape(-1, vertex_dim)

        if raw.dim() == 2 and raw.shape[1] == vertex_dim:
            points = raw[:, :vertex_dim]
        elif raw.dim() == 3 and raw.shape[-1] == vertex_dim:
            points = raw.reshape(-1, vertex_dim)
        else:
            return torch.zeros((0, num_vertices, vertex_dim), device=device, dtype=dtype)

        if points.shape[0] >= num_vertices:
            return points[:num_vertices, :vertex_dim]
        padded = torch.zeros((num_vertices, vertex_dim), device=device, dtype=dtype)
        padded[: points.shape[0]] = points.to(dtype=dtype)
        return padded

    def _collect_mesh_targets(self, raw: object, device: torch.device, dtype: torch.dtype) -> list[torch.Tensor]:
        meshes: list[torch.Tensor] = []
        if raw is None:
            return meshes
        if isinstance(raw, torch.Tensor):
            if raw.dim() == 3:
                for idx in range(raw.shape[0]):
                    mesh = self._mesh_to_tensor(raw[idx], self.mesh_num_vertices, self.mesh_vertex_dim, device, dtype)
                    if mesh.numel() > 0:
                        meshes.append(mesh)
            else:
                mesh = self._mesh_to_tensor(raw, self.mesh_num_vertices, self.mesh_vertex_dim, device, dtype)
                if mesh.numel() > 0:
                    meshes.append(mesh)
            return meshes

        if isinstance(raw, (list, tuple)):
            candidate = None
            try:
                candidate = torch.as_tensor(raw, dtype=dtype, device=device)
            except (TypeError, ValueError):
                candidate = None
            if candidate is None:
                for entry in raw:
                    candidate_mesh = self._mesh_to_tensor(
                        entry, self.mesh_num_vertices, self.mesh_vertex_dim, device, dtype
                    )
                    if candidate_mesh.numel() > 0:
                        meshes.append(candidate_mesh)
                return meshes
            if candidate.dim() == 3:
                for idx in range(candidate.shape[0]):
                    mesh = self._mesh_to_tensor(candidate[idx], self.mesh_num_vertices, self.mesh_vertex_dim, device, dtype)
                    if mesh.numel() > 0:
                        meshes.append(mesh)
            elif candidate.dim() == 2 and candidate.shape[1] == self.mesh_vertex_dim:
                mesh = self._mesh_to_tensor(candidate, self.mesh_num_vertices, self.mesh_vertex_dim, device, dtype)
                if mesh.numel() > 0:
                    meshes.append(mesh)
            elif candidate.dim() == 1:
                mesh = self._mesh_to_tensor(candidate, self.mesh_num_vertices, self.mesh_vertex_dim, device, dtype)
                if mesh.numel() > 0:
                    meshes.append(mesh)
            else:
                for entry in raw:
                    candidate_mesh = self._mesh_to_tensor(
                        entry, self.mesh_num_vertices, self.mesh_vertex_dim, device, dtype
                    )
                    if candidate_mesh.numel() > 0:
                        meshes.append(candidate_mesh)
        return meshes

    @staticmethod
    def _align_vector(vector: torch.Tensor, target_dim: int, *, fill_value: float = 0.0) -> torch.Tensor:
        if vector.numel() == 0:
            return torch.zeros((0, target_dim), dtype=vector.dtype, device=vector.device)
        if vector.dim() == 1:
            vector = vector.reshape(1, -1)
        if vector.shape[1] == target_dim:
            return vector
        aligned = torch.full((vector.shape[0], target_dim), fill_value, dtype=vector.dtype, device=vector.device)
        keep = min(vector.shape[1], target_dim)
        aligned[:, :keep] = vector[:, :keep]
        return aligned

    @staticmethod
    def _coerce_to_tensor(value: object, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if torch.is_tensor(value):
            return value.to(device=device, dtype=dtype)
        if isinstance(value, (list, tuple)):
            try:
                return torch.as_tensor(value, dtype=dtype, device=device)
            except (TypeError, ValueError):
                return torch.empty((0,), device=device, dtype=dtype)
        return torch.empty((0,), device=device, dtype=dtype)

    def _reid_loss(self, outputs: dict[str, torch.Tensor], targets: list[dict[str, object]]) -> torch.Tensor:
        pred_reid = outputs["pred_reid"]
        matches = self.matcher(outputs, targets)
        losses: list[torch.Tensor] = []
        for batch_index, (pred_idx, target_idx) in enumerate(matches):
            if pred_idx.numel() == 0:
                continue
            raw = targets[batch_index].get("reid_embedding")
            if raw is None:
                raw = targets[batch_index].get("reid")
            if raw is None:
                continue
            gt_reid = self._coerce_to_tensor(raw, pred_reid.device, pred_reid.dtype)
            if gt_reid.numel() == 0:
                continue
            if gt_reid.dim() == 1 and gt_reid.numel() == self.reid_embedding_dim:
                gt_reid = gt_reid.reshape(1, -1)
            if gt_reid.dim() != 2:
                continue
            if gt_reid.shape[1] != self.reid_embedding_dim:
                gt_reid = self._align_vector(gt_reid, self.reid_embedding_dim)
            if target_idx.max().item() >= gt_reid.shape[0]:
                continue
            selected_pred = pred_reid[batch_index, pred_idx]
            selected_gt = gt_reid[target_idx]
            if selected_gt.shape != selected_pred.shape:
                selected_gt = self._align_vector(
                    selected_gt,
                    selected_pred.shape[-1],
                )
                selected_gt = selected_gt.reshape_as(selected_pred)
            losses.append(F.mse_loss(selected_pred, selected_gt))
        if not losses:
            return self._zero(pred_reid)
        return torch.stack(losses).mean()

    def _depth_loss(self, outputs: dict[str, torch.Tensor], targets: list[dict[str, object]]) -> torch.Tensor:
        pred_depth = outputs["pred_depth"]
        losses: list[torch.Tensor] = []
        for batch_index, target in enumerate(targets):
            raw = target.get("depth")
            if raw is None:
                raw = target.get("depth_map")
            if raw is None:
                continue
            gt_depth = self._coerce_to_tensor(raw, pred_depth.device, pred_depth.dtype)
            if gt_depth.numel() == 0:
                continue
            if gt_depth.dim() == 1:
                side = int(math.sqrt(gt_depth.numel()))
                if side * side == gt_depth.numel():
                    gt_depth = gt_depth.reshape(1, 1, side, side)
                else:
                    gt_depth = gt_depth.reshape(1, 1, 1, -1)
            elif gt_depth.dim() == 2:
                gt_depth = gt_depth.unsqueeze(0).unsqueeze(0)
            elif gt_depth.dim() == 3:
                if gt_depth.shape[0] != 1:
                    gt_depth = gt_depth[0:1]
                gt_depth = gt_depth.unsqueeze(0) if gt_depth.dim() == 3 else gt_depth
            elif gt_depth.dim() == 4:
                pass
            else:
                continue
            if gt_depth.shape[-2:] != pred_depth.shape[-2:]:
                gt_depth = F.interpolate(
                    gt_depth.to(dtype=pred_depth.dtype),
                    size=pred_depth.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            losses.append(F.mse_loss(pred_depth[batch_index : batch_index + 1], gt_depth.to(device=pred_depth.device)))
        if not losses:
            return self._zero(pred_depth)
        return torch.stack(losses).mean()

    def _quality_loss(self, outputs: dict[str, torch.Tensor], targets: list[dict[str, object]]) -> torch.Tensor:
        pred_quality = outputs["pred_quality"]
        matches = self.matcher(outputs, targets)
        losses: list[torch.Tensor] = []
        for batch_index, (pred_idx, target_idx) in enumerate(matches):
            if pred_idx.numel() == 0:
                continue
            raw = targets[batch_index].get("quality")
            if raw is None:
                continue
            gt_quality = self._coerce_to_tensor(raw, pred_quality.device, pred_quality.dtype).reshape(-1)
            if gt_quality.numel() == 0 or target_idx.max().item() >= gt_quality.shape[0]:
                continue
            selected_pred = pred_quality[batch_index, pred_idx].squeeze(-1)
            selected_gt = gt_quality[target_idx]
            if selected_gt.shape != selected_pred.shape:
                continue
            losses.append(F.mse_loss(selected_pred, selected_gt))
        if not losses:
            return self._zero(pred_quality)
        return torch.stack(losses).mean()

    def _mesh_loss(self, outputs: dict[str, torch.Tensor], targets: list[dict[str, torch.Tensor]]) -> torch.Tensor:
        pred_mesh = outputs["pred_mesh"]
        device = pred_mesh.device
        dtype = pred_mesh.dtype
        mesh_losses: list[torch.Tensor] = []
        for batch_index, target in enumerate(targets):
            mesh_candidates: list[torch.Tensor] = []
            gt_mesh = target.get("mesh")
            if gt_mesh is not None:
                mesh_candidates.extend(self._collect_mesh_targets(gt_mesh, device, dtype))

            if self.mesh_distill_from_teachers:
                teacher_mesh = target.get("mesh_teachers", target.get("mesh_teacher"))
                teacher_meshes = self._collect_mesh_targets(teacher_mesh, device, dtype)
                mesh_candidates.extend(teacher_meshes)

            if not mesh_candidates:
                continue
            target_mesh = torch.stack(mesh_candidates, dim=0).mean(dim=0)
            target_mesh = target_mesh.to(device=device, dtype=dtype)
            mesh_losses.append(F.mse_loss(pred_mesh[batch_index], target_mesh))

        if not mesh_losses:
            return self._zero(pred_mesh)
        return torch.stack(mesh_losses).mean()

    def forward(
        self, outputs: dict[str, torch.Tensor], targets: list[dict[str, torch.Tensor]]
    ) -> dict[str, torch.Tensor]:
        losses: dict[str, torch.Tensor] = {}
        reference = next(iter(outputs.values()))
        total_loss = torch.zeros((), device=reference.device)

        if "detection" in self.active_tasks and {"pred_logits", "pred_boxes"} <= outputs.keys():
            detection_losses = self.detection_criterion(outputs, targets)
            det_weight = torch.as_tensor(self.task_weights["detection"], device=reference.device)
            total_loss = total_loss + detection_losses["loss_total"] * det_weight
            losses["loss_det_total"] = detection_losses["loss_total"] * det_weight
            losses["loss_cls"] = detection_losses["loss_cls"]
            losses["loss_bbox"] = detection_losses["loss_bbox"]
            losses["loss_giou"] = detection_losses["loss_giou"]

        if "segmentation" in self.active_tasks and "pred_seg" in outputs:
            seg_weight = torch.as_tensor(self.task_weights["segmentation"], device=reference.device)
            seg_loss = self._segmentation_loss(outputs["pred_seg"], targets)
            losses["loss_seg"] = seg_loss
            losses["loss_seg_total"] = seg_loss * seg_weight
            total_loss = total_loss + losses["loss_seg_total"]
        elif "segmentation" in self.active_tasks:
            losses["loss_seg"] = self._zero(outputs["pred_seg"]) if "pred_seg" in outputs else self._zero(reference)
            losses["loss_seg_total"] = losses["loss_seg"]

        if "pose" in self.active_tasks and {"pred_keypoints", "pred_logits", "pred_boxes"} <= outputs.keys():
            pose_weight = torch.as_tensor(self.task_weights["pose"], device=reference.device)
            pose_loss = self._pose_loss(outputs, targets)
            losses["loss_pose"] = pose_loss
            losses["loss_pose_total"] = pose_loss * pose_weight
            total_loss = total_loss + losses["loss_pose_total"]
        elif "pose" in self.active_tasks:
            losses["loss_pose"] = self._zero(outputs["pred_keypoints"]) if "pred_keypoints" in outputs else self._zero(reference)
            losses["loss_pose_total"] = losses["loss_pose"]

        if "mesh" in self.active_tasks and "pred_mesh" in outputs:
            mesh_weight = torch.as_tensor(self.task_weights["mesh"], device=reference.device)
            mesh_loss = self._mesh_loss(outputs, targets)
            losses["loss_mesh"] = mesh_loss
            losses["loss_mesh_total"] = mesh_loss * mesh_weight
            total_loss = total_loss + losses["loss_mesh_total"]

        if "reid" in self.active_tasks and "pred_reid" in outputs:
            reid_weight = torch.as_tensor(self.task_weights["reid"], device=reference.device)
            reid_loss = self._reid_loss(outputs, targets)
            losses["loss_reid"] = reid_loss
            losses["loss_reid_total"] = reid_loss * reid_weight
            total_loss = total_loss + losses["loss_reid_total"]
        elif "reid" in self.active_tasks:
            losses["loss_reid"] = self._zero(outputs["pred_reid"]) if "pred_reid" in outputs else self._zero(reference)
            losses["loss_reid_total"] = losses["loss_reid"]

        if "depth" in self.active_tasks and "pred_depth" in outputs:
            depth_weight = torch.as_tensor(self.task_weights["depth"], device=reference.device)
            depth_loss = self._depth_loss(outputs, targets)
            losses["loss_depth"] = depth_loss
            losses["loss_depth_total"] = depth_loss * depth_weight
            total_loss = total_loss + losses["loss_depth_total"]
        elif "depth" in self.active_tasks:
            losses["loss_depth"] = self._zero(outputs["pred_depth"]) if "pred_depth" in outputs else self._zero(reference)
            losses["loss_depth_total"] = losses["loss_depth"]

        if "quality" in self.active_tasks and "pred_quality" in outputs:
            quality_weight = torch.as_tensor(self.task_weights["quality"], device=reference.device)
            quality_loss = self._quality_loss(outputs, targets)
            losses["loss_quality"] = quality_loss
            losses["loss_quality_total"] = quality_loss * quality_weight
            total_loss = total_loss + losses["loss_quality_total"]
        elif "quality" in self.active_tasks:
            losses["loss_quality"] = (
                self._zero(outputs["pred_quality"]) if "pred_quality" in outputs else self._zero(reference)
            )
            losses["loss_quality_total"] = losses["loss_quality"]

        if "loss_det_total" not in losses:
            losses["loss_det_total"] = self._zero(reference)
        if "loss_seg_total" not in losses:
            losses["loss_seg_total"] = losses.get("loss_seg", self._zero(reference))
        if "loss_pose_total" not in losses:
            losses["loss_pose_total"] = losses.get("loss_pose", self._zero(reference))
        if "loss_mesh_total" not in losses:
            losses["loss_mesh_total"] = losses.get("loss_mesh", self._zero(reference))
        if "loss_reid_total" not in losses:
            losses["loss_reid_total"] = losses.get("loss_reid", self._zero(reference))
        if "loss_depth_total" not in losses:
            losses["loss_depth_total"] = losses.get("loss_depth", self._zero(reference))
        if "loss_quality_total" not in losses:
            losses["loss_quality_total"] = losses.get("loss_quality", self._zero(reference))

        if "loss_seg" in losses and losses["loss_seg"].dtype != total_loss.dtype:
            losses["loss_seg"] = losses["loss_seg"].to(dtype=total_loss.dtype)
        if "loss_pose" in losses and losses["loss_pose"].dtype != total_loss.dtype:
            losses["loss_pose"] = losses["loss_pose"].to(dtype=total_loss.dtype)
        if "loss_mesh" in losses and losses["loss_mesh"].dtype != total_loss.dtype:
            losses["loss_mesh"] = losses["loss_mesh"].to(dtype=total_loss.dtype)
        if "loss_reid" in losses and losses["loss_reid"].dtype != total_loss.dtype:
            losses["loss_reid"] = losses["loss_reid"].to(dtype=total_loss.dtype)
        if "loss_depth" in losses and losses["loss_depth"].dtype != total_loss.dtype:
            losses["loss_depth"] = losses["loss_depth"].to(dtype=total_loss.dtype)
        if "loss_quality" in losses and losses["loss_quality"].dtype != total_loss.dtype:
            losses["loss_quality"] = losses["loss_quality"].to(dtype=total_loss.dtype)
        if "loss_cls" in losses and losses["loss_cls"].dtype != total_loss.dtype:
            losses["loss_cls"] = losses["loss_cls"].to(dtype=total_loss.dtype)

        losses["loss_total"] = total_loss
        return losses
