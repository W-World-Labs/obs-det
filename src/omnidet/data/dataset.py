import math
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageEnhance
from torch.utils.data import Dataset

from omnidet.data.visdrone import parse_visdrone_annotation


def _random_scale_jitter_crop(
    image: Image.Image,
    boxes: list[tuple[float, float, float, float]],
    labels: list[int],
    target_size: int,
    *,
    scale_range: tuple[float, float] = (0.5, 1.5),
    min_box_keep: float = 0.3,
) -> tuple[Image.Image, list[tuple[float, float, float, float]], list[int]]:
    """Random crop on the original image, then resize to ``target_size``.

    Equivalent to scale jitter at the input — the single highest-impact
    augmentation for small-object detection in the VisDrone/UAV domain. Boxes
    are translated and scaled with the crop; boxes with less than
    ``min_box_keep`` of their original area inside the crop are dropped (and
    their corresponding labels dropped in sync).
    """
    width, height = image.size
    scale = random.uniform(*scale_range)
    crop_w = max(1, int(round(width * scale)))
    crop_h = max(1, int(round(height * scale)))
    x0 = random.randint(0, max(0, width - crop_w))
    y0 = random.randint(0, max(0, height - crop_h))
    cropped = image.crop((x0, y0, x0 + crop_w, y0 + crop_h)).resize(
        (target_size, target_size), Image.Resampling.BILINEAR
    )
    new_boxes: list[tuple[float, float, float, float]] = []
    new_labels: list[int] = []
    for (cx, cy, bw, bh), label in zip(boxes, labels, strict=True):
        ox, oy = cx * width, cy * height
        ow, oh = bw * width, bh * height
        x1, y1 = ox - ow / 2, oy - oh / 2
        x2, y2 = ox + ow / 2, oy + oh / 2
        ix1 = max(x1, x0)
        iy1 = max(y1, y0)
        ix2 = min(x2, x0 + crop_w)
        iy2 = min(y2, y0 + crop_h)
        if ix2 <= ix1 or iy2 <= iy1:
            continue
        orig_area = max(1e-6, (x2 - x1) * (y2 - y1))
        kept_area = (ix2 - ix1) * (iy2 - iy1)
        if kept_area / orig_area < min_box_keep:
            continue
        new_cx = ((ix1 + ix2) / 2 - x0) / crop_w
        new_cy = ((iy1 + iy2) / 2 - y0) / crop_h
        new_bw = (ix2 - ix1) / crop_w
        new_bh = (iy2 - iy1) / crop_h
        new_boxes.append((new_cx, new_cy, new_bw, new_bh))
        new_labels.append(label)
    return cropped, new_boxes, new_labels


def _apply_augmentation(
    image: Image.Image, boxes: list[tuple[float, float, float, float]]
) -> tuple[Image.Image, list[tuple[float, float, float, float]]]:
    """CPU-side augmentation applied on the already-resized target-size image.

    Random horizontal flip + photometric jitter (brightness, contrast). Scale
    jitter is handled separately on the *original* image via
    ``_random_scale_jitter_crop`` — calling it on the post-resize image
    would compound resize artifacts.
    """
    if random.random() < 0.5:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        boxes = [(1.0 - cx, cy, bw, bh) for cx, cy, bw, bh in boxes]
    if random.random() < 0.5:
        image = ImageEnhance.Brightness(image).enhance(random.uniform(0.7, 1.3))
    if random.random() < 0.5:
        image = ImageEnhance.Contrast(image).enhance(random.uniform(0.7, 1.3))
    return image, boxes


def _mosaic_compose(
    samples: list[tuple[Image.Image, list[tuple[float, float, float, float]], list[int]]],
    target_size: int,
) -> tuple[Image.Image, list[tuple[float, float, float, float]], list[int]]:
    """4-image Mosaic: tile 4 samples into a 2x2 canvas at ``target_size``.

    Each input is resized to ``target_size // 2`` and placed in a quadrant.
    Normalized cxcywh boxes are halved and translated to the corresponding
    quadrant. Classic Mosaic gives 4x object density per training step — the
    dominant augmentation for dense small-object detection.
    """
    if len(samples) != 4:
        raise ValueError(f"Mosaic requires exactly 4 samples, got {len(samples)}")
    half = target_size // 2
    canvas = Image.new("RGB", (target_size, target_size), color=(127, 127, 127))
    offsets = [(0, 0), (half, 0), (0, half), (half, half)]
    out_boxes: list[tuple[float, float, float, float]] = []
    out_labels: list[int] = []
    for (image, boxes, labels), (x0, y0) in zip(samples, offsets, strict=True):
        small = image.resize((half, half), Image.Resampling.BILINEAR)
        canvas.paste(small, (x0, y0))
        x0_n = x0 / target_size
        y0_n = y0 / target_size
        for (cx, cy, bw, bh), label in zip(boxes, labels, strict=True):
            out_boxes.append((x0_n + cx * 0.5, y0_n + cy * 0.5, bw * 0.5, bh * 0.5))
            out_labels.append(label)
    return canvas, out_boxes, out_labels


def _copy_paste_objects(
    image: Image.Image,
    boxes: list[tuple[float, float, float, float]],
    labels: list[int],
    source_image: Image.Image,
    source_boxes: list[tuple[float, float, float, float]],
    source_labels: list[int],
    *,
    min_paste: int = 1,
    max_paste: int = 8,
) -> tuple[Image.Image, list[tuple[float, float, float, float]], list[int]]:
    """Paste rectangular crops of source objects into ``image`` at random spots.

    Bbox-level copy (no instance mask) — simplest copy-paste variant that
    still gives most of the AP win for small-object density. Pasted objects
    get added to the target's box list.
    """
    if not source_boxes:
        return image, boxes, labels
    target_w, target_h = image.size
    src_w, src_h = source_image.size
    # Small-object priority: weight each source box by 1 / max(area, eps) and
    # sample without replacement. Tiny boxes dominate the candidate pool so
    # the augmentation densifies the part of the size distribution where
    # the model has the most to gain (VisDrone tail is heavy with sub-32-px
    # targets). Falls back to uniform when all candidates have similar size.
    src_areas = [max(bw * bh, 1e-6) for _, _, bw, bh in source_boxes]
    inv_area = np.array([1.0 / a for a in src_areas], dtype=np.float64)
    weights = inv_area / inv_area.sum()
    n = random.randint(min_paste, min(max_paste, len(source_boxes)))
    chosen = np.random.choice(
        len(source_boxes), size=n, replace=False, p=weights
    ).tolist()
    canvas = image.copy()
    new_boxes = list(boxes)
    new_labels = list(labels)
    for src_idx in chosen:
        cx, cy, bw, bh = source_boxes[src_idx]
        sx1 = max(0, int((cx - bw / 2) * src_w))
        sy1 = max(0, int((cy - bh / 2) * src_h))
        sx2 = min(src_w, int((cx + bw / 2) * src_w))
        sy2 = min(src_h, int((cy + bh / 2) * src_h))
        if sx2 <= sx1 or sy2 <= sy1:
            continue
        crop = source_image.crop((sx1, sy1, sx2, sy2))
        crop_w, crop_h = crop.size
        if crop_w > target_w or crop_h > target_h:
            continue
        px = random.randint(0, target_w - crop_w)
        py = random.randint(0, target_h - crop_h)
        canvas.paste(crop, (px, py))
        new_boxes.append(
            (
                (px + crop_w / 2) / target_w,
                (py + crop_h / 2) / target_h,
                crop_w / target_w,
                crop_h / target_h,
            )
        )
        new_labels.append(source_labels[src_idx])
    return canvas, new_boxes, new_labels


class VisDroneDetectionDataset(Dataset[tuple[torch.Tensor, dict[str, torch.Tensor]]]):
    def __init__(
        self,
        images_dir: str | Path,
        annotations_dir: str | Path,
        image_size: int,
        *,
        augment: bool = False,
        mosaic_prob: float = 0.5,
        copy_paste_prob: float = 0.3,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.annotations_dir = Path(annotations_dir)
        self.image_size = image_size
        self.augment = augment
        # Mosaic + copy-paste only fire when augment=True (val stays clean).
        self.mosaic_prob = mosaic_prob if augment else 0.0
        self.copy_paste_prob = copy_paste_prob if augment else 0.0
        self.image_paths = sorted(
            path
            for path in self.images_dir.iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        )
        if not self.image_paths:
            raise ValueError(f"No images found in {self.images_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def _load_sample(
        self, index: int
    ) -> tuple[Image.Image, list[tuple[float, float, float, float]], list[int]]:
        """Load one (image, boxes, labels) sample with boxes in normalized cxcywh."""
        image_path = self.image_paths[index]
        annotation_path = self.annotations_dir / f"{image_path.stem}.txt"
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        records = parse_visdrone_annotation(annotation_path)
        labels: list[int] = []
        boxes: list[tuple[float, float, float, float]] = []
        for record in records:
            x1, y1, x2, y2 = record.bbox_xyxy
            cx = ((x1 + x2) * 0.5) / width
            cy = ((y1 + y2) * 0.5) / height
            box_w = (x2 - x1) / width
            box_h = (y2 - y1) / height
            labels.append(record.label)
            boxes.append((cx, cy, box_w, box_h))
        return image, boxes, labels

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # 4-image Mosaic path: tiles 4 samples to target_size, skips scale jitter
        # (output already at target_size). Mosaic gives the biggest small-object
        # density win, applied independently of the per-image augmentations.
        # Requires at least 4 unique images in the dataset (test fixtures
        # sometimes have fewer, in which case mosaic is skipped).
        if self.mosaic_prob > 0.0 and len(self) >= 4 and random.random() < self.mosaic_prob:
            others = random.sample(range(len(self)), 3)
            samples = [self._load_sample(i) for i in (index, *others)]
            image, boxes, labels = _mosaic_compose(samples, self.image_size)
            image, boxes = _apply_augmentation(image, boxes)
        else:
            image, boxes, labels = self._load_sample(index)
            if self.augment:
                # Copy-paste objects from a random other image (works on the
                # ORIGINAL-size image, before the scale-jitter crop).
                if self.copy_paste_prob > 0.0 and random.random() < self.copy_paste_prob:
                    other_idx = random.randint(0, len(self) - 1)
                    src_image, src_boxes, src_labels = self._load_sample(other_idx)
                    image, boxes, labels = _copy_paste_objects(
                        image, boxes, labels, src_image, src_boxes, src_labels
                    )
                image, boxes, labels = _random_scale_jitter_crop(
                    image, boxes, labels, self.image_size
                )
                image, boxes = _apply_augmentation(image, boxes)
            else:
                image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        # Return uint8 HWC so the DataLoader worker skips the expensive float
        # conversion + HWC->CHW transpose; train_one_step does both on the GPU.
        image_tensor = torch.from_numpy(np.array(image, dtype=np.uint8))

        target = {
            "labels": torch.as_tensor(labels, dtype=torch.long),
            "boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "image_id": torch.tensor(index, dtype=torch.long),
        }
        return image_tensor, target


class COCODetectionDataset(Dataset[tuple[torch.Tensor, dict[str, torch.Tensor]]]):
    def __init__(
        self,
        images_dir: str | Path,
        annotations_path: str | Path,
        image_size: int,
        *,
        include_segmentation: bool = False,
        include_keypoints: bool = False,
        include_mesh: bool = False,
        include_reid: bool = False,
        include_depth: bool = False,
        include_quality: bool = False,
        num_keypoints: int = 17,
        reid_embedding_dim: int = 128,
        mesh_num_vertices: int = 478,
        mesh_vertex_dim: int = 3,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.annotations_path = Path(annotations_path)
        self.image_size = image_size
        self.include_segmentation = include_segmentation
        self.include_keypoints = include_keypoints
        self.include_mesh = include_mesh
        self.include_reid = include_reid
        self.include_depth = include_depth
        self.include_quality = include_quality
        self.num_keypoints = num_keypoints
        self.reid_embedding_dim = max(1, reid_embedding_dim)
        self.mesh_num_vertices = max(1, mesh_num_vertices)
        self.mesh_vertex_dim = max(1, mesh_vertex_dim)
        payload = json.loads(self.annotations_path.read_text(encoding="utf-8"))
        categories = payload.get("categories", [])
        self.category_ids = tuple(sorted(int(category["id"]) for category in categories))
        self.category_id_to_label = {
            category_id: label for label, category_id in enumerate(self.category_ids)
        }
        annotations_by_image: dict[int, list[dict[str, object]]] = {}
        for annotation in payload.get("annotations", []):
            if int(annotation.get("iscrowd", 0)) != 0:
                continue
            annotations_by_image.setdefault(int(annotation["image_id"]), []).append(annotation)
        self.samples: list[tuple[Path, int, int, int, list[dict[str, object]]]] = []
        self.depth_by_image_id: dict[int, torch.Tensor] = {}
        for image_info in payload.get("images", []):
            image_id = int(image_info["id"])
            raw_depth = image_info.get("depth") or image_info.get("depth_map")
            if raw_depth is None:
                continue
            depth_tensor = self._parse_depth(raw_depth)
            if depth_tensor is not None:
                self.depth_by_image_id[image_id] = depth_tensor
        for image_info in payload.get("images", []):
            image_path = self.images_dir / str(image_info["file_name"])
            if not image_path.exists():
                continue
            image_id = int(image_info["id"])
            self.samples.append(
                (
                    image_path,
                    image_id,
                    int(image_info["width"]),
                    int(image_info["height"]),
                    annotations_by_image.get(image_id, []),
                )
            )
        if not self.samples:
            raise ValueError(f"No COCO images found in {self.images_dir} for {self.annotations_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        image_path, image_id, width, height, annotations = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        resized = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        # Return uint8 HWC so the DataLoader worker skips the expensive float
        # conversion + HWC->CHW transpose; train_one_step does both on the GPU.
        image_tensor = torch.from_numpy(np.array(resized, dtype=np.uint8))

        labels: list[int] = []
        boxes: list[tuple[float, float, float, float]] = []
        keypoints: list[torch.Tensor] = []
        for annotation in annotations:
            category_id = int(annotation["category_id"])
            if category_id not in self.category_id_to_label:
                continue
            x, y, box_w, box_h = [float(value) for value in annotation["bbox"]]
            if box_w <= 0.0 or box_h <= 0.0:
                continue
            labels.append(self.category_id_to_label[category_id])
            boxes.append(
                (
                    (x + box_w * 0.5) / width,
                    (y + box_h * 0.5) / height,
                    box_w / width,
                    box_h / height,
                )
            )
            if self.include_keypoints:
                keypoints.append(self._parse_keypoints(annotation.get("keypoints")))

        target: dict[str, torch.Tensor] = {
            "labels": torch.as_tensor(labels, dtype=torch.long),
            "boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "image_id": torch.tensor(image_id, dtype=torch.long),
            "orig_size": torch.tensor([width, height], dtype=torch.long),
        }
        if self.include_segmentation:
            target["seg_mask"] = self._build_segmentation_mask(width, height, annotations)
        if self.include_keypoints:
            if keypoints:
                target["keypoints"] = torch.stack(keypoints, dim=0)
            else:
                target["keypoints"] = torch.zeros((0, self.num_keypoints, 3), dtype=torch.float32)
        if self.include_reid:
            target["reid_embedding"] = self._build_reid_targets(annotations)
        if self.include_quality:
            target["quality"] = self._build_quality_targets(annotations)
        if self.include_depth and image_id in self.depth_by_image_id:
            target["depth"] = self.depth_by_image_id[image_id]
        if self.include_mesh:
            target["mesh"] = self._build_mesh_target(annotations)
            target["mesh_teachers"] = self._build_mesh_teacher_targets(annotations)
        return image_tensor, target

    def _parse_keypoints(self, value: Any) -> torch.Tensor:
        if not isinstance(value, list) or len(value) == 0:
            return torch.zeros((self.num_keypoints, 3), dtype=torch.float32)
        raw = torch.tensor(value, dtype=torch.float32)
        if raw.numel() % 3 != 0:
            return torch.zeros((self.num_keypoints, 3), dtype=torch.float32)
        raw = raw.reshape(-1, 3)
        output = torch.zeros((self.num_keypoints, 3), dtype=torch.float32)
        keep = min(self.num_keypoints, raw.shape[0])
        if keep > 0:
            output[:keep] = raw[:keep]
        return output

    def _build_segmentation_mask(
        self, width: int, height: int, annotations: list[dict[str, object]]
    ) -> torch.Tensor:
        mask = Image.new("L", (self.image_size, self.image_size), 0)
        drawer = ImageDraw.Draw(mask)
        scale_x = self.image_size / max(1, width)
        scale_y = self.image_size / max(1, height)
        for annotation in annotations:
            category_id = int(annotation["category_id"])
            if category_id not in self.category_id_to_label:
                continue
            class_id = self.category_id_to_label[category_id] + 1
            segmentation = annotation.get("segmentation", [])
            if not isinstance(segmentation, list):
                continue
            for raw_polygon in segmentation:
                if not isinstance(raw_polygon, list) or len(raw_polygon) < 6 or len(raw_polygon) % 2 != 0:
                    continue
                points: list[tuple[float, float]] = []
                for index in range(0, len(raw_polygon), 2):
                    points.append((raw_polygon[index] * scale_x, raw_polygon[index + 1] * scale_y))
                if points:
                    drawer.polygon(points, fill=class_id)
        return torch.from_numpy(np.array(mask, dtype=np.int64))


    def _build_reid_targets(self, annotations: list[dict[str, object]]) -> torch.Tensor:
        embeddings: list[torch.Tensor] = []
        for annotation in annotations:
            raw = (
                annotation.get("reid_embedding")
                or annotation.get("reid")
                or annotation.get("reid_features")
                or annotation.get("track_embedding")
            )
            if raw is None:
                continue
            try:
                vector = torch.as_tensor(raw, dtype=torch.float32).reshape(-1)
            except (TypeError, ValueError):
                continue
            if vector.numel() == 0:
                continue
            embeddings.append(vector)
        if not embeddings:
            return torch.empty((0, self.reid_embedding_dim), dtype=torch.float32)
        max_dim = max(vector.numel() for vector in embeddings)
        max_dim = max(max_dim, self.reid_embedding_dim)
        padded = torch.zeros((len(embeddings), max_dim), dtype=torch.float32)
        for index, vector in enumerate(embeddings):
            padded[index, : vector.numel()] = vector
        return padded

    def _build_quality_targets(self, annotations: list[dict[str, object]]) -> torch.Tensor:
        values: list[float] = []
        for annotation in annotations:
            raw = annotation.get("quality") or annotation.get("score") or annotation.get("track_quality")
            if raw is None:
                continue
            try:
                values.append(float(raw))
            except (TypeError, ValueError):
                continue
        if not values:
            return torch.empty((0,), dtype=torch.float32)
        return torch.tensor(values, dtype=torch.float32)

    def _parse_depth(self, value: Any) -> torch.Tensor | None:
        if not isinstance(value, (list, tuple, torch.Tensor, int, float)):
            return None
        try:
            tensor = torch.as_tensor(value, dtype=torch.float32)
        except (TypeError, ValueError):
            return None
        if tensor.numel() == 0:
            return None
        if tensor.dim() == 0:
            return torch.full((1, 1), float(tensor.item()), dtype=torch.float32)
        if tensor.dim() == 1:
            side = int(math.sqrt(tensor.numel()))
            if side * side == tensor.numel():
                return tensor.reshape(side, side)
            if tensor.numel() == 1:
                return torch.full((1, 1), float(tensor.item()), dtype=torch.float32)
            return None
        if tensor.dim() == 2:
            return tensor
        return None

    def _default_mesh(self, tensor_device: torch.device | None = None) -> torch.Tensor:
        mesh = torch.zeros((0, self.mesh_num_vertices, self.mesh_vertex_dim), dtype=torch.float32)
        if tensor_device is not None:
            mesh = mesh.to(device=tensor_device)
        return mesh

    def _build_mesh_target(self, annotations: list[dict[str, object]]) -> torch.Tensor:
        for annotation in annotations:
            mesh = annotation.get("mesh")
            if mesh is None:
                continue
            mesh_tensor = self._parse_mesh(mesh)
            if mesh_tensor.numel() > 0:
                return mesh_tensor
        return self._default_mesh()

    def _build_mesh_teacher_targets(self, annotations: list[dict[str, object]]) -> list[torch.Tensor]:
        teachers: list[torch.Tensor] = []
        for annotation in annotations:
            raw = annotation.get("mesh_teacher") or annotation.get("mesh_teachers")
            if raw is None:
                continue
            for entry in (raw if isinstance(raw, list) else [raw]):
                mesh_teacher = self._parse_mesh(entry)
                if mesh_teacher.numel() > 0:
                    teachers.append(mesh_teacher)
        return teachers

    def _parse_mesh(self, value: Any) -> torch.Tensor:
        if not isinstance(value, (list, tuple, torch.Tensor)):
            return self._default_mesh()
        raw = torch.as_tensor(value, dtype=torch.float32)
        if raw.numel() == 0:
            return self._default_mesh()
        if raw.dim() == 1:
            if raw.numel() % self.mesh_vertex_dim != 0:
                return self._default_mesh()
            num_points = raw.numel() // self.mesh_vertex_dim
            raw = raw.reshape(-1, self.mesh_vertex_dim)
            if num_points != self.mesh_num_vertices:
                padded = torch.zeros((self.mesh_num_vertices, self.mesh_vertex_dim), dtype=torch.float32)
                keep = min(num_points, self.mesh_num_vertices)
                padded[:keep] = raw[:keep]
                raw = padded
            return raw[: self.mesh_num_vertices].to(dtype=torch.float32)
        if raw.dim() == 2 and raw.shape[1] == self.mesh_vertex_dim:
            num_points = raw.shape[0]
            if num_points >= self.mesh_num_vertices:
                return raw[: self.mesh_num_vertices].to(dtype=torch.float32)
            padded = torch.zeros((self.mesh_num_vertices, self.mesh_vertex_dim), dtype=torch.float32)
            padded[:num_points] = raw.to(dtype=torch.float32)
            return padded
        return self._default_mesh()


def _image_to_float_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image, dtype=np.float32) / 255.0


def collate_detections(
    batch: list[tuple[torch.Tensor, dict[str, torch.Tensor]]],
) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
    images, targets = zip(*batch, strict=True)
    return torch.stack(list(images), dim=0), list(targets)
