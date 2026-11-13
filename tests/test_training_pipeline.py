import json
from pathlib import Path

import torch
from PIL import Image

from omnidet.config import ModelConfig
from omnidet.data.dataset import COCODetectionDataset, VisDroneDetectionDataset, collate_detections
from omnidet.evaluation.coco_eval import collect_coco_predictions
from omnidet.losses.criterion import DetectionCriterion, MultiTaskCriterion
from omnidet.models.detector import OmniDet
from omnidet.training.engine import eval_one_epoch, train_one_step


def _write_sample(root: Path) -> tuple[Path, Path]:
    images = root / "images"
    annotations = root / "annotations"
    images.mkdir()
    annotations.mkdir()
    Image.new("RGB", (80, 40), color=(32, 48, 64)).save(images / "000001.jpg")
    annotations.joinpath("000001.txt").write_text(
        "10,5,20,10,1,4,0,0\n",
        encoding="utf-8",
    )
    return images, annotations


def _write_coco_sample(root: Path) -> tuple[Path, Path]:
    images = root / "images"
    images.mkdir()
    Image.new("RGB", (100, 50), color=(48, 64, 80)).save(images / "000001.jpg")
    annotations = root / "instances.json"
    annotations.write_text(
        json.dumps(
            {
                "images": [
                    {"id": 1, "file_name": "000001.jpg", "width": 100, "height": 50, "depth_map": [[0.5] * 16 for _ in range(16)]}
                ],
                "annotations": [
                    {
                        "id": 10,
                        "image_id": 1,
                        "category_id": 3,
                        "bbox": [20, 10, 30, 15],
                        "area": 450,
                        "iscrowd": 0,
                    }
                ],
                "categories": [{"id": 1, "name": "person"}, {"id": 3, "name": "car"}],
            }
        ),
        encoding="utf-8",
    )
    return images, annotations


def _write_coco_multitask_sample(root: Path) -> tuple[Path, Path]:
    images = root / "images"
    images.mkdir()
    Image.new("RGB", (100, 50), color=(48, 64, 80)).save(images / "000001.jpg")
    annotations = root / "instances.json"
    keypoints = []
    for index in range(17):
        keypoints.extend([20.0 + index * 0.2, 10.0 + index * 0.1, 1.0])
    mesh = [0.001 * index for index in range(478 * 3)]
    reid_embedding = [0.01 * index for index in range(8)]
    annotations.write_text(
        json.dumps(
            {
                "images": [
                    {
                        "id": 1,
                        "file_name": "000001.jpg",
                        "width": 100,
                        "height": 50,
                        "depth_map": [[0.5] * 16 for _ in range(16)],
                    }
                ],
                "annotations": [
                    {
                        "id": 10,
                        "image_id": 1,
                        "category_id": 3,
                        "bbox": [20, 10, 30, 15],
                        "area": 450,
                        "iscrowd": 0,
                        "segmentation": [[20, 10, 50, 10, 50, 25, 20, 25]],
                        "keypoints": keypoints,
                        "mesh": mesh,
                        "mesh_teacher": [mesh, [value * 0.8 for value in mesh]],
                        "reid_embedding": reid_embedding,
                        "quality": 0.9,
                    }
                ],
                "categories": [{"id": 1, "name": "person"}, {"id": 3, "name": "car"}],
            }
        ),
        encoding="utf-8",
    )
    return images, annotations


def test_visdrone_dataset_returns_tensor_and_normalized_target(tmp_path: Path) -> None:
    images, annotations = _write_sample(tmp_path)
    dataset = VisDroneDetectionDataset(images, annotations, image_size=64)

    image, target = dataset[0]

    # uint8 HWC contract: float conversion + CHW transpose happen on the GPU
    # in images_to_device, not in the DataLoader worker.
    assert image.shape == (64, 64, 3)
    assert image.dtype == torch.uint8
    assert target["labels"].tolist() == [2]
    assert torch.allclose(target["boxes"], torch.tensor([[0.25, 0.25, 0.25, 0.25]]))


def test_visdrone_dataset_augment_preserves_shape_and_box_validity(tmp_path: Path) -> None:
    import random as _random
    images, annotations = _write_sample(tmp_path)
    dataset = VisDroneDetectionDataset(images, annotations, image_size=64, augment=True)

    # Augmentation combines scale-jitter/crop, hflip, and photometric — verify
    # the contract holds across many seeds: shape stays uint8 HWC, surviving
    # boxes are well-formed normalised cxcywh, and across seeds the box can be
    # either dropped (cropped out) or kept (with a transformed cx position).
    saw_surviving_box = False
    for seed in range(32):
        _random.seed(seed)
        image, target = dataset[0]
        assert image.shape == (64, 64, 3)
        assert image.dtype == torch.uint8
        # Labels MUST stay in sync with boxes — augmentation that drops boxes
        # without dropping the corresponding labels caused a matcher crash in
        # production training (size mismatch in cost matrix).
        assert target["labels"].shape[0] == target["boxes"].shape[0]
        for cx, cy, bw, bh in target["boxes"].tolist():
            assert 0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0
            assert 0.0 < bw <= 1.0 and 0.0 < bh <= 1.0
            saw_surviving_box = True
    # 32 seeds with crop scale 0.6-1.0: at least some boxes must survive.
    assert saw_surviving_box


def test_collate_detections_stacks_images(tmp_path: Path) -> None:
    images, annotations = _write_sample(tmp_path)
    dataset = VisDroneDetectionDataset(images, annotations, image_size=64)

    batch_images, targets = collate_detections([dataset[0], dataset[0]])

    assert batch_images.shape == (2, 64, 64, 3)
    assert batch_images.dtype == torch.uint8
    assert len(targets) == 2


def test_coco_dataset_returns_tensor_and_contiguous_target(tmp_path: Path) -> None:
    images, annotations = _write_coco_sample(tmp_path)
    dataset = COCODetectionDataset(images, annotations, image_size=64)

    image, target = dataset[0]

    assert image.shape == (64, 64, 3)
    assert image.dtype == torch.uint8
    assert target["labels"].tolist() == [1]
    assert torch.allclose(target["boxes"], torch.tensor([[0.35, 0.35, 0.3, 0.3]]))
    assert target["orig_size"].tolist() == [100, 50]


def test_coco_dataset_returns_mesh_fields_when_enabled(tmp_path: Path) -> None:
    images, annotations = _write_coco_multitask_sample(tmp_path)
    dataset = COCODetectionDataset(
        images,
        annotations,
        image_size=64,
        include_segmentation=True,
        include_keypoints=True,
        include_mesh=True,
        include_reid=True,
        include_depth=True,
        include_quality=True,
        num_keypoints=17,
        reid_embedding_dim=8,
        mesh_num_vertices=478,
        mesh_vertex_dim=3,
    )

    image, target = dataset[0]

    assert image.shape == (64, 64, 3)
    assert image.dtype == torch.uint8
    assert target["seg_mask"].shape == (64, 64)
    assert target["keypoints"].shape == (1, 17, 3)
    assert target["mesh"].shape == (478, 3)
    assert target["reid_embedding"].shape[0] == 1
    assert target["quality"].shape == (1,)
    assert target["depth"].shape == (16, 16)


def test_train_one_step_returns_finite_loss(tmp_path: Path) -> None:
    images, annotations = _write_sample(tmp_path)
    dataset = VisDroneDetectionDataset(images, annotations, image_size=64)
    batch_images, targets = collate_detections([dataset[0]])
    cfg = ModelConfig(input_size=64, num_queries=12, neck_channels=32)
    model = OmniDet(cfg)
    criterion = DetectionCriterion(num_classes=cfg.num_classes)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    losses = train_one_step(model, criterion, optimizer, batch_images, targets, device="cpu")

    assert losses["loss_total"] > 0.0


def test_train_one_step_accepts_coco_num_classes(tmp_path: Path) -> None:
    images, annotations = _write_coco_sample(tmp_path)
    dataset = COCODetectionDataset(images, annotations, image_size=64)
    batch_images, targets = collate_detections([dataset[0]])
    cfg = ModelConfig(input_size=64, num_queries=12, neck_channels=32, num_classes=80)
    model = OmniDet(cfg)
    criterion = DetectionCriterion(num_classes=cfg.num_classes)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    losses = train_one_step(model, criterion, optimizer, batch_images, targets, device="cpu")

    assert losses["loss_total"] > 0.0


def test_multi_task_train_one_step_supports_joint_outputs(tmp_path: Path) -> None:
    images, annotations = _write_coco_multitask_sample(tmp_path)
    dataset = COCODetectionDataset(
        images,
        annotations,
        image_size=64,
        include_segmentation=True,
        include_keypoints=True,
        include_mesh=True,
        include_reid=True,
        include_depth=True,
        include_quality=True,
        num_keypoints=17,
        reid_embedding_dim=8,
        mesh_num_vertices=478,
        mesh_vertex_dim=3,
    )
    batch_images, targets = collate_detections([dataset[0]])
    cfg = ModelConfig(
        input_size=64,
        num_queries=12,
        neck_channels=32,
        num_classes=80,
        active_tasks=("detection", "segmentation", "pose", "mesh", "reid", "depth", "quality"),
        task_loss_weights=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        num_keypoints=17,
        mesh_enabled=True,
        reid_embedding_dim=8,
        depth_head_channels=16,
        quality_head_channels=16,
    )
    model = OmniDet(cfg)
    criterion = MultiTaskCriterion(
        num_classes=cfg.num_classes,
        active_tasks=("detection", "segmentation", "pose", "mesh", "reid", "depth", "quality"),
        seg_num_classes=None,
        num_keypoints=17,
        reid_embedding_dim=8,
        mesh_num_vertices=478,
        mesh_vertex_dim=3,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    losses = train_one_step(model, criterion, optimizer, batch_images, targets, device="cpu")

    assert losses["loss_total"] > 0.0
    assert "loss_seg_total" in losses
    assert "loss_pose_total" in losses
    assert "loss_mesh_total" in losses
    assert "loss_reid_total" in losses
    assert "loss_depth_total" in losses
    assert "loss_quality_total" in losses


def test_eval_one_epoch_returns_averaged_losses(tmp_path: Path) -> None:
    from torch.utils.data import DataLoader
    images, annotations = _write_coco_sample(tmp_path)
    dataset = COCODetectionDataset(images, annotations, image_size=64)
    loader = DataLoader(dataset, batch_size=1, collate_fn=collate_detections)
    cfg = ModelConfig(input_size=64, num_queries=12, neck_channels=32, num_classes=2)
    model = OmniDet(cfg)
    criterion = DetectionCriterion(num_classes=cfg.num_classes)

    val_losses = eval_one_epoch(model, criterion, loader, device="cpu")

    assert "loss_total" in val_losses
    assert val_losses["loss_total"] > 0.0


def test_collect_coco_predictions_returns_list(tmp_path: Path) -> None:
    from torch.utils.data import DataLoader
    images, annotations = _write_coco_sample(tmp_path)
    dataset = COCODetectionDataset(images, annotations, image_size=64)
    loader = DataLoader(dataset, batch_size=1, collate_fn=collate_detections)
    cfg = ModelConfig(input_size=64, num_queries=12, neck_channels=32, num_classes=2)
    model = OmniDet(cfg)

    preds = collect_coco_predictions(model, loader, device="cpu", score_thresh=0.0)

    assert isinstance(preds, list)
    for pred in preds:
        assert {"image_id", "category_id", "bbox", "score"} <= pred.keys()
        assert len(pred["bbox"]) == 4
