import torch

from omnidet.losses.criterion import DetectionCriterion, MultiTaskCriterion


def test_detection_criterion_returns_finite_losses() -> None:
    criterion = DetectionCriterion(num_classes=8)
    outputs = {
        "pred_logits": torch.randn(2, 10, 9),
        "pred_boxes": torch.rand(2, 10, 4),
    }
    targets = [
        {
            "labels": torch.tensor([1, 2], dtype=torch.long),
            "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.2, 0.2, 0.1, 0.1]]),
        },
        {
            "labels": torch.tensor([3], dtype=torch.long),
            "boxes": torch.tensor([[0.4, 0.4, 0.3, 0.2]]),
        },
    ]

    losses = criterion(outputs, targets)

    assert losses["loss_cls"].isfinite()
    assert losses["loss_bbox"].isfinite()
    assert losses["loss_giou"].isfinite()
    assert losses["loss_total"].isfinite()


def test_detection_criterion_handles_empty_targets() -> None:
    criterion = DetectionCriterion(num_classes=8)
    outputs = {
        "pred_logits": torch.randn(1, 5, 9),
        "pred_boxes": torch.rand(1, 5, 4),
    }
    targets = [{"labels": torch.empty(0, dtype=torch.long), "boxes": torch.empty(0, 4)}]

    losses = criterion(outputs, targets)

    assert losses["loss_total"].isfinite()
    assert losses["loss_bbox"].item() == 0.0


def test_multitask_criterion_returns_all_task_losses() -> None:
    criterion = MultiTaskCriterion(
        num_classes=8,
        active_tasks=("detection", "segmentation", "pose", "mesh", "reid", "depth", "quality"),
        seg_num_classes=2,
        num_keypoints=3,
        mesh_num_vertices=4,
        mesh_vertex_dim=3,
        reid_embedding_dim=4,
        mesh_distill_from_teachers=False,
        task_loss_weights=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    )
    outputs = {
        "pred_logits": torch.randn(1, 10, 9),
        "pred_boxes": torch.rand(1, 10, 4),
        "pred_seg": torch.randn(1, 2, 8, 8),
        "pred_keypoints": torch.rand(1, 10, 3, 3),
        "pred_mesh": torch.randn(1, 4, 3),
        "pred_reid": torch.randn(1, 10, 4),
        "pred_depth": torch.randn(1, 1, 8, 8),
        "pred_quality": torch.sigmoid(torch.randn(1, 10, 1)),
    }
    targets = [
        {
            "labels": torch.tensor([1, 2], dtype=torch.long),
            "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.2, 0.2, 0.1, 0.1]]),
            "seg_mask": torch.randint(0, 2, (8, 8)),
            "keypoints": torch.tensor(
                [
                    [0.5, 0.5, 1.0],
                    [0.3, 0.4, 1.0],
                    [0.4, 0.6, 1.0],
                ]
            ),
            "mesh": torch.randn(4, 3),
            "reid_embedding": torch.tensor(
                [
                    [0.01, 0.02, 0.03, 0.04],
                    [0.05, 0.06, 0.07, 0.08],
                ]
            ),
            "depth": torch.randn(8, 8),
            "quality": torch.tensor([0.5, 0.6]),
        }
    ]

    losses = criterion(outputs, targets)

    assert losses["loss_det_total"].isfinite()
    assert losses["loss_seg_total"].isfinite()
    assert losses["loss_pose_total"].isfinite()
    assert losses["loss_mesh_total"].isfinite()
    assert losses["loss_reid_total"].isfinite()
    assert losses["loss_depth_total"].isfinite()
    assert losses["loss_quality_total"].isfinite()


def test_multitask_criterion_uses_mesh_teacher_ensemble() -> None:
    criterion = MultiTaskCriterion(
        num_classes=8,
        active_tasks=("mesh",),
        task_loss_weights=(1.0, 1.0, 1.0, 1.0),
        mesh_num_vertices=4,
        mesh_vertex_dim=3,
    )
    outputs = {
        "pred_logits": torch.randn(1, 10, 9),
        "pred_boxes": torch.rand(1, 10, 4),
        "pred_mesh": torch.zeros(1, 4, 3),
    }
    teacher_a = torch.ones(4, 3)
    teacher_b = torch.full((4, 3), 2.0)
    targets = [{"labels": torch.empty(0, dtype=torch.long), "boxes": torch.empty((0, 4)), "mesh_teacher": torch.stack([teacher_a, teacher_b])}]

    losses = criterion(outputs, targets)

    assert losses["loss_mesh_total"].isfinite()
