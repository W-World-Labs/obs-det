from collections.abc import Mapping
import shutil
from pathlib import Path
from typing import Any

from PIL import Image


_NAME_TO_VISDRONE_CATEGORY = {
    "pedestrian": 1,
    "people": 2,
    "person": 2,
    "bicycle": 3,
    "car": 4,
    "van": 5,
    "truck": 6,
    "tricycle": 7,
    "awning-tricycle": 8,
    "bus": 9,
    "motor": 10,
    "motorcycle": 10,
}


def export_hf_sample(sample: Mapping[str, Any], output_dir: str | Path, index: int) -> None:
    output_path = Path(output_dir)
    images_dir = output_path / "images"
    annotations_dir = output_path / "annotations"
    images_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)

    image = sample["image"]
    if not isinstance(image, Image.Image):
        raise TypeError(f"Expected PIL image, got {type(image)!r}")
    stem = f"{index:07d}"
    image.convert("RGB").save(images_dir / f"{stem}.jpg", quality=95)

    lines = []
    for detection in _iter_detections(sample):
        label = str(detection["label"]).lower()
        category_id = _NAME_TO_VISDRONE_CATEGORY.get(label)
        if category_id is None:
            continue
        x, y, width, height = _box_to_pixels(detection["bounding_box"], image.width, image.height)
        lines.append(f"{x},{y},{width},{height},1,{category_id},0,0")
    (annotations_dir / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def export_fiftyone_sample(
    sample: Mapping[str, Any],
    snapshot_dir: str | Path,
    output_dir: str | Path,
    index: int,
) -> bool:
    source_path = Path(snapshot_dir) / str(sample["filepath"])
    if not source_path.exists():
        return False
    output_path = Path(output_dir)
    images_dir = output_path / "images"
    annotations_dir = output_path / "annotations"
    images_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{index:07d}"
    shutil.copy2(source_path, images_dir / f"{stem}{source_path.suffix.lower()}")

    width = int(sample["metadata"]["width"])
    height = int(sample["metadata"]["height"])
    lines = []
    for detection in _iter_detections(sample):
        label = str(detection["label"]).lower()
        category_id = _NAME_TO_VISDRONE_CATEGORY.get(label)
        if category_id is None:
            continue
        x, y, box_width, box_height = _box_to_pixels(detection["bounding_box"], width, height)
        truncation = int(detection.get("truncation", 0))
        occlusion = int(detection.get("occlusion", 0))
        lines.append(f"{x},{y},{box_width},{box_height},1,{category_id},{truncation},{occlusion}")
    (annotations_dir / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return True


def _iter_detections(sample: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    ground_truth = sample.get("ground_truth", [])
    if isinstance(ground_truth, Mapping):
        detections = ground_truth.get("detections", ground_truth.get("objects", []))
        return list(detections)
    return list(ground_truth)


def _box_to_pixels(box: Any, image_width: int, image_height: int) -> tuple[int, int, int, int]:
    if isinstance(box, Mapping):
        x = float(box.get("x", box.get("xmin", box.get("left", 0.0))))
        y = float(box.get("y", box.get("ymin", box.get("top", 0.0))))
        width = float(box.get("width", box.get("w", 0.0)))
        height = float(box.get("height", box.get("h", 0.0)))
    else:
        x, y, width, height = (float(value) for value in box)
    if max(x, y, width, height) <= 1.0:
        x *= image_width
        width *= image_width
        y *= image_height
        height *= image_height
    return round(x), round(y), round(width), round(height)
