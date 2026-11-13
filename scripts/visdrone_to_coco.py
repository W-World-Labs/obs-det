"""Convert VisDrone text annotations into a COCO-format JSON file.

The conversion is one-time data prep so that downstream evaluation can use the
standard pycocotools mAP machinery (the same metric set Ultralytics YOLO
reports), making OmniDet results directly comparable to any external baseline
run on the same ground truth.

Categories follow OMNIDET_CLASSES order with COCO 1-based ids; the
VisDrone-to-OmniDet category mapping in src/omnidet/data/classes.py is reused
so the labels exactly match what the model was trained on.
"""

import argparse
import json
from pathlib import Path

from PIL import Image

from omnidet.data.classes import OMNIDET_CLASSES
from omnidet.data.visdrone import parse_visdrone_annotation


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--images", type=Path, required=True, help="VisDrone images dir")
    parser.add_argument("--annotations", type=Path, required=True, help="VisDrone annotations dir (.txt)")
    parser.add_argument("--out", type=Path, required=True, help="output COCO JSON path")
    args = parser.parse_args()

    categories = [
        {"id": index + 1, "name": name}
        for index, name in enumerate(OMNIDET_CLASSES)
    ]
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    image_paths = sorted(
        path for path in args.images.iterdir() if path.suffix.lower() in image_extensions
    )
    if not image_paths:
        raise SystemExit(f"no images found in {args.images}")

    images: list[dict] = []
    annotations: list[dict] = []
    annotation_id = 0
    missing_annotations = 0

    for image_id, image_path in enumerate(image_paths):
        with Image.open(image_path) as image:
            width, height = image.size
        images.append(
            {"id": image_id, "file_name": image_path.name, "width": width, "height": height}
        )
        annotation_path = args.annotations / f"{image_path.stem}.txt"
        if not annotation_path.exists():
            missing_annotations += 1
            continue
        for record in parse_visdrone_annotation(annotation_path):
            x1, y1, x2, y2 = record.bbox_xyxy
            box_w, box_h = x2 - x1, y2 - y1
            if box_w <= 0 or box_h <= 0:
                continue
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": record.label + 1,
                    "bbox": [x1, y1, box_w, box_h],
                    "area": box_w * box_h,
                    "iscrowd": 0,
                }
            )
            annotation_id += 1

    payload = {"images": images, "annotations": annotations, "categories": categories}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload), encoding="utf-8")
    print(
        f"wrote {args.out}: {len(images)} images, {len(annotations)} annotations, "
        f"{len(categories)} categories ({missing_annotations} images had no annotation file)"
    )


if __name__ == "__main__":
    main()
