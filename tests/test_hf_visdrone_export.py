from pathlib import Path

from PIL import Image

from omnidet.data.hf_visdrone import export_fiftyone_sample, export_hf_sample


def test_export_hf_sample_writes_image_and_visdrone_annotation(tmp_path: Path) -> None:
    sample = {
        "image": Image.new("RGB", (100, 80), color=(8, 16, 24)),
        "ground_truth": [
            {
                "label": "car",
                "bounding_box": [0.1, 0.2, 0.3, 0.25],
            },
            {
                "label": "ignored-region",
                "bounding_box": [0.0, 0.0, 0.1, 0.1],
            },
        ],
    }

    export_hf_sample(sample, tmp_path, index=7)

    image_path = tmp_path / "images" / "0000007.jpg"
    annotation_path = tmp_path / "annotations" / "0000007.txt"
    assert image_path.exists()
    assert annotation_path.read_text(encoding="utf-8").strip() == "10,16,30,20,1,4,0,0"


def test_export_fiftyone_sample_copies_image_and_annotations(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    data = snapshot / "data"
    data.mkdir(parents=True)
    Image.new("RGB", (100, 80), color=(8, 16, 24)).save(data / "sample.jpg")
    sample = {
        "filepath": "data/sample.jpg",
        "metadata": {"width": 100, "height": 80},
        "ground_truth": {
            "detections": [
                {
                    "label": "pedestrian",
                    "bounding_box": [0.1, 0.2, 0.3, 0.25],
                    "truncation": 1,
                    "occlusion": 2,
                }
            ]
        },
    }

    exported = export_fiftyone_sample(sample, snapshot, tmp_path / "out", index=3)

    assert exported is True
    assert (tmp_path / "out" / "images" / "0000003.jpg").exists()
    assert (
        tmp_path / "out" / "annotations" / "0000003.txt"
    ).read_text(encoding="utf-8").strip() == "10,16,30,20,1,1,1,2"
