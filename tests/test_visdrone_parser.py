from pathlib import Path

from omnidet.data.visdrone import parse_visdrone_annotation


def test_parse_visdrone_annotation_filters_ignored_classes(tmp_path: Path) -> None:
    ann = tmp_path / "000001.txt"
    ann.write_text(
        "10,20,30,40,1,1,0,0\n"
        "50,60,20,10,1,4,0,0\n"
        "70,80,5,5,1,0,0,0\n",
        encoding="utf-8",
    )

    records = parse_visdrone_annotation(ann)

    assert len(records) == 2
    assert records[0].class_name == "pedestrian"
    assert records[0].bbox_xyxy == (10.0, 20.0, 40.0, 60.0)
    assert records[1].class_name == "car"
    assert records[1].bbox_xyxy == (50.0, 60.0, 70.0, 70.0)


def test_parse_visdrone_annotation_rejects_malformed_lines(tmp_path: Path) -> None:
    ann = tmp_path / "bad.txt"
    ann.write_text("10,20,30\n", encoding="utf-8")

    try:
        parse_visdrone_annotation(ann)
    except ValueError as exc:
        assert "Expected 8 comma-separated fields" in str(exc)
    else:
        raise AssertionError("malformed VisDrone line should raise ValueError")
