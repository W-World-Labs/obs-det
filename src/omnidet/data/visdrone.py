from dataclasses import dataclass
from pathlib import Path

from omnidet.data.classes import class_name_to_index, visdrone_category_to_omnidet


@dataclass(frozen=True)
class VisDroneObject:
    class_name: str
    bbox_xyxy: tuple[float, float, float, float]
    score: float
    truncation: int
    occlusion: int

    @property
    def label(self) -> int:
        return class_name_to_index(self.class_name)


def parse_visdrone_annotation(path: str | Path) -> list[VisDroneObject]:
    objects: list[VisDroneObject] = []
    annotation_path = Path(path)
    for raw_line in annotation_path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        fields = [item.strip() for item in raw_line.split(",")]
        if len(fields) != 8:
            raise ValueError(
                f"Expected 8 comma-separated fields in {annotation_path}: {raw_line}"
            )
        x, y, w, h = (float(fields[index]) for index in range(4))
        score = float(fields[4])
        category_id = int(fields[5])
        class_name = visdrone_category_to_omnidet(category_id)
        if class_name is None:
            continue
        objects.append(
            VisDroneObject(
                class_name=class_name,
                bbox_xyxy=(x, y, x + w, y + h),
                score=score,
                truncation=int(fields[6]),
                occlusion=int(fields[7]),
            )
        )
    return objects
