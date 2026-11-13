OMNIDET_CLASSES: tuple[str, ...] = (
    "person",
    "pedestrian",
    "car",
    "van",
    "truck",
    "bus",
    "motorcycle",
    "bicycle",
)

_CLASS_TO_INDEX = {name: index for index, name in enumerate(OMNIDET_CLASSES)}

_VISDRONE_TO_OMNIDET: dict[int, str | None] = {
    0: None,
    1: "pedestrian",
    2: "person",
    3: "bicycle",
    4: "car",
    5: "van",
    6: "truck",
    7: "motorcycle",
    8: "motorcycle",
    9: "bus",
    10: "motorcycle",
    11: None,
}


def class_name_to_index(class_name: str) -> int:
    return _CLASS_TO_INDEX[class_name]


def visdrone_category_to_omnidet(category_id: int) -> str | None:
    return _VISDRONE_TO_OMNIDET.get(category_id)
