from pathlib import Path

import onnx

from omnidet.config import ModelConfig
from omnidet.export.onnx import export_onnx


def test_export_onnx_writes_valid_model(tmp_path: Path) -> None:
    path = tmp_path / "omnidet.onnx"

    export_onnx(path, ModelConfig(input_size=64, accuracy_input_size=96, num_queries=12, neck_channels=32))

    model = onnx.load(path)
    onnx.checker.check_model(model)
    assert path.stat().st_size > 0
