from pathlib import Path

import torch

from omnidet.config import ModelConfig
from omnidet.models.detector import OmniDet


class OnnxWrapper(torch.nn.Module):
    def __init__(self, model: OmniDet) -> None:
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.model(images)
        return outputs["pred_logits"], outputs["pred_boxes"]


def export_onnx(path: str | Path, cfg: ModelConfig | None = None) -> Path:
    cfg = cfg or ModelConfig()
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model = OnnxWrapper(OmniDet(cfg)).eval()
    sample = torch.randn(1, 3, cfg.input_size, cfg.input_size)
    torch.onnx.export(
        model,
        sample,
        output_path,
        input_names=["images"],
        output_names=["pred_logits", "pred_boxes"],
        opset_version=17,
        dynamic_axes=None,
        dynamo=False,
    )
    return output_path
