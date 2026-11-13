import torch
from torch import nn

from omnidet.config import ModelConfig
from omnidet.models.backbone import build_backbone
from omnidet.models.head import MultiTaskHead
from omnidet.models.neck import EdgeMambaFPN


class OmniDet(nn.Module):
    def __init__(self, cfg: ModelConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or ModelConfig()
        self.backbone = build_backbone(self.cfg)
        self.neck = EdgeMambaFPN(self.cfg)
        self.head = MultiTaskHead(self.cfg, self.neck.feature_spec)

    def forward(
        self,
        images: torch.Tensor,
        targets: list[dict[str, torch.Tensor]] | None = None,
    ) -> dict[str, torch.Tensor]:
        # ``targets`` is training-only — passed through to the head so
        # DN-queries can be seeded from GT. At inference targets is None (and
        # the head also skips DN when not in train mode), so the ONNX export
        # contract via OnnxWrapper(model(images)) is unchanged.
        return self.head(self.neck(self.backbone(images)), targets=targets)
