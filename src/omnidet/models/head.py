import math

import torch
from torch import nn
import torch.nn.functional as F

from omnidet.config import ModelConfig
from omnidet.models.features import FeatureSpec


def _inv_sigmoid(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Inverse-sigmoid (logit), with a small clamp to avoid +/-inf at the edges."""
    x = x.clamp(min=eps, max=1.0 - eps)
    return torch.log(x / (1.0 - x))


def _build_dn_queries(
    targets: list[dict[str, torch.Tensor]],
    num_classes: int,
    *,
    num_dn_groups: int = 3,
    max_gt_per_image: int = 50,
    box_noise_scale: float = 0.4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int] | None:
    """Build DN-query inputs and supervision targets from a batch of GT.

    Each image's GT (up to ``max_gt_per_image``) is replicated into
    ``num_dn_groups`` noised copies, used as additional decoder queries during
    training. The model learns to *denoise* them back to the clean GT, which
    provides strong direct supervision and dramatically speeds DETR-family
    convergence (DN-DETR / DINO, 2022). Returns ``None`` if no image in the
    batch has any GT.
    """
    reference = next((t["boxes"] for t in targets if t["boxes"].numel() > 0), None)
    if reference is None:
        return None
    device = reference.device

    batch_size = len(targets)
    per_group = max_gt_per_image
    num_dn_total = per_group * num_dn_groups

    dn_ref = torch.zeros(batch_size, num_dn_total, 4, device=device)
    target_boxes = torch.zeros(batch_size, num_dn_total, 4, device=device)
    target_labels = torch.full(
        (batch_size, num_dn_total), num_classes, dtype=torch.long, device=device
    )
    valid_mask = torch.zeros(batch_size, num_dn_total, dtype=torch.bool, device=device)

    for b, target in enumerate(targets):
        gt_boxes = target["boxes"]
        gt_labels = target["labels"]
        n = min(gt_boxes.shape[0], per_group)
        if n == 0:
            continue
        gt_boxes = gt_boxes[:n].float().to(device)
        gt_labels = gt_labels[:n].to(device)
        for group in range(num_dn_groups):
            start = group * per_group
            # Noise: per-component, magnitude proportional to the box scale.
            box_diff = (torch.rand_like(gt_boxes) * 2 - 1) * box_noise_scale
            box_diff[:, :2] *= gt_boxes[:, 2:]  # cx, cy proportional to w, h
            box_diff[:, 2:] *= gt_boxes[:, 2:]  # w, h proportional to themselves
            noisy = (gt_boxes + box_diff).clamp(min=1e-3, max=1.0 - 1e-3)
            dn_ref[b, start : start + n] = torch.log(noisy / (1.0 - noisy))
            target_boxes[b, start : start + n] = gt_boxes
            target_labels[b, start : start + n] = gt_labels
            valid_mask[b, start : start + n] = True
    return dn_ref, target_boxes, target_labels, valid_mask, num_dn_total


def _build_dn_self_attn_mask(num_dn: int, num_regular: int, device: torch.device) -> torch.Tensor:
    """Self-attention mask for the [DN; regular] query layout.

    DN queries cannot see each other (no GT leak between independent DN slots)
    and cannot see the regular learnable queries (no GT leak to the inference
    path). Regular queries cannot see DN queries either. Returns a (N, N) bool
    where True = mask out.

    Implemented with fully vectorised broadcast (no in-place slice writes /
    ``fill_diagonal_``) so ``torch.compile`` traces it cleanly.
    """
    total = num_dn + num_regular
    if num_dn == 0:
        return torch.zeros(total, total, dtype=torch.bool, device=device)
    idx = torch.arange(total, device=device)
    is_dn = idx < num_dn  # (total,)
    i_dn = is_dn.unsqueeze(1)  # (total, 1)
    j_dn = is_dn.unsqueeze(0)  # (1, total)
    same_index = idx.unsqueeze(1) == idx.unsqueeze(0)
    # Mask when exactly one of (i, j) is DN, OR both are DN but different.
    return (i_dn ^ j_dn) | (i_dn & j_dn & ~same_index)


class _PositionEmbedding2DLearnable(nn.Module):
    """2D learnable positional encoding (row + column embeddings).

    Cheaper to export to TensorRT than the sinusoidal variant — it's just a
    parameter slice and a concat. ``max_size`` is sized to comfortably cover the
    largest decoder memory level (P3 at the design-doc 1280 accuracy input is
    160 tokens per side; we keep 256 as headroom).
    """

    def __init__(self, dim: int, max_size: int = 256) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"dim must be even, got {dim}")
        half = dim // 2
        self.row_embed = nn.Parameter(torch.randn(max_size, half) * 0.02)
        self.col_embed = nn.Parameter(torch.randn(max_size, half) * 0.02)

    def forward(self, h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        row = self.row_embed[:h].to(device=device, dtype=dtype)
        col = self.col_embed[:w].to(device=device, dtype=dtype)
        # (h, w, dim) via outer broadcast, then flatten spatial dims
        pos = torch.cat(
            [row.unsqueeze(1).expand(h, w, -1), col.unsqueeze(0).expand(h, w, -1)],
            dim=-1,
        )
        return pos.reshape(h * w, -1)


class _DecoderLayer(nn.Module):
    """Pre-norm transformer decoder layer with plain MHA (TensorRT-safe)."""

    def __init__(self, dim: int, num_heads: int, dim_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim_ff),
            nn.GELU(),
            nn.Linear(dim_ff, dim),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        content: torch.Tensor,
        memory: torch.Tensor,
        query_pos: torch.Tensor | None = None,
        memory_pos: torch.Tensor | None = None,
        self_attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Pre-norm self-attention; positional embedding injected into Q and K only.
        # ``self_attn_mask`` is used during DN-query training so DN queries
        # cannot see each other or the regular learnable queries.
        normed = self.norm1(content)
        q_k = normed + query_pos if query_pos is not None else normed
        attn_out, _ = self.self_attn(
            q_k, q_k, normed, need_weights=False, attn_mask=self_attn_mask
        )
        content = content + self.dropout(attn_out)

        # Pre-norm cross-attention; Q gets query_pos, K gets memory_pos, V unchanged
        normed = self.norm2(content)
        q = normed + query_pos if query_pos is not None else normed
        k = memory + memory_pos if memory_pos is not None else memory
        cross_out, _ = self.cross_attn(q, k, memory, need_weights=False)
        content = content + self.dropout(cross_out)

        # Pre-norm FFN
        content = content + self.dropout(self.ffn(self.norm3(content)))
        return content


class DetectionHead(nn.Module):
    """DETR-style detection head with a small transformer decoder.

    Design choices, drawn from 2024-2025 real-time DETR family papers
    (RT-DETR, DAB-DETR, DINO, DEIM) but implemented clean-room and kept
    TensorRT-friendly per the OmniDet v0 deployment contract:

    * Multi-scale memory from P3 + P4 + P5. P2 is dropped from the decoder
      memory to keep the token count manageable at 960 input — the
      high-resolution small-object signal still flows into the decoder via the
      neck's edge-guided fusion back into P3.
    * Learnable **content queries** plus learnable **reference points**. The
      per-layer query positional embedding is derived from the running refined
      box (DAB-DETR / Conditional DETR style), so query positions track the
      object location each layer claims.
    * N-layer transformer decoder, plain multi-head self+cross attention.
      **No deformable attention** — the deployment contract forbids custom
      plugins and scatter-heavy ops; plain MHA exports cleanly to TensorRT.
    * **Iterative box refinement** (Deformable DETR): each layer outputs a
      delta added to the running reference in inv-sigmoid space. The final
      layer's box_embed is zero-initialised so the first forward pass returns
      the unchanged reference, easing optimisation start.
    * Pre-norm layers for training stability under bf16/AMP.

    Multi-task token reuse: pose, reid, and quality heads consume the **final
    decoder query tokens** (real per-object semantic queries), which is what
    the "amortize the shared trunk like a KV cache" argument actually needs to
    hold — much stronger than the previous fixed-spatial-pool tokens.
    """

    def __init__(
        self,
        cfg: ModelConfig,
        feature_spec: FeatureSpec,
        *,
        num_layers: int = 6,
        num_heads: int = 8,
        dim_ff_mult: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_queries = cfg.num_queries
        self.num_classes = cfg.num_classes
        self.feature_spec = feature_spec
        self.dim = feature_spec.channels
        if self.dim % num_heads != 0:
            raise ValueError(
                f"neck_channels ({self.dim}) must be divisible by num_heads ({num_heads})"
            )

        # Decoder memory. P2 is opt-in (cfg.use_p2_memory) — it pays for itself
        # only on small-object-heavy datasets because at 960² it brings 57.6k
        # extra tokens. Cross-attention with N=300 queries stays affordable
        # (O(N · n) ≈ 23M ops vs 5M for P3-P5 alone). Encoder self-attn over
        # P2 is what's forbidden — we don't run one, so the cost is bounded.
        if getattr(cfg, "use_p2_memory", False):
            self.memory_levels = feature_spec.levels
        else:
            self.memory_levels = tuple(level for level in feature_spec.levels if level != "p2")
            if not self.memory_levels:
                self.memory_levels = feature_spec.levels

        self.level_embed = nn.Parameter(torch.zeros(len(self.memory_levels), self.dim))
        nn.init.normal_(self.level_embed, std=0.02)
        # P2 at the accuracy input (1280) is 320 tokens/side; size up the
        # learnable pos-embed accordingly when P2 is enabled.
        max_pos = max(256, cfg.accuracy_input_size // 4 if getattr(cfg, "use_p2_memory", False) else 0)
        self.pos_embed = _PositionEmbedding2DLearnable(self.dim, max_size=max_pos)

        # Learnable queries + reference boxes (stored in inv-sigmoid space).
        self.content_queries = nn.Parameter(torch.randn(self.num_queries, self.dim) * 0.02)
        ref_init = torch.zeros(self.num_queries, 4)
        # cx, cy: roughly uniform across the image (sigmoid([-1,1]) ≈ [0.27, 0.73]).
        nn.init.uniform_(ref_init[:, :2], -1.0, 1.0)
        # w, h: log-uniform across multiple scales so the query bank covers
        # tiny UAV-target sizes (<2% of image) through medium sizes (~30%).
        # The previous single-scale init at 0.1 prevented the model from ever
        # predicting boxes smaller than ~0.07, which is fatal on VisDrone-like
        # data where most objects are ~1-3% wide.
        scales = torch.empty(self.num_queries, 2).uniform_(math.log(0.02), math.log(0.3)).exp()
        ref_init[:, 2:] = torch.log(scales / (1.0 - scales))
        self.ref_points = nn.Parameter(ref_init)

        self.ref_to_query_pos = nn.Sequential(
            nn.Linear(4, self.dim),
            nn.GELU(),
            nn.Linear(self.dim, self.dim),
        )

        dim_ff = self.dim * dim_ff_mult
        self.layers = nn.ModuleList(
            [
                _DecoderLayer(self.dim, num_heads=num_heads, dim_ff=dim_ff, dropout=dropout)
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(self.dim)

        self.class_embed = nn.Linear(self.dim, cfg.num_classes + 1)
        self.box_embed = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.GELU(),
            nn.Linear(self.dim, self.dim),
            nn.GELU(),
            nn.Linear(self.dim, 4),
        )
        # Zero-init the final box layer so the first forward returns the
        # reference unchanged — standard DETR-family init.
        nn.init.zeros_(self.box_embed[-1].weight)
        nn.init.zeros_(self.box_embed[-1].bias)

        # DN-query content embedding — single learnable vector broadcast across
        # all DN slots. They differ only by their (noised-GT-seeded) reference
        # points, so the decoder discriminates them via position; sharing
        # content keeps parameter count low (DN is training-only).
        self.dn_content = nn.Parameter(torch.randn(self.dim) * 0.02)
        # Per-instance DN config — kept here so a future hyper-tuner can tweak
        # without touching the function body. Disable DN by setting num_dn_groups=0.
        self.num_dn_groups = 3
        self.max_dn_per_image = 50
        self.dn_box_noise_scale = 0.4

    def _flatten_memory(
        self, features: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokens: list[torch.Tensor] = []
        positions: list[torch.Tensor] = []
        for level_idx, level in enumerate(self.memory_levels):
            x = features[level]  # (B, C, H, W)
            batch, _, h, w = x.shape
            tokens.append(x.flatten(2).transpose(1, 2))  # (B, H*W, C)
            pos = self.pos_embed(h, w, x.device, x.dtype) + self.level_embed[level_idx].to(x.dtype)
            positions.append(pos.unsqueeze(0).expand(batch, -1, -1))
        return torch.cat(tokens, dim=1), torch.cat(positions, dim=1)

    def decode(
        self,
        features: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor] | None]:
        """Run the decoder. Returns (final_query_tokens, pred_logits, pred_boxes, dn_outputs).

        If ``targets`` is given AND the head is in train mode AND
        ``num_dn_groups > 0``, DN queries are prepended to the regular queries
        for this forward pass: they're seeded with noised GT and the model is
        supervised to denoise them back. At inference (model.eval() or
        targets=None), the DN path is a no-op and ``dn_outputs`` is ``None``,
        so the export contract is unchanged.
        """
        feat_any = next(iter(features.values()))
        batch = feat_any.shape[0]
        dtype = feat_any.dtype
        device = feat_any.device

        memory, memory_pos = self._flatten_memory(features)
        content = self.content_queries.to(dtype).unsqueeze(0).expand(batch, -1, -1).contiguous()
        ref = self.ref_points.to(dtype).unsqueeze(0).expand(batch, -1, -1).contiguous()
        num_regular = content.shape[1]

        # Optional DN-query injection (training-only).
        dn_meta = None
        num_dn = 0
        self_attn_mask: torch.Tensor | None = None
        if self.training and targets is not None and self.num_dn_groups > 0:
            dn = _build_dn_queries(
                targets,
                self.num_classes,
                num_dn_groups=self.num_dn_groups,
                max_gt_per_image=self.max_dn_per_image,
                box_noise_scale=self.dn_box_noise_scale,
            )
            if dn is not None:
                dn_ref, dn_target_boxes, dn_target_labels, dn_valid, num_dn = dn
                dn_content = self.dn_content.to(dtype).view(1, 1, -1).expand(batch, num_dn, -1).contiguous()
                content = torch.cat([dn_content, content], dim=1)
                ref = torch.cat([dn_ref.to(dtype), ref], dim=1)
                self_attn_mask = _build_dn_self_attn_mask(num_dn, num_regular, device)
                dn_meta = (dn_target_boxes, dn_target_labels, dn_valid)

        for layer in self.layers:
            ref_box = torch.sigmoid(ref)
            query_pos = self.ref_to_query_pos(ref_box)
            content = layer(
                content, memory, query_pos=query_pos, memory_pos=memory_pos,
                self_attn_mask=self_attn_mask,
            )
            ref = ref + self.box_embed(self.final_norm(content))

        normed = self.final_norm(content)
        pred_logits = self.class_embed(normed)
        pred_boxes = torch.sigmoid(ref)

        dn_outputs: dict[str, torch.Tensor] | None = None
        if num_dn > 0:
            dn_outputs = {
                "pred_logits_dn": pred_logits[:, :num_dn],
                "pred_boxes_dn": pred_boxes[:, :num_dn],
                "dn_target_boxes": dn_meta[0],
                "dn_target_labels": dn_meta[1],
                "dn_valid_mask": dn_meta[2],
            }
            normed = normed[:, num_dn:]
            pred_logits = pred_logits[:, num_dn:]
            pred_boxes = pred_boxes[:, num_dn:]
        return normed, pred_logits, pred_boxes, dn_outputs

    def forward(
        self,
        features: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]] | None = None,
    ) -> dict[str, torch.Tensor]:
        _, pred_logits, pred_boxes, dn_outputs = self.decode(features, targets=targets)
        out = {"pred_logits": pred_logits, "pred_boxes": pred_boxes}
        if dn_outputs is not None:
            out.update(dn_outputs)
        return out

    def _query_tokens(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        """Backward-compat helper for callers that only want the final query tokens."""
        tokens, _, _, _ = self.decode(features)
        return tokens


class SegmentationHead(nn.Module):
    def __init__(self, cfg: ModelConfig, feature_spec: FeatureSpec) -> None:
        super().__init__()
        self.num_classes = cfg.seg_num_classes if cfg.seg_num_classes is not None else (cfg.num_classes + 1)
        self.hidden = max(16, cfg.seg_head_channels)
        self.levels = feature_spec.levels
        self.level_convs = nn.ModuleDict(
            {
                level: nn.Conv2d(feature_spec.channels, self.hidden, kernel_size=3, padding=1)
                for level in self.levels
            }
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(self.hidden * len(self.levels), self.hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden, self.num_classes, kernel_size=1),
        )
        self.output_size = cfg.input_size

    def forward(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        upsampled: list[torch.Tensor] = []
        for level in self.levels:
            projected = self.level_convs[level](features[level])
            upsampled.append(
                F.interpolate(
                    projected,
                    size=(self.output_size, self.output_size),
                    mode="bilinear",
                    align_corners=False,
                )
            )
        return self.fuse(torch.cat(upsampled, dim=1))


class PoseHead(nn.Module):
    def __init__(self, cfg: ModelConfig, feature_spec: FeatureSpec) -> None:
        super().__init__()
        self.num_queries = cfg.num_queries
        self.num_keypoints = max(1, cfg.num_keypoints)
        channels = feature_spec.channels
        self.coord_embed = nn.Sequential(
            nn.Linear(channels, channels),
            nn.SiLU(inplace=True),
            nn.Linear(channels, self.num_keypoints * 3),
        )

    def forward(self, query_tokens: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.coord_embed(query_tokens)).view(
            query_tokens.shape[0], query_tokens.shape[1], self.num_keypoints, 3
        )


class MeshHead(nn.Module):
    def __init__(self, cfg: ModelConfig, feature_spec: FeatureSpec) -> None:
        super().__init__()
        self.num_vertices = max(1, cfg.mesh_num_vertices)
        self.vertex_dim = max(1, cfg.mesh_vertex_dim)
        self.mesh_output_dim = max(1, self.num_vertices * self.vertex_dim)
        if cfg.mesh_output_dim > 0:
            self.mesh_output_dim = cfg.mesh_output_dim
            self.num_vertices = max(1, int(math.ceil(self.mesh_output_dim / self.vertex_dim)))
        self.levels = feature_spec.levels
        channels = feature_spec.channels
        self.pool = nn.ModuleDict(
            {level: nn.AdaptiveAvgPool2d((1, 1)) for level in self.levels}
        )
        self.proj = nn.ModuleDict(
            {level: nn.Linear(channels, channels) for level in self.levels}
        )
        self.body = nn.Sequential(
            nn.Linear(channels * len(self.levels), channels),
            nn.SiLU(inplace=True),
            nn.Linear(channels, self.mesh_output_dim),
        )

    def forward(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        fused = []
        for level in self.levels:
            pooled = self.pool[level](features[level]).flatten(1)
            fused.append(self.proj[level](pooled))
        mesh = self.body(torch.cat(fused, dim=1))
        target_elements = self.num_vertices * self.vertex_dim
        if mesh.shape[1] < target_elements:
            mesh = F.pad(mesh, (0, target_elements - mesh.shape[1]))
        elif mesh.shape[1] > target_elements:
            mesh = mesh[:, :target_elements]
        return mesh.view(-1, self.num_vertices, self.vertex_dim)


class ReIDHead(nn.Module):
    def __init__(self, cfg: ModelConfig, feature_spec: FeatureSpec) -> None:
        super().__init__()
        self.reid_dim = max(1, cfg.reid_embedding_dim)
        channels = feature_spec.channels
        self.embed = nn.Sequential(
            nn.Linear(channels, channels),
            nn.SiLU(inplace=True),
            nn.Linear(channels, self.reid_dim),
        )

    def forward(self, query_tokens: torch.Tensor) -> torch.Tensor:
        return self.embed(query_tokens)


class DepthHead(nn.Module):
    def __init__(self, cfg: ModelConfig, feature_spec: FeatureSpec) -> None:
        super().__init__()
        hidden = max(16, cfg.depth_head_channels)
        self.source_level = feature_spec.finest_level
        self.net = nn.Sequential(
            nn.Conv2d(feature_spec.channels, hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )
        self.output_size = cfg.input_size

    def forward(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        pred = self.net(features[self.source_level])
        if pred.shape[-1] != self.output_size or pred.shape[-2] != self.output_size:
            pred = F.interpolate(pred, size=(self.output_size, self.output_size), mode="bilinear", align_corners=False)
        return pred


class QualityHead(nn.Module):
    def __init__(self, cfg: ModelConfig, feature_spec: FeatureSpec) -> None:
        super().__init__()
        hidden = max(16, cfg.quality_head_channels)
        channels = feature_spec.channels
        self.score = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, query_tokens: torch.Tensor) -> torch.Tensor:
        return self.score(query_tokens)


class MultiTaskHead(nn.Module):
    def __init__(self, cfg: ModelConfig, feature_spec: FeatureSpec) -> None:
        super().__init__()
        self.feature_spec = feature_spec
        self.active_tasks = tuple(cfg.active_tasks)
        self.use_detection = "detection" in self.active_tasks
        self.use_segmentation = "segmentation" in self.active_tasks
        self.use_pose = "pose" in self.active_tasks
        self.use_mesh = "mesh" in self.active_tasks and cfg.mesh_enabled
        self.use_reid = "reid" in self.active_tasks
        self.use_depth = "depth" in self.active_tasks
        self.use_quality = "quality" in self.active_tasks

        # The detection head doubles as the source of the shared query tokens
        # for any task that wants to consume them (pose/reid/quality).
        self.detection_head = DetectionHead(cfg, feature_spec) if (
            self.use_detection or self.use_pose or self.use_reid or self.use_quality
        ) else None
        self.segmentation_head = SegmentationHead(cfg, feature_spec) if self.use_segmentation else None
        self.pose_head = PoseHead(cfg, feature_spec) if self.use_pose else None
        self.mesh_head = MeshHead(cfg, feature_spec) if self.use_mesh else None
        self.reid_head = ReIDHead(cfg, feature_spec) if self.use_reid else None
        self.depth_head = DepthHead(cfg, feature_spec) if self.use_depth else None
        self.quality_head = QualityHead(cfg, feature_spec) if self.use_quality else None

    def forward(
        self,
        features: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]] | None = None,
    ) -> dict[str, torch.Tensor]:
        self.feature_spec.validate(features)
        outputs: dict[str, torch.Tensor] = {}
        detection_tokens: torch.Tensor | None = None

        if self.detection_head is not None:
            # Single decoder pass — produces both the detection outputs AND the
            # shared per-object query tokens used by pose/reid/quality.
            detection_tokens, pred_logits, pred_boxes, dn_outputs = self.detection_head.decode(
                features, targets=targets
            )
            if self.use_detection:
                outputs["pred_logits"] = pred_logits
                outputs["pred_boxes"] = pred_boxes
                if dn_outputs is not None:
                    outputs.update(dn_outputs)
        if self.segmentation_head is not None:
            outputs["pred_seg"] = self.segmentation_head(features)
        if self.pose_head is not None:
            if detection_tokens is None:
                raise RuntimeError("pose requires detection-like query tokens, but none is enabled")
            outputs["pred_keypoints"] = self.pose_head(detection_tokens)
        if self.mesh_head is not None:
            outputs["pred_mesh"] = self.mesh_head(features)
        if self.reid_head is not None:
            if detection_tokens is None:
                raise RuntimeError("reid requires detection-like query tokens, but none is enabled")
            outputs["pred_reid"] = self.reid_head(detection_tokens)
        if self.depth_head is not None:
            outputs["pred_depth"] = self.depth_head(features)
        if self.quality_head is not None:
            if detection_tokens is None:
                raise RuntimeError("quality requires detection-like query tokens, but none is enabled")
            outputs["pred_quality"] = self.quality_head(detection_tokens)
        return outputs
