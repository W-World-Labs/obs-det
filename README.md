# OmniDet

Clean-room NVIDIA-first detector for small person and vehicle targets in UAV
and surveillance video.

## Current Scope

- Backbone: lightweight clean-room P2/P3/P4/P5 pyramid, plus a reviewed
  pretrained `timm` feature-backbone adapter for 2026 ECDet/DINOv3-style
  experiments.
- Neck: `EdgeMambaFPN` with deployable `conv_gated` path and `ssm_lite`
  research path.
- Head: `MultiTaskHead` on a shared backbone/neck. Detection (dense one-to-one
  DETR/DEIM-style) is the anchor task and the default; segmentation, pose, mesh,
  reid, depth, and quality heads are opt-in via `active_tasks` and reuse the
  shared trunk features instead of recomputing them.
- Data: VisDrone text annotations and COCO json training dataset.
- Training: single-machine PyTorch training entrypoint.
- Export: static ONNX export for TensorRT (detection head only in v0).

No Ultralytics, Mamba-YOLO, YOLO11-Mamba, YOLO configs, YOLO decode logic, or
YOLO weights are used.

## Public distribution and external assets

This repository contains the complete project-owned implementation, all original
regression checks, training/evaluation scripts and experiment configurations.
The owner newly licenses this project-owned source under MIT; see `LICENSE`.
It contains no checkpoints, ONNX/TensorRT engines, datasets, media, production
configuration or private history. The default clean-room backbone uses random
weights. Selecting a pretrained timm preset or `--pretrained-backbone` explicitly
requests separately obtained upstream weights and can fetch them through timm;
review their model-specific terms first. The timm code license does not license
its checkpoints: ConvNeXtV2 FCMAE weights are marked CC-BY-NC-4.0 and FastViT has
Apple-specific terms. See `THIRD-PARTY-NOTICES.md` before acquiring any asset.
VisDrone and COCO access and reuse rights must be established independently.
The auxiliary pose/reid/depth/mesh heads remain research paths; synthetic
shape/finite-output tests do not establish accuracy on real task data.

## Known research limitations

The current COCO 17-keypoint loader returns an `N x 17 x 3` tensor, while the
pose loss normalizer expects another representation; that path currently does
not produce pose supervision. ReID and quality targets with missing values can
also lose alignment with filtered boxes. Review these paths before auxiliary
real-data training. Existing synthetic tests exercise shape/finite-output and
loss-key contracts; they do not prove these task labels are aligned. The default
detection path remains the primary training and export scope.

## Development

Install the declared dependencies in your own environment. Configuration YAML
requires PyYAML; COCO metrics require the optional `evaluation` extra.

```bash
python -m pip install -e '.[dev,evaluation]'
PYTHONPATH=src python -m pytest -v
```

## Smoke Check (Flywheel Gate)

Before spending compute on a full training run, gate the architecture on an
untrained (random-weight) model. `scripts/smoke_check.py` runs, in order:
forward numerics + gradient flow, per-component latency profiling, a tiny
fixed-batch overfit (is the model learnable?), and v0 ONNX export. It exits
non-zero if any gate fails, so it can be wired into CI as an architecture gate.

```bash
PYTHONPATH=src python scripts/smoke_check.py --tasks detection
```

Multi-task run, smaller input for a fast iteration on the flywheel:

```bash
PYTHONPATH=src python scripts/smoke_check.py \
  --tasks all \
  --image-size 512 \
  --batch-size 2 \
  --overfit-steps 150 \
  --json-out logs/smoke_check.json
```

Speed-only pass on the RTX 5090 host (skip the slower gates):

```bash
PYTHONPATH=src python scripts/smoke_check.py --tasks detection --skip overfit onnx
```

## Backbone Sweep

`scripts/sweep_backbones.py` runs the profiling gate across every backbone
preset (plus the clean-room baseline) on random weights — no pretrained
download — and reports params, per-component latency, shared-trunk share, and
memory. Use it to reject backbones that blow the latency budget or cannot
expose the P2/P3/P4/P5 pyramid before spending training compute.

```bash
PYTHONPATH=src python scripts/sweep_backbones.py --csv-out logs/backbone_sweep.csv
```

## Train on your own GPU workstation

Download and convert the Hugging Face VisDrone mirror first:

```bash
PYTHONPATH=src HF_ENDPOINT=https://huggingface.co python scripts/download_visdrone_hf.py \
  --dataset Voxel51/VisDrone2019-DET \
  --split train \
  --out /data/omnidet_datasets/visdrone_hf/train \
  --cache-dir /data/omnidet_datasets/hf_cache
```

```bash
PYTHONPATH=src python scripts/train_visdrone.py \
  --images /data/omnidet_datasets/visdrone_hf/train/images \
  --annotations /data/omnidet_datasets/visdrone_hf/train/annotations \
  --hardware-profile rtx5090_32g \
  --epochs 150 \
  --device cuda \
  --lr-scheduler cosine \
  --lr-warmup-steps 500 \
  --lr-min 1e-6 \
  --compile \
  --out checkpoints/omnidet_v0.pt
```

训练时推荐的学习率观察输出可结合 `--log-csv` 导出 per-epoch 指标：

```bash
PYTHONPATH=src python scripts/train_visdrone.py \
  --images /data/omnidet_datasets/visdrone_hf/train/images \
  --annotations /data/omnidet_datasets/visdrone_hf/train/annotations \
  --hardware-profile rtx5090_32g \
  --epochs 150 \
  --device cuda \
  --lr-scheduler cosine \
  --lr-warmup-steps 500 \
  --lr-min 1e-6 \
  --grad-clip 1.0 \
  --log-csv logs/train_visdrone.csv \
  --out checkpoints/omnidet_v0.pt
```

For RTX 50-series Blackwell cards, the stable training path is BF16/FP16 AMP,
TF32, channels-last, cuDNN benchmark, fused AdamW, and optional `torch.compile`.
FP4 is kept for later TensorRT low-precision inference experiments rather than
the first training loop.

For faster convergence, use a reviewed pretrained feature backbone that exposes
4x/8x/16x/32x feature maps:

```bash
PYTHONPATH=src python scripts/train_visdrone.py \
  --images /data/omnidet_datasets/visdrone_hf/train/images \
  --annotations /data/omnidet_datasets/visdrone_hf/train/annotations \
  --hardware-profile rtx5090_32g \
  --epochs 20 \
  --device cuda \
  --backbone-preset fastvit_t8_dist \
  --out checkpoints/omnidet_pretrained_backbone.pt
```

Available pretrained backbone presets:

- `fastvit_t8_dist`: first RTX 5090 smoke-tested preset.
- `repvit_m1_dist`
- `efficientvit_b0`
- `convnextv2_atto_fcmae`
- `edgenext_xx_small`

The 2026 backbone priority is EdgeCrafter/ECDet-style compact ViT features
first, DEIMv2/DINOv3 distillation second, and RF-DETR as a teacher or benchmark.
ECDet and DEIMv2 expose 8x/16x/32x features in their public HF configs, so they
are used as teacher or benchmark models before a dedicated P2 adapter is added.
Mamba-style modules stay in the neck/global-context path for v0.

The hardware profiles are:

- `rtx5090_32g`: 960 input, batch 8, 300 queries, BF16.
- `rtxpro6000_96g`: 1280 input, batch 24, 600 queries, BF16.

To train directly on COCO with the same entrypoint:

```bash
PYTHONPATH=src python scripts/train_visdrone.py \
  --dataset coco \
  --images /data/coco/train2017 \
  --annotations /data/coco/annotations/instances_train2017.json \
  --hardware-profile rtx5090_32g \
  --epochs 12 \
  --device cuda \
  --compile \
  --out checkpoints/omnidet_coco.pt
```

## Export ONNX

`export_onnx(path, cfg)` constructs a new random-weight model and exports its
static detection graph. It does not load a trained checkpoint; this helper and
its CI check establish graph structure only, not trained-model predictions.

```bash
PYTHONPATH=src python - <<'PY'
from omnidet.export.onnx import export_onnx

export_onnx("artifacts/omnidet_960.onnx")
PY
```

## Build TensorRT FP16 Engine

Run on the RTX 5090 host with TensorRT installed:

```bash
scripts/build_trt_engine.sh artifacts/omnidet_960.onnx artifacts/omnidet_960_fp16.engine
```

## Profile PyTorch FP16

```bash
PYTHONPATH=src python scripts/profile_rtx5090.py --input-size 960 --warmup 20 --iters 100
```

## 快速做小数据集先跑通

Known limitation: the existing tiny-subset helper currently writes image and
annotation files together under `out/train` or `out/val`, while its pre-created
`images`/`annotations` subdirectories remain empty. The example chain below is
not an accepted end-to-end workflow; review/correct the output layout before
using it as training input. Source CI does not claim to validate that helper.

```bash
PYTHONPATH=src python scripts/make_tiny_visdrone.py \
  --images-dir /data/omnidet_datasets/visdrone_hf/train/images \
  --annotations-dir /data/omnidet_datasets/visdrone_hf/train/annotations \
  --out-dir /data/omnidet_datasets/visdrone_hf/tiny \
  --train-count 300 \
  --val-count 60 \
  --seed 42
```

对应训练方式（建议先跑 1-2 个 epoch）：

```bash
PYTHONPATH=src python scripts/train_visdrone.py \
  --dataset visdrone \
  --images /data/omnidet_datasets/visdrone_hf/tiny/train/images \
  --annotations /data/omnidet_datasets/visdrone_hf/tiny/train/annotations \
  --val-images /data/omnidet_datasets/visdrone_hf/tiny/val/images \
  --val-annotations /data/omnidet_datasets/visdrone_hf/tiny/val/annotations \
  --hardware-profile rtx5090_32g \
  --epochs 2 \
  --batch-size 2 \
  --image-size 512 \
  --device cuda \
  --lr-scheduler cosine \
  --lr-warmup-steps 200 \
  --grad-clip 1.0 \
  --log-csv logs/visdrone_tiny_smoke.csv \
  --out checkpoints/omnidet_tiny_smoke.pt
```

## Source CI and acceptance boundary

Standard public GitHub Actions scan committed source and complete reachable
history, install the hash-pinned CPU dependencies in `requirements-ci.txt`, run
all existing synthetic tests with loopback-only networking, and build wheel,
sdist and committed-source archives with a manifest and SHA256SUMS. CI downloads
no pretrained weights, datasets or media and performs no GPU profiling. Its
result does not establish real-data accuracy, GPU/driver compatibility, TensorRT
performance, model permissions or production activation. The training and
profiling commands above are explicit operator actions on their own inputs.
