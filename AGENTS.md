# OmniDet source

Keep the complete existing detector, data parsers, training/evaluation scripts,
experiment configurations and original assertions. Use the smallest direct change;
do not replace the detector with a wrapper or a second implementation.

- `OmniDet` composes the backbone, `EdgeMambaFPN` and `MultiTaskHead` through
  `FeatureSpec` P2/P3/P4/P5 strides 4/8/16/32. The default clean-room backbone
  uses random weights; detection is the primary task.
- Preserve plain-MHA decoding, iterative box refinement, denoising-query training,
  Hungarian matching, small-object weighting and optional NWD. The `ssm_lite`
  neck and auxiliary heads remain research paths with README limitations.
- The frozen timm extractor stays in eval/no-grad; projections remain trainable.
  A timm code license does not license its model checkpoints. Keep all upstream
  asset terms separate, including ConvNeXtV2 non-commercial and Apple FastViT terms.
- Never vendor Ultralytics, Mamba-YOLO, YOLO11-Mamba, protected TensorRT plugins,
  third-party checkpoints, datasets, media, credentials or private history.
  Selecting a pretrained preset is an explicit external-asset action. CI must
  remain offline for model/data access and must not download pretrained weights.
- Dataset loaders emit uint8 HWC; GPU conversion lives in `images_to_device`.
  Preserve class mappings, synthetic fixtures and the shared-trunk task contracts.
- `scripts/train_visdrone.py` owns training; configuration precedence is CLI over
  YAML/JSON over hardware-profile/default values. Preserve public flags.
- `export_onnx(path, cfg)` constructs a random-weight static detection graph;
  it does not load a trained checkpoint. TensorRT/trtexec and GPU profiling are
  separate operator actions, not source-CI acceptance.
- Load only checkpoints the operator produced or independently trusts: the
  existing evaluation path uses `torch.load(..., weights_only=False)`.
- Build, existing tests and packaging run in standard public GitHub Actions.
  Local agent work is source/metadata review and dependency resolution only;
  do not install/build/test the product, infer/train, or fetch models/data locally.
- Preserve existing tests; add no unit/integration tests for publication metadata.
  `requirements-ci.txt` pins CPU CI artifacts and hashes. Package only committed
  source, notices and generated package metadata; retain seven-day artifacts.
- Keep `README.md`, `PROVENANCE.md`, `THIRD-PARTY-NOTICES.md` and package metadata
  aligned with any entrypoint, dependency or public-scope change. The MIT grant
  covers confirmed project-owned source only; no production activation is implied.
