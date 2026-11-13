# OmniDet provenance

This public distribution contains the complete confirmed project-owned OmniDet
implementation, its existing regression checks, scripts and experiment recipes.
The owner newly grants MIT for that source under `LICENSE`; this does not change
any upstream dependency, model, data, platform or media license. No private Git
history, operational inventories, credentials, checkpoints, media or datasets are
included. The generated source manifest identifies the exact public tree.

## Implementation and design references

The backbone, gated/low-rank neck, dense one-to-one decoder, denoising queries,
small-object weighting, data conversion and orchestration remain the existing
project-owned implementation. Paper names and mathematical ideas are design
references, not copied source or bundled pretrained implementations:
EdgeCrafter/ECDet, DEIMv2/DINOv3, RF-DETR, Le-DETR, MambaFPN, EdgeFusionNet,
DLRMamba, RT-DETR/DEIM, DAB-DETR/DN-DETR and normalized Wasserstein distance.
No Ultralytics, Mamba-YOLO, YOLO11-Mamba, AGPL-derived implementation, proprietary
TensorRT plugin or external model implementation is vendored. The `ssm_lite`
neck is the existing low-rank convolutional research mixer, not a claim to bundle
a third-party Mamba engine. Public APIs call separately installed original
PyTorch, scipy, timm, ONNX and optional pycocotools distributions.

## Separate external assets

`THIRD-PARTY-NOTICES.md` and `THIRD-PARTY-DEPENDENCIES.json` record dependency
and model-card declarations. The CPU CI requirements lock external package
artifacts and hashes; the artifact distribution contains none of those installed
packages. Selecting a pretrained preset can ask timm to obtain upstream weights;
the operator must review the model-specific license before doing so. In
particular, ConvNeXtV2 FCMAE cards declare CC-BY-NC-4.0, while FastViT points to
Apple-specific terms. A library's Apache-2.0 code license does not relicense
weights, ImageNet training data, or subsequent trained products.

VisDrone/COCO parsing and conversion code does not grant rights to their inputs.
VisDrone's non-commercial dataset conditions are not removed by a mirror's
metadata. No actual examples from those datasets are copied into this repository;
the existing tests construct small synthetic images and annotations at runtime.
Source CI establishes the existing synthetic contracts and package integrity,
not real-data accuracy, all auxiliary-task training correctness, GPU performance,
TensorRT activation or an operator's asset permissions. See README limitations.
