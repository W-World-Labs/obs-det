# Third-party notices and external-asset boundaries

The root MIT license covers newly authorized, confirmed project-owned OmniDet
source. Original upstream libraries, checkpoints, datasets, media and platform
components keep their own terms. No third-party source, installed dependency,
model weight, dataset, sample media, native plugin or TensorRT binary is bundled.

## Python dependencies and build tools

`pyproject.toml` declares runtime and optional dependencies; `requirements-ci.txt`
locks the Linux CPython 3.12 CPU check environment with artifact hashes.
`THIRD-PARTY-DEPENDENCIES.json` preserves the exact upstream metadata license
expressions/fields/classifiers for those package versions, without replacing
upstream license files or inventing a BSD clause variant. PyTorch and torchvision
CPU artifacts use the original PyTorch CPU index; the remaining packages use
public PyPI. Original dependency licenses/notices must accompany any future
artifact that actually redistributes their code or native libraries. This source
wheel/sdist does not bundle them. Setuptools, wheel, build, pytest and Ruff are
build/check tools, not an implicit right to third-party runtime or model assets.

## Optional pretrained backbone inputs

The timm library's code license does not license its pretrained checkpoints.
The nine retained presets name separate upstream assets; their model-card
metadata and public revisions are recorded independently in the inventory.

- The four [ConvNeXtV2 FCMAE cards](https://huggingface.co/timm/convnextv2_tiny.fcmae_ft_in1k)
  used by the atto/tiny/base/large presets declare **CC-BY-NC-4.0**. They must not
  be described as Apache-2.0 weights or as a general commercial-use grant.
- [FastViT T8](https://huggingface.co/timm/fastvit_t8.apple_dist_in1k) declares
  `other` and points to the original [Apple license](https://github.com/apple-aiml-research/ml-fastvit/blob/main/LICENSE),
  including its retained notices, endorsement and rights limitations. Those
  independent terms are not replaced by this repository's MIT grant.
- RepViT M1, EfficientViT B0 and ResNet50 cards currently declare Apache-2.0;
  EdgeNeXt XX-small declares MIT. Consult the exact card and checkpoint terms
  again when acquiring an asset; neither the names nor the metadata are bundled
  weights or proof of ImageNet rights.

The default clean-room backbone requests no pretrained weights. Pretrained
presets remain explicit operator choices; no CI job exercises their download.
CI sets offline model/dataset flags and runs tests in a loopback-only namespace.
No NLP/image/detection model or dataset is installed or fetched by publication.

## Datasets and platforms

The [VisDrone dataset](https://github.com/VisDrone/VisDrone-Dataset) and its
[HF mirror](https://huggingface.co/datasets/Voxel51/VisDrone2019-DET) retain
upstream dataset conditions. The mirror metadata can label CC-BY-SA-3.0 while
its Copyright Information states Attribution-NonCommercial-ShareAlike 3.0 and
prohibits commercial use; do not use the metadata label to erase that restriction.
The downloader is an explicit operator utility, not a dataset license grant.
COCO images and annotations also require their own applicable permissions.
The tiny synthetic test images are generated from project-owned test code and
are not extracts from VisDrone, COCO, ImageNet or an external media collection.

NVIDIA TensorRT, CUDA, drivers and `trtexec` remain separately installed platform
components under their upstream terms. No protected/custom plugin is included,
and the static detection graph uses ordinary PyTorch/ONNX operators. GPU/driver
and TensorRT acceptance remain independent of the hosted CPU source checks.
