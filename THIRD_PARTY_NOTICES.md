# Third-Party Notices

AssemLM is released under the MIT License (see [LICENSE](LICENSE)). The
repository bundles or adapts code from the open-source projects below, and the
released checkpoints and datasets are built on third-party models and datasets.
Every component stays under its own license; the notes below provide the
attribution and point to the original sources.

If you believe a component is missing, mis-attributed, or used in a way that
conflicts with its license, please open an issue and we will correct it promptly.

## Bundled and adapted code

### ChamferDistancePytorch

- Location: `assemlm/model/modules/point_encoder/vn_dgcnn/models/ChamferDistancePytorch/`
  (the release keeps the `chamfer3D` extension; the archived copy under
  `assemlm/legacy/v1/...` keeps the full 2D/3D/5D/6D set together with the
  upstream `LICENSE` and `README.md`).
- Source: <https://github.com/ThibaultGROUEIX/ChamferDistancePytorch>
- License: MIT License, Copyright (c) 2019 ThibaultGROUEIX.
- Notes: the CUDA extension is compiled lazily (JIT) on first use.

### Vector Neurons / VN-DGCNN point encoder

- Location: `assemlm/model/modules/point_encoder/vn_dgcnn/`
  (`vn_layers.py`, `vn_dgcnn.py`, `utils.py`, ...).
- Source: the public Vector Neurons reference implementation,
  <https://github.com/FlyingGiraffe/vnn> (and its point-cloud variant
  `vnn-pc`).
- License: MIT License.
- Notes: adapted for AssemLM 1.1/2.0 — RMSNorm-based normalization, the patch
  encoder, patch sampling helpers, and the pose-path rotation utilities were
  modified or added here.

### StarVLA

- Location (archived V1 implementation):
  `assemlm/legacy/v1/model/modules/pose_head/Linear_Tokenizer.py`,
  `assemlm/legacy/v1/model/modules/vlm/QWen2_5.py`,
  `assemlm/legacy/v1/model/modules/vlm/QWen3.py`,
  `assemlm/legacy/v1/model/modules/vlm/Florence2.py`.
- Source: <https://github.com/starVLA/starVLA>
- License: MIT License — the file headers state
  "Copyright 2025 starVLA community. All rights reserved.
  Licensed under the MIT License, Version 1.0" (implemented by Jinhui YE, HKUST).

### OpenVLA (Prismatic)

- Location: `assemlm/legacy/v1/training/trainer_utils/overwatch.py`.
- Source: <https://github.com/openvla/openvla>
- License: MIT License — the file header states
  "Original file from OpenVLA project (Prismatic), licensed under MIT License."

### X-VLA

- Location: `assemlm/legacy/v1/model/modules/vlm/Florence2.py` contains code
  marked as originally coming from X-VLA.
- Source: <https://github.com/2toinf/X-VLA>
- License: see the upstream repository.

## Models and runtime dependencies

| Component | Role in this release | License |
|---|---|---|
| [Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) | base vision-language model of the released checkpoint | Apache-2.0 |
| [Transformers](https://github.com/huggingface/transformers) | modeling and tokenizer runtime | Apache-2.0 |
| [PyTorch](https://github.com/pytorch/pytorch) / [torchvision](https://github.com/pytorch/vision) | deep-learning runtime | BSD-3-Clause |
| [DeepSpeed](https://github.com/microsoft/DeepSpeed) / [Accelerate](https://github.com/huggingface/accelerate) | distributed training launchers | Apache-2.0 |
| [Flask](https://github.com/pallets/flask) | GUI server | BSD-3-Clause |
| [h5py](https://github.com/h5py/h5py), [NumPy](https://github.com/numpy/numpy), [matplotlib](https://github.com/matplotlib/matplotlib), [Pillow](https://github.com/python-pillow/Pillow), [trimesh](https://github.com/mikedh/trimesh), [Rich](https://github.com/Textualize/rich), [wandb](https://github.com/wandb/wandb), [qwen-vl-utils](https://github.com/QwenLM/Qwen2.5-VL) | data loading, visualization, UI, logging | see each project |

The released AssemLM 2.0 checkpoint is a fine-tune built on the
Qwen3-VL-2B-Instruct backbone; please follow the upstream model license in
addition to the notes in the model card.

## Datasets

| Dataset | Usage | License |
|---|---|---|
| [IKEA-Manual](https://cs.stanford.edu/~rcwang/projects/ikea_manual/) | shape-assembly objects and manual images | CC BY 4.0 |
| [PartNet](https://partnet.cs.stanford.edu/) | shape-assembly objects | MIT |
| [PartNeXt](https://github.com/AuthorityWang/PartNeXt) | shape-assembly objects | see the PartNeXt release |
| BiAssembly | shape-assembly objects | see the corresponding dataset release |
| [Two by Two](https://tea-lab.github.io/TwoByTwo/) | pairwise assembly objects | MIT |

The point clouds and rendered manuals in the AssemLM 2.0 datasets are
re-processed derivatives of these collections; see the dataset card for the
per-subset details. The MIT license of this repository does not supersede any
upstream dataset license, and users remain responsible for complying with the
terms of the original collections.
