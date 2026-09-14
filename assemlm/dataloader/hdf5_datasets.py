"""HDF5 datasets for the fixed AssemLM 2.0 training and evaluation contract."""

from __future__ import annotations

from collections import Counter
import logging
import os
from pathlib import Path
from typing import Dict, List

import h5py
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, Dataset, Subset
from tqdm import tqdm

from assemlm.dataloader.assemble.assemble_collator import AssembleCollator
import assemlm.dataloader.assemble.dataset_mixture as dataset_mixture
from assemlm.utils.rigid_geometry import matrix_to_rotation_6d


logger = logging.getLogger(__name__)

REQUIRED_FEATURES = (
    "src_pc",
    "tgt_pc",
    "imgs",
    "lang",
    "asset",
    "category",
    "pre_pose",
)
POSE_QUERY_TOKEN = "<pose_query>"


def _parse_features(raw_features) -> List[str]:
    features = (
        [value.strip() for value in raw_features.split(",") if value.strip()]
        if isinstance(raw_features, str)
        else list(raw_features)
    )
    if set(features) != set(REQUIRED_FEATURES) or len(features) != len(REQUIRED_FEATURES):
        raise ValueError(
            "AssemLM 2.0 requires data_features=" + ",".join(REQUIRED_FEATURES)
        )
    return features


def _validate_data_contract(data_args) -> None:
    if not bool(getattr(data_args, "import_rotation", True)):
        raise ValueError("AssemLM 2.0 requires import_rotation=true.")
    if str(getattr(data_args, "centering_mode", "mean")).lower() != "mean":
        raise ValueError("AssemLM 2.0 requires centering_mode=mean.")
    if str(getattr(data_args, "pose_target_mode", "legacy")).lower() != "legacy":
        raise ValueError("AssemLM 2.0 requires the legacy pose target contract.")


def _parse_mix(raw_mix: str) -> List[str]:
    """Resolve a ``+`` mixture, or discover all processed files for ``auto``."""
    requested = str(raw_mix).strip()
    if requested.lower() in {"auto", "all", "*"}:
        names = dataset_mixture.available_dataset_names()
    else:
        names = [name.strip() for name in requested.split("+") if name.strip()]
    if not names:
        raise ValueError("Dataset mixture must contain at least one dataset name.")
    return [dataset_mixture.resolve_dataset_name(name) for name in names]


def _build_datasets(names: List[str], split: str, features: List[str]):
    return [
        AssembleDataset(
            data_path=dataset_mixture.DATASETS_LEGACY[name].hdf5_path,
            data_features=features,
            split=split,
            dataset_name=name,
        )
        for name in names
    ]


def _concat(datasets):
    return datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)


def make_assemble_data_module(tokenizer, processor, data_args) -> Dict:
    """Build train/test mixtures from the processed AssemLM 2.0 datasets."""
    del processor
    _validate_data_contract(data_args)
    features = _parse_features(data_args.data_features)

    dataset_mixture.register_datasets_mixtures()
    train_names = _parse_mix(data_args.train_mix)
    test_names = _parse_mix(data_args.test_mix)
    train_dataset = _concat(_build_datasets(train_names, "train", features))
    eval_dataset = _concat(_build_datasets(test_names, "test", features))

    message = (
        "AssemLM 2.0 HDF5 data module built | "
        f"train_mix={data_args.train_mix} | test_mix={data_args.test_mix} | "
        f"train_samples={len(train_dataset)} | eval_samples={len(eval_dataset)}"
    )
    print(message)
    logger.info(message)
    return {
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": AssembleCollator(
            tokenizer=tokenizer,
            label_pad_token_id=data_args.label_pad_token_id,
        ),
    }


class AssembleDataset(Dataset):
    """Read one HDF5 split and apply the v2 moving-part augmentation."""

    def __init__(
        self,
        data_path: str,
        data_features,
        split: str = "train",
        dataset_name: str | None = None,
        augmentation_seed: int | None = None,
    ):
        super().__init__()
        self.data_path = data_path
        self.dataset_name = dataset_name or Path(data_path).stem
        self.data_features = data_features
        self.augmentation_seed = (
            None if augmentation_seed is None else int(augmentation_seed)
        )
        self.h5f = None
        self.list_data_dict = self._load_index(split)

    def set_augmentation_seed(self, seed: int | None) -> None:
        """Use per-index RNG streams for deterministic evaluation, or reset to random."""
        self.augmentation_seed = None if seed is None else int(seed)

    @staticmethod
    def _decode(value) -> str:
        text = value.decode("utf-8", errors="ignore") if isinstance(value, bytes) else str(value)
        return text.strip("\x00 \t\r\n")

    def _load_index(self, split: str):
        if Path(self.data_path).suffix.lower() not in {".hdf5", ".h5"}:
            raise ValueError(f"Expected an HDF5 dataset, got: {self.data_path}")
        if not os.path.isfile(self.data_path):
            raise FileNotFoundError(f"HDF5 dataset not found: {self.data_path}")

        with h5py.File(self.data_path, "r") as h5f:
            if "split" not in h5f or split not in h5f["split"]:
                available = list(h5f.get("split", {}).keys())
                raise KeyError(f"Split {split!r} not found. Available splits: {available}")
            if "objs" not in h5f:
                raise KeyError("Group 'objs' not found in HDF5 dataset.")

            asset_names = [self._decode(value) for value in h5f["split"][split][()]]
            abandon = (
                {self._decode(value) for value in h5f["split"]["abandon"][()]}
                if "abandon" in h5f["split"]
                else set()
            )
            objects = h5f["objs"]
            samples = []
            category_counts = Counter()
            missing = 0
            skipped = 0

            for asset_name in tqdm(asset_names, desc=f"Loading split={split}"):
                if asset_name in abandon:
                    skipped += 1
                    continue
                key = asset_name.replace("/", "_")
                if key not in objects:
                    missing += 1
                    continue
                group = objects[key]
                category = (
                    self._decode(group["category"][()])
                    if "category" in group
                    else key.rsplit("_", 1)[0]
                )
                category = category.strip() or "Unlabeled"
                samples.append({"asset": key, "category": category})
                category_counts[category] += 1

        print(
            f"Loaded HDF5 split={split} | samples={len(samples)} | "
            f"missing={missing} | abandoned={skipped} | path={self.data_path}"
        )
        if category_counts:
            top_categories = category_counts.most_common(20)
            print("Top categories: " + ", ".join(f"{name}:{count}" for name, count in top_categories))
        return samples

    def __len__(self):
        return len(self.list_data_dict)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["h5f"] = None
        return state

    def __del__(self):
        h5f = getattr(self, "h5f", None)
        if h5f is not None:
            h5f.close()

    @staticmethod
    def _resample_point_cloud(
        point_cloud: np.ndarray,
        sample_count: int = 1024,
        rng=None,
    ):
        if len(point_cloud) == sample_count:
            return point_cloud
        choice = np.random.choice if rng is None else rng.choice
        indices = choice(
            len(point_cloud), sample_count, replace=len(point_cloud) < sample_count
        )
        return point_cloud[indices]

    @staticmethod
    def _validate_point_cloud(
        point_cloud: np.ndarray,
        asset: str,
        key: str,
    ) -> np.ndarray:
        """Validate one HDF5 point cloud before resampling or augmentation."""
        value = np.asarray(point_cloud)
        if value.ndim != 2 or value.shape[1] != 3 or value.shape[0] == 0:
            raise ValueError(
                f"Asset {asset!r} has invalid {key!r} shape {value.shape}; "
                "expected a non-empty [N, 3] array."
            )
        value = value.astype(np.float32, copy=False)
        if not np.isfinite(value).all():
            raise ValueError(f"Asset {asset!r} contains non-finite values in {key!r}.")
        return value

    @staticmethod
    def _random_rotation(generator: torch.Generator | None = None) -> torch.Tensor:
        vectors = (
            torch.rand(1, 6, generator=generator)
            .reshape(-1, 2, 3)
            .permute(0, 2, 1)
        )
        first = F.normalize(vectors[:, :, 0], p=2, dim=1)
        second_raw = vectors[:, :, 1]
        second = F.normalize(
            second_raw
            - torch.bmm(first.view(1, 1, -1), second_raw.view(1, -1, 1)).view(1, 1)
            * first,
            p=2,
            dim=1,
        )
        third = torch.cross(first, second, dim=1)
        return torch.stack([first, second, third], dim=1).permute(0, 2, 1).reshape(3, 3)

    @classmethod
    def _augment_moving_part(
        cls,
        point_cloud: np.ndarray,
        generator: torch.Generator | None = None,
    ):
        center = point_cloud.mean(axis=0)
        rotation = cls._random_rotation(generator=generator).float()
        centered = torch.as_tensor(point_cloud - center, dtype=torch.float32)
        transformed = (rotation @ centered.T).T
        rotation_6d = matrix_to_rotation_6d(rotation).numpy()
        return transformed, center.astype(np.float32), rotation_6d.astype(np.float32)

    def _read_images(self, group):
        # twobytwo is the only released dataset whose preferred manual is
        # lineart.  All other datasets use freestyle.  A fallback keeps the
        # loader usable with older files that contain only one image pair.
        is_twobytwo = (
            "twobytwo" in str(self.dataset_name).lower()
            or "twobytwo" in Path(self.data_path).stem.lower()
        )
        preferred = (
            ("image_base_lineart", "image_assemble_lineart"),
            ("image_base_freestyle", "image_assemble_freestyle"),
        ) if is_twobytwo else (
            ("image_base_freestyle", "image_assemble_freestyle"),
            ("image_base_lineart", "image_assemble_lineart"),
        )
        for first_key, second_key in preferred:
            if first_key in group and second_key in group:
                images = []
                for key in (first_key, second_key):
                    value = np.asarray(group[key][()])
                    if (
                        value.ndim != 3
                        or value.shape[0] == 0
                        or value.shape[1] == 0
                        or value.shape[-1] not in (3, 4)
                        or value.dtype.kind not in "uif"
                    ):
                        raise ValueError(
                            f"Image {key!r} has invalid shape/dtype: "
                            f"shape={value.shape}, dtype={value.dtype}."
                        )
                    numeric = value.astype(np.float64, copy=False)
                    if (
                        not np.isfinite(numeric).all()
                        or numeric.min() < 0
                        or numeric.max() > 255
                    ):
                        raise ValueError(
                            f"Image {key!r} contains non-finite or out-of-range values."
                        )
                    images.append(Image.fromarray(value.astype(np.uint8)))
                return images
        return []

    @classmethod
    def _read_instruction(cls, group, category: str = ""):
        # New assemlm_*.hdf5 files intentionally omit the legacy instruction
        # dataset.  Return the exact prompt assembled by AssemLM2 for logs
        # and downstream result tables without reintroducing it into HDF5.
        if category:
            return f"Assemble the {category} object {POSE_QUERY_TOKEN}"
        return cls._decode(group["instruction"][()]) if "instruction" in group else ""

    def __getitem__(self, index):
        if self.h5f is None:
            self.h5f = h5py.File(self.data_path, "r")

        sample = self.list_data_dict[index]
        asset = sample["asset"]
        group = self.h5f["objs"][asset]
        if "partA-pc" not in group or "base_partB-pc" not in group:
            raise KeyError(f"Missing point clouds for asset: {asset}")

        category = sample["category"]
        rng = None
        torch_generator = None
        if self.augmentation_seed is not None:
            # RandomState accepts only unsigned 32-bit seeds. Modulo keeps
            # negative user-provided seeds well-defined while retaining the
            # per-index deterministic stream contract.
            sample_seed = (self.augmentation_seed + int(index)) % (2**32 - 1)
            rng = np.random.RandomState(sample_seed)
            torch_generator = torch.Generator(device="cpu")
            torch_generator.manual_seed(sample_seed)
        moving = self._resample_point_cloud(
            self._validate_point_cloud(group["partA-pc"][()], asset, "partA-pc"),
            rng=rng,
        )
        fixed = self._resample_point_cloud(
            self._validate_point_cloud(group["base_partB-pc"][()], asset, "base_partB-pc"),
            rng=rng,
        )
        moving, center, rotation_6d = self._augment_moving_part(
            moving, generator=torch_generator
        )
        # The reference loader also augments the fixed part before discarding
        # that result. Preserve its torch RNG consumption for exact replay.
        self._augment_moving_part(fixed, generator=torch_generator)

        values = {
            "src_pc": moving.T.float(),
            "tgt_pc": torch.from_numpy(fixed.T).float(),
            "imgs": self._read_images(group),
            "lang": self._read_instruction(group, category),
            "asset": asset,
            "category": category,
            "pre_pose": np.concatenate([center, rotation_6d]).astype(np.float32),
        }
        return {feature: values[feature] for feature in self.data_features}


def set_augmentation_seed(
    dataset,
    seed: int | None,
    _index_offset: int = 0,
) -> None:
    """Apply deterministic global-index seeds to nested dataset containers.

    ``ConcatDataset`` presents child datasets through one global index space.
    Carrying the cumulative child length into each child's seed avoids
    duplicate augmentation streams at the boundaries of mixed training sets,
    while leaving a standalone dataset (the evaluation case) unchanged.
    """
    if isinstance(dataset, AssembleDataset):
        if seed is None:
            dataset.set_augmentation_seed(None)
        else:
            dataset.set_augmentation_seed(int(seed) + int(_index_offset))
        return
    if isinstance(dataset, Subset):
        set_augmentation_seed(dataset.dataset, seed, _index_offset)
        return
    if isinstance(dataset, ConcatDataset):
        child_offset = int(_index_offset)
        for child in dataset.datasets:
            set_augmentation_seed(child, seed, child_offset)
            child_offset += len(child)
        return
    raise TypeError(f"Unsupported dataset container: {type(dataset).__name__}")
