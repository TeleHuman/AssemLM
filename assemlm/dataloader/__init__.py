"""HDF5 dataloader used by the AssemLM 2.0 train/eval path."""

from __future__ import annotations

import random

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _loader_seed(cfg, offset: int = 0):
    trainer_cfg = getattr(cfg, "trainer", None)
    data_cfg = getattr(getattr(cfg, "datasets", None), "assemble_data", None)
    reproducible = bool(getattr(trainer_cfg, "reproducible_training", False)) or hasattr(data_cfg, "seed")
    if not reproducible:
        return {}
    seed = int(getattr(data_cfg, "seed", getattr(cfg, "seed", 42))) + offset
    if dist.is_available() and dist.is_initialized():
        seed += dist.get_rank()
    generator = torch.Generator()
    generator.manual_seed(seed)
    return {"worker_init_fn": _seed_worker, "generator": generator}


def build_dataloader(cfg, dataset_py="hdf5datasets", tokenizer=None, processor=None):
    if dataset_py != "hdf5datasets":
        raise ValueError("The v2 release supports only dataset_py=hdf5datasets.")

    from assemlm.dataloader.hdf5_datasets import make_assemble_data_module

    data_cfg = cfg.datasets.assemble_data
    data_module = make_assemble_data_module(
        tokenizer=tokenizer,
        processor=processor,
        data_args=data_cfg,
    )
    common = dict(
        batch_size=data_cfg.per_device_batch_size,
        collate_fn=data_module["data_collator"],
        num_workers=int(getattr(data_cfg, "num_workers", 4)),
    )
    train_loader = DataLoader(
        data_module["train_dataset"],
        # IKEA is intentionally test-only in the processed v2 release. A
        # zero-length training split must use SequentialSampler; PyTorch's
        # RandomSampler rejects an empty dataset before evaluation can run.
        shuffle=bool(getattr(data_cfg, "shuffle", True))
        and len(data_module["train_dataset"]) > 0,
        drop_last=bool(getattr(data_cfg, "drop_last", True)),
        **common,
        **_loader_seed(cfg),
    )
    eval_dataset = data_module.get("eval_dataset")
    if eval_dataset is None:
        return train_loader
    eval_loader = DataLoader(
        eval_dataset,
        shuffle=False,
        drop_last=False,
        **common,
        **_loader_seed(cfg, offset=10000),
    )
    return train_loader, eval_loader
