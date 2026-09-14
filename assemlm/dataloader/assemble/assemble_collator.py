"""Collation for the seven fixed fields used by AssemLM 2.0."""

from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass
class AssembleCollator:
    tokenizer: object = None
    label_pad_token_id: int = -100

    def __call__(self, samples):
        if not samples:
            return {}
        batch = {}
        for key in samples[0]:
            values = [sample[key] for sample in samples]
            if key == "pre_pose":
                batch[key] = torch.stack([torch.as_tensor(value) for value in values])
            elif key in {"src_pc", "tgt_pc", "imgs"}:
                batch[key] = values
            else:
                batch[key] = values
        return batch
