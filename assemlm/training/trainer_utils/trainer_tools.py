"""Small training utilities used by the AssemLM 2.0 release."""

from __future__ import annotations

import torch
import torch.distributed as dist


def normalize_dotlist_args(args):
    """Convert argparse-style dotlist overrides to OmegaConf syntax."""
    normalized = []
    index = 0
    while index < len(args):
        argument = args[index]
        if argument.startswith("--"):
            key = argument[2:]
            if "=" in key:
                normalized.append(key)
            elif index + 1 < len(args) and not args[index + 1].startswith("--"):
                normalized.append(f"{key}={args[index + 1]}")
                index += 1
            else:
                normalized.append(f"{key}=true")
        index += 1
    return normalized


def build_param_lr_groups(model, cfg):
    """Build the release optimizer groups from module-specific learning rates."""
    learning_rate = cfg.trainer.learning_rate
    base_lr = float(learning_rate.base)
    freeze_patterns = [
        pattern.strip()
        for pattern in str(cfg.trainer.freeze_modules).split(",")
        if pattern.strip()
    ]
    frozen_ids = set()
    for path in freeze_patterns:
        module = model
        for name in path.split("."):
            module = getattr(module, name)
        frozen_ids.update(id(parameter) for parameter in module.parameters())

    used_ids = set()
    groups = []
    for module_name, module_lr in (
        ("pvlm_interface", learning_rate.pvlm_interface),
    ):
        module = getattr(model, module_name)
        parameters = [
            parameter for parameter in module.parameters() if id(parameter) not in frozen_ids
        ]
        if parameters:
            groups.append(
                {
                    "params": parameters,
                    "lr": float(module_lr),
                    "name": module_name,
                }
            )
            used_ids.update(id(parameter) for parameter in parameters)

    remaining = [
        parameter
        for parameter in model.parameters()
        if id(parameter) not in frozen_ids and id(parameter) not in used_ids
    ]
    if remaining:
        groups.append({"params": remaining, "lr": base_lr, "name": "base"})
    return groups


class TrainerUtils:
    """Operations shared by the one published training loop."""

    @staticmethod
    def freeze_backbones(model, freeze_modules=""):
        frozen = []
        for raw_path in str(freeze_modules).split(","):
            path = raw_path.strip()
            if not path:
                continue
            module = model
            for name in path.split("."):
                module = getattr(module, name)
            for parameter in module.parameters():
                parameter.requires_grad = False
            frozen.append(path)
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        if frozen:
            print(f"Frozen modules: {', '.join(frozen)}")
        return model

    @staticmethod
    def print_trainable_parameters(model):
        if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
            return
        total = sum(parameter.numel() for parameter in model.parameters())
        trainable = sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        )
        print(
            f"Parameters (millions): {total / 10**6:.3f} total, "
            f"{trainable / 10**6:.3f} trainable"
        )
        return total, trainable

    @staticmethod
    def setup_distributed_training(accelerator, *components):
        return accelerator.prepare(*components)
