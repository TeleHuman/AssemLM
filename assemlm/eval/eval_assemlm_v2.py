#!/usr/bin/env python3
"""Evaluate the fixed AssemLM 2.0 continuous-pose model."""

from __future__ import annotations

import argparse
import json
import os
import random
import time

from accelerate import Accelerator
import numpy as np
from omegaconf import OmegaConf
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from assemlm.dataloader import build_dataloader
from assemlm.dataloader.hdf5_datasets import set_augmentation_seed
from assemlm.model.framework import build_framework
from assemlm.model.modules.point_encoder.vn_dgcnn.models.ChamferDistancePytorch.chamfer3D import (
    dist_chamfer_3D,
)
from assemlm.model.modules.point_encoder.vn_dgcnn.utils import bgs, get_6d_rot_loss
from assemlm.training.trainer_utils.trainer_tools import normalize_dotlist_args


PA_METRIC_SPECS = (
    ("PA(0.01)", 0.01),
    ("PA(0.008)", 0.008),
    ("PA(0.005)", 0.005),
)


def _atomic_write_json(output_file: str | os.PathLike[str], payload: dict) -> None:
    """Publish evaluation JSON only after the complete document is serialized."""
    output_path = os.fspath(output_file)
    temporary_path = output_path + ".partial"
    try:
        with open(temporary_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
        os.replace(temporary_path, output_path)
    except Exception:
        try:
            os.unlink(temporary_path)
        except FileNotFoundError:
            pass
        raise


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _validate_config(cfg) -> None:
    errors = []
    if cfg.framework.name != "AssemLM2":
        errors.append("framework.name must be AssemLM2")
    if not bool(cfg.framework.pose_regression.enabled):
        errors.append("framework.pose_regression.enabled must be true")
    data_cfg = cfg.datasets.assemble_data
    if str(data_cfg.centering_mode).lower() != "mean":
        errors.append("centering_mode must be mean")
    if errors:
        raise ValueError("Invalid AssemLM 2.0 evaluation configuration: " + "; ".join(errors))


def _load_checkpoint(model, checkpoint_path: str, accelerator: Accelerator) -> None:
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state_dict, dict):
        raise TypeError("Checkpoint must contain a mapping of parameter names to tensors.")
    if "module" in state_dict and isinstance(state_dict["module"], dict):
        state_dict = state_dict["module"]
    state_dict = {
        key[7:] if key.startswith("module.") else key: value
        for key, value in state_dict.items()
    }
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            "Checkpoint does not match AssemLM 2.0: "
            f"missing_keys={missing_keys}, unexpected_keys={unexpected_keys}"
        )
    accelerator.print("Checkpoint loaded with no missing or unexpected keys.")


def _select_dataloader(cfg, model, split: str, accelerator: Accelerator):
    dataloaders = build_dataloader(
        cfg=cfg,
        dataset_py=cfg.datasets.assemble_data.dataset_py,
        tokenizer=model.pvlm_interface.tokenizer,
        processor=model.pvlm_interface.processor,
    )
    train_dataloader, test_dataloader = dataloaders
    accelerator.print(f"Using {split.upper()} split for evaluation.")
    return train_dataloader if split == "train" else test_dataloader


def _move_pose_targets(batch, device):
    pre_pose = batch["pre_pose"].to(device=device, dtype=torch.float32)
    return pre_pose[:, :3].reshape(-1, 3, 1), pre_pose[:, 3:9]


def _evaluate_batch(
    model,
    batch,
    device,
    chamfer_dist_fn,
    original_indices,
):
    instructions = batch["lang"]
    categories = batch["category"]
    assets = batch["asset"]
    gt_trans, gt_rot_6d = _move_pose_targets(batch, device)
    src_point_clouds = [point_cloud.to(device=device, dtype=torch.float32) for point_cloud in batch["src_pc"]]

    generated_poses = model.generate(batch)
    expected_batch = len(src_point_clouds)
    if not isinstance(generated_poses, torch.Tensor) or generated_poses.ndim != 2:
        raise RuntimeError(
            "AssemLM2.generate() must return a tensor with shape [batch, 9]; "
            f"got {type(generated_poses).__name__} with shape "
            f"{getattr(generated_poses, 'shape', None)}."
        )
    if generated_poses.shape != (expected_batch, 9):
        raise RuntimeError(
            "AssemLM2.generate() returned an unexpected pose shape: "
            f"expected ({expected_batch}, 9), got {tuple(generated_poses.shape)}."
        )
    generated_poses = generated_poses.to(device=device, dtype=torch.float32)
    valid_mask = (
        (generated_poses[:, 0] != -100.0)
        & (generated_poses != 0).any(dim=-1)
        & torch.isfinite(generated_poses).all(dim=-1)
    )
    pred_trans = generated_poses[:, :3].reshape(-1, 3, 1)
    pred_rot_6d = generated_poses[:, 3:9]

    rotation_error = get_6d_rot_loss(gt_rot_6d, pred_rot_6d)
    translation_error = (
        (pred_trans - gt_trans).pow(2).mean(dim=1).sqrt().squeeze(-1)
    )
    pred_rotation = bgs(pred_rot_6d.reshape(-1, 2, 3).permute(0, 2, 1))
    gt_rotation = bgs(gt_rot_6d.reshape(-1, 2, 3).permute(0, 2, 1))

    cd_values = []
    cd_r_values = []
    for sample_index, point_cloud in enumerate(src_point_clouds):
        predicted = (
            pred_rotation[sample_index].T @ point_cloud + pred_trans[sample_index]
        )
        target = gt_rotation[sample_index].T @ point_cloud + gt_trans[sample_index]
        predicted_star = (
            pred_rotation[sample_index].T @ point_cloud + gt_trans[sample_index]
        )
        d1, d2, _, _ = chamfer_dist_fn(
            target.unsqueeze(0).permute(0, 2, 1),
            predicted.unsqueeze(0).permute(0, 2, 1),
        )
        d1_star, d2_star, _, _ = chamfer_dist_fn(
            target.unsqueeze(0).permute(0, 2, 1),
            predicted_star.unsqueeze(0).permute(0, 2, 1),
        )
        cd_values.append(0.5 * (d1.mean() + d2.mean()))
        cd_r_values.append(0.5 * (d1_star.mean() + d2_star.mean()))
    cd_values = torch.stack(cd_values)
    cd_r_values = torch.stack(cd_r_values)
    pa_values = {
        metric_name: (cd_values < threshold).float()
        for metric_name, threshold in PA_METRIC_SPECS
    }

    rows = []
    for batch_index, original_index in enumerate(original_indices):
        valid = bool(valid_mask[batch_index].item())
        row = {
            "idx": original_index,
            "asset": assets[batch_index],
            "category": str(categories[batch_index]),
            "instruction": instructions[batch_index],
            "pose_loss": 0.0,
            "valid": valid,
            "true_T": gt_trans[batch_index].view(-1).cpu().tolist(),
            "true_R": gt_rotation[batch_index].cpu().tolist(),
            "pred_T": pred_trans[batch_index].view(-1).cpu().tolist() if valid else None,
            "pred_R": pred_rotation[batch_index].cpu().tolist() if valid else None,
        }
        if valid:
            row.update(
                {
                    "GD": rotation_error[batch_index].item(),
                    "RMSE(T)": translation_error[batch_index].item(),
                    "CD": cd_values[batch_index].item(),
                    "CD(R)": cd_r_values[batch_index].item(),
                    **{
                        metric_name: values[batch_index].item()
                        for metric_name, values in pa_values.items()
                    },
                }
            )
        else:
            row["error"] = "Model returned an invalid pose."
        rows.append(row)
    return rows, generated_poses


def _gather_results(results, accelerator: Accelerator):
    if not dist.is_initialized():
        return results
    shards = [None] * dist.get_world_size()
    dist.all_gather_object(shards, results)
    return [row for shard in shards for row in shard]


def _mean(rows, key):
    values = [row[key] for row in rows]
    return float(np.mean(values)) if values else 0.0


def _summarize_category(rows):
    summary = {
        "count": len(rows),
        "Avg GD": _mean(rows, "GD"),
        "RMSE(T)": _mean(rows, "RMSE(T)"),
        "CD": _mean(rows, "CD"),
        "CD(R)": _mean(rows, "CD(R)"),
    }
    for metric_name, _ in PA_METRIC_SPECS:
        summary[metric_name] = _mean(rows, metric_name)
    return summary


def _summarize(results, total_samples: int, selected_samples: int):
    valid_results = [row for row in results if row["valid"]]
    summary = {
        "Total Dataset Samples": total_samples,
        "Selected Samples": selected_samples,
        "Evaluated Samples": len(results),
        "Not Selected Samples": max(total_samples - selected_samples, 0),
        "Dropped Samples": max(selected_samples - len(results), 0),
        "Unevaluated Samples": max(total_samples - len(results), 0),
        "Valid Samples": len(valid_results),
        "Invalid Samples": len(results) - len(valid_results),
        "Avg Pose Loss": _mean(results, "pose_loss"),
    }
    if valid_results:
        summary.update(
            {
                "Avg GD": _mean(valid_results, "GD"),
                "RMSE(T)": _mean(valid_results, "RMSE(T)"),
                "CD": _mean(valid_results, "CD"),
                "CD(R)": _mean(valid_results, "CD(R)"),
            }
        )
        for metric_name, _ in PA_METRIC_SPECS:
            summary[metric_name] = _mean(valid_results, metric_name)

    category_names = sorted({row["category"] for row in valid_results})
    per_category = {
        category: _summarize_category(
            [row for row in valid_results if row["category"] == category]
        )
        for category in category_names
    }
    return summary, per_category


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config_yaml", required=True)
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--num_samples", type=int)
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--output_dir", default="eval_results")
    parser.add_argument("--dataset_name")
    parser.add_argument("--run_timestamp")
    parser.add_argument("--seed", type=int, default=42)
    args, overrides = parser.parse_known_args()
    if args.num_samples is not None and args.num_samples < 1:
        parser.error("--num_samples must be a positive integer when provided.")
    if args.seed < 0:
        parser.error("--seed must be non-negative.")
    return args, overrides


def evaluate() -> None:
    args, config_overrides = _parse_args()
    set_seed(args.seed)

    cfg = OmegaConf.merge(
        OmegaConf.load(args.config_yaml),
        OmegaConf.from_dotlist(normalize_dotlist_args(config_overrides)),
    )
    _validate_config(cfg)

    timestamp = args.run_timestamp or time.strftime("%Y%m%d_%H%M%S")
    try:
        time.strptime(timestamp, "%Y%m%d_%H%M%S")
    except ValueError as exc:
        raise ValueError("run_timestamp must use YYYYMMDD_HHMMSS format.") from exc
    output_dir = os.path.join(args.output_dir, timestamp)

    accelerator = Accelerator()
    device = accelerator.device
    accelerator.print(f"Output directory: {output_dir}")
    accelerator.print(f"Building model: {cfg.framework.vlm.base_vlm}")
    model = build_framework(cfg)
    _load_checkpoint(model, args.ckpt_path, accelerator)
    model.to(device)
    model.eval()
    # Model construction can consume torch/Python/NumPy RNG state. Reset it
    # before selecting and iterating the dataset so --seed controls evaluation
    # augmentation rather than incidental initialization draws.
    set_seed(args.seed)

    base_dataloader = _select_dataloader(cfg, model, args.split, accelerator)
    dataset = base_dataloader.dataset
    set_augmentation_seed(dataset, args.seed)
    indices = list(range(len(dataset)))
    if args.shuffle:
        random.shuffle(indices)
    selected_samples = min(args.num_samples or len(indices), len(indices))
    if args.num_samples is not None:
        indices = indices[:selected_samples]

    process_indices = np.array_split(indices, accelerator.num_processes)[
        accelerator.process_index
    ].tolist()
    eval_batch_size = max(
        1,
        int(
            cfg.datasets.assemble_data.get(
                "eval_batch_size",
                cfg.datasets.assemble_data.get("per_device_batch_size", 1),
            )
        ),
    )
    eval_dataloader = DataLoader(
        Subset(dataset, process_indices),
        batch_size=eval_batch_size,
        shuffle=False,
        collate_fn=base_dataloader.collate_fn,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )
    model = accelerator.prepare(model)
    model = accelerator.unwrap_model(model)
    model.eval()

    chamfer_dist_fn = dist_chamfer_3D.chamfer_3DDist()
    results = []
    processed = 0
    with torch.no_grad():
        for batch in tqdm(
            eval_dataloader,
            desc="Evaluating",
            disable=not accelerator.is_main_process,
        ):
            batch_size = len(batch["lang"])
            original_indices = process_indices[processed : processed + batch_size]
            batch_results, generated_poses = _evaluate_batch(
                model,
                batch,
                device,
                chamfer_dist_fn,
                original_indices,
            )
            results.extend(batch_results)
            processed += batch_size
            del generated_poses, batch
            torch.cuda.empty_cache()

    results = _gather_results(results, accelerator)
    if accelerator.is_main_process:
        summary, per_category = _summarize(
            results,
            total_samples=len(dataset),
            selected_samples=selected_samples,
        )
        print("Evaluation summary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")

        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "eval_results.json")
        _atomic_write_json(
            output_file,
            {
                "summary": summary,
                "per_category": per_category,
                "dataset_name": args.dataset_name,
                "split": args.split,
                "run_timestamp": timestamp,
                "seed": args.seed,
                "config": OmegaConf.to_container(cfg),
                "ckpt_path": args.ckpt_path,
                "details": results,
            },
        )
        print(f"Detailed results saved to {output_file}")


if __name__ == "__main__":
    evaluate()
