#!/usr/bin/env python3
"""Persistent inference worker for the AssemLM evaluation model."""

from __future__ import annotations

import json
import os
import shutil
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any


def _runtime_project_root() -> Path:
    """Resolve the workspace root for source checkouts and installed wheels."""
    configured = os.environ.get("ASSEMLM_PROJECT_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    source_root = Path(__file__).resolve().parents[1]
    if (source_root / "config" / "assemlm_v2.yaml").is_file():
        return source_root
    return Path.cwd().resolve()


PROJECT_ROOT = _runtime_project_root()
MODEL_ROOT = Path(os.environ.get("ASSEMLM_MODEL_ROOT", PROJECT_ROOT)).expanduser().resolve()
MODEL_CACHE: dict[tuple[str, int, int, str, int, int], Any] = {}


def _model_cache_key(config_file: Path, checkpoint_file: Path) -> tuple[str, int, int, str, int, int]:
    """Return a cache key that changes when either model artifact is replaced.

    Paths alone are not sufficient: a checkpoint/config can be updated in place
    while keeping the same filename.  Including file size and nanosecond mtime
    makes a subsequent GUI reload build the new model instead of silently
    reusing the previous process-local instance.
    """
    config_stat = config_file.stat()
    checkpoint_stat = checkpoint_file.stat()
    return (
        str(config_file),
        int(config_stat.st_mtime_ns),
        int(config_stat.st_size),
        str(checkpoint_file),
        int(checkpoint_stat.st_mtime_ns),
        int(checkpoint_stat.st_size),
    )


def _resolve_sample_file(sample_dir: Path, raw_path: str | os.PathLike[str], label: str) -> Path:
    """Resolve a sample artifact without allowing metadata to escape its directory."""
    root = sample_dir.expanduser().resolve()
    candidate = Path(raw_path)
    resolved = (candidate if candidate.is_absolute() else root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} path must stay inside the sample directory: {raw_path!r}") from exc
    return resolved


def _atomic_save_npy(path: Path, array) -> None:
    """Write one NumPy artifact and publish it only after serialization succeeds."""
    temporary_path = path.with_name(path.name + ".partial")
    temporary_path.unlink(missing_ok=True)
    try:
        import numpy as np

        with temporary_path.open("wb") as handle:
            np.save(handle, array, allow_pickle=False)
        temporary_path.replace(path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON to a private temporary file before publishing the result."""
    temporary_path = path.with_name(path.name + ".partial")
    temporary_path.unlink(missing_ok=True)
    try:
        temporary_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def _load_point_cloud(path: Path, label: str):
    """Load a saved point cloud and enforce the GUI's non-empty ``[N, 3]`` format."""
    import numpy as np

    try:
        point_cloud = np.load(path, allow_pickle=False)
    except FileNotFoundError:
        raise FileNotFoundError(f"{label} point cloud not found: {path}") from None
    if point_cloud.ndim != 2 or point_cloud.shape[1] != 3 or point_cloud.shape[0] == 0:
        raise ValueError(
            f"{label} point cloud has invalid shape {point_cloud.shape}; "
            "expected a non-empty [N, 3] array."
        )
    point_cloud = point_cloud.astype(np.float32, copy=False)
    if not np.isfinite(point_cloud).all():
        raise ValueError(f"{label} point cloud contains non-finite values: {path}")
    return point_cloud


def _load_rgb_image(path: Path):
    """Open an image while closing the underlying file descriptor promptly."""
    from PIL import Image

    with Image.open(path) as image:
        return image.convert("RGB")


def _rotation_6d_to_matrix(torch, rotation_6d):
    vectors = rotation_6d.reshape(-1, 2, 3)
    first = torch.nn.functional.normalize(vectors[:, 0], dim=-1, eps=1e-8)
    second_raw = vectors[:, 1]
    second = torch.nn.functional.normalize(
        second_raw - (first * second_raw).sum(dim=-1, keepdim=True) * first,
        dim=-1,
        eps=1e-8,
    )
    third = torch.cross(first, second, dim=-1)
    return torch.stack((first, second, third), dim=-1)


def _load_model(config_path: str, checkpoint_path: str):
    import torch
    from omegaconf import OmegaConf

    config_file = Path(config_path).expanduser().resolve()
    checkpoint_file = Path(checkpoint_path).expanduser().resolve()
    if not config_file.is_file():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    if not checkpoint_file.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")
    if not torch.cuda.is_available():
        raise RuntimeError("The reference AssemLM model requires CUDA for inference.")

    cache_key = _model_cache_key(config_file, checkpoint_file)
    cached = MODEL_CACHE.get(cache_key)
    if cached is not None:
        return cached, True

    if str(MODEL_ROOT) not in sys.path:
        sys.path.insert(0, str(MODEL_ROOT))

    from assemlm.model.framework import build_framework

    cfg = OmegaConf.load(str(config_file))
    model = build_framework(cfg)
    checkpoint = torch.load(str(checkpoint_file), map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if isinstance(state_dict, dict) and "module" in state_dict and isinstance(state_dict["module"], dict):
        state_dict = state_dict["module"]
    if not isinstance(state_dict, dict):
        raise TypeError("Checkpoint does not contain a state dictionary.")
    state_dict = {
        key[7:] if key.startswith("module.") else key: value
        for key, value in state_dict.items()
    }
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            "Checkpoint does not match the AssemLM 2.0 model: "
            f"missing_keys={missing_keys}, unexpected_keys={unexpected_keys}"
        )
    model.to("cuda")
    model.eval()
    loaded = {
        "model": model,
        "device": torch.device("cuda"),
        "config_path": str(config_file),
        "checkpoint_path": str(checkpoint_file),
        "missing_keys": len(missing_keys),
        "unexpected_keys": len(unexpected_keys),
    }
    MODEL_CACHE.clear()
    MODEL_CACHE[cache_key] = loaded
    return loaded, False


def _symmetric_chamfer(torch, first, second):
    distances = torch.cdist(first.unsqueeze(0), second.unsqueeze(0)).squeeze(0).pow(2)
    return float(distances.min(dim=1).values.mean() + distances.min(dim=0).values.mean())


def _compute_pose_metrics(
    torch,
    source_tensor,
    predicted_moving,
    pred_translation,
    pred_rotation,
    gt_translation,
    gt_rotation,
):
    """Compute the three GUI metrics from one prediction and one GT pose.

    ``_symmetric_chamfer`` returns the sum of the two directed means (the
    historical SCD value), so both reported CD values explicitly divide it by
    two.  ``CD_R`` removes translation error by using the GT translation while
    retaining the predicted rotation, matching Eval's ``CD_star`` definition.
    """
    ground_truth_moving = (
        gt_rotation.T @ source_tensor + gt_translation.reshape(3, 1)
    ).T
    relative_rotation = gt_rotation.T @ pred_rotation
    rotation_cosine = ((torch.trace(relative_rotation) - 1.0) * 0.5).clamp(
        -1.0 + 1e-6, 1.0 - 1e-6
    )
    rmse_t = float(torch.sqrt((pred_translation - gt_translation).pow(2).mean()).item())
    scd = _symmetric_chamfer(torch, ground_truth_moving, predicted_moving)
    predicted_rotation_only = (
        pred_rotation.T @ source_tensor + gt_translation.reshape(3, 1)
    ).T
    cd_r_scd = _symmetric_chamfer(torch, ground_truth_moving, predicted_rotation_only)
    metrics = {
        "rotation_error_deg": float(torch.rad2deg(torch.acos(rotation_cosine)).item()),
        "RMSE_T": rmse_t,
        "CD": scd / 2.0,
        "CD_R": cd_r_scd / 2.0,
        # Legacy aliases retained for previously written GUI result readers.
        "T_error": rmse_t,
        "SCD": scd,
        "CD_star": cd_r_scd,
        "T_correct": bool(rmse_t < 0.01),
    }
    return ground_truth_moving, metrics


def _select_saved_images(meta: dict[str, Any], files: dict[str, Any], sample_dir: Path):
    """Resolve the image pair saved by the GUI, including lineart samples."""
    requested = str(meta.get("manual_type", "")).strip().lower()
    styles = ["lineart", "freestyle"] if requested.startswith("line") else ["freestyle", "lineart"]
    if not requested:
        has_lineart = any(
            (sample_dir / str(files.get(key, filename))).is_file()
            for key, filename in (
                ("image_base_lineart", "image_base_lineart.png"),
                ("image_assemble_lineart", "image_assemble_lineart.png"),
            )
        )
        if has_lineart:
            styles = ["lineart", "freestyle"]
    for style in styles:
        base_key = f"image_base_{style}"
        assemble_key = f"image_assemble_{style}"
        base = files.get(base_key, f"{base_key}.png")
        assemble = files.get(assemble_key, f"{assemble_key}.png")
        base_path = _resolve_sample_file(sample_dir, base, "Base image")
        assemble_path = _resolve_sample_file(sample_dir, assemble, "Assembly image")
        if base_path.is_file() and assemble_path.is_file():
            return base_path, assemble_path, style.capitalize()
    raise FileNotFoundError(
        "Stage 01 sample must contain a complete image_base/image_assemble "
        "lineart or freestyle pair."
    )


def _infer(request: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    import torch

    sample_dir = Path(request["sample_dir"]).expanduser().resolve()
    meta_path = sample_dir / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Sample metadata not found: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    files = meta.get("files", {})

    def resolve_file(key: str, fallback: str) -> Path:
        return _resolve_sample_file(sample_dir, files.get(key, fallback), key)

    moving_path = resolve_file("partA_pc", "partA_pc.npy")
    fixed_path = resolve_file("base_partB_pc", "base_partB_pc.npy")
    base_image_path, assemble_image_path, manual_type = _select_saved_images(meta, files, sample_dir)
    moving = _load_point_cloud(moving_path, "Moving")
    fixed = _load_point_cloud(fixed_path, "Fixed")
    images = [[_load_rgb_image(base_image_path), _load_rgb_image(assemble_image_path)]]

    model_state, cache_hit = _load_model(request["config_path"], request["checkpoint_path"])
    model = model_state["model"]
    device = model_state["device"]
    source_tensor = torch.from_numpy(moving.T).float().to(device)
    fixed_tensor = torch.from_numpy(fixed.T).float().to(device)
    category = str(meta.get("category", "unknown"))
    batch = {
        "src_pc": [source_tensor],
        "tgt_pc": [fixed_tensor],
        "imgs": images,
        "category": [category],
        "asset": [str(meta.get("asset_name", sample_dir.name))],
        "pre_pose": None,
    }
    with torch.no_grad():
        generated = model.generate(batch)
    if not isinstance(generated, torch.Tensor) or generated.ndim != 2:
        raise RuntimeError(
            "The model must return a pose tensor with shape [batch, 9]; "
            f"got {type(generated).__name__} with shape {getattr(generated, 'shape', None)}."
        )
    if generated.shape[0] < 1 or generated.shape[1] != 9:
        raise RuntimeError(
            "The model must return exactly 9 pose values per sample; "
            f"got shape {tuple(generated.shape)}."
        )
    pose = generated[0].float()
    valid = bool(torch.isfinite(pose).all().item() and pose[0].item() != -100.0)
    if not valid:
        raise RuntimeError("The model returned an invalid pose prediction.")

    pred_translation = pose[:3]
    pred_rotation_6d = pose[3:9]
    pred_rotation = _rotation_6d_to_matrix(torch, pred_rotation_6d.unsqueeze(0))[0]
    predicted_moving = (pred_rotation.T @ source_tensor + pred_translation.reshape(3, 1)).T

    ground_truth = meta.get("ground_truth_pose")
    metrics: dict[str, Any] = {}
    ground_truth_moving = None
    if ground_truth:
        gt_translation = torch.tensor(ground_truth["translation"], dtype=torch.float32, device=device)
        gt_rotation_6d = torch.tensor(ground_truth["rotation_6d"], dtype=torch.float32, device=device)
        gt_rotation = _rotation_6d_to_matrix(torch, gt_rotation_6d.unsqueeze(0))[0]
        ground_truth_moving, metrics = _compute_pose_metrics(
            torch,
            source_tensor,
            predicted_moving,
            pred_translation,
            pred_rotation,
            gt_translation,
            gt_rotation,
        )

    result_dir = sample_dir / "inference_results" / f"run_{request.get('job_id', 'latest')}"
    temporary_result_dir = result_dir.with_name(result_dir.name + ".partial")
    if result_dir.exists():
        raise FileExistsError(f"Inference result directory already exists: {result_dir}")
    shutil.rmtree(temporary_result_dir, ignore_errors=True)
    temporary_result_dir.mkdir(parents=True, exist_ok=False)
    try:
        _atomic_save_npy(
            temporary_result_dir / "predicted_partA_pc.npy",
            predicted_moving.detach().cpu().numpy().astype(np.float32),
        )
        if ground_truth_moving is not None:
            _atomic_save_npy(
                temporary_result_dir / "ground_truth_partA_pc.npy",
                ground_truth_moving.detach().cpu().numpy().astype(np.float32),
            )
        prediction_payload = {
            "asset_name": meta.get("asset_name", sample_dir.name),
            "category": category,
            "manual_type": manual_type,
            "valid": valid,
            "model_cache_hit": cache_hit,
            "checkpoint_path": model_state["checkpoint_path"],
            "config_path": model_state["config_path"],
            "pred_pose_9": pose.detach().cpu().tolist(),
            "pred_translation": pred_translation.detach().cpu().tolist(),
            "pred_rotation_6d": pred_rotation_6d.detach().cpu().tolist(),
            "pred_rotation_matrix": pred_rotation.detach().cpu().tolist(),
            "metrics": metrics,
            "files": {
                "predicted_partA_pc": str(result_dir / "predicted_partA_pc.npy"),
                "ground_truth_partA_pc": (
                    str(result_dir / "ground_truth_partA_pc.npy")
                    if ground_truth_moving is not None
                    else None
                ),
            },
        }
        _atomic_write_json(temporary_result_dir / "prediction.json", prediction_payload)
        temporary_result_dir.replace(result_dir)
    except Exception:
        shutil.rmtree(temporary_result_dir, ignore_errors=True)
        raise
    prediction_payload["predicted_point_cloud"] = predicted_moving.detach().cpu().tolist()
    prediction_payload["fixed_point_cloud"] = fixed.tolist()
    if ground_truth_moving is not None:
        prediction_payload["ground_truth_point_cloud"] = ground_truth_moving.detach().cpu().tolist()
    return prediction_payload


def _load_model_only(request: dict[str, Any]) -> dict[str, Any]:
    model_state, cache_hit = _load_model(request["config_path"], request["checkpoint_path"])
    return {
        "checkpoint_path": model_state["checkpoint_path"],
        "config_path": model_state["config_path"],
        "model_cache_hit": cache_hit,
        "missing_keys": model_state["missing_keys"],
        "unexpected_keys": model_state["unexpected_keys"],
    }


def main() -> None:
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
            with redirect_stdout(sys.stderr):
                if request.get("action") == "load_model":
                    result = _load_model_only(request)
                else:
                    result = _infer(request)
            response = {"ok": True, "result": result}
        except Exception as exc:
            response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(response, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
