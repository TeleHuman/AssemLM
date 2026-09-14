"""Point-cloud resampling helpers for the AssemLM inference paths.

The training/evaluation dataloader
(:mod:`assemlm.dataloader.hdf5_datasets`) resamples every part to a fixed point
count before batching, but the released HDF5 files are not always uniform:
PartNet ships ``partA-pc`` with 1000 points while ``base_partB-pc`` holds 1024.
Inference paths that skip that resampling therefore hit::

    RuntimeError: stack expects each tensor to be equal size,
    but got [3, 1000] at entry 0 and [3, 1024] at entry 1

These helpers apply the same fixed-count resampling so that any inference entry
point (GUI worker, legacy interface, HTTP API) receives uniform point clouds.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple

import numpy as np

#: Point count used by the dataloader (``_resample_point_cloud`` default).
DEFAULT_NUM_POINTS = 1024


def _is_torch_tensor(value: Any) -> bool:
    return value.__class__.__module__.split(".")[0] == "torch" and hasattr(value, "index_select")


def _evenly_spaced_indices(point_count: int, num_points: int) -> np.ndarray:
    """Deterministic index draw used when no RNG is supplied."""
    if point_count <= 0:
        raise ValueError("Point cloud must contain at least one point.")
    if point_count == num_points:
        return np.arange(point_count, dtype=np.int64)
    return np.rint(np.linspace(0.0, point_count - 1, num_points)).astype(np.int64)


def _validate(point_cloud: Any, axis: int, label: str) -> Tuple[int, int]:
    shape = tuple(int(dim) for dim in point_cloud.shape)
    if len(shape) != 2:
        raise ValueError(f"{label} must be a 2D array, got shape {shape}.")
    if shape[axis] <= 0:
        raise ValueError(f"{label} contains no points (shape {shape}).")
    if shape[1 - axis] != 3:
        raise ValueError(f"{label} must have 3 coordinates, got shape {shape}.")
    return shape, axis


def resample_point_cloud(
    point_cloud: Any,
    num_points: int = DEFAULT_NUM_POINTS,
    rng: Optional[np.random.Generator] = None,
) -> Any:
    """Resample one point cloud to ``num_points`` points.

    Accepts ``(N, 3)`` / ``(3, N)`` numpy arrays and torch tensors, returning the
    same container type (torch tensors keep their device and dtype).  Sampling is
    deterministic (evenly spaced indices) unless ``rng`` -- a
    :class:`numpy.random.Generator` -- is supplied, in which case the dataloader's
    random choice behaviour is reproduced.
    """
    if num_points is None or int(num_points) <= 0:
        num_points = DEFAULT_NUM_POINTS
    num_points = int(num_points)

    if _is_torch_tensor(point_cloud):
        import torch

        axis = 0 if (point_cloud.shape[-1] == 3 and point_cloud.shape[0] != 3) else 1
        shape, axis = _validate(point_cloud, axis, "point cloud")
        if shape[axis] == num_points:
            return point_cloud
        if rng is None:
            indices = _evenly_spaced_indices(shape[axis], num_points)
        else:
            indices = rng.choice(shape[axis], num_points, replace=shape[axis] < num_points)
        index = torch.as_tensor(np.asarray(indices), dtype=torch.long, device=point_cloud.device)
        return point_cloud.index_select(axis, index)

    array = np.asarray(point_cloud)
    axis = 0 if (array.ndim == 2 and array.shape[-1] == 3 and array.shape[0] != 3) else 1
    shape, axis = _validate(array, axis, "point cloud")
    if shape[axis] == num_points:
        return array
    if rng is None:
        indices = _evenly_spaced_indices(shape[axis], num_points)
    else:
        indices = rng.choice(shape[axis], num_points, replace=shape[axis] < num_points)
    return np.take(array, indices, axis=axis)


def resample_point_cloud_pair(
    first: Any,
    second: Any,
    num_points: int = DEFAULT_NUM_POINTS,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[Any, Any]:
    """Resample the ``[partA, base_partB]`` pair to a shared point count."""
    return (
        resample_point_cloud(first, num_points=num_points, rng=rng),
        resample_point_cloud(second, num_points=num_points, rng=rng),
    )


def resample_point_cloud_sequence(
    point_clouds: Sequence[Any],
    num_points: int = DEFAULT_NUM_POINTS,
    rng: Optional[np.random.Generator] = None,
) -> list:
    """Resample every entry of a per-sample point-cloud sequence."""
    return [resample_point_cloud(pc, num_points=num_points, rng=rng) for pc in point_clouds]
