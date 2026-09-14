"""Rotation utilities used by the AssemLM 2.0 pose path."""

import numpy as np
import torch
import torch.nn.functional as F


def bgs(d6s: torch.Tensor) -> torch.Tensor:
    """Convert a batch of 6D rotations shaped ``[B, 3, 2]`` to matrices."""
    batch_size = d6s.shape[0]
    first = F.normalize(d6s[:, :, 0], p=2, dim=1)
    second_raw = d6s[:, :, 1]
    second = F.normalize(
        second_raw
        - torch.bmm(first.view(batch_size, 1, -1), second_raw.view(batch_size, -1, 1)).view(
            batch_size, 1
        )
        * first,
        p=2,
        dim=1,
    )
    third = torch.cross(first, second, dim=1)
    return torch.stack([first, second, third], dim=1).permute(0, 2, 1)


def bgdR(ground_truth: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
    """Return geodesic rotation error in radians for two matrix batches."""
    relative = torch.bmm(ground_truth.transpose(1, 2), prediction)
    trace = relative.diagonal(dim1=1, dim2=2).sum(dim=1)
    cosine = ((trace - 1.0) * 0.5).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    return torch.acos(cosine)


def get_6d_rot_loss(ground_truth: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
    """Return per-sample 6D rotation error in degrees."""
    predicted_matrices = bgs(prediction.reshape(-1, 2, 3).permute(0, 2, 1))
    ground_truth_matrices = bgs(ground_truth.reshape(-1, 2, 3).permute(0, 2, 1))
    return bgdR(ground_truth_matrices, predicted_matrices) * 180.0 / np.pi
