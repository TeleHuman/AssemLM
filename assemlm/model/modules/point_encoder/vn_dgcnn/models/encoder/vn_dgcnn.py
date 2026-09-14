"""VN-DGCNN patch encoder used by AssemLM 1.1."""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from .vn_layers import VNLinearLeakyReLU, VNStdFeature, mean_pool, normalize_vn_norm_type


LOGGER = logging.getLogger(__name__)


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    return pairwise_distance.topk(k=k, dim=-1)[1]


def get_graph_feature(x, k=20):
    batch_size = x.size(0)
    num_points = x.size(3)
    x = x.view(batch_size, -1, num_points)
    idx = knn(x, k=k)
    device = x.device
    idx_base = torch.arange(batch_size, device=device).view(-1, 1, 1) * num_points
    idx = (idx + idx_base).view(-1)
    _, num_dims, _ = x.size()
    num_dims //= 3
    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims, 3)
    x = x.view(batch_size, num_points, 1, num_dims, 3).expand(-1, -1, k, -1, -1)
    return torch.cat((feature - x, x), dim=3).permute(0, 3, 4, 1, 2).contiguous()


def index_points(points, idx):
    batch_size = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(batch_size, device=points.device).view(view_shape)
    batch_indices = batch_indices.repeat(repeat_shape)
    return points[batch_indices, idx]


def farthest_point_sample_indices(xyz, num_samples):
    batch_size, num_points, _ = xyz.shape
    if num_samples > num_points:
        raise ValueError(f"num_samples={num_samples} cannot exceed num_points={num_points}.")
    centroids = torch.zeros(batch_size, num_samples, dtype=torch.long, device=xyz.device)
    distance = torch.full((batch_size, num_points), float("inf"), device=xyz.device)
    batch_indices = torch.arange(batch_size, dtype=torch.long, device=xyz.device)
    center = xyz.mean(dim=1, keepdim=True)
    farthest = torch.sum((xyz - center) ** 2, dim=-1).max(dim=-1)[1]
    for index in range(num_samples):
        centroids[:, index] = farthest
        centroid = xyz[batch_indices, farthest, :].view(batch_size, 1, 3)
        distances = torch.sum((xyz - centroid) ** 2, dim=-1)
        distance = torch.minimum(distance, distances)
        farthest = torch.max(distance, dim=-1)[1]
    return centroids


def knn_point_indices(num_neighbors, xyz, centers):
    if num_neighbors > xyz.size(1):
        raise ValueError(
            f"num_neighbors={num_neighbors} cannot exceed num_points={xyz.size(1)}."
        )
    return torch.cdist(centers, xyz).topk(
        k=num_neighbors, dim=-1, largest=False
    )[1]


class VN_DGCNN_Patch(nn.Module):
    """Extract global and 128 local VN-DGCNN features from a point cloud."""

    def __init__(
        self,
        feat_dim,
        pooling="mean",
        norm_type="rmsnorm",
        num_patches=256,
        patch_size=4,
        patch_method="fps_knn",
        patch_sort_method="none",
        normalize_frame=False,
    ):
        super().__init__()
        if str(pooling).lower() != "mean":
            raise ValueError("AssemLM 1.1 requires mean VN-DGCNN pooling.")
        if str(patch_method).lower() != "fps_knn":
            raise ValueError("AssemLM 1.1 requires fps_knn patching.")
        if str(patch_sort_method).lower() != "none":
            raise ValueError("AssemLM 1.1 requires unsorted patch tokens.")
        self.n_knn = 20
        self.feat_dim = int(feat_dim)
        self.num_patches = int(num_patches)
        self.patch_size = int(patch_size)
        self.patch_method = "fps_knn"
        self.patch_sort_method = "none"
        self.normalize_frame = bool(normalize_frame)
        if self.num_patches <= 0 or self.patch_size <= 0:
            raise ValueError("num_patches and patch_size must be positive.")
        self.norm_type = normalize_vn_norm_type(norm_type)

        LOGGER.info(
            "Initialized VN-DGCNN patch encoder: feat_dim=%s, norm=%s, "
            "patches=%s, patch_size=%s, normalize_frame=%s",
            self.feat_dim,
            self.norm_type,
            self.num_patches,
            self.patch_size,
            self.normalize_frame,
        )

        self.conv1 = VNLinearLeakyReLU(2, 64 // 3, norm_type=self.norm_type)
        self.conv2 = VNLinearLeakyReLU(64 // 3, 64 // 3, norm_type=self.norm_type)
        self.conv3 = VNLinearLeakyReLU(64 // 3 * 2, 64 // 3, norm_type=self.norm_type)
        self.conv4 = VNLinearLeakyReLU(64 // 3, 64 // 3, norm_type=self.norm_type)
        self.conv5 = VNLinearLeakyReLU(64 // 3 * 2, 64 // 3, norm_type=self.norm_type)
        self.conv6 = VNLinearLeakyReLU(
            64 // 3 * 3,
            feat_dim,
            dim=4,
            share_nonlinearity=True,
            norm_type=self.norm_type,
        )
        self.VnInv = VNStdFeature(
            feat_dim,
            dim=3,
            normalize_frame=self.normalize_frame,
            norm_type=self.norm_type,
        )
        self.linear0 = nn.Linear(3, feat_dim)

    def _build_patch_tokens(self, local_equiv, xyz):
        batch_size, feat_dim, _, num_points = local_equiv.shape
        if self.num_patches > num_points:
            raise ValueError(
                f"num_patches={self.num_patches} cannot exceed input num_points={num_points}."
            )
        center_idx = farthest_point_sample_indices(xyz, self.num_patches)
        patch_centers = index_points(xyz, center_idx)
        group_idx = knn_point_indices(self.patch_size, xyz, patch_centers)
        point_features = local_equiv.permute(0, 3, 1, 2).contiguous()
        grouped_features = index_points(point_features, group_idx)
        patch_equiv = grouped_features.mean(dim=2)
        return patch_equiv, patch_centers

    def forward(self, x):
        if not torch.is_tensor(x) or x.ndim != 3:
            raise ValueError(
                "VN_DGCNN_Patch expects point clouds shaped [B, C, N], "
                f"got {type(x).__name__} with shape {getattr(x, 'shape', None)}."
            )
        if not x.is_floating_point():
            raise TypeError(
                f"VN_DGCNN_Patch expects a floating-point tensor, got {x.dtype}."
            )
        if x.size(1) < 3 or x.size(1) % 3 != 0:
            raise ValueError(
                "VN_DGCNN_Patch expects C to be a positive multiple of 3 "
                f"(at least xyz), got C={x.size(1)}."
            )
        if x.size(-1) < self.n_knn:
            raise ValueError(
                f"VN_DGCNN_Patch requires at least {self.n_knn} points for k-NN, "
                f"got N={x.size(-1)}."
            )
        if not bool(torch.isfinite(x).all().item()):
            raise ValueError("VN_DGCNN_Patch received a point cloud containing NaN or Inf.")
        xyz = x[:, :3].transpose(1, 2).contiguous()
        x = x.unsqueeze(1)
        x = get_graph_feature(x, k=self.n_knn)
        x = self.conv1(x)
        x = self.conv2(x)
        x1 = mean_pool(x)
        x = get_graph_feature(x1, k=self.n_knn)
        x = self.conv3(x)
        x = self.conv4(x)
        x2 = mean_pool(x)
        x = get_graph_feature(x2, k=self.n_knn)
        x = self.conv5(x)
        x3 = mean_pool(x)
        local_equiv = self.conv6(torch.cat((x1, x2, x3), dim=1))
        patch_equiv, patch_centers = self._build_patch_tokens(local_equiv, xyz)
        global_equiv = local_equiv.mean(dim=-1)
        global_inv_before, z0 = self.VnInv(global_equiv)
        global_inv = self.linear0(global_inv_before)
        return global_equiv, global_inv, z0, patch_equiv, patch_centers, local_equiv
