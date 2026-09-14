"""Vector-Neuron layers used by the AssemLM 1.1 patch encoder."""

from __future__ import annotations

import torch
import torch.nn as nn


EPS = 1e-6


def normalize_vn_norm_type(norm_type):
    normalized = str(norm_type).strip().lower().replace("_", "").replace("-", "")
    if normalized not in {"rms", "rmsnorm", "vnrmsnorm"}:
        raise ValueError("AssemLM 1.1 requires VN normalization type rmsnorm.")
    return "rmsnorm"


class VNRMSNorm(nn.Module):
    """SO(3)-equivariant RMSNorm for [B, C, 3, ...] vector features."""

    def __init__(self, num_features, eps=EPS, elementwise_affine=True):
        super().__init__()
        if num_features <= 0 or eps <= 0:
            raise ValueError("num_features and eps must be positive.")
        self.num_features = int(num_features)
        self.eps = float(eps)
        self.elementwise_affine = bool(elementwise_affine)
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.num_features))
        else:
            self.register_parameter("weight", None)

    def forward(self, x):
        if x.ndim < 3 or x.size(1) != self.num_features or x.size(2) != 3:
            raise ValueError(
                f"VNRMSNorm expects [B, {self.num_features}, 3, ...], got {tuple(x.shape)}."
            )
        input_dtype = x.dtype
        values = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        inverse_rms = torch.rsqrt(values.square().mean(dim=(1, 2), keepdim=True) + self.eps)
        output = values * inverse_rms
        if self.weight is not None:
            shape = (1, self.num_features, 1) + (1,) * (x.ndim - 3)
            output = output * self.weight.to(dtype=values.dtype).view(shape)
        return output.to(dtype=input_dtype)


def build_vn_norm(num_features, dim, norm_type="rmsnorm"):
    del dim
    normalize_vn_norm_type(norm_type)
    return VNRMSNorm(num_features)


def mean_pool(x, dim=-1, keepdim=False):
    return x.mean(dim=dim, keepdim=keepdim)


class VNLinearLeakyReLU(nn.Module):
    """Vector-channel linear projection, RMS normalization, and equivariant ReLU."""

    def __init__(
        self,
        in_channels,
        out_channels,
        dim=5,
        share_nonlinearity=False,
        negative_slope=0.2,
        norm_type="rmsnorm",
    ):
        super().__init__()
        del dim
        self.negative_slope = negative_slope
        self.map_to_feat = nn.Linear(in_channels, out_channels, bias=False)
        self.batchnorm = build_vn_norm(out_channels, dim=5, norm_type=norm_type)
        self.map_to_dir = nn.Linear(
            in_channels,
            1 if share_nonlinearity else out_channels,
            bias=False,
        )

    def forward(self, x):
        projected = self.map_to_feat(x.transpose(1, -1)).transpose(1, -1)
        projected = self.batchnorm(projected)
        direction = self.map_to_dir(x.transpose(1, -1)).transpose(1, -1)
        dot_product = (projected * direction).sum(2, keepdims=True)
        mask = (dot_product >= 0).to(dtype=projected.dtype)
        direction_norm_sq = (direction * direction).sum(2, keepdims=True)
        return self.negative_slope * projected + (1 - self.negative_slope) * (
            mask * projected
            + (1 - mask)
            * (projected - (dot_product / (direction_norm_sq + EPS)) * direction)
        )


class VNStdFeature(nn.Module):
    """Produce the invariant feature and canonical frame used by the projector."""

    def __init__(
        self,
        in_channels,
        dim=4,
        normalize_frame=False,
        share_nonlinearity=False,
        negative_slope=0.2,
        norm_type="rmsnorm",
    ):
        super().__init__()
        self.dim = dim
        self.normalize_frame = normalize_frame
        self.vn1 = VNLinearLeakyReLU(
            in_channels,
            in_channels // 2,
            dim=dim,
            share_nonlinearity=share_nonlinearity,
            negative_slope=negative_slope,
            norm_type=norm_type,
        )
        self.vn2 = VNLinearLeakyReLU(
            in_channels // 2,
            in_channels // 4,
            dim=dim,
            share_nonlinearity=share_nonlinearity,
            negative_slope=negative_slope,
            norm_type=norm_type,
        )
        self.vn_lin = nn.Linear(in_channels // 4, 2 if normalize_frame else 3, bias=False)

    def forward(self, x):
        frame = self.vn1(x)
        frame = self.vn2(frame)
        frame = self.vn_lin(frame.transpose(1, -1)).transpose(1, -1)
        if self.normalize_frame:
            first = frame[:, 0, :]
            first = first / (torch.sqrt((first * first).sum(1, keepdims=True)) + EPS)
            second = frame[:, 1, :]
            second = second - (second * first).sum(1, keepdims=True) * first
            second = second / (torch.sqrt((second * second).sum(1, keepdims=True)) + EPS)
            third = torch.cross(first, second, dim=-1)
            frame = torch.stack([first, second, third], dim=1).transpose(1, 2)
        else:
            frame = frame.transpose(1, 2)

        if self.dim == 4:
            invariant = torch.einsum("bijm,bjkm->bikm", x, frame)
        elif self.dim == 3:
            invariant = torch.einsum("bij,bjk->bik", x, frame)
        else:
            raise ValueError(f"VNStdFeature only supports dim=3 or dim=4, got {self.dim}.")
        return invariant, frame
