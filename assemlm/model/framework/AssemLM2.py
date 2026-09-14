"""AssemLM 2.0 continuous-pose framework."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from assemlm.model.framework.base_framework import baseframework
from assemlm.model.modules.pvlm import get_pvlm_model
from assemlm.model.tools import FRAMEWORK_REGISTRY


class _MLPResNetBlock(nn.Module):
    """Residual MLP block used by the pose head."""

    def __init__(self, dim: int):
        super().__init__()
        self.ffn = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.ReLU())

    def forward(self, x):
        return x + self.ffn(x)


class _MLPResNet(nn.Module):
    def __init__(self, num_blocks: int, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.blocks = nn.ModuleList(
            [_MLPResNetBlock(hidden_dim) for _ in range(num_blocks)]
        )
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.relu(self.fc1(self.layer_norm1(x)))
        for block in self.blocks:
            x = block(x)
        return self.fc2(self.layer_norm2(x))


class _PoseRegressionHead(nn.Module):
    """Predict one continuous 9D pose from a pose-query hidden state."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_blocks: int):
        super().__init__()
        self.output_dim = output_dim
        self.model = _MLPResNet(num_blocks, input_dim, hidden_dim, output_dim)

    def forward(self, hidden_state):
        batch_size, chunk_len, hidden_dim = hidden_state.shape
        output = self.model(hidden_state.reshape(batch_size * chunk_len, hidden_dim))
        return output.view(batch_size, chunk_len, self.output_dim)


@FRAMEWORK_REGISTRY.register("AssemLM2")
class AssemLM2(baseframework):
    """The fixed AssemLM 2.0 model used by the release scripts."""

    def __init__(self, config: Optional[dict] = None, **kwargs):
        super().__init__()
        self.config = config
        self.pvlm_interface = get_pvlm_model(config=config)

        pose_cfg = config.framework.get("pose_regression", {}) or {}
        if not self._cfg_bool(self._cfg_get(pose_cfg, "enabled", False)):
            raise ValueError("AssemLM 2.0 requires continuous pose regression.")
        self.pose_regression_query_token = str(
            self._cfg_get(
                pose_cfg,
                "query_token",
                getattr(self.pvlm_interface, "pose_query_token", "<pose_query>"),
            )
        )
        self.pose_regression_query_mode = str(
            self._cfg_get(pose_cfg, "query_mode", "pose_query_token")
        ).strip().lower()
        if self.pose_regression_query_mode not in {
            "pose_query",
            "pose_query_token",
            "query_token",
        }:
            raise ValueError("AssemLM 2.0 requires query_mode=pose_query_token.")
        self.pose_regression_enabled = True
        self.pose_regression_rot_loss_unit = str(
            self._cfg_get(pose_cfg, "rot_loss_unit", "radian")
        ).strip().lower()
        self.pose_regression_trans_loss_weight = float(
            self._cfg_get(pose_cfg, "trans_loss_weight", 1.0)
        )
        self.pose_regression_rot_loss_weight = float(
            self._cfg_get(pose_cfg, "rot_loss_weight", 1.0)
        )

        vlm_hidden_size = int(self.pvlm_interface.vlm.config.hidden_size)
        output_dim = int(self._cfg_get(pose_cfg, "output_dim", 9))
        if output_dim != 9:
            raise ValueError("AssemLM 2.0 pose regression output_dim must be 9.")
        self.pose_regression_head = _PoseRegressionHead(
            input_dim=int(self._cfg_get(pose_cfg, "input_dim", vlm_hidden_size)),
            hidden_dim=int(self._cfg_get(pose_cfg, "hidden_dim", vlm_hidden_size * 2)),
            output_dim=output_dim,
            num_blocks=int(self._cfg_get(pose_cfg, "num_blocks", 2)),
        )

    @staticmethod
    def _cfg_get(container, key, default=None):
        if container is None:
            return default
        if hasattr(container, "get"):
            return container.get(key, default)
        return getattr(container, key, default)

    @staticmethod
    def _cfg_bool(value, default=False):
        if value is None:
            return bool(default)
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "on"}
        return bool(value)

    def _prepare_inputs(self, batch_data):
        # Released datasets are not point-count uniform (e.g. PartNet ships
        # partA with 1000 points and base_partB with 1024), while the point
        # encoder expects a shared count. Mirror the dataloader resampling so
        # the per-sample stack cannot fail with [3, 1000] vs [3, 1024].
        from assemlm.utils.point_clouds import resample_point_cloud_pair

        point_clouds = [
            list(resample_point_cloud_pair(src, tgt))
            for src, tgt in zip(batch_data["src_pc"], batch_data["tgt_pc"])
        ]
        instructions = [
            f"Assemble the {category} object {self.pose_regression_query_token}"
            for category in batch_data["category"]
        ]
        return {
            "images": batch_data["imgs"],
            "point_clouds": point_clouds,
            "instructions": instructions,
        }

    @staticmethod
    def _rotation_6d_to_matrix(rot_6d: torch.Tensor) -> torch.Tensor:
        rot_6d = rot_6d.reshape(-1, 2, 3)
        first, second = rot_6d[:, 0], rot_6d[:, 1]
        first = F.normalize(first, p=2, dim=-1, eps=1e-6)
        second = F.normalize(
            second - (first * second).sum(dim=-1, keepdim=True) * first,
            p=2,
            dim=-1,
            eps=1e-6,
        )
        third = torch.cross(first, second, dim=-1)
        return torch.stack((first, second, third), dim=-1)

    def _geodesic_rotation_loss(self, predicted, target):
        predicted_matrix = self._rotation_6d_to_matrix(predicted.float())
        target_matrix = self._rotation_6d_to_matrix(target.float())
        relative = target_matrix.transpose(1, 2) @ predicted_matrix
        trace = relative.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        cosine = ((trace - 1.0) * 0.5).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        angle = torch.acos(cosine)
        if self.pose_regression_rot_loss_unit in {"degree", "degrees", "deg"}:
            angle = angle * (180.0 / math.pi)
        elif self.pose_regression_rot_loss_unit not in {"radian", "radians", "rad"}:
            raise ValueError("rot_loss_unit must be radian or degree.")
        return angle.mean()

    def _gather_pose_query_hidden(self, hidden_state, attention_mask, input_ids):
        token_id = self.pvlm_interface.pose_query_token_id
        query_mask = input_ids.to(hidden_state.device) == token_id
        if attention_mask is not None:
            valid_mask = attention_mask.to(hidden_state.device).bool()
            if valid_mask.ndim == 4:
                valid_mask = torch.diagonal(valid_mask[:, 0], dim1=1, dim2=2)
            query_mask &= valid_mask
        counts = query_mask.sum(dim=1)
        if bool((counts < 1).any().item()):
            raise RuntimeError("Every sample must contain the pose query token.")
        positions = torch.arange(hidden_state.size(1), device=hidden_state.device)
        positions = positions.unsqueeze(0).expand(hidden_state.size(0), -1)
        query_index = positions.masked_fill(~query_mask, -1).max(dim=1).values
        batch_index = torch.arange(hidden_state.size(0), device=hidden_state.device)
        return hidden_state[batch_index, query_index].unsqueeze(1).float()

    def _predict_pose(self, pvlm_outputs, qwen_inputs):
        query = self._gather_pose_query_hidden(
            pvlm_outputs.last_hidden_state,
            qwen_inputs.get("attention_mask"),
            qwen_inputs["input_ids"],
        )
        parameter = next(self.pose_regression_head.parameters())
        query = query.to(device=parameter.device, dtype=parameter.dtype)
        if query.is_cuda:
            with torch.autocast("cuda", enabled=False):
                pose = self.pose_regression_head(query)
        else:
            pose = self.pose_regression_head(query)
        return pose.squeeze(1)

    def _pose_loss(self, predicted, target):
        predicted = predicted.float()
        target = target.to(device=predicted.device, dtype=torch.float32)
        translation_loss = F.l1_loss(predicted[:, :3], target[:, :3])
        rotation_loss = self._geodesic_rotation_loss(predicted[:, 3:9], target[:, 3:9])
        total = (
            self.pose_regression_trans_loss_weight * translation_loss
            + self.pose_regression_rot_loss_weight * rotation_loss
        )
        return total, translation_loss, rotation_loss

    def _run_pvlm(self, batch_data):
        prepared = self._prepare_inputs(batch_data)
        qwen_inputs = self.pvlm_interface.build_qwenvl_inputs(**prepared)
        outputs = self.pvlm_interface(
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
            **qwen_inputs,
        )
        return outputs, qwen_inputs

    def forward(self, batch_data: dict = None, **kwargs):
        if batch_data is None or "pre_pose" not in batch_data:
            raise ValueError("The batch must contain src_pc, tgt_pc, imgs, category, and pre_pose.")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs, qwen_inputs = self._run_pvlm(batch_data)
        predicted = self._predict_pose(outputs, qwen_inputs)
        total, translation, rotation = self._pose_loss(predicted, batch_data["pre_pose"])
        return {
            "pose_loss": total,
            "pose_regression_loss": total.detach(),
            "translation_l1_loss": translation.detach(),
            "rotation_geodesic_loss": rotation.detach(),
            "pred_pose_mean_abs": predicted.detach().abs().mean(),
            "per_pose_loss": None,
        }

    @torch.no_grad()
    def generate(self, batch_data: dict = None, **kwargs):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs, qwen_inputs = self._run_pvlm(batch_data)
        return self._predict_pose(outputs, qwen_inputs).detach()
