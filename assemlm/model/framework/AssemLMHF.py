from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import re
import torch
from omegaconf import OmegaConf
from transformers import PreTrainedModel, PretrainedConfig

from assemlm.model.tools import FRAMEWORK_REGISTRY
from assemlm.training.trainer_utils import initialize_overwatch


logger = initialize_overwatch(__name__)


class AssemLMHFConfig(PretrainedConfig):
    """HF config for AssemLM.

    This keeps the legacy nested config layout so existing modules can still
    read `framework`, `datasets`, and `trainer` blocks.
    """

    model_type = "assemlm_hf"

    def __init__(
        self,
        framework: Optional[Dict[str, Any]] = None,
        datasets: Optional[Dict[str, Any]] = None,
        trainer: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.framework = framework or {}
        self.datasets = datasets or {}
        self.trainer = trainer or {}

    @classmethod
    def from_legacy_config(cls, config: Any) -> "AssemLMHFConfig":
        if hasattr(config, "to_dict"):
            payload = config.to_dict()
        elif isinstance(config, dict):
            payload = copy.deepcopy(config)
        else:
            payload = OmegaConf.to_container(OmegaConf.create(config), resolve=True)
        return cls(**payload)


@FRAMEWORK_REGISTRY.register("AssemLMHF")
class AssemLMHF(PreTrainedModel):
    """HF-style AssemLM wrapper.

    This class keeps the legacy modules intact, but exposes standard
    `save_pretrained()` and `from_pretrained()` through the Transformers base
    class.
    """

    config_class = AssemLMHFConfig
    base_model_prefix = "assemlm"

    def __init__(self, config: AssemLMHFConfig, **kwargs):
        super().__init__(config)

        self.config = config
        self.legacy_config = OmegaConf.create(config.to_dict())
        config_root = getattr(config, "_name_or_path", None) or getattr(config, "name_or_path", None)
        self._normalize_legacy_paths(self.legacy_config, config_root=config_root)

        from assemlm.model.modules.pose_head import get_action_model
        from assemlm.model.modules.assemlm import get_assemlm_model

        self.assemlm_interface = get_assemlm_model(config=self.legacy_config)
        self.action_model = get_action_model(config=self.legacy_config)

    @staticmethod
    def _normalize_legacy_paths(legacy_config, config_root: Optional[str] = None):
        vlm_cfg = legacy_config.framework.get("vlm", {})
        base_vlm = vlm_cfg.get("base_vlm", None)
        if not base_vlm:
            return

        base_vlm_path = Path(str(base_vlm))
        if base_vlm_path.is_absolute():
            legacy_config.framework.vlm.base_vlm = str(base_vlm_path)
            legacy_config.framework.vlm.local_files_only = True
            return

        repo_root = config_root or getattr(legacy_config, "_name_or_path", None) or getattr(legacy_config, "name_or_path", None)
        if repo_root:
            candidate = Path(str(repo_root)) / base_vlm_path
            if candidate.exists():
                legacy_config.framework.vlm.base_vlm = str(candidate)
                legacy_config.framework.vlm.local_files_only = True

    def forward(self, *args, **kwargs):
        return self.assemlm_interface(*args, **kwargs)

    @staticmethod
    def _filter_non_vlm_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Keep only non-VLM parameters for the top-level HF checkpoint."""
        filtered_state_dict = {}
        for key, value in state_dict.items():
            if ".vlm." in key:
                continue
            filtered_state_dict[key] = value
        return filtered_state_dict

    @staticmethod
    def _save_state_dict(state_dict: Dict[str, torch.Tensor], save_path: Path, safe_serialization: bool = False) -> None:
        """Save a state dict using either safetensors or PyTorch format."""
        if safe_serialization:
            from safetensors.torch import save_file

            save_file(state_dict, str(save_path / "model.safetensors"))
            return

        torch.save(state_dict, save_path / "pytorch_model.bin")

    def generate(self, batch_data: dict = None, **kwargs) -> List[Any]:

        input_ids = batch_data["input_ids"]
        attention_mask = batch_data["attention_mask"]
        point_clouds = batch_data["point_clouds"]
        pixel_values = batch_data["pixel_values"]
        image_grid_thw = batch_data["image_grid_thw"]
        gen_input_ids = input_ids
        gen_attention_mask = attention_mask

        generated_ids_batch = self.assemlm_interface.generate(
            input_ids=gen_input_ids,
            attention_mask=gen_attention_mask,
            point_clouds=point_clouds,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            max_new_tokens=128,
            do_sample=False,
            temperature=0,
        )
        batch_generated_poses = []
        for b_idx in range(len(generated_ids_batch)):
            generated_ids = generated_ids_batch[b_idx]
            generated_text = self.assemlm_interface.tokenizer.decode(
                generated_ids, skip_special_tokens=False
            )
            final_pose = torch.full((9,), -100.0, device=self.assemlm_interface.vlm.device, dtype=torch.float32)

            first_segment = generated_text.split("<|im_end|>")[0]
            action_tokens_matched = re.findall(r"<assemble_pose_\d+>", first_segment)
            num_tokens = len(action_tokens_matched)
            generate_pose = self.action_model.vlmtoken2action(generated_text)
            if generate_pose is not None:
                if not isinstance(generate_pose, torch.Tensor):
                    generate_pose = torch.from_numpy(generate_pose).to(device=self.assemlm_interface.vlm.device, dtype=torch.float32)
                else:
                    generate_pose = generate_pose.to(device=self.assemlm_interface.vlm.device, dtype=torch.float32)

                pose_flat = generate_pose.flatten()
                if pose_flat.shape[0] == 9:
                    final_pose = pose_flat
                    valid_prediction = True
                elif pose_flat.shape[0] > 9 and num_tokens == 9:
                    final_pose = pose_flat[:9]
                    valid_prediction = True

            batch_generated_poses.append(final_pose)
        batch_generated_poses = torch.stack(batch_generated_poses, dim=0)
        return batch_generated_poses

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        save_path = Path(pretrained_model_name_or_path)

        config = cls.config_class.from_pretrained(str(save_path))
        config._name_or_path = str(save_path)

        model = cls(config, **kwargs)

        state_dict = {}
        safetensors_path = save_path / "model.safetensors"
        pytorch_path = save_path / "pytorch_model.bin"
        if safetensors_path.exists():
            from safetensors.torch import load_file

            state_dict = load_file(str(safetensors_path))
        elif pytorch_path.exists():
            state_dict = torch.load(pytorch_path, map_location="cpu")

        if state_dict:
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            missing_keys = [key for key in missing_keys if ".vlm." not in key]
            unexpected_keys = [key for key in unexpected_keys if ".vlm." not in key]
            if missing_keys:
                logger.warning(f"AssemLMHF load - Missing keys: {missing_keys}")
            if unexpected_keys:
                logger.warning(f"AssemLMHF load - Unexpected keys: {unexpected_keys}")

        return model

    def save_pretrained(self, save_directory: str, **kwargs):
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)

        original_config = self.config
        top_level_state_dict = self._filter_non_vlm_state_dict(self.state_dict())
        hf_kwargs = dict(kwargs)
        hf_kwargs["safe_serialization"] = True
        temp_payload = original_config.to_dict()
        temp_payload.setdefault("framework", {})
        temp_payload["framework"].setdefault("vlm", {})
        temp_payload["framework"]["vlm"]["base_vlm"] = "vlm"
        temp_payload["framework"]["vlm"]["local_files_only"] = True

        try:
            self.config = AssemLMHFConfig(**temp_payload)
            super().save_pretrained(str(save_path), state_dict=top_level_state_dict, **hf_kwargs)
        finally:
            self.config = original_config

        if hasattr(self.assemlm_interface, "save_vlm_pretrained"):
            self.assemlm_interface.save_vlm_pretrained(save_path / "vlm", safe_serialization=True)

        return str(save_path)

    @classmethod
    def export_from_legacy_model(
        cls,
        legacy_model: Any,
        save_directory: str,
        *,
        safe_serialization: bool = True,
    ) -> str:
        """Export a legacy AssemLM model into an HF-style directory.

        The exported directory can be loaded with `AssemLMHF.from_pretrained(...)`.
        """
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)

        legacy_config = getattr(legacy_model, "config", None)
        if legacy_config is None:
            raise ValueError("Legacy model does not expose a config.")

        hf_config = AssemLMHFConfig.from_legacy_config(legacy_config)
        hf_payload = hf_config.to_dict()
        hf_payload.setdefault("framework", {})
        hf_payload["framework"].setdefault("vlm", {})
        hf_payload["framework"]["vlm"]["base_vlm"] = "vlm"
        hf_payload["framework"]["vlm"]["local_files_only"] = True
        hf_config = AssemLMHFConfig(**hf_payload)
        hf_config.save_pretrained(str(save_path))

        state_dict = legacy_model.state_dict()
        top_level_state_dict = cls._filter_non_vlm_state_dict(state_dict)
        cls._save_state_dict(top_level_state_dict, save_path, safe_serialization=True)

        legacy_interface = getattr(legacy_model, "assemlm_interface", None)
        if legacy_interface is not None and hasattr(legacy_interface, "save_vlm_pretrained"):
            legacy_interface.save_vlm_pretrained(save_path / "vlm", safe_serialization=True)

        return str(save_path)

    # @classmethod
    # def from_legacy_checkpoint(cls, pretrained_checkpoint: str, **kwargs):
    #     from assemlm.model.framework.base_framework import baseframework

    #     legacy_model = baseframework.from_pretrained(pretrained_checkpoint, **kwargs)
    #     legacy_config = getattr(legacy_model, "config", None)
    #     if legacy_config is None:
    #         raise ValueError("Legacy checkpoint did not expose a config.")
    #     return cls(AssemLMHFConfig.from_legacy_config(legacy_config))