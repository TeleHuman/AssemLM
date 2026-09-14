import torch
import torch.nn as nn
from typing import Optional
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

from dataclasses import dataclass

DEFAULT_POINT_TOKEN = "<point>"
DEFAULT_POINT_START_TOKEN = "<point_start>"
DEFAULT_POINT_END_TOKEN = "<point_end>"
DEFAULT_POSE_QUERY_TOKEN = "<pose_query>"

@dataclass
class AssemLMOutputWithPast(CausalLMOutputWithPast):
    last_hidden_state: Optional[torch.FloatTensor] = None

from assemlm.model.modules.point_projector import (
    PointCloudProjector,
    PointPatchFusionBridge,
    normalize_point_center_pair,
)
from assemlm.model.modules.point_encoder import get_point_encoder
class _PVLM_Interface(nn.Module):
    @staticmethod
    def _cfg_bool(value, default=False):
        if value is None:
            return bool(default)
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y"}
        return bool(value)

    def __init__(self, config: Optional[dict] = None, **kwargs):
        super().__init__()
        self.config = config
        vlm_config = config.framework.get("vlm", {})
        model_id = vlm_config.get("base_vlm", "Qwen/Qwen3-VL-4B-Instruct")

        point_mrope_config = config.framework.get("point_mrope", {}) or {}
        point_attention_config = config.framework.get("point_attention", {}) or {}
        pose_regression_config = config.framework.get("pose_regression", {}) or {}
        self.point_mrope_enabled = self._cfg_bool(self._cfg_get(point_mrope_config, "enabled", False))
        self.point_mrope_grid_size = max(2, int(self._cfg_get(point_mrope_config, "grid_size", 64)))
        self.point_mrope_normalize = str(self._cfg_get(point_mrope_config, "normalize", "per_sample")).strip().lower()
        self.point_mrope_base_mode = str(self._cfg_get(point_mrope_config, "base_mode", "first_point")).strip().lower()
        self.point_mrope_reflow_text_positions = self._cfg_bool(
            self._cfg_get(point_mrope_config, "reflow_text_positions", False)
        )
        self.pose_regression_enabled = self._cfg_bool(self._cfg_get(pose_regression_config, "enabled", False))
        if not self.pose_regression_enabled:
            raise ValueError("AssemLM 2.0 requires continuous pose regression.")
        self.pose_query_token = str(
            self._cfg_get(
                pose_regression_config,
                "query_token",
                DEFAULT_POSE_QUERY_TOKEN,
            )
        )
        self.pose_query_token_id = None
        self.point_bidirectional_attention_enabled = self._cfg_bool(
            self._cfg_get(point_attention_config, "bidirectional_enabled", False)
        )
        self.point_bidirectional_attention_scope = str(
            self._cfg_get(point_attention_config, "bidirectional_scope", "all_points")
        ).strip().lower()
        self.point_feature_attention_isolation = str(
            self._cfg_get(point_attention_config, "feature_isolation", "none")
        ).strip().lower()
        if self.point_feature_attention_isolation not in {"none", "protect_invariant"}:
            raise ValueError(
                "framework.point_attention.feature_isolation must be none or protect_invariant, "
                f"got {self.point_feature_attention_isolation!r}."
            )

        self.attn_implementation = vlm_config.get("attn_implementation", "eager")

        self.vlm = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            attn_implementation=self.attn_implementation,
            dtype=torch.bfloat16,
            device_map="cuda",
        )
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.processor.tokenizer.padding_side = "left"
        self.tokenizer = self.processor.tokenizer

        # alin qwen3 with qwen2.5
        self.vlm.config.hidden_size = self.vlm.config.text_config.hidden_size

        #  special tokens / pose regression token: <pose_query>
        additional_special_tokens = [
            DEFAULT_POINT_TOKEN,
            DEFAULT_POINT_START_TOKEN,
            DEFAULT_POINT_END_TOKEN,
        ]
        if self.pose_query_token not in additional_special_tokens:
            additional_special_tokens.append(self.pose_query_token)
        special_tokens_dict = {
            "additional_special_tokens": additional_special_tokens
        }

        num_added = self.tokenizer.add_special_tokens(special_tokens_dict)
        if num_added > 0:
            self.vlm.resize_token_embeddings(len(self.tokenizer))
        self.pose_query_token_id = self.tokenizer.convert_tokens_to_ids(self.pose_query_token)

        self.point_token_len = config.datasets.get("point_token_len", 513)

        self.pc_token_id = self.tokenizer.convert_tokens_to_ids(DEFAULT_POINT_TOKEN)
        self.pc_start_token_id = self.tokenizer.convert_tokens_to_ids(DEFAULT_POINT_START_TOKEN)
        self.pc_end_token_id = self.tokenizer.convert_tokens_to_ids(DEFAULT_POINT_END_TOKEN)

        point_encoder_config = config.framework.point_encoder
        if point_encoder_config.type != "vn_dgcnn_patch":
            raise ValueError("AssemLM 2.0 requires point_encoder.type=vn_dgcnn_patch.")
        point_encoder_type_config = point_encoder_config.vn_dgcnn_patch

        def _point_encoder_cfg_get(key, default=None):
            return point_encoder_type_config.get(key, default)

        if _point_encoder_cfg_get("fusion_module", "patch_transformer") != "patch_transformer":
            raise ValueError("AssemLM 2.0 requires fusion_module=patch_transformer.")
        self.point_encoder_force_fp32 = self._cfg_bool(
            _point_encoder_cfg_get("force_fp32", True)
        )
        EncoderClass = get_point_encoder("vn_dgcnn_patch")
        self.point_encoder = EncoderClass(
            feat_dim=_point_encoder_cfg_get("pc_feat_dim", 512),
            pooling=_point_encoder_cfg_get("pooling", "mean"),
            norm_type=_point_encoder_cfg_get("norm_type", "rmsnorm"),
            num_patches=_point_encoder_cfg_get("num_patches", 128),
            patch_size=_point_encoder_cfg_get("patch_size", 4),
            patch_method=_point_encoder_cfg_get("patch_method", "fps_knn"),
            patch_sort_method=_point_encoder_cfg_get("patch_sort_method", "none"),
            normalize_frame=self._cfg_bool(
                _point_encoder_cfg_get("normalize_frame", True)
            ),
        )
        if self.point_encoder_force_fp32:
            self._ensure_point_encoder_fp32()

        point_projector_config = config.framework.point_projector
        self.point_projector = PointCloudProjector(point_projector_config)
        patch_bridge_config = _point_encoder_cfg_get("bridge", {}) or {}
        self.point_patch_bridge = PointPatchFusionBridge(
            patch_bridge_config,
            input_dim=int(_point_encoder_cfg_get("pc_feat_dim", 512)) * 3,
            output_dim=point_projector_config.project_output_dim,
        )

    @staticmethod
    def _cfg_get(container, key, default=None):
        if container is None:
            return default
        if hasattr(container, "get"):
            return container.get(key, default)
        return getattr(container, key, default)

    def _quantize_point_mrope_centers(self, fixed_centers, moving_centers):
        """Convert continuous point patch centers to integer xyz ids for Qwen M-RoPE.

        fixed_centers/moving_centers: [B, P, 3].
        returns: two LongTensors [B, P, 3] in the range [0, grid_size - 1].
        """
        if not self.point_mrope_enabled:
            return None, None
        if fixed_centers is None or moving_centers is None:
            raise ValueError(
                "framework.point_mrope.enabled=true requires patch centers from vn_dgcnn_patch. "
                "vn_dgcnn_patch must return patch centers for point M-RoPE."
            )
        if fixed_centers.shape != moving_centers.shape or fixed_centers.size(-1) != 3:
            raise ValueError(
                "Point M-RoPE expects fixed/moving patch centers with the same [B, P, 3] shape. "
                f"Got fixed={tuple(fixed_centers.shape)}, moving={tuple(moving_centers.shape)}."
            )

        fixed = fixed_centers.float()
        moving = moving_centers.float()
        grid_max = float(self.point_mrope_grid_size - 1)
        eps = 1e-6

        def _quantize_with_bounds(centers, mins, spans):
            ids = ((centers - mins) / (spans + eps) * grid_max).round()
            return ids.clamp_(0, grid_max).long()

        if self.point_mrope_normalize in {"joint_isotropic", "shared_isotropic", "isotropic"}:
            fixed_normalized, moving_normalized = normalize_point_center_pair(
                fixed,
                moving,
                mode="joint_isotropic",
                eps=eps,
            )

            def _quantize_unit_ball(centers):
                # joint_isotropic  [-1, 1] [0, G-1]
                ids = ((centers + 1.0) * 0.5 * grid_max).round()
                return ids.clamp_(0, grid_max).long()

            return _quantize_unit_ball(fixed_normalized), _quantize_unit_ball(moving_normalized)

        if self.point_mrope_normalize in {"per_sample", "per_sample_all", "all_points"}:
            all_centers = torch.cat([fixed, moving], dim=1)
            mins = all_centers.amin(dim=1, keepdim=True)
            spans = all_centers.amax(dim=1, keepdim=True) - mins
            return _quantize_with_bounds(fixed, mins, spans), _quantize_with_bounds(moving, mins, spans)

        if self.point_mrope_normalize in {"per_segment", "per_cloud", "per_part"}:
            fixed_mins = fixed.amin(dim=1, keepdim=True)
            fixed_spans = fixed.amax(dim=1, keepdim=True) - fixed_mins
            moving_mins = moving.amin(dim=1, keepdim=True)
            moving_spans = moving.amax(dim=1, keepdim=True) - moving_mins
            return (
                _quantize_with_bounds(fixed, fixed_mins, fixed_spans),
                _quantize_with_bounds(moving, moving_mins, moving_spans),
            )

        raise ValueError(
            f"Unsupported framework.point_mrope.normalize={self.point_mrope_normalize!r}. "
            "Expected joint_isotropic, per_sample, or per_segment."
        )

    def _apply_point_mrope_position_ids(
        self,
        position_ids,
        point_token_context,
        attention_mask_2d=None,
    ):
        """Replace Qwen M-RoPE axes for <point> tokens with quantized xyz ids.

        Qwen3-VL accepts a special [4, B, L] position_ids layout:
        - axis 0 is the ordinary text/causal position used by the attention mask helper.
        - axes 1..3 are the T/H/W M-RoPE axes. For point tokens we reuse them as X/Y/Z.
        The ordinary text/causal axis is never changed. When
        ``reflow_text_positions=true``, every contiguous non-point span after
        the first point token is compacted behind the shared point xyz space.
        This removes the artificial cost of counting hundreds of point
        placeholders as text while preserving each text/image span's internal
        relative T/H/W positions.
        """
        if not self.point_mrope_enabled or point_token_context is None:
            return position_ids

        point_pos_masks = point_token_context.get("pos_masks", None)
        point_mrope_coords = point_token_context.get("mrope_coords", None)
        if point_pos_masks is None or point_mrope_coords is None:
            raise ValueError("Point M-RoPE is enabled, but point token metadata was not produced.")

        point_pos_masks = point_pos_masks.to(device=position_ids.device).bool()
        if attention_mask_2d is None:
            valid_tokens = torch.ones_like(point_pos_masks)
        else:
            if attention_mask_2d.ndim != 2 or attention_mask_2d.shape != point_pos_masks.shape:
                raise ValueError(
                    "Point M-RoPE expects attention_mask_2d shaped [B, L], "
                    f"got {tuple(attention_mask_2d.shape)} for point mask {tuple(point_pos_masks.shape)}."
                )
            valid_tokens = attention_mask_2d.to(device=position_ids.device).bool()
        point_mrope_coords = point_mrope_coords.to(device=position_ids.device, dtype=torch.long)
        coord_ready = (point_mrope_coords >= 0).all(dim=-1)
        point_mask = point_pos_masks & coord_ready & valid_tokens
        if not bool(point_pos_masks.any().item()):
            return position_ids
        if not bool(point_mask.any().item()):
            raise ValueError(
                "Point M-RoPE is enabled, but no point token received xyz RoPE coordinates. "
                "This path currently expects vn_dgcnn_patch + patch_transformer."
            )
        if bool((point_pos_masks & ~coord_ready).any().item()):
            raise ValueError("Some <point> tokens are missing xyz RoPE coordinates.")

        if position_ids.ndim == 2:
            base_mrope_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1).clone()
            text_position_ids = base_mrope_ids[:1].clone()
        elif position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[:1].clone()
            base_mrope_ids = position_ids[1:].clone()
        elif position_ids.ndim == 3 and position_ids.shape[0] == 3:
            text_position_ids = position_ids[:1].clone()
            base_mrope_ids = position_ids.clone()
        else:
            raise ValueError(f"Unsupported position_ids shape for point M-RoPE: {tuple(position_ids.shape)}")

        point_mrope_ids = point_mrope_coords.clone()
        segment_ids = point_token_context.get("segment_ids", None)
        if segment_ids is not None:
            segment_ids = segment_ids.to(device=position_ids.device)

        if self.point_mrope_base_mode in {"none", "zero", "no_offset"}:
            pass
        elif self.point_mrope_base_mode in {"first_point", "sample_start"}:
            for batch_idx in range(point_mask.size(0)):
                token_positions = torch.where(point_mask[batch_idx])[0]
                if token_positions.numel() == 0:
                    continue
                base = base_mrope_ids[0, batch_idx, token_positions[0]]
                point_mrope_ids[batch_idx, token_positions] += base
        elif self.point_mrope_base_mode in {"segment_start", "per_segment"}:
            if segment_ids is None:
                raise ValueError("point_mrope.base_mode=segment_start requires point segment ids.")
            for batch_idx in range(point_mask.size(0)):
                batch_segments = torch.unique(segment_ids[batch_idx][point_mask[batch_idx]])
                for segment_id in batch_segments:
                    if int(segment_id.item()) < 0:
                        continue
                    token_positions = torch.where(point_mask[batch_idx] & (segment_ids[batch_idx] == segment_id))[0]
                    if token_positions.numel() == 0:
                        continue
                    base = base_mrope_ids[0, batch_idx, token_positions[0]]
                    point_mrope_ids[batch_idx, token_positions] += base
        else:
            raise ValueError(
                f"Unsupported framework.point_mrope.base_mode={self.point_mrope_base_mode!r}. "
                "Expected first_point, segment_start, or none."
            )

        if self.point_mrope_reflow_text_positions:
            sequence_positions = torch.arange(point_mask.size(1), device=position_ids.device)
            for batch_idx in range(point_mask.size(0)):
                point_positions = torch.where(point_mask[batch_idx])[0]
                if point_positions.numel() == 0:
                    continue

                #  point segment  xyz  point
                #  point token end/start
                #  T/H/W
                first_point_position = point_positions[0]
                non_point_positions = torch.where(
                    valid_tokens[batch_idx]
                    & ~point_mask[batch_idx]
                    & (sequence_positions > first_point_position)
                )[0]
                if non_point_positions.numel() == 0:
                    continue

                split_after = torch.where(non_point_positions[1:] != non_point_positions[:-1] + 1)[0] + 1
                span_starts = torch.cat([non_point_positions.new_zeros(1), split_after])
                span_ends = torch.cat([split_after, non_point_positions.new_tensor([non_point_positions.numel()])])
                next_available_position = point_mrope_ids[batch_idx, point_positions].amax() + 1

                for start_offset, end_offset in zip(span_starts.tolist(), span_ends.tolist()):
                    span_positions = non_point_positions[start_offset:end_offset]
                    current_span_start = base_mrope_ids[:, batch_idx, span_positions[0]]
                    axis_shifts = next_available_position - current_span_start
                    base_mrope_ids[:, batch_idx, span_positions] += axis_shifts[:, None]
                    next_available_position = base_mrope_ids[:, batch_idx, span_positions].amax() + 1

        custom_mrope_ids = base_mrope_ids.clone()
        for axis in range(3):
            custom_mrope_ids[axis][point_mask] = point_mrope_ids[..., axis][point_mask]

        return torch.cat([text_position_ids, custom_mrope_ids], dim=0)

    @staticmethod
    def _recompute_mrope_position_deltas(position_ids, attention_mask_2d=None):
        """Recompute Qwen generation deltas after custom point-position reflow."""
        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            rotary_position_ids = position_ids[1:]
        elif position_ids.ndim == 3 and position_ids.shape[0] == 3:
            rotary_position_ids = position_ids
        elif position_ids.ndim == 2:
            rotary_position_ids = position_ids.unsqueeze(0)
        else:
            raise ValueError(
                f"Unsupported position_ids shape for M-RoPE delta recomputation: {tuple(position_ids.shape)}"
            )

        batch_size, seq_len = rotary_position_ids.shape[1:]
        if attention_mask_2d is None:
            valid_tokens = torch.ones(batch_size, seq_len, dtype=torch.bool, device=position_ids.device)
        else:
            if attention_mask_2d.ndim != 2 or tuple(attention_mask_2d.shape) != (batch_size, seq_len):
                raise ValueError(
                    "M-RoPE delta recomputation expects attention_mask_2d shaped [B, L], "
                    f"got {tuple(attention_mask_2d.shape)} for B={batch_size}, L={seq_len}."
                )
            valid_tokens = attention_mask_2d.to(device=position_ids.device).bool()

        masked_positions = rotary_position_ids.masked_fill(
            ~valid_tokens.unsqueeze(0),
            torch.iinfo(rotary_position_ids.dtype).min,
        )
        max_positions = masked_positions.amax(dim=(0, 2))
        has_valid_tokens = valid_tokens.any(dim=1)
        max_positions = torch.where(has_valid_tokens, max_positions, torch.zeros_like(max_positions))
        return (max_positions + 1 - seq_len).unsqueeze(1)

    def _build_point_bidirectional_attention_mask(self, attention_mask_2d, point_token_context, inputs_embeds):
        """Build the 4D eager mask for point bidirectionality and feature isolation.

        The normal lower-triangular causal pattern is preserved everywhere except
        pairs where both query and key are <point> tokens. Those point pairs are
        unmasked, so point patches can exchange information independent of their
        textual order before the action tokens read the prefix. When
        ``feature_isolation=protect_invariant``, invariant point queries are then
        prevented from reading equivariant point keys, regardless of token order.
        """
        if (
            not self.point_bidirectional_attention_enabled
            and self.point_feature_attention_isolation == "none"
        ) or point_token_context is None:
            return None
        if self.attn_implementation != "eager":
            raise ValueError(
                "Custom point attention masks require framework.vlm.attn_implementation=eager "
                "because Flash Attention only receives a 2D padding mask in this path."
            )

        point_pos_masks = point_token_context.get("pos_masks", None)
        if point_pos_masks is None or not bool(point_pos_masks.any().item()):
            return None

        device = inputs_embeds.device
        batch_size, seq_len = point_pos_masks.shape
        point_pos_masks = point_pos_masks.to(device=device).bool()
        if attention_mask_2d is None:
            valid_tokens = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
        else:
            valid_tokens = attention_mask_2d.to(device=device).bool()

        query_idx = torch.arange(seq_len, device=device).view(1, seq_len, 1)
        key_idx = torch.arange(seq_len, device=device).view(1, 1, seq_len)
        key_valid = valid_tokens[:, None, :]
        query_valid = valid_tokens[:, :, None]

        allowed = (key_idx <= query_idx) & key_valid & query_valid
        # all_point_pairs  point token
        # point_pairs  segment  attention
        all_point_pairs = point_pos_masks[:, :, None] & point_pos_masks[:, None, :] & key_valid & query_valid
        point_pairs = all_point_pairs

        if self.point_bidirectional_attention_enabled:
            if self.point_bidirectional_attention_scope in {"per_segment", "segment", "same_segment"}:
                segment_ids = point_token_context.get("segment_ids", None)
                if segment_ids is None:
                    raise ValueError("point_attention.bidirectional_scope=per_segment requires point segment ids.")
                segment_ids = segment_ids.to(device=device)
                same_segment = segment_ids[:, :, None] == segment_ids[:, None, :]
                valid_segment = (segment_ids[:, :, None] >= 0) & (segment_ids[:, None, :] >= 0)
                point_pairs = point_pairs & same_segment & valid_segment
            elif self.point_bidirectional_attention_scope in {"all_points", "all", "point"}:
                pass
            else:
                raise ValueError(
                    f"Unsupported framework.point_attention.bidirectional_scope="
                    f"{self.point_bidirectional_attention_scope!r}. Expected all_points or per_segment."
                )
            allowed = allowed | point_pairs

        if self.point_feature_attention_isolation == "protect_invariant":
            feature_type_ids = point_token_context.get("feature_type_ids", None)
            if feature_type_ids is None:
                raise ValueError(
                    "point_attention.feature_isolation=protect_invariant requires point feature type ids."
                )
            feature_type_ids = feature_type_ids.to(device=device, dtype=torch.long)
            missing_types = point_pos_masks & (feature_type_ids < 0)
            if bool(missing_types.any().item()):
                raise ValueError(
                    "Some point tokens are missing equivariant/invariant feature type ids while "
                    "protect_invariant isolation is enabled."
                )
            invariant_queries = feature_type_ids[:, :, None] == 1
            equivariant_keys = feature_type_ids[:, None, :] == 0
            protected_pairs = invariant_queries & equivariant_keys & all_point_pairs
            #  causal/bidirectional  invariant
            # token  causal attention  equivariant token
            allowed = allowed & ~protected_pairs
        # Padding query rows are ignored by the loss, but they still go through
        # attention. Let them attend to valid keys to avoid all -inf softmax rows.
        allowed = torch.where(query_valid, allowed, key_valid.expand_as(allowed))

        min_dtype = torch.finfo(inputs_embeds.dtype).min
        zero = torch.tensor(0.0, device=device, dtype=inputs_embeds.dtype)
        neg = torch.tensor(min_dtype, device=device, dtype=inputs_embeds.dtype)
        return torch.where(allowed[:, None, :, :], zero, neg)

    def _select_lm_attention_mask(self, attention_mask_raw, attention_mask_2d, point_token_context, inputs_embeds):
        custom_mask = self._build_point_bidirectional_attention_mask(
            attention_mask_2d=attention_mask_2d,
            point_token_context=point_token_context,
            inputs_embeds=inputs_embeds,
        )
        if custom_mask is not None:
            return custom_mask
        return attention_mask_raw if self.attn_implementation == "eager" else attention_mask_2d

    def _ensure_point_encoder_fp32(self):
        """Keep point encoder parameters/buffers in fp32 when force_fp32 is enabled."""
        needs_cast = False #
        #  FP32 needs_cast  True
        for tensor in self.point_encoder.parameters():
            if tensor.is_floating_point() and tensor.dtype != torch.float32:
                needs_cast = True
                break
        # buffer buffer
        if not needs_cast:
            for tensor in self.point_encoder.buffers():
                if tensor.is_floating_point() and tensor.dtype != torch.float32:
                    needs_cast = True
                    break
        if needs_cast:
            self.point_encoder.float()

    def _get_point_encoder_forward_dtype(self):
        if getattr(self, "point_encoder_force_fp32", False):
            self._ensure_point_encoder_fp32()
            return torch.float32
        param = next(self.point_encoder.parameters(), None)
        return param.dtype if param is not None else torch.float32

    def _run_point_encoder(self, pcs, dtype=None):
        """Run point encoder with explicit dtype control.

        When `force_fp32=true`, both encoder parameters and point-cloud inputs are
        fp32, and autocast is disabled only for this encoder forward.
        """
        if getattr(self, "point_encoder_force_fp32", False):
            self._ensure_point_encoder_fp32()
            dtype = torch.float32
        elif dtype is None:
            dtype = self._get_point_encoder_forward_dtype()
        pcs = pcs.to(dtype=dtype)
        if pcs.device.type == "cuda":
            with torch.autocast("cuda", enabled=False):
                return self.point_encoder(pcs)
        return self.point_encoder(pcs)


    def _preprocess_images(self, imgs):
        assert len(imgs) == 2, "Images for pose task should be 2"
        content = []

        for idx, img in enumerate(imgs):
            content.append({"type": "image", "image": img})
            if idx == 0:
                # content.append({"type": "text", "text": "  # image 1 (before state)\n"})
                content.append({"type": "text", "text": "\n"})
            else:
                # content.append({"type": "text", "text": f"  # image {idx + 1} (after state)\n"})
                content.append({"type": "text", "text": "\n"})

        return content

    def _preprocess_pointclouds(self, pointclouds):
        assert len(pointclouds) == 2, "Pointclouds for pose task should be 2"
        pcd_prompt = (
            DEFAULT_POINT_START_TOKEN +
            DEFAULT_POINT_TOKEN * self.point_token_len +
            DEFAULT_POINT_END_TOKEN
        )
        return [{"type": "text", "text": f"{pcd_prompt}\n"} for _ in range(2)]

    def build_qwenvl_inputs(self, images, point_clouds, instructions):
        """Build Qwen inputs for two point clouds and two images per sample."""
        if not (len(images) == len(point_clouds) == len(instructions)):
            raise ValueError("Images, point clouds, and instructions must have the same batch size.")

        messages = []
        point_cloud_tensor_list = []
        for imgs, pcs, instruction in zip(images, point_clouds, instructions):
            content = self._preprocess_images(imgs)
            content.extend(self._preprocess_pointclouds(pcs))
            content.append({"type": "text", "text": instruction})
            messages.append([{"role": "user", "content": content}])
            point_cloud_tensor_list.append(
                [torch.as_tensor(pc, dtype=torch.float32) for pc in pcs]
            )

        vlm_max_length = getattr(self.config.trainer, "vlm_max_length", 2048)
        vlm_padding = getattr(self.config.trainer, "vlm_padding", "max_length")
        if isinstance(vlm_padding, str):
            vlm_padding = vlm_padding.lower() not in {"false", "0", "no"}
        batch_inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            padding=vlm_padding,
            max_length=vlm_max_length,
            truncation=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        batch_inputs["point_clouds"] = torch.stack(
            [torch.stack(pcs) for pcs in point_cloud_tensor_list]
        )
        return batch_inputs.to(self.vlm.device)



    def _compose_deepstack_inputs(
        self,
        input_ids,
        hidden_dim,
        dtype,
        image_pos_masks=None,
        deepstack_image_embeds=None,
        point_pos_masks=None,
        deepstack_point_embeds=None,
    ):
        """Compose image and point DeepStack residuals in the exact token order.

         Qwen3-VL language_model  DeepStack
        - language_model
          `hidden_states[visual_pos_masks] += deepstack_visual_embeds[layer]`

        -  `visual_pos_masks`  DeepStack
          residual  token
        - `deepstack_visual_embeds[layer]`  0
          `hidden_states[visual_pos_masks]`


        - input_ids: [B, L]B=batch sizeL=padding/truncation  token
        - hidden_dim: HQwen text hidden size Qwen3-VL-2B  2048
        - image_pos_masks: [B, L]  NoneTrue  image placeholder token
        - point_pos_masks: [B, L]  NoneTrue  <point> token
        - deepstack_image_embeds: list length K_img  Tensor  None
           [N_img, H] N_img=image_pos_masks.sum()
        - deepstack_point_embeds: list length K_point  Tensor  None
          patch_transformer  [B, 2P, H]
           P=point patch  P=256 2P=512

         concat image/point residual
        - PyTorch  boolean mask  `[B, L, H]`  batch-major
           [N, H]
        -  `torch.cat([image_layer, point_layer])`
           image token point token
        -  `layer_full: [B, L, H]`
          image/point residual
          `combined_mask` gather [N_img+N_point, H]


        - combined_mask: [B, L]  None Qwen  visual_pos_masks
           image mask  point mask mask
        - combined_embeds: list length max(K_img, K_point)  None
           [combined_mask.sum(), H]
        """

        def _as_layer_list(layer_embeds):
            #  list[Tensor]
            # - None -> None
            # - Tensor [N,H]  [B,T,H] -> [Tensor]
            # - list/tuple[Tensor] -> list[Tensor]
            if layer_embeds is None:
                return None
            if isinstance(layer_embeds, (list, tuple)):
                return list(layer_embeds)
            return [layer_embeds]
        # image_layers: None  list [N_img, H]
        # point_layers: None  list [B, 2P, H] flatten  [B*2P, H]
        image_layers = _as_layer_list(deepstack_image_embeds)
        point_layers = _as_layer_list(deepstack_point_embeds)
        #  mask  bool [B, L]
        image_pos_masks = image_pos_masks.bool() if image_pos_masks is not None else None
        point_pos_masks = point_pos_masks.bool() if point_pos_masks is not None else None

        #  DeepStack residual  active_masks
        #  mask residual combined_mask
        #  mask + None residual
        active_masks = []
        if image_layers is not None and image_pos_masks is not None:
            active_masks.append(image_pos_masks)
        if point_layers is not None and point_pos_masks is not None:
            active_masks.append(point_pos_masks)
        if len(active_masks) == 0:
            #  DeepStack residual
            # -  image_pos_masks  Qwen visual path
            # -  point_pos_masks  None
            if image_pos_masks is not None:
                return image_pos_masks, None
            return point_pos_masks, None

        # combined_mask: [B, L]
        # True  DeepStack  image token  point token
        combined_mask = active_masks[0].clone()
        for mask in active_masks[1:]:
            # image token  point token
            #  mask
            if torch.any(combined_mask & mask):
                raise ValueError("Image and point DeepStack masks overlap; token ownership is ambiguous.")
            combined_mask = combined_mask | mask

        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        # K = max(K_img, K_point) residual
        num_layers = max(len(image_layers or []), len(point_layers or [])) # 3
        combined_embeds = []

        for layer_idx in range(num_layers):
            # layer_full
            # [B, L, H] language_model hidden_states
            #  image/point residual  token
            layer_full = torch.zeros(
                batch_size,
                seq_len,
                hidden_dim,
                device=device,
                dtype=dtype,
            )

            if image_layers is not None and layer_idx < len(image_layers):
                # image_layer  Qwen vision encoder DeepStack
                # [N_img, H]N_img=image_pos_masks.sum()
                image_layer = image_layers[layer_idx].to(device=device, dtype=dtype)
                expected = int(image_pos_masks.sum().item())
                if image_layer.size(0) != expected or image_layer.size(-1) != hidden_dim:
                    raise ValueError(
                        f"Image DeepStack layer {layer_idx} shape {tuple(image_layer.shape)} "
                        f"does not match mask count {expected} and hidden_dim {hidden_dim}."
                    )
                #  mask
                # layer_full[image_pos_masks]: [N_img, H]
                # image_layer:                 [N_img, H]
                #  layer_full  [B, L, H]
                layer_full[image_pos_masks] = image_layer

            if point_layers is not None and layer_idx < len(point_layers):
                # point_layer  PointPatchFusionBridge
                # - patch_transformer : [B, 2P, H]
                # -  flatten : [N_point, H]
                point_layer = point_layers[layer_idx].to(device=device, dtype=dtype)
                if point_layer.ndim == 3:
                    # [B, 2P, H] -> [B*2P, H]
                    #  point_pos_masks  True  batch-major
                    #  reshape  `layer_full[point_pos_masks]`
                    point_layer = point_layer.reshape(-1, point_layer.size(-1))
                expected = int(point_pos_masks.sum().item())
                if point_layer.size(0) != expected or point_layer.size(-1) != hidden_dim:
                    raise ValueError(
                        f"Point DeepStack layer {layer_idx} shape {tuple(point_layer.shape)} "
                        f"does not match mask count {expected} and hidden_dim {hidden_dim}."
                    )
                # layer_full[point_pos_masks]: [N_point, H]
                # point_layer:                  [N_point, H]
                #  image  point residual  token
                layer_full[point_pos_masks] = point_layer

            #  gather
            # layer_full:                [B, L, H]
            # combined_mask:             [B, L]
            # layer_full[combined_mask]: [N_img + N_point, H]
            #  language_model  `hidden_states[visual_pos_masks]`
            #  deepstack_visual_embeds
            combined_embeds.append(layer_full[combined_mask])

        return combined_mask, combined_embeds

    # ====================  inputs_embeds +  ====================
    def _build_inputs_embeds(
        self,
        input_ids,
        point_clouds=None,
        pixel_values=None,
        image_grid_thw=None,
    ):
        """
         input_ids / point_clouds / pixel_values  inputs_embeds

          -  token  lm embedding
          - image placeholder token  image features
          - point token  embedding
        """
        device = input_ids.device
        dtype = self.vlm.get_input_embeddings().weight.dtype
        # 1.  embedding
        inputs_embeds = self.vlm.get_input_embeddings()(input_ids)  # (B, L, H)
        # 2.  token embedding
        point_pos_masks = None
        point_token_context = None
        deepstack_point_embeds = None
        if point_clouds is not None:
            inputs_embeds, point_pos_masks, deepstack_point_embeds, point_token_context = self._replace_pointcloud_tokens(
                inputs_embeds,
                input_ids,
                point_clouds,
            )

        # 3.  image_features vision
        image_mask = None
        image_pos_masks = None
        visual_pos_masks = None
        deepstack_visual_embeds = None
        deepstack_image_embeds = None

        if pixel_values is not None:
            # Qwen3-VL  vision -> embedding
            image_embeds, deepstack_image_embeds = self.vlm.model.get_image_features(pixel_values, image_grid_thw)
            # print_var_info("image_embeds", image_embeds)
            # print_var_info("deepstack_image_embeds", deepstack_image_embeds)

            # image_embeds  list[Tensor] cat
            image_embeds = torch.cat(image_embeds, dim=0).to(device=device, dtype=dtype)

            # mask:  <|image_pad|>
            image_mask, _ = self.vlm.model.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            # print_var_info("image_mask", image_mask)

            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if image_mask is not None:
            image_pos_masks = image_mask[..., 0]

        visual_pos_masks, deepstack_visual_embeds = self._compose_deepstack_inputs(
            input_ids=input_ids,
            hidden_dim=inputs_embeds.size(-1),
            dtype=inputs_embeds.dtype,
            image_pos_masks=image_pos_masks,
            deepstack_image_embeds=deepstack_image_embeds,
            point_pos_masks=point_pos_masks,
            deepstack_point_embeds=deepstack_point_embeds,
        )

        # print_var_info("visual_pos_masks", visual_pos_masks)
        # print_var_info("deepstack_visual_embeds", deepstack_visual_embeds)
        return inputs_embeds, visual_pos_masks, deepstack_visual_embeds, point_token_context


    def _unpack_vn_dgcnn_outputs(self, enc_res):
        """ encoder


        - equiv_feats: [B, C, 3]
        - inv_feats: [B, C, C]  [B, T, C]/
        - z0: [B, 3, 3]  NoneVNStdFeature
        - patch_equiv: [B, P, C, 3]  Nonepatch
        - patch_centers: [B, P, 3]  Nonepatch
        - local_equiv: [B, C, 3, N]  None
        """
        if not isinstance(enc_res, (tuple, list)):
            raise ValueError(
                f"Point encoder must return a tuple/list, got {type(enc_res).__name__}."
            )
        if len(enc_res) == 2:
            equiv_feats, inv_feats = enc_res
            return equiv_feats, inv_feats, None, None, None, None
        if len(enc_res) >= 6:
            return enc_res[:6]
        if len(enc_res) >= 5:
            return (*enc_res[:5], None)
        raise ValueError(
            f"Point encoder returned {len(enc_res)} values; expected 2 or at least 5."
        )


    # ====================  embedding  ====================
    def _replace_pointcloud_tokens(self, inputs_embeds, input_ids, point_clouds):
        """Encode the moving/fixed pair and inject the configured bridge tokens."""
        if (
            not torch.is_tensor(point_clouds)
            or point_clouds.ndim != 4
            or point_clouds.size(1) != 2
        ):
            raise ValueError("AssemLM 2.0 expects point_clouds with shape [B, 2, C, N].")

        batch_size = inputs_embeds.size(0)
        device = inputs_embeds.device
        point_pos_masks = torch.zeros_like(input_ids, dtype=torch.bool)
        point_segment_ids = torch.full_like(input_ids, -1)
        point_feature_type_ids = torch.full_like(input_ids, -1)
        point_mrope_coords = (
            torch.full((*input_ids.shape, 3), -1, dtype=torch.long, device=device)
            if self.point_mrope_enabled
            else None
        )

        encoder_cfg = self.config.framework.point_encoder.vn_dgcnn_patch
        if str(encoder_cfg.fusion_module) != "patch_transformer":
            raise ValueError("AssemLM 2.0 requires fusion_module=patch_transformer.")
        if str(getattr(self.point_patch_bridge, "feature_layout", "")) != "four_branch_equiv_invariant":
            raise ValueError("AssemLM 2.0 requires the four_branch_equiv_invariant bridge layout.")

        pe_dtype = self._get_point_encoder_forward_dtype()
        fixed_pcs = point_clouds[:, 1].to(device=device, dtype=pe_dtype)
        moving_pcs = point_clouds[:, 0].to(device=device, dtype=pe_dtype)
        with torch.autocast("cuda", enabled=False):
            fixed = self._unpack_vn_dgcnn_outputs(
                self._run_point_encoder(fixed_pcs, dtype=pe_dtype)
            )
            moving = self._unpack_vn_dgcnn_outputs(
                self._run_point_encoder(moving_pcs, dtype=pe_dtype)
            )

        _, fixed_invariant, fixed_z0, fixed_patch, fixed_centers, _ = fixed
        _, _, moving_z0, moving_patch, moving_centers, _ = moving
        required = (
            fixed_z0,
            moving_z0,
            fixed_patch,
            moving_patch,
            fixed_centers,
            moving_centers,
        )
        if any(value is None for value in required):
            raise RuntimeError(
                "vn_dgcnn_patch did not return all features required by the release bridge."
            )

        fixed_patch_invariant = torch.einsum(
            "bpci,bij->bpcj", fixed_patch, fixed_z0
        )
        moving_patch_invariant = torch.einsum(
            "bpci,bij->bpcj", moving_patch, moving_z0
        )
        fixed_invariant_centers = torch.einsum(
            "bpi,bij->bpj", fixed_centers, fixed_z0
        )
        moving_invariant_centers = torch.einsum(
            "bpi,bij->bpj", moving_centers, moving_z0
        )

        patch_count = fixed_patch.size(1)
        if self.point_token_len != 2 * patch_count:
            raise ValueError(
                f"point_token_len must equal 2 * num_patches ({2 * patch_count}), "
                f"got {self.point_token_len}."
            )

        bridge_result = self.point_patch_bridge(
            fixed_patch,
            moving_patch,
            fixed_centers,
            moving_centers,
            return_deepstack=True,
            fixed_patch_invariant=fixed_patch_invariant,
            moving_patch_invariant=moving_patch_invariant,
            fixed_patch_invariant_centers=fixed_invariant_centers,
            moving_patch_invariant_centers=moving_invariant_centers,
        )
        point_embeddings, deepstack_point_embeds = bridge_result
        expected_tokens = 4 * patch_count
        if point_embeddings.size(1) != expected_tokens:
            raise ValueError(
                f"The four-branch bridge must return {expected_tokens} tokens, "
                f"got {point_embeddings.size(1)}."
            )
        if point_embeddings.size(-1) != inputs_embeds.size(-1):
            raise ValueError("Point bridge output and Qwen hidden dimensions must match.")

        fixed_embeddings = torch.cat(
            [
                point_embeddings[:, :patch_count],
                point_embeddings[:, patch_count:2 * patch_count],
            ],
            dim=1,
        )
        moving_embeddings = torch.cat(
            [
                point_embeddings[:, 2 * patch_count:3 * patch_count],
                point_embeddings[:, 3 * patch_count:],
            ],
            dim=1,
        )
        fixed_mrope, moving_mrope = self._quantize_point_mrope_centers(
            fixed_centers, moving_centers
        )
        fixed_invariant_mrope, moving_invariant_mrope = (
            self._quantize_point_mrope_centers(
                fixed_invariant_centers, moving_invariant_centers
            )
        )
        if fixed_mrope is not None:
            fixed_mrope = torch.cat([fixed_mrope, fixed_invariant_mrope], dim=1)
            moving_mrope = torch.cat([moving_mrope, moving_invariant_mrope], dim=1)

        feature_ids = torch.cat(
            [
                torch.zeros(patch_count, dtype=torch.long, device=device),
                torch.ones(patch_count, dtype=torch.long, device=device),
            ]
        )
        for batch_index in range(batch_size):
            starts = torch.where(input_ids[batch_index] == self.pc_start_token_id)[0]
            ends = torch.where(input_ids[batch_index] == self.pc_end_token_id)[0]
            if len(starts) != 2 or len(ends) != 2:
                raise ValueError("Each sample must contain two point-cloud segments.")
            for segment, (start, end) in enumerate(zip(starts, ends)):
                positions = torch.arange(start + 1, end, device=device)
                if positions.numel() != self.point_token_len:
                    raise ValueError(
                        f"Point segment {segment} contains {positions.numel()} tokens; "
                        f"expected {self.point_token_len}."
                    )
                embeddings = fixed_embeddings if segment == 0 else moving_embeddings
                inputs_embeds[batch_index, positions] = embeddings[batch_index].to(
                    dtype=inputs_embeds.dtype
                )
                point_pos_masks[batch_index, positions] = True
                point_segment_ids[batch_index, positions] = segment
                point_feature_type_ids[batch_index, positions] = feature_ids
                if point_mrope_coords is not None:
                    point_mrope = fixed_mrope if segment == 0 else moving_mrope
                    point_mrope_coords[batch_index, positions] = point_mrope[batch_index]

        point_token_context = {
            "pos_masks": point_pos_masks,
            "segment_ids": point_segment_ids,
            "feature_type_ids": point_feature_type_ids,
            "mrope_coords": point_mrope_coords,
        }
        return (
            inputs_embeds,
            point_pos_masks,
            deepstack_point_embeds,
            point_token_context,
        )

    # ==================== forward ====================
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        point_clouds=None,
        pixel_values=None,
        image_grid_thw=None,
        cache_position: Optional[torch.LongTensor] = None,
        past_key_values=None,
        **kwargs,
    ):
        if point_clouds is None:
            raise ValueError("AssemLM 2.0 requires point clouds in every model input.")

        with torch.autocast("cuda", dtype=torch.bfloat16):
            inputs_embeds, visual_pos_masks, deepstack_visual_embeds, point_token_context = (
                self._build_inputs_embeds(
                    input_ids=input_ids,
                    point_clouds=point_clouds,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                )
            )
            attention_mask_raw = (
                attention_mask
                if not isinstance(attention_mask, dict)
                else attention_mask.get("full_attention")
            )
            attention_mask_2d = attention_mask_raw
            if attention_mask_2d is not None and attention_mask_2d.ndim == 4:
                attention_mask_2d = torch.diagonal(
                    attention_mask_2d[:, 0], dim1=1, dim2=2
                )
                if attention_mask_2d.dtype.is_floating_point:
                    attention_mask_2d = (attention_mask_2d > -1.0).int()

            position_ids = kwargs.pop("position_ids", None)
            if position_ids is None:
                position_ids, rope_deltas = self.vlm.model.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    None,
                    attention_mask=attention_mask_2d,
                )
                kwargs["rope_deltas"] = rope_deltas
            position_ids = self._apply_point_mrope_position_ids(
                position_ids,
                point_token_context,
                attention_mask_2d=attention_mask_2d,
            )
            if (
                self.point_mrope_enabled
                and point_token_context is not None
                and self.point_mrope_reflow_text_positions
            ):
                kwargs["rope_deltas"] = self._recompute_mrope_position_deltas(
                    position_ids,
                    attention_mask_2d=attention_mask_2d,
                )

            language_outputs = self.vlm.model.language_model(
                input_ids=None,
                position_ids=position_ids,
                attention_mask=self._select_lm_attention_mask(
                    attention_mask_raw=attention_mask_raw,
                    attention_mask_2d=attention_mask_2d,
                    point_token_context=point_token_context,
                    inputs_embeds=inputs_embeds,
                ),
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                cache_position=cache_position,
                visual_pos_masks=visual_pos_masks,
                deepstack_visual_embeds=deepstack_visual_embeds,
                **kwargs,
            )

        return AssemLMOutputWithPast(
            loss=None,
            logits=None,
            past_key_values=language_outputs.past_key_values,
            hidden_states=language_outputs.hidden_states,
            attentions=language_outputs.attentions,
            last_hidden_state=language_outputs.last_hidden_state,
        )
