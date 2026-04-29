import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Optional
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import transformers.modeling_utils as modeling_utils

from accelerate.logging import get_logger
from PIL import Image
import trimesh
import numpy as np
import time
import importlib

logger = get_logger(__name__)


@contextmanager
def _temporarily_enable_hf_init():
    old_init_weights = modeling_utils._init_weights
    modeling_utils._init_weights = True
    try:
        yield
    finally:
        modeling_utils._init_weights = old_init_weights

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 151655
VIDEO_TOKEN_INDEX = 151656
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"
DEFAULT_POINT_TOKEN = "<point>"
DEFAULT_POINT_START_TOKEN = "<point_start>"
DEFAULT_POINT_END_TOKEN = "<point_end>"

_ACTION_TOKEN_MIN = 151936 # 
_ACTION_TOKEN_MAX = 153984 # 

from assemlm.model.modules.point_projector import PointCloudProjector
from assemlm.model.modules.point_encoder import get_point_encoder

class _AssemLM_Interface(nn.Module):
    def __init__(self, config: Optional[dict] = None, **kwargs):
        super().__init__()

        self.config = config
        vlm_config = config.framework.get("vlm", {})
        model_id = vlm_config.get("base_vlm", "Qwen/Qwen3-VL-4B-Instruct")

        # Check if we need to output attentions (requires eager implementation)
        attn_implementation = vlm_config.get("attn_implementation", "flash_attention_2")
        if kwargs.get("output_attentions", False):
            attn_implementation = "eager"
        self.attn_implementation = attn_implementation
        
        build_version = vlm_config.get("build_version", 1)
        save_version = vlm_config.get("save_version", 1)

        if "saved_version" in vlm_config and vlm_config["saved_version"] != -1:
            build_version = vlm_config["saved_version"]
            vlm_config["build_version"] = build_version
            self.config.framework.vlm.build_version = build_version

        vlm_config["saved_version"] = save_version
        self.config.framework.vlm["saved_version"] = save_version

        if build_version == 0:
            pass
        elif build_version == 1:
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.processor.tokenizer.padding_side = "left"
            self.tokenizer = self.processor.tokenizer

            with _temporarily_enable_hf_init():
                self.vlm = Qwen3VLForConditionalGeneration.from_pretrained(
                    model_id,
                    attn_implementation=self.attn_implementation,
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=False,
                )
            self._sync_vlm_tokenizer_and_embeddings()
            self._materialize_vlm_lm_head()

            if torch.cuda.is_available():
                self.vlm = self.vlm.cuda()

            self.vlm.config.hidden_size = self.vlm.config.text_config.hidden_size # 2048

        self.R = torch.tensor([[0.26726124, -0.57735027,  0.77151675],
            [0.53452248, -0.57735027, -0.6172134],
            [0.80178373,  0.57735027,  0.15430335]], dtype=torch.float64).unsqueeze(0)
        
        self.close_eps = 0.1

        self.point_token_len = config.datasets.get("point_token_len", 513) # 256
        self.pc_token_id = self.tokenizer.convert_tokens_to_ids(DEFAULT_POINT_TOKEN) # 151669
        self.pc_start_token_id = self.tokenizer.convert_tokens_to_ids(DEFAULT_POINT_START_TOKEN) # 151670
        self.pc_end_token_id = self.tokenizer.convert_tokens_to_ids(DEFAULT_POINT_END_TOKEN) # 151671
        
        point_encoder_config = config.framework.get("point_encoder", {})
        pe_type = point_encoder_config.get("type", None)
        if pe_type:
            EncoderClass = None
            try:
                try:
                    EncoderClass = get_point_encoder(pe_type)
                except Exception:
                    pkg = importlib.import_module("starVLA.model.modules.pvlm")
                    EncoderClass = getattr(pkg, pe_type, None)
            except Exception:
                EncoderClass = None
            if EncoderClass is None:
                EncoderClass = get_point_encoder("transformerbert")
        else:
            EncoderClass = get_point_encoder("transformerbert")

        if pe_type == "vn_dgcnn":
            self.point_encoder = EncoderClass(feat_dim=point_encoder_config.get("pc_feat_dim", 512))
            is_resum = point_encoder_config.get("resume_from_checkpoint", None)

            if is_resum:
                point_encoder_model_path = point_encoder_config.get("model_path", None)
                if point_encoder_model_path is not None:
                    self._load_point_encoder_checkpoint(point_encoder_model_path)


        point_projector_config = config.framework.get("point_projector", {})
        self.point_projector = PointCloudProjector(point_projector_config)
        
        self.point_encoder = self.point_encoder.to(self.vlm.device)
        self.point_projector = self.point_projector.to(self.vlm.device)


    def _sync_vlm_tokenizer_and_embeddings(self):
        """Keep tokenizer, embedding matrix, and lm_head aligned."""
        vocab_size = len(self.tokenizer)
        current_vocab_size = self.vlm.get_input_embeddings().weight.shape[0]

        if current_vocab_size != vocab_size:
            self.vlm.resize_token_embeddings(vocab_size)

        self.vlm.config.vocab_size = vocab_size
        if hasattr(self.vlm.config, "text_config") and self.vlm.config.text_config is not None:
            self.vlm.config.text_config.vocab_size = vocab_size

        if getattr(self.vlm.config, "tie_word_embeddings", False):
            self.vlm.tie_weights()

    def _materialize_vlm_lm_head(self):
        """Force tied lm_head weights to exist before any device move."""
        if not hasattr(self.vlm, "lm_head"):
            return

        input_embeddings = self.vlm.get_input_embeddings()
        if input_embeddings is None or not hasattr(input_embeddings, "weight"):
            return

        if getattr(input_embeddings.weight, "is_meta", False):
            raise RuntimeError("VLM input embeddings are still meta after loading.")

        lm_head = self.vlm.lm_head
        lm_weight = getattr(lm_head, "weight", None)
        if lm_weight is not None and getattr(lm_weight, "is_meta", False):
            lm_head.weight = input_embeddings.weight

        if getattr(self.vlm.config, "tie_word_embeddings", False):
            self.vlm.tie_weights()

    def save_vlm_pretrained(
        self,
        output_dir: str,
        *,
        safe_serialization: bool = True,
        include_lm_head: bool = True,
    ) -> str:
        """Export a self-contained VLM snapshot (weights + processor/tokenizer).

        This is what you want if you don't want users to separately download the base Qwen3-VL.

        Args:
            output_dir: Directory to write HF-style files into (e.g. `<model_path>/vlm`).
            safe_serialization: Save weights as `model.safetensors` when True.

        Returns:
            str: Output directory path.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        self._sync_vlm_tokenizer_and_embeddings()
        if include_lm_head and hasattr(self.vlm, "lm_head") and hasattr(self.vlm.lm_head, "weight"):
            self._materialize_vlm_lm_head()

        state_dict = self.vlm.state_dict()
        if include_lm_head and hasattr(self.vlm, "lm_head") and hasattr(self.vlm.lm_head, "weight"):
            state_dict["lm_head.weight"] = self.vlm.lm_head.weight.detach().clone().cpu()
        self.vlm.save_pretrained(out, state_dict=state_dict, safe_serialization=safe_serialization)
        self.processor.save_pretrained(out)
        self.tokenizer.save_pretrained(out)
        return str(out)

    def add_pose_tokens(self):
        pose_token_count = 2048
        pose_tokens = [f"<assemble_pose_{i}>" for i in range(pose_token_count)]
        print(f"[PVLM] Adding {pose_token_count} pose tokens to tokenizer...")
        num_added_pose = self.tokenizer.add_special_tokens({"additional_special_tokens": pose_tokens})
        if num_added_pose > 0:
            old_embed = self.vlm.get_input_embeddings()
            old_size = old_embed.weight.shape[0]
            new_vocab_size = len(self.tokenizer)
            print(f"[PVLM] Resizing token embeddings: {old_size} -> {new_vocab_size}")
            torch.cuda.empty_cache()
            self.vlm.resize_token_embeddings(new_vocab_size)
            new_embed = self.vlm.get_input_embeddings()
            with torch.no_grad():
                ref_vec = old_embed.weight.mean(dim=0, keepdim=True)
                for idx in range(old_size, new_vocab_size):
                    new_embed.weight[idx].copy_(ref_vec[0])
            print(f"[PVLM] Successfully initialized {num_added_pose} new embeddings with mean vector.")
        first_pose_id = self.tokenizer.convert_tokens_to_ids(pose_tokens[0])
        last_pose_id = self.tokenizer.convert_tokens_to_ids(pose_tokens[-1])
        self.pose_token_start_id = first_pose_id
        self.pose_token_end_id = last_pose_id
        print(f"[PVLM] Pose tokens registered. ID Range: [{first_pose_id}, {last_pose_id}]")
        try:
            globals()["_POSE_TOKEN_MIN"] = first_pose_id
            globals()["_POSE_TOKEN_MAX"] = last_pose_id
        except Exception:
            pass

    def _load_point_encoder_checkpoint(self, ckpt_path):
        logger.info(f"Loading point encoder checkpoint from {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location="cpu")
        
        if "model" in state_dict:
            state_dict = state_dict["model"]
        elif "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
            
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("point_encoder."):
                new_state_dict[k[len("point_encoder."):]] = v
            elif k.startswith("module.point_encoder."):
                new_state_dict[k[len("module.point_encoder."):]] = v
            elif k.startswith("encoder."):
                new_state_dict[k[len("encoder."):]] = v
            elif k.startswith("module."):
                k_no_mod = k[len("module."):]
                if k_no_mod.startswith("point_encoder."):
                    new_state_dict[k_no_mod[len("point_encoder."):]] = v
                else:
                    new_state_dict[k_no_mod] = v
            else:
                new_state_dict[k] = v
        
        missing_keys, unexpected_keys = self.point_encoder.load_state_dict(new_state_dict, strict=False)
        if len(missing_keys) > 0:
            logger.warning(f"Point encoder load - Missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            logger.warning(f"Point encoder load - Unexpected keys: {unexpected_keys}")
        
        logger.info(f"Successfully loaded point encoder weights from {ckpt_path}")

    def _preprocess_image_files(self, image_files):
        assert len(image_files) == 2, "Images for pose task should be 2"
        content = []

        def _preprocess_image(image_path):
            
            image = Image.open(image_path).convert("RGB")
            
            return image
        
        for idx, image_file in enumerate(image_files):
            content.append({"type": "image", "image": _preprocess_image(image_file)})
            if idx == 0:
                content.append({"type": "text", "text": "  # image 1 (before state)\n"})
            else:
                content.append({"type": "text", "text": f"  # image {idx + 1} (after state)\n"})

        return content
    
    def _preprocess_images(self, imgs):
        assert len(imgs) == 2, "Images for pose task should be 2"
        content = []
        
        for idx, img in enumerate(imgs):
            content.append({"type": "image", "image": img})
            if idx == 0:
                content.append({"type": "text", "text": "\n"})
            else:
                content.append({"type": "text", "text": "\n"})

        return content
    
    def _preprocess_pointcloud_files(self, pointcloud_files):
        assert len(pointcloud_files) == 2, "Pointclouds for pose task should be 2"
        content = []

        pcd_prompt = (
            DEFAULT_POINT_START_TOKEN + 
            DEFAULT_POINT_TOKEN * self.point_token_len + 
            DEFAULT_POINT_END_TOKEN
        )

        content.append({"type": "text", "text": f"{pcd_prompt}  # first point cloud = fixed part\n"})
        content.append({"type": "text", "text": f"{pcd_prompt}  # second point cloud = moving part\n"})
        return content
    
    def _preprocess_pointclouds(self, pointclouds):
        assert len(pointclouds) == 2, "Pointclouds for pose task should be 2"
        content = []

        pcd_prompt = (
            DEFAULT_POINT_START_TOKEN + 
            DEFAULT_POINT_TOKEN * self.point_token_len + 
            DEFAULT_POINT_END_TOKEN
        )

        content.append({"type": "text", "text": f"{pcd_prompt}\n"})
        content.append({"type": "text", "text": f"{pcd_prompt}\n"})
        return content

    def _load_ply_file(self, file_path):
        mesh_or_pointcloud = trimesh.load(file_path)
        if hasattr(mesh_or_pointcloud, 'vertices'):
            points = mesh_or_pointcloud.vertices
        else:
            points = np.array(mesh_or_pointcloud)

        return points

    def _load_point_cloud(self, point_cloud_path):
        
        if "ply" in point_cloud_path:
            point_cloud = np.array(self._load_ply_file(point_cloud_path), dtype=np.float32)
        else:
            point_cloud = np.load(point_cloud_path).astype(np.float32)

        assert point_cloud.ndim == 2 and point_cloud.shape[1] >= 3, \
            f"Point cloud shape should be (N, C>=3), got {point_cloud.shape}"

        N, C = point_cloud.shape

        xyz = point_cloud[:, :3]  # (N,3)

        if C >= 6:
            feat = point_cloud[:, 3:6]  # (N,3)
        elif C > 3:
            feat = np.zeros((N, 3), dtype=np.float32)
            feat[:, :C-3] = point_cloud[:, 3:]
        else:  # C == 3
            feat = np.zeros((N, 3), dtype=np.float32)
        
        point_cloud_6d = np.concatenate([xyz, feat], axis=-1)  # (N,6)

        return point_cloud_6d

    def build_assemlm_inputs(self, images, point_clouds, instructions, solutions=None, **kwargs):
        """
        Build model inputs from batched multimodal samples.

        Input format requirements:
        - `images`: list of length B. Each item must be a list with exactly 2 RGB images
            ordered as [base image, assemble image]. Each image can be a PIL image.
        - `point_clouds`: list of length B. Each item must contain exactly 2 point clouds
            ordered as [partA, base_partB]. Each point cloud must be an array-like object
            with shape (3, N) before any model-side processing.
        - `instructions`: list of length B. Each item must be a string instruction for
            the corresponding sample, for example: "Assemble the {category} object".
        - `solutions`: optional list of length B. If provided, each item must be the
            target text for the corresponding sample.
        - All three main inputs must have the same batch length B.
        """
        # Create messages: one message per sample
        messages = []
        point_cloud_tensor_list = []
        assert len(images) == len(instructions) == len(point_clouds), "Images, instructions, and point clouds must have the same length"
        
        for imgs, pcs, instruction in zip(images, point_clouds, instructions):
            content = self._preprocess_images(imgs)

            content.extend(self._preprocess_pointclouds(pcs))

            content.append({"type": "text", "text": instruction})
            msg = [{"role": "user", "content": content}]

            if solutions is not None:
                solution = solutions[len(messages)]
                msg.append({"role": "assistant", "content": [{"type": "text", "text": solution}]})
            messages.append(msg)
            point_clouds = pcs
            point_cloud_tensor_list.append([torch.as_tensor(pc, dtype=torch.float32) for pc in point_clouds]) # 4 * 2 * (1024, 3)

        label_mask_mode = kwargs.pop("label_mask_mode", None)
        if solutions is not None and label_mask_mode is None:
            label_mask_mode = "action_tokens"

        vlm_max_length = getattr(self.config.trainer, "vlm_max_length", 2048) # 1800
        vlm_padding = getattr(self.config.trainer, "vlm_padding", "max_length") # True
        
        if isinstance(vlm_padding, str) and vlm_padding.lower() == "true":
            vlm_padding = True
        elif isinstance(vlm_padding, str) and vlm_padding.lower() == "false":
            vlm_padding = False

        add_generation_prompt_full = True
        if solutions is not None and label_mask_mode == "prompt":
            add_generation_prompt_full = False

        batch_inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            padding=vlm_padding,
            max_length=vlm_max_length,
            truncation=True,
            add_generation_prompt=add_generation_prompt_full,
            return_dict=True,
            return_tensors="pt",
        )
        batch_inputs["point_clouds"] = torch.stack([torch.stack(pcs) for pcs in point_cloud_tensor_list]) # torch.Size([4, 2, 1024, 3])

        if solutions is not None:
            if label_mask_mode == "prompt":
                prompt_messages = [m[:1] for m in messages]
                prompt_inputs = self.processor.apply_chat_template(
                    prompt_messages,
                    tokenize=True,
                    padding=vlm_padding,
                    max_length=vlm_max_length,
                    truncation=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                )

                labels = batch_inputs["input_ids"].clone()
                prompt_lens = prompt_inputs["attention_mask"].sum(dim=-1)
                for i in range(labels.size(0)):
                    pl = int(prompt_lens[i].item())
                    labels[i, :pl] = IGNORE_INDEX
                labels[labels == self.processor.tokenizer.pad_token_id] = IGNORE_INDEX
                batch_inputs["labels"] = labels

            else:
                action_token_min = _ACTION_TOKEN_MIN
                action_token_max = _ACTION_TOKEN_MAX
                labels = batch_inputs["input_ids"].clone()
                for i in range(labels.size(0)):
                    seq = labels[i]
                    mask_seq = (seq >= action_token_min) & (seq <= action_token_max)
                    nonzero_indices = torch.nonzero(mask_seq, as_tuple=False)
                    if nonzero_indices.numel() > 0:
                        first_action_index = nonzero_indices[0].item()
                        seq[:first_action_index] = IGNORE_INDEX
                    else:
                        seq[:] = IGNORE_INDEX
                        logger.warning("Action tokens not found in your tokenizer, please check your setup.")

                labels[labels == self.processor.tokenizer.pad_token_id] = IGNORE_INDEX
                batch_inputs["labels"] = labels

        return batch_inputs.to(self.vlm.device)


    def _build_inputs_embeds(
        self,
        input_ids,
        point_clouds=None,
        pixel_values=None,
        image_grid_thw=None,
    ):
        # breakpoint()
        device = input_ids.device
        dtype = self.vlm.get_input_embeddings().weight.dtype

        inputs_embeds = self.vlm.get_input_embeddings()(input_ids)  # (B, L, H), example: torch.Size([4, 1189, 2048])

        if point_clouds is not None:
            
            inputs_embeds = self._replace_pointcloud_tokens(inputs_embeds, input_ids, point_clouds)

        image_mask = None
        visual_pos_masks = None
        deepstack_visual_embeds = None

        disable_deepstack = os.environ.get("STARVLA_DISABLE_DEEPSTACK_VISUAL_EMBEDS", "0") in (
            "1",
            "true",
            "True",
            "yes",
            "YES",
        )

        if pixel_values is not None:
            image_embeds, deepstack_image_embeds = self.vlm.model.get_image_features(pixel_values, image_grid_thw)

            image_embeds = torch.cat(image_embeds, dim=0).to(device=device, dtype=dtype)

            image_mask, _ = self.vlm.model.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )

            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if image_mask is not None:
            image_mask = image_mask[..., 0]
            visual_pos_masks = image_mask
            if not disable_deepstack:
                deepstack_visual_embeds = deepstack_image_embeds

        return inputs_embeds, visual_pos_masks, deepstack_visual_embeds

    def check_equiv(self, x, R, xR, name): 
        # breakpoint()
        mean_diff = torch.mean(torch.abs(torch.matmul(x, R) - xR))
        print("Equivariance mean_diff for {}: {}".format(name, mean_diff))
        if mean_diff > self.close_eps:
            print(f'---[Equiv check]--- {name}: {mean_diff}')
        return mean_diff
    
    def check_inv(self, x, R, xR, name): 
        mean_diff = torch.mean(torch.abs(x - xR))
        print("Invariance mean_diff for {}: {}".format(name, mean_diff))
        if mean_diff > self.close_eps:
            print(f'---[Equiv check]--- {name}: {mean_diff}')
        return mean_diff
    
    def check_pointencoder_property(self, gt_data, pred_data):
        metrics = {}
        with torch.no_grad():
            B, _, N = gt_data['src_pc'].shape # torch.Size([4, 3, 1024])
            R = self.R.float().repeat(B, 1, 1).to(gt_data['src_pc'].device) # torch.Size([4, 3, 3])
            pcs_R = torch.matmul(gt_data['src_pc'].permute(0, 2, 1), R).permute(0, 2, 1) #对「原始源点云」施加旋转变换，得到旋转后的源点云 pcs_R
            
            pred_data_R = {}
            pred_data_R['Fa'], pred_data_R['Ga'] = self.point_encoder(pcs_R)

            equiv_feats = pred_data['Fa']
            equiv_feats_R = pred_data_R['Fa']
            metrics['equiv_feats'] = self.check_equiv(equiv_feats, R, equiv_feats_R, 'equiv_feats')

            inv_feats = pred_data['Ga']
            inv_feats_R = pred_data_R['Ga']
            metrics['inv_feats'] = self.check_inv(inv_feats, R, inv_feats_R, 'inv_feats')

        return metrics

    def _replace_pointcloud_tokens(self, inputs_embeds, input_ids, point_clouds):
        
        batch_size = inputs_embeds.shape[0] # 4
        device = inputs_embeds.device # cuda:0
        pe_config = self.config.framework.get("point_encoder", {}) 
        pe_type = pe_config.get("type") # vn_dgcnn
        pe_dtype = next(self.point_encoder.parameters()).dtype
        
        batch_pcs_to_replace = [None] * batch_size
        
        if pe_type == "vn_dgcnn" and torch.is_tensor(point_clouds) and point_clouds.dim() == 4 and point_clouds.size(1) == 2:
            tgt_pcs = point_clouds[:, 1].to(device=device, dtype=pe_dtype) # torch.Size([4, 3, 1024])
            src_pcs = point_clouds[:, 0].to(device=device, dtype=pe_dtype) # torch.Size([4, 3, 1024])
            
            Fb, Gb = self.point_encoder(tgt_pcs) # fixed [batch, 1024, 3], [batch, 1024, 1024]
            Fa, Ga = self.point_encoder(src_pcs) # moving
            
            gt_data={}
            pred_data={}
            pred_data["Fa"], pred_data["Ga"], pred_data["Fb"], pred_data["Gb"] = Fa, Ga, Fb, Gb
            gt_data['tgt_pc'] = tgt_pcs
            gt_data['src_pc'] = src_pcs
            self.check_pointencoder_property(gt_data, pred_data)

            fused_moving = Fa * Gb[:, :, :3]
            feat_fixed_all = Fb.flatten(1).unsqueeze(1)
            feat_moving_all = fused_moving.flatten(1).unsqueeze(1)
            
            for i in range(batch_size):
                batch_pcs_to_replace[i] = [feat_fixed_all[i:i+1], feat_moving_all[i:i+1]]
            
        for batch_idx in range(batch_size):
            start_positions = torch.where(input_ids[batch_idx] == self.pc_start_token_id)[0]
            end_positions = torch.where(input_ids[batch_idx] == self.pc_end_token_id)[0]
            
            if len(start_positions) == 0 or len(end_positions) == 0:
                continue
                
            pcs_to_replace = batch_pcs_to_replace[batch_idx]
            
            if pcs_to_replace is None:
                pcs = point_clouds[batch_idx]
                if pcs is None: continue
                if not isinstance(pcs, (list, tuple)):
                    pcs = [pcs] if pcs.dim() == 3 else [pcs.unsqueeze(0)]
                
                pcs_to_replace = []
                was_training = self.point_encoder.training
                self.point_encoder.eval()
                for p in pcs:
                    p_in = p.to(device=device, dtype=pe_dtype)
                    if p_in.dim() == 2: p_in = p_in.unsqueeze(0)
                    
                    enc_res = self.point_encoder(p_in)
                    if isinstance(enc_res, tuple): enc_res = enc_res[0]
                    
                    if enc_res.dim() == 2:
                        enc_res = enc_res.unsqueeze(1)
                    elif enc_res.dim() == 3 and enc_res.size(-1) == 3:
                        enc_res = enc_res.flatten(1).unsqueeze(1)
                    pcs_to_replace.append(enc_res)

            for pc_idx, (start_pos, end_pos) in enumerate(zip(start_positions, end_positions)):
                
                if pc_idx >= len(pcs_to_replace):
                    break

                pc_token_positions = torch.arange(start_pos + 1, end_pos, device=device)
                expected_token_len = len(pc_token_positions)
                if expected_token_len == 0: continue

                pc_features = pcs_to_replace[pc_idx]

                if pc_features.size(1) != expected_token_len:
                    if pc_features.size(1) == 1:
                        total_dim = pc_features.size(-1)
                        if total_dim % expected_token_len == 0:
                            pc_features = pc_features.view(pc_features.size(0), expected_token_len, -1)
                        else:
                            new_single_dim = math.ceil(total_dim / expected_token_len)
                            pc_features = F.interpolate(pc_features, size=(expected_token_len * new_single_dim), 
                                                      mode='linear', align_corners=False)
                            pc_features = pc_features.view(pc_features.size(0), expected_token_len, -1)
                    elif pc_features.size(1) > expected_token_len:
                        idx = torch.linspace(0, pc_features.size(1) - 1, expected_token_len, dtype=torch.long, device=device)
                        pc_features = pc_features[:, idx, :]
                    else:
                        repeat_times = (expected_token_len // pc_features.size(1)) + 1
                        pc_features = pc_features.repeat(1, repeat_times, 1)[:, :expected_token_len, :]
                pc_embeddings = self.point_projector(pc_features).to(dtype=inputs_embeds.dtype)
                inputs_embeds[batch_idx, pc_token_positions] = pc_embeddings.squeeze(0)

        return inputs_embeds

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        point_clouds=None,
        pixel_values=None,
        image_grid_thw=None,
        cache_position: Optional[torch.LongTensor] = None,
        past_key_values = None,
        **kwargs,
    ):
        pass

    def generate(
        self,
        input_ids=None,
        attention_mask=None,
        point_clouds=None,
        pixel_values=None,
        image_grid_thw=None,
        max_new_tokens=40,
        do_sample=False,
        temperature=1.0,
        **kwargs,
    ):
        
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            inputs_embeds, visual_pos_masks, deepstack_visual_embeds = self._build_inputs_embeds(
                input_ids=input_ids,
                point_clouds=point_clouds,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
            )

            attention_mask_raw = (
                attention_mask if not isinstance(attention_mask, dict) else attention_mask.get("full_attention", None)
            )

            attention_mask_2d = attention_mask_raw
            if attention_mask_2d is not None and attention_mask_2d.ndim == 4:

                attention_mask_2d = torch.diagonal(attention_mask_2d[:, 0], dim1=1, dim2=2)
                if attention_mask_2d.dtype.is_floating_point:
                    attention_mask_2d = (attention_mask_2d > -1.0).int()

            position_ids, rope_deltas = self.vlm.model.get_rope_index(
                input_ids,
                image_grid_thw,
                None,
                attention_mask=attention_mask_2d,
            )

            generated_ids = []
            past_key_values = None
            
            cur_inputs_embeds = inputs_embeds
            cur_position_ids = position_ids

            step_zero_mask = (
                attention_mask_raw if self.attn_implementation == "eager" else attention_mask_2d
            )
            
            L = inputs_embeds.shape[1]
            cur_cache_position = torch.arange(L, device=inputs_embeds.device)
            
            v_masks = visual_pos_masks
            ds_embeds = deepstack_visual_embeds
            r_deltas = rope_deltas

            for i in range(max_new_tokens):

                lm_outputs = self.vlm.model.language_model(
                    input_ids=None,
                    position_ids=cur_position_ids,
                    attention_mask=step_zero_mask if i == 0 else None,
                    past_key_values=past_key_values,
                    inputs_embeds=cur_inputs_embeds,
                    cache_position=cur_cache_position,
                    visual_pos_masks=v_masks,
                    deepstack_visual_embeds=ds_embeds,
                    rope_deltas=r_deltas if i == 0 else None,
                    use_cache=True,
                )
                
                past_key_values = lm_outputs.past_key_values
                logits = self.vlm.lm_head(lm_outputs.last_hidden_state[:, -1, :])
                
                forbidden_ids = [self.pc_token_id, self.pc_start_token_id, self.pc_end_token_id, 
                                 IMAGE_TOKEN_INDEX, VIDEO_TOKEN_INDEX]
                logits[:, forbidden_ids] = -float("Inf")

                last_token_logits = logits
                
                if do_sample:
                    probs = torch.softmax(last_token_logits / temperature, dim=-1)
                    next_token_id = torch.multinomial(probs, num_samples=1)
                else:
                    next_token_id = torch.argmax(last_token_logits, dim=-1).unsqueeze(-1) # (B, 1)

                generated_ids.append(next_token_id)

                if (next_token_id == self.tokenizer.eos_token_id).all():
                    break

                cur_inputs_embeds = self.vlm.get_input_embeddings()(next_token_id) #[4, 1, 2048]
                
                cur_position_ids = cur_position_ids[..., -1:].clone()
                cur_position_ids += 1
                    
                v_masks = None
                ds_embeds = None
                
                cur_cache_position = cur_cache_position[-1:] + 1

            if not generated_ids:
                return torch.zeros((input_ids.shape[0], 0), dtype=torch.long, device=input_ids.device)
            return torch.cat(generated_ids, dim=1)