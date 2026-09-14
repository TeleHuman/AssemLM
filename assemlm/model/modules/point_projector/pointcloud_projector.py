import torch
import torch.nn as nn
from typing import Optional


def _cfg_get(config, key, default=None):
    if config is None:
        return default
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def _cfg_bool(config, key, default=False):
    value = _cfg_get(config, key, default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def point_center_pair_origin_radius(fixed_centers, moving_centers, eps=1e-6):
    """Return the shared centroid and isotropic radius of two point sets.

    Both results are computed in fp32.  The radius is a scalar per sample with
    shape ``[B, 1, 1]`` and is invariant to a joint rigid rotation.  Keeping it
    alongside normalized coordinates restores the single absolute-scale degree
    of freedom removed by isotropic normalization.
    """
    if fixed_centers is None or moving_centers is None:
        raise ValueError("Point-center statistics require both fixed and moving centers.")
    if (
        fixed_centers.ndim != 3
        or moving_centers.ndim != 3
        or fixed_centers.shape != moving_centers.shape
        or fixed_centers.size(-1) != 3
    ):
        raise ValueError(
            "Point-center statistics expect matching [B, P, 3] tensors, "
            f"got fixed={tuple(fixed_centers.shape)} and moving={tuple(moving_centers.shape)}."
        )

    fixed = fixed_centers.float()
    moving = moving_centers.float()
    all_centers = torch.cat([fixed, moving], dim=1)
    origin = all_centers.mean(dim=1, keepdim=True)
    radius = torch.linalg.vector_norm(all_centers - origin, dim=-1).amax(
        dim=1, keepdim=True
    ).unsqueeze(-1)
    return origin, radius.clamp_min(float(eps))


def point_center_pair_scale_features(fixed_centers, moving_centers, eps=1e-6):
    """Return four complementary geometric scales invariant to a joint rigid motion.

    The returned tensor has shape ``[B, 4]`` and uses this fixed order::

        [fixed_radius, moving_radius, center_distance, joint_radius]

    ``fixed_radius`` and ``moving_radius`` measure each part around its own
    centroid. ``center_distance`` separates relative part displacement from
    object size. ``joint_radius`` is the exact positive scale removed by the
    shared ``joint_isotropic`` normalization. All statistics are computed in
    fp32 and clamped away from zero so their logarithms remain finite. The exact
    joint radius may still change under an *independent* rotation of only one
    anisotropic part because it is the actual radius of the combined scene.
    """
    # Reuse the shared validator and joint statistic so this helper follows the
    # exact same geometric convention as normalize_point_center_pair().
    _, joint_radius = point_center_pair_origin_radius(
        fixed_centers,
        moving_centers,
        eps=eps,
    )
    fixed = fixed_centers.float()
    moving = moving_centers.float()

    fixed_origin = fixed.mean(dim=1, keepdim=True)
    moving_origin = moving.mean(dim=1, keepdim=True)
    fixed_radius = torch.linalg.vector_norm(fixed - fixed_origin, dim=-1).amax(
        dim=1, keepdim=True
    )
    moving_radius = torch.linalg.vector_norm(moving - moving_origin, dim=-1).amax(
        dim=1, keepdim=True
    )
    center_distance = torch.linalg.vector_norm(
        fixed_origin.squeeze(1) - moving_origin.squeeze(1),
        dim=-1,
        keepdim=True,
    )
    scales = torch.cat(
        [fixed_radius, moving_radius, center_distance, joint_radius.squeeze(-1)],
        dim=-1,
    )
    return scales.clamp_min(float(eps))


def normalize_point_center_pair(fixed_centers, moving_centers, mode="none", eps=1e-6):
    """Normalize two point-center sets in one shared, geometry-preserving frame.

    ``joint_isotropic`` subtracts the joint centroid of both parts and divides
    every xyz coordinate by one shared maximum Euclidean radius.  Unlike
    per-axis min-max scaling, this preserves aspect ratios, angles, and relative
    distances up to one global positive scale factor.
    """
    mode = str(mode).strip().lower()
    if mode in {"none", "raw", "identity"}:
        return fixed_centers, moving_centers
    if mode not in {"joint_isotropic", "shared_isotropic", "isotropic"}:
        raise ValueError(
            f"Unsupported point-center normalization mode: {mode!r}. "
            "Expected none or joint_isotropic."
        )
    # FP32 keeps the centroid/radius stable even when the surrounding model uses bf16.
    fixed = fixed_centers.float()
    moving = moving_centers.float()
    origin, radius = point_center_pair_origin_radius(fixed, moving, eps=eps)
    fixed_centered = fixed - origin
    moving_centered = moving - origin

    #  clamp [-1,1]
    #  clamp
    fixed_normalized = fixed_centered / radius
    moving_normalized = moving_centered / radius
    return fixed_normalized, moving_normalized


class PointCloudProjector(nn.Module):
    def __init__(self, config: Optional[dict] = None, **kwargs):
        super().__init__()

        self.projection_hidden_layer = config.projection_hidden_layer
        self.backbone_output_dim = config.backbone_output_dim
        self.projection_hidden_dim = config.projection_hidden_dim or []
        self.project_output_dim = config.project_output_dim

        # build projector structure
        if self.projection_hidden_layer > 0:
            layers = []
            last_dim = self.backbone_output_dim
            hidden_dims = self.projection_hidden_dim

            for i in range(self.projection_hidden_layer):
                layers.append(nn.LayerNorm(last_dim))
                layers.append(nn.Linear(last_dim, hidden_dims[i]))
                layers.append(nn.GELU())
                last_dim = hidden_dims[i]

            # final layer
            layers.append(nn.Linear(last_dim, self.project_output_dim))
            self.layers = nn.Sequential(*layers)

        else:
            # only one layer
            self.layers = nn.Linear(
                self.backbone_output_dim,
                self.project_output_dim
            )

    def forward(self, x, *args, **kwargs):
        return self.layers(x)


class PointPatchFusionBridge(nn.Module):
    """Fuse point patch tokens before injecting them into Qwen.

    The default ``fixed_fusion`` layout preserves the original two-branch
    implementation and checkpoint shapes.  The optional
    ``four_branch_equiv_invariant`` layout is used only when explicitly
    configured and injects four independent branches in this order:

        fixed equivariant, fixed invariant,
        moving equivariant, moving invariant.

    Shape flow for the default layout:
        fixed_patch_equiv:  [B, P, C, 3]
        fusion_patch_equiv: [B, P, C, 3]
        flattened tokens:   [B, P, C * 3] for each branch
        concatenated tokens:[B, 2P, C * 3]
        transformer hidden: [B, 2P, H]
        output embeddings:  [B, 2P, D]
        deepstack embeds:   list[[B, 2P, D]] from selected bridge layers

    B is batch size, P is patch count, C is VN channel count, H is bridge
    hidden_dim, and D is the target VLM hidden size.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        input_dim: int = 1536,
        output_dim: int = 2048,
        **kwargs,
    ):
        super().__init__()
        self.input_dim = int(_cfg_get(config, "input_dim", input_dim))
        self.output_dim = int(_cfg_get(config, "output_dim", output_dim))
        self.hidden_dim = int(_cfg_get(config, "hidden_dim", 1024))
        self.num_layers = int(_cfg_get(config, "num_layers", 2))
        self.num_heads = int(_cfg_get(config, "num_heads", 8))
        self.ffn_dim = int(_cfg_get(config, "ffn_dim", self.hidden_dim * 4))
        self.dropout = float(_cfg_get(config, "dropout", 0.1))
        self.pos_embed_type = str(_cfg_get(config, "pos_embed_type", "fourier")).lower()
        self.fourier_bands = int(_cfg_get(config, "fourier_bands", 6))
        self.fourier_max_freq = float(_cfg_get(config, "fourier_max_freq", 10.0))
        self.feature_layout = str(_cfg_get(config, "feature_layout", "fixed_fusion")).strip().lower()
        self.position_normalize = str(_cfg_get(config, "position_normalize", "none")).strip().lower()
        self.attention_isolation = str(_cfg_get(config, "attention_isolation", "none")).strip().lower()
        self.scale_encoding_enabled = _cfg_bool(config, "scale_encoding_enabled", False)
        self.scale_injection = str(_cfg_get(config, "scale_injection", "film")).strip().lower()
        self.scale_feature_mode = str(
            _cfg_get(config, "scale_feature_mode", "joint_radius")
        ).strip().lower()
        self.scale_reference = float(_cfg_get(config, "scale_reference", 1.0))
        self.scale_log_clip = float(_cfg_get(config, "scale_log_clip", 10.0))
        self.deepstack_enabled = _cfg_bool(config, "deepstack_enabled", False)
        self.deepstack_num_layers = int(_cfg_get(config, "deepstack_num_layers", 3))
        self.deepstack_layer_indexes = self._parse_layer_indexes(
            _cfg_get(config, "deepstack_layer_indexes", "auto")
        )
        self.deepstack_init_scale = float(_cfg_get(config, "deepstack_init_scale", 0.1))

        # input_dim must equal C * 3 because every VN patch token keeps C
        # 3D vectors before flattening: [B, P, C, 3] -> [B, P, C * 3].
        if self.input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {self.input_dim}.")
        if self.output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {self.output_dim}.")
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {self.hidden_dim}.")
        if self.num_layers < 0:
            raise ValueError(f"num_layers must be non-negative, got {self.num_layers}.")
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                f"hidden_dim={self.hidden_dim} must be divisible by num_heads={self.num_heads}."
            )
        if self.pos_embed_type not in {"fourier", "linear", "none"}:
            raise ValueError(
                f"Unsupported pos_embed_type: {self.pos_embed_type}. Expected: fourier, linear, or none."
            )
        if self.feature_layout not in {"fixed_fusion", "four_branch_equiv_invariant"}:
            raise ValueError(
                f"Unsupported feature_layout: {self.feature_layout}. Expected: fixed_fusion or "
                "four_branch_equiv_invariant."
            )
        if self.position_normalize not in {"none", "raw", "identity", "joint_isotropic", "shared_isotropic", "isotropic"}:
            raise ValueError(
                f"Unsupported position_normalize: {self.position_normalize}. "
                "Expected none or joint_isotropic."
            )
        if self.attention_isolation not in {"none", "protect_invariant"}:
            raise ValueError(
                f"Unsupported attention_isolation: {self.attention_isolation}. "
                "Expected: none or protect_invariant."
            )
        if self.attention_isolation == "protect_invariant" and self.feature_layout != "four_branch_equiv_invariant":
            raise ValueError(
                "attention_isolation=protect_invariant requires "
                "feature_layout=four_branch_equiv_invariant."
            )
        if self.scale_injection not in {"film", "add"}:
            raise ValueError(
                f"Unsupported scale_injection: {self.scale_injection}. Expected: film or add."
            )
        if self.scale_feature_mode not in {
            "joint_radius",
            "part_radii_center_distance_joint_radius",
        }:
            raise ValueError(
                f"Unsupported scale_feature_mode: {self.scale_feature_mode}. Expected: "
                "joint_radius or part_radii_center_distance_joint_radius."
            )
        if self.scale_reference <= 0:
            raise ValueError(f"scale_reference must be positive, got {self.scale_reference}.")
        if self.scale_log_clip <= 0:
            raise ValueError(f"scale_log_clip must be positive, got {self.scale_log_clip}.")
        if self.deepstack_num_layers < 0:
            raise ValueError(f"deepstack_num_layers must be non-negative, got {self.deepstack_num_layers}.")
        self.deepstack_layer_indexes = self._resolve_deepstack_layer_indexes()

        # Project each flattened patch vector [C * 3] to bridge hidden size H.
        # The sequence length is 2P in the default layout and 4P in the new layout.
        self.input_proj = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        #  type embedding checkpoint
        #  patch  type embedding
        num_feature_types = 4 if self.feature_layout == "four_branch_equiv_invariant" else 2
        self.type_embed = nn.Embedding(num_feature_types, self.hidden_dim)

        # Scale features are injected after VN-RMSNorm, so they cannot be
        # normalized away upstream. The legacy mode keeps one joint radius for
        # checkpoint compatibility. The multi-geometry mode separates the two
        # part sizes, their center distance, and the joint normalization radius.
        if self.scale_encoding_enabled:
            scale_feature_dim = (
                1
                if self.scale_feature_mode == "joint_radius"
                else 4
            )
            scale_output_dim = self.hidden_dim * 2 if self.scale_injection == "film" else self.hidden_dim
            self.scale_encoder = nn.Sequential(
                nn.Linear(scale_feature_dim, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, scale_output_dim),
            )
            nn.init.normal_(self.scale_encoder[-1].weight, mean=0.0, std=1e-3)
            nn.init.zeros_(self.scale_encoder[-1].bias)
        else:
            self.scale_encoder = None

        if self.pos_embed_type == "fourier":
            # Centers are [B, P, 3]. Fourier encoding expands xyz to
            # [B, P, 3, 2 * bands] and then flattens to [B, P, 6 * bands].
            pos_input_dim = 3 * 2 * self.fourier_bands
            self.register_buffer(
                "fourier_freqs",
                torch.linspace(1.0, self.fourier_max_freq, self.fourier_bands),
                persistent=False,
            )
            self.pos_proj = nn.Sequential(
                nn.Linear(pos_input_dim, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
        elif self.pos_embed_type == "linear":
            self.register_buffer("fourier_freqs", torch.empty(0), persistent=False)
            # Linear 3D position path: [B, P, 3] -> [B, P, H].
            self.pos_proj = nn.Linear(3, self.hidden_dim)
        else:
            self.register_buffer("fourier_freqs", torch.empty(0), persistent=False)
            self.pos_proj = None

        # Use an explicit ModuleList so we can expose selected intermediate
        # bridge layer states as point DeepStack residuals.
        self.transformer_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=self.hidden_dim,
                    nhead=self.num_heads,
                    dim_feedforward=self.ffn_dim,
                    dropout=self.dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(self.num_layers)
            ]
        )

        # Final projection converts bridge hidden tokens [B, 2P, H] to Qwen/VLM
        # token embeddings [B, 2P, D].
        self.output_proj = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.output_dim),
        )
        if self.deepstack_enabled and len(self.deepstack_layer_indexes) > 0:
            # Qwen3-VL DeepStack adds residual features into the first K decoder
            # layers at the same token positions. Each projection maps one
            # selected bridge intermediate state [B, 2P, H] to the corresponding
            # Qwen decoder residual [B, 2P, D].
            self.deepstack_output_projs = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.LayerNorm(self.hidden_dim),
                        nn.Linear(self.hidden_dim, self.output_dim),
                    )
                    for _ in self.deepstack_layer_indexes
                ]
            )
            self.deepstack_gates = nn.Parameter(
                torch.full((len(self.deepstack_layer_indexes),), self.deepstack_init_scale)
            )
        else:
            self.deepstack_output_projs = nn.ModuleList()
            self.deepstack_gates = None

    @staticmethod
    def _parse_layer_indexes(value):
        if value is None:
            return None
        if isinstance(value, str):
            value = value.strip()
            if value == "" or value.lower() in {"auto", "none", "null"}:
                return None
            return [int(item.strip()) for item in value.split(",") if item.strip() != ""]
        if isinstance(value, (list, tuple)):
            return [int(item) for item in value]
        raise TypeError(f"deepstack_layer_indexes must be a list, comma-separated string, or auto. Got {type(value)}.")

    def _resolve_deepstack_layer_indexes(self):
        if not self.deepstack_enabled or self.deepstack_num_layers <= 0 or self.num_layers <= 0:
            return []

        if self.deepstack_layer_indexes is not None:
            indexes = self.deepstack_layer_indexes
            invalid = [idx for idx in indexes if idx < 0 or idx >= self.num_layers]
            if invalid:
                raise ValueError(
                    f"deepstack_layer_indexes={indexes} contains invalid indexes {invalid}; "
                    f"valid range is [0, {self.num_layers - 1}]."
                )
            if len(set(indexes)) != len(indexes):
                raise ValueError(f"deepstack_layer_indexes must be unique, got {indexes}.")
            return indexes[: self.deepstack_num_layers]

        count = min(self.deepstack_num_layers, self.num_layers)
        if count == self.num_layers:
            return list(range(self.num_layers))

        # Match the Qwen3-VL idea of taking progressive intermediate features
        # before the final representation. For example, depth=24 and count=3
        # gives [5, 11, 17], which matches the local Qwen3-VL-2B visual config.
        indexes = []
        for layer_idx in range(count):
            idx = int((layer_idx + 1) * self.num_layers / (count + 1)) - 1
            idx = max(0, min(self.num_layers - 1, idx))
            while idx in indexes and idx < self.num_layers - 1:
                idx += 1
            while idx in indexes and idx > 0:
                idx -= 1
            indexes.append(idx)
        return sorted(indexes)

    def _encode_positions(self, centers):
        # centers: [B, P, 3] for one branch or [B, 2P, 3] after concat.
        if self.pos_embed_type == "none" or centers is None:
            return None
        centers = centers.to(dtype=self.input_proj[1].weight.dtype)
        if self.pos_embed_type == "linear":
            # [B, P, 3] or [B, 2P, 3] -> [B, P, H] or [B, 2P, H].
            return self.pos_proj(centers)
        freqs = self.fourier_freqs.to(device=centers.device, dtype=centers.dtype)
        # [B, P, 3] -> [B, P, 3, bands].
        phase = centers.unsqueeze(-1) * freqs.view(1, 1, 1, -1)
        # [B, P, 3, bands] -> [B, P, 3, 2 * bands].
        encoded = torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)
        # [B, P, 3, 2 * bands] -> [B, P, 6 * bands] -> [B, P, H].
        encoded = encoded.flatten(-2)
        return self.pos_proj(encoded)

    def forward(
        self,
        fixed_patch_equiv,
        fusion_patch_equiv,
        fixed_patch_centers=None,
        fusion_patch_centers=None,
        return_deepstack: bool = False,
        fixed_patch_invariant=None,
        moving_patch_invariant=None,
        fixed_patch_invariant_centers=None,
        moving_patch_invariant_centers=None,
    ):
        if fixed_patch_equiv.ndim != 4 or fusion_patch_equiv.ndim != 4:
            raise ValueError(
                "PointPatchFusionBridge expects patch features shaped [B, P, C, 3]. "
                f"Got fixed={tuple(fixed_patch_equiv.shape)}, fusion={tuple(fusion_patch_equiv.shape)}."
            )
        if fixed_patch_equiv.shape != fusion_patch_equiv.shape:
            raise ValueError(
                "fixed_patch_equiv and fusion_patch_equiv must have the same shape. "
                f"Got {tuple(fixed_patch_equiv.shape)} and {tuple(fusion_patch_equiv.shape)}."
            )
        if fixed_patch_equiv.size(-1) != 3:
            raise ValueError(f"Expected 3D vector axis, got shape {tuple(fixed_patch_equiv.shape)}.")

        batch_size, num_patches, _, _ = fixed_patch_equiv.shape
        scale_features = None
        if self.scale_encoding_enabled:
            if fixed_patch_centers is None or fusion_patch_centers is None:
                raise ValueError(
                    "scale_encoding_enabled=true requires fixed and moving patch centers."
                )
            #  z0
            #  VN-RMSNorm
            if self.scale_feature_mode == "joint_radius":
                _, joint_radius = point_center_pair_origin_radius(
                    fixed_patch_centers,
                    fusion_patch_centers,
                )
                scale_features = joint_radius.squeeze(-1)
            else:
                scale_features = point_center_pair_scale_features(
                    fixed_patch_centers,
                    fusion_patch_centers,
                )
        if self.feature_layout == "four_branch_equiv_invariant":
            if fixed_patch_invariant is None or moving_patch_invariant is None:
                raise ValueError(
                    "feature_layout=four_branch_equiv_invariant requires both "
                    "fixed_patch_invariant and moving_patch_invariant."
                )
            if fixed_patch_invariant_centers is None or moving_patch_invariant_centers is None:
                raise ValueError(
                    "feature_layout=four_branch_equiv_invariant requires canonical patch centers "
                    "for both invariant branches."
                )
            expected_shape = fixed_patch_equiv.shape
            if fixed_patch_invariant.shape != expected_shape or moving_patch_invariant.shape != expected_shape:
                raise ValueError(
                    "All four patch branches must have the same [B, P, C, 3] shape. "
                    f"Got fixed_equiv={tuple(expected_shape)}, "
                    f"fixed_invariant={tuple(fixed_patch_invariant.shape)}, "
                    f"moving_equiv={tuple(fusion_patch_equiv.shape)}, "
                    f"moving_invariant={tuple(moving_patch_invariant.shape)}."
                )
            branch_features = [
                fixed_patch_equiv,
                fixed_patch_invariant,
                fusion_patch_equiv,
                moving_patch_invariant,
            ]
            # [fixed-eq, fixed-inv, moving-eq, moving-inv] [B,P,C,3]
            tokens = torch.cat([feature.flatten(2) for feature in branch_features], dim=1)
            type_ids_1d = torch.arange(4, dtype=torch.long, device=tokens.device).repeat_interleave(
                num_patches
            )
            if fixed_patch_centers is not None and fusion_patch_centers is not None:
                expected_center_shape = fixed_patch_centers.shape
                if (
                    fusion_patch_centers.shape != expected_center_shape
                    or fixed_patch_invariant_centers.shape != expected_center_shape
                    or moving_patch_invariant_centers.shape != expected_center_shape
                ):
                    raise ValueError(
                        "All four patch-center branches must have the same [B, P, 3] shape. "
                        f"Got fixed_equiv={tuple(expected_center_shape)}, "
                        f"fixed_invariant={tuple(fixed_patch_invariant_centers.shape)}, "
                        f"moving_equiv={tuple(fusion_patch_centers.shape)}, "
                        f"moving_invariant={tuple(moving_patch_invariant_centers.shape)}."
                    )
                #  fixed/moving
                #
                fixed_position_centers, moving_position_centers = normalize_point_center_pair(
                    fixed_patch_centers,
                    fusion_patch_centers,
                    mode=self.position_normalize,
                )
                fixed_invariant_position_centers, moving_invariant_position_centers = normalize_point_center_pair(
                    fixed_patch_invariant_centers,
                    moving_patch_invariant_centers,
                    mode=self.position_normalize,
                )
                centers = torch.cat(
                    [
                        fixed_position_centers,
                        fixed_invariant_position_centers,
                        moving_position_centers,
                        moving_invariant_position_centers,
                    ],
                    dim=1,
                )
            else:
                centers = None
        else:
            # [fixed equiv, Gb @ moving equiv]
            fixed_tokens = fixed_patch_equiv.flatten(2)
            fusion_tokens = fusion_patch_equiv.flatten(2)
            tokens = torch.cat([fixed_tokens, fusion_tokens], dim=1)
            type_ids_1d = torch.cat(
                [
                    torch.zeros(num_patches, dtype=torch.long, device=tokens.device),
                    torch.ones(num_patches, dtype=torch.long, device=tokens.device),
                ],
                dim=0,
            )
            centers = None
            if fixed_patch_centers is not None and fusion_patch_centers is not None:
                fixed_position_centers, moving_position_centers = normalize_point_center_pair(
                    fixed_patch_centers,
                    fusion_patch_centers,
                    mode=self.position_normalize,
                )
                centers = torch.cat([fixed_position_centers, moving_position_centers], dim=1)

        if tokens.size(-1) != self.input_dim:
            raise ValueError(
                f"Patch token dim {tokens.size(-1)} does not match bridge input_dim={self.input_dim}."
            )

        # [B, K*P, C*3] -> [B, K*P, H]K  2 4
        hidden = self.input_proj(tokens)
        type_ids = type_ids_1d.unsqueeze(0).expand(batch_size, -1)
        hidden = hidden + self.type_embed(type_ids)

        if centers is not None:
            pos_embed = self._encode_positions(centers)
            if pos_embed is not None:
                hidden = hidden + pos_embed.to(dtype=hidden.dtype)

        if self.scale_encoder is not None:
            # [B,S] -> log(scale/reference), where S is 1 in legacy mode and 4
            # in multi-geometry mode. Logarithms turn multiplicative unit-scale
            # changes into additive offsets. Raw statistics were clamped above,
            # and this final clamp bounds unusual samples before the scale MLP.
            log_scale_features = torch.log(scale_features / self.scale_reference)
            log_scale_features = log_scale_features.clamp(
                -self.scale_log_clip,
                self.scale_log_clip,
            )
            scale_params = self.scale_encoder(
                log_scale_features.to(
                    device=hidden.device,
                    dtype=self.scale_encoder[0].weight.dtype,
                )
            ).to(dtype=hidden.dtype)
            if self.scale_injection == "film":
                scale_gamma, scale_beta = scale_params.chunk(2, dim=-1)
                hidden = hidden * (1.0 + scale_gamma.unsqueeze(1)) + scale_beta.unsqueeze(1)
            else:
                hidden = hidden + scale_params.unsqueeze(1)

        branch_attention_mask = None
        if self.attention_isolation == "protect_invariant":
            # type 0/2  fixed/moving type 1/3
            # PyTorch bool src_mask  True  query  key
            is_equivariant = (type_ids_1d == 0) | (type_ids_1d == 2)
            is_invariant = (type_ids_1d == 1) | (type_ids_1d == 3)
            branch_attention_mask = is_invariant[:, None] & is_equivariant[None, :]

        # Cross-patch and cross-branch fusion in hidden space. We keep selected
        # intermediate layer outputs for point DeepStack residuals; each selected
        # hidden is [B, K*P, H], where K is 2 or 4 according to feature_layout.
        selected_hiddens = []
        selected_index_to_slot = {
            layer_idx: slot_idx for slot_idx, layer_idx in enumerate(self.deepstack_layer_indexes)
        }
        for layer_idx, layer in enumerate(self.transformer_layers):
            hidden = layer(hidden, src_mask=branch_attention_mask)
            if layer_idx in selected_index_to_slot:
                selected_hiddens.append((selected_index_to_slot[layer_idx], hidden))
        selected_hiddens = [hidden_state for _, hidden_state in sorted(selected_hiddens, key=lambda item: item[0])]

        deepstack_embeds = None
        if return_deepstack and self.deepstack_enabled and len(self.deepstack_output_projs) > 0:
            # List length equals the selected DeepStack layer count. Each entry
            # is [B, K*P, D] and will be flattened by
            # PVLM according to the actual <point> token positions.
            deepstack_embeds = [
                proj(layer_hidden) * self.deepstack_gates[layer_idx].to(dtype=layer_hidden.dtype)
                for layer_idx, (proj, layer_hidden) in enumerate(zip(self.deepstack_output_projs, selected_hiddens))
            ]
        # [B, K*P, H] -> [B, K*P, D], ready to replace point tokens in Qwen.
        output = self.output_proj(hidden)
        if return_deepstack:
            return output, deepstack_embeds
        return output
