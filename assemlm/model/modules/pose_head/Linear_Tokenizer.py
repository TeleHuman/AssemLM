# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License"); 
# Implemented by [Jinhui YE / HKUST University] in [2025].

"""Fast Action Tokenizer Adapter
"this file is adapted from https://huggingface.co/physical-intelligence/fast"

Overview:
    This module encapsulates a lightweight "action → language model-readable sequence" converter (Fast_Action_Tokenizer).
    Its core objective is to convert continuous/discrete raw robot actions (raw_actions) into
    pseudo-natural language token strings like <assemble_pose_12><assemble_pose_3><assemble_pose_87> ...
    This facilitates direct integration into multimodal large models (VLM/LLM) dialogue templates,
    leveraging their language modeling capabilities for action prediction.
"""
import re 
import torch.nn as nn
from typing import List, Dict, Any, Callable, Optional
import os
import numpy as np
from transformers import AutoProcessor

from assemlm.model.modules.vlm.QWen3 import _ACTION_TOKEN_MAX, _ACTION_TOKEN_MIN

class Linear_Action_Tokenizer(nn.Module):
    """Linear Action Tokenizer that uniformly encodes values in [-1, 1] to tokens."""
    def __init__(self, vocab_size=200, expected_len=20, pad_token_id=2047, use_expected_len=True):
        super().__init__()
        self.vocab_size = vocab_size
        self.expected_len = expected_len
        self.pad_token_id = pad_token_id
        self.use_expected_len = use_expected_len
        self._ACTION_TOKEN_MIN = _ACTION_TOKEN_MIN
        self._ACTION_TOKEN_MAX = _ACTION_TOKEN_MAX

    def encoder_action2vlmtoken(self, raw_actions, expected_len=None):
        import ast
        if expected_len is None:
            expected_len = self.expected_len

        # raw_actions: list of strings like "[[x,y,z,...]]"
        raw_actions = [ast.literal_eval(action) for action in raw_actions]
        batch_actions = np.array(raw_actions)  # (B, T, D)
        
        B, T, D = batch_actions.shape
        batch_actions_flat = batch_actions.reshape(B, -1) # (B, T*D)
        
        # Uniform encoding from [-1, 1] to [0, vocab_size-1] with rounding
        batch_actions_flat = np.clip(batch_actions_flat, -1.0, 1.0)
        batch_tokens = np.round((batch_actions_flat + 1.0) / 2.0 * (self.vocab_size - 1)).astype(np.int64)
        
        batch_vlm_actions = []
        for tokens in batch_tokens:
            tokens_list = list(tokens)
            if self.use_expected_len:
                if len(tokens_list) < expected_len:
                    tokens_list = tokens_list + [self.pad_token_id] * (expected_len - len(tokens_list))
                elif len(tokens_list) > expected_len:
                    tokens_list = tokens_list[:expected_len]
            batch_vlm_actions.append(self.map_fast_token_to_vlm_action(tokens_list))
            
        return batch_vlm_actions # List[str]

    def decode(self, generated_ids):
        # generated_ids: usually 1D or 2D array of token IDs (VLM IDs)
        ids = np.array(generated_ids)
            
        # Subtract min to get tokens [0, vocab_size-1]
        tokens = ids
        tokens = np.clip(tokens, 0, self.vocab_size - 1)
        
        # Decode to [-1, 1]
        values = tokens.astype(np.float32) / (self.vocab_size - 1) * 2.0 - 1.0
        
        return values

    def decoder_action(self, generated_ids):
        # generated_ids: usually 1D or 2D array of token IDs (VLM IDs)
        ids = np.array(generated_ids)
            
        # Subtract min to get tokens [0, vocab_size-1]
        tokens = ids - self._ACTION_TOKEN_MIN
        tokens = np.clip(tokens, 0, self.vocab_size - 1)
        
        # Decode to [-1, 1]
        values = tokens.astype(np.float32) / (self.vocab_size - 1) * 2.0 - 1.0
        
        return values

    def map_fast_token_to_vlm_action(self, tokens) -> str:
        """Maps fast action tokens to the VLM action format."""
        return ''.join([f"<assemble_pose_{token}>" for token in tokens])

    def vlmtoken2action(self, vlm_string: str):
        """Parses a VLM output string back to raw actions."""
        token_ids = [int(x) for x in re.findall(r"<assemble_pose_(\d+)>", vlm_string)]
        if not token_ids:
            return None
        return self.decoder_action(np.array(token_ids) + self._ACTION_TOKEN_MIN)

    def fit_tokenizer_on_datasets(self, action_dataset, datasets_path="<your_local_path>"):
        """Placeholder for linear tokenizer which doesn't need fitting."""
        pass

def start_debugpy_once():
    """start debugpy once"""
    import debugpy
    if getattr(start_debugpy_once, "_started", False):
        return
    debugpy.listen(("0.0.0.0", 10094))
    print("🔍 Waiting for VSCode attach on 0.0.0.0:10094 ...")
    debugpy.wait_for_client()
    start_debugpy_once._started = True

if __name__ == "__main__":
    pass




