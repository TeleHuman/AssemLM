# Modified from OpenAI's diffusion repos
#     GLIDE: https://github.com/openai/glide-text2im/blob/main/glide_text2im/gaussian_diffusion.py
#     ADM:   https://github.com/openai/guided-diffusion/blob/main/guided_diffusion
#     IDDPM: https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py


from .Linear_Tokenizer import Linear_Action_Tokenizer


def get_action_model(config=None):
    """
    Factory: build ActionModel from global framework config.

    Args:
        config: Global config (expects config.framework.action_model namespace).
    Returns:
        ActionModel: Initialized diffusion action head.
    """
    fast_tokenizer_name = config["framework"]["action_model"]["model_path"]
    expected_len = config["framework"]["action_model"].get("expected_len", 20)
    pad_token_id = config["framework"]["action_model"].get("pad_token_id", 2047)
    use_expected_len = config["framework"]["action_model"].get("use_expected_len", True)
    tokenizer_type = config["framework"]["action_model"].get("type", "fast")
    tokenizer_vocab_size = config["framework"]["action_model"].get("tokenizer_vocab_size", 2047)

    if tokenizer_type == "linear":
        print("Using Linear Action Tokenizer")
        print(f"Vocab Size: {tokenizer_vocab_size}, Expected Len: {expected_len}, Pad Token ID: {pad_token_id}, Use Expected Len: {use_expected_len}")
        action_model = Linear_Action_Tokenizer(
            vocab_size=tokenizer_vocab_size,
            expected_len=expected_len,
            pad_token_id=pad_token_id,
            use_expected_len=use_expected_len
        )
        
    return action_model