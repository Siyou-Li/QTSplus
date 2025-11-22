from .tokenizer import QTSplusTokenizer, QTSplusTokenizerConfig

def build_qts_plus_tower(config, **kwargs):
    # Use LM head count directly; enforce divisibility with vision embedding dim
    lm_heads = getattr(config, "num_attention_heads", None)
    vision_dim = getattr(config, "vision_embed_size", None)
    if not isinstance(lm_heads, int) or lm_heads <= 0:
        raise ValueError("num_attention_heads must be provided by the Qwen2.5-VL config")
    if not isinstance(vision_dim, int) or vision_dim <= 0:
        raise ValueError("vision_embed_size must be a positive int before building QTS+")
    if vision_dim % lm_heads != 0:
        raise ValueError(
            f"vision_embed_size ({vision_dim}) must be divisible by LM num_attention_heads ({lm_heads})"
        )
    n_heads_eff = lm_heads

    # Try to honor LM's multi-query kv heads if provided
    kv_heads = getattr(config, "num_key_value_heads", None)

    cfg = QTSplusTokenizerConfig(
        embedding_dim = config.vision_embed_size,
        n_heads = n_heads_eff,
        num_kv_heads = kv_heads if isinstance(kv_heads, int) and kv_heads > 0 else None,
        tau_s = getattr(config, "qts_plus_tau_s", 0.1),
        nmax = getattr(config, "qts_plus_nmax", 2560),
        rho_min = getattr(config, "qts_plus_rho_min", 0.05),
        rho_max = getattr(config, "qts_plus_rho_max", 0.5),
        block_dropout = getattr(config, "qts_plus_block_dropout", 0.0),
        reencode = getattr(config, "qts_plus_reencode", True),
        scoring_layers = getattr(config, "qts_plus_scoring_layers", 1),
        reencode_layers = getattr(config, "qts_plus_reencode_layers", 1),
        lambda_t = getattr(config, "lambda_t", 1.0),
        lambda_m = getattr(config, "lambda_m", 1.7),
        lambda_s = getattr(config, "lambda_s", 0.05),
        project_text_if_needed = getattr(config, "project_text_if_needed", False),
    )
    return QTSplusTokenizer(cfg)
