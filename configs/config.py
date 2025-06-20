
small = {
    "d_model": 256,
    "decoder_hidden_dim": 128,
    "transformer_decoder_heads": 4,
    "transformer_decoder_layers": 4,
    "moe_experts": 4,
}

base = {
    "d_model": 512,
    "decoder_hidden_dim": 256,
    "transformer_decoder_heads": 8,
    "transformer_decoder_layers": 12,
    "moe_experts": 4,
}

large = {
    "d_model": 768,
    "decoder_hidden_dim": 512,
    "transformer_decoder_heads": 12,
    "transformer_decoder_layers": 16,
    "moe_experts": 8,
}

xlarge = {
    "d_model": 1024,
    "decoder_hidden_dim": 1024,
    "transformer_encoder_heads": 16,
    "transformer_encoder_layers": 16,
    "transformer_decoder_heads": 16,
    "transformer_decoder_layers": 16,
    "moe_experts": 8,
}

billion = {
    "d_model": 1536,                          # 嵌入维度
    "decoder_hidden_dim": 6144,              # FFN 隐藏层 = 4 * d_model 是标准做法
    "transformer_decoder_heads": 24,         # 注意力头数（最好能整除 d_model）
    "transformer_decoder_layers": 36,        # 解码器层数
    "moe_experts": 16,                        # MoE专家数（可以稀疏激活，节省推理成本）
    "transformer_encoder_heads": 24,         # 如果 encoder 是 ViT/Transformer
    "transformer_encoder_layers": 36         # 如果你想对称 encoder
}

model_configs = {
    "small": small,
    "base": base,
    "large": large,
    "xlarge": xlarge,
    "billion":billion,
}
