import torch
import torch.nn as nn

class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, kv, mask=None):
        # mask: [B, L_kv] -> key_padding_mask
        q_norm = self.norm(q)
        kv_norm = self.norm(kv)

        key_padding_mask = mask  # shape: [B, L_kv] or None

        out, _ = self.attn(q_norm, kv_norm, kv_norm, key_padding_mask=key_padding_mask)
        return q + self.dropout(out)  # residual

class SelfAttentionLayer(nn.Module):
    def __init__(self, d_model, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # mask: [B, L] -> key_padding_mask
        x_res = x
        x = self.norm1(x) 
        attn_out, _ = self.attn(x, x, x, key_padding_mask=mask)
        x = x_res + self.dropout(attn_out)

        # FFN
        x_res = x
        x = self.norm2(x)
        x = x_res + self.dropout(self.ffn(x))
        return x


class MzVisTransformer(nn.Module):
    def __init__(self, d_model, hidden_dim, num_heads, num_cross_layers, num_self_layers, dropout=0.1):
        super().__init__()
        self.cross_layers = nn.ModuleList([
            CrossAttentionLayer(d_model, num_heads, dropout)
            for _ in range(num_cross_layers)
        ])
        self.self_layers = nn.ModuleList([
            SelfAttentionLayer(d_model, hidden_dim, num_heads, dropout)
            for _ in range(num_self_layers)
        ])

    def forward(self, mz_embed, vis_embed=None, attn_mask = None):
        x = mz_embed  # query

        # Cross-attention block (if vis_embed is provided)
        if vis_embed is not None:
            for layer in self.cross_layers:
                x = layer(x, vis_embed, mask=attn_mask)

        # Self-attention block
        for layer in self.self_layers:
            x = layer(x, mask=attn_mask)

        return x

import torch
from flash_attn.bert_padding import unpad_input, pad_input
from flash_attn.modules.mha import FlashSelfAttention,FlashCrossAttention

def flash_attention_with_padding(qkv_proj, flash_attn_module, x, padding_mask, num_heads):
    """
    Args:
        qkv_proj: nn.Linear (D -> 3*D)
        flash_attn_module: FlashSelfAttention 或 FlashCrossAttention
        x: [B, L, D]
        padding_mask: [B, L] (bool), True 为 padding
        num_heads: int

    Returns:
        x_attn: [B, L, D], padding 0
    """
    B, L, D = x.shape
    head_dim = D // num_heads
    x = x.to(torch.float16)  # FlashAttention 只支持 float16 / bfloat16

    # ========== Step 1: Unpad ==========
    x_unpad, indices, cu_seqlens, max_seqlen, _ = unpad_input(x, ~padding_mask)

    # ========== Step 2: QKV ==========
    qkv = qkv_proj(x_unpad)  # [total_tokens, 3*D]
    qkv = qkv.view(-1, 3, num_heads, head_dim)

    # ========== Step 3: Flash Attention ==========
    out_unpad = flash_attn_module(qkv, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
    out_unpad = out_unpad.view(-1,D)
    # ========== Step 4: Pad back ==========
    out = pad_input(out_unpad, indices, B, L)

    return out

class FlashSelfAttentionLayer(nn.Module):
    def __init__(self, d_model, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.flash_attn = FlashSelfAttention(causal=False)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model)
        )
        self.num_heads = num_heads  # or configurable

    def forward(self, x, mask=None):
        x_res = x
        x = self.norm1(x)

        if mask is not None:
            # Flash attention with unpadding
            attn_out = flash_attention_with_padding(
                self.qkv_proj, self.flash_attn, x, mask.bool(), self.num_heads
            )
        else:
            # No mask, full FlashAttention
            qkv = self.qkv_proj(x).view(x.size(0), x.size(1), 3, self.num_heads, -1)
            q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
            attn_out = self.flash_attn(q, k, v)

        x = x_res + self.dropout(self.out_proj(attn_out))

        # FFN
        x_res = x
        x = self.norm2(x)
        x = x_res + self.dropout(self.ffn(x))
        return x


def flash_cross_attention_with_padding(q_proj, kv_proj, flash_attn_module, q, kv, kv_mask, num_heads):
    """
    Args:
        q_proj: nn.Linear, projects query -> [B, L_q, D]
        kv_proj: nn.Linear, projects kv -> [B, L_kv, 2*D]
        flash_attn_module: FlashCrossAttention(causal=False)
        q: [B, L_q, D]
        kv: [B, L_kv, D]
        kv_mask: [B, L_kv] (bool), True 为 padding
        num_heads: int

    Returns:
        out: [B, L_q, D]
    """
    B, L_q, D = q.shape
    _, L_kv, _ = kv.shape
    head_dim = D // num_heads

    q = q.to(torch.float16)
    kv = kv.to(torch.float16)

    kv_unpad, indices_kv, cu_seqlens_kv, max_seqlen_kv,_ = unpad_input(kv, ~kv_mask)

    # Step 2: project
    q_proj = q_proj(q)  # [B, L_q, D]
    kv_proj = kv_proj(kv_unpad)  # [total_kv_tokens, 2*D]

    # reshape
    q = q_proj.view(B, L_q, num_heads, head_dim)
    kv_proj = kv_proj.view(-1, 2, num_heads, head_dim)
    k, v = kv_proj[:, 0], kv_proj[:, 1]

    # Step 3: flash attention
    out_unpad = flash_attn_module(q, k, v, cu_seqlens_kv=cu_seqlens_kv, max_seqlen_kv=max_seqlen_kv)

    # Step 4: pad back
    out = out_unpad  # already [B, L_q, D]
    return out

class FlashCrossAttentionLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.kv_proj = nn.Linear(d_model, 2 * d_model)
        self.flash_attn = FlashCrossAttention(causal=False)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.num_heads = num_heads

    def forward(self, q, kv, mask=None):
        # q: [B, L_q, D], kv: [B, L_kv, D], mask: [B, L_kv]
        q_res = q
        q = self.norm_q(q)
        kv = self.norm_kv(kv)

        if mask is not None:
            out = flash_cross_attention_with_padding(
                self.q_proj, self.kv_proj, self.flash_attn,
                q, kv, mask.bool(), self.num_heads
            )
        else:
            # no mask: full sequence
            q_proj = self.q_proj(q).view(q.size(0), q.size(1), self.num_heads, -1)
            kv_proj = self.kv_proj(kv).view(kv.size(0), kv.size(1), 2, self.num_heads, -1)
            k, v = kv_proj[:, :, 0], kv_proj[:, :, 1]
            out = self.flash_attn(q_proj, k, v)

        return q_res + self.dropout(self.out_proj(out))
    

class FlashMzVisTransformer(nn.Module):
    def __init__(self, d_model, hidden_dim, num_heads, num_cross_layers, num_self_layers, dropout=0.1):
        super().__init__()
        self.cross_layers = nn.ModuleList([
            FlashCrossAttentionLayer(d_model, num_heads, dropout)
            for _ in range(num_cross_layers)
        ])
        self.self_layers = nn.ModuleList([
            FlashSelfAttentionLayer(d_model, hidden_dim, num_heads, dropout)
            for _ in range(num_self_layers)
        ])

    def forward(self, mz_embed, vis_embed=None, attn_mask = None):
        x = mz_embed  # query

        # Cross-attention block (if vis_embed is provided)
        if vis_embed is not None:
            for layer in self.cross_layers:
                x = layer(x, vis_embed, mask=attn_mask)

        # Self-attention block
        for layer in self.self_layers:
            x = layer(x, mask=attn_mask)

        return x