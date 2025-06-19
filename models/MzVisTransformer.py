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
