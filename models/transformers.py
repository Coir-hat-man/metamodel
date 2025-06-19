import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union
from torch.distributions import Bernoulli

class DenseTransformerLayer(nn.Module):
    def __init__(
            self, 
            d_model: int, 
            hidden_dim: int, 
            num_heads: int, 
            dropout=0.1):
        
        super(DenseTransformerLayer, self).__init__()

        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout

        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        possible_input_dims = [d_model * (i + 1) for i in range(12)]
        self.proj_layers = nn.ModuleDict()
        for in_dim in possible_input_dims:
            key = f"{in_dim}->{d_model}"
            self.proj_layers[key] = nn.Linear(in_dim, d_model)

    # def forward(self, x, previous_outputs):
    #     """
    #     x: current input tensor [batch_size, seq_len, d_model]
    #     previous_outputs: list of previous layer outputs (including original input)
    #     """
    #     # Dense connection: concat all previous outputs

    #     dense_input = torch.cat(previous_outputs + [x], dim=-1)

    #     in_dim = dense_input.size(-1)
    #     out_dim = x.size(-1)

    #     key = f"{in_dim}->{out_dim}"
    #     if key not in self.proj_layers:
    #         self.proj_layers[key] = nn.Linear(in_dim, out_dim).to(x.device)

    #     # Project to expected d_model
    #     projected_input = self.proj_layers[key](dense_input)

    #     # Self-Attention
    #     attn_output, _ = self.self_attn(projected_input, projected_input, projected_input)
    #     x = self.norm1(projected_input + self.dropout(attn_output))

    #     # Feed-forward network
    #     ffn_output = self.ffn(x)
    #     output = self.norm2(x + self.dropout(ffn_output))

    #     return output
    
    def forward(self, previous_outputs):
        """
        x: current input tensor [batch_size, seq_len, d_model]
        previous_outputs: list of previous layer outputs (including original input)
        """
        # Dense connection: concat all previous outputs

        out_dim = previous_outputs[-1].size(-1)
        dense_input = torch.cat(previous_outputs, dim=-1)
        in_dim = dense_input.size(-1)


        key = f"{in_dim}->{out_dim}"
        if key not in self.proj_layers:
            self.proj_layers[key] = nn.Linear(in_dim, out_dim).to(previous_outputs[-1].device)

        # Project to expected d_model
        projected_input = self.proj_layers[key](dense_input)

        # Self-Attention
        attn_output, _ = self.self_attn(projected_input, projected_input, projected_input)
        x = self.norm1(projected_input + self.dropout(attn_output))

        # Feed-forward network
        ffn_output = self.ffn(x)
        output = self.norm2(x + self.dropout(ffn_output))

        return output
    
class DenseTransformer(nn.Module):
    def __init__(
            self, 
            d_model: int, 
            hidden_dim: int, 
            num_heads: int, 
            num_layers: int,
            dropout=0.1,
            args = None
            ):
        
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList([DenseTransformerLayer(
            d_model*2 if args.use_image and args.image_combine_mz=="cat_dim" else d_model, 
            hidden_dim, 
            num_heads, 
            dropout,
            ) for _ in range(num_layers)])
    
    def forward(self, x):
        outputs = [x]
        for layer in self.layers:
            # out = layer(outputs[-1], outputs)
            out = layer(outputs)
            outputs.append(out)
        final_output = outputs[-1]
        return final_output
    
