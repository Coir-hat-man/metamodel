import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from torch import Tensor

class MoeDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_experts: int,
        args = None
    ):
        super().__init__()
        d_in = d_model * 2 if args.use_image and args.image_combine_mz == "cat_dim" else d_model
        self.moe = MoELayer(input_dim=d_in, 
                            expert_hidden_dim=d_model, 
                            output_dim=2*d_model if args.use_image and args.image_combine_mz=="cat_dim" else d_model, 
                            num_experts=num_experts)

    def forward(self, x: Tensor, topn: int = 2) -> Dict[str, Tensor]:
        """x is the output of the transformer, (batch, seq_len, d_model)"""
        recon = self.moe(x, topn).squeeze(-1)
        return recon
    


class MoELayer(nn.Module):
    def __init__(self, input_dim, expert_hidden_dim, output_dim, num_experts):
        super(MoELayer, self).__init__()
        self.experts = nn.ModuleList([Expert(input_dim, expert_hidden_dim, output_dim) for _ in range(num_experts)])
        self.gate = GatingNetwork(input_dim, num_experts)

    def forward(self, x, num_experts_per_tok):
        gating_scores = self.gate(x)
        topk_gating_scores, topk_indices = gating_scores.topk(num_experts_per_tok, dim=2, sorted=False)
        # Create a mask to zero out the contributions of non-topk experts
        mask = torch.zeros_like(gating_scores).scatter_(2, topk_indices, 1)
        # Use the mask to retain only the topk gating scores
        gating_scores = gating_scores * mask
        # Normalize the gating scores to sum to 1 across the selected top experts
        gating_scores = F.normalize(gating_scores, p=1, dim=2)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        expert_outputs = expert_outputs.transpose(1, 2)
        output = torch.einsum('bte,bteo->bto', gating_scores, expert_outputs)

        return output
    

class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Expert, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
    def forward(self, x):
        return self.fc(x)
    

class GatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts):
        super(GatingNetwork, self).__init__()
        self.gate = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        return F.softmax(self.gate(x), dim=2)