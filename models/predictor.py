import torch
import torch.nn as nn
from torch import Tensor
from typing import Mapping, Dict, Union, Optional
from torch.distributions import Bernoulli
import torch.nn.functional as F

# Spatial metabolomics predictor
class STMPredictor(nn.Module):
    def __init__(self, 
                 d_model: int = 512, 
                 emb_type: str = "cls",
                 use_cell_size = True,
                 model_style= "attn", #["attn", "mlp"]
                 query_activation: nn.Module = nn.Sigmoid,
                 args=None
            ):
        super().__init__()
        assert emb_type in ["cls", "all"], 'Invalid embedding type! Use "cls" or "all"'
        # assert pred_method in ["inner_product", "cross_attn"], 'Invalid pred_method! Use "inner_product" or "cross_attn"'

        # if emb_type == "all" and pred_method == "inner product":
        #     raise ValueError('"all" embedding type only supports "cross_attn" method.')
        # if emb_type == "cls" and pred_method == "cross_attn":
        #     raise ValueError('"cls" embedding type only supports "inner_product" method.')
        
        self.emb_type = emb_type
        self.model_style = model_style
        self.d_in = d_model
        self.use_cell_size = use_cell_size

        if args.use_image and args.image_combine_mz=="cat_dim":
            self.d_in += d_model
        if self.use_cell_size:
            self.d_in += d_model
        
        if self.emb_type == "cls":
            self.predictor = InnerProductDecoder(
                d_model,
                self.d_in, 
                query_activation=query_activation)
        else:
            if self.model_style == "attn":
                self.predictor = CrossAttnDecoder(
                    d_model,
                    self.d_in
                )
            elif self.model_style == "mlp":
                self.predictor = CellFMecoder(self.d_in, dropout=0.0, zero=False)

    def _get_emb_for_pred(
            self, 
            cell_emb: Tensor,
        ) -> Tensor:
        """
        Args:
            cell_emb(:obj:`Tensor`): shape (B, L+1, Din),
        """

        # 应对neighbor的情况，把neighbor沿着第1维取平均
        if len(cell_emb.shape) == 4: # [B, K，L，Din]
            cell_emb = cell_emb.mean(dim=1) # [B, L，Din]


        if self.emb_type == "cls":
            cell_emb = cell_emb[:, 0, :]  # [B, Din]
        else:
            cell_emb = cell_emb[:, 1:, :] # [B, L, Din]
        return cell_emb
    


    def _fuse_protocol_info(self, cell_emb, cell_size_emb):
        if self.emb_type == "cls":
            return torch.cat([cell_emb , cell_size_emb[:,0,:]], dim=1)
        else:
            return torch.cat([cell_emb , cell_size_emb[:,1:,:]], dim=2)
                            

    def forward(self, 
                cell_emb, 
                gene_embs, 
                cell_size_emb,
                args=None
        ):
        
        cell_emb = self._get_emb_for_pred(cell_emb)
        cell_emb = self._fuse_protocol_info(cell_emb, cell_size_emb)
        if self.emb_type=="all" and self.model_style == "mlp":
            pred_value = self.predictor(cell_emb)
        else:
            pred_value = self.predictor(cell_emb, gene_embs[:,1:,:])# 不使用gene_embs的cls
        
        return pred_value


class InnerProductDecoder(nn.Module):
    def __init__(
        self,
        d_model,
        d_in, 
        query_activation: nn.Module = nn.Sigmoid,
    ) -> None:
        
        super().__init__()
        self.gene2query = nn.Linear(d_model, d_model) # [B, L, Dmodel] -> [B, L, Dmodel]
        self.query_activation = query_activation()
        self.W = nn.Linear(d_model, d_in, bias=False) # [B, L, Dmodel] -> [B, L, Din]

    def forward(
        self, cell_emb: Tensor, gene_embs: Tensor, args=None) -> Union[Tensor, Dict[str, Tensor]]:
        query_vecs = self.query_activation(self.gene2query(gene_embs)) #[B, L, D]
        cell_emb = cell_emb.unsqueeze(2)  # [B, d_in, 1]
        pred_value = torch.bmm(self.W(query_vecs), cell_emb).squeeze(2) # [B, L]
        return pred_value
    
class CrossAttnDecoder(nn.Module):
    def __init__(
        self,
        d_model,
        d_in
    ):
        super().__init__()

        self.mlp1 = MLPBeforeCrossAttn(d_model, d_in)
        self.cross_attn = nn.MultiheadAttention(
                embed_dim=d_model, num_heads=8, batch_first=True
        )
        self.mlp2 = MLPAfterCrossAttn(d_model)

    def forward(
        self, cell_emb: Tensor, gene_embs: Tensor, args=None):
        cell_emb = self.mlp1(cell_emb)
        attn_output, _= self.cross_attn(query=gene_embs, key=cell_emb, value=cell_emb)
        pred_value = self.mlp2(attn_output).squeeze(2)
        return pred_value
    
    
class MLPBeforeCrossAttn(nn.Module):
    def __init__(
        self,
        d_model,
        d_in, 
    ):
        super().__init__()
        self.net = nn.Sequential(
        nn.Linear(d_in, d_model),
        nn.LayerNorm(d_model),
        nn.GELU(),
        nn.Linear(d_model, d_model)
    )
    def forward(self, x):

        return self.net(x)
    

class MLPAfterCrossAttn(nn.Module):
    def __init__(
        self,
        d_model,
    ):
        super().__init__()
        self.net = nn.Sequential(
        nn.Linear(d_model, d_model),
        nn.LayerNorm(d_model),
        nn.GELU(),
        nn.Linear(d_model, 1)
    )
    def forward(self, x):
        return self.net(x)
    

class CellFMecoder(nn.Module):
    def __init__(self, emb_dims, dropout=0.0, zero=False):
        super(CellFMecoder, self).__init__()
        self.zero = zero
        self.linear1 = nn.Linear(emb_dims, emb_dims, bias=True)
        self.activation = nn.LeakyReLU()
        self.linear2 = nn.Linear(emb_dims, 1, bias=True)

        if self.zero:
            self.zero_logit = nn.Sequential(
                nn.Linear(emb_dims, emb_dims),
                nn.LeakyReLU(),
                nn.Linear(emb_dims, emb_dims),
                nn.LeakyReLU(),
                nn.Linear(emb_dims, 1),
                nn.Sigmoid()
            )

    def forward(self, expr_emb):
        """
        expr_emb: Tensor of shape [B, L, D]
        """
        B, L, D = expr_emb.shape

        x = self.linear1(expr_emb)             # [B, L, D]
        x = self.activation(x)
        pred = self.linear2(x).squeeze(-1)     # [B, L]

        if not self.zero:
            return pred
        else:
            zero_prob = self.zero_logit(expr_emb).squeeze(-1)  # [B, L]
            return pred, zero_prob
    


