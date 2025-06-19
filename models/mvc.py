
import torch
import torch.nn as nn
from torch import Tensor
from typing import Mapping, Dict, Union, Optional
from torch.distributions import Bernoulli
import torch.nn.functional as F


class BatchAwareMVCDecoder(nn.Module):
    def __init__(self, d_model, use_cross_attn=True, use_modulation=True, query_activation=nn.Sigmoid, args=None):
        super().__init__()
        self.use_cross_attn = use_cross_attn
        self.use_modulation = use_modulation

        self.gene2query = nn.Linear(d_model, d_model)
        self.query_activation = query_activation()
        self.W = nn.Linear(d_model, d_model, bias=False)

        if use_cross_attn:
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=d_model, num_heads=8, batch_first=True
            )

        if use_modulation:
            self.mod_gamma = nn.Linear(d_model, 1)  # broadcast over seq_len
            self.mod_beta = nn.Linear(d_model, 1)

    def forward(self, cell_emb, gene_embs, batch_emb, args=None):
        # Step 1: Cross-attention (optional)
        if self.use_cross_attn:
            q = cell_emb.unsqueeze(1)         # (B, 1, D)
            k = v = batch_emb.unsqueeze(1)    # (B, 1, D)
            cell_emb_mod, _ = self.cross_attn(q, k, v)
            cell_emb = cell_emb_mod.squeeze(1)  # (B, D)

        # Step 2: Query vector from gene_embs
        query_vecs = self.query_activation(self.gene2query(gene_embs))  # (B, L, D)

        # Step 3: Inner product
        cell_emb_exp = cell_emb.unsqueeze(2)  # (B, D, 1)
        pred = torch.bmm(self.W(query_vecs), cell_emb_exp).squeeze(2)  # (B, L)

        # Step 4: Conditional modulation (optional)
        if self.use_modulation:
            gamma = self.mod_gamma(batch_emb)  # (B, 1)
            beta = self.mod_beta(batch_emb)    # (B, 1)
            pred = gamma * pred + beta         # (B, L)

        return dict(pred=pred)
    
    

class MVCDecoder(nn.Module):
    def __init__(
        self,
        d_model: int, 
        arch_style: str = "inner product", # 构造方式
        query_activation: nn.Module = nn.Sigmoid,
        hidden_activation: nn.Module = nn.PReLU,
        explicit_zero_prob: bool = False,
        args=None
    ) -> None:
        """
        Args:
            d_model (:obj:`int`): dimension of the gene embedding. 
            就是一个基因的id被embed到512维空间的那个向量

            arch_style (:obj:`str`): architecture style of the decoder, choice from
                1. "inner product" or 2. "concat query" or 3. "sum query".
            query_activation (:obj:`nn.Module`): activation function for the query
                vectors.
            hidden_activation (:obj:`nn.Module`): activation function for the hidden
                layers.
        """
        super().__init__()
        # 如果有测序协议的信息，则需要2倍的输入长度[gene embedding + batch embedding]，这里的batch embedding表示一个测序协议的id被embed到512维空间的向量
        d_in = d_model
        if args.use_batch_labels:
            d_in += d_model
        if args.use_image and args.image_combine_mz=="cat_dim":
            d_in += d_model


        if arch_style in ["inner product", "inner product, detach"]:
            self.gene2query = nn.Linear(d_model, d_model) # 512 -> 512
            self.query_activation = query_activation()
            self.W = nn.Linear(d_model, d_in, bias=False)
            if explicit_zero_prob:  # by default, gene-wise prob rate
                self.W_zero_logit = nn.Linear(d_model, d_in)


        elif arch_style == "concat query":
            self.gene2query = nn.Linear(d_model, 64)
            self.query_activation = query_activation()
            self.fc1 = nn.Linear(d_model + 64, 64)
            self.hidden_activation = hidden_activation()
            self.fc2 = nn.Linear(64, 1)
            
        elif arch_style == "sum query":
            self.gene2query = nn.Linear(d_model, d_model)
            self.query_activation = query_activation()
            self.fc1 = nn.Linear(d_model, 64)
            self.hidden_activation = hidden_activation()
            self.fc2 = nn.Linear(64, 1)
        else:
            raise ValueError(f"Unknown arch_style: {arch_style}")

        self.arch_style = arch_style
        self.do_detach = arch_style.endswith("detach")
        self.explicit_zero_prob = explicit_zero_prob


    def forward(
        self, cell_emb: Tensor, gene_embs: Tensor, args=None) -> Union[Tensor, Dict[str, Tensor]]:
        """
        Args:
            cell_emb: Tensor, shape (batch, embsize=d_model)
            cell embedding就是细胞的CLS

            gene_embs: Tensor, shape (batch, seq_len, embsize=d_model)
        """

        gene_embs = gene_embs.detach() if self.do_detach else gene_embs
        if self.arch_style in ["inner product", "inner product, detach"]:
            # 计算出query vector
            query_vecs = self.query_activation(self.gene2query(gene_embs))
            
            cell_emb = cell_emb.unsqueeze(2)  # (batch, d_model, 1)
            # the pred gene expr values, # (batch, seq_len)
            pred_value = torch.bmm(self.W(query_vecs), cell_emb).squeeze(2)
            # query_vecs: (batch, seq_len, d_model)
            # self.W(query_vecs): (batch, seq_len, d_in)
            # cell_emb: (batch, d_model, 1)
            # torch.bmm(self.W(query_vecs), cell_emb): (batch, seq_len, 1) 
            # pred_value : (batch, seq_len) 即每个样本预测每个 gene 的表达值

            if not self.explicit_zero_prob:
                return dict(pred=pred_value)
            # zero logits need to based on the cell_emb, because of input exprs
            zero_logits = torch.bmm(self.W_zero_logit(query_vecs), cell_emb).squeeze(2)
            zero_probs = torch.sigmoid(zero_logits) # (batch, seq_len) 即每个样本预测每个 gene 的表达值为0的概率
            return dict(pred=pred_value, zero_probs=zero_probs)
        
        elif self.arch_style == "concat query":
            query_vecs = self.query_activation(self.gene2query(gene_embs))
            # expand cell_emb to (batch, seq_len, embsize)
            cell_emb = cell_emb.unsqueeze(1).expand(-1, gene_embs.shape[1], -1)

            h = self.hidden_activation(
                self.fc1(torch.cat([cell_emb, query_vecs], dim=2))
            )
            if self.explicit_zero_prob:
                raise NotImplementedError
            pred_value = self.fc2(h).squeeze(2)
            return dict(pred=pred_value) # (batch, seq_len)
        
        elif self.arch_style == "sum query":
            query_vecs = self.query_activation(self.gene2query(gene_embs))
            cell_emb = cell_emb.unsqueeze(1)

            h = self.hidden_activation(self.fc1(cell_emb + query_vecs))
            if self.explicit_zero_prob:
                raise NotImplementedError
            pred_value = self.fc2(h).squeeze(2)
            return dict(pred=pred_value)  # (batch, seq_len)
        



class MVCpredictor(nn.Module):
    def __init__(self, 
                 d_model, 
                 mvc_decoder_style="inner product", 
                 explicit_zero_prob = False,
                 args=None
                 ):
        super().__init__()

        self.explicit_zero_prob = explicit_zero_prob
        self.use_batch_labels = args.use_batch_labels
        self.cell_emb_style = args.cell_emb_style

        self.use_batch_aware = args.use_batch_aware

        if self.use_batch_aware and args.image_combine_mz != "cat_dim":
            self.mvc_decoder = BatchAwareMVCDecoder(
                d_model, 
                use_cross_attn=True, 
                use_modulation=True, 
                query_activation=nn.Sigmoid,
                args = args
            )
            self.impute_mvc_decoder = BatchAwareMVCDecoder(
                d_model, 
                use_cross_attn=True, 
                use_modulation=True, 
                query_activation=nn.Sigmoid,
                args = args
            )

        else:
            self.mvc_decoder = MVCDecoder(
                    d_model, # 512
                    arch_style=mvc_decoder_style, # "inner product"
                    explicit_zero_prob=explicit_zero_prob, # False
                    args = args
                )
            
            self.impute_mvc_decoder = MVCDecoder(
                    d_model,
                    arch_style=mvc_decoder_style,
                    explicit_zero_prob=explicit_zero_prob,
                    args = args
                )


        # self.cur_gene_token_embs = cur_gene_token_embs

        


    def _get_cell_emb_from_layer(
        self, 
        layer_output: Tensor, 
        weights: Tensor = None,
        args = None
    ) -> Tensor:
        """
        Args:
            layer_output(:obj:`Tensor`): shape (batch, seq_len, embsize),表示 DenseTransformer 的输出
            weights(:obj:`Tensor`): shape (batch, seq_len), optional and only used
                when :attr:`self.cell_emb_style` is "w-pool".

        Returns:
            :obj:`Tensor`: shape (batch, embsize)
        """
        if self.cell_emb_style == "cls":
            cell_emb = layer_output[:, 0, :]  # (batch, embsize)
            
        elif self.cell_emb_style == "avg-pool":
            cell_emb = torch.mean(layer_output, dim=1)
        elif self.cell_emb_style == "w-pool":
            if weights is None:
                raise ValueError("weights is required when cell_emb_style is w-pool")
            if weights.dim() != 2:
                raise ValueError("weights should be 2D")
            cell_emb = torch.sum(layer_output * weights.unsqueeze(2), dim=1)
            cell_emb = F.normalize(cell_emb, p=2, dim=1)  # (batch, embsize)

        return cell_emb
    
    
    def forward(
        self,
        center_embedding : Tensor,
        neighbor_embedding: Tensor,
        center_gene_token_embs: Tensor,
        batch_emb: Optional[Tensor] = None,
        n_batch_emb: Optional[Tensor] = None,
        do_sample: bool = False,
        predict_by_self = True,
        predict_by_neighbor = True,
        coordinates = None,
        args=None
    ) -> Mapping[str, Tensor]:

        output = dict()


        if predict_by_self:
            center_emb = self._get_cell_emb_from_layer(center_embedding)
            output["center_emb"] = center_emb

            if self.use_batch_aware and args.image_combine_mz != "cat_dim":
                mvc_output = self.mvc_decoder(cell_emb=center_emb, 
                                              gene_embs=center_gene_token_embs, 
                                              batch_emb=batch_emb[:,0,:],
                                              args=args
                                              )
            else:
                mvc_output = self.mvc_decoder(
                    cell_emb = center_emb  if not self.use_batch_labels else torch.cat([center_emb , batch_emb[:,0,:]], dim=1),
                    gene_embs = center_gene_token_embs,
                    args=args
                )

            if self.explicit_zero_prob and do_sample:
                bernoulli = Bernoulli(probs=mvc_output["zero_probs"])
                output["mvc_output"] = bernoulli.sample() * mvc_output["pred"]
            else:
                output["mvc_output"] = mvc_output["pred"]  # (batch, seq_len)
            if self.explicit_zero_prob:
                output["mvc_zero_probs"] = mvc_output["zero_probs"]
        
        
        if predict_by_neighbor:
            
            neighbor_mean_embedding = neighbor_embedding.mean(dim=1)  # [B, L, D]
            neighbor_emb = self._get_cell_emb_from_layer(neighbor_mean_embedding)
            
            output["neighbor_emb"] = neighbor_emb

            if self.use_batch_aware and args.image_combine_mz != "cat_dim":
                neighbor_output = self.impute_mvc_decoder(cell_emb=neighbor_emb, 
                                              gene_embs=center_gene_token_embs, 
                                              batch_emb=n_batch_emb[:,0,0,:],
                                              args=args
                                              )
            else:
                neighbor_output = self.impute_mvc_decoder(
                    cell_emb = neighbor_emb if not self.use_batch_labels else torch.cat([neighbor_emb, n_batch_emb[:,0,0,:]], dim=1),
                    gene_embs = center_gene_token_embs,
                    args=args
                )

            if self.explicit_zero_prob and do_sample:
                bernoulli = Bernoulli(probs=neighbor_output["zero_probs"])
                output["impute_pred"] = bernoulli.sample() * neighbor_output["pred"]
            else:
                output["impute_pred"] = neighbor_output["pred"]  # (batch, seq_len)
            if self.explicit_zero_prob:
                output["impute_zero_probs"] = neighbor_output["zero_probs"]

        return output