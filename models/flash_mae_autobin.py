# Copyright 2023 BioMap (Beijing) Intelligence Technology Limited

import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.cuda.amp import autocast
from einops import rearrange, repeat

from functools import partial
from contextlib import contextmanager
from utils import *

def exists(val):
    return val is not None

def gatherData(data, labels, pad_token_id):
    value_nums = labels.sum(1)
    max_num = max(value_nums)

    print(data)
    fake_data = torch.full((data.shape[0], max_num), pad_token_id,
                           device=data.device)
    data = torch.hstack([data, fake_data])

    fake_label = torch.full((labels.shape[0], max_num), 1,
                            device=labels.device)
    none_labels = ~labels
    labels = labels.float()
    labels[none_labels] = torch.tensor(-float('Inf'), device=labels.device)

    tmp_data = torch.tensor([(i + 1) * 20000 for i in range(labels.shape[1], 0, -1)], device=labels.device)
    labels += tmp_data

    labels = torch.hstack([labels, fake_label])

    fake_label_gene_idx = labels.topk(max_num).indices

    new_data = torch.gather(data, 1, fake_label_gene_idx)

    padding_labels = (new_data == pad_token_id)
    print(new_data,padding_labels)
    return new_data, padding_labels

def getEncoerDecoderData(data, data_raw, pad_token_id, seq_len):
    decoder_data = data.clone().detach()
    decoder_data_padding = torch.full_like(data, False, dtype=torch.bool).to(data.device)

    encoder_data_labels = data_raw > 0

    encoder_data, encoder_data_padding = gatherData(decoder_data, encoder_data_labels,
                                                    pad_token_id)

    new_data_raw = data_raw
    data_gene_ids = torch.arange(data.shape[1], device=data.device).repeat(data.shape[0], 1)
    encoder_position_gene_ids, _ = gatherData(data_gene_ids, encoder_data_labels,
                                                pad_token_id)
    decoder_position_gene_ids = data_gene_ids
    data_mask_labels = None

    encoder_position_gene_ids[encoder_data_padding] = seq_len
    decoder_position_gene_ids[decoder_data_padding] = seq_len

    return encoder_data, encoder_position_gene_ids, encoder_data_padding, encoder_data_labels, decoder_data, decoder_data_padding, new_data_raw, data_mask_labels, decoder_position_gene_ids

def gen_padding_label(data, pad_token_id = 0):

    # 生成 mask_labels (1 表示该位置被掩码，0 表示未被掩码)
    mask_labels = (data == pad_token_id)

    return mask_labels

def mask_strategy(data, mask_ratio, mask_token_id, pad_token_id, args=None):
    # 生成与 data 形状相同的随机掩码矩阵，随机选择部分值进行掩码

    mask = (torch.rand(data.shape) < mask_ratio).to(args.device) & (data != pad_token_id)

    # 生成 mask_labels (1 表示该位置被掩码，0 表示未被掩码)
    mask_labels = mask.int()

    # 在 data 中应用掩码，将被选中的值替换为 mask_token_id
    masked_data = data.clone()
    masked_data[mask] = mask_token_id
    return masked_data, mask_labels

class AutoDiscretizationEmbedding2(nn.Module):
    def __init__(self, dim, max_seq_len, bin_num, bin_alpha, mask_token_id = None, pad_token_id = None):
        super().__init__()
        
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.bin_num = bin_num
        self.bin_alpha = bin_alpha
        
        self.mlp = nn.Linear(1, self.bin_num)
        self.mlp2 = nn.Linear(self.bin_num, self.bin_num)
        self.LeakyReLU = nn.LeakyReLU(0.1)
        self.Softmax = nn.Softmax(dim=-1)
        self.emb = nn.Embedding(self.bin_num, self.dim)
        
        self.emb_mask = nn.Embedding(1, self.dim)
        self.emb_pad = nn.Embedding(1, self.dim)
        
        self.bin_num_idx = torch.tensor(range(self.bin_num))
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
        # print('self.bin_num_idx',self.bin_num_idx, self.bin_num_idx.shape)

        self.tensor0 = torch.tensor(0, dtype=torch.long)

    def forward(self, x, output_weight=0):
        x_mask_idx = (x==self.mask_token_id).nonzero()
        x_pad_idx = (x==self.pad_token_id).nonzero()

        # print("x_mask",x_mask_idx.shape,x_mask_idx)
        
        x = self.mlp(x) # [B,N,1] -> [B,N,H]
        x = self.LeakyReLU(x) # [B,N,H]
        x_crosslayer = self.mlp2(x) # [B,N,H]
        x = self.bin_alpha * x + x_crosslayer # [B,N,H]
        weight = self.Softmax(x) # [B, N, H]
        # print('weight', weight.shape, weight, torch.sum(weight, 2))
        
        bin_num_idx = self.bin_num_idx.to(x.device) # [H,]
        # print('bin_num_idx', bin_num_idx.shape)
        
        token_emb = self.emb(bin_num_idx) # [H, D]
        # print('token_emb', token_emb.shape)
        x = torch.matmul(weight, token_emb) #[B, N, D]
    
        # print("x_emb",x.shape,x)
        
        tensor0 = torch.tensor(0, dtype=torch.long, device=x.device)

        mask_token_emb = self.emb_mask(tensor0).to(x.device).type(x.dtype)
        # print(mask_token_emb.dtype)
        # print("x", x.dtype)
        x[x_mask_idx[:,0],x_mask_idx[:,1],:] = mask_token_emb.repeat(x_mask_idx.shape[0],1)
        # print("x_emb",x.shape,x)

        pad_token_emb = self.emb_pad(tensor0).to(x.device).type(x.dtype)
        x[x_pad_idx[:,0],x_pad_idx[:,1],:] = pad_token_emb.repeat(x_pad_idx.shape[0],1)
    
        if output_weight:
            return x,weight
        return x

class RandomPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len):
        super().__init__()
        self.emb = nn.Embedding(max_seq_len, dim)

    def forward(self, x):
        t = torch.arange(x.shape[1], device=x.device)
        return self.emb(t)

from flash_attn.modules.mha import FlashSelfAttention
from flash_attn.bert_padding import unpad_input, pad_input

class FlashTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, ff_mult=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.flash_attn = FlashSelfAttention(causal=False)

        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_mult * d_model, d_model),
        )

    def forward(self, x, padding_mask=None):
        B, L, D = x.shape
        x_res = x
        x = self.norm1(x)
        if padding_mask is not None:
            unpad_x, indices, cu_seqlens, max_seqlen,_ = unpad_input(x, padding_mask)
            qkv = self.qkv_proj(unpad_x)
            qkv = qkv.view(-1, 3, self.n_heads, self.head_dim)
            out = self.flash_attn(qkv, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
            out = out.view(-1,self.n_heads*self.head_dim)
            out = self.out_proj(out)
            out = pad_input(out, indices, B, L)
        else:
            # 无需 unpad，直接 dense 模式
            qkv = self.qkv_proj(x)
            qkv = qkv.view(B * L, 3, self.n_heads, self.head_dim)
            out = self.flash_attn(qkv, cu_seqlens=torch.arange(0, (B + 1) * L, step=L, device=x.device),
                                max_seqlen=L)
            out = out.view(-1,self.n_heads*self.head_dim)
            out = self.out_proj(out)
            out = out.view(B, L, D)

        x = x_res + self.dropout(out)

        x_res = x
        x = self.norm2(x)
        x = x_res + self.dropout(self.ffn(x))
        return x

class FlashTransformerModule(nn.Module):
    def __init__(self, max_seq_len, dim, depth, heads, ff_mult=4):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.depth = depth
        self.layers = nn.ModuleList([
            FlashTransformerEncoderLayer(dim, heads, ff_mult=ff_mult)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, padding_mask=None):
        for layer in self.layers:
            x = layer(x, padding_mask)
        return self.norm(x)

class pytorchTransformerModule(nn.Module):
    def __init__(self,
                 max_seq_len,
                 dim,
                 depth,
                 heads,
                 ff_mult=4,
                 norm_first=False,
                 ):
        super(pytorchTransformerModule, self).__init__()

        self.max_seq_len = max_seq_len
        self.depth = depth
        layers = []
        for i in range(depth):
            layers.append(nn.TransformerEncoderLayer(d_model=dim, nhead=heads,
                                                     dim_feedforward=dim * ff_mult,
                                                     batch_first=True,
                                                     norm_first=norm_first,
                                                     #activation="gelu",
                                                     ))

        self.transformer_encoder = nn.ModuleList(layers)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, padding_mask):
        b, n, _, device = *x.shape, x.device

        # x get encodings [B, N, D] , batch_first is True
        for mod in self.transformer_encoder:
            x = mod(x, src_key_padding_mask=padding_mask) # , src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        # x = self.transformer_encoder(x)
        x = self.norm(x)

        return x

class MaeAutobin(nn.Module):
    def __init__(
            self,
            *,
            max_seq_len,  # max length of sequence
            embed_dim,  # encoder dim of tokens
            decoder_embed_dim,
            compound_repr_dim = 768,
            bias = True,
            bin_alpha = 1.0,
            bin_num = 10,
            pad_token_id = 0,
            mask_token_id = 1,
            depth = 12,
            heads = 8,
            args = None,

    ):
        super(MaeAutobin, self).__init__()

        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id
        self.args = args
        # encoder
        self.token_emb = AutoDiscretizationEmbedding2(embed_dim, max_seq_len, bin_num=bin_num, bin_alpha=bin_alpha, pad_token_id=self.pad_token_id, mask_token_id=self.mask_token_id)
        self.pos_emb = nn.Embedding(max_seq_len+1, embed_dim)  #RandomPositionalEmbedding(embed_dim, max_seq_len)
        self.compound_emb = nn.Linear(compound_repr_dim, embed_dim, bias)
        self.real_pos_embed = nn.Linear(2, embed_dim)

        self.encoder = pytorchTransformerModule(max_seq_len, embed_dim, depth, heads)
        if args.use_flash:
            self.encoder = FlashTransformerModule(max_seq_len, embed_dim, depth, heads)
        ##### decoder
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

    def forward(self, x, compound, ids, pos, padding_labels, id_padding_labels, mask_labels = None, args=None, **kwargs):

        B, N, C = compound.shape
        # x, encoder_position_gene_ids, encoder_data_padding, encoder_data_labels, decoder_data, decoder_data_padding, new_data_raw, data_mask_labels, decoder_position_gene_ids = getEncoerDecoderData(x, x, self.pad_token_id, self.max_seq_len)
        # padding_label = encoder_data_padding
        data_gene_ids = ids
        masked_x = x.clone()
        masked_x[mask_labels] = 1
        if mask_labels != None:
            mask_labels = mask_labels.to(x.device)
            id_mask_labels = mask_labels[:,1:]
            data_gene_ids[id_mask_labels] = args.mask_label_id
        x_pad_idx = (x==self.pad_token_id).to(x.device)
        if mask_labels  != None:
            padding_and_masking_idx =  x_pad_idx | mask_labels
        else:
            padding_and_masking_idx = x_pad_idx

        # x_pad_idx = None
        # padding_and_masking_idx = x_pad_idx

        # mask_index = torch.nonzero(~mask_labels, as_tuple=True)
        # masked_x = x[mask_index].reshape(B, -1)
        # print(masked_x.shape)

        b, n, device = *masked_x.shape, x.device
        pos_emb = self.real_pos_embed(pos)
        pos_emb = pos_emb.unsqueeze(1)  # Shape becomes (B, 1, 1, C)
        # token and positional embedding
        raw_x = self.token_emb(torch.unsqueeze(masked_x.float(), 2), output_weight = 0)
        pos_emb = pos_emb.expand(-1, masked_x.shape[1], -1) 
        raw_x += pos_emb
        position_emb = self.pos_emb(data_gene_ids.int())
        raw_x[:,1:] += position_emb
        compound_emb = self.compound_emb(compound)
        
        x = raw_x + compound_emb
        fusion_emb = x
        x = self.encoder(x, padding_mask=~padding_and_masking_idx)


        # decoder_data = self.token_emb(torch.unsqueeze(decoder_data, 2))
        # position_emb = self.pos_emb(decoder_position_gene_ids)

        # batch_idx, gen_idx = (encoder_labels == True).nonzero(as_tuple=True)
        # decoder_data[batch_idx, gen_idx] = x[~padding_label].to(decoder_data.dtype)

        # decoder_data += position_emb

        x = self.decoder_embed(x)

        # x = self.decoder(decoder_data, padding_mask=decoder_data_padding_labels)

        # # print("x0",x.shape) 
        # x = self.norm(x)
        # # print("x1",x.shape) 
        # if exists(self.to_final):
        #     x = self.to_final(x)
        #     return x.squeeze(2) 
        # else:

        return fusion_emb, x, x_pad_idx, raw_x, compound_emb
    # def forward(self, x,  ids, pos, padding_labels, id_padding_labels, mask_labels = None, args=None, **kwargs):

    #     # x, encoder_position_gene_ids, encoder_data_padding, encoder_data_labels, decoder_data, decoder_data_padding, new_data_raw, data_mask_labels, decoder_position_gene_ids = getEncoerDecoderData(x, x, self.pad_token_id, self.max_seq_len)
    #     # padding_label = encoder_data_padding
    #     data_gene_ids = ids
    #     masked_x = x.clone()
    #     masked_x[mask_labels] = 1
    #     if mask_labels != None:
    #         mask_labels = mask_labels.to(x.device)
    #         id_mask_labels = mask_labels[:,1:]
    #         data_gene_ids[id_mask_labels] = args.mask_label_id
    #     x_pad_idx = (x==self.pad_token_id).to(x.device)

    #     if mask_labels  != None:
    #         padding_and_masking_idx =  x_pad_idx | mask_labels
    #     else:
    #         padding_and_masking_idx = x_pad_idx
    #     data_gene_ids = ids

    #     # mask_index = torch.nonzero(~mask_labels, as_tuple=True)
    #     # masked_x = x[mask_index].reshape(B, -1)
    #     # print(masked_x.shape)

    #     b, n, device = *masked_x.shape, x.device

    #     pos_emb = self.real_pos_embed(pos)
    #     pos_emb = pos_emb.unsqueeze(1)  # Shape becomes (B, 1, 1, C)
    #     # token and positional embedding
    #     raw_x = self.token_emb(torch.unsqueeze(masked_x.float(), 2), output_weight = 0)
    #     pos_emb = pos_emb.expand(-1, masked_x.shape[1], -1) 
    #     raw_x += pos_emb
    #     position_emb = self.pos_emb(data_gene_ids.int())
    #     raw_x[:,1:] += position_emb

    #     x = raw_x
    #     fusion_emb = x

    #     x = self.encoder(x, padding_mask=padding_and_masking_idx)
        

    #     # decoder_data = self.token_emb(torch.unsqueeze(decoder_data, 2))
    #     # position_emb = self.pos_emb(decoder_position_gene_ids)

    #     # batch_idx, gen_idx = (encoder_labels == True).nonzero(as_tuple=True)
    #     # decoder_data[batch_idx, gen_idx] = x[~padding_label].to(decoder_data.dtype)

    #     # decoder_data += position_emb

    #     x = self.decoder_embed(x)

    #     # x = self.decoder(decoder_data, padding_mask=decoder_data_padding_labels)

    #     # # print("x0",x.shape) 
    #     # x = self.norm(x)
    #     # # print("x1",x.shape) 
    #     # if exists(self.to_final):
    #     #     x = self.to_final(x)
    #     #     return x.squeeze(2) 
    #     # else:

    #     return fusion_emb, x, raw_x