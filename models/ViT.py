from functools import partial

import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import numpy as np
from timm.models.layers import to_2tuple
from timm.models.vision_transformer import DropPath, Mlp

from models.Embed import SPEmb, SpatialPatchEmb,  get_2d_sincos_pos_embed, get_1d_sincos_pos_embed_from_grid

from mask_strategy import *
import copy

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        input_size=(4, 14, 14),
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        assert attn_drop == 0.0  # do not use
        self.proj = nn.Linear(dim, dim, bias= qkv_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.input_size = input_size
        assert input_size[1] == input_size[2]

    def forward(self, x):
        B, N, C = x.shape
        q = (
            self.q(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        k = (
            self.k(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        v = (
            self.v(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.view(B, -1, C)
        return x


class Block(nn.Module):
    """
    Transformer Block with specified Attention function
    """

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        attn_func=Attention,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_func(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    
class pytorchTransformerModule(nn.Module):
    def __init__(self,
                 dim,
                 depth,
                 heads,
                 ff_mult=4,
                 norm_first=False,
                 ):
        super(pytorchTransformerModule, self).__init__()

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
        # x get encodings [B, N, D] , batch_first is True
        for mod in self.transformer_encoder:
            x = mod(x, src_key_padding_mask=padding_mask) # , src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        # x = self.transformer_encoder(x)
        x = self.norm(x)

        return x

class ViT_encoder(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, patch_size=1, in_chans=1, max_seq_len= 100,
                 embed_dim=512, decoder_embed_dim=512, depth=12, decoder_depth=8, num_heads=8,  decoder_num_heads=4,
                 mlp_ratio=2, norm_layer=nn.LayerNorm, t_patch_size=1,
                 no_qkv_bias=False, args=None, ):
        super().__init__()

        self.args = args

        self.in_chans = in_chans

        self.max_seq_len = max_seq_len

        self.Embedding = SpatialPatchEmb(in_chans, embed_dim, patch_size)

        self.patch_size = patch_size

        self.embed_dim = embed_dim

        self.pixel_embed = nn.Linear(3, embed_dim)

        self.pos_embed_spatial = nn.Parameter(
            torch.zeros(1, 1024, embed_dim)
        )

        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=args.qkv_bias,
                    qk_scale=None,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.no_qkv_bias = no_qkv_bias
        self.norm_layer = norm_layer

        self.norm = norm_layer(embed_dim)

        self.pos_embed = nn.Linear(2, embed_dim)

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias= not self.args.qkv_bias)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.initialize_weights_trivial()


    def init_emb(self):
        w = self.Embedding.conv.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        torch.nn.init.normal_(self.mask_token, std=0.02)

    def get_weights_sincos(self, num_patch_1, num_patch_2):
        # initialize (and freeze) pos_embed by sin-cos embedding

        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed_spatial.shape[-1],
            grid_size1 = num_patch_1,
            grid_size2 = num_patch_2
        )

        pos_embed_spatial = nn.Parameter(
                torch.zeros(1, num_patch_1 * num_patch_2, self.embed_dim)
            )

        pos_embed_spatial.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        pos_embed_spatial.requires_grad = False

        return pos_embed_spatial, copy.deepcopy(pos_embed_spatial)
    
    def initialize_weights_trivial(self):
        torch.nn.init.trunc_normal_(self.pos_embed_spatial, std=0.02)

        w = self.Embedding.conv.weight.data

        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        torch.nn.init.normal_(self.mask_token, std=0.02)
        #torch.nn.init.normal_(self.mask_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 1, H, W)
        x: (N, L, patch_size**2 *1)
        """
        N, _, H, W = imgs.shape
        p = self.args.patch_size
        assert H % p == 0 and W % p == 0
        h = H // p
        w = W // p
        x = imgs.reshape(shape=(N, 1, h, p, w, p))
        x = torch.einsum("nchpwq->nhwpqc", x)
        x = x.reshape(shape=(N,  h * w,  p**2 * 1))
        self.patch_info = (N, 1, H, W, p, h, w)
        return x
    
    def unpatchify(self, imgs):
        """
        imgs: (N, L, patch_size**2 *1)
        x: (N, 1,  H, W)
        """
        N, _, H, W, p, h, w = self.patch_info
        imgs = imgs.reshape(shape=(N, h, w, p, p))
        imgs = torch.einsum("nhwpq->nhpwq", imgs)
        imgs = imgs.reshape(shape=(N, 1, H, W))
        return imgs
    
    
    # def forward_encoder(self, x,  mask_ratio, mask_strategy, seed=None, data=None, mode='backward'):
    #     # embed patches
    #     N, _, H, W = x.shape

    #     x = self.Embedding(x) # B, H*W/(p*p), C
    #     _, L, C = x.shape


    #     if mode=='backward':

    #         if mask_strategy == 'random':
    #             x, mask, ids_restore, ids_keep = random_masking(x, mask_ratio)

    #     elif mode == 'forward': # evaluation, fix random seed
    #         if mask_strategy == 'random':
    #             x, mask, ids_restore, ids_keep = random_masking_evaluate(x, mask_ratio)

    #     input_size = (H//self.patch_size, W//self.patch_size)
    #     pos_embed_sort = self.pos_embed_enc(ids_keep, N, input_size)

    #     assert x.shape == pos_embed_sort.shape

    #     x_attn = x + pos_embed_sort

    #     # apply Transformer blocks
    #     for index, blk in enumerate(self.blocks):
    #         x_attn = blk(x_attn)

    #     x_attn = self.norm(x_attn)

    #     return x_attn, mask, ids_restore, input_size
    
    def apply_random_mask(self, x, mask_ratio=0.5, seed=None):
        """
        Apply a random mask to the input sequence x.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, L, C)
            mask_ratio (float): Proportion of tokens to mask
            seed (int, optional): Random seed for reproducibility
        
        Returns:
            masked_x (torch.Tensor): Masked input tensor
            mask (torch.Tensor): Mask tensor indicating masked positions
        """
        if seed is not None:
            torch.manual_seed(seed)
        B, L, C = x.shape
        num_masked = int(L * mask_ratio)
        
        # Generate random indices for masking
        rand_indices = torch.rand(B, L).argsort(dim=1)
        mask = torch.zeros(B, L, dtype=torch.bool)
        
        for i in range(B):
            mask[i, rand_indices[i, :num_masked]] = True
        
        # Apply the mask
        masked_x = x.clone()
        masked_x[mask] = 0  # Masked positions are set to zero
    
        return masked_x, mask
    
    def forward(self, x, pos, pixel_emb, padding_labels, mask = None):
        # embed patches

        B, N, C = x.shape
        x = self.Embedding(x) # B, H*W/(p*p), C
        embed_x =x
        B, L, C = x.shape
        # pixel_emb = pixel_emb.expand(-1, x.shape[1], -1) 
        x += pixel_emb
        pos_emb = self.pos_embed(pos)
        pos_emb = pos_emb.unsqueeze(1)  # Shape becomes (B, 1, C)
        pos_emb = pos_emb.expand(-1, x.shape[1], -1) 
        x += pos_emb
        if mask != None:
            x_attn = x[~mask].reshape(B,-1,C)
            restored_masked_x = torch.zeros_like(x)  # 创建一个和 x 同形状的全零张量
            #     # apply Transformer blocks
            for index, blk in enumerate(self.blocks):
                x_attn = blk(x_attn)

            x_attn = self.norm(x_attn)
            index = torch.nonzero(~mask, as_tuple=True)  # 获取非掩码的位置
            # 将 x_attn 中的有效元素还原到 restored_x
            decoder_embed = self.decoder_embed(x_attn)
            restored_masked_x[index] = decoder_embed.reshape(-1,C)
            mask_index = torch.nonzero(~mask, as_tuple=True)
            restored_masked_x[mask_index] = self.mask_token  # 注意 mask_token 是 (1, C)
        else :
            x_attn = x
            restored_masked_x = torch.zeros_like(x)  # 创建一个和 x 同形状的全零张量
            #     # apply Transformer blocks
            for index, blk in enumerate(self.blocks):
                x_attn = blk(x_attn)

            x_attn = self.norm(x_attn)
            # 将 x_attn 中的有效元素还原到 restored_x
            decoder_embed = self.decoder_embed(x_attn)
            restored_masked_x = decoder_embed

        return embed_x, restored_masked_x, mask, pixel_emb
    
class vision_encoder(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, patch_size=1, in_chans=1, max_seq_len= 100,
                 embed_dim=512, decoder_embed_dim=512, depth=12, decoder_depth=8, num_heads=8,  decoder_num_heads=4,
                 mlp_ratio=2, norm_layer=nn.LayerNorm, t_patch_size=1,
                 no_qkv_bias=False, args=None, ):
        super().__init__()

        self.args = args

        self.in_chans = in_chans

        self.max_seq_len = max_seq_len

        self.Embedding = SPEmb(in_chans, embed_dim, patch_size)

        self.patch_size = patch_size

        self.embed_dim = embed_dim

        self.pixel_embed = nn.Linear(3, embed_dim)

        self.pos_embed_spatial = nn.Parameter(
            torch.zeros(1, 1024, embed_dim)
        )

        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=args.qkv_bias,
                    qk_scale=None,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.no_qkv_bias = no_qkv_bias
        self.norm_layer = norm_layer

        self.norm = norm_layer(embed_dim)

        self.pos_embed = nn.Linear(2, embed_dim)

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias= not self.args.qkv_bias)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.initialize_weights_trivial()


    def init_emb(self):
        w = self.Embedding.conv.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        torch.nn.init.normal_(self.mask_token, std=0.02)

    def get_weights_sincos(self, num_patch_1, num_patch_2):
        # initialize (and freeze) pos_embed by sin-cos embedding

        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed_spatial.shape[-1],
            grid_size1 = num_patch_1,
            grid_size2 = num_patch_2
        )

        pos_embed_spatial = nn.Parameter(
                torch.zeros(1, num_patch_1 * num_patch_2, self.embed_dim)
            )

        pos_embed_spatial.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        pos_embed_spatial.requires_grad = False

        return pos_embed_spatial, copy.deepcopy(pos_embed_spatial)
    
    def initialize_weights_trivial(self):
        torch.nn.init.trunc_normal_(self.pos_embed_spatial, std=0.02)

        w = self.Embedding.conv.weight.data

        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        torch.nn.init.normal_(self.mask_token, std=0.02)
        #torch.nn.init.normal_(self.mask_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 1, H, W)
        x: (N, L, patch_size**2 *1)
        """
        N, _, H, W = imgs.shape
        p = self.args.patch_size
        assert H % p == 0 and W % p == 0
        h = H // p
        w = W // p
        x = imgs.reshape(shape=(N, 1, h, p, w, p))
        x = torch.einsum("nchpwq->nhwpqc", x)
        x = x.reshape(shape=(N,  h * w,  p**2 * 1))
        self.patch_info = (N, 1, H, W, p, h, w)
        return x
    
    def unpatchify(self, imgs):
        """
        imgs: (N, L, patch_size**2 *1)
        x: (N, 1,  H, W)
        """
        N, _, H, W, p, h, w = self.patch_info
        imgs = imgs.reshape(shape=(N, h, w, p, p))
        imgs = torch.einsum("nhwpq->nhpwq", imgs)
        imgs = imgs.reshape(shape=(N, 1, H, W))
        return imgs
    
    
    # def forward_encoder(self, x,  mask_ratio, mask_strategy, seed=None, data=None, mode='backward'):
    #     # embed patches
    #     N, _, H, W = x.shape

    #     x = self.Embedding(x) # B, H*W/(p*p), C
    #     _, L, C = x.shape


    #     if mode=='backward':

    #         if mask_strategy == 'random':
    #             x, mask, ids_restore, ids_keep = random_masking(x, mask_ratio)

    #     elif mode == 'forward': # evaluation, fix random seed
    #         if mask_strategy == 'random':
    #             x, mask, ids_restore, ids_keep = random_masking_evaluate(x, mask_ratio)

    #     input_size = (H//self.patch_size, W//self.patch_size)
    #     pos_embed_sort = self.pos_embed_enc(ids_keep, N, input_size)

    #     assert x.shape == pos_embed_sort.shape

    #     x_attn = x + pos_embed_sort

    #     # apply Transformer blocks
    #     for index, blk in enumerate(self.blocks):
    #         x_attn = blk(x_attn)

    #     x_attn = self.norm(x_attn)

    #     return x_attn, mask, ids_restore, input_size
    
    def apply_random_mask(self, x, mask_ratio=0.5, seed=None):
        """
        Apply a random mask to the input sequence x.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, L, C)
            mask_ratio (float): Proportion of tokens to mask
            seed (int, optional): Random seed for reproducibility
        
        Returns:
            masked_x (torch.Tensor): Masked input tensor
            mask (torch.Tensor): Mask tensor indicating masked positions
        """
        if seed is not None:
            torch.manual_seed(seed)
        B, L, C = x.shape
        num_masked = int(L * mask_ratio)
        
        # Generate random indices for masking
        rand_indices = torch.rand(B, L).argsort(dim=1)
        mask = torch.zeros(B, L, dtype=torch.bool)
        
        for i in range(B):
            mask[i, rand_indices[i, :num_masked]] = True
        
        # Apply the mask
        masked_x = x.clone()
        masked_x[mask] = 0  # Masked positions are set to zero
    
        return masked_x, mask
    
    def forward(self, x, pos = None, pixel_emb = None, mask = None):
        # embed patches

        B, C , H , W = x.shape
        x = self.Embedding(x) # B, H*W/(p*p), C
        embed_x =x
        restored_masked_x = None
        B, L, C = x.shape
        if pixel_emb != None:
            pixel_emb = pixel_emb.unsqueeze(1)
            pixel_emb = pixel_emb.expand(-1, x.shape[1], -1) 
            x += pixel_emb
        if pos != None:
            pos_emb = self.pos_embed(pos)
            pos_emb = pos_emb.unsqueeze(1)  # Shape becomes (B, 1, C)
            pos_emb = pos_emb.expand(-1, x.shape[1], -1) 
            x += pos_emb
        if mask != None:
            x_attn = x[~mask].reshape(B,-1,C)
            restored_masked_x = torch.zeros_like(x)  # 创建一个和 x 同形状的全零张量
            #     # apply Transformer blocks
            for index, blk in enumerate(self.blocks):
                x_attn = blk(x_attn)

            x_attn = self.norm(x_attn)
            index = torch.nonzero(~mask, as_tuple=True)  # 获取非掩码的位置
            # 将 x_attn 中的有效元素还原到 restored_x
            decoder_embed = self.decoder_embed(x_attn)
            restored_masked_x[index] = decoder_embed.reshape(-1,C)
            mask_index = torch.nonzero(~mask, as_tuple=True)
            restored_masked_x[mask_index] = self.mask_token  # 注意 mask_token 是 (1, C)
        else :
            x_attn = x
            restored_masked_x = torch.zeros_like(x)  # 创建一个和 x 同形状的全零张量
            #     # apply Transformer blocks
            for index, blk in enumerate(self.blocks):
                x_attn = blk(x_attn)

            x_attn = self.norm(x_attn)
            # 将 x_attn 中的有效元素还原到 restored_x
            decoder_embed = self.decoder_embed(x_attn)
            restored_masked_x = decoder_embed

        return embed_x, restored_masked_x, mask, pixel_emb
    