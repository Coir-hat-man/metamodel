import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math
import numpy as np

class SpatialPatchEmb(nn.Module):
    def __init__(self, c_in, d_model, patch_size):
        super(SpatialPatchEmb, self).__init__()
        self.patch_size = patch_size
        self.conv = nn.Conv1d(in_channels=c_in, out_channels=d_model, kernel_size=patch_size, stride=patch_size)
        self.patch_size = patch_size
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = x.permute(0, 2, 1)
        B, C, N = x.shape
        x = self.conv(x)
        x = x.reshape(B, -1, N).permute(0,2,1)
        return x

class SPEmb(nn.Module):
    def __init__(self, c_in, d_model, patch_size):
        super(SPEmb, self).__init__()
        self.patch_size = patch_size
        self.conv = nn.Conv2d(in_channels=c_in, out_channels=d_model, kernel_size=patch_size, stride=patch_size)
        self.patch_size = patch_size
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.conv(x)
        x = x.reshape(B, -1, H*W//self.patch_size**2).permute(0,2,1)
        return x

class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1, args=None, size1 = 48, size2=7 ):
        super(DataEmbedding, self).__init__()
        self.args = args
        self.value_embedding = SpatialPatchEmb(c_in=c_in, d_model=d_model, patch_size=args.patch_size) 
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark, is_time=1):
        '''
        x: N, T, C, H, W
        x_mark: N, T, D
        '''
        N, T, C, H, W = x.shape
        TokenEmb = self.value_embedding(x)
        x = TokenEmb
        return self.dropout(x)
    

# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim, grid_size1, grid_size2, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim]
    """
    grid_h = np.arange(grid_size1, dtype=np.float32)
    grid_w = np.arange(grid_size2, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size1, grid_size2])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed


def get_2d_sincos_pos_embed_with_resolution(
    embed_dim, grid_size1, grid_size2, res, cls_token=False, device="cpu"
):
    """
    grid_size: int of the grid height and width
    res: array of size n, representing the resolution of a pixel (say, in meters),
    return:
    pos_embed: [n,grid_size*grid_size, embed_dim]
    """
    # res = torch.FloatTensor(res).to(device)
    res = res.to(device)
    grid_h = torch.arange(grid_size1, dtype=torch.float32, device=device)
    grid_w = torch.arange(grid_size2, dtype=torch.float32, device=device)
    grid = torch.meshgrid(
        grid_w, grid_h, indexing="xy"
    )  # here h goes first,direction reversed for numpy
    grid = torch.stack(grid, dim=0)  # 2 x h x w

    # grid = grid.reshape([2, 1, grid_size, grid_size])
    grid = torch.einsum("chw,n->cnhw", grid, res)  # 2 x n x h x w
    _, n, h, w = grid.shape
    pos_embed = get_2d_sincos_pos_embed_from_grid_torch(
        embed_dim, grid
    )  #  # (nxH*W, D/2)
    
    pos_embed = pos_embed.reshape(h * w, embed_dim).numpy()
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_2d_sincos_pos_embed_from_grid_torch(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid_torch(
        embed_dim // 2, grid[0]
    )  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid_torch(
        embed_dim // 2, grid[1]
    )  # (H*W, D/2)

    emb = torch.cat([emb_h, emb_w], dim=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid_torch(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    old_shape = pos
    omega = torch.arange(embed_dim // 2, dtype=torch.float32, device=pos.device)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

def get_1d_sincos_pos_embed_from_grid_with_resolution(embed_dim, pos, res):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    pos = pos * res
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb
