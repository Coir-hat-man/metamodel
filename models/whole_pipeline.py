import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from models.mae_autobin import MaeAutobin
from models.flash_mae_autobin import MaeAutobin as Flash_MaeAutobin
from models.ViT import ViT_encoder
from models.moe import MoeDecoder
from models.mvc import MVCpredictor
from models.transformers import DenseTransformer
from models.continuous_encoder import ContinuousValueEncoder
from contrastive_loss import calc_contrastive_loss
from utils import *
from typing import Dict, Optional, Mapping
from models.predictor import STMPredictor
from models.MzVisTransformer import MzVisTransformer
from models.FlashMzVisTransformer import FlashMzVisTransformer


class MetablisumModelV3(nn.Module):
    def __init__(self, d_model=512, 
                 decoder_hidden_dim = 256,
                 transformer_encoder_heads=8, 
                 transformer_encoder_layers=12,
                 transformer_decoder_heads=8, 
                 transformer_decoder_layers=12, 
                 moe_experts=4, 
                 args=None):
        super().__init__()
        self.args = args

        if args.use_cls_token:
            self.cls_token_spectrum = nn.Parameter(torch.full((1, 1), -1.0))
            self.cls_patch = nn.Parameter(torch.full((1, 1), -1.0))
            self.cls_token_mol = nn.Parameter(torch.randn(1, 1, 768))
        

        self.pixel_encoder = nn.Linear(3, d_model)

        self.vit_encoder = ViT_encoder(in_chans=args.in_chans, args=args)
        if args.use_flash:
            self.mz_encoder = Flash_MaeAutobin(
            max_seq_len=args.mask_label_id,
            embed_dim=d_model,
            decoder_embed_dim=d_model,
            compound_repr_dim=768,
            mask_token_id= args.mask_label_id,
            depth=transformer_encoder_layers,
            heads= transformer_encoder_heads,
            args=args
        )
        else:
            self.mz_encoder = MaeAutobin(
                max_seq_len=args.mask_label_id,
                embed_dim=d_model,
                decoder_embed_dim=d_model,
                compound_repr_dim=768,
                mask_token_id= args.mask_label_id,
                depth=transformer_encoder_layers,
                heads= transformer_encoder_heads,
                args=args
            )

        if args.decoder == 'dense_transformer':
            self.transformer_decoder = DenseTransformer(
                d_model=d_model,
                hidden_dim=256,
                num_heads=transformer_decoder_heads,
                num_layers=transformer_decoder_layers,
                args=args
            )
        if args.decoder == 'naive_transformer':
            if args.use_flash:
                self.transformer_decoder = FlashMzVisTransformer(
                    d_model=d_model, 
                    hidden_dim=decoder_hidden_dim, 
                    num_heads=transformer_decoder_heads, 
                    num_cross_layers=transformer_decoder_layers, 
                    num_self_layers=transformer_decoder_layers, 
                    dropout=0.1
                )
            else:
                self.transformer_decoder = MzVisTransformer(
                    d_model=d_model, 
                    hidden_dim=decoder_hidden_dim, 
                    num_heads=transformer_decoder_heads, 
                    num_cross_layers=transformer_decoder_layers, 
                    num_self_layers=transformer_decoder_layers, 
                    dropout=0.1
                )

        self.moe_decoder = MoeDecoder(
            d_model=d_model, 
            num_experts=moe_experts, 
            args=args
        )
        # self.predictor = MVCpredictor(
        #     d_model=d_model, 
        #     mvc_decoder_style="inner product", 
        #     explicit_zero_prob=False, 
        #     args=args
        # )

        self.predictor_use_cls = STMPredictor(
                 d_model=d_model, 
                 emb_type="cls",
                 use_cell_size=True,
                 query_activation=nn.Sigmoid,
                 args=args
            )
        
        self.predictor_use_all = STMPredictor(
                 d_model=d_model, 
                 emb_type="all",
                 use_cell_size=True,
                 model_style=args.model_style,
                 query_activation=nn.Sigmoid,
                 args=args
            )
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

    def align_data(self, ids, intensities, pixel, compound_repr):
        
        pad_value = self.args.pad_token_id
        args = self.args

        B, N = ids.shape
        device = ids.device
        if N < args.seq_len:
            pad_len = args.seq_len - N
            ids = torch.cat([ids, torch.full((B, pad_len,), pad_value, dtype=torch.float32).to(device)], dim=1)
            intensities = torch.cat([intensities, torch.full((B, pad_len,), pad_value, dtype=torch.float32).to(device)], dim=1)
            if self.args.use_image:
                padding_channel = torch.full((B, pad_len, pixel.shape[-1]), pad_value, dtype=torch.float32).to(device)
                pixel = torch.cat([pixel, padding_channel], dim=1)
            padding_repr = torch.full((B, pad_len, compound_repr.shape[-1]), pad_value, dtype=torch.float32).to(device)
            compound_repr = torch.cat([compound_repr, padding_repr], dim=1)
        else:
            ids = ids[:, :args.seq_len]
            intensities = intensities[:, :args.seq_len]
            if self.args.use_image:
                pixel = pixel[:, :args.seq_len, :]
            compound_repr = compound_repr[:, :args.seq_len, :]
        return ids, intensities, pixel, compound_repr

    def _apply_random_mask(self, x, mask_ratio=0.5, padding_mask=None, seed=None):
        """
        Apply a random mask to the input sequence x, ignoring padding positions.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, L)
            mask_ratio (float): Proportion of non-padding tokens to mask
            padding_mask (torch.Tensor, optional): Boolean mask of shape (B, L), where True indicates padding
            seed (int, optional): Random seed for reproducibility

        Returns:
            masked_x (torch.Tensor): Masked input tensor
            mask (torch.Tensor): Boolean mask indicating masked positions (True = masked)
        """
        if seed is not None:
            torch.manual_seed(seed)
        
        B, L = x.shape
        masked_x = x.clone()
        mask = torch.zeros(B, L, dtype=torch.bool, device=x.device)

        for i in range(B):
            if padding_mask is not None:
                valid_indices = (~padding_mask[i]).nonzero(as_tuple=False).squeeze(1)
            else:
                valid_indices = torch.arange(L, device=x.device)

            num_to_mask = int(len(valid_indices) * mask_ratio)
            if num_to_mask == 0:
                continue

            rand_perm = valid_indices[torch.randperm(len(valid_indices))[:num_to_mask]]
            mask[i, rand_perm] = True
            masked_x[i, rand_perm] = 0
        # print("x",x[0])
        # print("masked_x", masked_x[0])
        # print("mask",mask[0])
        return masked_x, mask

    def _apply_padding_and_masking(self, ids, spec, pixel, mol_repr):
        value_labels = ids >= 0
        ids, id_padding_labels = gatherData(ids, value_labels, self.args.pad_token_id)
        spec, _ = gatherData(spec, value_labels, self.args.pad_token_id, id_padding_labels)
        if self.args.use_image:
            pixel, _ = gatherData(pixel, value_labels, self.args.pad_token_id, id_padding_labels)
        mol_repr, _ = gatherData(mol_repr, value_labels, self.args.pad_token_id, id_padding_labels)
        spec_pad_idx = (spec==0).to(spec.device)
        masked_x, mask_labels = self._apply_random_mask(spec, self.args.mask_ratio, spec_pad_idx)
        padding_labels = id_padding_labels
        return ids, spec, pixel, mol_repr, mask_labels, padding_labels, id_padding_labels

    def _apply_cls_tokens(self, spec, pixel, mol_repr, padding_labels, mask_labels, B):
        cls_spec = self.cls_token_spectrum.expand(B, -1)
        cls_pixel = self.cls_patch.expand(B, -1, -1)
        cls_mol = self.cls_token_mol.expand(B, -1, -1)
        cls_true = torch.zeros((B, 1), dtype=torch.bool, device=padding_labels.device)
        cls_true_mask = torch.zeros((mask_labels.size(0), 1), dtype=torch.bool, device=mask_labels.device)
        if self.args.use_image:
            return (
                torch.cat([cls_spec, spec], dim=1),
                torch.cat([cls_pixel, pixel], dim=1),
                torch.cat([cls_mol, mol_repr], dim=1),
                torch.cat([cls_true, padding_labels], dim=1),
                torch.cat([cls_true_mask, mask_labels], dim=1)
            )
        else :
            return (
                torch.cat([cls_spec, spec], dim=1),
                None,
                torch.cat([cls_mol, mol_repr], dim=1),
                torch.cat([cls_true, padding_labels], dim=1),
                torch.cat([cls_true_mask, mask_labels], dim=1)
            )

    def _encode_center(self, center):
        B = center["spectrum"].size(0)
        pos = center["coord"]
        ids = center["ids"]
        spec = center["spectrum"]
        pixel = center["patch"]
        mol_repr = center["mol_repr"]
        pixel_size = center["real_pixel"]

        ids, spec, pixel, mol_repr = self.align_data(ids,spec,pixel,mol_repr)

        ids, spec, pixel, mol_repr, mask_labels, padding_labels, id_padding_labels = self._apply_padding_and_masking(ids, spec, pixel, mol_repr)
        
        # if self.args.value_normalization == "lg1p":
        #     spec = torch.log1p(spec)/ np.log(10)
        # elif self.args.value_normalization == "ln1p":
        #     spec = torch.log1p(spec)

        if self.args.use_cls_token:
            spec, pixel, mol_repr, padding_labels, mask_labels = self._apply_cls_tokens(spec, pixel, mol_repr, padding_labels, mask_labels, B)
        # print("spec",spec)
        vis_embed = None
        vis_true = None

        pixel_emb = self.pixel_encoder(pixel_size).unsqueeze(1).expand(-1, spec.shape[1], -1)

        mz_true, mz_embed, zero_paddings, *_ = self.mz_encoder(spec, mol_repr, ids, pos, padding_labels, id_padding_labels, mask_labels, args=self.args)
        if self.args.use_image:
            vis_true, vis_embed, _ = self.vit_encoder(pixel, pos, pixel_emb, mask_labels, padding_labels)
            combined = combine_vis_mz(vis_embed, mz_embed, self.args)
        else :
            combined = mz_embed
        # print('combined',combined)

        return vis_true, mz_true, vis_embed, mz_embed, combined, pixel_emb, mask_labels[:,1:], zero_paddings

    def _encode_neighbors(self, neighbors, real_pixel_size):
        B, k = neighbors["coord"].shape[:2]
        n_pos = neighbors["coord"].reshape(B * k, -1)
        n_ids = neighbors["ids"].reshape(B * k, -1)
        n_spec = neighbors["spectrum"].reshape(B * k, -1)
        if self.args.use_image:
            n_pixel = neighbors["patch"].reshape(B * k, *neighbors["patch"].shape[2:])
        else :
            n_pixel = None
        n_repr = neighbors["mol_repr"].reshape(B * k, *neighbors["mol_repr"].shape[2:])
        n_ids,n_spec,n_pixel,n_repr = self.align_data(n_ids, n_spec, n_pixel, n_repr)
        n_ids, n_spec, n_pixel, n_repr, n_mask_labels, n_padding_labels, n_id_padding_labels = self._apply_padding_and_masking(n_ids,n_spec,n_pixel,n_repr)

        # if self.args.value_normalization == "lg1p":
        #     n_spec = torch.log1p(n_spec)/ np.log(10)
        # elif self.args.value_normalization == "ln1p":
        #     n_spec = torch.log1p(n_spec)

        if self.args.use_cls_token:
            n_spec, n_pixel, n_repr, n_padding_labels, n_mask_labels = self._apply_cls_tokens(n_spec, n_pixel, n_repr, n_padding_labels, n_mask_labels, B * k)
        repeated_pixel_size = real_pixel_size.repeat_interleave(k, dim=0)
        n_batch_emb = self.pixel_encoder(repeated_pixel_size).unsqueeze(1).expand(-1, n_spec.shape[1], -1)
        _, n_mz_embed, *_ = self.mz_encoder(n_spec, n_repr, n_ids, n_pos, n_padding_labels, n_id_padding_labels, args=self.args)
        if self.args.use_image:
            _, n_vis_embed, _, n_batch_emb = self.vit_encoder(n_pixel, n_pos, n_batch_emb, n_padding_labels)
            combined = combine_vis_mz(n_vis_embed, n_mz_embed, self.args)
        else :
            combined = n_mz_embed
            
        return combined.view(B, k, *combined.shape[1:]), n_batch_emb.view(B, k, *n_batch_emb.shape[1:])
    
    def forward(self, batch, args = None, _print = False):
        """
        batch: dict, from custom DataLoader with collate_fn
        {
            "center": {
                "coord": [B, 2],
                "spectrum": [B, D],
                "patch": [B, C, H, W],
                "mol_repr": [B, D, M],
                "pixel_size":[B, 3]
            },
            "neighbors": {
                "coord": [B, k, 2],
                "spectrum": [B, k, D],
                "patch": [B, k, C, H, W],
                "mol_repr": [B, k, D, M]
            }
        }
        """
        center, neighbors = batch["center"], batch["neighbors"]

        vis_true, mz_true, vis_embed, mz_embed, center_embedding, batch_emb, mask_labels, zero_paddings= self._encode_center(center)
        neighbor_embedding, neighbor_batch_emb = self._encode_neighbors(neighbors, center["real_pixel"])
        # print("centershape",center_embedding.shape)
        # print("center_emb",center_embedding)
        decoded = self.transformer_decoder(center_embedding, attn_mask = zero_paddings)
        
        recon = self.moe_decoder(decoded)
        # recon = decoded

        # pred = self.predictor(
        #     center_embedding,
        #     neighbor_embedding,
        #     mz_embed,
        #     batch_emb=batch_emb,
        #     n_batch_emb=neighbor_batch_emb,
        #     do_sample=False,
        #     predict_by_self=True,
        #     predict_by_neighbor=True,
        #     coordinates=center["coord"],
        #     args=self.args
        # )

        zero_paddings = ~zero_paddings[:,1:]
  
        pred={}
        
        # print('recon',recon)
        # print('mz',mz_embed)
        pred['cls_self']=self.predictor_use_cls(center_embedding, mz_embed, batch_emb)
        pred['cls_neighbor']=self.predictor_use_cls(neighbor_embedding, mz_embed, batch_emb)
        pred['all_self']=self.predictor_use_all(recon, mz_embed, batch_emb)
        # print('pred',pred['all_self'])
        cl_loss = None

        if self.args.use_image:
            cl_loss = calc_contrastive_loss(vis_embed[:, 1:], mz_embed[:, 1:], self.args)

        return vis_true, mz_true, recon, pred, cl_loss, mask_labels, zero_paddings
    

class autoencoder(nn.Module):
    def __init__(self, d_model=512, 
                 transformer_decoder_heads=8, 
                 transformer_decoder_layers=12, 
                 moe_experts=4, 
                 args=None):
        super().__init__()
        self.args = args
        self.mask_embedding = nn.Parameter(torch.randn(1, 1, 512)*0.02)  # shape: (1, 1, 512)
        # positional embedding for sequence length
        self.spec_pos_embedding = nn.Parameter(torch.randn(1, self.args.seq_len, 512))

        # 输入 proj：将原始 spec 映射到 Transformer 维度
        self.spec_input_proj = nn.Linear(1, 512)

        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        self.spec_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)

        # Transformer 解码器
        decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8, batch_first=True)
        self.spec_decoder = nn.TransformerDecoder(decoder_layer, num_layers=4)

        # 输出 proj：将解码器输出映射回原始维度
        self.spec_output_proj = nn.Linear(512, 1)

    def _apply_random_mask(self, x, mask_ratio=0.5, padding_mask=None, seed=None):
        """
        Apply a random mask to the input sequence x, ignoring padding positions.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, L)
            mask_ratio (float): Proportion of non-padding tokens to mask
            padding_mask (torch.Tensor, optional): Boolean mask of shape (B, L), where True indicates padding
            seed (int, optional): Random seed for reproducibility

        Returns:
            masked_x (torch.Tensor): Masked input tensor
            mask (torch.Tensor): Boolean mask indicating masked positions (True = masked)
        """
        if seed is not None:
            torch.manual_seed(seed)
        
        B, L = x.shape
        masked_x = x.clone()
        mask = torch.zeros(B, L, dtype=torch.bool, device=x.device)

        for i in range(B):
            if padding_mask is not None:
                valid_indices = (~padding_mask[i]).nonzero(as_tuple=False).squeeze(1)
            else:
                valid_indices = torch.arange(L, device=x.device)

            num_to_mask = int(len(valid_indices) * mask_ratio)
            if num_to_mask == 0:
                continue

            rand_perm = valid_indices[torch.randperm(len(valid_indices))[:num_to_mask]]
            mask[i, rand_perm] = True
            masked_x[i, rand_perm] = 0

        return masked_x, mask

    def forward(self, batch, args):
        center, neighbors = batch["center"], batch["neighbors"]
        spec = center["spectrum"]
        B, L = spec.shape

        if L > args.seq_len:
            spec = spec[:,:args.seq_len]
            L = args.seq_len

        # 创建 padding mask：True 表示是 padding，需要被 mask 掉
        padding_mask = (spec == 0)

        # 应用随机 mask
        # masked_x, mask_labels = self._apply_random_mask(spec, self.args.mask_ratio, padding_mask)

        padding_and_masking = padding_mask   # True 的位置应被跳过

        # 将输入投影到 512 维
        spec_embed = self.spec_input_proj(spec.unsqueeze(-1))  # (B, L, 512)
        spec_embed = spec_embed + self.spec_pos_embedding[:, :L, :]  # 加 positional embedding


        # expand mask embedding to (B, L, 512)
        mask_token = self.mask_embedding.expand(B, L, -1)
        # spec_embed = torch.where(mask_labels.unsqueeze(-1), mask_token, spec_embed)

        # 编码器处理
        memory = self.spec_encoder(spec_embed, src_key_padding_mask=padding_and_masking)

        # 解码器处理
        decoded = self.spec_decoder(
            tgt=memory,
            memory=memory,
            tgt_key_padding_mask=padding_mask
        )

        # 输出映射回原始维度
        spec = self.spec_output_proj(decoded).squeeze(-1)
        return spec, padding_mask
    
    
