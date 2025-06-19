import torch as th
import numpy as np
import torch
import argparse
def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")

def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)

def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total}")
    print(f"Trainable parameters: {trainable}")

def get_allocated_mb():
    return torch.cuda.memory_allocated() / 1024**2  # 单位：MB

def track_memory_stepwise(label, prev_allocated):
    current = get_allocated_mb()
    delta = current - prev_allocated
    print(f"[{label}] Current: {current:.2f} MB | +{delta:.2f} MB since last")
    return current

def cat_vis_mz_with_cls(vis,mz):
    vis_cls = vis[:,0,:].unsqueeze(1)
    mz_cls = mz[:,0,:].unsqueeze(1)
    combine_cls = vis_cls + mz_cls
    embed = torch.cat((mz[:,1:,:], vis[:,1:,:]), dim=1)
    embed = torch.cat((combine_cls, embed), dim=1)
    return embed

def combine_vis_mz(vis, mz, args=None):
    if args.image_combine_mz == "cat_seq":
        if args.use_cls_token:
            vis_cls = vis[:,0,:].unsqueeze(1)
            mz_cls = mz[:,0,:].unsqueeze(1)
            combine_cls = vis_cls + mz_cls
            embed = torch.cat((mz[:,1:,:], vis[:,1:,:]), dim=1)
            embed = torch.cat((combine_cls, embed), dim=1)
        else:
            embed = torch.cat((mz, vis), dim=1)

    elif args.image_combine_mz == "cat_dim":
        embed = torch.cat((mz, vis), dim=2)
    elif args.image_combine_mz == "add":
        embed = mz + vis

    return embed
    
import torch

def gatherData(data, labels=None, pad_token_id=0, padding_labels=None):
    """
    data: [B, N] or [B, N, C]
    labels: [B, N]  -> bool tensor for selection
    padding_labels: [B, max_num] -> optional mask, if provided will align data based on this
    """

    is_3d = data.dim() == 3
    B, N = data.shape[:2]
    C = data.shape[2] if is_3d else None

    # case 1: 已提供 padding_labels -> 直接对齐
    if padding_labels is not None:
        max_num = padding_labels.shape[1]
        if is_3d:
            # [B, max_num, C]
            index = (~padding_labels).nonzero(as_tuple=False)  # 找到有效位置的 index
            output = torch.zeros(B, max_num, C, device=data.device, dtype=data.dtype)
            for b in range(B):
                valid_idx = (~padding_labels[b]).nonzero(as_tuple=False).squeeze(1)
                output[b, valid_idx] = data[b, :][valid_idx]
            return output, padding_labels
        else:
            # [B, max_num]
            output = torch.full((B, max_num), pad_token_id, device=data.device, dtype=data.dtype)
            for b in range(B):
                valid_idx = (~padding_labels[b]).nonzero(as_tuple=False).squeeze(1)
                output[b, valid_idx] = data[b, :][valid_idx]
            return output, padding_labels

    # case 2: 没有 padding_labels -> 正常处理逻辑
    assert labels is not None, "Either labels or padding_labels must be provided"

    value_nums = labels.sum(1)
    max_num = int(value_nums.max().item())

    if is_3d:
        fake_data = torch.full((B, max_num, C), pad_token_id, device=data.device, dtype=data.dtype)
        data = torch.cat([data, fake_data], dim=1)
    else:
        fake_data = torch.full((B, max_num), pad_token_id, device=data.device, dtype=data.dtype)
        data = torch.cat([data, fake_data], dim=1)

    fake_label = torch.full((B, max_num), 1, device=labels.device)
    none_labels = ~labels
    labels = labels.float()
    labels[none_labels] = -float('Inf')

    tmp_data = torch.tensor([(i + 1) * 20000 for i in range(labels.shape[1], 0, -1)], device=labels.device)
    labels += tmp_data

    labels = torch.cat([labels, fake_label], dim=1)
    fake_label_gene_idx = labels.topk(max_num).indices  # [B, max_num]

    if is_3d:
        index = fake_label_gene_idx.unsqueeze(-1).expand(-1, -1, C)  # [B, max_num, C]
        new_data = torch.gather(data, dim=1, index=index)  # [B, max_num, C]
    else:
        new_data = torch.gather(data, dim=1, index=fake_label_gene_idx)  # [B, max_num]

    padding_labels = (new_data == pad_token_id) if not is_3d else (new_data == pad_token_id).all(dim=-1)

    return new_data, padding_labels

def getEncoerDecoderData(data, data_raw, config):
    decoder_data = data.clone().detach()
    decoder_data_padding = torch.full_like(data, False, dtype=torch.bool).to(data.device)

    encoder_data_labels = data_raw > 0
    encoder_data, encoder_data_padding = gatherData(decoder_data, encoder_data_labels,
                                                    config['pad_token_id'])
    new_data_raw = data_raw
    data_gene_ids = torch.arange(data.shape[1], device=data.device).repeat(data.shape[0], 1)
    encoder_position_gene_ids, _ = gatherData(data_gene_ids, encoder_data_labels,
                                                config['pad_token_id'])
    decoder_position_gene_ids = data_gene_ids
    data_mask_labels = None

    encoder_position_gene_ids[encoder_data_padding] = config["seq_len"]
    decoder_position_gene_ids[decoder_data_padding] = config["seq_len"]

    return encoder_data, encoder_position_gene_ids, encoder_data_padding, encoder_data_labels, decoder_data, decoder_data_padding, new_data_raw, data_mask_labels, decoder_position_gene_ids

class EMA:
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_avg = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_avg.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}