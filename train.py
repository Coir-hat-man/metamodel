import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
import time
import sys
import math
from loader import *
import torch
from torch.optim.lr_scheduler import LambdaLR
from models.whole_pipeline import MetablisumModelV3
from torch.utils.tensorboard import SummaryWriter
from utils import *
from loader import collate_fn
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity as ssim

from utils import EMA
from configs.config import *
import matplotlib.pyplot as plt
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)


def log1pf_normalize_tensor(X: torch.Tensor, args) -> torch.Tensor:
    """
    对输入 Tensor 进行中位数归一化并进行 log1p 转换（log1PF）。

    参数:
        X (torch.Tensor): 2D Tensor，shape 为 (samples, features)，通常是表达谱或强度矩阵。
    
    返回:
        torch.Tensor: 归一化并 log1p 转换后的结果。
    """

        # print(X.max(),X[X > 0].min())
    # max_val = X.max()

    # # 非零最小值
    # nonzero_min_val = X[X > 0].min()
    # if max_val-nonzero_min_val< 100:
    #     x_norm = X
    # else: 
        # 每个样本的总强度

    X = torch.clamp(X, max=100000)

    total_counts = X.sum(dim=1, keepdim=True)

    # 计算非零样本的中位数 count depth
    flattened_total = total_counts.view(-1)
    nonzero_total = flattened_total[flattened_total > 0]
    if nonzero_total.numel() == 0:
        print("Warning: all total_counts are zero, using default median 1.0")
        median_count = torch.tensor(1.0, device=X.device)
    else:
        median_count = nonzero_total.median().detach()

    # 避免除以 0
    total_counts[total_counts == 0] = 1.0

    # 中位数归一化
    X = X / total_counts * median_count

    # log1p 转换
    if args.value_normalization == "lg1p":
        X_log1p = torch.log1p(X) / np.log(10)
    elif args.value_normalization == "ln1p":
        X_log1p = torch.log1p(X)
    x_norm = X_log1p

    # print(X_log1p.max(), X_log1p[X_log1p > 0].min())
    # if len(original_shape) == 3:
    #     X_log1p = X_log1p.reshape(original_shape)
        # Step 2: 按每个样本做标准化（可选按最后一维或第二维）
    # X_log1p = X
    # mean = X_log1p.mean(dim=-1, keepdim=True)
    # std = X_log1p.std(dim=-1, keepdim=True) + 1e-6  # 避免除0
    # x_norm = (X_log1p - mean) / std
    # print(X_log1p.max(),X_log1p.min())
    # x_min = X_log1p.min(dim=-1, keepdim=True).values
    # x_max = X_log1p.max(dim=-1, keepdim=True).values
    # x_norm = (X_log1p - x_min) / (x_max - x_min + 1e-6)  # 加 epsilon 避免除0
    # print(x_norm.max(),x_norm.min())
    return x_norm

def get_cosine_schedule_with_warmup(optimizer, batch_per_epoch, args):
    num_training_steps = args.total_epoches * batch_per_epoch

    if args.num_warmup_steps > 1:
        warmup_steps = args.num_warmup_steps
    else:
        warmup_steps = int(num_training_steps * args.num_warmup_steps)
    print("warmup_steps", warmup_steps)
    min_lr_ratio = args.min_lr / args.lr  # 假设 args.lr 是初始学习率

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, num_training_steps - warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        # print(cosine_decay)
        return cosine_decay * (1 - min_lr_ratio) + min_lr_ratio

    return LambdaLR(optimizer, lr_lambda), warmup_steps



def get_cosine_schedule_with_warmupv2(optimizer, batch_per_epoch, args):
    num_training_steps = batch_per_epoch

    if args.num_warmup_steps>1:
        warmup_steps = args.num_warmup_steps
    else:
        warmup_steps = int(num_training_steps * args.num_warmup_steps)

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, num_training_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))  # <-- 不做 min_lr 比例归一化

    return LambdaLR(optimizer, lr_lambda), warmup_steps


def move_to_device(batch, device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device).float()
    elif isinstance(batch, dict):
        return {k: move_to_device(v, device) for k, v in batch.items()}
    elif isinstance(batch, list):
        return [move_to_device(b, device) for b in batch]
    elif isinstance(batch, tuple):
        return tuple(move_to_device(b, device) for b in batch)
    else:
        return batch
    

def build_trained_model(model_dir='/home/huangrp/WorkSpace/Metabolic_model/saved_model',
                use_pretrain=True,
                resume_ckpt_path=None,     
                args=None):

    def get_dataloader(dataset_name):
        sample_data = torch.load(
            os.path.join(args.file_load_path, "cache_datasets", f"{dataset_name}.pt"),
            weights_only=False  # ✅ 明确允许完整反序列化
        )
        dataset = build_dataset_from_samples(samples=sample_data, args=args)
        return torch.utils.data.DataLoader(
            dataset,
            num_workers=4,
            batch_size=args.batch_size,
            shuffle=args.shuffle,
            drop_last=True,
            collate_fn=collate_fn
        )

    def log_losses(writer, losses_dict, step):
        for k, v in losses_dict.items():
            writer.add_scalar(k, v, step)

    dataset_list, args = data_load_main(args)
    if args.num_datasets != None:
        dataset_list = dataset_list[:args.num_datasets]
    if args.training_num != None:
        if args.start_dataset:
            dataset_index = dataset_list.index(args.start_dataset)
            dataset_list = dataset_list[dataset_index:dataset_index+args.training_num]
        else:
            dataset_list = dataset_list[:args.training_num]
    if args.model_size not in model_configs:
        raise ValueError(f"Unknown model size: {args.model_size}")

    config = model_configs[args.model_size]

    model = MetablisumModelV3(
        d_model=config["d_model"],
        decoder_hidden_dim=config["decoder_hidden_dim"],
        transformer_decoder_heads=config["transformer_decoder_heads"],
        transformer_decoder_layers=config["transformer_decoder_layers"],
        moe_experts=config["moe_experts"],
        args=args
    ).to(args.device).train()
    if use_pretrain and model_dir:
        model = torch.load(f'{model_dir}/metabolic_model.pth').to(args.device).train()
        print('\033[32mUse pretrained model.\033[0m')
        return model


    
    count_parameters(model)
    mse = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    step_counter = 0
    total_batches = 0
    # for dataset_name in dataset_list:
    #     print(dataset_name)
    #     dataset_size = len(torch.load(os.path.join(args.file_load_path, "cache_datasets", f"{dataset_name}.pt")))
    #     print(1)
    #     num_batches = dataset_size // args.batch_size  # 因为 drop_last=True
    #     total_batches += num_batches

    cache_file = os.path.join(args.file_load_path, "cache_datasets","dataset_sizes.json")
    if os.path.exists(cache_file):
        dataset_sizes = json.load(open(cache_file))
    else:
        dataset_sizes = {}
        print("\033[34mCaching dataset sizes...\033[0m")
        for dataset_name in tqdm(dataset_list, desc="Caching dataset sizes", dynamic_ncols=True):
            sample_data = torch.load(os.path.join(args.file_load_path, "cache_datasets", f"{dataset_name}.pt"))
            dataset_sizes[dataset_name] = len(sample_data)
        json.dump(dataset_sizes, open(cache_file, "w"))
    if args.num_datasets or args.start_dataset is not None:
        dataset_sizes = {name: dataset_sizes[name] for name in dataset_list}

    for dataset_name, lenth in dataset_sizes.items():
        num_batches = lenth  // args.batch_size  # 向上取整
        total_batches += num_batches
    best_model_overall = None
    best_loss_overall = float('inf')

    best_step_model = None
    best_step_pcc = -float('inf')
    best_step_path = None
    scheduler, warmup_steps = get_cosine_schedule_with_warmup(optimizer, total_batches, args)
    resume_dataset_index = 0
    resume_ckpt_path = args.resume_ckpt_path
    current_epoch = 0
    if resume_ckpt_path is not None and os.path.exists(resume_ckpt_path):
        checkpoint = torch.load(resume_ckpt_path, map_location=args.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.last_epoch = checkpoint['scheduler_step']
        step_counter = checkpoint['step']
        current_epoch = checkpoint['epoch']
        current_dataset_name = checkpoint['current_dataset_name']
        # current_dataset_name = '2020-12-09_23h20m46s'
        print(current_dataset_name)
        if current_dataset_name and current_dataset_name in dataset_list:
            resume_dataset_index = dataset_list.index(current_dataset_name) + 1
            if resume_dataset_index >= len(dataset_list):
                print("\033[33mLast dataset was completed. Restarting from beginning.\033[0m")
                resume_dataset_index = 0
                current_epoch = current_epoch + 1

        print(f"\033[33mResumed training from {resume_ckpt_path}, starting from dataset index {resume_dataset_index} ({dataset_list[resume_dataset_index]})\033[0m")
    if args.use_ema:
        ema = EMA(model, decay=0.999)
    writer = SummaryWriter(log_dir=f'{args.log_dir}/train_logs/')
    for epoch in range(args.total_epoches):
        if epoch == 0:
            epoch = current_epoch
        epoch_start_time = time.time()  # <-- 开始时间
        for dataset_index, dataset_name in enumerate(dataset_list[resume_dataset_index:], start=resume_dataset_index + 1):
            dataset_start_time = time.time()  # <-- 开始时间
            print(f"\033[32mTraining on dataset {dataset_index}/{len(dataset_list)}: {dataset_name}...\033[0m")
            writer_step = SummaryWriter(log_dir=f'{args.log_dir}/train_logs_steps/{dataset_name}')
            dataloader = get_dataloader(dataset_name)

            progress = tqdm(total=len(dataloader), dynamic_ncols=True, file=sys.stdout)
            best_model_current_dataset = None
            best_pcc_current_dataset = -float('inf')
            best_loss_current_dataset = float('inf')
            for batch in dataloader:
                batch = move_to_device(batch, args.device)
                # batch["center"]["spectrum"] = log1pf_normalize_tensor(batch["center"]["spectrum"], args)
                # batch["neighbors"]["spectrum"] = log1pf_normalize_tensor(batch["neighbors"]["spectrum"], args)
                with torch.amp.autocast('cuda'):
                    vis_emb, mz_emb, recon_pred, pred_dict, cl_loss, mask_labels, zero_paddings = model(batch, args=args)
                target_spec = batch["center"]["spectrum"]
                target_spec = target_spec[:, :args.seq_len] if args.seq_len < target_spec.shape[1] else target_spec
                
                cls_self_loss = mse_with_mask(pred_dict['cls_self'], target_spec, zero_paddings)
                cls_neighbor_loss = mse_with_mask(pred_dict['cls_neighbor'], target_spec, zero_paddings)
                # cls_self_loss = mse(pred_dict['cls_self'], target_spec)
                # cls_neighbor_loss = mse(pred_dict['cls_neighbor'], target_spec)
                all_self_loss = mse_with_mask(pred_dict['all_self'], target_spec, mask_labels)
                non_mask_loss = mse_with_mask(pred_dict['all_self'], target_spec, ~mask_labels)
                # print(pred_dict['all_self'][0],target_spec[0])

                cls_self_pcc = cal_pcc(pred_dict['cls_self'], target_spec, zero_paddings)
                cls_neighbor_pcc = cal_pcc(pred_dict['cls_neighbor'], target_spec, zero_paddings)
                all_self_pcc = cal_pcc(pred_dict['all_self'], target_spec, mask_labels)
                if not mask_labels.any():
                    all_self_loss = mse_with_mask(pred_dict['all_self'], target_spec, zero_paddings)
                    all_self_pcc = cal_pcc(pred_dict['all_self'], target_spec, zero_paddings)

                total_loss = (cls_self_loss * args.pred_loss_weight +
                            cls_neighbor_loss * args.pred_loss_weight +
                            all_self_loss * args.recon_loss_weight)
                
                if args.use_no_mask_loss:
                    total_loss = (cls_self_loss * args.pred_loss_weight +
                        cls_neighbor_loss * args.pred_loss_weight +
                        all_self_loss * args.recon_loss_weight + 
                        non_mask_loss * args.recon_loss_weight)


                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                # EMA 更新
                if args.use_ema:
                    ema.update()

                scheduler.step()
                step_counter += 1

                # ----------- Logging ----------
                loss_dict = {
                    "cls_self_loss": cls_self_loss.item(),
                    "cls_neighbor_loss": cls_neighbor_loss.item(),
                    "all_self_loss": all_self_loss.item(),
                    "non_mask_loss": non_mask_loss.item(),
                    "cls_self_pcc": cls_self_pcc,
                    "cls_neighbor_pcc": cls_neighbor_pcc,
                    "all_self_pcc": all_self_pcc,
                    "Loss": total_loss.item(),
                    "lr": optimizer.param_groups[0]['lr'],
                }
                log_losses(writer_step, loss_dict, step_counter)
                log_losses(writer, loss_dict, step_counter)

                # ----------- 保存 best 模型 per dataset ----------

                if all_self_pcc > best_pcc_current_dataset:
                    best_pcc_current_dataset = all_self_pcc

                    if args.use_ema:
                        ema.apply_shadow()
                
                    best_model_current_dataset = {
                        'epoch': epoch,
                        'model_state_dict': copy.deepcopy(model.state_dict()),
                        'optimizer_state_dict': copy.deepcopy(optimizer.state_dict()),
                        'scheduler_step': scheduler.last_epoch,
                        'loss': total_loss,
                        'step': step_counter,
                        'current_dataset_name': dataset_name
                    }

                    if args.use_ema:
                        ema.restore()


                # ----------- 保存最优 loss 模型 overall ----------
                if total_loss.item() < best_loss_overall:
                    best_loss_overall = total_loss.item()
                    if args.use_ema:
                        ema.apply_shadow()
                    best_model_overall = {
                        'epoch': epoch,
                        'model_state_dict': copy.deepcopy(model.state_dict()),
                        'optimizer_state_dict': copy.deepcopy(optimizer.state_dict()),
                        'scheduler_step': scheduler.last_epoch,
                        'loss': total_loss,
                        'step': step_counter,
                        'current_dataset_name': dataset_name
                    }
                    if args.use_ema:
                        ema.restore()

                # ----------- 按 step 保存（只保留更好的 step） ----------
                if all_self_pcc > best_step_pcc:
                    best_step_pcc = all_self_pcc
                    if args.use_ema:
                        ema.apply_shadow()
                    best_step_model = {
                        'epoch': epoch,
                        'model_state_dict': copy.deepcopy(model.state_dict()),
                        'optimizer_state_dict': copy.deepcopy(optimizer.state_dict()),
                        'scheduler_step': scheduler.last_epoch,
                        'loss': total_loss,
                        'step': step_counter,
                        'current_dataset_name': dataset_name
                    }
                    if args.use_ema:
                        ema.restore()

                    if best_step_path and os.path.exists(best_step_path):
                        os.remove(best_step_path)

                    best_step_path = os.path.join(model_dir, f'checkpoint{step_counter}.pth')
                    os.makedirs(model_dir, exist_ok=True)
                    torch.save(best_step_model, best_step_path)

                # ----------- 日志输出 ----------
                progress.set_description(
                    f"Epoch {epoch+1}/{args.total_epoches} | Dataset {dataset_index}/{len(dataset_list)} | Step {step_counter} | "
                    f"Loss: {total_loss.item():.4f}({cls_self_loss:.4f}+{cls_neighbor_loss:.4f}+{all_self_loss:.4f}+{cls_self_pcc:.4f}+{cls_neighbor_pcc:.4f}+{all_self_pcc:.4f}) | LR: {optimizer.param_groups[0]['lr']:.2e}"
                )

                progress.update(1)
                

            progress.close()
            writer.close()
            dataset_time = time.time() - dataset_start_time
            print(f"\033[36mDataset {dataset_name} training completed in {dataset_time:.2f} seconds ({dataset_time/60:.2f} minutes).\033[0m")
            if (dataset_index % 50 == 0) or (dataset_index == len(dataset_list)):
                ckpt_path = os.path.join(model_dir, f'checkpoint_dataset_{dataset_index}_epoch{epoch}.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_step': scheduler.last_epoch,
                    'loss': total_loss,
                    'step': step_counter,
                    'current_dataset_name': dataset_name
                }, ckpt_path)
                print(f"\033[36mSaved checkpoint at dataset {dataset_index}: {ckpt_path}\033[0m")
        resume_dataset_index = 0
        epoch_time = time.time() - epoch_start_time  # <-- 计算耗时
        print(f"\033[35mEpoch {epoch+1} completed in {epoch_time:.2f} seconds ({epoch_time/60:.2f} minutes).\033[0m")
            # test_one(args,model,dataset_name)

    # 最后保存 loss 最低的 overall 模型（使用 EMA 权重）
    os.makedirs(model_dir, exist_ok=True)
    if args.use_ema:
        ema.apply_shadow()
    torch.save(best_model_overall, os.path.join(model_dir, f'checkpoint_final_best_loss.pth'))
    if args.use_ema:
        ema.restore()
    
    return model




def mse_with_mask(pred, true, mask_label):
    mse = (pred - true) ** 2  # 逐元素的 MSE
    mse_masked = mse[mask_label]  # 选出 mask 为 True 的位置

    mean_mse = mse_masked.mean()
    return mean_mse

def mse_with_mask_inv(pred, true, mask_label):
    mse = (pred - true) ** 2  # 逐元素的 MSE
    mse_masked = mse[~mask_label]  # 选出 mask 为 True 的位置

    mean_mse = mse_masked.mean()
    return mean_mse

def cal_pcc(matrix1, matrix2, mask=None):
    if mask is not None:
        # 仅选取 mask 为 True 的位置
        vector_a = matrix1[mask]
        vector_b = matrix2[mask]
    else:
        # 默认使用所有元素
        vector_a = matrix1.flatten()
        vector_b = matrix2.flatten()

    mean_a = torch.mean(vector_a)
    mean_b = torch.mean(vector_b)
    
    covariance = torch.sum((vector_a - mean_a) * (vector_b - mean_b)) / (vector_a.numel() - 1)
    std_dev_a = torch.std(vector_a, unbiased=False)
    std_dev_b = torch.std(vector_b, unbiased=False)
    
    pcc = covariance / (std_dev_a * std_dev_b)
    return pcc


def cal_pcc_inv(matrix1, matrix2, mask=None):
    if mask is not None:
        mask = ~mask
        # 仅选取 mask 为 True 的位置
        vector_a = matrix1[mask]
        vector_b = matrix2[mask]
    else:
        # 默认使用所有元素
        vector_a = matrix1.flatten()
        vector_b = matrix2.flatten()

    mean_a = torch.mean(vector_a)
    mean_b = torch.mean(vector_b)
    
    covariance = torch.sum((vector_a - mean_a) * (vector_b - mean_b)) / (vector_a.numel() - 1)
    std_dev_a = torch.std(vector_a, unbiased=True)
    std_dev_b = torch.std(vector_b, unbiased=True)
    
    pcc = covariance / (std_dev_a * std_dev_b)
    return pcc

def build_autoencoder_model(model_dir='/home/huangrp/WorkSpace/Metabolic_model/saved_model',
                use_pretrain=True,
                args=None):

    def get_dataloader(dataset_name):
        sample_data = torch.load(os.path.join(args.file_load_path, "cache_datasets", f"{dataset_name}.pt"))
        dataset = build_dataset_from_samples(samples=sample_data, args=args)
        return torch.utils.data.DataLoader(
            dataset,
            num_workers=4,
            batch_size=args.batch_size,
            shuffle=args.shuffle,
            drop_last=True,
            collate_fn=collate_fn
        )

    def log_losses(writer, losses_dict, step):
        for k, v in losses_dict.items():
            writer.add_scalar(k, v, step)

    dataset_list, args = data_load_main(args)
    if args.num_datasets != None:
        dataset_list = dataset_list[:args.num_datasets]
    if args.training_num != None:
        if args.start_dataset:
            dataset_index = dataset_list.index(args.start_dataset)
            dataset_list = dataset_list[dataset_index:dataset_index+args.training_num]
        else:
            dataset_list = dataset_list[:args.training_num]

    model = autoencoder(args=args).to(args.device).train()
    if use_pretrain and model_dir:
        model = torch.load(f'{model_dir}/metabolic_model.pth').to(args.device).train()
        print('\033[32mUse pretrained model.\033[0m')
        return model


    count_parameters(model)
    mse = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    step_counter = 0
    total_batches = 0
    cache_file = os.path.join(args.file_load_path, "cache_datasets","test_dataset_sizes.json")
    if os.path.exists(cache_file):
        dataset_sizes = json.load(open(cache_file))
    else:
        dataset_sizes = {}
        print("\033[34mCaching dataset sizes...\033[0m")
        for dataset_name in tqdm(dataset_list, desc="Caching dataset sizes", dynamic_ncols=True):
            sample_data = torch.load(os.path.join(args.file_load_path, "cache_datasets", f"{dataset_name}.pt"))
            dataset_sizes[dataset_name] = len(sample_data)
        json.dump(dataset_sizes, open(cache_file, "w"))
    if args.num_datasets or args.start_dataset is not None:
        dataset_sizes = {name: dataset_sizes[name] for name in dataset_list}

    for dataset_name, lenth in dataset_sizes.items():
        num_batches = lenth  // args.batch_size  # 向上取整
        total_batches += num_batches
    best_model_overall = None
    best_loss_overall = float('inf')

    best_step_model = None
    best_step_pcc = -float('inf')
    best_step_path = None
    scheduler, warmup_steps = get_cosine_schedule_with_warmup(optimizer, total_batches, args)
    writer = SummaryWriter(log_dir=f'{args.log_dir}/train_logs/')
    for epoch in range(args.total_epoches):
        for dataset_index, dataset_name in enumerate(dataset_list, start=1):
            best_model_current_dataset = None
            best_pcc_current_dataset = -float('inf')
            best_loss_current_dataset = float('inf')
            print(f"\033[32mTraining on dataset {dataset_index}/{len(dataset_list)}: {dataset_name}...\033[0m")
            writer_step = SummaryWriter(log_dir=f'{args.log_dir}/train_logs_steps/{dataset_name}')
            dataloader = get_dataloader(dataset_name)

            progress = tqdm(total=len(dataloader), dynamic_ncols=True, file=sys.stdout)

            for batch in dataloader:
                batch = move_to_device(batch, args.device)

                # batch["center"]["spectrum"] = log1pf_normalize_tensor(batch["center"]["spectrum"],args)
                # batch["neighbors"]["spectrum"] = log1pf_normalize_tensor(batch["neighbors"]["spectrum"],args)

                spec, padding_mask = model(batch, args=args)
                
                mask = ~padding_mask
                target_spec = batch["center"]["spectrum"]
                target_spec = target_spec[:, :args.seq_len] if args.seq_len < target_spec.shape[1] else target_spec

                # print(spec[0],target_spec[0])
                # if args.value_normalization == "lg1p":
                #     target_spec = torch.log1p(target_spec) / np.log(10)
                # elif args.value_normalization == "ln1p":
                #     target_spec = torch.log1p(target_spec)
                if not mask.any():
                    all_self_loss = mse(spec, target_spec)
                    all_self_pcc = cal_pcc(spec, target_spec)
                    
                else:
                    all_self_loss = mse_with_mask(spec, target_spec, mask)
                    all_self_pcc = cal_pcc(spec, target_spec, mask)
                
                total_loss = (all_self_loss * args.pred_loss_weight
                              )
                            #   cl_loss * args.cl_loss_weight)

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                scheduler.step()

                step_counter += 1
                loss_dict = {
                    "all_self_loss": all_self_loss.item(),

                    # "CL loss": cl_loss.item(),
                    "all_self_pcc": all_self_pcc,
                    "Loss": total_loss.item(),
                    "lr": optimizer.param_groups[0]['lr'],
                }
                log_losses(writer_step, loss_dict, step_counter)
                log_losses(writer, loss_dict, step_counter)

                if all_self_pcc > best_pcc_current_dataset:
                    best_pcc_current_dataset = all_self_pcc

                
                    best_model_current_dataset = {
                        'epoch': epoch,
                        'model_state_dict': copy.deepcopy(model.state_dict()),
                        'optimizer_state_dict': copy.deepcopy(optimizer.state_dict()),
                        'scheduler_step': scheduler.last_epoch,
                        'loss': total_loss,
                        'step': step_counter,
                        'current_dataset_name': dataset_name
                    }
                # ----------- 保存最优 loss 模型 overall ----------
                if total_loss.item() < best_loss_overall:
                    best_loss_overall = total_loss.item()

                    best_model_overall = {
                        'epoch': epoch,
                        'model_state_dict': copy.deepcopy(model.state_dict()),
                        'optimizer_state_dict': copy.deepcopy(optimizer.state_dict()),
                        'scheduler_step': scheduler.last_epoch,
                        'loss': total_loss,
                        'step': step_counter,
                        'current_dataset_name': dataset_name
                    }


                # ----------- 按 step 保存（只保留更好的 step） ----------
                if all_self_pcc > best_step_pcc:
                    best_step_pcc = all_self_pcc

                    best_step_model = {
                        'epoch': epoch,
                        'model_state_dict': copy.deepcopy(model.state_dict()),
                        'optimizer_state_dict': copy.deepcopy(optimizer.state_dict()),
                        'scheduler_step': scheduler.last_epoch,
                        'loss': total_loss,
                        'step': step_counter,
                        'current_dataset_name': dataset_name
                    }


                    if best_step_path and os.path.exists(best_step_path):
                        os.remove(best_step_path)

                    best_step_path = os.path.join(model_dir, f'checkpoint{step_counter}.pth')
                    os.makedirs(model_dir, exist_ok=True)
                    torch.save(best_step_model, best_step_path)

                # ----------- 日志输出 ----------
                progress.set_description(
                    f"Epoch {epoch+1}/{args.total_epoches} | Dataset {dataset_index}/{len(dataset_list)} | Step {step_counter} | "
                    f"Loss: {total_loss.item():.4f}+{all_self_pcc:.4f}) | LR: {optimizer.param_groups[0]['lr']:.2e}"
                )

                progress.update(1)
                

            progress.close()
            writer.close()
            if best_model_current_dataset is not None:
                torch.save(best_model_current_dataset,os.path.join(model_dir, f'best_{dataset_name}.pth'))
            # test_one(args,model,dataset_name)

    # 最后保存 loss 最低的 overall 模型（使用 EMA 权重）
    os.makedirs(model_dir, exist_ok=True)
    torch.save(best_model_overall, os.path.join(model_dir, f'checkpoint_final_best_loss.pth'))
    return model

def test_model(args):
    # 加载模型
    file_load_path = args.file_load_path
    combined_files = os.path.join(file_load_path,"combined_table.pkl")
    combined_table = pd.read_pickle(combined_files)
    train_table = combined_table[["mol_formula"]][:].copy()
    args.mask_label_id = len(train_table)
    dataset_name = args.test_dataset_name
    checkpoint = torch.load(args.ckpt_path, map_location=args.device)
    config = model_configs[args.model_size]
    model = MetablisumModelV3(
        d_model=config["d_model"],
        decoder_hidden_dim=config["decoder_hidden_dim"],
        transformer_decoder_heads=config["transformer_decoder_heads"],
        transformer_decoder_layers=config["transformer_decoder_layers"],
        moe_experts=config["moe_experts"],
        args=args
    ).to(args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    # print(model.state_dict())
    results_dict = {
    'dataset': [],
    'mse': [],
    'pcc': [],
    'ssim': []
    }
    # 加载数据
    dataset_list, args = data_load_main(args)

    if args.start_dataset:
        dataset_index = dataset_list.index(args.start_dataset)
        test_dataset_list = dataset_list[dataset_index:dataset_index+args.test_dataset_num]
    else:
        test_dataset_list = dataset_list[-args.test_dataset_num:]
    for dataset_index, dataset_name in enumerate(test_dataset_list,start=1):
        test_start_time = time.time()
        print(f"\033[32mTesting on dataset {dataset_index}/{len(test_dataset_list)}: {dataset_name}...\033[0m")
        writer_step = SummaryWriter(log_dir=f'{args.log_dir}/train_logs_steps/{dataset_name}')
        sample_data = torch.load(
            os.path.join(args.file_load_path, "cache_datasets", f"{dataset_name}.pt"),
            weights_only=False  # ✅ 明确允许完整反序列化
        )
        dataset = build_dataset_from_samples(samples=sample_data, args=args)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            num_workers=4,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn
        )

        all_preds = []
        all_targets = []
        all_zero_paddings = []
        all_masks = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Testing on {dataset_name}"):
                batch = move_to_device(batch, args.device)
                # batch["center"]["spectrum"] = log1pf_normalize_tensor(batch["center"]["spectrum"],args)
                # batch["neighbors"]["spectrum"] = log1pf_normalize_tensor(batch["neighbors"]["spectrum"],args)
                with torch.amp.autocast('cuda'):
                    _, _, _, pred_dict, _, mask_labels, zero_paddings = model(batch, args=args)
                target_spec = batch["center"]["spectrum"]
                target_spec = target_spec[:, :args.seq_len] if args.seq_len < target_spec.shape[1] else target_spec


                pred = pred_dict["all_self"]
                # if torch.all(target_spec == 0):
                #     print(target_spec,target_spec)
                all_preds.append(pred.cpu())
                all_targets.append(target_spec.cpu())
                all_zero_paddings.append(zero_paddings.cpu())
                all_masks.append(mask_labels.cpu())
        dataset_time = time.time() - test_start_time
        print(f"\033[36mDataset {dataset_name} training completed in {dataset_time:.2f} seconds ({dataset_time/60:.2f} minutes).\033[0m")
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        all_zero_paddings = torch.cat(all_zero_paddings, dim=0)
        all_masks = torch.cat(all_masks, dim = 0)
        
        # all_preds[all_zero_paddings] = 0
        # 计算 MSE
        # mask = all_masks
        mask = all_zero_paddings
        # print('pred',all_preds[10])
        # print('target',all_targets[10])
        # print('mask',mask[10])
        print(all_preds[0][mask[0]],all_targets[0][mask[0]])
        # print(all_preds[0],all_targets[0])
        mse_value = mean_squared_error(
            all_targets[mask].flatten().cpu().numpy(),
            all_preds[mask].flatten().cpu().numpy()
        )

        # 计算 PCC
        mean_pcc = cal_pcc(all_preds, all_targets, mask)
        all_preds = all_preds.cpu().numpy()
        all_targets = all_targets.cpu().numpy()
        # 计算 SSIM（基于一维谱图）
        ssim_vals = []
        
        for j in range(all_targets.shape[1]):
            mask_j = mask[:, j]
            if mask_j.sum() == 0:
                continue  # 跳过全是 padding 的列
            target_j = all_targets[:, j][mask_j]
            pred_j = all_preds[:, j][mask_j]
            data_range = target_j.max() - target_j.min()
            try:
                ssim_val = ssim(target_j, pred_j, data_range=data_range)
                ssim_vals.append(ssim_val)
            except ValueError as e:
                print(f"Skipping due to small image size: {e}")
                continue
            ssim_vals.append(ssim_val)
        mean_ssim = np.mean(ssim_vals)
        results_dict['dataset'].append(dataset_name)
        results_dict['mse'].append(mse_value)
        results_dict['pcc'].append(mean_pcc)
        results_dict['ssim'].append(mean_ssim)
        print(f"\033[34mTest Results:\033[0m\n"
            f" - MSE: {mse_value:.6f}\n"
            f" - Mean PCC: {mean_pcc:.6f}\n"
            f" - Mean SSIM: {mean_ssim:.6f}")

    results_df = pd.DataFrame(results_dict)
    results_df.to_csv(os.path.join(args.log_dir, "test_metrics.csv"), index=False)

    # 绘制图像
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.bar(results_dict['dataset'], results_dict['mse'], color='tomato')
    plt.title("MSE per Dataset")
    plt.xticks(rotation=45)
    plt.ylabel("MSE")

    plt.subplot(1, 3, 2)
    plt.bar(results_dict['dataset'], results_dict['pcc'], color='green')
    plt.title("PCC per Dataset")
    plt.xticks(rotation=45)
    plt.ylabel("PCC")

    plt.subplot(1, 3, 3)
    plt.bar(results_dict['dataset'], results_dict['ssim'], color='royalblue')
    plt.title("SSIM per Dataset")
    plt.xticks(rotation=45)
    plt.ylabel("SSIM")

    plt.tight_layout()
    plt.savefig(os.path.join(args.log_dir, "test_metrics_plot.png"))
    plt.close()
    return {
        'mse': mse_value,
        'pcc': mean_pcc,
        'ssim': mean_ssim
    }

def test_autoencoder_model(args):
    # 加载模型
    file_load_path = args.file_load_path
    combined_files = os.path.join(file_load_path,"combined_table.pkl")
    combined_table = pd.read_pickle(combined_files)
    train_table = combined_table[["mol_formula"]][:].copy()
    args.mask_label_id = len(train_table)
    dataset_name = args.test_dataset_name
    checkpoint = torch.load(args.ckpt_path, map_location=args.device)
    model = autoencoder(args=args).to(args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    # print(model.state_dict())
    results_dict = {
    'dataset': [],
    'mse': [],
    'pcc': [],
    'ssim': []
    }
    # 加载数据
    dataset_list, args = data_load_main(args)

    if args.start_dataset:
        dataset_index = dataset_list.index(args.start_dataset)
        test_dataset_list = dataset_list[dataset_index:dataset_index+args.test_dataset_num]
    else:
        test_dataset_list = dataset_list[-args.test_dataset_num:]
    for dataset_index, dataset_name in enumerate(test_dataset_list,start=1):
        print(f"\033[32mTesting on dataset {dataset_index}/{len(test_dataset_list)}: {dataset_name}...\033[0m")
        writer_step = SummaryWriter(log_dir=f'{args.log_dir}/train_logs_steps/{dataset_name}')
        sample_data = torch.load(os.path.join(args.file_load_path, "cache_datasets", f"{dataset_name}.pt"))
        dataset = build_dataset_from_samples(samples=sample_data, args=args)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            num_workers=4,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn
        )

        all_preds = []
        all_targets = []
        all_zero_paddings = []
        all_masks = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Testing on {dataset_name}"):
                batch = move_to_device(batch, args.device)
                # batch["center"]["spectrum"] = log1pf_normalize_tensor(batch["center"]["spectrum"],args)
                # batch["neighbors"]["spectrum"] = log1pf_normalize_tensor(batch["neighbors"]["spectrum"],args)
                spec, padding_mask = model(batch, args=args)
                target_spec = batch["center"]["spectrum"]
                target_spec = target_spec[:, :args.seq_len] if args.seq_len < target_spec.shape[1] else target_spec


                pred = spec
                # if torch.all(target_spec == 0):
                #     print(target_spec,target_spec)
                all_preds.append(pred.cpu())
                all_targets.append(target_spec.cpu())
                all_zero_paddings.append(padding_mask.cpu())

        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        all_zero_paddings = torch.cat(all_zero_paddings, dim=0)
        
        # all_preds[all_zero_paddings] = 0
        # 计算 MSE
        # mask = all_masks
        mask = ~all_zero_paddings
        # print('pred',all_preds[10])
        # print('target',all_targets[10])
        # print('mask',mask[10])
        # print(all_preds[0][mask[0]],all_targets[0][mask[0]])
        print(all_preds[0],all_targets[0])
        mse_value = mean_squared_error(
            all_targets[mask].flatten().cpu().numpy(),
            all_preds[mask].flatten().cpu().numpy()
        )

        # 计算 PCC
        mean_pcc = cal_pcc(all_preds, all_targets, mask)
        all_preds = all_preds.cpu().numpy()
        all_targets = all_targets.cpu().numpy()
        # 计算 SSIM（基于一维谱图）
        ssim_vals = []
        
        for j in range(all_targets.shape[1]):
            mask_j = mask[:, j]
            if mask_j.sum() == 0:
                continue  # 跳过全是 padding 的列
            target_j = all_targets[:, j][mask_j]
            pred_j = all_preds[:, j][mask_j]
            data_range = target_j.max() - target_j.min()
            try:
                ssim_val = ssim(target_j, pred_j, data_range=data_range)
                ssim_vals.append(ssim_val)
            except ValueError as e:
                print(f"Skipping due to small image size: {e}")
                continue
            ssim_vals.append(ssim_val)
        mean_ssim = np.mean(ssim_vals)
        results_dict['dataset'].append(dataset_name)
        results_dict['mse'].append(mse_value)
        results_dict['pcc'].append(mean_pcc)
        results_dict['ssim'].append(mean_ssim)
        print(f"\033[34mTest Results:\033[0m\n"
            f" - MSE: {mse_value:.6f}\n"
            f" - Mean PCC: {mean_pcc:.6f}\n"
            f" - Mean SSIM: {mean_ssim:.6f}")

    results_df = pd.DataFrame(results_dict)
    results_df.to_csv(os.path.join(args.log_dir, "test_metrics.csv"), index=False)

    # 绘制图像
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.bar(results_dict['dataset'], results_dict['mse'], color='tomato')
    plt.title("MSE per Dataset")
    plt.xticks(rotation=45)
    plt.ylabel("MSE")

    plt.subplot(1, 3, 2)
    plt.bar(results_dict['dataset'], results_dict['pcc'], color='green')
    plt.title("PCC per Dataset")
    plt.xticks(rotation=45)
    plt.ylabel("PCC")

    plt.subplot(1, 3, 3)
    plt.bar(results_dict['dataset'], results_dict['ssim'], color='royalblue')
    plt.title("SSIM per Dataset")
    plt.xticks(rotation=45)
    plt.ylabel("SSIM")

    plt.tight_layout()
    plt.savefig(os.path.join(args.log_dir, "test_metrics_plot.png"))
    plt.close()
    return {
        'mse': mse_value,
        'pcc': mean_pcc,
        'ssim': mean_ssim
    }