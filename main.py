from argparse import Namespace
import argparse

import random
import os
import setproctitle
import torch
import numba
numba.config.DISABLE_JIT = True  # 仅调试时可用
import logging
logging.getLogger("numba").setLevel(logging.WARNING)
from loader import *
from utils import *
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from contrastive_loss import calc_contrastive_loss
from train import build_trained_model,test_model,build_autoencoder_model,test_autoencoder_model

# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
torch.set_num_threads(8) 

def setup_init(seed):
    random.seed(seed)
    os.environ['PYtorchONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_args():
    parser = argparse.ArgumentParser(description="Metabolic Model Training Arguments")

    # Mode and paths
    parser.add_argument('--mode', choices=['training', 'testing','testing_one'], default='training')
    parser.add_argument('--file_load_path', type=str, default='./data')
    parser.add_argument('--save_dir', type=str, default='./model_save')
    parser.add_argument('--log_dir', type=str, default='./logs/model_save')
    parser.add_argument('--resume_ckpt_path', type=str, default=None)
    parser.add_argument('--start_dataset', type=str, default=None)
    parser.add_argument('--training_dataset', type=str, default=None)
    parser.add_argument('--training_num', type=int, default=None)
    parser.add_argument('--autoencoder', type=str, default=False)

    
    # Test settings
    parser.add_argument('--test_dataset_name', type=str, default=None)
    parser.add_argument('--ckpt_path', type=str, default='./model_save/checkpoint50000.pth')
    parser.add_argument('--test_dataset_num', type=int, default=8)

    # Model settings
    parser.add_argument('--model_size', type=str, default='base')
    parser.add_argument('--use_autoencoder', type=bool, default=False)
    parser.add_argument('--in_chans', type=int, default=1)
    parser.add_argument('--mask_ratio', type=float, default=0.2)
    parser.add_argument('--patch_size', type=int, default=1)
    parser.add_argument('--qkv_bias', type=int, default=1)
    parser.add_argument('--seq_len', type=int, default=100)
    parser.add_argument('--pad_token_id', type=int, default=-1)
    parser.add_argument('--decoder', type=str, default='naive_transformer')
    parser.add_argument('--use_flash', type=bool, default=True)

    # Pretrain settings
    parser.add_argument('--random', type=bool, default=True)  # True if present
    parser.add_argument('--mask_strategy', type=str, default='random')
    parser.add_argument('--normalize', type=bool, default=True)
    parser.add_argument('--T', type=float, default=0.01)
    parser.add_argument('--margin', type=float, default=0.5)

    # Training parameters
    
    parser.add_argument('--num_datasets', type=int, default=None)
    parser.add_argument('--value_normalization', choices=['ln1p', 'lg1p'], default='ln1p')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_warmup_steps', type=float, default=0.01)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--early_stop', type=int, default=5)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--log_interval', type=int, default=20)
    parser.add_argument('--total_epoches', type=int, default=2)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--clip_grad', type=float, default=0.05)
    parser.add_argument('--lr_anneal_steps', type=int, default=200)
    parser.add_argument('--recon_loss_weight', type=float, default=1.0)
    parser.add_argument('--pred_loss_weight', type=float, default=1.0)
    parser.add_argument('--cl_loss_weight', type=float, default=1.0)
    parser.add_argument('--k_neighbors', type=int, default=9)
    parser.add_argument('--use_cls_token', type=bool, default=True)
    parser.add_argument('--use_batch_labels', type=bool, default=True)
    parser.add_argument('--use_batch_aware', type=bool, default=True)
    parser.add_argument('--use_image', type=bool, default=False)
    parser.add_argument('--image_combine_mz', choices=['cat_seq', 'cat_dim', 'add'], default='cat_dim')
    parser.add_argument('--cell_emb_style', type=str, default='cls')
    parser.add_argument('--use_ema', type=str, default='False')
    parser.add_argument('--use_no_mask_loss', type=str, default='False')
    parser.add_argument('--model_style', type=str, default='mlp')
    

    return parser.parse_args()

def main():

    args = get_args()
    # model =Metablisum_model(args=args)
    if args.mode == 'training':
        if args.autoencoder:
            return build_autoencoder_model(
                    model_dir = args.save_dir,
                    use_pretrain=False,
                    args=args)
        return build_trained_model(
                    model_dir = args.save_dir,
                    use_pretrain=False,
                    args=args)
    if args.mode == 'testing':
        if args.autoencoder:
            return test_autoencoder_model(args)
        return test_model(args)
    if args.mode == 'testing_one':
        return test_one(args)
if __name__ == "__main__":
    main()