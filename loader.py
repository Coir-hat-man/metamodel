import numpy as np
import os
import json
import torch
import copy
import random
import matplotlib.pyplot as plt
import m2aia as m2
from torchvision import transforms
import pandas as pd
import numpy as np
from scipy import signal
import math
import re
import gc
from sklearn.neighbors import NearestNeighbors  
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import numpy as np
from typing import List
from unimol_tools import UniMolRepr
from torch.utils.data import Dataset
from scipy.stats import normaltest
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)

def prefer_hmdb(df):
    def pick_row(group):
        hmdb_rows = group[group["moleculeIds"].astype(str).str.startswith("\"HMDB")]
        return hmdb_rows.iloc[0] if not hmdb_rows.empty else group.iloc[0]
    return df.groupby("mol_formula", group_keys=False).apply(pick_row).reset_index(drop=True)

def test_get_matrix(folder_path):
    df = pd.read_csv(folder_path, skiprows=2)
    df = prefer_hmdb(df)
    df = df.drop_duplicates(subset="mz")
    # 提取所有以 "x" 开头且格式为 "x数字_y数字" 的列名（即像素强度列）
    pixel_columns = [col for col in df.columns if col.startswith("x") and "_y" in col]
    # pixel_columns = pixel_columns[1:]
    # 处理moleculeIds列：
    # 1. 分割逗号取第一个值
    # 2. 去除双引号
    df["moleculeIds"] = (
        df["moleculeIds"].astype(str)        # 确保类型为字符串
        .str.replace('"', '')                # 去除所有双引号
    )

    keep_columns = ["moleculeIds"] + pixel_columns
    # 构建矩阵并填充NaN
    matrix = (
        df.set_index("mol_formula")  # 设置行索引
        [keep_columns]              # 选择像素列
        .fillna(0)                   # NaN填充为0
        .astype({col: float for col in pixel_columns})             # 确保数据类型为浮点数
        .reset_index()
    )

    # 重命名列名，明确区分 moleculeIds 和像素列
    pixel_columns = [col for col in pixel_columns if matrix[col].sum() != 0]
    matrix = matrix[["mol_formula", "moleculeIds"] + pixel_columns]
    matrix.columns.name = None  # 移除潜在的列名层级
    centroids = df["mz"].unique()
    matrix[pixel_columns] = np.where(matrix[pixel_columns] > 100000, 100000, matrix[pixel_columns])

    all_pixel_values = matrix[pixel_columns].values[matrix[pixel_columns].values != 0].flatten()
    # x = matrix[pixel_columns].values.flatten()
    # global_mean = x.mean()
    # global_std = x.std() + 1e-6

    # matrix[pixel_columns] = (matrix[pixel_columns] - global_mean) / global_std
    # 计算最大值，用于构建区间边界
    max_val = all_pixel_values.max()

    # 构造以 10000 为步长的 bin 边界
    bin_edges = np.arange(0, max_val, 1000)

    # 统计每个区间的数量
    counts, edges = np.histogram(all_pixel_values, bins=bin_edges)

    # 打印结果
    for i in range(len(counts)):
        print(f"区间 {edges[i]:.0f} - {edges[i+1]:.0f}: 数量 = {counts[i]}")
    # 排除全0或全常数情况（否则无法做正态检验）
    if np.all(all_pixel_values == all_pixel_values[0]):
        print("矩阵中所有像素值相同，无法判断正态性。")
    else:
        stat, p = normaltest(all_pixel_values)
        if p > 0.05:
            print(f"✅ 矩阵整体像素值符合正态分布（p = {p:.4f}）")
        else:
            print(f"❌ 矩阵整体像素值不符合正态分布（p = {p:.4f}）")
    # 从 matrix 中取出像素值子矩阵
    X = matrix[pixel_columns].values  # shape: (n_samples, n_pixels)

    #     # 提取所有像素值
    pixel_values = matrix[pixel_columns].values[matrix[pixel_columns].values != 0].flatten()
    # 绘制所有像素值的直方图
    plt.figure(figsize=(10, 6))  # 设置图形大小
    sns.histplot(pixel_values, kde=False, bins=50)  # kde=True 添加核密度估计曲线
    plt.title("Histogram of All Pixel Values")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.savefig("data_hist")
    # 最大值
    max_val = pixel_values.max()

    # 非零最小值
    nonzero_min_val = pixel_values[pixel_values > 0].min()

    print("最大值:", max_val)
    print("非零最小值:", nonzero_min_val)

    # 每个样本的总强度（即行和）
    total_counts = X.sum(axis=1, keepdims=True)

    # 中位数 count depth
    median_count = np.median(total_counts[total_counts != 0])

    # 避免除以 0
    total_counts[total_counts == 0] = 1.0

    # 中位数归一化
    X_scaled = X / total_counts * median_count
    X_scaled = np.log1p(X_scaled)
    # 更新 matrix 中的像素列
    matrix[pixel_columns] = X_scaled
            # 提取所有像素值

    # matrix[pixel_columns] = (matrix[pixel_columns] - matrix[pixel_columns].mean()) / matrix[pixel_columns].std()

    pixel_values = matrix[pixel_columns].values[matrix[pixel_columns].values != 0].flatten()
    # 绘制所有像素值的直方图
    plt.figure(figsize=(10, 6))  # 设置图形大小
    sns.histplot(pixel_values, kde=False, bins=50)  # kde=True 添加核密度估计曲线
    plt.title("Histogram of All Pixel Values")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.savefig("norm_data_hist")
    # 最大值
    max_val = pixel_values.max()

    # 非零最小值
    nonzero_min_val = pixel_values[pixel_values > 0].min()

    print("归一化后最大值:", max_val)
    print("归一化后非零最小值:", nonzero_min_val)
        # 取像素值矩阵并展开为一维数组
    all_pixel_values = pixel_values

    # 排除全0或全常数情况（否则无法做正态检验）
    if np.all(all_pixel_values == all_pixel_values[0]):
        print("矩阵中所有像素值相同，无法判断正态性。")
    else:
        stat, p = normaltest(all_pixel_values)
        if p > 0.05:
            print(f"✅ 矩阵整体像素值符合正态分布（p = {p:.4f}）")
        else:
            print(f"❌ 矩阵整体像素值不符合正态分布（p = {p:.4f}）")
    return matrix, centroids

def get_matrix(folder_path):
    df = pd.read_csv(folder_path, skiprows=2)
    df = prefer_hmdb(df)
    df = df.drop_duplicates(subset="mz")
    # 提取所有以 "x" 开头且格式为 "x数字_y数字" 的列名（即像素强度列）
    pixel_columns = [col for col in df.columns if col.startswith("x") and "_y" in col]
    # pixel_columns = pixel_columns[1:]
    # 处理moleculeIds列：
    # 1. 分割逗号取第一个值
    # 2. 去除双引号
    df["moleculeIds"] = (
        df["moleculeIds"].astype(str)        # 确保类型为字符串
        .str.replace('"', '')                # 去除所有双引号
    )

    keep_columns = ["moleculeIds"] + pixel_columns
    # 构建矩阵并填充NaN
    matrix = (
        df.set_index("mol_formula")  # 设置行索引
        [keep_columns]              # 选择像素列
        .fillna(0)                   # NaN填充为0
        .astype({col: float for col in pixel_columns})             # 确保数据类型为浮点数
        .reset_index()
    )

    # 重命名列名，明确区分 moleculeIds 和像素列
    pixel_columns = [col for col in pixel_columns if matrix[col].sum() != 0]
    matrix = matrix[["mol_formula", "moleculeIds"] + pixel_columns]
    matrix.columns.name = None  # 移除潜在的列名层级
    centroids = df["mz"].unique()
    matrix[pixel_columns] = np.where(matrix[pixel_columns] > 100000, 100000, matrix[pixel_columns])

 
    X = matrix[pixel_columns].values  # shape: (n_samples, n_pixels)

    # 每个样本的总强度（即行和）
    total_counts = X.sum(axis=1, keepdims=True)

    # 中位数 count depth
    median_count = np.median(total_counts[total_counts != 0])

    # 避免除以 0
    total_counts[total_counts == 0] = 1.0

    # 中位数归一化
    X_scaled = X / total_counts * median_count
    X_scaled = np.log1p(X_scaled)
    # 更新 matrix 中的像素列
    matrix[pixel_columns] = X_scaled
            # 提取所有像素值

    # matrix[pixel_columns] = (matrix[pixel_columns] - matrix[pixel_columns].mean()) / matrix[pixel_columns].std()

    return matrix, centroids

def get_mol(folder_path):
    df = pd.read_csv(folder_path, skiprows=2)
    df = prefer_hmdb(df)
    df = df.drop_duplicates(subset="mz")
    # 提取所有以 "x" 开头且格式为 "x数字_y数字" 的列名（即像素强度列）
    # pixel_columns = [col for col in df.columns if col.startswith("x") and "_y" in col]
    # pixel_columns = pixel_columns[1:]
    # 处理moleculeIds列：
    # 1. 分割逗号取第一个值
    # 2. 去除双引号
    df["moleculeIds"] = (
        df["moleculeIds"].astype(str)        # 确保类型为字符串
        .str.replace('"', '')                # 去除所有双引号
    )

    keep_columns = ["moleculeIds"] 
    # 构建矩阵并填充NaN
    matrix = (
        df.set_index("mol_formula")  # 设置行索引
        [keep_columns]              # 选择像素列
        .fillna(0)                   # NaN填充为0
        .reset_index()
    )

    # 重命名列名，明确区分 moleculeIds 和像素列
    matrix = matrix[["mol_formula", "moleculeIds"]]
    matrix.columns.name = None  # 移除潜在的列名层级
    centroids = df["mz"].unique()
    # matrix[pixel_columns] = np.where(matrix[pixel_columns] > 100000, 100000, matrix[pixel_columns])

 
    # X = matrix[pixel_columns].values  # shape: (n_samples, n_pixels)

    # # 每个样本的总强度（即行和）
    # total_counts = X.sum(axis=1, keepdims=True)

    # # 中位数 count depth
    # median_count = np.median(total_counts[total_counts != 0])

    # # 避免除以 0
    # total_counts[total_counts == 0] = 1.0

    # # 中位数归一化
    # X_scaled = X / total_counts * median_count
    # X_scaled = np.log1p(X_scaled)
    # # 更新 matrix 中的像素列
    # matrix[pixel_columns] = X_scaled
            # 提取所有像素值

    # matrix[pixel_columns] = (matrix[pixel_columns] - matrix[pixel_columns].mean()) / matrix[pixel_columns].std()

    return matrix, centroids

def get_file_content_or_default(molecule_ids, repr_dict, device='cuda'):
    default_tensor = torch.zeros(768, device=device)
    default_tensor_2d = torch.zeros(1, 768, device=device)

    unique_repr = []
    for item in molecule_ids:
        if item is None or item == -1:
            unique_repr.append(default_tensor)
            continue

        item_repr = []
        for molecule_id in item.split(","):
            molecule_id = molecule_id.strip()
            value = repr_dict.get(molecule_id)
            if value is not None:
                item_repr.append(value)
            else:
                item_repr.append(default_tensor_2d)
            break

        stacked = torch.sum(torch.cat(item_repr, dim=0), dim=0)
        unique_repr.append(stacked)

    return torch.stack(unique_repr).to("cpu")

import os
from tqdm import tqdm

def get_smiles_repr(folder_path):
    """
    遍历 folder_path/smiles/ 下的所有子文件夹（dic），提取其中所有 .smiles 文件的内容，
    并将文件名（去除后缀）作为 key，SMILES 字符串作为 value 存入字典中。
    
    参数：
        folder_path (str): 根目录路径，内部应包含 'smiles/' 文件夹。
    
    返回：
        dict: 键为 .smiles 文件名（无扩展名），值为对应的 SMILES 字符串。
        For example:
        smiles_dict = {
            "HMDB0000036": "[H][C@@]1(CC[C@@]2([H])[C@]3([H])[C@H](O)C[C@]4([H])C[C@H](O)CC[C@]4(C)[C@@]3([H])C[C@H](O)[C@]12C)[C@H](C)CCC(=O)NCCS(O)(=O)=O"
            "HMDB0000653": "[H][C@@]12CC[C@H]([C@H](C)CCCC(C)C)[C@@]1(C)CC[C@@]1([H])[C@@]2([H])CC=C2C[C@H](CC[C@]12C)OS(O)(=O)=O"
            "HMDB0000874": "[H][C@@]12CC[C@H]([C@H](C)CCC(=O)NCCS(O)(=O)=O)[C@@]1(C)CC[C@@]1([H])[C@@]2([H])[C@@H](O)CC2C[C@H](O)CC[C@]12C"
            ...
        }
    """
    smiles_dict = {}
    smiles_folder = os.path.join(folder_path, "smiles")

    for filename in os.listdir(smiles_folder):
        if filename.endswith(".smi"):  # 只处理以 .smiles 结尾的文件
            file_path = os.path.join(smiles_folder, filename)
            with open(file_path, 'r') as file:
                key = filename.split('.')[0]  # 去掉扩展名作为 key
                smiles_dict[key] = file.read().strip()  # 读取并去除前后空白字符

    return smiles_dict


def process_smiles_with_unimol(smiles_dict, file_load_path):
    result_file_path = os.path.join(file_load_path, "unimol_results.pt")

    # 如果缓存存在，直接加载
    if os.path.exists(result_file_path):
        print(f"Loading repr from {result_file_path} ......")
        return torch.load(result_file_path)

    # 自动检测是否使用 GPU
    use_gpu = torch.cuda.is_available()

    # UniMol 参数
    params = {
        'data_type': 'molecule',
        'remove_hs': False,   # 是否移除氢原子
        'model_name': 'unimolv2',
        'model_size': '84m',  # 模型大小
    }

    # 初始化 UniMolRepr 模型
    clf = UniMolRepr(use_gpu=use_gpu, **params)

    # 结果字典
    unimol_results = {}

    # 遍历 smiles_dict
    for filename, smiles in tqdm(smiles_dict.items(), desc="Processing SMILES with UniMol"):
        try:
            unimol_repr = clf.get_repr(
                data=[smiles],
                return_atomic_reprs=False
            )
            unimol_results[filename] = torch.tensor(unimol_repr["cls_repr"])
        except Exception as e:
            print(f"Processing {filename} error: {e}")
            unimol_results[filename] = None
        # 保存处理结果

    torch.save(unimol_results, result_file_path)
    print(f"处理完成，结果已保存至 {result_file_path}")

    return unimol_results

def collate_fn(batch):
    # batch is a list of dicts: [{'center': ..., 'neighbors': [...]}, ...]
    center_coords, center_mol_ids, center_spectra, center_patches, center_mols, center_pixels = [], [], [], [], [], []
    neighbor_coords, neighbor_mol_ids, neighbor_spectra, neighbor_patches, neighbor_mols = [], [], [], [], []

    for sample in batch:
        c_coord, c_id, c_spec, c_patch, c_mol, c_pixel = sample['center']
        center_pixels.append(torch.tensor(c_pixel))
        center_coords.append(c_coord)
        center_mol_ids.append(torch.tensor(c_id))
        center_spectra.append(torch.tensor(c_spec) if not torch.is_tensor(c_spec) else c_spec)
        center_patches.append(c_patch)
        center_mols.append(c_mol)

        n_coords, n_ids, n_specs, n_patches, n_mols = [], [], [], [], []
        for neighbor in sample['neighbors']:
            n_coord, n_id, n_spec, n_patch, n_mol, _ = neighbor
            n_coords.append(n_coord)
            n_ids.append(torch.tensor(n_id))
            n_specs.append(torch.tensor(n_spec) if not torch.is_tensor(n_spec) else n_spec)
            n_patches.append(n_patch)
            n_mols.append(n_mol)
        
        neighbor_coords.append(torch.stack(n_coords))
        neighbor_mol_ids.append(torch.stack(n_ids))
        neighbor_spectra.append(torch.stack(n_specs))

        # neighbor_patches.append(torch.stack(n_patches))
        neighbor_mols.append(torch.stack(n_mols))
    return {
        "center": {
            "coord": torch.stack(center_coords),           # [B, 2]
            "ids": torch.stack(center_mol_ids),            # [B, N]
            "spectrum": torch.stack(center_spectra),       # [B, N]
            # "patch": torch.stack(center_patches),          # [B, N]
            "patch": None,
            "mol_repr": torch.stack(center_mols),          # [B, N, M]
            "real_pixel": torch.stack(center_pixels)
        },
        "neighbors": {
            "coord": torch.stack(neighbor_coords),         # [B, k, 2]
            "ids": torch.stack(neighbor_mol_ids),          # [B, N]
            "spectrum": torch.stack(neighbor_spectra),     # [B, k, N]
            # "patch": torch.stack(neighbor_patches),        # [B, k, N]
            "patch": None,
            "mol_repr": torch.stack(neighbor_mols)         # [B, k, N, M]
        }
    }


class PixelWithNeighborsDataset(Dataset):
    def __init__(self, image = None, matrix = None, vocab = None, centroids = None, unimol_repr = None, samples = None,  args = None):
        if samples != None:
            self.data_list = samples
            coordinate_list = [sample[0] for sample in samples]
            
        else:
            self.image = image
            self.centroids = centroids
            self.args = args
            self.matrix = matrix
            self.unimol_repr = unimol_repr
            self.vocab = vocab
            self.seq_len = args.seq_len
            
            for key in self.unimol_repr:
                    rep = self.unimol_repr[key]
                    if rep is not None and isinstance(rep, torch.Tensor):
                        self.unimol_repr[key] = rep.to(args.device)
            self.pixel_size = self.image.GetSpacing()
            # self.img_dataset = self._prepare_img_dataset()
            self.data_list, coordinate_list = self._prepare_samples()
        
        self.coordinates = torch.stack(coordinate_list)
        self.k = args.k_neighbors + 1  # include self
        self.nn = NearestNeighbors(n_neighbors=self.k, metric='euclidean')
        self.nn.fit(self.coordinates.numpy())
        self.indices = self.nn.kneighbors(self.coordinates.numpy(), return_distance=False)

    def _prepare_img_dataset(self):
        transform_rm_outliers_center_crop_resize = transforms.Compose([
            transforms.Lambda(lambda x: torch.Tensor(np.clip(x / np.quantile(x, 0.999), 0, 1))),
        ])

        return m2.IonImageDataset(
            [self.image],
            centroids=self.centroids,
            tolerance=75,
            tolerance_type='ppm',
            buffer_type='memory',
            transforms=transform_rm_outliers_center_crop_resize
        )

        

    def _prepare_samples(self):
        samples = []
        coordinate_list = []


        # 用 get 方式替代 map，如果找不到 key，就给 padding_id
        mol_formula_ids = self.matrix["mol_formula"].apply(
            lambda x: self.vocab.get(x, self.args.pad_token_id)
        )
        mol_formula_ids = mol_formula_ids.tolist() if isinstance(mol_formula_ids, pd.Series) else mol_formula_ids
        mol_formula_ids = torch.tensor(mol_formula_ids, dtype=torch.long)
        print("Start sampling")
        # 2. 预提取图像维度
        height, width = self.image.GetShape()[1], self.image.GetShape()[0]


            # 4. moleculeIds 列缓存
        molecule_ids_series = self.matrix["moleculeIds"]

            # 5. 所有有效的列数据缓存为 dict
        column_data = {
            col: self.matrix[col].values
            for col in self.matrix.columns
            if col.startswith("x") and "_y" in col
        }
        
        # img_tensor = torch.stack([img for img in self.img_dataset])

        for y in tqdm(range(height), desc="Sampling along Y axis"):
            for x in range(width):
                column_name = f"x{x}_y{y}"
                if column_name in column_data:
                    result = column_data[column_name]
                    if result.sum() != 0:
                        # print(column_name)
                        # print(result)
                        current_coord = torch.tensor([x, y], dtype=torch.float32)
                        coordinate_list.append(current_coord)

                        pixel_intensity = torch.tensor(result, dtype=torch.float32)

                        # # 提取每个通道的像素值
                        # pixel_channel = img_tensor[:, :, y, x]
                        pixel_channel = None

                        # 获取 molecule id 并生成 unique_repr
                        molecule_ids = molecule_ids_series.where(result != 0, None).tolist()
                        unique_repr = get_file_content_or_default(molecule_ids, self.unimol_repr)

                        samples.append((
                            current_coord,
                            mol_formula_ids,
                            pixel_intensity,
                            pixel_channel,
                            unique_repr,
                            self.pixel_size
                        ))
        return samples, coordinate_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        neighbor_idxs = self.indices[idx][1:]  # exclude self
        center_data = self.data_list[idx]  # [coord, spectrum, patch, mol_repr]
        neighbor_data = [self.data_list[i] for i in neighbor_idxs]
        return {
            "center": center_data,
            "neighbors": neighbor_data  # list of [coord, spectrum, patch, mol_repr]
        }

def build_dataset_from_samples(samples,args):
    dataset = PixelWithNeighborsDataset(samples = samples, args = args)
    return dataset


from multiprocessing import cpu_count    # 查看cpu核心数
from multiprocessing import Pool         # 并行处理必备，进程池 
import pickle

from concurrent.futures import ProcessPoolExecutor, as_completed
def process_one_dataset(dir, file_load_path, mol_formula_inv_vocab, args):
    print(f"[{os.getpid()}] Start processing subset with {len(dir)} folders")
    smiles_dict = {}
    dataset_sizes_local = {}

    if not os.path.exists(os.path.join(file_load_path, "unimol_results.pt")):
        smiles_dict = get_smiles_repr(file_load_path)
    unimol_repr = process_smiles_with_unimol(smiles_dict, file_load_path)

    for f in dir:
        folder_path = os.path.join(file_load_path, f)
        cache_file = os.path.join(file_load_path, "cache_datasets", f"{f}.pt")

        img, matrix, centroids = None, None, None
        try:
            for file in os.listdir(folder_path):
                if file.endswith(".imzML"):
                    img = m2.ImzMLReader(os.path.join(folder_path, file))
                elif file.endswith(".csv"):
                    try:
                        matrix, centroids = get_matrix(os.path.join(folder_path, file))
                    except Exception as e:
                        print(f"[get_matrix ERROR] {folder_path}: {e}")
                        continue

            if img is not None and matrix is not None:
                dataset = PixelWithNeighborsDataset(
                    image=img, matrix=matrix, vocab=mol_formula_inv_vocab,
                    centroids=centroids, unimol_repr=unimol_repr, args=args
                )
                torch.save(dataset.data_list, cache_file)
                print(f"[{f}] ✔ Saved to {cache_file}")
                dataset_sizes_local[f] = len(dataset.data_list)
                del dataset
                gc.collect()
            else:
                print(f"[{f}] ✘ Missing imzML or CSV.")

        except Exception as e:
            print(f"[{f}] ✘ Error: {e}")
    return dataset_sizes_local

def process_one_dataset_wrapper(args):
    return process_one_dataset(*args)
    
def create_filled_matrix(matrix, train_table, padding_label):
    # 生成全为-1的表格（排除mol_formula列
    filled_table = pd.DataFrame(
        -1, 
        index=train_table.index,
        columns=[col for col in matrix.columns if col != "mol_formula"]
    )

    # 插入分子式列到第一列
    filled_table.insert(0, "mol_formula", train_table["mol_formula"])

    # 构建一个字典，快速查找 mol_formula 对应的行数据
    matrix_dict = matrix.set_index("mol_formula").to_dict(orient="index")
    # 逐行填充
    # 优化填充逻辑：整行填充
    for idx, row in tqdm(train_table.iterrows(), total=len(train_table), desc="Filling table"):
        formula = row["mol_formula"]
        if formula in matrix_dict:
            filled_table.loc[idx, filled_table.columns != "mol_formula"] = pd.Series(matrix_dict[formula])
            
    return filled_table

def data_load_main(args):
    print("Start loading......")
    timestamp_pattern = re.compile(r"\d{4}-\d{2}-\d{2}_\d{2}h\d{2}m\d{2}s")
    file_load_path = args.file_load_path
    # resume_data = '2018-04-06_00h49m06s'

    # 创建缓存目录
    cache_dir = os.path.join(file_load_path, "cache_datasets")
    os.makedirs(cache_dir, exist_ok=True)

    # 如果 cache 目录中已有 .pt 文件，直接加载
    cached_files = [f for f in os.listdir(cache_dir) if f.endswith(".pt")]
    combined_files = os.path.join(file_load_path,"combined_table.pkl")

    if cached_files:
        print("Found cached datasets. Loading all from cache...")
        combined_table = pd.read_pickle(combined_files)
        train_table = combined_table[["mol_formula"]][:].copy()
        args.mask_label_id = len(train_table)

    else: # 否则开始构建数据集
        print("No cached datasets found. Building from scratch...")
        construct_combined_file = False
        if not os.path.exists(combined_files):
            combined_table = pd.DataFrame(columns=["mol_formula"])  # 初始化空表

            # ---------- Step 1: 遍历 CSV 提取 mol_formula ----------
            print("Scanning CSV files to build combined_table...")
            mol_formula_list = []
            for f in os.listdir(file_load_path):
                folder_path = os.path.join(file_load_path, f)
                if os.path.isdir(folder_path) and timestamp_pattern.fullmatch(f):
                    for file in os.listdir(folder_path):
                        if file.endswith(".csv"):
                            try:
                                matrix, _ = get_mol(os.path.join(folder_path, file))
                                print(file)
                                mol_formula_df = matrix[["mol_formula"]]
                                mol_formula_list.append(mol_formula_df)
                            except Exception as e:
                                print(f"[CSV ERROR] {folder_path}: {e}")
                                continue
            if mol_formula_list:
                combined_table = pd.concat(mol_formula_list, ignore_index=True)
                combined_table = (
                    combined_table
                    .assign(count=combined_table.groupby("mol_formula")["mol_formula"].transform("count"))
                    .sort_values("count", ascending=False)
                    .drop_duplicates("mol_formula")
                    .reset_index(drop=True)
                )
                combined_table.to_pickle(combined_files)
                print("✔ combined_table.pkl saved.")
            else:
                raise RuntimeError("No CSV data found. Cannot build mol_formula vocab.")
        else :
            combined_table = pd.read_pickle(combined_files)
            # ---------- Step 2: 构建 train_table 和 vocab ----------
        train_table = combined_table[["mol_formula"]].copy()
        args.mask_label_id = len(train_table)
        train_table = pd.concat([train_table, pd.DataFrame([{"mol_formula": "padding"}])], ignore_index=True)
        train_table["mol_formula_id"], mol_formula_uniques = pd.factorize(train_table["mol_formula"])
        args.mask_label_id = len(train_table)

        mol_formula_vocab = dict(enumerate(mol_formula_uniques))
        mol_formula_inv_vocab = {v: k for k, v in mol_formula_vocab.items()}

        # ---------- Step 3: 并发处理每个数据集 ----------
        timestamp_dirs = sorted(
            [f for f in os.listdir(file_load_path)
            if timestamp_pattern.fullmatch(f)]
        )
        num_dirs = len(timestamp_dirs)
        print(f"Found {len(timestamp_dirs)} dataset folders.")
        num_cores = 8

        subset1 = timestamp_dirs[:num_dirs // 8]
        subset2 = timestamp_dirs[num_dirs // 8: num_dirs // 4]
        subset3 = timestamp_dirs[num_dirs // 4: (num_dirs * 3) // 8]
        subset4 = timestamp_dirs[(num_dirs * 3) // 8: num_dirs // 2]
        subset5 = timestamp_dirs[num_dirs // 2: (num_dirs * 5) // 8]
        subset6 = timestamp_dirs[(num_dirs * 5) // 8: (num_dirs * 6) // 8]
        subset7 = timestamp_dirs[(num_dirs * 6) // 8: (num_dirs * 7) // 8]
        subset8 = timestamp_dirs[(num_dirs * 7) // 8: ]
        List_subsets = [subset1,subset2,subset3,subset4,
                    subset5,subset6,subset7,subset8]
        dataset_sizes = {}

        with ProcessPoolExecutor(max_workers=8) as executor:
            futures = []
            for i in range(8):
                args_tuple = (List_subsets[i], file_load_path, mol_formula_inv_vocab, args)
                futures.append(executor.submit(process_one_dataset, *args_tuple))

            for future in futures:
                try:
                    result = future.result()  # result is a dict
                    dataset_sizes.update(result)
                except Exception as e:
                    print(f"❌ Error in process: {e}")

        with open("dataset_sizes.json", "w") as f:
            json.dump(dataset_sizes, f)
        print("✅ dataset_sizes saved.")
        # p = Pool(num_cores)
        # for i in range(num_cores):
        #     p.apply_async(process_one_dataset, args = (List_subsets[i],file_load_path, unimol_repr,mol_formula_inv_vocab, args))
        # p.close()
        # p.join()
        print("✔ All datasets processed.")
        # 构建完成后再统一加载
        print("Now loading all saved datasets...")
    # datasets = []
    dataset_name_list = []
    for file in sorted(os.listdir(cache_dir)):
        if file.endswith(".pt"):
            dataset_name_list.append(file.split(".")[0])
            # samples = torch.load(os.path.join(cache_dir, file))
            # dataset = build_dataset_from_samples(samples = samples, args = args)
            # datasets.append(dataset)
    print(f"Total datasets loaded: {len(dataset_name_list)}")
    # dataloaders= []
    # for dataset in datasets:
    #     dataloader = torch.utils.data.DataLoader(
    #         dataset,
    #         num_workers=4,
    #         batch_size=args.batch_size,
    #         shuffle=args.shuffle,
    #         drop_last=True,
    #         collate_fn=collate_fn
    #     )
    #     dataloaders.append(dataloader)

    return dataset_name_list, args