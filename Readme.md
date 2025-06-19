# 模型说明

- 模型结构图位于：`./model.png`  
- 当前重点关注 **模型结构图下半部分的矩阵重构效果**
- 运行训练代码命令：
  ```bash
  CUDA_VISIBLE_DEVICES=0 python main.py
- 主要模型定义位于：./model/whole_pipline.py 中的 MetablisumModel 类
使用python3.10
使用的torch ： pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
安装Flashattn：pip install flash_attn-2.7.3+cu11torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
需要安装timm,seaborn,tensorboard,pandas,scikit-image,matplotlib,scikit-learn,tqdm,numba,setproctitle

