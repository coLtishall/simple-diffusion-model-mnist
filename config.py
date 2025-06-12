import torch
import os
from pathlib import Path

class Config:
    # --- 基本设置 ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_path = "mnist_data"
    output_dir = "results"
    
    # --- 模型与图像参数 ---
    image_size = 28
    channels = 1
    num_classes = 10
    time_emb_dim = 32
    num_groups = 8  # GroupNorm中的组数

    # --- 训练参数 ---
    batch_size = 128
    epochs = 30
    lr = 1e-3
    cfg_dropout_rate = 0.1 # CFG训练时丢弃标签的概率
    
    # --- 扩散过程参数 ---
    timesteps = 500
    cosine_schedule_s = 0.008 # Cosine噪声策略的s参数
    guidance_scale = 7.5      # CFG引导强度(默认值)
    
    # --- 保存与采样预览 ---
    save_every_epoch = 5
    preview_num_images = 16   # 训练时预览生成的图片数量
    preview_target_digit = 4    # 训练时预览生成的目标数字

def setup_directories():
    """创建项目所需的输出目录"""
    Path(Config.output_dir).mkdir(exist_ok=True)
    Path(os.path.join(Config.output_dir, 'images')).mkdir(exist_ok=True)
    Path(os.path.join(Config.output_dir, 'models')).mkdir(exist_ok=True)