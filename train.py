import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm
import os
import argparse

# 从我们拆分的文件中导入
from config import Config, setup_directories
from model import SimpleUnet
from diffusion import Diffusion

def train(cfg: Config, args):
    print(f"使用设备: {cfg.device}")
    
    # --- 数据加载 ---
    transforms_list = transforms.Compose([
        transforms.Resize(cfg.image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = datasets.MNIST(root=cfg.dataset_path, train=True, download=True, transform=transforms_list)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # --- 初始化 ---
    model = SimpleUnet(cfg).to(cfg.device)
    diffusion = Diffusion(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    
    start_epoch = 0
    
    # --- 检查点加载逻辑 ---
    if args.load_checkpoint_path:
        if os.path.exists(args.load_checkpoint_path):
            print(f"从检查点加载模型: {args.load_checkpoint_path}")
            checkpoint = torch.load(args.load_checkpoint_path, map_location=cfg.device)
            
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                if 'optimizer_state_dict' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if 'epoch' in checkpoint:
                    start_epoch = checkpoint['epoch'] + 1
                print(f"模型加载完成。当前 epoch 为 {start_epoch - 1}。")
            else:
                model.load_state_dict(checkpoint)
                print("旧格式检查点加载完成，仅加载模型权重。")
        else:
            print(f"警告: 检查点路径 '{args.load_checkpoint_path}' 未找到，将从头开始训练。")
    
    # --- 恢复了 continue_training_epochs 的完整计算逻辑 ---
    if start_epoch > 0 and args.continue_training_epochs > 0:
        total_epochs_to_run = start_epoch + args.continue_training_epochs
        print(f"将从 epoch {start_epoch} 继续训练 {args.continue_training_epochs} 轮, 直到 epoch {total_epochs_to_run -1}。")
    elif start_epoch > 0 and args.continue_training_epochs == 0:
        print(f"已加载模型到 epoch {start_epoch - 1}。未指定 --continue_training_epochs，不进行额外训练。")
        return # 直接退出训练函数
    else: # 从头训练
        total_epochs_to_run = args.epochs
        print(f"将从头开始训练 {total_epochs_to_run} 个 epochs。")

    if start_epoch >= total_epochs_to_run and start_epoch > 0:
        print(f"加载的 epoch ({start_epoch - 1}) 已达到或超过目标 epoch ({total_epochs_to_run - 1})。无需额外训练。")
        return
    # --- 轮数计算逻辑结束 ---

    # --- 训练循环 ---
    for epoch in range(start_epoch, total_epochs_to_run):
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{total_epochs_to_run}")
        model.train()
        for step, (images, labels) in enumerate(progress_bar):
            optimizer.zero_grad()
            images, labels = images.to(cfg.device), labels.to(cfg.device)
            t = torch.randint(0, cfg.timesteps, (images.shape[0],), device=cfg.device).long()
            x_t, noise = diffusion.q_sample(x_start=images, t=t)
            
            if torch.rand(1).item() < cfg.cfg_dropout_rate:
                labels = None
            predicted_noise = model(x_t, t, labels)

            loss = criterion(noise, predicted_noise)
            loss.backward()
            optimizer.step()
            progress_bar.set_postfix(loss=loss.item())

        # --- 保存与采样 ---
        if (epoch + 1) % cfg.save_every_epoch == 0 or (epoch + 1) == total_epochs_to_run:
            print(f"\nEpoch {epoch+1}: 保存模型并生成样本...")
            model_path = os.path.join(cfg.output_dir, 'models', f"model_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, model_path)
            
            model.eval()
            with torch.no_grad():
                samples = diffusion.p_sample_loop_cfg(model, shape=(16, cfg.channels, cfg.image_size, cfg.image_size), class_labels=[cfg.preview_target_digit], guidance_scale=cfg.guidance_scale)
            
            samples = (samples + 1) / 0.5
            save_image(samples, os.path.join(cfg.output_dir, 'images', f'samples_epoch_{epoch+1}.png'), nrow=4)
            print(f"已保存样本到 {cfg.output_dir}/images/samples_epoch_{epoch+1}.png")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="训练 MNIST 扩散模型")
    parser.add_argument('--epochs', type=int, default=Config.epochs, help='从头训练时的总 epoch 数')
    parser.add_argument('--lr', type=float, default=Config.lr, help='学习率')
    parser.add_argument('--load_checkpoint_path', type=str, default=None, help='要加载的检查点模型文件的路径')
    parser.add_argument('--continue_training_epochs', type=int, default=0, help='如果加载检查点，额外训练的 epoch 数量 (0表示不额外训练)')
    
    args = parser.parse_args()
    
    cfg = Config()
    setup_directories()
    
    train(cfg, args)