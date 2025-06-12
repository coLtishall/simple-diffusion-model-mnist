# generate.py

import torch
from torchvision.utils import save_image
import os

from config import Config
from model import SimpleUnet
from diffusion import Diffusion

def generate():
    cfg = Config()
    print("=== MNIST 扩散模型数字生成器 ===")
    
    # --- 模型选择 ---
    models_dir = os.path.join(cfg.output_dir, 'models')
    if not os.path.exists(models_dir) or not os.listdir(models_dir):
        print("未找到模型文件，请先运行 train.py 进行训练！")
        return
    
    model_files = sorted(os.listdir(models_dir))
    print("\n可用的模型:")
    for i, model_file in enumerate(model_files):
        print(f"{i+1}. {model_file}")
    
    try:
        choice = input(f"\n选择模型 (1-{len(model_files)}) 或回车使用最新模型: ").strip()
        model_file = model_files[int(choice) - 1] if choice else model_files[-1]
    except (ValueError, IndexError):
        model_file = model_files[-1]
        print(f"无效选择，使用最新模型: {model_file}")
    
    model_path = os.path.join(models_dir, model_file)
    print(f"使用模型: {model_file}")
    
    # --- 加载模型 ---
    model = SimpleUnet(cfg).to(cfg.device)
    checkpoint = torch.load(model_path, map_location=cfg.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    diffusion = Diffusion(cfg)
    
    print(f"\n模型加载完成！设备: {cfg.device}")
    
    # --- 生成循环 ---
    while True:
        try:
            command = input("\n输入要生成的数字(0-9)，或输入 'q' 退出: ").strip().lower()
            if command == 'q':
                break
            
            digit = int(command)
            if not 0 <= digit <= 9:
                raise ValueError
                
            num_samples = int(input(f"生成多少张数字 {digit}？(默认16): ") or 16)
            guidance_scale = float(input("输入引导强度 (CFG Scale, 默认7.5): ") or 7.5)
            
            print(f"正在生成 {num_samples} 张数字 {digit} (CFG={guidance_scale})...")
            
            with torch.no_grad():
                sample_shape = (num_samples, cfg.channels, cfg.image_size, cfg.image_size)
                samples = diffusion.p_sample_loop_cfg(model, shape=sample_shape, class_labels=[digit], guidance_scale=guidance_scale)
                samples = (samples + 1) / 2.0
            
            save_path = os.path.join(cfg.output_dir, f"generated_{digit}.png")
            save_image(samples, save_path, nrow=4)
            print(f"图片已保存到: {save_path}")

        except ValueError:
            print("请输入一个有效的数字 (0-9)！")
        except KeyboardInterrupt:
            print("\n用户中断。")
            break
            
if __name__ == '__main__':
    generate()