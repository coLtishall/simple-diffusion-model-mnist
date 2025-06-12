# diffusion.py

import torch
from tqdm import tqdm
from config import Config

def get_cosine_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

class Diffusion:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = self.cfg.device
        self.timesteps = self.cfg.timesteps

        self.betas = get_cosine_schedule(self.timesteps, s=self.cfg.cosine_schedule_s).to(self.device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = torch.nn.functional.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

# diffusion.py 中修正后的版本
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        
        noisy_image = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
        # 返回加噪后的图片和所使用的噪声
        return noisy_image, noise
        
    @torch.no_grad()
    def p_sample_loop_cfg(self, model, shape, class_labels, guidance_scale):
        """使用 Classifier-Free Guidance 进行的完整采样循环"""
        img = torch.randn(shape, device=self.device)
        
        labels = torch.tensor(class_labels, device=self.device, dtype=torch.long)
        if labels.numel() == 1:
            labels = labels.repeat(shape[0])

        for i in tqdm(reversed(range(0, self.timesteps)), desc="CFG Sampling...", total=self.timesteps):
            t = torch.full((shape[0],), i, device=self.device, dtype=torch.long)
            
            predicted_noise_cond = model(img, t, labels)
            predicted_noise_uncond = model(img, t, None)
            
            noise_pred = predicted_noise_uncond + guidance_scale * (predicted_noise_cond - predicted_noise_uncond)
            
            betas_t = self.betas[t].view(-1, 1, 1, 1)
            sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
            sqrt_recip_alphas_t = self.sqrt_recip_alphas[t].view(-1, 1, 1, 1)
            
            model_mean = sqrt_recip_alphas_t * (img - betas_t * noise_pred / sqrt_one_minus_alphas_cumprod_t)
            
            if i == 0:
                img = model_mean
            else:
                posterior_variance_t = self.posterior_variance[t].view(-1, 1, 1, 1)
                noise = torch.randn_like(img)
                img = model_mean + torch.sqrt(posterior_variance_t) * noise
        
        return img