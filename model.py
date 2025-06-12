import torch
import torch.nn as nn
from config import Config # 使用相对导入

class SinusoidalPositionEmbeddings(nn.Module):
    """将时间步 t 编码为位置嵌入向量"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = torch.log(torch.tensor(10000.0, device=device)) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim,num_groups=8):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(num_groups, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups, out_ch)
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        self.class_mlp = nn.Linear(time_emb_dim,out_ch)
        
    def forward(self, x, t,c):
        h = self.relu(self.conv1(x))
        # 注入时间信息
        time_emb = self.relu(self.time_mlp(t))
        class_emb=self.relu(self.class_mlp(c))
        conditon = time_emb+class_emb
        h = h + conditon.unsqueeze(-1).unsqueeze(-1) # 扩展维度以匹配图像
        h = self.relu(self.conv2(h))
        return h

class SimpleUnet(nn.Module):
    # 使用下面这个完整的、修正后的 __init__ 方法
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg  # 保存配置对象的引用
        
        # 从配置对象中获取参数
        time_emb_dim = self.cfg.time_emb_dim
        self.num_classes = self.cfg.num_classes
        in_channels = self.cfg.channels

        # 类别嵌入
        self.class_emb = nn.Embedding(self.num_classes + 1, time_emb_dim) # 现在 num_classes 是一个整数，可以正常计算

        # 时间编码
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )
        
        # 下采样路径
        self.down1 = Block(in_channels, 64, time_emb_dim=time_emb_dim,num_groups=cfg.num_groups)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = Block(64, 128, time_emb_dim=time_emb_dim,num_groups=cfg.num_groups)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = Block(128, 256, time_emb_dim=time_emb_dim,num_groups=cfg.num_groups)
        self.pool3 = nn.MaxPool2d(2)
        
        # 瓶颈部分
        self.bot1 = Block(256, 512, time_emb_dim=time_emb_dim,num_groups=cfg.num_groups)

        # 上采样路径
        self.upconv1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2,output_padding=1)
        self.up1 = Block(512, 256, time_emb_dim=time_emb_dim,num_groups=cfg.num_groups)

        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.up2 = Block(256, 128, time_emb_dim=time_emb_dim,num_groups=cfg.num_groups)

        self.upconv3 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.up3 = Block(128, 64, time_emb_dim=time_emb_dim,num_groups=cfg.num_groups)

        # 输出层
        self.out = nn.Conv2d(64, in_channels, 1)

    def forward(self, x, timestep,class_labels):
        t = self.time_mlp(timestep)
        if class_labels is not None:
            c = self.class_emb(class_labels)
        else:
            # 当 labels 为 None 时, 使用无条件嵌入
            # 我们为 batch 中的每个样本都指定索引为 num_classes 的嵌入
            unconditional_embedding_index = self.num_classes
            c = self.class_emb(torch.tensor([unconditional_embedding_index] * x.shape[0], device=x.device))

        # 下采样
        x1 = self.down1(x, t,c)
        p1 = self.pool1(x1)
        x2 = self.down2(p1, t,c)
        p2 = self.pool2(x2)
        x3=self.down3(p2,t,c)
        p3=self.pool3(x3)

        # 瓶颈
        xb = self.bot1(p3, t,c)

        # 上采样和跳跃连接
        u1 = self.upconv1(xb)
        # 拼接来自下采样路径的特征图 (skip connection)
        u1_cat = torch.cat([u1, x3], dim=1) 
        u1_out = self.up1(u1_cat, t,c)
        
        u2 = self.upconv2(u1_out)
        u2_cat = torch.cat([u2, x2], dim=1) 
        u2_out = self.up2(u2_cat, t,c)

        u3=self.upconv3(u2_out)
        u3_cat=torch.cat([u3,x1],dim=1)
        u3_out=self.up3(u3_cat,t,c)
        return self.out(u3_out)
