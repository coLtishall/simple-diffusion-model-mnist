# MNIST 扩散模型

本项目实现了一个简单的扩散模型，用于生成 MNIST 手写数字图像。

## 项目结构

```
.
├── .gitignore          # Git 忽略文件配置
├── config.py           # 配置文件，包含模型、训练和数据路径等参数
├── diffusion.py        # 实现扩散过程的核心逻辑
├── generate.py         # 脚本：加载训练好的模型生成新的数字图像
├── model.py            # 定义 U-Net 模型结构
├── requirement.txt     # 项目依赖的 Python 包
├── train.py            # 脚本：训练扩散模型
├── mnist_data/         # 存储 MNIST 数据集
│   └── MNIST/
│       └── raw/        # 原始 MNIST 数据文件
├── results/            # 存储训练结果
│   ├── images/         # 存储生成的样本图像
│   └── models/         # 存储训练好的模型权重
└── __pycache__/        # Python 编译的缓存文件
```

## 安装

1.  克隆本仓库。
2.  安装所需的 Python 包：
    ```bash
    pip install -r requirement.txt
    ```
3.  如果 `mnist_data/MNIST/raw/` 目录下没有数据，程序在首次运行时会自动下载 MNIST 数据集。

## 配置

项目的核心参数可以在 [`config.py`](config.py) 文件中进行配置，例如：

*   `data_dir`: MNIST 数据集路径
*   `output_dir`: 训练结果（模型和图像）的输出路径
*   `epochs`: 训练轮数
*   `batch_size`: 批处理大小
*   `learning_rate`: 学习率
*   `img_size`: 图像尺寸
*   `device`: 训练设备 ("cuda" 或 "cpu")

## 训练模型

要训练扩散模型，请运行 `train.py` 脚本，脚本会自动下载保存MNIST数据集，并创建模型、样本图像的保存目录。每5epoch保存模型参数、生成指定数字图片

```bash
python train.py
```

训练好的模型将保存在 `results/models/` 目录下，生成的样本图像将保存在 `results/images/` 目录下。

### 命令行参数

`train.py` 脚本支持以下命令行参数：

*   `--epochs INT`: 从头训练时的总 epoch 数。默认值取自 `config.py`。
*   `--lr FLOAT`: 学习率。默认值取自 `config.py`。
*   `--load_checkpoint_path STR`: 要加载的检查点模型文件的路径。如果提供此参数，模型将从指定的检查点继续训练。
*   `--continue_training_epochs INT`: 如果加载了检查点，额外训练的 epoch 数量。如果设置为 0 (默认值)，并且加载了检查点，则不会进行额外的训练，程序会直接退出。



## 生成图像

使用训练好的模型生成新的数字图像，请运行 [`generate.py`](generate.py) 脚本：

```bash
python generate.py
```

脚本会列出 `results/models/` 目录中所有可用的模型。您可以选择一个特定的模型或默认使用最新的模型。然后，按照提示输入您想要生成的数字 (0-9)。

## 文件说明

*   **[`config.py`](config.py):** 存储所有超参数和配置。
*   **[`model.py`](model.py):** 定义了 U-Net 模型的架构，这是扩散模型的核心组件。
*   **[`diffusion.py`](diffusion.py):** 实现了扩散和逆扩散过程的数学逻辑。
*   **[`train.py`](train.py):** 负责加载数据、初始化模型、执行训练循环，并保存模型权重和生成的样本。
*   **[`generate.py`](generate.py):** 加载预训练的模型，并根据用户输入生成指定数字的图像。

## 注意

*   确保在运行训练或生成脚本之前，已正确配置 [`config.py`](config.py) 中的路径。
*   训练过程可能需要较长时间，具体取决于您的硬件配置和设定的训练轮数。