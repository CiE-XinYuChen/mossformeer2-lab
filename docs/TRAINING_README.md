# MossFormer2 Training Guide

完整复现论文 "MossFormer2: Combining Transformer and RNN-Free Recurrent Network" 的训练流程。

## 📋 目录

1. [环境安装](#环境安装)
2. [数据准备](#数据准备)
3. [配置说明](#配置说明)
4. [开始训练](#开始训练)
5. [监控训练](#监控训练)
6. [评估模型](#评估模型)
7. [常见问题](#常见问题)

---

## 🔧 环境安装

### 方法1: 使用pip安装

```bash
# 创建虚拟环境（推荐）
conda create -n mossformer2 python=3.8
conda activate mossformer2

# 安装PyTorch (根据你的CUDA版本选择)
# CUDA 11.3
pip install torch==1.12.0+cu113 torchaudio==0.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

# 安装其他依赖
pip install -r requirements.txt
```

### 方法2: 使用SpeechBrain框架（推荐）

```bash
# 克隆并安装SpeechBrain
git clone https://github.com/speechbrain/speechbrain.git
cd speechbrain
pip install -r requirements.txt
pip install -e .
```

---

## 📊 数据准备

### 支持的数据集

1. **WSJ0-2mix / WSJ0-3mix** (论文主要使用)
2. **Libri2Mix**
3. **WHAM! / WHAMR!**

### WSJ0-2mix 数据准备

```bash
# 1. 下载WSJ0数据集（需要LDC授权）
# https://catalog.ldc.upenn.edu/LDC93S6A
# https://catalog.ldc.upenn.edu/LDC94S13A

# 2. 生成WSJ0-2mix数据集
git clone https://github.com/mpariente/asteroid.git
cd asteroid/egs/wsj0-mix/generate_data
./create_wsj_mix.sh /path/to/wsj0 /path/to/output/wsj0-2mix 8000 2

# 数据集结构应该如下:
# wsj0-2mix/
#   ├── train/
#   │   ├── mix/
#   │   ├── s1/
#   │   └── s2/
#   ├── val/
#   │   ├── mix/
#   │   ├── s1/
#   │   └── s2/
#   └── test/
#       ├── mix/
#       ├── s1/
#       └── s2/
```

### Libri2Mix 数据准备

```bash
# 使用官方脚本生成
git clone https://github.com/JorisCos/LibriMix.git
cd LibriMix
./scripts/generate_librimix.sh /path/to/librispeech /path/to/output/libri2mix

# 或者下载预生成的数据集
# https://zenodo.org/record/3871592
```

---

## ⚙️ 配置说明

编辑 `configs/train_mossformer2.yaml`:

```yaml
# 1. 修改数据路径
data_folder: /path/to/your/wsj0-2mix  # 改为你的数据集路径

# 2. 选择数据集类型
dataset: wsj0-2mix  # wsj0-2mix, wsj0-3mix, libri2mix, wham, whamr
num_spks: 2  # 说话人数量

# 3. 训练参数（论文配置，建议不改）
N_epochs: 200
batch_size: 1  # 如果显存足够可以改为2或4
lr: 0.000015  # 15e-5
gradient_clip: 5.0
lr_decay_epoch: 85
lr_decay_factor: 0.5

# 4. 模型配置（完整版 MossFormer2）
encoder_kernel_size: 16
encoder_embedding_dim: 512
num_mossformer_layer: 24
# 这个配置对应论文中的 55.7M 参数

# 5. 动态混合（论文设置）
use_dynamic_mixing: True  # Libri2Mix设为False
```

### 小版本配置 (MossFormer2-S)

如果显存不足，可以使用小版本（37.8M参数）:

```yaml
encoder_embedding_dim: 384
mossformer_sequence_dim: 384
num_mossformer_layer: 25
```

---

## 🚀 开始训练

### 基本训练命令

```bash
# 使用默认配置
python train.py --config configs/train_mossformer2.yaml

# 指定GPU
CUDA_VISIBLE_DEVICES=0 python train.py --config configs/train_mossformer2.yaml
```

### 从断点恢复训练

训练脚本会自动保存检查点，重新运行相同命令即可恢复：

```bash
# 自动从 latest_checkpoint.pt 恢复
python train.py --config configs/train_mossformer2.yaml
```

### 多GPU训练（可选）

```bash
# 使用 DataParallel (简单但效率较低)
# 修改 train.py 中的模型初始化:
# self.model = nn.DataParallel(self.model)

# 或使用 DistributedDataParallel (推荐)
python -m torch.distributed.launch --nproc_per_node=4 train.py --config configs/train_mossformer2.yaml
```

---

## 📈 监控训练

### 使用TensorBoard

```bash
# 启动TensorBoard
tensorboard --logdir results/mossformer2/1234/logs --port 6006

# 在浏览器打开: http://localhost:6006
```

### 查看训练日志

```bash
# 实时查看日志
tail -f results/mossformer2/1234/train_log.txt

# 查看最近10行
tail -n 10 results/mossformer2/1234/train_log.txt
```

### 检查点位置

```
results/mossformer2/1234/save/
├── latest_checkpoint.pt  # 最新检查点
└── best_checkpoint.pt    # 最佳验证集检查点
```

---

## 📊 评估模型

创建评估脚本 `evaluate.py`:

```python
import torch
from train import MossFormer2Trainer
from loss import si_sdr_improvement

def evaluate():
    # 加载训练器
    trainer = MossFormer2Trainer('configs/train_mossformer2.yaml')

    # 加载最佳检查点
    checkpoint_path = 'results/mossformer2/1234/save/best_checkpoint.pt'
    trainer.load_checkpoint(checkpoint_path)

    # 在测试集上评估
    trainer.model.eval()
    total_si_sdri = 0.0
    num_samples = 0

    with torch.no_grad():
        for batch in trainer.test_loader:
            mixture = batch['mixture'].to(trainer.device)
            sources = [s.to(trainer.device) for s in batch['sources']]

            # 推理
            estimated = trainer.model(mixture)

            # 计算 SI-SDRi
            for i in range(len(sources)):
                si_sdri = si_sdr_improvement(estimated[i], sources[i], mixture)
                total_si_sdri += si_sdri.mean().item()

            num_samples += 1

    avg_si_sdri = total_si_sdri / (num_samples * len(sources))
    print(f"Test SI-SDRi: {avg_si_sdri:.2f} dB")

if __name__ == '__main__':
    evaluate()
```

运行评估:

```bash
python evaluate.py
```

---

## 🎯 预期结果

根据论文 Table 2，在不同数据集上的预期 SI-SDRi 结果：

| 数据集 | SI-SDRi (dB) |
|--------|-------------|
| WSJ0-2mix | 24.1 |
| WSJ0-3mix | 22.2 |
| Libri2Mix | 21.7 |
| WHAM! | 18.1 |
| WHAMR! | 17.0 |

**注意**: 达到这些结果需要：
- 完整训练200个epochs
- 正确的数据预处理
- 论文中的动态混合设置
- 可能需要多次运行取最佳结果

---

## ❓ 常见问题

### Q1: CUDA Out of Memory

**解决方案**:
```yaml
# 1. 减小batch size
batch_size: 1  # 已经是最小了

# 2. 使用小版本模型
encoder_embedding_dim: 384
num_mossformer_layer: 25

# 3. 减少音频长度
segment_length: 3.0  # 从4秒改为3秒

# 4. 使用梯度累积模拟大batch
# 修改train.py，每N步更新一次
```

### Q2: 训练太慢

**解决方案**:
```bash
# 1. 增加num_workers
num_workers: 8  # 根据CPU核心数调整

# 2. 使用混合精度训练
# 在train.py中添加:
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

# 3. 减少验证频率
# 每5个epoch验证一次而不是每个epoch
```

### Q3: SI-SDRi太低

**检查清单**:
- ✓ 数据集是否正确生成？
- ✓ 采样率是否匹配（8kHz）？
- ✓ 动态混合是否启用？
- ✓ 学习率调度是否正确？
- ✓ 是否训练足够的epochs？

### Q4: 如何使用16kHz数据？

```yaml
# 修改配置
sample_rate: 16000
encoder_kernel_size: 16  # 保持不变或调整为32

# 注意: 论文使用8kHz，16kHz可能需要重新调整参数
```

---

## 📝 训练检查清单

开始训练前确认：

- [ ] 数据集已正确下载和生成
- [ ] 配置文件中的路径已修改
- [ ] 已安装所有依赖（`pip install -r requirements.txt`）
- [ ] GPU显存足够（建议32GB V100或以上）
- [ ] 硬盘空间足够（检查点文件约2-3GB）
- [ ] 已设置正确的CUDA_VISIBLE_DEVICES

---

## 🔗 参考资源

- 论文: [arXiv:2312.11825](https://arxiv.org/abs/2312.11825)
- SpeechBrain: https://github.com/speechbrain/speechbrain
- WSJ0-mix生成: https://github.com/mpariente/asteroid
- LibriMix: https://github.com/JorisCos/LibriMix

---

## 📧 问题反馈

训练过程中遇到问题可以：
1. 查看上述常见问题部分
2. 检查训练日志和TensorBoard
3. 确认配置是否与论文一致

祝训练顺利！🎉
