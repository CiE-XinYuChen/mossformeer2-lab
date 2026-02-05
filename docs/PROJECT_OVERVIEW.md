# MossFormer2 项目总览

## 📁 项目结构

```
MossFormer2_SS_16K/
├── 📄 核心模型文件
│   ├── mossformer2.py              # 主模型 (Encoder/Decoder/MaskNet)
│   ├── mossformer2_block.py        # MossFormer 和 Recurrent 块
│   ├── conv_module.py              # 卷积模块
│   ├── fsmn.py                     # FSMN 实现
│   └── layer_norm.py               # 归一化层
│
├── 🎓 训练相关
│   ├── train.py                    # 训练主脚本
│   ├── loss.py                     # 损失函数 (SI-SDR + PIT)
│   ├── dataset.py                  # 数据加载器
│   └── configs/
│       └── train_mossformer2.yaml  # 训练配置
│
├── 🔬 推理和测试
│   ├── inference_16k.py            # 推理脚本 (16kHz)
│   ├── inference_clearvoice.py     # ClearVoice 推理
│   ├── main.py                     # 简单示例
│   ├── test_model.py               # 模型测试套件
│   └── create_test_mix.py          # 创建测试混合音频
│
├── 📚 文档
│   ├── TECHNICAL_DOC.md            # 技术文档 (本文档)
│   ├── TRAINING_README.md          # 训练指南
│   ├── PROJECT_OVERVIEW.md         # 项目总览
│   └── requirements.txt            # Python 依赖
│
├── 🚀 工具脚本
│   └── quick_start.sh              # 快速启动脚本
│
├── 💾 模型和数据
│   ├── model/
│   │   └── last_best_checkpoint.pt # 预训练模型 (670MB)
│   ├── mdoels/                     # 备用模型目录
│   └── output/                     # 输出目录
│
└── 📑 论文
    └── file/
        └── 2312.11825v2.pdf        # MossFormer2 论文
```

## 🎯 核心文件说明

### 1. 模型定义 (`mossformer2.py`)
- **MossFormer**: 主模型类
- **MossFormer_MaskNet**: 掩码网络
- **Computation_Block**: 计算块
- **Encoder/Decoder**: 编解码器
- **参数量**: 55.7M (完整版) / 37.8M (小版本)

### 2. MossFormer Block (`mossformer2_block.py`)
- **FLASH_ShareA_FFConvM**: 单头门控注意力
- **Gated_FSMN_Block_Dilated**: 门控 FSMN
- **MossformerBlock_GFSMN**: 混合块 (Attention + FSMN)
- **MossformerBlock**: 纯 Attention 块

### 3. 训练脚本 (`train.py`)
- **MossFormer2Trainer**: 训练器类
  - `train_epoch()`: 训练一个 epoch
  - `validate()`: 验证
  - `save_checkpoint()`: 保存检查点
  - `load_checkpoint()`: 加载检查点

### 4. 损失函数 (`loss.py`)
- **si_sdr()**: SI-SDR 计算
- **PITLossWrapper**: PIT 包装器
- **MossFormer2Loss**: 完整损失 (SI-SDR + PIT)

### 5. 数据加载 (`dataset.py`)
- **SeparationDataset**: 数据集类
  - 支持动态混合
  - 随机分段
  - 多种数据集格式
- **collate_fn**: 批处理函数
- **create_dataloaders()**: 创建数据加载器

## 🔧 快速开始

### 1. 环境安装
```bash
# 激活环境
conda activate mossformer2

# 安装依赖
pip install -r requirements.txt
```

### 2. 测试模型
```bash
# 运行模型测试
python test_model.py

# 预期输出:
# ✓ Model created - Parameters: 55.7M
# ✓ Forward pass successful
# ✓ All tests passed!
```

### 3. 准备数据
```bash
# 设置数据集路径
# 编辑 configs/train_mossformer2.yaml
data_folder: /path/to/wsj0-2mix
```

### 4. 开始训练
```bash
# 方法1: 直接运行
python train.py --config configs/train_mossformer2.yaml

# 方法2: 使用快速启动脚本
./quick_start.sh
```

## 📊 模型配置对照表

| 配置 | 完整版 | 小版本 | 说明 |
|------|--------|--------|------|
| **名称** | MossFormer2 | MossFormer2-S | - |
| **参数量** | 55.7M | 37.8M | - |
| **层数 (R)** | 24 | 25 | num_mossformer_layer |
| **嵌入维度 (N)** | 512 | 384 | encoder_embedding_dim |
| **序列维度** | 512 | 384 | mossformer_sequence_dim |
| **Kernel (K)** | 16 | 16 | encoder_kernel_size |
| **瓶颈维度 (N')** | 256 | 256 | recurrent_bottleneck_dim |
| **FSMN 层数 (L)** | 2 | 2 | recurrent_fsmn_layers |
| **显存需求** | ~32GB | ~16GB | 训练时 (batch=1) |
| **推理速度** | 0.34x RT | 0.28x RT | Real-time factor |

## 📈 性能指标

### 论文报告结果

| 数据集 | SI-SDRi (dB) | 说明 |
|--------|-------------|------|
| WSJ0-2mix | 24.1 | 2说话人，干净混合 |
| WSJ0-3mix | 22.2 | 3说话人，干净混合 |
| Libri2Mix | 21.7 | 2说话人，大规模数据集 |
| WHAM! | 18.1 | 带噪声 |
| WHAMR! | 17.0 | 带噪声和混响 |

### 实测性能 (NVIDIA A6000)

| 操作 | 时间 | 说明 |
|------|------|------|
| 前向传播 (4s 音频) | 1.365s | Batch=1, FP32 |
| 实时因子 (RTF) | 0.341 | < 1.0 表示快于实时 |
| 训练一个 epoch | ~30-40 分钟 | WSJ0-2mix, 30h 数据 |
| 完整训练 (200 epochs) | ~5-7 天 | 单卡 V100/A6000 |

## 🛠️ 工作流程

### 训练工作流

```
1. 数据准备
   ├── 下载数据集 (WSJ0, LibriSpeech)
   ├── 生成混合音频 (wsj0-2mix)
   └── 验证数据格式

2. 配置修改
   ├── 编辑 configs/train_mossformer2.yaml
   ├── 设置 data_folder
   └── 调整训练参数

3. 训练启动
   ├── python train.py --config configs/...
   ├── 监控 TensorBoard
   └── 等待训练完成

4. 模型评估
   ├── 加载最佳检查点
   ├── 在测试集上评估
   └── 计算 SI-SDRi

5. 模型部署
   ├── 导出模型 (ONNX/TorchScript)
   ├── 优化推理速度
   └── 集成到应用
```

### 推理工作流

```
1. 加载模型
   model = MossFormer2_SS_16K(args)
   checkpoint = torch.load('best_checkpoint.pt')
   model.load_state_dict(checkpoint['model_state_dict'])

2. 加载音频
   mixture, sr = torchaudio.load('mix.wav')
   if sr != 16000:
       mixture = resample(mixture, sr, 16000)

3. 推理
   model.eval()
   with torch.no_grad():
       separated = model(mixture)

4. 保存结果
   for i, src in enumerate(separated):
       torchaudio.save(f'speaker_{i+1}.wav', src, 16000)
```

## 📋 检查清单

### 训练前检查
- [ ] Python 环境已配置 (Python 3.8+)
- [ ] PyTorch 已安装 (1.10+)
- [ ] 数据集已准备好
- [ ] 配置文件已修改 (data_folder)
- [ ] GPU 可用 (推荐 V100/A100/A6000)
- [ ] 磁盘空间充足 (>100GB)
- [ ] 模型测试通过 (`python test_model.py`)

### 训练中监控
- [ ] Loss 是否下降？
- [ ] SI-SDRi 是否提升？
- [ ] 学习率是否正确调度？
- [ ] GPU 利用率是否充分？
- [ ] 是否有 NaN 或 Inf？

### 训练后评估
- [ ] 最佳检查点已保存
- [ ] 测试集 SI-SDRi 已计算
- [ ] 结果接近论文报告
- [ ] 生成的音频质量检查

## 🔍 调试技巧

### 1. 快速测试
```bash
# 使用小数据集快速测试流程
python train.py --config configs/train_mossformer2.yaml \
    --max-epochs 5 \
    --batch-size 2
```

### 2. 单步调试
```python
# 在 train.py 中添加断点
import pdb; pdb.set_trace()

# 或使用 IPython
from IPython import embed; embed()
```

### 3. 可视化中间结果
```python
# 保存注意力权重
import matplotlib.pyplot as plt

attn_weights = model.get_attention_weights(mixture)
plt.imshow(attn_weights[0].cpu().numpy())
plt.savefig('attention.png')
```

## 📞 支持

### 问题诊断
1. 查看日志文件: `results/mossformer2/1234/train_log.txt`
2. 检查 TensorBoard: `tensorboard --logdir results/*/logs`
3. 查看 GPU 状态: `nvidia-smi`
4. 阅读技术文档: `TECHNICAL_DOC.md`

### 常见问题
- **CUDA OOM**: 减小 batch_size 或使用小模型
- **Loss 为 NaN**: 检查梯度裁剪和学习率
- **训练太慢**: 使用混合精度训练或增加 num_workers
- **SI-SDRi 不收敛**: 检查数据集和 PIT 实现

## 📚 相关资源

- 论文: [arXiv:2312.11825](https://arxiv.org/abs/2312.11825)
- SpeechBrain: https://github.com/speechbrain/speechbrain
- WSJ0-mix: https://github.com/mpariente/asteroid
- LibriMix: https://github.com/JorisCos/LibriMix

---

**项目状态**: ✅ 完成
**最后更新**: 2025
**版本**: 1.0
