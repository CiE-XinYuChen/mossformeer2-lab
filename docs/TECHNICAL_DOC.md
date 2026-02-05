# MossFormer2 技术文档

**版本**: v1.0
**基于论文**: MossFormer2: Combining Transformer and RNN-Free Recurrent Network for Enhanced Time-Domain Monaural Speech Separation (arXiv:2312.11825v2)
**实现日期**: 2025

---

## 目录

1. [模型架构](#1-模型架构)
2. [训练方法](#2-训练方法)
3. [损失函数](#3-损失函数)
4. [数据处理](#4-数据处理)
5. [实现细节](#5-实现细节)
6. [API参考](#6-api参考)
7. [配置参数](#7-配置参数)
8. [性能优化](#8-性能优化)
9. [故障排除](#9-故障排除)

---

## 1. 模型架构

### 1.1 总体架构

MossFormer2 采用经典的 **Encoder-Separator-Decoder** 框架：

```
输入音频 [B, T]
    ↓
┌─────────────────────┐
│   Encoder (Conv1D)   │  → 特征提取
└─────────────────────┘
    ↓ [B, N, S]
┌─────────────────────┐
│   MaskNet           │  → 掩码预测
│  (Separator)        │
└─────────────────────┘
    ↓ [C, B, N, S]
┌─────────────────────┐
│  Decoder (ConvT1D)  │  → 波形重建
└─────────────────────┘
    ↓
分离音频 [B, T] × C
```

**符号说明**:
- `B`: Batch size
- `T`: 时间采样点数
- `N`: 嵌入维度 (encoder_embedding_dim)
- `S`: 编码序列长度 = 2T/K - 1
- `K`: Encoder kernel size
- `C`: 说话人数量 (num_spks)

### 1.2 Encoder

**类型**: 1D 卷积编码器

```python
nn.Conv1d(
    in_channels=1,
    out_channels=N,     # 512
    kernel_size=K,      # 16
    stride=K//2,        # 8
    bias=False
)
+ ReLU()
```

**特点**:
- 使用 50% 重叠的卷积窗口 (stride = kernel_size / 2)
- 输出非负特征 (经过 ReLU)
- 无可学习偏置

**输入/输出**:
- 输入: `[B, T]` 原始波形
- 输出: `[B, N, S]` 编码特征，其中 S = 2T/K - 1

### 1.3 MaskNet (分离器核心)

MaskNet 由两种模块交替组成：

#### 1.3.1 MossFormer 模块

**功能**: 全局长程依赖建模

**核心机制**: Joint Local-Global Self-Attention
- **局部注意力**: 在非重叠分块上执行全计算自注意力
- **全局注意力**: 在整个序列上执行线性化注意力

**实现**: `FLASH_ShareA_FFConvM` (mossformer2_block.py:164-346)

```python
# 单个 MossFormer Layer 的前向流程
input [B, S, N]
    ↓
Token Shift (可选)
    ↓
to_hidden() → [v, u]  # FFConvM
to_qk() → qk          # FFConvM
    ↓
OffsetScale → [quad_q, lin_q, quad_k, lin_k]
    ↓
cal_attention() → [att_v, att_u]
    ↓
out = (att_u * v) * sigmoid(att_v * u)
    ↓
residual + to_out(out)
```

**关键组件**:

1. **FFConvM** (Feedforward Conv Module):
   ```
   LayerNorm → Linear → SiLU → ConvModule → Dropout
   ```

2. **Attention 计算**:
   - **Quadratic Attention**: ReLU(QK^T/g)^2 V
   - **Linear Attention**: Q(K^T V)
   - 两者相加得到最终输出

3. **Gating Mechanism**:
   ```
   output = (att_u ⊙ v) ⊙ σ(att_v ⊙ u)
   ```

**超参数** (论文配置):
- `group_size`: 256 (分块大小)
- `query_key_dim`: 128 (注意力向量维度)
- `expansion_factor`: 4.0 (FFN 扩展因子)
- `attn_dropout`: 0.1

#### 1.3.2 Recurrent 模块 (仅在 MossFormer2 中)

**功能**: 细粒度循环模式建模

**核心**: Gated FSMN (Feedforward Sequential Memory Network)

**结构** (mossformer2_block.py:560-637):

```
input [B, S, N]
    ↓
Bottleneck (1x1 Conv) → [B, S, N']
    ↓
┌──────────────────────┐
│    GCU Layer         │
│  ┌────────────────┐  │
│  │ Conv-U → u     │  │
│  │ Conv-U → v     │  │
│  │ Dilated_FSMN(v)│  │
│  │ out = u ⊙ v    │  │
│  └────────────────┘  │
└──────────────────────┘
    ↓
Output (1x1 Conv) → [B, S, N]
    ↓
residual connection
```

**Dilated FSMN Block** (mossformer2_block.py:Fig.3B):

```
input
  ↓
FFN Layer (Linear → PReLU → Linear)
  ↓
Memory Layer (Dense Dilated 2D Conv)
  │
  ├─ 2D Conv (dilation=1)
  ├─ 2D Conv (dilation=2)
  ├─ 2D Conv (dilation=4)
  │    ...
  └─ 2D Conv (dilation=2^(L-1))
  ↓
Dense Connections (concat all outputs)
  ↓
output
```

**关键参数**:
- `N'` (bottleneck_dim): 256
- `L` (num_fsmn_layers): 2
- `lorder`: 20 (FSMN 滤波器长度)

### 1.4 混合架构流程

**完整的 MossFormer2 Block**:

```python
for layer in range(R):  # R = num_mossformer_layer = 24
    x = MossFormer_Module(x)    # 全局依赖
    x = Recurrent_Module(x)     # 局部循环
```

### 1.5 Decoder

**类型**: 1D 转置卷积解码器

```python
nn.ConvTranspose1d(
    in_channels=N,
    out_channels=1,
    kernel_size=K,      # 16
    stride=K//2,        # 8
    bias=False
)
```

**输入/输出**:
- 输入: `[B, N, S]` 掩码处理后的特征
- 输出: `[B, T]` 重建波形

### 1.6 模型变体

| 模型 | R | N | K | N' | L | 参数量 |
|------|---|---|---|----|----|--------|
| MossFormer2 (Full) | 24 | 512 | 16 | 256 | 2 | 55.7M |
| MossFormer2 (S) | 25 | 384 | 16 | 256 | 2 | 37.8M |

---

## 2. 训练方法

### 2.1 训练配置 (复现论文)

#### 2.1.1 优化器

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=15e-5,           # 初始学习率
    weight_decay=0.0    # 无权重衰减
)
```

#### 2.1.2 学习率调度

**策略**: Constant + Step Decay

```python
# 伪代码
if epoch < 85:
    lr = 15e-5          # 保持常量
else:
    lr = 15e-5 * (0.5 ** (epoch - 85))  # 每轮衰减 0.5
```

**实际实现**:
```python
def get_lr(epoch):
    if epoch < lr_decay_epoch:  # 85
        return initial_lr  # 15e-5
    else:
        decay_steps = epoch - lr_decay_epoch
        return initial_lr * (lr_decay_factor ** decay_steps)
```

#### 2.1.3 梯度裁剪

```python
torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm=5.0,
    norm_type=2.0
)
```

**作用**: 防止梯度爆炸，稳定训练

#### 2.1.4 训练超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| Max Epochs | 200 | 最大训练轮数 |
| Batch Size | 1 | 批大小 (论文使用单卡V100) |
| Gradient Clip | 5.0 | L2 梯度裁剪阈值 |
| Initial LR | 15e-5 | 初始学习率 |
| LR Constant Epochs | 85 | 学习率保持不变的轮数 |
| LR Decay Factor | 0.5 | 学习率衰减因子 |

### 2.2 数据增强

#### 2.2.1 动态混合 (Dynamic Mixing)

**定义**: 训练时实时混合干净源信号

**启用条件** (论文):
- ✅ WSJ0-2mix
- ✅ WSJ0-3mix
- ✅ WHAM!
- ✅ WHAMR!
- ❌ Libri2Mix (数据集已足够大)

**实现**:
```python
if dynamic_mixing and split == 'train':
    # 加载单独的源信号
    sources = [load_audio(src_path) for src_path in source_paths]
    # 实时混合
    mixture = sum(sources)
else:
    # 加载预混合音频
    mixture = load_audio(mix_path)
```

**优势**:
- 增加数据多样性
- 防止模型过拟合特定混合方式
- 等效于无限数据增强

#### 2.2.2 音频分段

**训练阶段**:
```python
segment_length = 4.0  # 秒
segment_samples = int(segment_length * sample_rate)

# 随机裁剪
start = random.randint(0, audio_len - segment_samples)
segment = audio[start : start + segment_samples]
```

**验证/测试阶段**:
- 使用完整音频（不裁剪）
- 或使用固定长度的段

### 2.3 硬件配置

**论文配置**:
- GPU: 单张 NVIDIA V100 (32GB)
- 训练时长: ~5-7天 (WSJ0-2mix, 200 epochs)
- 内存: 32GB 系统内存

**推荐配置**:
- GPU: V100 / A100 / RTX 3090 / A6000
- VRAM: 最低 16GB (小模型) / 推荐 32GB (完整模型)
- 系统内存: 32GB+

---

## 3. 损失函数

### 3.1 SI-SDR (Scale-Invariant SDR)

**定义**: 尺度不变的源失真比

**数学公式**:

给定估计信号 $\hat{s}$ 和目标信号 $s$:

1. **去均值**:
   $$\hat{s}' = \hat{s} - \mathbb{E}[\hat{s}]$$
   $$s' = s - \mathbb{E}[s]$$

2. **目标投影**:
   $$s_{\text{target}} = \frac{\langle \hat{s}', s' \rangle}{\|s'\|^2} s'$$

3. **噪声残差**:
   $$e_{\text{noise}} = \hat{s}' - s_{\text{target}}$$

4. **SI-SDR**:
   $$\text{SI-SDR} = 10 \log_{10} \frac{\|s_{\text{target}}\|^2}{\|e_{\text{noise}}\|^2}$$

**PyTorch 实现** (loss.py:9-34):

```python
def si_sdr(estimated, target, eps=1e-8):
    # 去均值
    estimated = estimated - estimated.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)

    # 计算缩放因子
    dot_product = (estimated * target).sum(dim=-1, keepdim=True)
    target_energy = (target ** 2).sum(dim=-1, keepdim=True) + eps
    scale = dot_product / target_energy

    # 投影
    s_target = scale * target

    # 噪声
    e_noise = estimated - s_target

    # SI-SDR (dB)
    si_sdr_value = 10 * torch.log10(
        (s_target ** 2).sum(dim=-1) /
        ((e_noise ** 2).sum(dim=-1) + eps) + eps
    )

    return si_sdr_value
```

**作用**:
- 对信号幅度缩放不敏感
- 更符合人耳感知
- 是语音分离领域的标准评估指标

### 3.2 PIT (Permutation Invariant Training)

**问题**: 多说话人分离中，输出顺序是任意的

**解决方案**: 对所有可能的排列组合计算损失，选择最小的

**算法流程**:

```
给定估计源 [s1_est, s2_est, ..., sC_est]
     目标源 [s1_tgt, s2_tgt, ..., sC_tgt]

1. 生成所有排列: C! 种
   例如 C=2: [(0,1), (1,0)]
   例如 C=3: [(0,1,2), (0,2,1), (1,0,2), (1,2,0), (2,0,1), (2,1,0)]

2. 对每个排列 π 计算总损失:
   L(π) = Σ loss(si_est, s_π(i)_tgt)

3. 选择最小损失:
   L_final = min_π L(π)
```

**PyTorch 实现** (loss.py:48-90):

```python
def pit_loss(est_sources, target_sources):
    num_sources = len(est_sources)
    perms = list(itertools.permutations(range(num_sources)))

    losses = []
    for perm in perms:
        perm_loss = 0
        for src_idx, tgt_idx in enumerate(perm):
            perm_loss += si_sdr_loss(
                est_sources[src_idx],
                target_sources[tgt_idx]
            ).mean()
        losses.append(perm_loss)

    losses = torch.stack(losses)
    min_loss, min_idx = torch.min(losses, dim=0)
    best_perm = perms[min_idx]

    return min_loss, best_perm
```

**复杂度**:
- 时间复杂度: O(C! × C × T)
- 空间复杂度: O(C!)
- 对于 C=2: 2种排列
- 对于 C=3: 6种排列

### 3.3 训练损失

**完整训练损失** = SI-SDR + PIT

```python
criterion = MossFormer2Loss(num_spks=2)
loss, best_perm = criterion(estimated_sources, target_sources)
```

**返回值**:
- `loss`: 标量张量，用于反向传播
- `best_perm`: 最佳排列，用于日志记录

---

## 4. 数据处理

### 4.1 数据集格式

**标准目录结构**:

```
dataset_root/
├── train/
│   ├── mix/
│   │   ├── mix_000001.wav
│   │   ├── mix_000002.wav
│   │   └── ...
│   ├── s1/
│   │   ├── s1_000001.wav
│   │   ├── s1_000002.wav
│   │   └── ...
│   └── s2/
│       ├── s2_000001.wav
│       ├── s2_000002.wav
│       └── ...
├── val/
│   ├── mix/
│   ├── s1/
│   └── s2/
└── test/
    ├── mix/
    ├── s1/
    └── s2/
```

### 4.2 数据加载流程

**类**: `SeparationDataset` (dataset.py:15-182)

**初始化参数**:
```python
dataset = SeparationDataset(
    data_dir='/path/to/wsj0-2mix',
    split='train',              # 'train', 'val', 'test'
    sample_rate=8000,
    segment_length=4.0,         # 秒
    num_spks=2,
    dynamic_mixing=True         # 训练时启用
)
```

**加载流程**:

```python
def __getitem__(self, idx):
    # 1. 加载源信号
    sources = [load_audio(src_path) for src_path in source_paths]

    # 2. 生成或加载混合信号
    if dynamic_mixing and split == 'train':
        mixture = sum(sources)
    else:
        mixture = load_audio(mix_path)

    # 3. 裁剪段（训练时）
    if split == 'train':
        start = random.randint(0, length - segment_samples)
        mixture = mixture[start : start + segment_samples]
        sources = [s[start : start + segment_samples] for s in sources]

    # 4. 返回
    return {
        'mixture': mixture,     # [T]
        'sources': sources,     # List of [T]
    }
```

### 4.3 Batch Collation

**函数**: `collate_fn` (dataset.py:185-226)

**功能**: 将变长序列填充到相同长度

```python
def collate_fn(batch):
    # 找到最大长度
    max_len = max(item['mixture'].shape[0] for item in batch)

    # 填充到 max_len
    mixtures = []
    for item in batch:
        mix = item['mixture']
        if mix.shape[0] < max_len:
            padding = max_len - mix.shape[0]
            mix = F.pad(mix, (0, padding))
        mixtures.append(mix)

    # 堆叠为张量
    mixtures_batch = torch.stack(mixtures)  # [B, T]

    return {'mixture': mixtures_batch, 'sources': sources_batch}
```

### 4.4 音频预处理

**重采样**:
```python
if sr != target_sr:
    resampler = torchaudio.transforms.Resample(sr, target_sr)
    waveform = resampler(waveform)
```

**单声道转换**:
```python
if waveform.shape[0] > 1:
    waveform = torch.mean(waveform, dim=0, keepdim=True)
```

**归一化** (可选):
```python
# 方案1: 峰值归一化
max_amp = waveform.abs().max()
if max_amp > 0:
    waveform = waveform / max_amp

# 方案2: RMS归一化
rms = torch.sqrt((waveform ** 2).mean())
waveform = waveform / (rms + 1e-8)
```

**注意**: 论文未明确说明是否使用归一化

---

## 5. 实现细节

### 5.1 关键实现文件

| 文件 | 功能 | 行数 |
|------|------|------|
| `mossformer2.py` | 主模型定义 (Encoder/Decoder/MaskNet) | 816 |
| `mossformer2_block.py` | MossFormer和Recurrent块 | 845 |
| `conv_module.py` | 卷积模块 | ~400 |
| `fsmn.py` | FSMN实现 | ~350 |
| `layer_norm.py` | 归一化层 | ~150 |
| `loss.py` | 损失函数 (SI-SDR + PIT) | 252 |
| `dataset.py` | 数据加载 | 303 |
| `train.py` | 训练脚本 | 324 |

### 5.2 模型前向传播详解

**完整流程**:

```python
def forward(self, mixture):
    # mixture: [B, T]

    # 1. Encoder
    x = self.enc(mixture)  # [B, N, S]

    # 2. MaskNet
    mask = self.mask_net(x)  # [C, B, N, S]

    # 3. 应用掩码
    x_expanded = torch.stack([x] * self.num_spks)  # [C, B, N, S]
    sep_x = x_expanded * mask  # [C, B, N, S]

    # 4. Decoder
    outputs = []
    for i in range(self.num_spks):
        output_i = self.dec(sep_x[i])  # [B, T']
        outputs.append(output_i)

    # 5. 长度对齐
    for i in range(len(outputs)):
        if outputs[i].shape[1] < T:
            # 填充
            outputs[i] = F.pad(outputs[i], (0, T - outputs[i].shape[1]))
        else:
            # 裁剪
            outputs[i] = outputs[i][:, :T]

    return outputs  # List of [B, T]
```

### 5.3 MaskNet 内部流程

```python
def forward(self, x):
    # x: [B, N, S]

    # 1. 归一化
    x = self.norm(x)  # GlobalLayerNorm or LayerNorm

    # 2. 1x1 卷积降维
    x = self.conv1d_encoder(x)  # [B, N, S]

    # 3. 位置编码 (可选)
    if self.use_global_pos_enc:
        x = x.transpose(1, -1)  # [B, S, N]
        emb = self.pos_enc(x)   # [S, N]
        emb = emb.transpose(0, -1)  # [N, S]
        x = x.transpose(1, -1) + emb  # [B, N, S]

    # 4. Computation Block
    x = self.mdl(x)  # [B, N, S]
    # 内部包含:
    #   - MossFormer 模块 (×R 次)
    #   - Recurrent 模块 (×R 次，交替)

    x = self.prelu(x)

    # 5. 多说话人掩码生成
    x = self.conv1d_out(x)  # [B, N*C, S]
    x = x.view(B * C, -1, S)  # [B*C, N, S]

    # 6. 门控输出
    x = self.output(x) * self.output_gate(x)  # [B*C, N, S]

    # 7. 解码回原始维度
    x = self.conv1_decoder(x)  # [B*C, N_encoder, S]

    # 8. 重塑为多说话人格式
    x = x.view(B, C, N_encoder, S)  # [B, C, N, S]
    x = self.activation(x)  # ReLU
    x = x.transpose(0, 1)  # [C, B, N, S]

    return x
```

### 5.4 内存优化技巧

**1. 梯度检查点 (Gradient Checkpointing)**:

```python
from torch.utils.checkpoint import checkpoint

# 在 Computation_Block 中
def forward(self, x):
    for i, (mossformer_layer, recurrent_layer) in enumerate(layers):
        # 使用检查点节省内存
        x = checkpoint(mossformer_layer, x)
        x = checkpoint(recurrent_layer, x)
    return x
```

**节省内存**: ~50% (但增加 ~20% 训练时间)

**2. 混合精度训练**:

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in train_loader:
    with autocast():
        outputs = model(mixture)
        loss = criterion(outputs, sources)

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    scaler.step(optimizer)
    scaler.update()
```

**节省内存**: ~30-40%
**加速**: ~2-3x

**3. 梯度累积**:

```python
accumulation_steps = 4  # 模拟 batch_size=4

optimizer.zero_grad()
for i, batch in enumerate(train_loader):
    outputs = model(batch['mixture'])
    loss = criterion(outputs, batch['sources'])
    loss = loss / accumulation_steps  # 归一化
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        optimizer.zero_grad()
```

---

## 6. API 参考

### 6.1 模型类

#### `MossFormer2_SS_16K`

**初始化**:

```python
from mossformer2 import MossFormer2_SS_16K
from argparse import Namespace

args = Namespace(
    encoder_embedding_dim=512,      # Encoder 输出通道数
    mossformer_sequence_dim=512,    # MossFormer 隐藏维度
    num_mossformer_layer=24,        # 重复次数
    encoder_kernel_size=16,         # Encoder 卷积核大小
    num_spks=2,                     # 说话人数量
)

model = MossFormer2_SS_16K(args)
```

**前向传播**:

```python
# 输入
mixture = torch.randn(batch_size, num_samples)  # [B, T]

# 前向
with torch.no_grad():
    separated_sources = model(mixture)

# 输出
# separated_sources: List of [B, T], 长度为 num_spks
```

**参数**:

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| encoder_embedding_dim | int | 512 | Encoder 输出维度 |
| mossformer_sequence_dim | int | 512 | MossFormer 序列维度 |
| num_mossformer_layer | int | 24 | MossFormer 层数 |
| encoder_kernel_size | int | 16 | Encoder 卷积核大小 |
| num_spks | int | 2 | 说话人数量 |

### 6.2 损失函数类

#### `MossFormer2Loss`

**初始化**:

```python
from loss import MossFormer2Loss

criterion = MossFormer2Loss(num_spks=2)
```

**调用**:

```python
loss, best_perm = criterion(estimated_sources, target_sources)
```

**参数**:
- `estimated_sources`: List of Tensor [B, T], 估计的源信号
- `target_sources`: List of Tensor [B, T], 目标源信号

**返回**:
- `loss`: Tensor, 标量损失值
- `best_perm`: Tuple, 最佳排列索引

### 6.3 训练器类

#### `MossFormer2Trainer`

**初始化**:

```python
from train import MossFormer2Trainer

trainer = MossFormer2Trainer(config_path='configs/train_mossformer2.yaml')
```

**主要方法**:

```python
# 训练
trainer.train()

# 单轮训练
train_loss = trainer.train_epoch()

# 验证
val_loss, val_si_sdri = trainer.validate()

# 保存检查点
trainer.save_checkpoint(is_best=True)

# 加载检查点
trainer.load_checkpoint(checkpoint_path)
```

### 6.4 数据集类

#### `SeparationDataset`

**初始化**:

```python
from dataset import SeparationDataset

dataset = SeparationDataset(
    data_dir='/path/to/dataset',
    split='train',
    sample_rate=8000,
    segment_length=4.0,
    num_spks=2,
    dynamic_mixing=True
)
```

**使用**:

```python
from torch.utils.data import DataLoader
from dataset import collate_fn

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=collate_fn
)

for batch in loader:
    mixture = batch['mixture']      # [B, T]
    sources = batch['sources']      # List of [B, T]
```

---

## 7. 配置参数

### 7.1 YAML 配置文件结构

```yaml
# configs/train_mossformer2.yaml

# 基本设置
seed: 1234
data_folder: /path/to/dataset
output_folder: results/mossformer2/1234

# 数据集
dataset: wsj0-2mix
num_spks: 2
sample_rate: 8000
segment_length: 4.0

# 训练参数
N_epochs: 200
batch_size: 1
lr: 0.000015
gradient_clip: 5.0
lr_decay_epoch: 85
lr_decay_factor: 0.5

# 模型参数
encoder_kernel_size: 16
encoder_embedding_dim: 512
mossformer_sequence_dim: 512
num_mossformer_layer: 24

# 数据增强
use_dynamic_mixing: True
```

### 7.2 完整参数列表

| 参数组 | 参数名 | 类型 | 默认值 | 说明 |
|--------|--------|------|--------|------|
| **基础** | seed | int | 1234 | 随机种子 |
| | data_folder | str | - | 数据集路径 |
| | output_folder | str | - | 输出目录 |
| | save_folder | str | - | 检查点保存目录 |
| | train_log | str | - | 训练日志文件 |
| **数据集** | dataset | str | wsj0-2mix | 数据集类型 |
| | num_spks | int | 2 | 说话人数 |
| | sample_rate | int | 8000 | 采样率 |
| | segment_length | float | 4.0 | 音频段长度(秒) |
| | use_dynamic_mixing | bool | True | 是否动态混合 |
| **训练** | N_epochs | int | 200 | 最大训练轮数 |
| | batch_size | int | 1 | 批大小 |
| | num_workers | int | 4 | 数据加载线程数 |
| | lr | float | 15e-5 | 初始学习率 |
| | gradient_clip | float | 5.0 | 梯度裁剪阈值 |
| | lr_decay_epoch | int | 85 | 开始衰减的轮数 |
| | lr_decay_factor | float | 0.5 | 学习率衰减因子 |
| **模型** | encoder_kernel_size | int | 16 | Encoder卷积核 |
| | encoder_embedding_dim | int | 512 | Encoder输出维度 |
| | mossformer_sequence_dim | int | 512 | MossFormer维度 |
| | num_mossformer_layer | int | 24 | MossFormer层数 |
| | recurrent_bottleneck_dim | int | 256 | Recurrent瓶颈维度 |
| | recurrent_fsmn_layers | int | 2 | FSMN层数 |

---

## 8. 性能优化

### 8.1 训练速度优化

#### 方法1: 混合精度训练

```python
# 在 train.py 中添加
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

# 训练循环中
with autocast():
    outputs = model(mixture)
    loss, _ = criterion(outputs, sources)

scaler.scale(loss).backward()
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
scaler.step(optimizer)
scaler.update()
```

**效果**: 加速 2-3x，节省内存 30-40%

#### 方法2: 增大 Batch Size

```yaml
# 如果显存允许
batch_size: 4  # 从1增大到4
```

**注意**: 需要相应调整学习率
- 线性缩放规则: `new_lr = base_lr × (new_batch_size / base_batch_size)`

#### 方法3: DataLoader 优化

```yaml
num_workers: 8  # 增加数据加载线程
```

```python
# 启用 pin_memory
DataLoader(..., pin_memory=True)
```

### 8.2 内存优化

#### 方法1: 梯度检查点

```python
# 在 Computation_Block 中
from torch.utils.checkpoint import checkpoint

for layer in self.layers:
    x = checkpoint(layer, x)
```

#### 方法2: 减小模型

```yaml
# 使用小版本
encoder_embedding_dim: 384
num_mossformer_layer: 25
```

#### 方法3: 减少序列长度

```yaml
segment_length: 3.0  # 从4秒减到3秒
```

### 8.3 推理优化

#### 方法1: 模型量化

```python
# 动态量化
import torch.quantization
model_quantized = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear, torch.nn.Conv1d},
    dtype=torch.qint8
)
```

#### 方法2: TorchScript

```python
# 转换为 TorchScript
model.eval()
example = torch.randn(1, 32000)
traced_model = torch.jit.trace(model, example)
traced_model.save('mossformer2_traced.pt')
```

#### 方法3: ONNX 导出

```python
torch.onnx.export(
    model,
    example,
    'mossformer2.onnx',
    input_names=['mixture'],
    output_names=['source1', 'source2'],
    dynamic_axes={'mixture': {1: 'time'}}
)
```

---

## 9. 故障排除

### 9.1 常见错误

#### Error 1: CUDA Out of Memory

**症状**:
```
RuntimeError: CUDA out of memory. Tried to allocate X.XX GiB
```

**解决方案**:

1. **减小 batch size**:
   ```yaml
   batch_size: 1
   ```

2. **使用小模型**:
   ```yaml
   encoder_embedding_dim: 384
   num_mossformer_layer: 16
   ```

3. **启用混合精度**:
   ```python
   from torch.cuda.amp import autocast
   with autocast():
       outputs = model(mixture)
   ```

4. **梯度检查点**:
   ```python
   from torch.utils.checkpoint import checkpoint
   ```

5. **减少序列长度**:
   ```yaml
   segment_length: 2.0
   ```

#### Error 2: Loss 为 NaN

**症状**:
```
Loss: nan
```

**原因**:
- 梯度爆炸
- 学习率过大
- 数据异常

**解决方案**:

1. **检查梯度裁剪**:
   ```python
   torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
   ```

2. **降低学习率**:
   ```yaml
   lr: 1e-5  # 从 15e-5 降低
   ```

3. **检查数据**:
   ```python
   # 打印数据统计
   print(f"Mixture: min={mixture.min()}, max={mixture.max()}")
   print(f"Sources: min={sources[0].min()}, max={sources[0].max()}")
   ```

4. **添加归一化**:
   ```python
   mixture = mixture / mixture.abs().max()
   ```

#### Error 3: 训练太慢

**症状**:
每个 epoch 需要数小时

**解决方案**:

1. **增加 num_workers**:
   ```yaml
   num_workers: 8
   ```

2. **使用混合精度**

3. **启用 pin_memory**:
   ```python
   DataLoader(..., pin_memory=True)
   ```

4. **检查 I/O 瓶颈**:
   ```bash
   # 监控磁盘IO
   iostat -x 1
   ```

#### Error 4: SI-SDRi 不收敛

**症状**:
训练多个 epoch 后 SI-SDRi 仍然很低

**检查清单**:

- [ ] 数据集是否正确加载？
- [ ] 采样率是否匹配（8kHz）？
- [ ] PIT 是否正确实现？
- [ ] 学习率是否合适？
- [ ] 是否使用了动态混合？

**调试步骤**:

```python
# 1. 打印一个 batch 的数据
for batch in train_loader:
    print(batch['mixture'].shape)
    print([s.shape for s in batch['sources']])
    break

# 2. 检查模型输出
outputs = model(batch['mixture'])
print([o.shape for o in outputs])

# 3. 检查损失计算
loss, perm = criterion(outputs, batch['sources'])
print(f"Loss: {loss.item()}, Perm: {perm}")
```

### 9.2 性能调优

#### 基准测试

```python
import time

# 测试前向速度
model.eval()
mixture = torch.randn(1, 32000).cuda()

# Warmup
for _ in range(10):
    with torch.no_grad():
        _ = model(mixture)

# 测试
torch.cuda.synchronize()
start = time.time()
for _ in range(100):
    with torch.no_grad():
        _ = model(mixture)
torch.cuda.synchronize()
elapsed = time.time() - start

print(f"Average forward time: {elapsed / 100 * 1000:.2f} ms")
```

#### TensorBoard 分析

```bash
# 启动 TensorBoard
tensorboard --logdir results/mossformer2/1234/logs

# Profiler (可选)
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    outputs = model(mixture)

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

---

## 附录

### A. 论文结果复现检查表

- [ ] 数据集: WSJ0-2mix (8kHz)
- [ ] 模型: R=24, N=512, K=16
- [ ] 优化器: Adam, lr=15e-5
- [ ] 学习率调度: 85轮常量 + 衰减0.5
- [ ] 梯度裁剪: 5.0
- [ ] Batch size: 1
- [ ] 训练轮数: 200
- [ ] 动态混合: 启用
- [ ] 损失函数: SI-SDR + PIT

**预期结果** (WSJ0-2mix):
- SI-SDRi: ~24.1 dB

### B. 相关资源

- 论文: https://arxiv.org/abs/2312.11825
- SpeechBrain: https://github.com/speechbrain/speechbrain
- WSJ0-mix: https://github.com/mpariente/asteroid
- LibriMix: https://github.com/JorisCos/LibriMix

### C. 版本信息

```
PyTorch: >= 1.10.0
Python: >= 3.8
torchaudio: >= 0.10.0
CUDA: >= 11.3 (推荐)
```

### D. 引用

```bibtex
@article{zhao2024mossformer2,
  title={MossFormer2: Combining Transformer and RNN-Free Recurrent Network for Enhanced Time-Domain Monaural Speech Separation},
  author={Zhao, Shengkui and Ma, Yukun and Ni, Chongjia and Zhang, Chong and Wang, Hao and Nguyen, Trung Hieu and Zhou, Kun and Yip, Jia Qi and Ng, Dianwen and Ma, Bin},
  journal={arXiv preprint arXiv:2312.11825},
  year={2024}
}
```

---

**文档版本**: 1.0
**最后更新**: 2025
**作者**: MossFormer2 复现团队
