# LibreSpeech 数据准备和训练指南

本指南详细说明如何使用 LibreSpeech 数据集训练 MossFormer2 模型。

---

## 📋 目录

1. [环境准备](#1-环境准备)
2. [数据准备](#2-数据准备)
3. [数据验证](#3-数据验证)
4. [开始训练](#4-开始训练)
5. [参数说明](#5-参数说明)
6. [常见问题](#6-常见问题)

---

## 1. 环境准备

### 安装依赖

```bash
# 激活环境
conda activate mossformer2

# 安装必需的库
pip install soundfile matplotlib tqdm
```

### 检查数据集

确保你已经将 LibreSpeech 数据集放在 `dataset/origin` 目录下：

```bash
# 目录结构示例
dataset/origin/
├── LibriSpeech/
│   ├── train-clean-100/
│   │   ├── 19/
│   │   ├── 26/
│   │   └── ...
│   ├── train-clean-360/
│   └── dev-clean/
```

---

## 2. 数据准备

### 2.1 基本用法

使用 `prepare_librespeech_data.py` 脚本准备数据：

```bash
python prepare_librespeech_data.py \
    --input-dir dataset/origin \
    --output-dir dataset/prepared \
    --sample-rate 16000 \
    --num-speakers 2 \
    --num-samples 10000
```

### 2.2 完整参数说明

```bash
python prepare_librespeech_data.py \
    --input-dir dataset/origin \          # LibreSpeech flac 文件所在目录
    --output-dir dataset/prepared \        # 输出目录
    --sample-rate 16000 \                  # 采样率（8000 或 16000）
    --num-speakers 2 \                     # 混合的说话人数量
    --num-samples 10000 \                  # 总样本数
    --train-ratio 0.8 \                    # 训练集比例
    --val-ratio 0.1 \                      # 验证集比例
    --test-ratio 0.1 \                     # 测试集比例
    --min-duration 3.0 \                   # 最小音频时长（秒）
    --max-duration 10.0 \                  # 最大音频时长（秒）
    --output-csv metadata.csv \            # 输出CSV文件名
    --seed 42                              # 随机种子
```

### 2.3 混合规则

脚本会随机选择 N 条音频（默认2条）进行混合：

- **基础音频（A）**: 第一条音频作为基础
- **插入音频（B）**: 从随机时刻插入到 A 中
- **插入位置**: 0 到 A的时长之间随机选择
- **总时长**: `max(A时长, B插入位置 + B时长)`

**示例**:
```
A音频: ████████████████ (4秒)
B音频:         ████████ (2秒，从2秒处插入)
混合后: ████████████████████ (6秒)
```

### 2.4 输出结构

数据准备完成后，会生成以下结构：

```
dataset/prepared/
├── metadata.csv                 # 元数据文件
├── train/
│   ├── mix/                     # 混合音频
│   │   ├── mix_000000.wav
│   │   ├── mix_000001.wav
│   │   └── ...
│   ├── s1/                      # 源音频 1
│   │   ├── s1_000000.wav
│   │   └── ...
│   └── s2/                      # 源音频 2
│       ├── s2_000000.wav
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

### 2.5 CSV 格式

`metadata.csv` 包含以下字段：

| 字段 | 说明 |
|------|------|
| `mix_path` | 混合音频相对路径 |
| `total_duration` | 混合音频总时长（秒） |
| `s1_path` | 源音频1相对路径 |
| `s1_duration` | 源音频1时长（秒） |
| `s1_insert_time` | 源音频1插入时间（秒） |
| `s2_path` | 源音频2相对路径 |
| `s2_duration` | 源音频2时长（秒） |
| `s2_insert_time` | 源音频2插入时间（秒） |

**示例行**:
```csv
mix_path,total_duration,s1_path,s1_duration,s1_insert_time,s2_path,s2_duration,s2_insert_time
train/mix/mix_000000.wav,6.5,train/s1/s1_000000.wav,4.2,0.0,train/s2/s2_000000.wav,3.1,1.5
```

---

## 3. 数据验证

### 3.1 运行验证脚本

数据准备完成后，建议运行验证脚本检查数据完整性：

```bash
python verify_prepared_data.py \
    --data-root dataset/prepared \
    --csv-file metadata.csv \
    --max-file-check 100 \
    --num-audio-check 10 \
    --sample-idx 0
```

### 3.2 验证内容

验证脚本会检查：

1. ✅ **CSV 格式**: 检查必需字段是否存在
2. ✅ **文件存在性**: 验证音频文件是否都存在
3. ✅ **音频属性**: 检查采样率、时长是否匹配CSV
4. ✅ **可视化**: 生成波形图查看混合结果

### 3.3 预期输出

```
============================================================
LibreSpeech Dataset Verification
============================================================
Data root: dataset/prepared
CSV file: dataset/prepared/metadata.csv
============================================================

Step 1: Verifying CSV format...
✓ CSV file found: dataset/prepared/metadata.csv
  Headers: ['mix_path', 'total_duration', 's1_path', ...]
✓ All required headers present
✓ Total rows: 10000

Step 2: Verifying audio files (checking up to 100 samples)...
TRAIN split:
  Total samples checked: 80
  Missing mix files: 0
  Missing source files: 0
  ✓ All files present!

...

Verification Summary
============================================================
  CSV_FORMAT: ✓ PASS
  FILES: ✓ PASS
  AUDIO: ✓ PASS
  VISUALIZATION: ✓ PASS

✓ All checks passed! Dataset is ready for training.
```

### 3.4 查看可视化结果

验证脚本会在 `verification_plots/` 目录下生成波形图：

```bash
# 查看生成的图片
ls verification_plots/
# sample_0.png

# 打开查看
# 可以看到混合音频和各个源音频的波形
```

---

## 4. 开始训练

### 4.1 使用配置文件训练

```bash
# 使用 LibreSpeech 配置
python train.py --config configs/train_librespeech.yaml
```

### 4.2 修改配置

编辑 `configs/train_librespeech.yaml`：

```yaml
# 数据集配置
dataset_type: csv                    # 使用 CSV 格式数据集
data_root: dataset/prepared          # 数据集根目录
csv_file: metadata.csv               # CSV 文件

# 音频参数
sample_rate: 16000                   # 采样率（与准备数据时一致）
segment_length: 4.0                  # 音频段长度（秒）
num_spks: 2                          # 说话人数量

# 训练参数
N_epochs: 200                        # 训练轮数
batch_size: 1                        # 批大小
lr: 0.000015                         # 学习率
```

### 4.3 监控训练

```bash
# 启动 TensorBoard
tensorboard --logdir results/mossformer2_librespeech/1234/logs --port 6006

# 在浏览器打开
# http://localhost:6006
```

### 4.4 查看日志

```bash
# 实时查看训练日志
tail -f results/mossformer2_librespeech/1234/train_log.txt
```

---

## 5. 参数说明

### 5.1 数据准备参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input-dir` | `dataset/origin` | FLAC文件目录 |
| `--output-dir` | `dataset/prepared` | 输出目录 |
| `--sample-rate` | 16000 | 采样率（Hz） |
| `--num-speakers` | 2 | 混合说话人数 |
| `--num-samples` | 10000 | 总样本数 |
| `--train-ratio` | 0.8 | 训练集比例 |
| `--val-ratio` | 0.1 | 验证集比例 |
| `--test-ratio` | 0.1 | 测试集比例 |
| `--min-duration` | 3.0 | 最小时长（秒） |
| `--max-duration` | 10.0 | 最大时长（秒） |
| `--seed` | 42 | 随机种子 |

### 5.2 推荐配置

#### 小规模测试
```bash
python prepare_librespeech_data.py \
    --num-samples 1000 \
    --min-duration 2.0 \
    --max-duration 5.0
```

#### 中等规模
```bash
python prepare_librespeech_data.py \
    --num-samples 10000 \
    --min-duration 3.0 \
    --max-duration 8.0
```

#### 大规模（类似论文）
```bash
python prepare_librespeech_data.py \
    --num-samples 50000 \
    --min-duration 3.0 \
    --max-duration 10.0
```

---

## 6. 常见问题

### Q1: 找不到 FLAC 文件

**症状**:
```
Found 0 FLAC files
```

**解决方案**:
1. 检查 `--input-dir` 路径是否正确
2. 确认 LibreSpeech 已完整下载并解压
3. 检查目录权限

```bash
# 检查文件数量
find dataset/origin -name "*.flac" | wc -l
```

### Q2: 内存不足

**症状**:
```
MemoryError: Unable to allocate array
```

**解决方案**:
1. 减少 `--num-samples`
2. 减少 `--max-duration`
3. 增加系统swap空间

### Q3: 验证失败

**症状**:
```
✗ Some checks failed
```

**解决方案**:
1. 查看详细错误信息
2. 检查磁盘空间是否充足
3. 确认音频文件没有损坏

```bash
# 检查磁盘空间
df -h dataset/prepared

# 重新生成问题样本
python prepare_librespeech_data.py --num-samples 100
```

### Q4: 训练时数据加载错误

**症状**:
```
Error loading audio: ...
```

**解决方案**:
1. 确认配置文件中的路径正确
2. 检查 CSV 文件格式
3. 验证音频文件完整性

```yaml
# configs/train_librespeech.yaml
dataset_type: csv                    # 必须设置为 csv
data_root: dataset/prepared          # 正确的路径
csv_file: metadata.csv               # 正确的CSV文件名
```

### Q5: 采样率不匹配

**症状**:
```
RuntimeError: Sample rate mismatch
```

**解决方案**:
确保配置文件中的采样率与数据准备时一致：

```yaml
# configs/train_librespeech.yaml
sample_rate: 16000  # 与 prepare_librespeech_data.py 的 --sample-rate 一致
```

---

## 7. 完整工作流示例

### 从头到尾的完整步骤

```bash
# Step 1: 准备数据（假设 LibreSpeech 已在 dataset/origin）
python prepare_librespeech_data.py \
    --input-dir dataset/origin \
    --output-dir dataset/prepared \
    --sample-rate 16000 \
    --num-speakers 2 \
    --num-samples 10000 \
    --seed 42

# Step 2: 验证数据
python verify_prepared_data.py \
    --data-root dataset/prepared \
    --csv-file metadata.csv

# Step 3: 查看可视化结果（可选）
# 打开 verification_plots/sample_0.png

# Step 4: 开始训练
python train.py --config configs/train_librespeech.yaml

# Step 5: 监控训练（另一个终端）
tensorboard --logdir results/mossformer2_librespeech/1234/logs
```

---

## 8. 性能提示

### 数据准备加速

1. **使用 SSD**: 将数据存储在 SSD 上可显著加快读写速度
2. **并行处理**: 修改脚本使用多进程加速
3. **预先过滤**: 先筛选符合时长要求的文件

### 训练加速

1. **增大 batch_size**: 如果显存允许
   ```yaml
   batch_size: 4  # 从1增加到4
   ```

2. **增加 num_workers**: 加快数据加载
   ```yaml
   num_workers: 8  # 根据CPU核心数调整
   ```

3. **混合精度训练**: 在 train.py 中启用 AMP

---

## 9. 数据集统计

准备好的数据集应该有以下统计信息：

| 指标 | 值（示例，num_samples=10000） |
|------|------------------------------|
| 训练样本 | 8000 |
| 验证样本 | 1000 |
| 测试样本 | 1000 |
| 平均时长 | ~6秒 |
| 总数据量 | ~16GB（16kHz wav） |
| 采样率 | 16000 Hz |
| 通道数 | 单声道 |
| 格式 | WAV PCM |

---

## 10. 下一步

数据准备完成后，可以：

1. ✅ 开始训练 MossFormer2 模型
2. ✅ 调整超参数优化性能
3. ✅ 准备更多数据提升效果
4. ✅ 在测试集上评估模型

**祝训练顺利！** 🎉

有问题请参考：
- `TECHNICAL_DOC.md` - 技术文档
- `TRAINING_README.md` - 训练指南
- `PROJECT_OVERVIEW.md` - 项目总览
