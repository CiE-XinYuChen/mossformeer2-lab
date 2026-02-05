# LibreSpeech 数据准备完成总结

---

## ✅ 已完成的工作

我已经为你创建了完整的 LibreSpeech 数据准备和训练流程，所有功能都按照你的要求实现。

### 📦 新增文件列表

#### 1. **核心脚本**

| 文件 | 功能 | 大小 |
|------|------|------|
| `prepare_librespeech_data.py` | 数据准备主脚本 | ~16KB |
| `verify_prepared_data.py` | 数据验证脚本 | ~11KB |
| `test_librespeech_pipeline.sh` | 快速测试脚本 | ~4KB |

#### 2. **配置和文档**

| 文件 | 功能 |
|------|------|
| `configs/train_librespeech.yaml` | LibreSpeech 训练配置 |
| `LIBRESPEECH_GUIDE.md` | 完整使用指南 |
| `LIBRESPEECH_SUMMARY.md` | 本文档 |

#### 3. **修改的文件**

| 文件 | 修改内容 |
|------|----------|
| `dataset.py` | 添加 `CSVSeparationDataset` 类和 `create_csv_dataloaders` 函数 |
| `train.py` | 添加对 CSV 数据集的支持 |

---

## 🎯 功能实现

### ✅ 数据准备 (`prepare_librespeech_data.py`)

**实现的功能**:

1. ✅ **扫描 FLAC 文件**: 递归扫描 `dataset/origin` 下所有 `.flac` 文件
2. ✅ **转换为 WAV**: 自动转换为指定采样率（8000 或 16000 Hz）
3. ✅ **随机混合**: 支持混合任意数量的说话人（默认2个）
4. ✅ **随机插入**: B 音频从 A 音频的随机时刻插入
5. ✅ **生成 CSV**: 包含所有元数据（路径、时长、插入时间）
6. ✅ **数据集分割**: 自动分割为 train/val/test
7. ✅ **归一化**: 混合后的音频自动归一化

**混合规则**:
```
基础音频 A: ████████████████ (4秒)
插入音频 B:         ████████ (2秒，从随机位置插入)
混合结果:   ████████████████████ (总时长 = max(A长度, B插入位置 + B长度))
```

### ✅ CSV 格式

生成的 `metadata.csv` 包含以下字段：

```csv
mix_path,total_duration,s1_path,s1_duration,s1_insert_time,s2_path,s2_duration,s2_insert_time
train/mix/mix_000000.wav,6.5,train/s1/s1_000000.wav,4.2,0.0,train/s2/s2_000000.wav,3.1,1.5
```

**字段说明**:
- `mix_path`: 混合音频相对路径
- `total_duration`: 总时长（秒）
- `s{i}_path`: 第i个源音频路径
- `s{i}_duration`: 第i个源音频时长
- `s{i}_insert_time`: 第i个源音频插入时间（第一个为0）

### ✅ 数据验证 (`verify_prepared_data.py`)

**验证内容**:

1. ✅ **CSV 格式检查**: 验证必需字段
2. ✅ **文件完整性**: 检查所有音频文件是否存在
3. ✅ **音频属性**: 验证采样率、时长是否与 CSV 一致
4. ✅ **可视化**: 生成波形图查看混合效果

### ✅ 数据加载 (`dataset.py`)

**新增类和函数**:

```python
class CSVSeparationDataset(Dataset):
    """CSV 格式数据集加载器"""
    def __init__(self, data_root, csv_file, split, ...)
    def __getitem__(self, idx) -> {'mixture': Tensor, 'sources': List[Tensor]}

def create_csv_dataloaders(config):
    """创建 CSV 数据集的 DataLoader"""
    return train_loader, valid_loader, test_loader
```

### ✅ 训练支持 (`train.py`)

**修改内容**:

```python
# 自动检测数据集类型
dataset_type = config.get('dataset_type', 'standard')

if dataset_type == 'csv':
    # 使用 CSV 数据集
    loaders = create_csv_dataloaders(config)
else:
    # 使用标准数据集
    loaders = create_dataloaders(config)
```

---

## 🚀 快速开始

### 方法1: 使用测试脚本（推荐）

```bash
# 快速测试整个流程（生成10个样本）
./test_librespeech_pipeline.sh
```

### 方法2: 完整流程

```bash
# Step 1: 准备数据（10000个样本）
python prepare_librespeech_data.py \
    --input-dir dataset/origin \
    --output-dir dataset/prepared \
    --sample-rate 16000 \
    --num-speakers 2 \
    --num-samples 10000

# Step 2: 验证数据
python verify_prepared_data.py \
    --data-root dataset/prepared \
    --csv-file metadata.csv

# Step 3: 开始训练
python train.py --config configs/train_librespeech.yaml
```

---

## 📊 输出目录结构

运行 `prepare_librespeech_data.py` 后会生成：

```
dataset/prepared/
├── metadata.csv                 # 元数据文件
├── train/
│   ├── mix/
│   │   ├── mix_000000.wav
│   │   ├── mix_000001.wav
│   │   └── ... (8000个文件)
│   ├── s1/
│   │   ├── s1_000000.wav
│   │   └── ...
│   └── s2/
│       ├── s2_000000.wav
│       └── ...
├── val/
│   ├── mix/ (1000个文件)
│   ├── s1/
│   └── s2/
└── test/
    ├── mix/ (1000个文件)
    ├── s1/
    └── s2/
```

---

## ⚙️ 配置说明

### 数据准备参数

```bash
python prepare_librespeech_data.py \
    --input-dir dataset/origin \      # LibreSpeech FLAC 文件目录
    --output-dir dataset/prepared \   # 输出目录
    --sample-rate 16000 \             # 采样率（8000 或 16000）
    --num-speakers 2 \                # 混合说话人数
    --num-samples 10000 \             # 总样本数
    --train-ratio 0.8 \               # 训练集比例
    --val-ratio 0.1 \                 # 验证集比例
    --test-ratio 0.1 \                # 测试集比例
    --min-duration 3.0 \              # 最小音频时长（秒）
    --max-duration 10.0 \             # 最大音频时长（秒）
    --output-csv metadata.csv \       # CSV 文件名
    --seed 42                         # 随机种子
```

### 训练配置

编辑 `configs/train_librespeech.yaml`:

```yaml
# 关键配置
dataset_type: csv                    # 必须设置为 'csv'
data_root: dataset/prepared          # 数据集根目录
csv_file: metadata.csv               # CSV 文件
sample_rate: 16000                   # 采样率（与准备数据时一致）
num_spks: 2                          # 说话人数量
```

---

## 📈 性能预估

### 数据准备时间

| 样本数 | 采样率 | 预估时间 | 磁盘占用 |
|--------|--------|----------|----------|
| 1,000 | 16kHz | ~5 分钟 | ~1.5 GB |
| 10,000 | 16kHz | ~30 分钟 | ~15 GB |
| 50,000 | 16kHz | ~2.5 小时 | ~75 GB |

### 训练时间

| 样本数 | 轮数 | GPU | 预估时间 |
|--------|------|-----|----------|
| 10,000 | 200 | V100 | ~3-4 天 |
| 50,000 | 200 | V100 | ~12-15 天 |

---

## ✅ 验证清单

在开始训练前，请确认：

- [ ] LibreSpeech 数据已放在 `dataset/origin`
- [ ] Python 环境已安装必需的库（`soundfile`, `matplotlib`, `tqdm`）
- [ ] 磁盘空间充足（10000样本约需15GB）
- [ ] 数据准备脚本运行成功
- [ ] 验证脚本通过所有检查
- [ ] 配置文件中的路径正确
- [ ] GPU 可用且显存充足

---

## 🔧 常见问题

### Q: 找不到 FLAC 文件？

```bash
# 检查文件数量
find dataset/origin -name "*.flac" | wc -l

# 确认目录结构
ls -R dataset/origin | grep ".flac" | head -10
```

### Q: 内存不足？

**解决方案**:
1. 减少 `--num-samples`
2. 减少 `--max-duration`
3. 分批处理

### Q: 训练时加载数据失败？

**检查配置**:
```yaml
# configs/train_librespeech.yaml
dataset_type: csv                    # ← 必须是 'csv'
data_root: dataset/prepared          # ← 路径正确
csv_file: metadata.csv               # ← 文件存在
sample_rate: 16000                   # ← 与准备数据一致
```

### Q: 如何修改混合的说话人数量？

```bash
# 准备3说话人数据
python prepare_librespeech_data.py \
    --num-speakers 3 \
    --num-samples 10000

# 修改配置
# configs/train_librespeech.yaml
num_spks: 3  # 改为 3
```

---

## 📚 相关文档

- **`LIBRESPEECH_GUIDE.md`**: 详细使用指南
- **`TECHNICAL_DOC.md`**: 技术文档
- **`TRAINING_README.md`**: 训练指南
- **`PROJECT_OVERVIEW.md`**: 项目总览

---

## 🎉 总结

你现在拥有一个完整的 LibreSpeech 数据准备和训练流程：

1. ✅ **自动数据准备**: 扫描、转换、混合、生成 CSV
2. ✅ **数据验证**: 完整性检查和可视化
3. ✅ **灵活配置**: 支持任意说话人数、采样率、混合规则
4. ✅ **训练集成**: 无缝集成到 MossFormer2 训练流程
5. ✅ **完整文档**: 详细的使用指南和问题解决方案

**开始使用**:

```bash
# 快速测试
./test_librespeech_pipeline.sh

# 准备完整数据集
python prepare_librespeech_data.py --num-samples 10000

# 开始训练
python train.py --config configs/train_librespeech.yaml
```

**祝训练顺利！** 🚀
