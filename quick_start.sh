#!/bin/bash
# Quick start script for MossFormer2 training
# 快速启动训练的脚本

set -e  # Exit on error

echo "=========================================="
echo "MossFormer2 Training Quick Start"
echo "=========================================="
echo ""

# 1. 检查环境
echo "Step 1: Checking environment..."
python -c "import torch; print(f'✓ PyTorch {torch.__version__} installed')"
python -c "import torchaudio; print(f'✓ torchaudio installed')"
python -c "import yaml; print('✓ PyYAML installed')" || echo "⚠ PyYAML not found, installing..." && pip install pyyaml
python -c "import tensorboard; print('✓ TensorBoard installed')" || echo "⚠ TensorBoard not found, installing..." && pip install tensorboard
echo ""

# 2. 检查配置文件
echo "Step 2: Checking configuration..."
if [ ! -f "configs/train_mossformer2.yaml" ]; then
    echo "✗ Config file not found!"
    echo "  Please create configs/train_mossformer2.yaml first"
    exit 1
fi
echo "✓ Configuration file found"
echo ""

# 3. 检查数据路径（从配置中读取）
echo "Step 3: Checking data folder..."
DATA_FOLDER=$(python -c "import yaml; config=yaml.safe_load(open('configs/train_mossformer2.yaml')); print(config.get('data_folder', ''))")
if [ -z "$DATA_FOLDER" ] || [ ! -d "$DATA_FOLDER" ]; then
    echo "⚠ Warning: Data folder not found or not set in config"
    echo "  Data folder: $DATA_FOLDER"
    echo "  Please update 'data_folder' in configs/train_mossformer2.yaml"
    echo "  Or use dummy data for testing (not recommended)"
    read -p "  Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "✓ Data folder found: $DATA_FOLDER"
fi
echo ""

# 4. 创建输出目录
echo "Step 4: Creating output directories..."
mkdir -p results/mossformer2
echo "✓ Output directories created"
echo ""

# 5. 显示训练配置
echo "Step 5: Training configuration:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python -c "
import yaml
config = yaml.safe_load(open('configs/train_mossformer2.yaml'))
print(f\"  Dataset: {config.get('dataset', 'N/A')}\")
print(f\"  Speakers: {config.get('num_spks', 'N/A')}\")
print(f\"  Epochs: {config.get('N_epochs', 'N/A')}\")
print(f\"  Batch size: {config.get('batch_size', 'N/A')}\")
print(f\"  Learning rate: {config.get('lr', 'N/A')}\")
print(f\"  Model layers: {config.get('num_mossformer_layer', 'N/A')}\")
print(f\"  Embedding dim: {config.get('encoder_embedding_dim', 'N/A')}\")
"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 6. 询问是否开始训练
echo "Ready to start training!"
echo ""
echo "Options:"
echo "  1) Start training (default)"
echo "  2) Start training and open TensorBoard"
echo "  3) Cancel"
echo ""
read -p "Choose option [1-3] (default=1): " option
option=${option:-1}

case $option in
    1)
        echo ""
        echo "Starting training..."
        python train.py --config configs/train_mossformer2.yaml
        ;;
    2)
        echo ""
        echo "Starting TensorBoard in background..."
        OUTPUT_FOLDER=$(python -c "import yaml; config=yaml.safe_load(open('configs/train_mossformer2.yaml')); print(config['output_folder'])")
        tensorboard --logdir "$OUTPUT_FOLDER/logs" --port 6006 --bind_all &
        TB_PID=$!
        echo "✓ TensorBoard started (PID: $TB_PID)"
        echo "  Access at: http://localhost:6006"
        echo ""
        sleep 2
        echo "Starting training..."
        python train.py --config configs/train_mossformer2.yaml
        # Kill tensorboard when training finishes
        kill $TB_PID 2>/dev/null || true
        ;;
    3)
        echo "Cancelled."
        exit 0
        ;;
    *)
        echo "Invalid option. Cancelled."
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "Training completed or stopped!"
echo "=========================================="
