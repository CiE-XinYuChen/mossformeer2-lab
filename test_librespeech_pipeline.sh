#!/bin/bash
# LibreSpeech 数据准备和训练流程快速测试脚本
# Quick test for LibreSpeech data preparation and training pipeline

set -e  # Exit on error

echo "=========================================="
echo "LibreSpeech Pipeline Quick Test"
echo "=========================================="
echo ""

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 配置
INPUT_DIR="dataset/origin"
OUTPUT_DIR="dataset/prepared_test"
NUM_SAMPLES=10  # 测试只生成10个样本
SAMPLE_RATE=16000
NUM_SPEAKERS=2

echo -e "${YELLOW}Configuration:${NC}"
echo "  Input dir: $INPUT_DIR"
echo "  Output dir: $OUTPUT_DIR"
echo "  Num samples: $NUM_SAMPLES"
echo "  Sample rate: $SAMPLE_RATE Hz"
echo "  Num speakers: $NUM_SPEAKERS"
echo ""

# Step 1: 检查输入目录
echo -e "${YELLOW}Step 1: Checking input directory...${NC}"
if [ ! -d "$INPUT_DIR" ]; then
    echo -e "${RED}✗ Input directory not found: $INPUT_DIR${NC}"
    echo "  Please place LibreSpeech data in $INPUT_DIR"
    exit 1
fi

# 统计 FLAC 文件数量
FLAC_COUNT=$(find "$INPUT_DIR" -name "*.flac" 2>/dev/null | wc -l)
echo -e "${GREEN}✓ Found $FLAC_COUNT FLAC files${NC}"

if [ "$FLAC_COUNT" -lt "$NUM_SPEAKERS" ]; then
    echo -e "${RED}✗ Not enough FLAC files for testing${NC}"
    echo "  Need at least $NUM_SPEAKERS files, found $FLAC_COUNT"
    exit 1
fi
echo ""

# Step 2: 准备数据
echo -e "${YELLOW}Step 2: Preparing dataset (${NUM_SAMPLES} samples)...${NC}"
python prepare_librespeech_data.py \
    --input-dir "$INPUT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --sample-rate "$SAMPLE_RATE" \
    --num-speakers "$NUM_SPEAKERS" \
    --num-samples "$NUM_SAMPLES" \
    --min-duration 2.0 \
    --max-duration 5.0 \
    --seed 42

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Data preparation completed${NC}"
else
    echo -e "${RED}✗ Data preparation failed${NC}"
    exit 1
fi
echo ""

# Step 3: 验证数据
echo -e "${YELLOW}Step 3: Verifying prepared data...${NC}"
python verify_prepared_data.py \
    --data-root "$OUTPUT_DIR" \
    --csv-file metadata.csv \
    --max-file-check "$NUM_SAMPLES" \
    --num-audio-check 5 \
    --sample-idx 0

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Data verification passed${NC}"
else
    echo -e "${RED}✗ Data verification failed${NC}"
    exit 1
fi
echo ""

# Step 4: 检查生成的文件
echo -e "${YELLOW}Step 4: Checking generated files...${NC}"

# 检查 CSV
if [ -f "$OUTPUT_DIR/metadata.csv" ]; then
    CSV_LINES=$(wc -l < "$OUTPUT_DIR/metadata.csv")
    echo -e "${GREEN}✓ CSV file: $((CSV_LINES-1)) samples${NC}"
else
    echo -e "${RED}✗ CSV file not found${NC}"
    exit 1
fi

# 检查音频文件
for split in train val test; do
    MIX_COUNT=$(find "$OUTPUT_DIR/$split/mix" -name "*.wav" 2>/dev/null | wc -l)
    S1_COUNT=$(find "$OUTPUT_DIR/$split/s1" -name "*.wav" 2>/dev/null | wc -l)
    S2_COUNT=$(find "$OUTPUT_DIR/$split/s2" -name "*.wav" 2>/dev/null | wc -l)

    if [ $MIX_COUNT -gt 0 ]; then
        echo -e "${GREEN}✓ $split: $MIX_COUNT mix, $S1_COUNT s1, $S2_COUNT s2${NC}"
    fi
done
echo ""

# Step 5: 测试数据加载
echo -e "${YELLOW}Step 5: Testing data loader...${NC}"
python -c "
import sys
sys.path.insert(0, '.')
from dataset import CSVSeparationDataset

try:
    dataset = CSVSeparationDataset(
        data_root='$OUTPUT_DIR',
        csv_file='metadata.csv',
        split='train',
        sample_rate=$SAMPLE_RATE,
        segment_length=4.0,
        num_spks=$NUM_SPEAKERS
    )
    print(f'✓ Dataset loaded: {len(dataset)} samples')

    # 测试加载一个样本
    sample = dataset[0]
    print(f'✓ Sample loaded:')
    print(f'  Mixture shape: {sample[\"mixture\"].shape}')
    print(f'  Sources: {len(sample[\"sources\"])} sources')
    print(f'  Source shapes: {[s.shape for s in sample[\"sources\"]]}')

except Exception as e:
    print(f'✗ Data loader test failed: {e}')
    sys.exit(1)
"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Data loader test passed${NC}"
else
    echo -e "${RED}✗ Data loader test failed${NC}"
    exit 1
fi
echo ""

# Step 6: 显示统计信息
echo -e "${YELLOW}Step 6: Dataset statistics${NC}"
python -c "
import csv
from pathlib import Path

csv_file = Path('$OUTPUT_DIR/metadata.csv')
with open(csv_file, 'r') as f:
    reader = list(csv.DictReader(f))

    total = len(reader)
    train = sum(1 for r in reader if r['mix_path'].startswith('train/'))
    val = sum(1 for r in reader if r['mix_path'].startswith('val/'))
    test = sum(1 for r in reader if r['mix_path'].startswith('test/'))

    durations = [float(r['total_duration']) for r in reader]
    avg_duration = sum(durations) / len(durations) if durations else 0

    print(f'Total samples: {total}')
    print(f'  Train: {train}')
    print(f'  Val: {val}')
    print(f'  Test: {test}')
    print(f'Average duration: {avg_duration:.2f}s')
    print(f'Duration range: {min(durations):.2f}s - {max(durations):.2f}s')
"
echo ""

# 完成
echo "=========================================="
echo -e "${GREEN}✓ All tests passed!${NC}"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Prepare full dataset:"
echo "   python prepare_librespeech_data.py --num-samples 10000"
echo ""
echo "2. Train model:"
echo "   python train.py --config configs/train_librespeech.yaml"
echo ""
echo "3. Monitor training:"
echo "   tensorboard --logdir results/mossformer2_librespeech/*/logs"
echo ""
