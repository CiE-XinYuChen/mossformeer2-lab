"""
LibreSpeech 数据准备脚本
功能：
1. 扫描 dataset/origin 下的 flac 文件
2. 转换为指定采样率的 wav
3. 随机混合N条语音（默认2条）
4. 生成训练/验证/测试数据集
5. 输出 CSV 元数据文件
"""

import os
import random
import csv
import argparse
from pathlib import Path
from tqdm import tqdm
import soundfile as sf
import numpy as np
from typing import List, Tuple, Dict


class LibreSpeechDataPreparator:
    """LibreSpeech 数据准备器"""

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        sample_rate: int = 16000,
        num_speakers: int = 2,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        min_duration: float = 3.0,
        max_duration: float = 10.0,
        seed: int = 42
    ):
        """
        Args:
            input_dir: 输入目录（dataset/origin）
            output_dir: 输出目录（dataset/prepared）
            sample_rate: 目标采样率（8000 或 16000）
            num_speakers: 混合的说话人数量（默认2）
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            min_duration: 最小音频时长（秒）
            max_duration: 最大音频时长（秒）
            seed: 随机种子
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.sample_rate = sample_rate
        self.num_speakers = num_speakers
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.seed = seed

        # 设置随机种子
        random.seed(seed)
        np.random.seed(seed)

        # 创建输出目录
        self.create_output_dirs()

    def create_output_dirs(self):
        """创建输出目录结构"""
        for split in ['train', 'val', 'test']:
            (self.output_dir / split / 'mix').mkdir(parents=True, exist_ok=True)
            for i in range(1, self.num_speakers + 1):
                (self.output_dir / split / f's{i}').mkdir(parents=True, exist_ok=True)

    def scan_flac_files(self) -> List[Path]:
        """扫描所有 flac 文件"""
        print(f"Scanning FLAC files in {self.input_dir}...")
        flac_files = list(self.input_dir.rglob('*.flac'))
        print(f"Found {len(flac_files)} FLAC files")
        return flac_files

    def load_and_resample(self, file_path: Path) -> Tuple[np.ndarray, int]:
        """
        加载音频并重采样

        Returns:
            audio: 音频数据 (samples,)
            sr: 采样率
        """
        # 读取音频
        audio, sr = sf.read(file_path)

        # 转换为单声道
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        # 重采样
        if sr != self.sample_rate:
            # 使用简单的线性插值重采样
            duration = len(audio) / sr
            target_length = int(duration * self.sample_rate)
            audio = np.interp(
                np.linspace(0, len(audio) - 1, target_length),
                np.arange(len(audio)),
                audio
            )

        return audio.astype(np.float32), self.sample_rate

    def filter_audio_by_duration(self, audio: np.ndarray, sr: int) -> bool:
        """检查音频时长是否在范围内"""
        duration = len(audio) / sr
        return self.min_duration <= duration <= self.max_duration

    def mix_audios(self, audios: List[np.ndarray]) -> Tuple[np.ndarray, List[Dict]]:
        """
        混合多条音频
        混合规则：第一条为基础，其他音频从随机位置插入

        Args:
            audios: 音频列表，每个元素为 (samples,)

        Returns:
            mixed: 混合后的音频
            metadata: 混合元数据列表，包含每条音频的时长和插入时间
        """
        if len(audios) < 2:
            raise ValueError("至少需要2条音频进行混合")

        # 基础音频（第一条）
        base_audio = audios[0].copy()
        base_duration = len(base_audio) / self.sample_rate

        metadata = []
        metadata.append({
            'duration': base_duration,
            'insert_time': 0.0  # 第一条从0开始
        })

        # 混合其他音频
        for i, audio in enumerate(audios[1:], start=1):
            audio_duration = len(audio) / self.sample_rate

            # 随机选择插入位置（秒）
            # 插入时间范围：[0, base_duration]
            max_insert_time = base_duration
            insert_time = random.uniform(0, max_insert_time)
            insert_sample = int(insert_time * self.sample_rate)

            metadata.append({
                'duration': audio_duration,
                'insert_time': insert_time
            })

            # 计算混合后的总长度
            required_length = insert_sample + len(audio)

            # 扩展基础音频长度（如果需要）
            if required_length > len(base_audio):
                base_audio = np.pad(
                    base_audio,
                    (0, required_length - len(base_audio)),
                    mode='constant'
                )

            # 混合音频
            base_audio[insert_sample:insert_sample + len(audio)] += audio

        # 归一化混合音频
        max_val = np.abs(base_audio).max()
        if max_val > 0:
            base_audio = base_audio / max_val * 0.9  # 留10%余量

        total_duration = len(base_audio) / self.sample_rate

        return base_audio, metadata, total_duration

    def save_mixture(
        self,
        mixed_audio: np.ndarray,
        source_audios: List[np.ndarray],
        split: str,
        idx: int,
        metadata: List[Dict],
        total_duration: float
    ) -> Dict:
        """
        保存混合音频和源音频

        Returns:
            record: CSV记录字典
        """
        # 文件名
        mix_filename = f'mix_{idx:06d}.wav'
        source_filenames = [f's{i+1}_{idx:06d}.wav' for i in range(len(source_audios))]

        # 保存路径
        mix_path = self.output_dir / split / 'mix' / mix_filename
        source_paths = [
            self.output_dir / split / f's{i+1}' / source_filenames[i]
            for i in range(len(source_audios))
        ]

        # 保存混合音频
        sf.write(mix_path, mixed_audio, self.sample_rate)

        # 保存源音频（需要pad到相同长度）
        mix_length = len(mixed_audio)
        for i, (audio, path) in enumerate(zip(source_audios, source_paths)):
            # Pad到混合音频长度
            if len(audio) < mix_length:
                audio_padded = np.pad(
                    audio,
                    (0, mix_length - len(audio)),
                    mode='constant'
                )
            else:
                audio_padded = audio[:mix_length]

            sf.write(path, audio_padded, self.sample_rate)

        # 构建CSV记录
        record = {
            'mix_path': f'{split}/mix/{mix_filename}',
            'total_duration': total_duration,
        }

        # 添加每个源的信息
        for i in range(len(source_audios)):
            record[f's{i+1}_path'] = f'{split}/s{i+1}/{source_filenames[i]}'
            record[f's{i+1}_duration'] = metadata[i]['duration']
            record[f's{i+1}_insert_time'] = metadata[i]['insert_time']

        return record

    def prepare_dataset(
        self,
        num_samples: int = 10000,
        output_csv: str = 'metadata.csv'
    ):
        """
        准备完整数据集

        Args:
            num_samples: 生成的样本总数
            output_csv: 输出CSV文件名
        """
        print("\n" + "=" * 60)
        print("LibreSpeech Data Preparation")
        print("=" * 60)
        print(f"Configuration:")
        print(f"  Input dir: {self.input_dir}")
        print(f"  Output dir: {self.output_dir}")
        print(f"  Sample rate: {self.sample_rate} Hz")
        print(f"  Num speakers: {self.num_speakers}")
        print(f"  Num samples: {num_samples}")
        print(f"  Train/Val/Test: {self.train_ratio}/{self.val_ratio}/{self.test_ratio}")
        print("=" * 60 + "\n")

        # 扫描所有音频文件
        all_files = self.scan_flac_files()

        if len(all_files) < self.num_speakers:
            raise ValueError(
                f"文件数量不足！需要至少 {self.num_speakers} 个文件，"
                f"但只找到 {len(all_files)} 个"
            )

        # 加载并过滤音频
        print("\nLoading and filtering audio files...")
        valid_audios = []
        valid_files = []

        for file_path in tqdm(all_files, desc="Loading audio"):
            try:
                audio, sr = self.load_and_resample(file_path)
                if self.filter_audio_by_duration(audio, sr):
                    valid_audios.append(audio)
                    valid_files.append(file_path)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue

        print(f"Valid audio files: {len(valid_audios)}")

        if len(valid_audios) < num_samples * self.num_speakers:
            print(f"Warning: 只有 {len(valid_audios)} 个有效音频文件，"
                  f"可能不足以生成 {num_samples} 个样本")
            print(f"将使用重复采样")

        # 分配样本到各个集合
        num_train = int(num_samples * self.train_ratio)
        num_val = int(num_samples * self.val_ratio)
        num_test = num_samples - num_train - num_val

        splits_config = [
            ('train', num_train),
            ('val', num_val),
            ('test', num_test)
        ]

        # CSV 字段
        fieldnames = ['mix_path', 'total_duration']
        for i in range(1, self.num_speakers + 1):
            fieldnames.extend([
                f's{i}_path',
                f's{i}_duration',
                f's{i}_insert_time'
            ])

        # 打开CSV文件
        csv_path = self.output_dir / output_csv
        csv_file = open(csv_path, 'w', newline='', encoding='utf-8')
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        csv_writer.writeheader()

        # 生成每个集合的数据
        global_idx = 0

        for split, num_split_samples in splits_config:
            print(f"\nGenerating {split} set ({num_split_samples} samples)...")

            for i in tqdm(range(num_split_samples), desc=f"Creating {split}"):
                # 随机选择 num_speakers 条音频
                selected_indices = random.choices(
                    range(len(valid_audios)),
                    k=self.num_speakers
                )
                selected_audios = [valid_audios[idx] for idx in selected_indices]

                # 混合音频
                mixed_audio, metadata, total_duration = self.mix_audios(selected_audios)

                # 保存
                record = self.save_mixture(
                    mixed_audio,
                    selected_audios,
                    split,
                    global_idx,
                    metadata,
                    total_duration
                )

                # 写入CSV
                csv_writer.writerow(record)

                global_idx += 1

        csv_file.close()

        print("\n" + "=" * 60)
        print("✓ Data preparation completed!")
        print("=" * 60)
        print(f"Output directory: {self.output_dir}")
        print(f"Metadata CSV: {csv_path}")
        print(f"Total samples: {global_idx}")
        print(f"  Train: {num_train}")
        print(f"  Val: {num_val}")
        print(f"  Test: {num_test}")
        print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Prepare LibreSpeech data for speech separation training'
    )

    # 路径参数
    parser.add_argument(
        '--input-dir',
        type=str,
        default='dataset/origin',
        help='输入目录（包含 LibreSpeech flac 文件）'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='dataset/prepared',
        help='输出目录'
    )

    # 音频参数
    parser.add_argument(
        '--sample-rate',
        type=int,
        default=16000,
        choices=[8000, 16000],
        help='目标采样率（Hz）'
    )
    parser.add_argument(
        '--num-speakers',
        type=int,
        default=2,
        help='混合的说话人数量'
    )

    # 数据集参数
    parser.add_argument(
        '--num-samples',
        type=int,
        default=10000,
        help='生成的样本总数'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.8,
        help='训练集比例'
    )
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.1,
        help='验证集比例'
    )
    parser.add_argument(
        '--test-ratio',
        type=float,
        default=0.1,
        help='测试集比例'
    )

    # 音频时长限制
    parser.add_argument(
        '--min-duration',
        type=float,
        default=3.0,
        help='最小音频时长（秒）'
    )
    parser.add_argument(
        '--max-duration',
        type=float,
        default=10.0,
        help='最大音频时长（秒）'
    )

    # 其他参数
    parser.add_argument(
        '--output-csv',
        type=str,
        default='metadata.csv',
        help='输出CSV文件名'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子'
    )

    args = parser.parse_args()

    # 检查比例总和
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 0.01:
        raise ValueError(f"比例总和必须为1.0，当前为 {total_ratio}")

    # 创建数据准备器
    preparator = LibreSpeechDataPreparator(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        sample_rate=args.sample_rate,
        num_speakers=args.num_speakers,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        seed=args.seed
    )

    # 准备数据集
    preparator.prepare_dataset(
        num_samples=args.num_samples,
        output_csv=args.output_csv
    )


if __name__ == '__main__':
    main()
