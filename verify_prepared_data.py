"""
验证准备好的 LibreSpeech 数据集
检查CSV格式、音频文件完整性、采样率等
"""

import os
import csv
import argparse
from pathlib import Path
import soundfile as sf
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


class DatasetVerifier:
    """数据集验证器"""

    def __init__(self, data_root, csv_file):
        self.data_root = Path(data_root)
        self.csv_file = csv_file if os.path.isabs(csv_file) else self.data_root / csv_file

    def verify_csv_format(self):
        """验证CSV格式"""
        print("\n" + "="*60)
        print("Step 1: Verifying CSV format...")
        print("="*60)

        if not self.csv_file.exists():
            print(f"✗ CSV file not found: {self.csv_file}")
            return False

        with open(self.csv_file, 'r') as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames

            print(f"✓ CSV file found: {self.csv_file}")
            print(f"  Headers: {headers}")

            # Check required headers
            required_headers = ['mix_path', 'total_duration', 's1_path', 's1_duration', 's1_insert_time']
            missing = [h for h in required_headers if h not in headers]

            if missing:
                print(f"✗ Missing headers: {missing}")
                return False

            print("✓ All required headers present")

            # Count rows
            row_count = sum(1 for _ in reader)
            print(f"✓ Total rows: {row_count}")

        return True

    def verify_files(self, max_samples=100):
        """验证音频文件存在性"""
        print("\n" + "="*60)
        print(f"Step 2: Verifying audio files (checking up to {max_samples} samples)...")
        print("="*60)

        stats = {
            'train': {'total': 0, 'missing_mix': 0, 'missing_sources': 0},
            'val': {'total': 0, 'missing_mix': 0, 'missing_sources': 0},
            'test': {'total': 0, 'missing_mix': 0, 'missing_sources': 0}
        }

        with open(self.csv_file, 'r') as f:
            reader = csv.DictReader(f)

            for i, row in enumerate(tqdm(reader, desc="Checking files", total=max_samples)):
                if i >= max_samples:
                    break

                # Determine split
                split = row['mix_path'].split('/')[0]
                stats[split]['total'] += 1

                # Check mix file
                mix_path = self.data_root / row['mix_path']
                if not mix_path.exists():
                    stats[split]['missing_mix'] += 1

                # Check source files
                num_sources = sum(1 for key in row.keys() if key.startswith('s') and key.endswith('_path'))
                for i in range(1, num_sources + 1):
                    src_path = self.data_root / row[f's{i}_path']
                    if not src_path.exists():
                        stats[split]['missing_sources'] += 1

        # Print statistics
        for split, stat in stats.items():
            if stat['total'] > 0:
                print(f"\n{split.upper()} split:")
                print(f"  Total samples checked: {stat['total']}")
                print(f"  Missing mix files: {stat['missing_mix']}")
                print(f"  Missing source files: {stat['missing_sources']}")

                if stat['missing_mix'] == 0 and stat['missing_sources'] == 0:
                    print(f"  ✓ All files present!")
                else:
                    print(f"  ✗ Some files are missing!")

        return all(stat['missing_mix'] == 0 and stat['missing_sources'] == 0
                   for stat in stats.values() if stat['total'] > 0)

    def verify_audio_properties(self, num_samples=10):
        """验证音频属性（采样率、时长等）"""
        print("\n" + "="*60)
        print(f"Step 3: Verifying audio properties ({num_samples} samples)...")
        print("="*60)

        durations = []
        sample_rates = []
        issues = []

        with open(self.csv_file, 'r') as f:
            reader = csv.DictReader(f)

            for i, row in enumerate(reader):
                if i >= num_samples:
                    break

                try:
                    # Load mix file
                    mix_path = self.data_root / row['mix_path']
                    mix_audio, mix_sr = sf.read(mix_path)

                    sample_rates.append(mix_sr)
                    durations.append(len(mix_audio) / mix_sr)

                    # Check if duration matches CSV
                    csv_duration = float(row['total_duration'])
                    actual_duration = len(mix_audio) / mix_sr

                    if abs(csv_duration - actual_duration) > 0.1:  # 100ms tolerance
                        issues.append(f"Duration mismatch in {mix_path.name}: "
                                      f"CSV={csv_duration:.2f}s, Actual={actual_duration:.2f}s")

                    # Check source files
                    num_sources = sum(1 for key in row.keys() if key.startswith('s') and key.endswith('_path'))
                    for j in range(1, num_sources + 1):
                        src_path = self.data_root / row[f's{j}_path']
                        src_audio, src_sr = sf.read(src_path)

                        if src_sr != mix_sr:
                            issues.append(f"Sample rate mismatch in {src_path.name}: "
                                          f"Mix={mix_sr}Hz, Source={src_sr}Hz")

                except Exception as e:
                    issues.append(f"Error reading {row['mix_path']}: {e}")

        # Print statistics
        if sample_rates:
            print(f"\nSample rate(s): {set(sample_rates)}")
            print(f"Duration range: {min(durations):.2f}s - {max(durations):.2f}s")
            print(f"Average duration: {np.mean(durations):.2f}s")

        if issues:
            print(f"\n✗ Found {len(issues)} issues:")
            for issue in issues[:10]:  # Show first 10 issues
                print(f"  - {issue}")
            if len(issues) > 10:
                print(f"  ... and {len(issues) - 10} more")
            return False
        else:
            print("\n✓ All audio properties are correct!")
            return True

    def visualize_mixture(self, sample_idx=0, output_dir='verification_plots'):
        """可视化混合结果"""
        print("\n" + "="*60)
        print(f"Step 4: Visualizing sample {sample_idx}...")
        print("="*60)

        os.makedirs(output_dir, exist_ok=True)

        with open(self.csv_file, 'r') as f:
            reader = csv.DictReader(f)

            for i, row in enumerate(reader):
                if i != sample_idx:
                    continue

                # Load audio files
                mix_path = self.data_root / row['mix_path']
                mix_audio, sr = sf.read(mix_path)

                num_sources = sum(1 for key in row.keys() if key.startswith('s') and key.endswith('_path'))
                sources = []
                for j in range(1, num_sources + 1):
                    src_path = self.data_root / row[f's{j}_path']
                    src_audio, _ = sf.read(src_path)
                    sources.append(src_audio)

                # Plot
                fig, axes = plt.subplots(num_sources + 1, 1, figsize=(15, 3 * (num_sources + 1)))

                time_axis = np.arange(len(mix_audio)) / sr

                # Plot mixture
                axes[0].plot(time_axis, mix_audio)
                axes[0].set_title(f"Mixture (Total duration: {len(mix_audio)/sr:.2f}s)")
                axes[0].set_ylabel("Amplitude")
                axes[0].grid(True)

                # Plot sources
                for j, src_audio in enumerate(sources):
                    axes[j + 1].plot(time_axis[:len(src_audio)], src_audio)
                    insert_time = float(row[f's{j+1}_insert_time'])
                    duration = float(row[f's{j+1}_duration'])
                    axes[j + 1].axvline(x=insert_time, color='r', linestyle='--', label=f'Insert time: {insert_time:.2f}s')
                    axes[j + 1].set_title(f"Source {j+1} (Duration: {duration:.2f}s, Insert at: {insert_time:.2f}s)")
                    axes[j + 1].set_ylabel("Amplitude")
                    axes[j + 1].grid(True)
                    axes[j + 1].legend()

                axes[-1].set_xlabel("Time (s)")

                plt.tight_layout()
                plot_path = os.path.join(output_dir, f'sample_{sample_idx}.png')
                plt.savefig(plot_path, dpi=150)
                print(f"✓ Plot saved to: {plot_path}")
                plt.close()

                # Print metadata
                print(f"\nMetadata for sample {sample_idx}:")
                print(f"  Mix file: {row['mix_path']}")
                print(f"  Total duration: {row['total_duration']}s")
                for j in range(1, num_sources + 1):
                    print(f"  Source {j}:")
                    print(f"    Path: {row[f's{j}_path']}")
                    print(f"    Duration: {row[f's{j}_duration']}s")
                    print(f"    Insert time: {row[f's{j}_insert_time']}s")

                break

    def run_full_verification(self, max_file_check=100, num_audio_check=10, sample_idx=0):
        """运行完整验证"""
        print("\n" + "="*60)
        print("LibreSpeech Dataset Verification")
        print("="*60)
        print(f"Data root: {self.data_root}")
        print(f"CSV file: {self.csv_file}")
        print("="*60)

        results = {}

        # 1. CSV format
        results['csv_format'] = self.verify_csv_format()

        # 2. File existence
        results['files'] = self.verify_files(max_samples=max_file_check)

        # 3. Audio properties
        results['audio'] = self.verify_audio_properties(num_samples=num_audio_check)

        # 4. Visualization
        try:
            self.visualize_mixture(sample_idx=sample_idx)
            results['visualization'] = True
        except Exception as e:
            print(f"✗ Visualization failed: {e}")
            results['visualization'] = False

        # Summary
        print("\n" + "="*60)
        print("Verification Summary")
        print("="*60)
        for check, passed in results.items():
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"  {check.upper()}: {status}")

        if all(results.values()):
            print("\n✓ All checks passed! Dataset is ready for training.")
        else:
            print("\n✗ Some checks failed. Please fix the issues before training.")

        return all(results.values())


def main():
    parser = argparse.ArgumentParser(description='Verify prepared LibreSpeech dataset')

    parser.add_argument(
        '--data-root',
        type=str,
        default='dataset/prepared',
        help='Root directory of the prepared dataset'
    )
    parser.add_argument(
        '--csv-file',
        type=str,
        default='metadata.csv',
        help='CSV metadata file'
    )
    parser.add_argument(
        '--max-file-check',
        type=int,
        default=100,
        help='Maximum number of files to check for existence'
    )
    parser.add_argument(
        '--num-audio-check',
        type=int,
        default=10,
        help='Number of audio files to check properties'
    )
    parser.add_argument(
        '--sample-idx',
        type=int,
        default=0,
        help='Sample index to visualize'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='verification_plots',
        help='Output directory for plots'
    )

    args = parser.parse_args()

    verifier = DatasetVerifier(args.data_root, args.csv_file)
    success = verifier.run_full_verification(
        max_file_check=args.max_file_check,
        num_audio_check=args.num_audio_check,
        sample_idx=args.sample_idx
    )

    import sys
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
