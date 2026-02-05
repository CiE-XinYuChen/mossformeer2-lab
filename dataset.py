"""
Dataset loaders for speech separation
Supports: WSJ0-2mix, WSJ0-3mix, Libri2Mix, WHAM!, WHAMR!
Includes dynamic mixing as used in the MossFormer2 paper
Also supports CSV-based datasets (e.g., prepared LibreSpeech)
"""

import os
import csv
import torch
import torchaudio
import random
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader


class SeparationDataset(Dataset):
    """
    Generic speech separation dataset
    Supports both pre-mixed and dynamic mixing
    """

    def __init__(
        self,
        data_dir,
        split='train',
        sample_rate=8000,
        segment_length=4.0,
        num_spks=2,
        dynamic_mixing=False
    ):
        """
        Args:
            data_dir: Root directory of the dataset
            split: 'train', 'val', or 'test'
            sample_rate: Sampling rate (8000 or 16000)
            segment_length: Length of audio segments in seconds
            num_spks: Number of speakers (2 or 3)
            dynamic_mixing: If True, mix sources on-the-fly during training
        """
        self.data_dir = data_dir
        self.split = split
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.segment_samples = int(segment_length * sample_rate)
        self.num_spks = num_spks
        self.dynamic_mixing = dynamic_mixing and split == 'train'

        # Load file lists
        self.samples = self._load_samples()

    def _load_samples(self):
        """
        Load sample file paths
        Expected directory structure:
        data_dir/
            train/
                mix/
                    mix_1.wav, mix_2.wav, ...
                s1/
                    s1_1.wav, s1_2.wav, ...
                s2/
                    s2_1.wav, s2_2.wav, ...
        """
        samples = []
        mix_dir = os.path.join(self.data_dir, self.split, 'mix')

        if not os.path.exists(mix_dir):
            raise ValueError(f"Data directory not found: {mix_dir}")

        # Get all mixture files
        mix_files = sorted([f for f in os.listdir(mix_dir) if f.endswith('.wav')])

        for mix_file in mix_files:
            sample = {
                'mix': os.path.join(mix_dir, mix_file),
                'sources': []
            }

            # Get corresponding source files
            for spk_idx in range(1, self.num_spks + 1):
                src_file = mix_file.replace('mix', f's{spk_idx}')
                src_path = os.path.join(self.data_dir, self.split, f's{spk_idx}', src_file)

                if os.path.exists(src_path):
                    sample['sources'].append(src_path)
                else:
                    # Alternative naming convention
                    src_path = os.path.join(self.data_dir, self.split, f's{spk_idx}', mix_file)
                    if os.path.exists(src_path):
                        sample['sources'].append(src_path)

            # Only add if all sources are found
            if len(sample['sources']) == self.num_spks:
                samples.append(sample)

        if len(samples) == 0:
            raise ValueError(f"No valid samples found in {self.data_dir}/{self.split}")

        print(f"Loaded {len(samples)} samples for {self.split} split")
        return samples

    def __len__(self):
        return len(self.samples)

    def _load_audio(self, path):
        """Load and preprocess audio file"""
        waveform, sr = torchaudio.load(path)

        # Resample if necessary
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        # Convert to mono if necessary
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        return waveform.squeeze(0)  # [time]

    def _random_segment(self, audio):
        """Extract random segment from audio"""
        audio_len = audio.shape[-1]

        if audio_len >= self.segment_samples:
            # Random crop
            start = random.randint(0, audio_len - self.segment_samples)
            return audio[start:start + self.segment_samples]
        else:
            # Pad if too short
            padding = self.segment_samples - audio_len
            return torch.nn.functional.pad(audio, (0, padding))

    def __getitem__(self, idx):
        """
        Returns:
            mixture: Mixed audio [time]
            sources: List of source audios, each [time]
        """
        sample = self.samples[idx]

        # Load sources
        sources = [self._load_audio(src_path) for src_path in sample['sources']]

        if self.dynamic_mixing:
            # Dynamic mixing: Create new mixture from sources
            # Optionally randomize gains (not mentioned in paper, but common practice)
            mixture = sum(sources)
        else:
            # Load pre-mixed audio
            mixture = self._load_audio(sample['mix'])

        # Extract segments for training
        if self.split == 'train':
            # Find common length
            min_len = min([s.shape[0] for s in sources] + [mixture.shape[0]])

            # Truncate all to same length
            mixture = mixture[:min_len]
            sources = [s[:min_len] for s in sources]

            # Random segment
            start = random.randint(0, max(0, min_len - self.segment_samples))
            end = start + self.segment_samples

            if end > min_len:
                # Pad if necessary
                pad_len = end - min_len
                mixture = torch.nn.functional.pad(mixture, (0, pad_len))
                sources = [torch.nn.functional.pad(s, (0, pad_len)) for s in sources]

            mixture = mixture[start:end]
            sources = [s[start:end] for s in sources]
        else:
            # For validation/test, use full audio (or pad to minimum length)
            min_len = min([s.shape[0] for s in sources] + [mixture.shape[0]])
            mixture = mixture[:min_len]
            sources = [s[:min_len] for s in sources]

        # Normalize (optional - can help training stability)
        # max_amp = max([s.abs().max() for s in sources] + [mixture.abs().max()])
        # if max_amp > 0:
        #     mixture = mixture / max_amp
        #     sources = [s / max_amp for s in sources]

        return {
            'mixture': mixture,  # [time]
            'sources': sources,  # List of [time]
        }


def collate_fn(batch):
    """
    Custom collate function to handle variable-length sequences
    Pads to the longest sequence in the batch
    """
    mixtures = [item['mixture'] for item in batch]
    sources_list = [item['sources'] for item in batch]

    # Find max length in batch
    max_len = max([m.shape[0] for m in mixtures])

    # Pad mixtures
    mixtures_padded = []
    for mix in mixtures:
        if mix.shape[0] < max_len:
            padding = max_len - mix.shape[0]
            mix = torch.nn.functional.pad(mix, (0, padding))
        mixtures_padded.append(mix)

    # Pad sources
    num_sources = len(sources_list[0])
    sources_padded = [[] for _ in range(num_sources)]

    for sources in sources_list:
        for i, src in enumerate(sources):
            if src.shape[0] < max_len:
                padding = max_len - src.shape[0]
                src = torch.nn.functional.pad(src, (0, padding))
            sources_padded[i].append(src)

    # Stack into tensors
    mixtures_batch = torch.stack(mixtures_padded)  # [batch, time]
    sources_batch = [torch.stack(sources_padded[i]) for i in range(num_sources)]

    return {
        'mixture': mixtures_batch,
        'sources': sources_batch,
    }


def create_dataloaders(config):
    """
    Create train, validation, and test dataloaders

    Args:
        config: Configuration dictionary or namespace

    Returns:
        train_loader, valid_loader, test_loader
    """
    # Training set
    train_dataset = SeparationDataset(
        data_dir=config['data_folder'],
        split='train',
        sample_rate=config['sample_rate'],
        segment_length=config['segment_length'],
        num_spks=config['num_spks'],
        dynamic_mixing=config.get('use_dynamic_mixing', False)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config.get('num_workers', 4),
        collate_fn=collate_fn,
        drop_last=True,
        pin_memory=True
    )

    # Validation set
    valid_dataset = SeparationDataset(
        data_dir=config['data_folder'],
        split='val',
        sample_rate=config['sample_rate'],
        segment_length=config['segment_length'],
        num_spks=config['num_spks'],
        dynamic_mixing=False  # No dynamic mixing for validation
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        collate_fn=collate_fn,
        pin_memory=True
    )

    # Test set
    test_dataset = SeparationDataset(
        data_dir=config['data_folder'],
        split='test',
        sample_rate=config['sample_rate'],
        segment_length=config['segment_length'],
        num_spks=config['num_spks'],
        dynamic_mixing=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        collate_fn=collate_fn,
        pin_memory=True
    )

    return train_loader, valid_loader, test_loader


class CSVSeparationDataset(Dataset):
    """
    CSV-based speech separation dataset
    Loads data from a CSV metadata file created by prepare_librespeech_data.py

    CSV format:
    mix_path, total_duration, s1_path, s1_duration, s1_insert_time, s2_path, s2_duration, s2_insert_time, ...
    """

    def __init__(
        self,
        data_root,
        csv_file,
        split='train',
        sample_rate=16000,
        segment_length=4.0,
        num_spks=2
    ):
        """
        Args:
            data_root: Root directory of the prepared dataset
            csv_file: CSV metadata file path (relative to data_root or absolute)
            split: 'train', 'val', or 'test'
            sample_rate: Sampling rate (should match the prepared data)
            segment_length: Length of audio segments in seconds
            num_spks: Number of speakers
        """
        self.data_root = Path(data_root)
        self.split = split
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.segment_samples = int(segment_length * sample_rate)
        self.num_spks = num_spks

        # Load CSV metadata
        csv_path = csv_file if os.path.isabs(csv_file) else self.data_root / csv_file
        self.samples = self._load_csv(csv_path)

        print(f"Loaded {len(self.samples)} samples for {split} split from CSV")

    def _load_csv(self, csv_path):
        """Load samples from CSV file"""
        samples = []

        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            for row in reader:
                # Filter by split
                if not row['mix_path'].startswith(f"{self.split}/"):
                    continue

                # Parse the row
                sample = {
                    'mix_path': self.data_root / row['mix_path'],
                    'total_duration': float(row['total_duration']),
                    'sources': []
                }

                # Load source information
                for i in range(1, self.num_spks + 1):
                    source_info = {
                        'path': self.data_root / row[f's{i}_path'],
                        'duration': float(row[f's{i}_duration']),
                        'insert_time': float(row[f's{i}_insert_time'])
                    }
                    sample['sources'].append(source_info)

                # Verify files exist
                if sample['mix_path'].exists():
                    all_sources_exist = all(
                        src['path'].exists() for src in sample['sources']
                    )
                    if all_sources_exist:
                        samples.append(sample)

        if len(samples) == 0:
            raise ValueError(f"No valid samples found for split '{self.split}' in {csv_path}")

        return samples

    def __len__(self):
        return len(self.samples)

    def _load_audio(self, path):
        """Load audio file"""
        waveform, sr = torchaudio.load(str(path))

        # Convert to mono if necessary
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample if necessary
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        return waveform.squeeze(0)  # [time]

    def _random_segment(self, audio):
        """Extract random segment from audio"""
        audio_len = audio.shape[-1]

        if audio_len >= self.segment_samples:
            # Random crop
            start = random.randint(0, audio_len - self.segment_samples)
            return audio[start:start + self.segment_samples]
        else:
            # Pad if too short
            padding = self.segment_samples - audio_len
            return torch.nn.functional.pad(audio, (0, padding))

    def __getitem__(self, idx):
        """
        Returns:
            mixture: Mixed audio [time]
            sources: List of source audios, each [time]
        """
        sample = self.samples[idx]

        # Load mixture
        mixture = self._load_audio(sample['mix_path'])

        # Load sources
        sources = [
            self._load_audio(src['path'])
            for src in sample['sources']
        ]

        # Extract segments for training
        if self.split == 'train':
            # Find common length
            min_len = min([s.shape[0] for s in sources] + [mixture.shape[0]])

            # Random segment
            if min_len > self.segment_samples:
                start = random.randint(0, min_len - self.segment_samples)
                end = start + self.segment_samples

                mixture = mixture[start:end]
                sources = [s[start:end] for s in sources]
            else:
                # Pad if too short
                if mixture.shape[0] < self.segment_samples:
                    pad_len = self.segment_samples - mixture.shape[0]
                    mixture = torch.nn.functional.pad(mixture, (0, pad_len))
                    sources = [
                        torch.nn.functional.pad(s, (0, pad_len))
                        for s in sources
                    ]
        else:
            # For validation/test, use full audio
            min_len = min([s.shape[0] for s in sources] + [mixture.shape[0]])
            mixture = mixture[:min_len]
            sources = [s[:min_len] for s in sources]

        return {
            'mixture': mixture,  # [time]
            'sources': sources,  # List of [time]
        }


def create_csv_dataloaders(config):
    """
    Create train, validation, and test dataloaders for CSV-based dataset

    Args:
        config: Configuration dictionary with:
            - data_root: Root directory of prepared dataset
            - csv_file: CSV metadata file
            - sample_rate: Sampling rate
            - segment_length: Segment length in seconds
            - num_spks: Number of speakers
            - batch_size: Batch size
            - num_workers: Number of data loading workers

    Returns:
        train_loader, valid_loader, test_loader
    """
    # Training set
    train_dataset = CSVSeparationDataset(
        data_root=config['data_root'],
        csv_file=config['csv_file'],
        split='train',
        sample_rate=config['sample_rate'],
        segment_length=config['segment_length'],
        num_spks=config['num_spks']
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config.get('num_workers', 4),
        collate_fn=collate_fn,
        drop_last=True,
        pin_memory=True
    )

    # Validation set
    valid_dataset = CSVSeparationDataset(
        data_root=config['data_root'],
        csv_file=config['csv_file'],
        split='val',
        sample_rate=config['sample_rate'],
        segment_length=config['segment_length'],
        num_spks=config['num_spks']
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        collate_fn=collate_fn,
        pin_memory=True
    )

    # Test set
    test_dataset = CSVSeparationDataset(
        data_root=config['data_root'],
        csv_file=config['csv_file'],
        split='test',
        sample_rate=config['sample_rate'],
        segment_length=config['segment_length'],
        num_spks=config['num_spks']
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        collate_fn=collate_fn,
        pin_memory=True
    )

    return train_loader, valid_loader, test_loader


if __name__ == "__main__":
    # Test dataset loading
    print("Testing dataset loader...")

    config = {
        'data_folder': '/path/to/wsj0-2mix',
        'sample_rate': 8000,
        'segment_length': 4.0,
        'num_spks': 2,
        'batch_size': 2,
        'num_workers': 0,
        'use_dynamic_mixing': True,
    }

    try:
        train_loader, valid_loader, test_loader = create_dataloaders(config)
        print(f"✓ Train batches: {len(train_loader)}")
        print(f"✓ Valid batches: {len(valid_loader)}")
        print(f"✓ Test batches: {len(test_loader)}")

        # Test one batch
        batch = next(iter(train_loader))
        print(f"\nBatch shapes:")
        print(f"  Mixture: {batch['mixture'].shape}")
        print(f"  Sources: {[s.shape for s in batch['sources']]}")

    except Exception as e:
        print(f"✗ Error: {e}")
        print("  (This is expected if data path doesn't exist)")
