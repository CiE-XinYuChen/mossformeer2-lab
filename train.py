"""
Training script for MossFormer2
Replicates the training setup from the paper (Section 3.2)
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import yaml
from argparse import Namespace

# Import model and utilities
from mossformer2 import MossFormer2_SS_16K
from loss import MossFormer2Loss, si_sdr_improvement
from dataset import create_dataloaders, create_csv_dataloaders


class MossFormer2Trainer:
    """
    Trainer class for MossFormer2
    Implements training procedure from paper Section 3.2
    """

    def __init__(self, config_path):
        """
        Args:
            config_path: Path to YAML configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Set random seed
        torch.manual_seed(self.config['seed'])
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config['seed'])

        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Create output directories
        os.makedirs(self.config['output_folder'], exist_ok=True)
        os.makedirs(self.config['save_folder'], exist_ok=True)

        # Initialize model
        self.model = self._create_model()
        self.model = self.model.to(self.device)

        # Print model info
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model parameters: {num_params / 1e6:.1f}M")

        # Create data loaders
        print("Creating data loaders...")
        dataset_type = self.config.get('dataset_type', 'standard')  # 'standard' or 'csv'

        if dataset_type == 'csv':
            # CSV-based dataset (e.g., prepared LibreSpeech)
            self.train_loader, self.valid_loader, self.test_loader = create_csv_dataloaders(self.config)
        else:
            # Standard dataset (WSJ0-2mix, etc.)
            self.train_loader, self.valid_loader, self.test_loader = create_dataloaders(self.config)

        # Initialize loss function
        self.criterion = MossFormer2Loss(num_spks=self.config['num_spks'])

        # Initialize optimizer (Adam with lr=15e-5 from paper)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config['lr'],
            weight_decay=0  # Paper doesn't mention weight decay
        )

        # Learning rate scheduler (constant for 85 epochs, then decay by 0.5)
        self.current_epoch = 0
        self.best_val_loss = float('inf')

        # Tensorboard logger
        self.writer = SummaryWriter(log_dir=os.path.join(self.config['output_folder'], 'logs'))

        # Training log file
        self.log_file = open(self.config['train_log'], 'w')

    def _create_model(self):
        """Create MossFormer2 model"""
        args = Namespace(
            encoder_embedding_dim=self.config['encoder_embedding_dim'],
            mossformer_sequence_dim=self.config['mossformer_sequence_dim'],
            num_mossformer_layer=self.config['num_mossformer_layer'],
            encoder_kernel_size=self.config['encoder_kernel_size'],
            num_spks=self.config['num_spks'],
        )
        return MossFormer2_SS_16K(args)

    def _get_lr(self):
        """
        Get current learning rate based on epoch
        Paper: constant for 85 epochs, then decay by 0.5
        """
        if self.current_epoch < self.config['lr_decay_epoch']:
            return self.config['lr']
        else:
            # Number of decay steps
            decay_steps = self.current_epoch - self.config['lr_decay_epoch']
            return self.config['lr'] * (self.config['lr_decay_factor'] ** decay_steps)

    def _update_lr(self):
        """Update learning rate for optimizer"""
        new_lr = self._get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        return new_lr

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0

        # Update learning rate
        current_lr = self._update_lr()

        # Progress bar
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch + 1}/{self.config["N_epochs"]}')

        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            mixture = batch['mixture'].to(self.device)  # [batch, time]
            sources = [s.to(self.device) for s in batch['sources']]  # List of [batch, time]

            # Forward pass
            self.optimizer.zero_grad()
            estimated_sources = self.model(mixture)  # List of [batch, time]

            # Compute loss (with PIT)
            loss, best_perm = self.criterion(estimated_sources, sources)

            # Backward pass
            loss.backward()

            # Gradient clipping (L2 norm = 5.0 from paper)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config['gradient_clip']
            )

            # Optimizer step
            self.optimizer.step()

            # Accumulate loss
            epoch_loss += loss.item()
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{current_lr:.2e}'
            })

        # Average loss
        avg_loss = epoch_loss / num_batches

        # Log to tensorboard
        self.writer.add_scalar('Train/Loss', avg_loss, self.current_epoch)
        self.writer.add_scalar('Train/LR', current_lr, self.current_epoch)

        return avg_loss

    @torch.no_grad()
    def validate(self):
        """Validate on validation set"""
        self.model.eval()
        val_loss = 0.0
        val_si_sdri = 0.0
        num_batches = 0

        pbar = tqdm(self.valid_loader, desc='Validation')

        for batch in pbar:
            mixture = batch['mixture'].to(self.device)
            sources = [s.to(self.device) for s in batch['sources']]

            # Forward pass
            estimated_sources = self.model(mixture)

            # Compute loss
            loss, _ = self.criterion(estimated_sources, sources)
            val_loss += loss.item()

            # Compute SI-SDRi for each speaker
            batch_si_sdri = 0.0
            for i in range(len(sources)):
                si_sdri = si_sdr_improvement(
                    estimated_sources[i],
                    sources[i],
                    mixture
                )
                batch_si_sdri += si_sdri.mean().item()

            val_si_sdri += batch_si_sdri / len(sources)
            num_batches += 1

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # Average metrics
        avg_loss = val_loss / num_batches
        avg_si_sdri = val_si_sdri / num_batches

        # Log to tensorboard
        self.writer.add_scalar('Val/Loss', avg_loss, self.current_epoch)
        self.writer.add_scalar('Val/SI-SDRi', avg_si_sdri, self.current_epoch)

        return avg_loss, avg_si_sdri

    def save_checkpoint(self, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
        }

        # Save latest checkpoint
        latest_path = os.path.join(self.config['save_folder'], 'latest_checkpoint.pt')
        torch.save(checkpoint, latest_path)

        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.config['save_folder'], 'best_checkpoint.pt')
            torch.save(checkpoint, best_path)
            print(f"✓ Saved best checkpoint (epoch {self.current_epoch + 1})")

    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.current_epoch = checkpoint['epoch'] + 1
            self.best_val_loss = checkpoint['best_val_loss']
            print(f"✓ Loaded checkpoint from epoch {checkpoint['epoch']}")
            return True
        return False

    def train(self):
        """Main training loop"""
        print(f"\nStarting training for {self.config['N_epochs']} epochs...")
        print(f"Initial LR: {self.config['lr']:.2e}")
        print(f"LR decay after epoch: {self.config['lr_decay_epoch']}")
        print(f"Gradient clip: {self.config['gradient_clip']}")
        print("-" * 60)

        # Try to resume from checkpoint
        checkpoint_path = os.path.join(self.config['save_folder'], 'latest_checkpoint.pt')
        self.load_checkpoint(checkpoint_path)

        for epoch in range(self.current_epoch, self.config['N_epochs']):
            self.current_epoch = epoch

            # Train one epoch
            train_loss = self.train_epoch()

            # Validate
            val_loss, val_si_sdri = self.validate()

            # Log results
            log_msg = (
                f"Epoch {epoch + 1}/{self.config['N_epochs']} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val SI-SDRi: {val_si_sdri:.2f} dB | "
                f"LR: {self._get_lr():.2e}"
            )
            print(log_msg)
            self.log_file.write(log_msg + '\n')
            self.log_file.flush()

            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss

            self.save_checkpoint(is_best=is_best)

        print("\n✓ Training completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        self.log_file.close()
        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description='Train MossFormer2 for speech separation')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/train_mossformer2.yaml',
        help='Path to configuration file'
    )
    args = parser.parse_args()

    # Check if config exists
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)

    # Create trainer and start training
    trainer = MossFormer2Trainer(args.config)
    trainer.train()


if __name__ == '__main__':
    main()
