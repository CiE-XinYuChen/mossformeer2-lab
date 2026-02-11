"""
Training script for MossFormer2
Replicates the training setup from the paper (Section 3.2)
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import yaml
from argparse import Namespace

# Import model and utilities
from mossformer2 import MossFormer2_SS_16K
from loss import MossFormer2Loss, si_sdr_improvement
from dataset import create_dataloaders, create_csv_dataloaders


def setup_distributed():
    """Initialize distributed training environment.
    Environment variables RANK, LOCAL_RANK, WORLD_SIZE are set by torchrun.
    Returns (rank, local_rank, world_size). Returns (-1, -1, 1) for single-GPU.
    """
    if 'RANK' not in os.environ or 'WORLD_SIZE' not in os.environ:
        return -1, -1, 1

    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')
    dist.barrier()

    return rank, local_rank, world_size


def cleanup_distributed():
    """Clean up distributed training environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


class MossFormer2Trainer:
    """
    Trainer class for MossFormer2
    Implements training procedure from paper Section 3.2
    """

    def __init__(self, config_path, local_rank=-1):
        """
        Args:
            config_path: Path to YAML configuration file
            local_rank: Local rank for distributed training (-1 for single-GPU)
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Distributed training setup
        self.local_rank = local_rank
        self.is_distributed = local_rank != -1
        self.rank = dist.get_rank() if self.is_distributed else 0
        self.world_size = dist.get_world_size() if self.is_distributed else 1

        # Set random seed (different per rank for data diversity)
        seed = self.config['seed'] + self.rank
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        # Setup device
        if self.is_distributed:
            self.device = torch.device(f'cuda:{local_rank}')
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if self.rank == 0:
            print(f"Using device: {self.device}")
            if self.is_distributed:
                print(f"Distributed training: {self.world_size} GPUs")

        # Create output directories (only rank 0)
        if self.rank == 0:
            os.makedirs(self.config['output_folder'], exist_ok=True)
            os.makedirs(self.config['save_folder'], exist_ok=True)
        if self.is_distributed:
            dist.barrier()

        # Initialize model
        self.model = self._create_model()
        self.model = self.model.to(self.device)

        # Print model info
        if self.rank == 0:
            num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"Model parameters: {num_params / 1e6:.1f}M")

        # Wrap model with DDP
        if self.is_distributed:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False
            )

        # Create data loaders
        if self.rank == 0:
            print("Creating data loaders...")
        dist_kwargs = dict(
            is_distributed=self.is_distributed,
            world_size=self.world_size,
            rank=self.rank
        )
        dataset_type = self.config.get('dataset_type', 'standard')

        if dataset_type == 'csv':
            self.train_loader, self.valid_loader, self.test_loader = \
                create_csv_dataloaders(self.config, **dist_kwargs)
        else:
            self.train_loader, self.valid_loader, self.test_loader = \
                create_dataloaders(self.config, **dist_kwargs)

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

        # Tensorboard logger and log file (only rank 0)
        self.writer = None
        self.log_file = None
        if self.rank == 0:
            self.writer = SummaryWriter(log_dir=os.path.join(self.config['output_folder'], 'logs'))
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

        # Set epoch for DistributedSampler
        if self.is_distributed:
            self.train_loader.sampler.set_epoch(self.current_epoch)

        # Progress bar (only rank 0)
        if self.rank == 0:
            pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch + 1}/{self.config["N_epochs"]}')
        else:
            pbar = self.train_loader

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
            if self.rank == 0:
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'lr': f'{current_lr:.2e}'
                })

        # Average loss
        avg_loss = epoch_loss / num_batches

        # Log to tensorboard (only rank 0)
        if self.rank == 0 and self.writer is not None:
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

        if self.rank == 0:
            pbar = tqdm(self.valid_loader, desc='Validation')
        else:
            pbar = self.valid_loader

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

            if self.rank == 0:
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # Synchronize metrics across all processes
        if self.is_distributed:
            metrics = torch.tensor(
                [val_loss, val_si_sdri, num_batches],
                dtype=torch.float32, device=self.device
            )
            dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
            val_loss = metrics[0].item()
            val_si_sdri = metrics[1].item()
            num_batches = int(metrics[2].item())

        # Average metrics
        avg_loss = val_loss / num_batches
        avg_si_sdri = val_si_sdri / num_batches

        # Log to tensorboard (only rank 0)
        if self.rank == 0 and self.writer is not None:
            self.writer.add_scalar('Val/Loss', avg_loss, self.current_epoch)
            self.writer.add_scalar('Val/SI-SDRi', avg_si_sdri, self.current_epoch)

        return avg_loss, avg_si_sdri

    def save_checkpoint(self, is_best=False):
        """Save model checkpoint (only rank 0)"""
        if self.rank != 0:
            return

        # Get raw model state (unwrap DDP if needed)
        model_state = self.model.module.state_dict() if self.is_distributed \
            else self.model.state_dict()

        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': model_state,
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
            print(f"Saved best checkpoint (epoch {self.current_epoch + 1})")

    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            # Load into raw model (unwrap DDP if needed)
            if self.is_distributed:
                self.model.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.current_epoch = checkpoint['epoch'] + 1
            self.best_val_loss = checkpoint['best_val_loss']
            if self.rank == 0:
                print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
            return True
        return False

    def train(self):
        """Main training loop"""
        if self.rank == 0:
            print(f"\nStarting training for {self.config['N_epochs']} epochs...")
            print(f"Initial LR: {self.config['lr']:.2e}")
            print(f"LR decay after epoch: {self.config['lr_decay_epoch']}")
            print(f"Gradient clip: {self.config['gradient_clip']}")
            if self.is_distributed:
                print(f"World size: {self.world_size}")
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

            # Log results (only rank 0)
            if self.rank == 0:
                log_msg = (
                    f"Epoch {epoch + 1}/{self.config['N_epochs']} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Val SI-SDRi: {val_si_sdri:.2f} dB | "
                    f"LR: {self._get_lr():.2e}"
                )
                print(log_msg)
                if self.log_file is not None:
                    self.log_file.write(log_msg + '\n')
                    self.log_file.flush()

            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss

            self.save_checkpoint(is_best=is_best)

            # Synchronize before next epoch
            if self.is_distributed:
                dist.barrier()

        if self.rank == 0:
            print("\nTraining completed!")
            print(f"Best validation loss: {self.best_val_loss:.4f}")
            if self.log_file is not None:
                self.log_file.close()
            if self.writer is not None:
                self.writer.close()


def main():
    parser = argparse.ArgumentParser(description='Train MossFormer2 for speech separation')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/train_mossformer2.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--distributed',
        action='store_true',
        help='Enable distributed training (use with torchrun)'
    )
    args = parser.parse_args()

    # Check if config exists
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)

    # Setup distributed training
    local_rank = -1
    if args.distributed:
        rank, local_rank, world_size = setup_distributed()
        if rank == 0:
            print(f"Distributed training initialized: {world_size} processes")
    else:
        rank = 0

    # Create trainer and start training
    trainer = MossFormer2Trainer(args.config, local_rank=local_rank)
    try:
        trainer.train()
    finally:
        cleanup_distributed()


if __name__ == '__main__':
    main()
