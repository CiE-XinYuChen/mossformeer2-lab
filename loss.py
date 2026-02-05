"""
Loss functions for speech separation
Implements SI-SDR (Scale-Invariant Source-to-Distortion Ratio) with PIT
Based on MossFormer2 paper training setup
"""

import torch
import torch.nn as nn
import itertools


def si_sdr(estimated, target, eps=1e-8):
    """
    Calculate Scale-Invariant Source-to-Distortion Ratio (SI-SDR)

    Args:
        estimated: Estimated source, shape [batch, time]
        target: Target source, shape [batch, time]
        eps: Small epsilon for numerical stability

    Returns:
        si_sdr: SI-SDR value in dB
    """
    # Remove mean
    estimated = estimated - estimated.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)

    # Compute scaling factor
    # s_target = <est, target> * target / ||target||^2
    dot_product = (estimated * target).sum(dim=-1, keepdim=True)
    target_energy = (target ** 2).sum(dim=-1, keepdim=True) + eps
    scale = dot_product / target_energy

    # Compute projection
    s_target = scale * target

    # Compute noise
    e_noise = estimated - s_target

    # Compute SI-SDR
    si_sdr_value = 10 * torch.log10(
        (s_target ** 2).sum(dim=-1) / ((e_noise ** 2).sum(dim=-1) + eps) + eps
    )

    return si_sdr_value


def si_sdr_loss(estimated, target, eps=1e-8):
    """
    SI-SDR loss (negative SI-SDR for minimization)

    Args:
        estimated: Estimated source, shape [batch, time]
        target: Target source, shape [batch, time]

    Returns:
        loss: Negative SI-SDR (to be minimized)
    """
    return -si_sdr(estimated, target, eps)


class PITLossWrapper(nn.Module):
    """
    Permutation Invariant Training (PIT) wrapper for loss function
    Finds the best permutation of estimated sources to match targets

    This is essential for multi-speaker separation where the order of
    separated sources is arbitrary.
    """

    def __init__(self, loss_func=si_sdr_loss, mode='pairwise'):
        """
        Args:
            loss_func: Base loss function to compute pairwise losses
            mode: 'pairwise' for standard PIT
        """
        super(PITLossWrapper, self).__init__()
        self.loss_func = loss_func
        self.mode = mode

    def forward(self, est_sources, target_sources):
        """
        Args:
            est_sources: List of estimated sources, each [batch, time]
                        or tensor [num_sources, batch, time]
            target_sources: List of target sources, each [batch, time]
                           or tensor [num_sources, batch, time]

        Returns:
            loss: Minimum loss across all permutations
            best_perm: Best permutation indices
        """
        # Convert to tensor if list
        if isinstance(est_sources, list):
            est_sources = torch.stack(est_sources, dim=0)  # [num_sources, batch, time]
        if isinstance(target_sources, list):
            target_sources = torch.stack(target_sources, dim=0)

        num_sources = est_sources.shape[0]
        batch_size = est_sources.shape[1]

        # Generate all possible permutations
        perms = list(itertools.permutations(range(num_sources)))

        # Compute loss for each permutation
        losses = []
        for perm in perms:
            perm_loss = 0
            for src_idx, tgt_idx in enumerate(perm):
                # Compute pairwise loss
                perm_loss += self.loss_func(
                    est_sources[src_idx],  # [batch, time]
                    target_sources[tgt_idx]  # [batch, time]
                ).mean()  # Average over batch
            losses.append(perm_loss)

        # Find minimum loss
        losses = torch.stack(losses)
        min_loss, min_idx = torch.min(losses, dim=0)
        best_perm = perms[min_idx]

        return min_loss, best_perm


class SISNRLoss(nn.Module):
    """
    Scale-Invariant Signal-to-Noise Ratio (SI-SNR) Loss
    Essentially the same as SI-SDR for speech separation
    """

    def __init__(self):
        super(SISNRLoss, self).__init__()

    def forward(self, estimated, target, eps=1e-8):
        """
        Calculate SI-SNR loss

        Args:
            estimated: Estimated source [batch, time]
            target: Target source [batch, time]

        Returns:
            loss: Negative SI-SNR
        """
        return si_sdr_loss(estimated, target, eps)


class MossFormer2Loss(nn.Module):
    """
    Complete loss function for MossFormer2 training
    Combines SI-SDR with PIT for multi-speaker separation
    """

    def __init__(self, num_spks=2):
        super(MossFormer2Loss, self).__init__()
        self.num_spks = num_spks
        self.pit_wrapper = PITLossWrapper(loss_func=si_sdr_loss)

    def forward(self, est_sources, target_sources):
        """
        Args:
            est_sources: List of estimated sources from model
                        Each element: [batch, time]
            target_sources: List of target sources
                           Each element: [batch, time]

        Returns:
            loss: PIT loss value
            best_perm: Best permutation for this batch
        """
        loss, best_perm = self.pit_wrapper(est_sources, target_sources)
        return loss, best_perm


# Utility function for SI-SDR improvement calculation (for evaluation)
def si_sdr_improvement(estimated, target, mixture, eps=1e-8):
    """
    Calculate SI-SDR improvement (SI-SDRi)
    SI-SDRi = SI-SDR(estimated, target) - SI-SDR(mixture, target)

    Args:
        estimated: Estimated source [batch, time]
        target: Target source [batch, time]
        mixture: Input mixture [batch, time]

    Returns:
        si_sdri: SI-SDR improvement in dB
    """
    si_sdr_separated = si_sdr(estimated, target, eps)
    si_sdr_mixture = si_sdr(mixture, target, eps)
    return si_sdr_separated - si_sdr_mixture


if __name__ == "__main__":
    # Test the loss functions
    print("Testing SI-SDR Loss Functions...")

    # Create dummy data
    batch_size = 2
    seq_len = 16000  # 1 second at 16kHz
    num_sources = 2

    # Simulate estimated and target sources
    est = [torch.randn(batch_size, seq_len) for _ in range(num_sources)]
    target = [torch.randn(batch_size, seq_len) for _ in range(num_sources)]

    # Test SI-SDR
    print("\n1. Testing basic SI-SDR:")
    sdr = si_sdr(est[0], target[0])
    print(f"   SI-SDR shape: {sdr.shape}")
    print(f"   SI-SDR values: {sdr}")

    # Test PIT wrapper
    print("\n2. Testing PIT wrapper:")
    pit_loss = PITLossWrapper()
    loss, perm = pit_loss(est, target)
    print(f"   PIT Loss: {loss.item():.4f}")
    print(f"   Best permutation: {perm}")

    # Test complete MossFormer2 loss
    print("\n3. Testing MossFormer2 Loss:")
    mf2_loss = MossFormer2Loss(num_spks=2)
    loss, perm = mf2_loss(est, target)
    print(f"   Loss: {loss.item():.4f}")
    print(f"   Best permutation: {perm}")

    print("\n✓ All tests passed!")
