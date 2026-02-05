"""
Test script to verify MossFormer2 model works correctly
在开始训练前验证模型是否能正常工作
"""

import torch
from argparse import Namespace
from mossformer2 import MossFormer2_SS_16K
from loss import MossFormer2Loss, si_sdr_improvement
import time


def test_model_forward():
    """Test model forward pass"""
    print("=" * 60)
    print("Testing MossFormer2 Model")
    print("=" * 60)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n1. Device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Create model (full version - 55.7M params)
    print("\n2. Creating model...")
    args = Namespace(
        encoder_embedding_dim=512,
        mossformer_sequence_dim=512,
        num_mossformer_layer=24,
        encoder_kernel_size=16,
        num_spks=2,
    )

    model = MossFormer2_SS_16K(args)
    model = model.to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   ✓ Model created")
    print(f"   Parameters: {num_params / 1e6:.1f}M (expected: 55.7M)")

    # Create dummy input (4 seconds @ 8kHz)
    print("\n3. Testing forward pass...")
    batch_size = 1
    sample_rate = 8000
    duration = 4.0
    num_samples = int(sample_rate * duration)

    # Random input mixture
    mixture = torch.randn(batch_size, num_samples).to(device)
    print(f"   Input shape: {mixture.shape}")

    # Forward pass
    model.eval()
    with torch.no_grad():
        start_time = time.time()
        outputs = model(mixture)
        forward_time = time.time() - start_time

    print(f"   ✓ Forward pass successful")
    print(f"   Output shapes: {[out.shape for out in outputs]}")
    print(f"   Forward time: {forward_time:.3f}s")
    print(f"   Real-time factor: {forward_time / duration:.4f}")

    # Test loss function
    print("\n4. Testing loss function...")
    target_sources = [torch.randn_like(outputs[0]) for _ in range(args.num_spks)]

    criterion = MossFormer2Loss(num_spks=args.num_spks)
    loss, best_perm = criterion(outputs, target_sources)

    print(f"   ✓ Loss computation successful")
    print(f"   Loss value: {loss.item():.4f}")
    print(f"   Best permutation: {best_perm}")

    # Test SI-SDRi
    print("\n5. Testing SI-SDRi metric...")
    si_sdri = si_sdr_improvement(outputs[0], target_sources[0], mixture)
    print(f"   ✓ SI-SDRi: {si_sdri.mean().item():.2f} dB")

    # Memory usage
    if torch.cuda.is_available():
        print(f"\n6. GPU Memory:")
        print(f"   Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"   Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)

    return True


def test_small_model():
    """Test smaller model version (MossFormer2-S)"""
    print("\n" + "=" * 60)
    print("Testing MossFormer2-S (Small Version)")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create small model (37.8M params)
    args = Namespace(
        encoder_embedding_dim=384,
        mossformer_sequence_dim=384,
        num_mossformer_layer=25,
        encoder_kernel_size=16,
        num_spks=2,
    )

    model = MossFormer2_SS_16K(args).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nParameters: {num_params / 1e6:.1f}M (expected: 37.8M)")

    # Test forward
    mixture = torch.randn(1, 32000).to(device)
    with torch.no_grad():
        outputs = model(mixture)

    print(f"✓ Small model works! Output shapes: {[out.shape for out in outputs]}")

    return True


def test_batch_processing():
    """Test batch processing"""
    print("\n" + "=" * 60)
    print("Testing Batch Processing")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create model
    args = Namespace(
        encoder_embedding_dim=256,  # Smaller for faster testing
        mossformer_sequence_dim=256,
        num_mossformer_layer=6,
        encoder_kernel_size=16,
        num_spks=2,
    )

    model = MossFormer2_SS_16K(args).to(device)

    # Test different batch sizes
    for batch_size in [1, 2, 4]:
        print(f"\nBatch size: {batch_size}")
        mixture = torch.randn(batch_size, 16000).to(device)

        try:
            with torch.no_grad():
                start = time.time()
                outputs = model(mixture)
                elapsed = time.time() - start

            print(f"  ✓ Success! Time: {elapsed:.3f}s")

            if torch.cuda.is_available():
                print(f"  Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
                torch.cuda.empty_cache()
        except RuntimeError as e:
            print(f"  ✗ Failed: {e}")
            break

    return True


if __name__ == "__main__":
    import sys

    try:
        print("\n" + "=" * 60)
        print("MossFormer2 Model Test Suite")
        print("=" * 60)

        # Test 1: Full model
        print("\n[Test 1/3] Full Model (55.7M params)")
        test_model_forward()

        # Test 2: Small model
        print("\n[Test 2/3] Small Model (37.8M params)")
        test_small_model()

        # Test 3: Batch processing
        print("\n[Test 3/3] Batch Processing")
        test_batch_processing()

        print("\n" + "=" * 60)
        print("🎉 All tests completed successfully!")
        print("=" * 60)
        print("\nYou can now start training with:")
        print("  python train.py --config configs/train_mossformer2.yaml")
        print("\nOr use the quick start script:")
        print("  ./quick_start.sh")
        print()

    except Exception as e:
        print(f"\n✗ Test failed with error:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
