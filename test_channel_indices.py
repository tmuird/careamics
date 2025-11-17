"""
Quick test to verify arbitrary channel indices implementation.
Run this to ensure the modifications work correctly.ma
"""

import torch
import numpy as np
from careamics.config.transformations import N2VManipulateModel
from careamics.transforms import N2VManipulateTorch


def test_sequential_channels():
    """Test with sequential channels (backward compatibility)."""
    print("=" * 60)
    print("Test 1: Sequential channels (n_data_channels=2)")
    print("=" * 60)

    config = N2VManipulateModel(
        n_data_channels=2,
        masked_pixel_percentage=10.0,  # 10% masking (max allowed)
        roi_size=7,
    )

    manipulator = N2VManipulateTorch(
        n2v_manipulate_config=config,
        seed=42,
        device="cpu"
    )

    # Create test batch: 4 samples, 5 channels, 512 pixels (1D)
    batch = torch.randn(4, 5, 512)

    masked, original, mask = manipulator(batch)

    # Check that only first 2 channels are masked
    assert mask[:, 0, :].sum() > 0, "Channel 0 should have masks"
    assert mask[:, 1, :].sum() > 0, "Channel 1 should have masks"
    assert mask[:, 2, :].sum() == 0, "Channel 2 should NOT have masks"
    assert mask[:, 3, :].sum() == 0, "Channel 3 should NOT have masks"
    assert mask[:, 4, :].sum() == 0, "Channel 4 should NOT have masks"

    # Check that auxiliary channels are copied
    assert torch.allclose(masked[:, 2:, :], original[:, 2:, :]), \
        "Auxiliary channels should be unchanged"

    print(f"✓ Data channel indices: {manipulator.data_channel_indices}")
    print(f"✓ Channel 0 masked pixels: {mask[:, 0, :].sum().item()}")
    print(f"✓ Channel 1 masked pixels: {mask[:, 1, :].sum().item()}")
    print(f"✓ Channel 2 masked pixels: {mask[:, 2, :].sum().item()}")
    print(f"✓ Auxiliary channels preserved correctly")
    print("✓ Test PASSED\n")


def test_arbitrary_channels():
    """Test with arbitrary channel indices."""
    print("=" * 60)
    print("Test 2: Arbitrary channels (indices=[0, 3, 4])")
    print("=" * 60)

    config = N2VManipulateModel(
        data_channel_indices=[0, 3, 4],
        masked_pixel_percentage=10.0,  # 10% masking (max allowed)
        roi_size=7,
    )

    manipulator = N2VManipulateTorch(
        n2v_manipulate_config=config,
        seed=42,
        device="cpu"
    )

    # Create test batch: 4 samples, 6 channels, 512 pixels (1D)
    batch = torch.randn(4, 6, 512)

    masked, original, mask = manipulator(batch)

    # Check that only channels 0, 3, 4 are masked
    assert mask[:, 0, :].sum() > 0, "Channel 0 should have masks"
    assert mask[:, 1, :].sum() == 0, "Channel 1 should NOT have masks"
    assert mask[:, 2, :].sum() == 0, "Channel 2 should NOT have masks"
    assert mask[:, 3, :].sum() > 0, "Channel 3 should have masks"
    assert mask[:, 4, :].sum() > 0, "Channel 4 should have masks"
    assert mask[:, 5, :].sum() == 0, "Channel 5 should NOT have masks"

    # Check that non-data channels are copied
    assert torch.allclose(masked[:, 1, :], original[:, 1, :]), \
        "Channel 1 should be unchanged"
    assert torch.allclose(masked[:, 2, :], original[:, 2, :]), \
        "Channel 2 should be unchanged"
    assert torch.allclose(masked[:, 5, :], original[:, 5, :]), \
        "Channel 5 should be unchanged"

    print(f"✓ Data channel indices: {manipulator.data_channel_indices}")
    print(f"✓ Channel 0 masked pixels: {mask[:, 0, :].sum().item()}")
    print(f"✓ Channel 3 masked pixels: {mask[:, 3, :].sum().item()}")
    print(f"✓ Channel 4 masked pixels: {mask[:, 4, :].sum().item()}")
    print(f"✓ Non-data channels (1, 2, 5) have 0 masked pixels")
    print(f"✓ Auxiliary channels preserved correctly")
    print("✓ Test PASSED\n")


def test_2d_data():
    """Test with 2D data."""
    print("=" * 60)
    print("Test 3: 2D data (128x128) with arbitrary channels")
    print("=" * 60)

    config = N2VManipulateModel(
        data_channel_indices=[0, 2],
        masked_pixel_percentage=10.0,  # 10% masking (max allowed)
        roi_size=11,
    )

    manipulator = N2VManipulateTorch(
        n2v_manipulate_config=config,
        seed=42,
        device="cpu"
    )

    # Create test batch: 2 samples, 4 channels, 128x128 pixels (2D)
    batch = torch.randn(2, 4, 128, 128)

    masked, original, mask = manipulator(batch)

    # Check shapes
    assert masked.shape == batch.shape, "Masked shape should match input"
    assert mask.shape == batch.shape, "Mask shape should match input"

    # Check that only channels 0, 2 are masked
    assert mask[:, 0, :, :].sum() > 0, "Channel 0 should have masks"
    assert mask[:, 1, :, :].sum() == 0, "Channel 1 should NOT have masks"
    assert mask[:, 2, :, :].sum() > 0, "Channel 2 should have masks"
    assert mask[:, 3, :, :].sum() == 0, "Channel 3 should NOT have masks"

    # Check that non-data channels are copied
    assert torch.allclose(masked[:, 1, :, :], original[:, 1, :, :]), \
        "Channel 1 should be unchanged"
    assert torch.allclose(masked[:, 3, :, :], original[:, 3, :, :]), \
        "Channel 3 should be unchanged"

    print(f"✓ Data channel indices: {manipulator.data_channel_indices}")
    print(f"✓ Input shape: {batch.shape}")
    print(f"✓ Output shape: {masked.shape}")
    print(f"✓ Channel 0 masked pixels: {mask[:, 0, :, :].sum().item()}")
    print(f"✓ Channel 2 masked pixels: {mask[:, 2, :, :].sum().item()}")
    print(f"✓ Auxiliary channels preserved correctly")
    print("✓ Test PASSED\n")


def test_loss_computation():
    """Test that loss computation works with channel selection."""
    print("=" * 60)
    print("Test 4: Loss computation with mismatched channels")
    print("=" * 60)

    from careamics.losses.fcn.losses import n2v_loss

    # Simulate: 7 input channels, 1 output channel
    # Only channel 0 is masked
    batch_size = 4
    n_input_channels = 7
    n_output_channels = 1
    width = 256

    # Create masks (only channel 0 has non-zero masks)
    masks = torch.zeros(batch_size, n_input_channels, width)
    masks[:, 0, 50:100] = 1  # Mask pixels 50-100 in channel 0

    # Create original batch
    original = torch.randn(batch_size, n_input_channels, width)

    # Create prediction (only 1 channel)
    prediction = torch.randn(batch_size, n_output_channels, width)

    # Compute loss (should handle channel mismatch)
    loss = n2v_loss(prediction, original, masks)

    assert not torch.isnan(loss), "Loss should not be NaN"
    assert not torch.isinf(loss), "Loss should not be Inf"
    assert loss >= 0, "Loss should be non-negative"

    print(f"✓ Input channels: {n_input_channels}")
    print(f"✓ Output channels: {n_output_channels}")
    print(f"✓ Loss computed successfully: {loss.item():.6f}")
    print(f"✓ Total masked pixels: {masks.sum().item()}")
    print("✓ Test PASSED\n")


def test_config_validation():
    """Test configuration validation."""
    print("=" * 60)
    print("Test 5: Configuration validation")
    print("=" * 60)

    # Valid configuration
    config = N2VManipulateModel(data_channel_indices=[0, 2, 5])
    print(f"✓ Valid config created: {config.data_channel_indices}")

    # Test that duplicates are rejected
    try:
        config = N2VManipulateModel(data_channel_indices=[0, 1, 1, 2])
        print("✗ Should have rejected duplicate indices")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"✓ Correctly rejected duplicates: {e}")

    # Test that negative indices are rejected
    try:
        config = N2VManipulateModel(data_channel_indices=[0, -1, 2])
        print("✗ Should have rejected negative indices")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"✓ Correctly rejected negative indices: {e}")

    # Test that empty list is rejected
    try:
        config = N2VManipulateModel(data_channel_indices=[])
        print("✗ Should have rejected empty list")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"✓ Correctly rejected empty list: {e}")

    print("✓ Test PASSED\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Testing Arbitrary Channel Indices Implementation")
    print("=" * 60 + "\n")

    try:
        test_sequential_channels()
        test_arbitrary_channels()
        test_2d_data()
        test_loss_computation()
        test_config_validation()

        print("=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
        print("\nThe implementation is working correctly for:")
        print("  ✓ Sequential channel masking (backward compatible)")
        print("  ✓ Arbitrary channel indices")
        print("  ✓ 1D and 2D data")
        print("  ✓ Loss computation with channel mismatch")
        print("  ✓ Configuration validation")
        print("\nYou can now use arbitrary channel indices in your N2V training!")

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
