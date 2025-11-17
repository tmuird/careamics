# Using Arbitrary Channel Indices with N2V

This guide explains how to use the enhanced N2V implementation that supports arbitrary data channel indices.

## Overview

The modified CAREamics N2V implementation now supports:
- **Arbitrary channel indices**: Specify exactly which channels are data channels (e.g., `[0, 3, 5]`)
- **Auxiliary channels**: Non-data channels (like positional encodings) are passed through without masking
- **Independent output channels**: Model can output fewer channels than input (e.g., 1 output channel from 7 input channels)
- **Dimension agnostic**: Works with 1D, 2D, and 3D data

> **Important Note on Masking Percentage**: The `masked_pixel_percentage` parameter represents a **percentage value** (not a fraction). Valid range is **0.05% to 10%**. For example:
> - `masked_pixel_percentage=0.25` → 0.25% of pixels masked (very sparse)
> - `masked_pixel_percentage=5.0` → 5% of pixels masked (recommended range)
> - `masked_pixel_percentage=10.0` → 10% of pixels masked (maximum allowed)

## Configuration Options

### Option 1: Using `n_data_channels` (Simple, Sequential Channels)

If your data channels are the **first N channels** starting from index 0:

```python
from careamics.config.configuration_factories import create_n2v_configuration

config = create_n2v_configuration(
    experiment_name="my_experiment",
    data_type="array",
    axes="SYX",  # For 2D: SYX, For 1D: SXC
    n_channels=7,  # Total channels (1 data + 6 auxiliary)
    patch_size=[256, 256],  # [256, 256] for 2D, [1024] for 1D
    batch_size=32,
    num_epochs=100,
    masked_pixel_percentage=0.25,  # 0.25% of pixels masked
    roi_size=11,
    independent_channels=False,
)

# Set n_data_channels in the N2V config (masks channels 0 to n_data_channels-1)
config.algorithm_config.n2v_config.n_data_channels = 1  # Only mask channel 0
```

### Option 2: Using `data_channel_indices` (Flexible, Arbitrary Channels)

If your data channels are at **arbitrary positions**:

```python
from careamics.config.configuration_factories import create_n2v_configuration

config = create_n2v_configuration(
    experiment_name="my_experiment",
    data_type="array",
    axes="SYX",  # For 2D: SYX, For 1D: SXC
    n_channels=7,  # Total channels
    patch_size=[256, 256],
    batch_size=32,
    num_epochs=100,
    masked_pixel_percentage=0.25,  # 0.25% of pixels masked
    roi_size=11,
    independent_channels=False,
)

# Specify exact channel indices to mask
config.algorithm_config.n2v_config.data_channel_indices = [0, 3, 5]  # Mask channels 0, 3, and 5
```

## Complete Example with Positional Encoding

### 1D Data Example

```python
from careamics.config.configuration_factories import create_n2v_configuration
import numpy as np

# Prepare data: 1 Raman channel + 6 positional encoding channels
X_train_pe = np.random.randn(100, 1024, 7)  # (samples, width, channels)

# Create configuration
config = create_n2v_configuration(
    experiment_name="raman_pe_experiment",
    data_type="array",
    axes="SXC",  # Sample, X-dimension, Channels
    n_channels=7,  # 1 data + 6 PE channels
    patch_size=[1024],
    batch_size=64,
    num_epochs=100,
    masked_pixel_percentage=0.25,  # 0.25% of pixels masked
    roi_size=7,
    independent_channels=False,
    model_params={
        "depth": 3,
    },
)

# Option A: Mask only the first channel (channel 0)
config.algorithm_config.n2v_config.n_data_channels = 1

# Option B: Explicitly specify data channel indices
# config.algorithm_config.n2v_config.data_channel_indices = [0]

# Set model output to 1 channel (reconstructs only the data channel)
config.algorithm_config.model.num_classes = 1

print(f"Data channels to mask: {config.algorithm_config.n2v_config.data_channel_indices or list(range(config.algorithm_config.n2v_config.n_data_channels))}")
print(f"Model output channels: {config.algorithm_config.model.num_classes}")
```

### 2D Data Example

```python
from careamics.config.configuration_factories import create_n2v_configuration
import numpy as np

# Prepare data: 3 image channels + 2 auxiliary channels
X_train = np.random.randn(100, 512, 512, 5)  # (samples, height, width, channels)

# Create configuration for 2D data
config = create_n2v_configuration(
    experiment_name="2d_multi_channel_experiment",
    data_type="array",
    axes="SYX",  # Sample, Y-dimension, X-dimension
    n_channels=5,  # 3 data + 2 auxiliary channels
    patch_size=[256, 256],
    batch_size=32,
    num_epochs=100,
    masked_pixel_percentage=0.2,
    roi_size=11,
    independent_channels=False,
)

# Mask only specific channels (e.g., channels 0, 1, 2 are data, 3, 4 are auxiliary)
config.algorithm_config.n2v_config.data_channel_indices = [0, 1, 2]

# Model outputs 3 channels (one for each data channel)
config.algorithm_config.model.num_classes = 3

print(f"Data channels to mask: {config.algorithm_config.n2v_config.data_channel_indices}")
print(f"Model output channels: {config.algorithm_config.model.num_classes}")
```

## How It Works

### 1. Masking

Only channels specified in `data_channel_indices` (or the first `n_data_channels`) are masked during training:
- **Data channels**: Random pixels are masked and replaced according to N2V strategy
- **Auxiliary channels**: Copied without modification to the masked batch

### 2. Loss Computation

The loss is computed **only on masked pixels** in data channels:
```python
# Loss function automatically handles channel selection
errors = (original_batch - prediction) ** 2
loss = torch.sum(errors * masks) / torch.sum(masks)
```

The `masks` tensor has non-zero values only for:
- Pixels that were masked
- Channels that are data channels

### 3. Model Output

The model can output:
- **Same number of channels as input**: Each input channel reconstructed independently
- **Fewer channels than input**: Only data channels are reconstructed
  - Example: 7 input channels (1 data + 6 PE) → 1 output channel

## Advanced Usage: Multiple Non-Sequential Data Channels

```python
# Example: Channels 1, 3, 7 are data channels; others are auxiliary
config.algorithm_config.n2v_config.data_channel_indices = [1, 3, 7]

# Model outputs 3 channels (for the 3 data channels)
config.algorithm_config.model.num_classes = 3
```

## Validation

The validation metrics are computed using the same masking strategy as training:
- Masks are generated for validation patches
- Loss is computed only on masked pixels in data channels
- Metrics (PSNR, SSIM) compare predictions to original data for masked regions

## Migration from Old API

### Before (only first N channels)
```python
config.algorithm_config.n2v_config.n_data_channels = 3  # Masks channels 0, 1, 2
```

### After (arbitrary channels)
```python
# Equivalent to above
config.algorithm_config.n2v_config.data_channel_indices = [0, 1, 2]

# Or mask non-sequential channels
config.algorithm_config.n2v_config.data_channel_indices = [0, 3, 5]  # Skip channels 1, 2, 4
```

The old API (`n_data_channels`) is still supported for backward compatibility!

## Troubleshooting

### Shape Mismatch Errors

If you get shape mismatch errors during training:
1. Check that `num_classes` matches the number of data channels (or is 1 for single-channel output)
2. Verify that all channel indices in `data_channel_indices` are less than `n_channels`

### Loss is NaN or Inf

This can happen if:
1. No pixels are masked (increase `masked_pixel_percentage`)
2. All data channels have zero masks (check `data_channel_indices`)

## Compatibility

- **1D data**: Axes = "SXC", patch_size = `[width]`
- **2D data**: Axes = "SYX", patch_size = `[height, width]`
- **3D data**: Axes = "SZYX", patch_size = `[depth, height, width]`

All implementations work identically across dimensions!
