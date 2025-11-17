# Understanding Output Channel Mapping in N2V with Arbitrary Data Channels

## Introduction

When using the arbitrary data channel feature in CAREamics N2V, it's important to understand how model output channels are mapped to input data channels. This guide explains the mapping behavior and provides best practices for configuring your model correctly.

## Core Concept

In N2V training with multiple channels:
- **Input**: Your data may have `N` channels (e.g., 7 channels: 1 Raman + 6 positional encodings)
- **Data channels**: Only certain channels contain data to denoise (specified by `data_channel_indices`)
- **Output**: The model outputs reconstructed versions of the data channels only

**Key principle**: Model output channel `i` corresponds to data channel at index `data_channel_indices[i]`.

## Channel Mapping Behavior

### Automatic Sorting

The `data_channel_indices` list is **automatically sorted** when you set it, ensuring predictable and consistent mapping:

```python
from careamics.config.transformations import N2VManipulateModel

# You specify in any order
config = N2VManipulateModel(data_channel_indices=[5, 1, 3])

# It's automatically stored in sorted order
print(config.data_channel_indices)  # Output: [1, 3, 5]
```

**Why sorting?** This ensures that:
- The mapping between model outputs and input channels is always in ascending order
- Behavior is consistent regardless of how you specify the indices
- Loss computation is predictable and efficient

### Output-to-Input Mapping

After sorting, model output channels map to input channels as follows:

```
Model Output Channel 0 → Input Channel data_channel_indices[0]
Model Output Channel 1 → Input Channel data_channel_indices[1]
Model Output Channel 2 → Input Channel data_channel_indices[2]
...
```

**Example**:
```python
data_channel_indices = [1, 3, 5]  # After auto-sorting
model.num_classes = 3

# The model learns to:
# - Output[0] reconstructs Input[1]
# - Output[1] reconstructs Input[3]
# - Output[2] reconstructs Input[5]
```

## Configuration Guidelines

### Guideline 1: Match Output Channels to Data Channels

**Rule**: `model.num_classes` should equal the number of data channels

```python
# Calculate the required number of output channels
num_data_channels = len(data_channel_indices)
model.num_classes = num_data_channels
```

### Guideline 2: Common Configurations

#### Single Data Channel (Most Common)

Use this when you have one type of data to denoise with auxiliary channels:

```python
config = create_n2v_configuration(
    experiment_name="raman_denoising",
    axes="SXC",
    n_channels=7,  # 1 Raman + 6 positional encodings
    # ... other params ...
)

# Denoise only the Raman channel (index 0)
config.algorithm_config.n2v_config.data_channel_indices = [0]

# Model outputs 1 channel
config.algorithm_config.model.num_classes = 1
```

#### Multiple Independent Data Channels

Use this when you have multiple channels that all need denoising:

```python
config = create_n2v_configuration(
    experiment_name="multi_channel_microscopy",
    axes="SYX",
    n_channels=6,  # 3 fluorescence + 3 auxiliary
    # ... other params ...
)

# Denoise fluorescence channels at indices 0, 2, 4
config.algorithm_config.n2v_config.data_channel_indices = [0, 2, 4]

# Model outputs 3 channels
config.algorithm_config.model.num_classes = 3
```

### Guideline 3: What to Avoid

#### ❌ Mismatched Channel Counts

```python
# WRONG: 3 data channels but only 2 outputs
config.algorithm_config.n2v_config.data_channel_indices = [0, 2, 4]  # 3 channels
config.algorithm_config.model.num_classes = 2  # Only 2 outputs

# This will cause errors or unexpected behavior!
```

#### ❌ Single Output for Multiple Different Data Types

```python
# QUESTIONABLE: Broadcasting one output to multiple different channels
config.algorithm_config.n2v_config.data_channel_indices = [0, 3, 5]  # 3 different types
config.algorithm_config.model.num_classes = 1  # Same output for all

# The model will try to fit all three channels with one output
# Only use this if all data channels represent the same type of data
```

## Step-by-Step Setup Guide

### Step 1: Identify Your Data Structure

First, understand your data layout:

```python
# Example: Raman spectroscopy with positional encoding
# Channel 0: Raman spectrum (data to denoise)
# Channels 1-6: Positional encodings (auxiliary features)

total_channels = 7
data_channel_index = 0  # Only channel 0 needs denoising
auxiliary_channels = [1, 2, 3, 4, 5, 6]  # These are just features
```

### Step 2: Create Base Configuration

```python
from careamics.config.configuration_factories import create_n2v_configuration

config = create_n2v_configuration(
    experiment_name="my_experiment",
    data_type="array",
    axes="SXC",  # For 1D: SXC, For 2D: SYX
    n_channels=total_channels,
    patch_size=[1024],  # Adjust based on your data
    batch_size=64,
    num_epochs=100,
    masked_pixel_percentage=0.25,  # 0.25% (range: 0.05-10%)
    roi_size=7,
    independent_channels=False,  # Must be False for auxiliary channels
)
```

### Step 3: Specify Data Channels

```python
# Specify which channels contain data to denoise
config.algorithm_config.n2v_config.data_channel_indices = [data_channel_index]

# For multiple data channels:
# config.algorithm_config.n2v_config.data_channel_indices = [0, 2, 4]
```

### Step 4: Set Model Output Channels

```python
# Set number of output channels to match number of data channels
num_data_channels = len(config.algorithm_config.n2v_config.data_channel_indices)
config.algorithm_config.model.num_classes = num_data_channels
```

### Step 5: Verify Configuration

```python
# Print configuration to verify
print(f"Total input channels: {config.data_config.n_channels}")
print(f"Data channel indices: {config.algorithm_config.n2v_config.data_channel_indices}")
print(f"Model output channels: {config.algorithm_config.model.num_classes}")

# Verify they match
assert len(config.algorithm_config.n2v_config.data_channel_indices) == \
       config.algorithm_config.model.num_classes, \
       "Number of data channels must match model output channels!"
```

## Complete Working Examples

### Example 1: Raman Spectroscopy with Positional Encoding (1D)

```python
from careamics.config.configuration_factories import create_n2v_configuration

# Data structure:
# - 1 Raman spectrum channel (index 0)
# - 6 positional encoding channels (indices 1-6)

config = create_n2v_configuration(
    experiment_name="raman_pe_denoising",
    data_type="array",
    axes="SXC",  # Sample, X-position, Channels
    n_channels=7,
    patch_size=[1024],
    batch_size=64,
    num_epochs=100,
    masked_pixel_percentage=0.25,  # 0.25% of pixels masked
    roi_size=7,
    independent_channels=False,
    model_params={"depth": 3},
)

# Configure data channels
config.algorithm_config.n2v_config.data_channel_indices = [0]  # Only Raman channel
config.algorithm_config.model.num_classes = 1  # Output 1 channel

# Set normalization (optional)
config.data_config.image_means = [0.0] * 7
config.data_config.image_stds = [1.0] * 7

print("Configuration Summary:")
print(f"✓ Input: 7 channels (1 Raman + 6 PE)")
print(f"✓ Masking: Channel 0 only")
print(f"✓ Output: 1 channel (reconstructed Raman)")
print(f"✓ Auxiliary channels (1-6) used as features, not masked")
```

### Example 2: Multi-Channel Microscopy (2D)

```python
from careamics.config.configuration_factories import create_n2v_configuration

# Data structure:
# - 3 fluorescence channels (indices 0, 1, 2)
# - 2 texture feature channels (indices 3, 4)

config = create_n2v_configuration(
    experiment_name="multi_channel_microscopy",
    data_type="array",
    axes="SYX",  # Sample, Y, X
    n_channels=5,
    patch_size=[256, 256],
    batch_size=32,
    num_epochs=100,
    masked_pixel_percentage=5.0,  # 5% of pixels masked
    roi_size=11,
    independent_channels=False,
)

# Configure data channels
config.algorithm_config.n2v_config.data_channel_indices = [0, 1, 2]  # Fluorescence
config.algorithm_config.model.num_classes = 3  # Output 3 channels

print("Configuration Summary:")
print(f"✓ Input: 5 channels (3 fluorescence + 2 texture)")
print(f"✓ Masking: Channels 0, 1, 2")
print(f"✓ Output: 3 channels (reconstructed fluorescence)")
print(f"✓ Channel mapping:")
print(f"  - Output 0 → Input 0 (fluorescence 1)")
print(f"  - Output 1 → Input 1 (fluorescence 2)")
print(f"  - Output 2 → Input 2 (fluorescence 3)")
print(f"✓ Texture channels (3-4) used as features, not masked")
```

### Example 3: Non-Sequential Data Channels (2D)

```python
from careamics.config.configuration_factories import create_n2v_configuration

# Data structure:
# - Fluorescence channels at indices 0, 3, 5
# - Other channels are auxiliary features

config = create_n2v_configuration(
    experiment_name="non_sequential_channels",
    data_type="array",
    axes="SYX",
    n_channels=7,
    patch_size=[256, 256],
    batch_size=32,
    num_epochs=100,
    masked_pixel_percentage=5.0,
    roi_size=11,
    independent_channels=False,
)

# Configure data channels (will be auto-sorted to [0, 3, 5])
config.algorithm_config.n2v_config.data_channel_indices = [5, 0, 3]  # Any order
config.algorithm_config.model.num_classes = 3

# Verify auto-sorting
print(f"Specified: [5, 0, 3]")
print(f"Stored as: {config.algorithm_config.n2v_config.data_channel_indices}")  # [0, 3, 5]

print("\nConfiguration Summary:")
print(f"✓ Input: 7 channels")
print(f"✓ Masking: Channels 0, 3, 5 (auto-sorted)")
print(f"✓ Output: 3 channels")
print(f"✓ Channel mapping:")
print(f"  - Output 0 → Input 0")
print(f"  - Output 1 → Input 3")
print(f"  - Output 2 → Input 5")
print(f"✓ Channels 1, 2, 4, 6 are auxiliary features")
```

## How Loss is Computed

Understanding the loss computation helps you verify your configuration is correct:

1. **Masking**: Only pixels in data channels specified by `data_channel_indices` are masked
2. **Forward Pass**: Masked input goes through the model → prediction
3. **Channel Selection**: Loss function identifies which channels have masks
4. **Comparison**: Prediction is compared to original at masked pixels only
5. **Gradient Flow**: Only data channels contribute gradients

```python
# Conceptually, the loss computation does this:
# for i, channel_idx in enumerate(sorted(data_channel_indices)):
#     loss += MSE(prediction[:, i], original[:, channel_idx], mask[:, channel_idx])
```

## Troubleshooting

### Problem: Shape Mismatch Errors

**Symptom**: Errors during training about incompatible shapes

**Solution**: Ensure `model.num_classes` matches the number of data channels:
```python
assert len(config.algorithm_config.n2v_config.data_channel_indices) == \
       config.algorithm_config.model.num_classes
```

### Problem: All Channels Being Masked

**Symptom**: Auxiliary channels show masked pixels

**Solution**: Verify `data_channel_indices` is set correctly and `independent_channels=False`:
```python
config.algorithm_config.n2v_config.data_channel_indices = [0]  # Only channel 0
config.data_config.independent_channels = False  # Important!
```

### Problem: Loss Not Decreasing

**Symptom**: Training loss stays high or doesn't improve

**Possible causes**:
1. **Masking percentage too low**: Try increasing `masked_pixel_percentage` (e.g., from 0.25 to 5.0)
2. **Wrong channels masked**: Verify `data_channel_indices` points to actual data channels
3. **Insufficient data**: Ensure you have enough training samples

### Problem: Understanding Which Channel Is Which

**Solution**: Test your configuration before training:
```python
import torch
from careamics.transforms import N2VManipulateTorch

# Create manipulator
manipulator = N2VManipulateTorch(
    n2v_manipulate_config=config.algorithm_config.n2v_config,
    seed=42,
    device='cpu'
)

# Test with dummy data
dummy_batch = torch.randn(4, 7, 256)  # (batch, channels, width)
masked, original, mask = manipulator(dummy_batch)

# Check which channels have masks
print("Channels with masks:")
for c in range(7):
    n_masked = mask[:, c, :].sum().item()
    if n_masked > 0:
        print(f"  Channel {c}: {n_masked} masked pixels")
    else:
        print(f"  Channel {c}: 0 masked pixels (auxiliary)")
```

## Summary

**Key Takeaways:**

1. ✅ `data_channel_indices` is automatically sorted
2. ✅ Model output channel `i` reconstructs input channel `data_channel_indices[i]`
3. ✅ Set `model.num_classes` equal to number of data channels
4. ✅ Auxiliary channels are used as features but not masked or reconstructed
5. ✅ Works identically for 1D, 2D, and 3D data

**Quick Reference:**

| Scenario | data_channel_indices | num_classes | Output Mapping |
|----------|---------------------|-------------|----------------|
| Single data channel | `[0]` | `1` | Out[0] → In[0] |
| Sequential data channels | `[0, 1, 2]` | `3` | Out[i] → In[i] |
| Non-sequential data channels | `[0, 3, 5]` | `3` | Out[0]→In[0], Out[1]→In[3], Out[2]→In[5] |

**For most users**: If you have one data channel with auxiliary features, simply use:
```python
data_channel_indices = [0]
num_classes = 1
```
