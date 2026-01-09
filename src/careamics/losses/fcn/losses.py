"""
Loss submodule.

This submodule contains the various losses used in CAREamics.
"""

import torch
import torch.nn.functional as F
from torch.nn import L1Loss, MSELoss

from careamics.models.lvae.noise_models import GaussianMixtureNoiseModel


def mse_loss(source: torch.Tensor, target: torch.Tensor, *args) -> torch.Tensor:
    """
    Mean squared error loss.

    Parameters
    ----------
    source : torch.Tensor
        Source patches.
    target : torch.Tensor
        Target patches.
    *args : Any
        Additional arguments.

    Returns
    -------
    torch.Tensor
        Loss value.
    """
    loss = MSELoss()
    return loss(source, target)


def anscombe_mse_loss(
    source: torch.Tensor, target: torch.Tensor, *args
) -> torch.Tensor:
    """
    Mean squared error loss in Anscombe-transformed space.

    This loss applies the Anscombe variance-stabilizing transform to both
    predictions and targets before computing MSE. The transform converts
    Poisson-distributed photon counts to approximately Gaussian-distributed
    values with uniform variance, making MSE optimization more appropriate.

    The forward Anscombe transform is: f(x) = 2*sqrt(x + 3/8)
    No inverse transform is applied here - that happens during inference.

    Parameters
    ----------
    source : torch.Tensor
        Predicted photon counts (model output, assumed in count space).
    target : torch.Tensor
        Ground truth photon counts.
    *args : Any
        Optional arguments:
        - args[0]: image_means (list[float]) - Mean values for denormalization.
        - args[1]: image_stds (list[float]) - Std values for denormalization.
        If provided, data is denormalized before applying Anscombe transform.

    Returns
    -------
    torch.Tensor
        MSE loss computed in Anscombe-transformed space.

    Notes
    -----
    The loss workflow is:
    1. Denormalize predictions and targets (if normalization stats provided)
    2. Apply Anscombe transform: 2*sqrt(x + 3/8)
    3. Compute MSE in transformed space
    4. Model trains to minimize error in this stabilized space

    During inference, predictions are inverse-transformed back to count space.
    """
    # Extract normalization statistics if provided
    image_means = None
    image_stds = None
    if len(args) >= 2:
        image_means = args[0]
        image_stds = args[1]

    # Denormalize to photon count scale if needed
    if image_means is not None and image_stds is not None:
        device = source.device
        means = torch.tensor(image_means, device=device, dtype=source.dtype)
        stds = torch.tensor(image_stds, device=device, dtype=source.dtype)

        # Reshape for broadcasting: (1, C, 1, 1, ...) to match (B, C, D, H, W, ...)
        stats_shape = [1, len(means)] + [1] * (len(source.shape) - 2)
        means = means.view(stats_shape)
        stds = stds.view(stats_shape)

        # Denormalize: x_original = x_normalized * std + mean
        source = (source * stds) + means
        target = (target * stds) + means

    # Clamp to prevent negative values (Anscombe requires non-negative input)
    source = torch.clamp(source, min=0.0)
    target = torch.clamp(target, min=0.0)

    # Apply Anscombe variance-stabilizing transform
    source_anscombe = 2.0 * torch.sqrt(source + 3.0 / 8.0)
    target_anscombe = 2.0 * torch.sqrt(target + 3.0 / 8.0)

    # Compute MSE in transformed space
    loss = MSELoss()
    return loss(source_anscombe, target_anscombe)


def n2v_masked_signal_loss(
    manipulated_batch: torch.Tensor,
    original_batch: torch.Tensor,
    masks: torch.Tensor,
    *args,
    signal_threshold: float = 0.5,
) -> torch.Tensor:
    """N2V loss computed only on pixels with signal above threshold.

    Background pixels below the threshold are excluded from loss calculation.

    Parameters
    ----------
    manipulated_batch : torch.Tensor
        Model predictions. Shape: (B, C_out, ...).
    original_batch : torch.Tensor
        Ground truth data. Shape: (B, C_in, ...).
    masks : torch.Tensor
        Binary mask for N2V blind spots. Shape: (B, C_in, ...).
    *args : Any
        Optional normalization stats (image_means, image_stds).
    signal_threshold : float, optional
        Minimum value for a pixel to be considered signal, by default 0.5.

    Returns
    -------
    torch.Tensor
        Mean squared error over masked signal pixels only.
    """
    # [handle channel mismatches same as before]

    if masks.sum() == 0:
        return torch.tensor(0.0, device=manipulated_batch.device, requires_grad=True)

    # Denormalize if stats provided
    if len(args) >= 2:
        image_means = args[0]
        image_stds = args[1]
        device = manipulated_batch.device
        means = torch.tensor(image_means, device=device, dtype=manipulated_batch.dtype)
        stds = torch.tensor(image_stds, device=device, dtype=manipulated_batch.dtype)

        stats_shape = [1, len(means)] + [1] * (len(manipulated_batch.shape) - 2)
        means = means.view(stats_shape)
        stds = stds.view(stats_shape)

        manipulated_batch = (manipulated_batch * stds) + means
        original_batch = (original_batch * stds) + means

    # Signal mask: only pixels with photons contribute to loss
    signal_mask = (original_batch > signal_threshold).float()
    combined_mask = masks * signal_mask

    mask_sum = combined_mask.sum()
    if mask_sum == 0:
        return torch.tensor(0.0, device=manipulated_batch.device, requires_grad=True)

    # Simple MSE on signal pixels only
    errors = (original_batch - manipulated_batch) ** 2
    loss = torch.sum(errors * combined_mask) / mask_sum

    return loss


def n2v_anscombe_loss(
    manipulated_batch: torch.Tensor,
    original_batch: torch.Tensor,
    masks: torch.Tensor,
    *args,
) -> torch.Tensor:
    """
    N2V loss with Anscombe variance-stabilizing transform.

    This combines N2V masking with Anscombe-transformed MSE loss. The transform
    stabilizes variance across intensity levels, making MSE optimization more
    appropriate for Poisson-distributed photon counting data.

    Parameters
    ----------
    manipulated_batch : torch.Tensor
        Model predictions in count space. Shape: (B, C_out, ...).
    original_batch : torch.Tensor
        Ground truth photon counts. Shape: (B, C_in, ...).
    masks : torch.Tensor
        Binary mask indicating which pixels were masked. Shape: (B, C_in, ...).
    *args : Any
        Optional arguments:
        - args[0]: image_means (list[float]) - Mean values for DATA channels only.
        - args[1]: image_stds (list[float]) - Std values for DATA channels only.

    Returns
    -------
    torch.Tensor
        Mean squared error over masked pixels in Anscombe-transformed space.

    Notes
    -----
    Loss computation steps:
    1. Handle channel dimension mismatches (same as standard N2V)
    2. Denormalize to photon count scale (if stats provided)
    3. Apply Anscombe transform: 2*sqrt(x + 3/8)
    4. Compute squared error on transformed values
    5. Average over masked pixels only

    When using auxiliary channels (positional encoding):
    - Only pass normalization statistics for DATA channels
    - Do NOT include statistics for auxiliary channels
    """
    # Handle channel dimension mismatch (same logic as n2v_loss)
    if manipulated_batch.shape[1] < original_batch.shape[1]:
        channel_has_mask = (
            masks.sum(dim=[d for d in range(len(masks.shape)) if d != 1]) > 0
        )
        masked_channel_indices = torch.where(channel_has_mask)[0]
        original_batch = original_batch[:, masked_channel_indices, ...]
        masks = masks[:, masked_channel_indices, ...]

        if manipulated_batch.shape[1] == 1 and original_batch.shape[1] > 1:
            manipulated_batch = manipulated_batch.expand(
                -1, original_batch.shape[1], *[-1] * (len(original_batch.shape) - 2)
            )

    # Check for empty masks early
    mask_sum = masks.sum()
    if mask_sum == 0:
        return torch.tensor(0.0, device=manipulated_batch.device, requires_grad=True)

    # Extract normalization statistics if provided
    image_means = None
    image_stds = None
    if len(args) >= 2:
        image_means = args[0]
        image_stds = args[1]

    # Denormalize to photon count scale if needed
    if image_means is not None and image_stds is not None:
        device = manipulated_batch.device
        means = torch.tensor(image_means, device=device, dtype=manipulated_batch.dtype)
        stds = torch.tensor(image_stds, device=device, dtype=manipulated_batch.dtype)

        stats_shape = [1, len(means)] + [1] * (len(manipulated_batch.shape) - 2)
        means = means.view(stats_shape)
        stds = stds.view(stats_shape)

        manipulated_batch = (manipulated_batch * stds) + means
        original_batch = (original_batch * stds) + means

    if (original_batch < 0).any():
        print(f"Negative values in original_batch: {(original_batch < 0).sum().item()}")
    # Clamp to prevent negative values
    manipulated_batch = torch.clamp(manipulated_batch, min=0.0)
    original_batch = torch.clamp(original_batch, min=0.0)

    # Apply Anscombe variance-stabilizing transform
    pred_anscombe = 2.0 * torch.sqrt(manipulated_batch + 3.0 / 8.0)
    target_anscombe = 2.0 * torch.sqrt(original_batch + 3.0 / 8.0)

    # Compute squared error in transformed space
    errors = (target_anscombe - pred_anscombe) ** 2

    # Average over masked pixels only
    loss = torch.sum(errors * masks) / mask_sum

    return loss


def n2v_loss(
    manipulated_batch: torch.Tensor,
    original_batch: torch.Tensor,
    masks: torch.Tensor,
    *args,
) -> torch.Tensor:
    """
    N2V Loss function described in A Krull et al 2018.

    Parameters
    ----------
    manipulated_batch : torch.Tensor
        Batch after manipulation function applied. Shape: (B, C_out, ...).
    original_batch : torch.Tensor
        Original images. Shape: (B, C_in, ...).
    masks : torch.Tensor
        Coordinates of changed pixels. Shape: (B, C_in, ...).
    *args : Any
        Additional arguments.

    Returns
    -------
    torch.Tensor
        Loss value.

    Notes
    -----
    When C_out < C_in (e.g., model outputs 1 channel but input has multiple channels),
    the loss is computed only on channels where masks are non-zero (data channels).
    The manipulated_batch is broadcast or only the relevant channels are used.
    """
    # If output channels < input channels, only compute loss on masked channels
    if manipulated_batch.shape[1] < original_batch.shape[1]:
        # Find which channels have non-zero masks
        # Sum over all dimensions except channel dimension
        channel_has_mask = (
            masks.sum(dim=[d for d in range(len(masks.shape)) if d != 1]) > 0
        )

        # Get indices of channels with masks
        masked_channel_indices = torch.where(channel_has_mask)[0]

        # Select only the masked channels from original and masks
        original_batch = original_batch[:, masked_channel_indices, ...]
        masks = masks[:, masked_channel_indices, ...]

        # If model outputs 1 channel and there are multiple masked channels,
        # we need to expand the prediction to match
        if manipulated_batch.shape[1] == 1 and original_batch.shape[1] > 1:
            manipulated_batch = manipulated_batch.expand(
                -1, original_batch.shape[1], *[-1] * (len(original_batch.shape) - 2)
            )

    errors = (original_batch - manipulated_batch) ** 2
    # Average over pixels and batch
    loss = torch.sum(errors * masks) / torch.sum(masks)
    return loss  # TODO change output to dict ?


def n2v_poisson_loss(
    manipulated_batch: torch.Tensor,  # model predictions (rates)
    original_batch: torch.Tensor,  # observed counts
    masks: torch.Tensor,
    *args,
) -> torch.Tensor:
    """
    N2V Loss with Poisson NLL for photon counting data.

    Uses PyTorch's optimized poisson_nll_loss with masked averaging.
    This implementation:
    - Leverages PyTorch's C++ backend for efficiency
    - Uses the same masking pattern as standard N2V loss
    - Computes mean loss over masked pixels only

    Parameters
    ----------
    manipulated_batch : torch.Tensor
        Predicted photon rates (must be positive). Shape: (B, C_out, ...).
    original_batch : torch.Tensor
        Observed photon counts. Shape: (B, C_in, ...).
    masks : torch.Tensor
        Binary mask indicating which pixels were masked. Shape: (B, C_in, ...).
    *args : Any
        Optional arguments:
        - args[0]: image_means (list[float]) - Mean values for DATA channels only.
                   For N2V with auxiliary channels (e.g., positional encoding),
                   only pass stats for channels containing photon count data.
        - args[1]: image_stds (list[float]) - Std values for DATA channels only.

    Returns
    -------
    torch.Tensor
        Scalar loss value (mean Poisson NLL over masked pixels).

    Notes
    -----
    The Poisson NLL formula is: λ - target*log(λ) (simplified, no factorial term).
    We compute this per-pixel, then take the mean over masked pixels only.

    Why sum(loss * mask) / sum(mask) instead of mean()?
    - Only ~2% of pixels are masked in N2V
    - We want the mean over MASKED pixels only, not all pixels
    - sum(loss * mask) / sum(mask) = weighted mean over masked pixels

    IMPORTANT: When using auxiliary channels (positional encoding, etc.):
    - Only pass normalization statistics for DATA channels (photon counts)
    - Do NOT include statistics for auxiliary channels
    - Auxiliary channels are not photon data and should not be denormalized
    """
    # Handle channel dimension mismatch
    if manipulated_batch.shape[1] < original_batch.shape[1]:
        channel_has_mask = (
            masks.sum(dim=[d for d in range(len(masks.shape)) if d != 1]) > 0
        )
        masked_channel_indices = torch.where(channel_has_mask)[0]
        original_batch = original_batch[:, masked_channel_indices, ...]
        masks = masks[:, masked_channel_indices, ...]

        if manipulated_batch.shape[1] == 1 and original_batch.shape[1] > 1:
            manipulated_batch = manipulated_batch.expand(
                -1, original_batch.shape[1], *[-1] * (len(original_batch.shape) - 2)
            )

    # Check for empty masks early
    mask_sum = masks.sum()
    if mask_sum == 0:
        return torch.tensor(0.0, device=manipulated_batch.device, requires_grad=True)

    # Denormalize to photon count scale before Poisson NLL
    # Poisson requires non-negative counts, but normalized data can be negative!
    # Extract normalization stats from args if provided
    image_means = None
    image_stds = None
    if len(args) >= 2:
        image_means = args[0]
        image_stds = args[1]

    if image_means is not None and image_stds is not None:
        # Denormalize both predictions and targets to count scale
        device = manipulated_batch.device
        means = torch.tensor(image_means, device=device, dtype=manipulated_batch.dtype)
        stds = torch.tensor(image_stds, device=device, dtype=manipulated_batch.dtype)

        # Reshape for broadcasting: (1, C, 1, 1, ...) to match (B, C, D, H, W, ...)
        stats_shape = [1, len(means)] + [1] * (len(manipulated_batch.shape) - 2)
        means = means.view(stats_shape)
        stds = stds.view(stats_shape)

        # Denormalize: x_original = x_normalized * std + mean
        # Model outputs normalized values, we denormalize to photon count scale
        pred_denormalized = (manipulated_batch * stds) + means
        target_counts = (original_batch * stds) + means

        # Apply ReLU + epsilon to ensure predictions are positive
        # (required for Poisson λ > 0)
        # ReLU has NO floor (unlike Softplus floor ~0.693), crucial for sparse data
        # Small epsilon (1e-6) prevents log(0) in Poisson NLL computation
        pred_counts = F.relu(pred_denormalized) + 1e-6

    else:
        # No normalization - apply ReLU + epsilon (backward compatible)
        pred_counts = F.relu(manipulated_batch) + 1e-6
        target_counts = original_batch.clamp(min=0.0)

    # Compute Poisson NLL on denormalized photon count scale
    # Predictions are positive photon rates (after ReLU + epsilon)
    # Targets are clamped photon counts
    # This matches MSE's physical interpretation (both work in photon count space)
    nll_per_pixel = F.poisson_nll_loss(
        pred_counts,
        target_counts,
        log_input=False,  # Predictions are photon rates (not log rates)
        full=False,
        reduction="none",
    )

    # Apply mask and compute mean over masked pixels only
    loss = torch.sum(nll_per_pixel * masks) / mask_sum

    return loss


def pn2v_loss(
    samples: torch.Tensor,
    labels: torch.Tensor,
    masks: torch.Tensor,
    noise_model: GaussianMixtureNoiseModel,
) -> torch.Tensor:
    """
    Probabilistic N2V loss function described in A Krull et al., CVF (2019).

    Parameters
    ----------
    samples : torch.Tensor # TODO this naming is confusing
        Predicted pixel values from the network.
    labels : torch.Tensor
        Original pixel values.
    masks : torch.Tensor
        Coordinates of manipulated pixels.
    noise_model : GaussianMixtureNoiseModel
        Noise model for computing likelihood.

    Returns
    -------
    torch.Tensor
        Loss value.
    """
    likelihoods = noise_model.likelihood(labels, samples)
    likelihoods_avg = torch.log(torch.mean(likelihoods, dim=1, keepdim=True))

    # Average over pixels and batch
    loss = -torch.sum(likelihoods_avg * masks) / torch.sum(masks)
    return loss


def mae_loss(samples: torch.Tensor, labels: torch.Tensor, *args) -> torch.Tensor:
    """
    N2N Loss function described in to J Lehtinen et al 2018.

    Parameters
    ----------
    samples : torch.Tensor
        Raw patches.
    labels : torch.Tensor
        Different subset of noisy patches.
    *args : Any
        Additional arguments.

    Returns
    -------
    torch.Tensor
        Loss value.
    """
    loss = L1Loss()
    return loss(samples, labels)


# def dice_loss(
#     samples: torch.Tensor, labels: torch.Tensor, mode: str = "multiclass"
# ) -> torch.Tensor:
#     """Dice loss function."""
#     return DiceLoss(mode=mode)(samples, labels.long())
