import torch
import torch.nn.functional as F
from probe_torch import create_centered_circle

def vscale_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """
    scale for display
    """
    tensor = tensor + .01
    tensor = torch.log(tensor)
    tensor_min = tensor.min()
    tensor_max = tensor.max()
    tensor = (tensor - tensor_min) / (tensor_max - tensor_min)
    return tensor

def extract_amplitude_phase(tensor):
    """
    Extract amplitude and phase from a tensor of shape [N, C, H, W] (C = 3).
    tensor[:, 0, :, :] encodes amplitude via softplus.
    tensor[:, 1:3, :, :].mean(axis=1) encodes phase.
    Returns amplitude and phase tensors.
    """
    assert tensor.shape[1] == 3, "Channel dimension should be 3."
    
    # Calculate amplitude using softplus
    amplitude = F.softplus(tensor[:, 0, :, :])
    
    # Calculate phase by taking the mean of the second and third channels
    phase = tensor[:, 1:3, :, :].mean(dim=1)
    
    return amplitude, phase

def hsv_to_rgb(h, s, v):
    """
    Convert HSV to RGB.
    h, s, v should be PyTorch tensors with values in [0, 1].
    Returns r, g, b tensors of the same shape as input.
    """
    c = v * s
    x = c * (1 - torch.abs((h * 6) % 2 - 1))
    m = v - c
    
    r, g, b = 0, 0, 0
    mask = (h < 1/6)
    r, g, b = torch.where(mask, c, r), torch.where(mask, x, g), torch.where(mask, 0, b)
    mask = (h >= 1/6) & (h < 1/3)
    r, g, b = torch.where(mask, x, r), torch.where(mask, c, g), torch.where(mask, 0, b)
    mask = (h >= 1/3) & (h < 1/2)
    r, g, b = torch.where(mask, 0, r), torch.where(mask, c, g), torch.where(mask, x, b)
    mask = (h >= 1/2) & (h < 2/3)
    r, g, b = torch.where(mask, 0, r), torch.where(mask, x, g), torch.where(mask, c, b)
    mask = (h >= 2/3) & (h < 5/6)
    r, g, b = torch.where(mask, x, r), torch.where(mask, 0, g), torch.where(mask, c, b)
    mask = (h >= 5/6)
    r, g, b = torch.where(mask, c, r), torch.where(mask, 0, g), torch.where(mask, x, b)
    
    r, g, b = r + m, g + m, b + m
    return r, g, b

def tensor_to_hsv(tensor):
    """
    Convert a tensor of shape [N, C, H, W] (C = 3) with the given encoding to RGB.
    Uses extracted amplitude and phase to convert to HSV, then to RGB.
    """
    amplitude, phase = extract_amplitude_phase(tensor)
    
    # Normalize phase to be within [0, 1] for hue
    phase = (phase - phase.min()) / (phase.max() - phase.min())
    
    # Set saturation to a constant value, e.g., 1 for visualization
    saturation = torch.ones_like(amplitude)
    
    # Convert amplitude to value, scaling between [0, 1]
    value = (amplitude - amplitude.min()) / (amplitude.max() - amplitude.min())
    
    # Convert HSV to RGB
    r, g, b = hsv_to_rgb(phase, saturation, value)
    
    # Stack to create RGB image
    return torch.stack((r, g, b), dim=1)

def apply_mask_to_hsv_tensor(hsv_tensor, mask):
    """
    Apply a 2D mask to a 4D HSV tensor (in RGB format).
    Pixels corresponding to mask values of 0 are set to black.
    
    Args:
    hsv_tensor (torch.Tensor): A tensor of shape [N, 3, H, W] representing images in RGB format.
    mask (torch.Tensor): A 2D tensor of shape [H, W] with values between 0 and 1, where 1 means keep and 0 means mask.

    Returns:
    torch.Tensor: The masked HSV tensor.
    """
    if hsv_tensor.dim() != 4 or mask.dim() != 2:
        raise ValueError("hsv_tensor must be 4D and mask must be 2D.")

    # Ensure mask is broadcastable to the size of hsv_tensor
    mask = mask[None, None, :, :]  # Expand dims to [1, 1, H, W]
    mask = mask.expand_as(hsv_tensor)  # Expand across batch and channel dims

    # Apply the mask: pixels where mask is 0 will be set to zero (black)
    masked_tensor = hsv_tensor * mask
    return masked_tensor

def visualize_realspace(outputs, mask):
    """
    Process the tensor by applying a mask, converting to HSV in RGB format, and extracting amplitude and phase.
    
    Args:
    outputs (dict): Dictionary containing a tensor under the key 'intermediate_img'.
    extract_func (callable): Function to extract amplitude and phase from the tensor.
    
    Returns:
    dict: Contains the amplitude, phase, and RGB representation of the HSV tensor.
    """
    # Apply mask to the intermediate image tensor
    intermediate_img = outputs['intermediate_img']
    smooth_tensor = apply_mask_to_hsv_tensor(intermediate_img.cpu(), mask.cpu())

    # Convert to HSV and then to RGB
    rgb_tensor = tensor_to_hsv(smooth_tensor)

    # Apply mask to the RGB tensor
    rgb_tensor = apply_mask_to_hsv_tensor(rgb_tensor, mask)

    # Extract amplitude and phase for plotting
    amplitude, phase = extract_amplitude_phase(smooth_tensor)

    return {
        'amplitude': amplitude,
        'phase': phase,
        'rgb_tensor': rgb_tensor
    }


def cat_images(val_pre_img, val_diff_img, outputs, args, device):
    predicted_val_img = outputs['predicted_amplitude'] 
    to_cat = [val_pre_img,
                 visualize_realspace(outputs,
                     create_centered_circle(args.input_size))['rgb_tensor'].to(device),
                 vscale_tensor((val_diff_img.sqrt() / outputs['intensity_scale'])) * (1 - outputs['mask']),
                 vscale_tensor(predicted_val_img),
                 vscale_tensor((val_diff_img.sqrt() / outputs['intensity_scale']))]
    ncat = len(to_cat)
    return torch.cat(to_cat, dim=0), ncat
