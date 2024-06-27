import torch
import numpy as np
import params
from scipy.ndimage.filters import gaussian_filter as gf

def lowpass_g(size, y, sym = False):
    from scipy.signal.windows import gaussian
    L = gaussian(len(y), std = len(y) / (size * np.pi**2), sym = sym)
    L /= L.max()
    return L

default_probe_scale = params.cfg['default_probe_scale']

def get_default_probe(fmt='torch', N = None, probe_scale = 0.7):
    if N is None:
        N = params.get('N')
    filt = lowpass_g(probe_scale, np.ones(N), sym=True)
    probe_np = gf(((np.einsum('i,j->ij', filt, filt)) > .5).astype(float), 2) + 1e-9
#    norm = float(torch.mean(torch.abs(probe_np)))
#    probe_np = probe_np / norm

    if fmt == 'np':
        return probe_np
    elif fmt == 'torch':
        return torch.tensor(probe_np, dtype=torch.float32)
    else:
        raise ValueError

def get_probe(fmt='torch'):
    probe_torch = params.get('probe')
    assert len(probe_torch.shape) == 3
    if fmt == 'np':
        return probe_torch.detach().cpu().numpy()[:, :, 0]
    elif fmt == 'torch':
        return probe_torch
    else:
        raise ValueError

def to_np(probe):
    assert len(probe.shape) == 3
    return probe[:, :, 0].detach().cpu().numpy()

def get_squared_distance():
    N = params.get('N')
    filt = lowpass_g(default_probe_scale, np.ones(N), sym=True)
    centered_indices = np.arange(N) - N // 2 + .5
    x, y = torch.meshgrid(torch.tensor(centered_indices), torch.tensor(centered_indices))
    d = torch.sqrt(x ** 2 + y ** 2)
    return d

def create_centered_square(N: int = 64) -> torch.Tensor:
    """
    Creates a tensor of size [N, N] where the center N/2 x N/2 pixels are set to 1 and all others are 0.

    Args:
    N (int): The size of the tensor's width and height. Default is 64.

    Returns:
    torch.Tensor: A tensor where the middle N/2 x N/2 region is filled with 1s.
    """
    # Validate that N is even to ensure the middle block can be perfectly centered
    if N % 2 != 0:
        raise ValueError("N must be an even number.")

    # Create an N x N tensor of zeros
    tensor = torch.zeros((N, N), dtype=torch.float)

    # Calculate start and end indices for the middle block
    start_idx = N // 4
    end_idx = start_idx + N // 2

    # Set the middle N/2 x N/2 pixels to 1
    tensor[start_idx:end_idx, start_idx:end_idx] = 1

    return tensor

import scipy.ndimage as ndimage

def create_centered_circle(N: int = 64, sigma = 1) -> torch.Tensor:
    """
    Creates a tensor of size [N, N] where a circle with diameter N/2 is centered and its inside is set to 1, all others are 0.
    Applies a Gaussian filter to smooth the edges of the circle.

    Args:
    N (int): The size of the tensor's width and height. Default is 64.
    sigma (float): The sigma value for the Gaussian filter. Default is 1.

    Returns:
    torch.Tensor: A tensor where the central circle with diameter N/2 is filled with 1s and smoothed.
    """
    # Validate that N is even to ensure the circle can be perfectly centered
    if N % 2 != 0:
        raise ValueError("N must be an even number.")

    # Create an N x N tensor of zeros
    tensor = torch.zeros((N, N), dtype=torch.float)

    # Calculate the center of the tensor
    center = N / 2

    # Calculate the radius of the circle (quarter of the tensor size)
    radius = N / 4

    # Iterate over each element in the tensor
    for i in range(N):
        for j in range(N):
            # Calculate the distance from the center
            distance = ((i - center + 0.5)**2 + (j - center + 0.5)**2)**0.5
            # If the distance is less than the radius, set the element to 1
            if distance < radius:
                tensor[i, j] = 1

    # Apply Gaussian filter to smooth the edges of the circle
    smoothed_tensor = torch.from_numpy(ndimage.gaussian_filter(tensor.numpy(), sigma=sigma))

    return smoothed_tensor

params.cfg['N'] = 32
probe = get_default_probe()#[:, :, 0]

##probe_mask_real = (get_squared_distance() < N // 4)[..., None]
##
##def get_probe_mask():
##    probe_mask = torch.tensor(probe_mask_real, dtype=torch.complex64)
##    return probe_mask[..., None]
##
#def set_probe(probe):
#    mask = get_probe_mask().to(probe.dtype)
#    probe_scale = params.get('probe_scale')
#    tamped_probe = mask * probe
#    norm = float(probe_scale * torch.mean(torch.abs(tamped_probe)))
#    params.set('probe', probe / norm)

#def set_probe_guess(X_train, probe_guess=None):
#    N = params.get('N')
#    if probe_guess is None:
#        mu = 0.
#        tmp = X_train.mean(axis=(0, 3))
#        probe_fif = np.absolute(f.fftshift(f.ifft2(f.ifftshift(tmp))))[N // 2, :]
#        d_second_moment = (probe_fif / probe_fif.sum()) * ((np.arange(N) - N // 2) ** 2)
#        probe_sigma_guess = np.sqrt(d_second_moment.sum())
#        probe_guess = np.exp(-((get_squared_distance() - mu) ** 2 / (2.0 * probe_sigma_guess ** 2)))[..., None] + 1e-9
#        probe_guess *= probe_mask_real
#        probe_guess *= (np.sum(get_default_probe()) / np.sum(probe_guess))
#        t_probe_guess = torch.tensor(probe_guess, dtype=torch.float32)
#    else:
#        probe_guess = probe_guess[..., None]
#        t_probe_guess = torch.tensor(probe_guess, dtype=torch.complex64)
#    set_probe(t_probe_guess)
#    return t_probe_guess

#params.set('probe_mask', get_probe_mask())
#
#def set_default_probe():
#    set_probe(get_default_probe())
