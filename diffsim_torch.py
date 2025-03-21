import torch.nn.functional as F
from typing import Tuple
import torch
import torch.nn as nn
import torch.fft
from skimage import draw, morphology
import matplotlib.pyplot as plt
import numpy as np
#from ptycho.misc import memoize_disk_and_memory

import params as p
debug = False

def dprint(*args):
    if debug:
        print(*args)

N = p.get('N')

def observe_amplitude(amplitude):
    return torch.sqrt(torch.poisson(amplitude**2))
#def observe_amplitude(amplitude):
#    return torch.sqrt(torch.poisson(amplitude**2).sample())

def count_photons(obj):
    assert len(obj.shape) == 4
    # TODO obj needs to be in NCHW format
    return torch.sum(obj**2, dim=(2, 3))

# TODO in the data generating phase, this should be over the entire dataset
# (not just a batch)
def scale_nphotons(padded_obj):
    mean_photons = torch.mean(count_photons(padded_obj))
    norm = torch.sqrt(p.get('nphotons') / mean_photons)
    return norm

def pad_obj(input: torch.Tensor, h: int, w: int) -> torch.Tensor:
    return nn.ZeroPad2d((h // 4, h // 4, w // 4, w // 4))(input)

def pad_and_diffract(input: torch.Tensor, h: int, w: int, pad: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    print('input shape', input.shape)
    if pad:
        input = pad_obj(input, h, w)
    padded = input
    #assert input.shape[-1] == 1
    input = torch.fft.fft2(input[..., 0].to(torch.complex64))
    input = torch.real(torch.conj(input) * input) / (h * w)
    input = torch.sqrt(torch.fft.fftshift(input, dim=(-2, -1)))
    return padded, input

def combine_complex(amp: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    output = amp.to(torch.complex64) * torch.exp(1j * phi.to(torch.complex64))
    return output

def mk_rand(N):
    return int(N * np.random.uniform())

def mk_lines_img(N=64, nlines=10):
    image = np.zeros((N, N))
    for _ in range(nlines):
        rr, cc = draw.line(mk_rand(N), mk_rand(N), mk_rand(N), mk_rand(N))
        image[rr, cc] = 1
    res = np.zeros((N, N, 3))
    res[:, :, :] = image[..., None]
    return f.gf(res, 1) + 2 * f.gf(res, 5) + 5 * f.gf(res, 10)

def mk_noise(N=64, nlines=10):
    return np.random.uniform(size=N * N).reshape((N, N, 1))

def add_position_jitter(coords, jitter_scale):
    shape = coords.shape
    jitter = jitter_scale * torch.normal(torch.zeros(shape), torch.ones(shape))
    return jitter + coords

import math
def dummy_phi(Y_I):
    return torch.tensor(math.pi) * torch.tanh((Y_I - torch.max(Y_I) / 2) / (3 * torch.mean(Y_I)))

def sim_object_image(size, which='train'):
    if p.get('data_source') == 'lines':
        return mk_lines_img(2 * size, nlines=400)[size // 2: -size // 2, size // 2: -size // 2, :1]
    elif p.get('data_source') == 'grf':
        from .datagen import grf
        return grf.mk_grf(size)
    elif p.get('data_source') == 'points':
        from .datagen import points
        return points.mk_points(size)
    elif p.get('data_source') == 'testimg':
        from .datagen import testimg
        if which == 'train':
            return testimg.get_img(size)
        elif which == 'test':
            return testimg.get_img(size, reverse=True)
        else:
            raise ValueError
    elif p.get('data_source') == 'testimg_reverse':
        from .datagen import testimg
        return testimg.get_img(size, reverse=True)
    elif p.get('data_source') == 'diagonals':
        from .datagen import diagonals
        return diagonals.mk_diags(size)
    elif p.get('data_source') == 'V':
        from .datagen import vendetta
        return vendetta.mk_vs(size)
    else:
        raise ValueError

def pad_and_diffract(input: torch.Tensor, h: int, w: int, pad: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    if pad:
        input = pad_obj(input, h, w)
    padded = input
    input = torch.fft.fft2(input[...].to(torch.complex64))
    input = torch.real(torch.conj(input) * input) / (h * w)
    input = torch.sqrt(torch.fft.fftshift(input, dim=(-2, -1)))
    return padded, input

def diffract_obj(sample, draw_poisson=True):
    N = p.get('N')
    amplitude = pad_and_diffract(sample, N, N, pad=False)[1]
    if draw_poisson:
        observed_amp = observe_amplitude(amplitude)
        return observed_amp
    else:
        return amplitude

def illuminate_and_diffract(Y_complex, probe, intensity_scale=None,
                            draw_poisson = True):
    if intensity_scale is None:
        intensity_scale = scale_nphotons(torch.abs(Y_complex) * probe).item()
    obj_x_probe = Y_complex * probe.to(Y_complex.dtype)
    obj = intensity_scale * obj_x_probe

    X = diffract_obj(obj, draw_poisson = draw_poisson)
    X = X / intensity_scale

    # TODO: return the pre-illumination object as well, or else return the pre-illumination
    # object and multiply by the probe further up the stack
    return Y_complex, X

def map_to_pi(tensor: torch.Tensor) -> torch.Tensor:
    """
    Maps a tensor of float values to the interval [-π, π] using the hyperbolic tangent function.
    """
    assert tensor.dtype == torch.float, "Input tensor must be of type float."
    
    # Apply the hyperbolic tangent function to scale values to [-1, 1]
    scaled_tensor = torch.tanh(tensor)
    
    # Scale and shift the output to fit the [-π, π] interval
    pi_tensor = scaled_tensor * torch.pi
    
    return pi_tensor

def combine_amp_phase(amplitudes: torch.Tensor, phases: torch.Tensor) -> torch.Tensor:
    """
    Returns:
    torch.Tensor: A complex tensor where each element is formed from corresponding amplitude and phase.
    """
    assert amplitudes.dtype == torch.float, "Amplitudes must be a tensor of type float."
    assert phases.dtype == torch.float, "Phases must be a tensor of type float."
    
    # Convert amplitude and phase to real and imaginary components
    real = amplitudes * torch.cos(phases)
    imag = amplitudes * torch.sin(phases)
    
    # Create complex tensor
    complex_tensor = torch.complex(real, imag)
    return complex_tensor

def map_to_unit_interval(tensor: torch.Tensor) -> torch.Tensor:
    """
    Maps a tensor of float values to the interval [0, 1] using the sigmoid function.
    """
    assert tensor.dtype == torch.float, "Input tensor must be of type float."
    # Apply the sigmoid function to scale values to [0, 1]
    sigmoid_tensor = torch.sigmoid(tensor)
    return sigmoid_tensor

def symmetric_zero_pad(tensor: torch.Tensor) -> torch.Tensor:
    """
    Symmetrically zero-pads a 4D tensor from [N, C, size, size] to [N, C, 2*size, 2*size]
    
    Args:
    - tensor (torch.Tensor): Input tensor of shape [N, C, size, size].
    
    Returns:
    - torch.Tensor: Output tensor of shape [N, C, 2*size, 2*size] after padding.
    """
    size = tensor.shape[-1]
    padding_size = size // 2  # Halving the padding size

    # Pad symmetrically on both sides of the last two dimensions
    padded_tensor = F.pad(tensor, (padding_size, padding_size, padding_size, padding_size), mode='constant', value=0)
    
    return padded_tensor

def diffraction_from_channels(batch, probe, intensity_scale=1000., draw_poisson=True, bias=0., pad_before_diffraction=False):
    """
    Simulates the diffraction pattern from channels, with an option to pad Y_complex tensor before diffraction.
    
    Args:
    - batch (torch.Tensor): The input batch tensor.
    - probe (torch.Tensor): The probe tensor.
    - intensity_scale (float): Scaling factor for the intensity.
    - draw_poisson (bool): Flag to simulate Poisson noise.
    - bias (float): Bias to adjust the phase.
    - pad_before_diffraction (bool): Whether to apply symmetric zero-padding to Y_complex.

    Returns:
    - Tuple[torch.Tensor, torch.Tensor]: The illuminated complex tensor and the diffracted batch tensor.
    """
    dprint(f"Input batch shape: {batch.shape}, Data type: {batch.dtype}")

    # Processing intensity and phase channels
    Y_I = torch.nn.functional.softplus(batch[:, 0] - bias)
    Y_phi_input = (batch[:, 1] + batch[:, 2]) / 2
    Y_phi = Y_phi_input  # Phase component

    dprint(f"Y_I shape: {Y_I.shape}, Data type: {Y_I.dtype}")
    dprint(f"Y_phi shape: {Y_phi.shape}, Data type: {Y_phi.dtype}")

    # Combining amplitude and phase to form a complex tensor
    Y_complex = combine_amp_phase(Y_I, Y_phi)

    dprint(f"Y_complex shape: {Y_complex.shape}, Data type: {Y_complex.dtype}")

    # Optional symmetric zero-padding
    if pad_before_diffraction:
        Y_complex = symmetric_zero_pad(Y_complex)
        dprint(f"Padded Y_complex shape: {Y_complex.shape}")

    dprint(f"Probe shape: {probe.shape}, Data type: {probe.dtype}")
    
    # Illumination and diffraction simulation
    Y_complex, X = illuminate_and_diffract(Y_complex, probe, intensity_scale=intensity_scale, draw_poisson=draw_poisson)
    
    dprint(f"Diffracted X shape: {X.shape}, Data type: {X.dtype}")
    
    # Reshaping output to match the expected format (N, C, H, W)
    diffracted_batch = X.view(-1, 1, X.shape[1], X.shape[2]).repeat(1, 3, 1, 1)
    
    dprint(f"Diffracted batch shape: {diffracted_batch.shape}, Data type: {diffracted_batch.dtype}")
    
    return Y_complex, diffracted_batch

def complex_to_channels(complex_data):
    amplitude = torch.abs(complex_data)
    phase = torch.angle(complex_data)
    return torch.stack([amplitude, phase, phase], dim=1)
