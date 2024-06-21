from typing import Tuple, Optional, Union, Callable, Any
import torch
import torch.nn as nn
import torch.fft
from skimage import draw, morphology
import matplotlib.pyplot as plt
import numpy as np

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
#def pad_and_diffract(input: torch.Tensor, h: int, w: int, pad: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
#    print('input shape', input.shape)
#    if pad:
#        input = pad_obj(input, h, w)
#    padded = input
#    #assert input.shape[-1] == 1
#    input = torch.fft.fft2(input[..., 0].to(torch.complex64))
#    input = torch.real(torch.conj(input) * input) / (h * w)
#    input = torch.sqrt(torch.fft.fftshift(input, (-2, -1))).unsqueeze(-1)
#    return padded, input


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

#from ptycho.misc import memoize_disk_and_memory

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
    #print('input shape', input.shape)
    if pad:
        input = pad_obj(input, h, w)
        #print('After padding:', input.shape)
    padded = input
    input = torch.fft.fft2(input[...].to(torch.complex64))
    #print('After fft2:', input.shape)
    input = torch.real(torch.conj(input) * input) / (h * w)
    #print('After element-wise multiplication and division:', input.shape)
    input = torch.sqrt(torch.fft.fftshift(input, dim=(-2, -1)))
    #print('After fftshift and sqrt:', input.shape)
    return padded, input

def diffract_obj(sample, draw_poisson=True):
    N = p.get('N')
    amplitude = pad_and_diffract(sample, N, N, pad=False)[1]
    #print('After pad_and_diffract:', amplitude.shape)
    if draw_poisson:
        observed_amp = observe_amplitude(amplitude)
        #print('After observe_amplitude:', observed_amp.shape)
        return observed_amp
    else:
        return amplitude

def illuminate_and_diffract(Y_complex, probe, intensity_scale=None,
                            draw_poisson = True):
    if intensity_scale is None:
        intensity_scale = scale_nphotons(torch.abs(Y_complex) * probe).item()
    obj = intensity_scale * Y_complex
    #print('After intensity scaling:', obj.shape)
    obj = obj * probe.to(obj.dtype)
    #print('After probe multiplication:', obj.shape)

    X = diffract_obj(obj, draw_poisson = draw_poisson)
    #print('After diffract_obj:', X.shape)
    X = X / intensity_scale
    #print('After intensity scaling:', X.shape)

    return X

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



def diffraction_from_channels(batch, probe, intensity_scale = 1000.,
                              draw_poisson = True):
    dprint(f"Input batch shape: {batch.shape}, Data type: {batch.dtype}")

    Y_I = map_to_unit_interval(batch[:, 0])# + batch[:, 2]) / 2  # Calculate Y_phi as the average of the second and third channels
    Y_phi_input = (batch[:, 1] + batch[:, 2]) / 2
    Y_phi = Y_phi_input #* allow phase wrapping instead of squashing with tanh
    #Y_phi = map_to_pi(Y_phi_input)

#    Y_I = (batch[:, 0]  + batch[:, 1] + batch[:, 2]) / 3 
#    Y_phi = torch.zeros_like(Y_I)#(batch[:, 1])# + batch[:, 2]) / 2  # Calculate Y_phi as the average of the second and third channels

    dprint(f"Y_I shape: {Y_I.shape}, Data type: {Y_I.dtype}")
    dprint(f"Y_phi shape: {Y_phi.shape}, Data type: {Y_phi.dtype}")
    
#    # Create a complex tensor by combining Y_I and Y_phi
#    Y_complex = torch.complex(Y_I, Y_phi)
    Y_complex = combine_amp_phase(Y_I, Y_phi)
    
    dprint(f"Y_complex shape: {Y_complex.shape}, Data type: {Y_complex.dtype}")

    dprint(f"Probe shape: {probe.shape}, Data type: {probe.dtype}")
    
    # Apply the illuminate_and_diffract() function
    X = illuminate_and_diffract(Y_complex, probe, intensity_scale= intensity_scale,
                                draw_poisson=draw_poisson)
    
    dprint(f"Diffracted X shape: {X.shape}, Data type: {X.dtype}")
    
    # Reshape the output to match the expected shape (N, C, H, W)
    diffracted_batch = X.view(-1, 1, X.shape[1], X.shape[2]).repeat(1, 3, 1, 1)
    
    dprint(f"Diffracted batch shape: {diffracted_batch.shape}, Data type: {diffracted_batch.dtype}")
    
    return diffracted_batch # diffracted amplitude
