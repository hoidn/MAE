import torch

def mae_mse(inputdict):
    assert 'predicted_amplitude' in inputdict
    assert 'target_amplitude' in inputdict
    assert 'mask' in inputdict
    assert 'mask_ratio' in inputdict
    pred = inputdict['predicted_amplitude']
    target = inputdict['target_amplitude']
    return torch.mean((pred - target) ** 2 * inputdict['mask']) / inputdict['mask_ratio']

def mae_mae(inputdict):
    assert 'predicted_amplitude' in inputdict
    assert 'target_amplitude' in inputdict
    assert 'mask' in inputdict
    assert 'mask_ratio' in inputdict
    pred = inputdict['predicted_amplitude']
    target = inputdict['target_amplitude']
    return torch.mean(torch.abs((pred - target)) * inputdict['mask']) / inputdict['mask_ratio']
