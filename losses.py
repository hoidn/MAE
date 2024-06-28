import torch

def mae_mse(inputdict):
    assert 'predicted_amplitude' in inputdict
    assert 'target_amplitude' in inputdict
    assert 'mask' in inputdict
    assert 'mask_ratio' in inputdict
    pred = inputdict['predicted_amplitude']
    target = inputdict['target_amplitude']
    mask_ratio = inputdict['mask_ratio']
    
    if mask_ratio > 0:
        return torch.mean((pred - target) ** 2 * inputdict['mask']) / mask_ratio
    else:
        return torch.mean((pred - target) ** 2)

def mae_mae(inputdict):
    assert 'predicted_amplitude' in inputdict
    assert 'target_amplitude' in inputdict
    assert 'mask' in inputdict
    assert 'mask_ratio' in inputdict
    pred = inputdict['predicted_amplitude']
    target = inputdict['target_amplitude']
    mask_ratio = inputdict['mask_ratio']
    
    if mask_ratio > 0:
        return torch.mean(torch.abs(pred - target) * inputdict['mask']) / mask_ratio
    else:
        return torch.mean(torch.abs(pred - target))
