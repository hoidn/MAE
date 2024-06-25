import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import torch

def to_float(x):
    return x.item() if isinstance(x, torch.Tensor) else float(x)

def evaluate(model, dataloader, loss_fns, loss_weights=None, device='cuda'):
    model.eval()
    if loss_weights is None:
        loss_weights = [1.0] * len(loss_fns)
    elif len(loss_weights) < len(loss_fns):
        loss_weights = loss_weights + [1.0] * (len(loss_fns) - len(loss_weights))
    
    total_losses = {fn.__name__: 0.0 for fn in loss_fns}
    total_combined_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for (_, img) in dataloader:
            img = img.to(device)
            output = model(img)
            
            batch_combined_loss = 0.0
            for loss_fn, weight in zip(loss_fns, loss_weights):
                loss = loss_fn(output)
                total_losses[loss_fn.__name__] += to_float(loss)
                batch_combined_loss += weight * to_float(loss)
            
            total_combined_loss += batch_combined_loss
            num_batches += 1
    
    avg_losses = {name: total / num_batches for name, total in total_losses.items()}
    avg_combined_loss = total_combined_loss / num_batches
    
    return avg_combined_loss

## TODO update loss
#def evaluate(model, dataloader, mask_ratio, device):
#    model.eval()
#    total_loss = 0
#    with torch.no_grad():
#        for pre_img, diff_img in dataloader:
#            pre_img, diff_img = pre_img.to(device), diff_img.to(device)
#            output = model(diff_img)
#            predicted_img, mask = output['predicted_img'], output['mask']
#            loss = torch.mean((predicted_img - pre_img) ** 2 * mask) / mask_ratio
#            total_loss += loss.item()
#    return total_loss / len(dataloader)

class PreDiffractionDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.pre_files = [f for f in os.listdir(root_dir) if f.startswith('train_pre_') or f.startswith('test_pre_') and f.endswith('.npy')]
        self.diff_files = [f for f in os.listdir(root_dir) if f.startswith('train_diff_') or f.startswith('test_diff_') and f.endswith('.npy')]
        
        if len(self.pre_files) == 0:
            raise ValueError(f"No pre-diffraction .npy files found in {root_dir}")
        if len(self.diff_files) == 0:
            raise ValueError(f"No diffracted .npy files found in {root_dir}")
        if len(self.pre_files) != len(self.diff_files):
            raise ValueError(f"Mismatch in the number of pre-diffraction and diffracted .npy files in {root_dir}")
        
        print(f"Found {len(self.pre_files)} pre-diffraction and {len(self.diff_files)} diffracted .npy files in {root_dir}")
    
    def load_array(self, file_path):
        try:
            array = np.load(file_path)
            return torch.from_numpy(array)
        except Exception as e:
            print(f"Error loading .npy file {file_path}: {e}")
            return None

    def __len__(self):
        return len(self.pre_files)

    def __getitem__(self, idx):
        pre_file_name = os.path.join(self.root_dir, self.pre_files[idx])
        diff_file_name = os.path.join(self.root_dir, self.diff_files[idx])
        
        pre_image = self.load_array(pre_file_name)
        diff_image = self.load_array(diff_file_name)
        
        if pre_image is None or diff_image is None:
            # Return a placeholder or skip this item
            return None
        
        return pre_image, diff_image

def load_datasets_and_dataloaders(train_dir, in_dist_val_dir, out_dist_val_dir, batch_size=128, num_workers=4):
    train_dataset = PreDiffractionDataset(root_dir=train_dir)
    in_dist_val_dataset = PreDiffractionDataset(root_dir=in_dist_val_dir)
    out_dist_val_dataset = PreDiffractionDataset(root_dir=out_dist_val_dir)

    # Custom collate function to handle None values
    def collate_fn(batch):
        batch = list(filter(lambda x: x is not None, batch))
        return torch.utils.data.dataloader.default_collate(batch)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
    in_dist_val_dataloader = DataLoader(in_dist_val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    out_dist_val_dataloader = DataLoader(out_dist_val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

    return train_dataset, in_dist_val_dataset, out_dist_val_dataset, train_dataloader, in_dist_val_dataloader, out_dist_val_dataloader

def mae_mse(inputdict):
    assert 'predicted_amplitude' in inputdict
    assert 'target_amplitude' in inputdict
    assert 'mask' in inputdict
    assert 'mask_ratio' in inputdict
    pred = inputdict['predicted_amplitude']
    target = inputdict['target_amplitude']
#    print('pred mean', pred.mean())
#    print('target mean', target.mean())
    return torch.mean((pred - target) ** 2 * inputdict['mask']) / inputdict['mask_ratio']
