from torch.utils.data import Dataset, DataLoader
import os
import torch
import numpy as np

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
        for (_, img, _) in dataloader:
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

class PreDiffractionDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.pre_files = [f for f in os.listdir(root_dir) if f.startswith('train_pre_') or f.startswith('test_pre_') and f.endswith('.npz')]
        self.diff_files = [f for f in os.listdir(root_dir) if f.startswith('train_diff_') or f.startswith('test_diff_') and f.endswith('.npz')]
        
        if len(self.pre_files) == 0:
            raise ValueError(f"No pre-diffraction .npz files found in {root_dir}")
        if len(self.diff_files) == 0:
            raise ValueError(f"No diffracted .npz files found in {root_dir}")
        if len(self.pre_files) != len(self.diff_files):
            raise ValueError(f"Mismatch in the number of pre-diffraction and diffracted .npz files in {root_dir}")
        
        print(f"Found {len(self.pre_files)} pre-diffraction and {len(self.diff_files)} diffracted images in {root_dir}")
    
    def load_array(self, file_path):
        try:
            data = np.load(file_path, allow_pickle=True)
            array = data['array']
            intensity_scale = data['intensity_scale'].item() if 'intensity_scale' in data else None
            probe = data['probe'] if 'probe' in data else None
            return torch.from_numpy(array), intensity_scale, probe
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            return None, None, None

    def __len__(self):
        return len(self.pre_files)

    def __getitem__(self, idx):
        pre_file_name = os.path.join(self.root_dir, self.pre_files[idx])
        diff_file_name = os.path.join(self.root_dir, self.diff_files[idx])
        
        pre_image, _, _ = self.load_array(pre_file_name)
        diff_image, intensity_scale, probe = self.load_array(diff_file_name)
        
        if pre_image is None or diff_image is None:
            return None
        
        if probe is not None:
            probe = torch.from_numpy(probe).float()
        
        return pre_image, diff_image, intensity_scale, probe

def load_datasets_and_dataloaders(train_dir, in_dist_val_dir, out_dist_val_dir, batch_size=128, num_workers=4, use_probe=False):
    """
    Load datasets and create dataloaders for training and validation.

    Args:
        train_dir (str): Directory containing training data
        in_dist_val_dir (str): Directory containing in-distribution validation data
        out_dist_val_dir (str): Directory containing out-of-distribution validation data
        batch_size (int): Batch size for dataloaders
        num_workers (int): Number of worker processes for data loading
        use_probe (bool): Whether to use probe data from the dataset

    Returns:
        dict: A dictionary containing the following key-value pairs:
            - 'train_dataset': PreDiffractionDataset for training data
            - 'in_dist_val_dataset': PreDiffractionDataset for in-distribution validation data
            - 'out_dist_val_dataset': PreDiffractionDataset for out-of-distribution validation data
            - 'train_dataloader': DataLoader for training data
            - 'in_dist_val_dataloader': DataLoader for in-distribution validation data
            - 'out_dist_val_dataloader': DataLoader for out-of-distribution validation data
            - 'probe': Probe data if use_probe is True and probe data is available, else None
    """
    train_dataset = PreDiffractionDataset(root_dir=train_dir)
    in_dist_val_dataset = PreDiffractionDataset(root_dir=in_dist_val_dir)
    out_dist_val_dataset = PreDiffractionDataset(root_dir=out_dist_val_dir)

    if use_probe:
        first_data_point = train_dataset[1]
        probe = first_data_point[-1] if first_data_point is not None else None
    else:
        probe = None

    def collate_fn(batch):
        batch = list(filter(lambda x: x is not None, batch))
        if len(batch) == 0:
            return None
        pre_images, diff_images, intensity_scales, _ = zip(*batch)
        pre_images = torch.stack(pre_images)
        diff_images = torch.stack(diff_images)
        intensity_scales = torch.tensor(intensity_scales)
        return pre_images, diff_images, intensity_scales

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
    in_dist_val_dataloader = DataLoader(in_dist_val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    out_dist_val_dataloader = DataLoader(out_dist_val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

    return {
        'train_dataset': train_dataset,
        'in_dist_val_dataset': in_dist_val_dataset,
        'out_dist_val_dataset': out_dist_val_dataset,
        'train_dataloader': train_dataloader,
        'in_dist_val_dataloader': in_dist_val_dataloader,
        'out_dist_val_dataloader': out_dist_val_dataloader,
        'probe': probe
    }
