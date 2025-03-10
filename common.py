from torch.utils.data import Dataset, DataLoader
import os
import torch
import numpy as np
from typing import Optional, Tuple, List, Union

def to_float(x: Union[torch.Tensor, float]) -> float:
    return x.item() if isinstance(x, torch.Tensor) else float(x)

def evaluate(model: torch.nn.Module, dataloader: DataLoader, loss_fns: List[callable], loss_weights: Optional[List[float]] = None, device: str = 'cuda') -> float:
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
    def __init__(self, root_dir: str, transform: Optional[callable] = None):
        self.root_dir = root_dir
        self.data_files = [f for f in os.listdir(root_dir) if f.endswith('.npz')]
        
        if len(self.data_files) == 0:
            raise ValueError(f"No .npz files found in {root_dir}")
        
        print(f"Found {len(self.data_files)} data files in {root_dir}")
    
    def load_array(self, file_path: str) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[float], Optional[torch.Tensor], Optional[np.ndarray]]:
        try:
            data = np.load(file_path, allow_pickle=True)
            pre_array = data['pre_array']
            diff_array = data['diff_array']
            intensity_scale = data['intensity_scale'].item()
            probe = data['probe']
            coords = data['coords'] if 'coords' in data else None
            return torch.from_numpy(pre_array), torch.from_numpy(diff_array), intensity_scale, torch.from_numpy(probe), coords
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            return None, None, None, None, None

    def __len__(self) -> int:
        return len(self.data_files)

    def __getitem__(self, idx: int) -> Optional[Tuple[torch.Tensor, torch.Tensor, float, torch.Tensor, Optional[np.ndarray]]]:
        file_name = os.path.join(self.root_dir, self.data_files[idx])
        
        pre_image, diff_image, intensity_scale, probe, coords = self.load_array(file_name)
        
        if pre_image is None or diff_image is None:
            return None
        
        return pre_image, diff_image, intensity_scale, probe, coords

def load_datasets_and_dataloaders(train_dir: str, in_dist_val_dir: str, out_dist_val_dir: str, batch_size: int = 128, num_workers: int = 4, use_probe: bool = False) -> dict:
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
        probe = first_data_point[3] if first_data_point is not None else None
    else:
        probe = None

    def collate_fn(batch: List[Optional[Tuple[torch.Tensor, torch.Tensor, float, torch.Tensor, Optional[np.ndarray]]]]) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[List[Optional[np.ndarray]]]]]:
        batch = list(filter(lambda x: x is not None, batch))
        if len(batch) == 0:
            return None
        pre_images, diff_images, intensity_scales, probes, coords = zip(*batch)
        pre_images = torch.stack(pre_images)
        diff_images = torch.stack(diff_images)
        intensity_scales = torch.tensor(intensity_scales)
        if use_probe:
            probes = torch.stack(probes)
        else:
            probes = None
        return pre_images, diff_images, intensity_scales#, probes, coords

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
