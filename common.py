from torchvision.transforms import ToTensor, Compose, Normalize
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torch.utils.data import Dataset
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

class PreDiffractionDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.pre_files = [f for f in os.listdir(root_dir) if f.startswith('train_pre_') or f.startswith('test_pre_')]
        self.diff_files = [f for f in os.listdir(root_dir) if f.startswith('train_diff_')  or f.startswith('test_diff_')]
        
        if len(self.pre_files) == 0:
            raise ValueError(f"No pre-diffraction images found in {root_dir}")
        if len(self.diff_files) == 0:
            raise ValueError(f"No diffracted images found in {root_dir}")
        if len(self.pre_files) != len(self.diff_files):
            raise ValueError(f"Mismatch in the number of pre-diffraction and diffracted images in {root_dir}")
        
        print(f"Found {len(self.pre_files)} pre-diffraction and {len(self.diff_files)} diffracted images in {root_dir}")
    
    def __len__(self):
        return len(self.pre_files)
    
    def __getitem__(self, idx):
        pre_img_name = os.path.join(self.root_dir, self.pre_files[idx])
        diff_img_name = os.path.join(self.root_dir, self.diff_files[idx])
        pre_image = Image.open(pre_img_name).convert("RGB")
        diff_image = Image.open(diff_img_name).convert("RGB")
        if self.transform:
            pre_image = self.transform(pre_image)
            diff_image = self.transform(diff_image)
        return pre_image, diff_image

def load_datasets_and_dataloaders(train_dir, in_dist_val_dir, out_dist_val_dir, batch_size=128, num_workers=4):
    transform = Compose([ToTensor()])

    train_dataset = PreDiffractionDataset(root_dir=train_dir, transform=transform)
    in_dist_val_dataset = PreDiffractionDataset(root_dir=in_dist_val_dir, transform=transform)
    out_dist_val_dataset = PreDiffractionDataset(root_dir=out_dist_val_dir, transform=transform)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    in_dist_val_dataloader = DataLoader(in_dist_val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    out_dist_val_dataloader = DataLoader(out_dist_val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_dataset, in_dist_val_dataset, out_dist_val_dataset, train_dataloader, in_dist_val_dataloader, out_dist_val_dataloader
