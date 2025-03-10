import numpy as np
import os
import torch
import torchvision
from torchvision.transforms import ToTensor, Compose
from torch.utils.tensorboard import SummaryWriter
from diffsim_torch import diffraction_from_channels, symmetric_zero_pad
from probe_torch import get_default_probe
from synthetic_lines import (
    create_synthetic_lines_dataloader
)

intensity_scale = 1000.

def save_array(pre_array, diff_array, save_path, intensity_scale, probe, coords=None):
    """
    Save pre_array, diff_array, intensity_scale, probe, and optionally coords to a single .npz file.
    """
    if coords is not None:
        np.savez(save_path, pre_array=pre_array.astype('float32'), diff_array=diff_array.astype('float32'), intensity_scale=intensity_scale, probe=probe.astype('float32'), coords=coords)
    else:
        np.savez(save_path, pre_array=pre_array.astype('float32'), diff_array=diff_array.astype('float32'), intensity_scale=intensity_scale, probe=probe.astype('float32'))

def generate_datasets(intensity_scale=intensity_scale, probe_scale=0.55, N = 32, use_synthetic_lines=False, num_objects=10, object_size=392, num_lines=10):
    if N == 32:
        pad_before_diffraction=False
    else:
        pad_before_diffraction=True
    save_dir_train = 'diffracted_images/train'
    save_dir_test = 'diffracted_images/test'
    os.makedirs(save_dir_train, exist_ok=True)
    os.makedirs(save_dir_test, exist_ok=True)

    # Dataset and DataLoader
    batch_size = 128
    load_batch_size = min(512, batch_size)

    if use_synthetic_lines:
        print("Using synthetic lines dataset")
        train_loader = create_synthetic_lines_dataloader(batch_size=load_batch_size, num_objects=num_objects, object_size=object_size, num_lines=num_lines, patch_size = N // (2 if pad_before_diffraction else 1))
        test_loader = create_synthetic_lines_dataloader(batch_size=load_batch_size, num_objects=num_objects // 2, object_size=object_size, num_lines=num_lines, patch_size = N // (2 if pad_before_diffraction else 1))
    else:
        print("Using CIFAR10 dataset")
        # Train dataset and DataLoader
        transform = Compose([ToTensor()])
        train_dataset = torchvision.datasets.CIFAR10('data', train=True, download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, load_batch_size, shuffle=False, num_workers=4)

        # Test dataset and DataLoader
        test_dataset = torchvision.datasets.CIFAR10('data', train=False, download=True, transform=transform)
        test_loader = torch.utils.data.DataLoader(test_dataset, load_batch_size, shuffle=False, num_workers=4)

    # SummaryWriter
    writer = SummaryWriter(os.path.join('logs', 'cifar10', 'mae-pretrain'))

    probe = get_default_probe(probe_scale=probe_scale, N = N)
    #probe = torch.tensor(np.load('xprobe0.npz.npy')) / 5

    def process_and_save(dataloader, save_dir, phase):
        for batch_idx, batch_data in enumerate(dataloader):
            if use_synthetic_lines:
                pre_diffraction_batch, centers = batch_data
            else:
                batch, _ = batch_data
                pre_diffraction_batch = (batch + torch.flip(batch, dims=[2, 3])) / 2
                pre_diffraction_batch[:, 1:3, :, :] -= 0.5
                centers = None  # No centers for CIFAR10 data

                # Average G and B channels
                pre_diffraction_batch[:, 1, :, :] = (pre_diffraction_batch[:, 1, :, :] + pre_diffraction_batch[:, 2, :, :]) / 2
                pre_diffraction_batch[:, 2, :, :] = pre_diffraction_batch[:, 1, :, :]
            
            _, diffracted_batch = diffraction_from_channels(pre_diffraction_batch, probe, intensity_scale=intensity_scale, pad_before_diffraction=pad_before_diffraction)
            print(f"Processing {phase} batch {batch_idx + 1}")
            print(f"Probe mean: {probe.mean().item()}")
            for i, (pre_img, diff_img) in enumerate(zip(pre_diffraction_batch, diffracted_batch)):
                save_path = os.path.join(save_dir, f'{phase}_{batch_idx}_{i}.npz')
                pre_array = torch.abs((probe * symmetric_zero_pad(pre_img)).cpu()).numpy()
                diff_array = diff_img.cpu().numpy()
                coords = centers[i].cpu().numpy() if centers is not None else None
                save_array(pre_array, diff_array, save_path, intensity_scale, probe.cpu().numpy(), coords)
            
            writer.add_images(f'Pre-diffraction Images/{phase}', pre_diffraction_batch, batch_idx, dataformats='NCHW')
            writer.add_images(f'Diffracted Images/{phase}', diffracted_batch, batch_idx, dataformats='NCHW')
        
        return pre_diffraction_batch, diffracted_batch, centers

    # Process and save train data
    pre_diffraction_batch, diffracted_batch, centers = process_and_save(train_loader, save_dir_train, 'train')

    # Process and save test data
    pre_diffraction_batch, diffracted_batch, centers = process_and_save(test_loader, save_dir_test, 'test')

    writer.close()
    return probe, pre_diffraction_batch, diffracted_batch, centers

# Execute the generate_datasets function when the script is run directly
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate datasets for training and testing")
    parser.add_argument('--use_synthetic_lines', action='store_true', help='Use synthetic lines objects instead of CIFAR10')
    parser.add_argument('--num_objects', type=int, default=2, help='Number of synthetic lines objects to generate')
    parser.add_argument('--object_size', type=int, default=392, help='Size of each synthetic lines object')
    parser.add_argument('--num_lines', type=int, default=400, help='Number of lines in each synthetic object')
    args = parser.parse_args()

    probe, pre_diffraction_batch, diffracted_batch, centers = generate_datasets(
        intensity_scale=intensity_scale,
        probe_scale=0.55,
        use_synthetic_lines=args.use_synthetic_lines,
        num_objects=args.num_objects,
        object_size=args.object_size,
        num_lines=args.num_lines,
        N = 32
    )
