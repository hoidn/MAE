import numpy as np
import os
import torch
import torchvision
from torchvision.transforms import ToTensor, Compose
from torch.utils.tensorboard import SummaryWriter
from diffsim_torch import diffraction_from_channels
from probe_torch import get_default_probe

intensity_scale = 10.

def save_array(array, save_path):
    """
    Save a numpy array to a .npy file.
    """
    np.save(save_path, array)

def generate_datasets(intensity_scale=intensity_scale, probe_scale=0.55):
    """
    Encapsulates the logic for generating training and test datasets.
    
    Args:
        intensity_scale (float): Scale factor for the diffracted image intensity. Default is 1000.
        probe_scale (float): Scale factor for the probe. Default is 0.55.
    """
    save_dir_train = 'diffracted_images/train'
    save_dir_test = 'diffracted_images/test'
    os.makedirs(save_dir_train, exist_ok=True)
    os.makedirs(save_dir_test, exist_ok=True)

    # Dataset and DataLoader
    batch_size = 128
    load_batch_size = min(512, batch_size)
    transform = Compose([ToTensor()])

    # Train dataset and DataLoader
    train_dataset = torchvision.datasets.CIFAR10('data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, load_batch_size, shuffle=False, num_workers=4)

    # Test dataset and DataLoader
    test_dataset = torchvision.datasets.CIFAR10('data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, load_batch_size, shuffle=False, num_workers=4)

    # SummaryWriter
    writer = SummaryWriter(os.path.join('logs', 'cifar10', 'mae-pretrain'))

    # Device setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    probe = get_default_probe(probe_scale=probe_scale)
    
    def process_and_save(dataloader, save_dir, phase):
        for batch_idx, (batch, _) in enumerate(dataloader):
            pre_diffraction_batch = (batch + torch.flip(batch, dims=[2, 3])) / 2
            pre_diffraction_batch[:, 1:3, :, :] -= 0.5
            diffracted_batch = diffraction_from_channels(pre_diffraction_batch, probe, intensity_scale=intensity_scale)**2 # convert to intensity

            print(f"Processing {phase} batch {batch_idx + 1}")
            print(f"Probe mean: {probe.mean().item()}")
            for i, (pre_img, diff_img) in enumerate(zip(pre_diffraction_batch, diffracted_batch)):
                pre_save_path = os.path.join(save_dir, f'{phase}_pre_{batch_idx}_{i}.npy')
                diff_save_path = os.path.join(save_dir, f'{phase}_diff_{batch_idx}_{i}.npy')
                save_array(pre_img.cpu().numpy(), pre_save_path)
                save_array(diff_img.cpu().numpy(), diff_save_path)
            writer.add_images(f'Pre-diffraction Images/{phase}', pre_diffraction_batch, batch_idx, dataformats='NCHW')
            writer.add_images(f'Diffracted Images/{phase}', diffracted_batch, batch_idx, dataformats='NCHW')
        return pre_diffraction_batch, diffracted_batch

    # Process and save train data
    pre_diffraction_batch, diffracted_batch = process_and_save(train_loader, save_dir_train, 'train')

    # Process and save test data
    pre_diffraction_batch, diffracted_batch = process_and_save(test_loader, save_dir_test, 'test')

    writer.close()
    return probe, pre_diffraction_batch, diffracted_batch

# Execute the generate_datasets function when the script is run directly
if __name__ == "__main__":
    probe, pre_diffraction_batch, diffracted_batch = generate_datasets(intensity_scale=intensity_scale, probe_scale=0.55)
    print("Dataset generation complete.")
