import os
import torch
import torchvision
from torchvision.transforms import ToTensor, Compose
from torch.utils.tensorboard import SummaryWriter
from diffsim_torch import illuminate_and_diffract
import torchvision.utils as vutils
from diffsim_torch import diffraction_from_channels
from torch_probe import probe

# Parameters
intensity_scale = 1000.
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


def process_and_save(dataloader, save_dir, writer, phase):
    for batch_idx, (batch, _) in enumerate(dataloader):
        batch = (batch + torch.flip(batch, dims=[2, 3])) / 2
        diffracted_batch = diffraction_from_channels(batch, probe)
        print(f"Processing {phase} batch {batch_idx + 1}")
        for i, img in enumerate(diffracted_batch):
            vutils.save_image(img, os.path.join(save_dir, f'{phase}_{batch_idx}_{i}.png'))
        writer.add_images(f'Diffracted Images/{phase}', diffracted_batch, batch_idx, dataformats='NCHW')
    return batch, diffracted_batch

# Process and save train data
process_and_save(train_loader, save_dir_train, writer, 'train')

# Process and save test data
batch, diffracted_batch = process_and_save(test_loader, save_dir_test, writer, 'test')

writer.close()

