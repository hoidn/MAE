import os
import argparse
import math
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor, Compose, Normalize
from tqdm import tqdm

from model import *
from utils import setup_seed
from common import evaluate, PreDiffractionDataset

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Compose
import os

def load_datasets_and_dataloaders(train_dir, test_dir, batch_size=128, num_workers=4):
    transform = Compose([ToTensor()])

    train_dataset = PreDiffractionDataset(root_dir=train_dir, transform=transform)
    test_dataset = PreDiffractionDataset(root_dir=test_dir, transform=transform)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_dataset, test_dataset, train_dataloader, test_dataloader



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--max_device_batch_size', type=int, default=512)
    parser.add_argument('--base_learning_rate', type=float, default=1.5e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--mask_ratio', type=float, default=0.75)
    parser.add_argument('--total_epoch', type=int, default=2000)
    parser.add_argument('--warmup_epoch', type=int, default=200)
    parser.add_argument('--model_path', type=str, default='vit-t-mae.pt')
    parser.add_argument('--val_interval', type=int, default=1)

    args = parser.parse_args()

    setup_seed(args.seed)

    batch_size = args.batch_size
    load_batch_size = min(args.max_device_batch_size, batch_size)

    assert batch_size % load_batch_size == 0
    steps_per_update = batch_size // load_batch_size

#    train_dataset = torchvision.datasets.CIFAR10('data', train=True, download=True, transform=Compose([ToTensor(), Normalize(0.5, 0.5)]))
#    val_dataset = torchvision.datasets.CIFAR10('data', train=False, download=True, transform=Compose([ToTensor(), Normalize(0.5, 0.5)]))
    train_dataset, val_dataset, dataloader, test_dataloader = load_datasets_and_dataloaders(
        train_dir='bigprobe_images/train',
        test_dir='diffracted_images/test',
        batch_size = batch_size
    )
#    train_dataset = torchvision.datasets.CIFAR10('data', train=True, download=True, transform=Compose([ToTensor()]))
#    val_dataset = torchvision.datasets.CIFAR10('data', train=False, download=True, transform=Compose([ToTensor()]))
#    dataloader = torch.utils.data.DataLoader(train_dataset, load_batch_size, shuffle=True, num_workers=4)

    tboard_name = args.model_path.split('.')[0]
    writer = SummaryWriter(os.path.join('logs', 'vanilla', tboard_name))
    #writer = SummaryWriter(os.path.join('logs', 'cifar10', 'mae-pretrain'))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = MAE_ViT(mask_ratio=args.mask_ratio).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.base_learning_rate * args.batch_size / 256, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    lr_func = lambda epoch: min((epoch + 1) / (args.warmup_epoch + 1e-8), 0.5 * (math.cos(epoch / args.total_epoch * math.pi) + 1))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func, verbose=True)

    step_count = 0
    optim.zero_grad()

    for e in range(args.total_epoch):
        model.train()
        losses = []
        running_loss = 0.0
        for i, (_, diff_img) in enumerate(tqdm(iter(dataloader)), 1):
            step_count += 1
            diff_img = diff_img.to(device)
            predicted_img, mask = model(diff_img)
            loss = torch.mean((predicted_img - diff_img) ** 2 * mask) / args.mask_ratio
            loss.backward()
            if step_count % steps_per_update == 0:
                optim.step()
                optim.zero_grad()
            losses.append(loss.item())

        lr_scheduler.step()
        avg_loss = sum(losses) / len(losses)
        writer.add_scalar('Loss/train', avg_loss, global_step=e)
        print(f'In epoch {e}, average training loss is {avg_loss}.')

        if e % args.val_interval == 0:
            model.eval()
            with torch.no_grad():
                val_loss = evaluate(model, test_dataloader, args.mask_ratio, device)
            writer.add_scalar('Loss/validation', val_loss, global_step=e)
            print(f'In epoch {e}, validation loss is {val_loss}.')

            ''' visualize the first 16 predicted images on val dataset '''
            with torch.no_grad():
                val_pre_img, val_diff_img = zip(*[val_dataset[i] for i in range(16)])
                val_pre_img = torch.stack(val_pre_img).to(device)
                val_diff_img = torch.stack(val_diff_img).to(device)
                predicted_val_img, mask = model.forward(val_diff_img)
                predicted_val_img = predicted_val_img * mask + val_diff_img * (1 - mask)
                img = torch.cat([val_pre_img, val_diff_img, predicted_val_img], dim=0)
                img = rearrange(img, '(v h1 w1) c h w -> c (h1 h) (w1 v w)', w1=3, v=1)
                writer.add_image('MAE Image Comparison', (img + 1) / 2, global_step=e)


    writer.close()
