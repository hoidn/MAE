import os
import argparse
import math
from typing import Dict, Any, List, Tuple, Optional

import torch
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from einops import rearrange

from model_diff import MAE_ViT
from utils import setup_seed
from common import evaluate, load_datasets_and_dataloaders
from losses import mae_mse, mae_mae
from torch.distributions import Poisson

from probe_torch import create_centered_circle, get_default_probe
from visualization import cat_images

poisson_inflation = 0.1

def poisson_distribution(rate: torch.Tensor) -> Poisson:
    """Create a Poisson distribution with the given rate."""
    return Poisson(rate + poisson_inflation)

def negative_log_likelihood(inputdict: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Calculate the negative log-likelihood loss."""
    assert all(key in inputdict for key in ['predicted_amplitude', 'target_amplitude', 'intensity_scale']), \
        "Missing required keys in input dictionary"
    
    pred_intensity = (inputdict['predicted_amplitude'] * inputdict['intensity_scale'])**2
    poisson_dist = poisson_distribution(pred_intensity)
    log_prob = poisson_dist.log_prob(inputdict['target_amplitude'].to(torch.int64))
    return -torch.sum(log_prob)

def validate_and_visualize(model: MAE_ViT, 
                           val_dataset: Dataset, 
                           val_dataloader: DataLoader, 
                           writer: SummaryWriter, 
                           device: torch.device, 
                           args: argparse.Namespace, 
                           epoch: int, 
                           prefix: str,
                           probe: torch.Tensor) -> float:
    """Validate the model and visualize results."""
    model.eval()
    with torch.no_grad():
        val_loss = evaluate(model, val_dataloader, [mae_mae], [1.])
    writer.add_scalar(f'Loss/{prefix}_validation', val_loss, global_step=epoch)
    print(f'In epoch {epoch}, {prefix} validation loss is {val_loss}.')

    # Visualize the first 16 predicted images
    with torch.no_grad():
        val_data = [val_dataset[i] for i in range(min(16, len(val_dataset)))]
        if not val_data:
            print(f"Warning: {prefix} validation dataset is empty.")
            return val_loss
        
        val_pre_img, val_diff_img, _, val_probe, _ = zip(*val_data)
        # TODO: in the future, the model's forward method should take val_probe
        # as an explicit argument. This will allow associating a different illumination
        # with each sample. We will have to add a 'global_probe' parameter for backwards
        # compatibility with the current behavior, which is to use the same probe 
        # throughout the whole training instance 

        val_pre_img = torch.stack(val_pre_img).to(device)
        val_diff_img = torch.stack(val_diff_img).to(device)
        outputs = model(val_diff_img)
        outputs['probe'] = probe  # Add probe to outputs
        img, ncat = cat_images(val_pre_img, val_diff_img, outputs, args, device)
        img = rearrange(img, '(v h1 w1) c h w -> c (h1 h) (w1 v w)', w1=1, v=ncat)
        writer.add_image(f'{prefix} MAE Image Comparison', img, global_step=epoch)
        return val_loss

    return val_loss

def main(args: argparse.Namespace) -> None:
    """Main function to run the training process."""
    intensity_scale = 1000.
    N = args.input_size
    probe_scale = 0.5

    setup_seed(args.seed)

    batch_size = args.batch_size
    load_batch_size = min(args.max_device_batch_size, batch_size)

    assert batch_size % load_batch_size == 0, "Batch size must be divisible by max_device_batch_size"
    steps_per_update = batch_size // load_batch_size

    data = load_datasets_and_dataloaders(
        train_dir='bigprobe_images/train',
        in_dist_val_dir='bigprobe_images/test',
        out_dist_val_dir='diffracted_images/test',
        batch_size=batch_size,
        use_probe=True
    )

    # Validate that all expected keys are present in the data dictionary
    expected_keys = ['train_dataset', 'in_dist_val_dataset', 'out_dist_val_dataset',
                     'train_dataloader', 'in_dist_val_dataloader', 'out_dist_val_dataloader', 'probe']
    assert all(key in data for key in expected_keys), f"Missing keys in data dictionary. Expected: {expected_keys}"

    # Use dictionary unpacking for more concise data assignment
    train_dataset: Dataset
    in_dist_val_dataset: Dataset
    out_dist_val_dataset: Dataset
    train_dataloader: DataLoader
    in_dist_val_dataloader: DataLoader
    out_dist_val_dataloader: DataLoader
    probe_data: Optional[torch.Tensor]
    
    train_dataset, in_dist_val_dataset, out_dist_val_dataset, train_dataloader, in_dist_val_dataloader, out_dist_val_dataloader, probe_data = \
        [data.get(key) for key in expected_keys]

    # Check if dataloaders are None or empty
    if train_dataloader is None or len(train_dataloader) == 0:
        raise ValueError("Training dataloader is empty or None")

    tboard_name = args.model_path.split('.')[0]
    writer = SummaryWriter(os.path.join('logs', 'pinn', tboard_name))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if probe_data is None:
        print('INFO:', 'no probe found in dataset')
    if args.use_probe and probe_data is not None:
        probe = probe_data.to(device)
        print('INFO:', 'using probe from dataset')
    else:
        probe = create_centered_circle(N=N).to(device)
        print('INFO:', 'using generic / uninformative probe')
        #probe = get_default_probe(probe_scale=probe_scale).to(device)

    model = MAE_ViT(mask_ratio=args.mask_ratio, intensity_scale=intensity_scale, probe=probe,
                    input_size=args.input_size).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.base_learning_rate * args.batch_size / 256, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    lr_func = lambda epoch: min((epoch + 1) / (args.warmup_epoch + 1e-8), 0.5 * (math.cos(epoch / args.total_epoch * math.pi) + 1))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func, verbose=True)

    step_count = 0
    optim.zero_grad()

    for e in range(args.total_epoch):
        model.train()
        losses = []
        for i, data_batch in enumerate(tqdm(train_dataloader), 1):
            step_count += 1
            if data_batch is None:
                continue
            _, diff_img, _ = data_batch
            diff_img = diff_img.to(device)
            outputs = model(diff_img)
            mae_mse_loss = mae_mse(outputs)
            total_loss = mae_mse_loss
#            mae_mae_loss = mae_mae(outputs)
#            total_loss = mae_mae_loss
            total_loss.backward()
            if step_count % steps_per_update == 0:
                optim.step()
                optim.zero_grad()
            losses.append(total_loss.item())

        lr_scheduler.step()
        avg_loss = sum(losses) / len(losses) if losses else 0
        writer.add_scalar('Loss/train', avg_loss, global_step=e)
        print(f'In epoch {e}, average training loss is {avg_loss}.')

        if e % args.val_interval == 0:
            in_dist_val_loss = validate_and_visualize(model, in_dist_val_dataset, in_dist_val_dataloader, writer, device, args, e, 'In_dist', probe)
            out_dist_val_loss = validate_and_visualize(model, out_dist_val_dataset, out_dist_val_dataloader, writer, device, args, e, 'Out_dist', probe)

        model.save_model(args.model_path)

    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training script for MAE model")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument('--batch_size', type=int, default=4096, help="Batch size for training")
    parser.add_argument('--max_device_batch_size', type=int, default=512, help="Maximum batch size for device")
    parser.add_argument('--base_learning_rate', type=float, default=1.5e-4, help="Base learning rate")
    parser.add_argument('--weight_decay', type=float, default=0.05, help="Weight decay for optimizer")
    parser.add_argument('--mask_ratio', type=float, default=0.75, help="Masking ratio for MAE")
    parser.add_argument('--total_epoch', type=int, default=2000, help="Total number of training epochs")
    parser.add_argument('--warmup_epoch', type=int, default=200, help="Number of warmup epochs")
    parser.add_argument('--model_path', type=str, default='vit-t-mae.pt', help="Path to save the model")
    parser.add_argument('--val_interval', type=int, default=1, help="Interval for validation")
    parser.add_argument('--input_size', type=int, default=32, help='Size of the input images')
    parser.add_argument('--use_probe', action='store_true', help='Use probe from data')

    args = parser.parse_args()
    main(args)
