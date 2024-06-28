import os
import argparse
import math
import torch
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision.transforms import ToTensor, Compose, Normalize
from tqdm import tqdm
from einops import rearrange

from model_diff import MAE_ViT
from utils import setup_seed
from common import evaluate, load_datasets_and_dataloaders
from visualization import cat_images, vscale_tensor, visualize_realspace
from probe_torch import create_centered_circle, create_centered_square
from probe_torch import get_default_probe
from produce_dataset import probe_scale

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
    parser.add_argument('--input_size', type=int, default=32, help='Size of the input images')

    args = parser.parse_args()
    intensity_scale = 1000.
    N = args.input_size
    #probe = create_centered_square(N=N)
    probe = create_centered_circle(N=N)

    setup_seed(args.seed)

    batch_size = args.batch_size
    load_batch_size = min(args.max_device_batch_size, batch_size)

    assert batch_size % load_batch_size == 0
    steps_per_update = batch_size // load_batch_size

    train_dataset, in_dist_val_dataset, out_dist_val_dataset, dataloader, in_dist_val_dataloader, out_dist_val_dataloader = load_datasets_and_dataloaders(
        train_dir='bigprobe_images/train',
        in_dist_val_dir='bigprobe_images/test',
        out_dist_val_dir='diffracted_images/test',
        batch_size=batch_size
    )

    tboard_name = args.model_path.split('.')[0]
    writer = SummaryWriter(os.path.join('logs', 'pinn', tboard_name))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    probe = probe.to(device)

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
                in_dist_val_loss = evaluate(model, in_dist_val_dataloader, args.mask_ratio, device)
                out_dist_val_loss = evaluate(model, out_dist_val_dataloader, args.mask_ratio, device)
            writer.add_scalar('Loss/in_dist_validation', in_dist_val_loss, global_step=e)
            writer.add_scalar('Loss/out_dist_validation', out_dist_val_loss, global_step=e)
            print(f'In epoch {e}, in-distribution validation loss is {in_dist_val_loss}, out-of-distribution validation loss is {out_dist_val_loss}.')

            ''' visualize the first 16 predicted images on in-distribution val dataset '''
            with torch.no_grad():
                val_pre_img, val_diff_img = zip(*[in_dist_val_dataset[i] for i in range(16)])
                val_pre_img = torch.stack(val_pre_img).to(device)
                val_diff_img = torch.stack(val_diff_img).to(device)
                predicted_val_img, mask, intermediate_img = model.forward_with_intermediate(val_diff_img)
                outputs_dict = {
                    'predicted_amplitude': predicted_val_img,
                    'intermediate_img': intermediate_img,
                    'mask': mask,
                    'probe': get_default_probe(probe_scale=probe_scale).to(device)
                }
                img, ncat = cat_images(val_pre_img,
                                       # (val_diff_img.sqrt() / outputs['intensity_scale'])
                                       val_diff_img,
                                       outputs_dict, args, device)
                img = rearrange(img, '(v h1 w1) c h w -> c (h1 h) (w1 v w)', w1=1, v=ncat)
                writer.add_image('In-dist MAE Image Comparison', img, global_step=e)

            ''' visualize the first 16 predicted images on out-of-distribution val dataset '''
            with torch.no_grad():
                val_pre_img, val_diff_img = zip(*[out_dist_val_dataset[i] for i in range(16)])
                val_pre_img = torch.stack(val_pre_img).to(device)
                val_diff_img = torch.stack(val_diff_img).to(device)
                predicted_val_img, mask, intermediate_img = model.forward_with_intermediate(val_diff_img)
                outputs_dict = {
                    'predicted_amplitude': predicted_val_img,
                    'intermediate_img': intermediate_img,
                    'mask': mask,
                    'probe': get_default_probe(probe_scale=probe_scale).to(device)
                }
                img, ncat = cat_images(val_pre_img,
                                       # (val_diff_img.sqrt() / outputs['intensity_scale'])
                                       val_diff_img,
                                       outputs_dict, args, device)
                img = rearrange(img, '(v h1 w1) c h w -> c (h1 h) (w1 v w)', w1=1, v=ncat)
                writer.add_image('Out-dist MAE Image Comparison', img, global_step=e)

        model.save_model(args.model_path, probe)

    writer.close()
