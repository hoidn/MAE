### Incremental work on top of [*KaiMing He el.al. Masked Autoencoders Are Scalable Vision Learners*](https://arxiv.org/abs/2111.06377).

# Diffraction-Aware Masked Autoencoder (Diff-MAE)

This repository contains implementations of diffraction-aware Masked Autoencoders (MAE) for phase retrieval and diffraction imaging tasks. The code builds on the Vision Transformer (ViT) architecture with specialized components for diffraction physics simulation and inverse problem solving.

## Features

### Core Components

- **Diffraction-Aware MAE**: Enhanced Masked Autoencoder architecture that incorporates diffraction physics into the reconstruction process
- **Physics-Informed Neural Networks**: Models that respect the underlying physics of light diffraction
- **Mixed Precision Training**: Efficient training with automatic mixed precision using PyTorch's GradScaler
- **Comprehensive Visualization**: Tools for visualizing diffraction patterns, phase, amplitude, and reconstructions

### Datasets

- Support for both synthetic data (line patterns) and real-world datasets (CIFAR10)
- Dataset generation tools that simulate diffraction patterns from source images
- Efficient data loading pipelines with customizable preprocessing

### Training

- Modular training loops with validation and visualization
- Custom loss functions tailored for diffraction-aware tasks
- Tensorboard integration for monitoring training and results
- Support for various illumination probes and diffraction conditions

## Key Files

### Models and Architecture

- `model_diff.py`: Diffraction-aware Masked Autoencoder architecture
- `model.py`: Base Masked Autoencoder architecture

### Training Scripts

- `mae_diff.py`: Training script for diffraction-aware MAE
- `mae_pretrain.py`: Training script for standard MAE pretraining

### Data Handling

- `common.py`: Dataset classes and utilities for loading diffraction data
- `produce_dataset.py`: Tools for generating synthetic diffraction datasets

### Physics Simulation

- `diffsim_torch.py`: PyTorch implementation of diffraction physics
- `probe_torch.py`: Functions for creating and managing illumination probes

### Visualization and Evaluation

- `visualization.py`: Tools for visualizing diffraction data and reconstructions
- `losses.py`: Specialized loss functions for diffraction imaging tasks

## Usage

### Dataset Generation

Generate synthetic diffraction datasets:

```bash
python produce_dataset.py --use_synthetic_lines --num_objects 10 --object_size 392 --num_lines 400
```

### Training a Diffraction-Aware MAE

Train the diffraction-aware MAE model:

```bash
python mae_diff.py --batch_size 512 --mask_ratio 0.75 --input_size 32 --total_epoch 1000 --model_path "diff-mae-model.pt"
```

### Visualization

The training process automatically generates visualizations showing:
- Original objects
- Diffracted patterns
- Reconstructed amplitudes and phases
- Masked regions

All visualizations are logged to Tensorboard for easy monitoring.

## Model Details

### Architecture

The diffraction-aware MAE consists of:

1. **Encoder**: Vision Transformer that encodes diffraction patterns with masked patch embedding
2. **Decoder**: Transformer decoder that reconstructs the original image and integrates with a physical diffraction model
3. **Physics Layer**: A differentiable layer that simulates the physics of light diffraction

### Training Process

The training consists of:
1. Random masking of diffraction patterns
2. Encoding the unmasked patches
3. Reconstructing the full image via the decoder
4. Applying diffraction physics to the reconstruction
5. Computing loss between predicted and actual diffraction patterns

## Requirements

- PyTorch >= 1.9.0
- torchvision
- numpy
- scipy
- scikit-image
- tensorboard
- einops
- tqdm
- timm

## Citations

If using this code, please cite the related papers:

```
@article{he2022masked,
  title={Masked autoencoders are scalable vision learners},
  author={He, Kaiming and Chen, Xinlei and Xie, Saining and Li, Yanghao and Doll{\'a}r, Piotr and Girshick, Ross},
  journal={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2022}
}
```


