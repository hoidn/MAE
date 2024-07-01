import torch
import numpy as np
from scipy.ndimage import gaussian_filter as gf
from skimage import draw
from typing import List, Tuple


def mk_rand(N: int) -> int:
    """Generate a random integer between 0 and N-1."""
    return int(N * np.random.uniform())

def mk_lines_img(N: int = 64, nlines: int = 10) -> np.ndarray:
    """Generate a synthetic 'lines' image."""
    image = np.zeros((N, N))
    for _ in range(nlines):
        rr, cc = draw.line(mk_rand(N), mk_rand(N), mk_rand(N), mk_rand(N))
        image[rr, cc] = 1
    res = np.zeros((N, N, 1))
    res[:, :, :] = image[..., None]
    return gf(res, 1) + 2 * gf(res, 5) + 5 * gf(res, 10)

def create_tensor_image(size: int = 64, nlines: int = 10) -> torch.Tensor:
    """Create a tensor image of synthetic lines."""
    image = mk_lines_img(size * 2, nlines)[size // 2: -size // 2, size // 2: -size // 2, :1]
    tensor = torch.from_numpy(image).float().permute(2, 0, 1)
    print('nlines:', nlines)
    return tensor.repeat(3, 1, 1)

class SyntheticLinesDataset(torch.utils.data.Dataset):
    def __init__(self, num_objects: int = 1, object_size: int = 64,
                 patch_size: int = 16, num_lines: int = 400, patch_offset: int = 8):
        """Initialize the dataset with the ability to create overlapping patches."""
        self.num_objects = num_objects
        self.object_size = object_size
        self.patch_size = patch_size
        self.num_lines = num_lines
        self.patch_offset = patch_offset
        
        self.patches: List[torch.Tensor] = []
        self.centers: List[torch.Tensor] = []
        
        for _ in range(num_objects):
            obj = create_tensor_image(object_size, num_lines)
            
            # Generate patches ensuring they do not exceed image boundaries
            i = 0
            while i * patch_offset + patch_size <= object_size:
                j = 0
                while j * patch_offset + patch_size <= object_size:
                    y = i * patch_offset
                    x = j * patch_offset
                    patch = obj[:, y:y+patch_size, x:x+patch_size]
                    self.patches.append(patch)
                    
                    # Calculate center for the patch
                    center_y = y + patch_size // 2
                    center_x = x + patch_size // 2
                    self.centers.append(torch.tensor([center_y, center_x], dtype=torch.float32))
                    j += 1
                i += 1

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.patches[idx], self.centers[idx]

def create_synthetic_lines_dataloader(batch_size: int, num_objects: int = None,
                                      object_size: int = None, patch_size: int = None, num_lines: int = None,
                                      existing_dataset: torch.utils.data.Dataset = None) -> torch.utils.data.DataLoader:
    """
    Create a DataLoader for synthetic lines objects.
    
    Args:
        batch_size (int): The batch size for the DataLoader.
        num_objects (int, optional): The number of synthetic objects to generate if no dataset is provided.
        object_size (int, optional): The size of each object if no dataset is provided.
        patch_size (int, optional): The size of the patches to extract if no dataset is provided.
        num_lines (int, optional): The number of lines in each object if no dataset is provided.
        existing_dataset (torch.utils.data.Dataset, optional): An existing dataset to use.
    
    Returns:
        torch.utils.data.DataLoader: A DataLoader for the synthetic lines dataset.
    """
    # Use the provided dataset or create a new one if not provided
    if existing_dataset is not None:
        dataset = existing_dataset
    else:
        if num_objects is None or object_size is None or patch_size is None or num_lines is None:
            raise ValueError("num_objects, object_size, patch_size, and num_lines must be specified if no existing dataset is provided.")
        dataset = SyntheticLinesDataset(num_objects=num_objects, object_size=object_size,
                                        patch_size=patch_size, num_lines=num_lines)

    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

