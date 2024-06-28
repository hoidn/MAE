import torch
import numpy as np
from einops import repeat, rearrange
from einops.layers.torch import Rearrange
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Block
from functools import partial
from diffsim_torch import diffraction_from_channels

def random_indexes(size: int):
    forward_indexes = np.arange(size)
    np.random.shuffle(forward_indexes)
    backward_indexes = np.argsort(forward_indexes)
    return forward_indexes, backward_indexes

def take_indexes(sequences, indexes):
    return torch.gather(sequences, 0, repeat(indexes, 't b -> t b c', c=sequences.shape[-1]))

class PatchShuffle(torch.nn.Module):
    def __init__(self, ratio) -> None:
        super().__init__()
        self.ratio = ratio

    def forward(self, patches: torch.Tensor):
        T, B, C = patches.shape
        remain_T = int(T * (1 - self.ratio))

        indexes = [random_indexes(T) for _ in range(B)]
        forward_indexes = torch.as_tensor(np.stack([i[0] for i in indexes], axis=-1), dtype=torch.long).to(patches.device)
        backward_indexes = torch.as_tensor(np.stack([i[1] for i in indexes], axis=-1), dtype=torch.long).to(patches.device)

        patches = take_indexes(patches, forward_indexes)
        patches = patches[:remain_T]

        return patches, forward_indexes, backward_indexes

class MAE_Encoder(torch.nn.Module):
    def __init__(self,
                 input_size: int = 64,
                 emb_dim: int = 192,
                 num_layer: int = 12,
                 num_head: int = 3,
                 mask_ratio: float = 0.75,
                 ) -> None:
        super().__init__()

        self.patch_size = input_size // 16
        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros((input_size // self.patch_size) ** 2, 1, emb_dim))
        self.shuffle = PatchShuffle(mask_ratio)

        self.patchify = torch.nn.Conv2d(3, emb_dim, self.patch_size, self.patch_size)

        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])

        self.layer_norm = torch.nn.LayerNorm(emb_dim)

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, img: torch.Tensor):
        patches = self.patchify(img)
        patches = rearrange(patches, 'b c h w -> (h w) b c')
        patches = patches + self.pos_embedding

        patches, forward_indexes, backward_indexes = self.shuffle(patches)

        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
        patches = rearrange(patches, 't b c -> b t c')
        features = self.layer_norm(self.transformer(patches))
        features = rearrange(features, 'b t c -> t b c')

        return features, backward_indexes

class MAE_Decoder(torch.nn.Module):
    def __init__(self,
                 input_size: int = 32,
                 emb_dim: int = 192,
                 num_layer: int = 4,
                 num_head: int = 3,
                 intensity_scale: float = 1000.,
                 probe: torch.Tensor = None
                 ) -> None:
        super().__init__()

        self.patch_size = input_size // 16
        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros((input_size // self.patch_size) ** 2 + 1, 1, emb_dim))

        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])

        self.head = torch.nn.Linear(emb_dim, 3 * self.patch_size ** 2)
        self.patch2img = Rearrange('(h w) b (c p1 p2) -> b c (h p1) (w p2)', p1=self.patch_size, p2=self.patch_size, h=input_size//self.patch_size)
        self.diffract = partial(diffraction_from_channels, intensity_scale=intensity_scale, draw_poisson=False, bias = 1.)
        if probe is None:
            raise ValueError("Probe cannot be None")
        self.probe = probe
        self.intensity_scale = intensity_scale

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.mask_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, features: torch.Tensor, backward_indexes: torch.Tensor) -> dict:
        T = features.shape[0]
        backward_indexes = torch.cat([torch.zeros(1, backward_indexes.shape[1]).to(backward_indexes), backward_indexes + 1], dim=0)
        features = torch.cat([features, self.mask_token.expand(backward_indexes.shape[0] - features.shape[0], features.shape[1], -1)], dim=0)
        features = take_indexes(features, backward_indexes)
        features = features + self.pos_embedding

        features = rearrange(features, 't b c -> b t c')
        features = self.transformer(features)
        features = rearrange(features, 'b t c -> t b c')
        features = features[1:]  # remove global feature

        patches = self.head(features)
        mask = torch.zeros_like(patches)
        mask[T-1:] = 1
        mask = take_indexes(mask, backward_indexes[1:] - 1)
        img = self.patch2img(patches)
        mask = self.patch2img(mask)

        Y_complex, amplitude = self.diffract(img, self.probe)
        
        return {
            'predicted_amplitude': amplitude,
            'mask': mask,
            'intensity_scale': self.intensity_scale,
            'intermediate_img': img,
            'predicted_Y_complex': Y_complex
        }

class MAE_ViT(torch.nn.Module):
    def __init__(self,
                 input_size: int = 64,
                 emb_dim: int = 192,
                 encoder_layer: int = 12,
                 encoder_head: int = 3,
                 decoder_layer: int = 4,
                 decoder_head: int = 3,
                 mask_ratio: float = 0.75,
                 intensity_scale: float = 1000.,
                 probe: torch.Tensor = None
                 ) -> None:
        super().__init__()

        self.input_size = input_size
        self.patch_size = input_size // 16
        self.mask_ratio = mask_ratio
        self.intensity_scale = intensity_scale
        self.probe = probe

        self.encoder = MAE_Encoder(input_size, emb_dim, encoder_layer, encoder_head, mask_ratio)
        self.decoder = MAE_Decoder(input_size, emb_dim, decoder_layer, decoder_head,
                                   intensity_scale=intensity_scale, probe=probe)

    def forward(self, img: torch.Tensor) -> dict:
        features, backward_indexes = self.encoder(img)
        output = self.decoder(features, backward_indexes)
        output['target_amplitude'] = img  # Add target amplitude to the output dictionary
        output['mask_ratio'] = self.mask_ratio
        return output

    def save_model(self, file_path: str):
        save_dict = {
            'model_state_dict': self.state_dict(),
            'probe': self.probe
        }
        torch.save(save_dict, file_path)

    @classmethod
    def load_model(cls, file_path: str):
        checkpoint = torch.load(file_path)
        probe = checkpoint['probe']
        model = cls(probe=probe)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model

class ViT_Classifier(torch.nn.Module):
    def __init__(self, encoder: MAE_Encoder, num_classes: int = 10) -> None:
        super().__init__()
        self.cls_token = encoder.cls_token
        self.pos_embedding = encoder.pos_embedding
        self.patchify = encoder.patchify
        self.transformer = encoder.transformer
        self.layer_norm = encoder.layer_norm
        self.head = torch.nn.Linear(self.pos_embedding.shape[-1], num_classes)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        patches = self.patchify(img)
        patches = rearrange(patches, 'b c h w -> (h w) b c')
        patches = patches + self.pos_embedding
        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
        patches = rearrange(patches, 't b c -> b t c')
        features = self.layer_norm(self.transformer(patches))
        features = rearrange(features, 'b t c -> t b c')
        logits = self.head(features[0])
        return logits

if __name__ == '__main__':
    shuffle = PatchShuffle(0.75)
    a = torch.rand(16, 2, 10)
    b, forward_indexes, backward_indexes = shuffle(a)
    print(b.shape)

    img = torch.rand(2, 3, 32, 32)
    probe = torch.rand(32, 32)
    model = MAE_ViT(input_size=32, probe=probe)
    output = model(img)
    print(output['predicted_amplitude'].shape)
    loss = torch.mean((output['predicted_amplitude'] - output['target_amplitude']) ** 2 * output['mask'] / output['mask_ratio'])
    print(loss)
