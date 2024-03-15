import torch
import torch.nn as nn

from lightning_attn.ops import lightning_attn_func
from lightning_attn.utils import _build_slope_tensor


class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class PatchEmbedding(nn.Module):
    r"""
    Patch Embedding

    Args:
        image_size: int, Input image size
        patch_size: int, Patch size
        in_channels: int, Number of input channels
        embed_dim: int, Embedding dimension
        dropout: float, Dropout rate
    """
    def __init__(self, image_size, patch_size, in_channels, embed_dim, dropout):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.dropout = dropout

        self.num_patches = (image_size / patch_size) ** 2

    def forward(self, x):
        b, c, h, w = x.shape
        assert h == self.image_size and w == self.image_size, f"Image Shape Doesn't Match: {h}x{w} != {self.image_size}x{self.image_size}"
        x = x.flatten(2)        # B, C, H*W
        x = x.transpose(1, 2)   # B, H*W, C
        return x


class UNet(nn.Module):
    r"""
    UNet

    Args:
        image_size: int, Input image size
        patch_size: int, Patch size
        in_channels: int, Number of input channels
        embed_dim: int, Embedding dimension
        dropout: float, Dropout rate
        ape: bool, Absolute position embedding
    """
    def __init__(self,
                 image_size=512,
                 patch_size=1,
                 in_channels=3,
                 embed_dim=192,
                 dropout=0.,
                 ape=False):

        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.ape = ape

        self.patch_embedding = PatchEmbedding(image_size, patch_size, in_channels, embed_dim, dropout)
        num_patches = self.patch_embedding.num_patches
        if self.ape:
            self.absolute_position_embedding = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        self.pos_drop = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.pos_drop(x)
        return x
