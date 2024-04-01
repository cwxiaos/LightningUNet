import torch
import torch.nn as nn
from torch.nn.functional import interpolate
from torch.nn.functional import normalize
from torchvision import transforms

from .blocks import UNet


class LightningUnet(nn.Module):
    def __init__(self,
                 img_size=512,
                 patch_size=4,
                 in_channels=3,
                 embed_dim=96,
                 num_classes=14,
                 ape=True):
        super().__init__()
        self.image_size = img_size
        self.unet = UNet(img_size=img_size,
                         patch_size=patch_size,
                         in_chans=in_channels,
                         num_classes=num_classes,
                         embed_dim=embed_dim,
                         depths=[2, 2, 2, 2],
                         num_heads=[3, 6, 12, 24],
                         mlp_ratio=4,
                         qkv_bias=True,
                         qk_scale=None,
                         drop_rate=0.,
                         drop_path_rate=0.1,
                         norm_layer=nn.LayerNorm,
                         ape=ape)
        # self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # self.normalize = transforms.Compose([transforms.ToTensor(), self.norm])
        self.normalize = normalize

    def forward(self, x):
        # print(f"x shape: {x.shape}")
        b, c, h, w = x.shape
        if h != self.image_size or w != self.image_size:
            x = interpolate(x, size=(self.image_size, self.image_size), mode='bilinear', align_corners=True)
            
        # print(f"x shape: {x.shape}")

        # For RGB images, the shape of x is (B, C, H, W)
        # For grayscale images, the shape of x is (B, 1, H, W)， So repeat 3 times
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        x = self.normalize(x)

        # print(x.shape)
        x = self.unet(x)

        if h != x.shape[2] or w != x.shape[3]:
            x = interpolate(x, size=(h, w), mode='bilinear', align_corners=True)

        return x

