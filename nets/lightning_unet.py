import torch
import torch.nn as nn

from nets.blocks import UNet


class LightningUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.unet = UNet(image_size=512,
                         patch_size=1,
                         in_channels=3,
                         embed_dim=32,
                         dropout=0.0,
                         num_classes=14,
                         ape=False)

    def forward(self, x):
        # For RGB images, the shape of x is (B, C, H, W)
        # For grayscale images, the shape of x is (B, 1, H, W)， So repeat 3 times
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        # TODO : Add Image Zooming

        x = self.unet(x)

        return x
