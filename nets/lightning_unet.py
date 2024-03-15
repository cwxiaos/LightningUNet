import torch
import torch.nn as nn

from nets.blocks import UNet


class LightningUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.unet = UNet()

    def forward(self, x):
        x = self.unet(x)
        return x
