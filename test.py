import torch

from nets import lightning_unet as unet

if __name__ == "__main__":
    net = unet.LightningUNet()

    ch, H, W = 1, 512, 512
    x = torch.randn(1, ch, H, W)
    x = x.repeat(1, 3, 1, 1)

    # print(x)
    # print(x.shape)

    print(net)

    print(net(x))
    print(net(x).shape)

