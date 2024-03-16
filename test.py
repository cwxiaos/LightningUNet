import torch

from nets import lightning_unet as unet

if __name__ == "__main__":
    net = unet.LightningUNet().cuda()

    ch, H, W = 1, 1024, 1024
    x = torch.randn(1, ch, H, W).cuda()

    # print(x)
    # print(x.shape)

    # print(net)

    # print(net(x))
    print(net(x).shape)

