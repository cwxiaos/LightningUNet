import torch

from nets import lightning_unet as unet

device = torch.device("cuda")

if __name__ == "__main__":
    net = unet.LightningUNet().to(device)

    ch, H, W = 1, 512, 512
    x = torch.randn(1, ch, H, W).to(device)

    # print(x)
    # print(x.shape)

    # print(net)

    # print(net(x))
    print(net(x).shape)

