import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset.Dataloader_Synapse import BaseDataset
from networks.lightning_unet import LightningUnet
from utils import DiceLoss

parser = argparse.ArgumentParser()

parser.add_argument("--dataset", type=str, default="dataset")
parser.add_argument("--output", type=str, default="model")
parser.add_argument("--cfg", type=str, default="configs/tiny.yml")
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--base_lr", type=float, default=0.01)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--log_dir", type=str, default="logs")
parser.add_argument("--n_gpu", type=int, default=1)
parser.add_argument("--pretrained", type=str, default=None)

args = parser.parse_args()

assert torch.cuda.is_available(), f"CUDA is essential"
device = torch.device("cuda")
torch.cuda.empty_cache()

if __name__ == "__main__":
    if not os.path.exists(args.output):
        os.mkdir(args.output)

    with open(args.cfg, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

        config_img_size = yaml_cfg['model']['image_size']
        config_patch_size = yaml_cfg['model']['patch_size']
        config_in_channels = yaml_cfg['model']['in_channels']
        config_embed_dim = yaml_cfg['model']['embed_dim']
        config_ape = yaml_cfg['model']['ape']
        config_num_classes = yaml_cfg['model']['num_classes']

        config_seed = yaml_cfg['train']['seed']

    random.seed(config_seed)
    np.random.seed(config_seed)
    torch.manual_seed(config_seed)
    torch.cuda.manual_seed(config_seed)

    model = LightningUnet(img_size=config_img_size,
                          patch_size=config_patch_size,
                          in_channels=config_in_channels,
                          embed_dim=config_embed_dim,
                          ape=config_ape,
                          num_classes=config_num_classes).to(device)

    # print(model)
    if args.pretrained is not None:
        model.load_state_dict(torch.load(args.pretrained))

    writer = SummaryWriter(args.log_dir)

    train_dataset = BaseDataset(root=args.dataset, train=True)
    # val_dataset = BaseDataset(args.dataset, train=False)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    # val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

    if args.n_gpu > 1:
        model = nn.DataParallel(model)

    model.train()

    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(config_num_classes)

    optimizer = optim.SGD(model.parameters(),
                          lr=args.base_lr,
                          momentum=0.9,
                          weight_decay=0.0001)

    iter_num = 0
    max_iterations = args.epochs * len(train_dataloader)

    best_loss = 1

    for epoch in range(args.epochs):
        progress_bar = tqdm(train_dataloader,
                            desc=f"Epoch: {epoch + 1}/{args.epochs}",
                            ncols=170)
        for i, sample in enumerate(progress_bar):
            image, label = sample['image'], sample['label']
            image, label = image.to(device), label.to(device)
            image, label = image.permute(0, 3, 1, 2), label.permute(0, 3, 1, 2)
            label = label[:, 0, :, :]
            image, label = image.to(torch.float32), label.to(torch.float32)
            # B, 1, H, W    B, H, W

            output = model(image)

            loss_ce = ce_loss(output, label.long())
            loss_dice = dice_loss(output, label, softmax=True)
            loss = 0.4 * loss_ce + 0.6 * loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = args.base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            progress_bar.set_postfix_str(
                f"loss: {loss:.4f}, loss_ce: {loss_ce:.4f}, loss_dice: {loss_dice:.4f}, lr: {lr_:.4f}, progress: {iter_num / max_iterations * 100:.2f}%")

            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar("info/loss", loss, iter_num)
            writer.add_scalar("info/loss_ce", loss_ce, iter_num)
            writer.add_scalar("info/loss_dice", loss_dice, iter_num)

            _image = image[1, 0:1, :, :]
            _image = (_image - _image.min()) / (_image.max() - _image.min())
            writer.add_image('train/Image', _image, iter_num)
            _output = torch.argmax(torch.softmax(output, dim=1), dim=1, keepdim=True)
            writer.add_image('train/Prediction', _output[1, ...] * 50, iter_num)
            writer.add_image('train/GroundTruth', label[1, ...].unsqueeze(0) * 50, iter_num)

            iter_num += 1

        if loss < best_loss:
            torch.save(model.state_dict(), os.path.join(args.output, f"best.pth"))
            best_loss = loss

    torch.save(model.state_dict(), os.path.join(args.output, f"unet_{args.epochs}.pth"))
