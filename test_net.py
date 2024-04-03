import argparse
import random

import numpy as np
import torch
import yaml
from networks.lightning_unet import LightningUnet

parser = argparse.ArgumentParser()

parser.add_argument("--cfg", type=str, default="configs/tiny.yml")

args = parser.parse_args()

assert torch.cuda.is_available(), f"CUDA is essential"
device = torch.device("cuda")
torch.cuda.empty_cache()

if __name__ == "__main__":
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

    print(f"Model Params: {sum(p.numel() for p in model.parameters())}")