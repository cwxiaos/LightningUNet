import argparse
import os
from time import sleep

import SimpleITK as sitk
import h5py
import nibabel as nib
import numpy as np
import torch
import yaml
from medpy import metric
from torch.nn.functional import interpolate
from tqdm import trange

from networks.lightning_unet import LightningUnet

parser = argparse.ArgumentParser()

parser.add_argument('--data', type=str, help="Data To Inference")
parser.add_argument('--output', type=str, help="Output Dir")
parser.add_argument('--model', type=str, help="Path to Model")
parser.add_argument('--cfg', type=str, help="Config File", default="configs/m.yml")
parser.add_argument('--batch', type=int, default=12, help="Batch Size")
parser.add_argument('--label', type=str, help="Label Path", default=None)

args = parser.parse_args()

assert torch.cuda.is_available(), f"CUDA is essential"
device = torch.device("cuda")
torch.cuda.empty_cache()


def inference(file, dir_output, model, num_classes=14, img_size=512, batch=12, patch_size=4, label=None):
    net = LightningUnet(num_classes=num_classes, ape=False, img_size=img_size, patch_size=patch_size).to(device)
    net.eval()

    msg = net.load_state_dict(torch.load(model))
    print(f"Load Weights: {msg}")

    # To use SwinUnet h5 data, Load h5 file
    # data = h5py.File("/root/autodl-tmp/Swin-Unet/data/test_vol_h5/case0001.npy.h5")
    # image = data['image']

    # Use nibabel
    data = nib.load(file)
    image = data.get_fdata()
    image = np.transpose(image, (2, 1, 0))

    # Use SimpleITK
    # data = sitk.ReadImage(file)
    # image_filter = sitk.RescaleIntensityImageFilter()
    # image_filter.SetOutputMaximum(255)
    # image_filter.SetOutputMinimum(0)
    # data = image_filter.Execute(data)
    # image = sitk.GetArrayFromImage(data)

    prediction = np.zeros_like(image)
    s, h, w = image.shape

    print(f"File {file} shape: {s}, {h}, {w}")

    for i_slice in trange(0, s, batch, ncols=140):
        if s - i_slice < batch:
            batch_range = range(i_slice, s)
        else:
            batch_range = range(i_slice, i_slice + batch)
        # print(batch_range)

        slice_image = image[batch_range, :, :]
        slice_image = torch.from_numpy(slice_image)
        slice_image = slice_image.unsqueeze(0)
        slice_image = slice_image.permute(1, 0, 2, 3)
        slice_image = slice_image.to(device).to(torch.float32)

        # print(slice_image.shape)

        with torch.no_grad():
            # if h != img_size or w != img_size:
            #     slice_image = interpolate(slice_image, size=(img_size, img_size), mode='bilinear', align_corners=True)
            output = net(slice_image)

            out = torch.argmax(torch.softmax(output, dim=1), dim=1)
            # print(out.shape)

            # if h != img_size or w != img_size:
            #     out = out.unsqueeze(0)
            #     out = interpolate(out, size=(h, w), mode='bilinear', align_corners=True)
            #     out = out.squeeze(0)

            # nibabel
            # out = out.permute(1, 2, 0)
            out = out.cpu().detach().numpy()
            # print(out.shape)

            prediction[batch_range, :, :] = out

    # print(prediction.shape)
    if label is not None:
        data = nib.load(label)
        label = data.get_fdata()
        label = np.transpose(label, (2, 1, 0))

        dice = metric.binary.dc(prediction, label)
        hd95 = metric.binary.hd95(prediction, label)

        print(f"Test Dice: {dice}, Test HD95: {hd95}")

    save_path = os.path.join(dir_output, f"{os.path.basename(file)[:-7]}_pred.nii.gz")
    print(f"Save Path: {save_path}")

    # prediction = np.transpose(prediction, (2, 1, 0))
    # data = nib.Nifti1Image(prediction, affine=np.eye(4))
    # nib.save(data, save_path)

    sitk.WriteImage(sitk.GetImageFromArray(prediction), save_path)


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

    inference(file=args.data,
              dir_output=args.output,
              model=args.model,
              num_classes=config_num_classes,
              img_size=config_img_size,
              batch=args.batch,
              label=args.label,
              patch_size=config_patch_size)
