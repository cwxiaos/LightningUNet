import os

import nibabel as nib
import numpy as np
from tqdm import trange

from Dataloader_mr import BaseDataset

# TODO: Dataset702_AbdomenMR is Not 512x512, Need to Resize to 512x512 or Expand to 512x512
# This Script is Not Availiable Now

# TODO: 此脚本需要添加数据形状处理

def process_dataset(dataset, save_dir, train=True):
    print(f"Now Processing {'Train' if train else 'Val'} Dataset, Total: {len(dataset)}")
    for i in range(0, 60):
        volume = dataset[i]

        data = volume['image']
        label = volume['label']

        assert data.shape == label.shape, f"An Error Occurred in {dataset.data[i]}"

        h, w, s = data.shape
        
        # FIXME: Add Resize or Expand Insted of Drop Data with Undesired Shape
        if h == 512 and w == 512:
            for j in trange(data.shape[2], desc=f"Index: {i + 1}/{len(dataset)}", ncols=140):
                # Volume Shape is [H, W, I], Slice Shape is [H, W], So we need to expand dim to [H, W, 1]
                slice_data = data[:, :, j]
                slice_label = label[:, :, j]

                slice_data = np.expand_dims(slice_data, axis=2)
                slice_label = np.expand_dims(slice_label, axis=2)

                # print(slice_data.shape, slice_label.shape)

                save_data = os.path.join(save_dir, f"images{'Tr' if train else 'Val'}/{dataset.data[i].replace('_0000.nii.gz', '')}_slice{j:03d}.nii.gz")
                save_label = os.path.join(save_dir, f"labels{'Tr' if train else 'Val'}/{dataset.label[i].replace('.nii.gz', '')}_slice{j:03d}.nii.gz")

                # print(save_data, save_label)

                slice_data = nib.Nifti1Image(slice_data, affine=np.eye(4))
                slice_label = nib.Nifti1Image(slice_label, affine=np.eye(4))

                nib.save(slice_data, save_data)
                nib.save(slice_label, save_label)
        else:
            print(f"Skip {dataset.data[i]}, Shape: {data.shape}")

"""
This Script is used to convert Dataset701_AbdomenMR to slices
"""
if __name__ == "__main__":
    dataset_dir = "/root/autodl-tmp/Dataset702_AbdomenMR"
    save_dir = "/root/autodl-tmp/Dataset"

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if not os.path.exists(os.path.join(save_dir, "imagesVal")):
        os.mkdir(os.path.join(save_dir, "imagesVal"))
    if not os.path.exists(os.path.join(save_dir, "labelsVal")):
        os.mkdir(os.path.join(save_dir, "labelsVal"))
    if not os.path.exists(os.path.join(save_dir, "imagesTr")):
        os.mkdir(os.path.join(save_dir, "imagesTr"))
    if not os.path.exists(os.path.join(save_dir, "labelsTr")):
        os.mkdir(os.path.join(save_dir, "labelsTr"))

    train_data = BaseDataset(dataset_dir, train=True)
    process_dataset(train_data, save_dir, train=True)
    # val_data = BaseDataset(dataset_dir, train=False)
    # process_dataset(val_data, save_dir, train=False)

