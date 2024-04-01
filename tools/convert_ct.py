import os

import nibabel as nib
import numpy as np
from tqdm import trange

from Dataloader_ct import BaseDataset

def process_dataset(dataset, save_dir, train=True):
    print(f"Now Processing {'Train' if train else 'Val'} Dataset, Total: {len(dataset)}")
    # for i in range(len(dataset)):
    for i in range(0, 25):
        volume = dataset[i]

        data = volume['image']
        label = volume['label']

        assert data.shape == label.shape, f"An Error Occurred in {dataset.data[i]}"
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


"""
This Script is used to convert Dataset701_AbdomenCT to slices
"""
if __name__ == "__main__":
    dataset_dir = "/root/autodl-tmp/Dataset701_AbdomenCT"
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

