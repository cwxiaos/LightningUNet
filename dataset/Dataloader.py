import os
import random

import nibabel as nib
import numpy as np
from scipy import ndimage
from torch.utils.data import Dataset

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class BaseDataset(Dataset):
    def __init__(self, root, train=True):
        self.root = root
        self.train = train

        if train == True:
            self.dir_data = os.path.join(root, 'imagesTr')
            self.dir_label = os.path.join(root, 'labelsTr')
            self.data = os.listdir(self.dir_data)
            self.label = os.listdir(self.dir_label)
        else:
            self.dir_data = os.path.join(root, 'imagesVal')
            self.dir_label = os.path.join(root, 'labelsVal')
            self.data = os.listdir(self.dir_data)
            self.label = os.listdir(self.dir_label)

        self.data.sort()
        self.label.sort()

        assert len(self.data) == len(self.label), f"Data length {len(self.data)} Doesn't Match Label length {len(self.label)}"
        assert len(self.data) > 0, f"No Data Found in {self.root}"
        print(f"BaseDataset: Num Data {'Train' if self.train else 'Val'} = {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        name_data, name_label = self.data[index], self.label[index]

        assert name_data.endswith('.nii.gz') and name_label.endswith('.nii.gz'), f"Unsupported file {name_data} {name_label}"
        assert name_label[:-7] in name_data[:-7], f"Data {name_data} doesn't match Label {name_label}"

        data = nib.load(os.path.join(self.dir_data, name_data)).get_fdata()
        label = nib.load(os.path.join(self.dir_label, name_label)).get_fdata()

        # data = data.transpose(2, 0, 1)
        # label = label.transpose(2, 0, 1)
        if self.train:
            if random.random() > 0.5:
                data, label = random_rot_flip(data, label)
            elif random.random() > 0.5:
                data, label = random_rotate(data, label)
        sample = {'image': data, 'label': label}

        # print(sample['image'].shape)

        return sample