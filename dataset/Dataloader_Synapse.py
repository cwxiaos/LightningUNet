import os
import random

import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
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
        self.train = train
        self.root = root

        self.list = os.path.join(self.root, "train.txt")
        self.root = os.path.join(self.root, 'train_npz')

        self.output_size = [224, 224]

        self.sample_list = open(self.list).readlines()

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.train:
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.root, slice_name+'.npz')
            data = np.load(data_path)

            image, label = data['image'], data['label']
            # image = np.expand_dims(image, 2)
            # label = np.expand_dims(label, 2)

            # print(image.shape, label.shape)

            if random.random() > 0.2:
                image, label = random_rot_flip(image, label)
            elif random.random() > 0.2:
                image, label = random_rotate(image, label)

            x, y = image.shape
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)

            image = np.expand_dims(image, 2)
            label = np.expand_dims(label, 2)

        else:
            pass

        sample = {'image': image, 'label': label}
        sample['case_name'] = self.sample_list[idx].strip('\n')

        return sample
