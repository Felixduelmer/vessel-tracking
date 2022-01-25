import torch
import torch.utils.data as data
import h5py
import numpy as np
import datetime
from skimage import color, io
from matplotlib import pyplot as plt
import cv2

from os import listdir
from os.path import join
from .utils import check_exceptions


class UltraSoundDataset(data.Dataset):
    def __init__(self, root_path, split, transform=None, preload_data=False):
        super(UltraSoundDataset, self).__init__()

        f = h5py.File(root_path, 'r')

        self.images = f['x_'+split]
        print(self.images.shape)

        if preload_data:
            self.images = np.array(self.images[:50])

        self.labels = np.expand_dims(
            np.array(f['y_'+split][:50], dtype=np.int64), axis=1)  # [:1000]

        # print(class_weight)
        assert len(self.images) == len(self.labels)

        # data augmentation
        self.transform = transform

        # report the number of images in the dataset
        print('Number of images: {0}'.format(self.__len__()))

    def __getitem__(self, index):
        # update the seed to avoid workers sample the same augmentation parameters
        np.random.seed(datetime.datetime.now().second +
                       datetime.datetime.now().microsecond)

        # load the images
        input = np.array([cv2.cvtColor(
            self.images[index, runner, :, :, :], cv2.COLOR_BGR2GRAY) for runner in range(2)])
        target = np.uint8(self.labels[index])

        # handle exceptions
        #check_exceptions(input, target)
        if self.transform:
            input = self.transform(input)

        # print(input.shape, torch.from_numpy(np.array([target])))
        # print("target", np.int64(target))
        return input, target

    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    dataset = UltraSoundDataset(
        '/home/felix/projects/ma/data/h5_datasets/ultrasound.h5')

    from torch.utils.data import DataLoader, sampler
    ds = DataLoader(dataset=dataset, num_workers=1, batch_size=2)
