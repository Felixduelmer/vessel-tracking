import cv2
import torch
import torch.utils.data as data
import h5py
import numpy as np
import datetime
from skimage import color, io
from matplotlib import pyplot as plt

from os import listdir
from os.path import join
from .utils import check_exceptions


class UltraSoundDataset(data.Dataset):
    def __init__(self, root_path, split, transform=None, preload_data=False):
        super(UltraSoundDataset, self).__init__()

        f = h5py.File(root_path, 'r')

        self.images = f['x_'+split]

        self.seq_len = 20

        if preload_data:
            self.images = np.array(self.images[:])

        self.labels = np.expand_dims(
            np.array(f['y_'+split][:], dtype=np.int64), axis=1)  # [:1000]

        assert len(self.images) == len(self.labels)

        # reduce image to gray scale
        self.images = np.array([cv2.cvtColor(
            self.images[i, j, :, :, :], cv2.COLOR_BGR2GRAY) for j in range(self.images.shape[1]) for i in range(self.images.shape[0])]).reshape(*self.images.shape[:4])
        # prepare sequences
        if len(self.images) % self.seq_len != 0:
            if split == 'train':
                self.images = np.concatenate(
                    (self.images, np.zeros((self.seq_len - (len(self.images) % self.seq_len), *self.images.shape[1:]))))
                self.labels = np.concatenate(
                    (self.labels, np.zeros((self.seq_len - (len(self.labels) % self.seq_len), *self.labels.shape[1:]))))
            else:
                print((len(self.images)-int(len(self.images)/self.seq_len)))
                self.images = self.images[:(
                    len(self.images)-(len(self.images) % self.seq_len))]
                self.labels = self.labels[:(
                    len(self.labels)-(len(self.labels) % self.seq_len))]

        self.images = self.images.reshape(int(np.ceil(len(self.images)/self.seq_len)),
                                          self.seq_len, *self.images.shape[1:])
        self.labels = self.labels.reshape(int(np.ceil(len(self.labels)/self.seq_len)),
                                          self.seq_len, *self.labels.shape[1:])

        print(self.images.shape, self.labels.shape)

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
        input = self.images[index]
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
