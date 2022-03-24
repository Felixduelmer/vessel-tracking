import cv2
import torch
import torch.utils.data as data
import h5py
import numpy as np
import datetime

from os import listdir
from os.path import join
from .utils import check_exceptions


class UltraSoundDataset(data.Dataset):
    def __init__(self, root_path, split, transform=None, preload_data=False):
        super(UltraSoundDataset, self).__init__()

        f = h5py.File(root_path, 'r')

        self.images = f['x_' + split]

        if preload_data:
            self.images = np.array(self.images[:])

        self.labels = f['y_' + split][:]

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
        inp = self.images[index]
        target = self.labels[index]

        # handle exceptions
        # check_exceptions(input, target)
        # random number generator seed is saved to have same transformation for all images in one sequence
        # https://github.com/pytorch/vision/issues/9
        if self.transform:
            state = torch.get_rng_state()
            for i in range(len(inp)):
                torch.set_rng_state(state)
                inp[i][0] = self.transform(torch.from_numpy(inp[i, [0]]))
                torch.set_rng_state(state)
                inp[i][1] = self.transform(torch.from_numpy(inp[i, [1]]))
                torch.set_rng_state(state)
                target[i] = self.transform(torch.from_numpy(target[i]))

        # print(input.shape, torch.from_numpy(np.array([target])))
        # print("target", np.int64(target))
        return inp, target

    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    dataset = UltraSoundDataset(
        '/home/felix/projects/ma/data/h5_datasets/ultrasound.h5')

    from torch.utils.data import DataLoader, sampler

    ds = DataLoader(dataset=dataset, num_workers=1, batch_size=2)
