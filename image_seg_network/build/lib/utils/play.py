from sklearn import datasets
import torch
from torch.functional import Tensor
from torch.nn.modules import conv
import h5py
import os.path
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
from torch.utils.data import DataLoader, BatchSampler, dataloader, TensorDataset
from torch.nn.utils.rnn import pad_sequence


def main():
    seq_len = 20
    array = np.arange(16).reshape(4, 2, 2)
    print(array)
    array = array.reshape(2, 2, 2, 2)
    print(array)
    array = array.reshape(4, 2, 2)
    print(array)

    # dataset = TensorDataset(torch.arange(10), torch.randn(10))
    # print(dataset)
    # sampler = BatchSampler()
    # dataLoader = DataLoader(dataset, batch_size=2)
    # for samples, targets in dataLoader:
    #     print(samples, targets)


if __name__ == "__main__":

    main()
