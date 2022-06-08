import os

import h5py
import torch
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
from tracker import stabilize_doppler
from utils.util import json_file_to_pyobj
from models import get_model

def run_seg(arguments):

    streams = ['/data1/volume1/data/felix_data/results/patient1/original_sweep_30_05_17_52_29.h5', ]
    target_dir = "/data1/volume1/data/felix_data/results/patient1/original_sweep_30_05_17_52_29_unet/"

    def resizer(img):
        return cv2.resize(img, (320, 320))

    def resizer2(img):
        return cv2.resize(img, (501, 699))

    json_filename = arguments.config
    network_debug = arguments.debug

    # Load options
    json_opts = json_file_to_pyobj(json_filename)
    # Setup the NN Model
    model = get_model(json_opts.model)
    model.net.eval()

    images_h5 = h5py.File(streams[0], 'r')
    # loading and transposing
    images_h5 = images_h5["original_sweep_30_05_17_52_29"][:].transpose((0, 2, 3, 1))
    # cropping
    images_h5_us = images_h5[:, 136:835, 472:973, :]
    images_h5_doppler = images_h5[:, 136:835, 1004:1505, :]
    # resizing
    images_h5_us = np.array([resizer(xi) for xi in images_h5_us])
    images_h5_doppler = np.array([resizer(xi) for xi in images_h5_doppler])
    for counter, image in enumerate(images_h5_doppler):
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        # Threshold the HSV image to get only blue colors
        mask = cv2.inRange(
            hsv, np.array([0, 100, 20]), np.array([180, 255, 255]))
        # Bitwise-AND mask and original image
        images_h5_doppler[counter] = cv2.bitwise_and(image, image, mask=mask)
    # images_h5_doppler = stabilize_doppler(images_h5_doppler)
    # convert to greyscale image
    images_h5_us_tmp = []
    images_h5_doppler_tmp = []
    for i in range(len(images_h5_us)):
        images_h5_us_tmp.append(cv2.cvtColor(images_h5_us[i, :, :, :], cv2.COLOR_BGR2GRAY))
        images_h5_doppler_tmp.append(cv2.cvtColor(images_h5_doppler[i, :, :, :], cv2.COLOR_BGR2GRAY))
    images_h5_us = np.array(images_h5_us_tmp)
    images_h5_doppler = np.array(images_h5_doppler_tmp)
    # concatenate the two images
    images_h5 = np.concatenate(
        [images_h5_us[:, None, :, :], images_h5_doppler[:, None, :, :]], axis=1)

    for idx, image in enumerate(images_h5):
        model.set_input(torch.from_numpy(image[None, :, :, :]))
        model.forward("test")
        res = (torch.round(model.net.apply_sigmoid(model.prediction).data) * 255).cpu().detach().numpy()
        path = target_dir + 'label_' + str(idx) + '.png'
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cv2.imwrite(path, resizer2(res[0][0]))






if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='CNN Seg Training Function')

    parser.add_argument(
        '-c', '--config', help='training config file', required=True)
    parser.add_argument(
        '-d', '--debug', help='returns number of parameters and bp/fp runtime', action='store_true')
    args = parser.parse_args()

    run_seg(args)
