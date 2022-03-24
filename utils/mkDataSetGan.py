import os
from pathlib import Path
import shutil
import h5py
import numpy as np
from PIL import Image
import cv2

def resizer(img):
    return cv2.resize(img, (320, 320))

def trainA():

    directory = '/home/robotics-verse/projects/felix/vessel-tracking/pytorch-CycleGAN-and-pix2pix/results/bmode_cyclegan/test_latest/images'

    targetDir = '/home/robotics-verse/projects/felix/DataSet/felix_data/external_data/nmi_vasc_robot'   
    for file in os.listdir(directory):
        if 'fake' not in file:
            continue

        shutil.copy(directory + '/' + file, targetDir + '/' +file)

def trainB():

    directory = '/home/robotics-verse/projects/felix/DataSet/nmi-vasc-robot/data/dus_test'

    targetDir = '/home/robotics-verse/projects/felix/DataSet/felix_data/external_data/nmi_vasc_robot/doppler'

    for subdir in os.listdir(directory):
        for file in os.listdir(directory + '/' + subdir):
            if 'doppler' not in file:
                continue

            shutil.copy(directory + '/' + subdir + '/' + file, targetDir + '/' + subdir + '_' + file)

def removeChars():

    directory = '/home/robotics-verse/projects/felix/DataSet/doppler/seq1'

    targetDir = '/home/robotics-verse/projects/felix/DataSet/doppler/seq1'   
    for file in os.listdir(directory):

        shutil.copy(directory + '/' + file, targetDir + '/' +file[2:])

def labeltotrainA():

    targetDir = '/home/robotics-verse/projects/felix/vessel-tracking/pytorch-CycleGAN-and-pix2pix/datasets/label2bmode/trainA'

    h5_labels = ['/home/robotics-verse/projects/felix/DataSet/felix_data/Zhongliang/labels.h5',
                 '/home/robotics-verse/projects/felix/DataSet/felix_data/Felix/labels.h5', ]
    h5_images_start = [0, 36]
    h5_images_end = [-1, 421]

    assert len(h5_labels) == len(h5_images_start) == len(h5_images_end)

    for idx, path in enumerate(h5_labels):
        labels_h5 = h5py.File(path, 'r')
        # loading and transposing
        labels_h5 = labels_h5['labels'][h5_images_start[idx]:h5_images_end[idx]].transpose((0, 2, 3, 1))
        # cropping
        labels_h5 = np.squeeze(labels_h5[:, 136:835, 472:973, :], axis=3)
        # resizing
        list_labels_h5 = [Image.fromarray(xi*255, mode='L').resize((320, 320)) for xi in labels_h5]
        for counter, image in enumerate(list_labels_h5):
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)    
            # Threshold the HSV image to get only blue colors
            mask = cv2.inRange(
                hsv, np.array([0, 100, 20]), np.array([180, 255, 255]))
            # Bitwise-AND mask and original image
            res = cv2.bitwise_and(
                image, image, mask=mask)
            list_labels_h5[counter] = res
        [im.save('{}/{}_{}.png'.format(targetDir, idx, i)) for i, im in enumerate(list_labels_h5)]



def labeltotrainB():

    targetDir = '/home/robotics-verse/projects/felix/vessel-tracking/pytorch-CycleGAN-and-pix2pix/datasets/label2bmode/trainA'

    h5_labels = ['/home/robotics-verse/projects/felix/DataSet/felix_data/Zhongliang/labels.h5',
                 '/home/robotics-verse/projects/felix/DataSet/felix_data/Felix/labels.h5', ]
    h5_images_start = [0, 36]
    h5_images_end = [-1, 421]

    assert len(h5_labels) == len(h5_images_start) == len(h5_images_end)

    for idx, path in enumerate(h5_labels):
        labels_h5 = h5py.File(path, 'r')
        # loading and transposing
        labels_h5 = labels_h5['labels'][h5_images_start[idx]:h5_images_end[idx]].transpose((0, 2, 3, 1))
        # cropping
        labels_h5 = np.squeeze(labels_h5[:, 136:835, 472:973, :], axis=3)
        # resizing
        list_labels_h5 = [Image.fromarray(xi*255, mode='L').resize((320, 320)) for xi in labels_h5]
        [im.save('{}/{}_{}.png'.format(targetDir, idx, i)) for i, im in enumerate(list_labels_h5)]



if __name__ == "__main__":

    removeChars()