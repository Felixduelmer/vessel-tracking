import os

import h5py
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
from tracker import stabilize_doppler


def main():
    images = []
    labels = []
    seq_len = 20
    seq_number = 0
    seq_str = 'sequence'
    tmp_labels = None
    tmp_images = None

    h5_images = ['/data1/volume1/data/felix_data/sweeps_imfusion/patient_1/us_images.h5',
                 '/data1/volume1/data/felix_data/sweeps_imfusion/patient_2/0.h5',
                 '/data1/volume1/data/felix_data/sweeps_imfusion/patient_3/sweep_1/0.h5',
                 '/data1/volume1/data/felix_data/sweeps_imfusion/patient_3/sweep_2/0.h5',
                 '/data1/volume1/data/felix_data/sweeps_imfusion/patient_4/sweep_1/0.h5',
                 '/data1/volume1/data/felix_data/sweeps_imfusion/patient_4/sweep_2/0.h5',
                 '/data1/volume1/data/felix_data/sweeps_imfusion/patient_5/sweep_1/0.h5',
                 '/data1/volume1/data/felix_data/sweeps_imfusion/patient_5/sweep_2/0.h5',
                 '/data1/volume1/data/felix_data/sweeps_imfusion/patient_6/sweep_1/0.h5',
                 '/data1/volume1/data/felix_data/sweeps_imfusion/patient_6/sweep_2/0.h5',
                 '/data1/volume1/data/felix_data/sweeps_imfusion/patient_7/sweep_1/0.h5',
                 '/data1/volume1/data/felix_data/sweeps_imfusion/patient_7/sweep_2/0.h5',
                 ]
    h5_images_name = ['us_images', 'data', 'data', '0', 'data', 'data', '0', 'data', 'data', 'data', 'data',
                      'data', ]
    h5_labels = ['/data1/volume1/data/felix_data/sweeps_imfusion/patient_1/labels.h5',
                 '/data1/volume1/data/felix_data/sweeps_imfusion/patient_2/0-labels.h5',
                 '/data1/volume1/data/felix_data/sweeps_imfusion/patient_3/sweep_1/0-labels.h5',
                 '/data1/volume1/data/felix_data/sweeps_imfusion/patient_3/sweep_2/0-labels.h5',
                 '/data1/volume1/data/felix_data/sweeps_imfusion/patient_4/sweep_1/0-labels.h5',
                 '/data1/volume1/data/felix_data/sweeps_imfusion/patient_4/sweep_2/0-labels.h5',
                 '/data1/volume1/data/felix_data/sweeps_imfusion/patient_5/sweep_1/0-labels.h5',
                 '/data1/volume1/data/felix_data/sweeps_imfusion/patient_5/sweep_2/0-labels.h5',
                 '/data1/volume1/data/felix_data/sweeps_imfusion/patient_6/sweep_1/0-labels.h5',
                 '/data1/volume1/data/felix_data/sweeps_imfusion/patient_6/sweep_2/0-labels.h5',
                 '/data1/volume1/data/felix_data/sweeps_imfusion/patient_7/sweep_1/0-labels.h5',
                 '/data1/volume1/data/felix_data/sweeps_imfusion/patient_7/sweep_2/0-labels.h5',
                 ]

    h5_labels_name = ['labels', 'label', 'label', '0-labels', 'label', 'label', '0-labels', 'label',
                      'label', 'label', 'label', 'label', ]

    h5_images_start = [0, 0, 0, 0, 0, 0, 43, 0, 55, 0, 49, 38]
    h5_images_end = [-1, -1, -1, -1, -1, -1, 366, -1, 401, -1, 611, 361]

    external_images = ['/home/robotics-verse/projects/felix/DataSet/felix_data/external_data/nmi_vasc_robot', ]

    def resizer(img):
        return cv2.resize(img, (320, 320))

    def fill_sequence(fill_images):
        if len(fill_images) % seq_len != 0:
            return np.concatenate(
                (fill_images, np.zeros((seq_len - (len(fill_images) % seq_len), *fill_images.shape[1:]))))
        else:
            return fill_images

    assert len(h5_images) == len(h5_labels) == len(h5_images_start) == len(h5_images_end)

    for idx, path in enumerate(h5_images):
        images_h5 = h5py.File(path, 'r')
        # loading and transposing
        images_h5 = images_h5[h5_images_name[idx]][h5_images_start[idx]:h5_images_end[idx]].transpose((0, 2, 3, 1))
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
        # fill the sequence with dark images so the sequence length is reached
        images_h5 = fill_sequence(images_h5)
        if idx % 2 == 0 and idx > 1:
            tmp_images = images_h5
        else:
            if tmp_images is not None:
                images_h5 = np.concatenate((tmp_images, images_h5))
            images_h5 = images_h5.reshape(((images_h5.shape[0] // seq_len), seq_len, *images_h5.shape[1:]))
            images.append(images_h5)
            tmp_images = None

    for idx, path in enumerate(h5_labels):
        labels_h5 = h5py.File(path, 'r')
        # loading and transposing
        labels_h5 = labels_h5[h5_labels_name[idx]][h5_images_start[idx]:h5_images_end[idx]].transpose((0, 2, 3, 1))
        # cropping
        labels_h5 = labels_h5[:, 136:835, 472:973, :]
        # resizing
        labels_h5 = np.array([resizer(xi) for xi in labels_h5])
        labels_h5 = np.expand_dims(np.array(labels_h5, dtype=np.uint8), axis=1)
        # fill the sequence with dark images so the sequence length is reached
        labels_h5 = fill_sequence(labels_h5)
        if idx % 2 == 0 and idx > 1:
            tmp_labels = labels_h5
        else:
            if tmp_labels is not None:
                labels_h5 = np.concatenate((tmp_labels, labels_h5))
            labels_h5 = labels_h5.reshape(((labels_h5.shape[0] // seq_len), seq_len, *labels_h5.shape[1:]))
            labels.append(labels_h5)
            tmp_labels = None

    # for idx, path in enumerate(external_images):
    #     for directory in os.listdir(path):
    #         subdir = path + '/' + directory
    #         b_mode_tmp = []
    #         doppler_tmp = []
    #         label_tmp = []
    #         for file in sorted(os.listdir(subdir)):
    #             if 'bmode' in file:
    #                 b_mode_tmp.append(cv2.imread(os.path.join(subdir, file), cv2.IMREAD_GRAYSCALE))
    #             if 'doppler' in file:
    #                 doppler_tmp.append(cv2.imread(os.path.join(subdir, file), cv2.IMREAD_GRAYSCALE))
    #             if 'Label' in file:
    #                 label_tmp.append(cv2.imread(os.path.join(subdir, file), cv2.IMREAD_GRAYSCALE))
    #         assert len(b_mode_tmp) == len(label_tmp) == len(doppler_tmp)
    #         b_mode_tmp = np.array([resizer(xi) for xi in b_mode_tmp])
    #         doppler_tmp = np.array([resizer(xi) for xi in doppler_tmp])
    #         img_tmp = np.concatenate([b_mode_tmp[:, None, :, :], doppler_tmp[:, None, :, :]], axis=1)
    #         # fill the sequence with dark images so the sequence length is reached
    #         img_tmp = fill_sequence(img_tmp)
    #         images.append(img_tmp)
    #         # resizing
    #         label_tmp = np.array([resizer(xi) for xi in label_tmp])
    #         label_tmp = np.expand_dims(np.array(label_tmp, dtype=np.uint8), axis=1)
    #         # fill the sequence with dark images so the sequence length is reached
    #         label_tmp = fill_sequence(label_tmp)
    #         labels.append(label_tmp)

    # labels = np.concatenate(labels)
    # # labels = labels.reshape(((labels.shape[0] // seq_len), seq_len, *labels.shape[1:]))
    #
    # images = np.concatenate(images)
    # # images = images.reshape(((images.shape[0] // seq_len), seq_len, *images.shape[1:]))
    #
    # target_dir = "/data1/volume1/data/felix_data/sequences/"
    # names = ['label', 'bmode', 'doppler']
    # arrays = [labels, images[:, :, [0], :, :], images[:, :, [1], :, :]]
    #
    # for i in range(3):
    #     for seq_index, seq_elem in enumerate(arrays[i]):
    #         for index, element in enumerate(seq_elem):
    #             path = target_dir + 'sequence' + str(seq_index) + '/' + names[i] + '_' + str(index) + '.png'
    #             os.makedirs(os.path.dirname(path), exist_ok=True)
    #             cv2.imwrite(path, element.transpose((1, 2, 0)) * 255 if i == 0 else element.transpose((1, 2, 0)))

    # x_ids = list(range(len(images)))
    # y_ids = list(range(len(labels)))
    #
    # x_train, x_rem, y_train, y_rem = train_test_split(
    #     x_ids, y_ids, train_size=0.8, shuffle=True)
    #
    # x_valid, x_test, y_valid, y_test = train_test_split(
    #     x_rem, y_rem, test_size=0.5, shuffle=True)

    with h5py.File('/data1/volume1/data/felix_data/h5_files/ultrasound_patient_original.h5', mode='w') as h5fw:
        for idx in range(len(images)):
            h5fw.create_dataset(f'x_{idx}', data=images[idx])
            h5fw.create_dataset(f'y_{idx}', data=labels[idx])
        # h5fw.create_dataset('x_test', data=images[x_test])
        # h5fw.create_dataset('y_test', data=labels[y_test])
        # h5fw.create_dataset('x_valid', data=images[x_valid])
        # h5fw.create_dataset('y_valid', data=labels[y_valid])
    print(np.concatenate(images).shape)
    print(np.concatenate(labels).shape)


if __name__ == "__main__":
    main()
