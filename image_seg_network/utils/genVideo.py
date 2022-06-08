import os

import h5py
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
from tracker import stabilize_doppler

def createVideo():
    image_folder = '/data1/volume1/data/felix_data/results/patient2/sequence/'
    video_name = 'video.avi'

    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, 0, 10, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()

def main():
    images = []
    labels = []
    seq_len = 20
    seq_number = 0
    seq_str = 'sequence'
    tmp_labels = None
    tmp_images = None

    seg = False

    streams = ['/data1/volume1/data/felix_data/results_sweeps/original_sweep_31_05_14_53_38.imf', ]

    def resizer(img):
        return cv2.resize(img, (320, 320))

    def fill_sequence(fill_images):
        if len(fill_images) % seq_len != 0:
            return np.concatenate(
                (fill_images, np.zeros((seq_len - (len(fill_images) % seq_len), *fill_images.shape[1:]))))
        else:
            return fill_images

    if not seg:
        for idx, path in enumerate(streams):
            images_h5 = h5py.File(path, 'r')
            # loading and transposing
            print(images_h5.keys())
            images_h5 = images_h5["original_sweep_31_05_14_53_38"][:].transpose((0, 2, 3, 1))
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
            images_h5_doppler[images_h5_doppler > 0] = 255
    else:
        for idx, path in enumerate(streams):
            labels_h5 = h5py.File(path, 'r')
            # loading and transposing
            labels_h5 = labels_h5["sweep_30_05_17_53_10"][:].transpose((0, 2, 3, 1))
            # resizing
            labels_h5 = np.array([resizer(xi) for xi in labels_h5])
            # labels_h5 = np.expand_dims(np.array(labels_h5, dtype=np.uint8), axis=1)

    target_dir = "/data1/volume1/data/felix_data/results/patient1/sweep_30_05_17_53_10/"
    if seg:
        names = ["label"]
        arrays = [labels_h5]
    else:
        names = ['bmode', 'doppler']
        arrays = [images_h5_us, images_h5_doppler]

    size = 320, 320
    fps = 10

    for i in range(len(names)):
        out = cv2.VideoWriter(f'/data1/volume1/data/felix_data/results/patient1/sweep_30_05_17_53_10__{i}.mp4',
                              cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1], size[0]), False)
        for seq_index, seq_elem in enumerate(arrays[i]):
            path = target_dir + names[i] + '_' + str(seq_index) + '.png'
            os.makedirs(os.path.dirname(path), exist_ok=True)
            cv2.imwrite(path, seq_elem)
            out.write(seq_elem)
        out.release()

    # size = 320, 320
    # fps = 10
    # out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1], size[0]), False)
    # for index in range(len(arrays[0])):
    #     out.write(arrays[0][index])
    # x_ids = list(range(len(images)))
    # y_ids = list(range(len(labels)))
    #
    # x_train, x_rem, y_train, y_rem = train_test_split(
    #     x_ids, y_ids, train_size=0.8, shuffle=True)
    #
    # x_valid, x_test, y_valid, y_test = train_test_split(
    #     x_rem, y_rem, test_size=0.5, shuffle=True)

    # with h5py.File('/data1/volume1/data/felix_data/h5_files/ultrasound_patient_original.h5', mode='w') as h5fw:
    #     for idx in range(len(images)):
    #         h5fw.create_dataset(f'x_{idx}', data=images[idx])
    #         h5fw.create_dataset(f'y_{idx}', data=labels[idx])
    #     # h5fw.create_dataset('x_test', data=images[x_test])
    #     # h5fw.create_dataset('y_test', data=labels[y_test])
    #     # h5fw.create_dataset('x_valid', data=images[x_valid])
    #     # h5fw.create_dataset('y_valid', data=labels[y_valid])
    # print(np.concatenate(images).shape)
    # print(np.concatenate(labels).shape)


if __name__ == "__main__":
    createVideo()
