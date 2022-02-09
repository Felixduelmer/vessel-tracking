import h5py
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
from skimage import transform


def main():
    images_h5 = h5py.File('/home/robotics-verse/projects/felix/DataSet/felix_data/Zhongliang/images.h5', 'r')
    images = images_h5['Zhongliang-01'][:].transpose(
        (0, 2, 3, 1))
    labels_h5 = h5py.File('/home/robotics-verse/projects/felix/DataSet/felix_data/Zhongliang/labels.h5', 'r')
    cropped_labels = labels_h5['Images'][:, :, 136:854,
                                         472:973].transpose((0, 2, 3, 1))
    cropped_bmode_images = images[:, 136:854, 472:973, :]
    cropped_doppler_images = images[:, 136:854, 1004:1505, :]

    def resizer(img): return cv2.resize(
        img, (320, 320))
    resized_bmode_images = np.array([resizer(xi)
                                     for xi in cropped_bmode_images])
    resized_doppler_images = np.array([resizer(xi)
                                       for xi in cropped_doppler_images])
    label_array = np.array([resizer(xi)
                            for xi in cropped_labels])

    for counter, image in enumerate(resized_doppler_images):
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        # Threshold the HSV image to get only blue colors
        mask = cv2.inRange(
            hsv, np.array([0, 100, 20]), np.array([180, 255, 255]))
        # Bitwise-AND mask and original image
        res = cv2.bitwise_and(
            image, image, mask=mask)
        resized_doppler_images[counter] = res

    image_array = np.concatenate(
        [resized_bmode_images[:, None, :, :, :], resized_doppler_images[:, None, :, :, :]], axis=1)

    X_train, X_rem, y_train, y_rem = train_test_split(
        image_array, label_array, train_size=0.8, shuffle=False)

    X_valid, X_test, y_valid, y_test = train_test_split(
        X_rem, y_rem, test_size=0.5, shuffle=False)

    with h5py.File('ultrasound.h5', mode='w') as h5fw:
        h5fw.create_dataset('x_train', data=X_train)
        h5fw.create_dataset('y_train', data=y_train)
        h5fw.create_dataset('x_test', data=X_test)
        h5fw.create_dataset('y_test', data=y_test)
        h5fw.create_dataset('x_valid', data=X_valid)
        h5fw.create_dataset('y_valid', data=y_valid)
        print(h5fw['x_train'].shape)
        print(h5fw['x_test'].shape)
        print(h5fw['x_valid'].shape)


if __name__ == "__main__":

    main()