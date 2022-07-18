import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter
from matplotlib.gridspec import GridSpec


def main():
    seg_results = ['/data1/volume1/data/felix_data/results/zhongliang_reidentification/sweep_16_06_16_44_31.h5',
                   '/data1/volume1/data/felix_data/results/zhongliang_reidentification/sweep_16_06_16_45_18.h5'
                   ]
    seg_results_name = ['sweep_16_06_16_44_31', 'sweep_16_06_16_45_18']
    seg_lab = ['/data1/volume1/data/felix_data/results/zhongliang_reidentification/sweep_16_06_16_44_31_labelled.h5',
               '/data1/volume1/data/felix_data/results/zhongliang_reidentification/sweep_16_06_16_45_18_labelled.h5'
               ]

    seg_lab_name = ['sweep_16_06_16_44_31_labelled', 'sweep_16_06_16_45_18_labelled']

    seg_results_start = [270, 0]

    for (idx_r, path_r), (idx_l, path_l) in zip(enumerate(seg_results), enumerate(seg_lab)):
        seg_result = h5py.File(path_r, 'r')
        seg_lab = h5py.File(path_l, 'r')
        # loading
        seg_result = np.squeeze(
            seg_result[seg_results_name[idx_r]][seg_results_start[idx_r]:].transpose(
                (0, 2, 3, 1)))
        seg_lab = np.squeeze(seg_lab[seg_lab_name[idx_l]][seg_results_start[idx_l]:].transpose(
            (0, 2, 3, 1)))
        dice_score = []
        iou = []

        for idx, (l_seg, l_truth) in enumerate(zip(seg_result, seg_lab)):
            img_A = np.array(l_seg[:-1, :] >= 10, dtype=bool).flatten()
            img_B = np.array(l_truth[:, 10:507] >= 1, dtype=bool).flatten()
            dice_score.append(2.0 * np.sum(img_A * img_B) / (np.sum(img_A) + np.sum(img_B) + 1e-6))
            iou.append(np.sum(img_A * img_B) / float(np.sum(img_A + img_B)))

        print(np.average(dice_score), np.average(iou))
        print(np.std(dice_score), np.std(iou))
        # plt_1 = plt.figure(figsize=(12, 12))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

        ax1.plot(dice_score, label='Dice Score')
        ax1.plot(iou, label='IoU')
        ax1.legend()
        # plt.tick_params(
        # axis='x',  # changes apply to the x-axis
        # which='both',  # both major and minor ticks are affected
        # bottom=False,  # ticks along the bottom edge are off
        # top=False,  # ticks along the top edge are off
        # labelbottom=False)  # labels along the bottom edge are off
        ax1.grid(False)

        ax2.hist([dice_score, iou], label=['Dice Score', "IoU"], bins=10, orientation='horizontal',
                 weights=[np.ones(len(dice_score)) / len(dice_score), np.ones(len(iou)) / len(iou)])

        # ax2.hist(iou, bins=30, label='IoU', orientation='horizontal',weights=np.ones(len(dice_score)) / len(dice_score), alpha=0.5)
        ax2.xaxis.set_major_formatter(PercentFormatter(1, decimals=0, symbol=""))
        # plt.hist(dice_score, density=False, bins=30)  # density=False would make counts
        ax1.set(ylabel='Dice Score / IoU', xlabel='Frames')
        ax2.set(xlabel='Frequency (%)')
        ax2.legend()
        fig.tight_layout()
        plt.show()
        fig.savefig(
            '/data1/volume1/data/felix_data/results/zhongliang_reidentification/' + seg_results_name[idx_r] + ".png")
        fig.savefig(
            '/data1/volume1/data/felix_data/results/zhongliang_reidentification/' + seg_results_name[idx_r] + ".svg",
            format='svg', dpi=1200)
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
    main()
