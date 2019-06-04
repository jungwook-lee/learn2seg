import numpy as np
import cv2 as cv
import copy
import skimage.io as io
from matplotlib import pyplot as plt
import matplotlib.patches as patches

from learn2seg.data.TeethBoxes import TeethBoxesDataset


def main():
    dataset_root = '/home/jung/data/teeth/pan-teeth/train/image/'
    out_root = '/home/jung/data/out/'

    for i in range(5):
        fig = plt.figure()
        ax1 = fig.add_subplot(3, 1, 1)

        print("Visualizing Example #" + str(i))

        # work with the first instances of images
        orig_img = io.imread(dataset_root + str(i) + '.jpeg')
        plt.imshow(orig_img)

        ax2 = fig.add_subplot(3, 1, 2)
        out_img = io.imread(out_root + str(i) + '_predict.png',
                            as_gray=True)
        print(out_img.shape)
        # out_img = (((out_img / 255) > 0.25) * 255).astype(np.uint8)
        out_img = (out_img / 255).astype(float)
        print(np.max(out_img), np.min(out_img))
        out_img = out_img > 125
        plt.imshow(out_img)

        # Make a binary mask
        print(out_img.shape)
        print(out_img)

        ax3 = fig.add_subplot(3, 1, 3)
        out_img = out_img[0:, 0:2440]
        out_img = np.expand_dims(out_img, axis=2)
        out_mask = np.repeat(out_img, 3, axis=2)
        print(out_mask.shape)
        print(orig_img.shape)
        orig_img[:, :, 0] = orig_img[:, :, 0] * out_mask[:, :, 0]
        orig_img[:, :, 1] = orig_img[:, :, 1] * out_mask[:, :, 1]
        orig_img[:, :, 2] = orig_img[:, :, 2] * out_mask[:, :, 2]

        print(out_mask.shape)
        print(orig_img.shape)
        plt.imshow(orig_img)

        # boxes = teeth_data.get_boxes(im_index)
        # # print("Number of bounding boxes: " + str(boxes.shape[0]))
        # rect = patches.Rectangle((boxes[bb_index, 0], boxes[bb_index, 1]),
        #                           boxes[bb_index, 2], boxes[bb_index, 3],
        #                          linewidth=1, edgecolor='r',
        #                          facecolor='none')
        # ax1.add_patch(rect)
        #
        # # img = crop_im(im, boxes[0, :])
        # mask = np.zeros(img.shape[:2], np.uint8)
        # bgdModel = np.zeros((1, 65), np.float64)
        # fgdModel = np.zeros((1, 65), np.float64)
        #
        # rect = tuple(boxes[bb_index, :])
        #
        # cv.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)
        # mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        # img = img * mask2[:, :, np.newaxis]
        #
        # ax2 = fig.add_subplot(2, 1, 2)
        # rect = patches.Rectangle((boxes[bb_index, 0], boxes[bb_index, 1]),
        #                           boxes[bb_index, 2], boxes[bb_index, 3],
        #                          linewidth=1, edgecolor='r',
        #                          facecolor='none')
        # ax2.add_patch(rect)
        #
        # plt.imshow(img)
        #
        # # plt.pause(.5)
        # plt.savefig('/home/jung/Pictures/grabcut_demo/exp_'
        #             + str(im_index) + '_'
        #             + str(bb_index), dpi=300)
        plt.show()
        plt.pause(.5)
        plt.close(fig)
    #


if __name__ == '__main__':
    main()
