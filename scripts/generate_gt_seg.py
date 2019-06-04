import random
import copy
import os
import imageio
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches

from learn2seg.data.TeethBoxes import TeethBoxesDataset

# VISUALIZE = True


def main():
    teeth_data = TeethBoxesDataset()

    for i in range(teeth_data.size):
        # fig = plt.figure()
        # ax1 = fig.add_subplot(3, 1, 1)
        # retrieve all files
        print("Processing Image #" + str(i))
        print(teeth_data.img_files[i])

        # Show the image with bounding boxes
        im = teeth_data.get_image(i)
        # plt.imshow(im)
        save_path = '/home/jung/data/teeth/pan-teeth/train/image/'
        imageio.imsave(save_path + str(i) + '.jpeg', im)

        boxes = teeth_data.get_boxes(i)
        print("Number of bounding boxes: " + str(boxes.shape[0]))

        for b in range(boxes.shape[0]):
            rect = patches.Rectangle((boxes[b, 0], boxes[b, 1]),
                                      boxes[b, 2], boxes[b, 3],
                                     linewidth=1, edgecolor='r',
                                     facecolor='none')
            # ax1.add_patch(rect)
            plt.draw()

        # Create a new image with the teeth as the labels
        gt_mask = np.zeros(im.shape).astype(np.uint8)
        for b in range(boxes.shape[0]):
            x_min, y_min = boxes[b, 0], boxes[b, 1]
            x_max = boxes[b, 0] + boxes[b, 2]
            y_max = boxes[b, 1] + boxes[b, 3]
            gt_mask[y_min:y_max, x_min:x_max, :] = 255

        save_path = '/home/jung/data/teeth/pan-teeth/train/label/'
        imageio.imsave(save_path + str(i) + '.png', gt_mask)

        # ax2 = fig.add_subplot(3, 1, 2)
        # plt.imshow(gt_mask)
        # gt_mask = gt_mask[:, :, :].astype(bool)
        # print(gt_mask.shape)
        # masked_im = im[:, :, :] * gt_mask
        # ax3 = fig.add_subplot(3, 1, 3)
        # plt.imshow(masked_im)

        # plt.pause(.5)
        # plt.close(fig)
        # plt.show()


if __name__ == '__main__':
    main()
