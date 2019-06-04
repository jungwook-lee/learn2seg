import numpy as np
import cv2 as cv
import copy
from matplotlib import pyplot as plt
import matplotlib.patches as patches

from learn2seg.data.TeethBoxes import TeethBoxesDataset


def grabcut_demo(im_index, bb_index):
    teeth_data = TeethBoxesDataset()

    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)

    # print("Visualizing Example #" + str(im_index))

    # work with the first instances of images
    teeth_img = teeth_data.get_image(im_index)
    img = copy.deepcopy(teeth_img)
    plt.imshow(teeth_img)

    boxes = teeth_data.get_boxes(im_index)
    # print("Number of bounding boxes: " + str(boxes.shape[0]))
    rect = patches.Rectangle((boxes[bb_index, 0], boxes[bb_index, 1]),
                              boxes[bb_index, 2], boxes[bb_index, 3],
                             linewidth=1, edgecolor='r',
                             facecolor='none')
    ax1.add_patch(rect)

    # img = crop_im(im, boxes[0, :])
    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    rect = tuple(boxes[bb_index, :])

    cv.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img = img * mask2[:, :, np.newaxis]

    ax2 = fig.add_subplot(2, 1, 2)
    rect = patches.Rectangle((boxes[bb_index, 0], boxes[bb_index, 1]),
                              boxes[bb_index, 2], boxes[bb_index, 3],
                             linewidth=1, edgecolor='r',
                             facecolor='none')
    ax2.add_patch(rect)

    plt.imshow(img)

    # plt.pause(.5)
    plt.savefig('/home/jung/Pictures/grabcut_demo/exp_'
                + str(im_index) + '_'
                + str(bb_index), dpi=300)

    plt.close(fig)
    # plt.show()


if __name__ == '__main__':
    # Run grab cut on the first image with 27 teeth bounding boxes
    bb_index, im_index = 0, 0
    while True:
        print('Running grabcut on teeth #' + str(bb_index))
        grabcut_demo(im_index, bb_index)
        bb_index += 1
        if bb_index > 27:
            break
