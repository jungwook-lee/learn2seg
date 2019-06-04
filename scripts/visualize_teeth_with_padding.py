import random
import copy
from matplotlib import pyplot as plt
import matplotlib.patches as patches

from learn2seg.data.TeethBoxes import TeethBoxesDataset
from learn2seg.tools.tools import *


def main():
    teeth_data = TeethBoxesDataset()
    width_inf = 1.4
    height_inf = 1.2

    while True:
        fig, ax = plt.subplots(1)

        index = random.randint(0, teeth_data.size - 1)
        print("Visualizing Example #" + str(index))

        # work with the first instances of images
        im = teeth_data.get_image(index)

        boxes = teeth_data.get_boxes(index)
        inf_boxes = copy.deepcopy(boxes)
        teeth_index = random.randint(0, boxes.shape[0] - 1)
        print("Number of bounding boxes: " + str(boxes.shape[0]))

        # Inflate the bounding box region to get back/fore ground
        inf_range = inf_box_range(inf_boxes[teeth_index, :],
                                  width_inf, height_inf)
        # print(inf_boxes)
        plt.imshow(im)

        box_range = boxes[teeth_index, :]
        rect = patches.Rectangle((box_range[0], box_range[1]),
                                  box_range[2], box_range[3],
                                  linewidth=1, edgecolor='r',
                                  facecolor='none')
        ax.add_patch(rect)

        rect = patches.Rectangle((inf_range[0], inf_range[1]),
                                  inf_range[2], inf_range[3],
                                  linewidth=1, edgecolor='g',
                                  facecolor='none')
        ax.add_patch(rect)
        # plt.draw()

        plt.pause(10)
        plt.close(fig)


if __name__ == '__main__':
    main()
