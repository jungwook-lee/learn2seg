import random
from matplotlib import pyplot as plt
import matplotlib.patches as patches

from learn2seg.data.TeethBoxes import TeethBoxesDataset


def main():
    fig, ax = plt.subplots(1)
    teeth_data = TeethBoxesDataset()

    # retrieve all files
    index = random.randint(0, teeth_data.size)
    print("Visualizing Example #" + str(index))

    # work with the first instances of images
    im = teeth_data.get_image(index)
    plt.imshow(im)

    boxes = teeth_data.get_boxes(index)
    print("Number of bounding boxes: " + str(boxes.shape[0]))

    for i in range(boxes.shape[0]):
        rect = patches.Rectangle((boxes[i, 0], boxes[i, 1]), boxes[i, 2],
                                 boxes[i, 3], linewidth=1, edgecolor='r',
                                 facecolor='none')
        ax.add_patch(rect)
        plt.draw()

    plt.show()


if __name__ == '__main__':
    main()
