import random
from matplotlib import pyplot as plt

from learn2seg.data.TeethBoxes import TeethBoxesDataset
from learn2seg.tools.tools import crop_im


def main():
    teeth_data = TeethBoxesDataset()

    while True:
        index = random.randint(0, teeth_data.size - 1)
        print("Visualizing Example #" + str(index))

        # work with the first instances of images
        im = teeth_data.get_image(index)

        boxes = teeth_data.get_boxes(index)
        print("Number of bounding boxes: " + str(boxes.shape[0]))

        cropped_im = crop_im(im, boxes[0, :])

        plt.imshow(cropped_im)

        plt.pause(.5)
        plt.draw()


if __name__ == '__main__':
    main()
