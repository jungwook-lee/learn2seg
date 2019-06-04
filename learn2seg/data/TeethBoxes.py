# Class to handle teeth data calls
import os
import cv2
import json
import numpy as np

# TODO: Don't save it as a hard-coded location

module_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(module_path)
root_path = os.path.dirname(root_path)
dataset_root = os.path.join(root_path, 'dataset/orig_teeth/pan-teeth')


class TeethBoxesDataset(object):
    img_path = os.path.join(dataset_root, 'img')
    ann_path = os.path.join(dataset_root, 'ann')  # ann for annotations

    # Public Variables
    size = None
    bb_size = None  # TODO: Fix this later
    img_files = None
    ann_files = None

    def __init__(self):
        # Check if the root path exists first
        if not os.path.exists(dataset_root):
            raise RuntimeError("Teeth Dataset is missing at: " + dataset_root)

        # Check all filename and sort
        self.img_files = sorted(os.listdir(self.img_path))
        self.ann_files = sorted(os.listdir(self.ann_path))
        assert (len(self.img_files) == len(self.ann_files))
        self.size = len(self.img_files)

        # Count all teeth bb's
        bb_total = 0
        for n in range(self.size):
            boxes = self.get_boxes(n)
            bb_total += len(boxes)
        self.bb_size = bb_total

    def get_image(self, index):
        """ Given index of the data, return the corresponding image """
        img_path = os.path.join(self.img_path, self.img_files[index])

        # Read in as a greyscale image
        im = cv2.imread(img_path, 1)

        # Code to turn the image to Grayscale
        # im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY, im)
        # im = np.expand_dims(im, axis=2)
        return im

    def get_boxes(self, index):
        """ Given index of the data, return the corresponding BB """
        boxes = []
        filename = os.path.join(self.ann_path, self.ann_files[index])

        if filename:
            with open(filename, 'r') as f:
                label_dict = json.load(f)

        for i in range(len(label_dict)):
            json_dict = label_dict[i]['shape']
            bottom, left, width, height = get_box_from_ann(json_dict)
            boxes.append([bottom, left, width, height])

        return np.asarray(boxes)


def get_box_from_ann(box_dict):
    """ Function to retrieve bounding boxes given an image Id"""
    x_left = box_dict['startX']
    y_bottom = box_dict['startY']
    x_width = box_dict['endX'] - box_dict['startX']
    y_height = box_dict['endY'] - box_dict['startY']

    return x_left, y_bottom, x_width, y_height
