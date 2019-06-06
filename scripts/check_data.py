# Script to check class imbalance in the data

import os
import os.path as path

import numpy as np
import skimage.io as io


def check_class_balance(label_dir, im_size):
    """ Checks the class imbalance of given label file images for segmentation

    Args:
        label_dir (str): path to the label directory
        im_size (tuple): image size as (height, width)

    """
    total_pixels = im_size[0] * im_size[1]
    files = sorted(os.listdir(label_dir))

    total_pos = 0
    total_neg = 0

    for f_name in files:
        img = io.imread(path.join(label_dir, f_name), as_gray=True)
        img = (img / 255).astype(int)

        pos = np.sum(img)
        total_pos += pos
        total_neg += (total_pixels - pos)

    all_pixels = total_pixels * len(files)
    print("Percent of total labels positives: ", total_pos/all_pixels)
    print("Percent of total labels negatives: ", total_neg/all_pixels)


if __name__ == '__main__':
    data_path = '/home/jung/dataset/seg_teeth_v2'
    im_size = (496, 352)

    # Checking class balance for train set
    print("Checking binary balance for train set ...")
    label_dir = path.join(data_path, 'train/label')
    check_class_balance(label_dir, im_size)

    # Checking class balance for validation set
    print("Checking binary balance for validation set ...")
    label_dir = path.join(data_path, 'val/label')
    check_class_balance(label_dir, im_size)

    # Checking class balance for test
    print("Checking binary balance for test set ...")
    label_dir = path.join(data_path, 'test/label')
    check_class_balance(label_dir, im_size)
