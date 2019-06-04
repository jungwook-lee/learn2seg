# Check class imbalance in the current dataset
import os
import cv2
import json
import numpy as np
import skimage.io as io

import imageio
import random
import copy
from matplotlib import pyplot as plt
import matplotlib.patches as patches

from learn2seg.data.TeethBoxes import TeethBoxesDataset
from learn2seg.image.tools import *

# TODO: Don't save it as a hard-coded location
# dataset_root = 'dataset/data/teeth/pan-teeth'

module_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(module_path)
data_path = os.path.join(root_path, 'dataset/seg_teeth_v2')
orig_path = os.path.join(root_path, 'dataset/orig_teeth')

train_path = os.path.join(data_path, 'train/')
train_im_path = os.path.join(train_path, 'image/')
train_la_path = os.path.join(train_path, 'label/')

val_path = os.path.join(data_path, 'val/')
val_im_path = os.path.join(val_path, 'image/')
val_la_path = os.path.join(val_path, 'label/')

test_path = os.path.join(data_path, 'test/')
test_im_path = os.path.join(test_path, 'image/')
test_la_path = os.path.join(test_path, 'label/')


VISUALIZE = False
IMG_SIZE = (512, 512)


def check_class_balance(path):
    total_pixels = IMG_SIZE[0] * IMG_SIZE[1]
    files = sorted(os.listdir(path))

    total_pos = 0
    total_neg = 0
    for f_name in files:
        img = io.imread(os.path.join(path, f_name), as_gray=True)
        img = (img / 255).astype(int)
        # Get all summation of 0, 1's
        pos = np.sum(img)
        total_pos += pos
        total_neg += (total_pixels - pos)
    all_pixels = total_pixels * len(files)
    pos_ratio = total_pos / all_pixels
    neg_ratio = total_neg / all_pixels

    print("Percent of total labels positives, negative: ", (pos_ratio, neg_ratio))

    return


if __name__ == '__main__':
    check_class_balance(train_la_path)
    check_class_balance(val_la_path)
    check_class_balance(test_la_path)
