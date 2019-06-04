import os
import os.path as path
import numpy as np
import skimage.io as io

import learn2seg.tools as tools


def check_class_balance(label_path, im_size):
    total_pixels = im_size[0] * im_size[1]
    files = sorted(os.listdir(label_path))

    total_pos = 0
    total_neg = 0

    for f_name in files:
        img = io.imread(path.join(label_path, f_name), as_gray=True)
        img = (img / 255).astype(int)
        # Get all summation of 0, 1's
        pos = np.sum(img)
        total_pos += pos
        total_neg += (total_pixels - pos)

    all_pixels = total_pixels * len(files)
    print("Percent of total labels positives: ", total_pos/all_pixels)
    print("Percent of total labels negatives: ", total_neg/all_pixels)


if __name__ == '__main__':
    module_path = path.dirname(path.abspath(__file__))
    root_path = path.dirname(module_path)
    config_file = path.join(root_path, 'configs/unet_config.yml')

    config = tools.get_configs(config_file)
    data_path = config['data_path']
    im_size = tuple(config['image_size'])

    # Checking class balance for train set
    print("Checking binary balance for train set ...")
    label_path = path.join(data_path, 'train/label')
    check_class_balance(label_path, im_size)

    # Checking class balance for validation set
    print("Checking binary balance for validation set ...")
    label_path = path.join(data_path, 'val/label')
    check_class_balance(label_path, im_size)

    # Checking class balance for test
    print("Checking binary balance for test set ...")
    label_path = path.join(data_path, 'test/label')
    check_class_balance(label_path, im_size)
