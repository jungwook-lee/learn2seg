import numpy as np
import os

import skimage.io as io
import learn2seg.metrics as metrics


# TODO: Use this method in train/evalutor
def get_label_dirs(config, iteration):
    datset_config = config['dataset_config']
    out_dir = config['out_path']

    if iteration == 0:
        dataset_path = datset_config['data_path']
        pre_dir = dataset_path
    else:
        it_str = 'eval_{}'.format(iteration - 1)
        pre_dir = os.path.join(out_dir, it_str)

    it_str = 'eval_{}'.format(iteration)
    cur_dir = os.path.join(out_dir, it_str)

    return pre_dir, cur_dir


def iou_filter(pre_dir, cur_dir, iou_thresh):
    """ Given the a dataset dir for a dataset, compare the labels (masks) and
    replace the old one if does not meet the iou threshold

    Assumptions:
        dataset -> [train, val, test]
                    -> label
                        -> 000000.png ...
    """
    # Error check
    if not os.path.exists(pre_dir) and not os.path.exists(cur_dir):
        raise ValueError('Given folder to filter does not exist!')

    for split in ['train', 'val', 'test']:
        pre_path = os.path.join(pre_dir, split, 'label')
        cur_path = os.path.join(cur_dir, split, 'label')

        # get all files in the directories
        pre_dir_files = sorted(os.listdir(pre_path))
        cur_dir_files = sorted(os.listdir(cur_path))

        # read the index file to obtain the data labels
        index_file = np.genfromtxt(os.path.join(pre_dir, split, 'index.txt'),
                             delimiter=',')

        if len(index_file.shape) == 1:
            index_file = np.asarray([index_file])

        # make sure they are the same length when sorted
        assert (len(pre_dir_files) == len(cur_dir_files))

        # get iou's between the two
        for i in range(len(pre_dir_files)):
            pre_im_path = os.path.join(pre_path, pre_dir_files[i])
            cur_im_path = os.path.join(cur_path, cur_dir_files[i])

            pre_im = io.imread(pre_im_path, as_gray=True)
            cur_im = io.imread(cur_im_path, as_gray=True)

            if index_file[i, 0] == 1.:
                # None background bounding boxes
                iou_score = metrics.np_bin_iou(pre_im, cur_im)

                if iou_score < iou_thresh:
                    # Replace the prev im with cur im
                    os.remove(cur_im_path)
                    io.imsave(cur_im_path, pre_im.astype(np.uint8))
                    # print("Replacing file number: {}".format(i))
            else:
                # background class, propagate all the negatives
                os.remove(cur_im_path)
                io.imsave(cur_im_path, pre_im.astype(np.uint8))
