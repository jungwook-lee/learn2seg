import os
import cv2
import json
import numpy as np

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
data_path = os.path.join(root_path, 'dataset/seg_teeth')
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

width_inf = 1.2
height_inf = 1.2

VISUALIZE = False


def sample_for_negatives(im, boxes, neg_ratio=1):
    # fig = plt.figure()

    # if image shape is smaller than default size, half the samples
    if im.shape[0] < 1280:
        neg_ratio = 0.5

    bin_im = np.zeros(im.shape).astype(np.uint8)

    # Get a binary image file masked with annotations
    for b in range(boxes.shape[0]):
        cur_box = boxes[b, :]
        end_width = cur_box[0] + cur_box[2]
        end_height = cur_box[1] + cur_box[3]
        bin_im[cur_box[1]:end_height, cur_box[0]:end_width] = 255

    # for i in range(boxes.shape[0] * neg_ratio):
    total_neg_bb = 0
    neg_boxes = []
    while True:
        # Find and select a random box size
        avg_width = math.floor(np.sum(boxes[:, 2]) / boxes.shape[0])
        avg_height = math.floor(np.sum(boxes[:, 3]) / boxes.shape[0])

        # Check if the box contains inside the image
        width = random.randint(0, math.floor(im.shape[1] - avg_width))
        height = random.randint(0, math.floor(im.shape[0] - avg_height))

        # Check range of the box contains any pixel in the teeth
        if not (width + (avg_width / 2)) < im.shape[1] and (width - (avg_width / 2)) > 0:
            continue
        if not (height + (avg_height / 2)) < im.shape[0] and (height - (avg_height / 2)) > 0:
            continue

        bin_box = bin_im[height:height + avg_height, width:width + avg_width]

        if np.sum(bin_box) <= 0:
            total_neg_bb += 1
            neg_boxes.append([width, height, avg_width, avg_height, ])

            # print(neg_boxes)
            #             # print(total_neg_bb)

            test_box = neg_boxes[total_neg_bb - 1]
            # bin_box[:, :]

            # Sampled area, so set a value
            bin_im[test_box[1]:(test_box[1] + test_box[3]),
                   test_box[0]:(test_box[0] + test_box[2])] = 128

        if total_neg_bb >= (neg_ratio * boxes.shape[0]):
            break

    # # # Return the box sizes
    # plt.imshow(bin_im)
    # plt.pause(1)
    # plt.close(fig)

    return np.asarray(neg_boxes)


def main():
    teeth_data = TeethBoxesDataset()

    # Define paths
    train_index_file = os.path.join(data_path, 'train/index.txt')
    if not os.path.exists(train_index_file):
        open(train_index_file, 'a').close()

    val_index_file = os.path.join(data_path, 'val/index.txt')
    if not os.path.exists(val_index_file):
        open(val_index_file, 'a').close()

    test_index_file = os.path.join(data_path, 'test/index.txt')
    if not os.path.exists(test_index_file):
        open(test_index_file, 'a').close()

    # Split to train/val/test sets
    ratio_list = np.asarray([0.8, 0.15, 0.05])
    split_nums = (teeth_data.bb_size * ratio_list).astype(np.int)
    left_over = teeth_data.bb_size - np.sum(split_nums)
    split_nums[0] = split_nums[0] + left_over
    assert(np.sum(split_nums) == teeth_data.bb_size)

    # Make randomized splits to make the data
    # Calculate the number of bounding boxes
    nums = [x for x in range(teeth_data.bb_size)]
    random.shuffle(nums)

    # Total bounding boxes in the dataset
    train_nums = nums[:split_nums[0]]
    val_nums = nums[split_nums[0]:np.sum(split_nums[:2])]
    test_nums = nums[np.sum(split_nums[:2]):np.sum(split_nums)]
    print('Splits divided into', len(train_nums), len(val_nums), len(test_nums))

    out_index = 0
    train_index = 0
    val_index = 0
    test_index = 0

    max_width = 0
    max_height = 0

    for i in range(teeth_data.size):
    # for i in range(1):
        if VISUALIZE:
            fig = plt.figure()

        index = i
        print("Processing Image #" + str(index))

        # work with the first instances of images
        im = teeth_data.get_image(index)
        print("Image size: ", im.shape)
        boxes = teeth_data.get_boxes(index)

        print("Number of bounding boxes: " + str(boxes.shape[0]))
        inf_boxes = copy.deepcopy(boxes)

        for b in range(boxes.shape[0]):
            # Inflate the bounding box region to get back/fore ground
            box_range = boxes[b, :]
            inf_range = inf_box_range(inf_boxes[b, :],
                                      width_inf, height_inf)

            # Check for whether the inflated range is within the network size
            assert(inf_range[0] > 0 and inf_range[1] > 0)
            assert (inf_range[2] < 512 and inf_range[3] < 512)

            # image i axis is axis 2 for numpy
            assert (inf_range[0] + inf_range[2] < im.shape[1])
            # image j axis is axis 1 for numpy
            assert (inf_range[1] + inf_range[3] < im.shape[0])

            if VISUALIZE:
                ax = fig.add_subplot(2, 2, 1)
                plt.imshow(im)
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
                plt.draw()

            # Create 2 512^2 numpy arrays to save the images
            im_out = np.zeros((512, 512, 3)).astype(np.uint8)

            # Save the image as the part of the dataset
            cropped_im = crop_im(im, inf_range)

            width = inf_range[2]
            length = inf_range[3]

            # Find the maximum size of image
            if width > max_width:
                max_width = width
            if length > max_height:
                max_height = length

            # Position the cropped image to the center
            new_x, new_y = math.floor((512 - width) / 2), \
                           math.floor((512 - length) / 2)

            end_x, end_y = new_x + width, new_y + length
            im_out[new_y:end_y, new_x:end_x] = cropped_im

            # Save the bounding box as the label image
            la_out = np.zeros((512, 512, 1)).astype(np.uint8)
            width = box_range[2]
            length = box_range[3]
            new_x, new_y = math.floor((512 - width) / 2), \
                           math.floor((512 - length) / 2)
            end_x, end_y = new_x + width, new_y + length
            la_out[new_y:end_y, new_x:end_x] = 255

            ### -------- Saving boxes -------- ###
            if out_index in train_nums:
                imageio.imsave(train_im_path + str(train_index) + '.png', im_out)
                imageio.imsave(train_la_path + str(train_index) + '.png', la_out)

                with open(train_index_file, 'ab') as f:
                    np.savetxt(f, [(1, i, b)], delimiter=',', fmt='%d')

                train_index += 1
            elif out_index in val_nums:
                imageio.imsave(val_im_path + str(val_index) + '.png', im_out)
                imageio.imsave(val_la_path + str(val_index) + '.png', la_out)

                with open(val_index_file, 'ab') as f:
                    np.savetxt(f, [(1, i, b)], delimiter=',', fmt='%d')

                val_index += 1
            else:
                imageio.imsave(test_im_path + str(test_index) + '.png', im_out)
                imageio.imsave(test_la_path + str(test_index) + '.png', la_out)

                with open(test_index_file, 'ab') as f:
                    np.savetxt(f, [(1, i, b)], delimiter=',', fmt='%d')

                test_index += 1

            out_index += 1
            # print(out_index)

            if VISUALIZE:
                # Display cropped image
                ax = fig.add_subplot(2, 2, 2)
                plt.imshow(im_out)

                # Display label
                la_out = np.repeat(la_out, 3, axis=2)
                # print(la_out.shape)
                ax = fig.add_subplot(2, 2, 3)
                plt.imshow(la_out)

                # Display image with label
                ax = fig.add_subplot(2, 2, 4)
                la_out = la_out[:, :, :].astype(bool)
                masked_im = im_out[:, :, :] * la_out
                plt.imshow(masked_im)
                plt.pause(.5)

        ### -------- Negative Sampling --- ###
        neg_boxes = sample_for_negatives(im, boxes)

        # Get respective division of train/val/test
        neg_split = (neg_boxes.shape[0] * ratio_list).astype(np.int)
        left = neg_boxes.shape[0] - np.sum(neg_split)
        neg_split[0] = neg_split[0] + left
        assert(np.sum(neg_split) == neg_boxes.shape[0])

        neg_nums = [x for x in range(neg_boxes.shape[0])]
        random.shuffle(neg_nums)

        # Total bounding boxes in the dataset
        neg_train_nums = neg_nums[:neg_split[0]]
        neg_val_nums = neg_nums[neg_split[0]:np.sum(neg_split[:2])]
        # neg_test_nums = nums[np.sum(neg_nums[:2]):np.sum(neg_nums)]

        for b in range(neg_boxes.shape[0]):
            cur_neg_range = neg_boxes[b, :]
            im_out = np.zeros((512, 512, 3)).astype(np.uint8)
            cropped_im = crop_im(im, neg_boxes[b])

            width = cur_neg_range[2]
            length = cur_neg_range[3]

            # Position the cropped image to the center
            new_x, new_y = math.floor((512 - width) / 2), \
                           math.floor((512 - length) / 2)

            end_x, end_y = new_x + width, new_y + length
            im_out[new_y:end_y, new_x:end_x] = cropped_im

            # For negatives all the pixels are negatives
            la_out = np.zeros((512, 512, 3)).astype(np.uint8)

            # Save image, label, and update index file
            if b in neg_train_nums:
                imageio.imsave(train_im_path + str(train_index) + '.png',
                               im_out)
                imageio.imsave(train_la_path + str(train_index) + '.png',
                               la_out)

                with open(train_index_file, 'ab') as f:
                    np.savetxt(f, [(0, i, b)], delimiter=',', fmt='%d')

                train_index += 1

            elif b in neg_val_nums:
                imageio.imsave(val_im_path + str(val_index) + '.png',
                               im_out)
                imageio.imsave(val_la_path + str(val_index) + '.png',
                               la_out)

                with open(val_index_file, 'ab') as f:
                    np.savetxt(f, [(0, i, b)], delimiter=',', fmt='%d')

                val_index += 1
            else:
                imageio.imsave(test_im_path + str(test_index) + '.png',
                               im_out)
                imageio.imsave(test_la_path + str(test_index) + '.png',
                               la_out)

                with open(test_index_file, 'ab') as f:
                    np.savetxt(f, [(0, i, b)], delimiter=',', fmt='%d')

                test_index += 1

        # print(max_width, max_height)
        if VISUALIZE:
            plt.close(fig)


if __name__ == '__main__':
    # Check if dir exists
    if os.path.exists(orig_path):
        print('Data path exists!')
    else:
        raise RuntimeError()

    if os.path.exists(data_path):
        raise RuntimeError('Seg_Teeth Data path exists! Delete it first!')

    os.mkdir(data_path)

    os.mkdir(train_path)
    os.mkdir(train_im_path)
    os.mkdir(train_la_path)

    os.mkdir(val_path)
    os.mkdir(val_im_path)
    os.mkdir(val_la_path)

    os.mkdir(test_path)
    os.mkdir(test_im_path)
    os.mkdir(test_la_path)

    print('Proceed to process the images')
    main()
