""" Unet training code from https://github.com/zhixuhao/unet """

from learn2seg.feeder import *

from keras.preprocessing.image import ImageDataGenerator
import numpy as np


def check_iou():
    gt_path = "/home/jung/dataset/seg_teeth_eq_v2"
    eval_path = "/home/jung/output/seg_model/eval_0"

    gt_train_path = os.path.join(gt_path, 'val')
    eval_train_path = os.path.join(eval_path)

    # grab images
    image_datagen = ImageDataGenerator()
    gt_gen = image_datagen.flow_from_directory(
        gt_train_path,
        classes=['label'],
        class_mode=None,
        color_mode="grayscale",
        target_size=(496, 352),
        batch_size=1,
        shuffle=False)

    out_gen = image_datagen.flow_from_directory(
        eval_train_path,
        classes=['val'],
        class_mode=None,
        color_mode="grayscale",
        target_size=(496, 352),
        batch_size=1,
        shuffle=False)

    total_iou = 0
    count = 0

    for gt, out in zip(gt_gen, out_gen):

        # Normalize the intensity values
        gt /= 255
        out /= 255

        # Threshold and reshape
        gt = (gt > 0.5).reshape(496, 352)
        out = (out > 0.5).reshape(496, 352)

        # Calculate Intersection over Union
        intersec = np.sum(np.bitwise_and(gt, out))
        union = np.sum(np.bitwise_or(gt, out))
        iou = intersec/union
        total_iou += iou

        print('Calculating IoU for image: ' + str(count), iou)
        count += 1

        if count > 10:
            break

    print(total_iou/count)


def check_iou_file():
    gt_path = "/home/jung/dataset/seg_teeth_eq_v2"
    eval_path = "/home/jung/output/seg_model/eval_0"

    gt_train_path = os.path.join(gt_path, 'val/label')
    eval_train_path = os.path.join(eval_path, 'val')

    n = 10

    for i in range(n):

        gt_im = io.imread(os.path.join(gt_train_path,
                                       '{:06d}.png'.format(i)), as_gray=True)
        out_im = io.imread(os.path.join(eval_train_path,
                                        '{:06d}.png'.format(i)), as_gray=True)

        # Threshold and reshape
        gt = (gt_im > 0.5).reshape(496, 352)
        out = (out_im > 0.5).reshape(496, 352)

        # Calculate Intersection over Union
        intersec = np.sum(np.bitwise_and(gt, out))
        union = np.sum(np.bitwise_or(gt, out))
        iou = intersec/union

        print('Calculating IoU for image: ' + str(i), iou)


if __name__ == "__main__":
    check_iou()
    check_iou_file()
