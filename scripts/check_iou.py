""" Unet training code from https://github.com/zhixuhao/unet """

from learn2seg.feeder import *

import learn2seg.metrics as iou_eval

from keras.preprocessing.image import ImageDataGenerator
import numpy as np


def check_iou():
    gt_path = "/home/jung/dataset/seg_teeth_eq_v2"
    eval_path = "/home/jung/output/teeth_eq_v2/eval_0"

    gt_train_path = os.path.join(gt_path, 'train')
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
        seed=1)

    out_gen = image_datagen.flow_from_directory(
        eval_train_path,
        classes=['train'],
        class_mode=None,
        color_mode="grayscale",
        target_size=(496, 352),
        batch_size=1,
        seed=1)

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

        count += 1
        print('Calculating IoU for image: ' + str(count + 1), iou)

        if count >= gt_gen.n:
            break

    print(total_iou/count)


if __name__ == "__main__":
    check_iou()
