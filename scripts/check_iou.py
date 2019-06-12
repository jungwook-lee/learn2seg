""" Unet training code from https://github.com/zhixuhao/unet """

from learn2seg.feeder import *

import learn2seg.metrics as iou_eval

from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session


def check_iou():
    # Setup TF configs
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)
    set_session(sess)

    gt_path = "/home/jung/dataset/seg_teeth_eq_v2"
    eval_path = "/home/jung/output/teeth_eq_v2/eval_0"

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
        seed=1)

    out_gen = image_datagen.flow_from_directory(
        eval_train_path,
        classes=['val'],
        class_mode=None,
        color_mode="grayscale",
        target_size=(496, 352),
        batch_size=1,
        seed=1)

    total_iou = 0
    count = 0

    for gt, out in zip(gt_gen, out_gen):

        k_iou = sess.run(iou_eval.iou_score(gt, out))
        print(k_iou)

        # Threshold
        gt = gt.reshape(496, 352).astype(np.bool)
        out = out.reshape(496, 352).astype(np.bool)

        # Calculate Intersection over Union
        intersec = np.sum(np.bitwise_and(gt, out))
        union = np.sum(np.bitwise_or(gt, out))

        iou = intersec/union
        print(iou)
        total_iou += iou

        count += 1
        print('Calculating IoU for image: ' + str(count + 1))

        if count >= 10:
            break

        # k_iou = sess.run(iou_eval.iou_score(pred, true))
        # total_keras_iou += k_iou

    print(total_iou/count)

    # Try evaluating with the model version

    # 0.577065347837544 for train
    # 0.580457608891385 for val


if __name__ == "__main__":
    check_iou()
