# Copied from https://gist.github.com/Kautenja/69d306c587ccdf464c45d28c1545e580
from keras import backend as K
import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session


def castF(x):
    return K.cast(x, K.floatx())


def castB(x):
    return K.cast(x, bool)


def iou_loss_core(true,pred):  #this can be used as a loss if you make it negative
    intersection = true * pred
    notTrue = 1 - true
    union = true + (notTrue * pred)

    return (K.sum(intersection, axis=-1) + K.epsilon()) / (K.sum(union, axis=-1) + K.epsilon())


def iou_score(true, pred): #any shape can go - can't be a loss function

    tresholds = [0.5 + (i*.05) for i in range(10)]

    #flattened images (batch, pixels)
    true = K.batch_flatten(true)
    pred = K.batch_flatten(pred)
    pred = castF(K.greater(pred, 0.5))

    #total white pixels - (batch,)
    trueSum = K.sum(true, axis=-1)
    predSum = K.sum(pred, axis=-1)

    #has mask or not per image - (batch,)
    true1 = castF(K.greater(trueSum, 1))
    pred1 = castF(K.greater(predSum, 1))

    #to get images that have mask in both true and pred
    truePositiveMask = castB(true1 * pred1)

    #separating only the possible true positives to check iou
    testTrue = tf.boolean_mask(true, truePositiveMask)
    testPred = tf.boolean_mask(pred, truePositiveMask)

    # print(testPred)

    #getting iou and threshold comparisons
    iou = iou_loss_core(testTrue, testPred)
    truePositives = [castF(K.greater(iou, tres)) for tres in tresholds]

    #mean of thressholds for true positives and total sum
    truePositives = K.mean(K.stack(truePositives, axis=-1), axis=-1)
    truePositives = K.sum(truePositives)

    #to get images that don't have mask in both true and pred
    trueNegatives = (1-true1) * (1 - pred1) # = 1 -true1 - pred1 + true1*pred1
    trueNegatives = K.sum(trueNegatives)

    return (truePositives + trueNegatives) / castF(K.shape(true)[0])


def binary_iou(true, pred):
    true = K.batch_flatten(true)
    pred = K.batch_flatten(pred)
    # pred = castF(K.greater(pred, 0.5))
    print(true)
    print(pred)

    return


if __name__ == '__main__':
    # Setup TF configs
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)
    set_session(sess)

    # test the iou output
    true = [[0, 0, 0, 0],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [0, 0, 0, 0]
            ]

    true = np.asarray(true, dtype=np.float32)
    # change to [batch, im_height, im_width, channel]
    true = np.expand_dims(true, axis=0)
    true = np.expand_dims(true, axis=3)
    print(true.shape)

    pred = [[0, 0, 0, 0],
            [0, 1, 1, 1],
            [1, 0, 1, 0],
            [0, 0, 0, 0]
            ]

    pred = np.asarray(pred, dtype=np.float32)
    pred = np.expand_dims(pred, axis=0)
    pred = np.expand_dims(pred, axis=3)
    print(pred.shape)

    print(sess.run(iou_score(pred, true)))