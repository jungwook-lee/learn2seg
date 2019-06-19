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

    #getting iou and threshold comparisons
    iou = iou_loss_core(testTrue, testPred)
    truePositives = [castF(K.greater(iou, tres)) for tres in tresholds]

    #mean of thressholds for true positives and total sum
    truePositives = K.mean(K.stack(truePositives, axis=-1), axis=-1)
    truePositives = K.sum(truePositives)

    #to get images that don't have mask in both true and pred
    trueNegatives = (1 - true1) * (1 - pred1) # = 1 -true1 - pred1 + true1*pred1
    trueNegatives = K.sum(trueNegatives)

    return (truePositives + trueNegatives) / castF(K.shape(true)[0])


# https://stackoverflow.com/questions/48455338/tensorflow-mean-iou-how-to-get-consistent-results
def bin_iou(y_true, y_pred):
    yt0 = y_true[:, :, :, 0]
    yp0 = K.cast(y_pred[:, :, :, 0] > 0.5, 'float32')

    inter = tf.count_nonzero(tf.logical_and(tf.equal(yt0, 1), tf.equal(yp0, 1)))
    union = tf.count_nonzero(tf.add(yt0, yp0))
    iou = tf.where(tf.equal(union, 0), 1., tf.cast(inter/union, 'float32'))
    return iou


def np_bin_iou(gt_im, out_im):
    """ Numpy implementation of binary IoU

    Args:
        gt_im (np.ndarray): 2D array image with 0 - 1 intensity value
        out_im (np.ndarray): 2D array image with 0 - 1 intensity value

    Returns:

    """
    # Threshold and reshape
    gt = (gt_im > 0.5)
    out = (out_im > 0.5)

    # Calculate Intersection over Union
    intersec = np.sum(np.bitwise_and(gt, out))
    union = np.sum(np.bitwise_or(gt, out))
    iou = intersec / union
    return iou