""" Unet model code from https://github.com/zhixuhao/unet """

import numpy as np
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras

import learn2seg.metric.iou as iou

def unet(pretrained_weights=None, input_size=(512, 512, 1), weight_div=1, double_layer=False, lr=1e-4):

    inputs = Input(input_size)

    convs = int(64/weight_div)
    conv1 = Conv2D(convs, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    if double_layer:
        conv1 = Conv2D(convs, 3, activation='relu', padding='same', 
                       kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    convs = int(128/weight_div)
    conv2 = Conv2D(convs, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    if double_layer:
        conv2 = Conv2D(convs, 3, activation='relu', padding='same', 
                       kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    convs = int(256/weight_div)
    conv3 = Conv2D(convs, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    if double_layer:
        conv3 = Conv2D(convs, 3, activation='relu', padding='same', 
                       kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    convs = int(512/weight_div)
    conv4 = Conv2D(convs, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    if double_layer:
        conv4 = Conv2D(convs, 3, activation='relu', padding='same', 
                       kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    
    convs = int(1024/weight_div)
    conv5 = Conv2D(convs, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    if double_layer:
        conv5 = Conv2D(convs, 3, activation='relu', padding='same', 
                       kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    convs = int(512/weight_div)
    up6 = Conv2D(convs, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(convs, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    if double_layer:
        conv6 = Conv2D(convs, 3, activation='relu', padding='same', 
                       kernel_initializer='he_normal')(conv6)

    convs = int(256/weight_div)
    up7 = Conv2D(convs, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(convs, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    if double_layer:
        conv7 = Conv2D(convs, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal')(conv7)
    
    convs = int(128/weight_div)
    up8 = Conv2D(convs, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(convs, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    if double_layer:
        conv8 = Conv2D(convs, 3, activation='relu', padding='same', 
                       kernel_initializer='he_normal')(conv8)

    convs = int(64/weight_div)
    up9 = Conv2D(convs, 2, activation='relu', padding='same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(convs, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    if double_layer:   
        conv9 = Conv2D(convs, 3, activation='relu', padding='same', 
                       kernel_initializer='he_normal')(conv9)

    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)
    #activation = Activation('softmax')(conv9)

    model = Model(inputs=inputs, outputs=conv10)

    model.compile(optimizer=Adam(lr=1e-4),
                  loss='binary_crossentropy',
                  #loss=jaccard_distance_loss,
                  metrics=['acc', iou.iou_score]) # binary_accuracy
    
    #model.summary()

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model


def jaccard_distance_loss(y_true, y_pred, smooth=100):
    """
    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))
    
    The jaccard distance loss is usefull for unbalanced datasets. This has been
    shifted so it converges on 0 and is smoothed to avoid exploding or disapearing
    gradient.
    
    Ref: https://en.wikipedia.org/wiki/Jaccard_index
    
    @url: https://gist.github.com/wassname/f1452b748efcbeb4cb9b1d059dce6f96
    @author: wassname
    """
    intersection = keras.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = keras.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth
