""" Unet training code from https://github.com/zhixuhao/unet """

import copy
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

from learn2seg.model import unet
from learn2seg.feeder import *


def eval(dataset, train_config):
    config = tf.ConfigProto()
    # dynamically grow the memory used on the GPU
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    set_session(sess)

    input_size = copy.deepcopy(dataset.im_size)
    input_size.append(1)
    learn_rate = float(train_config['learning_rate'])

    model = unet(weight_div=train_config['weight_div'],
                 double_layer=train_config['double_layer'],
                 input_size=input_size,
                 lr=learn_rate
                 )

    weight_path = os.path.join(train_config['out_path'], "checkpoint.hdf5")
    model.load_weights(weight_path)

    # Generate results for all three splits [train/val/test]

    # Validation Split
    data_gene = valGenerator(dataset.split_path['val'], 'image',
                             target_size=dataset.im_size)

    results = model.predict_generator(data_gene,
                                      steps=train_config['val_steps'],
                                      verbose=1
                                      )

    eval_path = os.path.join(train_config['out_path'], 'eval', 'val')
    os.mkdir(eval_path)
    saveResult(eval_path, results)
