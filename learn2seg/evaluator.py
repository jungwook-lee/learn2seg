""" Unet training code from https://github.com/zhixuhao/unet """

import copy
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

from learn2seg.model import unet
from learn2seg.feeder import *


def eval(dataset, train_config, iteration=0):
    out_path = train_config['out_path']
    eval_path = os.path.join(out_path, 'eval_%d' % iteration)
    if os.path.exists(eval_path):
        raise ValueError("eval_path already exists!")
    else:
        os.mkdir(eval_path)

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
    splits = ['train', 'val', 'test']
    for split in splits:
        data_gene = valGenerator(dataset.split_path[split], 'image',
                                 target_size=dataset.im_size)

        steps_key = split + '_steps'
        results = model.predict_generator(data_gene,
                                          steps=train_config[steps_key],
                                          verbose=1
                                          )

        eval_path = os.path.join(train_config['out_path'],
                                 'eval_%d' % iteration,
                                 split)
        os.mkdir(eval_path)
        saveResult(eval_path, results)