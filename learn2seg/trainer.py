""" Unet training code from https://github.com/zhixuhao/unet """
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import ModelCheckpoint

import matplotlib.pyplot as plt

from learn2seg.model import *
from learn2seg.feeder import *


def train_model(dataset, train_config):

    # Setup TF configs
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)
    set_session(sess)

    # Add option for data_aug
    data_gen_args = dict()
    if train_config['apply_aug']:
        data_gen_args = dict(
                            rotation_range=0.2,
                            width_shift_range=0.05,
                            height_shift_range=0.05,
                            shear_range=0.05,
                            zoom_range=0.05,
                            horizontal_flip=True,
                            vertical_flip=True,
                            fill_mode='nearest'
                            )

    train_gen = trainGenerator(train_config['batch_size'],
                               dataset.split_path['train'],
                               'image', 'label', data_gen_args,
                               target_size=dataset.im_size)

    val_gen = trainGenerator(train_config['batch_size'],
                             dataset.split_path['val'],
                             'image', 'label', dict(),
                             target_size=dataset.im_size)

    # Extend the dimension to (width, height, channel)
    input_size = copy.deepcopy(dataset.im_size)
    input_size.append(1)
    learn_rate = float(train_config['learning_rate'])

    model = unet(weight_div=train_config['weight_div'],
                 double_layer=train_config['double_layer'],
                 input_size=input_size,
                 lr=learn_rate)

    file_path = os.path.join(train_config['out_path'], "checkpoint.hdf5")
    model_checkpoint = ModelCheckpoint(file_path,
                                       monitor='val_acc',
                                       verbose=False,
                                       save_best_only=True)

    # train samples - 7821, val - 1345, test - 398
    # unit samples - 98, val - 15
    # warp unit (92/12/1)/(3839/719)
    history = model.fit_generator(train_gen,
                                  validation_data=val_gen,
                                  validation_steps=train_config['val_steps'],
                                  steps_per_epoch=train_config['train_steps'],
                                  epochs=train_config['train_epoch'],
                                  callbacks=[model_checkpoint])
    return history


def plot_model(history, train_config):
    plot_dir = os.path.join(train_config['out_path'], 'plot')
    os.mkdir(plot_dir)

    # plot loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model_loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    out_path = os.path.join(plot_dir, 'loss.png')
    plt.savefig(out_path)
    plt.clf()

    # plot accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    out_path = os.path.join(plot_dir, 'acc.png')
    plt.savefig(out_path)
    plt.clf()

    # plot iou
    plt.plot(history.history['iou_score'])
    plt.plot(history.history['val_iou_score'])
    plt.title('iou')
    plt.ylabel('iou')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    out_path = os.path.join(plot_dir, 'iou.png')
    plt.savefig(out_path)
    plt.clf()
