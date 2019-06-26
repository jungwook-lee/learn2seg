""" Unet training code from https://github.com/zhixuhao/unet """
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard

import matplotlib.pyplot as plt

from learn2seg.model import *
from learn2seg.feeder import *


def train_model(dataset, model_config, train_config, out_path, train_it):

    # Setup TF configs
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)
    set_session(sess)

    # Extract parameters
    train_epoch = train_config['train_epoch']
    train_steps = train_config['train_steps']
    val_steps = train_config['val_steps']
    batch_size = train_config['batch_size']

    learning_rate = model_config['learning_rate']
    weight_div = model_config['weight_div']
    double_layer = model_config['double_layer']

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

    # for iterative training change the output gt path
    train_path = dataset.split_path['train']
    val_path = dataset.split_path['val']
    if train_it == 0:
        train_mask_path = train_path
        val_mask_path = val_path
    else:
        # Iteration get output path
        it_str = 'eval_{}'.format(train_it-1)
        mask_path = os.path.join(out_path, it_str)
        train_mask_path = os.path.join(mask_path, 'train')
        val_mask_path = os.path.join(mask_path, 'val')

        # Check if path exists
        if not os.path.exists(mask_path):
            raise ValueError('Next iteration mask path does not exist!')

    train_gen = trainGenerator(batch_size=batch_size,
                               train_path=train_path,
                               mask_path=train_mask_path,
                               image_folder='image',
                               mask_folder='label',
                               aug_dict=data_gen_args,
                               target_size=dataset.im_size,
                               shuffle=True,
                               )

    val_gen = trainGenerator(batch_size=1,
                             train_path=val_path,
                             mask_path=val_mask_path,
                             image_folder='image',
                             mask_folder='label',
                             aug_dict=dict(),
                             target_size=dataset.im_size,
                             shuffle=True,
                             )

    # Extend the dimension to (width, height, channel)
    input_size = copy.deepcopy(dataset.im_size)
    input_size.append(1)
    learn_rate = float(learning_rate)

    model = unet(weight_div=weight_div,
                 double_layer=double_layer,
                 input_size=input_size,
                 lr=learn_rate)

    if train_it > 0:
        # Initialize with previously trained weights if not first iteration
        file_str = "checkpoint_{}.hdf5".format(train_it - 1)
        file_path = os.path.join(out_path, file_str)
        weight_path = os.path.join(file_path)
        model.load_weights(weight_path)

    file_str = "checkpoint_{}.hdf5".format(train_it)
    file_path = os.path.join(out_path, file_str)
    model_checkpoint = ModelCheckpoint(file_path,
                                       monitor='val_bin_iou',
                                       verbose=True,
                                       mode='max',
                                       save_best_only=True)

    # train samples - 7821, val - 1345, test - 398
    # unit samples - 98, val - 15
    # warp unit (92/12/1)/(3839/719)

    # Setup tensorboard callbacks
    log_str = "tf_log_" + str(train_it)
    logdir = os.path.join(out_path, log_str)
    tensorboard_callback = TensorBoard(log_dir=logdir)

    history = model.fit_generator(train_gen,
                                  validation_data=val_gen,
                                  validation_steps=val_steps,
                                  steps_per_epoch=train_steps,
                                  epochs=train_epoch,
                                  callbacks=[model_checkpoint,
                                             tensorboard_callback])
    return history


def plot_model(history, out_path, train_it):
    plot_dir = os.path.join(out_path, 'plot')
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)

    # plot loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model_loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    out_path = os.path.join(plot_dir, 'loss_{}.png'.format(train_it))
    plt.savefig(out_path)
    plt.clf()

    # plot accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    out_path = os.path.join(plot_dir, 'acc_{}.png'.format(train_it))
    plt.savefig(out_path)
    plt.clf()

    # plot iou
    plt.plot(history.history['bin_iou'])
    plt.plot(history.history['val_bin_iou'])
    plt.title('binary_iou')
    plt.ylabel('iou')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    out_path = os.path.join(plot_dir, 'binary_iou_{}.png'.format(train_it))
    plt.savefig(out_path)
    plt.clf()
