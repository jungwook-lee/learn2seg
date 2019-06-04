""" Unet training code from https://github.com/zhixuhao/unet """

from learn2seg.model.unet import *
from learn2seg.data.feeder import *

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

# configurations 
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
                                    # (nothing gets printed in Jupyter, only if you run it standalone)
sess = tf.Session(config=config)
set_session(sess)

data_gen_args = dict(rotation_range=0.2,
                     width_shift_range=0.05,
                     height_shift_range=0.05,
                     shear_range=0.05,
                     zoom_range=0.05,
                     horizontal_flip=True,
                     fill_mode='nearest')

# Need to use size (1280, 2440, 3)
myGene = trainGenerator(1, 'dataset/teeth/pan-teeth/train',
                        'image',
                        'label',
                        data_gen_args,
                        num_class=2,
                        flag_multi_class=False,
                        target_size=(1280, 2448),
                        save_to_dir=None)

model = unet()
model_checkpoint = ModelCheckpoint('unet_membrane.hdf5',
                                   monitor='loss',
                                   verbose=0, save_best_only=True)
model.fit_generator(myGene,
                    steps_per_epoch=800,
                    epochs=1, callbacks=[model_checkpoint])

#testGene = testGenerator("/home/jung/data/teeth/pan-teeth/test/image")
#results = model.predict_generator(testGene, 1, verbose=1)
#saveResult("out/teeth/test", results)
