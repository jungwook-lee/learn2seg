""" Unet training code from https://github.com/zhixuhao/unet """

from learn2seg.unet2 import *
from learn2seg.feeder import *

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = False  # to log device placement (on which device the operation ran)
                                    # (nothing gets printed in Jupyter, only if you run it standalone)

sess = tf.Session(config=config)
set_session(sess)

model = unet(weight_div=2, double_layer=False, input_size=(496, 352, 1))
model.load_weights('weights/checkpoint.hdf5')

# 7821/1345/398, 92/12
valGene = valGenerator('dataset/seg_teeth_unit_warp/train/', 'image', target_size=(496, 352))
results = model.predict_generator(valGene, steps=92, verbose=1)
saveResult("out/teeth_warp/train", results)
