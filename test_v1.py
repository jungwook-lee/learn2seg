""" Unet training code from https://github.com/zhixuhao/unet """

from learn2seg.model.unet import *
from learn2seg.data.feeder import *

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
                                    # (nothing gets printed in Jupyter, only if you run it standalone)
sess = tf.Session(config=config)
set_session(sess)

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# data_gen_args = dict(rotation_range=0.2,
#                      width_shift_range=0.05,
#                      height_shift_range=0.05,
#                      shear_range=0.05,
#                      zoom_range=0.05,
#                      horizontal_flip=True,
#                      fill_mode='nearest')
# 
# # Need to use size (1280, 2440, 3)
# myGene = trainGenerator(1, '/home/ubuntu/data/teeth/pan-teeth/train',
#                         'image',
#                         'label',
#                         data_gen_args,
#                         target_size=(1280, 2448),
#                         save_to_dir=None)
# 

model = unet()
# model_checkpoint = ModelCheckpoint('unet_membrane.hdf5',
#                                    monitor='loss',
#                                    verbose=1, save_best_only=True)
model.load_weights('unet_membrane.hdf5')
#model.fit_generator(myGene,
#                    steps_per_epoch=10,
#                    epochs=1, callbacks=[model_checkpoint])

testGene = testGenerator("dataset/data/teeth_one/pan-teeth/train/image",
                         num_image=100)
results = model.predict_generator(testGene, 100, verbose=1)
saveResult("out/teeth/test", results)
