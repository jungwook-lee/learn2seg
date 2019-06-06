""" Unet training code from https://github.com/zhixuhao/unet """
import matplotlib.pyplot as plt
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import ModelCheckpoint

from learn2seg.model import *
from learn2seg.feeder import *

# configurations 
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
                                    # (nothing gets printed in Jupyter, only if you run it standalone)
sess = tf.Session(config=config)
set_session(sess)

data_gen_args = dict( \
#                     rotation_range=0.2,
#                     width_shift_range=0.05,
#                     height_shift_range=0.05,
#                     shear_range=0.05,
#                     zoom_range=0.05,
#                     horizontal_flip=True,
#                     vertical_flip=True,
#                     fill_mode='nearest'
                     )
# (352, 496)
myGene = trainGenerator(1, 'dataset/seg_teeth_warp/train/',
                        'image',
                        'label',
                        data_gen_args,
                        num_class=2,
                        flag_multi_class=False,
                        target_size=(496, 352),
                        save_to_dir=None)

valGen = trainGenerator(1, 'dataset/seg_teeth_warp/val/',
                      'image',
                      'label',
                      dict(),
                      num_class=2,
                      flag_multi_class=False,
                      target_size=(496, 352),
                      save_to_dir=None)

model = unet(weight_div=2, double_layer=False, 
             input_size=(496, 352, 1), lr=1e-5)

filepath="weights/checkpoint.hdf5"
model_checkpoint = ModelCheckpoint(filepath,
                                   monitor='val_acc',
                                   verbose=False, 
                                   save_best_only=True)

# train samples - 7821, val - 1345, test - 398
# unit samples - 98, val - 15
# warp unit (92/12/1)/(3839/719)
history = model.fit_generator(myGene,
                              validation_data=valGen,
                              validation_steps=719,
                              steps_per_epoch=3839,
                              epochs=100, callbacks=[model_checkpoint])

# plot the metrics here
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model_loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('out/loss.png')
plt.clf()

# plot accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('out/acc.png')
plt.clf()
