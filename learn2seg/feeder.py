""" Unet model code from https://github.com/zhixuhao/unet """

from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os

import skimage.io as io
import skimage.transform as trans


def adjustData(img, mask):
    img = img / 255
    mask = mask / 255
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0
    return img, mask


def trainGenerator(batch_size,
                   train_path,
                   image_folder,
                   mask_folder,
                   aug_dict,
                   image_color_mode="grayscale",
                   mask_color_mode="grayscale",
                   image_save_prefix="image",
                   mask_save_prefix="mask",
                   target_size=(512, 512),
                   seed=1):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the
    transformation for image and mask is the same
    if you want to visualize the results of generator,
    set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes=[image_folder],
        class_mode=None,
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_prefix=image_save_prefix,
        seed=seed)

    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes=[mask_folder],
        class_mode=None,
        color_mode=mask_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_prefix=mask_save_prefix,
        seed=seed)

    train_generator = zip(image_generator, mask_generator)
    for (img, mask) in train_generator:
        img, mask = adjustData(img, mask)
        yield (img, mask)


def valGenerator(val_path, image_folder, target_size):
   # aug_dict for validation is not needed
    aug_dict = dict()
    val_datagen = ImageDataGenerator(**aug_dict)
    image_generator = val_datagen.flow_from_directory(
        val_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = "grayscale",
        target_size = target_size,
        batch_size = 1,
        save_to_dir = False,
        seed=1)

    gen = image_generator
    for img in gen:
        img = img / 255
        img = img
        yield img

    # for i in range(num_image):
    #    img = io.imread(os.path.join(test_path,"%d.jpeg"%i),as_gray = as_gray)
    #    img = img / 255
    #    img = trans.resize(img,target_size)
    #    img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
    #    img = np.reshape(img,(1,)+img.shape)
    #    yield img


def testGenerator(test_path, num_image,target_size = (512, 512),flag_multi_class = False,as_gray = True):
    for i in range(num_image):
        img = io.imread(os.path.join(test_path,"%d.jpeg"%i),as_gray = as_gray)
        img = img / 255
        img = trans.resize(img,target_size)
        img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
        img = np.reshape(img,(1,)+img.shape)
        yield img


def testGenerator_png(test_path, num_image,target_size = (512, 512),flag_multi_class = False,as_gray = True):
    for i in range(num_image):
        img = io.imread(os.path.join(test_path,"%d.png"%i),as_grey = as_gray)
        img = img / 255
        img = np.reshape(img, (1,)+img.shape+(1,))
        #img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
        yield img


def saveResult(save_path, npyfile, flag_multi_class = False, num_class = 2):
    for i, item in enumerate(npyfile):
        img = item[:,:,0]
        #img = ((img*10.0))
        #factor = 1.0/np.max(img)
        #img = img*factor
        #img[:, :] = (img[:, :] >) * 255
        img = (img > 0.5) * 255
        io.imsave(os.path.join(save_path, "%d.png" % i), img.astype(np.uint8))


def labelVisualize(num_class,color_dict,img):
    img = img[:,:,0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    for i in range(num_class):
        img_out[img == i,:] = color_dict[i]
    return img_out / 255
