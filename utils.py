from keras.preprocessing.image import *
from PIL import Image
from scipy.misc import imresize
import numpy as np

def crop_mnist_image(image56, input_shape):
    imgs = []

    image56 = img_to_array(image56, data_format='channels_last')
    image56 = image56.astype(dtype=float) / 255
    imgs1 = image56[:28, :28, 0]
    imgs2 = image56[28:, :28, 0]
    imgs3 = image56[:28, 28:, 0]
    imgs4 = image56[28:, 28:, 0]

    imgs1 = imresize(imgs1, (input_shape[0], input_shape[1]), mode='F')
    imgs2 = imresize(imgs2, (input_shape[0], input_shape[1]), mode='F')
    imgs3 = imresize(imgs3, (input_shape[0], input_shape[1]), mode='F')
    imgs4 = imresize(imgs4, (input_shape[0], input_shape[1]), mode='F')

    imgs.append(imgs1[:, :, np.newaxis])
    imgs.append(imgs2[:, :, np.newaxis])
    imgs.append(imgs3[:, :, np.newaxis])
    imgs.append(imgs4[:, :, np.newaxis])
    return imgs

def combine_mnist_image(results, shape=[28, 28]):
    combine_image = np.zeros([shape[0]*2, shape[1]*2])
    combine_image[:shape[0], :shape[1]] = results[0]
    combine_image[shape[0]:, :shape[1]] = results[1]
    combine_image[:shape[0], shape[1]:] = results[2]
    combine_image[shape[0]:, shape[1]:] = results[3]

    return combine_image