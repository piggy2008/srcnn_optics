from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.layers import Input
from keras.models import Model
import os

def srcnn(input_shape=None, kernel_size=[3, 3]):

    img_input = Input(shape=input_shape)

    x = Convolution2D(64, kernel_size[0], kernel_size[1], activation='relu', border_mode='same', name='block1_conv1')(img_input)
    x = Convolution2D(32, kernel_size[0], kernel_size[1], activation='relu', border_mode='same', name='block1_conv2')(x)
    x = Convolution2D(1, 1, 1, border_mode='same')(x)
    model = Model(img_input, x)

    # weights_path = os.path.expanduser(os.path.join('~', '.keras/models/fcn_vgg16_weights_tf_dim_ordering_tf_kernels.h5'))
    # model.load_weights(weights_path, by_name=True)
    return model