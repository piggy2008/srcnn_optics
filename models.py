from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, BatchNormalization, UpSampling2D, merge
from keras.utils import np_utils
from keras import backend as K
from keras.layers import Input
from keras.models import Model
from keras.layers.core import Reshape
import os
from resnet_helpers import *
from BilinearUpSampling import *

def srcnn(input_shape=None, kernel_size=[3, 3]):

    img_input = Input(shape=input_shape)

    x = Convolution2D(64, kernel_size[0], kernel_size[1], border_mode='same', name='block1_conv1')(img_input)
    # x = BatchNormalization(axis=3, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = Convolution2D(32, kernel_size[0], kernel_size[1], activation='relu', border_mode='same', name='block1_conv2')(x)
    # x = BatchNormalization(axis=3, name='bn_conv2')(x)
    x = Convolution2D(1, 1, 1, border_mode='same')(x)
    x = Activation('relu')(x)
    model = Model(img_input, x)

    # weights_path = os.path.expanduser(os.path.join('~', '.keras/models/fcn_vgg16_weights_tf_dim_ordering_tf_kernels.h5'))
    # model.load_weights(weights_path, by_name=True)
    return model

def srcnn_fc(input_shape=None, kernel_size=[3, 3]):

    img_input = Input(shape=input_shape)

    x = Convolution2D(64, kernel_size[0], kernel_size[1], border_mode='same', name='block1_conv1')(img_input)
    # x = BatchNormalization(axis=3, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = Convolution2D(32, kernel_size[0], kernel_size[1], activation='relu', border_mode='same', name='block1_conv2')(x)
    # x = BatchNormalization(axis=3, name='bn_conv2')(x)
    x = Flatten()(x)
    x = Dense(48*48)(x)
    x = Activation('relu')(x)
    x = Reshape((48, 48, 1))(x)
    # x = Convolution2D(1, 1, 1, border_mode='same')(x)
    model = Model(img_input, x)

    # weights_path = os.path.expanduser(os.path.join('~', '.keras/models/fcn_vgg16_weights_tf_dim_ordering_tf_kernels.h5'))
    # model.load_weights(weights_path, by_name=True)
    return model

def cgi(input_shape=None):

    img_input = Input(shape=input_shape)

    x = Convolution2D(32, 9, 9, border_mode='same', activation='relu')(img_input)
    x = Convolution2D(32, 9, 9, activation='relu', border_mode='same')(x)
    x = Convolution2D(32, 9, 9, activation='relu', border_mode='same')(x)
    shortcut1 = Convolution2D(32, 1, 1, activation='relu', border_mode='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Convolution2D(64, 7, 7, border_mode='same', activation='relu')(x)
    x = Convolution2D(64, 7, 7, activation='relu', border_mode='same')(x)
    x = Convolution2D(64, 7, 7, activation='relu', border_mode='same')(x)
    shortcut2 = Convolution2D(64, 1, 1, border_mode='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Convolution2D(128, 5, 5, border_mode='same', activation='relu')(x)
    x = Convolution2D(128, 5, 5, activation='relu', border_mode='same')(x)
    x = Convolution2D(128, 5, 5, activation='relu', border_mode='same')(x)
    shortcut3 = Convolution2D(128, 1, 1, border_mode='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Convolution2D(256, 5, 5, border_mode='same', activation='relu')(x)
    x = Convolution2D(256, 5, 5, activation='relu', border_mode='same')(x)
    x = Convolution2D(256, 5, 5, activation='relu', border_mode='same')(x)
    shortcut4 = Convolution2D(256, 1, 1, border_mode='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    # x = BatchNormalization(axis=3, name='bn_conv2')(x)

    x = Convolution2D(512, 5, 5, border_mode='same', activation='relu')(x)
    x = Convolution2D(512, 5, 5, activation='relu', border_mode='same')(x)
    x = Convolution2D(512, 5, 5, activation='relu', border_mode='same')(x)

    x = Convolution2D(256, 1, 1, activation='relu', border_mode='same')(x)
    x = UpSampling2D(size=(2, 2))(x)

    x = merge([x, shortcut4], mode='sum')
    x = Convolution2D(256, 5, 5, border_mode='same', activation='relu')(x)
    x = Convolution2D(256, 5, 5, activation='relu', border_mode='same')(x)
    x = Convolution2D(256, 5, 5, activation='relu', border_mode='same')(x)

    x = Convolution2D(128, 1, 1, activation='relu', border_mode='same')(x)
    x = UpSampling2D(size=(2, 2))(x)
    x = merge([x, shortcut3], mode='sum')
    x = Convolution2D(128, 5, 5, border_mode='same', activation='relu')(x)
    x = Convolution2D(128, 5, 5, activation='relu', border_mode='same')(x)
    x = Convolution2D(128, 5, 5, activation='relu', border_mode='same')(x)

    x = Convolution2D(64, 1, 1, activation='relu', border_mode='same')(x)
    x = UpSampling2D(size=(2, 2))(x)
    x = merge([x, shortcut2], mode='sum')
    x = Convolution2D(64, 7, 7, border_mode='same', activation='relu')(x)
    x = Convolution2D(64, 7, 7, activation='relu', border_mode='same')(x)
    x = Convolution2D(64, 7, 7, activation='relu', border_mode='same')(x)

    x = Convolution2D(32, 1, 1, activation='relu', border_mode='same')(x)
    x = UpSampling2D(size=(2, 2))(x)
    x = merge([x, shortcut1], mode='sum')
    x = Convolution2D(32, 9, 9, border_mode='same', activation='relu')(x)
    x = Convolution2D(32, 9, 9, activation='relu', border_mode='same')(x)
    x = Convolution2D(32, 9, 9, activation='relu', border_mode='same')(x)

    x = Convolution2D(1, 1, 1, border_mode='same', activation='relu')(x)
    model = Model(img_input, x)

    # weights_path = os.path.expanduser(os.path.join('~', '.keras/models/fcn_vgg16_weights_tf_dim_ordering_tf_kernels.h5'))
    # model.load_weights(weights_path, by_name=True)
    return model

def unet(input_shape=None):

    img_input = Input(shape=input_shape)

    x = Convolution2D(32, 9, 9, border_mode='same', activation='relu')(img_input)
    x = Convolution2D(32, 9, 9, activation='relu', border_mode='same')(x)
    x = Convolution2D(32, 9, 9, activation='relu', border_mode='same')(x)
    shortcut1 = MaxPooling2D(pool_size=(2, 2))(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Convolution2D(64, 7, 7, border_mode='same', activation='relu')(x)
    x = Convolution2D(64, 7, 7, activation='relu', border_mode='same')(x)
    x = Convolution2D(64, 7, 7, activation='relu', border_mode='same')(x)
    shortcut2 = MaxPooling2D(pool_size=(2, 2))(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Convolution2D(128, 5, 5, border_mode='same', activation='relu')(x)
    x = Convolution2D(128, 5, 5, activation='relu', border_mode='same')(x)
    x = Convolution2D(128, 5, 5, activation='relu', border_mode='same')(x)
    shortcut3 = MaxPooling2D(pool_size=(2, 2))(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Convolution2D(256, 5, 5, border_mode='same', activation='relu')(x)
    x = Convolution2D(256, 5, 5, activation='relu', border_mode='same')(x)
    x = Convolution2D(256, 5, 5, activation='relu', border_mode='same')(x)
    shortcut4 = MaxPooling2D(pool_size=(2, 2))(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    # x = BatchNormalization(axis=3, name='bn_conv2')(x)

    x = Convolution2D(512, 5, 5, border_mode='same', activation='relu')(x)
    x = Convolution2D(512, 5, 5, activation='relu', border_mode='same')(x)
    x = Convolution2D(512, 5, 5, activation='relu', border_mode='same')(x)

    x = Convolution2D(256, 1, 1, activation='relu', border_mode='same')(x)
    x = merge([x, shortcut4], mode='concat')
    x = UpSampling2D(size=(2, 2))(x)
    x = Convolution2D(512, 5, 5, border_mode='same', activation='relu')(x)
    x = Convolution2D(512, 5, 5, activation='relu', border_mode='same')(x)
    x = Convolution2D(512, 5, 5, activation='relu', border_mode='same')(x)

    x = Convolution2D(128, 1, 1, activation='relu', border_mode='same')(x)
    x = merge([x, shortcut3], mode='concat')
    x = UpSampling2D(size=(2, 2))(x)
    x = Convolution2D(256, 5, 5, border_mode='same', activation='relu')(x)
    x = Convolution2D(256, 5, 5, activation='relu', border_mode='same')(x)
    x = Convolution2D(256, 5, 5, activation='relu', border_mode='same')(x)

    x = Convolution2D(64, 1, 1, activation='relu', border_mode='same')(x)
    x = merge([x, shortcut2], mode='concat')
    x = UpSampling2D(size=(2, 2))(x)
    x = Convolution2D(128, 7, 7, border_mode='same', activation='relu')(x)
    x = Convolution2D(128, 7, 7, activation='relu', border_mode='same')(x)
    x = Convolution2D(128, 7, 7, activation='relu', border_mode='same')(x)

    x = Convolution2D(32, 1, 1, activation='relu', border_mode='same')(x)
    x = merge([x, shortcut1], mode='concat')
    x = UpSampling2D(size=(2, 2))(x)
    x = Convolution2D(64, 9, 9, border_mode='same', activation='relu')(x)
    x = Convolution2D(64, 9, 9, activation='relu', border_mode='same')(x)
    x = Convolution2D(64, 9, 9, activation='relu', border_mode='same')(x)

    x = Convolution2D(1, 1, 1, border_mode='same')(x)
    model = Model(img_input, x)

    # weights_path = os.path.expanduser(os.path.join('~', '.keras/models/fcn_vgg16_weights_tf_dim_ordering_tf_kernels.h5'))
    # model.load_weights(weights_path, by_name=True)
    return model

def unet_limit(input_shape=None):

    img_input = Input(shape=input_shape)

    x = Convolution2D(32, 9, 9, border_mode='same', activation='relu')(img_input)
    x = Convolution2D(32, 9, 9, activation='relu', border_mode='same')(x)
    x = Convolution2D(32, 9, 9, activation='relu', border_mode='same')(x)
    shortcut1 = MaxPooling2D(pool_size=(2, 2))(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Convolution2D(64, 7, 7, border_mode='same', activation='relu')(x)
    x = Convolution2D(64, 7, 7, activation='relu', border_mode='same')(x)
    x = Convolution2D(64, 7, 7, activation='relu', border_mode='same')(x)
    shortcut2 = MaxPooling2D(pool_size=(2, 2))(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Convolution2D(128, 5, 5, border_mode='same', activation='relu')(x)
    x = Convolution2D(128, 5, 5, activation='relu', border_mode='same')(x)
    x = Convolution2D(128, 5, 5, activation='relu', border_mode='same')(x)
    shortcut3 = MaxPooling2D(pool_size=(2, 2))(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Convolution2D(256, 5, 5, border_mode='same', activation='relu')(x)
    x = Convolution2D(256, 5, 5, activation='relu', border_mode='same')(x)
    x = Convolution2D(256, 5, 5, activation='relu', border_mode='same')(x)

    x = Convolution2D(128, 1, 1, activation='relu', border_mode='same')(x)
    x = merge([x, shortcut3], mode='concat')
    x = UpSampling2D(size=(2, 2))(x)
    x = Convolution2D(256, 5, 5, border_mode='same', activation='relu')(x)
    x = Convolution2D(256, 5, 5, activation='relu', border_mode='same')(x)
    x = Convolution2D(256, 5, 5, activation='relu', border_mode='same')(x)

    x = Convolution2D(64, 1, 1, activation='relu', border_mode='same')(x)
    x = merge([x, shortcut2], mode='concat')
    x = UpSampling2D(size=(2, 2))(x)
    x = Convolution2D(128, 7, 7, border_mode='same', activation='relu')(x)
    x = Convolution2D(128, 7, 7, activation='relu', border_mode='same')(x)
    x = Convolution2D(128, 7, 7, activation='relu', border_mode='same')(x)

    x = Convolution2D(32, 1, 1, activation='relu', border_mode='same')(x)
    x = merge([x, shortcut1], mode='concat')
    x = UpSampling2D(size=(2, 2))(x)
    x = Convolution2D(64, 9, 9, border_mode='same', activation='relu')(x)
    x = Convolution2D(64, 9, 9, activation='relu', border_mode='same')(x)
    x = Convolution2D(64, 9, 9, activation='relu', border_mode='same')(x)

    x = Convolution2D(1, 1, 1, border_mode='same', activation='relu')(x)
    model = Model(img_input, x)

    # weights_path = os.path.expanduser(os.path.join('~', '.keras/models/fcn_vgg16_weights_tf_dim_ordering_tf_kernels.h5'))
    # model.load_weights(weights_path, by_name=True)
    return model

def unet_limit_dialate(input_shape=None):

    img_input = Input(shape=input_shape)

    x = Convolution2D(32, 9, 9, border_mode='same', activation='relu')(img_input)
    x = Convolution2D(32, 9, 9, activation='relu', border_mode='same')(x)
    x = Convolution2D(32, 9, 9, activation='relu', border_mode='same')(x)
    shortcut1 = MaxPooling2D(pool_size=(2, 2))(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Convolution2D(64, 7, 7, border_mode='same', activation='relu')(x)
    x = Convolution2D(64, 7, 7, activation='relu', border_mode='same')(x)
    x = Convolution2D(64, 7, 7, activation='relu', border_mode='same')(x)
    shortcut2 = MaxPooling2D(pool_size=(2, 2))(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Convolution2D(128, 5, 5, border_mode='same', activation='relu')(x)
    x = Convolution2D(128, 5, 5, activation='relu', border_mode='same')(x)
    x = Convolution2D(128, 5, 5, activation='relu', border_mode='same')(x)
    shortcut3 = Convolution2D(128, 1, 1, activation='relu', border_mode='same')(x)
    # x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Convolution2D(256, 5, 5, border_mode='same', activation='relu', dilation_rate=1)(x)
    x = Dropout(0.75)(x)
    x = Convolution2D(256, 5, 5, activation='relu', border_mode='same', dilation_rate=1)(x)
    x = Dropout(0.75)(x)
    x = Convolution2D(256, 5, 5, activation='relu', border_mode='same', dilation_rate=1)(x)
    x = Dropout(0.75)(x)

    x = Convolution2D(128, 1, 1, activation='relu', border_mode='same')(x)
    x = merge([x, shortcut3], mode='concat')
    # x = UpSampling2D(size=(2, 2))(x)
    x = Convolution2D(256, 5, 5, border_mode='same', activation='relu')(x)
    x = Convolution2D(256, 5, 5, activation='relu', border_mode='same')(x)
    x = Convolution2D(256, 5, 5, activation='relu', border_mode='same')(x)


    x = Convolution2D(64, 1, 1, activation='relu', border_mode='same')(x)
    x = merge([x, shortcut2], mode='concat')
    x = UpSampling2D(size=(2, 2))(x)
    x = Convolution2D(128, 7, 7, border_mode='same', activation='relu')(x)
    x = Convolution2D(128, 7, 7, activation='relu', border_mode='same')(x)
    x = Convolution2D(128, 7, 7, activation='relu', border_mode='same')(x)

    x = Convolution2D(32, 1, 1, activation='relu', border_mode='same')(x)
    x = merge([x, shortcut1], mode='concat')
    x = UpSampling2D(size=(2, 2))(x)
    x = Convolution2D(64, 9, 9, border_mode='same', activation='relu')(x)
    x = Convolution2D(64, 9, 9, activation='relu', border_mode='same')(x)
    x = Convolution2D(64, 9, 9, activation='relu', border_mode='same')(x)

    x = Convolution2D(1, 1, 1, border_mode='same', activation='relu')(x)
    model = Model(img_input, x)

    # weights_path = os.path.expanduser(os.path.join('~', '.keras/models/fcn_vgg16_weights_tf_dim_ordering_tf_kernels.h5'))
    # model.load_weights(weights_path, by_name=True)
    return model

def unet_limit_shortcut_dialate(input_shape=None):

    img_input = Input(shape=input_shape)

    y1 = Convolution2D(32, 9, 9, activation='relu', border_mode='same')(img_input)
    x = Convolution2D(32, 9, 9, border_mode='same', activation='relu')(img_input)
    x = Convolution2D(32, 9, 9, activation='relu', border_mode='same')(x)
    x = Convolution2D(32, 9, 9, activation='relu', border_mode='same')(x)
    x = merge([x, y1], mode='sum')
    shortcut1 = MaxPooling2D(pool_size=(2, 2))(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    y2 = Convolution2D(64, 7, 7, activation='relu', border_mode='same')(x)
    x = Convolution2D(64, 7, 7, border_mode='same', activation='relu')(x)
    x = Convolution2D(64, 7, 7, activation='relu', border_mode='same')(x)
    x = Convolution2D(64, 7, 7, activation='relu', border_mode='same')(x)
    x = merge([x, y2], mode='sum')
    shortcut2 = MaxPooling2D(pool_size=(2, 2))(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    y3 = Convolution2D(128, 5, 5, activation='relu', border_mode='same')(x)
    x = Convolution2D(128, 5, 5, border_mode='same', activation='relu')(x)
    x = Convolution2D(128, 5, 5, activation='relu', border_mode='same')(x)
    x = Convolution2D(128, 5, 5, activation='relu', border_mode='same')(x)
    x = merge([x, y3], mode='sum')
    shortcut3 = Convolution2D(128, 1, 1, activation='relu', border_mode='same')(x)
    # x = MaxPooling2D(pool_size=(2, 2))(x)
    y4 = Convolution2D(256, 5, 5, activation='relu', border_mode='same')(x)
    x = Convolution2D(256, 5, 5, border_mode='same', activation='relu', dilation_rate=1)(x)
    x = Convolution2D(256, 5, 5, activation='relu', border_mode='same', dilation_rate=1)(x)
    x = Convolution2D(256, 5, 5, activation='relu', border_mode='same', dilation_rate=1)(x)
    x = merge([x, y4], mode='sum')


    x = Convolution2D(128, 1, 1, activation='relu', border_mode='same')(x)
    x = merge([x, shortcut3], mode='concat')
    # x = UpSampling2D(size=(2, 2))(x)
    y5 = Convolution2D(256, 5, 5, activation='relu', border_mode='same')(x)
    x = Convolution2D(256, 5, 5, border_mode='same', activation='relu')(x)
    x = Convolution2D(256, 5, 5, activation='relu', border_mode='same')(x)
    x = Convolution2D(256, 5, 5, activation='relu', border_mode='same')(x)
    x = merge([x, y5], mode='sum')


    x = Convolution2D(64, 1, 1, activation='relu', border_mode='same')(x)
    x = merge([x, shortcut2], mode='concat')
    x = UpSampling2D(size=(2, 2))(x)

    y6 = Convolution2D(128, 7, 7, activation='relu', border_mode='same')(x)
    x = Convolution2D(128, 7, 7, border_mode='same', activation='relu')(x)
    x = Convolution2D(128, 7, 7, activation='relu', border_mode='same')(x)
    x = Convolution2D(128, 7, 7, activation='relu', border_mode='same')(x)
    x = merge([x, y6], mode='sum')

    x = Convolution2D(32, 1, 1, activation='relu', border_mode='same')(x)
    x = merge([x, shortcut1], mode='concat')
    x = UpSampling2D(size=(2, 2))(x)

    y7 = Convolution2D(64, 9, 9, activation='relu', border_mode='same')(x)
    x = Convolution2D(64, 9, 9, border_mode='same', activation='relu')(x)
    x = Convolution2D(64, 9, 9, activation='relu', border_mode='same')(x)
    x = Convolution2D(64, 9, 9, activation='relu', border_mode='same')(x)
    x = merge([x, y7], mode='sum')

    x = Convolution2D(1, 1, 1, border_mode='same', activation='relu')(x)
    model = Model(img_input, x)

    # weights_path = os.path.expanduser(os.path.join('~', '.keras/models/fcn_vgg16_weights_tf_dim_ordering_tf_kernels.h5'))
    # model.load_weights(weights_path, by_name=True)
    return model

def unet_limit_shortcut_dialate3x3(input_shape=None):

    img_input = Input(shape=input_shape)

    y1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(img_input)
    x = Convolution2D(32, 3, 3, border_mode='same', activation='relu')(img_input)
    x = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(x)
    x = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(x)
    x = merge([x, y1], mode='sum')
    shortcut1 = MaxPooling2D(pool_size=(2, 2))(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    y2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(x)
    x = Convolution2D(64, 3, 3, border_mode='same', activation='relu')(x)
    x = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(x)
    x = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(x)
    x = merge([x, y2], mode='sum')
    shortcut2 = MaxPooling2D(pool_size=(2, 2))(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    y3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(x)
    x = Convolution2D(128, 3, 3, border_mode='same', activation='relu')(x)
    x = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(x)
    x = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(x)
    x = merge([x, y3], mode='sum')
    shortcut3 = Convolution2D(128, 1, 1, activation='relu', border_mode='same')(x)
    # x = MaxPooling2D(pool_size=(2, 2))(x)
    y4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(x)
    x = Convolution2D(256, 3, 3, border_mode='same', activation='relu', dilation_rate=1)(x)
    x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', dilation_rate=1)(x)
    x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', dilation_rate=1)(x)
    x = merge([x, y4], mode='sum')


    x = Convolution2D(128, 1, 1, activation='relu', border_mode='same')(x)
    x = merge([x, shortcut3], mode='concat')
    # x = UpSampling2D(size=(2, 2))(x)
    y5 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(x)
    x = Convolution2D(256, 3, 3, border_mode='same', activation='relu')(x)
    x = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(x)
    x = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(x)
    x = merge([x, y5], mode='sum')


    x = Convolution2D(64, 1, 1, activation='relu', border_mode='same')(x)
    x = merge([x, shortcut2], mode='concat')
    x = UpSampling2D(size=(2, 2))(x)

    y6 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(x)
    x = Convolution2D(128, 3, 3, border_mode='same', activation='relu')(x)
    x = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(x)
    x = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(x)
    x = merge([x, y6], mode='sum')

    x = Convolution2D(32, 1, 1, activation='relu', border_mode='same')(x)
    x = merge([x, shortcut1], mode='concat')
    x = UpSampling2D(size=(2, 2))(x)

    y7 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(x)
    x = Convolution2D(64, 3, 3, border_mode='same', activation='relu')(x)
    x = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(x)
    x = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(x)
    x = merge([x, y7], mode='sum')

    x = Convolution2D(1, 1, 1, border_mode='same', activation='relu')(x)
    model = Model(img_input, x)

    # weights_path = os.path.expanduser(os.path.join('~', '.keras/models/fcn_vgg16_weights_tf_dim_ordering_tf_kernels.h5'))
    # model.load_weights(weights_path, by_name=True)
    return model

def unet_limit_dialate_multiscale(input_shape=None):

    img_input = Input(shape=input_shape)

    x = Convolution2D(32, 9, 9, border_mode='same', activation='relu')(img_input)
    x = Convolution2D(32, 9, 9, activation='relu', border_mode='same')(x)
    x = Convolution2D(32, 9, 9, activation='relu', border_mode='same')(x)

    scale1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(x)
    scale1 = Convolution2D(1, 1, 1, activation='relu', border_mode='same')(scale1)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Convolution2D(64, 7, 7, border_mode='same', activation='relu')(x)
    x = Convolution2D(64, 7, 7, activation='relu', border_mode='same')(x)
    x = Convolution2D(64, 7, 7, activation='relu', border_mode='same')(x)

    scale2 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(x)
    scale2 = Convolution2D(1, 1, 1, activation='relu', border_mode='same')(scale2)

    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Convolution2D(128, 5, 5, border_mode='same', activation='relu')(x)
    x = Convolution2D(128, 5, 5, activation='relu', border_mode='same')(x)
    x = Convolution2D(128, 5, 5, activation='relu', border_mode='same')(x)


    scale3 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(x)
    scale3 = Convolution2D(1, 1, 1, activation='relu', border_mode='same')(scale3)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Convolution2D(256, 5, 5, border_mode='same', activation='relu')(x)
    x = Convolution2D(256, 5, 5, activation='relu', border_mode='same')(x)
    x = Convolution2D(256, 5, 5, activation='relu', border_mode='same')(x)

    scale4 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(x)
    scale4 = Convolution2D(1, 1, 1, activation='relu', border_mode='same')(scale4)
    scale4 = UpSampling2D(size=(2, 2))(scale4)

    x = merge([scale3, scale4], mode='concat')
    x = Convolution2D(1, 1, 1, activation='relu', border_mode='same')(x)

    x = UpSampling2D(size=(2, 2))(x)
    x = merge([x, scale2], mode='concat')
    x = Convolution2D(1, 1, 1, activation='relu', border_mode='same')(x)

    x = UpSampling2D(size=(2, 2))(x)
    x = merge([x, scale1], mode='concat')
    x = Convolution2D(1, 1, 1, activation='relu', border_mode='same')(x)

    model = Model(img_input, x)
    return model

def FCN_Resnet50_32s(input_shape = None, weight_decay=0., batch_momentum=0.9, batch_shape=None, classes=21):
    if batch_shape:
        img_input = Input(batch_shape=batch_shape)
        image_size = batch_shape[1:3]
    else:
        img_input = Input(shape=input_shape)
        image_size = input_shape[0:2]

    bn_axis = 3

    x = Convolution2D(64, 7, 7, subsample=(2, 2), border_mode='same', name='conv1', W_regularizer=l2(weight_decay))(img_input)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(3, [64, 64, 256], stage=2, block='a', strides=(1, 1))(x)
    x = identity_block(3, [64, 64, 256], stage=2, block='b')(x)
    x = identity_block(3, [64, 64, 256], stage=2, block='c')(x)

    x = conv_block(3, [128, 128, 512], stage=3, block='a')(x)
    x = identity_block(3, [128, 128, 512], stage=3, block='b')(x)
    x = identity_block(3, [128, 128, 512], stage=3, block='c')(x)
    x = identity_block(3, [128, 128, 512], stage=3, block='d')(x)

    x = conv_block(3, [256, 256, 1024], stage=4, block='a')(x)
    x = identity_block(3, [256, 256, 1024], stage=4, block='b')(x)
    x = identity_block(3, [256, 256, 1024], stage=4, block='c')(x)
    x = identity_block(3, [256, 256, 1024], stage=4, block='d')(x)
    x = identity_block(3, [256, 256, 1024], stage=4, block='e')(x)
    x = identity_block(3, [256, 256, 1024], stage=4, block='f')(x)

    x = conv_block(3, [512, 512, 2048], stage=5, block='a')(x)
    x = identity_block(3, [512, 512, 2048], stage=5, block='b')(x)
    x = identity_block(3, [512, 512, 2048], stage=5, block='c')(x)
    #classifying layer
    x = Convolution2D(classes, 1, 1, init='he_normal', activation='linear', border_mode='valid', subsample=(1, 1), W_regularizer=l2(weight_decay))(x)

    x = BilinearUpSampling2D(target_size=tuple(image_size))(x)

    model = Model(img_input, x)
    # weights_path = os.path.expanduser(os.path.join('~', '.keras/models/fcn_resnet50_weights_tf_dim_ordering_tf_kernels.h5'))
    # model.load_weights(weights_path)
    return model

def AtrousFCN_Resnet50_16s(input_shape = None, weight_decay=0., batch_momentum=0.9, batch_shape=None, classes=21):
    if batch_shape:
        img_input = Input(batch_shape=batch_shape)
        image_size = batch_shape[1:3]
    else:
        img_input = Input(shape=input_shape)
        image_size = input_shape[0:2]

    bn_axis = 3

    x = Convolution2D(64, 7, 7, subsample=(2, 2), border_mode='same', name='conv1_new', W_regularizer=l2(weight_decay))(img_input)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1', momentum=batch_momentum)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(3, [64, 64, 256], stage=2, block='a', weight_decay=weight_decay, strides=(1, 1), batch_momentum=batch_momentum)(x)
    x = identity_block(3, [64, 64, 256], stage=2, block='b', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)
    x = identity_block(3, [64, 64, 256], stage=2, block='c', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)

    x = conv_block(3, [128, 128, 512], stage=3, block='a', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)
    x = identity_block(3, [128, 128, 512], stage=3, block='b', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)
    x = identity_block(3, [128, 128, 512], stage=3, block='c', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)
    x = identity_block(3, [128, 128, 512], stage=3, block='d', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)

    x = conv_block(3, [256, 256, 1024], stage=4, block='a', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)
    x = identity_block(3, [256, 256, 1024], stage=4, block='b', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)
    x = identity_block(3, [256, 256, 1024], stage=4, block='c', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)
    x = identity_block(3, [256, 256, 1024], stage=4, block='d', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)
    x = identity_block(3, [256, 256, 1024], stage=4, block='e', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)
    x = identity_block(3, [256, 256, 1024], stage=4, block='f', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)

    x = atrous_conv_block(3, [512, 512, 2048], stage=5, block='a', weight_decay=weight_decay, atrous_rate=(2, 2), batch_momentum=batch_momentum)(x)
    x = atrous_identity_block(3, [512, 512, 2048], stage=5, block='b', weight_decay=weight_decay, atrous_rate=(2, 2), batch_momentum=batch_momentum)(x)
    x = atrous_identity_block(3, [512, 512, 2048], stage=5, block='c', weight_decay=weight_decay, atrous_rate=(2, 2), batch_momentum=batch_momentum)(x)
    #classifying layer
    #x = AtrousConvolution2D(classes, 3, 3, atrous_rate=(2, 2), init='normal', activation='linear', border_mode='same', subsample=(1, 1), W_regularizer=l2(weight_decay))(x)
    x = Convolution2D(classes, 1, 1, init='he_normal', activation='linear', border_mode='same', subsample=(1, 1), W_regularizer=l2(weight_decay))(x)
    x = BilinearUpSampling2D(target_size=tuple(image_size))(x)

    model = Model(img_input, x)
    weights_path = os.path.expanduser(os.path.join('~', '.keras/models/fcn_resnet50_weights_tf_dim_ordering_tf_kernels.h5'))
    model.load_weights(weights_path, by_name=True)
    return model



def DnCNN(input_shape=None):
    img_input = Input(shape=input_shape)

    x = Convolution2D(64, 3, 3, border_mode='same', name='conv1', activation='relu')(img_input)

    x = Convolution2D(64, 3, 3, border_mode='same', name='conv2')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = Convolution2D(64, 3, 3, border_mode='same', name='conv3')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = Convolution2D(64, 3, 3, border_mode='same', name='conv4')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = Convolution2D(64, 3, 3, border_mode='same', name='conv5')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = Convolution2D(64, 3, 3, border_mode='same', name='conv6')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = Convolution2D(64, 3, 3, border_mode='same', name='conv7')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = Convolution2D(64, 3, 3, border_mode='same', name='conv8')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = Convolution2D(64, 3, 3, border_mode='same', name='conv9')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = Convolution2D(64, 3, 3, border_mode='same', name='conv10')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = Convolution2D(64, 3, 3, border_mode='same', name='conv11')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = Convolution2D(64, 3, 3, border_mode='same', name='conv12')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = Convolution2D(64, 3, 3, border_mode='same', name='conv13')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = Convolution2D(64, 3, 3, border_mode='same', name='conv14')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = Convolution2D(64, 3, 3, border_mode='same', name='conv15')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = Convolution2D(64, 3, 3, border_mode='same', name='conv16')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = Convolution2D(1, 3, 3, border_mode='same', name='conv17')(x)


    model = Model(img_input, x)
    return model