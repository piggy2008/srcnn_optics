from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, BatchNormalization, UpSampling2D, merge
from keras.utils import np_utils
from keras import backend as K
from keras.layers import Input
from keras.models import Model
from keras.layers.core import Reshape
import os

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

    x = Convolution2D(1, 1, 1, border_mode='same')(x)
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

    x = Convolution2D(1, 1, 1, border_mode='same')(x)
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
    x = Convolution2D(256, 5, 5, activation='relu', border_mode='same', dilation_rate=1)(x)
    x = Convolution2D(256, 5, 5, activation='relu', border_mode='same', dilation_rate=1)(x)

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

def unet_limit_dialate_multiscale(input_shape=None):

    img_input = Input(shape=input_shape)

    x = Convolution2D(32, 9, 9, border_mode='same', activation='relu')(img_input)
    x = Convolution2D(32, 9, 9, activation='relu', border_mode='same')(x)
    x = Convolution2D(32, 9, 9, activation='relu', border_mode='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    scale1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(x)
    scale1 = Convolution2D(1, 1, 1, activation='relu', border_mode='same')(scale1)
    scale1 = UpSampling2D(size=(2, 2))(scale1)

    x = Convolution2D(64, 7, 7, border_mode='same', activation='relu')(x)
    x = Convolution2D(64, 7, 7, activation='relu', border_mode='same')(x)
    x = Convolution2D(64, 7, 7, activation='relu', border_mode='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    scale2 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(x)
    scale2 = Convolution2D(1, 1, 1, activation='relu', border_mode='same')(scale2)
    scale2 = UpSampling2D(size=(4, 4))(scale2)

    x = Convolution2D(128, 5, 5, border_mode='same', activation='relu')(x)
    x = Convolution2D(128, 5, 5, activation='relu', border_mode='same')(x)
    x = Convolution2D(128, 5, 5, activation='relu', border_mode='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    scale3 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(x)
    scale3 = Convolution2D(1, 1, 1, activation='relu', border_mode='same')(scale3)
    scale3 = UpSampling2D(size=(8, 8))(scale3)

    x = Convolution2D(256, 5, 5, border_mode='same', activation='relu', dilation_rate=1)(x)
    x = Convolution2D(256, 5, 5, activation='relu', border_mode='same', dilation_rate=1)(x)
    x = Convolution2D(256, 5, 5, activation='relu', border_mode='same', dilation_rate=1)(x)

    scale4 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(x)
    scale4 = Convolution2D(1, 1, 1, activation='relu', border_mode='same')(scale4)
    scale4 = UpSampling2D(size=(8, 8))(scale4)

    x = merge([scale1, scale2, scale3, scale4], mode='concat')
    x = Convolution2D(1, 1, 1, activation='relu', border_mode='same')(x)
    model = Model(img_input, x)
    return model

