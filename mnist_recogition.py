'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
import cv2

batch_size = 128
nb_classes = 10
nb_epoch = 12

# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)
dilation_kernel = np.ones([3, 3], np.uint8)
# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train_resize = np.zeros([len(X_train), 64, 64])
X_test_resize = np.zeros([len(X_test), 64, 64])
if K.image_dim_ordering() == 'th':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    for i in range(len(X_train)):
        tmp = cv2.resize(X_train[i, :, :], (64, 64), interpolation=cv2.INTER_LINEAR)
        tmp = cv2.dilate(tmp, dilation_kernel, iterations=1)
        X_train_resize[i, :, :] = tmp

    for i in range(len(X_test)):
        tmp = cv2.resize(X_test[i, :, :], (64, 64), interpolation=cv2.INTER_LINEAR)
        tmp = cv2.dilate(tmp, dilation_kernel, iterations=1)
        X_test_resize[i, :, :] = tmp

    X_train_resize = X_train_resize.reshape(X_train_resize.shape[0], 64, 64, 1)
    X_test_resize = X_test_resize.reshape(X_test_resize.shape[0], 64, 64, 1)
    # input_shape = (img_rows, img_cols, 1)
input_shape = (64, 64, 1)
X_train_resize = X_train_resize.astype('float32')
X_test_resize = X_test_resize.astype('float32')
X_train_resize /= 255
X_test_resize /= 255
print('X_train_resize shape:', X_train_resize.shape)
print(X_train_resize.shape[0], 'train samples')
print(X_test_resize.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()

model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                        border_mode='valid',
                        input_shape=input_shape))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

model.fit(X_train_resize, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_data=(X_test_resize, Y_test))
score = model.evaluate(X_test_resize, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

model.save_weights('mnist_recognition_64.h5')
