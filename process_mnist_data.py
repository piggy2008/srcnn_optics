import numpy as np
import cv2
from keras.datasets import mnist
import scipy.io as sio

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train_resize = np.zeros([len(X_train), 64, 64])
X_test_resize = np.zeros([len(X_test), 64, 64])


# dilation_kernel = np.ones([3, 3], np.uint8)

for i in range(len(X_train)):
    tmp = cv2.resize(X_train[i, :, :], (64, 64), interpolation=cv2.INTER_LINEAR)
    # tmp = cv2.dilate(tmp, dilation_kernel, iterations=1)
    X_train_resize[i, :, :] = tmp

for i in range(len(X_test)):
    tmp = cv2.resize(X_test[i, :, :], (64, 64), interpolation=cv2.INTER_LINEAR)
    # tmp = cv2.dilate(tmp, dilation_kernel, iterations=1)
    X_test_resize[i, :, :] = tmp

sio.savemat('data/mnist_dilate_64_train.mat', {'X_train': X_train_resize, 'y_train': y_train})
sio.savemat('data/mnist_dilate_64_test.mat', {'X_test': X_test_resize, 'y_test': y_test})