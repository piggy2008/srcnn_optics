from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from SRDataGenerator import SRDataGenerator
import os
from models import srcnn, cgi, unet, unet_limit, unet_limit_dialate, srcnn_fc, unet_limit_shortcut_dialate3x3, unet_limit_shortcut_dialate, DnCNN
from losses import custom_loss, mean_squared_error, mean_absolute_error, cos_distance
from keras.preprocessing.image import *
lr_base = 0.01
lr_power = 0.9
from PIL import Image
from matplotlib import pyplot as plt
from utils.utils import crop_mnist_image

import scipy.io as sio


# ###############learning rate scheduler####################
def lr_scheduler(epoch, mode='power_decay'):
    '''if lr_dict.has_key(epoch):
        lr = lr_dict[epoch]
        print 'lr: %f' % lr'''

    if mode is 'power_decay':
        # original lr scheduler
        lr = lr_base * ((1 - float(epoch)/epochs) ** lr_power)
    if mode is 'exp_decay':
        # exponential decay
        lr = (float(lr_base) ** float(lr_power)) ** float(epoch+1)
    # adam default lr
    if mode is 'adam':
        lr = 0.001

    if mode is 'progressive_drops':
        # drops as progression proceeds, good for sgd
        if epoch > 0.9 * epochs:
            lr = 0.0001
        elif epoch > 0.75 * epochs:
            lr = 0.001
        elif epoch > 0.5 * epochs:
            lr = 0.01
        else:
            lr = 0.1

    print('lr: %f' % lr)
    return lr

nb_filters_conv1 = 64
nb_filters_conv2 = 32
kernel_size = (3, 3)
classes = 1
input_shape = (64, 64, 1)
target_shape = [64, 64]
batch_size = 32
epochs = 40

save_path = '/home/ty/code/srcnn_optics'

train_file_path = '/home/ty/data/mnist_data/mnist_data_64/noise_100/train'
train_label_path = '/home/ty/data/mnist_data/mnist_data_64/image/train'

# data_dir = '/home/ty/data/MSRA5000/image_noise'
# label_dir = '/home/ty/data/MSRA5000/image_intensity'

mat_train_path = '/home/public/noise_64/mnist_noise_64_train.mat'
mat_test_path = '/home/public/noise_64/mnist_noise_64_test.mat'

def generate_data_from_dir(train_file_path, train_label_path):
    all_images = os.listdir(train_label_path)
    all_images.sort()
    train_images = all_images
    input_data = np.zeros((len(train_images),) + input_shape)
    input_label = np.zeros((len(train_images),) + input_shape)
    # train_data = [load_img(os.path.join(train_file_path, x + '.bmp')) for x in images]
    # train_label = [load_img(os.path.join(train_label_path, x)) for x in images]

    for i, image in enumerate(train_images):
        img = load_img(os.path.join(train_file_path, image), grayscale=True)
        # img = img.resize((input_shape[1], input_shape[0]), Image.BILINEAR)
        # imgs = crop_mnist_image(img, input_shape)

        data = img_to_array(img, data_format='channels_last')
        input_data[i] = data.astype(dtype=float) / 255

        label = load_img(os.path.join(train_label_path, image), grayscale=True)
        # label = label.resize((input_shape[1], input_shape[0]), Image.BILINEAR)
        # labels = crop_mnist_image(label, input_shape)
        label_arr = img_to_array(label, data_format='channels_last')
        input_label[i] = label_arr.astype(dtype=float) / 255

        # for j in range(len(imgs)):
        #     input_data[i * 4 + j] = imgs[j]
        #     input_label[i * 4 + j] = labels[j]
    return input_data, input_label


def generate_data_from_mat(mat_train_path, mat_test_path):
    train_mat = sio.loadmat(mat_test_path)
    input_data = train_mat['X_noise']
    input_label = train_mat['X']

    return input_data, input_label

input_data, input_label = generate_data_from_dir(train_file_path, train_label_path)


# train_datagen = SRDataGenerator(crop_mode='random',
#                                 crop_size=target_shape,
#                                 # pad_size=(505, 505),
#                                 rotation_range=0.,
#                                 shear_range=0,
#                                 horizontal_flip=True,
#                                 nomal=True,
#                                  fill_mode='constant')
# generator = train_datagen.flow_from_directory(file_path=train_file_path,
#             data_dir=data_dir, data_suffix=data_suffix,
#             label_dir=label_dir, label_suffix=label_suffix,
#             target_size=target_shape, color_mode='grayscale',
#             batch_size=batch_size, shuffle=True,
#             loss_shape=None)

scheduler = LearningRateScheduler(lr_scheduler)
callbacks = [scheduler]


# ################### checkpoint saver#######################
checkpoint = ModelCheckpoint(filepath=os.path.join(save_path, 'checkpoint_weights.h5'),
                             save_weights_only=True)  # .{epoch:d}
callbacks.append(checkpoint)

# model = srcnn(input_shape=input_shape, kernel_size=[3, 3])
model = unet_limit(input_shape=input_shape)
# model.load_weights('unet_optics_l2.h5')
model.compile(loss=mean_squared_error, optimizer='adadelta')
model.summary()
history = model.fit(input_data, input_label, batch_size=batch_size, nb_epoch=epochs,
                    callbacks=callbacks,
                    verbose=1)


model.save_weights('unet_limit_dialate_l2_mnist_noise100_64.h5')
