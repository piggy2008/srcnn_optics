from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from SRDataGenerator import SRDataGenerator
import os
from models import srcnn, cgi, unet, unet_limit, unet_limit_dialate, unet_limit_shortcut_dialate, AtrousFCN_Resnet50_16s, unet_limit_shortcut_dialate3x3
from losses import custom_loss, mean_squared_error, mean_absolute_error, cos_distance
from generate_batch_data import *
lr_base = 0.01
lr_power = 0.9

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
input_shape = (56, 56, 1)
target_shape = [56, 56]
batch_size = 100
epochs = 8


data_dir = '/home/public/mnist_data/mnist_data_56/noise_200/train'
label_dir = '/home/public/mnist_data/mnist_data_56/combine_image/train'
data_suffix = '.jpg.bmp'
label_suffix = '.jpg.bmp'
save_path = '/home/ty/code/srcnn_optics'
image_list = os.listdir(data_dir)
num_images = len(image_list)

# model = srcnn(input_shape=input_shape, kernel_size=[3, 3])
model = unet_limit_shortcut_dialate3x3(input_shape=input_shape)
# model.load_weights('unet_optics_l2.h5')
model.compile(loss=mean_squared_error, optimizer="rmsprop")
model.summary()


for i in range(epochs):
    print 'Epoch ---> ', i
    print 'Number of batches in 1 epoch:', num_images / batch_size

    for index in range(num_images / batch_size):
        batch_x, batch_y = batch_data_generator(data_dir, label_dir, image_list, batch_size, input_shape, index)
        loss = model.train_on_batch(batch_x, batch_y)
        print 'batch %d d_loss : %f' % (index, loss)

    model.save_weights(os.path.join(save_path, 'checkpoint_weights.h5'))

model.save_weights('unet_limit_shortcut_dialate3x3_optics_l2.h5')



