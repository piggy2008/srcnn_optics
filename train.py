from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from SRDataGenerator import SRDataGenerator
import os

from models import srcnn, cgi, unet, unet_limit, unet_limit_dialate, \
    unet_limit_shortcut_dialate, unet_limit_dialate_multiscale, \
    unet_limit_shortcut_dialate3x3, FCN_Resnet50_32s, AtrousFCN_Resnet50_16s

from losses import custom_loss, mean_squared_error, mean_absolute_error, cos_distance

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
input_shape = [64, 64, 1]
target_shape = [64, 64]
batch_size = 10
epochs = 8

train_file_path = '/home/public/noise_data/train_MSRA.txt'
data_dir = '/home/public/noise_data/image_noise'
label_dir = '/home/public/noise_data/image_intensity'
data_suffix = '.jpg.bmp'
label_suffix = '.jpg.bmp'
save_path = '/home/ty/code/srcnn_optics'


train_datagen = SRDataGenerator(crop_mode='random',
                                crop_size=target_shape,
                                # pad_size=(505, 505),
                                rotation_range=0.,
                                shear_range=0,
                                horizontal_flip=True,
                                nomal=True,
                                 fill_mode='constant')
generator = train_datagen.flow_from_directory(file_path=train_file_path,
            data_dir=data_dir, data_suffix=data_suffix,
            label_dir=label_dir, label_suffix=label_suffix,
            target_size=target_shape, color_mode='grayscale',
            batch_size=batch_size, shuffle=True,
            loss_shape=None)

scheduler = LearningRateScheduler(lr_scheduler)
callbacks = [scheduler]


# ################### checkpoint saver#######################
checkpoint = ModelCheckpoint(filepath=os.path.join(save_path, 'checkpoint_weights.h5'),
                             save_weights_only=True)  # .{epoch:d}
callbacks.append(checkpoint)

# model = srcnn(input_shape=input_shape, kernel_size=[3, 3])

model = unet_limit_shortcut_dialate3x3(input_shape=input_shape)

# model.load_weights('unet_optics_l2.h5')
model.compile(loss=mean_squared_error, optimizer='adadelta')
model.summary()
history = model.fit_generator(
        generator=generator,
        steps_per_epoch=5000,
        epochs=epochs,
        callbacks=callbacks,
        nb_worker=4)


model.save_weights('unet_limit_shortcut_dialate3x3_optics_l2.h5')

