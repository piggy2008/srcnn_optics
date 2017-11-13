import os
from models import srcnn, cgi, unet, unet_limit, unet_limit_dialate, srcnn_fc
# from train import mean_squared_error
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from skimage.measure import compare_psnr, compare_ssim
from scipy.ndimage.filters import gaussian_filter, median_filter
from keras.preprocessing.image import *

nb_filters_conv1 = 64
nb_filters_conv2 = 32
kernel_size = (3, 3)
classes = 1
input_shape = (64, 64, 1)
target_shape = [128, 128]
batch_size = 128
epochs = 50

save_path = '/home/ty/code/srcnn_optics'
train_file_path = '/home/ty/data/mnist_data/mnist_data_28/noise_200/test'
train_label_path = '/home/ty/data/mnist_data/mnist_data_28/image/test'
save_dir = '/home/ty/data/mnist_data/mnist_denoise'
input_data = np.zeros((10000,) + input_shape)
input_label = np.zeros((10000,) + input_shape)
all_images = os.listdir(train_label_path)
all_images.sort()
test_images = all_images
model = unet_limit_dialate(input_shape=input_shape)
# model = srcnn(input_shape=input_shape, kernel_size=[3, 3])

model.load_weights('unet_limit_dialate_l2_mnist_noise200.h5')

total_psnr = 0.0
total_ssim = 0.0

total_psnr_median = 0.0
count = 0

for image in test_images:
    img = load_img(os.path.join(train_file_path, image), grayscale=True)
    img = img.resize((input_shape[1], input_shape[0]), Image.BILINEAR)
    x = img_to_array(img, data_format='channels_last')
    x /= 255

    y = load_img(os.path.join(train_label_path, image), grayscale=True)
    y = y.resize((input_shape[1], input_shape[0]), Image.BILINEAR)
    label_arr = img_to_array(y, data_format='channels_last')
    x = x[np.newaxis, ...]
    result = model.predict(x)
    result = result[0, :, :, 0]

    psnr_image = compare_psnr(label_arr[:, :, 0] / 255, result)
    ssim_image = compare_ssim(label_arr[:, :, 0] / 255, result)
    total_psnr += psnr_image
    total_ssim += ssim_image
    count += 1

    result = result * 255
    result = result.astype('uint8')
    img = Image.fromarray(result, mode='P')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    img.save(os.path.join(save_dir, image))

print 'mean psnr:', (total_psnr / count)
print 'mean ssim:', (total_ssim / count)

print 'mean psnr median:', (total_psnr_median / count)
# x, w, h = generate_test_data(os.path.join(test_dir, file_names[0]))
# result = model.predict(x)
# print np.shape(result)

# plt.subplot(1, 3, 1)
# plt.imshow(result[0, :h, :w, 0])
# plt.subplot(1, 3, 2)
# plt.imshow(x[0, :h, :w, 0])
# plt.subplot(1, 3, 3)
# plt.imshow(x[0, :h, :w, 0])
# plt.show()

'''


'''
