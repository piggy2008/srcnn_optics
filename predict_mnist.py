import os
from models import srcnn, cgi, unet, unet_limit, unet_limit_dialate, srcnn_fc, DnCNN, mlp
# from train import mean_squared_error
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from skimage.measure import compare_psnr, compare_ssim
from scipy.ndimage.filters import gaussian_filter, median_filter
from keras.preprocessing.image import *
from utils.utils import crop_mnist_image
from utils.utils import combine_mnist_image
from scipy.misc import imresize

def predict_multiscale(input_shape, model, img):
    img = imresize(img[:, :, 0], (input_shape[0], input_shape[1]), mode='F')
    img = img[np.newaxis, ..., np.newaxis]
    result = model.predict(img)
    result = imresize(result[0, :, :, 0], (28, 28), mode='F')
    return result

def predict_multiscale_noise(input_shape, model, img):
    img = imresize(img[:, :, 0], (input_shape[0], input_shape[1]), mode='F')
    img = img[np.newaxis, ..., np.newaxis]
    result = model.predict(img)
    result = img - result
    result = imresize(result[0, :, :, 0], (28, 28), mode='F')
    return result


nb_filters_conv1 = 64
nb_filters_conv2 = 32
kernel_size = (3, 3)
classes = 1
input_shape = (64, 64, 1)
input_shape2 = (48, 48, 1)
input_shape3 = (28, 28, 1)

batch_size = 128
epochs = 50

save_path = '/home/ty/code/srcnn_optics'
train_file_path = '/home/public/mnist_data/mnist_data_64/noise_100/test'
# train_file_path = '/home/ty/data/mnist_data/mnist_denoise'
train_label_path = '/home/public/mnist_data/mnist_data_64/combine_image/test'
save_dir = '/home/ty/data/mnist_data/mnist_denoise_mlp'

all_images = os.listdir(train_file_path)
all_images.sort()
test_images = all_images
# model = unet_limit(input_shape=input_shape)
# model2 = DnCNN(input_shape=input_shape2)
# model3 = DnCNN(input_shape=input_shape3)
# model = srcnn(input_shape=input_shape, kernel_size=[3, 3])
model = mlp()
model.load_weights('mlp_noise100_64.h5')
# model2.load_weights('unet_limit_dialate_l2_mnist_combinenoise300_32.h5')
# model3.load_weights('unet_limit_dialate_l2_mnist_combinenoise300_32.h5')


total_psnr = 0.0
total_ssim = 0.0

total_psnr_median = 0.0
count = 0

for image in test_images:
    img = load_img(os.path.join(train_file_path, image), grayscale=True)
    # imgs = crop_mnist_image(img, input_shape)
    # img = img.resize((input_shape[1], input_shape[0]), Image.BILINEAR)
    x = img_to_array(img, data_format='channels_last')
    x /= 255
    x = x[np.newaxis, ...]
    x = x.reshape(1, 4096)
    result = model.predict(x)

    # y = load_img(os.path.join(train_label_path, image), grayscale=True)
    # y = y.resize((input_shape[1], input_shape[0]), Image.BILINEAR)
    # labels = crop_mnist_image(y, input_shape)
    # label_arr = img_to_array(y, data_format='channels_last')
    # results = []
    # for crop_image in imgs:
    #     result = predict_multiscale_noise(input_shape, model, crop_image)
        # result2 = predict_multiscale(input_shape2, model2, crop_image)
        # result3 = predict_multiscale(input_shape3, model3, crop_image)
        # result = (result + result + result3) / 3

        # results.append(result)
    # combine_image = combine_mnist_image(results)
    # plt.subplot(2, 2, 1)
    # plt.imshow(results[0])
    # plt.subplot(2, 2, 2)
    # plt.imshow(results[1])
    # plt.subplot(2, 2, 3)
    # plt.imshow(results[2])
    # plt.subplot(2, 2, 4)
    # plt.imshow(results[3])
    # plt.show()

    # psnr_image = compare_psnr(label_arr[:, :, 0] / 255, result)
    # ssim_image = compare_ssim(label_arr[:, :, 0] / 255, result)
    # total_psnr += psnr_image
    # total_ssim += ssim_image
    count += 1

    # combine_image = combine_image * 255
    # combine_image = combine_image.astype('uint8')
    result = result.reshape(1, 64, 64)
    result = result[0, :, :] * 255
    result = result.astype('uint8')
    img = Image.fromarray(result, mode='P')
    # img = img.resize((56, 56), Image.BILINEAR)
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
