import os
from models import srcnn, cgi, unet, unet_limit, unet_limit_dialate, \
    unet_limit_shortcut_dialate, unet_limit_dialate_multiscale, unet_limit_shortcut_dialate3x3, AtrousFCN_Resnet50_16s, FCN_Resnet50_32s

# from train import mean_squared_error
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from skimage.measure import compare_psnr, compare_ssim
from scipy.ndimage.filters import gaussian_filter, median_filter


def read_test_file(path, suffix='.jpg.bmp'):
    files = []

    fp = open(path)
    lines = fp.readlines()
    fp.close()

    for line in lines:
        line = line.strip('\n').strip('\r')
        files.append(line + suffix)

    return files

def generate_test_data(image_path):
    im = Image.open(image_path)
    w, h = im.size
    if max(w, h) > 510:
        if w > h:
            im = im.resize([510, int(510. / w * h)])

        else:
            im = im.resize([int(510. / h * w), 510])

    w, h = im.size
    in_ = np.array(im, dtype=np.float32)
    print np.shape(in_)
    in_ = in_/ 255
    npad = ([0, 512 - h], [0, 512 - w])
    in_ = np.pad(in_, npad, 'constant')
    in_ = in_[np.newaxis, ..., np.newaxis]
    print np.shape(in_)
    return in_, w, h


test_dir = '/home/public/noise_data/image_noise'
gt_dir = '/home/public/noise_data/image_intensity'
test_file_path = '/home/public/noise_data/test_MSRA.txt'
save_dir = '/home/public/noise_data/image_denoise'

file_names = read_test_file(test_file_path)
input_shape = [512, 512, 1]
image_path = os.path.join(test_dir, file_names[0])
gt_path = os.path.join(gt_dir, file_names[0])
# x, w, h = generate_test_data(image_path)
# y, _, _ = generate_test_data(gt_path)
# model = srcnn(input_shape=input_shape, kernel_size=[3, 3])

model = FCN_Resnet50_32s(input_shape=input_shape, classes=1)
# model.compile(loss=mean_squared_error, optimizer='adadelta')
# model.summary()
model.load_weights('unet_limit_shortcut_dialate3x3_optics_l2.h5')


total_psnr = 0.0
total_ssim = 0.0

total_psnr_median = 0.0
count = 0
for name in file_names:
    x, w, h = generate_test_data(os.path.join(test_dir, name))
    result = model.predict(x)
    result = result * 255
    result = result.astype('uint8')
    result = result[0, :h, :w, 0]

    # median filter
    median_result = median_filter(x, 3)
    median_result = median_result * 255
    median_result = median_result.astype('uint8')
    median_result = median_result[0, :h, :w, 0]

    gt = np.array(Image.open(os.path.join(gt_dir, name)))
    psnr_image = compare_psnr(gt, result)
    ssim_image = compare_ssim(gt, result)

    psnr_image_median = compare_psnr(gt, median_result)


    total_psnr += psnr_image
    total_ssim += ssim_image

    total_psnr_median += psnr_image_median

    count += 1
    # img = Image.fromarray(result, mode='P')
    # if not os.path.exists(save_dir):
    #     os.mkdir(save_dir)
    # img.save(os.path.join(save_dir, name))
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
lr_base = 0.001
lr_power = 0.95
psnr: 14.1  ssim: 0.21

lr_base = 0.01
lr_power = 0.90
psnr: 16.8  ssim: 0.35

lr_base = 0.01
lr_power = 0.90
l1: 0.55 l2:0.45
no bn
psnr: 18.37  ssim: 0.47

lr_base = 0.01
lr_power = 0.90
l1: 1 l2:0
no bn
psnr: 18.74  ssim: 0.48

lr_base = 0.01
lr_power = 0.90
l1: 0 l2:1
no bn
psnr: 18.46  ssim: 0.47

lr_base = 0.01
lr_power = 0.90
l1: 0.45 l2:0.55
no bn
psnr: 18.50  ssim: 0.47

lr_base = 0.01
lr_power = 0.90
l1: 0.2 l2:0.8
no bn
psnr: 18.83  ssim: 0.47

lr_base = 0.01
lr_power = 0.90
l1: 0.8 l2:0.2
no bn
psnr: 18.72  ssim: 0.48

lr_base = 0.01
lr_power = 0.90
l1: head l2: tail
no bn
psnr: 18.77  ssim: 0.48

lr_base = 0.01
lr_power = 0.90
l1: 0 l2: 1
cgi model
psnr: 19.92  ssim: 0.56

lr_base = 0.01
lr_power = 0.90
l1: 0 l2: 1
unet model
psnr: 20.18  ssim: 0.57

lr_base = 0.01
lr_power = 0.90
l1: 0 l2: 1
unet_limit model
psnr: 20.17  ssim: 0.61

lr_base = 0.01
lr_power = 0.90
l1: 0 l2: 1
unet_limit_dialate model dialate=1 with 5*5 kernel
psnr: 19.79  ssim: 0.61

lr_base = 0.01
lr_power = 0.90
l1: 0 l2: 1
unet_limit_dialate model dialate=1 with 3*3 kernel
psnr: 20.10  ssim: 0.61


lr_base = 0.01
lr_power = 0.90
l1: 0.2 l2: 0.8
unet_limit_dialate model dialate=1 with 3*3 kernel
psnr: 20.12  ssim: 0.61

lr_base = 0.01
lr_power = 0.90
l1: 0 l2: 1
unet_limit_shortcut_dialate model
psnr: 20.04  ssim: 0.54

'''
