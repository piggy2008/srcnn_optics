import os
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import preprocess_input
import numpy as np
from PIL import Image


def batch_data_generator(image_path, gt_path, image_list, batch_size, image_shape, num):
    batch_x = np.zeros((batch_size,) + image_shape)
    batch_y = np.zeros((batch_size,) + (image_shape[0], image_shape[1], 1,))
    cur_image_list = image_list[num*batch_size: (num+1)*batch_size]
    i = 0
    for name in cur_image_list:
        img = load_img(os.path.join(image_path, name), grayscale=True)
        x = img_to_array(img)
        x = x.astype(dtype=float) / 255
        batch_x[i] = x

        gt = load_img(os.path.join(gt_path, name), grayscale=True)
        y = img_to_array(gt)
        y = y.astype(dtype=float) / 255
        batch_y[i] = y
        i = i + 1

    # batch_x = preprocess_input(batch_x)
    return batch_x, batch_y

def batch_data_generator_3d(image_path, gt_path, image_list, batch_size, image_shape, num):
    batch_x = np.zeros((batch_size,) + image_shape)
    batch_y = np.zeros((batch_size,) + (image_shape[1], image_shape[2], 1,))
    cur_image_list = image_list[num*batch_size: (num+1)*batch_size]
    i = 0
    for name in cur_image_list:
        name_temp = name.split(' ')
        img_first = load_img(image_path + name_temp[0] + '.jpg')
        img_second = load_img(image_path + name_temp[1] + '.jpg')
        x_first = img_to_array(img_first.resize((image_shape[1], image_shape[2]), Image.BILINEAR))
        x_second = img_to_array(img_second.resize((image_shape[1], image_shape[2]), Image.BILINEAR))
        batch_x[i, 0, :, :, :] = x_first
        batch_x[i, 1, :, :, :] = x_second

        gt = load_img(gt_path + name_temp[1] + '.png', grayscale=True)
        y = img_to_array(gt.resize((image_shape[1], image_shape[2]), Image.BILINEAR)).astype(int)

        batch_y[i] = y
        i = i + 1

    batch_x = preprocess_input(batch_x)
    return batch_x, batch_y

def batch_data_generator_2f(image_path, gt_path, image_list, batch_size, image_shape, num):
    batch_x = np.zeros((batch_size,) + image_shape)
    batch_y = np.zeros((batch_size,) + (image_shape[0], image_shape[1], 1,))
    cur_image_list = image_list[num*batch_size: (num+1)*batch_size]
    i = 0
    for name in cur_image_list:
        name_temp = name.split(' ')
        img_first = load_img(image_path + name_temp[0] + '.jpg')
        img_second = load_img(image_path + name_temp[1] + '.jpg')
        x_first = img_to_array(img_first.resize((image_shape[0], image_shape[1]), Image.BILINEAR))
        x_second = img_to_array(img_second.resize((image_shape[0], image_shape[1]), Image.BILINEAR))
        batch_x[i, :, :, :3] = x_first
        batch_x[i, :, :, 3:] = x_second

        gt = load_img(gt_path + name_temp[1] + '.png', grayscale=True)
        y = img_to_array(gt.resize((image_shape[0], image_shape[1]), Image.BILINEAR)).astype(float)

        batch_y[i] = y / 255
        i = i + 1

    batch_x = preprocess_input(batch_x)
    return batch_x, batch_y

def extract_patches(X, image_dim_ordering, patch_size):

    # Now extract patches form X_disc
    if image_dim_ordering == "th":
        X = X.transpose(0, 2, 3, 1)

    list_X = []
    list_row_idx = [(i * patch_size[0], (i + 1) * patch_size[0]) for i in range(X.shape[1] / patch_size[0])]
    list_col_idx = [(i * patch_size[1], (i + 1) * patch_size[1]) for i in range(X.shape[2] / patch_size[1])]

    for row_idx in list_row_idx:
        for col_idx in list_col_idx:
            list_X.append(X[:, row_idx[0]:row_idx[1], col_idx[0]:col_idx[1], :])

    if image_dim_ordering == "th":
        for i in range(len(list_X)):
            list_X[i] = list_X[i].transpose(0, 3, 1, 2)

    return list_X

if __name__ == '__main__':
    image_path = '/home/ty/data/FBMS/FBMS_Trainingset/'
    gt_path = '/home/ty/data/FBMS/FBMS_GT2_fuse/'

    file = open('/home/ty/data/FBMS/train_FBMS_3d.txt', mode='r')

    lines = file.readlines()
    file.close()
    image_list = []
    for line in lines:
        line = line.strip('\n')
        image_list.append(line)

    print image_list
    import random
    random.shuffle(image_list)
    print image_list
    print np.shape(image_list)
    image_shape = (256, 256, 6)
    num = 0
    batch_x, batch_y = batch_data_generator_2f(image_path, gt_path, image_list, 10, image_shape, num)
    batch_y = batch_y * 255
    print np.shape(batch_y)
    print np.shape(batch_x)