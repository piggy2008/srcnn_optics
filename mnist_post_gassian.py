import os
import cv2
from PIL import Image

train_file_path = '/home/ty/data/mnist_data/mnist_denoise'
images = os.listdir(train_file_path)
save_dir = '/home/ty/data/mnist_data/mnist_denoise_bilateral'


for image in images:
    img = cv2.imread(os.path.join(train_file_path, image), 0)
    result = cv2.bilateralFilter(img, 9, 75, 75)

    result = result.astype('uint8')
    img = Image.fromarray(result, mode='P')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    img.save(os.path.join(save_dir, image))