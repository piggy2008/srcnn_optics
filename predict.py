import os
from models import srcnn
# from train import mean_squared_error
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

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

test_dir = '/home/ty/data/MSRA5000/image_noise'
gt_dir = '/home/ty/data/MSRA5000/image_intensity'
test_file_path = '/home/ty/data/MSRA5000/test_MSRA.txt'

file_names = read_test_file(test_file_path)
input_shape = [512, 512, 1]
image_path = os.path.join(test_dir, file_names[0])
gt_path = os.path.join(gt_dir, file_names[0])
x, w, h = generate_test_data(image_path)
y, _, _ = generate_test_data(gt_path)
model = srcnn(input_shape=input_shape, kernel_size=[3, 3])
# model.compile(loss=mean_squared_error, optimizer='adadelta')
# model.summary()
model.load_weights('checkpoint_weights_backup.h5')
result = model.predict(x)
print np.shape(result)

plt.subplot(1, 3, 1)
plt.imshow(result[0, :h, :w, 0])
plt.subplot(1, 3, 2)
plt.imshow(y[0, :h, :w, 0])
plt.subplot(1, 3, 3)
plt.imshow(x[0, :h, :w, 0])
plt.show()
