"""
Random Erasingで前処理してから　WideResNet28-10で学習する
"""
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

from keras import models, layers, datasets, utils, backend, optimizers, initializers
backend.set_session(session)
import numpy as np
import time
from keras.models import Sequential
import wide_resnet
from keras.utils import np_utils
from tensorflow.python.client import device_lib
import math

print(device_lib.list_local_devices())

(Xtr, Ytr), (Xts, Yts) = datasets.cifar10.load_data()
Xtr = Xtr.astype('float32')
Xts = Xts.astype('float32')
Xtr /= 255
Xts /= 255
Ytr = np_utils.to_categorical(Ytr, 10)
Yts = np_utils.to_categorical(Yts, 10)

def model():
	model=Sequential()
	model.add(wide_resnet.WideResidualNetwork(depth=28, width=10, dropout_rate=0.1,include_top=True, weights=None,input_shape=None,classes=10, activation='softmax'))
	return model

def randomErasing(img,erasing_prob = 0.5, sl = 0.5, sh = 0.5, r1 = 0.3):
    prob = np.random.rand()
    mean = np.mean(img, axis=(0, 1))
    if prob > erasing_prob:
        return img
    else:
        area = img.shape[0] * img.shape[1]
        target_area = np.random.uniform(sl, sh) * area
        aspect_ratio = np.random.uniform(r1, 1/r1)
        h = int(round(math.sqrt(target_area * aspect_ratio)))
        w = int(round(math.sqrt(target_area // aspect_ratio)))
        if w < img.shape[1] and h < img.shape[0]:
            x1 = np.random.randint(0, img.shape[0] - h)
            y1 = np.random.randint(0, img.shape[1] - w)
            if img.shape[2] == 3:
                img[x1:x1+h, y1:y1+w,0] = mean[0]
                img[x1:x1+h, y1:y1+w,1] = mean[1]
                img[x1:x1+h, y1:y1+w,2] = mean[2]
            else:
                img[0, x1:x1+h, y1:y1+w] = mean[1]
            #plt.imshow(img)
            #plt.savefig(str(datetime.datetime.now().strftime('%s'))+".png")
            return img
        return img

sample_img = Xtr[0]
min_area_height = (sample_img.shape[0] / 100) * 2
min_area_width = (sample_img.shape[1] / 100) * 2
max_area_height = (sample_img.shape[0] / 100) * 40
max_area_width = (sample_img.shape[1] / 100) * 40
musk_target_one_height = (max_area_height - min_area_height)
musk_target_one_width = (max_area_width - min_area_width)
magnitude = int(np.random.rand() * 10)
t_width = int(musk_target_one_width/10) * magnitude / sample_img.shape[1]
t_height = int(musk_target_one_height/10) * magnitude / sample_img.shape[0]
for img in Xtr:
    img= randomErasing(img,erasing_prob = 0.5, sl = t_width, sh = t_height, r1 = 0.3)


model = model()
model.compile(optimizers.SGD(decay=1e-4), 'categorical_crossentropy', ['accuracy'])
start = time.time()
result = model.fit(Xtr,Ytr,batch_size=128,epochs=100,verbose=2,validation_split=0.1)
end = time.time() - start
print(end)
print('--------')
print(result.history)
score = model.evaluate(Xts,Yts,verbose=2)
print('===Final Test Score===')
print('Test loss:', score[0])
print('Test accuracy:', score[1])

