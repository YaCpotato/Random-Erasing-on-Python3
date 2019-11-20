"""
basicなCNNでやる用です。画像のサンプルプロットなどもここで行いました。
"""
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import datetime
from keras.datasets import cifar10
import keras
import keras.backend as K
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils


def P(y_true, y_pred):
    true_positives = K.sum(K.cast(K.greater(K.clip(y_true * y_pred, 0, 1), 0.20), 'float32'))
    pred_positives = K.sum(K.cast(K.greater(K.clip(y_pred, 0, 1), 0.20), 'float32'))

    precision = true_positives / (pred_positives + K.epsilon())
    return precision

#recall
def R(y_true, y_pred):
    true_positives = K.sum(K.cast(K.greater(K.clip(y_true * y_pred, 0, 1), 0.20), 'float32'))
    poss_positives = K.sum(K.cast(K.greater(K.clip(y_true, 0, 1), 0.20), 'float32'))

    recall = true_positives / (poss_positives + K.epsilon())
    return recall


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

def model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding="same", input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(10))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy',P,R])

    model.summary()
    return model

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    sample_img = x_train[0]
    min_area_height = (sample_img.shape[0] / 100) * 2
    min_area_width = (sample_img.shape[1] / 100) * 2
    max_area_height = (sample_img.shape[0] / 100) * 40
    max_area_width = (sample_img.shape[1] / 100) * 40
    musk_target_one_height = (max_area_height - min_area_height)
    musk_target_one_width = (max_area_width - min_area_width)
    magnitude = int(np.random.rand() * 10)
    t_width = int(musk_target_one_width/10) * magnitude / sample_img.shape[1]
    t_height = int(musk_target_one_height/10) * magnitude / sample_img.shape[0]
    #i=0
    for img in x_train:
        img= randomErasing(img,erasing_prob = 0.5, sl = t_width, sh = t_height, r1 = 0.3)
        #i+=1
        #if(i%7 == 0):#これを外すと7枚ごとにサンプル画像をpngとして保存するのでオススメしません。フォルダを圧迫します。したい方はどうぞ
            #plt.imshow(img)
            #plt.savefig(str(i)+'.png')
    
    #plt.figure()
    #plt.imshow(x_train[0])
    #plt.savefig('RandomErasingEffectedSample.png')
    #plt.figure()
    #plt.subplot(321)
    #plt.imshow(x_train[0])
    #plt.subplot(322)
    #plt.imshow(x_train[1])
    #plt.subplot(323)
    #plt.imshow(x_train[2])
    #plt.subplot(324)
    #plt.imshow(x_train[3])
    #plt.subplot(325)
    #plt.imshow(x_train[4])
    #plt.subplot(326)
    #plt.imshow(x_train[5])
    #plt.savefig('RandomErasingEffectedSample2.png')
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)


    model = model()
    model.fit(x_train, y_train, batch_size=10, epochs=1, validation_split=0.1)
    pred = model.predict(x_test)
    pred = np.argmax(pred, axis=1)
    Y = np.argmax(y_test,axis=1)
    score = model.evaluate(x_test, y_test)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    print(confusion_matrix(pred,Y))