#-*- coding:utf-8 -*-
import os
import numpy as np
from imp import reload
from PIL import Image, ImageOps

from keras.layers import Input
from keras.models import Model
# import keras.backend as K

from . import keys
from . import densenet

reload(densenet)

characters = keys.alphabet[:]
characters = f'{characters[1:]}卍'
nclass = len(characters)

input = Input(shape=(32, None, 1), name='the_input')
y_pred= densenet.dense_cnn(input, nclass)
basemodel = Model(inputs=input, outputs=y_pred)

modelPath = os.path.join(os.getcwd(), 'densenet/models/weights_densenet.h5')
if os.path.exists(modelPath):
    basemodel.load_weights(modelPath)

def decode(pred):
    pred_text = pred.argmax(axis=2)[0]
    char_list = [
        characters[pred_text[i]]
        for i in range(len(pred_text))
        if pred_text[i] != nclass - 1
        and (
            i <= 0
            or pred_text[i] != pred_text[i - 1]
            or (i > 1 and pred_text[i] == pred_text[i - 2])
        )
    ]
    return u''.join(char_list)

def predict(img):
    width, height = img.size[0], img.size[1]
    scale = height * 1.0 / 32
    width = int(width / scale)

    img = img.resize([width, 32], Image.ANTIALIAS)

    '''
    img_array = np.array(img.convert('1'))
    boundary_array = np.concatenate((img_array[0, :], img_array[:, width - 1], img_array[31, :], img_array[:, 0]), axis=0)
    if np.median(boundary_array) == 0:  # 将黑底白字转换为白底黑字
        img = ImageOps.invert(img)
    '''

    img = np.array(img).astype(np.float32) / 255.0 - 0.5

    X = img.reshape([1, 32, width, 1])

    y_pred = basemodel.predict(X)
    y_pred = y_pred[:, :, :]

    return decode(y_pred)
