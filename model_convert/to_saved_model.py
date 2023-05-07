import tensorflow as tf
import os
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Dense,ReLU,Dropout,Activation,concatenate,Softmax,Conv2D,MaxPool2D,Add,\
    BatchNormalization,MaxPooling2D,ZeroPadding2D,AveragePooling2D
import tensorflow.keras.layers as layers


def ResNet50(nb_class, input_shape):
    # define your model
    return model
model_ResNet50 = ResNet50(6, (256, 256, 1))

class_names = ['CSNJ', 'CW', 'LFM', 'MTJ', 'PBNJ', 'PPNJ']
Version = 'ResNet'
# CNN训练网络调整

with open("STFT.csv") as file_name:
    test_img = np.loadtxt(file_name, delimiter=",")

test_img = np.array(test_img)
test_img = np.expand_dims(test_img, axis=2)
test_img = np.expand_dims(test_img, 0)  # 将三维输入图像拓展成四维张量
# 引入网络模型
model = model_ResNet50
filepath = os.path.join(Version + '.h5')

# load model weight from .h5 file
model.load_weights(filepath)

# save network and weight to saved_model
tf.saved_model.save(model, "saved_model")
