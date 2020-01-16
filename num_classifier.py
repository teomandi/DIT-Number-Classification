import pickle
import matplotlib.pyplot as plt
import time
import numpy as np
import cv2
import time

from tensorflow import keras
import tensorflow.keras.layers as l
import tensorflow.keras.backend as K

import tensorflow.keras.optimizers as o
import tensorflow.keras.models as m
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import to_categorical





mnist = keras.datasets.mnist 
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = keras.utils.normalize(x_train, axis=1)
x_test = keras.utils.normalize(x_test, axis=1)

x_train = x_train.reshape([-1, 28, 28, 1])
x_test = x_test.reshape([-1, 28, 28, 1])


y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = m.Sequential()

model.add(l.Conv2D(49, (3,3), padding='same', activation='relu', input_shape = (28,28,1)))
model.add(l.MaxPooling2D((3,3)))
model.add(l.Conv2D(36, (3,3), padding='same', activation='relu'))
model.add(l.MaxPooling2D((3,3)))

model.add(l.Flatten())
model.add(l.Dense(32, activation="relu"))
model.add(l.Dropout(rate=0.5))
model.add(l.Dense(64, activation="relu"))

model.add(l.Dense(10, activation='softmax'))

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()


name = "tf_{}.log".format(int(time.time()))
# print(name)
tensorboard = TensorBoard(log_dir='logs/{}'.format(name))


model.fit(x_train, y_train, epochs=10, callbacks=[tensorboard], validation_data = (x_test,y_test))
