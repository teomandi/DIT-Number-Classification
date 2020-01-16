import pickle
import matplotlib.pyplot as plt
import time
import numpy as np
import cv2
import time
import os

from tensorflow import keras
import tensorflow.keras.layers as l
import tensorflow.keras.backend as K

import tensorflow.keras.optimizers as o
import tensorflow.keras.models as m
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import to_categorical

from kerastuner.tuners import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters


LOG_DIR="tuner_"+f"{int(time.time())}"
if not os.path.isdir(LOG_DIR):
	os.mkdir(LOG_DIR)

# load dataset
mnist = keras.datasets.mnist 
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = keras.utils.normalize(x_train, axis=1)
x_test = keras.utils.normalize(x_test, axis=1)

x_train = x_train.reshape([-1, 28, 28, 1])
x_test = x_test.reshape([-1, 28, 28, 1])

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


def build_model(hp):
	model = m.Sequential()
	
	model.add(
		l.Conv2D(
			hp.Int("conv_input_units",min_value=32,max_value=256, step=32),
			(3,3),
			padding='same',
			activation='relu',
			input_shape = (28,28,1)
		)
	)
	model.add(l.MaxPooling2D((3,3)))
	
	for i in range(hp.Int("conv_n_layers", 1, 4)):
		model.add(
			l.Conv2D(
				hp.Int("conv_"+str(i+1)+"_units",min_value=32,max_value=256, step=32),
				(3,3),
				padding='same',
				activation='relu'
			)
		)
		# model.add(l.MaxPooling2D((3,3)))
	
	model.add(l.Flatten())
	model.add(
		l.Dense(
			hp.Int("dense_input_units",min_value=32,max_value=256, step=32),
			activation="relu"
			)
		)
	model.add(l.Dropout(rate=0.5))
	for i in range(hp.Int("dense_n_layers", 1, 4)):
		model.add(
			l.Dense(
				hp.Int("dense_"+str(i+1)+"_units",min_value=32,max_value=256, step=32),
				activation='relu'
			)
		)


	model.add(l.Dense(10, activation='softmax'))
	model.compile(
	    optimizer='adam',
	    loss='categorical_crossentropy',
	    metrics=['accuracy']
	)
	# model.summary()
	return model


# call backs
name = "tf_{}.log".format(int(time.time()))
print("lon_name: ",name)
tensorboard = TensorBoard(log_dir='logs/{}'.format(name))
earlystop = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=0, patience=3,
    verbose=0, mode='auto',
    baseline=None,
    restore_best_weights=False
)

# model = build_model()
# model.fit(x_train, y_train, 
# 	epochs=5,
# 	batch_size=64, #maybe not at all
# 	callbacks=[tensorboard, earlystop],
# 	validation_data = (x_test,y_test)
# )


tuner = RandomSearch(
	build_model,
	objective="val_accuracy",
	max_trials = 10,
	executions_per_trial = 3, # <=too much
	directory = LOG_DIR
	)

start = time.time()

tuner.search(
	x = x_train,
	y = y_train,
	epochs = 5,
	batch_size = 64,
	validation_data = (x_test, y_test)
)

print("TOOK: ", time.time()-start)