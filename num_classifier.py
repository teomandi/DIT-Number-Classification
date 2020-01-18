import pickle
import matplotlib.pyplot as plt
import time
import numpy as np
import cv2
import time
import tensorflow as tf

import tensorflow.keras as keras
import tensorflow.keras.layers as l
import tensorflow.keras.backend as K

import tensorflow.keras.optimizers as o
import tensorflow.keras.models as m
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import to_categorical

from kerastuner.tuners import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters

from scipy.io import loadmat
import random
import os


LOG_DIR="tuner_"+f"{int(time.time())}"
if not os.path.isdir(LOG_DIR):
	os.mkdir(LOG_DIR)

# mnist
mnist = keras.datasets.mnist 
(mnist_x_train, mnist_y_train), (mnist_x_test, mnist_y_test) = mnist.load_data()

mnist_x_train = np.asarray(mnist_x_train)
mnist_x_test = np.asarray(mnist_x_test)

mnist_y_train = mnist_y_train.reshape((mnist_y_train.shape[0], 1))
mnist_y_test = mnist_y_test.reshape((mnist_y_test.shape[0], 1))

# emnist
EMNIST_digits_file = os.path.join("..", "number_dataset","emnist-digits.mat")
mat = loadmat(EMNIST_digits_file)
data = mat['dataset']

emnist_x_train = data['train'][0,0]['images'][0,0]
emnist_x_test = data['test'][0,0]['images'][0,0]
emnist_x_train = emnist_x_train.reshape( (emnist_x_train.shape[0], 28, 28), order='F')
emnist_x_test = emnist_x_test.reshape( (emnist_x_test.shape[0], 28, 28), order='F')

emnist_y_train = data['train'][0,0]['labels'][0,0]
emnist_y_test = data['test'][0,0]['labels'][0,0]

# custom one
custom_dataset_path = os.path.join("..", "number_dataset", "COMPLETE_DATASET_clean")
custom_dataset_names = os.listdir(custom_dataset_path)

IMG_SIZE = 28
custom_dataset = []
custom_labels = []

for i,img_name in enumerate(custom_dataset_names):
    # load image gray scale
    img = cv2.imread(os.path.join(custom_dataset_path,img_name), cv2.IMREAD_GRAYSCALE)
    # resize to 28x28
    img = cv2.resize(img, (IMG_SIZE,IMG_SIZE), 1)
    #thress hold it 
    ret, t_img = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)# THRESH_TOZERO_INV or THRESH_BINARY_INV
    ret, img = cv2.threshold(img, 200, 255, cv2.THRESH_TOZERO_INV)

    # append it
    custom_dataset.append(img)

    # label
    current_label = np.zeros(10)
    current_label[int(img_name[0])]=1
    custom_labels.append(current_label)


my_seed = int(10*random.random())
random.Random(my_seed).shuffle(custom_dataset)
random.Random(my_seed).shuffle(custom_labels)
custom_x_train = np.asarray(custom_dataset[:int(0.75*len(custom_dataset))])
custom_y_train = np.asarray(custom_labels[:int(0.75*len(custom_labels))])
custom_x_test = np.asarray(custom_dataset[int(0.75*len(custom_dataset)):])
custom_y_test = np.asarray(custom_labels[int(0.75*len(custom_labels)):])

print("~~initial data~~")
print(emnist_y_train.shape)
print(mnist_y_train.shape)
print(custom_y_train.shape)


# concat them all
x_train, x_test = np.concatenate((emnist_x_train, mnist_x_train)), np.concatenate((emnist_x_test, mnist_x_test))
x_train, x_test = np.concatenate((x_train, custom_x_train)), np.concatenate((x_test, custom_x_test))
y_train, y_test = np.concatenate((emnist_y_train, mnist_y_train)), np.concatenate((emnist_y_test, mnist_y_test))
y_train, y_test = keras.utils.to_categorical(y_train, 10), keras.utils.to_categorical(y_test, 10)
y_train, y_test = np.concatenate((y_train, custom_y_train)), np.concatenate((y_test, custom_y_test))


def mirro_rotate(dataset, dataset_labels, ROT_DEGGREES=20):
    augmented_image = []
    augmented_image_labels = []
    
    for num in range (0, dataset.shape[0]):
        # original image:
        augmented_image.append(dataset[num])
        augmented_image_labels.append(dataset_labels[num])
        #rotate 
        height, width = dataset[num].shape
        rotation_matrix = cv2.getRotationMatrix2D((width/2,height/2), ROT_DEGGREES ,1)
        rotated_image = cv2.warpAffine(dataset[num], rotation_matrix, (width, height))
        augmented_image.append(rotated_image)
        augmented_image_labels.append(dataset_labels[num])
 		#----second one       
        rotation_matrix = cv2.getRotationMatrix2D((width/2,height/2), (-1)*ROT_DEGGREES ,1)
        rotated_image = cv2.warpAffine(dataset[num], rotation_matrix, (width, height))
        augmented_image.append(rotated_image)
        augmented_image_labels.append(dataset_labels[num])
 
    return np.array(augmented_image), np.array(augmented_image_labels)

x_train, y_train = mirro_rotate(x_train, y_train, 10)
print("~~ratated~~")
print(x_train.shape)
print(y_train.shape)

x_train, x_test = ((1.0/255.0)*x_train).reshape([-1, 28, 28, 1]), ((1.0/255.0)*x_test).reshape([-1, 28, 28, 1]) 

# -----------------------------------------------
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
		model.add(l.MaxPooling2D((3,3)))
	
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
	executions_per_trial = 1, # <=too much
	directory = LOG_DIR
	)

start = time.time()

tuner.search(
	x = x_train,
	y = y_train,
	epochs = 4,
	batch_size = 64,
	validation_data = (x_test, y_test)
)

print("TOOK: ", time.time()-start)