import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix
import itertools
import os
import shutil
import random
import glob
import matplotlib.pyplot as plt
import warnings
import numpy as np
warnings.simplefilter(action='ignore', category=FutureWarning)

os.chdir('recognise-pets')

model = load_model('./models/pets_model_vgg16.h5')
model.summary()
model.optimizer

train_path = 'pet-images/train'        
valid_path = 'pet-images/valid'        
test_path = 'pet-images/test'

train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(directory=train_path, 
                                                                                                                            target_size=(224,224), 
                                                                                                                            classes=['cat','dog'], 
                                                                                                                            batch_size=10)

valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(directory=valid_path,
                                                                                                                            target_size=(224,224), 
                                                                                                                            classes=['cat','dog'], 
                                                                                                                            batch_size=10)

test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(directory=test_path,
                                                                                                                           target_size=(224,224), 
                                                                                                                           classes=['cat','dog'], 
                                                                                                                           batch_size=10, 
                                                                                                                           shuffle=False)
