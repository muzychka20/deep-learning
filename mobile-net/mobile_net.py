import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import VGG16, imagenet_utils
from sklearn.metrics import confusion_matrix
from IPython.display import Image
import itertools
import os
import shutil
import random
import glob
import matplotlib.pyplot as plt
import warnings
import numpy as np
warnings.simplefilter(action="ignore", category=FutureWarning)

mobile = tf.keras.applications.mobilenet.MobileNet()

def prepare_image(file):
    img_path = "./mobile-net/samples/"
    img = image.load_img(img_path + file, target_size=(224,224))
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return tf.keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)


preprocessed_image = prepare_image("1.jpg")
predictions = mobile.predict(preprocessed_image)
results = imagenet_utils.decode_predictions(predictions)
print("1.jpg", results)


preprocessed_image = prepare_image("2.jpg")
predictions = mobile.predict(preprocessed_image)
results = imagenet_utils.decode_predictions(predictions)
print("2.jpg", results)


preprocessed_image = prepare_image("3.jpg")
predictions = mobile.predict(preprocessed_image)
results = imagenet_utils.decode_predictions(predictions)
print("3.jpg", results)