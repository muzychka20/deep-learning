import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensoflow.keras.layers import Activation, Dense, Flatten
from tensoflow.keras.optimizers import Adam
from tensoflow.keras.metrics import categorical_crossetropy
from tensoflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import itertools
import os
import shutil
import random
import glob
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

physical_devices = tf.config.experimental.list_physical_devices("GPU")
print('Num GPUs Available')
tf.config.experimental.set_memory_growth(physical_devices[0], True)