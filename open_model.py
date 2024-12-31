import numpy as np
from random import randint
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy

from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt

import os.path

from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json

# open full model 
new_model = load_model('models/medical_trial_model.h5')
new_model.summary()
new_model.optimizer

# save as JSON
json_string = new_model.to_json()
model_architecture = model_from_json(json_string)
model_architecture.summary()

# open model, having only weights
model2 = Sequential([
    Dense(units=16, input_shape=(1,), activation='relu'),
    Dense(units=32, activation='relu'),
    Dense(units=2, activation='softmax')
])
model2.load_weights('./models/medical_trial_model_weights.h5')
print(model2.get_weights())