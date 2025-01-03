from tensorflow import keras
from tensorflow.keras.models import model_from_json  # для загрузки архитектуры модели из JSON
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model  # для загрузки полной модели
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from random import randint
import tensorflow as tf
import numpy as np
import itertools
import os.path

# Открытие полной модели из файла
new_model = load_model('models/medical_trial_model.h5')  # Загружаем модель целиком из файла .h5
new_model.summary()  # Выводим сводку модели (архитектура, количество параметров)
new_model.optimizer  # Получаем информацию о текущем оптимизаторе модели

# Сохранение архитектуры модели в формате JSON
json_string = new_model.to_json()  # Сохраняем архитектуру модели в формате JSON
model_architecture = model_from_json(json_string)  # Загружаем модель из JSON
model_architecture.summary()  # Выводим сводку модели, загруженной из JSON

# Открытие модели с только весами
# Сначала создаем пустую модель с той же архитектурой
model2 = Sequential([
    Dense(units=16, input_shape=(1,), activation='relu'),  # Слой с 16 нейронами и функцией активации ReLU
    Dense(units=32, activation='relu'),  # Слой с 32 нейронами и функцией активации ReLU
    Dense(units=2, activation='softmax')  # Выходной слой с 2 нейронами и функцией активации Softmax
])

# Загружаем только веса для модели из файла
model2.load_weights('./models/medical_trial_model_weights.h5')  # Загружаем веса модели из файла
print(model2.get_weights())  # Выводим веса модели для проверки
