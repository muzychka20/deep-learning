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

# Отключаем предупреждения
warnings.simplefilter(action="ignore", category=FutureWarning)

# Загружаем предобученную модель MobileNet
mobile = tf.keras.applications.mobilenet.MobileNet()

# Функция для подготовки изображения
def prepare_image(file):
    # Указываем путь к изображению
    img_path = "./mobile-net/samples/"
    # Загружаем изображение с нужными размерами для модели
    img = image.load_img(img_path + file, target_size=(224,224))
    # Преобразуем изображение в массив
    img_array = image.img_to_array(img)
    # Расширяем размерность массива для соответствия входу модели
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    # Применяем предобработку для модели MobileNet
    return tf.keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)

# Пример предсказания для первого изображения
preprocessed_image = prepare_image("1.jpg")
# Получаем предсказания модели для обработанного изображения
predictions = mobile.predict(preprocessed_image)
# Декодируем результаты предсказания (классы и вероятности)
results = imagenet_utils.decode_predictions(predictions)
# Выводим результаты для первого изображения
print("1.jpg", results)

# Пример предсказания для второго изображения
preprocessed_image = prepare_image("2.jpg")
predictions = mobile.predict(preprocessed_image)
results = imagenet_utils.decode_predictions(predictions)
# Выводим результаты для второго изображения
print("2.jpg", results)

# Пример предсказания для третьего изображения
preprocessed_image = prepare_image("3.jpg")
predictions = mobile.predict(preprocessed_image)
results = imagenet_utils.decode_predictions(predictions)
# Выводим результаты для третьего изображения
print("3.jpg", results)
