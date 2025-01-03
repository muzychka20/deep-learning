import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Activation, Dense, Resizing, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, imagenet_utils, MobileNet
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from IPython.display import Image
from os import listdir
from imageio import imread
import itertools
import os
import shutil
import random
import glob
import matplotlib.pyplot as plt
import warnings
import numpy as np
warnings.simplefilter(action="ignore", category=FutureWarning)

# Переходим в директорию с примерами изображений жестов
os.chdir("sign-language/samples")

# Если директории для тренировочных, валидирующих и тестовых данных не существуют, создаём их
if not os.path.isdir("train"):
    os.mkdir("train")
    os.mkdir("valid")
    os.mkdir("test")
    
    # Для каждой папки (цифры от 0 до 9) переносим данные в train, valid и test
    for i in range(0, 10):
        shutil.move(f'{i}', 'train')  # Перемещаем изображения в тренировочную папку
        os.mkdir(f'valid/{i}')  # Создаём папку для валидации
        os.mkdir(f'test/{i}')  # Создаём папку для тестирования
        
        # Перемещаем случайные 30 изображений в папку valid
        valid_samples = random.sample(os.listdir(f'train/{i}'), 30)
        for j in valid_samples:
            shutil.move(f"train/{i}/{j}", f"valid/{i}")
            
        # Перемещаем случайные 5 изображений в папку test
        test_samples = random.sample(os.listdir(f'train/{i}'), 5)
        for j in test_samples:
            shutil.move(f"train/{i}/{j}", f"test/{i}")

# Переходим обратно в основную директорию
os.chdir("../../sign-language")


# Устанавливаем пути к директориям с изображениями
train_path = "samples/train"
valid_path = "samples/valid"
test_path = "samples/test"

# Подготавливаем генераторы данных для тренировочных, валидационных и тестовых данных
train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(directory=train_path, target_size=(224,224), batch_size=10)
valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(directory=valid_path, target_size=(224,224), batch_size=10)
test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(directory=test_path, target_size=(224,224), batch_size=10, shuffle=False)

# Загружаем предобученную модель MobileNet
mobile = tf.keras.applications.MobileNet(weights='imagenet')
# Выводим summary модели, чтобы понять её структуру
mobile.summary()

# Извлекаем выход с 6-го слоя с конца (перед классификационным слоем)
x = mobile.layers[-6].output
# Применяем Flatten, чтобы выпрямить данные перед подачей в новый плотный слой
x = Flatten()(x)
# Добавляем новый плотный слой для классификации (10 классов, активация softmax)
output = Dense(units=10, activation='softmax')(x)

# Создаём новую модель, используя входы от MobileNet и новый выходной слой
model = Model(inputs=mobile.input, outputs=output)

# Замораживаем все слои модели, кроме последних 23
for layer in model.layers[:-23]:
    layer.trainable = False

# Выводим структуру новой модели
model.summary()

# Компилируем модель: используем Adam с маленьким шагом (lr=0.0001), функцию потерь categorical_crossentropy и метрику accuracy
model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Обучаем модель на тренировочных данных, используя валидационные данные для мониторинга
model.fit(x=train_batches, validation_data=valid_batches, epochs=30, verbose=2)

# Если модель ещё не сохранена, сохраняем её
if not os.path.isfile("models/sign_model.h5"):
    model.save("models/sign_model.h5")
