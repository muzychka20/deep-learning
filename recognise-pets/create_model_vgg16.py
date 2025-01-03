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

# Игнорирование предупреждений FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)

# Функция для отображения 10 изображений
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 10, figsize=(20,20))  # Создание фигуры с 10 подграфиками
    axes = axes.flatten()  # Выпрямление массива подграфиков для удобства
    for img, ax in zip(images_arr, axes):  # Отображение каждого изображения
        ax.imshow(img)
        ax.axis('off')  # Отключение осей
    plt.tight_layout()  # Автоматическое выравнивание подграфиков
    plt.show()

# Установка рабочей директории
os.chdir('recognise-pets')

# Пути к данным
train_path = 'pet-images/train'        
valid_path = 'pet-images/valid'        
test_path = 'pet-images/test'

# Использование ImageDataGenerator для загрузки и обработки изображений
# Это позволяет подготавливать изображения для обучения модели
train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(
    directory=train_path, 
    target_size=(224,224),  # Изменение размера изображений до 224x224 пикселей
    classes=['cat','dog'],  # Классы для классификации (кошки и собаки)
    batch_size=10)  # Размер пакета (batch size)

valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(
    directory=valid_path,
    target_size=(224,224),
    classes=['cat','dog'],
    batch_size=10)

test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(
    directory=test_path,
    target_size=(224,224),
    classes=['cat','dog'],
    batch_size=10, 
    shuffle=False)  # Не перемешиваем тестовые данные для корректной оценки

# Загружаем тестовые изображения и метки для визуализации
test_imgs, test_labels = next(test_batches)
plotImages(test_imgs)  # Отображаем изображения
print(test_labels)  # Печатаем метки классов (0 для кошки, 1 для собаки)

# Загружаем предобученную модель VGG16 (без верхней части)
# Модель VGG16 уже обучена на большом наборе данных ImageNet и будет использоваться для извлечения признаков
vgg16_model = tf.keras.applications.vgg16.VGG16()
vgg16_model.summary()  # Выводим архитектуру модели

# Создаем новую модель, которая будет включать все слои VGG16, кроме последнего (полносвязного)
model = Sequential()
for layer in vgg16_model.layers[:-1]:  # Добавляем все слои VGG16, кроме последнего
    model.add(layer)

model.summary()  # Выводим структуру новой модели

# Замораживаем веса слоев VGG16, чтобы не обучать их
# Это необходимо, потому что модель VGG16 уже обучена, и мы не хотим изменять её веса
for layer in model.layers:
    layer.trainable = False

# Добавляем новый полносвязный слой для классификации на 2 класса (кошки и собаки)
model.add(Dense(units=2, activation='softmax'))  # Слой с 2 нейронами и активацией softmax для классификации

model.summary()  # Выводим итоговую структуру модели

# Компиляция модели с использованием Adam оптимизатора и функции потерь categorical_crossentropy
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Обучение модели
# Мы обучаем модель на данных train_batches с валидацией на valid_batches
model.fit(x=train_batches, validation_data=valid_batches, epochs=5, verbose=2)

# Сохранение модели в файл .h5
# Если модель уже существует, то она не будет перезаписана
if os.path.isfile('./models/pets_model_vgg16.h5') is False:
    model.save('./models/pets_model_vgg16.h5')  # Сохранение модели в указанную папку
