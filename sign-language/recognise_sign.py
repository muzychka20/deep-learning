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

# Отключаем предупреждения о будущем
warnings.simplefilter(action="ignore", category=FutureWarning)

# Функция для визуализации матрицы ошибок (confusion matrix)
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    # Визуализируем матрицу ошибок с использованием цветовой карты
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)  # Метки для оси X (предсказания)
    plt.yticks(tick_marks, classes)  # Метки для оси Y (истинные значения)
    
    # Если normalize=True, нормализуем матрицу ошибок (по строкам)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    
    # Выводим матрицу ошибок
    print(cm)
    
    # Определяем порог для цвета текста на ячейках матрицы
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        # Добавляем текст в каждую ячейку с соответствующим значением
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
    
    # Подгоняем визуализацию, чтобы все элементы поместились
    plt.tight_layout()
    plt.ylabel("True label")  # Подпись для оси Y (истинные значения)
    plt.xlabel("Predicted label")  # Подпись для оси X (предсказанные значения)
    plt.show()  # Отображаем график

# Переходим в основную папку проекта (где находятся данные и модель)
os.chdir("sign-language")

# Загружаем ранее сохраненную модель из файла
model = load_model('./models/sign_model.h5')
# Выводим сводку модели для ознакомления с ее архитектурой
model.summary()

# Устанавливаем пути к директориям с изображениями для тренировочных, валидационных и тестовых данных
train_path = "samples/train"
valid_path = "samples/valid"
test_path = "samples/test"

# Подготавливаем генераторы данных для тренировочных, валидационных и тестовых данных с нужной предобработкой изображений
train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(directory=train_path, target_size=(224,224), batch_size=10)
valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(directory=valid_path, target_size=(224,224), batch_size=10)
test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(directory=test_path, target_size=(224,224), batch_size=10, shuffle=False)

# Получаем метки для тестовых данных (классы, которые должны быть предсказаны)
test_labels = test_batches.classes

# Прогнозируем результаты для тестовых данных с использованием обученной модели
predictions = model.predict(x=test_batches, verbose=0)

# Строим матрицу ошибок (confusion matrix) для тестовых данных
cm = confusion_matrix(y_true=test_labels, y_pred=predictions.argmax(axis=1))

# Выводим индексы классов (соответствия меток с индексами)
test_batches.class_indices

# Создаем список меток для отображения на оси x и y матрицы ошибок (0-9 для цифр)
cm_plot_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# Визуализируем и выводим матрицу ошибок с помощью ранее созданной функции
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')
