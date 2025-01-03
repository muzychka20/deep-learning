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

# Отключаем предупреждения типа FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)

# Переход в директорию с проектом (если скрипт выполняется в другой директории)
os.chdir('recognise-pets')

# Загружаем обученную модель из файла (предполагается, что модель была сохранена ранее)
model = load_model('./models/pets_model_vgg16.h5')
model.summary()  # Выводим архитектуру модели
model.optimizer  # Получаем текущий оптимизатор модели

# Задание путей к данным для обучения, валидации и тестирования
train_path = 'pet-images/train'        
valid_path = 'pet-images/valid'        
test_path = 'pet-images/test'

# Создание генераторов данных для обучения, валидации и тестирования изображений
train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(
    directory=train_path,  # Путь к директории с изображениями для обучения
    target_size=(224, 224),  # Изменяем размер изображений до 224x224
    classes=['cat', 'dog'],  # Классы, которые мы обучаем (кошка и собака)
    batch_size=10  # Размер пакета
)

valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(
    directory=valid_path,  # Путь к директории с изображениями для валидации
    target_size=(224, 224),  # Изменяем размер изображений до 224x224
    classes=['cat', 'dog'],  # Классы
    batch_size=10  # Размер пакета
)

test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(
    directory=test_path,  # Путь к директории с тестовыми изображениями
    target_size=(224, 224),  # Изменяем размер изображений до 224x224
    classes=['cat', 'dog'],  # Классы
    batch_size=10,  # Размер пакета
    shuffle=False  # Для тестирования изображения не перемешиваются
)

# Получаем предсказания от модели на тестовом наборе
predictions = model.predict(x=test_batches, verbose=2)
print(test_batches.classes)  # Выводим истинные метки классов для тестовых данных

# Генерация матрицы путаницы
cm = confusion_matrix(y_true=test_batches.classes, y_pred=np.argmax(predictions, axis=-1))
print(test_batches.class_indices)  # Выводим индексы классов (для интерпретации матрицы путаницы)

# Метки для классов в матрице путаницы
cm_plot_labels = ['cat', 'dog']

# Функция для визуализации матрицы путаницы
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)  # Отображаем матрицу как изображение
    plt.title(title)  # Название графика
    plt.colorbar()  # Добавляем цветовую шкалу
    tick_marks = np.arange(len(classes))  # Метки для оси X и Y
    plt.xticks(tick_marks, classes, rotation=45)  # Подписи для оси X
    plt.yticks(tick_marks, classes)  # Подписи для оси Y
    
    # Нормализуем матрицу путаницы, если необходимо
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    
    print(cm)  # Выводим саму матрицу путаницы
    
    # Добавляем числа в клетки матрицы для лучшего восприятия
    thresh = cm.max() / 2.  # Порог для изменения цвета текста (черный или белый)
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
    
    # Добавляем подписи для осей и показываем график
    plt.tight_layout()
    plt.ylabel("True label")  # Подпись для оси Y (истинные метки)
    plt.xlabel("Predicted label")  # Подпись для оси X (предсказанные метки)
    plt.show()  # Отображаем график

# Визуализация и вывод матрицы путаницы
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')
