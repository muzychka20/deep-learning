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

# Отключение предупреждений типа FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)

# Функция для визуализации изображений
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 10, figsize=(20,20))  # Создание графика с 10 изображениями
    axes = axes.flatten()  # Уплощение массива для удобного обращения
    for img, ax in zip(images_arr, axes):  # Цикл по изображению и соответствующей оси
        ax.imshow(img)  # Отображение изображения
        ax.axis('off')  # Убираем оси
    plt.tight_layout()  # Автоматически подгоняем расположение изображений
    plt.show()  # Отображаем график

# Переход в директорию с проектом
os.chdir('recognise-pets')

# Загрузка модели, ранее обученной и сохраненной в файл
model = load_model('./models/pets_model.h5')  # Загружаем модель
model.summary()  # Выводим структуру модели
model.optimizer  # Выводим информацию об оптимизаторе модели

# Пути к папкам с данными для обучения, валидации и тестирования
train_path = 'pet-images/train'        
valid_path = 'pet-images/valid'        
test_path = 'pet-images/test'

# Генератор данных для обучения
train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(
    directory=train_path,  # Путь к папке с изображениями для обучения
    target_size=(224, 224),  # Изменение размера изображений до 224x224
    classes=['cat', 'dog'],  # Классы: кошка и собака
    batch_size=10  # Размер батча
)

# Генератор данных для валидации
valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(
    directory=valid_path,  # Путь к папке с изображениями для валидации
    target_size=(224, 224),  # Изменение размера изображений до 224x224
    classes=['cat', 'dog'],  # Классы
    batch_size=10  # Размер батча
)

# Генератор данных для тестирования
test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(
    directory=test_path,  # Путь к папке с изображениями для тестирования
    target_size=(224, 224),  # Изменение размера изображений до 224x224
    classes=['cat', 'dog'],  # Классы
    batch_size=10,  # Размер батча
    shuffle=False  # Для теста изображения не будут перемешиваться
)

# Извлекаем несколько изображений и их метки из тестового набора
test_imgs, test_labels = next(test_batches)

# Отображаем тестовые изображения
plotImages(test_imgs)
print(test_labels)  # Печатаем метки для отображаемых изображений
print(test_batches.classes)  # Печатаем все метки в тестовом наборе

# Получаем предсказания модели для тестового набора
predictions = model.predict(x=test_batches, verbose=0)  # Прогнозируем классы для всех тестовых изображений
print(np.round(predictions))  # Печатаем округленные предсказания модели

# Создание матрицы ошибок для анализа ошибок
cm = confusion_matrix(y_true=test_batches.classes, y_pred=np.argmax(predictions, axis=-1))

# Функция для визуализации матрицы ошибок
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    # Отображаем матрицу ошибок как изображение
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)  # Название графика
    plt.colorbar()  # Добавляем цветовую шкалу
    tick_marks = np.arange(len(classes))  # Создаем метки для осей
    plt.xticks(tick_marks, classes, rotation=45)  # Подписи для оси X
    plt.yticks(tick_marks, classes)  # Подписи для оси Y
    
    # Нормализация матрицы ошибок, если указано
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    
    # Печать матрицы ошибок
    print(cm)
    
    # Добавление значений в клетки матрицы для лучшего восприятия
    thresh = cm.max() / 2.  # Порог для изменения цвета текста
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()  # Подгонка графика
    plt.ylabel("True label")  # Подпись для оси Y (истинные метки)
    plt.xlabel("Predicted label")  # Подпись для оси X (предсказанные метки)
    plt.show()  # Отображение графика

# Печатаем индексы классов для понимания, какие метки к каким индексам принадлежат
print(test_batches.class_indices)

# Метки классов для отображения в матрице ошибок
cm_plot_labels = ['cat', 'dog']

# Визуализация матрицы ошибок
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')
