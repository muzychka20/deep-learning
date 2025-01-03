import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import itertools
import os
import shutil
import random
import glob
import matplotlib.pyplot as plt
import warnings

# Игнорирование предупреждений FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)

# Устанавливаем рабочую директорию
os.chdir('recognise-pets/pet-images')

# Проверка наличия нужных директорий и создание их, если нужно
if os.path.isdir("train/dog") is False:
    # Создание директорий для тренировочных, валидационных и тестовых данных
    os.makedirs("train/dog")
    os.makedirs("train/cat")
    os.makedirs("valid/dog")
    os.makedirs("valid/cat")
    os.makedirs("test/dog")
    os.makedirs("test/cat")
    
    # Определяем пути для поиска изображений кошек и собак
    cat_files = 'train/cat*'
    dog_files = 'train/dog*'
    
    # Перемещаем случайные файлы кошек и собак в соответствующие каталоги
    for c in random.sample(glob.glob(cat_files), 500):
        shutil.move(c, 'train/cat')        
    for c in random.sample(glob.glob(dog_files), 500):
        shutil.move(c, 'train/dog')

    for c in random.sample(glob.glob(cat_files), 100):
        shutil.move(c, 'valid/cat')        
    for c in random.sample(glob.glob(dog_files), 100):
        shutil.move(c, 'valid/dog')

    for c in random.sample(glob.glob(cat_files), 50):
        shutil.move(c, 'test/cat')
    for c in random.sample(glob.glob(dog_files), 50):
        shutil.move(c, 'test/dog')

# Возвращаемся в основную директорию проекта
os.chdir('../../recognise-pets')

# Пути к данным
train_path = 'pet-images/train'        
valid_path = 'pet-images/valid'        
test_path = 'pet-images/test'

# Создание генераторов для подачи данных в модель
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

# Проверяем правильность распределения данных
assert train_batches.n == 1000  # 1000 изображений в тренировочной выборке
assert valid_batches.n == 200  # 200 изображений в валидационной выборке
assert test_batches.n == 100  # 100 изображений в тестовой выборке
assert train_batches.num_classes == valid_batches.num_classes == test_batches.num_classes == 2  # 2 класса (кошка, собака)

# Извлекаем 10 изображений и их метки для отображения
imgs, labels = next(train_batches)

# Функция для отображения изображений
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 10, figsize=(20,20))  # Создаем фигуру с 10 подграфиками
    axes = axes.flatten()  # Выпрямляем массив подграфиков для удобства
    for img, ax in zip(images_arr, axes):  # Отображаем изображения
        ax.imshow(img)
        ax.axis('off')  # Отключаем оси
    plt.tight_layout()  # Автоматическое выравнивание подграфиков
    plt.show()

# Выводим изображения
plotImages(imgs)
# Печатаем метки классов для изображений
print(labels)

# Создание модели с использованием сверточных слоев
model = Sequential([
    # Первый сверточный слой (32 фильтра, размер ядра 3x3, активация ReLU)
    Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(224,224,3)),
    # Слой подвыборки (max-pooling) для уменьшения размерности
    MaxPool2D(pool_size=(2, 2), strides=2),
    
    # Второй сверточный слой (64 фильтра, размер ядра 3x3, активация ReLU)
    Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
    # Еще один слой подвыборки
    MaxPool2D(pool_size=(2, 2), strides=2),
    
    # Преобразуем многомерный тензор в одномерный для подачи в полносвязный слой
    Flatten(),
    
    # Полносвязный слой для классификации на 2 класса (кошки и собаки)
    Dense(units=2, activation='softmax'),
])

# Выводим архитектуру модели
model.summary()

# Компиляция модели с оптимизатором Adam, функцией потерь categorical_crossentropy и метрикой accuracy
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Обучение модели на тренировочных данных, валидация на валидационных данных
model.fit(x=train_batches, validation_data=valid_batches, epochs=10, verbose=2)

# Сохранение модели, если она еще не сохранена
if os.path.isfile('./models/pets_model.h5') is False:
    model.save('./models/pets_model.h5')  # Сохраняем модель в файл
