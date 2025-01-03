import matplotlib.pyplot as plt
import numpy as np
import os
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator, save_img

# Функция для отображения изображений
def plotImages(images_arr):
    # Создаем фигуру с 1 строкой и 10 столбцами для отображения изображений
    fig, axes = plt.subplots(1, 10, figsize=(20,20))  
    axes = axes.flatten()  # Выпрямляем массив подграфиков для удобства
    for img, ax in zip(images_arr, axes):  # Перебираем изображения и оси
        ax.imshow(img)  # Отображаем изображение
        ax.axis('off')  # Отключаем оси (чтобы не отображались метки)
    plt.tight_layout()  # Автоматическое выравнивание подграфиков
    plt.show()  # Показываем изображение

# Меняем рабочую директорию на "data-augmentation" для работы с данными
os.chdir("data-augmentation")

# Создаем генератор изображений для аугментации
gen = ImageDataGenerator(
    rotation_range=10,       # Случайный поворот на 10 градусов
    width_shift_range=0.1,   # Сдвиг изображения по горизонтали на 10% от ширины
    height_shift_range=0.1,  # Сдвиг изображения по вертикали на 10% от высоты
    shear_range=0.15,        # Сдвиг изображения с углом наклона 15%
    zoom_range=0.1,          # Увеличение или уменьшение изображения на 10%
    channel_shift_range=10., # Изменение цветовых каналов на 10
    horizontal_flip=True     # Горизонтальное отражение изображения
)

# Указываем путь к изображению для аугментации
image_path = "img/cat.jpg"

# Загружаем изображение и расширяем его размерность для использования в генераторе
image = np.expand_dims(plt.imread(image_path), 0)

# Генерируем аугментированные изображения
aug_iter = gen.flow(image)

# Создаем папку для сохранения аугментированных изображений
output_dir = 'augmented_images'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Получаем 10 аугментированных изображений и сохраняем их в список
aug_images = []
for i in range(10):
    aug_image = next(aug_iter)[0].astype(np.uint8)  # Генерация нового изображения
    aug_images.append(aug_image)  # Добавляем изображение в список
    # Сохраняем изображение в формате PNG
    save_img(f"{output_dir}/aug_image_{i+1}.png", aug_image)

# Отображаем аугментированные изображения с помощью функции plotImages
plotImages(aug_images)
