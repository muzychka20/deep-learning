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

# Списки для хранения данных обучающего набора
train_labels = []  # метки классов
train_samples = []  # образцы (возраст)

# Генерация данных для обучающего набора
for i in range(50):
    # Генерация случайных значений для более молодых (13-64 года)
    random_younger = randint(13, 64)
    train_samples.append(random_younger)
    train_labels.append(1)  # 1 - более молодой
    
    # Генерация случайных значений для более старших (65-100 лет)
    random_older = randint(65, 100)
    train_samples.append(random_older)
    train_labels.append(0)  # 0 - более старший
    
# Генерация дополнительных данных (тысяча примеров)
for i in range(1000):
    random_younger = randint(13, 64)
    train_samples.append(random_younger)
    train_labels.append(0)  # 0 - более старший
    
    random_older = randint(65, 100)
    train_samples.append(random_older)
    train_labels.append(1)  # 1 - более молодой

# Преобразуем списки в массивы numpy
train_labels = np.array(train_labels)
train_samples = np.array(train_samples)

# Перемешиваем данные
train_labels, train_samples = shuffle(train_labels, train_samples)

# Масштабируем данные с использованием MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_train_samples = scaler.fit_transform(train_samples.reshape(-1, 1))

# Создание модели
model = Sequential([
    Dense(units=16, input_shape=(1,), activation='relu'),  # Слой с 16 нейронами и функцией активации ReLU
    Dense(units=32, activation='relu'),  # Слой с 32 нейронами и функцией активации ReLU
    Dense(units=2, activation='softmax')  # Выходной слой с 2 нейронами (для двух классов) и функцией активации Softmax
])

# Вывод структуры модели
model.summary()

# Компиляция модели
model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Обучение модели
model.fit(x=scaled_train_samples, y=train_labels, batch_size=10, validation_split=0.1, epochs=30, shuffle=True, verbose=2)

# Тестирование модели
test_labels = []  # метки классов для тестового набора
test_samples = []  # образцы для тестового набора

# Генерация данных для тестового набора
for i in range(10):
    random_younger = randint(13, 64)
    test_samples.append(random_younger)
    test_labels.append(1)  # 1 - более молодой
    
    random_older = randint(65, 100)
    test_samples.append(random_older)
    test_labels.append(0)  # 0 - более старший
    
# Генерация дополнительных данных для теста (200 примеров)
for i in range(200):
    random_younger = randint(13, 64)
    test_samples.append(random_younger)
    test_labels.append(0)  # 0 - более старший
    
    random_older = randint(65, 100)
    test_samples.append(random_older)
    test_labels.append(1)  # 1 - более молодой

# Преобразуем списки в массивы numpy
test_labels = np.array(test_labels)
test_samples = np.array(test_samples)

# Перемешиваем тестовые данные
test_labels, test_samples = shuffle(test_labels, test_samples)

# Масштабируем тестовые данные
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_test_samples = scaler.fit_transform(test_samples.reshape(-1, 1))

# Получение предсказаний от модели
predictions = model.predict(x=scaled_test_samples, batch_size=10, verbose=0)

# Вывод предсказаний
for i in predictions:
    print(i)

# Округление предсказаний для получения класса (0 или 1)
rounded_predictions = np.argmax(predictions, axis=1)

# Вывод округленных предсказаний
for i in rounded_predictions:
    print(i)

# Матрица путаницы (Confusion Matrix)
cm = confusion_matrix(y_true=test_labels, y_pred=rounded_predictions)

# Функция для отображения матрицы путаницы
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    
    print(cm)
    
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.show()

# Метки классов для матрицы путаницы
cm_plot_labels = ["no side effects", "had_side_effects"]    
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')

# Сохранение модели и весов
if os.path.isfile('./models/medical_trial_model.h5') is False:
    model.save('./models/medical_trial_model.h5')

if os.path.isfile('./models/medical_trial_model_weights.h5') is False:    
    model.save_weights('./models/medical_trial_model_weights.h5')
