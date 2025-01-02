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
warnings.simplefilter(action='ignore', category=FutureWarning)

# physical_devices = tf.config.experimental.list_physical_devices("GPU")
# print('Num GPUs Available')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

os.chdir('recognise-pets/pet-images')
if os.path.isdir("train/dog") is False:
    os.makedirs("train/dog")
    os.makedirs("train/cat")
    os.makedirs("valid/dog")
    os.makedirs("valid/cat")
    os.makedirs("test/dog")
    os.makedirs("test/cat")
        
    cat_files ='train/cat*'
    dog_files = 'train/dog*'
    
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

os.chdir('../../recognise-pets')
        
train_path = 'pet-images/train'        
valid_path = 'pet-images/valid'        
test_path = 'pet-images/test'

train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(directory=train_path, 
                                                                                                                            target_size=(224,224), 
                                                                                                                            classes=['cat','dog'], 
                                                                                                                            batch_size=10)

valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(directory=valid_path,
                                                                                                                            target_size=(224,224), 
                                                                                                                            classes=['cat','dog'], 
                                                                                                                            batch_size=10)

test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(directory=test_path,
                                                                                                                           target_size=(224,224), 
                                                                                                                           classes=['cat','dog'], 
                                                                                                                           batch_size=10, 
                                                                                                                           shuffle=False)

assert train_batches.n == 1000
assert valid_batches.n == 200
assert test_batches.n == 100
assert train_batches.num_classes == valid_batches.num_classes == test_batches.num_classes == 2

imgs, labels = next(train_batches)


def plotImages(images_arr):
    fig, axes = plt.subplots(1, 10, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
    
# plotImages(imgs)
# print(labels)

model = Sequential([
    Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(224,224,3)),
    MaxPool2D(pool_size=(2, 2), strides=2),
    Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
    MaxPool2D(pool_size=(2, 2), strides=2),
    Flatten(),
    Dense(units=2, activation='softmax'),
])

model.summary()

model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x=train_batches, validation_data=valid_batches, epochs=10, verbose=2)

if os.path.isfile('./models/pets_model.h5') is False:
    model.save('./models/pets_model.h5')
