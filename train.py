import os
import random
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dropout, Dense, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from cv2 import imread

def display_imgs_from_path(path='', rows = 1, cols = 1):
    fig = plt.figure(figsize=(8, 5))
    for i , img_name in enumerate(random.sample(os.listdir(path), rows * cols)):
        img = imread(path + '/' + img_name)
        fig.add_subplot(rows, cols, i+1)
        plt.imshow(img)
        plt.axis('off')
        plt.title(img_name[:8])

# Set image dimensions and batch size
WIDTH = 128
HEIGHT = 128
IMG_SIZE = (WIDTH, HEIGHT)
BATCH = 32

def build_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(WIDTH, HEIGHT, 3)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Dense(1, activation='sigmoid'))
    
    model.summary()

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train_model(model, callbacks, epochs = 1):
    train_datagen = ImageDataGenerator(rescale=1. / 255, rotation_range=15, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    validation_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory('working/train', 
                                                        target_size=(128, 128), 
                                                        classes=['Cats', 'Dogs'], 
                                                        class_mode='binary', 
                                                        batch_size=32)

    validation_generator = validation_datagen.flow_from_directory('working/validation', 
                                                                target_size=(128, 128), 
                                                                classes=['Cats', 'Dogs'], 
                                                                class_mode='binary', 
                                                                batch_size=32)

    EPOCHS = epochs
    
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH,
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH,
        callbacks=callbacks
    )
    
    print(f"Training done")

    return history

def save_model(model, model_path):
    model.save(model_path)

def plot_history(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    plt.plot(history.history['lr'])
    plt.title('Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.legend(['Learning Rate'], loc='upper left')
    plt.show()
