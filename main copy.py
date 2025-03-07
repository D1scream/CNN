import shutil
import random 
import os
import time
from PIL import Image
import cv2
import numpy as np
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dropout, Dense, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau , CSVLogger
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import tensorflow as tf
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

kukuk

def reset_directory(dir_path):
    """
    Deletes the contents of a directory and then recreates the directory.
    
    Arguments:
    dir_path (str): The path of the directory to reset.
    
    Returns:
    None
    """

    # If the directory is already empty, print a message and return

    if len(os.listdir(dir_path)) == 0 :
        print(dir_path + " is already empty")
        return

    # Print a message and record the starting time
    print("resetting "+ dir_path)

    # Delete the directory and all its contents
    shutil.rmtree(dir_path)

    # Create an empty directory in the same location
    os.makedirs(dir_path)
    print(dir_path + " is now empty")


def copy_clean(src = '', dest ='', ignore = []):
    """
    Copies all the files from the source directory to the destination directory, ignoring the files specified in the ignore list.
    
    Parameters:
    src (str): The path of the source directory.
    dest (str): The path of the destination directory.
    ignore (list): A list of file names to ignore.
    
    Returns:
    None
    """
    print("Copying file from " + src + " to " + dest)
    i = 0
    j = 0
    for filename in (os.listdir(src)):
        i += 1
        if filename not in ignore:
            shutil.copy(src + '/' + filename, dest + '/' + filename)
            j+=1
        if (i % 1000 == 0):
            print(i, end = " ")
        if (i % 10000 == 0):
            print()
        
    print()
    print(j)
    print("Copying {} files finished".format(len(os.listdir(dest))))

def split_data_to_dir(class_str = '',src_path ='', dest_path ='',samples ={}):
  """
  Copies samples (represented by a dict) from one source directory to a destination directory 
  
    Arguments:
        class_str: a string representing the class
        src_path: a string representing the path for the source directory
        dest_path: a string representing the path for the destination directory
        sample: a python Dict representing either test or train or validation samples 

    Returns:
        None

  """

  src_path = src_path + '/' + class_str
  dest_path = dest_path + '/' + class_str
  print(" Sending test samples to  " + dest_path)
  i = 0 # to track haw many imgs are copied
  for filename in samples[class_str]:
    shutil.copy(src_path + '/' + filename, dest_path + '/'+ class_str + "_" + filename)
    i+=1
    if (i % 250 == 0 ):
      print(i, end = " ")
    if (i % 5000 == 0):
      print()

  print("nb of test samples for {} is {}".format(class_str, str(i)))
  print("Sending {} test samples complete ".format(str(i)))

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

dog_train_path = "OriginalDataset/train/dogs"
cat_train_path = "OriginalDataset/train/cats"

dog_test_path = "OriginalDataset/test/dogs"
cat_test_path = "OriginalDataset/test/cats"

original_dataset_path = 'OriginalDataset/train'
clean_dataset_path = 'working/dataset'

test_path = 'working/test'
train_path = 'working/train'
validation_path = 'working/validation'


print("Cat samples : {}".format(len(os.listdir(cat_train_path))))
print("Dog samples : {}".format(len(os.listdir(dog_train_path))))

try:
    os.makedirs('working/dataset')

    os.makedirs('working/dataset/cats')
    os.makedirs('working/dataset/dogs')

    os.makedirs('working/train/cats')
    os.makedirs('working/test/cats')

    os.makedirs('working/train/dogs')
    os.makedirs('working/test/dogs')

    os.makedirs('working/validation/dogs')
    os.makedirs('working/validation/cats')

    os.makedirs('working/data/train')
    os.makedirs('working/data/test')
except FileExistsError :
    print("files already exists")
    pass

reset_directory(clean_dataset_path +'/cats')
copy_clean(src = original_dataset_path + '/cats',
           dest = clean_dataset_path +'/cats')

print(len(os.listdir(clean_dataset_path +'/cats')))
reset_directory(clean_dataset_path +'/dogs')
copy_clean(src = original_dataset_path + '/dogs',
           dest = clean_dataset_path +'/dogs')
print(len(os.listdir(clean_dataset_path +'/dogs')))

def train_validation_split(data_path='', validation_split=0.15):
    data_files = os.listdir(data_path)

    data_size = len(data_files)
    validation_size = int(validation_split * data_size)
    
    validation_sample = random.sample(data_files, validation_size)

    train_sample = list(set(data_files) - set(validation_sample))
    
    print(f'Train size: {len(train_sample)}')
    print(f'Validation size: {len(validation_sample)}')
    
    return train_sample, validation_sample

cat_train_sample, cat_validation_sample = train_validation_split(data_path = clean_dataset_path +'/cats')
dog_train_sample , dog_validation_sample = train_validation_split(data_path = clean_dataset_path +'/dogs')
cat_test_sample = os.listdir(clean_dataset_path +'/cats')
dog_test_sample = os.listdir(clean_dataset_path +'/dogs')

print(f'Cats test size: {len(cat_test_sample)}')
print(f'Dogs test size: {len(dog_test_sample)}')

test_samples = { 'Cats': cat_test_sample,
                 'Dogs': dog_test_sample,
                }
validation_samples = { 'Cats': cat_validation_sample,
                       'Dogs': dog_validation_sample,
                      }
train_samples = { 'Cats': cat_train_sample,
                  'Dogs': dog_train_sample,
                      }

reset_directory(test_path +"/cats")
split_data_to_dir(class_str = "Cats", src_path = clean_dataset_path, dest_path = test_path, samples = test_samples)
print(len(os.listdir(test_path +"/cats")))

reset_directory(train_path +"/cats")
split_data_to_dir(class_str = "Cats", src_path = clean_dataset_path, dest_path = train_path, samples = train_samples)
print(len(os.listdir(train_path +"/cats")))

reset_directory(validation_path +"/cats")
split_data_to_dir(class_str = "Cats", src_path = clean_dataset_path, dest_path = validation_path, samples = validation_samples)
print(len(os.listdir(validation_path  +"/cats")))

reset_directory(test_path +"/dogs")
split_data_to_dir(class_str = "Dogs", src_path = clean_dataset_path, dest_path = test_path, samples = test_samples)
print(len(os.listdir(test_path +"/dogs")))

reset_directory(train_path +"/dogs")
split_data_to_dir(class_str = "Dogs", src_path = clean_dataset_path, dest_path = train_path, samples = train_samples)
print(len(os.listdir(train_path +"/dogs")))

reset_directory(validation_path +"/dogs")
split_data_to_dir(class_str = "Dogs", src_path = clean_dataset_path, dest_path = validation_path, samples = validation_samples)
print(len(os.listdir(validation_path  +"/dogs")))

print("Folders in test " + str(len(os.listdir(test_path))))
print("Samples in test/cats: " + str(len(os.listdir(test_path + "/cats"))))
print("Samples in test/dogs: " + str(len(os.listdir(test_path + "/dogs"))))

print("Folders in validation " + str(len(os.listdir(validation_path))))
print("Samples in validation/cats: " + str(len(os.listdir(validation_path + "/cats"))))
print("Samples in validation/dogs: " + str(len(os.listdir(validation_path + "/dogs"))))

print("Folders in train " + str(len(os.listdir(train_path))))
print("Samples in train/cats: " + str(len(os.listdir(train_path + "/cats"))))
print("Samples in train/dogs: " + str(len(os.listdir(train_path + "/dogs"))))

display_imgs_from_path(path=test_path +'/cats', rows = 1, cols = 5)
display_imgs_from_path(path=train_path +'/cats', rows = 1, cols = 5)
display_imgs_from_path(path=validation_path +'/cats', rows = 1, cols = 5)

display_imgs_from_path(path=test_path +'/dogs', rows = 1, cols = 5)
display_imgs_from_path(path=train_path +'/dogs', rows = 1, cols = 5)
display_imgs_from_path(path=validation_path +'/dogs', rows = 1, cols = 5)


# Set the image dimensions and Batch size
WIDTH = 128
HEIGHT = 128
IMG_SIZE = (WIDTH , HEIGHT)
BATCH = 32

# Create an ImageDataGenerator object for the validation set
validation_datagen = ImageDataGenerator(rescale=1. / 255)
# Create an ImageDataGenerator object for the test set
test_datagen = ImageDataGenerator(rescale=1. / 255)
# Create an ImageDataGenerator object for the training set

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=15,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(train_path, 
                                                    target_size = IMG_SIZE,
                                                    classes=['Cats' , 'Dogs'],
                                                    class_mode='binary',
                                                    batch_size=BATCH,
                                                    #save_to_dir=aug_data_path,
                                                    #save_prefix='aug_',
                                                    #save_format="jpg",
                                                    seed = 1)
print(train_generator.class_indices)
print(train_generator.num_classes)
print(train_generator.samples)

validation_generator = validation_datagen.flow_from_directory(validation_path, 
                                                    target_size = IMG_SIZE,
                                                    classes=['Cats' , 'Dogs'],
                                                    class_mode='binary',
                                                    batch_size=BATCH,
                                                    seed = 1 )
print(validation_generator.class_indices)

test_generator = test_datagen.flow_from_directory(test_path, 
                                                    target_size = IMG_SIZE,
                                                    classes=['Cats' , 'Dogs'],
                                                    class_mode='binary',
                                                    batch_size=BATCH,
                                                    seed = 1 )
print(test_generator.class_indices)


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
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

models_path = '/working/models'
try:
    os.makedirs(models_path)
except FileExistsError :
    print("already exists")
    pass

learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 
                                            patience= 4, 
                                            verbose= 1, 
                                            factor= 0.75, 
                                            min_lr= 0.00001)

early_stopping = EarlyStopping(patience = 10)

"""checkpoint_loss = ModelCheckpoint(filepath = models_path + '/model_best_loss.keras',
    monitor = "val_loss",
    save_best_only = True, save_freq= 'epoch' )"""

checkpoint_acc = ModelCheckpoint(filepath = models_path + '/model_best_acc.keras',
    monitor = "val_accuracy",
    save_best_only = True, save_freq= 'epoch' )

filename='log.csv'
history_logger= CSVLogger(filename, separator=",", append=True)


callbacks = [learning_rate_reduction, early_stopping , #checkpoint_loss , 
             checkpoint_acc, history_logger]

EPOCHS = 1
beg = int(time.time())

history = model.fit(
        train_generator,
        steps_per_epoch = train_generator.samples  // BATCH,
        epochs = EPOCHS,
        validation_data = validation_generator,
        validation_steps = validation_generator.samples // BATCH,
        callbacks = callbacks)

end = int(time.time())
t = end - beg
hrs = t // 3600
mins = (t - 3600 * hrs) // 60
secs = t % 60
print("training took {} hrs -- {} mins -- {} secs".format(hrs,mins,secs))
model.save_weights(models_path + '/first_try.weights.h5')
model.save(models_path + '/first_try.h5')

scores = model.evaluate(test_generator)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(history.history['lr'])
plt.title('Learning rate')
plt.ylabel('learnin rate')
plt.xlabel('epoch')
plt.legend(['learning rate'], loc='upper left')
plt.show()

labels = ["Cat","Dog"]

fig = plt.figure(figsize=(8, 5))

    
for j in range(10):
    i = random.randint(0,1800)
    dog_test_img = cv2.imread(test_path + '/Dog'+'/'+
                              os.listdir(test_path + '/Dog')[i])
    #print(os.listdir(test_path + '/Dog')[5])
    #print(type(dog_test_img))
    #print(dog_test_img.shape)
    dog_test_img = cv2.cvtColor(dog_test_img,cv2.COLOR_BGR2RGB)
    fig.add_subplot(2, 5, j+1)
    plt.imshow(dog_test_img)
    plt.axis('off')
    
    dog_test_img = cv2.resize(dog_test_img,(128,128))
    dog_test_img = np.reshape(dog_test_img,(1,128,128,3))
    #print(dog_test_img.shape)
    
    results = model.predict(dog_test_img,verbose = 0)
    results = np.squeeze(results)
    plt.title(labels[results.astype(int)])

    #print(results.astype(int))
    #print(type(results))