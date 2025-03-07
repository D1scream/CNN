from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau , CSVLogger
from init import preprocess_data
from train import build_model, train_model, save_model, plot_history
from tensorflow.keras.models import load_model

tf.config.threading.set_intra_op_parallelism_threads(3)
tf.config.threading.set_inter_op_parallelism_threads(3)

#preprocess_data()

#test_datagen = ImageDataGenerator(rescale=1. / 255)

#test_generator = test_datagen.flow_from_directory('working/test', target_size=(128, 128), classes=['Cats', 'Dogs'], class_mode='binary', batch_size=32)

#model = build_model()
#history = train_model(model, callbacks, epochs=5)
#save_model(model, 'working/models/model50epochs.keras')
#plot_history(history)
'''
callbacks = [
    ReduceLROnPlateau(monitor='val_accuracy', patience=4, verbose=1, factor=0.75, min_lr=0.00001),
    EarlyStopping(patience=10),
    ModelCheckpoint(filepath='working/models/model_best_acc.keras', monitor='val_accuracy', save_best_only=True, save_freq='epoch'),
    CSVLogger('log.csv', separator=",", append=True)
]
'''
model_path = 'working/models/model50epochs.keras'
model = load_model(model_path)

def predict(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    
    probability = prediction[0][0]
    label = "Dog" if probability > 0.5 else "Cat"
    
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Предсказание: {label}. Probability {max(probability,1-probability):.2f}")
    plt.show()

predict('AnimeCat.jpg')