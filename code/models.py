# Suppress some of the "standard" tensorflow output
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D, Activation, Flatten, Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model


from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

# Prevent tensorflow from using too much GPU memory
config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
config.gpu_options.allow_growth = True
session = InteractiveSession(config = config)

from code.misc import pick_samples_1D

def LeNet():
    """
    Create a LeNet inspired model
    
    Inputs
    ----------
        N/A
        
    Outputs
    -------
    model : tf.keras.Sequential
        LeNet model
    """

    model = Sequential()

    # Conv 32x32x1 -> 28x28x6.
    model.add(Conv2D(filters = 6, kernel_size = (5, 5), strides = (1, 1), padding = 'valid', 
                            data_format = 'channels_last', input_shape = (32, 32, 1)))
    model.add(Activation("relu"))

    # Maxpool 28x28x6 -> 14x14x6
    model.add(MaxPooling2D((2, 2)))

    # Conv 14x14x6 -> 10x10x16
    model.add(Conv2D(filters = 16, kernel_size = (5, 5), padding = 'valid'))
    model.add(Activation("relu"))

    # Maxpool 10x10x16 -> 5x5x16
    model.add(MaxPooling2D(pool_size = (2, 2)))

    # Flatten 5x5x16 -> 400
    model.add(Flatten())

    # FC Layer: 400 -> 120
    model.add(Dense(120))
    model.add(Activation("relu"))

    # FC Layer: 120 -> 84
    model.add(Dense(84))
    model.add(Activation("relu"))

    # Dropout
    model.add(Dropout(0.2))
    
    # FC Layer: layer 84-> 43
    model.add(Dense(43))
    model.add(Activation("softmax"))

    return model


def VGG16():
    """
    Create a VGG16 inspired model
    
    Inputs
    ----------
        N/A
        
    Outputs
    -------
    model : tf.keras.Sequential
        Custom model
    """

    model = Sequential()
    norm_axis = -1
    
    # Conv Layer: 32x32x1 -> 32x32x8.
    model.add(Conv2D(filters = 8, kernel_size = (5,5), padding = "same", input_shape = (32, 32, 1)))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis = norm_axis))

    # Layer: 32x32x1 -> 16x16x8.
    model.add(MaxPooling2D(pool_size = (2, 2)))
    
    # Conv Layer: 16x16x8 -> 16x16x16
    model.add(Conv2D(filters = 16, kernel_size = (3,3), padding = "same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis = norm_axis))
    model.add(Conv2D(filters = 16, kernel_size = (3,3), padding = "same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis = norm_axis))

    # Layer: 16x16x8 -> 8x8x16
    model.add(MaxPooling2D(pool_size = (2,2)))
    
    # Layer: 8x8x16 -> 8x8x32
    model.add(Conv2D(filters = 32, kernel_size = (3,3), padding = "same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis = norm_axis))
    model.add(Conv2D(filters = 32, kernel_size = (3,3), padding = "same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis = norm_axis))
    model.add(Conv2D(filters = 32, kernel_size = (3,3), padding = "same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis = norm_axis))

    # Layer: 8x8x32 -> 4x4x32
    model.add(MaxPooling2D(pool_size = (2,2)))
    
    # Flatten Layer: 4x4x32 -> 512
    model.add(Flatten())

    # FC Layer: 512 -> 128
    model.add(Dense(128))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.7))
    
    # Flatten Layer:: 128 -> 128
    model.add(Flatten())

    # FC Layer: 128 -> 128
    model.add(Dense(128))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    
    # FC Layer: 128 -> 43
    model.add(Dense(43))
    model.add(Activation("softmax"))
    
    return model


def train_model(file_path_model, X_train, y_train, X_valid, y_valid, lrn_rate, n_max_epochs, batch_size):

    if file_path_model.find("LeNet") != -1:
        print("INFO: code.models.train_model(): Model type is LeNet!")
        model = LeNet()
    elif file_path_model.find("VGG16") !=-1:
        print("INFO: code.models.train_model(): Model type is VGG16!")
        model = VGG16()
    else:
        return None, None


    optimizer = Adam(learning_rate = lrn_rate)

    model.compile(optimizer = optimizer, loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

    early_stopping = EarlyStopping(monitor = 'val_accuracy', 
                                   patience = 5, min_delta = 0.001, 
                                   mode = 'max', restore_best_weights = True)

    history = model.fit(X_train, y_train, batch_size = batch_size, epochs = n_max_epochs, 
                        validation_data = (X_valid, y_valid), callbacks = [early_stopping])

    return model, history

def save_model(file_path, model):

    model.save(file_path)

def evaluate_model(model, X_test, y_test, batch_size):

    model.evaluate(X_test, y_test, batch_size = batch_size)

def predict_signs(model, X, y_metadata, indices, top_k = 5):

    predictions = model.predict(X)

    signs = []

    for i, index in enumerate(indices):
        prediction = predictions[index]

        top_k_predictions  = prediction.argsort()[-top_k:][::-1]
        top_k_probabilities = np.sort(prediction)[-top_k:][::-1]

        y_metadata_samples = pick_samples_1D(y_metadata, top_k_predictions, dtype = 'U25')

        sign = ''
        for k, prob in enumerate(top_k_probabilities):
            sign += y_metadata_samples[k] + " " + "P:" + str(prob) + "\n"

        signs.append(sign)

    return signs

