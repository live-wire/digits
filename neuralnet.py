from PIL import Image
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from time import sleep
import scipy.io as sio
import itertools

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

class kerasNN:
    def __init__(self,data):
        self.data = data
    def generator(self):
        return (self.data['train'][0],self.data['train'][1])
    def create(self):
        model = Sequential()

        model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same',
                         activation='relu', input_shape=(15, 15, 1)))
        model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same',
                         activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same',
                         activation='relu'))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same',
                         activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(256, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(10, activation="softmax"))

        # Define the optimizer
        optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
        # Compile the model
        model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
        # Set a learning rate annealer
        learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                                    patience=3,
                                                    verbose=1,
                                                    factor=0.5,
                                                    min_lr=0.00001)

        epochs = 10  # Turn epochs to 30 to get 0.9967 accuracy
        batch_size = 86
        # model.fit(self.data['train'][0].reshape(-1,15,15,1),to_categorical(self.data['train'][1], num_classes = 10),epochs=epochs,batch_size=batch_size,callbacks=[learning_rate_reduction])
        # serialize model to JSON
        # model_json = model.to_json()
        # with open("model.json", "w") as json_file:
        #     json_file.write(model_json)
        # # serialize weights to HDF5
        # model.save_weights("model.h5")
        # print("Saved model to disk")
        # later...

        # load json and create model
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights("model.h5")
        print("Loaded model from disk")
        loaded_model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

        score = loaded_model.evaluate(self.data['test'][0].reshape(-1,15,15,1),to_categorical(self.data['test'][1]))

        # history = model.fit_generator(self.generator(),
        #                               epochs=epochs, validation_data=(self.data['test'][0], self.data['test'][1]),
        #                               verbose=2, steps_per_epoch=1000 // batch_size
        #                               , callbacks=[learning_rate_reduction])

        print(score)

loaded_model_global = None
def load_model():
    global loaded_model_global
    if not loaded_model_global==None:
        return loaded_model_global
    else:
        optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights("model.h5")
        print("Loaded model from disk")
        loaded_model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
        loaded_model_global = loaded_model
        return loaded_model_global


def predictcnn(X):
    loaded_model = load_model()
    predict_cnn = loaded_model.predict(X)
    p = []
    for row in predict_cnn:
        for index, item in enumerate(row):
            if item == np.max(row):
                p.append(index)
                break
    return p

