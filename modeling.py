
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras.wrappers.scikit_learn import KerasClassifier

im_size = 28
num_classes = 10

def create_model(dropout=0.5, lr=0.001, num_filters=64, num_layers=3 , use_batch_norm = False):
    model = Sequential()
    for i in range(num_layers):
        if i == 0:
            model.add(Conv2D(num_filters, (3, 3), input_shape=(im_size, im_size, 1)))
        else:
            model.add(Conv2D(num_filters, (3, 3)))
        if use_batch_norm:
            model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(128))
    model.add(Activation('relu'))
    if dropout > 0:
        model.add(Dropout(dropout))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=lr), metrics=['accuracy'])
    return model