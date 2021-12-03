# import keras
# import tensorflow as tf


# from keras.utils import plot_model
# import keras.backend as K
from keras.models import Model, Sequential
import keras.layers as layers

num_classes = 2

def msg_model(input_shape):
    # first input model
    visible = layers.Input(shape=input_shape, name='input')

    net = layers.Conv2D(32, kernel_size=3, padding='same', kernel_initializer='he_normal')(visible)
    # net = layers.Conv2D(256, kernel_size=3, padding='same', kernel_initializer='he_normal')(visible)
    # net = layers.BatchNormalization()(net)
    net = layers.LeakyReLU()(net)

    net = layers.Conv2D(64, kernel_size=3, padding='same', kernel_initializer='he_normal')(net)
    # net = layers.Conv2D(128, kernel_size=3, padding='same', kernel_initializer='he_normal')(net)
    # net = layers.BatchNormalization()(net)
    # net = layers.Activation('relu')(net)
    net = layers.LeakyReLU()(net)
    # net = layers.MaxPool2D(pool_size=2)(net)
    # net = layers.AveragePooling2D(padding='same')(net)

    shortcut_1 = net

    # net = layers.Conv2D(64, kernel_size=3, padding='same', kernel_initializer='he_normal')(net)
    net = layers.Conv2D(128, kernel_size=3, padding='same', kernel_initializer='he_normal')(net)
    net = layers.LeakyReLU()(net)

    net = layers.Conv2D(256, kernel_size=3, padding='same', kernel_initializer='he_normal')(net)
    # net = layers.BatchNormalization()(net)
    # net = layers.Activation('relu')(net)
    net = layers.LeakyReLU()(net)
    # net = layers.MaxPool2D(pool_size=2)(net)

    shortcut_2 = net

    #     net = layers.Conv2D(256, kernel_size=3, padding='same')(net)
    #     # net = layers.Activation('relu')(net)
    #     net = layers.LeakyReLU()(net)
    #     net = layers.MaxPool2D(pool_size=2)(net)

    #     shortcut_3 = net

    #     net = layers.Conv2D(128, kernel_size=1, padding='same')(net)
    #     # net = layers.Activation('relu')(net)
    #     net = layers.LeakyReLU()(net)
    #     net = layers.MaxPool2D(pool_size=2)(net)

    net = layers.Flatten()(net)
    net = layers.Dense(128)(net)
    net = layers.LeakyReLU()(net)

    net = layers.Dense(64)(net)
    net = layers.LeakyReLU()(net)

    net = layers.Dense(num_classes, activation='softmax')(net)

    # create model
    model = Model(inputs=visible, outputs=net)
    # summary layers
    # print(model.summary())

    return model