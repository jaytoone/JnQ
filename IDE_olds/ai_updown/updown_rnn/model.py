from keras.models import Model
import keras.layers as layers

num_classes = 2


def lstm_model(input_shape):
    # first input model
    visible = layers.Input(shape=input_shape, name='input')

    # net = layers.LSTM(32, return_sequences=False)(visible)
    net = layers.LSTM(10, return_sequences=False)(visible)
    net = layers.LeakyReLU()(net)

    net = layers.Dense(64)(net)
    net = layers.LeakyReLU()(net)
    net = layers.BatchNormalization()(net)

    net = layers.Dense(128)(net)
    net = layers.LeakyReLU()(net)

    net = layers.Dense(256)(net)
    net = layers.LeakyReLU()(net)
    net = layers.BatchNormalization()(net)

    net = layers.Dense(128)(net)
    net = layers.LeakyReLU()(net)

    net = layers.Dense(64)(net)
    net = layers.LeakyReLU()(net)

    net = layers.Dropout(0.3)(net)

    net = layers.Dense(num_classes, activation='softmax')(net)

    # create model
    model = Model(inputs=visible, outputs=net)
    # summary layers
    print(model.summary())

    return model