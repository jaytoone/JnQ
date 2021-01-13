from keras.models import Model
import keras.layers as layers


def Basic_Model(input_shape, num_classes):
    # first input model
    visible = layers.Input(shape=input_shape, name='input')

    net = layers.Dense(64, activation='relu')(visible)
    # net = layers.Dropout(0.2)(net)

    # net = layers.LSTM(5, return_sequences=True)(net)
    net = layers.Dense(32, activation='relu')(net)
    net = layers.Dropout(0.2)(net)

    net = layers.Flatten()(net)
    net = layers.Dense(64)(net)
    net = layers.LeakyReLU()(net)
    net = layers.Dense(num_classes, activation='softmax')(net)

    # create model
    model = Model(inputs=visible, outputs=net)
    # summary layers
    # print(model.summary())

    return model
