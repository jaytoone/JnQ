import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import pyplot
import scipy.misc
from math import sqrt
import itertools
from IPython.display import display
from PIL import Image

input_data_length = 54
model_num = 27

Made_X = np.load('Made_X/Made_X %s_%s.npy' % (input_data_length, model_num))
Made_Y = np.load('Made_X/Made_Y %s_%s.npy' % (input_data_length, model_num))

print(Made_X.shape)
print(Made_Y.shape)
print(np.sum(Made_Y))

# pyplot.figure(figsize=(10, 5))
# pyplot.plot(Made_Y)
# pyplot.show()

row = Made_X.shape[1]
col = Made_X.shape[2]

total_len = len(Made_X)
train_len = int(total_len * 0.7)
val_len = int(total_len * 0.15)
test_len = total_len - (train_len + val_len)

X_train = Made_X[:train_len].astype('float32').reshape(-1, input_data_length, col, 1)
X_val = Made_X[train_len:train_len + val_len].astype('float32').reshape(-1, input_data_length, col, 1)
X_test = Made_X[train_len + val_len:].astype('float32').reshape(-1, input_data_length, col, 1)

print(X_train.shape)
print(X_val.shape)
print(X_test.shape)

from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator

Y_train = Made_Y[:train_len].astype('float32')
Y_val = Made_Y[train_len:train_len + val_len].astype('float32')
Y_test = Made_Y[train_len + val_len:].astype('float32')
num_classes = 3
Y_train = np_utils.to_categorical(Y_train, num_classes)
Y_val = np_utils.to_categorical(Y_val, num_classes)
Y_test = np_utils.to_categorical(Y_test, num_classes)
print(Y_train.shape)
print(Y_val.shape)
print(Y_test.shape)

datagen = ImageDataGenerator(
    rotation_range=60,
    horizontal_flip=True,
    width_shift_range=0.6,
    height_shift_range=0.6,
    fill_mode='nearest'
)

testgen = ImageDataGenerator(
)
datagen.fit(X_train)
batch_size = 128

# for X_batch, _ in datagen.flow(X_train, Y_train, batch_size=9):
#     for i in range(0, 9):
#         pyplot.axis('off')
#         pyplot.subplot(330 + 1 + i)
#         pyplot.imshow(X_batch[i].reshape(input_data_length, col), cmap=pyplot.get_cmap('gray'))
#     pyplot.axis('off')
#     pyplot.show()
#     break

#       dataset 분리      #       다차원 배열을 함수의 인자로 받는 방법을 모릅니다..    #
#       list 순서 : price, vol, sto, macd     #
def data_split_flow(dataX, dataY):
    split_data = [dataX[:, :, :4], dataX[:, :, [4, -5]], dataX[:, :, [-6, -4]], dataX[:, :, -3:]]

    flow_list = [] * len(split_data)
    for i in range(len(split_data)):
        datagen.fit(split_data[i])
        flow_data = datagen.flow(split_data[i], dataY, batch_size=batch_size)
        flow_list.append(flow_data)

    return flow_list


train_flow_list = data_split_flow(X_train, Y_train)
val_flow_list = data_split_flow(X_val, Y_val)
test_flow_list = data_split_flow(X_test, Y_test)


from keras.utils import plot_model
import keras.backend as K
from keras.models import Model, Sequential
from keras.layers import LSTM, TimeDistributed, Input, Dense, Flatten, Dropout, BatchNormalization, Conv1D
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D, MaxPooling1D
from keras.layers.merge import concatenate
from keras.optimizers import Adam, SGD
from keras.regularizers import l1, l2
from sklearn.metrics import confusion_matrix


def ohlc_Model(input_shape=(input_data_length, 4, 1)):
    # first input model
    visible = Input(shape=input_shape, name='input')
    conv1_fit = 100
    conv2_fit = 100
    conv3_fit = 128
    # conv4_fit = 256
    # conv5_fit = 256

    # the 1-st block
    conv1_1 = Conv2D(conv1_fit, kernel_size=3, activation='relu', padding='same', name='conv1_1')(visible)
    conv1_1 = BatchNormalization()(conv1_1)
    # conv1_2 = Conv2D(conv1_fit, kernel_size=3, activation='relu', padding='same', name = 'conv1_2')(conv1_1)
    # conv1_2 = BatchNormalization()(conv1_2)
    pool1_1 = MaxPooling2D(pool_size=(2, 2), name='pool1_1')(conv1_1)
    # drop1_1 = Dropout(0.3, name = 'drop1_1')(pool1_1)

    # the 2-nd block
    conv2_1 = Conv2D(conv2_fit, kernel_size=3, activation='relu', padding='same', name='conv2_1')(pool1_1)
    conv2_1 = BatchNormalization()(conv2_1)
    # conv2_2 = Conv2D(conv2_fit, kernel_size=3, activation='relu', padding='same', name = 'conv2_2')(conv2_1)
    # conv2_2 = BatchNormalization()(conv2_2)
    # conv2_3 = Conv2D(conv2_fit, kernel_size=3, activation='relu', padding='same', name = 'conv2_3')(conv2_2)
    # conv2_3 = BatchNormalization()(conv2_3)
    pool2_1 = MaxPooling2D(pool_size=(2, 2), name='pool2_1')(conv2_1)
    drop2_1 = Dropout(0.3, name='drop2_1')(pool2_1)

    # Flatten and output
    flatten = Flatten(name='flatten')(drop2_1)
    dense = Dense(100, activation='relu', name='dense')(flatten)
    output = Dense(num_classes, activation='softmax', name='output')(dense)

    # create model
    model = Model(inputs=visible, outputs=output)
    # summary layers
    print(model.summary())

    return model


def rest_Model(input_shape=(input_data_length, 2, 1)):
    # first input model
    visible = Input(shape=input_shape, name='input')
    conv1_fit = 100
    conv2_fit = 100
    conv3_fit = 128
    # conv4_fit = 256
    # conv5_fit = 256

    # the 1-st block
    conv1_1 = Conv2D(conv1_fit, kernel_size=3, activation='relu', padding='same', name='conv1_1')(visible)
    conv1_1 = BatchNormalization()(conv1_1)
    # conv1_2 = Conv2D(conv1_fit, kernel_size=3, activation='relu', padding='same', name = 'conv1_2')(conv1_1)
    # conv1_2 = BatchNormalization()(conv1_2)
    pool1_1 = MaxPooling2D(pool_size=(2, 2), name='pool1_1')(conv1_1)
    drop1_1 = Dropout(0.3, name = 'drop1_1')(pool1_1)

    # the 2-nd block
    # conv2_1 = Conv2D(conv2_fit, kernel_size=3, activation='relu', padding='same', name='conv2_1')(pool1_1)
    # conv2_1 = BatchNormalization()(conv2_1)
    # # conv2_2 = Conv2D(conv2_fit, kernel_size=3, activation='relu', padding='same', name = 'conv2_2')(conv2_1)
    # # conv2_2 = BatchNormalization()(conv2_2)
    # # conv2_3 = Conv2D(conv2_fit, kernel_size=3, activation='relu', padding='same', name = 'conv2_3')(conv2_2)
    # # conv2_3 = BatchNormalization()(conv2_3)
    # pool2_1 = MaxPooling2D(pool_size=(2, 2), name='pool2_1')(conv2_1)
    # drop2_1 = Dropout(0.3, name='drop2_1')(pool2_1)

    # Flatten and output
    flatten = Flatten(name='flatten')(drop1_1)
    dense = Dense(100, activation='relu', name='dense')(flatten)
    output = Dense(num_classes, activation='softmax', name='output')(dense)

    # create model
    model = Model(inputs=visible, outputs=output)
    # summary layers
    print(model.summary())

    return model


model = rest_Model()
opt = Adam(lr=0.0001, decay=1e-6)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# callbacks log 를 저장시키는 방법 예시
from keras.callbacks import Callback
import pickle


class Checkpoint_History(Callback):

    def on_train_begin(self, logs={}):
        self.loss = []
        self.val_loss = []
        self.acc = []
        self.val_acc = []

    def on_batch_end(self, batch, logs={}):
        self.loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))
        self.acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))


from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

filepath = "model/rapid_ascending %s_%s.hdf5" % (input_data_length, model_num)
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
checkpoint2 = TensorBoard(log_dir='Tensorboard_graph',
                          histogram_freq=0,
                          write_graph=True,
                          write_images=True)
checkpoint3 = EarlyStopping(monitor='val_loss', patience=15)
callbacks_list = [checkpoint, checkpoint2, checkpoint3]

# keras.callbacks.Callback 로 부터 log 를 받아와 history log 를 작성할 수 있다.

# we iterate 200 times over the entire training set
num_epochs = 100
history = model.fit_generator(train_flow_list[0],
                              steps_per_epoch=len(X_train) / batch_size,
                              epochs=num_epochs,
                              verbose=2,
                              callbacks=callbacks_list,
                              validation_data=val_flow_list[0],
                              validation_steps=len(X_val) / batch_size,
                              shuffle=False)
