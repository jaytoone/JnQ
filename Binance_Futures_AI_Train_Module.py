# import pandas as pd
# import numpy as np
# from matplotlib import pyplot as plt
# import scipy.misc
# import itertools
import random
# from keras.utils import plot_model
from easydict import EasyDict
from binance_futures_modules import *
from Funcs_Indicator import *
import time
import pathlib
from datetime import datetime


from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split
from Binance_Futures_AI_NN_models import *
# from Funcs_For_Trade import *
# from keras_self_attention import SeqSelfAttention
from keras.models import load_model

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'

import warnings
warnings.filterwarnings(action='ignore')

# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())
# quit()

import gc
import tensorflow as tf

# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)
# 연산에 사용한 디바이스 정보 출력 설정
# tf.debugging.set_log_device_placement(True)

tf_config = tf.compat.v1.ConfigProto()
tf_config.gpu_options.allow_growth = True
# tf_config.gpu_options.per_process_gpu_memory_fraction = 0.7
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=tf_config))

#           GPU Set         #
tf.device('/device:XLA_GPU:0')

# Data Class Weight
from sklearn.utils import class_weight
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard


seed = 1218
np.random.seed(seed)
random.seed(seed)

batch_size = 42
verbose = 0

test_num = 0
feature_i = 0

model_init = True
model_renew = False
while 1:

    #            Configuration             #
    with open('binance_futures_bot_config.json', 'r') as cfg:
        config = EasyDict(json.load(cfg))

    fundamental = config.FUNDMTL
    ai = config.AI

    model_range = [ai.model_num]

    for model_num in model_range:

        time.sleep(.5)

        #           Dataset Name            #
        data_name = pathlib.Path('Made_X/Made_X %s.npy' % model_num)
        data_time = data_name.stat().st_mtime

        if model_init:
            model_init = False
            prev_data_time = data_time
            model_renew = True
        else:
            #           Check Renewed DataSet            #
            if prev_data_time != data_time:
                prev_data_time = data_time
                model_renew = True
            else:
                model_renew = False

        if model_renew:
            model_renew = False

            try:

                Made_X = np.load('Made_X/Made_X %s.npy' % model_num)
                Made_Y = np.load('Made_X/Made_Y %s.npy' % model_num).reshape(-1, 1)

                Made_Y = Made_Y.reshape(-1, 1)

                #          Data Preprocessing       #
                nan_row = np.argwhere(np.isnan(Made_Y))
                Made_X2 = np.delete(Made_X, nan_row, axis=0)
                Made_Y2 = np.delete(Made_Y, nan_row, axis=0)

                #         Feature Selection      #
                print('Made_X.shape :', Made_X.shape)
                Made_X3 = Made_X2[:, :, :4]

                row = Made_X3.shape[1]
                col = Made_X3.shape[2]

                X_train, X_val, Y_train, Y_val = train_test_split(Made_X3, Made_Y2, test_size=0.3,
                                                                  shuffle=False)

                print('X_train.shape :', X_train.shape)
                print('X_val.shape :', X_val.shape)

                label = Y_train.reshape(-1, )
                # print(label.shape)
                num_classes = len(np.unique(label))

                Y_train = np_utils.to_categorical(Y_train, num_classes)
                Y_val = np_utils.to_categorical(Y_val, num_classes)

                print('Y_train.shape :', Y_train.shape)
                print('Y_val.shape :', Y_val.shape)

                #         Get Class_Weights (--> should be considered only for TrainSet)       #
                class_weights = class_weight.compute_class_weight('balanced',
                                                                  classes=np.unique(label),
                                                                  y=label)
                class_weights = dict(enumerate(class_weights))
                print(class_weights)
                # quit()

                model = Basic_Model((row, col), num_classes=num_classes)
                opt = Adam(lr=0.0001, decay=1e-6)

                model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

                filepath = "model/rapid_ascending %s_%s_%s_futures_rnn.hdf5" % (model_num, test_num, feature_i)
                checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=verbose, save_best_only=True, mode='auto')
                checkpoint2 = TensorBoard(log_dir='Tensorboard_graph',
                                          histogram_freq=0,
                                          write_graph=True,
                                          write_images=True)
                checkpoint3 = EarlyStopping(monitor='val_accuracy', patience=100)
                callbacks_list = [checkpoint, checkpoint3]

                # keras.callbacks.Callback 로 부터 log 를 받아와 history log 를 작성할 수 있다.

                # we iterate 200 times over the entire training set
                num_epochs = 1000
                history = model.fit(X_train, Y_train,
                                    steps_per_epoch=int(len(X_train) / batch_size),
                                    epochs=num_epochs,
                                    verbose=verbose,
                                    callbacks=callbacks_list,
                                    class_weight=class_weights,
                                    validation_data=(X_val, Y_val),
                                    validation_steps=int(len(X_val) / batch_size),
                                    shuffle=False)

                print('train completed')

                #       Get thr_precision       #
                best_model = load_model(filepath)

                val_y_pred_ = model.predict(X_val, verbose=verbose)

                for target_index in [1, 2]:
                    target_score = val_y_pred_[:, [target_index]]
                    t_te = np.argmax(Y_val, axis=1)
                    t_te = np.where(t_te == target_index, 1, 0)

                    precision, _, threshold = precision_recall_curve(t_te, target_score)
                    precision = precision[:-1]

                    precision_arg = np.argwhere(precision >= ai.target_precision)
                    if len(precision_arg) != 0:
                        thr_precision = float(threshold[np.min(precision_arg)])
                    else:
                        thr_precision = None

                    if target_index == 1:
                        config.AI.long_thr_precision = thr_precision
                        print('long_thr_precision :', thr_precision, end=' ')
                    else:
                        config.AI.short_thr_precision = thr_precision
                        print('short_thr_precision :', thr_precision)

                with open('binance_futures_bot_config.json', 'w') as cfg:
                    json.dump(config, cfg)

                print('time :', datetime.now())

                #       Release GPU Memory      #
                del model
                del best_model
                del history
                # tf.compat.v1.keras.backend.clear_session()
                # sess.close()
                gc.collect()

            except Exception as e:
                print('Error in Train Module :', e)





