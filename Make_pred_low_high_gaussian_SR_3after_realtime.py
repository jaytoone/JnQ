import os

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import numpy as np
import pandas as pd
from keras.models import load_model
from matplotlib import pyplot as plt
from Make_X_low_high_gaussian_support_resistance_3after import low_high
import mpl_finance as mf
from datetime import datetime
import pybithumb
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def min_max_scaler(x):
    scaled_x = (x - x.min()) / (x.max() - x.min())
    return scaled_x


def max_abs_scaler(x):
    scaled_x = x / abs(x).max()
    return scaled_x


if __name__ == '__main__':

    #           Get TopCoin List         #

    TopCoin = ['DAC', 'EGG', 'CON', 'PLX', 'WTC', 'XVG', 'DASH', 'XLM', 'OMG', 'STRAT']
    # TopCoin = ['STEEM']

    #           PARAMS           #
    input_data_length = 15
    model_num = '127_ohlc'
    # crop_size_low = 200
    # crop_size_high = 100
    # crop_size_sudden_death = 100
    # limit_line_low = 0.97
    # limit_line_high = 0.9
    # limit_line_sudden_death = 0.45
    get_fig = 1

    #       Make folder      #

    try:
        os.mkdir('./Figure_trade/%s_%s/' % (input_data_length, model_num))

    except Exception as e:
        pass

    #       LOAD MODEL      #
    model = load_model('./model/rapid_ascending %s_%s.hdf5' % (input_data_length, model_num))

    for Coin in TopCoin:

        # try:
        dataX, _, chart, _ = low_high(Coin, input_data_length, crop_size=input_data_length)
        X_test = np.array(dataX)
        # print(X_test)
        # quit()

        row = X_test.shape[1]
        col = X_test.shape[2]
        X_test = X_test.astype('float32').reshape(-1, row, col, 1)

        if 'ohlcmacd' in model_num:
            X_test = X_test[:, :, [0, 1, 2, 3, -2]]
        elif 'macd_osc' in model_num:
            X_test = X_test[:, :, [11]]
        elif 'rsimacd' in model_num:
            X_test = X_test[:, :, -5:]
        elif 'ohlccmo' in model_num:
            X_test = X_test[:, :, [0, 1, 2, 3, 5]]
        elif 'ohlc' in model_num:
            X_test = X_test[:, :, [0, 1, 2, 3]]
        elif 'volume' in model_num:
            X_test = X_test[:, :, [0, 1, 2, 3, 4]]
        elif 'cmo' in model_num:
            X_test = X_test[:, :, [5]]
        elif 'obv' in model_num:
            X_test = X_test[:, :, [0, 1, 2, 3, 6]]
        elif 'rsi' in model_num:
            X_test = X_test[:, :, [0, 1, 2, 3, 7]]
        elif 'macd' in model_num:
            X_test = X_test[:, :, -4:]
        # elif 'ohlcmacd' in
        # print(X_test.shape)
        # quit()
        # X_test2, _, closeprice2, _ = low_high(Coin, input_data_length, crop_size=crop_size_high, sudden_death=0.)
        # X_test3, _, closeprice3, _ = low_high(Coin, input_data_length, crop_size=crop_size_sudden_death, sudden_death=0.)
        # # X_test, _ = low_high(Coin, input_data_length, sudden_death=1.)
        # closeprice = np.roll(np.array(list(map(lambda x: x[-1][[1]][0], X_test))), -1)

        if X_test is None:
            continue

        # except Exception as e:
        #     print('Error in getting data from made_x :', e)
        #     continue

        # dataX 에 담겨있는 value 에 [-1] : 바로 이전의 행 x[-1][:].shape = (1, 6)
        # sliced_ohlcv = np.array(list(map(lambda x: x[-1][:], X_test)))
        # print(sliced_ohlcv)
        # quit()

        if len(X_test) != 0:

            Y_pred_ = model.predict(X_test, verbose=1)
            # Y_pred_[:, [-1]] *= 0.9

            # print(Y_pred)
            # quit()
            #           LABELING BY SORTING         #
            n = 10
            Y_pred = np.zeros(len(Y_pred_))
            for label in [0, 1, 2]:
                max_index_list = Y_pred_[:, label].argsort()[::-1][:n]
                for i in max_index_list:
                    Y_pred[i] = label

            #          LABELING BY PERCENTAGE       #
            # max_value = np.max(Y_pred_, axis=0)
            # print(max_value)
            # Y_pred = np.zeros(len(Y_pred_))
            # for i in range(len(Y_pred_)):
            #     if Y_pred_[i][0] > max_value[0] * 0.9:
            #         Y_pred[i] = 0
            #     if Y_pred_[i][1] > max_value[1] * 0.9:
            #     # if Y_pred_[i][1] > 0.8:  # 0.8
            #         Y_pred[i] = 1
            #     if Y_pred_[i][2] > max_value[2] * 0.97:
            #     # if Y_pred_[i][2] > 0.8:  # 0.8
            #         Y_pred[i] = 2

            if get_fig == 1:

                spanlist = []
                spanlist_low = []
                spanlist_high = []
                spanlist_trend_up = list()
                for m in range(len(Y_pred)):
                    if Y_pred[m] == 0.:
                        if m + 1 < len(Y_pred):
                            spanlist.append((m, m + 1))
                        else:
                            spanlist.append((m - 1, m))

                for m in range(len(Y_pred)):
                    if Y_pred[m] == 1.:
                        if m + 1 < len(Y_pred):
                            spanlist_low.append((m, m + 1))
                        else:
                            spanlist_low.append((m - 1, m))
                for m in range(len(Y_pred)):
                    if Y_pred[m] == 2.:
                        if m + 1 < len(Y_pred):
                            spanlist_high.append((m, m + 1))
                        else:
                            spanlist_high.append((m - 1, m))

                osc = chart[:, [-3]]  # >> osc가 아니라 curve_osc의 기울기로 해야함
                for m in range(len(osc)):
                    if osc[m] > 0:
                        if m + 1 < len(osc):
                            spanlist_trend_up.append((m, m + 1))
                        else:
                            spanlist_trend_up.append((m - 1, m))

                plt.figure(figsize=(10, 8))
                # plt.subplot(311)
                # plt.plot(chart[:, [1]], 'gold', label='close')
                # # plt.plot(chart[:, [-4]], '', label='close')
                # # plt.plot(chart[:, [-3]], 'g', label='close')
                # # plt.plot(chart[:, [-2]], 'gold', label='close')
                # plt.legend(loc='upper right')
                # for i in range(len(spanlist)):
                #     plt.axvspan(spanlist[i][0], spanlist[i][1], facecolor='b', alpha=0.7)

                plt.subplot(411)
                plt.plot(min_max_scaler(chart[:, [1]]), 'gold', label='close')
                # plt.plot(chart[:, [-4]], 'r', label='close')
                # plt.plot(chart[:, [-3]], 'r', label='close')
                # plt.plot(chart[:, [-2]], 'gold', label='close')
                plt.legend(loc='upper right')
                for i in range(len(spanlist_low)):
                    plt.axvspan(spanlist_low[i][0], spanlist_low[i][1], facecolor='c', alpha=0.7)
                # for i in range(len(spanlist_trend_up)):
                #     plt.axvspan(spanlist_trend_up[i][0], spanlist_trend_up[i][1], facecolor='b', alpha=0.3)
                plt.subplot(412)
                plt.plot(Y_pred_[:, [1]], 'limegreen', label='Y_pred')
                plt.legend(loc='upper right')

                plt.subplot(413)
                plt.plot(min_max_scaler(chart[:, [1]]), 'gold', label='close')
                # plt.plot(chart[:, [-3]], 'r', label='close')
                # plt.plot(chart[:, [-5]], 'gold', label='close')
                plt.legend(loc='upper right')
                for i in range(len(spanlist_high)):
                    plt.axvspan(spanlist_high[i][0], spanlist_high[i][1], facecolor='m', alpha=0.7)
                # for i in range(len(spanlist_trend_up)):
                #     plt.axvspan(spanlist_trend_up[i][0], spanlist_trend_up[i][1], facecolor='b', alpha=0.3)
                #
                plt.subplot(414)
                plt.plot(Y_pred_[:, [2]], 'limegreen', label='Y_pred')
                plt.legend(loc='upper right')

                plt.show()
                # plt.savefig('./Figure_trade/%s_%s/%s %s.png' % (input_data_length, model_num, datetime.now().date(), Coin), dpi=500)
                # plt.close()


