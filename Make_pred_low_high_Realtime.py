import os

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import numpy as np
import pandas as pd
from keras.models import load_model
from matplotlib import pyplot as plt
from Make_X_total import low_high
from datetime import datetime
import pybithumb
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


if __name__ == '__main__':

    #           Making TopCoin List         #
    # Coinlist = pybithumb.get_tickers()
    # Fluclist = []
    # while True:
    #     try:
    #         for Coin in Coinlist:
    #             tickerinfo = pybithumb.PublicApi.ticker(Coin)
    #             data = tickerinfo['data']
    #             fluctate = data['fluctate_rate_24H']
    #             Fluclist.append(fluctate)
    #             time.sleep(1 / 90)
    #         break
    #
    #     except Exception as e:
    #         Fluclist.append(None)
    #         print('Error in making Topcoin :', e)
    #
    # Fluclist = list(map(float, Fluclist))
    # series = pd.Series(Fluclist, Coinlist)
    # series = series.sort_values(ascending=False)
    #
    # series = series[0:10]
    # TopCoin = list(series.index)
    # TopCoin = [input('Input Coin Name : ').upper()]

    for Coin in TopCoin:
        # Coin = input('Input Coin Name : ').upper()
        # input_data_length = int(input("Input Data Length : "))
        input_data_length = 54
        # model_num = input('Press model number : ')
        model_num = 16

        #           PARAMS           #
        check_span = 30
        get_fig = 1

        #       LOAD MODEL      #
        model_low = load_model('model/rapid_ascending_low %s_%s.hdf5' % (input_data_length, model_num))
        model_high = load_model('model/rapid_ascending_high %s_%s.hdf5' % (input_data_length, model_num))

        try:
            X_test, _ = low_high(Coin, input_data_length, 'proxy')

        except Exception as e:
            print('Error in getting data from made_x :', e)

        closeprice = np.roll(np.array(list(map(lambda x: x[-1][[1]][0], X_test))), -1)
        OBV = np.roll(np.array(list(map(lambda x: x[-1][[-1]][0], X_test))), -1)

        # dataX 에 담겨있는 value 에 [-1] : 바로 이전의 행 x[-1][:].shape = (1, 6)
        # sliced_ohlcv = np.array(list(map(lambda x: x[-1][:], X_test)))
        # print(sliced_ohlcv)
        # quit()

        if len(X_test) != 0:

            #       Data Preprocessing      #
            Y_pred_low_ = model_low.predict(X_test, verbose=1)
            Y_pred_high_ = model_high.predict(X_test, verbose=1)

            # max_value = np.max(Y_pred_[:, [-1]])
            Y_pred_low = np.array(list(map(classifier, Y_pred_low_)))
            Y_pred_low = np.argmax(Y_pred_low, axis=1)
            print(Y_pred_low)
            quit()
            Y_pred_high = np.array(list(map(classifier, Y_pred_high_)))
            Y_pred_high = np.argmax(Y_pred_high, axis=1)

            # limit_line = 0.9
            # limit_line_low = 0.9
            # limit_line_high = 0.9
            #
            # Y_pred_low = np.where(Y_pred_low_[:, [-1]] > max_value_low * limit_line_low, 1, 0)
            # Y_pred_high = np.where(Y_pred_high_[:, [-1]] > max_value_high * limit_line_high, 1, 0)

            #       Save Pred_ohlcv      #
            # 기존에 pybithumb 을 통해서 제공되던 ohlcv 와는 조금 다르다 >> 이전 데이터와 현재 y 데이터 행이 같다.
            sliced_Y_low = Y_pred_low.reshape(-1, 1)
            sliced_Y_high = Y_pred_high.reshape(-1, 1)

            if get_fig == 1:
                spanlist_low = []
                spanlist_high = []

                for m in range(len(Y_pred_low)):
                    if Y_pred_low[m] > 0.5:
                        if m + 1 < len(Y_pred_low):
                            spanlist_low.append((m, m + 1))
                        else:
                            spanlist_low.append((m - 1, m))

                for m in range(len(Y_pred_high)):
                    if Y_pred_high[m] > 0.5:
                        print()
                        if m + 1 < len(Y_pred_high):
                            spanlist_high.append((m, m + 1))
                        else:
                            spanlist_high.append((m - 1, m))

                # ----------- 인덱스 초기화 됨 -----------#

                plt.subplot(211)
                # plt.subplot(312)
                plt.plot(closeprice, 'r', label='close')
                plt.plot(OBV, 'b', label='OBV')
                plt.legend(loc='upper right')
                for i in range(len(spanlist_low)):
                    plt.axvspan(spanlist_low[i][0], spanlist_low[i][1], facecolor='m', alpha=0.5)

                plt.subplot(212)
                # plt.subplot(313)
                plt.plot(closeprice, 'r', label='close')
                plt.plot(OBV, 'b', label='OBV')
                plt.legend(loc='upper right')
                for i in range(len(spanlist_high)):
                    plt.axvspan(spanlist_high[i][0], spanlist_high[i][1], facecolor='c', alpha=0.5)

                plt.savefig('./Figure_trade/%s_%s/%s %s.png' % (input_data_length, model_num, datetime.now().date(), Coin), dpi=500)
                plt.close()

