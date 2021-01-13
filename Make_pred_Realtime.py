import os

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import numpy as np
import pandas as pd
from keras.models import load_model
from matplotlib import pyplot as plt
from Make_X_rapid_ascend import low_high
import pybithumb
import time
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if __name__ == '__main__':

    #           Making TopCoin List         #
    Coinlist = pybithumb.get_tickers()
    Fluclist = []
    while True:
        try:
            for Coin in Coinlist:
                tickerinfo = pybithumb.PublicApi.ticker(Coin)
                data = tickerinfo['data']
                fluctate = data['fluctate_rate_24H']
                Fluclist.append(fluctate)
                time.sleep(1 / 90)
            break

        except Exception as e:
            Fluclist.append(None)
            print('Error in making Topcoin :', e)

    Fluclist = list(map(float, Fluclist))
    series = pd.Series(Fluclist, Coinlist)
    series = series.sort_values(ascending=False)

    series = series[0:10]
    TopCoin = list(series.index)

    for Coin in TopCoin:
        # Coin = input('Input Coin Name : ').upper()
        # input_data_length = int(input("Input Data Length : "))
        input_data_length = 54
        # model_num = input('Press model number : ')
        model_num = 19

        try:
            os.mkdir('./Figure_trade/%s_%s/' % (input_data_length, model_num))
        except Exception as e:
            pass

        #           PARAMS           #
        check_span = 30
        get_fig = 1

        #       LOAD MODEL      #
        model = load_model('model/rapid_ascending %s_%s.hdf5' % (input_data_length, model_num))

        try:
            X_test, _ = low_high(Coin, input_data_length)

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
            Y_pred_ = model.predict(X_test, verbose=1)

            max_value = np.max(Y_pred_[:, [-1]])
            limit_line = 0.9
            Y_pred = np.where(Y_pred_[:, [-1]] > max_value * limit_line, 1, 0)

            if get_fig == 1:
                spanlist = []

                for m in range(len(Y_pred)):
                    if Y_pred[m] > 0.5:
                        if m + 1 < len(Y_pred):
                            spanlist.append((m, m + 1))
                        else:
                            spanlist.append((m - 1, m))

                # ----------- 인덱스 초기화 됨 -----------#
                # plt.subplot(312)
                plt.plot(closeprice, 'r', label='close')
                plt.plot(OBV, 'b', label='OBV')
                plt.legend(loc='upper right')
                for i in range(len(spanlist)):
                    plt.axvspan(spanlist[i][0], spanlist[i][1], facecolor='m', alpha=0.5)

                plt.savefig('./Figure_trade/%s_%s/%s %s.png' % (input_data_length, model_num, datetime.now().date(), Coin), dpi=500)
                plt.close()

