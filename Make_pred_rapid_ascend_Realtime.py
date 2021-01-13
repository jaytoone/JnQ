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
    # series = series[:20]
    # TopCoin = list(series.index)
    TopCoin = list(map(str.upper, ['fct']))

    for Coin in TopCoin:

        #           PARAMS           #
        input_data_length = 54
        # input_data_length = int(input("Input Data Length : "))
        # model_num = input('Press model number : ')
        model_num = '23_400000'
        crop_size_low = input_data_length
        crop_size_high = input_data_length
        crop_size_sudden_death = 100
        limit_line_low = 0.97
        limit_line_high = 0.9
        limit_line_sudden_death = 0.45
        get_fig = 1

        #       Make folder      #

        try:
            os.mkdir('./Figure_trade/%s_%s/' % (input_data_length, model_num))

        except Exception as e:
            pass

        #       LOAD MODEL      #
        model = load_model('./model/rapid_ascending %s_%s.hdf5' % (input_data_length, model_num))

        try:
            X_test, _, closeprice, _ = low_high(Coin, input_data_length, crop_size=crop_size_low)
            X_test2, _, closeprice2, _ = low_high(Coin, input_data_length, crop_size=crop_size_high)
            X_test3, _, closeprice3, _ = low_high(Coin, input_data_length, crop_size=crop_size_sudden_death)
            # X_test, _ = low_high(Coin, input_data_length, sudden_death=1.)
            # closeprice = np.roll(np.array(list(map(lambda x: x[-1][[1]][0], X_test))), -1)
            # print(X_test.shape)

            if X_test is None:
                continue

        except Exception as e:
            print('Error in getting data from made_x :', e)

        OBV = np.roll(np.array(list(map(lambda x: x[-1][[-1]][0], X_test))), -1)

        # dataX 에 담겨있는 value 에 [-1] : 바로 이전의 행 x[-1][:].shape = (1, 6)
        # sliced_ohlcv = np.array(list(map(lambda x: x[-1][:], X_test)))
        # print(sliced_ohlcv)
        # quit()

        if len(X_test) != 0:

            Y_pred_ = model.predict(X_test, verbose=1)
            Y_pred2_ = model.predict(X_test2, verbose=1)
            Y_pred3_ = model.predict(X_test3, verbose=1)
            # print(Y_pred2_.shape)
            # quit()

            max_value = np.max(Y_pred_, axis=0)
            # print(max_value)
            max_value2 = np.max(Y_pred2_, axis=0)
            max_value3 = np.max(Y_pred3_, axis=0)
            Y_pred = np.zeros(len(Y_pred_))
            Y_pred2 = np.zeros(len(Y_pred2_))
            Y_pred3 = np.zeros(len(Y_pred3_))
            for i in range(len(Y_pred_)):
                if Y_pred_[i][1] > max_value[1] * limit_line_low:
                    Y_pred[i] = 1
            for i in range(len(Y_pred2_)):
                if Y_pred2_[i][2] > max_value2[2] * limit_line_high:
                    Y_pred2[i] = 2
            for i in range(len(Y_pred3_)):
                if Y_pred3_[i][2] > max_value3[2] * limit_line_sudden_death:
                    Y_pred3[i] = 2

            if get_fig == 1:

                spanlist_low = []
                spanlist_high = []
                spanlist_sudden_death = []

                for m in range(len(Y_pred)):
                    if (Y_pred[m] > 0.5) and (Y_pred[m] < 1.5):
                        if m + 1 < len(Y_pred):
                            spanlist_low.append((m, m + 1))
                        else:
                            spanlist_low.append((m - 1, m))
                for m in range(len(Y_pred2)):
                    if Y_pred2[m] == 2.:
                        if m + 1 < len(Y_pred2):
                            spanlist_high.append((m, m + 1))
                        else:
                            spanlist_high.append((m - 1, m))
                for m in range(len(Y_pred3)):
                    if Y_pred3[m] == 2.:
                        if m + 1 < len(Y_pred3):
                            spanlist_sudden_death.append((m, m + 1))
                        else:
                            spanlist_sudden_death.append((m - 1, m))

                plt.subplot(311)
                # plt.subplot(313)
                plt.plot(closeprice, 'gold', label='close')
                # plt.plot(OBV, 'b', label='OBV')
                plt.legend(loc='upper right')
                for i in range(len(spanlist_low)):
                    plt.axvspan(spanlist_low[i][0], spanlist_low[i][1], facecolor='c', alpha=0.7)

                plt.subplot(312)
                # plt.subplot(313)
                plt.plot(closeprice2, 'gold', label='close')
                # plt.plot(OBV, 'b', label='OBV')
                plt.legend(loc='upper right')
                for i in range(len(spanlist_high)):
                    plt.axvspan(spanlist_high[i][0], spanlist_high[i][1], facecolor='m', alpha=0.7)

                plt.subplot(313)
                # plt.subplot(313)
                plt.plot(closeprice3, 'gold', label='close')
                # plt.plot(OBV, 'b', label='OBV')
                plt.legend(loc='upper right')
                for i in range(len(spanlist_sudden_death)):
                    plt.axvspan(spanlist_sudden_death[i][0], spanlist_sudden_death[i][1], facecolor='m', alpha=0.7)

                # plt.show()
                plt.savefig('./Figure_trade/%s_%s/%s %s.png' % (input_data_length, model_num, datetime.now().date(), Coin), dpi=500)
                plt.close()


