import os

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import numpy as np
import pandas as pd
from keras.models import load_model
from matplotlib import pyplot as plt
from Make_X_MACD_OSC import low_high
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

    series = series[:10]
    TopCoin = list(series.index)
    # TopCoin = list(map(str.upper, ['xem']))

    #           PARAMS           #
    input_data_length = 30
    model_num = '88_macd_osc'
    model_num2 = '87_macd_osc'
    # crop_size_low = 200
    # crop_size_high = 100
    # crop_size_sudden_death = 100
    # limit_line_low = 0.97
    # limit_line_high = 0.9
    # limit_line_sudden_death = 0.45
    get_fig = 0
    get_excel = 1

    Date = str(datetime.now()).split()[0]

    #       Make folder      #

    try:
        os.mkdir('./Figure_trade/%s_%s/' % (input_data_length, model_num))

    except Exception as e:
        pass

    #       LOAD MODEL      #
    model = load_model('./model/rapid_ascending %s_%s.hdf5' % (input_data_length, model_num))
    model2 = load_model('./model/rapid_ascending %s_%s.hdf5' % (input_data_length, model_num2))

    for Coin in TopCoin:

        try:
            ohlcv_excel = pybithumb.get_ohlcv(Coin, 'KRW', 'minute1')
            dataX, chart = low_high(ohlcv_excel, input_data_length, crop_size=input_data_length)
            X_test = np.array(dataX)
            # print(X_test)
            # quit()

            row = X_test.shape[1]
            col = X_test.shape[2]
            X_test = X_test.astype('float32').reshape(-1, row, col, 1)

            if 'ohlcmacd' in model_num:
                X_test = X_test[:, :, [0, 1, 2, 3, 9, 10, 11, 12]]
            elif 'macd_osc' in model_num:
                X_test = X_test[:, :, [11]]
            elif 'rsimacd' in model_num:
                X_test = X_test[:, :, [8, 9, 10, 11, 12]]
            elif 'ohlc' in model_num:
                X_test = X_test[:, :, [0, 1, 2, 3]]
            elif 'volume' in model_num:
                X_test = X_test[:, :, [0, 1, 2, 3, 4]]
            elif 'ma20' in model_num:
                X_test = X_test[:, :, [0, 1, 2, 3, 5]]
            elif 'cmo' in model_num:
                X_test = X_test[:, :, [0, 1, 2, 3, 6]]
            elif 'obv' in model_num:
                X_test = X_test[:, :, [0, 1, 2, 3, 7]]
            elif 'rsi' in model_num:
                X_test = X_test[:, :, [0, 1, 2, 3, 8]]
            elif 'macd' in model_num:
                X_test = X_test[:, :, [9, 10, 11, 12]]
            # elif 'ohlcmacd' in
            # print(X_test.shape)
            # quit()
            # X_test2, _, closeprice2, _ = low_high(Coin, input_data_length, crop_size=crop_size_high, sudden_death=0.)
            # X_test3, _, closeprice3, _ = low_high(Coin, input_data_length, crop_size=crop_size_sudden_death, sudden_death=0.)
            # # X_test, _ = low_high(Coin, input_data_length, sudden_death=1.)
            # closeprice = np.roll(np.array(list(map(lambda x: x[-1][[1]][0], X_test))), -1)

            if X_test is None:
                continue

        except Exception as e:
            print('Error in getting data from made_x :', e)

        # dataX 에 담겨있는 value 에 [-1] : 바로 이전의 행 x[-1][:].shape = (1, 6)
        # sliced_ohlcv = np.array(list(map(lambda x: x[-1][:], X_test)))
        # print(sliced_ohlcv)
        # quit()

        if len(X_test) != 0:

            Y_pred_ = model.predict(X_test, verbose=1)
            Y_pred2_ = model2.predict(X_test, verbose=1)
            # Y_pred_[:, [-1]] *= 0.7

            Y_pred = np.argmax(Y_pred_, axis=1)
            Y_pred2 = np.argmax(Y_pred2_, axis=1)
            # print(Y_pred)
            # quit()

            # max_value = np.max(Y_pred_, axis=0)
            # print(max_value)
            # Y_pred = np.zeros(len(Y_pred_))
            # for i in range(len(Y_pred_)):
            #     if Y_pred_[i][0] > max_value[0] * 0.9:
            #         Y_pred[i] = 0
            #     if Y_pred_[i][1] > max_value[1] * 0.95:
            #         Y_pred[i] = 1

            # for i in range(len(Y_pred2_)):
            #     if Y_pred2_[i][2] > max_value2[2] * limit_line_high:
            #         Y_pred2[i] = 2
            # for i in range(len(Y_pred3_)):
            #     if Y_pred3_[i][2] > max_value3[2] * limit_line_sudden_death:
            #         Y_pred3[i] = 2

            if get_fig == 1:

                for index, Y_predict in enumerate([Y_pred, Y_pred2]):
                    spanlist_low = []
                    spanlist_high = []
                    spanlist = []
                    for m in range(len(Y_predict)):
                        if Y_predict[m] == 0.:
                            if m + 1 < len(Y_predict):
                                spanlist.append((m, m + 1))
                            else:
                                spanlist.append((m - 1, m))

                    for m in range(len(Y_predict)):
                        if Y_predict[m] == 1.:
                            if m + 1 < len(Y_predict):
                                spanlist_low.append((m, m + 1))
                            else:
                                spanlist_low.append((m - 1, m))

                    for m in range(len(Y_predict)):
                        if Y_predict[m] == 2.:
                            if m + 1 < len(Y_predict):
                                spanlist_high.append((m, m + 1))
                            else:
                                spanlist_high.append((m - 1, m))

                # plt.subplot(311)
                # # plt.subplot(313)
                # # plt.plot(chart[:, [1]], 'gold', label='close')
                # plt.plot(chart[:, [-3]], 'g', label='close')
                # # plt.plot(chart[:, [-4]], 'r', label='close')
                # plt.plot(chart[:, [-2]], 'gold', label='close')
                # plt.legend(loc='upper right')
                # for i in range(len(spanlist)):
                #     plt.axvspan(spanlist[i][0], spanlist[i][1], facecolor='b', alpha=0.7)

                    if index == 0:
                        plt.subplot(211)
                        # plt.subplot(312)
                        # plt.subplot(313)
                        # plt.plot(chart[:, [1]], 'gold', label='close')
                        plt.plot(chart[:, [-3]], 'g', label='close')
                        # plt.plot(chart[:, [-4]], 'r', label='close')
                        plt.plot(chart[:, [-2]], 'gold', label='close')
                        plt.legend(loc='upper right')
                        for i in range(len(spanlist_low)):
                            plt.axvspan(spanlist_low[i][0], spanlist_low[i][1], facecolor='c', alpha=0.7)

                    elif index == 1:
                        plt.subplot(212)
                        # plt.subplot(313)
                        # plt.subplot(313)
                        plt.plot(chart[:, [-3]], 'g', label='close')
                        # plt.plot(chart[:, [-4]], 'r', label='close')
                        plt.plot(chart[:, [-2]], 'gold', label='close')
                        plt.legend(loc='upper right')
                        for i in range(len(spanlist_high)):
                            plt.axvspan(spanlist_high[i][0], spanlist_high[i][1], facecolor='m', alpha=0.7)
                #
                # plt.subplot(313)
                # # plt.subplot(313)
                # plt.plot(closeprice3, 'gold', label='close')
                # # plt.plot(OBV, 'b', label='OBV')
                # plt.legend(loc='upper right')
                # for i in range(len(spanlist_sudden_death)):
                #     plt.axvspan(spanlist_sudden_death[i][0], spanlist_sudden_death[i][1], facecolor='m', alpha=0.7)

                plt.show()
                # plt.savefig('./Figure_trade/%s_%s/%s %s.png' % (input_data_length, model_num, datetime.now().date(), Coin), dpi=500)
                # plt.close()

            elif get_excel == 1:

                df = pd.DataFrame(np.hstack((ohlcv_excel.values[-len(chart):], Y_pred2.reshape(-1, 1))),
                                  columns=['open', 'close', 'high', 'low', 'volume', 'MA20', 'CMO', 'OBV',
                                           'RSI', 'MACD', 'MACD_Signal', 'MACD_OSC', 'MACD_Zero', 'trade_state', 'Y_pred'])
                df.to_excel('./BestSet/Pred_ohlc/%s %s ohlcv.xlsx' % (Date, Coin))
                df = pd.DataFrame(np.hstack((ohlcv_excel.values[-len(chart):], Y_pred.reshape(-1, 1))),
                                  columns=['open', 'close', 'high', 'low', 'volume', 'MA20', 'CMO', 'OBV',
                                           'RSI', 'MACD', 'MACD_Signal', 'MACD_OSC', 'MACD_Zero', 'trade_state', 'Y_pred'])
                df.to_excel('./BestSet/Pred_ohlc88/%s %s ohlcv.xlsx' % (Date, Coin))

