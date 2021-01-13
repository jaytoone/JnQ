import numpy as np
import pandas as pd
from keras.models import load_model
from matplotlib import pyplot as plt
import os
from Make_X_Cross_Long import made_x
import Make_X_Cross_Long_no_candle
from Funcs_CNN4 import ema_cross
from datetime import datetime
import pybithumb
import sys
np.set_printoptions(threshold=sys.maxsize)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


home_dir = os.path.expanduser('~')
dir = home_dir + '/OneDrive/CoinBot/ohlcv/'
ohlcv_list = os.listdir(dir)

if __name__ == '__main__':

    #           PARAMS           #
    input_data_length = 30
    model_num = 67
    model_num2 = '77_ohlc'
    crop_size = input_data_length

    #       Make folder      #
    folder_name_list = ['pred_ohlcv', 'Figure_trade']

    for folder_name in folder_name_list:
        try:
            os.mkdir('./%s/%s_%s/' % (folder_name, input_data_length, model_num))

        except Exception as e:
            print(e)
            continue

    except_list = os.listdir('./pred_ohlcv/%s_%s' % (input_data_length, model_num))

    Coin_list = ['chr'.upper()]

    #       LOAD MODEL      #
    model = load_model('./model/rapid_ascending %s_%s_cross_long.hdf5' % (input_data_length, model_num))
    model2 = load_model('./model/rapid_ascending %s_%s.hdf5' % (input_data_length, model_num2))

    for Coin in Coin_list:

        ohlcv_excel = pybithumb.get_ohlcv(Coin, 'KRW', 'minute1')

        #         if file in except_list:
        #             continue

        print('loading %s' % Coin)

        Date = str(datetime.now()).split()[0]

        try:
            result = made_x(ohlcv_excel, input_data_length, model_num, crop_size=input_data_length)
            result2 = Make_X_Cross_Long_no_candle.made_x(ohlcv_excel, input_data_length, model_num, crop_size=input_data_length)
            X_test = np.array(result[0])
            X_test2 = np.array(result2[0])
            row = X_test2.shape[1]
            col = X_test2.shape[2]
            # col2 = X_test2.shape[2]

            X_test2 = X_test2.astype('float32').reshape(-1, row, col, 1)[:, :, [0, 1, 2, 3]]
            # quit()

        except Exception as e:
            print('Error in getting data from made_x :', e)
            continue

        if X_test is not None:
            if len(X_test) != 0:

                Y_pred_ = model.predict(X_test, verbose=1)
                Y_pred_2 = model2.predict(X_test2, verbose=1)
                #
                # max_value = np.max(Y_pred_, axis=0)
                # print(max_value)
                # print(Y_pred_)
                Y_pred = np.argmax(Y_pred_, axis=1)
                Y_pred2 = np.argmax(Y_pred_2, axis=1)

                # Y_pred = np.zeros(len(Y_pred_))
                # for i in range(len(Y_pred_)):
                #     if Y_pred_[i][1] > 0.7:
                #         Y_pred[i] = 1.
                #     else:
                #         Y_pred[i] = 0.

                # ohlcv_excel = pd.read_excel(dir + file, index_col=0)
                ema_cross(ohlcv_excel)
                trade_state = [0] * len(ohlcv_excel)
                for i in range(2, len(ohlcv_excel)):
                    #   이전에 역순이 존재하고 바로 전이 정순이면,
                    if ohlcv_excel['EMA_1'][i - 2] <= ohlcv_excel['EMA_2'][i - 2]:
                        if ohlcv_excel['EMA_1'][i - 1] > ohlcv_excel['EMA_2'][i - 1]:
                            trade_state[i] = 1

                    if ohlcv_excel['EMA_1'][i - 2] >= ohlcv_excel['EMA_2'][i - 2]:
                        if ohlcv_excel['EMA_1'][i - 1] < ohlcv_excel['EMA_2'][i - 1]:
                            trade_state[i] = 2
                ohlcv_excel['trade_state'] = trade_state
                ohlcv_data = ohlcv_excel.values[sum(ohlcv_excel.EMA_2.isna()):].astype(np.float)[crop_size:]
                y = ohlcv_data[:, [-1]]
                span_list = list()
                crop_span = np.full(len(y), np.NaN)
                for i in range(len(crop_span)):
                    if 0.5 < y[i] < 1.5:  # 1을 찾았을 때
                        for j in range(i + 1, len(crop_span)):
                            if 1.5 < y[j] < 2.5:  # 뒤에 2가 존재한다면,
                                #   i + 1 부터 j + 1 까지 0의 값을 부여한다.
                                # print(i, j)
                                # crop_span[i + 1:j + 2] = 0
                                span_list.append(ohlcv_data[i + 1:j + 2])
                                # plt.plot(ohlcv_data[i + 1:j + 2, [1]])
                                # plt.plot(ohlcv_data[i + 1:j + 2, [5]], 'g')
                                # plt.plot(ohlcv_data[i + 1:j + 2, [6]], 'r')
                                # plt.show()
                                # plt.close()
                                break

                for i in range(len(span_list)):

                    ohlcv_data = span_list[i]

                    plt.subplot(211)
                    # plt.subplot(313)
                    plt.title('%s' % Y_pred[i])
                    plt.plot((ohlcv_data[:, [1]]), 'gold', label='close')
                    plt.plot((ohlcv_data[:, [5]]), 'g', label='ema1')
                    plt.plot((ohlcv_data[:, [6]]), 'r', label='ema2')
                    plt.legend(loc='upper right')

                    plt.subplot(212)
                    # plt.subplot(313)
                    plt.title('%s' % Y_pred2[i])
                    plt.plot((ohlcv_data[:, [1]]), 'gold', label='close')
                    plt.plot((ohlcv_data[:, [5]]), 'g', label='ema1')
                    plt.plot((ohlcv_data[:, [6]]), 'r', label='ema2')
                    plt.legend(loc='upper right')

                    plt.savefig('./Figure_trade/%s_%s/%s %s_%s.png' % (input_data_length, model_num, Date, Coin, i), dpi=500)
                    plt.close()





