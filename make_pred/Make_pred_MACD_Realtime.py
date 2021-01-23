import numpy as np
import pandas as pd
from keras.models import load_model
from matplotlib import pyplot as plt
import os
from Funcs_CNN4 import macd
import sys
from Make_X_MACD import made_x
from datetime import datetime
import pybithumb
np.set_printoptions(threshold=sys.maxsize)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


home_dir = os.path.expanduser('~')
dir = home_dir + '/OneDrive/CoinBot/ohlcv/'
ohlcv_list = os.listdir(dir)

if __name__ == '__main__':

    #           PARAMS           #
    input_data_length = 30
    model_num = '83_macd'
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

    Coin_list = ['knc'.upper()]

    #       LOAD MODEL      #
    model = load_model('./model/rapid_ascending %s_%s.hdf5' % (input_data_length, model_num))

    for Coin in Coin_list:

        ohlcv_excel = pybithumb.get_ohlcv(Coin, 'KRW', 'minute1')

        #         if file in except_list:
        #             continue

        print('loading %s' % Coin)

        Date = str(datetime.now()).split()[0]

        try:
            # X_test = np.load('./Made_Chart_to_np/%s_%s/%s %s.npy' % (input_data_length, model_num,
            #                                                          Date, 'Realtime')).astype(np.float32) / 255.
            result = made_x(ohlcv_excel, input_data_length, model_num, crop_size=input_data_length)
            X_test = np.array(result[0])
            row = X_test.shape[1]
            col = X_test.shape[2]
            X_test = X_test.astype('float32').reshape(-1, row, col, 1)

            if 'ohlcmacd' in model_num:
                X_test = X_test[:, :, [0, 1, 2, 3, 9, 10, 11]]
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
                X_test = X_test[:, :, [9, 10, 11]]
            # X_test = X_test[:100]
            # quit()

        except Exception as e:
            print('Error in getting data from made_x :', e)
            continue

        if X_test is not None:
            if len(X_test) != 0:

                Y_pred_ = model.predict(X_test, verbose=1)
                #
                # print(max_value)
                # print(Y_pred_)
                Y_pred = np.argmax(Y_pred_, axis=1)

                # ohlcv_excel = pd.read_excel(dir + file, index_col=0)
                macd(ohlcv_excel, short=5, long=35, signal=5)
                trade_state = [0] * len(ohlcv_excel)
                for i in range(2, len(ohlcv_excel)):
                    if ohlcv_excel['MACD'][i - 1] <= 0:
                        if ohlcv_excel['MACD'][i] > 0:
                            trade_state[i] = 1

                    if ohlcv_excel['MACD'][i - 1] >= 0:
                        if ohlcv_excel['MACD'][i] < 0:
                            trade_state[i] = 2
                ohlcv_excel['trade_state'] = trade_state
                ohlcv_data = ohlcv_excel.values[sum(ohlcv_excel.MACD_Signal.isna()):].astype(np.float)[crop_size:]
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
                                span_list.append(ohlcv_data[i:j + 1])
                                # plt.plot(ohlcv_data[i + 1:j + 2, [1]])
                                # plt.plot(ohlcv_data[i + 1:j + 2, [5]], 'g')
                                # plt.plot(ohlcv_data[i + 1:j + 2, [6]], 'r')
                                # plt.show()
                                # plt.close()
                                break

                len_cnt = 0
                for i in range(len(span_list)):

                    ohlcv_data = span_list[i]

                    spanlist_low = []
                    spanlist_high = []

                    Y_pred_span = Y_pred[len_cnt:len_cnt + len(ohlcv_data)]

                    # Y_pred_span_ = Y_pred_[len_cnt:len_cnt + len(ohlcv_data)]
                    # print(len(Y_pred_), len_cnt)
                    # max_value = np.max(Y_pred_span_, axis=0)
                    # Y_pred_span = np.zeros(len(Y_pred_span_))
                    # for j in range(len(Y_pred_span_)):
                    #     if Y_pred_span_[j][0] > max_value[0] * 0.9:
                    #         Y_pred_span[j] = 0
                    #     if Y_pred_span_[j][1] > max_value[1] * 0.95:
                    #         Y_pred_span[j] = 1

                    # print(Y_pred_span)

                    len_cnt = len_cnt + len(ohlcv_data)

                    for m in range(len(Y_pred_span)):
                        if Y_pred_span[m] == 0.:
                            if m + 1 < len(Y_pred_span):
                                spanlist_low.append((m, m + 1))
                            else:
                                spanlist_low.append((m - 1, m))

                    for m in range(len(Y_pred_span)):
                        if Y_pred_span[m] == 1.:
                            if m + 1 < len(Y_pred_span):
                                spanlist_high.append((m, m + 1))
                            else:
                                spanlist_high.append((m - 1, m))

                    # print(spanlist_low)
                    # print(spanlist_high)

                    plt.subplot(211)
                    # plt.subplot(313)
                    plt.plot((ohlcv_data[:, [-5]]), 'gold', label='macd')
                    plt.plot((ohlcv_data[:, [-2]]), 'g', label='0')
                    # plt.plot((ohlcv_data[:, [6]]), 'r', label='ema2')
                    plt.legend(loc='upper right')
                    for j in range(len(spanlist_low)):
                        plt.axvspan(spanlist_low[j][0], spanlist_low[j][1], facecolor='c', alpha=0.7)

                    plt.subplot(212)
                    # plt.subplot(313)
                    plt.plot((ohlcv_data[:, [-5]]), 'gold', label='macd')
                    plt.plot((ohlcv_data[:, [-2]]), 'g', label='0')
                    # plt.plot((ohlcv_data[:, [6]]), 'r', label='ema2')
                    plt.legend(loc='upper right')
                    for j in range(len(spanlist_high)):
                        plt.axvspan(spanlist_high[j][0], spanlist_high[j][1], facecolor='m', alpha=0.7)

                    plt.savefig('./Figure_trade/%s_%s/%s %s_%s.png' % (input_data_length, model_num, Date, Coin, i), dpi=500)
                    plt.close()





