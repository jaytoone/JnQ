import numpy as np
import pandas as pd
from keras.models import load_model
from matplotlib import pyplot as plt
import os
from Make_X_Cross_Long_no_candle import made_x
from keras.utils import np_utils
from Funcs_CNN4 import ema_cross
import pybithumb
from datetime import datetime
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


home_dir = os.path.expanduser('~')
dir = home_dir + '/OneDrive/CoinBot/ohlcv/'
ohlcv_list = os.listdir(dir)

if __name__ == '__main__':

    #           PARAMS           #
    input_data_length = 30
    model_num = '73'
    model_name_list = list()
    for file in os.listdir('./model'):
        if file.find(model_num) is not -1:  # 해당 파일이면 temp[i] 에 넣겠다.
            model_name_list.append(file)
    # print(model_list)
    # print('macd' in model_list[1])
    # quit()

    crop_size = input_data_length     # 이전 저점까지 slicing

    #       Make folder      #
    folder_name_list = ['pred_ohlcv', 'Figure_trade']

    for folder_name in folder_name_list:
        try:
            os.mkdir('./%s/%s_%s/' % (folder_name, input_data_length, model_num))

        except Exception as e:
            print(e)
            continue

    except_list = os.listdir('./pred_ohlcv/%s_%s' % (input_data_length, model_num))

    Coin_list = ['ipx'.upper()]

    #       LOAD MODEL      #
    model_list = list()
    for model_name in model_name_list:
        print('load model %s' % model_name)
        model_list.append(load_model('./model/%s' % model_name))
    print()

    for Coin in Coin_list:

        ohlcv_excel = pybithumb.get_ohlcv(Coin, 'KRW', 'minute1')

        #         if file in except_list:
        #             continue

        print('loading %s' % Coin)

        Date = str(datetime.now()).split()[0]

        try:
            result = made_x(ohlcv_excel, input_data_length, model_num, crop_size=input_data_length)
            X_test = np.array(result[0])
            row = X_test.shape[1]
            col = X_test.shape[2]
            # col2 = X_test.shape[2]

            X_test = X_test.astype('float32').reshape(-1, row, col, 1)

        except Exception as e:
            print('Error in getting data from made_x :', e)
            continue

        # OBV = np.roll(np.array(list(map(lambda x: x[-1][[5]][0], X_test))), -1)

        # dataX 에 담겨있는 value 에 [-1] : 바로 이전의 행 x[-1][:].shape = (1, 6)
        # sliced_ohlcv = np.array(list(map(lambda x: x[-1][:], X_test)))
        # print(sliced_ohlcv)
        # quit()

        if X_test is not None:
            if len(X_test) != 0:

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
                    fig_cnt = 1
                    for model_name in model_name_list:

                        if 'ohlc' in model_name:
                            X_test_resize = X_test[:, :, [0, 1, 2, 3]]
                        # elif 'volume' in model_name:
                        #     X_test_resize = X_test[:, :, [0, 1, 2, 3, 4]]
                        elif 'ema' in model_name:
                            X_test_resize = X_test[:, :, [0, 1, 2, 3, 5, 6]]
                        elif 'cmo' in model_name:
                            X_test_resize = X_test[:, :, [0, 1, 2, 3, 7]]
                        elif 'obv' in model_name:
                            X_test_resize = X_test[:, :, [0, 1, 2, 3, 8]]
                        elif 'rsi' in model_name:
                            X_test_resize = X_test[:, :, [0, 1, 2, 3, 9]]
                        elif 'macd' in model_name:
                            X_test_resize = X_test[:, :, [0, 1, 2, 3, 10, 11, 12]]

                        Y_pred_ = model_list[fig_cnt - 1].predict(X_test_resize, verbose=1)
                        Y_pred = np.argmax(Y_pred_, axis=1)

                        try:
                            plt.subplot(len(model_list), 1, fig_cnt)
                            plt.title('%s %s' % (Y_pred[i], model_name))
                            plt.plot((ohlcv_data[:, [1]]), 'gold', label='close')
                            plt.plot((ohlcv_data[:, [5]]), 'g', label='ema1')
                            plt.plot((ohlcv_data[:, [6]]), 'r', label='ema2')
                            plt.legend(loc='upper right')

                        except Exception as e:
                            print(e)

                        fig_cnt += 1

                    plt.savefig('./Figure_trade/%s_%s/%s %s_%s.png' % (input_data_length, model_num, Date, Coin, i),
                                dpi=500)
                    plt.close()





