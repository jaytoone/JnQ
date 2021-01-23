import numpy as np
import pandas as pd
from keras.models import load_model
from matplotlib import pyplot as plt
import os
from Make_X_total import made_x, made_x_origin
from keras.utils import np_utils
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


home_dir = os.path.expanduser('~')
dir = home_dir + '/OneDrive/CoinBot/ohlcv/'
ohlcv_list = os.listdir(dir)

if __name__ == '__main__':

    #           PARAMS           #
    input_data_length = 54
    model_num = '23 - 400000'
    crop_size = 0     # 이전 저점까지 slicing
    crop_size2 = crop_size

    limit_line = 1.  # 다음 저점을 고르기 위한 limit_line
    limit_line2 = 1.  # Y_pred 에 저점이 존재할때까지 limit_line 을 낮춘다.
    sudden_death = 0
    sudden_death2 = 0
    check_span = 30
    get_fig = 1

    #       Make folder      #
    try:
        os.mkdir('./pred_ohlcv/%s_%s/' % (input_data_length, model_num))
    except Exception as e:
        pass

    try:
        os.mkdir('./Figure_pred/%s_%s/' % (input_data_length, model_num))
    except Exception as e:
        pass

    except_list = os.listdir('./pred_ohlcv/%s_%s' % (input_data_length, model_num))

    #       LOAD MODEL      #
    model = load_model('./model/rapid_ascending %s_%s.hdf5' % (input_data_length, model_num))

    ohlcv_list = ['2019-10-09 ZRX ohlcv.xlsx']

    # def find_low_point(file, input_data_length, model_num, check_span, crop_size=crop_size, sudden_death=sudden_death):
    #     return made_x_origin(file, input_data_length, model_num, check_span, 0, crop_size=crop_size, sudden_death=sudden_death)
    result_ohlcv = None
    for file in ohlcv_list:

        if result_ohlcv is None:
            Date = file.split()[0]
            Coin = file.split()[1].split('.')[0]

        #         if file in except_list:
        #             continue

        spanlist_low = []
        spanlist_high = []
        print('loading %s' % file)

        #           데이터가 끝날때까지 반복          #
        while True:
            # try:
            print(type(file))

            X_test, _, sliced_ohlc = made_x_origin(file, input_data_length, model_num, check_span, 0, crop_size=crop_size, sudden_death=sudden_death)
            X_test2, _, sliced_ohlc2 = made_x_origin(file, input_data_length, model_num, check_span, 0, crop_size=crop_size2, sudden_death=sudden_death2)
                # X_test, _ = low_high(Coin, input_data_length, sudden_death=1.)
                # closeprice = np.roll(np.array(list(map(lambda x: x[-1][[1]][0], X_test))), -1)
                # print(X_test)

            # except Exception as e:
            #     print('Error in getting data from made_x :', e)

            # OBV = np.roll(np.array(list(map(lambda x: x[-1][[5]][0], X_test))), -1)

            if X_test is None:
                break

            #       저점이 생길때까지 limit_line 을 낮추고 있으면 자르고 자른 저점은 spanlist 에 넣어두고      #
            if len(X_test) != 0:

                Y_pred_ = model.predict(X_test, verbose=1)
                Y_pred2_ = model.predict(X_test2, verbose=1)

                max_value = np.max(Y_pred_, axis=0)
                max_value2 = np.max(Y_pred2_, axis=0)
                Y_pred = np.zeros(len(Y_pred_))
                Y_pred2 = np.zeros(len(Y_pred2_))

                break_out = 0
                while True:
                    for i in range(len(Y_pred_)):
                        if Y_pred_[i][1] > max_value[1] * limit_line:
                            Y_pred[i] = 1
                            break_out = 1   # 하나라도 있으면, spanlist 에 넣고 차트 잘라.

                    for j in range(len(Y_pred2_)):

                        if Y_pred2_[j][2] > max_value2[2] * limit_line2:
                            Y_pred2[j] = 2

                    limit_line -= 0.05
                    limit_line2 -= 0.05

                    if break_out == 1:
                        break

                #   Y_pred 에 1 이 존재하면, Plot Comparing
                #   저점을 누적해서 찍어준다.
                #                   찍을때 인덱스 번호가 달라진다.               #

                for m in np.arange(len(Y_pred)):
                    if (Y_pred[m] > 0.5) and (Y_pred[m] < 1.5):
                        m += crop_size
                        if m + 1 < len(Y_pred):
                            spanlist_low.append((m, m + 1))
                        else:
                            spanlist_low.append((m - 1, m))

                for m in np.arange(len(Y_pred2)):
                    if (Y_pred2[m] > 1.5) and (Y_pred2[m] < 2.5):
                        m += crop_size
                        if m + 1 < len(Y_pred2):
                            spanlist_high.append((m, m + 1))
                        else:
                            spanlist_high.append((m - 1, m))

                crop_size = spanlist_low[-1][0]
                crop_size2 = spanlist_low[-1][0]

                if type(file) is str:
                    result_ohlcv = sliced_ohlc

                file = sliced_ohlc

    if get_fig == 1:

        plt.subplot(211)
        # plt.subplot(313)
        plt.plot(result_ohlcv[:, [1]], 'gold', label='close')
        plt.plot(result_ohlcv[:, [-1]], 'b', label='MA')
        plt.legend(loc='upper right')
        for i in range(len(spanlist_low)):
            plt.axvspan(spanlist_low[i][0], spanlist_low[i][1], facecolor='c', alpha=0.7)

        plt.subplot(212)
        # plt.subplot(313)
        plt.plot(result_ohlcv[:, [1]], 'gold', label='close')
        plt.plot(result_ohlcv[:, [-1]], 'b', label='MA')
        plt.legend(loc='upper right')
        for i in range(len(spanlist_high)):
            plt.axvspan(spanlist_high[i][0], spanlist_high[i][1], facecolor='m', alpha=0.7)

        plt.savefig('./Figure_pred/%s_%s/%s %s.png' % (input_data_length, model_num, Date, Coin), dpi=500)
        plt.close()





