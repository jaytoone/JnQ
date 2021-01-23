import numpy as np
import pandas as pd
import pybithumb
import os
import matplotlib.pyplot as plt
import warnings
from datetime import datetime
from Funcs_Indicator import *

warnings.filterwarnings("ignore")

pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 2500)
pd.set_option('display.max_columns', 2500)


def min_max_scaler(x):
    scaled_x = (x - x.min()) / (x.max() - x.min())
    return scaled_x


def max_abs_scaler(x):
    scaled_x = x / abs(x).max()
    return scaled_x


def made_x(file, input_data_length, model_num):
    ohlcv_excel = pd.read_excel(dir + file, index_col=0)

    new_column = ['open', 'close', 'high', 'low', 'MA_SHORT', 'MA_LONG', 'SUPERTREND',
                  'TRIX', 'MACD', 'MACD_Signal', 'MACD_OSC', 'FISHER_LONG', 'FISHER_LONGER',
                  'STOCHASTIC_D', 'STOCHASTIC_D2', 'STOCH_RSI', 'STOCH_RSI2', 'STOCH_RSI3', 'label']

    ohlcv_excel = ohlcv_excel[new_column]

    if 'LT_Trend' in new_column:
        ohlcv_excel['LT_Trend'] = np.where(ohlcv_excel['LT_Trend'] == 'Bull', 1, 0)
    # print(ohlcv_excel)
    # quit()

    # ----------- dataX, dataY 추출하기 -----------#
    ohlcv_excel = ohlcv_excel.iloc[sum(ohlcv_excel['STOCH_RSI3'].isna()):, :]
    if sum(sum(np.isnan(ohlcv_excel.values[:, :-1]))) != 0:
        print('None value imported')
        print(sum(np.isnan(ohlcv_excel.values[:, :-1])))
        return

    # print(ohlcv_excel.info())
    # quit()
    # ohlcv_excel.to_excel('test.xlsx')

    # NaN 제외하고 데이터 자르기 (데이터가 PIXEL 로 들어간다고 생각하면 된다)
    ohlcv_data = ohlcv_excel.values.astype(np.float)

    # plt.plot(ohlcv_data[:, :7])
    # plt.show()
    # quit()

    # 결측 데이터 제외
    if len(ohlcv_data) != 0:

        #          GLOBAL SCALING         #

        #    X_data    #
        # PRICE MA SUPERTREND
        ohlcv_data[:, [0, 1, 2, 3, 4, 5, 6]] = min_max_scaler(ohlcv_data[:, [0, 1, 2, 3, 4, 5, 6]])
        # ohlcv_data[:, [0, 1, 2, 3, 4, 5, 6]] = max_abs_scaler(ohlcv_data[:, [0, 1, 2, 3, 4, 5, 6]])
        # plt.subplot(212)
        # plt.plot(ohlcv_data[:, :7])
        # plt.show()
        # quit()

        #       TRIX / FISHER   7, 11, 12    #
        for column_number in [7, 11, 12]:
            ohlcv_data[:, [column_number]] = max_abs_scaler(ohlcv_data[:, [column_number]])

        #      STOCH SERIES     #
        for column_number in [10, 13, 14, 15, 16, 17]:
            ohlcv_data[:, [column_number]] = max_abs_scaler(ohlcv_data[:, [column_number]])

        #       MACD, SIGNAL   8, 9      #
        ohlcv_data[:, [8, 9]] = max_abs_scaler(ohlcv_data[:, [8, 9]])

        # for plot_i in range(7, 18):
        #     plt.plot(ohlcv_data[:, [plot_i]])
        #     plt.show()
        #     plt.close()
        # plt.plot(ohlcv_data[:, [8, 9]])
        # plt.show()
        # plt.close()
        # quit()

        # print(x.shape, y_low.shape)  # (258, 6) (258, 1)
        # quit()

        dataX = []  # input_data length 만큼 담을 dataX 그릇
        dataY = []  # Target 을 담을 그릇

        if input_data_length is None:
            dataX = ohlcv_data[:, :-1]
            dataY = ohlcv_data[:, [-1]]
            return dataX, dataY

        for i in range(input_data_length, len(ohlcv_data)):  # 마지막 데이터까지 다 긇어모은다.

            group_y = ohlcv_data[i, [-1]]
            if np.isnan(group_y):
                # print(group_y)
                continue
            # quit()

            group_x_ = ohlcv_data[i + 1 - input_data_length: i + 1, :-1]
            group_x = group_x_.copy()

            #       SHOULDN'T BE OVERWRAPPED BY SCALING        #
            # group_x[:, [0, 1, 2, 3, 4, 5, 6]] = min_max_scaler(group_x_[:, [0, 1, 2, 3, 4, 5, 6]])

            #         show chart       #
            # plt.subplot(211)
            # plt.plot(group_x_[:, :7])
            # plt.subplot(212)
            # plt.plot(group_x[:, [0, 1, 2, 3, 4, 5, 6]])
            # plt.title(i)
            # plt.show()
            # plt.close()

            #   데이터 값에 결측치가 존재하는 경우 #
            if sum(sum(np.isnan(group_x))) > 0:
                continue

            dataX.append(group_x)  # dataX 리스트에 추가
            dataY.append(group_y)

        return dataX, dataY


if __name__ == '__main__':

    # ----------- Params -----------#
    # input_data_length = None
    input_data_length = 100
    model_num = 131

    # home_dir = os.path.expanduser('~')
    # dir = home_dir + '/OneDrive/CoinBot/ohlcv/'
    dir = './labeled_data/Fisher/'
    ohlcv_list = os.listdir(dir)

    import random
    random.shuffle(ohlcv_list)

    #       Make folder      #
    try:
        os.mkdir('./figure_data/%s_%s/' % (input_data_length, model_num))

    except Exception as e:
        pass

    Made_X = []
    Made_Y = []

    # ohlcv_list = ['2020-07-04 META ohlcv.xlsx']

    for file in ohlcv_list:

        try:
            result = made_x(file, input_data_length, model_num)
            # result = low_high('dvp'.upper(), input_data_length, lowhigh_point='on')
            # print(result)
            # quit()

        except Exception as e:
            print(e)

        # ------------ 데이터가 있으면 dataX, dataY 병합하기 ------------#
        else:
            if result is not None:
                if result[0] is not None:
                    if input_data_length is not None:
                        Made_X += result[0]
                        Made_Y += result[1]
                    else:
                        if len(Made_X) == 0:
                            Made_X, Made_Y = result
                        else:
                            Made_X = np.vstack((Made_X, result[0]))
                            Made_Y = np.vstack((Made_Y, result[1]))

                # 누적 데이터량 표시
                print(file, len(Made_X))
                if len(Made_X) > 3000:
                    break

    # SAVING X, Y
    np.save('./Made_X/Made_X %s_%s' % (input_data_length, model_num), np.array(Made_X))
    np.save('./Made_X/Made_Y %s_%s' % (input_data_length, model_num), np.array(Made_Y))
