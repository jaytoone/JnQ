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


def made_x(file, input_data_length):

    ohlcv_excel = pd.read_excel(dir + file, index_col=0)

    new_column = ['open', 'high', 'low', 'close',
                  'minor_ST1_Up', 'minor_ST1_Down',
                  'minor_ST2_Up', 'minor_ST2_Down',
                  'minor_ST3_Up', 'minor_ST3_Down',
                  'major_ST1_Up', 'major_ST1_Down',
                  'major_ST2_Up', 'major_ST2_Down',
                  'major_ST3_Up', 'major_ST3_Down',
                  '30EMA', '100EMA',
                  'minor_ST1_Trend', 'minor_ST2_Trend', 'minor_ST3_Trend',
                  'major_ST1_Trend', 'major_ST2_Trend', 'major_ST3_Trend',
                  'CB', 'EMA_CB',
                  'Fisher1', 'Fisher2', 'Fisher3',
                  'Trix', 'minor_Trix',
                  'trade_state'
                  ]

    ohlcv_excel = ohlcv_excel[new_column]

    # print(ohlcv_excel.columns)
    # quit()
    # print(max(ohlcv_excel.iloc[:, :-1].isna().sum()))
    # print(ohlcv_excel.info())
    # quit()

    #     NaN 제외하고 데이터 자르기 (데이터가 PIXEL 로 들어간다고 생각하면 된다)     #
    ohlcv_excel = ohlcv_excel.iloc[max(ohlcv_excel.iloc[:, :-1].isna().sum()):, :]
    if max(ohlcv_excel.iloc[:, :-1].isna().sum()) > 0:
        print('None value imported')
        print(ohlcv_excel.iloc[:, :-1].isna().sum())
        return

    # print(ohlcv_excel.info())
    # quit()
    # ohlcv_excel.to_excel('test.xlsx')

    ohlcv_data = ohlcv_excel.values.astype(np.float)

    # plt.plot(ohlcv_data[:, :7])
    # plt.show()
    # quit()

    # 결측 데이터 제외
    if len(ohlcv_data) != 0:

        #    X_data    #
        #           OHLC EMA ST Up & Down         #
        ohlcv_data[:, :18] = min_max_scaler(ohlcv_data[:, :18])
        # ohlcv_data[:, [0, 1, 2, 3, 4, 5, 6]] = max_abs_scaler(ohlcv_data[:, [0, 1, 2, 3, 4, 5, 6]])
        # plt.subplot(212)
        # plt.plot(ohlcv_data[:, :7])
        # plt.show()
        # quit()

        #           ST Trend : 18 ~ 23      #
        column_index = [j for j in range(18, 24)]
        ohlcv_data[:, column_index] = max_abs_scaler(ohlcv_data[:, column_index])

        #           CB, EMA_CB : 24, 25   #
        column_index = [25, 25]
        ohlcv_data[:, [column_index]] = min_max_scaler(ohlcv_data[:, [column_index]])

        #           Multiple Fisher : 26 ~ 28   #
        column_index = [26, 27, 28]
        ohlcv_data[:, [column_index]] = max_abs_scaler(ohlcv_data[:, [column_index]])

        #           Trix, minor_Trix : 29, 30      #
        column_index = [29, 30]
        ohlcv_data[:, [column_index]] = max_abs_scaler(ohlcv_data[:, [column_index]])

        #           If Min, Max value differs, scale individually using 'for'      #
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

        dataX = []
        dataY = []

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

            #          Scaling in Scale_Window_Size         #
            group_x_ = ohlcv_data[i + 1 - input_data_length: i + 1, :-1]
            #       Original group_x_ should't be overwrapped by scaling      #
            group_x = group_x_.copy()

            #         show chart       #
            # plt.subplot(211)
            # plt.plot(group_x_[:, :7])
            # plt.subplot(212)
            # plt.plot(group_x[:, [0, 1, 2, 3, 4, 5, 6]])
            # plt.title(i)
            # plt.show()
            # plt.close()

            #   데이터 값에 결측치가 존재하는 경우 #
            if max(sum(np.isnan(group_x))) > 0:
                continue

            dataX.append(group_x)  # dataX 리스트에 추가
            dataY.append(group_y)

        return dataX, dataY


if __name__ == '__main__':

    # ----------- Params -----------#
    # input_data_length = None
    input_data_length = 30
    model_num = 259

    # home_dir = os.path.expanduser('~')
    # dir = home_dir + '/OneDrive/CoinBot/ohlcv/'
    dir = './labeled_data/'
    # dir = './labeled_data/Stack_TP_Ratio/'
    ohlcv_list = os.listdir(dir)

    import random

    # random.shuffle(ohlcv_list)

    #       Make folder      #
    try:
        os.mkdir('./figure_data/%s_%s/' % (input_data_length, model_num))

    except Exception as e:
        pass

    Made_X = []
    Made_Y = []

    # ohlcv_list = ['2020-12-03 COMPUSDT.xlsx']

    for file in ohlcv_list:

        model_num += 1
        try:
            result = made_x(file, input_data_length)
            # print(result)
            # quit()

            #       Save        #
            Made_X, Made_Y = result
            print(file, len(Made_X))
            np.save('./Made_X/Made_X %s_%s' % (input_data_length, model_num), np.array(Made_X))
            np.save('./Made_X/Made_Y %s_%s' % (input_data_length, model_num), np.array(Made_Y))
            # quit()
            continue

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
