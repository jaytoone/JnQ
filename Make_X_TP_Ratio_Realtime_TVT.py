# import numpy as np
# import pandas as pd
# import pybithumb
# import os
# import matplotlib.pyplot as plt
import warnings
from datetime import datetime
import time
from Funcs_Indicator import *
from Binance_Futures_concat_candlestick import concat_candlestick
from Binance_Futures_Bot_Config import LabelType

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


def made_x(file, input_data_length, scale_window_size, data_amount, label_type='Test'):

    # print('type(file) :', type(file))
    if type(file) != pd.DataFrame:
        ohlcv_excel = pd.read_excel(dir + file, index_col=0)
    else:
        ohlcv_excel = file

    new_column = ['open', 'high', 'low', 'close',
                  'minor_ST1_Up', 'minor_ST1_Down',
                  'minor_ST2_Up', 'minor_ST2_Down',
                  'minor_ST3_Up', 'minor_ST3_Down',
                  'major_ST1_Up', 'major_ST1_Down',
                  'major_ST2_Up', 'major_ST2_Down',
                  'major_ST3_Up', 'major_ST3_Down',
                  'minor_ST1', 'minor_ST2', 'minor_ST3',
                  'major_ST1', 'major_ST2', 'major_ST3',
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

    if data_amount > len(ohlcv_excel):
        print('data_amount > len(ohlcv_excel)')
        quit()

    ohlcv_data = ohlcv_excel.values.astype(np.float)[-data_amount:]
    # print('len(ohlcv_data) :', len(ohlcv_data))
    # plt.plot(ohlcv_data[:, :7])
    # plt.show()
    # quit()

    # 결측 데이터 제외
    if len(ohlcv_data) != 0:

        if label_type == 'Test':
            start_i = len(ohlcv_data) - 1
        else:
            start_i = scale_window_size

        dataX, dataY, dataZ = list(), list(), list()
        chunk_num = 0
        for i in range(start_i, len(ohlcv_data)):  # 마지막 데이터까지 다 긇어모은다.

            group_y = ohlcv_data[i, [-1]]
            group_z = ohlcv_data[i, :4]

            #       Shouldn't use this for real Trading Back-test        #
            # if label_type == 'Train':
            #     if np.isnan(group_y):
            #         # print(group_y)
            #         continue
            # quit()

            #          Scaling in Scale_Window_Size         #
            group_x_ = ohlcv_data[i + 1 - scale_window_size: i + 1, :-1]
            #       Original group_x_ should't be overwrapped by scaling      #
            group_x = group_x_.copy()

            #    X_data    #
            #           OHLC EMA ST Up & Down ST         #
            group_x[:, :24] = min_max_scaler(group_x[:, :24])
            #           Just for Tester_Scaling         #
            # group_x[:, :4] = min_max_scaler(group_x[:, :4])
            # group_x[:, 4:18] = min_max_scaler(group_x[:, 4:18])

            # ohlcv_data[:, [0, 1, 2, 3, 4, 5, 6]] = max_abs_scaler(ohlcv_data[:, [0, 1, 2, 3, 4, 5, 6]])
            # plt.subplot(212)
            # plt.plot(ohlcv_data[:, :7])
            # plt.show()
            # quit()

            #           ST Trend             #
            column_index = [j for j in range(24, 30)]
            group_x[:, column_index] = max_abs_scaler(group_x[:, column_index])

            #           CB, EMA_CB         #
            column_index = [30, 31]
            group_x[:, [column_index]] = min_max_scaler(group_x[:, [column_index]])

            #           Multiple Fisher           #
            column_index = [31, 32, 33]
            group_x[:, [column_index]] = max_abs_scaler(group_x[:, [column_index]])

            #           Trix, minor_Trix     #
            column_index = [33, 34]
            group_x[:, [column_index]] = max_abs_scaler(group_x[:, [column_index]])

            #           If Min, Max value differs, scale individually using 'for'      #

            #           Slice by input data length     #
            group_x = group_x[-input_data_length:, :]

            #         show chart       #
            # plt.subplot(211)
            # plt.plot(group_x_[:, :7])
            # plt.subplot(212)
            # plt.plot(group_x[:, [0, 1, 2, 3, 4, 5, 6]])
            # plt.title(i)
            # plt.show()
            # plt.close()

            dataX.append(group_x)  # dataX 리스트에 추가
            dataY.append(group_y)
            dataZ.append(group_z)

            #          Cover Memory Error           #
            if len(dataX) > 10000:
                np.save('./Made_X/Made_X %s_%s' % (model_num, chunk_num), np.array(dataX))
                np.save('./Made_X/Made_Y %s_%s' % (model_num, chunk_num), np.array(dataY))
                np.save('./Made_X/Made_Z %s_%s' % (model_num, chunk_num), np.array(dataZ))
                chunk_num += 1
                print('len(dataX) :', len(dataX))

                #       Empty dataX, Y, Z       #
                dataX, dataY, dataZ = list(), list(), list()

        return np.array(dataX), np.array(dataY), np.array(dataZ)


if __name__ == '__main__':

    from Funcs_for_TP_Ratio_TVT_Bias_Binarize_modified import profitage

    # ----------- Params -----------#
    scale_window_size = 3000
    input_data_length = 200
    model_num = 1223
    data_amount = 0

    days = 34
    end_date = '2020-12-22'
    symbol = 'DOTUSDT'
    label_type = LabelType.TRAIN

    startTime = time.time()

    first_df, _ = concat_candlestick(symbol, '1m', days, end_date=end_date, show_process=True)
    second_df, _ = concat_candlestick(symbol, '3m', days, end_date=end_date, show_process=True)
    third_df, _ = concat_candlestick(symbol, '30m', days, end_date=end_date, show_process=True)

    print('elapsed by concat_candlestick :', time.time() - startTime)

    df = profitage(first_df, second_df, third_df, symbol=symbol, label_type=label_type, get_fig=0, excel=0, show_time=True)
    print('elapsed by profitage :', time.time() - startTime)
    # quit()

    #       use it, if you want use selected excel      #
    # save_path = "Labeled_Data/TP_Ratio_TVT/%s/" % input_data_length
    # file = '2020-12-20 DOTUSDT.xlsx'
    # df = pd.read_excel(save_path + file, index_col=0)

    # try:

    for input_data_length in [30, 60, 100, 200]:

        result = made_x(df, input_data_length, scale_window_size, data_amount, label_type)
        print('elapsed by made_x :', time.time() - startTime)

        model_num += 1