import numpy as np
import pandas as pd
import pybithumb
import os
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from Funcs_CNN4 import cmo, rsi, obv, macd
warnings.filterwarnings("ignore")
pd.set_option('display.max_rows', 500)

Scaler = MinMaxScaler()

home_dir = os.path.expanduser('~')
dir = home_dir + '/OneDrive/CoinBot/ohlcv/'
ohlcv_list = os.listdir(dir)


def min_max_scaler(price):
    Scaler = MinMaxScaler()
    Scaler.fit(price)

    return Scaler.transform(price)


def low_high(Coin, input_data_length, ip_limit=None, trade_limit=None, crop_size=300, lowhigh_point=None):

    #   거래 제한은 고점과 저점을 분리한다.

    #   User-Agent Configuration
    #   IP - Change
    if ip_limit is None:
        ohlcv_excel = pybithumb.get_ohlcv(Coin, 'KRW', 'minute1')
    else:
        ohlcv_excel = pybithumb.get_ohlcv(Coin, 'KRW', 'minute1', 'proxyon')

    price_gap = ohlcv_excel.close.max() / ohlcv_excel.close.min()
    if (price_gap < 1.07) and (trade_limit is not None):
        return None, None, None, None

    ohlcv_excel['OBV'] = obv(ohlcv_excel)

    if lowhigh_point is not None:
        check_span = 20
        trade_state = [np.NaN] * len(ohlcv_excel)
        for i in range(check_span, len(ohlcv_excel) - check_span):
            #   저점
            if ohlcv_excel['close'][i - check_span:i].min() >= ohlcv_excel['close'][i]:
                if ohlcv_excel['close'][i + 1:i + 1 + check_span].min() >= ohlcv_excel['close'][i]:
                    # if ohlcv_excel['close'][i - check_span:i + 1 + check_span].value_counts().sort_index().iloc[0] <= 3:
                    if 1 not in trade_state[i - check_span:i]:
                        trade_state[i] = 1
                    else:
                        trade_state[i] = 0
                    # else:
                    #     trade_state[i] = 0
                else:
                    trade_state[i] = 0
            #   고점
            elif ohlcv_excel['close'][i - check_span:i].max() <= ohlcv_excel['close'][i]:
                if ohlcv_excel['close'][i + 1:i + 1 + check_span].max() <= ohlcv_excel['close'][i]:
                    # if ohlcv_excel['close'][i - check_span:i + 1 + check_span].value_counts().sort_index().iloc[-1] <= 3:
                    if 2 not in trade_state[i - check_span:i]:
                        trade_state[i] = 2
                    else:
                        trade_state[i] = 0
                # else:
                #     trade_state[i] = 0
                else:
                    trade_state[i] = 0

            #   거래 안함
            else:
                trade_state[i] = 0

        ohlcv_excel['trade_state'] = trade_state

        #   현재가가 이전 저점보다 0.1 이상 크지 않으면 거래하지 않는다.
        low_index = None
        for i in range(len(trade_state)):
            if trade_state[-i] == 1:
                low_index = i

    closeprice = ohlcv_excel['close'].iloc[-1]
    datetime_list = ohlcv_excel.index.values

    # ----------- dataX, dataY 추출하기 -----------#
    #   OBV :
    ohlcv_data = ohlcv_excel.values[1:].astype(np.float)

    scaled_close = min_max_scaler(ohlcv_data[:, [1]])
    prev_low = scaled_close[-low_index]
    pres_low = scaled_close[-1]
    if pres_low - prev_low < 0.1:
        return None, None, None, None

    # 결측 데이터 제외
    if len(ohlcv_data) != 0:

        #          데이터 전처리         #
        #   Fixed X_data    #
        # price = ohlcv_data[:, :4]
        # volume = ohlcv_data[:, [4]]
        # OBV = ohlcv_data[:, [-1]]
        #
        # scaled_price = min_max_scaler(price)
        # scaled_volume = min_max_scaler(volume)
        # scaled_OBV = min_max_scaler(OBV)
        # print(scaled_MA60.shape)

        # x = np.concatenate((scaled_price, scaled_volume, scaled_OBV), axis=1)  # axis=1, 세로로 합친다
        # print(x.shape)
        # print(ohlcv_data.shape)
        # quit()

        # if (x[-1][1] > 0.3) and (trade_limit is not None):
        #     return None, None

        # print(x.shape)  # (258, 6)
        # quit()

        dataX = []  # input_data length 만큼 담을 dataX 그릇
        for i in range(crop_size, len(ohlcv_data) + 1):  # 마지막 데이터까지 다 긇어모은다.
            group_x = ohlcv_data[i - crop_size: i]
            scaled_price = min_max_scaler(group_x[:, :4])
            # scaled_volume = min_max_scaler(group_x[:, [4]])
            # scaled_OBV = min_max_scaler(group_x[:, [-1]])
            # x = np.concatenate((scaled_price, scaled_volume, scaled_OBV), axis=1) + sudden_death  # axis=1, 세로로 합친다
            x = scaled_price  # axis=1, 세로로 합친다
            group_x = x[-input_data_length:]
            # print(group_x[-1:, 1:2])
            # print(group_x.shape)
            # print(group_x.max())
            dataX.append(group_x)  # dataX 리스트에 추가

        if len(dataX) < 100:
            return None, None, None, None

        # quit()
        X_test = np.array(dataX)
        row = X_test.shape[1]
        col = X_test.shape[2]

        X_test = X_test.astype('float32').reshape(-1, row, col, 1)
        # closeprice = np.roll(np.array(list(map(lambda x: x[-1][[1]][0], X_test))), -1)
        # plt.plot(closeprice)
        # plt.show()

        return X_test, closeprice, min_max_scaler(ohlcv_data[crop_size:, [1]]), datetime_list


# def low_high_origin(Coin, input_data_length, ip_limit=None, trade_limit=None, crop_size=300, sudden_death=0):
#
#     #   거래 제한은 고점과 저점을 분리한다.
#
#     #   User-Agent Configuration
#     #   IP - Change
#     if ip_limit is None:
#         ohlcv_excel = pybithumb.get_ohlcv(Coin, 'KRW', 'minute1')
#     else:
#         ohlcv_excel = pybithumb.get_ohlcv(Coin, 'KRW', 'minute1', 'proxyon')
#
#     price_gap = ohlcv_excel.close.max() / ohlcv_excel.close.min()
#     if (price_gap < 1.07) and (trade_limit is not None):
#         return None, None, None
#
#     obv = [0] * len(ohlcv_excel)
#     for m in range(1, len(ohlcv_excel)):
#         if ohlcv_excel['close'].iloc[m] > ohlcv_excel['close'].iloc[m - 1]:
#             obv[m] = obv[m - 1] + ohlcv_excel['volume'].iloc[m]
#         elif ohlcv_excel['close'].iloc[m] == ohlcv_excel['close'].iloc[m - 1]:
#             obv[m] = obv[m - 1]
#         else:
#             obv[m] = obv[m - 1] - ohlcv_excel['volume'].iloc[m]
#     ohlcv_excel['OBV'] = obv
#
#     closeprice = ohlcv_excel['close'].iloc[-1]
#     datetime_list = ohlcv_excel.index.values
#
#     # ----------- dataX, dataY 추출하기 -----------#
#     #   OBV :
#     ohlcv_data = ohlcv_excel.values[1:].astype(np.float)
#
#     # 결측 데이터 제외
#     if len(ohlcv_data) != 0:
#
#         #          데이터 전처리         #
#         #   Fixed X_data    #
#         price = ohlcv_data[:, :4]
#         volume = ohlcv_data[:, [4]]
#         OBV = ohlcv_data[:, [-1]]
#
#         scaled_price = min_max_scaler(price)
#         scaled_volume = min_max_scaler(volume)
#         scaled_OBV = min_max_scaler(OBV)
#         # print(scaled_MA60.shape)
#
#         # x = np.concatenate((scaled_price, scaled_volume, scaled_OBV), axis=1)  # axis=1, 세로로 합친다
#         x = scaled_price
#         # print(x.shape)
#         # print(ohlcv_data.shape)
#         # quit()
#
#         # if (x[-1][1] > 0.3) and (trade_limit is not None):
#         #     return None, None
#
#         # print(x.shape)  # (258, 6)
#         # quit()
#
#         dataX = []  # input_data length 만큼 담을 dataX 그릇
#         for i in range(input_data_length, len(x) + 1):  # 마지막 데이터까지 다 긇어모은다.
#             group_x = x[i - input_data_length:i]
#             dataX.append(group_x)  # dataX 리스트에 추가
#
#         if len(dataX) < 100:
#             return None, None, None
#
#         # quit()
#         X_test = np.array(dataX)
#         row = X_test.shape[1]
#         col = X_test.shape[2]
#
#         X_test = X_test.astype('float32').reshape(-1, row, col, 1)
#
#         return X_test, closeprice, min_max_scaler(ohlcv_data[input_data_length:, [1]]), datetime_list


def made_x(file, input_data_length, model_num, check_span, get_fig, crop_size=None, sudden_death=0):

    ohlcv_excel = pd.read_excel(dir + file, index_col=0)

    ohlcv_excel['MA20'] = ohlcv_excel['close'].rolling(20).mean()
    ohlcv_excel['CMO'] = cmo(ohlcv_excel)
    ohlcv_excel['OBV'] = obv(ohlcv_excel)
    ohlcv_excel['RSI'] = rsi(ohlcv_excel)
    macd(ohlcv_excel)

    # print(ohlcv_excel)
    # quit()

    #   이후 check_span 데이터와 현재 포인트를 비교해서 현재 포인트가 저가인지 고가인지 예측한다.
    #   진입, 저점, 고점, 거래 안함의 y_label 인 trade_state  >> [1, 2, 0]
    #   저점과 고점은 최대 3개의 중복 값을 허용한다.
    trade_state = [np.NaN] * len(ohlcv_excel)
    for i in range(len(ohlcv_excel) - check_span):
        #   저점
        if ohlcv_excel['close'][i - check_span:i].min() >= ohlcv_excel['close'][i]:
            if ohlcv_excel['close'][i + 1:i + 1 + check_span].min() >= ohlcv_excel['close'][i]:
                # if ohlcv_excel['close'][i - check_span:i + 1 + check_span].value_counts().sort_index().iloc[0] <= 3:
                if 1 not in trade_state[i - check_span:i]:
                    trade_state[i] = 1
                else:
                    trade_state[i] = 0
                # else:
                #     trade_state[i] = 0
            else:
                trade_state[i] = 0
        #   고점
        elif ohlcv_excel['close'][i - check_span:i].max() <= ohlcv_excel['close'][i]:
            if ohlcv_excel['close'][i + 1:i + 1 + check_span].max() <= ohlcv_excel['close'][i]:
                # if ohlcv_excel['close'][i - check_span:i + 1 + check_span].value_counts().sort_index().iloc[-1] <= 3:
                if 2 not in trade_state[i - check_span:i]:
                    trade_state[i] = 2
                else:
                    trade_state[i] = 0
            # else:
            #     trade_state[i] = 0
            else:
                trade_state[i] = 0

        #   거래 안함
        else:
            trade_state[i] = 0

    ohlcv_excel['trade_state'] = trade_state

    # ----------- dataX, dataY 추출하기 -----------#
    # print(ohlcv_excel.info())
    # quit()
    # ohlcv_excel.to_excel('test.xlsx')

    # NaN 제외하고 데이터 자르기 (데이터가 PIXEL 로 들어간다고 생각하면 된다)
    #   OBV : -CHECK_SPAN
    ohlcv_data = ohlcv_excel.values[sum(ohlcv_excel.MACD_Signal.isna()): -check_span].astype(np.float)
    # ohlcv_data = ohlcv_excel.values[1: -check_span].astype(np.float)
    # ohlcv_data = ohlcv_excel.values[sum(ohlcv_excel.CMO.isna()): -check_span].astype(np.float)
    # ohlcv_data = ohlcv_excel.values[sum(ohlcv_excel.RSI.isna()): -check_span].astype(np.float)

    # print(pd.DataFrame(ohlcv_data).info())
    # print(pd.DataFrame(ohlcv_data).to_excel('test.xlsx'))
    # print(list(map(float, ohlcv_data[0])))
    # quit()

    #       출력되는 Y 데이터 길이를 동일화 시킨다.     #
    # ohlcv_data = ohlcv_data[-300:]

    # 결측 데이터 제외
    if len(ohlcv_data) != 0:

        #          데이터 전처리         #
        #   Fixed X_data    #
        price = ohlcv_data[:, :4]
        volume = ohlcv_data[:, [4]]
        MA20 = ohlcv_data[:, [-8]]
        CMO = ohlcv_data[:, [-7]]
        OBV = ohlcv_data[:, [-6]]
        RSI = ohlcv_data[:, [-5]]
        MACD = ohlcv_data[:, [-4]]
        MACD_Signal = ohlcv_data[:, [-3]]
        MACD_OSC = ohlcv_data[:, [-2]]

        #   Flexible Y_data    #
        trade_state = ohlcv_data[:, [-1]]

        # scaled_price = min_max_scaler(price)
        # scaled_volume = min_max_scaler(volume)
        # scaled_CMO = min_max_scaler(CMO)
        # scaled_OBV = min_max_scaler(OBV)
        # scaled_RSI = min_max_scaler(RSI)

        # x = np.concatenate((scaled_price, scaled_volume, scaled_OBV), axis=1)  # axis=1, 세로로 합친다
        # x = scaled_price
        y = trade_state
        # print(x.shape, y_low.shape)  # (258, 6) (258, 1)
        # quit()

        dataX = []  # input_data length 만큼 담을 dataX 그릇
        dataY = []  # Target 을 담을 그릇
        for i in range(crop_size, len(ohlcv_data)):  # 마지막 데이터까지 다 긇어모은다.
            group_x = ohlcv_data[i - crop_size: i]
            group_y = y[i]
            price = group_x[:, :4]
            volume = group_x[:, [4]]
            MA20 = group_x[:, [-8]]
            CMO = group_x[:, [-7]]
            OBV = group_x[:, [-6]]
            RSI = group_x[:, [-5]]
            MACD = group_x[:, [-4]]
            MACD_Signal = group_x[:, [-3]]
            MACD_OSC = group_x[:, [-2]]
            scaled_price = min_max_scaler(price)
            scaled_volume = min_max_scaler(volume)
            scaled_MA20 = min_max_scaler(MA20)
            scaled_CMO = min_max_scaler(CMO)
            scaled_OBV = min_max_scaler(OBV)
            scaled_RSI = min_max_scaler(RSI)
            scaled_MACD = min_max_scaler(MACD)
            scaled_MACD_Signal = min_max_scaler(MACD_Signal)
            scaled_MACD_OSC = min_max_scaler(MACD_OSC)
            x = np.concatenate((scaled_price, scaled_volume, scaled_MA20, scaled_CMO, scaled_OBV, scaled_RSI,
                                scaled_MACD, scaled_MACD_Signal, scaled_MACD_OSC), axis=1)
            # x = scaled_x + sudden_death  # axis=1, 세로로 합친다
            group_x = x[-input_data_length:]
            # print(group_x[0])
            # quit()

            #   데이터 값에 결측치가 존재하는 경우 #
            if sum(sum(np.isnan(group_x))) > 0:
                return None, None, None

            dataX.append(group_x)  # dataX 리스트에 추가
            dataY.append(group_y)

        if len(dataX) < 100:
            print('len(dataX) < 100')
            return None, None, None

        # X_test = np.array(dataX)
        # row = X_test.shape[1]
        # col = X_test.shape[2]
        #
        # X_test = X_test.astype('float32').reshape(-1, row, col, 1)
        # print(X_test.shape)

        #       Exstracting fiexd X_data       #
        sliced_ohlcv = min_max_scaler(ohlcv_data[crop_size:, :x.shape[1]])

        #                      Get Figure                     #
        if get_fig == 1:
            spanlist_low = []
            spanlist_high = []

            for m in range(len(trade_state)):
                if (trade_state[m] > 0.5) and (trade_state[m] < 1.5):
                    if m + 1 < len(trade_state):
                        spanlist_low.append((m, m + 1))
                    else:
                        spanlist_low.append((m - 1, m))

            for m in range(len(trade_state)):
                if (trade_state[m] > 1.5) and (trade_state[m] < 2.5):
                    if m + 1 < len(trade_state):
                        spanlist_high.append((m, m + 1))
                    else:
                        spanlist_high.append((m - 1, m))

            # ----------- 인덱스 초기화 됨 -----------#

            # ----------- 공통된 Chart 그리기 -----------#

            plt.subplot(211)
            plt.plot(min_max_scaler(ohlcv_data[:, 1:2]), 'r', label='close')
            # plt.plot(scaled_OBV, 'b', label='OBV')
            plt.legend(loc='upper right')
            for i in range(len(spanlist_low)):
                plt.axvspan(spanlist_low[i][0], spanlist_low[i][1], facecolor='m', alpha=0.5)

            plt.subplot(212)
            plt.plot(min_max_scaler(ohlcv_data[:, 1:2]), 'r', label='close')
            # plt.plot(scaled_OBV, 'b', label='OBV')
            plt.legend(loc='upper right')
            for i in range(len(spanlist_high)):
                plt.axvspan(spanlist_high[i][0], spanlist_high[i][1], facecolor='c', alpha=0.5)

            Date = file.split()[0]
            Coin = file.split()[1].split('.')[0]
            plt.savefig('./Figure_data/%s_%s/%s %s.png' % (input_data_length, model_num, Date, Coin), dpi=500)
            plt.close()
            # plt.show()
            # ----------- Chart 그리기 -----------#

        return dataX, dataY, sliced_ohlcv


# def made_x_origin(file, input_data_length, model_num, check_span, get_fig, crop_size=0, sudden_death=0):
#
#     if type(file) is str:
#         ohlcv_excel = pd.read_excel(dir + file, index_col=0)
#
#         # ohlcv_excel['MA60'] = ohlcv_excel['close'].rolling(60).mean()
#         # ohlcv_excel['CMO'] = cmo(ohlcv_excel)
#         # ohlcv_excel['OBV'] = obv(ohlcv_excel)
#         # ohlcv_excel['RSI'] = rsi(ohlcv_excel)
#
#         # print(ohlcv_excel)
#         # quit()
#
#         #   이후 check_span 데이터와 현재 포인트를 비교해서 현재 포인트가 저가인지 고가인지 예측한다.
#         #   진입, 저점, 고점, 거래 안함의 y_label 인 trade_state  >> [1, 2, 0]
#         #   저점과 고점은 최대 3개의 중복 값을 허용한다.
#         trade_state = [np.NaN] * len(ohlcv_excel)
#         for i in range(len(ohlcv_excel) - check_span):
#             #   저점
#             if ohlcv_excel['close'][i + 1:i + 1 + check_span].min() >= ohlcv_excel['close'][i]:
#                 if ohlcv_excel['close'][i:i + 1 + check_span].value_counts().sort_index().iloc[0] <= 3:
#                     trade_state[i] = 1
#                 else:
#                     trade_state[i] = 0
#             #   고점
#             elif ohlcv_excel['close'][i + 1:i + 1 + check_span].max() <= ohlcv_excel['close'][i]:
#                 if ohlcv_excel['close'][i:i + 1 + check_span].value_counts().sort_index().iloc[-1] <= 3:
#                     trade_state[i] = 2
#                 else:
#                     trade_state[i] = 0
#             #   거래 안함
#             else:
#                 trade_state[i] = 0
#
#         ohlcv_excel['trade_state'] = trade_state
#
#         # ----------- dataX, dataY 추출하기 -----------#
#         # print(ohlcv_excel.info())
#         # ohlcv_excel.to_excel('test.xlsx')
#         # quit()
#
#         # NaN 제외하고 데이터 자르기 (데이터가 PIXEL 로 들어간다고 생각하면 된다)
#         #   OBV : -CHECK_SPAN
#         ohlcv_data = ohlcv_excel.values[: -check_span].astype(np.float)
#         # ohlcv_data = ohlcv_excel.values[ohlcv_excel['MA60'].isnull().sum():-check_span - 150].astype(np.float)
#         # ohlcv_data = ohlcv_excel.values[1: -check_span].astype(np.float)
#         # ohlcv_data = ohlcv_excel.values[sum(ohlcv_excel.CMO.isna()): -check_span].astype(np.float)
#         # ohlcv_data = ohlcv_excel.values[sum(ohlcv_excel.RSI.isna()): -check_span].astype(np.float)
#
#         # print(pd.DataFrame(ohlcv_data).info())
#         # print(pd.DataFrame(ohlcv_data).to_excel('test.xlsx'))
#         # print(list(map(float, ohlcv_data[0])))
#         # quit()
#     else:
#         ohlcv_data = file
#
#     #           ohlcv_data slicing      #
#     # ohlcv_data = ohlcv_data[-crop_size:]
#
#     # 결측 데이터 제외
#     if len(ohlcv_data) != 0:
#
#         #          데이터 전처리         #
#         #   Fixed X_data    #
#         price = ohlcv_data[:, :4]
#         # volume = ohlcv_data[:, [4]]
#         # MA60 = ohlcv_data[:, [-3]]
#         # CMO = ohlcv_data[:, [-2]]
#         # OBV = ohlcv_data[:, [-2]]
#         # RSI = ohlcv_data[:, [-2]]
#
#         #   Flexible Y_data    #
#         trade_state = ohlcv_data[:, [-1]]
#
#         scaled_price = min_max_scaler(price)
#         # scaled_volume = min_max_scaler(volume)
#         # scaled_MA60 = min_max_scaler(MA60)
#         # scaled_CMO = min_max_scaler(CMO)
#         # scaled_OBV = min_max_scaler(OBV)
#         # scaled_RSI = min_max_scaler(RSI)
#
#         # x = np.concatenate((scaled_price, scaled_MA60), axis=1)  # axis=1, 세로로 합친다
#         x = scaled_price
#         y = trade_state
#         # print(x.shape, y_low.shape)  # (258, 6) (258, 1)
#         # quit()
#
#         dataX = []  # input_data length 만큼 담을 dataX 그릇
#         dataY = []  # Target 을 담을 그릇
#         for i in range(input_data_length, len(x)):
#             group_x = x[i - input_data_length:i]
#             group_y = y[i]
#
#             dataX.append(group_x)  # dataX 리스트에 추가
#             dataY.append(group_y)
#
#         if len(dataX) < 100:
#             return None, None, None
#
#         X_test = np.array(dataX)
#         row = X_test.shape[1]
#         col = X_test.shape[2]
#
#         X_test = X_test.astype('float32').reshape(-1, row, col, 1)
#         # print(X_test.shape)
#
#         #       Exstracting fiexd X_data       #
#         sliced_ohlcv = ohlcv_data[input_data_length:, :x.shape[1]]
#
#         #                      Get Figure                     #
#         if get_fig == 1:
#             spanlist_low = []
#             spanlist_high = []
#
#             for m in range(len(trade_state)):
#                 if (trade_state[m] > 0.5) and (trade_state[m] < 1.5):
#                     if m + 1 < len(trade_state):
#                         spanlist_low.append((m, m + 1))
#                     else:
#                         spanlist_low.append((m - 1, m))
#
#             for m in range(len(trade_state)):
#                 if (trade_state[m] > 1.5) and (trade_state[m] < 2.5):
#                     if m + 1 < len(trade_state):
#                         spanlist_high.append((m, m + 1))
#                     else:
#                         spanlist_high.append((m - 1, m))
#
#             # ----------- 인덱스 초기화 됨 -----------#
#
#             # ----------- 공통된 Chart 그리기 -----------#
#
#             plt.subplot(211)
#             plt.plot(min_max_scaler(ohlcv_data[:, 1:2]), 'r', label='close')
#             # plt.plot(scaled_OBV, 'b', label='OBV')
#             plt.legend(loc='upper right')
#             for i in range(len(spanlist_low)):
#                 plt.axvspan(spanlist_low[i][0], spanlist_low[i][1], facecolor='c', alpha=0.5)
#
#             plt.subplot(212)
#             plt.plot(min_max_scaler(ohlcv_data[:, 1:2]), 'r', label='close')
#             # plt.plot(scaled_OBV, 'b', label='OBV')
#             plt.legend(loc='upper right')
#             for i in range(len(spanlist_high)):
#                 plt.axvspan(spanlist_high[i][0], spanlist_high[i][1], facecolor='m', alpha=0.5)
#
#             Date = file.split()[0]
#             Coin = file.split()[1].split('.')[0]
#             plt.savefig('./Figure_data/%s_%s/%s %s.png' % (input_data_length, model_num, Date, Coin), dpi=500)
#             plt.close()
#             # plt.show()
#             # ----------- Chart 그리기 -----------#
#
#         return X_test, dataY, sliced_ohlcv


if __name__ == '__main__':

    # ----------- Params -----------#
    input_data_length = 50
    model_num = '50'

    #       Make folder      #
    try:
        os.mkdir('./Figure_data/%s_%s/' % (input_data_length, model_num))

    except Exception as e:
        pass
    check_span = 30
    get_fig = 1

    Made_X = []
    Made_Y = []

    # ohlcv_list = ['2020-01-10 ETH ohlcv.xlsx']

    for file in ohlcv_list:

        if int(file.split()[0].split('-')[1]) != 1:
            continue

        result = made_x(file, input_data_length, model_num, check_span, get_fig, crop_size=input_data_length)
        # result = low_high('dvp'.upper(), input_data_length, lowhigh_point='on')
        # print(result)
        # quit()

        # ------------ 데이터가 있으면 dataX, dataY 병합하기 ------------#
        if result is not None:
            if result[0] is not None:

                Made_X += result[0]
                Made_Y += result[1]

            # 누적 데이터량 표시
            print(file, len(Made_X))

    # SAVING X, Y
    X = np.array(Made_X)
    Y = np.array(Made_Y)
    np.save('./Made_X/Made_X %s_%s' % (input_data_length, model_num), X)
    np.save('./Made_X/Made_Y %s_%s' % (input_data_length, model_num), Y)
    # np.save('./Made_X_high/Made_Y %s_%s' % (input_data_length, model_num), Y_high)

