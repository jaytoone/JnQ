import numpy as np
import pandas as pd
import pybithumb
import os
import warnings
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from Funcs_MACD_OSC import ema_cross, cmo, obv, rsi, macd, supertrend
import matplotlib.pyplot as plt
import sys

np.set_printoptions(threshold=sys.maxsize)
warnings.filterwarnings("ignore")
pd.set_option('display.max_rows', 1500)


home_dir = os.path.expanduser('~')
dir = home_dir + '/OneDrive/CoinBot/ohlcv/'
ohlcv_list = os.listdir(dir)


def min_max_scaler(x):
    scaled_x = (x - x.min()) / (x.max() - x.min())
    return scaled_x


def max_abs_scaler(x):
    scaled_x = x / abs(x).max()
    return scaled_x


def low_high(file, input_data_length, crop_size=None):

    if type(file) == str:
        ohlcv_excel = pd.read_excel(dir + file, index_col=0)
        Date = file.split()[0]
        Coin = file.split()[1].split('.')[0]
    else:
        ohlcv_excel = file
        Date = str(datetime.now()).split()[0]
        Coin = file.index.name

    ohlcv_excel['CMO'] = cmo(ohlcv_excel, period=11)
    ohlcv_excel['OBV'] = obv(ohlcv_excel)
    ohlcv_excel['RSI'] = rsi(ohlcv_excel, period=14)
    macd(ohlcv_excel, short=20, long=33, signal=5)
    # macd(ohlcv_excel, short=30, long=60, signal=30)

    # print(ohlcv_excel.columns)
    # quit()

    #   역순, 정순 라벨링
    trade_state = [0] * len(ohlcv_excel)
    for i in range(1, len(ohlcv_excel)):
        if ohlcv_excel['MACD_OSC'][i - 1] <= 0:
            if ohlcv_excel['MACD_OSC'][i] > 0:
                trade_state[i] = 1

        if ohlcv_excel['MACD_OSC'][i - 1] >= 0:
            if ohlcv_excel['MACD_OSC'][i] < 0:
                trade_state[i] = 3

    for i in range(len(trade_state)):
        if trade_state[i] == 1.:  # 1 찾았을 때
            for j in range(i + 1, len(trade_state)):
                if trade_state[j] == 3.:  # 뒤에 2 존재한다면,
                    # print(i, j)
                    max_index = np.argmax(ohlcv_excel['MACD_OSC'][i:j + 1].values)
                    if trade_state[i + max_index] != 1:  # 실시간 거래에서는 최고점을 알 수 없으니 1이 아닌 경우, 2로 표시한다.
                        trade_state[i + max_index] = 2
                    # print(trade_state[i:j + 1])
                    # print(ohlcv_excel['MACD_OSC'][i:j + 1].values)
                    # quit()
                    break

    ohlcv_excel['trade_state'] = trade_state

    # print(trade_state)
    # quit()
    # print(ohlcv_excel.info())
    # quit()

    # NaN 제외하고 데이터 자르기 (데이터가 PIXEL 로 들어간다고 생각하면 된다)
    ohlcv_data = ohlcv_excel.values[sum(ohlcv_excel.MACD_Signal.isna()):].astype(np.float)
    # print(sum(np.isnan(ohlcv_data)))
    # quit()

    # 결측 데이터 제외
    if len(ohlcv_data) != 0:

        #          데이터 전처리         #
        #   Fixed X_data    #
        # price ma
        # ohlcv_data[:, [0, 1, 2, 3, 5]] = min_max_scaler(ohlcv_data[:, [0, 1, 2, 3, 5]])
        # volume
        ohlcv_data[:, [4]] = min_max_scaler(ohlcv_data[:, [4]])
        #   CMO
        ohlcv_data[:, [-8]] = max_abs_scaler(ohlcv_data[:, [-8]])
        #   OBV
        ohlcv_data[:, [-7]] = min_max_scaler(ohlcv_data[:, [-7]])
        #   RSI
        ohlcv_data[:, [-6]] = min_max_scaler(ohlcv_data[:, [-6]])
        #   MACD
        ohlcv_data[:, -5:-1] = max_abs_scaler(ohlcv_data[:, -5:-1])

        #   Flexible Y_data    #
        trade_state = ohlcv_data[:, [-1]]
        y = trade_state
        # print(x.shape, y_low.shape)  # (258, 6) (258, 1)
        # quit()
        # print(ohlcv_data)
        # quit()

        dataX = []  # input_data length 만큼 담을 dataX 그릇
        dataY = []  # Target 을 담을 그릇
        for i in range(crop_size, len(ohlcv_data)):  # 마지막 데이터까지 다 긇어모은다.

            group_x = ohlcv_data[i - crop_size: i]
            group_y = y[i]
            price = group_x[:, :4]
            volume = group_x[:, [4]]

            scaled_pricema = min_max_scaler(price)

            CMO = group_x[:, [-8]]
            OBV = group_x[:, [-7]]
            RSI = group_x[:, [-6]]
            MACD = group_x[:, -5:-1]

            x = np.concatenate((scaled_pricema[:, :4], volume, scaled_pricema[:, 4:], CMO, OBV, RSI, MACD), axis=1)
            # x = scaled_x + sudden_death  # axis=1, 세로로 합친다
            group_x = x[-input_data_length:]

            #   데이터 값에 결측치가 존재하는 경우 #
            # if sum(sum(np.isnan(group_x))) > 0:
            #     continue

            dataX.append(group_x)  # dataX 리스트에 추가

        # if len(dataX) < 100:
        #     return None, None

        return dataX, ohlcv_data[crop_size:, ]


def low_high01(file, input_data_length, crop_size=None):

    if type(file) == str:
        ohlcv_excel = pd.read_excel(dir + file, index_col=0)
        Date = file.split()[0]
        Coin = file.split()[1].split('.')[0]
    else:
        ohlcv_excel = file
        Date = str(datetime.now()).split()[0]
        Coin = file.index.name

    ohlcv_excel['CMO'] = cmo(ohlcv_excel, period=9)
    ohlcv_excel['OBV'] = obv(ohlcv_excel)
    ohlcv_excel['RSI'] = rsi(ohlcv_excel, period=14)
    # macd(ohlcv_excel, short=100, long=168, signal=32)
    macd(ohlcv_excel, short=20, long=33, signal=5)
    # macd(ohlcv_excel, short=30, long=60, signal=30)

    # print(ohlcv_excel.columns)
    # quit()

    #   역순, 정순 라벨링
    trade_state = [0] * len(ohlcv_excel)
    for i in range(1, len(ohlcv_excel)):
        if ohlcv_excel['MACD_OSC'][i - 1] <= 0:
            if ohlcv_excel['MACD_OSC'][i] > 0:
                trade_state[i] = 1

        if ohlcv_excel['MACD_OSC'][i - 1] >= 0:
            if ohlcv_excel['MACD_OSC'][i] < 0:
                trade_state[i] = 3

    for i in range(len(trade_state)):
        if trade_state[i] == 1.:  # 1 찾았을 때
            for j in range(i + 1, len(trade_state)):
                if trade_state[j] == 3.:  # 뒤에 2 존재한다면,
                    # print(i, j)
                    max_index = np.argmax(ohlcv_excel['MACD_OSC'][i:j + 1].values)
                    if trade_state[i + max_index] != 1:  # 실시간 거래에서는 최고점을 알 수 없으니 1이 아닌 경우, 2로 표시한다.
                        trade_state[i + max_index] = 2
                    # print(trade_state[i:j + 1])
                    # print(ohlcv_excel['MACD_OSC'][i:j + 1].values)
                    # quit()
                    break

    for n, i in enumerate(trade_state):
        if i in [1, 3]:
            trade_state[n] = 0
        elif i in [2]:
            trade_state[n] = 1

    ohlcv_excel['trade_state'] = trade_state

    # print(trade_state)
    # quit()
    # print(ohlcv_excel.info())
    # quit()

    # NaN 제외하고 데이터 자르기 (데이터가 PIXEL 로 들어간다고 생각하면 된다)
    ohlcv_data = ohlcv_excel.values[sum(ohlcv_excel.MACD_Signal.isna()):].astype(np.float)
    # print(sum(np.isnan(ohlcv_data)))
    # quit()

    # 결측 데이터 제외
    if len(ohlcv_data) != 0:

        #          데이터 전처리         #
        #   Fixed X_data    #
        # price ma
        # ohlcv_data[:, [0, 1, 2, 3, 5]] = min_max_scaler(ohlcv_data[:, [0, 1, 2, 3, 5]])
        # volume
        ohlcv_data[:, [4]] = min_max_scaler(ohlcv_data[:, [4]])
        #   CMO
        ohlcv_data[:, [-8]] = max_abs_scaler(ohlcv_data[:, [-8]])
        #   OBV
        ohlcv_data[:, [-7]] = min_max_scaler(ohlcv_data[:, [-7]])
        #   RSI
        ohlcv_data[:, [-6]] = min_max_scaler(ohlcv_data[:, [-6]])
        #   MACD
        ohlcv_data[:, -5:-1] = max_abs_scaler(ohlcv_data[:, -5:-1])

        #   Flexible Y_data    #
        trade_state = ohlcv_data[:, [-1]]
        y = trade_state
        # print(x.shape, y_low.shape)  # (258, 6) (258, 1)
        # quit()
        # print(ohlcv_data)
        # quit()

        dataX = []  # input_data length 만큼 담을 dataX 그릇
        dataY = []  # Target 을 담을 그릇
        for i in range(crop_size, len(ohlcv_data)):  # 마지막 데이터까지 다 긇어모은다.

            group_x = ohlcv_data[i - crop_size: i]
            group_y = y[i]
            price = group_x[:, :4]
            volume = group_x[:, [4]]

            scaled_price = min_max_scaler(price)

            CMO = group_x[:, [-8]]
            OBV = group_x[:, [-7]]
            RSI = group_x[:, [-6]]
            MACD = group_x[:, -5:-1]
            # scaled_MACD = max_abs_scaler(MACD)

            x = np.concatenate((scaled_price, volume, CMO, OBV, RSI, MACD), axis=1)
            # x = scaled_x + sudden_death  # axis=1, 세로로 합친다
            group_x = x[-input_data_length:]

            #   데이터 값에 결측치가 존재하는 경우 #
            # if sum(sum(np.isnan(group_x))) > 0:
            #     continue

            dataX.append(group_x)  # dataX 리스트에 추가

        # if len(dataX) < 100:
        #     return None, None

        return dataX, ohlcv_data[crop_size:, ]


def high(file, input_data_length, crop_size=None):

    if type(file) == str:
        ohlcv_excel = pd.read_excel(dir + file, index_col=0)
        Date = file.split()[0]
        Coin = file.split()[1].split('.')[0]
    else:
        ohlcv_excel = file
        Date = str(datetime.now()).split()[0]
        Coin = file.index.name

    ohlcv_excel['CMO'] = cmo(ohlcv_excel, period=9)
    ohlcv_excel['OBV'] = obv(ohlcv_excel)
    ohlcv_excel['RSI'] = rsi(ohlcv_excel, period=14)
    macd(ohlcv_excel, short=105, long=163, signal=38)
    # macd(ohlcv_excel, short=30, long=60, signal=30)

    # print(ohlcv_excel.columns)
    # quit()

    #   역순, 정순 라벨링
    trade_state = [np.nan] * len(ohlcv_excel)
    for i in range(1, len(ohlcv_excel)):
        if ohlcv_excel['MACD_OSC'][i - 1] <= 0:
            if ohlcv_excel['MACD_OSC'][i] > 0:
                trade_state[i] = 1

        if ohlcv_excel['MACD_OSC'][i - 1] >= 0:
            if ohlcv_excel['MACD_OSC'][i] < 0:
                trade_state[i] = 3

    for i in range(len(trade_state)):
        if trade_state[i] == 1.:  # 1 찾았을 때
            for j in range(i + 1, len(trade_state)):
                if trade_state[j] == 3.:  # 뒤에 2 존재한다면,
                    # print(i, j)
                    for k in range(i, j):
                        #   전후로 각각 3개의 종가보다 크다면 fractal
                        check_span = 5
                        if ohlcv_excel['close'][k - check_span:k].max() <= ohlcv_excel['close'][k]:
                            if ohlcv_excel['close'][k + 1:k + 1 + check_span].max() <= ohlcv_excel['close'][k]:
                                if 2 not in trade_state[i - check_span:i]:
                                    trade_state[k] = 0

                    max_index = np.argmax(ohlcv_excel['close'][i:j].values)
                    trade_state[i + max_index] = 2

                    # print(trade_state[i:j + 1])
                    # quit()
                    # print(trade_state[i:j + 1])
                    # print(ohlcv_excel['MACD_OSC'][i:j + 1].values)
                    # quit()
                    break

    for n, i in enumerate(trade_state):
        if i in [1, 3]:
            trade_state[n] = np.nan
        elif i in [2]:
            trade_state[n] = 1

    ohlcv_excel['trade_state'] = trade_state

    # print(trade_state)
    # quit()
    # print(ohlcv_excel.info())
    # quit()

    # NaN 제외하고 데이터 자르기 (데이터가 PIXEL 로 들어간다고 생각하면 된다)
    ohlcv_data = ohlcv_excel.values[sum(ohlcv_excel.MACD_Signal.isna()):].astype(np.float)
    # print(sum(np.isnan(ohlcv_data)))
    # quit()

    # 결측 데이터 제외
    if len(ohlcv_data) != 0:

        #          데이터 전처리         #
        #   Fixed X_data    #
        # price ma
        # ohlcv_data[:, [0, 1, 2, 3, 5]] = min_max_scaler(ohlcv_data[:, [0, 1, 2, 3, 5]])
        # volume
        ohlcv_data[:, [4]] = min_max_scaler(ohlcv_data[:, [4]])
        #   CMO
        ohlcv_data[:, [-8]] = max_abs_scaler(ohlcv_data[:, [-8]])
        #   OBV
        ohlcv_data[:, [-7]] = min_max_scaler(ohlcv_data[:, [-7]])
        #   RSI
        ohlcv_data[:, [-6]] = min_max_scaler(ohlcv_data[:, [-6]])
        #   MACD
        ohlcv_data[:, -5:-1] = max_abs_scaler(ohlcv_data[:, -5:-1])

        #   Flexible Y_data    #
        trade_state = ohlcv_data[:, [-1]]
        y = trade_state
        # print(x.shape, y_low.shape)  # (258, 6) (258, 1)
        # quit()
        # print(ohlcv_data)
        # quit()

        dataX = []  # input_data length 만큼 담을 dataX 그릇
        dataY = []  # Target 을 담을 그릇
        for i in range(crop_size, len(ohlcv_data)):  # 마지막 데이터까지 다 긇어모은다.

            group_x = ohlcv_data[i - crop_size: i]
            group_y = y[i]
            price = group_x[:, :4]
            volume = group_x[:, [4]]

            scaled_pricema = min_max_scaler(price)

            CMO = group_x[:, [-8]]
            OBV = group_x[:, [-7]]
            RSI = group_x[:, [-6]]
            MACD = group_x[:, -5:-1]

            x = np.concatenate((scaled_pricema[:, :4], volume, scaled_pricema[:, 4:], CMO, OBV, RSI, MACD), axis=1)
            # x = scaled_x + sudden_death  # axis=1, 세로로 합친다
            group_x = x[-input_data_length:]

            #   데이터 값에 결측치가 존재하는 경우 #
            # if sum(sum(np.isnan(group_x))) > 0:
            #     continue

            dataX.append(group_x)  # dataX 리스트에 추가

        # if len(dataX) < 100:
        #     return None, None

        return dataX, ohlcv_data[crop_size:, ]


def made_x(file, input_data_length, model_num, crop_size=None):

    if type(file) == str:
        ohlcv_excel = pd.read_excel(dir + file, index_col=0)
        Date = file.split()[0]
        Coin = file.split()[1].split('.')[0]
    else:
        ohlcv_excel = file
        Date = str(datetime.now()).split()[0]
        Coin = file.index.name

    ohlcv_excel['CMO'] = cmo(ohlcv_excel, period=9)
    ohlcv_excel['OBV'] = obv(ohlcv_excel)
    ohlcv_excel['RSI'] = rsi(ohlcv_excel, period=14)
    macd(ohlcv_excel, short=100, long=158, signal=32)
    # supertrend(ohlcv_excel)
    # macd(ohlcv_excel, short=30, long=60, signal=30)

    # print(ohlcv_excel.columns)
    # quit()

    #   역순, 정순 라벨링
    trade_state = [np.nan] * len(ohlcv_excel)
    for i in range(1, len(ohlcv_excel)):
        if ohlcv_excel['MACD_OSC'][i - 1] <= 0:
            if ohlcv_excel['MACD_OSC'][i] > 0:
                trade_state[i] = 1

        if ohlcv_excel['MACD_OSC'][i - 1] >= 0:
            if ohlcv_excel['MACD_OSC'][i] < 0:
                trade_state[i] = 3

    for i in range(len(trade_state)):
        if trade_state[i] == 1.:  # 1 찾았을 때
            for j in range(i + 1, len(trade_state)):
                if trade_state[j] == 3.:  # 뒤에 2 존재한다면,
                    # print(i, j)
                    for k in range(i, j):
                        trade_state[k] = 0
                    #     #   전후로 각각 3개의 종가보다 크다면 fractal
                    #     check_span = 5
                    #     if ohlcv_excel['close'][k - check_span:k].max() <= ohlcv_excel['close'][k]:
                    #         if ohlcv_excel['close'][k + 1:k + 1 + check_span].max() <= ohlcv_excel['close'][k]:
                    #             if 2 not in trade_state[i - check_span:i]:
                    #                 trade_state[k] = 0
                    max_index = np.argmax(ohlcv_excel['MACD_OSC'][i:j].values)
                    trade_state[i + max_index] = 2

                    # print(trade_state[i:j + 1])
                    # quit()
                    # print(trade_state[i:j + 1])
                    # print(ohlcv_excel['MACD_OSC'][i:j + 1].values)
                    # quit()
                    break

    for n, i in enumerate(trade_state):
        if i in [1, 3]:
            trade_state[n] = np.nan
        elif i in [2]:
            trade_state[n] = 1

    ohlcv_excel['trade_state'] = trade_state

    # print(trade_state)
    # quit()

    # print(ohlcv_excel.iloc[:, [-2]])
    # plt.plot(ohlcv_excel.iloc[:, [-2]].values)
    # # plt.plot(df['MACD_Zero'].values, 'g')
    # # span_list = list()
    # # span_list2 = list()
    # # for i in range(len(trade_state)):
    # #     if trade_state[i] == 1.:
    # #         span_list.append((i, i + 1))
    # #     elif trade_state[i] == 2.:
    # #         span_list2.append((i, i + 1))
    # #
    # # for i in range(len(span_list)):
    # #     plt.axvspan(span_list[i][0], span_list[i][1], facecolor='c', alpha=0.7)
    # # for i in range(len(span_list2)):
    # #     plt.axvspan(span_list2[i][0], span_list2[i][1], facecolor='m', alpha=0.7)
    #
    # plt.show()
    # quit()

    # NaN 제외하고 데이터 자르기 (데이터가 PIXEL 로 들어간다고 생각하면 된다)
    ohlcv_data = ohlcv_excel.values[sum(ohlcv_excel.MACD_Signal.isna()):].astype(np.float)
    # print(sum(np.isnan(ohlcv_data)))
    # print(np.isnan(ohlcv_data[:, [5]]))
    # quit()

    # print(ohlcv_data[-5:])
    # quit()

    # plt.plot(ohlcv_data[:, [1]])
    # plt.plot(ohlcv_data[:, [-2]], 'g')
    # span_list = list()
    # span_list2 = list()
    # for i in range(len(ohlcv_data[:, [-1]])):
    #     if ohlcv_data[:, [-1]][i] == 0.:
    #         span_list.append((i, i + 1))
    #     elif ohlcv_data[:, [-1]][i] == 1.:
    #         span_list2.append((i, i + 1))
    #
    # for i in range(len(span_list)):
    #     plt.axvspan(span_list[i][0], span_list[i][1], facecolor='c', alpha=0.7)
    # for i in range(len(span_list2)):
    #     plt.axvspan(span_list2[i][0], span_list2[i][1], facecolor='m', alpha=0.7)
    #
    # plt.show()
    # quit()

    # 결측 데이터 제외
    if len(ohlcv_data) != 0:

        #          데이터 전처리         #
        #   Fixed X_data    #
        # price ma
        ohlcv_data[:, [0, 1, 2, 3]] = min_max_scaler(ohlcv_data[:, [0, 1, 2, 3]])
        # volume
        ohlcv_data[:, [4]] = min_max_scaler(ohlcv_data[:, [4]])
        #   CMO
        ohlcv_data[:, [-8]] = max_abs_scaler(ohlcv_data[:, [-8]])
        #   OBV
        ohlcv_data[:, [-7]] = min_max_scaler(ohlcv_data[:, [-7]])
        #   RSI
        ohlcv_data[:, [-6]] = min_max_scaler(ohlcv_data[:, [-6]])
        #   MACD
        ohlcv_data[:, -5:-1] = max_abs_scaler(ohlcv_data[:, -5:-1])

        #   Flexible Y_data    #
        trade_state = ohlcv_data[:, [-1]]
        y = trade_state
        # print(x.shape, y_low.shape)  # (258, 6) (258, 1)
        # quit()
        # print(ohlcv_data)
        # quit()

        dataX = []  # input_data length 만큼 담을 dataX 그릇
        dataY = []  # Target 을 담을 그릇
        for i in range(crop_size, len(ohlcv_data)):  # 마지막 데이터까지 다 긇어모은다.

            if not np.isnan(y[i]):
                group_x = ohlcv_data[i - crop_size: i]
                group_y = y[i]
                price = group_x[:, :4]
                volume = group_x[:, [4]]
                CMO = group_x[:, [-8]]
                OBV = group_x[:, [-7]]
                RSI = group_x[:, [-6]]
                MACD = group_x[:, -5:-2]

                x = np.concatenate((price, volume, CMO, OBV, RSI, MACD), axis=1)
                # x = scaled_x + sudden_death  # axis=1, 세로로 합친다
                group_x = x[-input_data_length:]
                # print(group_x[0])
                # quit()

                #   데이터 값에 결측치가 존재하는 경우 #
                if sum(sum(np.isnan(group_x))) > 0:
                    continue

                dataX.append(group_x)  # dataX 리스트에 추가
                dataY.append(group_y)

        # if len(dataX) < 100:
        #     print('len(dataX) < 100')
        #     return None, None, None

        return dataX, dataY


if __name__ == '__main__':

    # ----------- Params -----------#
    input_data_length = 30
    model_num = '101'

    Made_X = []
    Made_Y = []

    # ohlcv_list = [pybithumb.get_ohlcv('GNT', interval='minute1')]
    # exist_list = list()
    # except_list = os.listdir('./Made_Chart_to_np/%s_%s/' % (input_data_length, model_num))
    # for except_file in except_list:
    #     pull_filename = except_file.split('.')[0] + ' ohlcv.xlsx'
    #     if pull_filename in ohlcv_list:
    #         # ohlcv_list.remove(pull_filename)
    #         exist_list.append(pull_filename)

    for file in ohlcv_list:

        try:
            if int(file.split()[0].split('-')[1]) not in [1, 2]:
                continue

            # Date = file.split()[0]
            # Coin = file.split()[1].split('.')[0]

            result = made_x(file, input_data_length, model_num, crop_size=input_data_length)
            # result = low_high('dvp'.upper(), input_data_length, lowhigh_point='on')
            # print(result)
            # quit()

            Made_X += result[0]
            Made_Y += result[1]
            print(file, len(Made_X))

            if len(Made_X) > 400000:
                break

        except Exception as e:
            print('Error in made_x :', e)

    np.save('./Made_X/Made_X %s_%s' % (input_data_length, model_num), np.array(Made_X))
    np.save('./Made_X/Made_Y %s_%s' % (input_data_length, model_num), np.array(Made_Y))
