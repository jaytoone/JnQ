import numpy as np
import pandas as pd
import pybithumb
import os
import warnings
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from Funcs_CNN4 import ema_cross, cmo, obv, rsi, macd
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

    ohlcv_excel['MA20'] = ohlcv_excel['close'].rolling(20).mean()
    ohlcv_excel['CMO'] = cmo(ohlcv_excel)
    ohlcv_excel['OBV'] = obv(ohlcv_excel)
    ohlcv_excel['RSI'] = rsi(ohlcv_excel)
    macd(ohlcv_excel, short=5, long=35, signal=5)

    # print(ohlcv_excel.columns)
    # quit()

    #   역순, 정순 라벨링
    trade_state = [0] * len(ohlcv_excel)
    for i in range(2, len(ohlcv_excel)):
        if ohlcv_excel['MACD'][i - 1] <= 0:
            if ohlcv_excel['MACD'][i] > 0:
                trade_state[i] = 1

        if ohlcv_excel['MACD'][i - 1] >= 0:
            if ohlcv_excel['MACD'][i] < 0:
                trade_state[i] = 2

    ohlcv_excel['trade_state'] = trade_state
    # print(trade_state)
    # quit()
    # print(ohlcv_excel)

    # NaN 제외하고 데이터 자르기 (데이터가 PIXEL 로 들어간다고 생각하면 된다)
    ohlcv_data = ohlcv_excel.values[sum(ohlcv_excel.MACD_Signal.isna()):].astype(np.float)

    y = ohlcv_data[:, [-1]]
    crop_span = np.full(len(y), np.NaN)
    for i in range(len(crop_span)):
        if y[i] == 1.:  # 1을 찾았을 때
            for j in range(i + 1, len(crop_span)):
                if y[j] == 2.:  # 뒤에 2가 존재한다면,
                    # print(i, j)
                    crop_span[i:j + 1] = 0

                    #   i + 1 부터 j + 1 까지 중 최고점은 1의 값을 부여한다.
                    max_index = np.argmax(ohlcv_data[i:j + 1, [-5]])
                    crop_span[i:j + 1][max_index:] = 1
                    # plt.plot(ohlcv_data[i:j + 1, [1]])
                    # plt.plot(ohlcv_data[i:j + 1, [5]], 'g')
                    # plt.plot(ohlcv_data[i:j + 1, [6]], 'r')
                    # plt.show()
                    # plt.close()
                    # quit()
                    break

    # plt.plot(ohlcv_data[:, [-5]])
    # plt.plot(ohlcv_data[:, [-2]], 'g')
    # span_list = list()
    # for i in range(len(crop_span)):
    #     if crop_span[i] == 1.:
    #         span_list.append((i, i + 1))
    #
    # for i in range(len(span_list)):
    #     plt.axvspan(span_list[i][0], span_list[i][1], facecolor='c', alpha=0.7)
    #
    # plt.show()
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

            if not np.isnan(crop_span[i]):
                group_x = ohlcv_data[i - crop_size: i]
                group_y = crop_span[i]
                price = group_x[:, :4]
                volume = group_x[:, [4]]
                MA20 = group_x[:, [5]]

                pricema = np.concatenate((price, MA20), axis=1)
                scaled_pricema = min_max_scaler(pricema)

                CMO = group_x[:, [-8]]
                OBV = group_x[:, [-7]]
                RSI = group_x[:, [-6]]
                MACD = group_x[:, -5:-1]

                x = np.concatenate((scaled_pricema[:, :4], volume, scaled_pricema[:, 4:], CMO, OBV, RSI, MACD), axis=1)
                # x = scaled_x + sudden_death  # axis=1, 세로로 합친다
                group_x = x[-input_data_length:]

                #   데이터 값에 결측치가 존재하는 경우 #
                if sum(sum(np.isnan(group_x))) > 0:
                    continue

                dataX.append(group_x)  # dataX 리스트에 추가

        if len(dataX) < 100:
            return None, None

        return dataX, dataY


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


def made_x(file, input_data_length, model_num, crop_size=None):

    if type(file) == str:
        ohlcv_excel = pd.read_excel(dir + file, index_col=0)
        Date = file.split()[0]
        Coin = file.split()[1].split('.')[0]
    else:
        ohlcv_excel = file
        Date = str(datetime.now()).split()[0]
        Coin = file.index.name

    ohlcv_excel['MA20'] = ohlcv_excel['close'].rolling(20).mean()
    ohlcv_excel['CMO'] = cmo(ohlcv_excel, period=60)
    ohlcv_excel['OBV'] = obv(ohlcv_excel)
    ohlcv_excel['RSI'] = rsi(ohlcv_excel, period=60)
    macd(ohlcv_excel, short=5, long=8, signal=5)
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
    # print(trade_state)
    # quit()
    ohlcv_excel['trade_state'] = trade_state

    # print(trade_state)
    # quit()
    # print(ohlcv_excel.info())
    # quit()

    # NaN 제외하고 데이터 자르기 (데이터가 PIXEL 로 들어간다고 생각하면 된다)
    ohlcv_data = ohlcv_excel.values[sum(ohlcv_excel.CMO.isna()):].astype(np.float)
    # print(sum(np.isnan(ohlcv_data)))
    # quit()

    # print(ohlcv_data)
    # quit()

    # print(crop_span)
    # plt.plot(ohlcv_data[:, [-3]])
    # plt.plot(ohlcv_data[:, [-2]], 'g')
    # span_list = list()
    # span_list2 = list()
    # for i in range(len(ohlcv_data[:, [-1]])):
    #     if ohlcv_data[:, [-1]][i] == 1.:
    #         span_list.append((i, i + 1))
    #     elif ohlcv_data[:, [-1]][i] == 2.:
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
        ohlcv_data[:, [0, 1, 2, 3, 5]] = min_max_scaler(ohlcv_data[:, [0, 1, 2, 3, 5]])
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
            MA20 = group_x[:, [5]]

            CMO = group_x[:, [-8]]
            OBV = group_x[:, [-7]]
            RSI = group_x[:, [-6]]
            MACD = group_x[:, -5:-1]

            x = np.concatenate((price, volume, MA20, CMO, OBV, RSI, MACD), axis=1)
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
    model_num = '88'

    Made_X = []
    Made_Y = []

    # ohlcv_list = [pybithumb.get_ohlcv('IPX', interval='minute1')]
    # exist_list = list()
    # except_list = os.listdir('./Made_Chart_to_np/%s_%s/' % (input_data_length, model_num))
    # for except_file in except_list:
    #     pull_filename = except_file.split('.')[0] + ' ohlcv.xlsx'
    #     if pull_filename in ohlcv_list:
    #         # ohlcv_list.remove(pull_filename)
    #         exist_list.append(pull_filename)

    for file in ohlcv_list:

        try:
            if int(file.split()[0].split('-')[1]) not in [1]:
                continue
            #
            # Date = file.split()[0]
            # Coin = file.split()[1].split('.')[0]

            result = made_x(file, input_data_length, model_num, crop_size=input_data_length)
            # result = low_high('dvp'.upper(), input_data_length, lowhigh_point='on')
            # print(result)
            # quit()

            Made_X += result[0]
            Made_Y += result[1]
            print(file, len(Made_X))

        except Exception as e:
            print('Error in made_x :', e)

    np.save('./Made_X/Made_X %s_%s' % (input_data_length, model_num), np.array(Made_X))
    np.save('./Made_X/Made_Y %s_%s' % (input_data_length, model_num), np.array(Made_Y))
