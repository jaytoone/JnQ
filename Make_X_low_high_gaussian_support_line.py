import numpy as np
import pandas as pd
import pybithumb
import os
import matplotlib.pyplot as plt
import warnings
from datetime import datetime
from Funcs_CNN4 import cmo, rsi, obv, macd, clearance
import mpl_finance as mf
from scipy.ndimage.filters import gaussian_filter1d
from sklearn.preprocessing import StandardScaler
import random
import time

warnings.filterwarnings("ignore")
pd.set_option('display.max_rows', 500)


home_dir = os.path.expanduser('~')
dir = home_dir + '/OneDrive/CoinBot/ohlcv/'
ohlcv_list = os.listdir(dir)


def min_max_scaler(x):
    scaled_x = (x - x.min()) / (x.max() - x.min())
    return scaled_x


def max_abs_scaler(x):
    scaled_x = x / abs(x).max()
    return scaled_x


def low_high(Coin, input_data_length, interval="minute1", sigma=4, ip_limit=None, crop_size=None, fluc_limit=1.025):

    #   거래 제한은 고점과 저점을 분리한다.

    #   User-Agent Configuration
    #   IP - Change
    if ip_limit is None:
        ohlcv_excel = pybithumb.get_ohlcv(Coin, 'KRW', interval=interval)
    else:
        ohlcv_excel = pybithumb.get_ohlcv(Coin, 'KRW', interval, 'proxyon')

    # price_gap = ohlcv_excel.close.max() / ohlcv_excel.close.min()
    # if (price_gap < 1.07) and (trade_limit is not None):
    #     return None, None, None, None

    # ohlcv_excel['CMO'] = cmo(ohlcv_excel)
    # ohlcv_excel['OBV'] = obv(ohlcv_excel)
    # ohlcv_excel['RSI'] = rsi(ohlcv_excel)

    smoothed_curve = gaussian_filter1d(ohlcv_excel['close'], sigma=sigma)
    #           REALTIME CURVE          #
    # crop_size = 30
    # smoothed_curve = [np.nan] * len(ohlcv_excel)
    # for i in range(input_data_length, len(ohlcv_excel)):
    #     y = ohlcv_excel['close'].values[i + 1 - crop_size:i + 1]
    #     y_smooth = gaussian_filter1d(y, sigma=sigma, mode='reflect')
    #     smoothed_curve[i] = y_smooth[-1]

    ohlcv_excel['gaussian'] = smoothed_curve

    trade_state = [np.nan] * len(ohlcv_excel)
    # print(ohlcv_excel.head())
    # quit()

    support_line_index = list()
    resistance_exit_index = list()
    for i in range(2, len(ohlcv_excel)):
        #       윗봉우리 1        #
        if (smoothed_curve[i - 2] < smoothed_curve[i - 1]) and (smoothed_curve[i - 1] > smoothed_curve[i]):
            #       윗봉우리의 꼭지점 인덱스를 저장       #
            resistance_exit_index.append(i)

            up_hill_exist = 0
            for j in range(i + 1, len(ohlcv_excel)):
                #       윗봉우리 2        #
                if (smoothed_curve[j - 2] < smoothed_curve[j - 1]) and (smoothed_curve[j - 1] > smoothed_curve[j]):
                    # crossover 부터 under 까지 최고 최저가 각각 2, 1로 라벨링
                    min_index = np.argmin(ohlcv_excel['low'][i:j].values)
                    support_line_index.append(i + min_index)
                    trade_state[i + min_index] = 1
                    up_hill_exist = 1
                    break

            #       윗봉우리 1 다음에 윗봉우리 2가 없고 아랫봉우리가 있으면     #
            if up_hill_exist != 1:
                for j in range(i + 1, len(ohlcv_excel)):
                    #       아랫봉우리가 있으면    #
                    if (smoothed_curve[j - 2] > smoothed_curve[j - 1]) and (smoothed_curve[j - 1] < smoothed_curve[j]):
                        min_index = np.argmin(ohlcv_excel['low'][i:].values)
                        support_line_index.append(i + min_index)
                        trade_state[i + min_index] = 1

    # #       전전 지지선 > 전 지지선 && 전 지지선 다음에 윗봉우리가 존재하는 경우 && 현재 가격이 이전 지지선보다 높고          #
    # # print(support_line_index)
    # # quit()
    # if len(support_line_index) >= 2:
    #     if ohlcv_excel['low'].iloc[support_line_index[-2]] > ohlcv_excel['low'].iloc[support_line_index[-1]]:
    #         #   윗봉우리 존재 여부 검사 = 윗봉우리의 마지막 인덱스가 마지막 지지선의 인덱스보다 크면 된다.        #
    #         if support_line_index[-1] < resistance_exit_index[-1]:
    #             #   현재 저가가 이전 지지선보다 높으면,
    #             if ohlcv_excel['low'].iloc[-1] >= ohlcv_excel['low'].iloc[support_line_index[-1]]:
    #                 #   전전 고가, 전 고가의 최고가와 현재 저가의 변동폭이 2.5% 이상
    #                 if len(resistance_exit_index) >= 2:
    #                     high_price = max(ohlcv_excel['high'].iloc[resistance_exit_index[-2]], ohlcv_excel['high'].iloc[resistance_exit_index[-1]])
    #                 else:
    #                     high_price = ohlcv_excel['high'].iloc[resistance_exit_index[-1]]
    #
    #                 if high_price / ohlcv_excel['low'].iloc[support_line_index[-1]] >= fluc_limit:
    #                     trade_state[-1] = 2
    #       전전전 지지선 > 전전 지지선 && 전전 지지선 < 전 지지선      #
    if len(support_line_index) >= 3:
        if ohlcv_excel['low'].iloc[support_line_index[-3]] > ohlcv_excel['low'].iloc[support_line_index[-2]]:
            if ohlcv_excel['low'].iloc[support_line_index[-2]] < ohlcv_excel['low'].iloc[support_line_index[-1]]:
                #   전전 고가, 전 고가의 최고가와 현재 저가의 변동폭이 fluc_limit% 이상
                if len(resistance_exit_index) >= 2:
                    high_price = max(ohlcv_excel['high'].iloc[resistance_exit_index[-2]], ohlcv_excel['high'].iloc[resistance_exit_index[-1]])
                else:
                    high_price = ohlcv_excel['high'].iloc[resistance_exit_index[-1]]

                if high_price / ohlcv_excel['low'].iloc[support_line_index[-1]] >= fluc_limit:
                    trade_state[-1] = 2

    # for i in range(2, len(support_line_index)):
    #     if ohlcv_excel['low'].iloc[support_line_index[i - 2]] > ohlcv_excel['low'].iloc[support_line_index[i - 1]]:
    #         if ohlcv_excel['low'].iloc[support_line_index[i - 1]] < ohlcv_excel['low'].iloc[i]:
    #             trade_state[i] = 2

    # for n, i in enumerate(trade_state):
    #     if i not in [1, 2]:
    #         trade_state[n] = 0

    ohlcv_excel['trade_state'] = trade_state

    # ----------- dataX, dataY 추출하기 -----------#
    # print(ohlcv_excel.info())
    # quit()
    # ohlcv_excel.to_excel('test.xlsx')

    # NaN 제외하고 데이터 자르기 (데이터가 PIXEL 로 들어간다고 생각하면 된다)
    ohlcv_data = ohlcv_excel.values.astype(np.float)
    # ohlcv_data = ohlcv_excel.values[sum(ohlcv_excel.gaussian.isna()):].astype(np.float)
    # ohlcv_data = ohlcv_excel.values[sum(ohlcv_excel.MACD_Signal.isna()):].astype(np.float)
    # MACD = ohlcv_data[:, -5:-1]

    # if not 2 in trade_state:
    #     return None

    #           show Chart          #
    # fig = plt.figure(figsize=(10, 7))
    # ax = fig.add_subplot(111)
    # ochl = ohlcv_excel.iloc[:, :4]
    # index = np.arange(len(ochl))
    # ochl = np.hstack((np.reshape(index, (-1, 1)), ochl))
    # mf.candlestick_ochl(ax, ochl, width=0.5, colorup='r', colordown='b')
    # ax.plot(smoothed_curve, color='y')
    #
    # span_list = list()
    # for i in range(1, len(ochl)):
    #     if trade_state[i] == 1:
    #         span_list.append((i, i + 1))
    # for i in range(len(span_list)):
    #     plt.axvspan(span_list[i][0], span_list[i][1], facecolor='m', alpha=0.7)
    # span_list = list()
    # for i in range(1, len(ochl)):
    #     if trade_state[i] == 2:
    #         span_list.append((i - 1, i))
    # for i in range(len(span_list)):
    #     plt.axvspan(span_list[i][0], span_list[i][1], facecolor='c', alpha=1.)
    #
    # plt.show()
    # # plt.show(block=False)
    # # plt.pause(1.)
    # plt.close()
    # # quit()

    return trade_state

    # 결측 데이터 제외
    # if len(ohlcv_data) != 0:
    #
    #     #          데이터 전처리         #
    #
    #     #    X_data    #
    #     # price ma
    #
    #     # ohlcv_data[:, [0, 1, 2, 3, -2]] = min_max_scaler(ohlcv_data[:, [0, 1, 2, 3, -2]])
    #     # # plt.plot(ohlcv_data[:, [0, 1, 2, 3, -2]])
    #     # # plt.show()
    #     # buy_price = ohlcv_data[-1, [1]]
    #     # price_max = max(ohlcv_excel['high'])
    #     # price_min = min(ohlcv_excel['low'])
    #     # exit_price = (buy_price - 0.1) * (price_max - price_min) + price_min
    #     # # ohlcv_data[:, [0, 1, 2, 3]] = max_abs_scaler(ohlcv_data[:, [0, 1, 2, 3]])
    #     # # volume
    #     # ohlcv_data[:, [4]] = max_abs_scaler(ohlcv_data[:, [4]])
    #     # #   CMO
    #     # ohlcv_data[:, [-9]] = max_abs_scaler(ohlcv_data[:, [-9]])
    #     # #   OBV
    #     # ohlcv_data[:, [-8]] = max_abs_scaler(ohlcv_data[:, [-8]])
    #     # #   RSI
    #     # ohlcv_data[:, [-7]] = max_abs_scaler(ohlcv_data[:, [-7]])
    #     # #   MACD
    #     # ohlcv_data[:, -6:-2] = max_abs_scaler(ohlcv_data[:, -6:-2])
    #     #
    #     # #    Y_data    #
    #     # trade_state = ohlcv_data[:, [-1]]
    #     # y = trade_state
    #
    #     # print(x.shape, y_low.shape)  # (258, 6) (258, 1)
    #     # quit()
    #
    #     dataX = []  # input_data length 만큼 담을 dataX 그릇
    #     dataY = []  # Target 을 담을 그릇
    #     # for i in range(crop_size, len(ohlcv_data)):  # 마지막 데이터까지 다 긇어모은다.
    #     #
    #     #     group_x = ohlcv_data[i - crop_size: i]
    #     #     group_y = y[i]
    #     #     price = group_x[:, :4]
    #     #     volume = group_x[:, [4]]
    #     #     CMO = group_x[:, [-9]]
    #     #     OBV = group_x[:, [-8]]
    #     #     RSI = group_x[:, [-7]]
    #     #     MACD = group_x[:, -6:-2]
    #     #     gaussian_curve = group_x[:, [-2]]
    #     #
    #     #     # price = min_max_scaler(price)
    #     #
    #     #     x = np.concatenate((price, volume, CMO, OBV, RSI, MACD, gaussian_curve), axis=1)
    #     #     #          데이터 전처리         #
    #     #     #   Fixed X_data    #
    #     #     # .C), axis=1)  # axis=1, 세로로 합친다
    #     #
    #     #     group_x = x[-input_data_length:]
    #     #     # print(group_x[0])
    #     #     # quit()
    #     #
    #     #     #   데이터 값에 결측치가 존재하는 경우 #
    #     #     # if sum(sum(np.isnan(group_x))) > 0:
    #     #     #     continue
    #     #
    #     #     dataX.append(group_x)  # dataX 리스트에 추가
    #     #
    #     # #       low_high 는 데이터를 거르지 않는다.        #
    #     # if len(dataX) < 100:
    #     #     return None, None, None, None
    #
    #     # closeprice = np.roll(np.array(list(map(lambda x: x[-1][[1]][0], X_test))), -1)
    #     # plt.plot(closeprice)
    #     # plt.show()
    #
    #     return dataX, ohlcv_excel['close'].iloc[-1], ohlcv_data[crop_size:, ], None


def made_x(file, input_data_length, sigma, model_num, get_fig, crop_size=None):
    
    if not file.endswith('.xlsx'):
        ohlcv_excel = pybithumb.get_ohlcv(file, 'KRW', 'minute1')
    else:
        ohlcv_excel = pd.read_excel(dir + '%s' % file, index_col=0)

    # ohlcv_excel['CMO'] = cmo(ohlcv_excel)
    # ohlcv_excel['OBV'] = obv(ohlcv_excel)
    # ohlcv_excel['RSI'] = rsi(ohlcv_excel)
    # macd(ohlcv_excel, short=105, long=168, signal=32)

    # print(ohlcv_excel)
    # quit()

    #   이후 check_span 데이터와 현재 포인트를 비교해서 현재 포인트가 저가인지 고가인지 예측한다.
    #   진입, 저점, 고점, 거래 안함의 y_label 인 trade_state  >> [1, 2, 0]
    #   저점과 고점은 최대 3개의 중복 값을 허용한다.

    smoothed_curve = gaussian_filter1d(ohlcv_excel['close'], sigma=sigma)
    #           REALTIME CURVE          #
    # smoothed_curve = [np.nan] * len(ohlcv_excel)
    # for i in range(input_data_length, len(ohlcv_excel)):
    #     y = ohlcv_excel['close'].values[i + 1 - crop_size:i + 1]
    #     y_smooth = gaussian_filter1d(y, sigma=10, mode='reflect')
    #     smoothed_curve[i] = y_smooth[-1]

    ohlcv_excel['gaussian'] = smoothed_curve

    trade_state = [np.nan] * len(ohlcv_excel)
    # print(ohlcv_excel.head())
    # quit()

    support_line_index = list()
    resistance_exit_index = list()
    for i in range(2, len(ohlcv_excel)):
        #       윗봉우리 1        #
        if (smoothed_curve[i - 2] < smoothed_curve[i - 1]) and (smoothed_curve[i - 1] > smoothed_curve[i]):
            #       윗봉우리의 꼭지점 인덱스를 저장       #
            resistance_exit_index.append(i)

            up_hill_exist = 0
            for j in range(i + 1, len(ohlcv_excel)):
                #       윗봉우리 2        #
                if (smoothed_curve[j - 2] < smoothed_curve[j - 1]) and (smoothed_curve[j - 1] > smoothed_curve[j]):
                    # crossover 부터 under 까지 최고 최저가 각각 2, 1로 라벨링
                    min_index = np.argmin(ohlcv_excel['low'][i:j].values)
                    support_line_index.append(i + min_index)
                    trade_state[i + min_index] = 1
                    up_hill_exist = 1
                    break

            #       윗봉우리 1 다음에 윗봉우리 2가 없고 아랫봉우리가 있으면     #
            if up_hill_exist != 1:
                for j in range(i + 1, len(ohlcv_excel)):
                    #       아랫봉우리가 있으면    #
                    if (smoothed_curve[j - 2] > smoothed_curve[j - 1]) and (smoothed_curve[j - 1] < smoothed_curve[j]):
                        min_index = np.argmin(ohlcv_excel['low'][i:].values)
                        support_line_index.append(i + min_index)
                        trade_state[i + min_index] = 1

    #       지지선의 V자 형식 포착하기         #
    # print(support_line_index)
    # quit()
    # for i in range(2, len(support_line_index)):
    #     if ohlcv_excel['low'].iloc[support_line_index[i - 2]] > ohlcv_excel['low'].iloc[support_line_index[i - 1]]:
    #         if ohlcv_excel['low'].iloc[support_line_index[i - 1]] < ohlcv_excel['low'].iloc[support_line_index[i]]:
    #             trade_state[support_line_index[i]] = 2

    for n, i in enumerate(trade_state):
        if i not in [1]:
            trade_state[n] = 0

    ohlcv_excel['trade_state'] = trade_state

    # ----------- dataX, dataY 추출하기 -----------#
    # print(ohlcv_excel.info())
    # quit()
    # ohlcv_excel.to_excel('test.xlsx')

    # NaN 제외하고 데이터 자르기 (데이터가 PIXEL 로 들어간다고 생각하면 된다)
    #   OBV : -CHECK_SPAN
    ohlcv_data = ohlcv_excel.values.astype(np.float)
    # ohlcv_data = ohlcv_excel.values[sum(ohlcv_excel.MACD_Signal.isna()):].astype(np.float)
    # MACD = ohlcv_data[:, -5:-1]

    #           show Chart          #
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111)
    ochl = ohlcv_excel.iloc[:, :4]
    index = np.arange(len(ochl))
    ochl = np.hstack((np.reshape(index, (-1, 1)), ochl))
    mf.candlestick_ochl(ax, ochl, width=0.5, colorup='r', colordown='b')
    ax.plot(smoothed_curve, color='y')

    span_list = list()
    for i in range(1, len(ochl)):
        if trade_state[i] == 1:
            span_list.append((i, i + 1))
    for i in range(len(span_list)):
        plt.axvspan(span_list[i][0], span_list[i][1], facecolor='m', alpha=0.7)
    span_list = list()
    for i in range(1, len(ochl)):
        if trade_state[i] == 2:
            span_list.append((i, i + 1))
    for i in range(len(span_list)):
        plt.axvspan(span_list[i][0], span_list[i][1], facecolor='c', alpha=1.)

    plt.show()
    # plt.show(block=False)
    # plt.pause(1.)
    plt.close()
    # quit()

    # 결측 데이터 제외
    if len(ohlcv_data) != 0:

        #          데이터 전처리         #

        #    X_data    #
        # volume
        # ohlcv_data[:, [4]] = max_abs_scaler(ohlcv_data[:, [4]])
        # #   CMO
        # ohlcv_data[:, [-9]] = max_abs_scaler(ohlcv_data[:, [-9]])
        # #   OBV
        # ohlcv_data[:, [-8]] = max_abs_scaler(ohlcv_data[:, [-8]])
        # #   RSI
        # ohlcv_data[:, [-7]] = max_abs_scaler(ohlcv_data[:, [-7]])
        # #   MACD
        # ohlcv_data[:, -6:-2] = max_abs_scaler(ohlcv_data[:, -6:-2])

        #    Y_data    #
        trade_state = ohlcv_data[:, [-1]]
        y = trade_state

        # print(x.shape, y_low.shape)  # (258, 6) (258, 1)
        # quit()

        dataX = []  # input_data length 만큼 담을 dataX 그릇
        dataY = []  # Target 을 담을 그릇
        for i in range(crop_size, len(ohlcv_data)):  # 마지막 데이터까지 다 긇어모은다.

            group_x = ohlcv_data[i + 1 - crop_size: i + 1]
            group_y = y[i]
            price = group_x[:, :4]
            # volume = group_x[:, [4]]
            # CMO = group_x[:, [-9]]
            # OBV = group_x[:, [-8]]
            # RSI = group_x[:, [-7]]
            # MACD = group_x[:, -6:-2]
            # gaussian_curve = group_x[:, [-2]]

            #       어떤 scaler를 사용할지 생각해보아야한다.     #
            price = max_abs_scaler(price)

            x = price
            # x = np.concatenate((price, volume, CMO, OBV, RSI, MACD, gaussian_curve), axis=1)

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

        # X_test = np.array(dataX)
        # row = X_test.shape[1]
        # col = X_test.shape[2]
        #
        # X_test = X_test.astype('float32').reshape(-1, row, col, 1)
        # print(X_test.shape)

        #       Exstracting fiexd X_data       #
        # sliced_ohlcv = min_max_scaler(ohlcv_data[crop_size:, :x.shape[1]])

        return dataX, dataY


if __name__ == '__main__':

    # ----------- Params -----------#
    input_data_length = 15
    sigma = 4
    model_num = 124
    get_fig = 0

    #       Make folder      #
    try:
        os.mkdir('./Figure_data/%s_%s/' % (input_data_length, model_num))

    except Exception as e:
        pass

    Made_X = []
    Made_Y = []

    # ohlcv_list = ['2020-01-10 ETH ohlcv.xlsx']
    # random.shuffle(ohlcv_list)
    
    # Coinlist = pybithumb.get_tickers()
    # Coinlist = ['TMTG']

    # print(pybithumb.get_top_coin(10))
    # quit()
    TopCoin = ['ZIL', 'THETA', 'TRX', 'BTT', 'XEM']
    # TopCoin = ['DVP']
    # print(TopCoin)
    # quit()

    for file in TopCoin:

        # if int(file.split()[0].split('-')[1]) != 1:
        #     continue

        result = made_x(file, input_data_length, sigma, model_num, get_fig, crop_size=input_data_length)
        # result = low_high(file, input_data_length, sigma=4)
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
    np.save('./Made_X/Made_X %s_%s' % (input_data_length, model_num), np.array(Made_X))
    np.save('./Made_X/Made_Y %s_%s' % (input_data_length, model_num), np.array(Made_Y))

