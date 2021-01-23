import numpy as np
import pandas as pd
import pybithumb
import os
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from Funcs_CNN4 import cmo, rsi, obv, macd
import mpl_finance as mf
import PIL.Image as pilimg
import cv2
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


def low_high(Coin, input_data_length, ip_limit=None, trade_limit=None, crop_size=None, lowhigh_point=None):

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

    ohlcv_excel['MA60'] = ohlcv_excel['close'].rolling(20).mean()
    ohlcv_excel['CMO'] = cmo(ohlcv_excel)
    ohlcv_excel['OBV'] = obv(ohlcv_excel)
    ohlcv_excel['RSI'] = rsi(ohlcv_excel)
    macd(ohlcv_excel)

    if lowhigh_point is not None:
        check_span = 30
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
                break

    closeprice = ohlcv_excel['close'].iloc[-1]
    datetime_list = ohlcv_excel.index.values

    # ----------- dataX, dataY 추출하기 -----------#
    #   OBV :
    ohlcv_data = ohlcv_excel.values[sum(ohlcv_excel.MACD_Signal.isna()):].astype(np.float)
    # print(ohlcv_data.shape)

    #       현재까지 모든 종가들이 이전 저점 보다 커야한다.     #
    #       검사      #
    if lowhigh_point is not None:
        scaled_close = min_max_scaler(ohlcv_data[:, [1]])
        for i in range(low_index):
            if scaled_close[-low_index] >= scaled_close[-i]:
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
        for i in range(crop_size, len(ohlcv_data)):  # 마지막 데이터까지 다 긇어모은다.
            group_x = ohlcv_data[i - crop_size: i]
            price = group_x[:, :4]
            volume = group_x[:, [4]]
            MA60 = group_x[:, [-8]]
            CMO = group_x[:, [-7]]
            OBV = group_x[:, [-6]]
            RSI = group_x[:, [-5]]
            MACD = group_x[:, [-4]]
            MACD_Signal = group_x[:, [-3]]
            MACD_OSC = group_x[:, [-2]]
            scaled_price = min_max_scaler(price)
            scaled_volume = min_max_scaler(volume)
            scaled_MA20 = min_max_scaler(MA60)
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
                return None, None, None, None

            dataX.append(group_x)  # dataX 리스트에 추가

        if len(dataX) < 100:
            return None, None, None, None

        # quit()
        X_test = np.array(dataX)
        row = X_test.shape[1]
        col = X_test.shape[2]

        X_test = X_test.astype('float32').reshape(-1, row, col, 1)[:, :, :4]
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

    if type(file) == str:

        # ohlcv_excel = pd.read_excel(dir + file, index_col=0)

        # Date = file.split()[0]
        # Coin = file.split()[1].split('.')[0]

        # except_list case
        ohlcv_excel = pd.read_excel(dir + file.split('_')[0] + ' ohlcv.xlsx', index_col=0)

        Date = file.split()[0]
        Coin = file.split('_')[0].split()[1]

    else:
        ohlcv_excel = file
        Date = str(datetime.now()).split()[0]
        Coin = file.index.name

    ohlcv_excel['MA60'] = ohlcv_excel['close'].rolling(60).mean()
    ohlcv_excel['CMO'] = cmo(ohlcv_excel)
    ohlcv_excel['OBV'] = obv(ohlcv_excel)
    ohlcv_excel['RSI'] = rsi(ohlcv_excel)
    macd(ohlcv_excel)

    # print(ohlcv_excel.info)
    # quit()

    #   이후 check_span 데이터와 현재 포인트를 비교해서 현재 포인트가 저가인지 고가인지 예측한다.
    #   진입, 저점, 고점, 거래 안함의 y_label 인 trade_state  >> [1, 2, 0]
    #   저점과 고점은 최대 3개의 중복 값을 허용한다.
    trade_state = [0] * len(ohlcv_excel)
    max_index_list = list()
    for i in range(2, len(ohlcv_excel) - check_span):

        if (ohlcv_excel['close'][i - 2] <= ohlcv_excel['MA60'][i - 2]) and (ohlcv_excel['close'][i - 1] > ohlcv_excel['MA60'][i - 1]):
            # trade_state[i] = 1

            #   진입
            for i2 in range(i, i + check_span):
                if ohlcv_excel['close'][i2] <= ohlcv_excel['MA60'][i2]:
                    break
                if i2 == i + check_span - 1:

                    #   이탈
                    #   진입부터 MA선 밑으로 내려가는 SPAN POINT 찾기
                    for i3 in range(i, len(ohlcv_excel) - check_span):
                        # print(ohlcv_excel['close'][i3], ohlcv_excel['MA60'][i3])
                        if ohlcv_excel['close'][i3] <= ohlcv_excel['MA60'][i3]:
                            break

                    #   진입부터 SPAN_POINT 까지의 MAX값 INDEX 찾기
                    for i4 in range(i, i3):
                        if ohlcv_excel['close'][i4] == ohlcv_excel['close'][i:i3].max():
                            break
                    # print(i, i3, i4)
                    trade_state[i] = 1
                    max_index_list.append(i4)

        #   거래 안함
        else:
            trade_state[i] = 0

    for index in max_index_list:
        trade_state[index] = 2
    ohlcv_excel['trade_state'] = trade_state
    # print(trade_state)
    # quit()

    # ----------- dataX, dataY 추출하기 -----------#
    # print(ohlcv_excel.info())
    # quit()
    # ohlcv_excel.to_excel('test.xlsx')

    # NaN 제외하고 데이터 자르기 (데이터가 PIXEL 로 들어간다고 생각하면 된다)
    #   OBV : -CHECK_SPAN
    ohlcv_data = ohlcv_excel.values[sum(ohlcv_excel.MA60.isna()): -check_span].astype(np.float)
    # ohlcv_data = ohlcv_excel.values[sum(ohlcv_excel.MACD_Signal.isna()): -check_span].astype(np.float)
    # ohlcv_data = ohlcv_excel.values[1: -check_span].astype(np.float)
    # ohlcv_data = ohlcv_excel.values[sum(ohlcv_excel.CMO.isna()): -check_span].astype(np.float)
    # ohlcv_data = ohlcv_excel.values[sum(ohlcv_excel.RSI.isna()): -check_span].astype(np.float)
    # for i in range(len(ohlcv_data)):
    #     if ohlcv_data[:, [-1]][i] == 1:
    #         plt.plot(ohlcv_data[i - check_span:i, [0, 1, 2, 3, 5]])
    #         plt.show()

    # print(pd.DataFrame(ohlcv_data).info())
    # print(pd.DataFrame(ohlcv_data).to_excel('test.xlsx'))
    # print(list(map(float, ohlcv_data[0])))
    # print(ohlcv_data.shape)
    # quit()

    #       출력되는 Y 데이터 길이를 동일화 시킨다.     #
    # ohlcv_data = ohlcv_data[-300:]

    # 결측 데이터 제외
    if len(ohlcv_data) != 0:

        #          데이터 전처리         #
        #   Fixed X_data    #
        # pricema = ohlcv_data[:, [0, 1, 2, 3, 5]]
        # volume = ohlcv_data[:, [4]]
        # MA60 = ohlcv_data[:, [-8]]
        # CMO = ohlcv_data[:, [-7]]
        # OBV = ohlcv_data[:, [-6]]
        # RSI = ohlcv_data[:, [-5]]
        # MACD = ohlcv_data[:, [-4]]
        # MACD_Signal = ohlcv_data[:, [-3]]
        # MACD_OSC = ohlcv_data[:, [-2]]

        #   Flexible Y_data    #
        trade_state = ohlcv_data[:, [-1]]

        # scaled_pricema = min_max_scaler(pricema)
        # plt.plot(scaled_pricema)
        # plt.plot(trade_state)
        # plt.show()
        # scaled_volume = min_max_scaler(volume)
        # scaled_CMO = min_max_scaler(CMO)
        # scaled_OBV = min_max_scaler(OBV)
        # scaled_RSI = min_max_scaler(RSI)

        # x = np.concatenate((scaled_price, scaled_volume, scaled_OBV), axis=1)  # axis=1, 세로로 합친다
        # x = scaled_price
        y = trade_state
        # print(sum(y==1))
        # print(x.shape, y_low.shape)  # (258, 6) (258, 1)
        # quit()
        chart_to_np = list()
        y_label = list()
        for i in range(crop_size, len(ohlcv_data)):
            group_x = ohlcv_data[i - crop_size: i]

            #   데이터 값에 결측치가 존재하는 경우 #
            if sum(sum(np.isnan(group_x))) > 0:
                continue

            if y[i] in [1, 2]:
                pricema = group_x[:, [0, 1, 2, 3, 5]]
                volume = group_x[:, [4]]
                CMO = group_x[:, [-7]]
                OBV = group_x[:, [-6]]
                RSI = group_x[:, [-5]]
                MACD = group_x[:, [-4]]
                MACD_Signal = group_x[:, [-3]]
                MACD_OSC = group_x[:, [-2]]

                #       Let's make Chart        #
                index = np.arange(len(pricema))
                ochl = np.hstack((np.reshape(index, (-1, 1)), pricema[:, :-1]))
                fig = plt.figure(figsize=(3, 3))
                ax = fig.add_subplot(111)
                mf.candlestick_ochl(ax, ochl, width=0.5, colorup='r', colordown='b')
                plt.plot(pricema[:, [-1]])
                plt.axis('off')

                ax2 = ax.twinx()
                ax2.plot(OBV, color='g')
                plt.axis('off')

                # ax3 = ax.twinx()
                # ax3.plot(RSI, color='g')
                # plt.axis('off')

                fig.tight_layout()

                img_path = './Made_Chart_all/%s_%s/%s %s_%s.png' % (input_data_length, model_num,
                                                                    Date, Coin, i)
                # plt.show()
                plt.savefig(img_path)

                # #       저점과 고점 데이터는 따로 한번 더 저장해준다       #
                # if 0.5 < y[i] < 1.5:
                #     img_path = './Made_Chart_in/%s_%s/%s %s_%s.png' % (input_data_length, model_num,
                #                                                             Date, Coin, i)
                # elif 1.5 < y[i] < 2.5:
                #     img_path = './Made_Chart_out/%s_%s/%s %s_%s.png' % (input_data_length, model_num,
                #                                                             Date, Coin, i)

                # plt.savefig(img_path)
                plt.close()

                img = pilimg.open('./Made_Chart_all/%s_%s/%s %s_%s.png' % (input_data_length, model_num,
                                                                              Date, Coin, i))
                pixel = np.array(img)
                rgb = cv2.cvtColor(pixel, cv2.COLOR_RGBA2RGB)
                print(rgb.shape)
                if rgb.shape[0] != 300:
                    print(Date, Coin, 'under 300')
                    continue
                    # quit()
                chart_to_np.append(rgb)
                y_label.append(y[i])

        return chart_to_np, y_label


if __name__ == '__main__':

    # ----------- Params -----------#
    input_data_length = 30
    model_num = '64'

    #       Make folder      #
    # folder_name_list = ['Made_Chart_to_np', 'Made_Chart_all', 'Made_Chart_in', 'Made_Chart_out']
    #
    # for folder_name in folder_name_list:
    #     try:
    #         os.mkdir('./%s/%s_%s/' % (folder_name, input_data_length, model_num))
    #
    #     except Exception as e:
    #         print(e)
    #         continue

    check_span = 40
    get_fig = 1

    Made_X = []
    Made_Y = []

    # ohlcv_list = ['2020-01-10 BTC ohlcv.xlsx']
    except_list = os.listdir('./Made_Chart_all/30_64/')
    # for except_file in except_list:
    #     pull_filename = except_file.split('.')[0] + ' ohlcv.xlsx'
    #     if pull_filename in ohlcv_list:
    #         ohlcv_list.remove(pull_filename)

    # print(except_list)
    # quit()

    for file in except_list:

        try:

            # if int(file.split()[0].split('-')[1]) != 1:
            #     continue
            #
            # Date = file.split()[0]
            # Coin = file.split()[1].split('.')[0]

            # except_list case
            Date = file.split()[0]
            Coin = file.split('_')[0].split()[1]

            result = made_x(file, input_data_length, model_num, check_span, get_fig, crop_size=input_data_length)
            # result = low_high('dvp'.upper(), input_data_length, lowhigh_point='on')
            # print(result)
            # quit()

            Made_X += result[0]
            Made_Y += result[1]
            print(file, len(Made_X))
            print(Made_X[0].shape)

            # np.save('./Made_Chart_to_np/%s_%s/%s %s' % (input_data_length, model_num,
            #                                             Date, Coin), np.array(result[0]))
            # np.save('./Made_Chart_to_np/%s_%s/%s %s_y' % (input_data_length, model_num,
            #                                               Date, Coin), np.array(result[1]))

            if len(Made_X) > 5000:
                break

        except Exception as e:
            print('Error in made_x :', e)

    np.save('./Made_X/Made_X %s_%s' % (input_data_length, model_num), np.array(Made_X))
    np.save('./Made_X/Made_Y %s_%s' % (input_data_length, model_num), np.array(Made_Y))

