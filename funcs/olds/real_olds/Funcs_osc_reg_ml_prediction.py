import pybithumb
import numpy as np
import pandas as pd
from datetime import datetime
import os
from scipy import stats
from asq.initiators import query
import matplotlib.pyplot as plt
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler
import time
import mpl_finance as mf
from scipy.ndimage.filters import gaussian_filter1d
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import random
import warnings

warnings.filterwarnings(action='ignore')

# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import r2_score
# from sklearn.preprocessing import PolynomialFeatures

pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 2500)
pd.set_option('display.max_columns', 2500)


def min_max_scaler(x):
    scaled_x = (x - x.min()) / (x.max() - x.min())
    return scaled_x


def max_abs_scaler(x):
    scaled_x = x / abs(x).max()
    return scaled_x


def transh_hour(realtime, numb):
    Hour = realtime[numb].split(':')[0]
    return int(Hour)


def transh_min(realtime, numb):
    Minute = realtime[numb].split(':')[1]
    return int(Minute)


# print(transh_min(-1))
def transh_fluc(Coin):
    try:
        TransH = pybithumb.transaction_history(Coin)
        TransH = TransH['data']
        Realtime = query(TransH).select(lambda item: item['transaction_date'].split(' ')[1]).to_list()

        # 거래 활발한지 검사
        if (transh_hour(Realtime, -1) - transh_hour(Realtime, 0)) < 0:
            if 60 + (transh_min(Realtime, -1) - transh_min(Realtime, 0)) > 30:
                return 0, 0
        elif 60 * (transh_hour(Realtime, -1) - transh_hour(Realtime, 0)) + (
                transh_min(Realtime, -1) - transh_min(Realtime, 0)) > 30:
            return 0, 0

        # 1분 동안의 거래 이력을 조사하는 루프, 0 - 59 와 같은 음수처리를 해주어야한다.
        i = 1
        while True:
            i += 1
            if i > len(Realtime):
                m = i
                break
            # 음수 처리
            if (transh_min(Realtime, -1) - transh_min(Realtime, -i)) < 0:
                if (60 + transh_min(Realtime, -1) - transh_min(Realtime, -i)) > 1:
                    m = i - 1
                    break
            elif (transh_min(Realtime, -1) - transh_min(Realtime, -i)) > 1:
                m = i - 1
                break

        # Realtime = query(TransH[-i:]).select(lambda item: item['transaction_date'].split(' ')[1]).to_list()
        Price = list(map(float, query(TransH[-m:]).select(lambda item: item['price']).to_list()))

        # print(Realtime)
        # print(Price)
        fluc = max(Price) / min(Price)
        if TransH[-1]['type'] == 'ask':
            fluc = -fluc
        return fluc, min(Price)

    except Exception as e:
        print("Error in transh_fluc :", e)
        return 0, 0


def realtime_transaction(Coin, display=5):
    Transaction_history = pybithumb.transaction_history(Coin)
    Realtime = query(Transaction_history['data'][-display:]).select(
        lambda item: item['transaction_date'].split(' ')[1]).to_list()
    Realtime_Price = list(
        map(float, query(Transaction_history['data'][-display:]).select(lambda item: item['price']).to_list()))
    Realtime_Volume = list(
        map(float, query(Transaction_history['data'][-display:]).select(lambda item: item['units_traded']).to_list()))

    print("##### 실시간 체결 #####")
    print("{:^10} {:^10} {:^20}".format('시간', '가격', '거래량'))
    for i in reversed(range(display)):
        print("%-10s %10.2f %20.3f" % (Realtime[i], Realtime_Price[i], Realtime_Volume[i]))
    return


def realtime_hogachart(Coin, display=3):
    Hogachart = pybithumb.get_orderbook(Coin)

    print("##### 실시간 호가창 #####")
    print("{:^10} {:^20}".format('가격', '거래량'))
    for i in reversed(range(display)):
        print("%10.2f %20.3f" % (Hogachart['asks'][i]['price'], Hogachart['asks'][i]['quantity']))
    print('-' * 30)
    for j in range(display):
        print("%10.2f %20.3f" % (Hogachart['bids'][j]['price'], Hogachart['bids'][j]['quantity']))


def realtime_volume(Coin):
    Transaction_history = pybithumb.transaction_history(Coin)
    Realtime_Volume = query(Transaction_history['data']).where(lambda item: item['type'] == 'bid').select(
        lambda item: item['units_traded']).to_list()
    Realtime_Volume = sum(list(map(float, Realtime_Volume)))
    return Realtime_Volume


def realtime_volume_ratio(Coin):
    Transaction_history = pybithumb.transaction_history(Coin)
    Realtime_bid = query(Transaction_history['data']).where(lambda item: item['type'] == 'bid').select(
        lambda item: item['units_traded']).to_list()
    Realtime_ask = query(Transaction_history['data']).where(lambda item: item['type'] == 'ask').select(
        lambda item: item['units_traded']).to_list()
    Realtime_bid = sum(list(map(float, Realtime_bid)))
    Realtime_ask = sum(list(map(float, Realtime_ask)))
    Realtime_Volume_Ratio = Realtime_bid / Realtime_ask
    return Realtime_Volume_Ratio


def topcoinlist(Date):
    temp = []
    dir = 'C:/Users/장재원/OneDrive/Hacking/CoinBot/ohlcv/'
    ohlcv_list = os.listdir(dir)

    for file in ohlcv_list:
        if file.find(Date) is not -1:  # 해당 파일이면 temp[i] 에 넣겠다.
            filename = os.path.splitext(file)
            temp.append(filename[0].split(" ")[1])
    return temp


def get_ma_min(Coin):
    df = pybithumb.get_ohlcv(Coin, "KRW", 'minute1')

    df['MA20'] = df['close'].rolling(20).mean()

    DatetimeIndex = df.axes[0]
    period = 20
    if inthour(DatetimeIndex[-1]) - inthour(DatetimeIndex[-period]) < 0:
        if 60 + (intmin(DatetimeIndex[-1]) - intmin(DatetimeIndex[-period])) > 30:
            return 0
    elif 60 * (inthour(DatetimeIndex[-1]) - inthour(DatetimeIndex[-period])) + intmin(DatetimeIndex[-1]) - intmin(
            DatetimeIndex[-period]) > 30:
        return 0
    slope, intercept, r_value, p_value, stderr = stats.linregress([i for i in range(period)], df.MA20[-period:])

    return slope


def get_ma20_min(Coin):
    df = pybithumb.get_ohlcv(Coin, "KRW", 'minute1')

    maxAbsScaler = MaxAbsScaler()

    df['MA20'] = df['close'].rolling(20).mean()
    MA_array = np.array(df['MA20']).reshape(len(df.MA20), 1)
    maxAbsScaler.fit(MA_array)
    scaled_MA = maxAbsScaler.transform(MA_array)

    period = 5
    slope, intercept, r_value, p_value, stderr = stats.linregress([i for i in range(period)], scaled_MA[-period:])

    return slope


def get_obv_min(Coin):
    df = pybithumb.get_ohlcv(Coin, "KRW", "minute1")

    obv = [0] * len(df.index)
    for m in range(1, len(df.index)):
        if df['close'].iloc[m] > df['close'].iloc[m - 1]:
            obv[m] = obv[m - 1] + df['volume'].iloc[m]
        elif df['close'].iloc[m] == df['close'].iloc[m - 1]:
            obv[m] = obv[m - 1]
        else:
            obv[m] = obv[m - 1] - df['volume'].iloc[m]
    df['OBV'] = obv

    # 24시간의 obv를 잘라서 box 높이를 만들어주어야한다.
    DatetimeIndex = df.axes[0]
    boxheight = [0] * len(df.index)
    whaleincome = [0] * len(df.index)
    for m in range(len(df.index)):
        # 24시간 시작행 찾기, obv 데이터가 없으면 stop
        n = m
        while True:
            n -= 1
            if n < 0:
                n = 0
                break
            if inthour(DatetimeIndex[m]) - inthour(DatetimeIndex[n]) < 0:
                if 60 - (intmin(DatetimeIndex[m]) - intmin(DatetimeIndex[n])) >= 60 * 24:
                    break
            elif 60 * (inthour(DatetimeIndex[m]) - inthour(DatetimeIndex[n])) + intmin(DatetimeIndex[m]) - intmin(
                    DatetimeIndex[n]) >= 60 * 24:
                break
        obv_trim = obv[n:m]
        if len(obv_trim) != 0:
            boxheight[m] = max(obv_trim) - min(obv_trim)
            if obv[m] - min(obv_trim) != 0:
                whaleincome[m] = abs(max(obv_trim) - obv[m]) / abs(obv[m] - min(obv_trim))

    df['BoxHeight'] = boxheight
    df['Whaleincome'] = whaleincome

    period = 0
    while True:
        period += 1
        if period >= len(DatetimeIndex):
            break
        if inthour(DatetimeIndex[-1]) - inthour(DatetimeIndex[-period]) < 0:
            if 60 + (intmin(DatetimeIndex[-1]) - intmin(DatetimeIndex[-period])) >= 10:
                break
        elif 60 * (inthour(DatetimeIndex[-1]) - inthour(DatetimeIndex[-period])) + intmin(DatetimeIndex[-1]) - intmin(
                DatetimeIndex[-period]) >= 10:
            break

    slope, intercept, r_value, p_value, stderr = stats.linregress([i for i in range(period)], df.OBV[-period:])
    if period < 3:
        df['Whaleincome'].iloc[-1], slope = 0, 0
    else:
        slope = slope / df['BoxHeight'].iloc[-1]

    return df['Whaleincome'].iloc[-1], slope


def GetHogaunit(Hoga):
    if Hoga < 1:
        Hogaunit = 0.0001
    elif 1 <= Hoga < 10:
        Hogaunit = 0.001
    elif 10 <= Hoga < 100:
        Hogaunit = 0.01
    elif 100 <= Hoga < 1000:
        Hogaunit = 0.1
    elif 1000 <= Hoga < 5000:
        Hogaunit = 1
    elif 5000 <= Hoga < 10000:
        Hogaunit = 5
    elif 10000 <= Hoga < 50000:
        Hogaunit = 10
    elif 50000 <= Hoga < 100000:
        Hogaunit = 50
    elif 100000 <= Hoga < 500000:
        Hogaunit = 100
    elif 500000 <= Hoga < 1000000:
        Hogaunit = 500
    else:
        Hogaunit = 1000
    return Hogaunit


def clearance(price):
    try:
        Hogaunit = GetHogaunit(price)
        Htype = type(Hogaunit)
        if Hogaunit == 0.1:
            price2 = int(price * 10) / 10.0
        elif Hogaunit == 0.01:
            price2 = int(price * 100) / 100.0
        elif Hogaunit == 0.001:
            price2 = int(price * 1000) / 1000.0
        elif Hogaunit == 0.0001:
            price2 = int(price * 10000.0) / 10000.0
        else:
            return int(price) // Hogaunit * Hogaunit
        return Htype(price2)

    except Exception as e:
        return np.nan


def inthour(date):
    date = str(date)
    date = date.split(' ')
    hour = int(date[1].split(':')[0])  # 시
    return hour


def intmin(date):
    date = str(date)
    date = date.split(' ')
    min = int(date[1].split(':')[1])  # 분
    return min


def cmo(df, period=9):
    df['closegap_cunsum'] = (df['close'] - df['close'].shift(1)).cumsum()
    df['closegap_abs_cumsum'] = abs(df['close'] - df['close'].shift(1)).cumsum()
    # print(df)

    df['CMO'] = (df['closegap_cunsum'] - df['closegap_cunsum'].shift(period)) / (
            df['closegap_abs_cumsum'] - df['closegap_abs_cumsum'].shift(period)) * 100

    del df['closegap_cunsum']
    del df['closegap_abs_cumsum']

    return df['CMO']


def rsi(ohlcv_df, period=14):
    ohlcv_df['up'] = np.where(ohlcv_df.diff(1)['close'] > 0, ohlcv_df.diff(1)['close'], 0)
    ohlcv_df['down'] = np.where(ohlcv_df.diff(1)['close'] < 0, ohlcv_df.diff(1)['close'] * (-1), 0)
    ohlcv_df['au'] = ohlcv_df['up'].rolling(period).mean()
    ohlcv_df['ad'] = ohlcv_df['down'].rolling(period).mean()
    ohlcv_df['RSI'] = ohlcv_df.au / (ohlcv_df.ad + ohlcv_df.au) * 100

    del ohlcv_df['up']
    del ohlcv_df['down']
    del ohlcv_df['au']
    del ohlcv_df['ad']

    return ohlcv_df.RSI


def obv(df):
    obv = [0] * len(df)
    for m in range(1, len(df)):
        if df['close'].iloc[m] > df['close'].iloc[m - 1]:
            obv[m] = obv[m - 1] + df['volume'].iloc[m]
        elif df['close'].iloc[m] == df['close'].iloc[m - 1]:
            obv[m] = obv[m - 1]
        else:
            obv[m] = obv[m - 1] - df['volume'].iloc[m]

    return obv


def macd(df, short=12, long=26, signal=9, period=3):
    df['MACD'] = df['close'].ewm(span=short, min_periods=short - 1, adjust=False).mean() - \
                 df['close'].ewm(span=long, min_periods=long - 1, adjust=False).mean()
    df['MACD_Signal'] = df['MACD'].ewm(span=signal, min_periods=signal - 1, adjust=False).mean()
    df['MACD_OSC'] = df.MACD - df.MACD_Signal

    # df['OSC_dev'] = np.nan
    # for i in range(period - 1, len(df)):
    #     df['OSC_dev'].iloc[i], intercept, r_value, p_value, stderr = stats.linregress([x for x in range(period)], df.MACD_OSC[i + 1 - period:i + 1])

    df['MACD_Zero'] = 0

    return


def ema_ribbon(df, ema_1=5, ema_2=8, ema_3=13):
    df['EMA_1'] = df['close'].ewm(span=ema_1, min_periods=ema_1 - 1, adjust=False).mean()
    df['EMA_2'] = df['close'].ewm(span=ema_2, min_periods=ema_2 - 1, adjust=False).mean()
    df['EMA_3'] = df['close'].ewm(span=ema_3, min_periods=ema_3 - 1, adjust=False).mean()

    return


def ema_cross(df, ema_1=30, ema_2=60):
    df['EMA_1'] = df['close'].ewm(span=ema_1, min_periods=ema_1 - 1, adjust=False).mean()
    df['EMA_2'] = df['close'].ewm(span=ema_2, min_periods=ema_2 - 1, adjust=False).mean()

    return


def max_abs_scaler(x):
    scaled_x = x / abs(x).max()
    return scaled_x


def supertrend(df, n=100, f=3):  # df is the dataframe, n is the period, f is the factor; f=3, n=7 are commonly used.
    # Calculation of ATR
    df['H-L'] = abs(df['high'] - df['low'])
    df['H-PC'] = abs(df['high'] - df['close'].shift(1))
    df['L-PC'] = abs(df['low'] - df['close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df['ATR'] = np.nan
    df.ix[n - 1, 'ATR'] = df['TR'][:n].mean()  # .ix is deprecated from pandas verion- 0.19
    for i in range(n, len(df)):
        df['ATR'][i] = (df['ATR'][i - 1] * (n - 1) + df['TR'][i]) / n

    # Calculation of SuperTrend
    df['Upper Basic'] = (df['high'] + df['low']) / 2 + (f * df['ATR'])
    df['Lower Basic'] = (df['high'] + df['low']) / 2 - (f * df['ATR'])
    df['Upper Band'] = df['Upper Basic']
    df['Lower Band'] = df['Lower Basic']
    for i in range(n, len(df)):
        if df['close'][i - 1] <= df['Upper Band'][i - 1]:
            df['Upper Band'][i] = min(df['Upper Basic'][i], df['Upper Band'][i - 1])
        else:
            df['Upper Band'][i] = df['Upper Basic'][i]
    for i in range(n, len(df)):
        if df['close'][i - 1] >= df['Lower Band'][i - 1]:
            df['Lower Band'][i] = max(df['Lower Basic'][i], df['Lower Band'][i - 1])
        else:
            df['Lower Band'][i] = df['Lower Basic'][i]
    df['SuperTrend'] = np.nan
    for i in df['SuperTrend']:
        if df['close'][n - 1] <= df['Upper Band'][n - 1]:
            df['SuperTrend'][n - 1] = df['Upper Band'][n - 1]
        elif df['close'][n - 1] > df['Upper Band'][i]:
            df['SuperTrend'][n - 1] = df['Lower Band'][n - 1]
    for i in range(n, len(df)):
        if df['SuperTrend'][i - 1] == df['Upper Band'][i - 1] and df['close'][i] <= df['Upper Band'][i]:
            df['SuperTrend'][i] = df['Upper Band'][i]
        elif df['SuperTrend'][i - 1] == df['Upper Band'][i - 1] and df['close'][i] >= df['Upper Band'][i]:
            df['SuperTrend'][i] = df['Lower Band'][i]
        elif df['SuperTrend'][i - 1] == df['Lower Band'][i - 1] and df['close'][i] >= df['Lower Band'][i]:
            df['SuperTrend'][i] = df['Lower Band'][i]
        elif df['SuperTrend'][i - 1] == df['Lower Band'][i - 1] and df['close'][i] <= df['Lower Band'][i]:
            df['SuperTrend'][i] = df['Upper Band'][i]

    del df['H-L']
    del df['H-PC']
    del df['L-PC']
    del df['TR']
    del df['ATR']
    del df['Upper Basic']
    del df['Lower Basic']
    del df['Upper Band']
    del df['Lower Band']

    return df


def supertrend2(df, n=100, f=3):  # df is the dataframe, n is the period, f is the factor; f=3, n=7 are commonly used.
    # Calculation of ATR
    df['H-L'] = abs(df['high'] - df['low'])
    df['H-PC'] = abs(df['high'] - df['close'].shift(1))
    df['L-PC'] = abs(df['low'] - df['close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df['ATR'] = np.nan
    df.ix[n - 1, 'ATR'] = df['TR'][:n].mean()  # .ix is deprecated from pandas verion- 0.19
    for i in range(n, len(df)):
        df['ATR'][i] = (df['ATR'][i - 1] * (n - 1) + df['TR'][i]) / n

    # Calculation of SuperTrend
    df['Upper Basic'] = (df['high'] + df['low']) / 2 + (f * df['ATR'])
    df['Lower Basic'] = (df['high'] + df['low']) / 2 - (f * df['ATR'])
    df['Upper Band'] = df['Upper Basic']
    df['Lower Band'] = df['Lower Basic']
    for i in range(n, len(df)):
        if df['close'][i - 1] <= df['Upper Band'][i - 1]:
            df['Upper Band'][i] = min(df['Upper Basic'][i], df['Upper Band'][i - 1])
        else:
            df['Upper Band'][i] = df['Upper Basic'][i]
    for i in range(n, len(df)):
        if df['close'][i - 1] >= df['Lower Band'][i - 1]:
            df['Lower Band'][i] = max(df['Lower Basic'][i], df['Lower Band'][i - 1])
        else:
            df['Lower Band'][i] = df['Lower Basic'][i]
    df['SuperTrend2'] = np.nan
    for i in df['SuperTrend2']:
        if df['close'][n - 1] <= df['Upper Band'][n - 1]:
            df['SuperTrend2'][n - 1] = df['Upper Band'][n - 1]
        elif df['close'][n - 1] > df['Upper Band'][i]:
            df['SuperTrend2'][n - 1] = df['Lower Band'][n - 1]
    for i in range(n, len(df)):
        if df['SuperTrend2'][i - 1] == df['Upper Band'][i - 1] and df['close'][i] <= df['Upper Band'][i]:
            df['SuperTrend2'][i] = df['Upper Band'][i]
        elif df['SuperTrend2'][i - 1] == df['Upper Band'][i - 1] and df['close'][i] >= df['Upper Band'][i]:
            df['SuperTrend2'][i] = df['Lower Band'][i]
        elif df['SuperTrend2'][i - 1] == df['Lower Band'][i - 1] and df['close'][i] >= df['Lower Band'][i]:
            df['SuperTrend2'][i] = df['Lower Band'][i]
        elif df['SuperTrend2'][i - 1] == df['Lower Band'][i - 1] and df['close'][i] <= df['Lower Band'][i]:
            df['SuperTrend2'][i] = df['Upper Band'][i]

    del df['H-L']
    del df['H-PC']
    del df['L-PC']
    del df['TR']
    del df['ATR']
    del df['Upper Basic']
    del df['Lower Basic']
    del df['Upper Band']
    del df['Lower Band']

    return df


def low_high(df, model, crop_size=0, C=None, gamma=None, epsilon=None, chart=0):

    none_indexs = sum(np.isnan(df['MACD_OSC'].values))

    if len(df['MACD_OSC']) - none_indexs >= crop_size:
        pass
    else:
        print('not enough data length')
        return [None] * 2

    #               for realtime svr            #
    if crop_size != 0:

        realtime_osc_reg = [np.nan] * len(df['MACD_OSC'])
        trade_state = [np.nan] * len(df['MACD_OSC'])
        for i in range(none_indexs + crop_size, len(df['MACD_OSC'])):

            x = np.arange(len(df['MACD_OSC']))[i - crop_size:i].reshape(-1, 1)
            y = df['MACD_OSC'].values[i - crop_size:i].reshape(-1, 1)

            # print(x.shape)
            # print(y.shape)
            print(i)

            scaler_x = StandardScaler()
            scaler_y = StandardScaler()
            x = scaler_x.fit_transform(x)
            y = scaler_y.fit_transform(y)
            model.fit(x, y)
            osc_reg = model.predict(x)
            # print(osc_reg)
            # quit()

            # mean_period = 100
            # osc_reg[-1] = np.sum(osc_reg[-mean_period:]) / mean_period
            realtime_osc_reg[i] = osc_reg[-1]

            if chart == 1:
                #           SHOW CHART          #
                span_list = list()
                for j in range(1, len(osc_reg)):
                    if osc_reg[j - 1] <= osc_reg[j]:
                        span_list.append((j, j + 1))

                plt.plot(osc_reg)
                plt.plot(y)

                for j in range(len(span_list)):
                    plt.axvspan(span_list[j][0], span_list[j][1], facecolor='b', alpha=0.7)

                plt.show(block=False)
                # plt.savefig('figure_reg/realtime/%s %s %s %s.png' % (C, gamma, epsilon, i))
                plt.pause(0.3)
                plt.close()
                # quit()

            if osc_reg[-2] < osc_reg[-1]:
                trade_state[i] = 'up'
            else:
                trade_state[i] = 'down'

        return realtime_osc_reg, trade_state

    #           for non realtime svr        #
    else:
        predict_size = 5
        x = np.arange(len(df['MACD_OSC']))[none_indexs:].reshape(-1, 1)
        # x = df['MACD_OSC'].values[none_indexs + predict_size:-predict_size].reshape(-1, 1)
        y = df['MACD_OSC'].values[none_indexs:].reshape(-1, 1)
        # y = df['MACD_OSC'].shift(-predict_size).values[none_indexs + predict_size:-predict_size].reshape(-1, 1)

        scaler_x = StandardScaler()
        scaler_y = StandardScaler()
        x = scaler_x.fit_transform(x)
        y = scaler_y.fit_transform(y)
        # plt.plot(x, color='g')
        # plt.plot(y, color='r')
        # plt.show()
        # quit()

        model.fit(x, y)
        osc_reg = model.predict(x)

        x2 = y
        y2 = osc_reg
        # y2 = gaussian_filter1d(y.reshape(-1, ), sigma=10)
        model.fit(x2, y2)
        osc_reg = model.predict(x2)
        # plt.plot(osc_reg, color='g')
        #
        # x2 = osc_reg.reshape(-1, 1)
        # osc_reg = model.predict(x2)

        # plt.subplot(211)
        # plt.plot(osc_reg)
        # plt.plot(x2, color='orange')
        # # plt.plot(y2, color='r')
        # plt.plot(osc_reg, color='b')
        # plt.show()
        # #
        # # model.fit(x[:-370], y[:-370])
        # # osc_reg = model.predict(x[-370:])
        # # # osc_reg[-370:] = model.predict(x[-370:])
        # #
        # # # # osc_reg = scaler_y.inverse_transform(osc_reg)
        # # # # osc_reg = [np.nan] * none_indexs + list(osc_reg)
        # # # plt.plot(osc_reg)
        # # plt.plot(x)
        # # plt.plot(x[-370:])
        # # plt.show()
        # quit()
        # #
        # # osc_reg = model.predict(x[-100:])
        # # plt.plot(osc_reg)
        # # plt.show()
        #
        # train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3, shuffle=False)
        # model.fit(train_x, train_y)
        # y_pred = model.predict(test_x)
        # plt.plot(y_pred)
        # plt.show()
        # quit()

        trade_state = [np.nan] * len(osc_reg)
        for i in range(len(osc_reg)):
            if osc_reg[i - 1] < osc_reg[i]:
                trade_state[i] = 'up'
            else:
                trade_state[i] = 'down'

        return osc_reg, trade_state


def profitage(Coin, reg_target, short, long, signal, C, gamma, epsilon, model, crop_size, Date='2019-09-25', excel=0, get_fig=0):

    # try:
    #     df = pd.read_excel(dir + '%s %s ohlcv.xlsx' % (Date, Coin), index_col=0)
    #
    # except Exception as e:
    #     print('Error in loading ohlcv_data :', e)
    #     return 1.0, 1.0, 1.0

    if not Coin.endswith('.xlsx'):
        df = pybithumb.get_ohlcv(Coin, 'KRW', 'minute1')
    else:
        df = pd.read_excel(dir + '%s' % Coin, index_col=0)

    macd(df, short=short, long=long, signal=signal)
    print('data lengh : ', len(df))
    # quit()

    # print(df[reg_target])
    # quit()
    osc_reg, trade_state = low_high(df, model, crop_size, C, gamma, epsilon, chart=0)
    df = df.iloc[-len(trade_state):]

    df['osc_reg'] = osc_reg
    df['trade_state'] = trade_state
    # print(df)
    # quit()

    # 거래 수수료
    fee = 0.005

    # DatetimeIndex 를 지정해주기 번거로운 상황이기 때문에 틱을 기준으로 거래한다.
    # DatetimeIndex = df.axes[0]

    # ------------------- 상향 / 하향 매도 여부와 이익률 계산 -------------------#

    # high 가 SPP 를 건드리거나, low 가 SPM 을 건드리면 매도 체결 [ 매도 체결될 때까지 SPP 와 SPM 은 유지 !! ]
    length = len(df.index) - 1  # 데이터 갯수 = 1, m = 0  >> 데이터 갯수가 100 개면 m 번호는 99 까지 ( 1 - 100 )

    # 병합할 dataframe 초기화
    bprelay = pd.DataFrame(index=df.index, columns=['bprelay'])
    condition = pd.DataFrame(index=df.index, columns=["Condition"])
    Profits = pd.DataFrame(index=df.index, columns=["Profits"])
    # price_point = pd.DataFrame(index=np.arange(len(df)), columns=['Price_point'])

    Profits.Profits = 1.0
    Minus_Profits = 1.0

    # 오더라인과 매수가가 정해진 곳에서부터 일정시간까지 오더라인과 매수가를 만족할 때까지 대기  >> 일정시간,
    m = 0
    spanlist = []
    spanlist_limit = []
    spanlist_breakaway = []
    while m <= length:

        #               bp 찾기               #
        while True:

            #           Realtime SVR            #
            #   down >> up 이면 매수 등록    #
            if (df['trade_state'].iloc[m - 1] == 'down') and (df['trade_state'].iloc[m] == 'up'):
            # if (df['trade_state'].iloc[m - 2] == 'down') and (df['trade_state'].iloc[m] == 'up') and (df['trade_state'].iloc[m - 1] == 'up'):
                print('매수 등록')
                break

            m += 1
            if m > length:
                break

        if m > length:  # or pd.isnull(df.iloc[m]['BuyPrice']): << 이게 무슨 의미인지 모르겠음..
            break
        #       매수가는 55초 이후 up 이라고 결정된 봉의 종가    #
        bp = df['close'].iloc[m]

        #       매수 등록 완료, 매수 체결 대기     #
        start_m = m
        while True:
            bprelay["bprelay"].iloc[m] = bp

            #               매수 체결 조건 = 다음 봉의 저가가 매수가 이하이면 체결되었다고 볼 수 있다.            #
            try:
                if df.iloc[m + 1]['low'] <= bp:  # and (df.iloc[m]['high'] != df.iloc[m]['low']):  # 조건을 만족할 경우 spp 생성
                    # print(df['low'].iloc[m], '<=', bp, '?')

                    condition.iloc[m] = '매수 체결'
                    # bprelay["bprelay"].iloc[m] = bprelay["bprelay"].iloc[m - 1]

                    m += 1
                    break
                else:
                    m += 1
                    if m > length:
                        break

                    #       매수 대기 term      #
                    if m - start_m > 5:
                        break

                    condition.iloc[m] = '매수 대기'
                    bprelay["bprelay"].iloc[m] = bprelay["bprelay"].iloc[m - 1]

            except Exception as e:
                print('m > length :', e)
                break

        #               매수 체결, 매도 대기                #
        #           up >> down 될 때까지 (조건 만족시 매도 체결)         #
        start_m = m
        high_condition = 0
        while True:

            #      고점 매도       #
            # if df['trade_state'].iloc[m] == 'down' and df['trade_state'].iloc[m - 1] == 'down':
            # if df['trade_state'].iloc[m] == 'down' and df['trade_state'].iloc[m - 1] == 'down'\
            #         and df['trade_state'].iloc[m - 2] == 'down':
            if df['trade_state'].iloc[m] == 'down':
                print('매도 체결')
                high_condition = 1.
                break

            condition.iloc[m] = '매도 대기'
            bprelay["bprelay"].iloc[m] = bprelay["bprelay"].iloc[m - 1]

            m += 1
            if m > length:
                break

        if m > length:
            # print(condition.iloc[m - 1])
            break

        elif high_condition == 1.:

            if np.isnan(bprelay["bprelay"].iloc[m]):
                bprelay["bprelay"].iloc[m] = bprelay["bprelay"].iloc[m - 1]

            condition.iloc[m] = "매도 체결"
            Profits.iloc[m] = df.iloc[m]['close'] / bprelay["bprelay"].iloc[m] - fee

            if float(Profits.iloc[m]) < 1:
                Minus_Profits *= float(Profits.iloc[m])
                try:
                    if start_m == m:
                        m += 1
                    spanlist.append((start_m, m))
                    spanlist_breakaway.append((start_m, m))

                except Exception as e:
                    pass

            else:
                try:
                    if start_m == m:
                        m += 1
                    spanlist.append((start_m, m))
                    spanlist_limit.append((start_m, m))

                except Exception as e:
                    pass

        # 체결시 재시작
        m += 1

    df = pd.merge(df, bprelay, how='outer', left_index=True, right_index=True)
    df = pd.merge(df, condition, how='outer', left_index=True, right_index=True)
    df = pd.merge(df, Profits, how='outer', left_index=True, right_index=True)
    df = df.reset_index(drop=True)

    if Coin.endswith('.xlsx'):
        Coin = Coin.split()[1] + '_from_excel'

    if excel == 1:
        # df.to_excel("./BackTest/%s BackTest %s.xlsx" % (Date, Coin))
        df.to_excel("excel_check/%s BackTest %s.xlsx" % (Date, Coin))

    profits = Profits.cumprod()  # 해당 열까지의 누적 곱!
    # print(profits)

    if np.isnan(profits.iloc[-1].item()):
        return 1.0, 1.0, 1.0, 1.0, 1.0

    # [-1] 을 사용하려면 데이터가 존재해야 되는데 데이터 전체가 결측치인 경우가 존재한다.
    if len(profits) == 0:
        return 1.0, 1.0, 1.0, 1.0, 1.0

    elif float(profits.iloc[-1]) != 1.0 and get_fig == 1:

        # 거래 체결마다 subplot 1,2 저장
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(311)
        ochl = df.iloc[:, :4]
        index = np.arange(len(ochl))
        ochl = np.hstack((np.reshape(index, (-1, 1)), ochl))
        mf.candlestick_ochl(ax, ochl, width=0.5, colorup='r', colordown='b')
        # plt.plot(df['SuperTrend'], 'blue', label='ST', linewidth=1.0)
        # plt.plot(df['SuperTrend'] + df['R2G_Gap'] * offset_low , 'c', label='ST low', linewidth=1.0)
        # plt.plot(df['SuperTrend'] + df['R2G_Gap'] * offset_high, 'm', label='ST high', linewidth=1.0)
        plt.xlim(0, len(ochl))
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        # plt.legend(loc='upper right', fontsize=10)
        plt.title('%.3f %.3f %.3f' % (float(profits.iloc[-1]), float(profits.iloc[-1]) / Minus_Profits, Minus_Profits))

        for trade_num in range(len(spanlist_limit)):
            plt.axvspan(spanlist_limit[trade_num][0], spanlist_limit[trade_num][1], facecolor='c', alpha=0.5)
        for trade_num in range(len(spanlist_breakaway)):
            plt.axvspan(spanlist_breakaway[trade_num][0], spanlist_breakaway[trade_num][1], facecolor='m', alpha=0.5)

        plt.subplot(312)
        plt.plot(osc_reg, 'gold', label='oscillator', linewidth=1.0)
        plt.plot(df[reg_target])
        plt.xlim(0, len(ochl))
        # plt.title('Prefered MACD_OSC', fontsize=10)
        plt.title('Realtime %s' % reg_target, fontsize=10)
        # plt.plot(df[['MACD_Zero']], 'g', label='zero', linewidth=1.0)
        span_list = list()
        for i in range(1, len(df)):
            if osc_reg[i - 1] < osc_reg[i]:
                span_list.append((i, i + 1))
            # if trade_state[i] == 'up':
            #     span_list.append((i, i + 1))
        for i in range(len(span_list)):
            plt.axvspan(span_list[i][0], span_list[i][1], facecolor='c', alpha=0.5)

        plt.subplot(313)
        osc_reg, _ = low_high(df, model)
        # osc_reg, _ = low_high(df[reg_target], model)
        plt.plot(osc_reg, 'gold', label='oscillator', linewidth=1.0)
        plt.plot(df[reg_target])
        plt.xlim(0, len(ochl))
        plt.title('Prefered %s' % reg_target, fontsize=10)
        # plt.plot(df[['MACD_Zero']], 'g', label='zero', linewidth=1.0)
        span_list = list()
        for i in range(1, len(df)):
            if osc_reg[i - 1] < osc_reg[i]:
                span_list.append((i, i + 1))
        for i in range(len(span_list)):
            plt.axvspan(span_list[i][0], span_list[i][1], facecolor='c', alpha=0.5)

        # plot 저장 & 닫기
        plt.show()
        # plt.savefig("./figure_reg/reg_result/%s %s %s %s %s %s.png" % (Date, Coin, C, gamma, epsilon, crop_size), dpi=300)
        # plt.close()

    if profits.values[-1] != 1.:
        profits_sum = 0
        for i in range(len(Profits)):
            if Profits.values[i] != 1.:
                profits_sum += Profits.iloc[i]
        profits_avg = profits_sum / sum(Profits.values != 1.)

    else:
        profits_avg = [1.]

    # print(Profits.values.min())
    # quit()
    # print(Profits.values != 1.)
    # print(profits_avg)
    # quit()
    # print(std)

    return float(profits.iloc[-1]), float(profits.iloc[-1]) / Minus_Profits, Minus_Profits, profits_avg[
        0], Profits.values.min()


if __name__ == "__main__":

    home_dir = os.path.expanduser('~')
    dir = home_dir + '/OneDrive/CoinBot/ohlcv_concat/'
    ohlcv_list = os.listdir(dir)

    # Coinlist = pybithumb.get_tickers()
    # Fluclist = []
    # while True:
    #     try:
    #         for Coin in Coinlist:
    #             tickerinfo = pybithumb.PublicApi.ticker(Coin)
    #             data = tickerinfo['data']
    #             fluctate = data['fluctate_rate_24H']
    #             Fluclist.append(fluctate)
    #             time.sleep(1 / 90)
    #         break
    #
    #     except Exception as e:
    #         Fluclist.append(None)
    #         print('Error in making Topcoin :', e)
    #
    # Fluclist = list(map(float, Fluclist))
    # series = pd.Series(Fluclist, Coinlist)
    # series = series.sort_values(ascending=False)
    #
    # series = series[:10]
    # TopCoin = list(series.index)
    # print(TopCoin)
    # quit()
    # TopCoin = ['TMTG', 'FNB', 'APIX', 'LBA', 'SNT', 'MXC', 'BTG', 'QKC', 'LUNA', 'PIVX']
    TopCoin = ['FNB']

    #
    # print(Date)
    # quit()
    #     # Date = '2020-02-27'

    # ohlcv_list = ['2019-10-05 LAMB ohlcv.xlsx']
    ohlcv_concat = list()
    for file in ohlcv_list:
        if file.split()[1] in TopCoin:
            ohlcv_concat.append(file)
            # print(file)

    Date = str(datetime.now()).split()[0]
    # excel_list = os.listdir('./BestSet/Test_ohlc/')
    # random.shuffle(excel_list)
    # print(excel_list)
    # quit()
    total_df = pd.DataFrame(
        columns=['short', 'long', 'signal', 'total_profit_avg', 'plus_profit_avg', 'minus_profit_avg', 'avg_profit_avg',
                 'min_profit_avg'])

    C, gamma, epsilon = 100, 1000, 1.
    # model = SVR(kernel='rbf', C=C, gamma=gamma, epsilon=epsilon)     # kernel='rbf', C=100, gamma=10, epsilon=0.1
    # model = SVR(kernel='rbf', C=10, gamma=100, epsilon=0.1)     # kernel='rbf', C=100, gamma=10, epsilon=0.1
    # model = SVR(kernel='sigmoid')     # kernel='rbf', C=100, gamma=10, epsilon=0.1
    model = SVR(kernel='poly', gamma=1)     # kernel='rbf', C=100, gamma=10, epsilon=0.1
    #  500, 20, 1
    #  0.1, 100, 0.5

    for short in range(20, 100, 5):
        for long in range(short + 3, short + 100, 5):
            for signal in range(5, 50, 3):

                # short, long, signal = 12, 26, 9
                short, long, signal = 105, 168, 32

                #       Make folder      #
                try:
                    os.mkdir("./Figure_pred/%s_%s_%s/" % (short, long, signal))

                except Exception as e:
                    print(e)

                #
                for Coin in TopCoin:
                    # print(Coin)
                    # Coin = excel_list[3]
                    try:
                        print(Coin, profitage(Coin, 'MACD_OSC',
                                              short, long, signal,
                                              C, gamma, epsilon, model, crop_size=0, Date=Date, get_fig=1, excel=0))

                    except Exception as e:
                        print(e)
                        continue
                quit()

                total_profit = 0
                plus_profit = 0
                minus_profit = 0
                avg_profit = 0
                min_profit = 0
                for Coin in excel_list:
                    start = time.time()
                    result = profitage(Coin, short, long, signal, Date=Date, get_fig=1)
                    # quit()
                    total_profit += result[0]
                    plus_profit += result[1]
                    minus_profit += result[2]
                    avg_profit += result[3]
                    min_profit += result[4]
                total_profit_avg = total_profit / len(excel_list)
                plus_profit_avg = plus_profit / len(excel_list)
                minus_profit_avg = minus_profit / len(excel_list)
                avg_profit_avg = avg_profit / len(excel_list)
                min_profit_avg = min_profit / len(excel_list)
                print(short, long, signal, total_profit_avg, plus_profit_avg, minus_profit_avg, avg_profit_avg,
                      min_profit_avg, '%.3f second' % (time.time() - start))

                result_df = pd.DataFrame(data=[
                    [short, long, signal, total_profit_avg, plus_profit_avg, minus_profit_avg, avg_profit_avg,
                     min_profit_avg]],
                                         columns=['short', 'long', 'signal', 'total_profit_avg', 'plus_profit_avg',
                                                  'minus_profit_avg', 'avg_profit_avg', 'min_profit_avg'])
                total_df = total_df.append(result_df)

            total_df.to_excel('./BestSet/total_df %s.xlsx' % short)
            break
