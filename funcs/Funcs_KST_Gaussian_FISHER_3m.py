import pybithumb
import numpy as np
import pandas as pd
from datetime import datetime
import os
from scipy import stats
from asq.initiators import query
import matplotlib.pyplot as plt
from sklearn.preprocessing import MaxAbsScaler
import mpl_finance as mf
import time
from scipy.ndimage.filters import gaussian_filter1d

pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 2500)
pd.set_option('display.max_columns', 2500)


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


def stochrsi(ohlcv_df, period_rsi=14, period_sto=14, k=3, d=3):
    rsi(ohlcv_df, period_rsi)
    ohlcv_df['H_RSI'] = ohlcv_df.RSI.rolling(period_sto).max()
    ohlcv_df['L_RSI'] = ohlcv_df.RSI.rolling(period_sto).min()
    ohlcv_df['StochRSI'] = (ohlcv_df.RSI - ohlcv_df.L_RSI) / (ohlcv_df.H_RSI - ohlcv_df.L_RSI) * 100

    ohlcv_df['StochRSI_K'] = ohlcv_df.StochRSI.rolling(k).mean()
    ohlcv_df['StochRSI_D'] = ohlcv_df.StochRSI_K.rolling(d).mean()

    del ohlcv_df['H_RSI']
    del ohlcv_df['L_RSI']

    return


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


def macd(df, short=12, long=26, signal=9):
    df['MACD'] = df['close'].ewm(span=short, min_periods=short - 1, adjust=False).mean() - \
                 df['close'].ewm(span=long, min_periods=long - 1, adjust=False).mean()
    df['MACD_Signal'] = df['MACD'].ewm(span=signal, min_periods=signal - 1, adjust=False).mean()
    df['MACD_OSC'] = df.MACD - df.MACD_Signal
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
    df.ix[n - 1, 'ATR'] = df['TR'][:n - 1].mean()  # .ix is deprecated from pandas verion- 0.19
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

    return df


def to_timestamp(datetime_):
    timestamp = time.mktime(datetime.strptime(str(datetime_), '%Y-%m-%d %H:%M:%S').timetuple())

    return int(timestamp)


def timestamped_index(df):
    # df.index = pd.Series(df.index).apply(to_timestamp)

    return pd.Series(df.index).apply(to_timestamp)


def last_datetime_indexing(df, interval):
    last_datetime_index = df.index[-1]

    last_min = int(str(last_datetime_index).split(':')[1])
    while last_min % interval != 0:
        last_min -= 1

    modified_index = ":".join([str(last_datetime_index).split(':')[0], "%2f:00" % last_min])
    modified_index = pd.to_datetime(modified_index)

    df.index.values[-1] = modified_index

    return


from finta import TA


def profitage(Coin, long_signal_value, short_signal_value, long_period, short_period, wait_tick=3, Date='2019-09-25',
              excel=0, get_fig=0):

    global length
    if not Coin.endswith('.xlsx'):
        df = pybithumb.get_ohlcv(Coin, 'KRW', 'minute3')
        df_short = pybithumb.get_ohlcv(Coin, 'KRW', 'minute1')
        # df_short2 = pybithumb.get_ohlcv(Coin, 'KRW', 'minute5')
    else:
        df = pd.read_excel(dir + '%s' % Coin, index_col=0)
    # df = pd.read_excel('./BestSet/Test_ohlc/%s' % Coin, index_col=0)

    df['KST'] = TA.KST(df, r1=10, r2=15, r3=20, r4=30)['signal']
    df['KST'] = np.where(np.isnan(df['KST']), 0, df['KST'])
    df['KST_TREND'] = np.where(df['KST'].shift(1) <= df['KST'], 'UP', 'DOWN')

    df['KST_Origin'] = TA.KST(df, r1=10, r2=15, r3=20, r4=30)['KST']
    df['KST_Origin'] = np.where(np.isnan(df['KST_Origin']), 0, df['KST_Origin'])

    df['Fisher_long'] = TA.FISH(df, period=long_period)
    df['Fisher_long_ic'] = np.NaN
    df['Fisher_short'] = np.NaN
    # df['Fisher_short2'] = np.NaN

    df_short['Fisher_long'] = TA.FISH(df_short, period=long_period)
    df_short['Fisher_short'] = TA.FISH(df_short, period=short_period)

    # df_short2['Fisher_short'] = TA.FISH(df_short2, period=short_period)

    #       마지막 MINUTE이 해당 INTERVAL로 나누어떨어지지 않으면        #
    #       나누어 떨어지는 MINUTE으로 변경해준다.        #
    # last_datetime_indexing(df_short, 3)
    # # last_datetime_indexing(df_short2, 5)
    #
    # #           INTERVAL MATCHING           #
    # index_df = timestamped_index(df)
    # index_df_short = timestamped_index(df_short)
    # # index_df_short2 = timestamped_index(df_short2)
    #
    # stime = time.time()
    # start_j = 1
    # for i in range(len(index_df)):
    #     for j in range(start_j, len(index_df_short)):
    #
    #         if j == len(index_df_short) - 1:
    #             if index_df_short[j] <= index_df[i]:
    #                 df.values[[i], -2:] = df_short.values[[j], -2:]
    #
    #         if index_df_short[j - 1] <= index_df[i] < index_df_short[j]:
    #             df.values[[i], -2:] = df_short.values[[j - 1], -2:]
    #             start_j = j

    # start_j = 1
    # for i in range(len(index_df)):
    #     for j in range(start_j, len(index_df_short2)):
    #
    #         if j == len(index_df_short2) - 1:
    #             if index_df_short2[j] <= index_df[i]:
    #                 df.values[[i], -1:] = df_short2.values[[j], -1:]
    #
    #         if index_df_short2[j - 1] <= index_df[i] < index_df_short2[j]:
    #             df.values[[i], -1:] = df_short2.values[[j - 1], -1:]
    #             start_j = j

    # print(df.tail(10))
    # print(df_short2.tail(10))
    # quit()
    # print('elapsed time :', time.time() - stime)
    df['Fisher_long_ic_TREND'] = np.where(df['Fisher_long_ic'].shift(1) <= df['Fisher_long_ic'], 'UP', 'DOWN')
    for i in range(1, len(df)):
        if df['Fisher_long_ic'].iloc[i] == df['Fisher_long_ic'].iloc[i - 1]:
            df['Fisher_long_ic_TREND'].iloc[i] = df['Fisher_long_ic_TREND'].iloc[i - 1]

    # print(df)
    # quit()

    df['Fisher_trend'] = np.where(df['Fisher_long'] > df['Fisher_long'].shift(1), 'UP', 'DOWN')
    df['Fisher_IP'] = np.where(df['Fisher_trend'].shift(1) != df['Fisher_trend'], 1, 0)
    # df['Fisher_long_gaussian'] = df.Fisher_long.rolling(60).mean()
    # df['Fisher_short_gaussian'] = df.Fisher_short.rolling(60).mean()

    #           REALTIME CURVE          #
    smoothed_curve = [np.nan] * len(df)
    smoothed_curve_short = [np.nan] * len(df)
    close_list = [np.nan] * len(df)
    period = 5
    sigma = 5
    for i in range(period, len(df)):
        y = df['Fisher_long'].values[i + 1 - period:i + 1]
        y_short = df['KST_Origin'].values[i + 1 - period:i + 1]
        y_smooth = gaussian_filter1d(y, sigma=sigma)
        y_smooth_short = gaussian_filter1d(y_short, sigma=sigma)
        smoothed_curve[i] = y_smooth[-1]
        smoothed_curve_short[i] = y_smooth_short[-1]

        # close = df.close.values[i + 1 - period:i + 1]
        # close = gaussian_filter1d(close, sigma=sigma, mode='reflect')
        # close_list[i] = close[-1]

    df['Fisher_GAUSSIAN'] = smoothed_curve
    df['Fisher_GAUSSIAN_TREND'] = np.where(df['Fisher_GAUSSIAN'].shift(1) <= df['Fisher_GAUSSIAN'], 'UP', 'DOWN')

    df['KST_GAUSSIAN'] = smoothed_curve_short
    df['KST_GAUSSIAN_TREND'] = np.where(df['KST_GAUSSIAN'].shift(1) <= df['KST_GAUSSIAN'], 'UP', 'DOWN')

    kst_vertical_gap = 20
    kst_horizontal_ticks = 5
    upper_line = 10
    lower_line = -15

    # long_signal_value = 0
    # short_signal_value = 2.0
    long_ic_signal_value = -2.0
    detect_close_value = long_signal_value
    high_check_value = 2
    high_check_value_exit = 1.7
    high_check_value2 = 0
    high_check_value2_exit = 0
    exit_short_signal_value = -1.0

    #           SMMA DC          #
    smma_period = 200
    smma_accept_range = -0.08
    dc_accept_range = 0.05
    dc_accept_gap_range = -0.015
    df['SMMA'] = TA.SMMA(df, period=smma_period)
    dc_period_entry = 20
    df['DC_LOW_ENTRY'] = TA.DO(df, lower_period=dc_period_entry, upper_period=dc_period_entry)['LOWER']
    df['SMMA_DC_GAP'] = (df['DC_LOW_ENTRY'] - df['SMMA']) / (df['high'].max() - df['low'].min())

    #           DC CLOSE           #
    dc_period = 80
    df['DC_LOW'] = TA.DO(df, lower_period=dc_period, upper_period=dc_period)['LOWER']
    df['CLOSE_DC_GAP'] = (df['close'] - df['DC_LOW']) / (df['high'].max() - df['low'].min())
    df['DC_TREND'] = np.where(df['DC_LOW'].shift(1) <= df['DC_LOW'], 'UP', 'DOWN')
    for i in range(1, len(df)):
        if df['DC_LOW'].iloc[i] == df['DC_LOW'].iloc[i - 1]:
            df['DC_TREND'].iloc[i] = df['DC_TREND'].iloc[i - 1]

    # print(df)
    # quit()

    #                               LONG SHORT                              #
    trade_state = [0] * len(df)
    for i in range(2, len(df)):

        # if long_signal_value_under < df['Fisher_long_gaussian'].iloc[i] <= long_signal_value_upper:
        #     if df['Fisher_long_gaussian'].iloc[i] > df['Fisher_long_gaussian'].iloc[i - 1]:
        #         trade_state[i] = 1

        # if df['Fisher_long'].iloc[i - 1] <= long_signal_value:
        #     if df['Fisher_long'].iloc[i] > df['Fisher_long'].iloc[i - 1] < df['Fisher_long'].iloc[i - 2] \
        #             and df['close'].iloc[i] > clearance(df['SMMA'].iloc[i] * smma_accept_range):
            # if df['Fisher_long_gaussian'].iloc[i] > df['Fisher_long_gaussian'].iloc[i - 1] < df['Fisher_long_gaussian'].iloc[i - 2] \
            #         and df['close'].iloc[i] > clearance(df['SMMA'].iloc[i] * smma_accept_range):
            #     trade_state[i] = 1

        #       KST LONG      #
        # if df['KST_TREND'].iloc[i - 1] == 'DOWN' and df['KST_TREND'].iloc[i] == 'UP' and 20 > df['KST'].iloc[i] >= -15:
        #     trade_state[i] = 1.5

        #       3M FISHER LONG POSITION FINDING     #
        if df['Fisher_long'].iloc[i] <= long_signal_value < df['Fisher_long'].iloc[i - 1]: #and df['close'].iloc[i] > clearance(df['SMMA'].iloc[i] * smma_accept_range):
            for j in range(i + 1, len(df)):
                if df['KST_GAUSSIAN_TREND'].iloc[j - 1] == 'DOWN' and df['KST_GAUSSIAN_TREND'].iloc[j] == 'UP':
                    # if df['KST_GAUSSIAN'].iloc[j] - df['KST_GAUSSIAN'].iloc[j - 1] >= 1.0:
                        if df['Fisher_long'].iloc[j] < 0:
                            if df['DC_TREND'].iloc[j] == 'UP':
                            # and upper_line > df['KST_GAUSSIAN'].iloc[j] >= lower_line\
                            # and df['KST'].iloc[j] != 0:  # 초기에 진입하는거를 방지하고 KST와 비교하기 위해서
                                trade_state[j] = 1.5

                            #       DC가 DOWN일 경우 DOWN의 시작점과 진입 지점사이의 GAP을 조사한다.     #
                            else:
                                copy_j = j
                                while True:
                                    copy_j -= 1
                                    if df['DC_TREND'].iloc[copy_j] == 'UP':
                                        break
                                print('DC GAP :', copy_j, j, (df['DC_LOW'].iloc[j] - df['DC_LOW'].iloc[copy_j]) / (df['high'].max() - df['low'].min()))
                                if (df['DC_LOW'].iloc[j] - df['DC_LOW'].iloc[copy_j]) / (df['high'].max() - df['low'].min()) > dc_accept_gap_range:
                                    trade_state[j] = 1.5


                        # break

                if df['Fisher_long'].iloc[j] > 0:
                    break

                    #     #   이전 변곡점과의 높이차이가 10이상    #
                    # copy_j = j
                    # while True:
                    #     copy_j -= 1
                    #     if df['KST_GAUSSIAN_TREND'].iloc[copy_j - 1] == 'UP' and df['KST_GAUSSIAN_TREND'].iloc[copy_j] == 'DOWN':
                    #         break
                    #     if copy_j == 0:
                    #         break
                    #
                    # if copy_j != 0:
                    #     if df['KST_GAUSSIAN'].iloc[copy_j] - df['KST_GAUSSIAN'].iloc[j] >= kst_vertical_gap:
                    #         print('KST_GAUSSIAN GAP :', j, df['KST_GAUSSIAN'].iloc[copy_j] - df['KST_GAUSSIAN'].iloc[j])
                    #         trade_state[j] = 1.5

        # if df['KST_TREND'].iloc[i - 1] == 'UP' and df['KST_TREND'].iloc[i] == 'DOWN':
        #     trade_state[i] = 2
        if df['KST_GAUSSIAN_TREND'].iloc[i - 1] == 'UP' and df['KST_GAUSSIAN_TREND'].iloc[i] == 'DOWN':
            trade_state[i] = 2

        if df['Fisher_long'].iloc[i - 1] >= short_signal_value:
            if df['Fisher_long'].iloc[i] < df['Fisher_long'].iloc[i - 1]:
                trade_state[i] = 2.5

        # if df['Fisher_short'].iloc[i - 1] >= short_signal_value:
        #     if df['Fisher_short'].iloc[i] < df['Fisher_short'].iloc[i - 1]:
        #     # if df['Fisher_short'].iloc[i] < df['Fisher_short'].iloc[i - 1] > df['Fisher_short'].iloc[i - 2]:
        #         if i + 3 < len(df):
        #             while True:
        #                 if df['Fisher_short'].iloc[i] != df['Fisher_short'].iloc[i + 1]:
        #                     break
        #                 else:
        #                     i += 1
        #                     if i >= len(df):
        #                         break
        #             if i >= len(df):
        #                 break
        #             trade_state[i] = 2

        # if df['Fisher_short'].iloc[i] < df['Fisher_short'].iloc[i - 1]:
        #     trade_state[i] = 2

        # if df['Fisher_short'].iloc[i - 1] >= short_signal_value:
        #     if df['Fisher_short_gaussian'].iloc[i] < df['Fisher_short_gaussian'].iloc[i - 1]:
        #         trade_state[i + 2] = 2

        # if df['Fisher_long'].iloc[i] < long_signal_value:
        #     trade_state[i] = -1

    df['trade_state'] = trade_state
    # print(df.trade_state)
    # quit()

    span_list_target = list()
    span_list_target_ic = list()
    target = df['trade_state']
    for i in range(1, len(target)):
        if df['KST_GAUSSIAN_TREND'].iloc[i] != df['KST_GAUSSIAN_TREND'].iloc[i - 1]:
            span_list_target.append((i, i + 1))
        if target.iloc[i] == 1.5:
            span_list_target_ic.append((i, i + 1))

    span_list_target_fisher = list()
    # span_list_target_ic = list()
    target = df['trade_state']
    for i in range(2, len(target)):
        if df['Fisher_GAUSSIAN_TREND'].iloc[i] != df['Fisher_GAUSSIAN_TREND'].iloc[i - 1]:
            span_list_target_fisher.append((i, i + 1))
        # if target.iloc[i] == 1.5:
        #     span_list_target_ic.append((i, i + 1))

    # fig = plt.figure(figsize=(10, 7))
    # ax = fig.add_subplot(311)
    # ochl = df.iloc[:, :4]
    # index = np.arange(len(ochl))
    # ochl = np.hstack((np.reshape(index, (-1, 1)), ochl))
    # mf.candlestick_ochl(ax, ochl, width=0.5, colorup='r', colordown='b')
    #
    # for i in range(len(span_list_target)):
    #     plt.axvspan(span_list_target[i][0], span_list_target[i][1], facecolor='c', alpha=0.7)
    # for i in range(len(span_list_target_ic)):
    #     plt.axvspan(span_list_target_ic[i][0], span_list_target_ic[i][1], facecolor='m', alpha=0.7)
    #
    # plt.subplot(312)
    # plt.plot(df.Fisher_long.values)
    #
    # for i in range(len(span_list_target)):
    #     plt.axvspan(span_list_target[i][0], span_list_target[i][1], facecolor='c', alpha=0.7)
    # for i in range(len(span_list_target_ic)):
    #     plt.axvspan(span_list_target_ic[i][0], span_list_target_ic[i][1], facecolor='m', alpha=0.7)
    #
    # plt.subplot(313)
    # plt.plot(df.Fisher_long_gaussian.values)
    #
    # for i in range(len(span_list_target)):
    #     plt.axvspan(span_list_target[i][0], span_list_target[i][1], facecolor='c', alpha=0.7)
    # for i in range(len(span_list_target_ic)):
    #     plt.axvspan(span_list_target_ic[i][0], span_list_target_ic[i][1], facecolor='m', alpha=0.7)
    #
    # plt.show()
    # return

    # 매수 시점 = 급등 예상시, 매수가 = 이전 종가
    # df['BuyPrice'] = np.where(df['trade_state']== 1., df['close'], np.nan)
    df['BuyPrice'] = np.where(df['trade_state'] == 1.5, df['close'], np.nan)
    # df['BuyPrice'] = np.where(df['trade_state'] == 1.5, df['close'].shift(1), np.nan)
    # df['BuyPrice'] = df['BuyPrice'].apply(clearance)

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
    support = pd.DataFrame(index=df.index, columns=["support_line"])
    # price_point = pd.DataFrame(index=np.arange(len(df)), columns=['Price_point'])

    Profits.Profits = 1.0
    Minus_Profits = 1.0
    exit_fluc = 0
    exit_count = 0

    # 오더라인과 매수가가 정해진 곳에서부터 일정시간까지 오더라인과 매수가를 만족할 때까지 대기  >> 일정시간,
    m = 0
    while m <= length:

        while True:  # bp 찾기
            if pd.notnull(df.iloc[m]['BuyPrice']):
                break
            m += 1
            if m > length:
                break

        if (m > length) or pd.isnull(df.iloc[m]['BuyPrice']):
            break

        bp = df.iloc[m]['BuyPrice']
        bprelay["bprelay"].iloc[m] = bp

        # sp = min(df['low'].iloc[ip_before:ip_after])
        # bp = clearance(sp * 1.0025)

        #       매수 등록 완료, 매수 체결 대기     #
        start_m = m
        while True:
            # support["support_line"].iloc[m] = sp

            # 매수 체결 조건
            if df.iloc[m]['low'] <= bprelay["bprelay"].iloc[m]:  # and (df.iloc[m]['high'] != df.iloc[m]['low']):  # 조건을 만족할 경우 spp 생성
                # print(df['low'].iloc[m], '<=', bp, '?')

                condition.iloc[m] = '매수 체결'
                # bprelay["bprelay"].iloc[m] = bprelay["bprelay"].iloc[m - 1]

                m += 1
                break
            else:
                #   SINGAL 다음 시점에서 CLOSE와 매수가의 GAP이 0.6% 미만이면, CLOSE로 매수한다.
                if df['close'].iloc[m] / bp <= 1.006:
                    bprelay["bprelay"].iloc[m] = df['close'].iloc[m]
                    continue

                m += 1
                if m > length:
                    break
                if m - start_m >= wait_tick:
                    break

                condition.iloc[m] = '매수 대기'
                bprelay["bprelay"].iloc[m] = clearance(bprelay["bprelay"].iloc[m - 1] * 1.0015)
                support["support_line"].iloc[m] = support["support_line"].iloc[m - 1]

    # if excel == 1:
    #     df = pd.merge(df, condition, how='outer', left_index=True, right_index=True)
    #     df = pd.merge(df, bprelay, how='outer', left_index=True, right_index=True)
    #     df = pd.merge(df, support, how='outer', left_index=True, right_index=True)
    #     df.to_excel("./excel_check/%s BackTest %s.xlsx" % (Date, Coin))
    #     quit()

    #           지정 매도가 표시 완료           #

    #                               수익성 검사                                  #
    m = 0
    spanlist = []
    spanlist_limit = []
    spanlist_breakaway = []
    while m <= length:  # 초반 시작포인트 찾기

        while True:  # SPP 와 SPM 찾긴
            if condition.iloc[m]['Condition'] == '매수 체결':
                # and type(df.iloc[m]['SPP']) != str:  # null 이 아니라는 건 오더라인과 매수가로 캡쳐했다는 거
                break
            m += 1
            if m > length:  # 차트가 끊나는 경우, 만족하는 spp, spm 이 없는 경우
                break

        if (m > length) or pd.isnull(condition.iloc[m]['Condition']):
            break

        start_m = m
        buy_signal_m = start_m
        # exit_price = 0.975
        exit_signal = 0
        dc_descend_cnt = 0
        fisher_ic_cnt = 0
        sell_switch = False
        exit_switch = False

        if df['trade_state'].iloc[start_m] != 1.5:
            while True:
                buy_signal_m -= 1
                if df['trade_state'].iloc[buy_signal_m] == 1.5:
                    break
        while True:

            #       매도 시그널       #
            if df.trade_state.iloc[m] == 2:

                print('KST_GAUSSIAN GAP:', m, df['KST_GAUSSIAN'].iloc[m] - df['KST_GAUSSIAN'].iloc[start_m])
                if m - start_m > kst_horizontal_ticks:
                    #       자잘거림 보정     #
                    #       이전 변곡점과의 상하좌우 폭 비교      #
                    copy_m = m
                    while True:
                        copy_m -= 1
                        if df['KST_GAUSSIAN_TREND'].iloc[copy_m] != df['KST_GAUSSIAN_TREND'].iloc[copy_m - 1]:
                            break

                    if abs(df['KST_GAUSSIAN'].iloc[m] - df['KST_GAUSSIAN'].iloc[copy_m]) < kst_vertical_gap:
                        pass
                    else:
                        sell_switch = True
                        break

            if df['trade_state'].iloc[m] == 2.5:
                sell_switch = True
                break

            #       손절 시그널       #

            #       매수 시그널의 KST_GAUSSIAN 값보다 작아지면 손절        #
            if m - start_m > kst_horizontal_ticks \
                    and df['KST_GAUSSIAN'].iloc[m] - df['KST_GAUSSIAN'].iloc[buy_signal_m] < -1:
                exit_switch = True
                break

            #       1. 고점 지나고 SHORT 찍을 가망이 없어 보이는 경우       #
            # if df['Fisher_long'].iloc[m] >= high_check_value:
            #     exit_signal = 1
            #
            # elif df['Fisher_long'].iloc[m] >= high_check_value2:
            #     exit_signal = 1.5
            #
            # if exit_signal == 1 and df['Fisher_long'].iloc[m] < high_check_value_exit:
            #     exit_switch = True
            #     break
            #
            # elif exit_signal == 1.5 and df['Fisher_long'].iloc[m] < high_check_value2_exit:
            #     exit_switch = True
            #     break

            #       2. DC LOW LINE DOWN      #
            # if df['DC_LOW'].iloc[m] < df['DC_LOW'].iloc[m - 1]:
            #     dc_descend_cnt += 1
            #     print('start_m :', start_m)
            #     print('DC_GAP at %s :' % m, (df['DC_LOW'].iloc[m] - df['DC_LOW'].iloc[start_m]) / (df['high'].max() - df['low'].min()))
            #     # exit_signal = 2
            #     exit_switch = True
            #     break

            # if exit_signal == 2 and df['Fisher_long'].iloc[m] >= exit_short_signal_value > df['Fisher_long'].iloc[m - 1]:
            #     exit_switch = True
            #     break

            # if dc_descend_cnt == 1:
            #     exit_switch = True
            #     break

            # #       CROSSUNDER AT LONG SIGNAL 재발생시 손절       #
            # if df['Fisher_long'].iloc[m] <= long_signal_value < df['Fisher_long'].iloc[m - 1]:
            #     exit_switch = True
            #     break

            m += 1
            if m > length:
                break

            condition.iloc[m] = '매도 대기'
            bprelay["bprelay"].iloc[m] = bprelay["bprelay"].iloc[m - 1]
            support["support_line"].iloc[m] = support["support_line"].iloc[m - 1]

        if m > length:
            # print(condition.iloc[m - 1])
            condition.iloc[m - 1] = "매도 체결"

            #       매수 체결 후 바로 매도 체결 되어버리는 경우       #
            if m == start_m:
                Profits.iloc[m - 1] = df.iloc[m - 1]['close'] / bprelay["bprelay"].iloc[m - 1] - fee
            else:
                Profits.iloc[m - 1] = df.iloc[m - 1]['close'] / bprelay["bprelay"].iloc[m - 1] - fee

            if float(Profits.iloc[m - 1]) < 1:
                Minus_Profits *= float(Profits.iloc[m - 1])

                try:
                    spanlist.append((start_m, m - 1))
                    spanlist_breakaway.append((start_m, m - 1))

                except Exception as e:
                    pass

            else:
                exit_fluc += df.low.iloc[start_m:m - 1].min() / bprelay["bprelay"].iloc[m - 2]
                exit_count += 1

                try:
                    spanlist.append((start_m, m - 1))
                    spanlist_limit.append((start_m, m - 1))

                except Exception as e:
                    pass
            break

        elif sell_switch or exit_switch:
            condition.iloc[m] = "매도 체결"

            #       매수 체결 후 바로 매도 체결 되어버리는 경우       #
            if m == start_m:
                Profits.iloc[m] = df.iloc[m]['close'] / bprelay["bprelay"].iloc[m] - fee

            else:
                if exit_switch:
                    # Profits.iloc[m] = support['support_line'].iloc[m] / bprelay['bprelay'].iloc[m] - fee
                    Profits.iloc[m] = df.iloc[m]['close'] / bprelay["bprelay"].iloc[m - 1] - fee
                else:
                    Profits.iloc[m] = df.iloc[m]['close'] / bprelay["bprelay"].iloc[m - 1] - fee

            if float(Profits.iloc[m]) < 1:
                print('Minus Profits at %s %.3f' % (m, float(Profits.iloc[m])))
                Minus_Profits *= float(Profits.iloc[m])

                try:
                    spanlist.append((start_m, m))
                    spanlist_breakaway.append((start_m, m))

                except Exception as e:
                    pass

            else:
                exit_fluc += df.low.iloc[start_m:m].min() / bprelay["bprelay"].iloc[m - 1]
                exit_count += 1

                try:
                    spanlist.append((start_m, m))
                    spanlist_limit.append((start_m, m))

                except Exception as e:
                    pass

        # DC LOW 손절시, 탐색 종료 구간 벗어날때까지 m += 1
        if dc_descend_cnt == 2:
            while True:
                m += 1
                if m >= length:
                    break

                if df['Fisher_long'].iloc[m] >= detect_close_value:
                    break
        else:
            m += 1
            if m > length:
                break
            condition.iloc[m] = np.NaN

    df = pd.merge(df, bprelay, how='outer', left_index=True, right_index=True)
    df = pd.merge(df, support, how='outer', left_index=True, right_index=True)
    df = pd.merge(df, condition, how='outer', left_index=True, right_index=True)
    df = pd.merge(df, Profits, how='outer', left_index=True, right_index=True)
    # df = pd.merge(df, price_point, how='outer', left_index=True, right_index=True)
    df = df.reset_index(drop=True)

    if excel == 1:
        # df.to_excel("./BackTest/%s BackTest %s.xlsx" % (Date, Coin))
        df.to_excel("excel_check/%s BackTest %s.xlsx" % (Date, Coin))

    profits = Profits.cumprod()  # 해당 열까지의 누적 곱!

    if exit_count != 0:
        exit_fluc_mean = exit_fluc / exit_count
    else:
        exit_fluc_mean = 1.

    if np.isnan(profits.iloc[-1].item()):
        return 1.0, 1.0, 1.0, 1.0, 1.0, 1.0

    # [-1] 을 사용하려면 데이터가 존재해야 되는데 데이터 전체가 결측치인 경우가 존재한다.
    if len(profits) == 0:
        return 1.0, 1.0, 1.0, 1.0, 1.0, 1.0

    elif get_fig == 1:

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(311)
        ochl = df.iloc[:, :4]
        index = np.arange(len(ochl))
        ochl = np.hstack((np.reshape(index, (-1, 1)), ochl))
        mf.candlestick_ochl(ax, ochl, width=0.5, colorup='r', colordown='b')
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.plot(df.SMMA)
        plt.plot(df.DC_LOW_ENTRY, color='gold')
        plt.plot(df.DC_LOW, color='crimson')
        # plt.legend(loc='upper right', fontsize=5)
        plt.title('KST GAUSSIAN FISHER - %s %.3f %.3f %.3f' % (Coin,
        float(profits.iloc[-1]), float(profits.iloc[-1]) / Minus_Profits, Minus_Profits))
        # for trade_num in range(len(span_list_target)):
        #     plt.axvspan(span_list_target[trade_num][0], span_list_target[trade_num][1], facecolor='red', alpha=0.3)
        for trade_num in range(len(span_list_target_ic)):
            plt.axvspan(span_list_target_ic[trade_num][0], span_list_target_ic[trade_num][1], facecolor='blue', alpha=0.7)
        for trade_num in range(len(spanlist_limit)):
            plt.axvspan(spanlist_limit[trade_num][0], spanlist_limit[trade_num][1], facecolor='c', alpha=0.5)
        for trade_num in range(len(spanlist_breakaway)):
            plt.axvspan(spanlist_breakaway[trade_num][0], spanlist_breakaway[trade_num][1], facecolor='m', alpha=0.5)

        plt.subplot(312)
        plt.plot(df.Fisher_long)
        plt.plot(df['Fisher_GAUSSIAN'], color='limegreen')
        # plt.plot(df.Fisher_short2, color='gold')
        # plt.plot(df.Fisher_short_gaussian)
        plt.axhline(0, color='red', alpha=0.5)
        plt.axhline(long_signal_value)
        plt.axhline(short_signal_value)
        for trade_num in range(len(span_list_target_fisher)):
            plt.axvspan(span_list_target_fisher[trade_num][0], span_list_target_fisher[trade_num][1], facecolor='red', alpha=0.3)
        for trade_num in range(len(spanlist_limit)):
            plt.axvspan(spanlist_limit[trade_num][0], spanlist_limit[trade_num][1], facecolor='c', alpha=0.5)
        for trade_num in range(len(spanlist_breakaway)):
            plt.axvspan(spanlist_breakaway[trade_num][0], spanlist_breakaway[trade_num][1], facecolor='m', alpha=0.5)

        plt.subplot(313)
        plt.plot(df.KST)
        plt.plot(df.KST_GAUSSIAN, color='orange')
        # plt.axhline(high_check_value, color='blue', alpha=0.5)
        plt.axhline(0, color='red', alpha=0.5)
        # plt.axhline(upper_line)
        # plt.axhline(lower_line)
        for trade_num in range(len(span_list_target)):
            plt.axvspan(span_list_target[trade_num][0], span_list_target[trade_num][1], facecolor='red', alpha=0.3)
        for trade_num in range(len(span_list_target_ic)):
            plt.axvspan(span_list_target_ic[trade_num][0], span_list_target_ic[trade_num][1], facecolor='blue',
                        alpha=0.7)
        for trade_num in range(len(spanlist_limit)):
            plt.axvspan(spanlist_limit[trade_num][0], spanlist_limit[trade_num][1], facecolor='c', alpha=0.5)
        for trade_num in range(len(spanlist_breakaway)):
            plt.axvspan(spanlist_breakaway[trade_num][0], spanlist_breakaway[trade_num][1], facecolor='m', alpha=0.5)

        # plot 저장 & 닫기
        plt.show()
        # plt.show(block=False)
        # plt.pause(3)
        # plt.savefig("./Fisher_result/%s_%s_%s_%s/%s_revised_early_entry.png" % (long_signal_value, short_signal_value, long_period, short_period, Coin), dpi=300)
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
    # std = np.std(max_abs_scaler(df[df['trade_state'] == 2.]['MACD_OSC']))
    # print(std)

    return float(profits.iloc[-1]), float(profits.iloc[-1]) / Minus_Profits, Minus_Profits, profits_avg[
        0], Profits.values.min(), exit_fluc_mean


if __name__ == "__main__":

    home_dir = os.path.expanduser('~')
    dir = home_dir + '/OneDrive/CoinBot/ohlcv/'
    ohlcv_list = os.listdir(dir)

    import random

    random.shuffle(ohlcv_list)

    # TopCoin = pybithumb.get_top_coin(10)
    # print(TopCoin)
    # quit()
    TopCoin = ['DAC', 'CHR', 'LRC', 'ZIL', 'VET', 'HYC', 'FX', 'REP', 'STRAT']
    # TopCoin = [ 'ZIL', 'ZRX', 'BZNT', 'HYC', 'CON', 'CHR']
    # TopCoin = ['LRC', 'KNC', 'GXC', 'XEM']
    # TopCoin = [ 'XEM']

    # ohlcv_list = ['2020-03-13 DAC ohlcv.xlsx']

    # for file in ohlcv_list:
    #     Coin = file.split()[1].split('.')[0]
    #     Date = file.split()[0]
    Date = str(datetime.now()).split()[0]
    excel_list = os.listdir('./BestSet/Test_ohlc/')
    total_df = pd.DataFrame(
        columns=['long_signal_value', 'short_signal_value', 'long_period', 'short_period', 'total_profit_avg',
                 'plus_profit_avg', 'minus_profit_avg', 'avg_profit_avg',
                 'min_profit_avg', 'exit_fluc_mean'])

    for long_signal_value in np.arange(-4, -1.5, 0.5):
        for short_signal_value in np.arange(1.5, 4, 0.5):
            for short_period in np.arange(60, 500, 100):

                long_signal_value, short_signal_value = -0., 1.9
                long_period, short_period = 60, 60

                #       Make folder      #
                try:
                    os.mkdir("./Fisher_result/%s_%s_%s_%s/" % (
                    long_signal_value, short_signal_value, long_period, short_period))

                except Exception as e:
                    print(e)
                #
                for Coin in TopCoin:
                    # if Coin.split('-')[1] not in ['03']:
                    #     continue
                    # try:
                    print(Coin,
                          profitage(Coin, long_signal_value, short_signal_value, long_period, short_period, Date=Date,
                                    get_fig=1, excel=1))
                # except Exception as e:
                #     print('Error in profitage :', e)
                quit()

                total_profit = 0
                plus_profit = 0
                minus_profit = 0
                avg_profit = 0
                min_profit = 0
                exit_fluc_mean = 0
                for Coin in excel_list:
                    start = time.time()
                    try:
                        result = profitage(Coin, long_signal_value, short_signal_value, long_period, short_period,
                                           Date=Date, get_fig=1)
                        if result[0] == 1.:  # 거래가 없는 코인을 데이터프레임에 저장할 필요가 없다.
                            continue
                    except Exception as e:
                        continue
                    # quit()
                    total_profit += result[0]
                    plus_profit += result[1]
                    minus_profit += result[2]
                    avg_profit += result[3]
                    min_profit += result[4]
                    exit_fluc_mean += result[5]
                total_profit_avg = total_profit / len(excel_list)
                plus_profit_avg = plus_profit / len(excel_list)
                minus_profit_avg = minus_profit / len(excel_list)
                avg_profit_avg = avg_profit / len(excel_list)
                min_profit_avg = min_profit / len(excel_list)
                exit_fluc_avg = exit_fluc_mean / len(excel_list)
                print(long_signal_value, short_signal_value, long_period, short_period, total_profit_avg,
                      plus_profit_avg, minus_profit_avg, avg_profit_avg,
                      min_profit_avg, exit_fluc_avg, '%.3f second' % (time.time() - start))

                result_df = pd.DataFrame(data=[
                    [long_signal_value, short_signal_value, long_period, short_period, total_profit_avg,
                     plus_profit_avg, minus_profit_avg, avg_profit_avg,
                     min_profit_avg, exit_fluc_avg]],
                    columns=['long_signal_value', 'short_signal_value', 'long_period', 'short_period',
                             'total_profit_avg', 'plus_profit_avg',
                             'minus_profit_avg', 'avg_profit_avg', 'min_profit_avg', 'exit_fluc_avg'])
                total_df = total_df.append(result_df)
                print()

            total_df.to_excel('./BestSet/total_df %s.xlsx' % long_signal_value)
            break
