# import pybithumb
import numpy as np
import pandas as pd
from datetime import datetime
import os
from scipy import stats
# from asq.initiators import query
from sklearn.preprocessing import MaxAbsScaler
import time


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
# def transh_fluc(Coin):
#     try:
#         TransH = pybithumb.transaction_history(Coin)
#         TransH = TransH['data']
#         Realtime = query(TransH).select(lambda item: item['transaction_date'].split(' ')[1]).to_list()
#
#         # 거래 활발한지 검사
#         if (transh_hour(Realtime, -1) - transh_hour(Realtime, 0)) < 0:
#             if 60 + (transh_min(Realtime, -1) - transh_min(Realtime, 0)) > 30:
#                 return 0, 0
#         elif 60 * (transh_hour(Realtime, -1) - transh_hour(Realtime, 0)) + (
#                 transh_min(Realtime, -1) - transh_min(Realtime, 0)) > 30:
#             return 0, 0
#
#         # 1분 동안의 거래 이력을 조사하는 루프, 0 - 59 와 같은 음수처리를 해주어야한다.
#         i = 1
#         while True:
#             i += 1
#             if i > len(Realtime):
#                 m = i
#                 break
#             # 음수 처리
#             if (transh_min(Realtime, -1) - transh_min(Realtime, -i)) < 0:
#                 if (60 + transh_min(Realtime, -1) - transh_min(Realtime, -i)) > 1:
#                     m = i - 1
#                     break
#             elif (transh_min(Realtime, -1) - transh_min(Realtime, -i)) > 1:
#                 m = i - 1
#                 break
#
#         # Realtime = query(TransH[-i:]).select(lambda item: item['transaction_date'].split(' ')[1]).to_list()
#         Price = list(map(float, query(TransH[-m:]).select(lambda item: item['price']).to_list()))
#
#         # print(Realtime)
#         # print(Price)
#         fluc = max(Price) / min(Price)
#         if TransH[-1]['type'] == 'ask':
#             fluc = -fluc
#         return fluc, min(Price)
#
#     except Exception as e:
#         print("Error in transh_fluc :", e)
#         return 0, 0
#
#
# def realtime_transaction(Coin, display=5):
#     Transaction_history = pybithumb.transaction_history(Coin)
#     Realtime = query(Transaction_history['data'][-display:]).select(
#         lambda item: item['transaction_date'].split(' ')[1]).to_list()
#     Realtime_Price = list(
#         map(float, query(Transaction_history['data'][-display:]).select(lambda item: item['price']).to_list()))
#     Realtime_Volume = list(
#         map(float, query(Transaction_history['data'][-display:]).select(lambda item: item['units_traded']).to_list()))
#
#     print("##### 실시간 체결 #####")
#     print("{:^10} {:^10} {:^20}".format('시간', '가격', '거래량'))
#     for i in reversed(range(display)):
#         print("%-10s %10.2f %20.3f" % (Realtime[i], Realtime_Price[i], Realtime_Volume[i]))
#     return
#
#
# def realtime_hogachart(Coin, display=3):
#     Hogachart = pybithumb.get_orderbook(Coin)
#
#     print("##### 실시간 호가창 #####")
#     print("{:^10} {:^20}".format('가격', '거래량'))
#     for i in reversed(range(display)):
#         print("%10.2f %20.3f" % (Hogachart['asks'][i]['price'], Hogachart['asks'][i]['quantity']))
#     print('-' * 30)
#     for j in range(display):
#         print("%10.2f %20.3f" % (Hogachart['bids'][j]['price'], Hogachart['bids'][j]['quantity']))
#
#
# def realtime_volume(Coin):
#     Transaction_history = pybithumb.transaction_history(Coin)
#     Realtime_Volume = query(Transaction_history['data']).where(lambda item: item['type'] == 'bid').select(
#         lambda item: item['units_traded']).to_list()
#     Realtime_Volume = sum(list(map(float, Realtime_Volume)))
#     return Realtime_Volume
#
#
# def realtime_volume_ratio(Coin):
#     Transaction_history = pybithumb.transaction_history(Coin)
#     Realtime_bid = query(Transaction_history['data']).where(lambda item: item['type'] == 'bid').select(
#         lambda item: item['units_traded']).to_list()
#     Realtime_ask = query(Transaction_history['data']).where(lambda item: item['type'] == 'ask').select(
#         lambda item: item['units_traded']).to_list()
#     Realtime_bid = sum(list(map(float, Realtime_bid)))
#     Realtime_ask = sum(list(map(float, Realtime_ask)))
#     Realtime_Volume_Ratio = Realtime_bid / Realtime_ask
#     return Realtime_Volume_Ratio
#
#
# def topcoinlist(Date):
#     temp = []
#     dir = 'C:/Users/장재원/OneDrive/Hacking/CoinBot/ohlcv/'
#     ohlcv_list = os.listdir(dir)
#
#     for file in ohlcv_list:
#         if file.find(Date) is not -1:  # 해당 파일이면 temp[i] 에 넣겠다.
#             filename = os.path.splitext(file)
#             temp.append(filename[0].split(" ")[1])
#     return temp
#
#
# def get_ma_min(Coin):
#     df = pybithumb.get_ohlcv(Coin, "KRW", 'minute1')
#
#     df['MA20'] = df['close'].rolling(20).mean()
#
#     DatetimeIndex = df.axes[0]
#     period = 20
#     if inthour(DatetimeIndex[-1]) - inthour(DatetimeIndex[-period]) < 0:
#         if 60 + (intmin(DatetimeIndex[-1]) - intmin(DatetimeIndex[-period])) > 30:
#             return 0
#     elif 60 * (inthour(DatetimeIndex[-1]) - inthour(DatetimeIndex[-period])) + intmin(DatetimeIndex[-1]) - intmin(
#             DatetimeIndex[-period]) > 30:
#         return 0
#     slope, intercept, r_value, p_value, stderr = stats.linregress([i for i in range(period)], df.MA20[-period:])
#
#     return slope
#
#
# def get_ma20_min(Coin):
#     df = pybithumb.get_ohlcv(Coin, "KRW", 'minute1')
#
#     maxAbsScaler = MaxAbsScaler()
#
#     df['MA20'] = df['close'].rolling(20).mean()
#     MA_array = np.array(df['MA20']).reshape(len(df.MA20), 1)
#     maxAbsScaler.fit(MA_array)
#     scaled_MA = maxAbsScaler.transform(MA_array)
#
#     period = 5
#     slope, intercept, r_value, p_value, stderr = stats.linregress([i for i in range(period)], scaled_MA[-period:])
#
#     return slope
#
#
# def get_obv_min(Coin):
#     df = pybithumb.get_ohlcv(Coin, "KRW", "minute1")
#
#     obv = [0] * len(df.index)
#     for m in range(1, len(df.index)):
#         if df['close'].iloc[m] > df['close'].iloc[m - 1]:
#             obv[m] = obv[m - 1] + df['volume'].iloc[m]
#         elif df['close'].iloc[m] == df['close'].iloc[m - 1]:
#             obv[m] = obv[m - 1]
#         else:
#             obv[m] = obv[m - 1] - df['volume'].iloc[m]
#     df['OBV'] = obv
#
#     # 24시간의 obv를 잘라서 box 높이를 만들어주어야한다.
#     DatetimeIndex = df.axes[0]
#     boxheight = [0] * len(df.index)
#     whaleincome = [0] * len(df.index)
#     for m in range(len(df.index)):
#         # 24시간 시작행 찾기, obv 데이터가 없으면 stop
#         n = m
#         while True:
#             n -= 1
#             if n < 0:
#                 n = 0
#                 break
#             if inthour(DatetimeIndex[m]) - inthour(DatetimeIndex[n]) < 0:
#                 if 60 - (intmin(DatetimeIndex[m]) - intmin(DatetimeIndex[n])) >= 60 * 24:
#                     break
#             elif 60 * (inthour(DatetimeIndex[m]) - inthour(DatetimeIndex[n])) + intmin(DatetimeIndex[m]) - intmin(
#                     DatetimeIndex[n]) >= 60 * 24:
#                 break
#         obv_trim = obv[n:m]
#         if len(obv_trim) != 0:
#             boxheight[m] = max(obv_trim) - min(obv_trim)
#             if obv[m] - min(obv_trim) != 0:
#                 whaleincome[m] = abs(max(obv_trim) - obv[m]) / abs(obv[m] - min(obv_trim))
#
#     df['BoxHeight'] = boxheight
#     df['Whaleincome'] = whaleincome
#
#     period = 0
#     while True:
#         period += 1
#         if period >= len(DatetimeIndex):
#             break
#         if inthour(DatetimeIndex[-1]) - inthour(DatetimeIndex[-period]) < 0:
#             if 60 + (intmin(DatetimeIndex[-1]) - intmin(DatetimeIndex[-period])) >= 10:
#                 break
#         elif 60 * (inthour(DatetimeIndex[-1]) - inthour(DatetimeIndex[-period])) + intmin(DatetimeIndex[-1]) - intmin(
#                 DatetimeIndex[-period]) >= 10:
#             break
#
#     slope, intercept, r_value, p_value, stderr = stats.linregress([i for i in range(period)], df.OBV[-period:])
#     if period < 3:
#         df['Whaleincome'].iloc[-1], slope = 0, 0
#     else:
#         slope = slope / df['BoxHeight'].iloc[-1]
#
#     return df['Whaleincome'].iloc[-1], slope


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
            price2 = int(round(price * 10)) / 10.0
        elif Hogaunit == 0.01:
            price2 = int(round(price * 100)) / 100.0
        elif Hogaunit == 0.001:
            price2 = int(round(price * 1000)) / 1000.0
        elif Hogaunit == 0.0001:
            price2 = int(round(price * 10000.0)) / 10000.0
        else:
            return int(price) // Hogaunit * Hogaunit
        return Htype(price2)

    except Exception as e:
        return np.nan


def hclearance(price):
    try:
        Hogaunit = GetHogaunit(price)
        Htype = type(Hogaunit)
        if Hogaunit == 0.1:
            price2 = round(price, 1)
        elif Hogaunit == 0.01:
            price2 = round(price, 2)
        elif Hogaunit == 0.001:
            price2 = round(price, 3)
        elif Hogaunit == 0.0001:
            price2 = round(price, 4)
        else:
            return round(price)
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


def to_timestamp(datetime_):
    # timestamp = time.mktime(datetime.strptime(str(datetime_), '%Y-%m-%d %H:%M:%S').timetuple())
    timestamp = time.mktime(datetime_.timetuple())
    # print(type(timestamp))
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


def convert_df(df, second_df, second_interval_min=3):
    last_datetime_indexing(second_df, second_interval_min)

    #           INTERVAL MATCHING           #
    index_df = timestamped_index(df)
    index_second_df = timestamped_index(second_df)
    start_j = 1
    second_df_list = list()
    for i in range(len(index_df)):
        for j in range(start_j, len(index_second_df)):

            if j == len(index_second_df) - 1:
                if index_second_df[j] <= index_df[i]:
                    second_df_list.append(second_df.values[j - 1])
                    # df_copy.values[i] = second_df.values[j]
                    # df_copy.values[[i], -2:] = second_df.values[[j], -2:]

            if index_second_df[j - 1] <= index_df[i] < index_second_df[j]:
                second_df_list.append(second_df.values[j - 2])
                # df_copy.values[i] = second_df.values[j - 2]
                # df_copy.values[[i], -2:] = second_df.values[[j - 1], -2:]
                start_j = j

    length_gap = len(df.index) - len(second_df_list)
    # print('length_gap :', length_gap)
    if length_gap != 0:
        for i in range(length_gap):
            second_df_list.insert(0, np.full((len(second_df.columns), ), np.NaN))
        # print(second_df_list[:5])

    # df2 = df_copy
    df2 = pd.DataFrame(index=df.index, data=second_df_list, columns=second_df.columns)

    return df2


def to_lower_tf(ltf_df, htf_df, column):

    last_datetime_index = ltf_df.index[-1]

    last_min = int(str(last_datetime_index).split(':')[1])
    last_hour = int(str(last_datetime_index).split(':')[0].split(' ')[1])

    interval = int((to_timestamp(htf_df.index[-1]) - to_timestamp(htf_df.index[-2])) / 60)
    # print('interval :', interval)
    # print('type(interval) :', type(interval))

    if interval < 60:
        sliced_index_len = last_min % interval + 1
    else:
        sliced_index_len = (last_hour * 60 + last_min) % interval + 1

    # value_list = list()
    #           -2 => 완성된 이전 데이터를 사용하기 위함           #
    backing_i = -2
    # value_list = [htf_df[column].iloc[backing_i]] * sliced_index_len
    value_list = htf_df.iloc[[backing_i], column].values
    value_list = np.tile(value_list, (sliced_index_len, 1))
    # print(value_list)
    # print(type(value_list))
    # quit()
    while True:
        backing_i -= 1
        # print('backing_i :', backing_i)
        # temp_list = [htf_df[column].iloc[backing_i]] * interval
        temp_list = htf_df.iloc[[backing_i], column].values
        temp_list = np.tile(temp_list, (interval, 1))
        value_list = np.vstack((temp_list, value_list))
        # print(value_list)
        # print(value_list.shape)
        # quit()

        if len(value_list) > len(ltf_df):
            break

    # value_list = value_list[: -(len(value_list) - len(ltf_df))]
    value_list = value_list[(len(value_list) - len(ltf_df)):]
    # print(value_list.shape)
    # quit()
    # print(len(ltf_df), len(value_list))
    # ltf_df[column] = list(reversed(value_list))

    return value_list


def to_higher_candlestick(first_df, interval):

    assert interval < 60, 'Current fuction is only for below 1h interval'
    first_df_copy = first_df.copy()

    for i in range(len(first_df)):
        roll_i = intmin(first_df.index[i]) % interval + 1
        first_df_copy['open'].iloc[i] = first_df['open'].iloc[i - (roll_i - 1)]
        first_df_copy['high'].iloc[i] = first_df['high'].rolling(roll_i).max().iloc[i]
        first_df_copy['low'].iloc[i] = first_df['low'].rolling(roll_i).min().iloc[i]

    # print(pd.concat([first_df, first_df_copy], axis=1))

    return first_df_copy


def get_precision_by_price(price):

    try:
        precision = len(str(price).split('.')[1])

    except Exception as e:
        precision = 0

    return precision


def calc_win_ratio(win_cnt, lose_cnt):
    if win_cnt + lose_cnt == 0:
        win_ratio = 0
    else:
        win_ratio = win_cnt / (win_cnt + lose_cnt) * 100

    return win_ratio