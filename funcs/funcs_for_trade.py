# import pybithumb
# from datetime import datetime
# import os
# from scipy import stats
# from asq.initiators import query
# from sklearn.preprocessing import MaxAbsScaler
import numpy as np
import pandas as pd
import time
# from funcs.funcs_indicator import ffill
from datetime import datetime


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


def sharpe_ratio(pr, risk_free_rate=0.0, multiply_frq=False):
    pr_pct = pr - 1

    mean_pr_ = np.mean(pr_pct)
    s = np.std(pr_pct)

    sr_ = (mean_pr_ - risk_free_rate) / s

    if multiply_frq:
        sr_ = len(pr_pct) ** (1 / 2) * sr_

    return sr_


def ffill(arr):  # 이전 row 의 데이터로 현 missing_value 를 채움
    arr = np.array(arr)
    # mask = np.isnan(arr)
    mask = pd.isna(arr)
    # print(mask.shape)
    # print(type(arr))
    idx = np.where(~mask, np.arange(mask.shape[1]), 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    # print("idx :", idx)
    out = arr[np.arange(idx.shape[0])[:,None], idx]
    return out


def bfill(arr):
    arr = np.array(arr)
    # mask = np.isnan(arr)
    mask = pd.isna(arr)

    idx = np.where(~mask, np.arange(mask.shape[1]), mask.shape[1] - 1)
    idx = np.minimum.accumulate(idx[:, ::-1], axis=1)[:, ::-1]
    out = arr[np.arange(idx.shape[0])[:,None], idx]
    return out


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


def itv_bn2ub(itv):
    if itv == "1m":
        return "minute1"
    elif itv == "3m":
        return "minute3"
    elif itv == "5m":
        return "minute5"
    elif itv == "15m":
        return "minute15"
    elif itv == "30m":
        return "minute30"
    elif itv == "1h":
        return "minute60"
    elif itv == "4h":
        return "minute240"


def limit_by_itv(interval):
    if interval == "1m":
        limit = 1500
    elif interval == "3m":
        limit = 500
    elif interval == "5m":
        limit = 300
    else:
        limit = 100

    return limit


def consecutive_df(res_df, itv_num):
    np_idx_ts = np.array(list(map(lambda x: datetime.timestamp(x), res_df.index)))
    ts_start = np_idx_ts[0]
    ts_end = np_idx_ts[-1]

    # print(ts_start)
    # print(ts_end)

    ts_gap = 60.0 * itv_num
    new_ts_line = [new_ts for new_ts in np.arange(ts_start, ts_end + ts_gap, ts_gap)]
    # print(new_ts_line[:10])

    new_res_idx = np.array(list(map(lambda x: pd.to_datetime(datetime.fromtimestamp(x)), new_ts_line)))
    # print(new_res_idx[:10])

    new_res_idx_df = pd.DataFrame(index=new_res_idx, columns=res_df.columns)
    # print(new_res_idx_df.tail())

    new_res_idx_df.loc[res_df.index, :] = res_df

    verify = np.sum(pd.isna(new_res_idx_df['open']))
    print("np.sum(pd.isna(new_res_idx_df['open']) :", verify)

    if verify:

        for col in new_res_idx_df.columns:
            new_res_idx_df[col] = ffill(new_res_idx_df[col].values.reshape(1, -1)).reshape(-1, 1)

        print("np.sum(pd.isna(new_res_idx_df['open']) :", np.sum(pd.isna(new_res_idx_df['open'])))

    return new_res_idx_df


def to_lower_tf(ltf_df, htf_df, column, output_len=None, show_info=False, backing_i=-2):

    start_0 = time.time()

    last_datetime_index = ltf_df.index[-1]

    last_min = int(str(last_datetime_index).split(':')[1])
    last_hour = int(str(last_datetime_index).split(':')[0].split(' ')[1])

    interval = int((to_timestamp(htf_df.index[-1]) - to_timestamp(htf_df.index[-2])) / 60)

    if show_info:
        print('last_min :', last_min)
        print('last_hour :', last_hour)
        print('interval :', interval)
        # print('type(interval) :', type(interval))

    if interval < 60:
        sliced_index_len = last_min % interval + 1  # +1 하는 이유는, 나머지가 0 인 경우에 대해서도 value 를 채워주어야하기 때문임
                                                    # 채우지않으면, ltf 에 np.nan 이 생김
        # ==> 59.9999 붙어서 그런걸로 알고 있음 (31 분이면 실제로 30:59.999999 니까)
    else:

        if interval == 240:  # 4h
            modi_last_hour = last_hour % 4 - 1
            if modi_last_hour < 0:
                modi_last_hour += 4

        elif interval == 1440:  # 1d
            modi_last_hour = last_hour - 9
            if modi_last_hour < 0:
                modi_last_hour += 24

        else:
            modi_last_hour = last_hour

        if show_info:
            print("modi_last_hour :", modi_last_hour)

        sliced_index_len = (modi_last_hour * 60 + last_min) % interval + 1  # 이렇게 하면 하루를 기준으로 잡는거내

        #       Todo        #
        #        1. 4h, 1d timestamp 시작 시간이 다르기 때문에 각 interval 의 기준으로 재정의해야할 것      #
        #           a. 1d 는 9:00:00 기준으로 현재까지의 시간을 모두 minute 으로 변경 (sliced_index_len)
        #               i. last_hour -> last_hour - 9, if < 0 => + 24
        #           b. 4h 는 1, 5, 9... 시 기준으로 현재까지의 시간을 minute 으로 변경 (sliced_index_len)
        #               i. last_hour -> last_hour % 4(h) - 1, if < 0 => + 4 
        #        2. 1h 는 아직

    #           -2 => 오차없는 이전 데이터를 사용하기 위함           #
    # backing_i = -2
    #           -1 => 완성된 future data, => realtime 은 아님      #
    # backing_i = -1

    if show_info:
        print("backing_i :", backing_i)
        print("sliced_index_len :", sliced_index_len)

    # value_list = [htf_df[column].iloc[backing_i]] * sliced_index_len

    #           1. backing_i 만큼 이전 데이터를 htf 로부터 추출함            #
    value_list = htf_df.iloc[[backing_i], column].values
    # print("value_list[-1] :", value_list[-1])

    #           2. 해당 htf data 를 ltf 로 변환함 (sliced_len 만큼 변환함)         #
    value_list = np.tile(value_list, (sliced_index_len, 1))

    # print(value_list)
    # print(type(value_list))
    # quit()
    while True:

        #       org backing_i 의 htf data 는 이미 value_list 에 채웠으니,
        #       다음 index 부터 htf data slicing 시작
        backing_i -= 1

        if show_info:
            print('backing_i :', backing_i)

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
    # value_list = value_list[(len(value_list) - len(ltf_df)):]
    value_list = value_list[-len(ltf_df):]
    # print(value_list.shape)
    # quit()
    # print(len(ltf_df), len(value_list))
    # ltf_df[column] = list(reversed(value_list))

    # print("elasped time in to_lower_tf :", time.time() - start_0)

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


def to_higher_candlestick_v2(first_df, interval):

    assert interval < 60, 'Current fuction is only for below 1h interval'
    first_df_copy = first_df.copy()

    for i in range(len(first_df)):
        roll_i = intmin(first_df.index[i]) % interval + 1
        first_df_copy['open'].iloc[i] = first_df['open'].iloc[i - (roll_i - 1)]
        first_df_copy['high'].iloc[i] = first_df['high'].rolling(roll_i).max().iloc[i]
        first_df_copy['low'].iloc[i] = first_df['low'].rolling(roll_i).min().iloc[i]

    # print(pd.concat([first_df, first_df_copy], axis=1))

    #       Todo        #
    #       1. interval 기준으로 align 해아함      #
    minute_index = np.array(list(map(lambda x: intmin(x), first_df_copy.index)))
    # print("minute_index :", minute_index)
    # quit()
    interval_index = np.argwhere(minute_index % interval == interval - 1).reshape(-1, )
    # print("interval_index :", interval_index)

    htf_df = first_df_copy.iloc[interval_index]
    htf_df_copy = htf_df.copy()

    #       2. htf_df 의 마지막 index 는, first_df_copy 의 마지막 데이터로 덮어씌운다   #
    #       2-1. last index 가 interval 시작인경우, append
    if minute_index[-1] % interval != interval - 1:
        htf_df_copy = htf_df_copy.append(first_df_copy.iloc[-1])

    #       2-2. interval 시작이 아닌 경우, 덮어씌우기
    # else:
    #     htf_df_copy.iloc[-1] = first_df_copy.iloc[-1]

    # print("first_df_copy.tail(10) :", first_df_copy.tail(10))
    # quit()

    return htf_df_copy


def to_itvnum(interval):

    if interval == '1m':
        int_minute = 1
    elif interval == '3m':
        int_minute = 3
    elif interval == '5m':
        int_minute = 5
    elif interval == '15m':
        int_minute = 15
    elif interval == '30m':
        int_minute = 30
    elif interval == '1h':
        int_minute = 60
    elif interval == '4h':
        int_minute = 240
    else:
        print("unacceptable interval :", interval)
        return None

    return int_minute


def calc_train_days(interval, use_rows):

    if interval == '15m':
        data_amt = 96
    elif interval == '30m':
        data_amt = 48
    elif interval == '1h':
        data_amt = 24
    elif interval == '4h':
        data_amt = 6
    else:
        print("unacceptable interval :", interval)
        return None

    days = int(use_rows / data_amt) + 1

    return days


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