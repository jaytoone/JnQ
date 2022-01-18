import numpy as np
import pandas as pd
import time
from datetime import datetime
import warnings
import logging

sys_log4 = logging.getLogger()

#        keep fundamental warnings quiet         #
warnings.simplefilter("ignore", category=RuntimeWarning)
np.seterr(invalid="ignore")


def min_max_scaler(x):
    scaled_x = (x - x.min()) / (x.max() - x.min())
    return scaled_x


def max_abs_scaler(x):
    scaled_x = x / abs(x).max()
    return scaled_x


# def transh_hour(realtime, numb):
#     Hour = realtime[numb].split(':')[0]
#     return int(Hour)
#
#
# def transh_min(realtime, numb):
#     Minute = realtime[numb].split(':')[1]
#     return int(Minute)
#
#
# def GetHogaunit(Hoga):
#     if Hoga < 1:
#         Hogaunit = 0.0001
#     elif 1 <= Hoga < 10:
#         Hogaunit = 0.001
#     elif 10 <= Hoga < 100:
#         Hogaunit = 0.01
#     elif 100 <= Hoga < 1000:
#         Hogaunit = 0.1
#     elif 1000 <= Hoga < 5000:
#         Hogaunit = 1
#     elif 5000 <= Hoga < 10000:
#         Hogaunit = 5
#     elif 10000 <= Hoga < 50000:
#         Hogaunit = 10
#     elif 50000 <= Hoga < 100000:
#         Hogaunit = 50
#     elif 100000 <= Hoga < 500000:
#         Hogaunit = 100
#     elif 500000 <= Hoga < 1000000:
#         Hogaunit = 500
#     else:
#         Hogaunit = 1000
#     return Hogaunit
#
#
# def clearance(price):
#     try:
#         Hogaunit = GetHogaunit(price)
#         Htype = type(Hogaunit)
#         if Hogaunit == 0.1:
#             price2 = int(round(price * 10)) / 10.0
#         elif Hogaunit == 0.01:
#             price2 = int(round(price * 100)) / 100.0
#         elif Hogaunit == 0.001:
#             price2 = int(round(price * 1000)) / 1000.0
#         elif Hogaunit == 0.0001:
#             price2 = int(round(price * 10000.0)) / 10000.0
#         else:
#             return int(price) // Hogaunit * Hogaunit
#         return Htype(price2)
#
#     except Exception as e:
#         return np.nan
#
#
# def hclearance(price):
#     try:
#         Hogaunit = GetHogaunit(price)
#         Htype = type(Hogaunit)
#         if Hogaunit == 0.1:
#             price2 = round(price, 1)
#         elif Hogaunit == 0.01:
#             price2 = round(price, 2)
#         elif Hogaunit == 0.001:
#             price2 = round(price, 3)
#         elif Hogaunit == 0.0001:
#             price2 = round(price, 4)
#         else:
#             return round(price)
#         return Htype(price2)
#
#     except Exception as e:
#         return np.nan


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
        
        #       Todo        #
        #        1. pd.ffill() 활용해서 latency 개선 가능할 것
        for col in new_res_idx_df.columns:
            new_res_idx_df[col] = ffill(new_res_idx_df[col].values.reshape(1, -1)).reshape(-1, 1)

        print("np.sum(pd.isna(new_res_idx_df['open']) :", np.sum(pd.isna(new_res_idx_df['open'])))

    return new_res_idx_df


# def to_lower_tf(ltf_df, htf_df, column, output_len=None, show_info=False, backing_i=-2):
# 
#     start_0 = time.time()
# 
#     last_datetime_index = ltf_df.index[-1]
# 
#     last_min = int(str(last_datetime_index).split(':')[1])
#     last_hour = int(str(last_datetime_index).split(':')[0].split(' ')[1])
# 
#     interval = int((to_timestamp(htf_df.index[-1]) - to_timestamp(htf_df.index[-2])) / 60)
# 
#     if show_info:
#         print('last_min :', last_min)
#         print('last_hour :', last_hour)
#         print('interval :', interval)
#         # print('type(interval) :', type(interval))
# 
#     if interval < 60:
#         sliced_index_len = last_min % interval + 1  # +1 하는 이유는, 나머지가 0 인 경우에 대해서도 value 를 채워주어야하기 때문임
#                                                     # 채우지않으면, ltf 에 np.nan 이 생김
#         # ==> 59.9999 붙어서 그런걸로 알고 있음 (31 분이면 실제로 30:59.999999 니까)
#     else:
# 
#         if interval == 240:  # 4h
#             modi_last_hour = last_hour % 4 - 1        #  Todo - 여기 이상해서, to_ltf_v2 만듬
#             if modi_last_hour < 0:
#                 modi_last_hour += 4
# 
#         elif interval == 1440:  # 1d
#             modi_last_hour = last_hour - 9
#             if modi_last_hour < 0:
#                 modi_last_hour += 24
# 
#         else:
#             modi_last_hour = last_hour
# 
#         if show_info:
#             print("modi_last_hour :", modi_last_hour)
# 
#         sliced_index_len = (modi_last_hour * 60 + last_min) % interval + 1  # 이렇게 하면 하루를 기준으로 잡는거내
# 
#         #       Todo        #
#         #        1. 4h, 1d timestamp 시작 시간이 다르기 때문에 각 interval 의 기준으로 재정의해야할 것      #
#         #           a. 1d 는 9:00:00 기준으로 현재까지의 시간을 모두 minute 으로 변경 (sliced_index_len)
#         #               i. last_hour -> last_hour - 9, if < 0 => + 24
#         #           b. 4h 는 1, 5, 9... 시 기준으로 현재까지의 시간을 minute 으로 변경 (sliced_index_len)
#         #               i. last_hour -> last_hour % 4(h) - 1, if < 0 => + 4 
#         #        2. 1h 는 아직
# 
#     #           -2 => 오차없는 이전 데이터를 사용하기 위함           #
#     # backing_i = -2
#     #           -1 => 완성된 future data, => realtime 은 아님      #
#     # backing_i = -1
# 
#     if show_info:
#         print("backing_i :", backing_i)
#         print("sliced_index_len :", sliced_index_len)
# 
#     # value_list = [htf_df[column].iloc[backing_i]] * sliced_index_len
# 
#     #           1. backing_i 만큼 이전 데이터를 htf 로부터 추출함            #
#     value_list = htf_df.iloc[[backing_i], column].values
#     # print("value_list[-1] :", value_list[-1])
# 
#     #           2. 해당 htf data 를 ltf 로 변환함 (sliced_len 만큼 변환함)         #
#     value_list = np.tile(value_list, (sliced_index_len, 1))
# 
#     # print(value_list)
#     # print(type(value_list))
#     # quit()
#     while True:
# 
#         #       org backing_i 의 htf data 는 이미 value_list 에 채웠으니,
#         #       다음 index 부터 htf data slicing 시작
#         backing_i -= 1
# 
#         if show_info:
#             print('backing_i :', backing_i)
# 
#         # temp_list = [htf_df[column].iloc[backing_i]] * interval
# 
#         temp_list = htf_df.iloc[[backing_i], column].values
#         temp_list = np.tile(temp_list, (interval, 1))
#         value_list = np.vstack((temp_list, value_list))
# 
#         # print(value_list)
#         # print(value_list.shape)
#         # quit()
# 
#         if len(value_list) > len(ltf_df):
#             break
# 
#     # value_list = value_list[: -(len(value_list) - len(ltf_df))]
#     # value_list = value_list[(len(value_list) - len(ltf_df)):]
#     value_list = value_list[-len(ltf_df):]
#     # print(value_list.shape)
#     # quit()
#     # print(len(ltf_df), len(value_list))
#     # ltf_df[column] = list(reversed(value_list))
# 
#     # print("elasped time in to_lower_tf :", time.time() - start_0)
# 
#     return value_list


def to_lower_tf_v2(ltf_df, htf_df, column, backing_i=1, show_info=False):
    #       Todo        #
    #        1. 현재 downsampled df 만 허용, direct_df 사용시 issue 발생 가능할 것
    assert type(column[0]) == int, "column value should be integer"
    # if not replace_last_idx:
    #     assert datetime.timestamp(htf_df.index[-1]) >= datetime.timestamp(
    #         ltf_df.index[-1]), "htf_lastidx should >= ltf_lastidx"  # data sync confirmation
    #   htf idx ts 가 ltf 보다 작을 경우, intersec_idx 를 구할 수 없음

    #   downsampled htf_df 만 허용 - 통일성
    #   non_resampled df 사용시 "cannot reindex a non-unique index with a method or limit" 라는 error_msg 확인

    cols = htf_df.columns[column]  # to_lower_tf_v1 의 int col 반영

    ltf_itv = pd.infer_freq(ltf_df.index)
    assert ltf_itv == 'T', "currently only -> 'T' allowed.."

    if show_info:
        print("backing_i :", backing_i)

    #        1. htf last_ts_idx 의 second 가 :00 format (all rows same format)이 아닌 경우, bfill() 시 NaN 발생
    #           --> 상관 없음, 다른 윈인임
    #        2. single positional indexer is out-of-bounds => 대게 len(data) 부족 문제임 (row 증분)
    # iloc 뒤에 붙여야함, 아니면 timeidx 정상 출력 안됨

    renamed_last_row = htf_df.rename(index={htf_df.index[-1]: ltf_df.index[-1]}, inplace=False).iloc[[-1]]
    if htf_df.index[-1] != renamed_last_row.index[-1]:  # cannot reindex a non-unique index with a method or limit 방지
        htf_df = htf_df.append(renamed_last_row)

    # print(htf_df.tail())

    # downsampled htf 의 freq_offset 기준으로 앞에서 뒤로 채우는게 맞음
    # --> hh:mm:00 format 을 사용해서 그럼 (59:999 면 bfill() 이였을 것)
    resampled_df = htf_df[cols].shift(backing_i).resample(ltf_itv).ffill()
    # print(resampled_df.tail())
    # print()

    #        2. htf 가 downsampled 된 df 를 default 로 사용하기에, slicing 불필요해짐
    #        3. Shape of passed values is (799, 3), indices imply (3000, 3) -> len(ltf_df) > len(resampled_df) 란 의미
    #        4. solution => reindexing for inner join   #
    #         4_1. len(resampled_df) > len(ltf_df) 경우 slicing 진행
    #           4_1_1. to_htf 를 upsampling 하면서 발생하는 length 오차 교정 위함임 -> 애초에 input_rows 차이 개념이 아님
    #           4_1_2. sliced_ltf_df & sliced_htf_df 의 경우,
    if len(resampled_df) > len(ltf_df):
        resampled_df = resampled_df.iloc[-len(ltf_df):]
    # print("len(ltf_df) :", len(ltf_df))
    # print("len(resampled_df) :", len(resampled_df))
    resampled_df.index = ltf_df.index[-len(resampled_df):]
    # assert len(ltf_df) <= len(resampled_df), "for join method, assert len(ltf_df) <= len(resampled_df)"

    #       check last row's validity       #
    assert np.sum(
        pd.isnull(resampled_df.iloc[-1].values)) == 0, "np.nan value occured, more {} rows might be reguired".format(
        cols)

    # if datetime.timestamp(htf_df.index[-1]) < datetime.timestamp(ltf_df.index[-1]):
    #     # resampled_df.rename(index={resampled_df.index[-1]: ltf_df.index[-1]}, inplace=True)
    #     print(resampled_df.tail())
    #     print("-----------")

    #        1. 필요한 len 계산해서 pre_proc. 진행 --> open_idx 를 동일하게 맞춰놓았고, shift 적용 상태이기 때문에 불필요함

    #        1. ltf_df 의 마지막 timeidx 와 sync. 맞춰주어야함
    #           a. itv '1T' 가 아닌경우, 교집합 timeidx 가 존재하지 않을 수 있음
    # ltf_lastidx = ltf_df.tail(1).resample('1T').asfreq().index
    # intersec_idx_arr = np.argwhere(resampled_df.index == ltf_lastidx.item())
    # intersec_idx = intersec_idx_arr.item()
    # print("intersec_idx :", intersec_idx)
    # print("len(resampled_df) :", len(resampled_df))
    #
    # assert len(intersec_idx_arr) >= 1, "len(intersec_idx_arr) is zero"
    #
    # sliced_resampled_df = resampled_df[:intersec_idx + 1]
    # return sliced_resampled_df.values[-len(ltf_df):]

    # return resampled_df.values[-len(ltf_df):]
    return resampled_df


def to_htf(df, itv_, offset):

    h_res_df = df.resample(itv_, offset=offset).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    })

    return h_res_df


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


def calc_rows_and_days(itv_list, row_list, min_days=1440):

    itv_arr = np.array([to_itvnum(itv_) for itv_ in itv_list])
    row_arr = np.array(row_list)

    max_rows = np.max(itv_arr * row_arr)
    days = int(max_rows / min_days) + 1

    return max_rows, days


def calc_tp_out_fee(config_):

    if config_.ep_set.entry_type == 'MARKET':
        if config_.tp_set.tp_type != 'MARKET':  # Todo : 실제로, tp_fee 가 아닌 spread const. 를 위한 spread_fee1 임 (추후 수정 권고)
            tp_fee = config_.trader_set.market_fee + config_.trader_set.limit_fee
        else:
            tp_fee = config_.trader_set.market_fee + config_.trader_set.market_fee
        out_fee = config_.trader_set.market_fee + config_.trader_set.market_fee
    else:
        if config_.tp_set.tp_type != 'MARKET':
            tp_fee = config_.trader_set.limit_fee + config_.trader_set.limit_fee
        else:
            tp_fee = config_.trader_set.limit_fee + config_.trader_set.market_fee
        out_fee = config_.trader_set.limit_fee + config_.trader_set.market_fee

    return tp_fee, out_fee


def calc_train_days(interval, use_rows):    # --> for ai

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