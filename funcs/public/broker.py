import numpy as np
import pandas as pd
import time
from datetime import datetime
import warnings
import logging
import pickle

sys_log = logging.getLogger()

#        keep fundamental warnings quiet         #
warnings.simplefilter("ignore", category=RuntimeWarning)
np.seterr(invalid="ignore")

def save_pkl(input_list, f_path):
  with open(f_path, 'wb') as pickle_file:
    pickle.dump(input_list, pickle_file)
    print(f_path, "saved !")

def load_pkl(f_path):
  with open(f_path, 'rb') as pickle_load:
      return pickle.load(pickle_load)

def dup_col_value(x, invalid_value, dup_x):
  return dup_x if x == invalid_value else x


def chng_bin_col(np_bin, inval_list, input_arr, col_i):
  np_bin[:, col_i] = [dup_col_value(x, inval_list[col_i], input_arr[col_i]) for x in np_bin[:, col_i]]
  # short_bin[:, col_i] = [dup_col_value(x, -11, input_arr[col_i]) for x in short_bin[:, col_i]]
  # short_bin
  return


def preproc_bin(np_bin, inval_list, input_arr):
  _ = [chng_bin_col(np_bin, inval_list, input_arr, col_i) for col_i in range(np_bin.shape[-1])]


def save_bin(bin_path_, data):  # funcs_trader 에 numpy 존재해서, 이곳에 배치해놓음, 그리고 funcs_duration 은 앞으로 deprecate 될 .py 임
  with open(bin_path_, 'wb') as f:
    np.save(f, data)
    print(bin_path_, 'saved !')


def load_bin(bin_path_):
  with open(bin_path_, 'rb') as f:
    return np.load(f, allow_pickle=True)


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


def intmin_np(date):
    date = str(date)
    date = date.split('T')
    min = int(date[1].split(':')[1])  # 분
    return min


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


def itv_binance_to_upbit(itv):
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


def itv_to_number(interval):

    if interval in ['1m', 'T']:
        int_minute = 1
    elif interval in ['3m', '3T']:
        int_minute = 3
    elif interval in ['5m', '5T']:
        int_minute = 5
    elif interval in ['15m', '15T']:
        int_minute = 15
    elif interval in ['30m', '30T']:
        int_minute = 30
    elif interval in ['1h', 'H']:
        int_minute = 60
    elif interval in ['4h', '4H']:
        int_minute = 240
    else:
        print("unacceptable interval :", interval)
        return None

    return int_minute


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


def to_lower_tf_v4(ltf_df, htf_df, cols, backing_i=1, show_info=False, ltf_itv=None):
    """
    v3 -> v4 : downsampled_df & ltf_df timeindex 의 format sync. 진행.
    """

    if ltf_itv is None:
        ltf_itv = pd.infer_freq(ltf_df.index)
    assert ltf_itv == 'T', "currently only -> 'T' allowed.."
    # assert type(column[0]) in [int, np.int64], "column value should be integer"

    # cols = htf_df.columns[column]  # to_lower_tf_v1 의 int col 반영

    if show_info:
        print("backing_i :", backing_i)

    # 1. 15T 기준 : 1:30 -> ~1:30 까지 downsample 됨 / T 는 1:43 까지 있는데.
    # 2. 따라서 htf col 형식을 유지하기 위해서 rename 을 사용하고,
    # 3. 43 까지 downsampling 을 위해서, ltf_df 마지막 행을 추가해줌
    renamed_last_row = htf_df.rename(index={htf_df.index[-1]: ltf_df.index[-1]}, inplace=False).iloc[[-1]]
    if htf_df.index[-1] != renamed_last_row.index[-1]:  # cannot reindex a non-unique index with a method or limit 방지
        htf_df = htf_df.append(renamed_last_row)

    downsampled_df = htf_df[cols].shift(backing_i).resample(ltf_itv).ffill()

    downsampled_time_format = str(downsampled_df.index[-1]).split(":")[-1]
    ltf_time_format = str(ltf_df.index[-1]).split(":")[-1]

    if downsampled_time_format != ltf_time_format:
        downsampled_df.index = pd.to_datetime(list(map(lambda x: "{}:{}:{}".format(*str(x).split(":")[:-1], ltf_time_format), downsampled_df.index)))

        intersec_index = sorted(list(set(downsampled_df.index) & set(ltf_df.index)))
        downsampled_df = downsampled_df.loc[intersec_index]

        # ------ check last row's validity ------ #
        assert np.sum(~pd.isnull(downsampled_df.iloc[-1].values)) > 0, "assert np.sum(~pd.isnull(downsampled_df.iloc[-1].values)) > 0"

    return downsampled_df


def to_lower_tf_v3(ltf_df, htf_df, cols, backing_i=1, show_info=False, ltf_itv=None):
    if ltf_itv is None:
        ltf_itv = pd.infer_freq(ltf_df.index)
    assert ltf_itv == 'T', "currently only -> 'T' allowed.."
    # assert type(column[0]) in [int, np.int64], "column value should be integer"

    # cols = htf_df.columns[column]  # to_lower_tf_v1 의 int col 반영

    if show_info:
        print("backing_i :", backing_i)

    # 1. 15T 기준 : 1:30 -> ~1:30 까지 downsample 됨 / T 는 1:43 까지 있는데.
    # 2. 따라서 htf col 형식을 유지하기 위해서 rename 을 사용하고,
    # 3. 43 까지 downsampling 을 위해서, ltf_df 마지막 행을 추가해줌
    renamed_last_index = htf_df.rename(index={htf_df.index[-1]: ltf_df.index[-1]}, inplace=False).iloc[[-1]]
    if htf_df.index[-1] != renamed_last_index.index[-1]:  # cannot reindex a non-unique index with a method or limit 방지
        htf_df = htf_df.append(renamed_last_index)

    downsampled_df = htf_df[cols].shift(backing_i).resample(ltf_itv).ffill()

    # Todo. 정확한 timestamp sync. 를 위해서 len 이 아닌 timeindex 로 접근하는게 맞음.
    intersec_index = sorted(list(set(downsampled_df.index) & set(ltf_df.index)))
    downsampled_df = downsampled_df.loc[intersec_index]

    """
    new version (upper) -> kiwoom data 사용과정에서 범용성 위해 개선함.
    old version (lower)
    """
    # if len(downsampled_df) > len(ltf_df):
    #     # downsampled_df = downsampled_df.iloc[-len(ltf_df):]
    #     downsampled_df = downsampled_df.loc[ltf_df.index]

    # downsampled_df.index = ltf_df.index[-len(downsampled_df):]
    # assert len(ltf_df) <= len(downsampled_df), "for join method, assert len(ltf_df) <= len(downsampled_df)"

    # ------ check last row's validity ------ #
    assert np.sum(
        ~pd.isnull(downsampled_df.iloc[-1].values)) > 0, "assert np.sum(~pd.isnull(downsampled_df.iloc[-1].values)) > 0"

    return downsampled_df


def to_lower_tf_v2(ltf_df, htf_df, column, backing_i=1, show_info=False):
    #       Todo        #
    #        0. T 에 대한 backing_i 는 제공하지 않음, assert htf_df's itv != T
    #        1. 현재 downsampled df 만 허용, direct_df 사용시 issue 발생 가능할 것
    assert type(column[0]) in [int, np.int64], "column value should be integer"
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
    #         a. len(resampled_df) > len(ltf_df) 경우 slicing 진행
    #           -> ltf_index 를 다가져와도 resampled_indx 를 채울 수 없음 => error
    #           i. resampled length 를 ltf length 로 줄임
    #               1. 이미 htf_indi. 는 계산된 상태이고, 
    #                   a. trader - 마지막 index 만 사용, 전혀 무리없음
    #                   b. IDEP - resampled_df.head(itv) 만큼만 소실된 것 -> 큰 무리없음
    #         b. len(resampled_df) < len(ltf_df)-> 상관없음 (ltf_index 에서 resampled_df length 만큼만 때가면 되니까)
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

    return resampled_df


def to_htf(df, itv, offset):

    h_res_df = df.resample(itv, offset=offset).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    })

    return h_res_df


def calc_rows_and_days(itv_list, row_list, rec_row_list, rows_per_request=1440):

    itv_arr = np.array([itv_to_number(itv_) for itv_ in itv_list])
    row_arr = np.maximum(np.array(row_list), np.array(rec_row_list))

    max_rows = np.max(itv_arr * row_arr)
    days = int(max_rows / rows_per_request) + 1

    return max_rows, days


def calc_tp_out_fee_v2(config_):
    if config_.ep_set.entry_type == 'MARKET':
        if not config_.tp_set.non_tp:  # Todo : 실제로, tp_fee 가 아닌 spread const. 를 위한 spread_fee1 임 (추후 수정 권고)
            tp_fee = config_.trader_set.market_fee + config_.trader_set.limit_fee
        else:
            tp_fee = config_.trader_set.market_fee + config_.trader_set.market_fee
        out_fee = config_.trader_set.market_fee + config_.trader_set.market_fee
    else:
        if not config_.tp_set.non_tp:
            tp_fee = config_.trader_set.limit_fee + config_.trader_set.limit_fee
        else:
            tp_fee = config_.trader_set.limit_fee + config_.trader_set.market_fee
        out_fee = config_.trader_set.limit_fee + config_.trader_set.market_fee

    return tp_fee, out_fee

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


def get_hoga_unit(price):

    if price < 2000:
        return 1
    elif price < 5000:
        return 5
    elif price < 20000:
        return 10
    elif price < 50000:
        return 50
    elif price < 200000:
        return 100
    elif price < 500000:
        return 500
    else:
        return 1000


def calc_with_hoga_unit(price, mode="top"):  # Stock 에서는 price_precision 이 무효함.

    """
    1. 체결을 고려해 올림을 기본으로 함 (+ 1 을 한 이유.)
        a. 예로, 22580 -> 22600 이 매수가로 적당할 건데, 22500 은 매수되지 않을 가능성 있음.
            i. 일단은, tp / ep / out 모두 + 1 hoga_unit 상태.
    2. integer
    """
    if pd.isnull(price):
        return np.nan
    else:
        hoga_unit = get_hoga_unit(price)
        if mode == "top":
            return int(((price // hoga_unit) + 1) * hoga_unit)
        elif mode == "middle":
            return int((price // hoga_unit) * hoga_unit)
        else:
            return int(((price // hoga_unit) - 1) * hoga_unit)


def get_precision_by_price(price):

    try:
        precision = len(str(price).split('.')[1])

    except Exception as e:
        precision = 0

    return precision
