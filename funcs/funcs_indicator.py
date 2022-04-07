import numpy as np
import pandas as pd
from funcs.funcs_trader import to_lower_tf_v2, intmin, ffill, bfill, to_itvnum


def nz(x, y=0):
    # print(x)
    if np.isnan(x):
        return y
    else:
        return x


def round(x):
    x = float(x)
    if x > 0.99:
        return 0.999
    else:
        if x < -0.99:
            return -0.999
        else:
            return x


def iszero(val, eps):
    return abs(val) <= eps


def tozero(fst, snd):
    eps = 1e-10
    result = fst + snd
    if iszero(result, eps):
        result = 0
    else:
        if not iszero(result, 1e-4):
            pass
        else:
            result = 1e-15

    return result


def tozero_v2(result):

    eps = 1e-10

    if iszero(result, eps):
        result = 0
    else:
        if not iszero(result, 1e-4):
            pass
        else:
            result = 1e-15

    return result


def stdev(df, period):
    avg = df['close'].rolling(period).mean().to_numpy()
    std = np.zeros(len(df))

    #       Todo        #
    #        1. rolling - one_line 보류
    close = df['close'].to_numpy()
    for i in range(period - 1, len(df)):
        sum_series = close[i + 1 - period:i + 1] - avg[i]
        sum2_series = [tozero_v2(sum_) ** 2 for sum_ in sum_series]
        #   rolling minus avg
        std[i] = np.sum(sum2_series)
    std = np.sqrt(std / period)

    return std  # 1.9082684516906738 25.619273900985718


def dev(data, period):

    mean_ = data.rolling(period).mean()
    dev_ = pd.Series(index=data.index)

    for i in range(len(data)):
        sum = 0.0
        for backing_i in range(period):
            val_ = data.iloc[i - backing_i]
            sum += abs(val_ - mean_.iloc[i])

        dev_.iloc[i] = sum / period

        # temp_sum = abs(data.iloc[i - backing_i] - mean_.iloc[i])

    return dev_

def imb_ratio_v2(df, itv):  # watch 3 candle

  itv_num = to_itvnum(itv)
  b2_itv_num = itv_num * 2

  b1_high = df['high_{}'.format(itv)].shift(itv_num).to_numpy()
  b1_low = df['low_{}'.format(itv)].shift(itv_num).to_numpy()
  b1_candle_range = b1_high - b1_low

  high = df['high_{}'.format(itv)].to_numpy()
  low = df['low_{}'.format(itv)].to_numpy()
  b2_high = df['high_{}'.format(itv)].shift(b2_itv_num).to_numpy()
  b2_low = df['low_{}'.format(itv)].shift(b2_itv_num).to_numpy()

  open = df['open_{}'.format(itv)].to_numpy()
  close = df['close_{}'.format(itv)].to_numpy()

  # 추후에 통계 측정해야함 -> bir 에 따른 개별 trader 의 epout / tpep 이라던가 => short 에 양봉은 취급안함 (why use np.nan)
  df['short_ir_{}'.format(itv)] = np.where(close < open, (b2_low - high) / b1_candle_range, np.nan) # close < open & close < b1_low
  df['long_ir_{}'.format(itv)] = np.where(close > open, (low - b2_high) / b1_candle_range, np.nan) # close > open & close > b1_high

  return

def imb_ratio(df, itv):

  itv_num = to_itvnum(itv)

  high = df['high_{}'.format(itv)].to_numpy()
  low = df['low_{}'.format(itv)].to_numpy()
  candle_range = high - low

  open = df['open_{}'.format(itv)].to_numpy()
  close = df['close_{}'.format(itv)].to_numpy()
  b1_high = df['high_{}'.format(itv)].shift(itv_num).to_numpy()
  b1_low = df['low_{}'.format(itv)].shift(itv_num).to_numpy()

  # 추후에 통계 측정해야함 -> bir 에 따른 개별 trader 의 epout / tpep 이라던가 => short 에 양봉은 취급안함 (why use np.nan)
  df['short_ir_{}'.format(itv)] = np.where(close < open, (b1_low - close) / candle_range, np.nan) # close < open & close < b1_low
  df['long_ir_{}'.format(itv)] = np.where(close > open, (close - b1_high) / candle_range, np.nan) # close > open & close > b1_high

  return

def norm_data(res_df, target_data, target_col, minmax_col=None, norm_period=100):
    if minmax_col is None:
        minmax_col = ['low', 'high']
    res_df['norm_min'] = res_df[minmax_col[0]].rolling(norm_period).min()
    res_df['norm_max'] = res_df[minmax_col[1]].rolling(norm_period).max()
    res_df['norm_' + target_col] = target_data / (res_df['norm_max'].to_numpy() - res_df['norm_min'].to_numpy()) * 100

    return

def norm_hohlc(res_df, h_c_itv, norm_period=120):
    try:
        res_df.norm_max
    except:
        res_df['norm_max'] = res_df['high'].rolling(norm_period).max()
        res_df['norm_min'] = res_df['low'].rolling(norm_period).min()

    norm_min_np = res_df['norm_min'].to_numpy()
    for hcol in h_candle_cols(h_c_itv):
        res_df['norm_' + hcol] = (res_df[hcol].to_numpy() - norm_min_np) / (res_df['norm_max'].to_numpy() - norm_min_np) * 100

    return

#    개별 candle_score 조회를 위해 _ratio 와 분리함        #
def get_candle_score(o, h, l, c, updown=None, unsigned=True):

    #   check up / downward
    up = np.where(c >= o, 1, 0)

    total_len = h - l
    upper_wick = (h - np.maximum(c, o)) / total_len
    lower_wick = (np.minimum(c, o) - l) / total_len

    body_score = abs(c - o) / total_len * 100

    up_score = (1 - upper_wick) * 100
    dn_score = (1 - lower_wick) * 100

    if updown is not None:
        if updown == "up":
            wick_score = up_score
        else:
            if unsigned:
                wick_score = dn_score
            else:
                wick_score = -dn_score
    else:
        if unsigned:
            wick_score = np.where(up, up_score, dn_score)
        else:
            wick_score = np.where(up, up_score, -dn_score)

    return wick_score, body_score


def candle_score_v2(res_df, itv, ohlc_col=None, updown=None, unsigned=True):
    if ohlc_col is None:
        ohlc_col = ["open", "high", "low", "close"]

    # ohlcs = res_df[ohlc_col]
    # o, h, l, c = np.split(ohlcs.values, 4, axis=1)
    o, h, l, c = [res_df[col_].to_numpy() for col_ in ohlc_col]
    #   check up / downward
    up = np.where(c >= o, 1, 0)

    total_len = h - l
    upper_wick = (h - np.maximum(c, o)) / total_len
    lower_wick = (np.minimum(c, o) - l) / total_len

    body_score = abs(c - o) / total_len * 100

    up_score = (1 - upper_wick) * 100
    dn_score = (1 - lower_wick) * 100

    if updown is not None:
        if updown == "up":
            wick_score = up_score
        else:
            if unsigned:
                wick_score = dn_score
            else:
                wick_score = -dn_score
    else:
        if unsigned:
            wick_score = np.where(up, up_score, dn_score)
        else:
            wick_score = np.where(up, up_score, -dn_score)

    res_df['wick_score_{}'.format(itv)] = wick_score
    res_df['body_score_{}'.format(itv)] = body_score

    return

def candle_score(res_df, ohlc_col=None, updown=None, unsigned=True):

    if ohlc_col is None:
        ohlc_col = ["open", "high", "low", "close"]

    ohlcs = res_df[ohlc_col]
    o, h, l, c = np.split(ohlcs.values, 4, axis=1)

    return get_candle_score(o, h, l, c, updown, unsigned)

def rel_abs_ratio(res_df, itv, norm_period=120):
    if itv == 'T':
        candle_range = res_df['high'.format(itv)].to_numpy() - res_df['low'.format(itv)].to_numpy()
    else:
        candle_range = res_df['high_{}'.format(itv)].to_numpy() - res_df['low_{}'.format(itv)].to_numpy()

    res_df['rel_ratio_{}'.format(itv)] = candle_range / pd.Series(candle_range).shift(to_itvnum(itv)).to_numpy()

    norm_max = res_df['high'].rolling(norm_period).max().to_numpy()
    norm_min = res_df['low'].rolling(norm_period).min().to_numpy()
    res_df['abs_ratio_{}'.format(itv)] = candle_range / (norm_max - norm_min)

    return

def h_ratio_cols(h_c_itv):   # 아래 함수와 통일성 유지
  return ['hrel_ratio_{}'.format(h_c_itv), 'habs_ratio_{}'.format(h_c_itv)]

def ratio_cols(h_c_itv):   # 아래 함수와 통일성 유지
  return ['rel_ratio_{}'.format(h_c_itv), 'abs_ratio_{}'.format(h_c_itv)]

def h_oc_cols(h_c_itv):   # 아래 함수와 통일성 유지
  return ['hopen_{}'.format(h_c_itv), 'hclose_{}'.format(h_c_itv)]

def oc_cols(h_c_itv):   # 아래 함수와 통일성 유지
  return ['open_{}'.format(h_c_itv), 'close_{}'.format(h_c_itv)]

def h_candle_cols(h_c_itv):
    return ['hopen_{}'.format(h_c_itv), 'hhigh_{}'.format(h_c_itv), 'hlow_{}'.format(h_c_itv), 'hclose_{}'.format(h_c_itv)]

def ohlc_cols(h_c_itv):
    return ['open_{}'.format(h_c_itv), 'high_{}'.format(h_c_itv), 'low_{}'.format(h_c_itv), 'close_{}'.format(h_c_itv)]

def score_cols(h_c_itv):
    return ['front_wick_score_{}'.format(h_c_itv), 'body_score_{}'.format(h_c_itv), 'back_wick_score_{}'.format(h_c_itv)]


def sma(data, period):
    return data.rolling(period).mean()


#       Todo - recursive       #
def ema(data, period, adjust=False):
    return pd.Series(data.ewm(span=period, adjust=adjust).mean())

# def ema(data, period):
#
#     alpha = 2 / (period + 1)
#     avg = sma(data, period)
#
#     sum_ = alpha * data + (1 - alpha) * avg
#
#     return sum_

# # @functools.lru_cache(10)
# @jit(nopython=True)
# def ema(data, avg, period):
#
#     alpha = 2 / (period + 1)
#
#     # avg = sma(data, period).values
#     assert not pd.isnull(avg[-period]), "avg.iloc[-period] should not be nan"
#     # print(avg.iloc[-period])
#
#     sum_ = pd.Series(index=data.index).values
#     # sum_ = 0
#     len_data = len(data)
#     # start_idx = len_data - period
#
#     for i in range(period, len_data):    # back_pr 문제로 1 부터 해야할 것
#
#         #   i idx 의 ema 를 결정   # - each ema's start_value = avg.iloc[j]
#         temp_sum = pd.Series(index=data.index).values
#         for j in range(i + 1 - period, i + 1):
#             if np.isnan(temp_sum[j - 1]):
#                 temp_sum[j] = avg[j]
#             else:
#                 temp_sum[j] = alpha * data[j] + (1 - alpha) * nz(temp_sum[j - 1])
#                 # temp_sum[i] = alpha * data[i] + (1 - alpha) * nz(avg[i - 1])
#         sum_[i] = temp_sum[j]
#
#     return sum_


#       Todo - recursive       #
# def ema(data, period):    # start_value 인 avg.iloc[i] 가 days 에 따라 변경됨
#
#     alpha = 2 / (period + 1)
#
#     # avg = sma(data, period)
#     # assert not np.sum(pd.isnull(avg.iloc[-period])), "assert not np.sum(pd.isnull(avg.iloc[-period]))"
#     # print(np.sum(pd.isnull(avg.iloc[-period])))
#
#     sum_ = pd.Series(index=data.index, dtype=np.float64)
#
#     len_data = len(data)
#     sum_.iloc[0] = 0
#     for i in range(1, len_data):
#         sum_.iloc[i] = alpha * data.iloc[i] + (1 - alpha) * sum_.iloc[i - 1]
#
#     return sum_

def cloud_bline(df, period):
    return (df['high'].rolling(period).max() + df['low'].rolling(period).min()) / 2

def cloud_bline_v2(df, period):  # dc_base 와 다를게 없는데 ?
    itv = pd.infer_freq(df.index)
    df['cbline_{}{}'.format(itv, period)] = (df['high'].rolling(period).max().to_numpy() + df['low'].rolling(period).min().to_numpy()) / 2

    return

def heikinashi(df):

    ha_df = df.copy()
    ha_df['close'] = (df['open'] + df['close'] + df['high'] + df['low']) / 4
    ha_df['open'] = np.nan
    for i in range(1, len(df)):
        if pd.isna(ha_df['open'].iloc[i - 1]):
            ha_df['open'].iloc[i] = (df['open'].iloc[i] + df['close'].iloc[i]) / 2
        else:
            ha_df['open'].iloc[i] = (ha_df['open'].iloc[i - 1] + ha_df['close'].iloc[i - 1]) / 2
    ha_df['high'] = np.max(ha_df.iloc[:, [0, 1, 3]].values, axis=1)
    ha_df['low'] = np.min(ha_df.iloc[:, [0, 2, 3]].values, axis=1)

    return ha_df

def heikinashi_v2(res_df_):
  data_cols = ['open', 'high', 'low', 'close']
  o, h, l, c = [res_df_[col_].to_numpy() for col_ in data_cols]

  ha_c = (o + h + l + c) / 4
  ha_o = np.full(len(o), np.nan)
  ha_o[0] = (o[0] + c[0]) / 2
  #       Todo - recursive -> numba 사용해서 for loop 돌리는게 합리적이라고 함       #
  for i in range(1, len(ha_o)):
      ha_o[i] = (ha_o[i - 1] + ha_c[i - 1]) / 2
  ha_h = np.max(np.vstack((ha_o, h, ha_c)), axis=0)  # ohc
  ha_l = np.min(np.vstack((ha_o, l, ha_c)), axis=0)

  for cols, data in zip(data_cols, [ha_o, ha_h, ha_l, ha_c]):
    res_df_['ha' + cols] = data

  return

def roc(data, period):
    roc_ = 100 * (data - data.shift(period)) / data.shift(period)

    return roc_


def ema_roc(data, roc_period, period):
    ema_roc_ = ema(roc(data, roc_period), period)

    return ema_roc_


#       Todo - recursive       #
def trix_hist(df, period, multiplier, signal_period):
    triple = ema(ema(ema(df['close'], period), period), period)

    roc_ = 100 * (triple.diff() / triple)
    trix = multiplier * roc_
    signal = trix.rolling(signal_period).mean()

    hist = trix - signal

    return hist


def dtk_plot(res_df, dtk_itv2, hhtf_entry, use_dtk_line, np_timeidx=None):

    if np_timeidx is None:  # temporary adjusted for "utils_v3_1216.py"
        np_timeidx = np.array(list(map(lambda x: intmin(x), res_df.index)))

    res_df['short_dtk_plot_1'] = res_df['bb_lower_%s' % dtk_itv2]
    res_df['short_dtk_plot_0'] = res_df['dc_upper_%s' % dtk_itv2]
    res_df['short_dtk_plot_gap'] = res_df['short_dtk_plot_0'] - res_df['short_dtk_plot_1']

    res_df['long_dtk_plot_1'] = res_df['bb_upper_%s' % dtk_itv2]
    res_df['long_dtk_plot_0'] = res_df['dc_lower_%s' % dtk_itv2]
    res_df['long_dtk_plot_gap'] = res_df['long_dtk_plot_1'] - res_df['long_dtk_plot_0']

    res_df['hh_entry'] = np.zeros(len(res_df))

    res_df['hh_entry'] = np.where(  # (res_df['open'] >= res_df['bb_lower_%s' % dtk_itv2]) &
        (res_df['close'].shift(hhtf_entry * 1) >= res_df['bb_lower_%s' % dtk_itv2]) &
        (res_df['close'] < res_df['bb_lower_%s' % dtk_itv2]) &
        (np_timeidx % hhtf_entry == (hhtf_entry - 1))
        , res_df['hh_entry'] - 1, res_df['hh_entry'])

    res_df['hh_entry'] = np.where(  # (res_df['open'] <= res_df['bb_upper_%s' % dtk_itv2]) &
        (res_df['close'].shift(hhtf_entry * 1) <= res_df['bb_upper_%s' % dtk_itv2]) &
        (res_df['close'] > res_df['bb_upper_%s' % dtk_itv2]) &
        (np_timeidx % hhtf_entry == (hhtf_entry - 1))
        , res_df['hh_entry'] + 1, res_df['hh_entry'])

    if use_dtk_line:
        res_df['short_dtk_plot_1'] = np.where(res_df['hh_entry'] == -1, res_df['short_dtk_plot_1'], np.nan)
        res_df['short_dtk_plot_1'] = ffill(res_df['short_dtk_plot_1'].values.reshape(1, -1)).reshape(-1, 1)
        res_df['short_dtk_plot_gap'] = np.where(res_df['hh_entry'] == -1, res_df['short_dtk_plot_gap'], np.nan)
        res_df['short_dtk_plot_gap'] = ffill(res_df['short_dtk_plot_gap'].values.reshape(1, -1)).reshape(-1, 1)

        res_df['long_dtk_plot_1'] = np.where(res_df['hh_entry'] == 1, res_df['long_dtk_plot_1'], np.nan)
        res_df['long_dtk_plot_1'] = ffill(res_df['long_dtk_plot_1'].values.reshape(1, -1)).reshape(-1, 1)
        res_df['long_dtk_plot_gap'] = np.where(res_df['hh_entry'] == 1, res_df['long_dtk_plot_gap'], np.nan)
        res_df['long_dtk_plot_gap'] = ffill(res_df['long_dtk_plot_gap'].values.reshape(1, -1)).reshape(-1, 1)

    return res_df


# def h_candle(res_df, interval_): .

def h_candle_v3(res_df_, itv):
    h_res_df = res_df_.resample(itv).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    })

    h_res_df = pd.concat([h_res_df, res_df_[['open', 'high', 'low', 'close']].iloc[-1:]])

    #       downsampled h_res_df 의 offset 기준이 00:00 이라서     #
    h_res_df2 = h_res_df.resample('T').ffill()  # 기본적으로 'T' 단위로만 resampling, ffill 이 맞음

    h_candle_col = ['open_{}'.format(itv), 'high_{}'.format(itv), 'low_{}'.format(itv), 'close_{}'.format(itv)]

    #        1. res_df_ & resampled data idx unmatch
    #           1_1. res_df_ 의 남는 rows 를 채워주어야하는데
    # print("res_df_.tail(15) :", res_df_[['open', 'high', 'low', 'close']].tail(15))
    # print("h_res_df.tail(5) :", h_res_df.tail(5))
    # print("h_res_df2.tail(15) :", h_res_df2.tail(15))
    res_df_[h_candle_col] = h_res_df2.values[-len(res_df_):]

    return

def h_candle_v2(res_df_, itv):

    h_res_df = res_df_.resample(itv).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    })

    h_res_df = pd.concat([h_res_df, res_df_[['open', 'high', 'low', 'close']].iloc[-1:]])

    #       downsampled h_res_df 의 offset 기준이 00:00 이라서     #
    h_res_df2 = h_res_df.resample('T').ffill()  # 기본적으로 'T' 단위로만 resampling, ffill 이 맞음

    h_candle_col = ['hopen_{}'.format(itv), 'hhigh_{}'.format(itv), 'hlow_{}'.format(itv), 'hclose_{}'.format(itv)]

    #        1. res_df_ & resampled data idx unmatch
    #           1_1. res_df_ 의 남는 rows 를 채워주어야하는데
    # print("res_df_.tail(15) :", res_df_[['open', 'high', 'low', 'close']].tail(15))
    # print("h_res_df.tail(5) :", h_res_df.tail(5))
    # print("h_res_df2.tail(15) :", h_res_df2.tail(15))
    res_df_[h_candle_col] = h_res_df2.values[-len(res_df_):]

    return res_df_


def cct_bbo(df, period, smooth):

    avg_ = df['close'].rolling(period).mean()
    stdev_ = stdev(df, period)
    # print("len(stdev_) :", len(stdev_))
    # print("len(stdev_.apply(lambda x: 2 * x)) :", len(stdev_.apply(lambda x: 2 * x)))

    cctbbo = 100 * (df['close'] + stdev_.apply(lambda x: 2 * x) - avg_) / (stdev_.apply(lambda x: 4 * x))
    ema_cctbbo = ema(cctbbo, smooth)

    return cctbbo, ema_cctbbo


def donchian_channel(df, period):
    hh = df['high'].rolling(period).max().to_numpy()
    ll = df['low'].rolling(period).min().to_numpy()
    base = (hh + ll) / 2

    return hh, ll, base

def donchian_channel_v2(df, period):
  itv = pd.infer_freq(df.index)
  upper_name = 'dc_upper_{}{}'.format(itv, period)
  lower_name = 'dc_lower_{}{}'.format(itv, period)

  try:
    df.drop([upper_name, lower_name], inplace=True, axis=1)
  except:
    pass

  df[upper_name] = df['high'].rolling(period).max()
  df[lower_name] = df['low'].rolling(period).min()

  return

def sd_dc(df, period1, period2, ltf_df=None):
    assert period1 <= period2, "assert period1 <= period2"
    donchian_channel_v2(df, period1)
    donchian_channel_v2(df, period2)
    itv = pd.infer_freq(df.index)
    if itv != 'T':
      assert ltf_df is not None, "assert ltf_df is not None"
      ltf_df = ltf_df.join(to_lower_tf_v2(ltf_df, df, [-4, -3, -2, -1]), how='inner')
    else:
      ltf_df = df

    ltf_df['short_base_{}'.format(itv)] = (ltf_df['dc_lower_{}{}'.format(itv, period1)].to_numpy() + ltf_df['dc_upper_{}{}'.format(itv, period2)].to_numpy()) / 2
    ltf_df['long_base_{}'.format(itv)] = (ltf_df['dc_upper_{}{}'.format(itv, period1)].to_numpy() + ltf_df['dc_lower_{}{}'.format(itv, period2)].to_numpy()) / 2

    return ltf_df

def get_line(touch_idx, rtc_):
  touch_idx_copy = touch_idx.copy()
  touch_line = np.full_like(rtc_, np.nan)

  nan_idx = np.isnan(touch_idx_copy)
  touch_idx_copy[nan_idx] = 0   # for indexing array
  touch_line = rtc_[touch_idx_copy.astype(int)].copy()
  touch_line[nan_idx] = np.nan   # for true comp.

  return touch_line


def wave_range_v6(df, period1, period2, ltf_df=None, touch_period=50):  # v2 for period1 only

    itv = pd.infer_freq(df.index)

    donchian_channel_v2(df, period1)
    donchian_channel_v2(df, period2)
    dc_lower_, dc_upper_ = df['dc_lower_{}{}'.format(itv, period1)].to_numpy(), df['dc_upper_{}{}'.format(itv, period1)].to_numpy()
    dc_lower2_, dc_upper2_ = df['dc_lower_{}{}'.format(itv, period2)].to_numpy(), df['dc_upper_{}{}'.format(itv, period2)].to_numpy()

    short_base = (dc_lower_ + dc_upper2_) / 2
    long_base = (dc_upper_ + dc_lower2_) / 2
    df['short_base_{}{}{}'.format(itv, period1, period2)] = short_base
    df['long_base_{}{}{}'.format(itv, period1, period2)] = long_base

    len_df = len(df)
    short_open_res = np.ones(len_df)
    long_open_res = np.ones(len_df)

    data_cols = ['open', 'high', 'low']
    open, high, low = [df[col_].to_numpy() for col_ in data_cols]

    short_a_touch_idx = pd.Series(np.where(high >= dc_upper2_, np.arange(len_df), np.nan)).rolling(touch_period,
                                                                                                   min_periods=1).max().to_numpy()  # min -> max
    short_b_touch_idx = pd.Series(np.where(low <= dc_lower_, np.arange(len_df), np.nan)).rolling(touch_period,
                                                                                                 min_periods=1).max().to_numpy()  # min 으로 하면 a 이전에 b 있는건 모두 제외하게됨
    short_open_res *= (high >= short_base) & (short_base >= open) & (short_a_touch_idx < short_b_touch_idx)
    df['short_a_touch_idx_{}{}{}'.format(itv, period1, period2)] = short_a_touch_idx
    df['short_b_touch_idx_{}{}{}'.format(itv, period1, period2)] = short_b_touch_idx
    df['short_a_line_{}{}{}'.format(itv, period1, period2)] = get_line(short_a_touch_idx, dc_upper2_)
    df['short_b_line_{}{}{}'.format(itv, period1, period2)] = get_line(short_b_touch_idx, dc_lower_)
    df['short_wave_point_{}{}{}'.format(itv, period1, period2)] = short_open_res

    long_a_touch_idx = pd.Series(np.where(low <= dc_lower2_, np.arange(len_df), np.nan)).rolling(touch_period, min_periods=1).max().to_numpy()
    long_b_touch_idx = pd.Series(np.where(high >= dc_upper_, np.arange(len_df), np.nan)).rolling(touch_period, min_periods=1).max().to_numpy()
    long_open_res *= (open >= long_base) & (long_base >= low) & (long_a_touch_idx < long_b_touch_idx)
    df['long_a_touch_idx_{}{}{}'.format(itv, period1, period2)] = long_a_touch_idx
    df['long_b_touch_idx_{}{}{}'.format(itv, period1, period2)] = long_b_touch_idx
    df['long_a_line_{}{}{}'.format(itv, period1, period2)] = get_line(long_a_touch_idx, dc_lower2_)
    df['long_b_line_{}{}{}'.format(itv, period1, period2)] = get_line(long_b_touch_idx, dc_upper_)
    df['long_wave_point_{}{}{}'.format(itv, period1, period2)] = long_open_res

    if itv != 'T':
        assert ltf_df is not None, "assert ltf_df is not None"
        join_cols = np.arange(-16, 0, 1).astype(int)  # points & donchian_channels
        ltf_df.drop(df.columns[join_cols], inplace=True, axis=1, errors='ignore')
        try:
            ltf_df = ltf_df.join(to_lower_tf_v2(ltf_df, df, join_cols), how='inner')
        except Exception as e:
            print("error in wave_range_v5's join() :", e)
    else:
        ltf_df = df

    return ltf_df

def wave_range_v5(df, period1, period2, ltf_df=None, touch_period=50):  # v2 for period1 only

    itv = pd.infer_freq(df.index)

    donchian_channel_v2(df, period1)
    donchian_channel_v2(df, period2)
    dc_lower_, dc_upper_ = df['dc_lower_{}{}'.format(itv, period1)].to_numpy(), df['dc_upper_{}{}'.format(itv, period1)].to_numpy()
    dc_lower2_, dc_upper2_ = df['dc_lower_{}{}'.format(itv, period2)].to_numpy(), df['dc_upper_{}{}'.format(itv, period2)].to_numpy()

    short_base = (dc_lower_ + dc_upper2_) / 2
    long_base = (dc_upper_ + dc_lower2_) / 2
    df['short_base_{}{}{}'.format(itv, period1, period2)] = short_base
    df['long_base_{}{}{}'.format(itv, period1, period2)] = long_base

    len_df = len(df)
    short_open_res = np.ones(len_df)
    long_open_res = np.ones(len_df)

    data_cols = ['open', 'high', 'low']
    open, high, low = [df[col_].to_numpy() for col_ in data_cols]

    short_a_touch_idx = pd.Series(np.where(high >= dc_upper2_, np.arange(len_df), np.nan)).rolling(touch_period,
                                                                                                   min_periods=1).max().to_numpy()  # min -> max
    short_b_touch_idx = pd.Series(np.where(low <= dc_lower_, np.arange(len_df), np.nan)).rolling(touch_period,
                                                                                                 min_periods=1).max().to_numpy()  # min 으로 하면 a 이전에 b 있는건 모두 제외하게됨
    short_open_res *= (high >= short_base) & (short_base >= open) & (short_a_touch_idx < short_b_touch_idx)
    df['short_a_touch_idx_{}{}{}'.format(itv, period1, period2)] = short_a_touch_idx
    df['short_b_touch_idx_{}{}{}'.format(itv, period1, period2)] = short_b_touch_idx
    df['short_wave_point_{}{}{}'.format(itv, period1, period2)] = short_open_res

    long_a_touch_idx = pd.Series(np.where(low <= dc_lower2_, np.arange(len_df), np.nan)).rolling(touch_period, min_periods=1).max().to_numpy()
    long_b_touch_idx = pd.Series(np.where(high >= dc_upper_, np.arange(len_df), np.nan)).rolling(touch_period, min_periods=1).max().to_numpy()
    long_open_res *= (open >= long_base) & (long_base >= low) & (long_a_touch_idx < long_b_touch_idx)
    df['long_a_touch_idx_{}{}{}'.format(itv, period1, period2)] = long_a_touch_idx
    df['long_b_touch_idx_{}{}{}'.format(itv, period1, period2)] = long_b_touch_idx
    df['long_wave_point_{}{}{}'.format(itv, period1, period2)] = long_open_res

    if itv != 'T':
        assert ltf_df is not None, "assert ltf_df is not None"
        join_cols = np.arange(-12, 0, 1).astype(int)  # points & donchian_channels
        ltf_df.drop(df.columns[join_cols], inplace=True, axis=1, errors='ignore')
        try:
            ltf_df = ltf_df.join(to_lower_tf_v2(ltf_df, df, join_cols), how='inner')
            # ltf_df = ltf_df.merge(to_lower_tf_v2(ltf_df, df, [-7, -6, -5, -4, -3, -2, -1]), how='inner')
        except Exception as e:
            print("error in wave_range_v5's join() :", e)
    else:
        ltf_df = df

    return ltf_df

def wave_range_v4(df, period1, ltf_df=None, touch_lbperiod=50):  # v2 for period1 only

    itv = pd.infer_freq(df.index)

    donchian_channel_v2(df, period1)
    dc_lower_, dc_upper_ = df['dc_lower_{}{}'.format(itv, period1)].to_numpy(), df['dc_upper_{}{}'.format(itv, period1)].to_numpy()

    df['wave_base_{}'.format(itv)] = (dc_lower_ + dc_upper_) / 2  # dc_base touch 사용하면 point 를 좀 더 일찍 잡을 수 있음
    wave_base_ = df['wave_base_{}'.format(itv)].to_numpy()
    # short_base_ = df['short_base__{}'.format(itv)].to_numpy()
    # long_base_ = df['long_base__{}'.format(itv)].to_numpy()

    len_df = len(df)
    short_open_res = np.ones(len_df)
    long_open_res = np.ones(len_df)

    data_cols = ['open', 'high', 'low']
    open, high, low = [df[col_].to_numpy() for col_ in data_cols]

    dc_upper_touch_idx = pd.Series(np.where(high >= dc_upper_, np.arange(len_df), np.nan)).rolling(touch_lbperiod,
                                                                                                   min_periods=1).max().to_numpy()  # min -> max
    dc_lower_touch_idx = pd.Series(np.where(low <= dc_lower_, np.arange(len_df), np.nan)).rolling(touch_lbperiod,
                                                                                                  min_periods=1).max().to_numpy()  # min 으로 하면 a 이전에 b 있는건 모두 제외하게됨

    # df['wave_upper_{}'.format(itv)] = pd.Series(get_line(dc_upper_touch_idx, dc_upper_)).ffill().to_numpy()
    # df['wave_lower_{}'.format(itv)] = pd.Series(get_line(dc_lower_touch_idx, dc_lower_)).ffill().to_numpy()

    short_open_res *= (high >= wave_base_) & (wave_base_ >= open) & (dc_upper_touch_idx < dc_lower_touch_idx)
    df['short_wave_point_{}{}'.format(itv, period1)] = short_open_res

    long_open_res *= (open >= wave_base_) & (wave_base_ >= low) & (dc_lower_touch_idx < dc_upper_touch_idx)
    df['long_wave_point_{}{}'.format(itv, period1)] = long_open_res

    if itv != 'T':
        assert ltf_df is not None, "assert ltf_df is not None"

        join_cols = np.arange(-5, 0, 1).astype(int)
        ltf_df.drop(df.columns[join_cols], inplace=True, axis=1, errors='ignore')
        try:
            ltf_df = ltf_df.join(to_lower_tf_v2(ltf_df, df, join_cols), how='inner')
            # ltf_df = ltf_df.merge(to_lower_tf_v2(ltf_df, df, [-7, -6, -5, -4, -3, -2, -1]), how='inner')
        except Exception as e:
            print("error in wave_range_v4's join() :", e)
    else:
        ltf_df = df

    return ltf_df

def wave_range_v3(df, period1, ltf_df=None, touch_lbperiod=50):  # v2 for period1 only

    itv = pd.infer_freq(df.index)

    donchian_channel_v2(df, period1)
    dc_lower_, dc_upper_ = df['dc_lower_{}{}'.format(itv, period1)].to_numpy(), df['dc_upper_{}{}'.format(itv, period1)].to_numpy()

    df['wave_base_{}'.format(itv)] = (dc_lower_ + dc_upper_) / 2  # dc_base touch 사용하면 point 를 좀 더 일찍 잡을 수 있음
    wave_base_ = df['wave_base_{}'.format(itv)].to_numpy()
    # short_base_ = df['short_base__{}'.format(itv)].to_numpy()
    # long_base_ = df['long_base__{}'.format(itv)].to_numpy()

    len_df = len(df)
    short_open_res = np.ones(len_df)
    long_open_res = np.ones(len_df)

    data_cols = ['open', 'high', 'low']
    open, high, low = [df[col_].to_numpy() for col_ in data_cols]

    dc_upper_touch_idx = pd.Series(np.where(high >= dc_upper_, np.arange(len_df), np.nan)).rolling(touch_lbperiod,
                                                                                                   min_periods=1).max().to_numpy()  # min -> max
    dc_lower_touch_idx = pd.Series(np.where(low <= dc_lower_, np.arange(len_df), np.nan)).rolling(touch_lbperiod,
                                                                                                  min_periods=1).max().to_numpy()  # min 으로 하면 a 이전에 b 있는건 모두 제외하게됨

    df['wave_upper_{}'.format(itv)] = pd.Series(get_line(dc_upper_touch_idx, dc_upper_)).ffill().to_numpy()
    df['wave_lower_{}'.format(itv)] = pd.Series(get_line(dc_lower_touch_idx, dc_lower_)).ffill().to_numpy()

    short_open_res *= (high >= wave_base_) & (wave_base_ >= open) & (dc_upper_touch_idx < dc_lower_touch_idx)
    df['short_wave_point_{}'.format(itv)] = short_open_res

    long_open_res *= (open >= wave_base_) & (wave_base_ >= low) & (dc_lower_touch_idx < dc_upper_touch_idx)
    df['long_wave_point_{}'.format(itv)] = long_open_res

    if itv != 'T':
        assert ltf_df is not None, "assert ltf_df is not None"

        join_cols = np.arange(-7, 0, 1).astype(int)
        ltf_df.drop(df.columns[join_cols], inplace=True, axis=1, errors='ignore')
        try:
            ltf_df = ltf_df.join(to_lower_tf_v2(ltf_df, df, join_cols), how='inner')
            # ltf_df = ltf_df.merge(to_lower_tf_v2(ltf_df, df, [-7, -6, -5, -4, -3, -2, -1]), how='inner')
        except Exception as e:
            print("error in wave_range_v3's join() :", e)
            print(df.iloc[-1])
    else:
        ltf_df = df

    return ltf_df

def wave_range_v2(df, period1, ltf_df=None, touch_lbperiod=50):  # v2 for period1 only

    donchian_channel_v2(df, period1)

    itv = pd.infer_freq(df.index)
    if itv != 'T':
        assert ltf_df is not None, "assert ltf_df is not None"
        try:
            ltf_df = ltf_df.join(to_lower_tf_v2(ltf_df, df, [-2, -1]), how='inner')
        except Exception as e:
            print("error in wave_range_v2's join() :", e)
    else:
        ltf_df = df

    dc_lower_, dc_upper_ = ltf_df['dc_lower_{}{}'.format(itv, period1)].to_numpy(), ltf_df['dc_upper_{}{}'.format(itv, period1)].to_numpy()

    ltf_df['wave_base_{}'.format(itv)] = (dc_lower_ + dc_upper_) / 2  # dc_base touch 사용하면 point 를 좀 더 일찍 잡을 수 있음

    wave_base_ = ltf_df['wave_base_{}'.format(itv)].to_numpy()
    # short_base_ = ltf_df['short_base__{}'.format(itv)].to_numpy()
    # long_base_ = ltf_df['long_base__{}'.format(itv)].to_numpy()

    len_df = len(ltf_df)
    short_open_res = np.ones(len_df)
    long_open_res = np.ones(len_df)

    data_cols = ['open', 'high', 'low']
    open, high, low = [ltf_df[col_].to_numpy() for col_ in data_cols]

    dc_upper_touch_idx = pd.Series(np.where(high >= dc_upper_, np.arange(len_df), np.nan)).rolling(touch_lbperiod,
                                                                                                   min_periods=1).max().to_numpy()  # min -> max
    dc_lower_touch_idx = pd.Series(np.where(low <= dc_lower_, np.arange(len_df), np.nan)).rolling(touch_lbperiod,
                                                                                                  min_periods=1).max().to_numpy()  # min 으로 하면 a 이전에 b 있는건 모두 제외하게됨

    ltf_df['wave_upper_{}'.format(itv)] = pd.Series(get_line(dc_upper_touch_idx, dc_upper_)).ffill().to_numpy()
    ltf_df['wave_lower_{}'.format(itv)] = pd.Series(get_line(dc_lower_touch_idx, dc_lower_)).ffill().to_numpy()

    short_open_res *= (high >= wave_base_) & (wave_base_ >= open) & (dc_upper_touch_idx < dc_lower_touch_idx)
    ltf_df['short_wave_point_{}'.format(itv)] = short_open_res.astype(bool)

    long_open_res *= (open >= wave_base_) & (wave_base_ >= low) & (dc_lower_touch_idx < dc_upper_touch_idx)
    ltf_df['long_wave_point_{}'.format(itv)] = long_open_res.astype(bool)

    return ltf_df


def wave_range(df, period1, period2, ltf_df=None, touch_lbperiod=50):
    assert period1 <= period2, "assert period1 <= period2"
    donchian_channel_v2(df, period1)
    donchian_channel_v2(df, period2)
    itv = pd.infer_freq(df.index)
    if itv != 'T':
        assert ltf_df is not None, "assert ltf_df is not None"
        ltf_df = ltf_df.join(to_lower_tf_v2(ltf_df, df, [-4, -3, -2, -1]), how='inner')
    else:
        ltf_df = df

    short_tp_1_, short_tp_0_ = ltf_df['dc_lower_{}{}'.format(itv, period1)].to_numpy(), ltf_df['dc_upper_{}{}'.format(itv, period2)].to_numpy()
    long_tp_1_, long_tp_0_ = ltf_df['dc_upper_{}{}'.format(itv, period1)].to_numpy(), ltf_df['dc_lower_{}{}'.format(itv, period2)].to_numpy()

    ltf_df['short_base_{}'.format(itv)] = (short_tp_1_ + short_tp_0_) / 2  # dc_base touch 사용하면 point 를 좀 더 일찍 잡을 수 있음
    ltf_df['long_base_{}'.format(itv)] = (long_tp_1_ + long_tp_0_) / 2

    short_base = ltf_df['short_base_{}'.format(itv)].to_numpy()
    long_base = ltf_df['long_base_{}'.format(itv)].to_numpy()

    len_df = len(ltf_df)
    short_open_res = np.ones(len_df)
    long_open_res = np.ones(len_df)

    data_cols = ['open', 'high', 'low']
    open, high, low = [ltf_df[col_].to_numpy() for col_ in data_cols]

    short_a_touch_idx = pd.Series(np.where(high >= short_tp_0_, np.arange(len_df), np.nan)).rolling(touch_lbperiod,
                                                                                                    min_periods=1).max().to_numpy()  # min -> max
    short_b_touch_idx = pd.Series(np.where(low <= short_tp_1_, np.arange(len_df), np.nan)).rolling(touch_lbperiod,
                                                                                                   min_periods=1).max().to_numpy()  # min 으로 하면 a 이전에 b 있는건 모두 제외하게됨
    short_open_res *= (high >= short_base) & (short_base >= open) & (short_a_touch_idx < short_b_touch_idx)

    ltf_df['short_wave_1_{}'.format(itv)] = pd.Series(get_line(short_b_touch_idx, short_tp_1_)).ffill().to_numpy()  # ffill() 다음 to_numpy() 안하면 None 값
    ltf_df['short_wave_0_{}'.format(itv)] = pd.Series(get_line(short_a_touch_idx, short_tp_0_)).ffill().to_numpy()
    ltf_df['short_wave_point_{}'.format(itv)] = short_open_res.astype(bool)

    long_a_touch_idx = pd.Series(np.where(low <= long_tp_0_, np.arange(len_df), np.nan)).rolling(touch_lbperiod, min_periods=1).max().to_numpy()
    long_b_touch_idx = pd.Series(np.where(high >= long_tp_1_, np.arange(len_df), np.nan)).rolling(touch_lbperiod, min_periods=1).max().to_numpy()
    long_open_res *= (open >= long_base) & (long_base >= low) & (long_a_touch_idx < long_b_touch_idx)

    ltf_df['long_wave_1_{}'.format(itv)] = pd.Series(get_line(long_b_touch_idx, long_tp_1_)).ffill().to_numpy()
    ltf_df['long_wave_0_{}'.format(itv)] = pd.Series(get_line(long_a_touch_idx, long_tp_0_)).ffill().to_numpy()
    ltf_df['long_wave_point_{}'.format(itv)] = long_open_res.astype(bool)

    return ltf_df

def dc_line(ltf_df, htf_df, interval, dc_period=20):

    dc_upper = 'dc_upper_%s' % interval
    dc_lower = 'dc_lower_%s' % interval
    dc_base = 'dc_base_%s' % interval

    if interval not in  ['T', '1m']:
        htf_df[dc_upper], htf_df[dc_lower], htf_df[dc_base] = donchian_channel(htf_df, dc_period)
        joined_ltf_df = ltf_df.join(to_lower_tf_v2(ltf_df, htf_df, [i for i in range(-3, 0, 1)]), how='inner')
    else:
        joined_ltf_df = ltf_df.copy()
        joined_ltf_df[dc_upper], joined_ltf_df[dc_lower], joined_ltf_df[dc_base] = donchian_channel(ltf_df, dc_period)

    return joined_ltf_df


def dc_level(ltf_df, interval, dc_gap_multiple):

    dc_upper = 'dc_upper_%s' % interval
    dc_lower = 'dc_lower_%s' % interval
    dc_base = 'dc_base_%s' % interval

    dc_gap = 'dc_gap_%s' % interval
    dc_upper2 = 'dc_upper2_%s' % interval
    dc_lower2 = 'dc_lower2_%s' % interval
    dc_upper3 = 'dc_upper3_%s' % interval
    dc_lower3 = 'dc_lower3_%s' % interval

    # dc_width_ = 'dc_width_%s' % interval

    ltf_df[dc_gap] = (ltf_df[dc_upper] - ltf_df[dc_base]) * dc_gap_multiple

    ltf_df[dc_upper] = ltf_df[dc_base] + ltf_df[dc_gap]
    ltf_df[dc_lower] = ltf_df[dc_base] - ltf_df[dc_gap]

    # ltf_df[dc_width_] = (ltf_df[dc_upper] - ltf_df[dc_lower]) / ltf_df[dc_base]

    ltf_df[dc_upper2] = ltf_df[dc_base] + ltf_df[dc_gap] * 2
    ltf_df[dc_lower2] = ltf_df[dc_base] - ltf_df[dc_gap] * 2

    ltf_df[dc_upper3] = ltf_df[dc_base] + ltf_df[dc_gap] * 3
    ltf_df[dc_lower3] = ltf_df[dc_base] - ltf_df[dc_gap] * 3

    return ltf_df

def rs_channel(df, period=20, itv='T'):  # htf ohlc 는 itv 조사가 안되서 itv 변수 필요함
    if itv == 'T':
        rolled_data = df[['open', 'close']].rolling(period)
    else:
        rolled_data = df[['open_{}'.format(itv), 'close_{}'.format(itv)]].rolling(period)
    df['resi_{}'.format(itv)] = rolled_data.max().max(1)
    df['sup_{}'.format(itv)] = rolled_data.min().min(1)

    return

def rs_channel_v2(df, period=20, itv='T', type='TP'):
    if itv == 'T':
        rolled_data = df['close'].rolling(period)
    else:
        rolled_data = df['close_{}'.format(itv)].rolling(period)

    if type == 'TP':
        df['resi_{}{}'.format(itv, period)] = rolled_data.max()
        df['sup_{}{}'.format(itv, period)] = rolled_data.min()
    else:
        df['resi_out_{}{}'.format(itv, period)] = rolled_data.max()
        df['sup_out_{}{}'.format(itv, period)] = rolled_data.min()

    return

def bb_width(df, period, multiple):

    basis = df['close'].rolling(period).mean()
    # print(stdev(df, period))
    # quit()
    dev_ = multiple * stdev(df, period)
    upper = basis + dev_
    lower = basis - dev_
    bbw = 2 * dev_ / basis

    return upper, lower, bbw


def bb_width_v2(df, period, multiple, return_bbw=False):
    basis = df['close'].rolling(period).mean().to_numpy()
    dev_ = multiple * stdev(df, period)
    upper = basis + dev_
    lower = basis - dev_

    itv = pd.infer_freq(df.index)
    df['bb_upper_{}{}'.format(itv, period)] = upper
    df['bb_lower_{}{}'.format(itv, period)] = lower
    if return_bbw:
        df['bbw_{}{}'.format(itv, period)] = 2 * dev_ / basis

    return



def bb_line(ltf_df, htf_df, interval, period=20, multi=1):  # deprecated - join 을 따로 사용해도 될터인데.. (v2 만들면서 느낀점)

    bb_upper = 'bb_upper_%s' % interval
    bb_lower = 'bb_lower_%s' % interval

    if interval not in ['T', '1m']:
        htf_df[bb_upper], htf_df[bb_lower], _ = bb_width(htf_df, period, multi)
        joined_ltf_df = ltf_df.join(to_lower_tf_v2(ltf_df, htf_df, [i for i in range(-2, 0, 1)]), how='inner')
    else:
        joined_ltf_df = ltf_df.copy()
        joined_ltf_df[bb_upper], joined_ltf_df[bb_lower], _ = bb_width(ltf_df, period, multi)

    return joined_ltf_df

# def bb_line_v2(ltf_df, htf_df, period=20, multi=1):
#     if interval != '1m':
#         bb_width_v2(htf_df, period, multi)
#         ltf_df = ltf_df.join(to_lower_tf_v2(ltf_df, htf_df, [i for i in range(-2, 0, 1)]), how='inner')
#     else:
#         bb_width(ltf_df, period, multi)
#
#     return

def bb_level(ltf_df, interval, bb_gap_multiple):

    bb_upper = 'bb_upper_%s' % interval
    bb_lower = 'bb_lower_%s' % interval
    bb_base = 'bb_base_%s' % interval

    ltf_df[bb_base] = (ltf_df[bb_upper] + ltf_df[bb_lower]) / 2

    bb_gap = 'bb_gap_%s' % interval
    bb_upper2 = 'bb_upper2_%s' % interval
    bb_lower2 = 'bb_lower2_%s' % interval
    bb_upper3 = 'bb_upper3_%s' % interval
    bb_lower3 = 'bb_lower3_%s' % interval

    # bb_width_ = 'bb_width_%s' % interval

    ltf_df[bb_gap] = (ltf_df[bb_upper] - ltf_df[bb_base]) * bb_gap_multiple

    ltf_df[bb_upper] = ltf_df[bb_base] + ltf_df[bb_gap]
    ltf_df[bb_lower] = ltf_df[bb_base] - ltf_df[bb_gap]

    # ltf_df[bb_width_] = (ltf_df[bb_upper] - ltf_df[bb_lower]) / ltf_df[bb_base]

    ltf_df[bb_upper2] = ltf_df[bb_base] + ltf_df[bb_gap] * 2
    ltf_df[bb_lower2] = ltf_df[bb_base] - ltf_df[bb_gap] * 2

    ltf_df[bb_upper3] = ltf_df[bb_base] + ltf_df[bb_gap] * 3
    ltf_df[bb_lower3] = ltf_df[bb_base] - ltf_df[bb_gap] * 3

    return ltf_df

#       Todo - recursive -> not fixed      #
def fisher(df, period):
    hl2 = (df['high'] + df['low']) / 2
    high_ = hl2.rolling(period).max()
    low_ = hl2.rolling(period).min()

    # print(type(hl2))
    # print(high_)
    value = pd.Series(index=hl2.index)
    fish = pd.Series(index=hl2.index)
    value.iloc[0] = 0.0
    fish.iloc[0] = 0.0

    # print(value)

    for i in range(1, len(df)):
        value.iloc[i] = (0.66 * ((hl2.iloc[i] - low_.iloc[i]) / max(high_.iloc[i] - low_.iloc[i], .001) - .5)
                         + .67 * nz(value.iloc[i - 1]))
        # print(value.iloc[i])
        value.iloc[i] = round(value.iloc[i])
        # print(value.iloc[i])
        # print()
        fish.iloc[i] = .5 * np.log((1 + value.iloc[i]) / max(1 - value.iloc[i], .001)) + .5 * nz(fish.iloc[i - 1])

    fish.iloc[0] = np.nan

    return fish

def bbwp(bb_gap, st_gap, ma_period=10):

    bbwp_ = bb_gap / st_gap
    bbwp_ma = bbwp_.rolling(ma_period).mean()

    return bbwp_, bbwp_ma


#       Todo - recursive ?       #
def fisher_trend(df, column, tc_upper, tc_lower):
    fisher_cross_series = np.where((df[column].shift(1) < tc_upper) & (tc_upper < df[column]), 'CO', np.nan)
    fisher_cross_series = np.where((df[column].shift(1) > tc_lower) & (tc_lower > df[column]), 'CU',
                                   fisher_cross_series)
    fisher_trend_df = pd.DataFrame([np.nan] * len(df))

    start_i = -1
    while True:
        start_i += 1
        if start_i >= len(df):
            break
        # print(start_i)
        if not pd.isna(fisher_cross_series[start_i]):
            if fisher_cross_series[start_i] == 'CO':
                for j in range(start_i + 1, len(df)):
                    if fisher_cross_series[j] == 'CU':
                        fisher_trend_df.iloc[start_i: j] = 'Long'
                        start_i = j - 1
                        break
                    elif j == len(df) - 1:
                        fisher_trend_df.iloc[start_i:] = 'Long'
                        start_i = j - 1
            elif fisher_cross_series[start_i] == 'CU':
                for j in range(start_i + 1, len(df)):
                    if fisher_cross_series[j] == 'CO':
                        fisher_trend_df.iloc[start_i: j] = 'Short'
                        start_i = j - 1
                        break
                    elif j == len(df) - 1:
                        fisher_trend_df.iloc[start_i:] = 'Short'
                        start_i = j - 1

    return fisher_trend_df.values


#       Todo - recursive ?       #
def lucid_sar(df, af_initial=0.02, af_increment=0.02, af_maximum=0.2, return_uptrend=True):
    uptrend = pd.Series(True, index=df.index)
    new_trend = pd.Series(False, index=df.index)
    reversal_state = pd.Series(0, index=df.index)
    af = pd.Series(af_initial, index=df.index)

    ep = df['high'].copy()
    sar = df['low'].copy()

    for i in range(1, len(df)):
        # if not pd.isna(uptrend.iloc[i - 1]) and pd.isna(new_trend.iloc[i - 1]):
        if reversal_state.iloc[i] == 0:
            if uptrend.iloc[i - 1]:
                ep.iloc[i] = max(df['high'].iloc[i], ep.iloc[i - 1])
            else:
                ep.iloc[i] = min(df['low'].iloc[i], ep.iloc[i - 1])
            if new_trend.iloc[i - 1]:
                af.iloc[i] = af_initial
            else:
                if ep.iloc[i] != ep.iloc[i - 1]:
                    af.iloc[i] = min(af_maximum, af.iloc[i - 1] + af_increment)
                else:
                    af.iloc[i] = af.iloc[i - 1]
            sar.iloc[i] = sar.iloc[i - 1] + af.iloc[i] * (ep.iloc[i] - sar.iloc[i - 1])

            if uptrend.iloc[i - 1]:
                sar.iloc[i] = min(sar.iloc[i], df['low'].iloc[i - 1])
                # if not pd.isna(df['low'].iloc[i - 2]):
                if i >= 2:
                    sar.iloc[i] = min(sar.iloc[i], df['low'].iloc[i - 2])
                if sar.iloc[i] > df['low'].iloc[i]:
                    uptrend.iloc[i] = False
                    new_trend.iloc[i] = True
                    sar.iloc[i] = max(df['high'].iloc[i], ep.iloc[i - 1])
                    ep.iloc[i] = min(df['low'].iloc[i], df['low'].iloc[i - 1])
                    reversal_state.iloc[i] = 2
                else:
                    uptrend.iloc[i] = True
                    new_trend.iloc[i] = False

            else:
                sar.iloc[i] = max(sar.iloc[i], df['high'].iloc[i - 1])
                # if not pd.isna(df['high'].iloc[i - 2]):
                if i >= 2:
                    sar.iloc[i] = max(sar.iloc[i], df['high'].iloc[i - 2])
                if sar.iloc[i] < df['high'].iloc[i]:
                    uptrend.iloc[i] = True
                    new_trend.iloc[i] = True
                    sar.iloc[i] = min(df['low'].iloc[i],  ep.iloc[i - 1])
                    ep.iloc[i] = max(df['high'].iloc[i], df['high'].iloc[i - 1])
                    reversal_state.iloc[i] = 1
                else:
                    uptrend.iloc[i] = False
                    new_trend.iloc[i] = False

        else:
            if reversal_state.iloc[i] == 1:
                ep.iloc[i] = df['high'].iloc[i]
                if df['low'].iloc[i] < sar.iloc[i]:
                    sar.iloc[i] = ep.iloc[i]
                    ep.iloc[i] = df['low'].iloc[i]
                    reversal_state.iloc[i] = 2
                    uptrend.iloc[i] = False
            else:
                ep.iloc[i] = df['low'].iloc[i]
                if df['high'].iloc[i] > sar.iloc[i]:
                    sar.iloc[i] = ep.iloc[i]
                    ep.iloc[i] = df['high'].iloc[i]
                    reversal_state.iloc[i] = 1
                    uptrend.iloc[i] = True

    if return_uptrend:
        return sar, uptrend

    return sar

def lucid_sar_v2(df, af_initial=0.03, af_increment=0.02, af_maximum=0.2, return_uptrend=True):
    len_df = len(df)
    uptrend = np.ones(len_df)
    new_trend = np.zeros(len_df)
    reversal_state = np.zeros(len_df)
    af = np.full(len_df, af_initial)

    h = df['high'].to_numpy()
    l = df['low'].to_numpy()
    ep = h.copy()
    sar = l.copy()

    for i in range(1, len(df)):
        # if not pd.isna(uptrend.iloc[i - 1]) and pd.isna(new_trend.iloc[i - 1]):
        if reversal_state[i] == 0:
            # ------ ep set ------ #
            if uptrend[i - 1]:
                ep[i] = max(h[i], ep[i - 1])
            else:
                ep[i] = min(l[i], ep[i - 1])
            # ------ af set ------ #
            if new_trend[i - 1]:
                af[i] = af_initial
            else:
                if ep[i] != ep[i - 1]:
                    af[i] = min(af_maximum, af[i - 1] + af_increment)
                else:
                    af[i] = af[i - 1]

            # ------ sar set ------ #
            sar[i] = sar[i - 1] + af[i] * (ep[i] - sar[i - 1])   # recursive
            if uptrend[i - 1]:
                sar[i] = min(sar[i], l[i - 1])
                # if not pd.isna(l[i - 2]):
                if i >= 2:
                    sar[i] = min(sar[i], l[i - 2])
                if sar[i] > l[i]:
                    uptrend[i] = 0
                    new_trend[i] = 1
                    sar[i] = max(h[i], ep[i - 1])
                    ep[i] = min(l[i], l[i - 1])
                    reversal_state[i] = 2
                else:
                    uptrend[i] = 1
                    new_trend[i] = 0

            else:
                sar[i] = max(sar[i], h[i - 1])
                # if not pd.isna(h[i - 2]):
                if i >= 2:
                    sar[i] = max(sar[i], h[i - 2])
                if sar[i] < h[i]:
                    uptrend[i] = 1
                    new_trend[i] = 1
                    sar[i] = min(l[i],  ep[i - 1])
                    ep[i] = max(h[i], h[i - 1])
                    reversal_state[i] = 1
                else:
                    uptrend[i] = 0
                    new_trend[i] = 0

        else:
            if reversal_state[i] == 1:
                ep[i] = h[i]
                if l[i] < sar[i]:
                    sar[i] = ep[i]
                    ep[i] = l[i]
                    reversal_state[i] = 2
                    uptrend[i] = 0
            else:
                ep[i] = l[i]
                if h[i] > sar[i]:
                    sar[i] = ep[i]
                    ep[i] = h[i]
                    reversal_state[i] = 1
                    uptrend[i] = 1

    itv = pd.infer_freq(df.index)
    df['sar_{}'.format(itv)] = sar
    if return_uptrend:
        df['sar_uptrend_{}'.format(itv)] = uptrend
    return


def cmo(df, period=9):

    df['closegap_cunsum'] = (df['close'] - df['close'].shift(1)).cumsum()
    df['closegap_abs_cumsum'] = abs(df['close'] - df['close'].shift(1)).cumsum()
    # print(df)

    df['CMO'] = (df['closegap_cunsum'] - df['closegap_cunsum'].shift(period)) / (
            df['closegap_abs_cumsum'] - df['closegap_abs_cumsum'].shift(period)) * 100

    del df['closegap_cunsum']
    del df['closegap_abs_cumsum']

    return df['CMO']


#       Todo - recursive       #
def rma(series, period):

    alpha = 1 / period
    rma_ = pd.Series(index=series.index, dtype=np.float64)
    avg_ = series.rolling(period).mean()
    
    rma_.iloc[0] = 0
    for i in range(1, len(series)):
        if np.isnan(rma_.iloc[i - 1]):
            rma_.iloc[i] = avg_.iloc[i]
        else:
            rma_.iloc[i] = series.iloc[i] * alpha + (1 - alpha) * nz(rma_.iloc[i - 1])  # recursive
            
    # rma_.iloc[0] = avg_.iloc[0]
    # rma_ = series * alpha + (1 - alpha) * rma_.shift(1).apply(nz)

    return rma_


#       Todo - recursive       #
def rsi(ohlcv_df, period=14):

    #       rma 에 rolling, index 가 필요해서 series 로 남겨두고 del 하는 것임     #
    ohlcv_df['up'] = np.where(ohlcv_df['close'].diff(1) > 0, ohlcv_df['close'].diff(1), 0)
    ohlcv_df['down'] = np.where(ohlcv_df['close'].diff(1) < 0, ohlcv_df['close'].diff(1) * (-1), 0)

    rs = rma(ohlcv_df['up'], period) / rma(ohlcv_df['down'], period)
    rsi_ = 100 - 100 / (1 + rs)

    del ohlcv_df['up']
    del ohlcv_df['down']

    return rsi_


def cci(df, period=20):

    hlc3 = (df['high'] + df['low'] + df['close']) / 3
    ma_ = hlc3.rolling(period).mean()
    cci_ = (hlc3 - ma_) / (0.015 * dev(hlc3, period))

    return cci_


def stoch(ohlcv_df, period_sto=13, k=3, d=3):

    hh = ohlcv_df['high'].rolling(period_sto).max()
    ll = ohlcv_df['low'].rolling(period_sto).min()
    stoch = (ohlcv_df['close'] - ll) / (hh - ll) * 100

    stoch_k = stoch.rolling(k).mean()
    stock_d = stoch_k.rolling(d).mean()

    return stock_d

def stoch_v2(ohlcv_df, period_sto=13, k=3, d=3):

    hh = ohlcv_df['high'].rolling(period_sto).max()
    ll = ohlcv_df['low'].rolling(period_sto).min()
    stoch = (ohlcv_df['close'] - ll) / (hh - ll) * 100

    stoch_k = stoch.rolling(k).mean()
    stock_d = stoch_k.rolling(d).mean()

    return stock_d

def stochrsi(ohlcv_df, period_rsi=14, period_sto=14, k=3, d=3):
    rsi(ohlcv_df, period_rsi)
    ohlcv_df['H_RSI'] = ohlcv_df.RSI.rolling(period_sto).max()
    ohlcv_df['L_RSI'] = ohlcv_df.RSI.rolling(period_sto).min()
    ohlcv_df['StochRSI'] = (ohlcv_df.RSI - ohlcv_df.L_RSI) / (ohlcv_df.H_RSI - ohlcv_df.L_RSI) * 100

    ohlcv_df['StochRSI_K'] = ohlcv_df.StochRSI.rolling(k).mean()
    ohlcv_df['StochRSI_D'] = ohlcv_df.StochRSI_K.rolling(d).mean()

    del ohlcv_df['H_RSI']
    del ohlcv_df['L_RSI']

    return ohlcv_df['StochRSI_D']


#       Todo - recursive       #
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


# def macd(df, short=9, long=19, signal=9):
#     macd = df['close'].ewm(span=short, min_periods=short - 1, adjust=False).mean() - \
#                  df['close'].ewm(span=long, min_periods=long - 1, adjust=False).mean()
#     macd_signal = macd.ewm(span=signal, min_periods=signal - 1, adjust=False).mean()
#     macd_hist = macd - macd_signal
#
#     return macd_hist


#       Todo - recursive       #
def macd(df, short=9, long=19, signal=9):

    macd = ema(df['close'], short) - ema(df['close'], long)
    macd_signal = ema(macd, signal)
    macd_hist = macd - macd_signal

    return macd_hist


#       Todo - recursive       #
def ema_ribbon(df, ema_1=5, ema_2=8, ema_3=13):

    # ema1 = df['close'].ewm(span=ema_1, min_periods=ema_1 - 1, adjust=False).mean()
    # ema2 = df['close'].ewm(span=ema_2, min_periods=ema_2 - 1, adjust=False).mean()
    # ema3 = df['close'].ewm(span=ema_3, min_periods=ema_3 - 1, adjust=False).mean()
    ema1 = ema(df['close'], ema_1)
    ema2 = ema(df['close'], ema_2)
    ema3 = ema(df['close'], ema_3)

    return ema1, ema2, ema3


#       Todo - recursive       #
def ema_cross(df, ema_1=30, ema_2=60):

    # df['EMA_1'] = df['close'].ewm(span=ema_1, min_periods=ema_1 - 1, adjust=False).mean()
    # df['EMA_2'] = df['close'].ewm(span=ema_2, min_periods=ema_2 - 1, adjust=False).mean()
    df['EMA_1'] = ema(df['close'], ema_1)
    df['EMA_2'] = ema(df['close'], ema_2)

    return


#       Todo - recursive       #
def atr(df, period):
    tr = pd.Series(index=df.index)
    tr.iloc[0] = df['high'].iloc[0] - df['low'].iloc[0]

    # for i in range(1, len(df)):
    #     if pd.isna(df['high'].iloc[i - 1]):
    #         tr.iloc[i] = df['high'].iloc[i] - df['low'].iloc[i]
    #     else:
    #         tr.iloc[i] = max(
    #             max(df['high'].iloc[i] - df['low'].iloc[i], abs(df['high'].iloc[i] - df['close'].iloc[i - 1])),
    #             abs(df['low'].iloc[i] - df['close'].iloc[i - 1]))

    tr = np.maximum(np.maximum(df['high'] - df['low'], abs(df['high'] - df['close'].shift(1))),
        abs(df['low'] - df['close'].shift(1)))
    # print(tr.tail(10))
    # quit()
    atr = rma(tr, period)

    return atr


#       Todo - recursive       #
def supertrend(df, period, multiplier, cal_st=False):

    hl2 = (df['high'] + df['low']) / 2
    # print(hl2)
    # print(atr(df, period))
    # quit()
    atr_down = hl2 - (multiplier * atr(df, period))
    # max_atr_down = np.maximum(atr_down, atr_down.shift(1))
    # atr_down_v2 = np.where(df['close'].shift(1) > atr_down.shift(1), max_atr_down, atr_down)
    # print(atr_down_v2[-20:])

    #       오차를 없애기 위해서, for loop 사용함       #
    for i in range(1, len(df)):
        if df['close'].iloc[i - 1] > atr_down[i - 1]:
            atr_down[i] = max(atr_down[i], atr_down[i - 1])  # --> 여기도 누적형, 만든 atr_down[i] 가 [i + 1] 에 영향을 줌
    # print(atr_down[-20:])
    # print(atr_down[-20:].values == atr_down_v2[-20:])
    # quit()

    atr_up = hl2 + (multiplier * atr(df, period))
    # atr_up = np.where(df['close'].shift(1) < atr_up.shift(1), min(atr_up, atr_up.shift(1)), atr_up)
    for i in range(1, len(df)):
        if df['close'].iloc[i - 1] < atr_up[i - 1]:
            atr_up[i] = min(atr_up[i], atr_up[i - 1])

    atr_trend = pd.Series(index=df.index)
    atr_trend.iloc[0] = 0

    #       Todo        #
    #        이부분을 주석 처리한 이유가 있음
    # atr_trend = np.where(atr_trend.shift(1) == np.nan, atr_trend, atr_trend)
    for i in range(1, len(df)):
        if df['close'].iloc[i] > atr_up[i - 1]:
            atr_trend.iloc[i] = 1
        else:
            if df['close'].iloc[i] < atr_down[i - 1]:
                atr_trend.iloc[i] = -1
            else:
                atr_trend.iloc[i] = atr_trend.iloc[i - 1]

    if not cal_st:
        return atr_up, atr_down, atr_trend
    else:
        # st = np.where(atr_trend == -1, pd.Series(atr_up), np.nan)
        # st = np.where(atr_trend == 1, pd.Series(atr_down), st)
        # st = np.where(atr_trend == -1, atr_up, atr_down)
        return np.where(atr_trend == -1, atr_up, atr_down)


def st_price_line(ltf_df, htf_df, interval):

    ha_htf_df = heikinashi(htf_df)

    st1_up, st2_up, st3_up = 'ST1_Up_%s' % interval, 'ST2_Up_%s' % interval, 'ST3_Up_%s' % interval
    st1_down, st2_down, st3_down = 'ST1_Down_%s' % interval, 'ST2_Down_%s' % interval, 'ST3_Down_%s' % interval
    st1_trend, st2_trend, st3_trend = 'ST1_Trend_%s' % interval, 'ST2_Trend_%s' % interval, 'ST3_Trend_%s' % interval

    htf_df[st1_up], htf_df[st1_down], htf_df[st1_trend] = supertrend(htf_df, 10, 2)
    htf_df[st2_up], htf_df[st2_down], htf_df[st2_trend] = supertrend(ha_htf_df, 7, 2)
    htf_df[st3_up], htf_df[st3_down], htf_df[st3_trend] = supertrend(ha_htf_df, 7, 2.5)
    # print(df.head(20))
    # quit()

    # startTime = time.time()

    joined_ltf_df = ltf_df.join(to_lower_tf_v2(ltf_df, htf_df, [i for i in range(-9, 0, 1)]), how='inner')

    return joined_ltf_df

def st_level(ltf_df, interval, st_gap_multiple):

    st1_up, st2_up, st3_up = 'ST1_Up_%s' % interval, 'ST2_Up_%s' % interval, 'ST3_Up_%s' % interval
    st1_down, st2_down, st3_down = 'ST1_Down_%s' % interval, 'ST2_Down_%s' % interval, 'ST3_Down_%s' % interval

    min_upper, max_lower = 'min_upper_%s' % interval, 'max_lower_%s' % interval
    st_base = 'st_base_%s' % interval
    st_gap = 'st_gap_%s' % interval
    st_upper, st_lower = 'st_upper_%s' % interval, 'st_lower_%s' % interval
    st_upper2, st_lower2 = 'st_upper2_%s' % interval, 'st_lower2_%s' % interval
    st_upper3, st_lower3 = 'st_upper3_%s' % interval, 'st_lower3_%s' % interval

    ltf_df[min_upper] = np.min(ltf_df[[st1_up, st2_up, st3_up]], axis=1)
    ltf_df[max_lower] = np.max(ltf_df[[st1_down, st2_down, st3_down]], axis=1)

    ltf_df[st_base] = (ltf_df[min_upper] + ltf_df[max_lower]) / 2
    ltf_df[st_gap] = (ltf_df[min_upper] - ltf_df[st_base]) * st_gap_multiple

    # --------------- levels --------------- #
    ltf_df[st_upper] = ltf_df[st_base] + ltf_df[st_gap]
    ltf_df[st_lower] = ltf_df[st_base] - ltf_df[st_gap]

    ltf_df[st_upper2] = ltf_df[st_base] + ltf_df[st_gap] * 2
    ltf_df[st_lower2] = ltf_df[st_base] - ltf_df[st_gap] * 2

    ltf_df[st_upper3] = ltf_df[st_base] + ltf_df[st_gap] * 3
    ltf_df[st_lower3] = ltf_df[st_base] - ltf_df[st_gap] * 3

    return ltf_df

# def mmh_st(df, mp1, mp2, pd1=10, pd2=10):   # makemoney_hybrid
#       Todo - recursive       #
def mmh_st(df, mp1, pd1=10):   # makemoney_hybrid

    hlc3 = (df['high'] + df['low'] + df['close']) / 3

    up1 = hlc3 - (mp1 * atr(df, pd1))
    # up2 = hlc3 - (mp2 * atr(df, pd2))

    down1 = hlc3 + (mp1 * atr(df, pd1))
    # down2 = hlc3 + (mp2 * atr(df, pd2))

    # 이전 결과를 recursively 하게 사용하기 때문에 for loop 가 맞을 것임
    for i in range(1, len(df)):
        if df['close'].iloc[i - 1] > up1[i - 1]:
            up1[i] = max(up1[i], up1[i - 1])
        # if df['close'].iloc[i - 1] > up2[i - 1]:
        #     up2[i] = max(up2[i], up2[i - 1])

        if df['close'].iloc[i - 1] < down1[i - 1]:
            down1[i] = min(down1[i], down1[i - 1])
        # if df['close'].iloc[i - 1] < down2[i - 1]:
        #     down2[i] = min(down2[i], down2[i - 1])

    trend1 = pd.Series(index=df.index)
    trend1.iloc[0] = 1
    for i in range(1, len(df)):
        if df['close'].iloc[i - 1] > down1[i - 1]:
            trend1.iloc[i] = 1
        else:
            if df['close'].iloc[i] < up1[i - 1]:
                trend1.iloc[i] = -1
            else:
                trend1.iloc[i] = trend1.iloc[i - 1]

    tls = np.where(trend1 == 1, up1, down1)

    return tls


def ichimoku(ohlc, tenkan_period=9, kijun_period=26, senkou_period=52, chikou_period=1):

    #       Conversion Line     #
    tenkan_sen = pd.Series(
        (
                ohlc["high"].rolling(window=tenkan_period).max()
                + ohlc["low"].rolling(window=tenkan_period).min()
        )
        / 2,
        name="TENKAN",
    )

    #       Base Line       #
    kijun_sen = pd.Series(
        (
                ohlc["high"].rolling(window=kijun_period).max()
                + ohlc["low"].rolling(window=kijun_period).min()
        )
        / 2,
        name="KIJUN",
    )

    #       Leading Span        #
    senkou_span_a = pd.Series(
        ((tenkan_sen + kijun_sen) / 2), name="senkou_span_a"
    )
    senkou_span_b = pd.Series(
        (
                (
                        ohlc["high"].rolling(window=senkou_period).max()
                        + ohlc["low"].rolling(window=senkou_period).min()
                )
                / 2
        ),
        name="SENKOU",
    )

    # chikou_span = pd.Series(
    #     ohlc["close"].shift(chikou_period).rolling(window=chikou_period).mean(),
    #     name="CHIKOU",
    # )

    return senkou_span_a.shift(chikou_period - 1), senkou_span_b.shift(chikou_period - 1)


def ad(df):
    ad_value = (2 * df['close'] - df['low'] - df['high']) / (df['high'] - df['low']) * df['volume']
    ad_ = np.where((df['close'] == df['high']) & (df['close'] == df['low']) | (df['high'] == df['low']), 0, ad_value)
    ad_ = ad_.cumsum()

    return ad_


# def lt_trend(df, lambda_bear=0.06, lambda_bull=0.08):
#     df['LT_Trend'] = np.NaN
#     t_zero = 0
#     trend_state = None
#     for i in range(1, len(df)):
#         pmax = df['close'].iloc[t_zero:i].max()
#         pmin = df['close'].iloc[t_zero:i].min()
#         delta_bear = (pmax - df['close'].iloc[i]) / pmax
#         delta_bull = (df['close'].iloc[i] - pmin) / pmin
#         # print(pmax, pmin, delta_bear, delta_bull)
#
#         if delta_bear > lambda_bear and trend_state != 'Bear':
#             t_peak = df['close'].iloc[t_zero:i].idxmax()
#             t_peak = df.index.to_list().index(t_peak)
#             df['LT_Trend'].iloc[t_zero + 1:t_peak + 1] = 'Bull'
#             t_zero = t_peak
#             trend_state = 'Bear'
#
#         elif delta_bull > lambda_bull and trend_state != 'Bull':
#             t_trough = df['close'].iloc[t_zero:i].idxmin()
#             t_trough = df.index.to_list().index(t_trough)
#             df['LT_Trend'].iloc[t_zero + 1:t_trough + 1] = 'Bear'
#             t_zero = t_trough
#             trend_state = 'Bull'
#
#         if i == len(df) - 1:
#             if pd.isnull(df['LT_Trend'].iloc[i]):
#                 back_i = i
#                 while True:
#                     back_i -= 1
#                     if not pd.isnull(df['LT_Trend'].iloc[back_i]):
#                         if df['LT_Trend'].iloc[back_i] == 'Bull':
#                             df['LT_Trend'].iloc[back_i + 1:] = 'Bear'
#                         else:
#                             df['LT_Trend'].iloc[back_i + 1:] = 'Bull'
#                         break
#
#             df['LT_Trend'].iloc[0] = df['LT_Trend'].iloc[1]
#
#
# from scipy.ndimage.filters import gaussian_filter1d
#
#
# def support_line(df, sigma=10):
#     #           Gaussian Curve          #
#     df['Gaussian_close'] = gaussian_filter1d(df['close'], sigma=sigma)
#     df['Gaussian_close_Trend'] = np.where(df['Gaussian_close'] > df['Gaussian_close'].shift(1), 1, 0)
#
#     df['Support_Line'] = np.NaN
#     i = 0
#     while i < len(df):
#         # print('i :', i)
#         #           Find Red           #
#         if not df['Gaussian_close_Trend'].iloc[i]:
#             #     Find end of the Red       #
#             for j in range(i + 1, len(df)):
#                 if df['Gaussian_close_Trend'].iloc[j]:
#                     min_price = df['low'].iloc[i:j].min()
#                     # print('i, j, min_price :', i, j, min_price)
#                     #       Find Red to Green          #
#                     for k in range(j + 1, len(df)):
#                         if df['Gaussian_close_Trend'].iloc[k] == 1 and df['Gaussian_close_Trend'].iloc[k - 1] == 0:
#                             df['Support_Line'].iloc[j:k] = min_price
#                             # print('One S.P Drawing Done!')
#                             i = j
#                             break
#                         else:
#                             if k == len(df) - 1:
#                                 df['Support_Line'].iloc[j:] = min_price
#                                 i = len(df)
#                                 break
#                     break
#         else:
#             i += 1
#
#     return
