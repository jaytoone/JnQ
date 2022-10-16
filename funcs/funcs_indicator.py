import numpy as np
import pandas as pd
from funcs.funcs_trader import to_lower_tf_v2, to_lower_tf_v3, intmin, ffill, bfill, to_itvnum, to_htf
import talib


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

def get_wave_time_ratio(res_df, wave_itv1, wave_period1):

  wave_cu_prime_idx_fill_ = res_df['wave_cu_prime_idx_fill_{}{}'.format(wave_itv1, wave_period1)].shift(1).to_numpy()
  wave_co_prime_idx_fill_ = res_df['wave_co_prime_idx_fill_{}{}'.format(wave_itv1, wave_period1)].shift(1).to_numpy()

  wave_cu_post_idx_fill_ = res_df['wave_cu_post_idx_fill_{}{}'.format(wave_itv1, wave_period1)].shift(1).to_numpy()
  wave_co_post_idx_fill_ = res_df['wave_co_post_idx_fill_{}{}'.format(wave_itv1, wave_period1)].shift(1).to_numpy()

  wave_cu_idx_ = res_df['wave_cu_idx_{}{}'.format(wave_itv1, wave_period1)].to_numpy()
  wave_co_idx_ = res_df['wave_co_idx_{}{}'.format(wave_itv1, wave_period1)].to_numpy()

  res_df['short_wave_time_ratio_{}{}'.format(wave_itv1, wave_period1)] = (wave_co_post_idx_fill_ - wave_cu_prime_idx_fill_) / (wave_cu_idx_- wave_co_post_idx_fill_)
  res_df['long_wave_time_ratio_{}{}'.format(wave_itv1, wave_period1)] = (wave_cu_post_idx_fill_ - wave_co_prime_idx_fill_) / (wave_co_idx_- wave_cu_post_idx_fill_)

  return res_df

def add_roll_idx(res_df, valid_prime_idx, roll_idx_arr, data_col, roll_hl_cnt):

    # data = res_df[data_col].to_numpy()
    len_res_df = len(res_df)
    roll_cols = [data_col + '_-{}'.format(cnt_ + 1) for cnt_ in reversed(range(roll_hl_cnt))]

    roll_data = pd.DataFrame(index=res_df.index, data=np.full((len_res_df, roll_hl_cnt), np.nan))
    roll_data.iloc[valid_prime_idx[roll_hl_cnt - 1:], :] = roll_idx_arr  # 제한된 idx 를 제외한 row 에 roll_hl 입력

    res_df[roll_cols] = roll_data.ffill()

    return res_df


def tc_cci_v2(t_df, wave_period):  # high_confirmation
    t_df = cci_v2(t_df, wave_period)
    itv = pd.infer_freq(t_df.index)

    cci_ = t_df['cci_{}{}'.format(itv, wave_period)].to_numpy()
    b1_cci_ = t_df['cci_{}{}'.format(itv, wave_period)].shift(1).to_numpy()

    baseline = 0
    band_width = 100
    upper_band = baseline + band_width
    lower_band = baseline - band_width

    data_cols = ['open', 'high', 'low', 'close']
    ohlc_list = [t_df[col_].to_numpy() for col_ in data_cols]
    open, high, low, close = ohlc_list

    t_df['short_tc_{}{}'.format(itv, wave_period)] = (b1_cci_ < lower_band) & (lower_band < cci_)
    t_df['long_tc_{}{}'.format(itv, wave_period)] = (b1_cci_ > upper_band) & (upper_band > cci_)

    return t_df

def tc_cci(t_df, wave_period):
    t_df = cci_v2(t_df, wave_period)
    itv = pd.infer_freq(t_df.index)

    cci_ = t_df['cci_{}{}'.format(itv, wave_period)].to_numpy()
    b1_cci_ = t_df['cci_{}{}'.format(itv, wave_period)].shift(1).to_numpy()

    baseline = 0
    band_width = 100
    upper_band = baseline + band_width
    lower_band = baseline - band_width

    data_cols = ['open', 'high', 'low', 'close']
    ohlc_list = [t_df[col_].to_numpy() for col_ in data_cols]
    open, high, low, close = ohlc_list

    t_df['short_tc_{}{}'.format(itv, wave_period)] = (b1_cci_ > lower_band) & (lower_band > cci_)
    t_df['long_tc_{}{}'.format(itv, wave_period)] = (b1_cci_ < upper_band) & (upper_band < cci_)

    return t_df

def tc_dc_base(t_df, dc_period):
    t_df = donchian_channel_v4(t_df, dc_period)
    itv = pd.infer_freq(t_df.index)

    dc_base_ = t_df['dc_base_{}{}'.format(itv, dc_period)].to_numpy()

    close = t_df['close'].to_numpy()
    b1_close = t_df['close'].shift(1).to_numpy()

    t_df['short_tc_{}{}'.format(itv, dc_period)] = (b1_close > dc_base_) & (dc_base_ > close)
    t_df['long_tc_{}{}'.format(itv, dc_period)] = (b1_close < dc_base_) & (dc_base_ < close)

    return t_df

def ma(df, period=30):
    itv = pd.infer_freq(df.index)
    close = df['close'].to_numpy()

    df['ma_{}{}'.format(itv, period)] = talib.MA(close, timeperiod=period)

    return df

def macd_hist(df, short=5, long=35, signal=5):
    itv = pd.infer_freq(df.index)
    close = df['close'].to_numpy()
    macd = talib.MA(close, timeperiod=short) - talib.MA(close, timeperiod=long)
    macd_signal = talib.MA(macd, timeperiod=signal)

    df['macd_{}{}{}'.format(itv, short, long)] = macd
    df['macd_hist_{}{}{}{}'.format(itv, short, long, signal)] = macd - macd_signal

    return df

def stoch_v2(df, fastk_period=13, slowk_period=3, slowd_period=3):
    itv = pd.infer_freq(df.index)
    high, low, close = [df[col_].to_numpy() for col_ in ['high', 'low', 'close']]
    slowk, slowd = talib.STOCH(high, low, close, fastk_period=fastk_period, slowk_period=slowk_period,
                               slowk_matype=0, slowd_period=slowd_period, slowd_matype=0)

    df['stoch_{}{}{}{}'.format(itv, fastk_period, slowk_period, slowd_period)] = slowd

    return df

def cci_v2(df, period=20, smooth=None):
    itv = pd.infer_freq(df.index)
    high, low, close = [df[col_].to_numpy() for col_ in ['high', 'low', 'close']]

    if smooth is None:
        df['cci_{}{}'.format(itv, period)] = talib.CCI(high, low, close, timeperiod=period)
    else:
        df['cci_{}{}'.format(itv, period)] = talib.MA(talib.CCI(high, low, close, timeperiod=period), timeperiod=smooth)

    return df


def get_wave_length(res_df, valid_co_prime_idx, valid_cu_prime_idx, roll_co_idx_arr, roll_cu_idx_arr, wave_itv, wave_period, roll_hl_cnt=3):
    res_df['short_wave_length_{}{}'.format(wave_itv, wave_period)] = np.nan
    res_df['long_wave_length_{}{}'.format(wave_itv, wave_period)] = np.nan
    res_df['short_wave_length_{}{}'.format(wave_itv, wave_period)].iloc[valid_cu_prime_idx[roll_hl_cnt - 1:]] = roll_cu_idx_arr[:,
                                                                                                                -1] - roll_cu_idx_arr[:, -2]
    res_df['long_wave_length_{}{}'.format(wave_itv, wave_period)].iloc[valid_co_prime_idx[roll_hl_cnt - 1:]] = roll_co_idx_arr[:,
                                                                                                               -1] - roll_co_idx_arr[:, -2]
    res_df['short_wave_length_fill_{}{}'.format(wave_itv, wave_period)] = res_df['short_wave_length_{}{}'.format(wave_itv, wave_period)].ffill()
    res_df['long_wave_length_fill_{}{}'.format(wave_itv, wave_period)] = res_df['long_wave_length_{}{}'.format(wave_itv, wave_period)].ffill()

    return res_df

def wave_range_ratio_v4_3(res_df, wave_itv, wave_period, roll_hl_cnt=4):

    wave_high_fill_ = res_df['wave_high_fill_{}{}'.format(wave_itv, wave_period)].to_numpy()
    wave_low_fill_ = res_df['wave_low_fill_{}{}'.format(wave_itv, wave_period)].to_numpy()

    roll_highs = [res_df['wave_high_fill_{}{}_-{}'.format(wave_itv, wave_period, cnt_ + 1)].to_numpy() for cnt_ in reversed(range(roll_hl_cnt))]
    roll_lows = [res_df['wave_low_fill_{}{}_-{}'.format(wave_itv, wave_period, cnt_ + 1)].to_numpy() for cnt_ in reversed(range(roll_hl_cnt))]

    cu_wave0_range = roll_highs[-2] - roll_lows[-2]
    cu_wave1_range = roll_highs[-1] - roll_lows[-2]   # cu's roll_high_[:, -1] = prev_high & cu's roll_low_[:, -1] = current_low
    cu_wave2_range = roll_highs[-1] - wave_low_fill_     # for short, cu

    co_wave0_range = roll_highs[-2] - roll_lows[-2]   # co's roll_low_[:, -1] = prev_low & co's roll_high_[:, -1] = current_high
    co_wave1_range = roll_highs[-2] - roll_lows[-1]   # co's roll_low_[:, -1] = prev_low & co's roll_high_[:, -1] = current_high
    co_wave2_range = wave_high_fill_ - roll_lows[-1]     # for long, co

    wave3_range = wave_high_fill_ - wave_low_fill_

    res_df['cu_wrr_10_{}{}'.format(wave_itv, wave_period)] = cu_wave1_range / cu_wave0_range
    res_df['cu_wrr_21_{}{}'.format(wave_itv, wave_period)] = cu_wave2_range / cu_wave1_range
    res_df['cu_wrr_32_{}{}'.format(wave_itv, wave_period)] = wave3_range / cu_wave2_range

    res_df['co_wrr_10_{}{}'.format(wave_itv, wave_period)] = co_wave1_range / co_wave0_range
    res_df['co_wrr_21_{}{}'.format(wave_itv, wave_period)] = co_wave2_range / co_wave1_range
    res_df['co_wrr_32_{}{}'.format(wave_itv, wave_period)] = wave3_range / co_wave2_range

    return res_df

def wave_range_ratio_v4_2(res_df, wave_itv, wave_period, roll_hl_cnt=3):

    wave_high_fill_ = res_df['wave_high_fill_{}{}'.format(wave_itv, wave_period)].to_numpy()
    wave_low_fill_ = res_df['wave_low_fill_{}{}'.format(wave_itv, wave_period)].to_numpy()

    roll_highs = [res_df['wave_high_fill_{}{}_-{}'.format(wave_itv, wave_period, cnt_ + 1)].to_numpy() for cnt_ in reversed(range(roll_hl_cnt))]
    roll_lows = [res_df['wave_low_fill_{}{}_-{}'.format(wave_itv, wave_period, cnt_ + 1)].to_numpy() for cnt_ in reversed(range(roll_hl_cnt))]

    cu_wave1_range = roll_highs[-1] - roll_lows[-2]   # cu's roll_high_[:, -1] = prev_high & cu's roll_low_[:, -1] = current_low
    cu_wave2_range = roll_highs[-1] - wave_low_fill_     # for short, cu
    co_wave1_range = roll_highs[-2] - roll_lows[-1]   # co's roll_low_[:, -1] = prev_low & co's roll_high_[:, -1] = current_high
    co_wave2_range = wave_high_fill_ - roll_lows[-1]     # for long, co
    wave3_range = wave_high_fill_ - wave_low_fill_

    res_df['cu_wrr_21_{}{}'.format(wave_itv, wave_period)] = cu_wave2_range / cu_wave1_range
    res_df['cu_wrr_32_{}{}'.format(wave_itv, wave_period)] = wave3_range / cu_wave2_range

    res_df['co_wrr_21_{}{}'.format(wave_itv, wave_period)] = co_wave2_range / co_wave1_range
    res_df['co_wrr_32_{}{}'.format(wave_itv, wave_period)] = wave3_range / co_wave2_range

    return res_df


def get_terms_info_v4(cu_idx, co_idx, len_df, len_df_range):

    cu_fill_idx = fill_arr(cu_idx)
    co_fill_idx = fill_arr(co_idx)

    notnan_cu_bool = ~np.isnan(cu_idx)
    notnan_co_bool = ~np.isnan(co_idx)

    valid_cu_bool = notnan_cu_bool * ~np.isnan(co_fill_idx)  # high_terms 를 위해 pair 되는 fill & idx 의 nan 제거
    valid_co_bool = notnan_co_bool * ~np.isnan(cu_fill_idx)

    # ------ 생략된 idx 에 대한 prime_idx 탐색 ------ #
    high_bool = cu_fill_idx < co_fill_idx  # 이렇게 해야 high_terms[:, 1] 이 cu_idx 가 나옴
    low_bool = co_fill_idx < cu_fill_idx

    high_terms_vec = get_index_bybool(high_bool, len_df_range)
    low_terms_vec = get_index_bybool(low_bool, len_df_range)  # -> low_terms

    high_terms_list = using_clump(high_terms_vec)
    low_terms_list = using_clump(low_terms_vec)

    valid_co_prime_idx = np.array([terms.min() for terms in high_terms_list])
    valid_cu_prime_idx = np.array([terms.min() for terms in low_terms_list])

    # arrays for indice must be of integer (or boolean) type --> more rows required
    assert len(valid_co_prime_idx) > 1 and len(valid_cu_prime_idx) > 1, "len(valid_co_prime_idx) > 1 and len(valid_cu_prime_idx) > 1"

    # valid_co_post_idx = np.array([terms.max() for terms in high_terms_list])   # 이곳은 cross_idx 가 아님, 단지 chunknized 된 filled_idx 일뿐
    # valid_cu_post_idx = np.array([terms.max() for terms in low_terms_list])

    cu_prime_idx = np.full(len_df, np.nan)
    co_prime_idx = np.full(len_df, np.nan)

    cu_prime_idx[valid_cu_prime_idx] = valid_cu_prime_idx
    co_prime_idx[valid_co_prime_idx] = valid_co_prime_idx

    cu_prime_fill_idx = fill_arr(cu_prime_idx)
    co_prime_fill_idx = fill_arr(co_prime_idx)

    # cu_post_idx = np.full(len_df, np.nan)  # --> Todo, unavailable : not cross_idx
    # co_post_idx = np.full(len_df, np.nan)

    # cu_post_idx[valid_cu_post_idx] = valid_cu_post_idx
    # co_post_idx[valid_co_post_idx] = valid_co_post_idx

    # cu_post_fill_idx = fill_arr(cu_post_idx)
    # co_post_fill_idx = fill_arr(co_post_idx)

    # ------------------------------------ #
    valid_cu_bool *= ~np.isnan(co_prime_fill_idx)
    valid_co_bool *= ~np.isnan(cu_prime_fill_idx)

    return cu_fill_idx, co_fill_idx, cu_prime_idx, co_prime_idx, cu_prime_fill_idx, co_prime_fill_idx, valid_cu_bool, valid_co_bool
    # return cu_fill_idx, co_fill_idx, cu_prime_idx, co_prime_idx, cu_prime_fill_idx, co_prime_fill_idx, \
    #   cu_post_idx, co_post_idx, cu_post_fill_idx, co_post_fill_idx, valid_cu_bool, valid_co_bool


def get_roll_wave_data_v2(res_df, valid_prime_idx, roll_idx_arr, data_col, roll_hl_cnt):

    data = res_df[data_col].to_numpy()
    len_res_df = len(res_df)
    roll_cols = [data_col + '_-{}'.format(cnt_ + 1) for cnt_ in reversed(range(roll_hl_cnt))]

    roll_data = pd.DataFrame(index=res_df.index, data=np.full((len_res_df, roll_hl_cnt), np.nan))
    roll_data.iloc[valid_prime_idx[roll_hl_cnt - 1:], :] = data[roll_idx_arr]  # 제한된 idx 를 제외한 row 에 roll_hl 입력

    res_df[roll_cols] = roll_data.ffill()

    return res_df


def get_roll_wave_data(valid_prime_idx, roll_idx_arr, len_df, data, roll_hl_cnt):
    roll_data = pd.DataFrame(np.full((len_df, roll_hl_cnt), np.nan))
    roll_data.iloc[valid_prime_idx[roll_hl_cnt - 1:], :] = data[roll_idx_arr]  # 제한된 idx 를 제외한 row 에 roll_hl 입력

    return roll_data.ffill().to_numpy()


def roll_wave_hl_idx_v5(t_df, wave_itv, wave_period, roll_hl_cnt=4):
    co_prime_idx = t_df['wave_co_prime_idx_{}{}'.format(wave_itv, wave_period)].to_numpy()
    cu_prime_idx = t_df['wave_cu_prime_idx_{}{}'.format(wave_itv, wave_period)].to_numpy()

    # Todo, roll_prev_high_idx 로 valid_co_prime_idx 를 사용하는 이유는 최종 high 를 선정하려면 co_prime_idx 를 이용해야하기 때문
    valid_co_prime_idx = co_prime_idx[~np.isnan(co_prime_idx)].astype(int)  # roll_high 를 위한 prime_idx, this should be "unique"
    valid_cu_prime_idx = cu_prime_idx[~np.isnan(cu_prime_idx)].astype(int)  # roll_low 를 위한 prime_idx

    # roll_co/cu_idx_arr 는 말그대로, valid_co/cu_prime_idx 를 roll 한 것
    roll_co_idx_arr = np.array([valid_co_prime_idx[idx_ + 1 - roll_hl_cnt:idx_ + 1] for idx_ in range(len(valid_co_prime_idx)) if
                                idx_ + 1 >= roll_hl_cnt])  # cnt 수를 만족시키기 위해 idx 제한
    roll_cu_idx_arr = np.array(
        [valid_cu_prime_idx[idx_ + 1 - roll_hl_cnt:idx_ + 1] for idx_ in range(len(valid_cu_prime_idx)) if idx_ + 1 >= roll_hl_cnt])

    assert len(roll_co_idx_arr) > 0 and len(roll_cu_idx_arr) > 0

    return valid_co_prime_idx, valid_cu_prime_idx, roll_co_idx_arr, roll_cu_idx_arr


def backing_future_data(res_df, future_cols, itv_list):  # itv 자동 조사 가능 ? (future_work)

    for col_, itv_ in zip(future_cols, itv_list):
        back_col_ = 'b1_' + col_
        res_df[back_col_] = res_df[col_].shift(to_itvnum(itv_))

    return res_df


def wave_loc_pct_v2(res_df, config, itv, period):
    wave_itv = pd.infer_freq(res_df.index)
    wave_period = config.tr_set.wave_period

    wave_high_fill_ = res_df['wave_high_fill_{}{}'.format(wave_itv, wave_period)].to_numpy()
    wave_low_fill_ = res_df['wave_low_fill_{}{}'.format(wave_itv, wave_period)].to_numpy()

    bb_upper_ = res_df['bb_upper_{}{}'.format(itv, period)].to_numpy()
    bb_lower_ = res_df['bb_lower_{}{}'.format(itv, period)].to_numpy()

    bb_gap = bb_upper_ - bb_lower_

    cu_prime_idx_fill_ = res_df['wave_cu_prime_idx_fill_{}{}'.format(wave_itv, wave_period)].to_numpy()
    co_prime_idx_fill_ = res_df['wave_co_prime_idx_fill_{}{}'.format(wave_itv, wave_period)].to_numpy()
    cu_prime_bb_gap = get_line(cu_prime_idx_fill_, bb_gap)
    co_prime_bb_gap = get_line(co_prime_idx_fill_, bb_gap)

    res_df['wave_high_loc_pct_{}{}'.format(wave_itv, wave_period)] = (bb_upper_ - wave_high_fill_) / cu_prime_bb_gap
    res_df['wave_low_loc_pct_{}{}'.format(wave_itv, wave_period)] = (wave_low_fill_ - bb_lower_) / co_prime_bb_gap

    return res_df


def wave_loc_pct(res_df, config, itv, period):
    wave_itv = pd.infer_freq(res_df.index)
    wave_period = config.tr_set.wave_period

    wave_high_fill_ = res_df['wave_high_fill_{}{}'.format(wave_itv, wave_period)].to_numpy()
    wave_low_fill_ = res_df['wave_low_fill_{}{}'.format(wave_itv, wave_period)].to_numpy()

    bb_upper_ = res_df['bb_upper_{}{}'.format(itv, period)].to_numpy()
    bb_lower_ = res_df['bb_lower_{}{}'.format(itv, period)].to_numpy()

    bb_gap = bb_upper_ - bb_lower_

    res_df['wave_high_loc_pct_{}{}'.format(wave_itv, wave_period)] = (bb_upper_ - wave_high_fill_) / bb_gap
    res_df['wave_low_loc_pct_{}{}'.format(wave_itv, wave_period)] = (wave_low_fill_ - bb_lower_) / bb_gap

    return res_df




def enough_space(res_df, config, itv, period):

  dc_upper_ = res_df['dc_upper_{}{}'.format(itv, period)].to_numpy()
  dc_base_ = res_df['dc_base_{}{}'.format(itv, period)].to_numpy()
  dc_lower_ = res_df['dc_lower_{}{}'.format(itv, period)].to_numpy()
  high_ = res_df['high_{}'.format(config.loc_set.point.tf_entry)].to_numpy()
  low_ = res_df['low_{}'.format(config.loc_set.point.tf_entry)].to_numpy()

  half_dc_gap = dc_upper_ - dc_base_

  res_df['cu_es_{}{}'.format(itv, period)] = (low_ - dc_lower_) / half_dc_gap
  res_df['co_es_{}{}'.format(itv, period)] = (dc_upper_ - high_) / half_dc_gap

  return res_df


# Todo, future_data
def candle_range_ratio(res_df, c_itv, bb_itv, bb_period):
    itv_num = to_itvnum(c_itv)

    b1_bb_upper_ = res_df['bb_upper_{}{}'.format(bb_itv, bb_period)].shift(itv_num).to_numpy()
    b1_bb_lower_ = res_df['bb_lower_{}{}'.format(bb_itv, bb_period)].shift(itv_num).to_numpy()
    bb_range = b1_bb_upper_ - b1_bb_lower_  # <-- h_candle's open_idx 의 bb_gap 사용

    high_ = res_df['high_{}'.format(c_itv)].to_numpy()
    low_ = res_df['low_{}'.format(c_itv)].to_numpy()
    candle_range = high_ - low_  # 부호로 양 / 음봉 구분 (양봉 > 0)

    res_df['crr_{}'.format(c_itv)] = candle_range / bb_range

    return res_df

def candle_pattern_pkg(ltf_df, htf_df):

    columns = talib.get_function_groups()['Pattern Recognition']
    for col in columns:
      func_ = eval("talib.{}".format(col))
      htf_df[col] = func_(htf_df.open, htf_df.high, htf_df.low, htf_df.close)
      # print(col, 'succeed')

    return h_candle_v4(ltf_df, htf_df, columns)

def body_rel_ratio(res_df, c_itv):
    itv_num = to_itvnum(c_itv)

    b1_close_ = res_df['close_{}'.format(c_itv)].shift(itv_num).to_numpy()
    b1_open_ = res_df['open_{}'.format(c_itv)].shift(itv_num).to_numpy()
    b1_body_range = abs(b1_close_ - b1_open_)

    close_ = res_df['close_{}'.format(c_itv)].to_numpy()
    open_ = res_df['open_{}'.format(c_itv)].to_numpy()
    body_range = abs(close_ - open_)

    res_df['body_rel_ratio_{}'.format(c_itv)] = body_range / b1_body_range

def dc_over_body_ratio(res_df, c_itv, dc_itv, dc_period):
    close_ = res_df['close_{}'.format(c_itv)].to_numpy()
    open_ = res_df['open_{}'.format(c_itv)].to_numpy()
    body_range = abs(close_ - open_)

    dc_upper_ = res_df['dc_upper_{}{}'.format(dc_itv, dc_period)].to_numpy()
    dc_lower_ = res_df['dc_lower_{}{}'.format(dc_itv, dc_period)].to_numpy()

    res_df['dc_upper_body_ratio'] = (close_ - dc_upper_) / body_range
    res_df['dc_lower_body_ratio'] = (close_ - dc_lower_) / body_range

    return res_df

def candle_pumping_ratio_v2(res_df, c_itv, dc_itv, period):
    res_df = dc_line_v3(res_df, dc_itv, dc_period=period)

    dc_upper_ = res_df['dc_upper_{}{}'.format(dc_itv, period)].to_numpy()
    dc_lower_ = res_df['dc_lower_{}{}'.format(dc_itv, period)].to_numpy()
    dc_range = dc_upper_ - dc_lower_

    open_ = res_df['open_{}'.format(c_itv)].to_numpy()
    close_ = res_df['close_{}'.format(c_itv)].to_numpy()
    body = close_ - open_  # 부호로 양 / 음봉 구분 (양봉 > 0)

    res_df['cppr_{}'.format(c_itv)] = body / dc_range

    return res_df


# Todo, future_data
def candle_pumping_ratio(res_df, c_itv, bb_itv, period):
    itv_num = to_itvnum(c_itv)

    # 여기에도 v2 처럼 bb_indi. 추가 (자동화)

    b1_bb_upper_ = res_df['bb_upper_{}{}'.format(bb_itv, period)].shift(itv_num).to_numpy()
    b1_bb_lower_ = res_df['bb_lower_{}{}'.format(bb_itv, period)].shift(itv_num).to_numpy()
    bb_range = b1_bb_upper_ - b1_bb_lower_

    open_ = res_df['open_{}'.format(c_itv)].to_numpy()
    close_ = res_df['close_{}'.format(c_itv)].to_numpy()
    body = close_ - open_  # 부호로 양 / 음봉 구분 (양봉 > 0)

    res_df['cppr_{}'.format(c_itv)] = body / bb_range

    return res_df

def pumping_ratio(res_df, config, itv, period1, period2):
    bb_lower_5T = res_df['bb_lower_5T'].to_numpy()
    bb_upper_5T = res_df['bb_upper_5T'].to_numpy()
    bb_range = bb_upper_5T - bb_lower_5T

    selection_id = config.selection_id

    res_df['short_ppr_{}'.format(selection_id)] = res_df['short_tp_gap_{}'.format(selection_id)].to_numpy() / get_line(
        res_df['short_wave_high_idx_{}{}{}'.format(itv, period1, period2)].to_numpy(), bb_range)
    res_df['long_ppr_{}'.format(selection_id)] = res_df['long_tp_gap_{}'.format(selection_id)].to_numpy() / get_line(
        res_df['long_wave_low_idx_{}{}{}'.format(itv, period1, period2)].to_numpy(), bb_range)


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
  df['short_ir_{}'.format(itv)] = np.where(close < open, (b2_low - high) / b1_candle_range, np.nan)  # close < open & close < b1_low
  df['long_ir_{}'.format(itv)] = np.where(close > open, (low - b2_high) / b1_candle_range, np.nan)  # close > open & close > b1_high

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
  df['short_ir_{}'.format(itv)] = np.where(close < open, (b1_low - close) / candle_range, np.nan)  # close < open & close < b1_low
  df['long_ir_{}'.format(itv)] = np.where(close > open, (close - b1_high) / candle_range, np.nan)  # close > open & close > b1_high

  return


def wave_range_stoch_v1(t_df, wave_period, slowk_period=3, slowd_period=3):
    t_df = stoch_v2(t_df, fastk_period=wave_period)
    itv = pd.infer_freq(t_df.index)

    stoch_ = t_df['stoch_{}{}{}{}'.format(itv, wave_period, slowk_period, slowd_period)].to_numpy()
    b1_stoch_ = t_df['stoch_{}{}{}{}'.format(itv, wave_period, slowk_period, slowd_period)].shift(1).to_numpy()

    baseline = 50
    band_width = 17
    upper_band = baseline + band_width
    lower_band = baseline - band_width

    data_cols = ['open', 'high', 'low', 'close']
    ohlc_list = [t_df[col_].to_numpy() for col_ in data_cols]
    open, high, low, close = ohlc_list

    # ============ modules ============ #
    # ------ define co, cu ------ # <- point missing 과 관련해 정교해아함
    cu_bool = (b1_stoch_ > upper_band) & (upper_band > stoch_)
    co_bool = (b1_stoch_ < lower_band) & (lower_band < stoch_)

    return wave_publics(t_df, cu_bool, co_bool, ohlc_list, wave_period)

def wave_range_dc_envel_v1(t_df, wave_period):

    t_df = donchian_channel_v4(t_df, wave_period)
    itv = pd.infer_freq(t_df.index)

    dc_upper_ = t_df['dc_upper_{}{}'.format(itv, wave_period)].to_numpy()
    dc_lower_ = t_df['dc_lower_{}{}'.format(itv, wave_period)].to_numpy()

    data_cols = ['open', 'high', 'low', 'close']
    ohlc_list = [t_df[col_].to_numpy() for col_ in data_cols]
    open, high, low, close = ohlc_list
    # b1_close = t_df.close.shift(itv_num).to_numpy()

    # ============ modules ============ #
    # ------ define co, cu ------ # <- point missing 과 관련해 정교해아함
    cu_bool = low <= dc_lower_
    co_bool = high >= dc_upper_

    return wave_publics(t_df, cu_bool, co_bool, ohlc_list, wave_period)

def wave_range_cci_v4_1(t_df, wave_period, band_width=100):
    t_df = cci_v2(t_df, wave_period)
    itv = pd.infer_freq(t_df.index)

    cci_ = t_df['cci_{}{}'.format(itv, wave_period)].to_numpy()
    b1_cci_ = t_df['cci_{}{}'.format(itv, wave_period)].shift(1).to_numpy()

    baseline = 0
    upper_band = baseline + band_width
    lower_band = baseline - band_width

    data_cols = ['open', 'high', 'low', 'close']
    ohlc_list = [t_df[col_].to_numpy() for col_ in data_cols]
    open, high, low, close = ohlc_list

    # ============ modules ============ #
    # ------ define co, cu ------ # <- point missing 과 관련해 정교해아함
    cu_bool = (b1_cci_ > upper_band) & (upper_band > cci_)
    co_bool = (b1_cci_ < lower_band) & (lower_band < cci_)

    return wave_publics_v3(t_df, cu_bool, co_bool, ohlc_list, wave_period)

def wave_range_cci_v4(t_df, wave_period):
    t_df = cci_v2(t_df, wave_period)
    itv = pd.infer_freq(t_df.index)

    cci_ = t_df['cci_{}{}'.format(itv, wave_period)].to_numpy()
    b1_cci_ = t_df['cci_{}{}'.format(itv, wave_period)].shift(1).to_numpy()

    baseline = 0
    band_width = 100
    upper_band = baseline + band_width
    lower_band = baseline - band_width

    data_cols = ['open', 'high', 'low', 'close']
    ohlc_list = [t_df[col_].to_numpy() for col_ in data_cols]
    open, high, low, close = ohlc_list

    # ============ modules ============ #
    # ------ define co, cu ------ # <- point missing 과 관련해 정교해아함
    cu_bool = (b1_cci_ > upper_band) & (upper_band > cci_)
    co_bool = (b1_cci_ < lower_band) & (lower_band < cci_)

    return wave_publics(t_df, cu_bool, co_bool, ohlc_list, wave_period)


def wave_publics_v3(t_df, cu_bool, co_bool, ohlc_list, wave_period):
    itv = pd.infer_freq(t_df.index)

    len_df = len(t_df)
    len_df_range = np.arange(len_df).astype(int)

    cu_idx = get_index_bybool(cu_bool, len_df_range)
    co_idx = get_index_bybool(co_bool, len_df_range)

    open, high, low, close = ohlc_list

    cu_fill_idx, co_fill_idx, cu_prime_idx, co_prime_idx, cu_prime_fill_idx, co_prime_fill_idx, valid_cu_bool, valid_co_bool = get_terms_info_v4(
        cu_idx, co_idx, len_df, len_df_range)
    # cu_fill_idx, co_fill_idx, cu_prime_idx, co_prime_idx, cu_prime_fill_idx, co_prime_fill_idx, \
    #   cu_post_idx, co_post_idx, cu_post_fill_idx, co_post_fill_idx, valid_cu_bool, valid_co_bool = get_terms_info_v5(cu_idx, co_idx, len_df, len_df_range)

    # ------ get post_terms ------ #
    high_post_terms = np.vstack((co_fill_idx[valid_cu_bool], cu_idx[valid_cu_bool])).T.astype(int)
    low_post_terms = np.vstack((cu_fill_idx[valid_co_bool], co_idx[valid_co_bool])).T.astype(int)

    high_post_terms_cnt = high_post_terms[:, 1] - high_post_terms[:, 0]
    low_post_terms_cnt = low_post_terms[:, 1] - low_post_terms[:, 0]

    # ------ get post_idx ------ #
    paired_cu_post_idx = high_post_terms[:, 1]  # Todo, 여기는 cross_idx (위에서 vstack 으로 cross_idx 입력함)
    paired_co_post_idx = low_post_terms[:, 1]

    cu_post_idx = np.full(len_df, np.nan)  # --> Todo, unavailable : not cross_idx
    co_post_idx = np.full(len_df, np.nan)

    cu_post_idx[paired_cu_post_idx] = paired_cu_post_idx
    co_post_idx[paired_co_post_idx] = paired_co_post_idx

    cu_post_fill_idx = fill_arr(cu_post_idx)
    co_post_fill_idx = fill_arr(co_post_idx)

    # ------ get prime_terms ------ # # 기본은 아래 logic 으로 수행하고, update_hl 도 해당 term 구간의 hl 이 더 작거나 클경우 적용 가능할 것
    # high_prime_terms = np.vstack((co_prime_fill_idx[valid_cu_bool], cu_idx[valid_cu_bool])).T.astype(int)
    # low_prime_terms = np.vstack((cu_prime_fill_idx[valid_co_bool], co_idx[valid_co_bool])).T.astype(int)

    # high_prime_terms_cnt = high_prime_terms[:, 1] - high_prime_terms[:, 0]
    # low_prime_terms_cnt = low_prime_terms[:, 1] - low_prime_terms[:, 0]

    # paired_prime_cu_idx = high_prime_terms[:, 1]
    # paired_prime_co_idx = low_prime_terms[:, 1]

    # ====== get wave_hl & terms ====== #
    wave_high_ = np.full(len_df, np.nan)
    wave_low_ = np.full(len_df, np.nan)

    wave_highs = np.array([high[iin:iout + 1].max() for iin, iout in high_post_terms])
    wave_lows = np.array([low[iin:iout + 1].min() for iin, iout in low_post_terms])

    wave_high_[paired_cu_post_idx] = wave_highs
    wave_low_[paired_co_post_idx] = wave_lows

    wave_high_fill_ = fill_arr(wave_high_)
    wave_low_fill_ = fill_arr(wave_low_)

    wave_high_idx = np.full(len_df, np.nan)
    wave_low_idx = np.full(len_df, np.nan)

    wave_highs_idx = np.array([high[iin:iout + 1].argmax() + iin for iin, iout in high_post_terms])
    wave_lows_idx = np.array([low[iin:iout + 1].argmin() + iin for iin, iout in low_post_terms])

    wave_high_idx[paired_cu_post_idx] = wave_highs_idx
    wave_low_idx[paired_co_post_idx] = wave_lows_idx

    wave_high_idx_fill_ = fill_arr(wave_high_idx)
    wave_low_idx_fill_ = fill_arr(wave_low_idx)

    # ------ Todo, update_hl 에 대해서, post_terms_hl 적용 ------ #
    wave_high_terms_low_ = np.full(len_df, np.nan)
    wave_low_terms_high_ = np.full(len_df, np.nan)

    wave_high_terms_lows = np.array([low[iin:iout + 1].min() for iin, iout in high_post_terms])  # for point rejection, Todo, min_max 설정 항상 주의
    wave_low_terms_highs = np.array([high[iin:iout + 1].max() for iin, iout in low_post_terms])

    wave_high_terms_low_[paired_cu_post_idx] = wave_high_terms_lows
    wave_low_terms_high_[paired_co_post_idx] = wave_low_terms_highs

    update_low_cu_bool = wave_high_terms_low_ < wave_low_fill_
    update_high_co_bool = wave_low_terms_high_ > wave_high_fill_

    # ------ term cnt ------ #
    wave_high_terms_cnt_ = np.full(len_df, np.nan)
    wave_low_terms_cnt_ = np.full(len_df, np.nan)

    wave_high_terms_cnt_[paired_cu_post_idx] = high_post_terms_cnt
    wave_low_terms_cnt_[paired_co_post_idx] = low_post_terms_cnt

    wave_high_terms_cnt_fill_ = fill_arr(wave_high_terms_cnt_)
    wave_low_terms_cnt_fill_ = fill_arr(wave_low_terms_cnt_)

    # ------ hl_fill 의 prime_idx 를 찾아야함 ------ #
    # b1_wave_high_fill_ = pd.Series(wave_high_fill_).shift(1).to_numpy()
    # b1_wave_low_fill_ = pd.Series(wave_low_fill_).shift(1).to_numpy()
    # wave_high_prime_idx = np.where((wave_high_fill_ != b1_wave_high_fill_) & ~np.isnan(wave_high_fill_), len_df_range, np.nan)
    # wave_low_prime_idx = np.where((wave_low_fill_ != b1_wave_low_fill_) & ~np.isnan(wave_low_fill_), len_df_range, np.nan)
    #
    # high_prime_idx_fill_ = fill_arr(wave_high_prime_idx)
    # low_prime_idx_fill_ = fill_arr(wave_low_prime_idx)

    # ============ enlist to df_cols ============ #
    t_df['wave_high_fill_{}{}'.format(itv, wave_period)] = wave_high_fill_
    t_df['wave_low_fill_{}{}'.format(itv, wave_period)] = wave_low_fill_
    t_df['wave_high_terms_cnt_fill_{}{}'.format(itv, wave_period)] = wave_high_terms_cnt_fill_
    t_df['wave_low_terms_cnt_fill_{}{}'.format(itv, wave_period)] = wave_low_terms_cnt_fill_

    t_df['wave_high_idx_fill_{}{}'.format(itv, wave_period)] = wave_high_idx_fill_
    t_df['wave_low_idx_fill_{}{}'.format(itv, wave_period)] = wave_low_idx_fill_

    t_df['wave_update_low_cu_bool_{}{}'.format(itv, wave_period)] = update_low_cu_bool  # temporary, for plot_check
    t_df['wave_update_high_co_bool_{}{}'.format(itv, wave_period)] = update_high_co_bool

    t_df['wave_cu_{}{}'.format(itv, wave_period)] = cu_bool  # * ~update_low_cu_bool
    t_df['wave_co_{}{}'.format(itv, wave_period)] = co_bool  # * ~update_high_co_bool

    t_df['wave_co_idx_{}{}'.format(itv, wave_period)] = co_idx
    t_df['wave_cu_idx_{}{}'.format(itv, wave_period)] = cu_idx

    t_df['wave_co_post_idx_{}{}'.format(itv, wave_period)] = co_post_idx  # paired_
    t_df['wave_cu_post_idx_{}{}'.format(itv, wave_period)] = cu_post_idx  # paired_
    t_df['wave_co_post_idx_fill_{}{}'.format(itv, wave_period)] = co_post_fill_idx
    t_df['wave_cu_post_idx_fill_{}{}'.format(itv, wave_period)] = cu_post_fill_idx

    # Todo, idx 저장은 sync. 가 맞는 tf_df 에 대하여 적용하여야함
    # ------ for roll prev_hl ------ #
    # high_post_idx 를 위해 co_prime_idx 입력 = 뜻 : high_term's prime co_idx (high_prime_idx = wave_high 를 만들기 위한 가장 앞단의 co_idx)
    t_df['wave_co_prime_idx_{}{}'.format(itv,
                                         wave_period)] = co_prime_idx  # co_prime_idx wave_high_prime_idx  # high 갱신을 고려해, prev_hl 는 prime_idx 기준으로 진행
    t_df['wave_cu_prime_idx_{}{}'.format(itv,
                                         wave_period)] = cu_prime_idx  # cu_prime_idx wave_low_prime_idx  # cu_prime_idx's low 를 사용하겠다라는 의미, 즉 roll_prev 임
    t_df['wave_co_prime_idx_fill_{}{}'.format(itv, wave_period)] = co_prime_fill_idx  # co_prime_fill_idx high_prime_idx_fill_
    t_df['wave_cu_prime_idx_fill_{}{}'.format(itv, wave_period)] = cu_prime_fill_idx  # cu_prime_fill_idx low_prime_idx_fill_

    # ------ for plot_checking ------ #
    t_df['wave_cu_marker_{}{}'.format(itv, wave_period)] = get_line(cu_idx, close)
    t_df['wave_co_marker_{}{}'.format(itv, wave_period)] = get_line(co_idx, close)

    return t_df

def wave_publics_v2(t_df, cu_bool, co_bool, ohlc_list, wave_period):
    itv = pd.infer_freq(t_df.index)

    len_df = len(t_df)
    len_df_range = np.arange(len_df).astype(int)

    cu_idx = get_index_bybool(cu_bool, len_df_range)
    co_idx = get_index_bybool(co_bool, len_df_range)

    open, high, low, close = ohlc_list

    cu_fill_idx, co_fill_idx, cu_prime_idx, co_prime_idx, cu_prime_fill_idx, co_prime_fill_idx, valid_cu_bool, valid_co_bool = get_terms_info_v4(
        cu_idx, co_idx, len_df, len_df_range)
    # cu_fill_idx, co_fill_idx, cu_prime_idx, co_prime_idx, cu_prime_fill_idx, co_prime_fill_idx, \
    #   cu_post_idx, co_post_idx, cu_post_fill_idx, co_post_fill_idx, valid_cu_bool, valid_co_bool = get_terms_info_v5(cu_idx, co_idx, len_df, len_df_range)

    # ------ get post_terms ------ #
    high_post_terms = np.vstack((co_fill_idx[valid_cu_bool], cu_idx[valid_cu_bool])).T.astype(int)
    low_post_terms = np.vstack((cu_fill_idx[valid_co_bool], co_idx[valid_co_bool])).T.astype(int)

    high_post_terms_cnt = high_post_terms[:, 1] - high_post_terms[:, 0]
    low_post_terms_cnt = low_post_terms[:, 1] - low_post_terms[:, 0]

    # ------ get post_idx ------ #
    paired_cu_post_idx = high_post_terms[:, 1]  # Todo, 여기는 cross_idx (위에서 vstack 으로 cross_idx 입력함)
    paired_co_post_idx = low_post_terms[:, 1]

    cu_post_idx = np.full(len_df, np.nan)  # --> Todo, unavailable : not cross_idx
    co_post_idx = np.full(len_df, np.nan)

    cu_post_idx[paired_cu_post_idx] = paired_cu_post_idx
    co_post_idx[paired_co_post_idx] = paired_co_post_idx

    cu_post_fill_idx = fill_arr(cu_post_idx)
    co_post_fill_idx = fill_arr(co_post_idx)

    # ------ get prime_terms ------ # # 기본은 아래 logic 으로 수행하고, update_hl 도 해당 term 구간의 hl 이 더 작거나 클경우 적용 가능할 것
    # high_prime_terms = np.vstack((co_prime_fill_idx[valid_cu_bool], cu_idx[valid_cu_bool])).T.astype(int)
    # low_prime_terms = np.vstack((cu_prime_fill_idx[valid_co_bool], co_idx[valid_co_bool])).T.astype(int)

    # high_prime_terms_cnt = high_prime_terms[:, 1] - high_prime_terms[:, 0]
    # low_prime_terms_cnt = low_prime_terms[:, 1] - low_prime_terms[:, 0]

    # paired_prime_cu_idx = high_prime_terms[:, 1]
    # paired_prime_co_idx = low_prime_terms[:, 1]

    # ====== get wave_hl & terms ====== #
    wave_high_ = np.full(len_df, np.nan)
    wave_low_ = np.full(len_df, np.nan)

    wave_highs = np.array([high[iin:iout + 1].max() for iin, iout in high_post_terms])
    wave_lows = np.array([low[iin:iout + 1].min() for iin, iout in low_post_terms])

    wave_high_[paired_cu_post_idx] = wave_highs
    wave_low_[paired_co_post_idx] = wave_lows

    wave_high_fill_ = fill_arr(wave_high_)
    wave_low_fill_ = fill_arr(wave_low_)

    # ------ Todo, update_hl 에 대해서, post_terms_hl 적용 ------ #
    wave_high_terms_low_ = np.full(len_df, np.nan)
    wave_low_terms_high_ = np.full(len_df, np.nan)

    wave_high_terms_lows = np.array([low[iin:iout + 1].min() for iin, iout in high_post_terms])  # for point rejection, Todo, min_max 설정 항상 주의
    wave_low_terms_highs = np.array([high[iin:iout + 1].max() for iin, iout in low_post_terms])

    wave_high_terms_low_[paired_cu_post_idx] = wave_high_terms_lows
    wave_low_terms_high_[paired_co_post_idx] = wave_low_terms_highs

    update_low_cu_bool = wave_high_terms_low_ < wave_low_fill_
    update_high_co_bool = wave_low_terms_high_ > wave_high_fill_

    # ------ term cnt ------ #
    wave_high_terms_cnt_ = np.full(len_df, np.nan)
    wave_low_terms_cnt_ = np.full(len_df, np.nan)

    wave_high_terms_cnt_[paired_cu_post_idx] = high_post_terms_cnt
    wave_low_terms_cnt_[paired_co_post_idx] = low_post_terms_cnt

    wave_high_terms_cnt_fill_ = fill_arr(wave_high_terms_cnt_)
    wave_low_terms_cnt_fill_ = fill_arr(wave_low_terms_cnt_)

    # ------ hl_fill 의 prime_idx 를 찾아야함 ------ #
    # b1_wave_high_fill_ = pd.Series(wave_high_fill_).shift(1).to_numpy()
    # b1_wave_low_fill_ = pd.Series(wave_low_fill_).shift(1).to_numpy()
    # wave_high_prime_idx = np.where((wave_high_fill_ != b1_wave_high_fill_) & ~np.isnan(wave_high_fill_), len_df_range, np.nan)
    # wave_low_prime_idx = np.where((wave_low_fill_ != b1_wave_low_fill_) & ~np.isnan(wave_low_fill_), len_df_range, np.nan)
    #
    # high_prime_idx_fill_ = fill_arr(wave_high_prime_idx)
    # low_prime_idx_fill_ = fill_arr(wave_low_prime_idx)

    # ============ enlist to df_cols ============ #
    t_df['wave_high_fill_{}{}'.format(itv, wave_period)] = wave_high_fill_
    t_df['wave_low_fill_{}{}'.format(itv, wave_period)] = wave_low_fill_
    t_df['wave_high_terms_cnt_fill_{}{}'.format(itv, wave_period)] = wave_high_terms_cnt_fill_
    t_df['wave_low_terms_cnt_fill_{}{}'.format(itv, wave_period)] = wave_low_terms_cnt_fill_

    t_df['wave_update_low_cu_bool_{}{}'.format(itv, wave_period)] = update_low_cu_bool  # temporary, for plot_check
    t_df['wave_update_high_co_bool_{}{}'.format(itv, wave_period)] = update_high_co_bool

    t_df['wave_cu_{}{}'.format(itv, wave_period)] = cu_bool  # * ~update_low_cu_bool
    t_df['wave_co_{}{}'.format(itv, wave_period)] = co_bool  # * ~update_high_co_bool

    t_df['wave_co_idx_{}{}'.format(itv, wave_period)] = co_idx
    t_df['wave_cu_idx_{}{}'.format(itv, wave_period)] = cu_idx

    t_df['wave_co_post_idx_{}{}'.format(itv, wave_period)] = co_post_idx  # paired_
    t_df['wave_cu_post_idx_{}{}'.format(itv, wave_period)] = cu_post_idx  # paired_
    t_df['wave_co_post_idx_fill_{}{}'.format(itv, wave_period)] = co_post_fill_idx
    t_df['wave_cu_post_idx_fill_{}{}'.format(itv, wave_period)] = cu_post_fill_idx

    # Todo, idx 저장은 sync. 가 맞는 tf_df 에 대하여 적용하여야함
    # ------ for roll prev_hl ------ #
    # high_post_idx 를 위해 co_prime_idx 입력 = 뜻 : high_term's prime co_idx (high_prime_idx = wave_high 를 만들기 위한 가장 앞단의 co_idx)
    t_df['wave_co_prime_idx_{}{}'.format(itv,
                                         wave_period)] = co_prime_idx  # co_prime_idx wave_high_prime_idx  # high 갱신을 고려해, prev_hl 는 prime_idx 기준으로 진행
    t_df['wave_cu_prime_idx_{}{}'.format(itv,
                                         wave_period)] = cu_prime_idx  # cu_prime_idx wave_low_prime_idx  # cu_prime_idx's low 를 사용하겠다라는 의미, 즉 roll_prev 임
    t_df['wave_co_prime_idx_fill_{}{}'.format(itv, wave_period)] = co_prime_fill_idx  # co_prime_fill_idx high_prime_idx_fill_
    t_df['wave_cu_prime_idx_fill_{}{}'.format(itv, wave_period)] = cu_prime_fill_idx  # cu_prime_fill_idx low_prime_idx_fill_

    # ------ for plot_checking ------ #
    t_df['wave_cu_marker_{}{}'.format(itv, wave_period)] = get_line(cu_idx, close)
    t_df['wave_co_marker_{}{}'.format(itv, wave_period)] = get_line(co_idx, close)

    return t_df

def wave_publics(t_df, cu_bool, co_bool, ohlc_list, wave_period):
    itv = pd.infer_freq(t_df.index)

    len_df = len(t_df)
    len_df_range = np.arange(len_df).astype(int)

    cu_idx = get_index_bybool(cu_bool, len_df_range)
    co_idx = get_index_bybool(co_bool, len_df_range)

    open, high, low, close = ohlc_list

    cu_fill_idx, co_fill_idx, cu_prime_idx, co_prime_idx, cu_prime_fill_idx, co_prime_fill_idx, valid_cu_bool, valid_co_bool = get_terms_info_v4(
        cu_idx, co_idx, len_df, len_df_range)
    # cu_fill_idx, co_fill_idx, cu_prime_idx, co_prime_idx, cu_prime_fill_idx, co_prime_fill_idx, \
    #   cu_post_idx, co_post_idx, cu_post_fill_idx, co_post_fill_idx, valid_cu_bool, valid_co_bool = get_terms_info_v5(cu_idx, co_idx, len_df, len_df_range)

    # ------ get post_terms ------ #
    high_post_terms = np.vstack((co_fill_idx[valid_cu_bool], cu_idx[valid_cu_bool])).T.astype(int)
    low_post_terms = np.vstack((cu_fill_idx[valid_co_bool], co_idx[valid_co_bool])).T.astype(int)

    high_post_terms_cnt = high_post_terms[:, 1] - high_post_terms[:, 0]
    low_post_terms_cnt = low_post_terms[:, 1] - low_post_terms[:, 0]

    # ------ get post_idx ------ #
    paired_cu_post_idx = high_post_terms[:, 1]  # Todo, 여기는 cross_idx (위에서 vstack 으로 cross_idx 입력함)
    paired_co_post_idx = low_post_terms[:, 1]

    cu_post_idx = np.full(len_df, np.nan)  # --> Todo, unavailable : not cross_idx
    co_post_idx = np.full(len_df, np.nan)

    cu_post_idx[paired_cu_post_idx] = paired_cu_post_idx
    co_post_idx[paired_co_post_idx] = paired_co_post_idx

    cu_post_fill_idx = fill_arr(cu_post_idx)
    co_post_fill_idx = fill_arr(co_post_idx)

    # ------ get prime_terms ------ # # 기본은 아래 logic 으로 수행하고, update_hl 도 해당 term 구간의 hl 이 더 작거나 클경우 적용 가능할 것
    # high_prime_terms = np.vstack((co_prime_fill_idx[valid_cu_bool], cu_idx[valid_cu_bool])).T.astype(int)
    # low_prime_terms = np.vstack((cu_prime_fill_idx[valid_co_bool], co_idx[valid_co_bool])).T.astype(int)

    # high_prime_terms_cnt = high_prime_terms[:, 1] - high_prime_terms[:, 0]
    # low_prime_terms_cnt = low_prime_terms[:, 1] - low_prime_terms[:, 0]

    # paired_prime_cu_idx = high_prime_terms[:, 1]
    # paired_prime_co_idx = low_prime_terms[:, 1]

    # ====== get wave_hl & terms ====== #
    wave_high_ = np.full(len_df, np.nan)
    wave_low_ = np.full(len_df, np.nan)

    wave_highs = np.array([high[iin:iout + 1].max() for iin, iout in high_post_terms])
    wave_lows = np.array([low[iin:iout + 1].min() for iin, iout in low_post_terms])

    wave_high_[paired_cu_post_idx] = wave_highs
    wave_low_[paired_co_post_idx] = wave_lows

    wave_high_fill_ = fill_arr(wave_high_)
    wave_low_fill_ = fill_arr(wave_low_)

    # ------ Todo, update_hl 에 대해서, post_terms_hl 적용 ------ #
    wave_high_terms_low_ = np.full(len_df, np.nan)
    wave_low_terms_high_ = np.full(len_df, np.nan)

    wave_high_terms_lows = np.array([low[iin:iout + 1].min() for iin, iout in high_post_terms])  # for point rejection, Todo, min_max 설정 항상 주의
    wave_low_terms_highs = np.array([high[iin:iout + 1].max() for iin, iout in low_post_terms])

    wave_high_terms_low_[paired_cu_post_idx] = wave_high_terms_lows
    wave_low_terms_high_[paired_co_post_idx] = wave_low_terms_highs

    update_low_cu_bool = wave_high_terms_low_ < wave_low_fill_
    update_high_co_bool = wave_low_terms_high_ > wave_high_fill_

    # ------ term cnt ------ #
    wave_high_terms_cnt_ = np.full(len_df, np.nan)
    wave_low_terms_cnt_ = np.full(len_df, np.nan)

    wave_high_terms_cnt_[paired_cu_post_idx] = high_post_terms_cnt
    wave_low_terms_cnt_[paired_co_post_idx] = low_post_terms_cnt

    wave_high_terms_cnt_fill_ = fill_arr(wave_high_terms_cnt_)
    wave_low_terms_cnt_fill_ = fill_arr(wave_low_terms_cnt_)

    # ------ hl_fill 의 prime_idx 를 찾아야함 ------ #
    # b1_wave_high_fill_ = pd.Series(wave_high_fill_).shift(1).to_numpy()
    # b1_wave_low_fill_ = pd.Series(wave_low_fill_).shift(1).to_numpy()
    # wave_high_prime_idx = np.where((wave_high_fill_ != b1_wave_high_fill_) & ~np.isnan(wave_high_fill_), len_df_range, np.nan)
    # wave_low_prime_idx = np.where((wave_low_fill_ != b1_wave_low_fill_) & ~np.isnan(wave_low_fill_), len_df_range, np.nan)
    #
    # high_prime_idx_fill_ = fill_arr(wave_high_prime_idx)
    # low_prime_idx_fill_ = fill_arr(wave_low_prime_idx)

    # ============ enlist to df_cols ============ #
    t_df['wave_high_fill_{}{}'.format(itv, wave_period)] = wave_high_fill_
    t_df['wave_low_fill_{}{}'.format(itv, wave_period)] = wave_low_fill_
    t_df['wave_high_terms_cnt_fill_{}{}'.format(itv, wave_period)] = wave_high_terms_cnt_fill_
    t_df['wave_low_terms_cnt_fill_{}{}'.format(itv, wave_period)] = wave_low_terms_cnt_fill_

    t_df['wave_update_low_cu_bool_{}{}'.format(itv, wave_period)] = update_low_cu_bool  # temporary, for plot_check
    t_df['wave_update_high_co_bool_{}{}'.format(itv, wave_period)] = update_high_co_bool

    t_df['wave_cu_{}{}'.format(itv, wave_period)] = cu_bool  # * ~update_low_cu_bool
    t_df['wave_co_{}{}'.format(itv, wave_period)] = co_bool  # * ~update_high_co_bool

    t_df['wave_co_post_idx_{}{}'.format(itv, wave_period)] = co_post_idx  # paired_
    t_df['wave_cu_post_idx_{}{}'.format(itv, wave_period)] = cu_post_idx  # paired_
    t_df['wave_co_post_idx_fill_{}{}'.format(itv, wave_period)] = co_post_fill_idx
    t_df['wave_cu_post_idx_fill_{}{}'.format(itv, wave_period)] = cu_post_fill_idx

    # Todo, idx 저장은 sync. 가 맞는 tf_df 에 대하여 적용하여야함
    # ------ for roll prev_hl ------ #
    # high_post_idx 를 위해 co_prime_idx 입력 = 뜻 : high_term's prime co_idx (high_prime_idx = wave_high 를 만들기 위한 가장 앞단의 co_idx)
    t_df['wave_co_prime_idx_{}{}'.format(itv,
                                         wave_period)] = co_prime_idx  # co_prime_idx wave_high_prime_idx  # high 갱신을 고려해, prev_hl 는 prime_idx 기준으로 진행
    t_df['wave_cu_prime_idx_{}{}'.format(itv,
                                         wave_period)] = cu_prime_idx  # cu_prime_idx wave_low_prime_idx  # cu_prime_idx's low 를 사용하겠다라는 의미, 즉 roll_prev 임
    t_df['wave_co_prime_idx_fill_{}{}'.format(itv, wave_period)] = co_prime_fill_idx  # co_prime_fill_idx high_prime_idx_fill_
    t_df['wave_cu_prime_idx_fill_{}{}'.format(itv, wave_period)] = cu_prime_fill_idx  # cu_prime_fill_idx low_prime_idx_fill_

    # ------ for plot_checking ------ #
    t_df['wave_cu_marker_{}{}'.format(itv, wave_period)] = get_line(cu_idx, close)
    t_df['wave_co_marker_{}{}'.format(itv, wave_period)] = get_line(co_idx, close)

    return t_df


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

def wick_ratio(res_df, itv):
    if itv == "T":
        ohlc_col = ["open", "high", "low", "close"]
    else:
        ohlc_col = ['open_{}'.format(itv), 'high_{}'.format(itv), 'low_{}'.format(itv), 'close_{}'.format(itv)]

    open, high, low, close = [res_df[col_].to_numpy() for col_ in ohlc_col]

    total_len = high - low
    res_df['body_ratio_{}'.format(itv)] = abs(close - open) / total_len
    res_df['upper_wick_ratio_{}'.format(itv)] = (high - np.maximum(close, open)) / total_len
    res_df['lower_wick_ratio_{}'.format(itv)] = (np.minimum(close, open) - low) / total_len
    res_df['candle_updown_{}'.format(itv)] = np.where(close >= open, 1, 0)

    return res_df

def candle_score_v3(res_df, itv, updown=None, unsigned=True):
    if itv == "T":
        ohlc_col = ["open", "high", "low", "close"]
    else:
        ohlc_col = ['open_{}'.format(itv), 'high_{}'.format(itv), 'low_{}'.format(itv), 'close_{}'.format(itv)]

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


def h_candle_v4(ltf_df, htf_df, columns=None):   # v4 는 htf_df 를 입력해야함, colab 상에서 to_htf 를 중복하지 않기 위해 만든걸로 보임

    if columns is None:
      columns = ['open', 'high', 'low', 'close']

    last_row_ltf_df = pd.DataFrame(index=[ltf_df.index[-1]], data=np.full((1, len(columns)), np.nan), columns=columns)
    htf_df2 = pd.concat([htf_df[columns], last_row_ltf_df[columns].iloc[-1:]])  # downsampling 과정에서 timeindex sync 를 맞추기 위해 ltf last_row's ts 를 덧붙임
    # print(last_row_ltf_df)

    itv = pd.infer_freq(htf_df.index)

    #       downsampled h_res_df 의 offset 기준이 00:00 이라서     #
    ltf_df2 = htf_df2.resample('T').ffill()  # 기본적으로 'T' 단위로만 resampling, ffill 이 맞음

    to_ltf_col = [col + '_{}'.format(itv) for col in columns]

    #        1. res_df_ & resampled data idx unmatch
    #           1_1. res_df_ 의 남는 rows 를 채워주어야하는데
    ltf_df[to_ltf_col] = ltf_df2.to_numpy()[-len(ltf_df):]

    return ltf_df

def h_candle_v3(res_df_, itv):   # v3 는 htf_df 를 입력하지 않아도 됨
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
    res_df_[h_candle_col] = h_res_df2.to_numpy()[-len(res_df_):]

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

def donchian_channel_v4(df, period):
  itv = pd.infer_freq(df.index)
  upper_name = 'dc_upper_{}{}'.format(itv, period)
  lower_name = 'dc_lower_{}{}'.format(itv, period)
  base_name = 'dc_base_{}{}'.format(itv, period)

  try:
    df.drop([upper_name, lower_name, base_name], inplace=True, axis=1)
  except:
    pass

  if period != 1:
      hh = talib.MAX(df.high, period).to_numpy()
      ll = talib.MIN(df.low, period).to_numpy()
  else:
      hh = df.high.to_numpy()
      ll = df.low.to_numpy()

  df[upper_name] = hh
  df[lower_name] = ll
  df[base_name] = (hh + ll) / 2

  return df

def donchian_channel_v3(df, period):
  itv = pd.infer_freq(df.index)
  upper_name = 'dc_upper_{}{}'.format(itv, period)
  lower_name = 'dc_lower_{}{}'.format(itv, period)
  base_name = 'dc_base_{}{}'.format(itv, period)

  try:
    df.drop([upper_name, lower_name, base_name], inplace=True, axis=1)
  except:
    pass

  hh = df['high'].rolling(period).max().to_numpy()
  ll = df['low'].rolling(period).min().to_numpy()

  df[upper_name] = hh
  df[lower_name] = ll
  df[base_name] = (hh + ll) / 2

  return df

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

def donchian_channel(df, period):
    hh = df['high'].rolling(period).max().to_numpy()
    ll = df['low'].rolling(period).min().to_numpy()
    base = (hh + ll) / 2

    return hh, ll, base

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
  # touch_line = np.full_like(rtc_, np.nan)

  nan_idx = np.isnan(touch_idx_copy)
  touch_idx_copy[nan_idx] = 0   # for indexing array
  touch_line = rtc_[touch_idx_copy.astype(int)].copy()
  touch_line[nan_idx] = np.nan   # for true comp.

  return touch_line

def to_value_index(prev_wave_point_idx2, len_df_range):
  valid_idx = ~np.isnan(prev_wave_point_idx2)
  # np.sum(~np.isnan(prev_wave_point_idx2))
  prevwp_valid_value = prev_wave_point_idx2[valid_idx].astype(int)   # inner box value
  prev_wave_point_ = np.full(len(valid_idx), False)
  prev_wave_point_[prevwp_valid_value] = True

  return np.where(prev_wave_point_, len_df_range, np.nan)


def get_index_bybool(bool_arr, len_df_range):
  return np.where(bool_arr, len_df_range, np.nan)

def using_clump(a):
    return [a[s].astype(int) for s in np.ma.clump_unmasked(np.ma.masked_invalid(a))]


def fill_arr(arr_, mode='ffill'):
    if mode == 'ffill':
        return pd.Series(arr_).ffill().to_numpy()
    else:
        return pd.Series(arr_).bfill().to_numpy()


def dc_line_v4(ltf_df, htf_df, dc_period=20):

    interval = pd.infer_freq(htf_df.index)
    if interval not in ['T', '1m']:
        htf_df = donchian_channel_v4(htf_df, dc_period)
        return ltf_df.join(to_lower_tf_v2(ltf_df, htf_df, [i for i in range(-3, 0, 1)]), how='inner')
    else:
        ltf_df = donchian_channel_v4(ltf_df, dc_period)
        return ltf_df

def dc_line_v3(ltf_df, htf_df, dc_period=20):

    interval = pd.infer_freq(htf_df.index)
    if interval not in ['T', '1m']:
        htf_df = donchian_channel_v3(htf_df, dc_period)
        return ltf_df.join(to_lower_tf_v2(ltf_df, htf_df, [i for i in range(-3, 0, 1)]), how='inner')
    else:
        ltf_df = donchian_channel_v3(ltf_df, dc_period)
        return ltf_df

def dc_line_v2(ltf_df, htf_df, interval, dc_period=20):

    if interval not in ['T', '1m']:
        htf_df = donchian_channel_v3(htf_df, dc_period)
        return ltf_df.join(to_lower_tf_v2(ltf_df, htf_df, [i for i in range(-3, 0, 1)]), how='inner')
    else:
        ltf_df = donchian_channel_v3(ltf_df, dc_period)
        return ltf_df


def dc_line(ltf_df, htf_df, interval, dc_period=20):

    dc_upper = 'dc_upper_%s' % interval
    dc_lower = 'dc_lower_%s' % interval
    dc_base = 'dc_base_%s' % interval

    if interval not in ['T', '1m']:
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


def bb_width_v3(df, period, multiple):

    upper, base, lower = talib.BBANDS(df.close, timeperiod=period, nbdevup=multiple, nbdevdn=multiple, matype=0)

    itv = pd.infer_freq(df.index)
    upper_name = 'bb_upper_{}{}'.format(itv, period)
    lower_name = 'bb_lower_{}{}'.format(itv, period)
    base_name = 'bb_base_{}{}'.format(itv, period)

    try:
        df.drop([upper_name, lower_name, base_name], inplace=True, axis=1)
    except:
        pass

    df['bb_upper_{}{}'.format(itv, period)] = upper
    df['bb_lower_{}{}'.format(itv, period)] = lower
    df['bb_base_{}{}'.format(itv, period)] = base

    return df

def bb_width_v2(df, period, multiple, return_bbw=False):
    basis = df['close'].rolling(period).mean().to_numpy()
    dev_ = multiple * stdev(df, period)
    upper = basis + dev_
    lower = basis - dev_

    itv = pd.infer_freq(df.index)
    upper_name = 'bb_upper_{}{}'.format(itv, period)
    lower_name = 'bb_lower_{}{}'.format(itv, period)
    base_name = 'bb_base_{}{}'.format(itv, period)

    try:
        df.drop([upper_name, lower_name, base_name], inplace=True, axis=1)
    except:
        pass

    df['bb_upper_{}{}'.format(itv, period)] = upper
    df['bb_lower_{}{}'.format(itv, period)] = lower
    df['bb_base_{}{}'.format(itv, period)] = basis

    if return_bbw:
        df['bbw_{}{}'.format(itv, period)] = 2 * dev_ / basis

def bb_width(df, period, multiple):

    basis = df['close'].rolling(period).mean()
    # print(stdev(df, period))
    # quit()
    dev_ = multiple * stdev(df, period)
    upper = basis + dev_
    lower = basis - dev_
    bbw = 2 * dev_ / basis

    return upper, lower, basis, bbw

def bb_line(ltf_df, htf_df, interval, period=20, multi=1):  # deprecated - join 을 따로 사용해도 될터인데.. (v2 만들면서 느낀점)

    bb_upper = 'bb_upper_%s' % interval
    bb_lower = 'bb_lower_%s' % interval
    bb_base = 'bb_base_%s' % interval

    if interval not in ['T', '1m']:
        htf_df[bb_upper], htf_df[bb_lower], htf_df[bb_base], _ = bb_width(htf_df, period, multi)
        joined_ltf_df = ltf_df.join(to_lower_tf_v2(ltf_df, htf_df, [i for i in range(-3, 0, 1)]), how='inner')
    else:
        joined_ltf_df = ltf_df.copy()
        joined_ltf_df[bb_upper], joined_ltf_df[bb_lower], joined_ltf_df[bb_base], _ = bb_width(ltf_df, period, multi)

    return joined_ltf_df

def bb_line_v2(ltf_df, htf_df, interval, period=20, multi=1):
    # interval = pd.infer_freq(htf_df.index)
    if interval not in ['T', '1m']:
        bb_width_v2(htf_df, period, multi)
        return ltf_df.join(to_lower_tf_v2(ltf_df, htf_df, [i for i in range(-3, 0, 1)]), how='inner')
    else:
        bb_width(ltf_df, period, multi)
        return ltf_df

def bb_line_v3(ltf_df, htf_df, period=20, multiple=1):

    interval = pd.infer_freq(htf_df.index)
    if interval not in ['T', '1m']:
        bb_width_v3(htf_df, period, multiple)
        return ltf_df.join(to_lower_tf_v2(ltf_df, htf_df, [i for i in range(-3, 0, 1)]), how='inner')
    else:
        bb_width_v3(ltf_df, period, multiple)
        return ltf_df

def bb_level_v2(res_df, itv, period):

    bb_base = res_df['bb_base_{}{}'.format(itv, period)].to_numpy()

    bb_upper2 = 'bb_upper2_{}{}'.format(itv, period)
    bb_lower2 = 'bb_lower2_{}{}'.format(itv, period)
    bb_upper3 = 'bb_upper3_{}{}'.format(itv, period)
    bb_lower3 = 'bb_lower3_{}{}'.format(itv, period)

    level_gap = res_df['bb_upper_{}{}'.format(itv, period)].to_numpy() - bb_base

    res_df[bb_upper2] = bb_base + level_gap * 2
    res_df[bb_lower2] = bb_base - level_gap * 2

    res_df[bb_upper3] = bb_base + level_gap * 3
    res_df[bb_lower3] = bb_base - level_gap * 3

    return res_df

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
