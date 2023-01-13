import numpy as np
import pandas as pd
from funcs.public.broker import to_lower_tf, intmin, ffill


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


def stdev(df, period):
    avg = df['close'].rolling(period).mean()
    std = pd.Series(index=df.index)
    for i in range(len(df)):
        std.iloc[i] = 0.0
        for backing_i in range(period):
            sum = tozero(df['close'].iloc[i - backing_i], -avg.iloc[i])
            std.iloc[i] += sum ** 2
        std.iloc[i] = np.sqrt(std.iloc[i] / period)

    return std


def dev(data, period):

    mean_ = data.rolling(period).mean()
    dev_ = pd.Series(index=data.index)

    for i in range(len(data)):
        sum = 0.0
        for backing_i in range(period):
            val_ = data.iloc[i - backing_i]
            sum += abs(val_ - mean_.iloc[i])

        dev_.iloc[i] = sum / period

    return dev_


# # wick_score -> 정수자리, body_score -> 소수점 자리
# def candle_score(o, h, l, c, updown=None, unsigned=True):
#
#     #   check up / downward
#     up = np.where(c >= o, 1, 0)
#
#     total_len = h - l
#     upper_wick = (h - np.maximum(c, o)) / total_len
#     lower_wick = (np.minimum(c, o) - l) / total_len
#     body = abs(c - o) / total_len
#
#     up_score = (1 - upper_wick) * 100 + body
#     dn_score = (1 - lower_wick) * 100 + body
#
#     if updown is not None:
#         if updown == "up":
#             score = up_score
#         else:
#             if unsigned:
#                 score = dn_score
#             else:
#                 score = -dn_score
#     else:
#         if unsigned:
#             score = np.where(up, up_score, dn_score)
#         else:
#             score = np.where(up, up_score, -dn_score)
#
#     return score, body * 100


# def candle_score(o, h, l, c, updown=None, unsigned=True):
#
#     #   check up / downward
#     up = np.where(c >= o, 1, 0)
#
#     total_len = h - l
#     upper_wick = (h - np.maximum(c, o)) / total_len
#     lower_wick = (np.minimum(c, o) - l) / total_len
#     body = abs(c - o) / total_len
#
#     up_score = ((1 - upper_wick) * 100).astype(np.int64) + body
#     dn_score = ((1 - lower_wick) * 100).astype(np.int64) + body
#
#     if updown is not None:
#         if updown == "up":
#             score = up_score
#         else:
#             if unsigned:
#                 score = dn_score
#             else:
#                 score = -dn_score
#     else:
#         if unsigned:
#             score = np.where(up, up_score, dn_score)
#         else:
#             score = np.where(up, up_score, -dn_score)
#
#     return score, body * 100


def candle_score(o, h, l, c, updown=None, unsigned=True):

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


def candle_ratio(res_df, ohlc_col=None, updown=None, unsigned=True):

    if ohlc_col is None:
        ohlc_col = ["open", "high", "low", "close"]

    ohlcs = res_df[ohlc_col]
    o, h, l, c = np.split(ohlcs.values, 4, axis=1)

    # score = candle_score(o, h, l, c, updown, unsigned)

    return candle_score(o, h, l, c, updown, unsigned)


def ema(data, period, adjust=True):
    return pd.Series(data.ewm(span=period, adjust=adjust).mean())


def cloud_bline(df, period):

    h = df['high'].rolling(period).max()
    l = df['low'].rolling(period).min()
    avg = (h + l) / 2

    return avg


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


def roc(data, period):
    roc_ = 100 * (data - data.shift(period)) / data.shift(period)

    return roc_


def ema_roc(data, roc_period, period):
    ema_roc_ = ema(roc(data, roc_period), period)

    return ema_roc_


def trix_hist(df, period, multiplier, signal_period):
    triple = ema(ema(ema(df['close'], period), period), period)

    roc_ = 100 * (triple.diff() / triple)
    trix = multiplier * roc_
    signal = trix.rolling(signal_period).mean()

    hist = trix - signal

    return hist


def dtk_plot(res_df, dtk_itv2, hhtf_entry, use_dtk_line):

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


def h_candle(res_df, interval_):

    np_timeidx = np.array(list(map(lambda x: intmin(x), res_df.index)))

    h_hroll = res_df['high'].rolling(interval_).max()
    h_lroll = res_df['low'].rolling(interval_).min()

    hclose = res_df['close'].iloc[np.argwhere(np_timeidx % interval_ == interval_ - 1).reshape(-1, )]
    repeated_df = np.repeat(hclose, interval_)
    row_idx = np.argwhere(res_df.index == repeated_df.index[0]).item() + 1
    res_df['hclose_%s' % interval_] = np.nan

    # print(len(repeated_df))
    # print(len(res_df))
    # if len(res_df) >= len(repeated_df):
    #     res_df['hclose_%s' % interval_].iloc[:len(repeated_df)] = repeated_df.shift(row_idx - interval_).values
    # else:
    #     res_df['hclose_%s' % interval_] = repeated_df.shift(row_idx - interval_).values[:len(res_df)]
    res_df['hclose_%s' % interval_].iloc[row_idx:] = repeated_df.shift(1 - interval_).values[
                                                     :len(res_df['hclose_%s' % interval_].iloc[row_idx:])]

    hhigh = h_hroll.iloc[np.argwhere(np_timeidx % interval_ == interval_ - 1).reshape(-1, )]
    repeated_df = np.repeat(hhigh, interval_)
    res_df['hhigh_%s' % interval_] = np.nan
    res_df['hhigh_%s' % interval_].iloc[row_idx:] = repeated_df.shift(1 - interval_).values[
                                                     :len(res_df['hhigh_%s' % interval_].iloc[row_idx:])]

    hlow = h_lroll.iloc[np.argwhere(np_timeidx % interval_ == interval_ - 1).reshape(-1, )]
    repeated_df = np.repeat(hlow, interval_)
    res_df['hlow_%s' % interval_] = np.nan
    res_df['hlow_%s' % interval_].iloc[row_idx:] = repeated_df.shift(1 - interval_).values[
                                                    :len(res_df['hlow_%s' % interval_].iloc[row_idx:])]
    # length unmatch error occurs
    # res_df['hclose_%s' % interval_] = repeated_df.shift(row_idx - interval_).values[:len(res_df)]

    hopen = res_df['open'].iloc[np.argwhere(np_timeidx % interval_ == 0).reshape(-1, )]
    repeated_df = np.repeat(hopen, interval_)
    row_idx = np.argwhere(res_df.index == repeated_df.index[0]).item()
    res_df['hopen_%s' % interval_] = np.nan
    # res_df['hopen_%s' % interval_].iloc[row_idx:] = repeated_df.values
    res_df['hopen_%s' % interval_].iloc[row_idx:] = repeated_df.values[
                                                    :len(res_df['hopen_%s' % interval_].iloc[row_idx:])]

    return res_df


def cct_bbo(df, period, smooth):
    avg_ = df['close'].rolling(period).mean()
    stdev_ = stdev(df, period)
    # print("len(stdev_) :", len(stdev_))
    # print("len(stdev_.apply(lambda x: 2 * x)) :", len(stdev_.apply(lambda x: 2 * x)))
    cctbbo = 100 * (df['close'] + stdev_.apply(lambda x: 2 * x) - avg_) / (stdev_.apply(lambda x: 4 * x))
    ema_cctbbo = ema(cctbbo, smooth)

    return cctbbo, ema_cctbbo


def donchian_channel(df, period):

    hh = df['high'].rolling(period).max()
    ll = df['low'].rolling(period).min()
    base = (hh + ll) / 2

    return hh, ll, base


def bb_width(df, period, multiple):

    basis = df['close'].rolling(period).mean()
    # print(stdev(df, period))
    # quit()
    dev_ = multiple * stdev(df, period)
    upper = basis + dev_
    lower = basis - dev_
    bbw = 2 * dev_ / basis

    return upper, lower, bbw


def dc_line(ltf_df, htf_df, interval, dc_period=20):

    dc_upper = 'dc_upper_%s' % interval
    dc_lower = 'dc_lower_%s' % interval
    dc_base = 'dc_base_%s' % interval

    if interval != '1m':
        htf_df[dc_upper], htf_df[dc_lower], htf_df[dc_base] = donchian_channel(htf_df, dc_period)
        joined_ltf_df = ltf_df.join(
            pd.DataFrame(index=ltf_df.index, data=to_lower_tf(ltf_df, htf_df, [i for i in range(-3, 0, 1)]),
                         columns=[dc_upper, dc_lower, dc_base]))
    else:
        joined_ltf_df = ltf_df.copy()
        joined_ltf_df[dc_upper], joined_ltf_df[dc_lower], joined_ltf_df[dc_base] = donchian_channel(ltf_df, dc_period)

    return joined_ltf_df


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

    joined_ltf_df = ltf_df.join(
        pd.DataFrame(index=ltf_df.index, data=to_lower_tf(ltf_df, htf_df, [i for i in range(-9, 0, 1)]),
                     columns=[st1_up, st1_down, st1_trend
                         , st2_up, st2_down, st2_trend
                         , st3_up, st3_down, st3_trend]))

    return joined_ltf_df


def bb_line(ltf_df, htf_df, interval):

    bb_upper = 'bb_upper_%s' % interval
    bb_lower = 'bb_lower_%s' % interval

    if interval != '1m':
        htf_df[bb_upper], htf_df[bb_lower], _ = bb_width(htf_df, 20, 1)
        joined_ltf_df = ltf_df.join(
            pd.DataFrame(index=ltf_df.index, data=to_lower_tf(ltf_df, htf_df, [i for i in range(-2, 0, 1)]),
                         columns=[bb_upper, bb_lower]))
    else:
        joined_ltf_df = ltf_df.copy()
        joined_ltf_df[bb_upper], joined_ltf_df[bb_lower], _ = bb_width(ltf_df, 20, 1)

    return joined_ltf_df


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


def bbwp(bb_gap, st_gap, ma_period=10):

    bbwp_ = bb_gap / st_gap
    bbwp_ma = bbwp_.rolling(ma_period).mean()

    return bbwp_, bbwp_ma


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


def cmo(df, period=9):

    df['closegap_cunsum'] = (df['close'] - df['close'].shift(1)).cumsum()
    df['closegap_abs_cumsum'] = abs(df['close'] - df['close'].shift(1)).cumsum()
    # print(df)

    df['CMO'] = (df['closegap_cunsum'] - df['closegap_cunsum'].shift(period)) / (
            df['closegap_abs_cumsum'] - df['closegap_abs_cumsum'].shift(period)) * 100

    del df['closegap_cunsum']
    del df['closegap_abs_cumsum']

    return df['CMO']


def rma(series, period):

    alpha = 1 / period
    rma = pd.Series(index=series.index)
    rma.iloc[0] = 0
    for i in range(1, len(series)):
        if np.isnan(rma.iloc[i - 1]):
            rma.iloc[i] = series.rolling(period).mean().iloc[i]
        else:
            rma.iloc[i] = series.iloc[i] * alpha + (1 - alpha) * nz(rma.iloc[i - 1])

    return rma


def rsi(ohlcv_df, period=14):

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


def macd(df, short=9, long=19, signal=9):
    macd = df['close'].ewm(span=short, min_periods=short - 1, adjust=False).mean() - \
                 df['close'].ewm(span=long, min_periods=long - 1, adjust=False).mean()
    macd_signal = macd.ewm(span=signal, min_periods=signal - 1, adjust=False).mean()
    macd_hist = macd - macd_signal

    return macd_hist


def ema_ribbon(df, ema_1=5, ema_2=8, ema_3=13):

    ema1 = df['close'].ewm(span=ema_1, min_periods=ema_1 - 1, adjust=False).mean()
    ema2 = df['close'].ewm(span=ema_2, min_periods=ema_2 - 1, adjust=False).mean()
    ema3 = df['close'].ewm(span=ema_3, min_periods=ema_3 - 1, adjust=False).mean()

    return ema1, ema2, ema3


def ema_cross(df, ema_1=30, ema_2=60):
    df['EMA_1'] = df['close'].ewm(span=ema_1, min_periods=ema_1 - 1, adjust=False).mean()
    df['EMA_2'] = df['close'].ewm(span=ema_2, min_periods=ema_2 - 1, adjust=False).mean()

    return


def atr(df, period):
    tr = pd.Series(index=df.index)
    tr.iloc[0] = df['high'].iloc[0] - df['low'].iloc[0]
    for i in range(1, len(df)):
        if pd.isna(df['high'].iloc[i - 1]):
            tr.iloc[i] = df['high'].iloc[i] - df['low'].iloc[i]
        else:
            tr.iloc[i] = max(
                max(df['high'].iloc[i] - df['low'].iloc[i], abs(df['high'].iloc[i] - df['close'].iloc[i - 1])),
                abs(df['low'].iloc[i] - df['close'].iloc[i - 1]))
    # print(tr)
    # quit()
    atr = rma(tr, period)

    return atr


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
            atr_down[i] = max(atr_down[i], atr_down[i - 1])
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


# def mmh_st(df, mp1, mp2, pd1=10, pd2=10):   # makemoney_hybrid
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


def lt_trend(df, lambda_bear=0.06, lambda_bull=0.08):
    df['LT_Trend'] = np.NaN
    t_zero = 0
    trend_state = None
    for i in range(1, len(df)):
        pmax = df['close'].iloc[t_zero:i].max()
        pmin = df['close'].iloc[t_zero:i].min()
        delta_bear = (pmax - df['close'].iloc[i]) / pmax
        delta_bull = (df['close'].iloc[i] - pmin) / pmin
        # print(pmax, pmin, delta_bear, delta_bull)

        if delta_bear > lambda_bear and trend_state != 'Bear':
            t_peak = df['close'].iloc[t_zero:i].idxmax()
            t_peak = df.index.to_list().index(t_peak)
            df['LT_Trend'].iloc[t_zero + 1:t_peak + 1] = 'Bull'
            t_zero = t_peak
            trend_state = 'Bear'

        elif delta_bull > lambda_bull and trend_state != 'Bull':
            t_trough = df['close'].iloc[t_zero:i].idxmin()
            t_trough = df.index.to_list().index(t_trough)
            df['LT_Trend'].iloc[t_zero + 1:t_trough + 1] = 'Bear'
            t_zero = t_trough
            trend_state = 'Bull'

        if i == len(df) - 1:
            if pd.isnull(df['LT_Trend'].iloc[i]):
                back_i = i
                while True:
                    back_i -= 1
                    if not pd.isnull(df['LT_Trend'].iloc[back_i]):
                        if df['LT_Trend'].iloc[back_i] == 'Bull':
                            df['LT_Trend'].iloc[back_i + 1:] = 'Bear'
                        else:
                            df['LT_Trend'].iloc[back_i + 1:] = 'Bull'
                        break

            df['LT_Trend'].iloc[0] = df['LT_Trend'].iloc[1]


from scipy.ndimage.filters import gaussian_filter1d


def support_line(df, sigma=10):
    #           Gaussian Curve          #
    df['Gaussian_close'] = gaussian_filter1d(df['close'], sigma=sigma)
    df['Gaussian_close_Trend'] = np.where(df['Gaussian_close'] > df['Gaussian_close'].shift(1), 1, 0)

    df['Support_Line'] = np.NaN
    i = 0
    while i < len(df):
        # print('i :', i)
        #           Find Red           #
        if not df['Gaussian_close_Trend'].iloc[i]:
            #     Find end of the Red       #
            for j in range(i + 1, len(df)):
                if df['Gaussian_close_Trend'].iloc[j]:
                    min_price = df['low'].iloc[i:j].min()
                    # print('i, j, min_price :', i, j, min_price)
                    #       Find Red to Green          #
                    for k in range(j + 1, len(df)):
                        if df['Gaussian_close_Trend'].iloc[k] == 1 and df['Gaussian_close_Trend'].iloc[k - 1] == 0:
                            df['Support_Line'].iloc[j:k] = min_price
                            # print('One S.P Drawing Done!')
                            i = j
                            break
                        else:
                            if k == len(df) - 1:
                                df['Support_Line'].iloc[j:] = min_price
                                i = len(df)
                                break
                    break
        else:
            i += 1

    return
