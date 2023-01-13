import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy import stats, signal
from ast import literal_eval
# from numba import jit, njit
# import torch


class OrderSide:
    BUY = "BUY"
    SELL = "SELL"
    INVALID = None


def ffill_line(line_, idx_):
    len_line = len(line_)
    total_en_idx = np.zeros(len_line)
    total_en_idx[idx_] = 1
    idx_line_ = np.where(total_en_idx, line_, np.nan)
    idx_line = pd.Series(idx_line_).ffill().to_numpy()

    return idx_line


def get_wave_bias_v2(res_df, config, high, low, len_df, short_obj, long_obj):
    short_op_idx = short_obj[-1].astype(int)
    short_en_idx = short_obj[2].astype(int)
    short_en_tp1 = ffill_line(res_df['short_wave_1_{}'.format(config.selection_id)].to_numpy(), short_op_idx)  # en_idx 에 sync 된 open_idx 를 사용해야함
    short_en_out0 = ffill_line(res_df['short_wave_0_{}'.format(config.selection_id)].to_numpy(), short_op_idx)

    long_op_idx = long_obj[-1].astype(int)
    long_en_idx = long_obj[2].astype(int)
    long_en_tp1 = ffill_line(res_df['long_wave_1_{}'.format(config.selection_id)].to_numpy(), long_op_idx)
    long_en_out0 = ffill_line(res_df['long_wave_0_{}'.format(config.selection_id)].to_numpy(), long_op_idx)

    bias_info_tick = config.tr_set.bias_info_tick

    # 1. min 에 초점을 맞추는 거니까, touch 없을시 len_df 로 설정
    # 2. future_data 사용이니까, shift(-bias_info_tick) 설정
    # 3. entry 다음 idx 부터 -> tp & out 체결 logic 이 현재 entry_idx 부터 되어있어서 취소
    len_df_range = np.arange(len_df)
    last_idx = len_df - 1
    shift_range = bias_info_tick - 1  # entry_idx 까지 포함해서 wave_bias check
    short_en_tp1_touch_idx = \
    pd.Series(np.where(low <= short_en_tp1, len_df_range, last_idx)).rolling(bias_info_tick, min_periods=1).min().shift(-shift_range).to_numpy()[
        short_en_idx]
    short_en_out0_touch_idx = \
    pd.Series(np.where(high >= short_en_out0, len_df_range, last_idx)).rolling(bias_info_tick, min_periods=1).min().shift(-shift_range).to_numpy()[
        short_en_idx]

    long_en_tp1_touch_idx = \
    pd.Series(np.where(high >= long_en_tp1, len_df_range, last_idx)).rolling(bias_info_tick, min_periods=1).min().shift(-shift_range).to_numpy()[
        long_en_idx]
    long_en_out0_touch_idx = \
    pd.Series(np.where(low <= long_en_out0, len_df_range, last_idx)).rolling(bias_info_tick, min_periods=1).min().shift(-shift_range).to_numpy()[
        long_en_idx]

    short_true_bias_idx = short_en_tp1_touch_idx < short_en_out0_touch_idx  # true_bias 의 조건
    long_true_bias_idx = long_en_tp1_touch_idx < long_en_out0_touch_idx

    short_false_bias_idx = short_en_tp1_touch_idx >= short_en_out0_touch_idx  # false_bias 의 조건
    long_false_bias_idx = long_en_tp1_touch_idx >= long_en_out0_touch_idx

    # return short_true_bias_idx.ravel(), short_false_bias_idx.ravel(), long_true_bias_idx.ravel(), long_false_bias_idx.ravel()
    return short_true_bias_idx, short_false_bias_idx, long_true_bias_idx, long_false_bias_idx, short_en_tp1[short_en_idx], short_en_out0[
        short_en_idx], long_en_tp1[long_en_idx], long_en_out0[long_en_idx]

def get_wave_bias(res_df, config, high, low, len_df, short_obj, long_obj):
    short_op_idx = short_obj[-1].astype(int)
    short_en_idx = short_obj[2].astype(int)
    short_en_tp1 = ffill_line(res_df['short_tp_1_{}'.format(config.selection_id)].to_numpy(), short_op_idx)  # en_idx 에 sync 된 open_idx 를 사용해야함
    short_en_out0 = ffill_line(res_df['short_epout_0_{}'.format(config.selection_id)].to_numpy(), short_op_idx)

    long_op_idx = long_obj[-1].astype(int)
    long_en_idx = long_obj[2].astype(int)
    long_en_tp1 = ffill_line(res_df['long_tp_1_{}'.format(config.selection_id)].to_numpy(), long_op_idx)
    long_en_out0 = ffill_line(res_df['long_epout_0_{}'.format(config.selection_id)].to_numpy(), long_op_idx)

    bias_info_tick = config.tr_set.bias_info_tick

    # 1. min 에 초점을 맞추는 거니까, touch 없을시 len_df 로 설정
    # 2. future_data 사용이니까, shift(-bias_info_tick) 설정
    # 3. entry 다음 idx 부터 -> tp & out 체결 logic 이 현재 entry_idx 부터 되어있어서 취소
    last_idx = len_df - 1
    short_en_tp1_touch_idx = pd.Series(np.where(low <= short_en_tp1, np.arange(len_df), last_idx)).rolling(bias_info_tick, min_periods=1).min().shift(
        -bias_info_tick).to_numpy()[short_en_idx]
    short_en_out0_touch_idx = \
    pd.Series(np.where(high >= short_en_out0, np.arange(len_df), last_idx)).rolling(bias_info_tick, min_periods=1).min().shift(
        -bias_info_tick).to_numpy()[short_en_idx]

    long_en_tp1_touch_idx = pd.Series(np.where(high >= long_en_tp1, np.arange(len_df), last_idx)).rolling(bias_info_tick, min_periods=1).min().shift(
        -bias_info_tick).to_numpy()[long_en_idx]
    long_en_out0_touch_idx = pd.Series(np.where(low <= long_en_out0, np.arange(len_df), last_idx)).rolling(bias_info_tick, min_periods=1).min().shift(
        -bias_info_tick).to_numpy()[long_en_idx]

    short_true_bias_idx = short_en_tp1_touch_idx < short_en_out0_touch_idx  # true_bias 의 조건
    long_true_bias_idx = long_en_tp1_touch_idx < long_en_out0_touch_idx

    short_false_bias_idx = short_en_tp1_touch_idx > short_en_out0_touch_idx  # false_bias 의 조건
    long_false_bias_idx = long_en_tp1_touch_idx > long_en_out0_touch_idx

    # return short_true_bias_idx.ravel(), short_false_bias_idx.ravel(), long_true_bias_idx.ravel(), long_false_bias_idx.ravel()
    return short_true_bias_idx, short_false_bias_idx, long_true_bias_idx, long_false_bias_idx, short_en_tp1[short_en_idx], short_en_out0[
        short_en_idx], long_en_tp1[long_en_idx], long_en_out0[long_en_idx]


def kde_plot_v2(v, c, kde_factor=0.15, num_samples=100):

  # start_0 = time.time()
  kde = stats.gaussian_kde(v,weights=c,bw_method=kde_factor)
  kdx = np.linspace(v.min(),v.max(),num_samples)
  kdy = kde(kdx)
  ticks_per_sample = (kdx.max() - kdx.min()) / num_samples
  # print("ticks_per_sample :", ticks_per_sample)  # sample 당 가격
  # print("kdy elapsed_time :", time.time() - start_0)

  hist_res = plt.hist(v, weights=c, bins=num_samples, alpha=.8, edgecolor='black')
  # print("len(hist_res) :", len(hist_res))
  # print("hist_res[1] :", hist_res[1])
  # print("hist_res[2] :", hist_res[2])

  peaks,_ = signal.find_peaks(kdy)
  pkx = kdx[peaks]
  rescaled_kdy = kdy * (hist_res[0].max() / kdy.max())
  # pky = kdy[peaks]
  pky = rescaled_kdy[peaks]
  print("pkx :", pkx)

  # plt.figure(figsize=(10,5))

  # plt.plot(kdx, kdy, color='white')
  plt.plot(kdx, rescaled_kdy, color='white')
  plt.plot(pkx, pky, 'bo', color='yellow')
  # plt.show()


def kde_plot(plot_data, kde_factor=0.15, num_samples=100):
  v, c = np.unique(plot_data, return_counts=True)

  # start_0 = time.time()
  kde = stats.gaussian_kde(v,weights=c,bw_method=kde_factor)
  kdx = np.linspace(v.min(),v.max(),num_samples)
  kdy = kde(kdx)
  ticks_per_sample = (kdx.max() - kdx.min()) / num_samples
  # print("ticks_per_sample :", ticks_per_sample)  # sample 당 가격
  # print("kdy elapsed_time :", time.time() - start_0)

  peaks,_ = signal.find_peaks(kdy)
  pkx = kdx[peaks]
  pky = kdy[peaks]

  # plt.figure(figsize=(10,5))
  plt.hist(v, weights=c, bins=num_samples, alpha=.8, edgecolor='black')
  plt.plot(kdx, kdy, color='white')
  plt.plot(pkx, pky, 'bo', color='yellow')
  # plt.show()

def get_max_outg_v4(open_side, config, ohlc_list, obj, tpout_arr, tp_0, out_gap):

    h, l = ohlc_list[1:3]

    np_obj = np.array(obj).T[0]
    assert len(np_obj.shape) == 2

    _, _, en_idxs, ex_idxs, open_idxs = obj

    if open_side == "SELL":
        max_high = np.array([np.max(h[int(en_idx):int(ex_idx + 1)]) for en_idx, ex_idx in zip(en_idxs, ex_idxs)]).reshape(-1, 1)  # reshape is important in np boradcasting   # outg 라서, iin + 1 이 아님
        max_outg = (tp_0 - max_high) / out_gap
    else:
        min_low = np.array([np.min(l[int(en_idx):int(ex_idx + 1)]) for en_idx, ex_idx in zip(en_idxs, ex_idxs)]).reshape(-1, 1)  # outg 라서, iin + 1 이 아님
        max_outg = (min_low - tp_0) / out_gap  # out_idx 포함

    return max_outg

def get_max_outg_v3(open_side, config, ohlc_list, obj, tpout_arr, epout_0, out_gap):
    h, l = ohlc_list[1:3]

    np_obj = np.array(obj).T[0]
    assert len(np_obj.shape) == 2

    # iin == iout 인 경우 분리
    # en_idx = np_obj[:, 2]
    # ex_idx = np_obj[:, 3]
    # equal_idx = en_idx == ex_idx

    _, _, en_idxs, ex_idxs, open_idxs = obj

    if open_side == "SELL":
        idxgs = [np.argwhere(l[int(en_idx):int(en_idx + config.tr_set.bias_info_tick)] <= tpout_arr[i, 0]) for i, en_idx in enumerate(en_idxs)]
    else:
        idxgs = [np.argwhere(h[int(en_idx):int(en_idx + config.tr_set.bias_info_tick)] >= tpout_arr[i, 0]) for i, en_idx in enumerate(en_idxs)]

    min_idxg = np.array([gaps.min() if len(gaps) != 0 else np.nan for gaps in idxgs])  # get 최소 idx_gap from en_idx
    nan_idx = np.isnan(min_idxg)  # .astype(int) -> false swing_bias idx
    # min_idxg[nan_idx] = 0  # fill na 0, for idex summation below

    if open_side == "SELL":
        # max_high = np.array([np.max(h[int(en_idx):int(en_idx + idx_gap + 1)]) for en_idx, idx_gap in zip(en_idxs, min_idxg)]).reshape(-1, 1)  # reshape is important in np boradcasting
        # max_high = np.array([np.max(h[int(en_idx):int(en_idx + idx_gap + 1)]) if not np.isnan(idx_gap) else np.max(h[int(en_idx):int(ex_idx + 1)])
        max_high = np.array([np.max(h[int(en_idx):int(en_idx + idx_gap + 1)]) if not np.isnan(idx_gap) else np.max(
            h[int(en_idx):int(en_idx + config.tr_set.bias_info_tick)])
                             for en_idx, ex_idx, idx_gap in zip(en_idxs, ex_idxs, min_idxg)]).reshape(-1,
                                                                                                      1)  # reshape is important in np boradcasting   # outg 라서, iin + 1 이 아님
        max_outg = (max_high - epout_0) / out_gap
    else:
        # min_low = np.array([np.min(l[int(en_idx):int(en_idx + idx_gap + 1)]) for en_idx, idx_gap in zip(en_idxs, min_idxg)]).reshape(-1, 1)
        # min_low = np.array([np.min(l[int(en_idx):int(en_idx + idx_gap + 1)]) if not np.isnan(idx_gap) else np.min(l[int(en_idx):int(ex_idx + 1)])
        min_low = np.array([np.min(l[int(en_idx):int(en_idx + idx_gap + 1)]) if not np.isnan(idx_gap) else np.min(
            l[int(en_idx):int(en_idx + config.tr_set.bias_info_tick)])
                            for en_idx, ex_idx, idx_gap in zip(en_idxs, ex_idxs, min_idxg)]).reshape(-1, 1)  # outg 라서, iin + 1 이 아님
        max_outg = (epout_0 - min_low) / out_gap  # out_idx 포함

    return max_outg, open_idxs.astype(int), ~nan_idx.astype(bool).reshape(-1, 1)  # true_bias 의 outg data 만 사용

def get_max_outg_v2(open_side, config, ohlc_list, obj, tpout_arr, epout_0, out_gap):

    h, l = ohlc_list[1:3]
    _, _, en_idxs, _, open_idxs = obj

    if open_side == "SELL":
      idxgs = [np.argwhere(l[int(en_idx):int(en_idx + config.tr_set.bias_info_tick)] <= tpout_arr[i, 0]) for i, en_idx in enumerate(en_idxs)]
    else:
      idxgs = [np.argwhere(h[int(en_idx):int(en_idx + config.tr_set.bias_info_tick)] >= tpout_arr[i, 0]) for i, en_idx in enumerate(en_idxs)]

    min_idxg = np.array([gaps.min() if len(gaps) != 0 else np.nan for gaps in idxgs])  # get 최소 idx_gap from en_idx
    nan_idx = np.isnan(min_idxg) # .astype(int)
    min_idxg[nan_idx] = 0  # fill na 0, for idex summation below

    if open_side == "SELL":
      max_high = np.array([np.max(h[int(en_idx):int(en_idx + idx_gap + 1)]) for en_idx, idx_gap in zip(en_idxs, min_idxg)]).reshape(-1, 1)  # reshape is important in np boradcasting
      max_outg = (max_high - epout_0) / out_gap
    else:
      max_low = np.array([np.min(l[int(en_idx):int(en_idx + idx_gap + 1)]) for en_idx, idx_gap in zip(en_idxs, min_idxg)]).reshape(-1, 1)
      max_outg = (epout_0 - max_low) / out_gap # out_idx 포함

    return max_outg[~nan_idx], open_idxs[~nan_idx].astype(int)  # true_bias 의 outg data 만 사용

def get_max_outg(open_side, config, ohlc_list, obj, tpout_arr, out_gap):

    h, l = ohlc_list[1:3]
    en_p, _, en_idxs, _, open_idxs = obj

    if open_side == "SELL":
      idxgs = [np.argwhere(l[int(en_idx):int(en_idx + config.tr_set.bias_info_tick)] <= tpout_arr[i, 0]) for i, en_idx in enumerate(en_idxs)]
    else:
      idxgs = [np.argwhere(h[int(en_idx):int(en_idx + config.tr_set.bias_info_tick)] >= tpout_arr[i, 0]) for i, en_idx in enumerate(en_idxs)]

    min_idxg = np.array([gaps.min() if len(gaps) != 0 else np.nan for gaps in idxgs])  # get 최소 idx_gap from en_idx
    nan_idx = np.isnan(min_idxg) # .astype(int)
    min_idxg[nan_idx] = 0  # fill na 0, for idex summation below

    if open_side == "SELL":
      max_outg = np.array([np.max(h[int(en_idx):int(en_idx + idx_gap + 1)]) - ep_ for en_idx, idx_gap, ep_ in zip(en_idxs, min_idxg, en_p)]) / out_gap
    else:
      max_outg = np.array([ep_ - np.min(l[int(en_idx):int(en_idx + idx_gap + 1)]) for en_idx, idx_gap, ep_ in zip(en_idxs, min_idxg, en_p)]) / out_gap # out_idx 포함

    return max_outg[~nan_idx], open_idxs[~nan_idx].astype(int)  # true_bias 의 outg data 만 사용

def get_max_tpg_v2(open_side, ohlc_list, pr_, obj, tp_1, tp_gap):  # much faster

    h, l = ohlc_list[1:3]

    # en_p = obj[0]
    # ex_p = obj[1]

    np_obj = np.array(obj).T[0]
    assert len(np_obj.shape) == 2

    # iin == iout 인 경우 분리
    en_idx = np_obj[:, 2]
    ex_idx = np_obj[:, 3]
    equal_idx = en_idx == ex_idx

    if open_side == "SELL":
        min_low = np.full_like(tp_1, np.nan)
        # min_low[~equal_idx] = np.array([np.min(l[int(iin + 1):int(iout + 1)]) for _, _, iin, iout in np_obj[~equal_idx, :4]]).reshape(-1, 1)  # start from iin + 1
        min_low[~equal_idx] = np.array([np.min(l[int(iin + 1):int(iout + 1)]) for _, _, iin, iout in np_obj[~equal_idx, :4]]).reshape(-1, 1)  # start from iin + 1
        max_tpg = (tp_1 - min_low) / tp_gap
    else:
        max_high = np.full_like(tp_1, np.nan)
        # max_high[~equal_idx] = np.array([np.max(h[int(iin + 1):int(iout + 1)]) for _, _, iin, iout in np_obj[~equal_idx, :4]]).reshape(-1, 1)
        max_high[~equal_idx] = np.array([np.max(h[int(iin + 1):int(iout + 1)]) for _, _, iin, iout in np_obj[~equal_idx, :4]]).reshape(-1, 1)
        max_tpg = (max_high - tp_1) / tp_gap

    return max_tpg[pr_ < 1]  # out 된 case 만 tpg 담음
    # return max_tpg

def get_max_tpg(open_side, ohlc_list, pr_, obj, tp_gap):  # much faster

    h, l = ohlc_list[1:3]

    en_p = obj[0]
    # ex_p = obj[1]

    np_obj = np.array(obj).T[0]
    assert len(np_obj.shape) == 2

    # iin == iout 인 경우 분리
    en_idx = np_obj[:, 2]
    ex_idx = np_obj[:, 3]
    equal_idx = en_idx == ex_idx

    if open_side == "SELL":
        min_low = np.full_like(en_p, np.nan)
        min_low[~equal_idx] = np.array([np.min(l[int(iin + 1):int(iout + 1)]) for _, _, iin, iout in np_obj[~equal_idx, :4]]).reshape(-1, 1)  # start from iin + 1
        max_tpg = (en_p - min_low) / tp_gap
    else:
        max_high = np.full_like(en_p, np.nan)
        max_high[~equal_idx] = np.array([np.max(h[int(iin + 1):int(iout + 1)]) for _, _, iin, iout in np_obj[~equal_idx, :4]]).reshape(-1, 1)
        max_tpg = (max_high - en_p) / tp_gap

    return max_tpg[pr_ < 1]  # out 된 case 만 tpg 담음
    # return max_tpg

# def get_max_tpg(open_side, ohlc_list, pr_, obj_, rtc_gap):  # much faster
#     if type(obj_) == list:
#         obj_ = list(zip(*obj_))
#
#     h, l = ohlc_list[1:3]
#
#     if open_side == "SELL":
#       max_tpg = np.array([ep_ - np.min(l[int(iin):int(iout + 1)]) for ep_, _, iin, iout in obj_]) / rtc_gap # out 포함
#       # max_outg = np.array([np.max(h[int(iin):int(iout + 1)]) - ep_ for ep_, _, iin, iout in obj_]) / rtc_gap
#     else:
#       max_tpg = np.array([np.max(h[int(iin):int(iout + 1)]) - ep_ for ep_, _, iin, iout in obj_]) / rtc_gap
#       # max_outg = np.array([ep_ - np.min(l[int(iin):int(iout + 1)]) for ep_, _, iin, iout in obj_]) / rtc_gap
#
#     return max_tpg[pr_ < 1]  # out 된 case 만 tpg 담음

def precision(pr_list, true_idx):   # true_pr in true_bias / true_bias
  true_bias_pr = pr_list[true_idx].ravel()
  return np.sum(true_bias_pr > 1) / len(true_bias_pr)  # 차원을 고려한 계산

def wave_bias(true_idx, false_idx):  # 정확하게 하려고, true & false 로 기준함
    true_sum = np.sum(true_idx)
    false_sum = np.sum(false_idx)
    return true_sum / (true_sum + false_sum)

def recall(true_idx):   # true_bias / total_entry
    return np.sum(true_idx) / len(true_idx) #  2.16 µs per loop (len) --> 3.78 µs per loop

def mr_res(input_data, rpsn_v, inval_v, np_ones):
    return input_data == rpsn_v if rpsn_v > inval_v else np_ones


def idep_plot_v14(res_df, len_df, config, high, low, open_idx, side_arr, paired_res, inversion=False, sample_ratio=0.7, title_position=(0.5, 0.5),
                  fontsize=15, signi=False):
    if not signi:
        plt.style.use(['dark_background', 'fast'])
        fig = plt.figure(figsize=(24, 8))
        gs = gridspec.GridSpec(nrows=2,  # row 몇 개
                               ncols=3,  # col 몇 개
                               height_ratios=[10, 1]
                               # height_ratios=[10, 10, 1]
                               )
    gs_idx = 0
    # plt.suptitle(key)

    p_ranges, p_qty_ratio = literal_eval(config.tp_set.p_ranges), literal_eval(config.tp_set.p_qty_ratio)
    assert np.sum(p_qty_ratio) == 1.0
    assert len(p_ranges) == len(p_qty_ratio)

    if sample_ratio is not None:
        sample_len = int(len_df * sample_ratio)
    else:
        sample_len = len_df

    # ------ short & long data preparation ------ #
    # start_0 = time.time()
    point1_arr, valid_openi_arr, pair_idx_arr, pair_price_arr, lvrg_arr, fee_arr, tpout_arr, tr_arr = paired_res
    assert len(valid_openi_arr) != 0, "assert len(valid_openi_arr) != 0"
    short_valid_openi_idx = np.where(side_arr[valid_openi_arr] == OrderSide.SELL)[0]  # valid_openi_arr 에 대한 idx, # side_arr,
    long_valid_openi_idx = np.where(side_arr[valid_openi_arr] == OrderSide.BUY)[0]

    valid_open_idx = open_idx[valid_openi_arr].reshape(-1, 1)

    short_obj = np.hstack((pair_price_arr, pair_idx_arr, valid_open_idx))[short_valid_openi_idx]
    long_obj = np.hstack((pair_price_arr, pair_idx_arr, valid_open_idx))[long_valid_openi_idx]
    both_obj = np.vstack((short_obj, long_obj))
    print("short_obj.shape :", short_obj.shape)
    print("long_obj.shape :", long_obj.shape)

    short_obj, long_obj, both_obj = [np.split(obj_, 5, axis=1) for obj_ in [short_obj, long_obj, both_obj]]

    short_point1_arr, long_point1_arr = [point1_arr[openi_idx_].reshape(-1, 1) for openi_idx_ in [short_valid_openi_idx, long_valid_openi_idx]]
    short_lvrg_arr, long_lvrg_arr = [lvrg_arr[openi_idx_].reshape(-1, 1) for openi_idx_ in [short_valid_openi_idx, long_valid_openi_idx]]
    short_fee_arr, long_fee_arr = [fee_arr[openi_idx_].reshape(-1, 1) for openi_idx_ in [short_valid_openi_idx, long_valid_openi_idx]]
    short_tpout_arr, long_tpout_arr = [tpout_arr[openi_idx_] for openi_idx_ in [short_valid_openi_idx, long_valid_openi_idx]]
    # short_bias_arr, long_bias_arr = [bias_arr[openi_idx_] for openi_idx_ in [short_valid_openi_idx, long_valid_openi_idx]]
    short_tr_arr, long_tr_arr = [tr_arr[openi_idx_] for openi_idx_ in [short_valid_openi_idx, long_valid_openi_idx]]
    # print("long_bias_arr.shape :", long_bias_arr.shape)
    # print("elapsed time :", time.time() - start_0)

    short_true_bias_idx, short_false_bias_idx, long_true_bias_idx, long_false_bias_idx, short_en_tp1, short_en_out0, long_en_tp1, long_en_out0 = \
        get_wave_bias_v2(res_df, config, high, low, len_df, short_obj, long_obj)

    len_short, len_long = len(short_valid_openi_idx), len(long_valid_openi_idx)

    # ------ plot_data ------ #
    try:
        # start_0 = time.time()
        if len_short == 0:
            short_pr = []
            gs_idx += 1
        else:
            short_tr = short_tr_arr.mean()
            short_pr, short_liqd = get_pr_v4(OrderSide.SELL, high, low, short_obj, short_tpout_arr, short_lvrg_arr, short_fee_arr, p_ranges,
                                             p_qty_ratio, inversion)
            short_total_pr = to_total_pr(len_df, short_pr, short_obj[-2])
            short_cum_pr = np.cumprod(short_total_pr)
            # short_liqd = liquidation_v2(OrderSide.SELL, h, short_obj[:4], short_lvrg_arr, short_fee_arr)
            short_prcn, short_rc = precision(short_pr, short_true_bias_idx), wave_bias(short_true_bias_idx, short_false_bias_idx)
            short_trade_ticks = np.mean(short_obj[-2] - short_obj[-1])
            if signi:
                short_idep_res_obj = (short_prcn, short_rc) + get_res_info_nb_v2(sample_len, short_pr, short_total_pr, short_cum_pr, short_liqd)
            else:
                gs_idx = plot_info_v6(gs, gs_idx, sample_len, short_tr, short_prcn, short_rc, short_trade_ticks, short_pr, short_total_pr,
                                      short_cum_pr, short_liqd, short_lvrg_arr[-1], title_position, fontsize)
        # print("elapsed time :", time.time() - start_0)

    except Exception as e:
        gs_idx += 1
        print("error in short plot_data :", e)

    try:
        # start_0 = time.time()
        if len_long == 0:
            long_pr = []
            gs_idx += 1
        else:
            long_tr = long_tr_arr.mean()
            long_pr, long_liqd = get_pr_v4(OrderSide.BUY, high, low, long_obj, long_tpout_arr, long_lvrg_arr, long_fee_arr, p_ranges, p_qty_ratio,
                                           inversion)
            long_total_pr = to_total_pr(len_df, long_pr, long_obj[-2])
            long_cum_pr = np.cumprod(long_total_pr)
            # long_liqd = liquidation_v2(OrderSide.BUY, l, long_obj[:4], long_lvrg_arr, long_fee_arr)
            long_prcn, long_rc = precision(long_pr, long_true_bias_idx), wave_bias(long_true_bias_idx, long_false_bias_idx)
            long_trade_ticks = np.mean(long_obj[-2] - long_obj[-1])
            if signi:
                long_idep_res_obj = (long_prcn, long_rc) + get_res_info_nb_v2(sample_len, long_pr, long_total_pr, long_cum_pr, long_liqd)
            else:
                gs_idx = plot_info_v6(gs, gs_idx, sample_len, long_tr, long_prcn, long_rc, long_trade_ticks, long_pr, long_total_pr, long_cum_pr,
                                      long_liqd, long_lvrg_arr[-1], title_position, fontsize)
        # print("elapsed time :", time.time() - start_0)
    except Exception as e:
        gs_idx += 1
        print("error in long plot_data :", e)

    try:
        # start_0 = time.time()
        if len_short * len_long == 0:
            both_pr = []
            gs_idx += 1
        else:
            both_tr = (short_tr + long_tr) / 2
            both_pr = np.vstack((short_pr, long_pr))  # for 2d arr, obj 를 1d 로 만들지 않는 이상, pr 은 2d 유지될 것
            both_total_pr = to_total_pr(len_df, both_pr, both_obj[-2])
            both_cum_pr = np.cumprod(both_total_pr)
            both_liqd = min(short_liqd, long_liqd)
            both_true_bias_idx = np.vstack((short_true_bias_idx, long_true_bias_idx))  # vstack for 2d arr
            both_false_bias_idx = np.vstack((short_false_bias_idx, long_false_bias_idx))
            both_prcn, both_rc = precision(both_pr, both_true_bias_idx), wave_bias(both_true_bias_idx, both_false_bias_idx)
            both_trade_ticks = np.mean(both_obj[-2] - both_obj[-1])
            if signi:
                both_idep_res_obj = (both_prcn, both_rc) + get_res_info_nb_v2(sample_len, both_pr, both_total_pr, both_cum_pr, both_liqd)
            else:
                gs_idx = plot_info_v6(gs, gs_idx, sample_len, both_tr, both_prcn, both_rc, both_trade_ticks, both_pr, both_total_pr, both_cum_pr,
                                      both_liqd, lvrg_arr[-1], title_position, fontsize)
        # print("elapsed time :", time.time() - start_0)
    except Exception as e:
        gs_idx += 1
        print("error in both plot_data :", e)

    if not signi:
        if len_short * len_long > 0:
            for obj, bias_arr, cum_pr in zip([short_obj, long_obj, both_obj], [short_true_bias_idx, long_true_bias_idx, both_true_bias_idx],
                                             [short_cum_pr, long_cum_pr, both_cum_pr]):
                try:
                    # start_0 = time.time()
                    gs_idx = frq_dev_plot_v4(gs, gs_idx, len_df, sample_len, obj[-2], bias_arr, cum_pr[-1], fontsize)
                    # print("elapsed time :", time.time() - start_0)
                except Exception as e:
                    gs_idx += 1
                    print("error in frq_dev_plot_v3 :", e)
            plt.show()
            plt.close()

        return short_pr, short_obj, short_lvrg_arr, short_fee_arr, short_tpout_arr, short_true_bias_idx, short_false_bias_idx, short_point1_arr, short_en_tp1, short_en_out0, \
               long_pr, long_obj, long_lvrg_arr, long_fee_arr, long_tpout_arr, long_true_bias_idx, long_false_bias_idx, long_point1_arr, long_en_tp1, long_en_out0

    else:
        return [short_idep_res_obj[:-1], long_idep_res_obj[:-1], both_idep_res_obj[:-1]]
    
def idep_plot_v13(res_df, len_df, config, high, low, open_idx, side_arr, paired_res, inversion=False, sample_ratio=0.7, title_position=(0.5, 0.5),
                  fontsize=15, signi=False):
    if not signi:
        plt.style.use(['dark_background', 'fast'])
        fig = plt.figure(figsize=(24, 8))
        gs = gridspec.GridSpec(nrows=2,  # row 몇 개
                               ncols=3,  # col 몇 개
                               height_ratios=[10, 1]
                               # height_ratios=[10, 10, 1]
                               )
    gs_idx = 0
    # plt.suptitle(key)

    p_ranges, p_qty_ratio = literal_eval(config.tp_set.p_ranges), literal_eval(config.tp_set.p_qty_ratio)
    assert np.sum(p_qty_ratio) == 1.0
    assert len(p_ranges) == len(p_qty_ratio)

    if sample_ratio is not None:
        sample_len = int(len_df * sample_ratio)
    else:
        sample_len = len_df

    # ------ short & long data preparation ------ #
    # start_0 = time.time()
    point1_arr, valid_openi_arr, pair_idx_arr, pair_price_arr, lvrg_arr, fee_arr, tpout_arr, tr_arr = paired_res
    assert len(valid_openi_arr) != 0, "assert len(valid_openi_arr) != 0"
    short_valid_openi_idx = np.where(side_arr[valid_openi_arr] == OrderSide.SELL)[0]  # valid_openi_arr 에 대한 idx, # side_arr,
    long_valid_openi_idx = np.where(side_arr[valid_openi_arr] == OrderSide.BUY)[0]

    valid_open_idx = open_idx[valid_openi_arr].reshape(-1, 1)

    short_obj = np.hstack((pair_price_arr, pair_idx_arr, valid_open_idx))[short_valid_openi_idx]
    long_obj = np.hstack((pair_price_arr, pair_idx_arr, valid_open_idx))[long_valid_openi_idx]
    both_obj = np.vstack((short_obj, long_obj))
    print("short_obj.shape :", short_obj.shape)
    print("long_obj.shape :", long_obj.shape)

    short_obj, long_obj, both_obj = [np.split(obj_, 5, axis=1) for obj_ in [short_obj, long_obj, both_obj]]

    short_point1_arr, long_point1_arr = [point1_arr[openi_idx_].reshape(-1, 1) for openi_idx_ in [short_valid_openi_idx, long_valid_openi_idx]]
    short_lvrg_arr, long_lvrg_arr = [lvrg_arr[openi_idx_].reshape(-1, 1) for openi_idx_ in [short_valid_openi_idx, long_valid_openi_idx]]
    short_fee_arr, long_fee_arr = [fee_arr[openi_idx_].reshape(-1, 1) for openi_idx_ in [short_valid_openi_idx, long_valid_openi_idx]]
    short_tpout_arr, long_tpout_arr = [tpout_arr[openi_idx_] for openi_idx_ in [short_valid_openi_idx, long_valid_openi_idx]]
    # short_bias_arr, long_bias_arr = [bias_arr[openi_idx_] for openi_idx_ in [short_valid_openi_idx, long_valid_openi_idx]]
    short_tr_arr, long_tr_arr = [tr_arr[openi_idx_] for openi_idx_ in [short_valid_openi_idx, long_valid_openi_idx]]
    # print("long_bias_arr.shape :", long_bias_arr.shape)
    # print("elapsed time :", time.time() - start_0)

    short_true_bias_idx, short_false_bias_idx, long_true_bias_idx, long_false_bias_idx, short_en_tp1, short_en_out0, long_en_tp1, long_en_out0 = \
        get_wave_bias_v2(res_df, config, high, low, len_df, short_obj, long_obj)

    len_short, len_long = len(short_valid_openi_idx), len(long_valid_openi_idx)

    # ------ plot_data ------ #
    try:
        # start_0 = time.time()
        if len_short == 0:
            short_pr = []
            gs_idx += 1
        else:
            short_tr = short_tr_arr.mean()
            short_pr, short_liqd = get_pr_v4(OrderSide.SELL, high, low, short_obj, short_tpout_arr, short_lvrg_arr, short_fee_arr, p_ranges,
                                             p_qty_ratio, inversion)
            short_total_pr = to_total_pr(len_df, short_pr, short_obj[-2])
            short_cum_pr = np.cumprod(short_total_pr)
            # short_liqd = liquidation_v2(OrderSide.SELL, h, short_obj[:4], short_lvrg_arr, short_fee_arr)
            short_prcn, short_rc = precision(short_pr, short_true_bias_idx), wave_bias(short_true_bias_idx, short_false_bias_idx)
            short_trade_ticks = np.mean(short_obj[-2] - short_obj[-1])
            if signi:
                short_idep_res_obj = (short_prcn, short_rc) + get_res_info_nb_v2(sample_len, short_pr, short_total_pr, short_cum_pr, short_liqd)
            else:
                gs_idx = plot_info_v6(gs, gs_idx, sample_len, short_tr, short_prcn, short_rc, short_trade_ticks, short_pr, short_total_pr,
                                      short_cum_pr, short_liqd, short_lvrg_arr[-1], title_position, fontsize)
        # print("elapsed time :", time.time() - start_0)

    except Exception as e:
        gs_idx += 1
        print("error in short plot_data :", e)

    try:
        # start_0 = time.time()
        if len_long == 0:
            long_pr = []
            gs_idx += 1
        else:
            long_tr = long_tr_arr.mean()
            long_pr, long_liqd = get_pr_v4(OrderSide.BUY, high, low, long_obj, long_tpout_arr, long_lvrg_arr, long_fee_arr, p_ranges, p_qty_ratio,
                                           inversion)
            long_total_pr = to_total_pr(len_df, long_pr, long_obj[-2])
            long_cum_pr = np.cumprod(long_total_pr)
            # long_liqd = liquidation_v2(OrderSide.BUY, l, long_obj[:4], long_lvrg_arr, long_fee_arr)
            long_prcn, long_rc = precision(long_pr, long_true_bias_idx), wave_bias(long_true_bias_idx, long_false_bias_idx)
            long_trade_ticks = np.mean(long_obj[-2] - long_obj[-1])
            if signi:
                long_idep_res_obj = (long_prcn, long_rc) + get_res_info_nb_v2(sample_len, long_pr, long_total_pr, long_cum_pr, long_liqd)
            else:
                gs_idx = plot_info_v6(gs, gs_idx, sample_len, long_tr, long_prcn, long_rc, long_trade_ticks, long_pr, long_total_pr, long_cum_pr,
                                      long_liqd, long_lvrg_arr[-1], title_position, fontsize)
        # print("elapsed time :", time.time() - start_0)
    except Exception as e:
        gs_idx += 1
        print("error in long plot_data :", e)

    try:
        # start_0 = time.time()
        if len_short * len_long == 0:
            both_pr = []
            gs_idx += 1
        else:
            both_tr = (short_tr + long_tr) / 2
            both_pr = np.vstack((short_pr, long_pr))  # for 2d arr, obj 를 1d 로 만들지 않는 이상, pr 은 2d 유지될 것
            both_total_pr = to_total_pr(len_df, both_pr, both_obj[-2])
            both_cum_pr = np.cumprod(both_total_pr)
            both_liqd = min(short_liqd, long_liqd)
            both_true_bias_idx = np.vstack((short_true_bias_idx, long_true_bias_idx))  # vstack for 2d arr
            both_false_bias_idx = np.vstack((short_false_bias_idx, long_false_bias_idx))
            both_prcn, both_rc = precision(both_pr, both_true_bias_idx), wave_bias(both_true_bias_idx, both_false_bias_idx)
            both_trade_ticks = np.mean(both_obj[-2] - both_obj[-1])
            if signi:
                both_idep_res_obj = (both_prcn, both_rc) + get_res_info_nb_v2(sample_len, both_pr, both_total_pr, both_cum_pr, both_liqd)
            else:
                gs_idx = plot_info_v6(gs, gs_idx, sample_len, both_tr, both_prcn, both_rc, both_trade_ticks, both_pr, both_total_pr, both_cum_pr,
                                      both_liqd, lvrg_arr[-1], title_position, fontsize)
        # print("elapsed time :", time.time() - start_0)
    except Exception as e:
        gs_idx += 1
        print("error in both plot_data :", e)

    if not signi:
        if len_short * len_long > 0:
            for obj, cum_pr in zip([short_obj, long_obj, both_obj], [short_cum_pr, long_cum_pr, both_cum_pr]):
                try:
                    # start_0 = time.time()
                    gs_idx = frq_dev_plot_v3(gs, gs_idx, len_df, sample_len, obj[-2], cum_pr[-1], fontsize)
                    # print("elapsed time :", time.time() - start_0)
                except Exception as e:
                    gs_idx += 1
                    print("error in frq_dev_plot_v3 :", e)
            plt.show()
            plt.close()

        return short_pr, short_obj, short_lvrg_arr, short_fee_arr, short_tpout_arr, short_true_bias_idx, short_false_bias_idx, short_point1_arr, short_en_tp1, short_en_out0, \
               long_pr, long_obj, long_lvrg_arr, long_fee_arr, long_tpout_arr, long_true_bias_idx, long_false_bias_idx, long_point1_arr, long_en_tp1, long_en_out0

    else:
        return [short_idep_res_obj[:-1], long_idep_res_obj[:-1], both_idep_res_obj[:-1]]

def idep_plot_v12(res_df, len_df, config, high, low, open_idx, side_arr, paired_res, inversion=False, sample_ratio=0.7, title_position=(0.5, 0.5),
                  fontsize=15, signi=False):
    if not signi:
        plt.style.use(['dark_background', 'fast'])
        fig = plt.figure(figsize=(24, 8))
        gs = gridspec.GridSpec(nrows=2,  # row 몇 개
                               ncols=3,  # col 몇 개
                               height_ratios=[10, 1]
                               # height_ratios=[10, 10, 1]
                               )
    gs_idx = 0
    # plt.suptitle(key)

    p_ranges, p_qty_ratio = literal_eval(config.tp_set.p_ranges), literal_eval(config.tp_set.p_qty_ratio)
    assert np.sum(p_qty_ratio) == 1.0
    assert len(p_ranges) == len(p_qty_ratio)

    if sample_ratio is not None:
        sample_len = int(len_df * sample_ratio)
    else:
        sample_len = len_df

    # ------ short & long data preparation ------ #
    # start_0 = time.time()
    point1_arr, valid_openi_arr, pair_idx_arr, pair_price_arr, lvrg_arr, fee_arr, tpout_arr, tr_arr = paired_res
    assert len(valid_openi_arr) != 0, "assert len(valid_openi_arr) != 0"
    short_valid_openi_idx = np.where(side_arr[valid_openi_arr] == OrderSide.SELL)[0]  # valid_openi_arr 에 대한 idx, # side_arr,
    long_valid_openi_idx = np.where(side_arr[valid_openi_arr] == OrderSide.BUY)[0]

    valid_open_idx = open_idx[valid_openi_arr].reshape(-1, 1)

    short_obj = np.hstack((pair_price_arr, pair_idx_arr, valid_open_idx))[short_valid_openi_idx]
    long_obj = np.hstack((pair_price_arr, pair_idx_arr, valid_open_idx))[long_valid_openi_idx]
    both_obj = np.vstack((short_obj, long_obj))
    print("short_obj.shape :", short_obj.shape)
    print("long_obj.shape :", long_obj.shape)

    short_obj, long_obj, both_obj = [np.split(obj_, 5, axis=1) for obj_ in [short_obj, long_obj, both_obj]]

    short_point1_arr, long_point1_arr = [point1_arr[openi_idx_].reshape(-1, 1) for openi_idx_ in [short_valid_openi_idx, long_valid_openi_idx]]
    short_lvrg_arr, long_lvrg_arr = [lvrg_arr[openi_idx_].reshape(-1, 1) for openi_idx_ in [short_valid_openi_idx, long_valid_openi_idx]]
    short_fee_arr, long_fee_arr = [fee_arr[openi_idx_].reshape(-1, 1) for openi_idx_ in [short_valid_openi_idx, long_valid_openi_idx]]
    short_tpout_arr, long_tpout_arr = [tpout_arr[openi_idx_] for openi_idx_ in [short_valid_openi_idx, long_valid_openi_idx]]
    # short_bias_arr, long_bias_arr = [bias_arr[openi_idx_] for openi_idx_ in [short_valid_openi_idx, long_valid_openi_idx]]
    short_tr_arr, long_tr_arr = [tr_arr[openi_idx_] for openi_idx_ in [short_valid_openi_idx, long_valid_openi_idx]]
    # print("long_bias_arr.shape :", long_bias_arr.shape)
    # print("elapsed time :", time.time() - start_0)

    short_true_bias_idx, short_false_bias_idx, long_true_bias_idx, long_false_bias_idx, short_en_tp1, short_en_out0, long_en_tp1, long_en_out0 = \
        get_wave_bias(res_df, config, high, low, len_df, short_obj, long_obj)

    len_short, len_long = len(short_valid_openi_idx), len(long_valid_openi_idx)

    # ------ plot_data ------ #
    try:
        # start_0 = time.time()
        if len_short == 0:
            short_pr = []
            gs_idx += 1
        else:
            short_tr = short_tr_arr.mean()
            short_pr, short_liqd = get_pr_v3(OrderSide.SELL, high, low, short_obj, short_tpout_arr, short_lvrg_arr, short_fee_arr, p_ranges,
                                             p_qty_ratio, inversion)
            short_total_pr = to_total_pr(len_df, short_pr, short_obj[-2])
            short_cum_pr = np.cumprod(short_total_pr)
            # short_liqd = liquidation_v2(OrderSide.SELL, h, short_obj[:4], short_lvrg_arr, short_fee_arr)
            short_prcn, short_rc = precision(short_pr, short_true_bias_idx), wave_bias(short_true_bias_idx, short_false_bias_idx)
            short_trade_ticks = np.mean(short_obj[-2] - short_obj[-1])
            if signi:
                short_idep_res_obj = (short_prcn, short_rc) + get_res_info_nb_v2(sample_len, short_pr, short_total_pr, short_cum_pr, short_liqd)
            else:
                gs_idx = plot_info_v6(gs, gs_idx, sample_len, short_tr, short_prcn, short_rc, short_trade_ticks, short_pr, short_total_pr,
                                      short_cum_pr, short_liqd, short_lvrg_arr[-1], title_position, fontsize)
        # print("elapsed time :", time.time() - start_0)

    except Exception as e:
        gs_idx += 1
        print("error in short plot_data :", e)

    try:
        # start_0 = time.time()
        if len_long == 0:
            long_pr = []
            gs_idx += 1
        else:
            long_tr = long_tr_arr.mean()
            long_pr, long_liqd = get_pr_v3(OrderSide.BUY, high, low, long_obj, long_tpout_arr, long_lvrg_arr, long_fee_arr, p_ranges, p_qty_ratio,
                                           inversion)
            long_total_pr = to_total_pr(len_df, long_pr, long_obj[-2])
            long_cum_pr = np.cumprod(long_total_pr)
            # long_liqd = liquidation_v2(OrderSide.BUY, l, long_obj[:4], long_lvrg_arr, long_fee_arr)
            long_prcn, long_rc = precision(long_pr, long_true_bias_idx), wave_bias(long_true_bias_idx, long_false_bias_idx)
            long_trade_ticks = np.mean(long_obj[-2] - long_obj[-1])
            if signi:
                long_idep_res_obj = (long_prcn, long_rc) + get_res_info_nb_v2(sample_len, long_pr, long_total_pr, long_cum_pr, long_liqd)
            else:
                gs_idx = plot_info_v6(gs, gs_idx, sample_len, long_tr, long_prcn, long_rc, long_trade_ticks, long_pr, long_total_pr, long_cum_pr,
                                      long_liqd, long_lvrg_arr[-1], title_position, fontsize)
        # print("elapsed time :", time.time() - start_0)
    except Exception as e:
        gs_idx += 1
        print("error in long plot_data :", e)

    try:
        # start_0 = time.time()
        if len_short * len_long == 0:
            both_pr = []
            gs_idx += 1
        else:
            both_tr = (short_tr + long_tr) / 2
            both_pr = np.vstack((short_pr, long_pr))  # for 2d arr, obj 를 1d 로 만들지 않는 이상, pr 은 2d 유지될 것
            both_total_pr = to_total_pr(len_df, both_pr, both_obj[-2])
            both_cum_pr = np.cumprod(both_total_pr)
            both_liqd = min(short_liqd, long_liqd)
            both_true_bias_idx = np.vstack((short_true_bias_idx, long_true_bias_idx))  # vstack for 2d arr
            both_false_bias_idx = np.vstack((short_false_bias_idx, long_false_bias_idx))
            both_prcn, both_rc = precision(both_pr, both_true_bias_idx), wave_bias(both_true_bias_idx, both_false_bias_idx)
            both_trade_ticks = np.mean(both_obj[-2] - both_obj[-1])
            if signi:
                both_idep_res_obj = (both_prcn, both_rc) + get_res_info_nb_v2(sample_len, both_pr, both_total_pr, both_cum_pr, both_liqd)
            else:
                gs_idx = plot_info_v6(gs, gs_idx, sample_len, both_tr, both_prcn, both_rc, both_trade_ticks, both_pr, both_total_pr, both_cum_pr,
                                      both_liqd, lvrg_arr[-1], title_position, fontsize)
        # print("elapsed time :", time.time() - start_0)
    except Exception as e:
        gs_idx += 1
        print("error in both plot_data :", e)

    if not signi:
        if len_short * len_long > 0:
            for obj, cum_pr in zip([short_obj, long_obj, both_obj], [short_cum_pr, long_cum_pr, both_cum_pr]):
                try:
                    # start_0 = time.time()
                    gs_idx = frq_dev_plot_v3(gs, gs_idx, len_df, sample_len, obj[-2], cum_pr[-1], fontsize)
                    # print("elapsed time :", time.time() - start_0)
                except Exception as e:
                    gs_idx += 1
                    print("error in frq_dev_plot_v3 :", e)
            plt.show()
            plt.close()

        return short_pr, short_obj, short_lvrg_arr, short_fee_arr, short_tpout_arr, short_true_bias_idx, short_false_bias_idx, short_point1_arr, short_en_tp1, short_en_out0, \
               long_pr, long_obj, long_lvrg_arr, long_fee_arr, long_tpout_arr, long_true_bias_idx, long_false_bias_idx, long_point1_arr, long_en_tp1, long_en_out0

    else:
        return [short_idep_res_obj[:-1], long_idep_res_obj[:-1], both_idep_res_obj[:-1]]

def idep_plot_v11(res_df, len_df, config, high, low, open_idx, side_arr, paired_res, inversion=False, sample_ratio=0.7, title_position=(0.5, 0.5),
                  fontsize=15, signi=False):
    if not signi:
        plt.style.use(['dark_background', 'fast'])
        fig = plt.figure(figsize=(24, 8))
        gs = gridspec.GridSpec(nrows=2,  # row 몇 개
                               ncols=3,  # col 몇 개
                               height_ratios=[10, 1]
                               # height_ratios=[10, 10, 1]
                               )
    gs_idx = 0
    # plt.suptitle(key)

    p_ranges, p_qty_ratio = literal_eval(config.tp_set.p_ranges), literal_eval(config.tp_set.p_qty_ratio)
    assert np.sum(p_qty_ratio) == 1.0
    assert len(p_ranges) == len(p_qty_ratio)

    if sample_ratio is not None:
        sample_len = int(len_df * sample_ratio)
    else:
        sample_len = len_df

    # ------ short & long data preparation ------ #
    # start_0 = time.time()
    point1_arr, valid_openi_arr, pair_idx_arr, pair_price_arr, lvrg_arr, fee_arr, tpout_arr, tr_arr = paired_res
    assert len(valid_openi_arr) != 0, "assert len(valid_openi_arr) != 0"
    short_valid_openi_idx = np.where(side_arr[valid_openi_arr] == OrderSide.SELL)[0]  # valid_openi_arr 에 대한 idx, # side_arr,
    long_valid_openi_idx = np.where(side_arr[valid_openi_arr] == OrderSide.BUY)[0]

    valid_open_idx = open_idx[valid_openi_arr].reshape(-1, 1)

    short_obj = np.hstack((pair_price_arr, pair_idx_arr, valid_open_idx))[short_valid_openi_idx]
    long_obj = np.hstack((pair_price_arr, pair_idx_arr, valid_open_idx))[long_valid_openi_idx]
    both_obj = np.vstack((short_obj, long_obj))
    print("short_obj.shape :", short_obj.shape)
    print("long_obj.shape :", long_obj.shape)

    short_obj, long_obj, both_obj = [np.split(obj_, 5, axis=1) for obj_ in [short_obj, long_obj, both_obj]]

    short_point1_arr, long_point1_arr = [point1_arr[openi_idx_].reshape(-1, 1) for openi_idx_ in [short_valid_openi_idx, long_valid_openi_idx]]
    short_lvrg_arr, long_lvrg_arr = [lvrg_arr[openi_idx_].reshape(-1, 1) for openi_idx_ in [short_valid_openi_idx, long_valid_openi_idx]]
    short_fee_arr, long_fee_arr = [fee_arr[openi_idx_].reshape(-1, 1) for openi_idx_ in [short_valid_openi_idx, long_valid_openi_idx]]
    short_tpout_arr, long_tpout_arr = [tpout_arr[openi_idx_] for openi_idx_ in [short_valid_openi_idx, long_valid_openi_idx]]
    # short_bias_arr, long_bias_arr = [bias_arr[openi_idx_] for openi_idx_ in [short_valid_openi_idx, long_valid_openi_idx]]
    short_tr_arr, long_tr_arr = [tr_arr[openi_idx_] for openi_idx_ in [short_valid_openi_idx, long_valid_openi_idx]]
    # print("long_bias_arr.shape :", long_bias_arr.shape)
    # print("elapsed time :", time.time() - start_0)

    short_true_bias_idx, short_false_bias_idx, long_true_bias_idx, long_false_bias_idx, short_en_tp1, short_en_out0, long_en_tp1, long_en_out0 = \
        get_wave_bias(res_df, config, high, low, len_df, short_obj, long_obj)

    len_short, len_long = len(short_valid_openi_idx), len(long_valid_openi_idx)

    # ------ plot_data ------ #
    try:
        # start_0 = time.time()
        if len_short == 0:
            short_pr = []
            gs_idx += 1
        else:
            short_tr = short_tr_arr.mean()
            short_pr, short_liqd = get_pr_v3(OrderSide.SELL, high, low, short_obj, short_tpout_arr, short_lvrg_arr, short_fee_arr, p_ranges,
                                             p_qty_ratio, inversion)
            short_total_pr = to_total_pr(len_df, short_pr, short_obj[-2])
            short_cum_pr = np.cumprod(short_total_pr)
            # short_liqd = liquidation_v2(OrderSide.SELL, h, short_obj[:4], short_lvrg_arr, short_fee_arr)
            short_prcn, short_rc = precision(short_pr, short_true_bias_idx), wave_bias(short_true_bias_idx, short_false_bias_idx)
            if signi:
                short_idep_res_obj = (short_prcn, short_rc) + get_res_info_nb_v2(sample_len, short_pr, short_total_pr, short_cum_pr, short_liqd)
            else:
                gs_idx = plot_info_v5(gs, gs_idx, sample_len, short_tr, short_prcn, short_rc, short_pr, short_total_pr, short_cum_pr, short_liqd,
                                      short_lvrg_arr[-1], title_position, fontsize)
        # print("elapsed time :", time.time() - start_0)

    except Exception as e:
        gs_idx += 1
        print("error in short plot_data :", e)

    try:
        # start_0 = time.time()
        if len_long == 0:
            long_pr = []
            gs_idx += 1
        else:
            long_tr = long_tr_arr.mean()
            long_pr, long_liqd = get_pr_v3(OrderSide.BUY, high, low, long_obj, long_tpout_arr, long_lvrg_arr, long_fee_arr, p_ranges, p_qty_ratio,
                                           inversion)
            long_total_pr = to_total_pr(len_df, long_pr, long_obj[-2])
            long_cum_pr = np.cumprod(long_total_pr)
            # long_liqd = liquidation_v2(OrderSide.BUY, l, long_obj[:4], long_lvrg_arr, long_fee_arr)
            long_prcn, long_rc = precision(long_pr, long_true_bias_idx), wave_bias(long_true_bias_idx, long_false_bias_idx)
            if signi:
                long_idep_res_obj = (long_prcn, long_rc) + get_res_info_nb_v2(sample_len, long_pr, long_total_pr, long_cum_pr, long_liqd)
            else:
                gs_idx = plot_info_v5(gs, gs_idx, sample_len, long_tr, long_prcn, long_rc, long_pr, long_total_pr, long_cum_pr, long_liqd,
                                      long_lvrg_arr[-1], title_position, fontsize)
        # print("elapsed time :", time.time() - start_0)
    except Exception as e:
        gs_idx += 1
        print("error in long plot_data :", e)

    try:
        # start_0 = time.time()
        if len_short * len_long == 0:
            both_pr = []
            gs_idx += 1
        else:
            both_tr = (short_tr + long_tr) / 2
            both_pr = np.vstack((short_pr, long_pr))  # for 2d arr, obj 를 1d 로 만들지 않는 이상, pr 은 2d 유지될 것
            both_total_pr = to_total_pr(len_df, both_pr, both_obj[-2])
            both_cum_pr = np.cumprod(both_total_pr)
            both_liqd = min(short_liqd, long_liqd)
            both_true_bias_idx = np.vstack((short_true_bias_idx, long_true_bias_idx))  # vstack for 2d arr
            both_false_bias_idx = np.vstack((short_false_bias_idx, long_false_bias_idx))
            both_prcn, both_rc = precision(both_pr, both_true_bias_idx), wave_bias(both_true_bias_idx, both_false_bias_idx)
            if signi:
                both_idep_res_obj = (both_prcn, both_rc) + get_res_info_nb_v2(sample_len, both_pr, both_total_pr, both_cum_pr, both_liqd)
            else:
                gs_idx = plot_info_v5(gs, gs_idx, sample_len, both_tr, both_prcn, both_rc, both_pr, both_total_pr, both_cum_pr, both_liqd,
                                      lvrg_arr[-1], title_position, fontsize)
        # print("elapsed time :", time.time() - start_0)
    except Exception as e:
        gs_idx += 1
        print("error in both plot_data :", e)

    if not signi:
        if len_short * len_long > 0:
            for obj, cum_pr in zip([short_obj, long_obj, both_obj], [short_cum_pr, long_cum_pr, both_cum_pr]):
                try:
                    # start_0 = time.time()
                    gs_idx = frq_dev_plot_v3(gs, gs_idx, len_df, sample_len, obj[-2], cum_pr[-1], fontsize)
                    # print("elapsed time :", time.time() - start_0)
                except Exception as e:
                    gs_idx += 1
                    print("error in frq_dev_plot_v3 :", e)
            plt.show()
            plt.close()

        return short_pr, short_obj, short_lvrg_arr, short_fee_arr, short_tpout_arr, short_true_bias_idx, short_false_bias_idx, short_point1_arr, short_en_tp1, short_en_out0, \
               long_pr, long_obj, long_lvrg_arr, long_fee_arr, long_tpout_arr, long_true_bias_idx, long_false_bias_idx, long_point1_arr, long_en_tp1, long_en_out0

    else:
        return [short_idep_res_obj[:-1], long_idep_res_obj[:-1], both_idep_res_obj[:-1]]

def idep_plot_v10(len_df, config, h, l, open_idx, side_arr, paired_res, inversion=False, sample_ratio=0.7, title_position=(0.5, 0.5), fontsize=15,
                  signi=False):
    if not signi:
        plt.style.use(['dark_background', 'fast'])
        fig = plt.figure(figsize=(24, 8))
        gs = gridspec.GridSpec(nrows=2,  # row 몇 개
                               ncols=3,  # col 몇 개
                               height_ratios=[10, 1]
                               # height_ratios=[10, 10, 1]
                               )
    gs_idx = 0
    # plt.suptitle(key)

    p_ranges, p_qty_ratio = literal_eval(config.tp_set.p_ranges), literal_eval(config.tp_set.p_qty_ratio)
    assert np.sum(p_qty_ratio) == 1.0
    assert len(p_ranges) == len(p_qty_ratio)

    if sample_ratio is not None:
        sample_len = int(len_df * sample_ratio)
    else:
        sample_len = len_df

    # ------ short & long data preparation ------ #
    # start_0 = time.time()
    point1_arr, valid_openi_arr, pair_idx_arr, pair_price_arr, lvrg_arr, fee_arr, tpout_arr, bias_arr, tr_arr = paired_res
    assert len(valid_openi_arr) != 0, "assert len(valid_openi_arr) != 0"
    short_valid_openi_idx = np.where(side_arr[valid_openi_arr] == OrderSide.SELL)[0]  # valid_openi_arr 에 대한 idx, # side_arr,
    long_valid_openi_idx = np.where(side_arr[valid_openi_arr] == OrderSide.BUY)[0]

    valid_open_idx = open_idx[valid_openi_arr].reshape(-1, 1)

    short_obj = np.hstack((pair_price_arr, pair_idx_arr, valid_open_idx))[short_valid_openi_idx]
    long_obj = np.hstack((pair_price_arr, pair_idx_arr, valid_open_idx))[long_valid_openi_idx]
    both_obj = np.vstack((short_obj, long_obj))
    print("short_obj.shape :", short_obj.shape)
    print("long_obj.shape :", long_obj.shape)

    short_obj, long_obj, both_obj = [np.split(obj_, 5, axis=1) for obj_ in [short_obj, long_obj, both_obj]]

    short_point1_arr, long_point1_arr = [point1_arr[openi_idx_].reshape(-1, 1) for openi_idx_ in [short_valid_openi_idx, long_valid_openi_idx]]
    short_lvrg_arr, long_lvrg_arr = [lvrg_arr[openi_idx_].reshape(-1, 1) for openi_idx_ in [short_valid_openi_idx, long_valid_openi_idx]]
    short_fee_arr, long_fee_arr = [fee_arr[openi_idx_].reshape(-1, 1) for openi_idx_ in [short_valid_openi_idx, long_valid_openi_idx]]
    short_tpout_arr, long_tpout_arr = [tpout_arr[openi_idx_] for openi_idx_ in [short_valid_openi_idx, long_valid_openi_idx]]
    short_bias_arr, long_bias_arr = [bias_arr[openi_idx_] for openi_idx_ in [short_valid_openi_idx, long_valid_openi_idx]]
    short_tr_arr, long_tr_arr = [tr_arr[openi_idx_] for openi_idx_ in [short_valid_openi_idx, long_valid_openi_idx]]
    # print("long_bias_arr.shape :", long_bias_arr.shape)
    # print("elapsed time :", time.time() - start_0)

    short_true_bias_idx = short_bias_arr[:, 0] <= short_bias_arr[:, 1]  # info, threshold (등호의 유무가 매우 유의미함)
    long_true_bias_idx = long_bias_arr[:, 0] >= long_bias_arr[:, 1]

    len_short, len_long = len(short_valid_openi_idx), len(long_valid_openi_idx)

    # ------ plot_data ------ #
    try:
        # start_0 = time.time()
        if len_short == 0:
            short_pr = []
            gs_idx += 1
        else:
            short_tr = short_tr_arr.mean()
            short_pr, short_liqd = get_pr_v3(OrderSide.SELL, h, l, short_obj, short_tpout_arr, short_lvrg_arr, short_fee_arr, p_ranges, p_qty_ratio,
                                             inversion)
            short_total_pr = to_total_pr(len_df, short_pr, short_obj[-2])
            short_cum_pr = np.cumprod(short_total_pr)
            # short_liqd = liquidation_v2(OrderSide.SELL, h, short_obj[:4], short_lvrg_arr, short_fee_arr)
            short_prcn, short_rc = precision(short_pr, short_true_bias_idx), recall(short_true_bias_idx)
            if signi:
                short_idep_res_obj = (short_prcn, short_rc) + get_res_info_nb_v2(sample_len, short_pr, short_total_pr, short_cum_pr, short_liqd)
            else:
                gs_idx = plot_info_v5(gs, gs_idx, sample_len, short_tr, short_prcn, short_rc, short_pr, short_total_pr, short_cum_pr, short_liqd,
                                      short_lvrg_arr[-1], title_position, fontsize)
        # print("elapsed time :", time.time() - start_0)

    except Exception as e:
        gs_idx += 1
        print("error in short plot_data :", e)

    try:
        # start_0 = time.time()
        if len_long == 0:
            long_pr = []
            gs_idx += 1
        else:
            long_tr = long_tr_arr.mean()
            long_pr, long_liqd = get_pr_v3(OrderSide.BUY, h, l, long_obj, long_tpout_arr, long_lvrg_arr, long_fee_arr, p_ranges, p_qty_ratio,
                                           inversion)
            long_total_pr = to_total_pr(len_df, long_pr, long_obj[-2])
            long_cum_pr = np.cumprod(long_total_pr)
            # long_liqd = liquidation_v2(OrderSide.BUY, l, long_obj[:4], long_lvrg_arr, long_fee_arr)
            long_prcn, long_rc = precision(long_pr, long_true_bias_idx), recall(long_true_bias_idx)
            if signi:
                long_idep_res_obj = (long_prcn, long_rc) + get_res_info_nb_v2(sample_len, long_pr, long_total_pr, long_cum_pr, long_liqd)
            else:
                gs_idx = plot_info_v5(gs, gs_idx, sample_len, long_tr, long_prcn, long_rc, long_pr, long_total_pr, long_cum_pr, long_liqd,
                                      long_lvrg_arr[-1], title_position, fontsize)
        # print("elapsed time :", time.time() - start_0)
    except Exception as e:
        gs_idx += 1
        print("error in long plot_data :", e)

    try:
        # start_0 = time.time()
        if len_short * len_long == 0:
            both_pr = []
            gs_idx += 1
        else:
            both_tr = (short_tr + long_tr) / 2
            both_pr = np.vstack((short_pr, long_pr))  # for 2d shape, obj 를 1d 로 만들지 않는 이상, pr 은 2d 유지될 것
            both_total_pr = to_total_pr(len_df, both_pr, both_obj[-2])
            both_cum_pr = np.cumprod(both_total_pr)
            both_liqd = min(short_liqd, long_liqd)
            both_true_bias_idx = np.hstack((short_true_bias_idx, long_true_bias_idx))  # for 1d shape
            both_prcn, both_rc = precision(both_pr, both_true_bias_idx), recall(both_true_bias_idx)
            if signi:
                both_idep_res_obj = (both_prcn, both_rc) + get_res_info_nb_v2(sample_len, both_pr, both_total_pr, both_cum_pr, both_liqd)
            else:
                gs_idx = plot_info_v5(gs, gs_idx, sample_len, both_tr, both_prcn, both_rc, both_pr, both_total_pr, both_cum_pr, both_liqd,
                                      lvrg_arr[-1], title_position, fontsize)
        # print("elapsed time :", time.time() - start_0)
    except Exception as e:
        gs_idx += 1
        print("error in both plot_data :", e)

    if not signi:
        if len_short * len_long > 0:
            for obj, cum_pr in zip([short_obj, long_obj, both_obj], [short_cum_pr, long_cum_pr, both_cum_pr]):
                try:
                    # start_0 = time.time()
                    gs_idx = frq_dev_plot_v3(gs, gs_idx, len_df, sample_len, obj[-2], cum_pr[-1], fontsize)
                    # print("elapsed time :", time.time() - start_0)
                except Exception as e:
                    gs_idx += 1
                    print("error in frq_dev_plot_v3 :", e)
            plt.show()
            plt.close()

        return short_pr, short_obj, short_lvrg_arr, short_fee_arr, short_tpout_arr, short_bias_arr, short_point1_arr, \
               long_pr, long_obj, long_lvrg_arr, long_fee_arr, long_tpout_arr, long_bias_arr, long_point1_arr

    else:
        return [short_idep_res_obj[:-1], long_idep_res_obj[:-1], both_idep_res_obj[:-1]]

def idep_plot_v9(len_df, config, h, l, open_idx, side_arr, paired_res, inversion=False, sample_ratio=0.7, title_position=(0.5, 0.5), fontsize=15,
                 signi=False):
    if not signi:
        plt.style.use(['dark_background', 'fast'])
        fig = plt.figure(figsize=(24, 8))
        gs = gridspec.GridSpec(nrows=2,  # row 몇 개
                               ncols=3,  # col 몇 개
                               height_ratios=[10, 1]
                               # height_ratios=[10, 10, 1]
                               )
    gs_idx = 0
    # plt.suptitle(key)

    p_ranges, p_qty_ratio = literal_eval(config.tp_set.p_ranges), literal_eval(config.tp_set.p_qty_ratio)
    assert np.sum(p_qty_ratio) == 1.0
    assert len(p_ranges) == len(p_qty_ratio)

    if sample_ratio is not None:
        sample_len = int(len_df * sample_ratio)
    else:
        sample_len = len_df

    # ------ short & long data preparation ------ #
    # start_0 = time.time()
    point1_arr, valid_openi_arr, pair_idx_arr, pair_price_arr, lvrg_arr, fee_arr, tpout_arr, bias_arr, tr_arr = paired_res
    assert len(valid_openi_arr) != 0, "assert len(valid_openi_arr) != 0"
    short_valid_openi_idx = np.where(side_arr[valid_openi_arr] == OrderSide.SELL)[0]  # valid_openi_arr 에 대한 idx, # side_arr,
    long_valid_openi_idx = np.where(side_arr[valid_openi_arr] == OrderSide.BUY)[0]

    valid_open_idx = open_idx[valid_openi_arr].reshape(-1, 1)

    short_obj = np.hstack((pair_price_arr, pair_idx_arr, valid_open_idx))[short_valid_openi_idx]
    long_obj = np.hstack((pair_price_arr, pair_idx_arr, valid_open_idx))[long_valid_openi_idx]
    both_obj = np.vstack((short_obj, long_obj))
    print("short_obj.shape :", short_obj.shape)
    print("long_obj.shape :", long_obj.shape)

    short_obj, long_obj, both_obj = [np.split(obj_, 5, axis=1) for obj_ in [short_obj, long_obj, both_obj]]

    short_point1_arr, long_point1_arr = [point1_arr[openi_idx_].reshape(-1, 1) for openi_idx_ in [short_valid_openi_idx, long_valid_openi_idx]]
    short_lvrg_arr, long_lvrg_arr = [lvrg_arr[openi_idx_].reshape(-1, 1) for openi_idx_ in [short_valid_openi_idx, long_valid_openi_idx]]
    short_fee_arr, long_fee_arr = [fee_arr[openi_idx_].reshape(-1, 1) for openi_idx_ in [short_valid_openi_idx, long_valid_openi_idx]]
    short_tpout_arr, long_tpout_arr = [tpout_arr[openi_idx_] for openi_idx_ in [short_valid_openi_idx, long_valid_openi_idx]]
    short_bias_arr, long_bias_arr = [bias_arr[openi_idx_] for openi_idx_ in [short_valid_openi_idx, long_valid_openi_idx]]
    short_tr_arr, long_tr_arr = [tr_arr[openi_idx_] for openi_idx_ in [short_valid_openi_idx, long_valid_openi_idx]]
    # print("long_bias_arr.shape :", long_bias_arr.shape)
    # print("elapsed time :", time.time() - start_0)

    short_true_bias_idx = short_bias_arr[:, 0] <= short_bias_arr[:, 1]  # info, threshold (등호의 유무가 매우 유의미함)
    long_true_bias_idx = long_bias_arr[:, 0] >= long_bias_arr[:, 1]

    # ------ plot_data ------ #
    try:
        # start_0 = time.time()
        short_tr = short_tr_arr.mean()
        short_pr, short_liqd = get_pr_v3(OrderSide.SELL, h, l, short_obj, short_tpout_arr, short_lvrg_arr, short_fee_arr, p_ranges, p_qty_ratio, inversion)
        short_total_pr = to_total_pr(len_df, short_pr, short_obj[-2])
        short_cum_pr = np.cumprod(short_total_pr)
        # short_liqd = liquidation_v2(OrderSide.SELL, h, short_obj[:4], short_lvrg_arr, short_fee_arr)
        short_prcn, short_rc = precision(short_pr, short_true_bias_idx), recall(short_true_bias_idx)
        if signi:
            short_idep_res_obj = (short_prcn, short_rc) + get_res_info_nb_v2(sample_len, short_pr, short_total_pr, short_cum_pr, short_liqd)
        else:
            gs_idx = plot_info_v5(gs, gs_idx, sample_len, short_tr, short_prcn, short_rc, short_pr, short_total_pr, short_cum_pr, short_liqd,
                                  short_lvrg_arr[-1], title_position, fontsize)
        # print("elapsed time :", time.time() - start_0)

    except Exception as e:
        gs_idx += 1
        print("error in short plot_data :", e)

    try:
        # start_0 = time.time()
        long_tr = long_tr_arr.mean()
        long_pr, long_liqd = get_pr_v3(OrderSide.BUY, h, l, long_obj, long_tpout_arr, long_lvrg_arr, long_fee_arr, p_ranges, p_qty_ratio, inversion)
        long_total_pr = to_total_pr(len_df, long_pr, long_obj[-2])
        long_cum_pr = np.cumprod(long_total_pr)
        # long_liqd = liquidation_v2(OrderSide.BUY, l, long_obj[:4], long_lvrg_arr, long_fee_arr)
        long_prcn, long_rc = precision(long_pr, long_true_bias_idx), recall(long_true_bias_idx)
        if signi:
            long_idep_res_obj = (long_prcn, long_rc) + get_res_info_nb_v2(sample_len, long_pr, long_total_pr, long_cum_pr, long_liqd)
        else:
            gs_idx = plot_info_v5(gs, gs_idx, sample_len, long_tr, long_prcn, long_rc, long_pr, long_total_pr, long_cum_pr, long_liqd,
                                  long_lvrg_arr[-1], title_position, fontsize)
        # print("elapsed time :", time.time() - start_0)
    except Exception as e:
        gs_idx += 1
        print("error in long plot_data :", e)

    try:
        # start_0 = time.time()
        both_tr = (short_tr + long_tr) / 2
        both_pr = np.vstack((short_pr, long_pr))  # for 2d shape, obj 를 1d 로 만들지 않는 이상, pr 은 2d 유지될 것
        both_total_pr = to_total_pr(len_df, both_pr, both_obj[-2])
        both_cum_pr = np.cumprod(both_total_pr)
        both_liqd = min(short_liqd, long_liqd)
        both_true_bias_idx = np.hstack((short_true_bias_idx, long_true_bias_idx))  # for 1d shape
        both_prcn, both_rc = precision(both_pr, both_true_bias_idx), recall(both_true_bias_idx)
        if signi:
            both_idep_res_obj = (both_prcn, both_rc) + get_res_info_nb_v2(sample_len, both_pr, both_total_pr, both_cum_pr, both_liqd)
        else:
            gs_idx = plot_info_v5(gs, gs_idx, sample_len, both_tr, both_prcn, both_rc, both_pr, both_total_pr, both_cum_pr, both_liqd, lvrg_arr[-1],
                                  title_position, fontsize)
        # print("elapsed time :", time.time() - start_0)
    except Exception as e:
        gs_idx += 1
        print("error in both plot_data :", e)

    if not signi:
        for obj, cum_pr in zip([short_obj, long_obj, both_obj], [short_cum_pr, long_cum_pr, both_cum_pr]):
            try:
                # start_0 = time.time()
                gs_idx = frq_dev_plot_v3(gs, gs_idx, len_df, sample_len, obj[-2], cum_pr[-1], fontsize)
                # print("elapsed time :", time.time() - start_0)
            except Exception as e:
                gs_idx += 1
                print("error in frq_dev_plot_v3 :", e)

        plt.show()
        plt.close()
        return short_pr, short_obj, short_lvrg_arr, short_fee_arr, short_tpout_arr, short_bias_arr, short_point1_arr, \
               long_pr, long_obj, long_lvrg_arr, long_fee_arr, long_tpout_arr, long_bias_arr, long_point1_arr

    else:
        return [short_idep_res_obj[:-1], long_idep_res_obj[:-1], both_idep_res_obj[:-1]]

def idep_plot_v8(len_df, config, h, l, open_idx, side_arr, paired_res, inversion=False, sample_ratio=0.7, title_position=(0.5, 0.5), fontsize=15,
                 signi=False):
    if not signi:
        plt.style.use(['dark_background', 'fast'])
        fig = plt.figure(figsize=(24, 8))
        gs = gridspec.GridSpec(nrows=2,  # row 몇 개
                               ncols=3,  # col 몇 개
                               height_ratios=[10, 1]
                               # height_ratios=[10, 10, 1]
                               )
    gs_idx = 0
    # plt.suptitle(key)

    p_ranges, p_qty_ratio = literal_eval(config.tp_set.p_ranges), literal_eval(config.tp_set.p_qty_ratio)
    assert np.sum(p_qty_ratio) == 1.0
    assert len(p_ranges) == len(p_qty_ratio)

    if sample_ratio is not None:
        sample_len = int(len_df * sample_ratio)
    else:
        sample_len = len_df

    # ------ short & long data preparation ------ #
    # start_0 = time.time()
    valid_openi_arr, pair_idx_arr, pair_price_arr, lvrg_arr, fee_arr, tpout_arr, bias_arr, tr_arr = paired_res
    assert len(valid_openi_arr) != 0, "assert len(valid_openi_arr) != 0"
    short_valid_openi_idx = np.where(side_arr[valid_openi_arr] == OrderSide.SELL)[0]  # valid_openi_arr 에 대한 idx, # side_arr,
    long_valid_openi_idx = np.where(side_arr[valid_openi_arr] == OrderSide.BUY)[0]

    valid_open_idx = open_idx[valid_openi_arr].reshape(-1, 1)

    short_obj = np.hstack((pair_price_arr, pair_idx_arr, valid_open_idx))[short_valid_openi_idx]
    long_obj = np.hstack((pair_price_arr, pair_idx_arr, valid_open_idx))[long_valid_openi_idx]
    both_obj = np.vstack((short_obj, long_obj))
    print("short_obj.shape :", short_obj.shape)
    print("long_obj.shape :", long_obj.shape)

    short_obj, long_obj, both_obj = [np.split(obj_, 5, axis=1) for obj_ in [short_obj, long_obj, both_obj]]

    short_lvrg_arr, long_lvrg_arr = [lvrg_arr[openi_idx_].reshape(-1, 1) for openi_idx_ in [short_valid_openi_idx, long_valid_openi_idx]]
    short_fee_arr, long_fee_arr = [fee_arr[openi_idx_].reshape(-1, 1) for openi_idx_ in [short_valid_openi_idx, long_valid_openi_idx]]
    short_tpout_arr, long_tpout_arr = [tpout_arr[openi_idx_] for openi_idx_ in [short_valid_openi_idx, long_valid_openi_idx]]
    short_bias_arr, long_bias_arr = [bias_arr[openi_idx_] for openi_idx_ in [short_valid_openi_idx, long_valid_openi_idx]]
    short_tr_arr, long_tr_arr = [tr_arr[openi_idx_] for openi_idx_ in [short_valid_openi_idx, long_valid_openi_idx]]
    # print("long_bias_arr.shape :", long_bias_arr.shape)
    # print("elapsed time :", time.time() - start_0)

    short_true_bias_idx = short_bias_arr[:, 0] <= short_bias_arr[:, 1]  # info, threshold (등호의 유무가 매우 유의미함)
    long_true_bias_idx = long_bias_arr[:, 0] >= long_bias_arr[:, 1]

    # ------ plot_data ------ #
    try:
        # start_0 = time.time()
        short_tr = short_tr_arr.mean()
        short_pr, short_liqd = get_pr_v3(OrderSide.SELL, h, l, short_obj, short_tpout_arr, short_lvrg_arr, short_fee_arr, p_ranges, p_qty_ratio, inversion)
        short_total_pr = to_total_pr(len_df, short_pr, short_obj[-2])
        short_cum_pr = np.cumprod(short_total_pr)
        # short_liqd = liquidation_v2(OrderSide.SELL, h, short_obj[:4], short_lvrg_arr, short_fee_arr)
        short_prcn, short_rc = precision(short_pr, short_true_bias_idx), recall(short_true_bias_idx)
        if signi:
            short_idep_res_obj = (short_prcn, short_rc) + get_res_info_nb_v2(sample_len, short_pr, short_total_pr, short_cum_pr, short_liqd)
        else:
            gs_idx = plot_info_v5(gs, gs_idx, sample_len, short_tr, short_prcn, short_rc, short_pr, short_total_pr, short_cum_pr, short_liqd,
                                  short_lvrg_arr[-1], title_position, fontsize)
        # print("elapsed time :", time.time() - start_0)

    except Exception as e:
        gs_idx += 1
        print("error in short plot_data :", e)

    try:
        # start_0 = time.time()
        long_tr = long_tr_arr.mean()
        long_pr, long_liqd = get_pr_v3(OrderSide.BUY, h, l, long_obj, long_tpout_arr, long_lvrg_arr, long_fee_arr, p_ranges, p_qty_ratio, inversion)
        long_total_pr = to_total_pr(len_df, long_pr, long_obj[-2])
        long_cum_pr = np.cumprod(long_total_pr)
        # long_liqd = liquidation_v2(OrderSide.BUY, l, long_obj[:4], long_lvrg_arr, long_fee_arr)
        long_prcn, long_rc = precision(long_pr, long_true_bias_idx), recall(long_true_bias_idx)
        if signi:
            long_idep_res_obj = (long_prcn, long_rc) + get_res_info_nb_v2(sample_len, long_pr, long_total_pr, long_cum_pr, long_liqd)
        else:
            gs_idx = plot_info_v5(gs, gs_idx, sample_len, long_tr, long_prcn, long_rc, long_pr, long_total_pr, long_cum_pr, long_liqd,
                                  long_lvrg_arr[-1], title_position, fontsize)
        # print("elapsed time :", time.time() - start_0)
    except Exception as e:
        gs_idx += 1
        print("error in long plot_data :", e)

    try:
        # start_0 = time.time()
        both_tr = (short_tr + long_tr) / 2
        both_pr = np.vstack((short_pr, long_pr))  # for 2d shape, obj 를 1d 로 만들지 않는 이상, pr 은 2d 유지될 것
        both_total_pr = to_total_pr(len_df, both_pr, both_obj[-2])
        both_cum_pr = np.cumprod(both_total_pr)
        both_liqd = min(short_liqd, long_liqd)
        both_true_bias_idx = np.hstack((short_true_bias_idx, long_true_bias_idx))  # for 1d shape
        both_prcn, both_rc = precision(both_pr, both_true_bias_idx), recall(both_true_bias_idx)
        if signi:
            both_idep_res_obj = (both_prcn, both_rc) + get_res_info_nb_v2(sample_len, both_pr, both_total_pr, both_cum_pr, both_liqd)
        else:
            gs_idx = plot_info_v5(gs, gs_idx, sample_len, both_tr, both_prcn, both_rc, both_pr, both_total_pr, both_cum_pr, both_liqd, lvrg_arr[-1],
                                  title_position, fontsize)
        # print("elapsed time :", time.time() - start_0)
    except Exception as e:
        gs_idx += 1
        print("error in both plot_data :", e)

    if not signi:
        for obj, cum_pr in zip([short_obj, long_obj, both_obj], [short_cum_pr, long_cum_pr, both_cum_pr]):
            try:
                # start_0 = time.time()
                gs_idx = frq_dev_plot_v3(gs, gs_idx, len_df, sample_len, obj[-2], cum_pr[-1], fontsize)
                # print("elapsed time :", time.time() - start_0)
            except Exception as e:
                gs_idx += 1
                print("error in frq_dev_plot_v3 :", e)

        plt.show()
        plt.close()
        return short_pr, short_obj, short_lvrg_arr, short_fee_arr, short_tpout_arr, short_bias_arr, \
               long_pr, long_obj, long_lvrg_arr, long_fee_arr, long_tpout_arr, long_bias_arr
    else:
        return [short_idep_res_obj[:-1], long_idep_res_obj[:-1], both_idep_res_obj[:-1]]

def idep_plot_v7(len_df, config, h, l, open_idx, side_arr, paired_res, inversion=False, sample_ratio=0.7, title_position=(0.5, 0.5), fontsize=15,
                 signi=False):
    if not signi:
        plt.style.use(['dark_background', 'fast'])
        fig = plt.figure(figsize=(24, 8))
        gs = gridspec.GridSpec(nrows=2,  # row 몇 개
                               ncols=3,  # col 몇 개
                               height_ratios=[10, 1]
                               # height_ratios=[10, 10, 1]
                               )
    gs_idx = 0
    # plt.suptitle(key)

    p_ranges, p_qty_ratio = literal_eval(config.tp_set.p_ranges), literal_eval(config.tp_set.p_qty_ratio)
    assert np.sum(p_qty_ratio) == 1.0
    assert len(p_ranges) == len(p_qty_ratio)

    if sample_ratio is not None:
        sample_len = int(len_df * sample_ratio)
    else:
        sample_len = len_df

    # ------ short & long data preparation ------ #
    # start_0 = time.time()
    valid_openi_arr, pair_idx_arr, pair_price_arr, lvrg_arr, fee_arr, tpout_arr, bias_arr = paired_res
    assert len(valid_openi_arr) != 0, "assert len(valid_openi_arr) != 0"
    short_valid_openi_idx = np.where(side_arr[valid_openi_arr] == OrderSide.SELL)[0]  # valid_openi_arr 에 대한 idx, # side_arr,
    long_valid_openi_idx = np.where(side_arr[valid_openi_arr] == OrderSide.BUY)[0]

    valid_open_idx = open_idx[valid_openi_arr].reshape(-1, 1)

    short_obj = np.hstack((pair_price_arr, pair_idx_arr, valid_open_idx))[short_valid_openi_idx]
    long_obj = np.hstack((pair_price_arr, pair_idx_arr, valid_open_idx))[long_valid_openi_idx]
    both_obj = np.vstack((short_obj, long_obj))
    print("short_obj.shape :", short_obj.shape)
    print("long_obj.shape :", long_obj.shape)

    short_obj, long_obj, both_obj = [np.split(obj_, 5, axis=1) for obj_ in [short_obj, long_obj, both_obj]]

    short_lvrg_arr, long_lvrg_arr = [lvrg_arr[openi_idx_].reshape(-1, 1) for openi_idx_ in [short_valid_openi_idx, long_valid_openi_idx]]
    short_fee_arr, long_fee_arr = [fee_arr[openi_idx_].reshape(-1, 1) for openi_idx_ in [short_valid_openi_idx, long_valid_openi_idx]]
    short_tpout_arr, long_tpout_arr = [tpout_arr[openi_idx_] for openi_idx_ in [short_valid_openi_idx, long_valid_openi_idx]]
    short_bias_arr, long_bias_arr = [bias_arr[openi_idx_] for openi_idx_ in [short_valid_openi_idx, long_valid_openi_idx]]
    # print("long_bias_arr.shape :", long_bias_arr.shape)
    # print("elapsed time :", time.time() - start_0)

    short_true_bias_idx = short_bias_arr[:, 0] <= short_bias_arr[:, 1]  # info, threshold (등호의 유무가 매우 유의미함)
    long_true_bias_idx = long_bias_arr[:, 0] >= long_bias_arr[:, 1]

    # ------ plot_data ------ #
    try:
        # start_0 = time.time()
        short_pr, short_liqd = get_pr_v3(OrderSide.SELL, h, l, short_obj, short_tpout_arr, short_lvrg_arr, short_fee_arr, p_ranges, p_qty_ratio, inversion)
        short_total_pr = to_total_pr(len_df, short_pr, short_obj[-2])
        short_cum_pr = np.cumprod(short_total_pr)
        # short_liqd = liquidation_v2(OrderSide.SELL, h, short_obj[:4], short_lvrg_arr, short_fee_arr)
        short_prcn, short_rc = precision(short_pr, short_true_bias_idx), recall(short_true_bias_idx)
        if signi:
            short_idep_res_obj = (short_prcn, short_rc) + get_res_info_nb_v2(sample_len, short_pr, short_total_pr, short_cum_pr, short_liqd)
        else:
            gs_idx = plot_info_v4(gs, gs_idx, sample_len, short_pr, short_total_pr, short_cum_pr, short_liqd, short_prcn, short_rc,
                                  short_lvrg_arr[-1], title_position, fontsize)
        # print("elapsed time :", time.time() - start_0)

    except Exception as e:
        gs_idx += 1
        print("error in short plot_data :", e)

    try:
        # start_0 = time.time()
        long_pr, long_liqd = get_pr_v3(OrderSide.BUY, h, l, long_obj, long_tpout_arr, long_lvrg_arr, long_fee_arr, p_ranges, p_qty_ratio, inversion)
        long_total_pr = to_total_pr(len_df, long_pr, long_obj[-2])
        long_cum_pr = np.cumprod(long_total_pr)
        # long_liqd = liquidation_v2(OrderSide.BUY, l, long_obj[:4], long_lvrg_arr, long_fee_arr)
        long_prcn, long_rc = precision(long_pr, long_true_bias_idx), recall(long_true_bias_idx)
        if signi:
            long_idep_res_obj = (long_prcn, long_rc) + get_res_info_nb_v2(sample_len, long_pr, long_total_pr, long_cum_pr, long_liqd)
        else:
            gs_idx = plot_info_v4(gs, gs_idx, sample_len, long_pr, long_total_pr, long_cum_pr, long_liqd, long_prcn, long_rc, long_lvrg_arr[-1],
                                  title_position, fontsize)
        # print("elapsed time :", time.time() - start_0)
    except Exception as e:
        gs_idx += 1
        print("error in long plot_data :", e)

    try:
        # start_0 = time.time()
        both_pr = np.vstack((short_pr, long_pr))  # for 2d shape, obj 를 1d 로 만들지 않는 이상, pr 은 2d 유지될 것
        both_total_pr = to_total_pr(len_df, both_pr, both_obj[-2])
        both_cum_pr = np.cumprod(both_total_pr)
        both_liqd = min(short_liqd, long_liqd)
        both_true_bias_idx = np.hstack((short_true_bias_idx, long_true_bias_idx))  # for 1d shape
        both_prcn, both_rc = precision(both_pr, both_true_bias_idx), recall(both_true_bias_idx)
        if signi:
            both_idep_res_obj = (both_prcn, both_rc) + get_res_info_nb_v2(sample_len, both_pr, both_total_pr, both_cum_pr, both_liqd)
        else:
            gs_idx = plot_info_v4(gs, gs_idx, sample_len, both_pr, both_total_pr, both_cum_pr, both_liqd, both_prcn, both_rc, lvrg_arr[-1],
                                  title_position, fontsize)
        # print("elapsed time :", time.time() - start_0)
    except Exception as e:
        gs_idx += 1
        print("error in both plot_data :", e)

    if not signi:
        for obj, cum_pr in zip([short_obj, long_obj, both_obj], [short_cum_pr, long_cum_pr, both_cum_pr]):
            try:
                # start_0 = time.time()
                gs_idx = frq_dev_plot_v3(gs, gs_idx, len_df, sample_len, obj[-2], cum_pr[-1], fontsize)
                # print("elapsed time :", time.time() - start_0)
            except Exception as e:
                gs_idx += 1
                print("error in frq_dev_plot_v3 :", e)

        plt.show()
        plt.close()
        return short_pr, short_obj, short_lvrg_arr, short_fee_arr, short_tpout_arr, short_bias_arr, \
               long_pr, long_obj, long_lvrg_arr, long_fee_arr, long_tpout_arr, long_bias_arr
    else:
        return [short_idep_res_obj[:-1], long_idep_res_obj[:-1], both_idep_res_obj[:-1]]

def idep_plot_v6(len_df, h, l, open_idx, side_arr, paired_res, inversion=False, sample_ratio=0.7, title_position=(0.5, 0.5), fontsize=15, signi=False):
    if not signi:
        plt.style.use(['dark_background', 'fast'])
        fig = plt.figure(figsize=(24, 8))
        gs = gridspec.GridSpec(nrows=2,  # row 몇 개
                               ncols=3,  # col 몇 개
                               height_ratios=[10, 1]
                               # height_ratios=[10, 10, 1]
                               )
    gs_idx = 0
    # plt.suptitle(key)

    if sample_ratio is not None:
      sample_len = int(len_df * sample_ratio)
    else:
      sample_len = len_df

    # ------ short & long data preparation ------ #
    # start_0 = time.time()
    valid_openi_arr, pair_idx_arr, pair_price_arr, lvrg_arr, fee_arr, tpout_arr, bias_arr = paired_res
    assert len(valid_openi_arr) != 0, "assert len(valid_openi_arr) != 0"
    short_valid_openi_idx = np.where(side_arr[valid_openi_arr] == OrderSide.SELL)[0]  # valid_openi_arr 에 대한 idx, # side_arr,
    long_valid_openi_idx = np.where(side_arr[valid_openi_arr] == OrderSide.BUY)[0]

    valid_open_idx = open_idx[valid_openi_arr].reshape(-1, 1)

    short_obj = np.hstack((pair_price_arr, pair_idx_arr, valid_open_idx))[short_valid_openi_idx]
    long_obj = np.hstack((pair_price_arr, pair_idx_arr, valid_open_idx))[long_valid_openi_idx]
    both_obj = np.vstack((short_obj, long_obj))
    print("short_obj.shape :", short_obj.shape)
    print("long_obj.shape :", long_obj.shape)

    short_obj, long_obj, both_obj = [np.split(obj_, 5, axis=1) for obj_ in [short_obj, long_obj, both_obj]]

    short_lvrg_arr, long_lvrg_arr = [lvrg_arr[openi_idx_].reshape(-1, 1) for openi_idx_ in [short_valid_openi_idx, long_valid_openi_idx]]
    short_fee_arr, long_fee_arr = [fee_arr[openi_idx_].reshape(-1, 1) for openi_idx_ in [short_valid_openi_idx, long_valid_openi_idx]]
    short_tpout_arr, long_tpout_arr = [tpout_arr[openi_idx_] for openi_idx_ in [short_valid_openi_idx, long_valid_openi_idx]]
    short_bias_arr, long_bias_arr = [bias_arr[openi_idx_] for openi_idx_ in [short_valid_openi_idx, long_valid_openi_idx]]
    # print("long_bias_arr.shape :", long_bias_arr.shape)
    # print("elapsed time :", time.time() - start_0)

    short_true_bias_idx = short_bias_arr[:, 0] <= short_bias_arr[:, 1] # info, threshold (등호의 유무가 매우 유의미함)
    long_true_bias_idx = long_bias_arr[:, 0] >= long_bias_arr[:, 1]

    # ------ plot_data ------ #
    try:
      # start_0 = time.time()
      short_pr = get_pr(OrderSide.SELL, *short_obj[:2], short_lvrg_arr, short_fee_arr, inversion)
      short_total_pr = to_total_pr(len_df, short_pr, short_obj[-2])
      short_cum_pr = np.cumprod(short_total_pr)
      short_liqd = liquidation(OrderSide.SELL, h, short_obj[:4], short_lvrg_arr, short_fee_arr)
      short_prcn, short_rc = precision(short_pr, short_true_bias_idx), recall(short_true_bias_idx)
      if signi:
        short_idep_res_obj = (short_prcn, short_rc) + get_res_info_nb_v2(sample_len, short_pr, short_total_pr, short_cum_pr, short_liqd)
      else:
        gs_idx = plot_info_v4(gs, gs_idx, sample_len, short_pr, short_total_pr, short_cum_pr, short_liqd, short_prcn, short_rc, short_lvrg_arr[-1], title_position, fontsize)
      # print("elapsed time :", time.time() - start_0)
    except Exception as e:
      gs_idx += 1
      print("error in short plot_data :", e)

    try:
      # start_0 = time.time()
      long_pr = get_pr(OrderSide.BUY, *long_obj[:2], long_lvrg_arr, long_fee_arr, inversion)
      long_total_pr = to_total_pr(len_df, long_pr, long_obj[-2])
      long_cum_pr = np.cumprod(long_total_pr)
      long_liqd = liquidation(OrderSide.BUY, l, long_obj[:4], long_lvrg_arr, long_fee_arr)
      long_prcn, long_rc = precision(long_pr, long_true_bias_idx), recall(long_true_bias_idx)
      if signi:
        long_idep_res_obj = (long_prcn, long_rc) + get_res_info_nb_v2(sample_len, long_pr, long_total_pr, long_cum_pr, long_liqd)
      else:
        gs_idx = plot_info_v4(gs, gs_idx, sample_len, long_pr, long_total_pr, long_cum_pr, long_liqd, long_prcn, long_rc, long_lvrg_arr[-1], title_position, fontsize)
      # print("elapsed time :", time.time() - start_0)
    except Exception as e:
      gs_idx += 1
      print("error in long plot_data :", e)

    try:
      # start_0 = time.time()
      both_pr = np.vstack((short_pr, long_pr))  # for 2d shape, obj 를 1d 로 만들지 않는 이상, pr 은 2d 유지될 것
      both_total_pr = to_total_pr(len_df, both_pr, both_obj[-2])
      both_cum_pr = np.cumprod(both_total_pr)
      both_liqd = min(short_liqd, long_liqd)
      both_true_bias_idx = np.hstack((short_true_bias_idx, long_true_bias_idx))  # for 1d shape
      both_prcn, both_rc = precision(both_pr, both_true_bias_idx), recall(both_true_bias_idx)
      if signi:
        both_idep_res_obj = (both_prcn, both_rc) + get_res_info_nb_v2(sample_len, both_pr, both_total_pr, both_cum_pr, both_liqd)
      else:
        gs_idx = plot_info_v4(gs, gs_idx, sample_len, both_pr, both_total_pr, both_cum_pr, both_liqd, both_prcn, both_rc, lvrg_arr[-1], title_position, fontsize)
      # print("elapsed time :", time.time() - start_0)
    except Exception as e:
      gs_idx += 1
      print("error in both plot_data :", e)

    if not signi:
        for obj, cum_pr in zip([short_obj, long_obj, both_obj], [short_cum_pr, long_cum_pr, both_cum_pr]):
          try:
            # start_0 = time.time()
            gs_idx = frq_dev_plot_v3(gs, gs_idx, len_df, sample_len, obj[-2], cum_pr[-1], fontsize)
            # print("elapsed time :", time.time() - start_0)
          except Exception as e:
            gs_idx += 1
            print("error in frq_dev_plot_v3 :", e)

        plt.show()
        plt.close()
        return short_pr, short_obj, short_lvrg_arr, short_fee_arr, short_tpout_arr, short_bias_arr, \
                long_pr, long_obj, long_lvrg_arr, long_fee_arr, long_tpout_arr, long_bias_arr
    else:
        return [short_idep_res_obj[:-1], long_idep_res_obj[:-1], both_idep_res_obj[:-1]]


def liquidation_v2(open_side, data_, obj_, lvrg, fee):  # v1 is deprecated
    if type(obj_) == list:
        obj = list(zip(*obj_))

    np_obj = np.array(obj_).T[0]
    assert len(np_obj.shape) == 2

    en_p = obj_[0]

    # iin == iout 인 경우 분리
    en_idx = np_obj[:, 2]
    ex_idx = np_obj[:, 3]
    equal_idx = en_idx == ex_idx

    if open_side == "SELL":
        max_high = np.full_like(en_p, np.nan)
        max_high[~equal_idx] = np.array([np.max(data_[int(iin):int(iout)]) for _, _, iin, iout in np_obj[~equal_idx, :4]]).reshape(-1, 1)
        return np.nanmin((en_p / max_high - fee - 1) * lvrg + 1)
    else:
        min_low = np.full_like(en_p, np.nan)
        min_low[~equal_idx] = np.array([np.min(data_[int(iin):int(iout)]) for _, _, iin, iout in np_obj[~equal_idx, :4]]).reshape(-1, 1)
        return np.nanmin((min_low / en_p - fee - 1) * lvrg + 1)

def liquidation(open_side, data_, obj_, lvrg, fee):  # much faster
    if type(obj_) == list:
        obj_ = list(zip(*obj_))

    if open_side == "SELL":
        return np.min([(ep_ / np.max(data_[int(iin):int(iout)]) - fee - 1) * lvrg + 1 for ep_, _, iin, iout in obj_ if iin != iout])
    else:
        return np.min([(np.min(data_[int(iin):int(iout)]) / ep_ - fee - 1) * lvrg + 1 for ep_, _, iin, iout in obj_ if iin != iout])


def frq_dev_plot_v4(gs, gs_idx, len_df, sample_len, exit_idx, bias_arr, acc_pr, fontsize):
    ax = plt.subplot(gs[gs_idx])

    exit_idx_ravel = exit_idx.ravel()
    bias_ravel = bias_arr.ravel()
    transform = ax.get_xaxis_transform()

    plt.fill_between(exit_idx_ravel, 0, 1, where=bias_ravel > 0,
                     facecolor='#00ff00', alpha=1, transform=transform)
    # plt.fill_between(exit_idx_ravel, 0, 1, where=bias_ravel < 1,
    #                 facecolor='#ff00ff', alpha=1, transform=transform)

    title_msg = "periodic_pr\n acc_day : {:.4f}\n month : {:.4f}\n year : {:.4f}"  # \n rev_acc_day : {:.4f}\n month : {:.4f}\n year : {:.4f}"
    plt.title(title_msg.format(*get_period_pr(sample_len, acc_pr), fontsize=fontsize))

    return gs_idx + 1

def frq_dev_plot_v3(gs, gs_idx, len_df, sample_len, exit_idx, acc_pr, fontsize):
    plt.subplot(gs[gs_idx])
    frq_dev = np.zeros(len_df)
    if type(exit_idx) != int:
      exit_idx = exit_idx.astype(int)
    frq_dev[exit_idx] = 1
    plt.plot(frq_dev)

    title_msg = "periodic_pr\n acc_day : {:.4f}\n month : {:.4f}\n year : {:.4f}"  # \n rev_acc_day : {:.4f}\n month : {:.4f}\n year : {:.4f}"
    plt.title(title_msg.format(*get_period_pr(sample_len, acc_pr), fontsize=fontsize))

    return gs_idx + 1


def to_total_pr(len_df, pr, exit_idx):
  total_pr = np.ones(len_df)
  if type(exit_idx) != int:
    exit_idx = exit_idx.astype(int)
  total_pr[exit_idx] = pr

  return total_pr

def plot_info_v6(gs, gs_idx, sample_len, tr, prcn, rc, bars_in, pr, total_pr, cum_pr, liqd, leverage, title_position, fontsize):
  try:
    plt.subplot(gs[gs_idx])
    idep_res_obj = get_res_info_nb_v2(sample_len, pr, total_pr, cum_pr, liqd)
    plt.plot(cum_pr)
    plt.plot(idep_res_obj[-1], color='gold')
    if sample_len is not None:
      plt.axvline(sample_len, alpha=1., linestyle='--', color='#ffeb3b')
    title_str = "tr : {:.3f}\n hhm : {:.3f}\n hlm : {:.3f}\n bars_in : {:.3f}\n len_pr : {}\n dpf : {:.3f}\n wr : {:.3f}\n sr : {:.3f}\n acc_pr : {:.3f}\n sum_pr : {:.3f}\n" +\
              "min_pr : {:.3f}\n liqd : {:.3f}\n acc_mdd : -{:.3f}\n sum_mdd : -{:.3f}\n leverage {}"
    plt.title(title_str.format(tr, rc, prcn, bars_in, *idep_res_obj[:-1], leverage), position=title_position, fontsize=fontsize)
  except Exception as e:
    print("error in plot_info :", e)

  return gs_idx + 1

def plot_info_v5(gs, gs_idx, sample_len, tr, prcn, rc, pr, total_pr, cum_pr, liqd, leverage, title_position, fontsize):
  try:
    plt.subplot(gs[gs_idx])
    idep_res_obj = get_res_info_nb_v2(sample_len, pr, total_pr, cum_pr, liqd)
    plt.plot(cum_pr)
    plt.plot(idep_res_obj[-1], color='gold')
    if sample_len is not None:
      plt.axvline(sample_len, alpha=1., linestyle='--', color='#ffeb3b')
    title_str = "tr : {:.3f}\n prcn : {:.3f}\n wave_bias : {:.3f}\n len_pr : {}\n dpf : {:.3f}\n wr : {:.3f}\n sr : {:.3f}\n acc_pr : {:.3f}\n sum_pr : {:.3f}\n" +\
              "min_pr : {:.3f}\n liqd : {:.3f}\n acc_mdd : -{:.3f}\n sum_mdd : -{:.3f}\n leverage {}"
    plt.title(title_str.format(tr, prcn, rc, *idep_res_obj[:-1], leverage), position=title_position, fontsize=fontsize)
  except Exception as e:
    print("error in plot_info :", e)

  return gs_idx + 1

def plot_info_v4(gs, gs_idx, sample_len, pr, total_pr, cum_pr, liqd, prcn, rc, leverage, title_position, fontsize):
  try:
    plt.subplot(gs[gs_idx])
    idep_res_obj = get_res_info_nb_v2(sample_len, pr, total_pr, cum_pr, liqd)
    plt.plot(cum_pr)
    plt.plot(idep_res_obj[-1], color='gold')
    if sample_len is not None:
      plt.axvline(sample_len, alpha=1., linestyle='--', color='#ffeb3b')
    # title_str = "prcn : {:.3f} rc : {:.3f}\n len_pr : {} dpf : {:.3f}\n wr : {:.3f} sr : {:.3f}\n acc_pr : {:.3f} sum_pr : {:.3f}\n" +\
    #           "min_pr : {:.3f} liqd : {:.3f}\n acc_mdd : -{:.3f} sum_mdd : -{:.3f}\n leverage {}"
    title_str = "prcn : {:.3f}\n rc : {:.3f}\n len_pr : {}\n dpf : {:.3f}\n wr : {:.3f}\n sr : {:.3f}\n acc_pr : {:.3f}\n sum_pr : {:.3f}\n" +\
              "min_pr : {:.3f}\n liqd : {:.3f}\n acc_mdd : -{:.3f}\n sum_mdd : -{:.3f}\n leverage {}"
    plt.title(title_str.format(prcn, rc, *idep_res_obj[:-1], leverage), position=title_position, fontsize=fontsize)
  except Exception as e:
    print("error in plot_info :", e)

  return gs_idx + 1


def get_res_info(len_df, np_pr):
  wr = len(np_pr[np_pr > 1]) / len(np_pr[np_pr != 1])
  sr = sharpe_ratio(np_pr)
  total_pr = np.cumprod(np_pr)
  sum_pr = get_sum_pr_nb(np_pr)
  min_pr = np.min(np_pr)

  len_pr = len(np_pr)
  assert len_pr != 0
  dpf = (len_df / 1440) / len_pr # devision zero warning

  acc_mdd = mdd(total_pr)
  sum_mdd = mdd(sum_pr)

  return len_pr, dpf, wr, sr, total_pr[-1], sum_pr[-1], min_pr, acc_mdd, sum_mdd, total_pr, sum_pr


# @jit  # almost equal
def get_res_info_nb_v2(len_df, np_pr, total_pr, acc_pr, liqd):
    wr = len(np_pr[np_pr > 1]) / len(np_pr[np_pr != 1])
    sr = sharpe_ratio(np_pr)
    sum_pr = get_sum_pr_nb(total_pr)
    min_pr = np.min(np_pr)

    len_pr = len(np_pr)
    assert len_pr != 0
    dpf = (len_df / 1440) / len_pr  # devision zero warning

    acc_mdd = mdd(acc_pr)
    sum_mdd = mdd(sum_pr)

    return len_pr, dpf, wr, sr, acc_pr[-1], sum_pr[-1], min_pr, liqd, acc_mdd, sum_mdd, sum_pr


# @jit  # almost equal
def get_res_info_nb(len_df, np_pr, acc_pr, liqd):
    wr = len(np_pr[np_pr > 1]) / len(np_pr[np_pr != 1])
    sr = sharpe_ratio(np_pr)
    sum_pr = get_sum_pr_nb(np_pr)
    min_pr = np.min(np_pr)

    len_pr = len(np_pr)
    assert len_pr != 0
    dpf = (len_df / 1440) / len_pr  # devision zero warning

    acc_mdd = mdd(acc_pr)
    sum_mdd = mdd(sum_pr)

    return len_pr, dpf, wr, sr, acc_pr[-1], sum_pr[-1], min_pr, liqd, acc_mdd, sum_mdd, sum_pr


# @jit
def get_sum_pr_nb(np_pr):
  for_sum_pr = np_pr - 1
  for_sum_pr[0] = 1
  sum_pr = np.cumsum(for_sum_pr)
  sum_pr = np.where(sum_pr < 0, 0, sum_pr)

  return sum_pr


def calc_win_ratio(win_cnt, lose_cnt):
    if win_cnt + lose_cnt == 0:
        win_ratio = 0
    else:
        win_ratio = win_cnt / (win_cnt + lose_cnt) * 100

    return win_ratio


def sharpe_ratio(pr_arr, risk_free_rate=0.0, multiply_frq=False):
  if len(pr_arr) == 0:
    return 0
  else:
    pr_pct = pr_arr - 1
    sr_ = (np.mean(pr_pct) - risk_free_rate) / np.std(pr_pct)
    if multiply_frq:
        sr_ = len(pr_pct) ** (1 / 2) * sr_
    return sr_


def mdd(pr):
  rollmax_pr = np.maximum.accumulate(pr)
  return np.max((rollmax_pr - pr) / rollmax_pr)


#     필요한 col index 정리    #
def get_col_idxs(df, cols):
  return [df.columns.get_loc(col) for col in cols]

def get_pr_v4(open_side, h, l, obj, tpout, lvrg, fee, p_ranges, p_qty_ratio, inversion=False):  # --> 여기서 사용하는 ex_p = ex_p

    en_p = obj[0]
    # ex_p = obj[1]
    tp, out = np.split(tpout, 2, axis=1)
    len_p = len(p_ranges)
    en_ps, tps, outs, lvrgs, fees = [np.tile(arr_, (1, len_p)) for arr_ in [en_p, tp, out, lvrg, fee]]

    np_obj = np.array(obj).T[0]
    assert len(np_obj.shape) == 2

    # iin == iout 인 경우 분리
    en_idx = np_obj[:, 2]
    ex_idx = np_obj[:, 3]
    equal_idx = en_idx == ex_idx    # equal_idx 는 어차피 out 임

    min_low = np.full_like(en_p, np.nan)
    min_low[~equal_idx] = np.array([np.min(l[int(iin + 1):int(iout + 1)]) for _, _, iin, iout in np_obj[~equal_idx, :4]]).reshape(-1, 1)  # start from iin + 1 (tp 체결을 entry_idx 부터 보지 않음)
    max_high = np.full_like(en_p, np.nan)
    max_high[~equal_idx] = np.array([np.max(h[int(iin + 1):int(iout + 1)]) for _, _, iin, iout in np_obj[~equal_idx, :4]]).reshape(-1, 1)

    if open_side == "SELL":
        p_tps = en_ps - (en_ps - tps) * p_ranges
        # min_low = np.full_like(en_p, np.nan)
        # min_low[~equal_idx] = np.array([np.min(l[int(iin + 1):int(iout + 1)]) for _, _, iin, iout in np_obj[~equal_idx, :4]]).reshape(-1, 1)  # start from iin + 1 (tp 체결을 entry_idx 부터 보지 않음)
        tp_idx = (np.tile(min_low, (1, len_p)) <= p_tps) * (np.tile(max_high, (1, len_p)) <= outs)  # entry_idx 포함해서 out touch 금지 (보수적 검증)
    else:
        p_tps = en_ps + (tps - en_ps) * p_ranges
        # max_high = np.full_like(en_p, np.nan)
        # max_high[~equal_idx] = np.array([np.max(h[int(iin + 1):int(iout + 1)]) for _, _, iin, iout in np_obj[~equal_idx, :4]]).reshape(-1, 1)
        tp_idx = (np.tile(max_high, (1, len_p)) >= p_tps) * (np.tile(min_low, (1, len_p)) >= outs)

    ex_ps = outs.copy()
    ex_ps[tp_idx] = p_tps[tp_idx]

    if open_side == "SELL":
        if not inversion:
            pr = ((en_ps / ex_ps - fees - 1) * lvrgs * p_qty_ratio).sum(axis=1) + 1
            # ------ liquidation ------ #
            max_high = np.full_like(en_p, np.nan)
            max_high[~equal_idx] = np.array([np.max(h[int(iin):int(iout)]) for _, _, iin, iout in np_obj[~equal_idx, :4]]).reshape(-1, 1)
            liqd = np.nanmin((en_p / max_high - fee - 1) * lvrg + 1)
        else:
            pr = ((ex_ps / en_ps - fees - 1) * lvrgs * p_qty_ratio).sum(axis=1) + 1
            # ------ liquidation ------ #
            min_low = np.full_like(en_p, np.nan)
            min_low[~equal_idx] = np.array([np.min(l[int(iin):int(iout)]) for _, _, iin, iout in np_obj[~equal_idx, :4]]).reshape(-1, 1)
            liqd = np.nanmin((min_low / en_p - fee - 1) * lvrg + 1)
    else:
        if not inversion:
            pr = ((ex_ps / en_ps - fees - 1) * lvrgs * p_qty_ratio).sum(axis=1) + 1
            # ------ liquidation ------ #
            min_low = np.full_like(en_p, np.nan)
            min_low[~equal_idx] = np.array([np.min(l[int(iin):int(iout)]) for _, _, iin, iout in np_obj[~equal_idx, :4]]).reshape(-1, 1)
            liqd = np.nanmin((min_low / en_p - fee - 1) * lvrg + 1)
        else:
            pr = ((en_ps / ex_ps - fees - 1) * lvrgs * p_qty_ratio).sum(axis=1) + 1
            # ------ liquidation ------ #
            max_high = np.full_like(en_p, np.nan)
            max_high[~equal_idx] = np.array([np.max(h[int(iin):int(iout)]) for _, _, iin, iout in np_obj[~equal_idx, :4]]).reshape(-1, 1)
            liqd = np.nanmin((en_p / max_high - fee - 1) * lvrg + 1)

    return pr.reshape(-1, 1), liqd

def get_pr_v3(open_side, h, l, obj, tpout, lvrg, fee, p_ranges, p_qty_ratio, inversion=False):  # --> 여기서 사용하는 ex_p = ex_p

    en_p = obj[0]
    # ex_p = obj[1]
    tp, out = np.split(tpout, 2, axis=1)
    len_p = len(p_ranges)
    en_ps, tps, outs, lvrgs, fees = [np.tile(arr_, (1, len_p)) for arr_ in [en_p, tp, out, lvrg, fee]]

    np_obj = np.array(obj).T[0]
    assert len(np_obj.shape) == 2

    # iin == iout 인 경우 분리
    en_idx = np_obj[:, 2]
    ex_idx = np_obj[:, 3]
    equal_idx = en_idx == ex_idx

    if open_side == "SELL":
        p_tps = en_ps - (en_ps - tps) * p_ranges
        # min_low = np.array([np.min(l[int(iin):int(iout + 1)]) for _, _, iin, iout in list(zip(*obj[:4]))]).reshape(-1, 1)  # -> deprecated, start from iin + 1
        min_low = np.full_like(en_p, np.nan)
        min_low[~equal_idx] = np.array([np.min(l[int(iin + 1):int(iout + 1)]) for _, _, iin, iout in np_obj[~equal_idx, :4]]).reshape(-1, 1)
        tp_idx = np.tile(min_low, (1, len_p)) <= p_tps

        # ------ liquidation ------ #
        max_high = np.full_like(en_p, np.nan)
        max_high[~equal_idx] = np.array([np.max(h[int(iin):int(iout)]) for _, _, iin, iout in np_obj[~equal_idx, :4]]).reshape(-1, 1)
        liqd = np.nanmin((en_p / max_high - fee - 1) * lvrg + 1)
    else:
        p_tps = en_ps + (tps - en_ps) * p_ranges
        # max_high = np.array([np.max(h[int(iin):int(iout + 1)]) for _, _, iin, iout in list(zip(*obj[:4]))]).reshape(-1, 1)
        max_high = np.full_like(en_p, np.nan)
        max_high[~equal_idx] = np.array([np.max(h[int(iin + 1):int(iout + 1)]) for _, _, iin, iout in np_obj[~equal_idx, :4]]).reshape(-1, 1)
        tp_idx = np.tile(max_high, (1, len_p)) >= p_tps

        # ------ liquidation ------ #
        min_low = np.full_like(en_p, np.nan)
        min_low[~equal_idx] = np.array([np.min(l[int(iin):int(iout)]) for _, _, iin, iout in np_obj[~equal_idx, :4]]).reshape(-1, 1)
        liqd = np.nanmin((min_low / en_p - fee - 1) * lvrg + 1)

    ex_ps = outs.copy()
    ex_ps[tp_idx] = p_tps[tp_idx]

    if open_side == "SELL":
        if not inversion:
            pr = ((en_ps / ex_ps - fees - 1) * lvrgs * p_qty_ratio).sum(axis=1) + 1
        else:
            pr = ((ex_ps / en_ps - fees - 1) * lvrgs * p_qty_ratio).sum(axis=1) + 1
    else:
        if not inversion:
            pr = ((ex_ps / en_ps - fees - 1) * lvrgs * p_qty_ratio).sum(axis=1) + 1
        else:
            pr = ((en_ps / ex_ps - fees - 1) * lvrgs * p_qty_ratio).sum(axis=1) + 1

    return pr.reshape(-1, 1), liqd

def get_pr_v2(open_side, h, l, obj, tpout, lvrg, fee, p_ranges, p_qty_ratio, inversion=False):  # --> 여기서 사용하는 ex_p = ex_p

    en_p = obj[0]
    # ex_p = obj[1]
    tp, out = np.split(tpout, 2, axis=1)
    len_p = len(p_ranges)
    en_ps, tps, outs, lvrgs, fees = [np.tile(arr_, (1, len_p)) for arr_ in [en_p, tp, out, lvrg, fee]]

    if open_side == "SELL":
        p_tps = en_ps - (en_ps - tps) * p_ranges
        min_low = np.array([np.min(l[int(iin):int(iout + 1)]) for _, _, iin, iout in list(zip(*obj[:4]))]).reshape(-1, 1)  # / rtc_gap
        res = np.tile(min_low, (1, len_p)) <= p_tps
    else:
        p_tps = en_ps + (tps - en_ps) * p_ranges
        max_high = np.array([np.max(h[int(iin):int(iout + 1)]) for _, _, iin, iout in list(zip(*obj[:4]))]).reshape(-1, 1)  # / rtc_gap
        res = np.tile(max_high, (1, len_p)) >= p_tps

    ex_ps = outs.copy()
    ex_ps[res] = p_tps[res]

    if open_side == "SELL":
        if not inversion:
            pr = ((en_ps / ex_ps - fees - 1) * lvrgs * p_qty_ratio).sum(axis=1) + 1
        else:
            pr = ((ex_ps / en_ps - fees - 1) * lvrgs * p_qty_ratio).sum(axis=1) + 1
    else:
        if not inversion:
            pr = ((ex_ps / en_ps - fees - 1) * lvrgs * p_qty_ratio).sum(axis=1) + 1
        else:
            pr = ((en_ps / ex_ps - fees - 1) * lvrgs * p_qty_ratio).sum(axis=1) + 1

    return pr.reshape(-1, 1)

def get_pr(open_side, en_p, ex_p, lvrg, fee, inversion=False):
  assert len(ex_p) == len(en_p)
  if open_side == "SELL":

    if not inversion:
      pr = (en_p / ex_p - fee - 1) * lvrg + 1
    else:
      pr = (ex_p / en_p - fee - 1) * lvrg + 1
  else:
    if not inversion:
      pr = (ex_p / en_p - fee - 1) * lvrg + 1
    else:
      pr = (en_p / ex_p - fee - 1) * lvrg + 1

  return pr


# @jit
def get_pr_nb(open_side, ep, tp, lvrg, fee):
  assert len(tp) == len(ep)
  if open_side == "SELL":
    pr = (ep / tp - fee - 1) * lvrg + 1
  else: #if open_side == "BUY":
    pr = (tp / ep - fee - 1) * lvrg + 1

  return pr


def get_period_pr(len_df, pr_):
    days = len_df / 1440
    months = days / 30
    years = days / 365

    return [pr_ ** (1 / period) for period in [days, months, years]]


def p_pr_plot(gs_, frq_dev_, res_df, pr_, rev_pr_, fontsize_):
    try:
        plt.subplot(gs_)
        plt.plot(frq_dev_)

        title_msg = "periodic_pr\n acc_day : {:.4f}\n month : {:.4f}\n year : {:.4f}\n rev_acc_day : {:.4f}\n month : {:.4f}\n year : {:.4f}"
        len_df = len(res_df)
        plt.title(title_msg.format(*get_period_pr(len_df, pr_[-1]), *get_period_pr(len_df, rev_pr_[-1])), fontsize=fontsize_)

    except Exception as e:
        print("error in p_pr_plot :", e)

    return


def frq_dev_plot(res_df, trade_list, side_list, plot=True, figsize=(16, 5)):

    np_trade_list = np.array(trade_list)
    np_side_list = np.array(side_list)
    l_idx = np.argwhere(np_side_list == 'l')
    s_idx = np.argwhere(np_side_list == 's')

    l_trade_list = np_trade_list[l_idx]
    s_trade_list = np_trade_list[s_idx]

    frq_dev = np.zeros(len(res_df))
    l_frq_dev = np.zeros(len(res_df))
    s_frq_dev = np.zeros(len(res_df))

    frq_dev[np_trade_list[:, [0], [0]]] = 1
    l_frq_dev[l_trade_list[:, [0], [0]]] = 1
    s_frq_dev[s_trade_list[:, [0], [0]]] = 1

    if plot:
        plt.figure(figsize=figsize)

        gs = gridspec.GridSpec(nrows=1,  # row 몇 개
                               ncols=3,  # col 몇 개
                               # height_ratios=[1, 1, 1]
                               )

        plt.subplot(gs[0])
        plt.plot(frq_dev)
        # plt.show()

        plt.subplot(gs[1])
        plt.plot(s_frq_dev)
        # plt.show()

        plt.subplot(gs[2])
        plt.plot(l_frq_dev)
        plt.show()

    else:
        return frq_dev, s_frq_dev, l_frq_dev
