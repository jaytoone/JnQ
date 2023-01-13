import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy import stats, signal
# from numba import jit, njit
# import torch


class OrderSide:
    BUY = "BUY"
    SELL = "SELL"
    INVALID = None

def kde_plot_v2(v, c, kde_factor=0.15, num_samples=100):

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
  print("pkx :", pkx)

  # plt.figure(figsize=(10,5))
  plt.hist(v, weights=c, bins=num_samples, alpha=.8, edgecolor='black')
  plt.plot(kdx, kdy, color='white')
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

def get_max_outg(open_side, config, ohlc_list, obj, tpout_arr, rtc_gap):

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
      max_outg = np.array([np.max(h[int(en_idx):int(en_idx + idx_gap + 1)]) - ep_ for en_idx, idx_gap, ep_ in zip(en_idxs, min_idxg, en_p)]) / rtc_gap
    else:
      max_outg = np.array([ep_ - np.min(l[int(en_idx):int(en_idx + idx_gap + 1)]) for en_idx, idx_gap, ep_ in zip(en_idxs, min_idxg, en_p)]) / rtc_gap # out_idx 포함

    return max_outg[~nan_idx], open_idxs[~nan_idx].astype(int)  # true_bias 의 outg data 만 사용

def get_max_tpg(open_side, ohlc_list, pr_, obj_, rtc_gap):  # much faster
    if type(obj_) == list:
        obj_ = list(zip(*obj_))

    h, l = ohlc_list[1:3]

    if open_side == "SELL":
      max_tpg = np.array([ep_ - np.min(l[int(iin):int(iout + 1)]) for ep_, _, iin, iout in obj_]) / rtc_gap # out 포함
      # max_outg = np.array([np.max(h[int(iin):int(iout + 1)]) - ep_ for ep_, _, iin, iout in obj_]) / rtc_gap
    else:
      max_tpg = np.array([np.max(h[int(iin):int(iout + 1)]) - ep_ for ep_, _, iin, iout in obj_]) / rtc_gap
      # max_outg = np.array([ep_ - np.min(l[int(iin):int(iout + 1)]) for ep_, _, iin, iout in obj_]) / rtc_gap

    return max_tpg[pr_ < 1]  # out 된 case 만 tpg 담음

def precision(pr_list, true_idx):   # true_pr in true_bias / true_bias
  true_bias_pr = pr_list[true_idx].ravel()
  return np.sum(true_bias_pr > 1) / len(true_bias_pr)  # 차원을 고려한 계산

def recall(true_idx):   # true_bias / total_entry
  return np.sum(true_idx) / len(true_idx) #  2.16 µs per loop (len) --> 3.78 µs per loop

def mr_res(input_data, rpsn_v, inval_v, np_ones):
    return input_data == rpsn_v if rpsn_v > inval_v else np_ones


# 511 µs 498  (v1) ->  515 507 µs per loop (v2) || nohlc : 1.12 ms (v1) -> 1.14 ms (v2)
def multi_mr_v3(in_data_list, param_list, inval_list, obj, np_ones):
    assert len(param_list) == len(in_data_list), "assert len(param_list) == len(in_data_list)"
    short_mr_res_idx = np_ones.copy()
    long_mr_res_idx = np_ones.copy()

    for in_obj in zip(in_data_list, param_list, inval_list):
        short_mr_res_idx *= mr_res(*in_obj, np_ones)
        long_mr_res_idx *= mr_res(*in_obj, np_ones)

    short_ep_loc_obj = adj_mr(obj[0], short_mr_res_idx.reshape(-1, ))
    long_ep_loc_obj = adj_mr(obj[1], long_mr_res_idx.reshape(-1, ))

    return short_ep_loc_obj, long_ep_loc_obj  # obj 는 plot_check + frq_dev 를 위해 필요

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

def idep_plot_v5(len_df, h, l, open_idx, side_arr, paired_res, inversion=False, sample_ratio=0.7, title_position=(0.5, 0.5), fontsize=15, signi=False):
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
    valid_openi_arr, pair_idx_arr, pair_price_arr, lvrg_arr, fee_arr, tpout_arr = paired_res
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
    # print("elapsed time :", time.time() - start_0)

    # ------ plot_data ------ #
    try:
      # start_0 = time.time()
      short_pr = get_pr(OrderSide.SELL, *short_obj[:2], short_lvrg_arr, short_fee_arr, inversion)
      short_total_pr = to_total_pr(len_df, short_pr, short_obj[-2])
      short_cum_pr = np.cumprod(short_total_pr)
      s_liqd = liquidation(OrderSide.SELL, h, short_obj[:4], short_lvrg_arr, short_fee_arr)
      if signi:
        short_idep_res_obj = get_res_info_nb_v2(sample_len, short_pr, short_total_pr, short_cum_pr, s_liqd)
      else:
        gs_idx = plot_info_v3(gs, gs_idx, sample_len, short_pr, short_total_pr, short_cum_pr, s_liqd, short_lvrg_arr[-1], title_position, fontsize)
      # print("elapsed time :", time.time() - start_0)
    except:
      gs_idx += 1

    try:
      # start_0 = time.time()
      long_pr = get_pr(OrderSide.BUY, *long_obj[:2], long_lvrg_arr, long_fee_arr, inversion)
      long_total_pr = to_total_pr(len_df, long_pr, long_obj[-2])
      long_cum_pr = np.cumprod(long_total_pr)
      l_liqd = liquidation(OrderSide.BUY, l, long_obj[:4], long_lvrg_arr, long_fee_arr)
      if signi:
        long_idep_res_obj = get_res_info_nb_v2(sample_len, long_pr, long_total_pr, long_cum_pr, l_liqd)
      else:
        gs_idx = plot_info_v3(gs, gs_idx, sample_len, long_pr, long_total_pr, long_cum_pr, l_liqd, long_lvrg_arr[-1], title_position, fontsize)
      # print("elapsed time :", time.time() - start_0)
    except:
      gs_idx += 1

    try:
      # start_0 = time.time()
      both_pr = np.vstack((short_pr, long_pr))
      both_total_pr = to_total_pr(len_df, both_pr, both_obj[-2])
      both_cum_pr = np.cumprod(both_total_pr)
      b_liqd = min(s_liqd, l_liqd)
      if signi:
        both_idep_res_obj = get_res_info_nb_v2(sample_len, both_pr, both_total_pr, both_cum_pr, b_liqd)
      else:
        gs_idx = plot_info_v3(gs, gs_idx, sample_len, both_pr, both_total_pr, both_cum_pr, b_liqd, lvrg_arr[-1], title_position, fontsize)
      # print("elapsed time :", time.time() - start_0)
    except:
      gs_idx += 1

    if not signi:
        for obj, cum_pr in zip([short_obj, long_obj, both_obj], [short_cum_pr, long_cum_pr, both_cum_pr]):
          try:
            # start_0 = time.time()
            gs_idx = frq_dev_plot_v3(gs, gs_idx, len_df, sample_len, obj[-2], cum_pr[-1], fontsize)
            # print("elapsed time :", time.time() - start_0)
          except:
            gs_idx += 1

        plt.show()
        plt.close()
        return short_pr, short_obj, short_lvrg_arr, short_fee_arr, short_tpout_arr, long_pr, long_obj, long_lvrg_arr, long_fee_arr, long_tpout_arr
    else:
        return [short_idep_res_obj[:-1], long_idep_res_obj[:-1], both_idep_res_obj[:-1]]


def idep_plot_v4(len_df, h, l, open_idx, side_arr, paired_res, inversion=False, sample_ratio=0.7, title_position=(0.5, 0.5), fontsize=15):
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
    valid_openi_arr, pair_idx_arr, pair_price_arr, lvrg_arr, fee_arr, tpout_arr = paired_res
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
    # print("elapsed time :", time.time() - start_0)

    # ------ plot_data ------ #
    try:
      # start_0 = time.time()
      short_pr = get_pr(OrderSide.SELL, *short_obj[:2], short_lvrg_arr, short_fee_arr, inversion)
      short_total_pr = to_total_pr(len_df, short_pr, short_obj[-2])
      short_cum_pr = np.cumprod(short_total_pr)
      s_liqd = liquidation(OrderSide.SELL, h, short_obj[:4], short_lvrg_arr, short_fee_arr)
      gs_idx = plot_info_v3(gs, gs_idx, sample_len, short_pr, short_total_pr, short_cum_pr, s_liqd, short_lvrg_arr[-1], title_position, fontsize)
      # print("elapsed time :", time.time() - start_0)
    except:
      gs_idx += 1

    try:
      # start_0 = time.time()
      long_pr = get_pr(OrderSide.BUY, *long_obj[:2], long_lvrg_arr, long_fee_arr, inversion)
      long_total_pr = to_total_pr(len_df, long_pr, long_obj[-2])
      long_cum_pr = np.cumprod(long_total_pr)
      l_liqd = liquidation(OrderSide.BUY, l, long_obj[:4], long_lvrg_arr, long_fee_arr)
      gs_idx = plot_info_v3(gs, gs_idx, sample_len, long_pr, long_total_pr, long_cum_pr, l_liqd, long_lvrg_arr[-1], title_position, fontsize)
      # print("elapsed time :", time.time() - start_0)
    except:
      gs_idx += 1

    try:
      # start_0 = time.time()
      both_pr = np.vstack((short_pr, long_pr))
      both_total_pr = to_total_pr(len_df, both_pr, both_obj[-2])
      both_cum_pr = np.cumprod(both_total_pr)
      b_liqd = min(s_liqd, l_liqd)
      gs_idx = plot_info_v3(gs, gs_idx, sample_len, both_pr, both_total_pr, both_cum_pr, b_liqd, lvrg_arr[-1], title_position, fontsize)
      # print("elapsed time :", time.time() - start_0)
    except:
      gs_idx += 1

    for obj, cum_pr in zip([short_obj, long_obj, both_obj], [short_cum_pr, long_cum_pr, both_cum_pr]):
      try:
        # start_0 = time.time()
        gs_idx = frq_dev_plot_v3(gs, gs_idx, len_df, sample_len, obj[-2], cum_pr[-1], fontsize)
        # print("elapsed time :", time.time() - start_0)
      except:
        gs_idx += 1

    plt.show()
    plt.close()

    return short_pr, short_obj, short_lvrg_arr, short_fee_arr, short_tpout_arr, long_pr, long_obj, long_lvrg_arr, long_fee_arr, long_tpout_arr

# 2.6 s per loop (torch) -->  2.36 s (cpu) # 1.58 s per loop --> 1.58 (param added)
def idep_plot_torch_v3(data, hl_col, len_df, short_ep_loc_obj, long_ep_loc_obj, leverage, fee, inversion=False,
                       to_cpu=False, sample_ratio=0.7):
    #     len(pr_info) not enough for gpu effect    #
    if to_cpu:
        short_ep_loc_obj = [gpu_to_cpu(res_) for res_ in short_ep_loc_obj]
        long_ep_loc_obj = [gpu_to_cpu(res_) for res_ in long_ep_loc_obj]
        # print(type(short_ep_loc_obj))

    plt.style.use(['dark_background', 'fast'])

    fig = plt.figure(figsize=(16, 8))
    gs = gridspec.GridSpec(nrows=2,  # row 몇 개
                           ncols=2,  # col 몇 개
                           height_ratios=[10, 1]
                           # height_ratios=[10, 10, 1]
                           )
    gs_idx = 0
    # plt.suptitle(key)

    fontsize = 15
    title_position = (0.5, 0.5)

    v_line = int(len_df * sample_ratio)

    short_pr = get_pr(OrderSide.SELL, *short_ep_loc_obj[:2], leverage, fee, inversion)  # 76.8 µs per loop (torch) -> 9.96 µs (with cpu) ...
    long_pr = get_pr(OrderSide.BUY, *long_ep_loc_obj[:2], leverage, fee, inversion)

    s_liqd = liquidation(OrderSide.SELL, data[:, hl_col[0]], short_ep_loc_obj, leverage, fee)
    short_total_pr = to_total_pr(len_df, short_pr, short_ep_loc_obj[-1])
    short_cum_pr = np.cumprod(short_total_pr)
    gs_idx = plot_info_v2(gs, gs_idx, len_df, short_pr, short_total_pr, short_cum_pr, s_liqd, leverage, v_line, title_position, fontsize)

    l_liqd = liquidation(OrderSide.BUY, data[:, hl_col[1]], long_ep_loc_obj, leverage, fee)
    long_total_pr = to_total_pr(len_df, long_pr, long_ep_loc_obj[-1])
    long_cum_pr = np.cumprod(long_total_pr)
    gs_idx = plot_info_v2(gs, gs_idx, len_df, long_pr, long_total_pr, long_cum_pr, l_liqd, leverage, v_line, title_position, fontsize)

    gs_idx = frq_dev_plot_v2(gs, gs_idx, len_df, short_ep_loc_obj[-1], short_cum_pr[-1], fontsize)
    gs_idx = frq_dev_plot_v2(gs, gs_idx, len_df, long_ep_loc_obj[-1], long_cum_pr[-1], fontsize)

    plt.show()
    plt.close()

def get_select_obj_v2(in_data_list, select_param, inval_list, obj, np_ones, side_col):
  res_obj = np.array([multi_mr_v3(in_data_list, pos_param, inval_list, obj, np_ones) for pos_param in select_param])
  _, unq_idx = np.unique(np.hstack(res_obj[:, side_col, -1]), return_index=True)  # 현재, multi_mr_v2 return value 가 short & long 같이 제공해서 side_col 필요함
  res = [np.hstack(res_obj[:, 0, col_])[unq_idx] for col_ in range(res_obj.shape[2])]

  if len(res_obj) == 1:
      res[-2:] = np.array(res[-2:]).astype(int)  # idx integer only supported
  return res



#   정확한 test_ratio 를 구하려면, 기존 res_df 의 test_ratio timeidx 를 구하고, new_res_df 에서 test_ratio 을 추출해야함
def test_ratio_by_tsidx(res_df, ts_str):
  test_idx = np.argwhere(res_df.index == pd.to_datetime(ts_str))
  assert len(test_idx) > 0, "assert len(test_idx) > 0"
  return 1 - test_idx.item() / len(res_df)

def get_tsidx_by_ratio(res_df, test_ratio):
  t_idx = int(len(res_df) * (1 - test_ratio))
  return res_df.index[t_idx]

def arr_pairing_v2(arr_):
    return np.vstack((np.zeros_like(arr_), arr_)).T

def arr_pairing(arr_):
  return np.vstack((arr_[:-1], arr_[1:])).T

def get_split_len_v2(arr_, split_idx_pairs):
    return [np.sum((pair[0] <= arr_) & (arr_ < pair[1])) for pair in split_idx_pairs]

def get_split_len(arr_, split_idx_pairs):
  return np.insert(np.cumsum([np.sum((pair[0] <= arr_) & (arr_ < pair[1])) for pair in split_idx_pairs]), 0, 0)

def split_data(data_, split_len_pairs):
  return np.array([[a_data[pair[0]:pair[1]] for pair in pairs] for a_data, pairs in zip(data_, split_len_pairs)])

#   Todo - 사용하지 않는 ep ~ exit col, 없애는게 좋지않을까 => monthly idep_plot 으로 보류
def to_sample_data_v3(survey_data, len_df, test_ratio, k, data_cols):
  pr_numb, frq_numb, sr_numb, pr_list_numb, ep_numb, tp_numb, entry_idx_numb, exit_idx_numb = data_cols

  fold_len = int(len_df * (1 - test_ratio))
  split_idx_pairs = arr_pairing(np.arange(0, fold_len, int(fold_len / k) - 1))  # -1 for including last index
  split_len_pairs = [arr_pairing(get_split_len(a_data, split_idx_pairs)) for a_data in survey_data[:, exit_idx_numb]]
  split_obj = [split_data(survey_data[:, numb], split_len_pairs) for numb in [pr_list_numb, ep_numb, tp_numb, entry_idx_numb, exit_idx_numb]]

  return [set_sample_data(survey_data, split_obj, col_i, data_cols) for col_i in range(k)]


def get_slice_len(arr_, thresh):
  return np.sum(arr_ < thresh)


def slice_data(data_, slice_len_list):
  return [a_data[:slice_len] for a_data, slice_len in zip(data_, slice_len_list)]


def get_acc_pr(pr_list_):
  if len(pr_list_) == 0:
    return 1
  else:
    return np.cumprod(pr_list_)[-1]

def slice_data_v2(data_, slice_len_list):   # return is, oos
  return np.array([[a_data[:slice_len], a_data[slice_len:]] for a_data, slice_len in zip(data_, slice_len_list)])

def set_sample_data_v2(survey_data, sample_pr_list, s_col):
  target_pr_list = sample_pr_list[:, s_col]

  sample_pr = [get_acc_pr(pr_list_) for pr_list_ in target_pr_list]  # len = 0 을 포함하는 방향
  sample_frq = [len(pr_list_) for pr_list_ in target_pr_list]
  sample_sr = [sharpe_ratio(pr_list_) for pr_list_ in target_pr_list]

  return np.array([sample_pr, sample_frq, sample_sr]).T

def to_sample_data_v5(survey_data, len_df, test_ratio, k, public_cols, data_cols, mode='SPLIT'):
  pr_list_numb, exit_idx_numb = data_cols

  fold_len = int(len_df * (1 - test_ratio))
  fold_gap = int(fold_len / k) - 1  # -1 for including last index
  if mode == 'CUM':
    split_idx_pairs = arr_pairing_v2(np.append(np.arange(fold_gap, fold_len, fold_gap), len_df))
    split_len_pairs = [arr_pairing_v2(get_split_len_v2(a_data, split_idx_pairs)) for a_data in survey_data[:, exit_idx_numb]]
    # return split_len_pairs
  else:
    split_idx_pairs = arr_pairing(np.append(np.arange(0, fold_len, fold_gap), len_df))  # -1 for including last index
    split_len_pairs = [arr_pairing(get_split_len(a_data, split_idx_pairs)) for a_data in survey_data[:, exit_idx_numb]]
  split_pr_list = split_data(survey_data[:, pr_list_numb], split_len_pairs)

  return survey_data[:, public_cols], [set_sample_data_v2(survey_data, split_pr_list, col_i) for col_i in range(k + 1)]  # k + 1 for test_set

def to_sample_data_v4(survey_data, len_df, test_ratio, k, public_cols, data_cols, mode='SPLIT'):
  pr_list_numb, exit_idx_numb = data_cols

  fold_len = int(len_df * (1 - test_ratio))
  fold_gap = int(fold_len / k) - 1  # -1 for including last index
  if mode == 'CUM':
    split_idx_pairs = arr_pairing_v2(np.arange(fold_gap, fold_len, fold_gap))
    split_len_pairs = [arr_pairing_v2(get_split_len_v2(a_data, split_idx_pairs)) for a_data in survey_data[:, exit_idx_numb]]
  else:
    split_idx_pairs = arr_pairing(np.arange(0, fold_len, fold_gap))  # -1 for including last index
    split_len_pairs = [arr_pairing(get_split_len(a_data, split_idx_pairs)) for a_data in survey_data[:, exit_idx_numb]]
  split_pr_list = split_data(survey_data[:, pr_list_numb], split_len_pairs)

  return survey_data[:, public_cols], [set_sample_data_v2(survey_data, split_pr_list, col_i) for col_i in range(k)]

def set_sample_data(survey_data, sample_obj, s_col, data_cols):
  sample_pr_list, sample_ep, sample_tp, sample_entry_idx, sample_exit_idx = sample_obj
  target_pr_list = sample_pr_list[:, s_col]

  sample_pr = [get_acc_pr(pr_list_) for pr_list_ in target_pr_list]  # len = 0 을 포함하는 방향
  sample_frq = [len(pr_list_) for pr_list_ in target_pr_list]
  sample_sr = [sharpe_ratio(pr_list_) for pr_list_ in target_pr_list]

  sample_survey_data = survey_data.copy()
  sample_survey_data[:, data_cols] = \
  np.array([sample_pr, sample_frq, sample_sr, target_pr_list, sample_ep[:, s_col], sample_tp[:, s_col], sample_entry_idx[:, s_col], sample_exit_idx[:, s_col]]).T

  return sample_survey_data


def to_sample_data_v2(survey_data, sample_len, data_cols):
  pr_numb, frq_numb, sr_numb, pr_list_numb, ep_numb, tp_numb, entry_idx_numb, exit_idx_numb = data_cols
  slice_len_list = [get_slice_len(a_data, sample_len) for a_data in survey_data[:, exit_idx_numb]]
  sample_obj = [slice_data_v2(survey_data[:, numb], slice_len_list) for numb in [pr_list_numb, ep_numb, tp_numb, entry_idx_numb, exit_idx_numb]]

  return set_sample_data(survey_data, sample_obj, 0, data_cols), set_sample_data(survey_data, sample_obj, 1, data_cols)


def to_sample_data(survey_data, sample_len, pr_numb, frq_numb, sr_numb, pr_list_numb, ep_numb, tp_numb, entry_idx_numb, exit_idx_numb):
  slice_len_list = [get_slice_len(a_data, sample_len) for a_data in survey_data[:, exit_idx_numb]]
  sample_pr_list, sample_ep, sample_tp, sample_entry_idx, sample_exit_idx = [slice_data(survey_data[:, numb], slice_len_list) \
                                                                            for numb in [pr_list_numb, ep_numb, tp_numb, entry_idx_numb, exit_idx_numb]]
  sample_pr = [get_acc_pr(pr_list_) for pr_list_ in sample_pr_list]  # len = 0 을 포함하는 방향
  sample_frq = [len(pr_list_) for pr_list_ in sample_pr_list]
  sample_sr = [sharpe_ratio(pr_list_) for pr_list_ in sample_pr_list]

  sample_survey_data = survey_data.copy()
  sample_survey_data[:, [pr_numb, frq_numb, sr_numb, pr_list_numb, ep_numb, tp_numb, entry_idx_numb, exit_idx_numb]] = \
    np.array([sample_pr, sample_frq, sample_sr, sample_pr_list, sample_ep, sample_tp, sample_entry_idx, sample_exit_idx]).T

  return sample_survey_data


def posrank5(survey_key, survey_data_list, col_numb, comp_value, target_col, up=True, sort=True,
             log=True):  # no_inf covered by frq_col - more specific (sr inf gen. by frq=1)
    t_idx = np.ones_like(survey_data_list[0][:, 0], dtype=bool)
    # if non_test:
    #   survey_data_list = [survey_data_list[-2]]  # 누적이라, -2 만 적어도됨
    if type(survey_data_list) == list:
        survey_data_list = np.array(survey_data_list)
    survey_data_list = survey_data_list[target_col]
    if up:
        for survey_data in survey_data_list:
            t_idx *= survey_data[:, col_numb] >= comp_value
    else:
        for survey_data in survey_data_list:
            t_idx *= survey_data[:, col_numb] < comp_value
    if log:
        print(np.sum(t_idx))

    pos_key = survey_key[t_idx]
    pos_data_list = [survey_data[t_idx] for survey_data in survey_data_list]

    # if sort:  - unsupported
    #     return [pos_data[(-pos_data[:, col_numb]).argsort()] for pos_data in pos_data_list]
    # else:
    return pos_key, pos_data_list

def posrank4(survey_key, survey_data_list, col_numb, comp_value, non_test=False, up=True, sort=True, log=True):  # no_inf covered by frq_col - more specific (sr inf gen. by frq=1)
    t_idx = np.ones_like(survey_data_list[0][:, 0], dtype=bool)
    if non_test:
      survey_data_list = [survey_data_list[-2]]
    # if last_only:
    #   survey_data_list = [survey_data_list[-1]]
    if up:
      for survey_data in survey_data_list:
        t_idx *= survey_data[:, col_numb] >= comp_value
    else:
      for survey_data in survey_data_list:
        t_idx *= survey_data[:, col_numb] < comp_value
    if log:
        print(np.sum(t_idx))

    pos_key = survey_key[t_idx]
    pos_data_list = [survey_data[t_idx] for survey_data in survey_data_list]

    # if sort:  - unsupported
    #     return [pos_data[(-pos_data[:, col_numb]).argsort()] for pos_data in pos_data_list]
    # else:
    return pos_key, pos_data_list


def posrank3(survey_data_list, col_numb, comp_value, up=True, sort=True, log=True):  # no_inf covered by frq_col - more specific (sr inf gen. by frq=1)
    t_idx = np.ones_like(survey_data_list[0][:, 0], dtype=bool)
    if up:
      for survey_data in survey_data_list:
        t_idx *= survey_data[:, col_numb] >= comp_value
    else:
      for survey_data in survey_data_list:
        t_idx *= survey_data[:, col_numb] < comp_value
    if log:
        print(np.sum(t_idx))
    pos_data_list = [survey_data[t_idx] for survey_data in survey_data_list]

    if sort:
        return [pos_data[(-pos_data[:, col_numb]).argsort()] for pos_data in pos_data_list]
    else:
        return pos_data_list

def posrank2(survey_data_tup, col_numb, comp_value, up=True, sort=True, log=True):  # no_inf covered by frq_col - more specific (sr inf gen. by frq=1)
    is_, oos_ = survey_data_tup
    if up:
        t_idx = (is_[:, col_numb] >= comp_value) & (oos_[:, col_numb] >= comp_value)
    else:
        t_idx = (is_[:, col_numb] < comp_value) & (oos_[:, col_numb] < comp_value)
    if log:
        print(np.sum(t_idx))
    pos_data_tup = (is_[t_idx], oos_[t_idx])

    if sort:
        is_pos, oos_pos = pos_data_tup
        return is_pos[(-is_pos[:, col_numb]).argsort()], oos_pos[(-oos_pos[:, col_numb]).argsort()]  # 길이 위해 is/oos 정의
    else:
        return pos_data_tup


def posrank(survey_data, col_numb, comp_value, up=True, sort=True):  # no_inf covered by frq_col - more specific (sr inf gen. by frq=1)
  if up:
    pos_data = survey_data[survey_data[:, col_numb] >= comp_value]
  else:
    pos_data = survey_data[survey_data[:, col_numb] < comp_value]
  if sort:
    return pos_data[(-pos_data[:, col_numb]).argsort()]
  else:
    return pos_data


# def cpu_to_gpu(arr_):
#     return torch.from_numpy(arr_).cuda()


def gpu_to_cpu(arr_):
    return arr_.cpu().detach().numpy()


def liquidation(open_side, data_, obj_, lvrg, fee):  # much faster
    if type(obj_) == list:
        obj_ = list(zip(*obj_))

    if open_side == "SELL":
        return np.min([(ep_ / np.max(data_[int(iin):int(iout)]) - fee - 1) * lvrg + 1 for ep_, _, iin, iout in obj_ if iin != iout])
    else:
        return np.min([(np.min(data_[int(iin):int(iout)]) / ep_ - fee - 1) * lvrg + 1 for ep_, _, iin, iout in obj_ if iin != iout])


# def liquidation_torch(open_side, data_, obj_):  # deprecated - lvrg & fee
#   if open_side == "SELL":
#     return torch.min(torch.tensor([ep_ / torch.max(data_[iin:iout]) for ep_, _, iin, iout in zip(*obj_)]))
#   else:
#     return torch.min(torch.tensor([torch.min(data_[iin:iout]) / ep_ for ep_, _, iin, iout in zip(*obj_)]))


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

def frq_dev_plot_v2(gs, gs_idx, len_df, exit_idx, acc_pr, fontsize):
    plt.subplot(gs[gs_idx])
    frq_dev = np.zeros(len_df)
    if type(exit_idx) != int:
      exit_idx = exit_idx.astype(int)
    frq_dev[exit_idx] = 1
    plt.plot(frq_dev)

    title_msg = "periodic_pr\n acc_day : {:.4f}\n month : {:.4f}\n year : {:.4f}"  # \n rev_acc_day : {:.4f}\n month : {:.4f}\n year : {:.4f}"
    plt.title(title_msg.format(*get_period_pr(len_df, acc_pr), fontsize=fontsize))

    return gs_idx + 1


def to_total_pr(len_df, pr, exit_idx):
  total_pr = np.ones(len_df)
  if type(exit_idx) != int:
    exit_idx = exit_idx.astype(int)
  total_pr[exit_idx] = pr

  return total_pr

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

def plot_info_v3(gs, gs_idx, sample_len, pr, total_pr, cum_pr, liqd, leverage, title_position, fontsize):
  try:
    plt.subplot(gs[gs_idx])
    idep_res_obj = get_res_info_nb_v2(sample_len, pr, total_pr, cum_pr, liqd)
    plt.plot(cum_pr)
    plt.plot(idep_res_obj[-1], color='gold')
    if sample_len is not None:
      plt.axvline(sample_len, alpha=1., linestyle='--', color='#ffeb3b')
    title_str = "len_pr : {}\n dpf : {:.3f}\n wr : {:.3f}\n sr : {:.3f}\n acc_pr : {:.3f}\n sum_pr : {:.3f}\n" +\
              "min_pr : {:.3f}\n liqd : {:.3f}\n acc_mdd : -{:.3f}\n sum_mdd : -{:.3f}\n leverage {}"
    plt.title(title_str.format(*idep_res_obj[:-1], leverage), position=title_position, fontsize=fontsize)
  except Exception as e:
    print("error in plot_info :", e)

  return gs_idx + 1

def plot_info_v2(gs, gs_idx, len_df, pr, total_pr, cum_pr, liqd, leverage, vline, title_position, fontsize):
  try:
    plt.subplot(gs[gs_idx])
    idep_res_obj = get_res_info_nb_v2(len_df, pr, total_pr, cum_pr, liqd)
    plt.plot(cum_pr)
    plt.plot(idep_res_obj[-1], color='gold')
    if vline is not None:
      plt.axvline(vline, alpha=1., linestyle='--', color='#ffeb3b')
    title_str = "len_pr : {}\n dpf : {:.3f}\n wr : {:.3f}\n sr : {:.3f}\n acc_pr : {:.3f}\n sum_pr : {:.3f}\n" +\
              "min_pr : {:.3f}\n liqd : {:.3f}\n acc_mdd : -{:.3f}\n sum_mdd : -{:.3f}\n leverage {}"
    plt.title(title_str.format(*idep_res_obj[:-1], leverage), position=title_position, fontsize=fontsize)
  except Exception as e:
    print("error in plot_info :", e)

  return gs_idx + 1


def plot_info(gs, gs_idx, len_df, pr, cum_pr, liqd, config, vline, title_position, fontsize):
  try:
    plt.subplot(gs[gs_idx])
    idep_res_obj = get_res_info_nb(len_df, pr, cum_pr, liqd)
    plt.plot(cum_pr)
    plt.plot(idep_res_obj[-1], color='gold')
    if vline is not None:
      plt.axvline(vline, alpha=1., linestyle='--', color='#ffeb3b')
    title_str = "len_pr : {}\n dpf : {:.3f}\n wr : {:.3f}\n sr : {:.3f}\n acc_pr : {:.3f}\n sum_pr : {:.3f}\n" +\
              "min_pr : {:.3f}\n liqd : {:.3f}\n acc_mdd : -{:.3f}\n sum_mdd : -{:.3f}\n leverage {}"
    plt.title(title_str.format(*idep_res_obj[:-1], config.lvrg_set.leverage), position=title_position, fontsize=fontsize)
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


# @jit    # slower
def sharpe_ratio_nb(pr, risk_free_rate=0.0, multiply_frq=False):
    pr_pct = pr - 1
    mean_pr_ = np.mean(pr_pct)
    sr_ = (mean_pr_ - risk_free_rate) / np.std(pr_pct)

    if multiply_frq:
        sr_ *= len(pr_pct) ** (1 / 2)

    return sr_


def mdd(pr):
  rollmax_pr = np.maximum.accumulate(pr)
  return np.max((rollmax_pr - pr) / rollmax_pr)


# @jit(parallel=True)  # slower
# def mdd_nb(pr):
#   rollmax_pr = np.maximum.accumulate(pr)
#   return np.max((rollmax_pr - pr) / rollmax_pr)


#     필요한 col index 정리    #
def get_col_idxs(df, cols):
  return [df.columns.get_loc(col) for col in cols]


def adj_mr(obj, mr_res_idx):
  return [res_[mr_res_idx] for res_ in obj]


# @jit  # warning - jit cannot be used by outer_scope's idx
def adj_mr_nb(obj, mr_res_idx):
    # result = np.empty(len(obj))
    result = []
    for i, ob_ in enumerate(obj):
        # result[i] = ob_[mr_res_idx]
        result.append(ob_[mr_res_idx])

    return result


def get_exc_info(res_df, config, short_ep, short_tp, long_ep, long_tp, np_timeidx, mode="banking"):
    strat_version = config.strat_version

    #   open & close init   #
    #        0. open -> entry(executed) | close -> exit(executed)
    #           a. => entry & exit 은 trader 단에서 설계 불가
    #      Todo      #
    #        1. open & close -> enlist_tr 을 통해 진행 - 이거 하려면, importlib 해야할 듯
    #        2. ep_tp_pairing_nb 는 ep_loc_entry_idx 기준으로 진행 (이때는 limit / market 두가지 허용)
    np_zeros = np.zeros(len(res_df)).astype(np.int8).copy()
    if mode != "CLOSE":
        res_df['short_open_{}'.format(strat_version)], res_df['long_open_{}'.format(strat_version)] = open_nb(np_zeros, np_zeros,
                                                                                                                 config.loc_set.point.tf_entry,
                                                                                                                 np_timeidx)
    if mode != "OPEN":
        res_df['short_close_{}'.format(strat_version)], res_df['long_close_{}'.format(strat_version)] = close_nb(np_zeros, np_zeros,
                                                                                                              config.loc_set.point.tf_entry, np_timeidx)

    # print(res_df['short_open_{}'.format(strat_version)].tail())
    # print(res_df['short_close_{}'.format(strat_version)].tail())

    s_open = res_df['short_open_{}'.format(strat_version)].to_numpy()
    s_close = res_df['short_close_{}'.format(strat_version)].to_numpy()
    l_open = res_df['long_open_{}'.format(strat_version)].to_numpy()
    l_close = res_df['long_close_{}'.format(strat_version)].to_numpy()

    short_slice_entry_idx, short_slice_exit_idx, long_slice_entry_idx, long_slice_exit_idx = ep_tp_pairing_nb(s_open, s_close, l_open, l_close)

    #   ep indi. length 는 res_df 와 동일해야함   #
    short_exc_ep = short_ep[short_slice_entry_idx.reshape(-1, )].to_numpy()
    long_exc_ep = long_ep[long_slice_entry_idx.reshape(-1, )].to_numpy()

    short_exc_tp = short_tp[short_slice_exit_idx.reshape(-1, )].to_numpy()
    long_exc_tp = long_tp[long_slice_exit_idx.reshape(-1, )].to_numpy()

    # short_obj = remove_nan(short_exc_ep, short_exc_tp, short_slice_entry_idx.reshape(-1, ), short_slice_exit_idx.reshape(-1, ))
    # long_obj = remove_nan(long_exc_ep, long_exc_tp, short_slice_entry_idx.reshape(-1, ), short_slice_exit_idx.reshape(-1, ))
    #       일일이 value define 해도 numba 라서 빠름     #
    short_obj = remove_nan_nb(short_exc_ep, short_exc_tp, short_slice_entry_idx.reshape(-1, ), short_slice_exit_idx.reshape(-1, ))
    long_obj = remove_nan_nb(long_exc_ep, long_exc_tp, short_slice_entry_idx.reshape(-1, ), short_slice_exit_idx.reshape(-1, ))

    return res_df, short_obj, long_obj


# @njit
def open_nb(short_open_np, long_open_np, tf_entry, np_timeidx):
  # ------------------ short = -1 ------------------ #
  # ------ point ------ #
  short_open_np = np.where((np_timeidx % tf_entry == 0), short_open_np - 1, short_open_np)
  # ------ point_dur ------ #

  # ------------------ long = 1 ------------------ #
  # ------ point ------ #
  #       Todo - orderside should be decided by dur. const_ (time point)  #
  long_open_np = np.where((np_timeidx % tf_entry == 0), long_open_np + 1, long_open_np)
  # ------ point_dur ------ #

  return short_open_np, long_open_np


# @njit
def close_nb(short_close_np, long_close_np, tf_entry, np_timeidx):
  # ------------------ short = -1 ------------------ #
  # ------ point ------ #
  short_close_np = np.where((np_timeidx % tf_entry == tf_entry - 1), short_close_np - 1, short_close_np)
  # ------ point_dur ------ #

  # ------------------ long = 1 ------------------ #
  # ------ point ------ #
  #       Todo - orderside should be decided by dur. const_ (time point)  #
  long_close_np = np.where((np_timeidx % tf_entry == tf_entry - 1), long_close_np + 1, long_close_np)
  # ------ point_dur ------ #

  return short_close_np, long_close_np


# @njit
def ep_tp_pairing_nb(s_entry, s_exit, l_entry, l_exit):  # 현재 규칙적인 ep & tp gap 에 의존하는 logic 임
  #  get entry & exit idx - strat_version 때문에, 함부로 var_name 못정하고 있는것 -> to_numpy() 사용하는 결과
  #  exit_idx's first idx should be greater than the entry   #
  short_entry_idx = np.argwhere(s_entry == -1)  #[10:]
  short_exit_idx = np.argwhere(s_exit == -1)
  short_mod_exit_idx = short_exit_idx[np.sum(short_entry_idx[0] > short_exit_idx):]

  long_entry_idx = np.argwhere(l_entry == 1) #[10:]
  long_exit_idx = np.argwhere(l_exit == 1)
  long_mod_exit_idx = long_exit_idx[np.sum(long_entry_idx[0] > long_exit_idx):]

  #   slicing for equal length - updown pairing logic    #
  short_min_len = min(len(short_entry_idx), len(short_mod_exit_idx))
  long_min_len = min(len(long_entry_idx), len(long_mod_exit_idx))

  return short_entry_idx[:short_min_len], short_mod_exit_idx[:short_min_len], long_entry_idx[:long_min_len], long_mod_exit_idx[:long_min_len]


def remove_nan(ep, tp, ep_idx, tp_idx):
    # invalid_idx = np.unique(np.concatenate((np.argwhere(np.isnan(res_)) for res_ in [ep, tp]), axis=None)) # hstack 으로 통일하려면 ndim = 1
    invalid_idx = np.unique(np.concatenate([np.argwhere(np.isnan(res_)) for res_ in [ep, tp]], axis=None))  # hstack 으로 통일하려면 ndim = 1
    valid_idx = np.delete(np.arange(len(ep)), invalid_idx)

    return [res_[valid_idx] for res_ in [ep, tp, ep_idx, tp_idx]]


# @jit
def remove_nan_nb(ep, tp, ep_idx, tp_idx):  # warning - numba 내부에서 [] oneline comprehension 을 허용하지 않는 것으로 보임
  ep_invalid_idx = np.argwhere(np.isnan(ep))  #.reshape(-1,)
  tp_invalid_idx = np.argwhere(np.isnan(tp))  #.reshape(-1,)
  invalid_idx = np.unique(np.concatenate((ep_invalid_idx, tp_invalid_idx), axis=None)) # hstack 으로 통일하려면 ndim = 1
  valid_idx = np.delete(np.arange(len(ep)), invalid_idx)
  # print(valid_idx)

  valid_ep = ep[valid_idx]
  valid_tp = tp[valid_idx]
  valid_ep_idx = ep_idx[valid_idx]
  valid_tp_idx = tp_idx[valid_idx]

  return valid_ep, valid_tp, valid_ep_idx, valid_tp_idx


def get_pr_v3(open_side, h, l, obj, tpout, lvrg, fee, p_ranges, p_qty, inversion=False):  # --> 여기서 사용하는 ex_p = ex_p

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
            pr = ((en_ps / ex_ps - fees - 1) * lvrgs * p_qty).sum(axis=1) + 1
        else:
            pr = ((ex_ps / en_ps - fees - 1) * lvrgs * p_qty).sum(axis=1) + 1
    else:
        if not inversion:
            pr = ((ex_ps / en_ps - fees - 1) * lvrgs * p_qty).sum(axis=1) + 1
        else:
            pr = ((en_ps / ex_ps - fees - 1) * lvrgs * p_qty).sum(axis=1) + 1

    return pr.reshape(-1, 1), liqd

def get_pr_v2(open_side, h, l, obj, tpout, lvrg, fee, p_ranges, p_qty, inversion=False):  # --> 여기서 사용하는 ex_p = ex_p

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
            pr = ((en_ps / ex_ps - fees - 1) * lvrgs * p_qty).sum(axis=1) + 1
        else:
            pr = ((ex_ps / en_ps - fees - 1) * lvrgs * p_qty).sum(axis=1) + 1
    else:
        if not inversion:
            pr = ((ex_ps / en_ps - fees - 1) * lvrgs * p_qty).sum(axis=1) + 1
        else:
            pr = ((en_ps / ex_ps - fees - 1) * lvrgs * p_qty).sum(axis=1) + 1

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
    a_day = len_df / 1440
    a_month = a_day / 30
    a_year = a_day / 365

    return [pr_ ** (1 / period) for period in [a_day, a_month, a_year]]


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
