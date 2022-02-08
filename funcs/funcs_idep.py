import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from numba import jit, njit
import torch


def get_slice_len(arr_, thresh):
  return np.sum(arr_ < thresh)


def slice_data(data_, slice_len_list):
  return [a_data[:slice_len] for a_data, slice_len in zip(data_, slice_len_list)]


def get_acc_pr(pr_list_):
  if len(pr_list_) == 0:
    return 1
  else:
    return np.cumprod(pr_list_)[-1]


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


def posrank(survey_data, col_numb, comp_value, up=True, sort=True):  # no_inf covered by frq_col - more specific (sr inf gen. by frq=1)
  if up:
    pos_data = survey_data[survey_data[:, col_numb] >= comp_value]
  else:
    pos_data = survey_data[survey_data[:, col_numb] < comp_value]
  if sort:
    return pos_data[(-pos_data[:, col_numb]).argsort()]
  else:
    return pos_data


def cpu_to_gpu(arr_):
    return torch.from_numpy(arr_).cuda()


def gpu_to_cpu(arr_):
    return arr_.cpu().detach().numpy()


def liquidation(open_side, data_, obj_, lvrg, fee):  # much faster
    if type(obj_) == list:
        obj_ = zip(*obj_)

    if open_side == "SELL":
        return np.min([(ep_ / np.max(data_[iin:iout]) - fee - 1) * lvrg + 1 for ep_, _, iin, iout in obj_])
    else:
        return np.min([(np.min(data_[iin:iout]) / ep_ - fee - 1) * lvrg + 1 for ep_, _, iin, iout in obj_])


def liquidation_torch(open_side, data_, obj_):  # deprecated - lvrg & fee
  if open_side == "SELL":
    return torch.min(torch.tensor([ep_ / torch.max(data_[iin:iout]) for ep_, _, iin, iout in zip(*obj_)]))
  else:
    return torch.min(torch.tensor([torch.min(data_[iin:iout]) / ep_ for ep_, _, iin, iout in zip(*obj_)]))


def frq_dev_plot_v2(gs, gs_idx, len_df, entry_idx, acc_pr, fontsize):
    plt.subplot(gs[gs_idx])
    frq_dev = np.zeros(len_df)
    frq_dev[entry_idx] = 1
    plt.plot(frq_dev)

    title_msg = "periodic_pr\n acc_day : {:.4f}\n month : {:.4f}\n year : {:.4f}"  # \n rev_acc_day : {:.4f}\n month : {:.4f}\n year : {:.4f}"
    plt.title(title_msg.format(*get_period_pr(len_df, acc_pr), fontsize=fontsize))

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


@jit  # almost equal
def get_res_info_nb(len_df, np_pr, total_pr, liqd):
    wr = len(np_pr[np_pr > 1]) / len(np_pr[np_pr != 1])
    sr = sharpe_ratio(np_pr)
    sum_pr = get_sum_pr_nb(np_pr)
    min_pr = np.min(np_pr)

    len_pr = len(np_pr)
    assert len_pr != 0
    dpf = (len_df / 1440) / len_pr  # devision zero warning

    acc_mdd = mdd(total_pr)
    sum_mdd = mdd(sum_pr)

    return len_pr, dpf, wr, sr, total_pr[-1], sum_pr[-1], min_pr, liqd, acc_mdd, sum_mdd, sum_pr


@jit
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


@jit    # slower
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


@jit(parallel=True)  # slower
def mdd_nb(pr):
  rollmax_pr = np.maximum.accumulate(pr)
  return np.max((rollmax_pr - pr) / rollmax_pr)


#     필요한 col index 정리    #
def get_col_idxs(df, cols):
  return [df.columns.get_loc(col) for col in cols]


def adj_mr(obj, mr_res_idx):
  return [res_[mr_res_idx] for res_ in obj]


@jit  # warning - jit cannot be used by outer_scope's idx
def adj_mr_nb(obj, mr_res_idx):
    # result = np.empty(len(obj))
    result = []
    for i, ob_ in enumerate(obj):
        # result[i] = ob_[mr_res_idx]
        result.append(ob_[mr_res_idx])

    return result


def get_exc_info(res_df, config, short_ep, short_tp, long_ep, long_tp, np_timeidx, mode="IDE"):
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


@njit
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


@njit
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


@njit
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


@jit
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


def get_pr(open_side, ep, tp, lvrg, fee):
  assert len(tp) == len(ep)
  if open_side == "SELL":
    pr = (ep / tp - fee - 1) * lvrg + 1
  else:
    pr = (tp / ep - fee - 1) * lvrg + 1

  return pr


@jit
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
        plt.title(title_msg.format(*get_period_pr(res_df, pr_[-1]), *get_period_pr(res_df, rev_pr_[-1])), fontsize=fontsize_)

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
