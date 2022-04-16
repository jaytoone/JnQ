import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import mpl_finance as mf
from funcs.funcs_idep import get_col_idxs
# from numba import jit


def strcol_tonumb(res_df, col_list):
  step_col_arr = np.array(col_list)
  if len(col_list) != 0:
    step_col_arr[:, 0] = [get_col_idxs(res_df, col_) for col_ in step_col_arr[:, 0]]   # str_col to number

  return step_col_arr

def nonstep_col_plot(np_col, alpha=1, color='#ffffff', linewidth=2):
  try:
    plt.plot(np.arange(len(np_col)), np_col, alpha=alpha, color=color, linewidth=linewidth)
  except Exception as e:
    print("error in nonstep_col_plot :", e)

def step_col_plot(np_col, alpha=1, color='#ffffff', linewidth=2):
  try:
    plt.step(np.arange(len(np_col)), np_col, alpha=alpha, color=color, linewidth=linewidth)
  except Exception as e:
    print("error in step_col_plot :", e)

def stepmark_col_plot(np_col, alpha=1, color='#ffffff', markersize=2):
  try:
    plt.step(np.arange(len(np_col)), np_col, 'c*', alpha=alpha, color=color, markersize=markersize)
  except Exception as e:
    print("error in stepmark_col_plot :", e)

def fillbw_col_plot(np_col, alpha=1, color='#ffffff', linewidth=2):
  try:
    plt.step(np.arange(len(np_col)), np_col, alpha=alpha, color=color, linewidth=linewidth)
  except Exception as e:
    print("error in fillbw_col_plot :", e)

def sort_bypr_v3(pr, obj, arr_list, descending=True):
  if descending:
    sort_idx = pr.ravel().argsort()[::-1]
  else:
    sort_idx = pr.ravel().argsort()

  return pr[sort_idx], [ob[sort_idx] for ob in obj], [arr_[sort_idx] for arr_ in arr_list]

def sort_bypr_v2(pr, obj, lvrg_arr, fee_arr, tpout_arr, descending=True):
  if descending:
    sort_idx = pr.ravel().argsort()[::-1]
  else:
    sort_idx = pr.ravel().argsort()

  return pr[sort_idx], [ob[sort_idx] for ob in obj], lvrg_arr[sort_idx], fee_arr[sort_idx], tpout_arr[sort_idx]

def sort_bypr(pr, obj, descending=True):
  if descending:
    sort_idx = pr.argsort()[::-1]
  else:
    sort_idx = pr.argsort()

  return pr[sort_idx], [ob[sort_idx] for ob in obj]


def eptp_hvline_v5(config, en_p, ex_p, entry_idx, exit_idx, open_idx, point1_idx, tp_line, out_line, en_tp1, en_out0,
                   front_plot, iin, iout, x_max, x_margin_mult, y_margin_mult, a_data, **col_idx_dict):
  # ------ get vertical ticks ------ #
  entry_tick = int(entry_idx - iin)
  exit_tick = entry_tick + int(exit_idx - entry_idx)
  open_tick = entry_tick - int(entry_idx - open_idx)
  point1_tick = open_tick - int(open_idx - point1_idx)
  bias_info_tick = entry_tick + config.tr_set.bias_info_tick

  if front_plot == 1:
    x_max = open_tick + 20
  elif front_plot == 2:
    x_max = entry_tick + 20
  elif front_plot == 3:
    x_max = exit_tick + 20
  elif front_plot == 4:
    x_max = bias_info_tick + 20

  if (iout - iin) > x_max:
    x_margin = x_max * x_margin_mult
    plt.xlim(0 - x_margin, x_max + x_margin)
  x0, x1 = plt.gca().get_xlim()

  # ------------ hlines ------------ #
  # ------ entry & exit ------ #
  en_xmin = entry_tick / x1
  ex_xmin = exit_tick / x1
  plt.axhline(en_p, x0, en_xmin, linewidth=2, linestyle='--', alpha=1, color='lime')  # en_p line axhline
  plt.text(x0, en_p, 'en_p :\n {:.3f} \n epg {}'.format(en_p, config.tr_set.ep_gap), ha='right', va='center', fontweight='bold',
           fontsize=15)  # en_p line label
  plt.axhline(ex_p, ex_xmin, 1, linewidth=2, linestyle='--', alpha=1, color='lime')  # ex_p line axhline (signal 도 포괄함, 존재 의미)
  plt.text(x1, ex_p, 'ex_p :\n {}'.format(ex_p), ha='left', va='center', fontweight='bold', fontsize=15)  # ex_p line label

  # ------ tpout_line ------ #
  plt.axhline(tp_line, 0.05, 1, linewidth=4, linestyle='-', alpha=1, color='#00ff00')  # ep 와 gap 비교 용이하기 위해 ex_xmin -> 0.1 사용
  plt.text(x0, tp_line, 'tpg {}'.format(config.tr_set.tp_gap), ha='right', va='center', fontweight='bold', fontsize=15, color='#00ff00')
  plt.axhline(out_line, 0.05, 1, linewidth=4, linestyle='-', alpha=1, color='#ff0000')
  plt.text(x0, out_line, 'outg {}'.format(config.tr_set.out_gap), ha='right', va='center', fontweight='bold', fontsize=15, color='#ff0000')

  # ------ bias_line ------ #
  text_x_pos = (x0 + x1) * 0.1
  plt.axhline(en_tp1, 0.2, 1, linewidth=4, linestyle='-', alpha=1, color='#ffffff')
  plt.text(text_x_pos, en_tp1, ' en_tp1', ha='right', va='bottom', fontweight='bold', fontsize=15)
  plt.axhline(en_out0, 0.2, 1, linewidth=4, linestyle='-', alpha=1, color='#ffffff')
  plt.text(text_x_pos, en_out0, ' en_out0', ha='right', va='bottom', fontweight='bold', fontsize=15)

  # ------ ylim ------ #
  if front_plot:
    y_lim_data = a_data[:x_max + 1, col_idx_dict['ylim_col_idxs']]  # +1 for including open_tick
  else:
    y_lim_data = a_data[:, col_idx_dict['ylim_col_idxs']]

  y_min = y_lim_data.min()
  y_max = y_lim_data.max()
  y_margin = (y_max - y_min) * y_margin_mult
  plt.ylim(y_min - y_margin, y_max + y_margin)

  # ------------ vline (open_tick, entry_tick, exit_tick) ------------ #
  y0, y1 = plt.gca().get_ylim()
  l_data = a_data[:exit_tick + 1, col_idx_dict['ohlc_col_idxs'][2]]  # +1 for including exit_tick
  point1_ymax, open_ymax, en_ymax, ex_ymax = [(l_data[tick_] - y0) / (y1 - y0) - .01 for tick_ in
                                              [point1_tick, open_tick, entry_tick, exit_tick]]  # -.05 for margin
  plt.axvline(point1_tick, 0, point1_ymax, alpha=1, linewidth=2, linestyle='--', color='#ff7722')
  plt.axvline(open_tick, 0, open_ymax, alpha=1, linewidth=2, linestyle='--', color='#ffeb3b')  # 추후, tick 별 세부 정의가 달라질 수 있음을 고려해 multi_line 작성 유지
  plt.axvline(entry_tick, 0, en_ymax, alpha=1, linewidth=2, linestyle='--', color='#ffeb3b')
  plt.axvline(exit_tick, 0, ex_ymax, alpha=1, linewidth=2, linestyle='--', color='#ffeb3b')
  plt.axvline(bias_info_tick, alpha=1, linewidth=2, linestyle='-', color='#ffeb3b')

  return

def eptp_hvline_v4(config, en_p, ex_p, entry_idx, exit_idx, open_idx, point1_idx, tp_line, out_line, bias_info, bias_thresh,
                   front_plot, iin, iout, x_max, x_margin_mult, y_margin_mult, a_data, **col_idx_dict):
  # ------ get vertical ticks ------ #
  entry_tick = int(entry_idx - iin)
  exit_tick = entry_tick + int(exit_idx - entry_idx)
  open_tick = entry_tick - int(entry_idx - open_idx)
  point1_tick = open_tick - int(open_idx - point1_idx)

  if front_plot == 1:
    x_max = open_tick
  elif front_plot == 2:
    x_max = entry_tick + 20
  elif front_plot == 3:
    x_max = exit_tick + 20
  elif front_plot == 4:
    x_max = entry_tick + config.tr_set.bias_info_tick

  if (iout - iin) > x_max:
    x_margin = x_max * x_margin_mult
    plt.xlim(0 - x_margin, x_max + x_margin)
  x0, x1 = plt.gca().get_xlim()
  # ------------ hlines ------------ #
  # ------ entry & exit ------ #
  en_xmin = entry_tick / x1
  ex_xmin = exit_tick / x1
  plt.axhline(en_p, x0, en_xmin, linewidth=2, linestyle='--', alpha=1, color='lime')  # en_p line axhline
  plt.text(x0, en_p, 'en_p :\n {:.3f} \n epg {}'.format(en_p, config.tr_set.ep_gap), ha='right', va='center', fontweight='bold',
           fontsize=15)  # en_p line label
  plt.axhline(ex_p, ex_xmin, 1, linewidth=2, linestyle='--', alpha=1, color='lime')  # ex_p line axhline (signal 도 포괄함, 존재 의미)
  plt.text(x1, ex_p, 'ex_p :\n {}'.format(ex_p), ha='left', va='center', fontweight='bold', fontsize=15)  # ex_p line label

  # ------ tpout_line ------ #
  plt.axhline(tp_line, 0.1, 1, linewidth=4, linestyle='-', alpha=1, color='#00ff00')  # ep 와 gap 비교 용이하기 위해 ex_xmin -> 0.1 사용
  plt.text(x0, tp_line, 'tpg {}'.format(config.tr_set.tp_gap), ha='right', va='center', fontweight='bold', fontsize=15, color='#00ff00')
  plt.axhline(out_line, 0.1, 1, linewidth=4, linestyle='-', alpha=1, color='#ff0000')
  plt.text(x0, out_line, 'outg {}'.format(config.tr_set.out_gap), ha='right', va='center', fontweight='bold', fontsize=15, color='#ff0000')

  # ------ bias_line ------ #
  plt.axhline(bias_info, 0.1, 1, linewidth=4, linestyle='-', alpha=1, color='#a231d4')
  plt.text(x0, bias_info, ' bias_info', ha='left', va='center', fontweight='bold')
  plt.axhline(bias_thresh, 0.1, 1, linewidth=4, linestyle='-', alpha=1, color='#ff8400')
  plt.text(x0, bias_thresh, ' bias_thresh', ha='left', va='center', fontweight='bold')

  # ------ ylim ------ #
  if front_plot:
    y_lim_data = a_data[:x_max + 1, col_idx_dict['ylim_col_idxs']]  # +1 for including open_tick
  else:
    y_lim_data = a_data[:, col_idx_dict['ylim_col_idxs']]

  y_min = y_lim_data.min()
  y_max = y_lim_data.max()
  y_margin = (y_max - y_min) * y_margin_mult
  plt.ylim(y_min - y_margin, y_max + y_margin)

  # ------------ vline (open_tick, entry_tick, exit_tick) ------------ #
  y0, y1 = plt.gca().get_ylim()
  l_data = a_data[:exit_tick + 1, col_idx_dict['ohlc_col_idxs'][2]]  # +1 for including exit_tick
  point1_ymax, open_ymax, en_ymax, ex_ymax = [(l_data[tick_] - y0) / (y1 - y0) - .01 for tick_ in
                                              [point1_tick, open_tick, entry_tick, exit_tick]]  # -.05 for margin
  plt.axvline(point1_tick, 0, point1_ymax, alpha=1, linewidth=2, linestyle='--', color='#ff7722')
  plt.axvline(open_tick, 0, open_ymax, alpha=1, linewidth=2, linestyle='--', color='#ffeb3b')  # 추후, tick 별 세부 정의가 달라질 수 있음을 고려해 multi_line 작성 유지
  plt.axvline(entry_tick, 0, en_ymax, alpha=1, linewidth=2, linestyle='--', color='#ffeb3b')
  plt.axvline(exit_tick, 0, ex_ymax, alpha=1, linewidth=2, linestyle='--', color='#ffeb3b')

  return

def eptp_hvline_v3(config, en_p, ex_p, entry_idx, exit_idx, open_idx, tp_line, out_line, bias_info, bias_thresh,
                   front_plot, iin, iout, x_max, x_margin_mult, y_margin_mult, a_data, **col_idx_dict):
  # ------ get vertical ticks ------ #
  entry_tick = int(entry_idx - iin)
  open_tick = entry_tick - int(entry_idx - open_idx)
  exit_tick = entry_tick + int(exit_idx - entry_idx)

  if front_plot == 1:
    x_max = open_tick
  elif front_plot == 2:
    x_max = entry_tick + 20  # temp
  if (iout - iin) > x_max:
    x_margin = x_max * x_margin_mult
    plt.xlim(0 - x_margin, x_max + x_margin)
  x0, x1 = plt.gca().get_xlim()
  # ------------ hlines ------------ #
  # ------ entry & exit ------ #
  en_xmin = entry_tick / x1
  ex_xmin = exit_tick / x1
  plt.axhline(en_p, x0, en_xmin, linewidth=2, linestyle='--', alpha=1, color='lime')  # en_p line axhline
  plt.text(x0, en_p, 'en_p :\n {:.3f} \n epg {}'.format(en_p, config.tr_set.ep_gap), ha='right', va='center', fontweight='bold',
           fontsize=15)  # en_p line label
  plt.axhline(ex_p, ex_xmin, 1, linewidth=2, linestyle='--', alpha=1, color='lime')  # ex_p line axhline (signal 도 포괄함, 존재 의미)
  plt.text(x1, ex_p, 'ex_p :\n {}'.format(ex_p), ha='left', va='center', fontweight='bold')  # ex_p line label

  # ------ tpout_line ------ #
  plt.axhline(tp_line, 0.1, 1, linewidth=4, linestyle='-', alpha=1, color='#00ff00')  # ep 와 gap 비교 용이하기 위해 ex_xmin -> 0.1 사용
  plt.text(x0, tp_line, 'tpg {}'.format(config.tr_set.tp_gap), ha='right', va='center', fontweight='bold', fontsize=15)
  plt.axhline(out_line, 0.1, 1, linewidth=4, linestyle='-', alpha=1, color='#ff0000')
  plt.text(x0, out_line, 'outg {}'.format(config.tr_set.out_gap), ha='right', va='center', fontweight='bold', fontsize=15)

  # ------ bias_line ------ #
  plt.axhline(bias_info, 0.1, 1, linewidth=4, linestyle='-', alpha=1, color='dodgerblue')
  plt.text(x0, bias_info, ' bias_info', ha='left', va='center', fontweight='bold')
  plt.axhline(bias_thresh, 0.1, 1, linewidth=4, linestyle='-', alpha=1, color='#ff8400')
  plt.text(x0, bias_thresh, ' bias_thresh', ha='left', va='center', fontweight='bold')

  # ------ ylim ------ #
  if front_plot:
    y_lim_data = a_data[:x_max + 1, col_idx_dict['ylim_col_idxs']]  # +1 for including open_tick
  else:
    y_lim_data = a_data[:, col_idx_dict['ylim_col_idxs']]

  y_min = y_lim_data.min()
  y_max = y_lim_data.max()
  y_margin = (y_max - y_min) * y_margin_mult
  plt.ylim(y_min - y_margin, y_max + y_margin)

  # ------------ vline (open_tick, entry_tick, exit_tick) ------------ #
  y0, y1 = plt.gca().get_ylim()
  l_data = a_data[:exit_tick + 1, col_idx_dict['ohlc_col_idxs'][2]]  # +1 for including exit_tick
  open_ymax, en_ymax, ex_ymax = [(l_data[tick_] - y0) / (y1 - y0) - .01 for tick_ in [open_tick, entry_tick, exit_tick]]  # -.05 for margin
  plt.axvline(open_tick, 0, open_ymax, alpha=1, linewidth=2, linestyle='--', color='#ffeb3b')  # 추후, tick 별 세부 정의가 달라질 수 있음을 고려해 multi_line 작성 유지
  plt.axvline(entry_tick, 0, en_ymax, alpha=1, linewidth=2, linestyle='--', color='#ffeb3b')
  plt.axvline(exit_tick, 0, ex_ymax, alpha=1, linewidth=2, linestyle='--', color='#ffeb3b')

  return

def eptp_hvline_v2(config, ep, tp, entry_idx, exit_idx, open_idx, tp_line, out_line, front_plot, iin, iout, x_max, x_margin_mult, y_margin_mult,
                   a_data, **col_idx_dict):
  # ------ vline entry & exit ------ #
  ep_tick = int(entry_idx - iin)
  open_tick = ep_tick - int(entry_idx - open_idx)
  tp_tick = ep_tick + int(exit_idx - entry_idx)

  if front_plot == 1:
    x_max = open_tick
  elif front_plot == 2:
    x_max = ep_tick
  if (iout - iin) > x_max:
    x_margin = x_max * x_margin_mult
    plt.xlim(0 - x_margin, x_max + x_margin)
  x0, x1 = plt.gca().get_xlim()
  # ------ hline entry & exit ------ #
  ep_xmin = ep_tick / x1
  tp_xmin = tp_tick / x1
  plt.axhline(ep, linestyle='--', xmin=ep_xmin, xmax=1, alpha=1, color='lime')  # ep line axhline
  plt.text(x1, ep, ' ep :\n {}'.format(ep), ha='left', va='center', fontweight='bold')  # ep line label
  plt.axhline(tp, linestyle='--', xmin=tp_xmin, xmax=1, alpha=1, color='lime')  # tp line axhline
  plt.text(x1, tp, ' tp :\n {}'.format(tp), ha='left', va='center', fontweight='bold')  # tp line label

  # ------ tpout_line ------ #
  plt.axhline(tp_line, linewidth=2, linestyle='-', alpha=1, color='#00ff00')
  plt.text(x0, tp_line, ' %s' % config.tr_set.tp_gap, ha='left', va='center', fontweight='bold')
  plt.axhline(out_line, linewidth=2, linestyle='-', alpha=1, color='#ff0000')
  plt.text(x0, out_line, ' %s' % config.tr_set.out_gap, ha='left', va='center', fontweight='bold')

  # ------ ylim ------ #
  if front_plot:
    y_lim_data = a_data[:x_max + 1, col_idx_dict['ylim_col_idxs']]  # +1 for including open_tick
  else:
    y_lim_data = a_data[:, col_idx_dict['ylim_col_idxs']]
  y_min = y_lim_data.min()
  y_max = y_lim_data.max()
  y_margin = (y_max - y_min) * y_margin_mult
  plt.ylim(y_min - y_margin, y_max + y_margin)

  # ------ open ep tp vline ------ #
  y0, y1 = plt.gca().get_ylim()
  l_data = a_data[:tp_tick + 1, col_idx_dict['ohlc_col_idxs'][2]]  # +1 for including tp_tick
  open_ymax, ep_ymax, tp_ymax = [(l_data[tick_] - y0) / (y1 - y0) - .05 for tick_ in [open_tick, ep_tick, tp_tick]]  # -.05 for margin
  plt.axvline(open_tick, 0, open_ymax, alpha=1, linestyle='--', color='#ffeb3b')  # 추후, tick 별 세부 정의가 달라질 수 있음을 고려해 multi_line 작성 유지
  plt.axvline(ep_tick, 0, ep_ymax, alpha=1, linestyle='--', color='#ffeb3b')
  plt.axvline(tp_tick, 0, tp_ymax, alpha=1, linestyle='--', color='#ffeb3b')

  return

def eptp_hvline(ep, tp, entry_idx, exit_idx, prev_plotsize):
  # ------ vline entry & exit ------ #
  ep_tick = prev_plotsize
  # tp_tick = prev_plotsize + (exit_idx - entry_idx)
  tp_tick = prev_plotsize + int(exit_idx - entry_idx)
  plt.axvline(ep_tick, alpha=0.7, linestyle='--', color='#ffeb3b')
  plt.axvline(tp_tick, alpha=1., linestyle='--', color='#ffeb3b')
  # ------ hline entry & exit ------ #
  x0, x1 = plt.gca().get_xlim()
  ep_xmin = ep_tick / x1
  tp_xmin = tp_tick / x1

  plt.axhline(ep, linestyle='--', xmin=ep_xmin, xmax=1, alpha=1, color='lime')  # ep line axhline
  plt.text(x1, ep, ' ep :\n {}'.format(ep), ha='left', va='center', fontweight='bold')  # ep line label

  plt.axhline(tp, linestyle='--', xmin=tp_xmin, xmax=1, alpha=1, color='lime')  # tp line axhline
  plt.text(x1, tp, ' tp :\n {}'.format(tp), ha='left', va='center', fontweight='bold')  # tp line label

  return ep_tick

def plot_check_v5(data, config, param_zip, pr_msg, x_max, x_margin_mult, y_margin_mult, front_plot, plot_check_dir=None, **col_idx_dict):
  # start_0 = time.time()
  plt.style.use(['dark_background', 'fast'])
  fig = plt.figure(figsize=(30, 18))
  nrows, ncols = 2, 2
  gs = gridspec.GridSpec(nrows=nrows,  # row 부터 index 채우고 col 채우는 순서임 (gs_idx)
                         ncols=ncols,
                         height_ratios=[3, 1]
                         )
  for gs_idx, params in enumerate(param_zip):
    ax = fig.add_subplot(gs[gs_idx])
    iin, iout, pr, ep, tp, entry_idx, exit_idx, open_idx, point1_idx, lvrg, fee, tp_line, out_line, en_tp1, en_out0 = params

    # if ep > out_line:  # for tp > ep > out plot_check
    #   break

    # ------------ add_col section ------------ #
    a_data = data[int(iin):int(iout + 1)]
    # a_data = data[iin:iout]
    # ------ candles ------ #
    candle_plot(a_data[:, col_idx_dict['ohlc_col_idxs']], ax, alpha=1.0, wickwidth=1.0)

    # ------ add cols ------ #
    [nonstep_col_plot(a_data[:, params[0]], *params[1:]) for params in col_idx_dict['nonstep_col_info']]
    [step_col_plot(a_data[:, params[0]], *params[1:]) for params in col_idx_dict['step_col_info']]
    [stepmark_col_plot(a_data[:, params[0]], *params[1:]) for params in col_idx_dict['stepmark_col_info']]

    # ------ ep, tp + xlim ------ #
    try:
      eptp_hvline_v5(config, ep, tp, entry_idx, exit_idx, open_idx, point1_idx, tp_line, out_line, en_tp1, en_out0,
                    front_plot, iin, iout, x_max, x_margin_mult, y_margin_mult, a_data, **col_idx_dict)
    except Exception as e:
      print("error in eptp_hvline_v3 :", e)

    #     Todo    #
    #     3. outer_price plot 일 경우, gs_idx + nrows 하면 됨

    # ------ trade_info ------ #
    plt.title(pr_msg.format(entry_idx, exit_idx, pr, lvrg, fee))

  if plot_check_dir is None:
    plt.show()
    print()
  else:
    fig_name = plot_check_dir + "/%s.png" % int(entry_idx)
    plt.savefig(fig_name)
    print(fig_name, "saved !")
  plt.close()
  # print("elapsed time :", time.time() - start_0)

  return

def plot_check_v4(data, config, param_zip, pr_msg, x_max, x_margin_mult, y_margin_mult, front_plot, plot_check_dir=None, **col_idx_dict):
  # start_0 = time.time()
  plt.style.use(['dark_background', 'fast'])
  fig = plt.figure(figsize=(30, 18))
  nrows, ncols = 2, 2
  gs = gridspec.GridSpec(nrows=nrows,  # row 부터 index 채우고 col 채우는 순서임 (gs_idx)
                         ncols=ncols,
                         height_ratios=[3, 1]
                         )
  for gs_idx, params in enumerate(param_zip):
    ax = fig.add_subplot(gs[gs_idx])
    iin, iout, pr, ep, tp, entry_idx, exit_idx, open_idx, point1_idx, lvrg, fee, tp_line, out_line, bias_info, bias_thresh = params

    # if ep > out_line:  # for tp > ep > out plot_check
    #   break

    # ------------ add_col section ------------ #
    a_data = data[int(iin):int(iout + 1)]
    # a_data = data[iin:iout]
    # ------ candles ------ #
    candle_plot(a_data[:, col_idx_dict['ohlc_col_idxs']], ax, alpha=1.0, wickwidth=1.0)

    # ------ add cols ------ #
    [nonstep_col_plot(a_data[:, params[0]], *params[1:]) for params in col_idx_dict['nonstep_col_info']]
    [step_col_plot(a_data[:, params[0]], *params[1:]) for params in col_idx_dict['step_col_info']]
    [stepmark_col_plot(a_data[:, params[0]], *params[1:]) for params in col_idx_dict['stepmark_col_info']]

    # ------ ep, tp + xlim ------ #
    try:
      eptp_hvline_v4(config, ep, tp, entry_idx, exit_idx, open_idx, point1_idx, tp_line, out_line, bias_info, bias_thresh,
                    front_plot, iin, iout, x_max, x_margin_mult, y_margin_mult, a_data, **col_idx_dict)
    except Exception as e:
      print("error in eptp_hvline_v3 :", e)

    #     Todo    #
    #     3. outer_price plot 일 경우, gs_idx + nrows 하면 됨

    # ------ trade_info ------ #
    plt.title(pr_msg.format(entry_idx, exit_idx, pr, lvrg, fee))

  if plot_check_dir is None:
    plt.show()
    print()
  else:
    fig_name = plot_check_dir + "/%s.png" % int(entry_idx)
    plt.savefig(fig_name)
    print(fig_name, "saved !")
  plt.close()
  # print("elapsed time :", time.time() - start_0)

  return

def plot_check_v3(data, config, param_zip, pr_msg, x_max, x_margin_mult, y_margin_mult, front_plot, plot_check_dir=None, **col_idx_dict):
  # start_0 = time.time()
  plt.style.use(['dark_background', 'fast'])
  fig = plt.figure(figsize=(30, 18))
  nrows, ncols = 2, 2
  gs = gridspec.GridSpec(nrows=nrows,  # row 부터 index 채우고 col 채우는 순서임 (gs_idx)
                         ncols=ncols,
                         height_ratios=[3, 1]
                         )
  for gs_idx, params in enumerate(param_zip):
    ax = fig.add_subplot(gs[gs_idx])
    iin, iout, pr, ep, tp, entry_idx, exit_idx, open_idx, lvrg, fee, tp_line, out_line, bias_info, bias_thresh = params

    # if ep > out_line:  # for tp > ep > out plot_check
    #   break

    # ------------ add_col section ------------ #
    a_data = data[int(iin):int(iout + 1)]
    # a_data = data[iin:iout]
    # ------ candles ------ #
    candle_plot(a_data[:, col_idx_dict['ohlc_col_idxs']], ax, alpha=1.0, wickwidth=1.0)

    # ------ add cols ------ #
    [nonstep_col_plot(a_data[:, params[0]], *params[1:]) for params in col_idx_dict['nonstep_col_info']]
    [step_col_plot(a_data[:, params[0]], *params[1:]) for params in col_idx_dict['step_col_info']]
    [stepmark_col_plot(a_data[:, params[0]], *params[1:]) for params in col_idx_dict['stepmark_col_info']]

    # ------ ep, tp + xlim ------ #
    try:
      eptp_hvline_v3(config, ep, tp, entry_idx, exit_idx, open_idx, tp_line, out_line, bias_info, bias_thresh,
                    front_plot, iin, iout, x_max, x_margin_mult, y_margin_mult, a_data, **col_idx_dict)
    except Exception as e:
      print("error in eptp_hvline_v3 :", e)

    #     Todo    #
    #     3. outer_price plot 일 경우, gs_idx + nrows 하면 됨

    # ------ trade_info ------ #
    plt.title(pr_msg.format(entry_idx, exit_idx, pr, lvrg, fee))

  if plot_check_dir is None:
    plt.show()
    print()
  else:
    fig_name = plot_check_dir + "/%s.png" % int(entry_idx)
    plt.savefig(fig_name)
    print(fig_name, "saved !")
  plt.close()
  # print("elapsed time :", time.time() - start_0)

  return

def plot_check_v2(data, config, param_zip, pr_msg, x_max, x_margin_mult, y_margin_mult, front_plot, plot_check_dir=None, **col_idx_dict):
  # start_0 = time.time()
  plt.style.use(['dark_background', 'fast'])
  fig = plt.figure(figsize=(30, 18))
  nrows, ncols = 2, 2
  gs = gridspec.GridSpec(nrows=nrows,  # row 부터 index 채우고 col 채우는 순서임 (gs_idx)
                         ncols=ncols,
                         height_ratios=[3, 1]
                         )
  for gs_idx, params in enumerate(param_zip):
    ax = fig.add_subplot(gs[gs_idx])
    iin, iout, pr, ep, tp, entry_idx, exit_idx, open_idx, lvrg, fee, tp_line, out_line = params

    # ------------ add_col section ------------ #
    a_data = data[int(iin):int(iout)]
    # a_data = data[iin:iout]
    # ------ candles ------ #
    candle_plot(a_data[:, col_idx_dict['ohlc_col_idxs']], ax, alpha=1.0, wickwidth=1.0)

    # ------ add cols ------ #
    [nonstep_col_plot(a_data[:, params[0]], *params[1:]) for params in col_idx_dict['nonstep_col_info']]
    [step_col_plot(a_data[:, params[0]], *params[1:]) for params in col_idx_dict['step_col_info']]
    [stepmark_col_plot(a_data[:, params[0]], *params[1:]) for params in col_idx_dict['stepmark_col_info']]

    # ------ ep, tp + xlim ------ #
    eptp_hvline_v2(config, ep, tp, entry_idx, exit_idx, open_idx, tp_line, out_line, front_plot, iin, iout, x_max, x_margin_mult, y_margin_mult,
                   a_data, **col_idx_dict)

    #     Todo    #
    #     3. outer_price plot 일 경우, gs_idx + nrows 하면 됨

    # ------ trade_info ------ #
    plt.title(pr_msg.format(entry_idx, exit_idx, pr, lvrg, fee))

  if plot_check_dir is None:
    plt.show()
    print()
  else:
    fig_name = plot_check_dir + "/%s.png" % int(entry_idx)
    plt.savefig(fig_name)
    print(fig_name, "saved !")
  plt.close()
  # print("elapsed time :", time.time() - start_0)

  return

def plot_check(data, iin, iout, pr, ep, tp, entry_idx, exit_idx, lvrg, fee, pr_msg, x_max, x_margin, front_plot, **col_idx_dict):
  plt.style.use(['dark_background', 'fast'])
  fig = plt.figure(figsize=(26, 18))
  gs = gridspec.GridSpec(nrows=2,  # row 몇 개
                         ncols=2,  # col 몇 개
                         height_ratios=[3, 1]
                         )

  ax = fig.add_subplot(gs[0])
  # ax = fig.add_subplot(gs[odds - 1])

  a_data = data[int(iin):int(iout)]
  # a_data = data[iin:iout]
  candle_plot(a_data[:, col_idx_dict['ohlc_col_idxs']], ax)
  _ = [hcandle_plot(a_data[:, hoc_col_idxs], square=sq) for (hoc_col_idxs, sq) in zip(col_idx_dict['hoc_col_idxs_list'], np.arange(3) + 1)]

  #     1. ep, tp    #
  #     Todo    #
  #     2. norm_range_col - decided by a_data's col
  ep_tick = eptp_hvline(ep, tp, entry_idx, exit_idx, entry_idx - iin)

  if front_plot:
    x_max = ep_tick
  if (iout - iin) > x_max:
    plt.xlim(0 - x_margin, x_max + x_margin)

  # ------ check pr ------ #
  # if not config.lvrg_set.static_lvrg:
  plt.title(pr_msg.format(entry_idx, exit_idx, pr, lvrg, fee))
  plt.show()
  plt.close()


def candle_plot(ohlc_np, ax, alpha=1.0, wickwidth=1.0):
  index = np.arange(len(ohlc_np))
  candle = np.hstack((np.reshape(index, (-1, 1)), ohlc_np))
  mf.candlestick_ohlc(ax, candle, width=0.5, colorup='#26a69a', colordown='#ef5350', alpha=alpha, wickwidth=wickwidth)


# @jit
def candle_plot_nb(ohlc_np, ax):  # slower
  index = np.arange(len(ohlc_np))
  candle = np.hstack((np.reshape(index, (-1, 1)), ohlc_np))
  mf.candlestick_ohlc(ax, candle, width=0.5, colorup='#26a69a', colordown='#ef5350', alpha=0.5)


def hcandle_plot(hcandle_np, alpha=1, square=1, color='#ffffff'):
  assert hcandle_np.shape[-1] == 2, "assert hcandle_np.shape[-1] == 2"
  try:
    plt.step(np.arange(len(hcandle_np)), hcandle_np, alpha=alpha, color=color, linewidth=2 ** square)
    # plt.fill_between(np.arange(len(plot_df)), plot_df['hclose_60'].values, plot_df['hopen_60'].values,
    #                     where=1, facecolor='#ffffff', alpha=alpha)
  except Exception as e:
    print("error in hcandle_plot :", e)


