import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import mpl_finance as mf
from funcs.funcs_idep import get_col_idxs
from scipy import stats
# from numba import jit


def strcol_tonumb(res_df, col_list):
  step_col_arr = np.array(col_list)
  if len(col_list) != 0:
    step_col_arr[:, 0] = [get_col_idxs(res_df, col_) for col_ in step_col_arr[:, 0]]   # str_col to number

  return step_col_arr


def candle_plot_v2(ax, ohlc_np, alpha=1.0, wickwidth=1.0):
  index = np.arange(len(ohlc_np))
  candle = np.hstack((np.reshape(index, (-1, 1)), ohlc_np))
  mf.candlestick_ohlc(ax, candle, width=0.5, colorup='#26a69a', colordown='#ef5350', alpha=alpha, wickwidth=wickwidth)


def nonstep_col_plot_v2(ax, np_col, alpha=1, color='#ffffff', linewidth=2):
  try:
    ax.plot(np.arange(len(np_col)), np_col, alpha=alpha, color=color, linewidth=linewidth)
  except Exception as e:
    print("error in nonstep_col_plot :", e)

def step_col_plot_v2(ax, np_col, alpha=1, color='#ffffff', linewidth=2):
  try:
    ax.step(np.arange(len(np_col)), np_col, alpha=alpha, color=color, linewidth=linewidth)
  except Exception as e:
    print("error in step_col_plot :", e)

def stepmark_col_plot_v2(ax, np_col, alpha=1, color='#ffffff', markersize=2, markerstyle='c*'):
  try:
    ax.step(np.arange(len(np_col)), np_col, markerstyle, alpha=alpha, color=color, markersize=markersize)
  except Exception as e:
    print("error in stepmark_col_plot :", e)

def fillbw_col_plot_v2(ax, np_col, alpha=1, color='#ffffff', linewidth=2):
  try:
    ax.step(np.arange(len(np_col)), np_col, alpha=alpha, color=color, linewidth=linewidth)
  except Exception as e:
    print("error in fillbw_col_plot :", e)

def sort_bypr_v4(pr, obj, arr_list, descending=True):

  pr_ravel = pr.ravel()
  descend_sort_idx = pr_ravel.argsort()[::-1]
  if descending:
    sort_idx = descend_sort_idx
  else:
    descend_pr = pr_ravel[descend_sort_idx]
    true_pr_idx = descend_pr > 1
    sort_idx = np.hstack((descend_sort_idx[true_pr_idx][::-1], descend_sort_idx[~true_pr_idx][::-1]))

  return pr[sort_idx], [ob[sort_idx] for ob in obj], [arr_[sort_idx] for arr_ in arr_list]

def sort_bypr_v3(pr, obj, arr_list, descending=True):
  if descending:
    sort_idx = pr.ravel().argsort()[::-1]
  else:
    sort_idx = pr.ravel().argsort()

  return pr[sort_idx], [ob[sort_idx] for ob in obj], [arr_[sort_idx] for arr_ in arr_list]

def whole_plot_check(data, win_idxs, selected_op_idxs, selected_ex_idxs, plot_check_dir=None, **col_idx_dict):
  # start_0 = time.time()
  plt.style.use(['dark_background', 'fast'])
  fig = plt.figure(figsize=(30, 12))
  nrows, ncols = 1, 1
  gs = gridspec.GridSpec(nrows=nrows,  # row 부터 index 채우고 col 채우는 순서임 (gs_idx)
                         ncols=ncols,
                         # height_ratios=[31, 1]
                         )

  ax = fig.add_subplot(gs[0])

  # ------------ add_col section ------------ #
  a_data = data[selected_op_idxs[0]:selected_op_idxs[-1] + 1]

  plot_op_idxs = selected_op_idxs - selected_op_idxs[0]
  plot_win_op_idxs = plot_op_idxs[win_idxs]
  plot_loss_op_idxs = plot_op_idxs[~win_idxs]

  plot_ex_idxs = selected_ex_idxs - selected_op_idxs[0]
  plot_win_ex_idxs = plot_ex_idxs[win_idxs]
  plot_loss_ex_idxs = plot_ex_idxs[~win_idxs]

  # ------ add cols ------ #
  [nonstep_col_plot(a_data[:, params[0]], *params[1:]) for params in col_idx_dict['nonstep_col_info']]
  [step_col_plot(a_data[:, params[0]], *params[1:]) for params in col_idx_dict['step_col_info']]
  [stepmark_col_plot(a_data[:, params[0]], *params[1:]) for params in col_idx_dict['stepmark_col_info']]

  # [plt.axvline(op_idx, color='#00ff00') for op_idx in plot_win_op_idxs]
  # [plt.axvline(op_idx, color='#ff0000') for op_idx in plot_loss_op_idxs]
  [plt.axvspan(op_idx, ex_idx, alpha=0.5, color='#00ff00') for op_idx, ex_idx in zip(plot_win_op_idxs, plot_win_ex_idxs)]
  [plt.axvspan(op_idx, ex_idx, alpha=0.5, color='#ff0000') for op_idx, ex_idx in zip(plot_loss_op_idxs, plot_loss_ex_idxs)]

  plt.show()

  if plot_check_dir is None:
    plt.show()
    print()
  else:
    fig_name = plot_check_dir + "/whole_plot_{}.png".format(selected_op_idxs[0])
    plt.savefig(fig_name)
    print(fig_name, "saved !")
  plt.close()
  # print("elapsed time :", time.time() - start_0)

  return


def eptp_hvline_v9_1(ax1, ax2, config, iin, iout, pr, en_p, ex_p, entry_idx, exit_idx, p1_idx, p2_idx, lvrg, fee, tp_line, out_line, tp_1, tp_0,
                     out_1, out_0, ep2_0,
                     back_plot, x_max, x_margin_mult, y_margin_mult, a_data, vp_info, **col_idx_dict):
  # ------ get vertical ticks ------ #
  entry_tick = int(entry_idx - iin)
  exit_tick = entry_tick + int(exit_idx - entry_idx)
  p1_tick = entry_tick - int(entry_idx - p1_idx)
  p2_tick = p1_tick + int(p2_idx - p1_idx)

  if back_plot == 1:
    x_max = p1_tick + 20
  elif back_plot == 2:
    x_max = p2_tick + 20
  elif back_plot == 3:
    x_max = entry_tick + 20
  elif back_plot == 4:
    x_max = exit_tick + 20

  # ============ xlim ============ #
  if (iout - iin) > x_max:
    x_margin = x_max * x_margin_mult
    ax1.set_xlim(0 - x_margin, x_max + x_margin)
    ax2.set_xlim(0 - x_margin, x_max + x_margin)
  x0, x1 = ax1.get_xlim()

  # ============ hlines ============ #
  # ------------ ax1 ------------ #
  # ------ entry & exit ------ #
  en_xmin = entry_tick / x1
  ex_xmin = exit_tick / x1
  ax1.axhline(en_p, x0, en_xmin, linewidth=2, linestyle='--', alpha=1, color='lime')  # en_p line axhline
  if config.tr_set.check_hlm in [0, 1]:
    ax1.text(x0, en_p, 'en_p :\n {:.3f} \n epg1 {}'.format(en_p, config.tr_set.ep_gap1), ha='right', va='center', fontweight='bold',
             fontsize=15)  # en_p line label
  else:
    ax1.text(x0, en_p, 'en_p :\n {:.3f} \n epg2 {}'.format(en_p, config.tr_set.ep_gap2), ha='right', va='center', fontweight='bold',
             fontsize=15)  # en_p line label
  ax1.axhline(ex_p, ex_xmin, 1, linewidth=2, linestyle='--', alpha=1, color='lime')  # ex_p line axhline (signal 도 포괄함, 존재 의미)
  ax1.text(x1, ex_p, 'ex_p :\n {}'.format(ex_p), ha='left', va='center', fontweight='bold', fontsize=15)  # ex_p line label

  # ------ tpout_line ------ #
  ax1.axhline(tp_line, 0.05, 1, linewidth=2, linestyle='-', alpha=1, color='#00ff00')  # ep 와 gap 비교 용이하기 위해 ex_xmin -> 0.1 사용
  ax1.text(x0, tp_line, 'tpg {}'.format(config.tr_set.tp_gap), ha='right', va='center', fontweight='bold', fontsize=15, color='#00ff00')
  ax1.axhline(out_line, 0.05, 1, linewidth=2, linestyle='-', alpha=1, color='#ff0000')
  ax1.text(x0, out_line, 'outg {}'.format(config.tr_set.out_gap), ha='right', va='center', fontweight='bold', fontsize=15, color='#ff0000')

  # ------ tp_box ------ #
  text_x_pos = (x0 + x1) * 0.1
  ax1.axhline(tp_1, 0.2, 1, linewidth=4, linestyle='-', alpha=1, color='#ffffff')
  ax1.text(text_x_pos, tp_1, ' tp_1', ha='right', va='bottom', fontweight='bold', fontsize=15)
  ax1.axhline(tp_0, 0.2, 1, linewidth=4, linestyle='-', alpha=1, color='#ffffff')
  ax1.text(text_x_pos, tp_0, ' tp_0', ha='right', va='bottom', fontweight='bold', fontsize=15)

  # ------ out_box ------ #
  # text_x_pos = (x0 + x1) * 0.1
  ax1.axhline(out_1, 0.2, 1, linewidth=2, linestyle='-', alpha=1, color='#ffffff')
  ax1.text(text_x_pos, out_1, ' out_1', ha='right', va='bottom', fontweight='bold', fontsize=15)
  ax1.axhline(out_0, 0.2, 1, linewidth=2, linestyle='-', alpha=1, color='#ffffff')
  ax1.text(text_x_pos, out_0, ' out_0', ha='right', va='bottom', fontweight='bold', fontsize=15)
  # ------ ep_box ------ #
  # text_x_pos = (x0 + x1) * 0.1
  ax1.axhline(ep2_0, 0.2, 1, linewidth=1, linestyle='-', alpha=1, color='#ffffff')
  ax1.text(text_x_pos, ep2_0, ' ep2_0', ha='right', va='bottom', fontweight='bold', fontsize=15)

  # ------ volume profile ------ #
  close, volume, kde_factor, num_samples = vp_info
  # if iin >= vp_range:
  # start_0 = time.time()
  kde = stats.gaussian_kde(close, weights=volume, bw_method=kde_factor)
  kdx = np.linspace(close.min(), close.max(), num_samples)
  kdy = kde(kdx)
  kdy_max = kdy.max()
  # print("kde elapsed_time :", time.time() - start_0)

  # peaks,_ = signal.find_peaks(kdy, prominence=kdy_max * 0.3)   # get peak_entries
  # peak_list = kdx[peaks]   # peak_list
  # [ax1.axhline(peak, linewidth=1, linestyle='-', alpha=1, color='orange') for peak in peak_list]

  kdy_ratio = p1_tick / kdy_max  # 30 / 0.0001   # max_value 가 p1_tick 까지 닿을 수 있게.
  # print("kdx :", kdx)
  # ax1.plot(kdy * kdy_ratio, kdx, color='white')  # Todo, bars 가능 ?
  # ax1.barh(kdy * kdy_ratio, kdx, color='white')  # Todo, bars 가능 ?
  ax1.barh(kdx, kdy * kdy_ratio, color='#00ff00', alpha=0.5)  # Todo, bars 가능 ?

  # ------------ ax2 ------------ #
  # ------ cci_band ------ #
  ax2.axhline(100, color="#ffffff")
  ax2.axhline(-100, color="#ffffff")

  # ------ stoch_band ------ #
  # ax2.axhline(67, color="#ffffff")
  # ax2.axhline(33, color="#ffffff")

  # ax2.axhline(0, color="#ffffff")

  # ============ ylim ============ # - ax1 only
  if back_plot:
    y_lim_data = a_data[:x_max + 1, col_idx_dict['ylim_col_idxs']]  # +1 for including p1_tick
  else:
    y_lim_data = a_data[:, col_idx_dict['ylim_col_idxs']]

  y_min = y_lim_data.min()
  y_max = y_lim_data.max()
  y_margin = (y_max - y_min) * y_margin_mult
  ax1.set_ylim(y_min - y_margin, y_max + y_margin)

  # ============ vline (p1_tick, entry_tick, exit_tick) ============ # - add p1_tick on ax2
  y0, y1 = ax1.get_ylim()
  low_data = a_data[:exit_tick + 1, col_idx_dict['ohlc_col_idxs'][2]]  # +1 for including exit_tick
  p2_ymax, en_ymax, ex_ymax = [(low_data[tick_] - y0) / (y1 - y0) - .01 for tick_ in [p2_tick, entry_tick, exit_tick]]  # -.05 for margin
  if p1_tick > 0:
    p1_ymax = (low_data[p1_tick] - y0) / (y1 - y0) - .01
    ax1.axvline(p1_tick, 0, p1_ymax, alpha=1, linewidth=2, linestyle='--', color='#ff0000')  # 추후, tick 별 세부 정의가 달라질 수 있음을 고려해 multi_line 작성 유지
    ax2.axvline(p1_tick, 0, 1, alpha=1, linewidth=2, linestyle='--', color='#ff0000')
  ax1.axvline(p2_tick, 0, p2_ymax, alpha=1, linewidth=2, linestyle='--', color='#2196f3')
  ax1.axvline(entry_tick, 0, en_ymax, alpha=1, linewidth=2, linestyle='--', color='#ffeb3b')
  ax1.axvline(exit_tick, 0, ex_ymax, alpha=1, linewidth=2, linestyle='--', color='#ffeb3b')
  ax2.axvline(p2_tick, 0, 1, alpha=1, linewidth=2, linestyle='--', color='#2196f3')
  ax2.axvline(entry_tick, 0, 1, alpha=1, linewidth=2, linestyle='--', color='#ffeb3b')
  ax2.axvline(exit_tick, 0, 1, alpha=1, linewidth=2, linestyle='--', color='#ffeb3b')

  return

def eptp_hvline_v9(ax1, ax2, config, iin, iout, pr, en_p, ex_p, entry_idx, exit_idx, p1_idx, p2_idx, lvrg, fee, tp_line, out_line, tp_1, tp_0, out_1,
                   out_0, ep2_0,
                   back_plot, x_max, x_margin_mult, y_margin_mult, a_data, **col_idx_dict):
  # ------ get vertical ticks ------ #
  entry_tick = int(entry_idx - iin)
  exit_tick = entry_tick + int(exit_idx - entry_idx)
  p1_tick = entry_tick - int(entry_idx - p1_idx)
  p2_tick = p1_tick + int(p2_idx - p1_idx)

  if back_plot == 1:
    x_max = p1_tick + 20
  elif back_plot == 2:
    x_max = p2_tick + 20
  elif back_plot == 3:
    x_max = entry_tick + 20
  elif back_plot == 4:
    x_max = exit_tick + 20

  # ============ xlim ============ #
  if (iout - iin) > x_max:
    x_margin = x_max * x_margin_mult
    ax1.set_xlim(0 - x_margin, x_max + x_margin)
    ax2.set_xlim(0 - x_margin, x_max + x_margin)
  x0, x1 = ax1.get_xlim()

  # ============ hlines ============ #
  # ------------ ax1 ------------ #
  # ------ entry & exit ------ #
  en_xmin = entry_tick / x1
  ex_xmin = exit_tick / x1
  ax1.axhline(en_p, x0, en_xmin, linewidth=2, linestyle='--', alpha=1, color='lime')  # en_p line axhline
  if config.tr_set.check_hlm in [0, 1]:
    ax1.text(x0, en_p, 'en_p :\n {:.3f} \n epg1 {}'.format(en_p, config.tr_set.ep_gap1), ha='right', va='center', fontweight='bold',
             fontsize=15)  # en_p line label
  else:
    ax1.text(x0, en_p, 'en_p :\n {:.3f} \n epg2 {}'.format(en_p, config.tr_set.ep_gap2), ha='right', va='center', fontweight='bold',
             fontsize=15)  # en_p line label
  ax1.axhline(ex_p, ex_xmin, 1, linewidth=2, linestyle='--', alpha=1, color='lime')  # ex_p line axhline (signal 도 포괄함, 존재 의미)
  ax1.text(x1, ex_p, 'ex_p :\n {}'.format(ex_p), ha='left', va='center', fontweight='bold', fontsize=15)  # ex_p line label

  # ------ tpout_line ------ #
  ax1.axhline(tp_line, 0.05, 1, linewidth=2, linestyle='-', alpha=1, color='#00ff00')  # ep 와 gap 비교 용이하기 위해 ex_xmin -> 0.1 사용
  ax1.text(x0, tp_line, 'tpg {}'.format(config.tr_set.tp_gap), ha='right', va='center', fontweight='bold', fontsize=15, color='#00ff00')
  ax1.axhline(out_line, 0.05, 1, linewidth=2, linestyle='-', alpha=1, color='#ff0000')
  ax1.text(x0, out_line, 'outg {}'.format(config.tr_set.out_gap), ha='right', va='center', fontweight='bold', fontsize=15, color='#ff0000')

  # ------ tp_box ------ #
  text_x_pos = (x0 + x1) * 0.1
  ax1.axhline(tp_1, 0.2, 1, linewidth=4, linestyle='-', alpha=1, color='#ffffff')
  ax1.text(text_x_pos, tp_1, ' tp_1', ha='right', va='bottom', fontweight='bold', fontsize=15)
  ax1.axhline(tp_0, 0.2, 1, linewidth=4, linestyle='-', alpha=1, color='#ffffff')
  ax1.text(text_x_pos, tp_0, ' tp_0', ha='right', va='bottom', fontweight='bold', fontsize=15)

  # ------ out_box ------ #
  # text_x_pos = (x0 + x1) * 0.1
  ax1.axhline(out_1, 0.2, 1, linewidth=2, linestyle='-', alpha=1, color='#ffffff')
  ax1.text(text_x_pos, out_1, ' out_1', ha='right', va='bottom', fontweight='bold', fontsize=15)
  ax1.axhline(out_0, 0.2, 1, linewidth=2, linestyle='-', alpha=1, color='#ffffff')
  ax1.text(text_x_pos, out_0, ' out_0', ha='right', va='bottom', fontweight='bold', fontsize=15)
  # ------ ep_box ------ #
  # text_x_pos = (x0 + x1) * 0.1
  ax1.axhline(ep2_0, 0.2, 1, linewidth=1, linestyle='-', alpha=1, color='#ffffff')
  ax1.text(text_x_pos, ep2_0, ' ep2_0', ha='right', va='bottom', fontweight='bold', fontsize=15)

  # ------------ ax2 ------------ #
  # ------ cci_band ------ #
  ax2.axhline(100, color="#ffffff")
  ax2.axhline(-100, color="#ffffff")

  # ------ stoch_band ------ #
  # ax2.axhline(67, color="#ffffff")
  # ax2.axhline(33, color="#ffffff")

  # ax2.axhline(0, color="#ffffff")

  # ============ ylim ============ # - ax1 only
  if back_plot:
    y_lim_data = a_data[:x_max + 1, col_idx_dict['ylim_col_idxs']]  # +1 for including p1_tick
  else:
    y_lim_data = a_data[:, col_idx_dict['ylim_col_idxs']]

  y_min = y_lim_data.min()
  y_max = y_lim_data.max()
  y_margin = (y_max - y_min) * y_margin_mult
  ax1.set_ylim(y_min - y_margin, y_max + y_margin)

  # ============ vline (p1_tick, entry_tick, exit_tick) ============ # - add p1_tick on ax2
  y0, y1 = ax1.get_ylim()
  low_data = a_data[:exit_tick + 1, col_idx_dict['ohlc_col_idxs'][2]]  # +1 for including exit_tick
  p2_ymax, en_ymax, ex_ymax = [(low_data[tick_] - y0) / (y1 - y0) - .01 for tick_ in [p2_tick, entry_tick, exit_tick]]  # -.05 for margin
  if p1_tick > 0:
    p1_ymax = (low_data[p1_tick] - y0) / (y1 - y0) - .01
    ax1.axvline(p1_tick, 0, p1_ymax, alpha=1, linewidth=2, linestyle='--', color='#ff0000')  # 추후, tick 별 세부 정의가 달라질 수 있음을 고려해 multi_line 작성 유지
    ax2.axvline(p1_tick, 0, 1, alpha=1, linewidth=2, linestyle='--', color='#ff0000')
  ax1.axvline(p2_tick, 0, p2_ymax, alpha=1, linewidth=2, linestyle='--', color='#2196f3')
  ax1.axvline(entry_tick, 0, en_ymax, alpha=1, linewidth=2, linestyle='--', color='#ffeb3b')
  ax1.axvline(exit_tick, 0, ex_ymax, alpha=1, linewidth=2, linestyle='--', color='#ffeb3b')
  ax2.axvline(p2_tick, 0, 1, alpha=1, linewidth=2, linestyle='--', color='#2196f3')
  ax2.axvline(entry_tick, 0, 1, alpha=1, linewidth=2, linestyle='--', color='#ffeb3b')
  ax2.axvline(exit_tick, 0, 1, alpha=1, linewidth=2, linestyle='--', color='#ffeb3b')

  return

def eptp_hvline_v8(ax1, ax2, config, iin, iout, pr, en_p, ex_p, entry_idx, exit_idx, p1_idx, p2_idx, lvrg, fee, tp_line, out_line, tp_1, tp_0, out_1,
                   out_0, ep2_0,
                   back_plot, x_max, x_margin_mult, y_margin_mult, a_data, **col_idx_dict):
  # ------ get vertical ticks ------ #
  entry_tick = int(entry_idx - iin)
  exit_tick = entry_tick + int(exit_idx - entry_idx)
  p1_tick = entry_tick - int(entry_idx - p1_idx)
  p2_tick = p1_tick + int(p2_idx - p1_idx)
  bias_info_tick = entry_tick + config.tr_set.bias_info_tick

  if back_plot == 1:
    x_max = p1_tick + 20
  elif back_plot == 2:
    x_max = p2_tick + 20
  elif back_plot == 3:
    x_max = entry_tick + 20
  elif back_plot == 4:
    x_max = exit_tick + 20
  elif back_plot == 5:
    x_max = bias_info_tick + 20

  # ============ xlim ============ #
  if (iout - iin) > x_max:
    x_margin = x_max * x_margin_mult
    ax1.set_xlim(0 - x_margin, x_max + x_margin)
    ax2.set_xlim(0 - x_margin, x_max + x_margin)
  x0, x1 = ax1.get_xlim()

  # ============ hlines ============ #
  # ------------ ax1 ------------ #
  # ------ entry & exit ------ #
  en_xmin = entry_tick / x1
  ex_xmin = exit_tick / x1
  ax1.axhline(en_p, x0, en_xmin, linewidth=2, linestyle='--', alpha=1, color='lime')  # en_p line axhline
  if config.tr_set.check_hlm in [0, 1]:
    ax1.text(x0, en_p, 'en_p :\n {:.3f} \n epg1 {}'.format(en_p, config.tr_set.ep_gap1), ha='right', va='center', fontweight='bold',
             fontsize=15)  # en_p line label
  else:
    ax1.text(x0, en_p, 'en_p :\n {:.3f} \n epg2 {}'.format(en_p, config.tr_set.ep_gap2), ha='right', va='center', fontweight='bold',
             fontsize=15)  # en_p line label
  ax1.axhline(ex_p, ex_xmin, 1, linewidth=2, linestyle='--', alpha=1, color='lime')  # ex_p line axhline (signal 도 포괄함, 존재 의미)
  ax1.text(x1, ex_p, 'ex_p :\n {}'.format(ex_p), ha='left', va='center', fontweight='bold', fontsize=15)  # ex_p line label

  # ------ tpout_line ------ #
  ax1.axhline(tp_line, 0.05, 1, linewidth=2, linestyle='-', alpha=1, color='#00ff00')  # ep 와 gap 비교 용이하기 위해 ex_xmin -> 0.1 사용
  ax1.text(x0, tp_line, 'tpg {}'.format(config.tr_set.tp_gap), ha='right', va='center', fontweight='bold', fontsize=15, color='#00ff00')
  ax1.axhline(out_line, 0.05, 1, linewidth=2, linestyle='-', alpha=1, color='#ff0000')
  ax1.text(x0, out_line, 'outg {}'.format(config.tr_set.out_gap), ha='right', va='center', fontweight='bold', fontsize=15, color='#ff0000')

  # ------ tp_box ------ #
  text_x_pos = (x0 + x1) * 0.1
  ax1.axhline(tp_1, 0.2, 1, linewidth=4, linestyle='-', alpha=1, color='#ffffff')
  ax1.text(text_x_pos, tp_1, ' tp_1', ha='right', va='bottom', fontweight='bold', fontsize=15)
  ax1.axhline(tp_0, 0.2, 1, linewidth=4, linestyle='-', alpha=1, color='#ffffff')
  ax1.text(text_x_pos, tp_0, ' tp_0', ha='right', va='bottom', fontweight='bold', fontsize=15)

  # ------ out_box ------ #
  # text_x_pos = (x0 + x1) * 0.1
  ax1.axhline(out_1, 0.2, 1, linewidth=2, linestyle='-', alpha=1, color='#ffffff')
  ax1.text(text_x_pos, out_1, ' out_1', ha='right', va='bottom', fontweight='bold', fontsize=15)
  ax1.axhline(out_0, 0.2, 1, linewidth=2, linestyle='-', alpha=1, color='#ffffff')
  ax1.text(text_x_pos, out_0, ' out_0', ha='right', va='bottom', fontweight='bold', fontsize=15)
  # ------ ep_box ------ #
  # text_x_pos = (x0 + x1) * 0.1
  ax1.axhline(ep2_0, 0.2, 1, linewidth=1, linestyle='-', alpha=1, color='#ffffff')
  ax1.text(text_x_pos, ep2_0, ' ep2_0', ha='right', va='bottom', fontweight='bold', fontsize=15)

  # ------------ ax2 ------------ #
  # ------ band ------ #
  ax2.axhline(100, color="#ffffff")
  ax2.axhline(-100, color="#ffffff")

  # ============ ylim ============ # - ax1 only
  if back_plot:
    y_lim_data = a_data[:x_max + 1, col_idx_dict['ylim_col_idxs']]  # +1 for including p1_tick
  else:
    y_lim_data = a_data[:, col_idx_dict['ylim_col_idxs']]

  y_min = y_lim_data.min()
  y_max = y_lim_data.max()
  y_margin = (y_max - y_min) * y_margin_mult
  ax1.set_ylim(y_min - y_margin, y_max + y_margin)

  # ============ vline (p1_tick, entry_tick, exit_tick) ============ # - add p1_tick on ax2
  y0, y1 = ax1.get_ylim()
  low_data = a_data[:exit_tick + 1, col_idx_dict['ohlc_col_idxs'][2]]  # +1 for including exit_tick
  p2_ymax, en_ymax, ex_ymax = [(low_data[tick_] - y0) / (y1 - y0) - .01 for tick_ in [p2_tick, entry_tick, exit_tick]]  # -.05 for margin
  if p1_tick > 0:
    p1_ymax = (low_data[p1_tick] - y0) / (y1 - y0) - .01
    ax1.axvline(p1_tick, 0, p1_ymax, alpha=1, linewidth=2, linestyle='--', color='#ff0000')  # 추후, tick 별 세부 정의가 달라질 수 있음을 고려해 multi_line 작성 유지
    ax2.axvline(p1_tick, 0, 1, alpha=1, linewidth=2, linestyle='--', color='#ff0000')
  ax1.axvline(p2_tick, 0, p2_ymax, alpha=1, linewidth=2, linestyle='--', color='#2196f3')
  ax1.axvline(entry_tick, 0, en_ymax, alpha=1, linewidth=2, linestyle='--', color='#ffeb3b')
  ax1.axvline(exit_tick, 0, ex_ymax, alpha=1, linewidth=2, linestyle='--', color='#ffeb3b')
  ax2.axvline(p2_tick, 0, 1, alpha=1, linewidth=2, linestyle='--', color='#2196f3')
  ax2.axvline(entry_tick, 0, 1, alpha=1, linewidth=2, linestyle='--', color='#ffeb3b')
  ax2.axvline(exit_tick, 0, 1, alpha=1, linewidth=2, linestyle='--', color='#ffeb3b')
  if back_plot == 5:
    ax1.axvline(bias_info_tick, alpha=1, linewidth=2, linestyle='-', color='#ffeb3b')

  return


def eptp_hvline_v7(ax1, ax2, config, en_p, ex_p, entry_idx, exit_idx, open_idx, p2_idx, tp_line, out_line, en_tp1, en_out0,
                   back_plot, iin, iout, x_max, x_margin_mult, y_margin_mult, a_data, **col_idx_dict):
  # ------ get vertical ticks ------ #
  entry_tick = int(entry_idx - iin)
  exit_tick = entry_tick + int(exit_idx - entry_idx)
  open_tick = entry_tick - int(entry_idx - open_idx)
  p2_tick = open_tick + int(p2_idx - open_idx)
  bias_info_tick = entry_tick + config.tr_set.bias_info_tick

  if back_plot == 1:
    x_max = open_tick + 20
  elif back_plot == 2:
    x_max = p2_tick + 20
  elif back_plot == 3:
    x_max = entry_tick + 20
  elif back_plot == 4:
    x_max = exit_tick + 20
  elif back_plot == 5:
    x_max = bias_info_tick + 20

  # ============ xlim ============ #
  if (iout - iin) > x_max:
    x_margin = x_max * x_margin_mult
    ax1.set_xlim(0 - x_margin, x_max + x_margin)
    ax2.set_xlim(0 - x_margin, x_max + x_margin)
  x0, x1 = ax1.get_xlim()

  # ============ hlines ============ #
  # ------------ ax1 ------------ #
  # ------ entry & exit ------ #
  en_xmin = entry_tick / x1
  ex_xmin = exit_tick / x1
  ax1.axhline(en_p, x0, en_xmin, linewidth=2, linestyle='--', alpha=1, color='lime')  # en_p line axhline
  ax1.text(x0, en_p, 'en_p :\n {:.3f} \n epg {}'.format(en_p, config.tr_set.ep_gap2), ha='right', va='center', fontweight='bold',
           fontsize=15)  # en_p line label
  ax1.axhline(ex_p, ex_xmin, 1, linewidth=2, linestyle='--', alpha=1, color='lime')  # ex_p line axhline (signal 도 포괄함, 존재 의미)
  ax1.text(x1, ex_p, 'ex_p :\n {}'.format(ex_p), ha='left', va='center', fontweight='bold', fontsize=15)  # ex_p line label

  # ------ tpout_line ------ #
  ax1.axhline(tp_line, 0.05, 1, linewidth=2, linestyle='-', alpha=1, color='#00ff00')  # ep 와 gap 비교 용이하기 위해 ex_xmin -> 0.1 사용
  ax1.text(x0, tp_line, 'tpg {}'.format(config.tr_set.tp_gap), ha='right', va='center', fontweight='bold', fontsize=15, color='#00ff00')
  ax1.axhline(out_line, 0.05, 1, linewidth=2, linestyle='-', alpha=1, color='#ff0000')
  ax1.text(x0, out_line, 'outg {}'.format(config.tr_set.out_gap), ha='right', va='center', fontweight='bold', fontsize=15, color='#ff0000')

  # ------ wave_line ------ #
  text_x_pos = (x0 + x1) * 0.1
  ax1.axhline(en_tp1, 0.2, 1, linewidth=2, linestyle='-', alpha=1, color='#ffffff')
  ax1.text(text_x_pos, en_tp1, ' wave_1', ha='right', va='bottom', fontweight='bold', fontsize=15)
  ax1.axhline(en_out0, 0.2, 1, linewidth=2, linestyle='-', alpha=1, color='#ffffff')
  ax1.text(text_x_pos, en_out0, ' wave_0', ha='right', va='bottom', fontweight='bold', fontsize=15)

  # ------------ ax2 ------------ #
  # ------ band ------ #
  ax2.axhline(100, color="#ffffff")
  ax2.axhline(-100, color="#ffffff")

  # ============ ylim ============ # - ax1 only
  if back_plot:
    y_lim_data = a_data[:x_max + 1, col_idx_dict['ylim_col_idxs']]  # +1 for including open_tick
  else:
    y_lim_data = a_data[:, col_idx_dict['ylim_col_idxs']]

  y_min = y_lim_data.min()
  y_max = y_lim_data.max()
  y_margin = (y_max - y_min) * y_margin_mult
  ax1.set_ylim(y_min - y_margin, y_max + y_margin)

  # ============ vline (open_tick, entry_tick, exit_tick) ============ # - add open_tick on ax2
  y0, y1 = ax1.get_ylim()
  low_data = a_data[:exit_tick + 1, col_idx_dict['ohlc_col_idxs'][2]]  # +1 for including exit_tick
  p2_ymax, en_ymax, ex_ymax = [(low_data[tick_] - y0) / (y1 - y0) - .01 for tick_ in [p2_tick, entry_tick, exit_tick]]  # -.05 for margin
  if open_tick > 0:
    open_ymax = (low_data[open_tick] - y0) / (y1 - y0) - .01
    ax1.axvline(open_tick, 0, open_ymax, alpha=1, linewidth=2, linestyle='--', color='#ffeb3b')  # 추후, tick 별 세부 정의가 달라질 수 있음을 고려해 multi_line 작성 유지
    ax2.axvline(open_tick, 0, open_ymax, alpha=1, linewidth=2, linestyle='--', color='#ffeb3b')
  ax1.axvline(p2_tick, 0, p2_ymax, alpha=1, linewidth=2, linestyle='--', color='#2196f3')
  ax1.axvline(entry_tick, 0, en_ymax, alpha=1, linewidth=2, linestyle='--', color='#ffeb3b')
  ax1.axvline(exit_tick, 0, ex_ymax, alpha=1, linewidth=2, linestyle='--', color='#ffeb3b')
  if back_plot == 5:
    ax1.axvline(bias_info_tick, alpha=1, linewidth=2, linestyle='-', color='#ffeb3b')

  return

def eptp_hvline_v6(ax1, ax2, config, en_p, ex_p, entry_idx, exit_idx, open_idx, point1_idx, tp_line, out_line, en_tp1, en_out0,
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

  # ============ xlim ============ #
  if (iout - iin) > x_max:
    x_margin = x_max * x_margin_mult
    ax1.set_xlim(0 - x_margin, x_max + x_margin)
    ax2.set_xlim(0 - x_margin, x_max + x_margin)
  x0, x1 = ax1.get_xlim()

  # ============ hlines ============ #
  # ------------ ax1 ------------ #
  # ------ entry & exit ------ #
  en_xmin = entry_tick / x1
  ex_xmin = exit_tick / x1
  ax1.axhline(en_p, x0, en_xmin, linewidth=2, linestyle='--', alpha=1, color='lime')  # en_p line axhline
  ax1.text(x0, en_p, 'en_p :\n {:.3f} \n epg {}'.format(en_p, config.tr_set.ep_gap), ha='right', va='center', fontweight='bold',
           fontsize=15)  # en_p line label
  ax1.axhline(ex_p, ex_xmin, 1, linewidth=2, linestyle='--', alpha=1, color='lime')  # ex_p line axhline (signal 도 포괄함, 존재 의미)
  ax1.text(x1, ex_p, 'ex_p :\n {}'.format(ex_p), ha='left', va='center', fontweight='bold', fontsize=15)  # ex_p line label

  # ------ tpout_line ------ #
  ax1.axhline(tp_line, 0.05, 1, linewidth=2, linestyle='-', alpha=1, color='#00ff00')  # ep 와 gap 비교 용이하기 위해 ex_xmin -> 0.1 사용
  ax1.text(x0, tp_line, 'tpg {}'.format(config.tr_set.tp_gap), ha='right', va='center', fontweight='bold', fontsize=15, color='#00ff00')
  ax1.axhline(out_line, 0.05, 1, linewidth=2, linestyle='-', alpha=1, color='#ff0000')
  ax1.text(x0, out_line, 'outg {}'.format(config.tr_set.out_gap), ha='right', va='center', fontweight='bold', fontsize=15, color='#ff0000')

  # ------ wave_line ------ #
  text_x_pos = (x0 + x1) * 0.1
  ax1.axhline(en_tp1, 0.2, 1, linewidth=2, linestyle='-', alpha=1, color='#ffffff')
  ax1.text(text_x_pos, en_tp1, ' wave_1', ha='right', va='bottom', fontweight='bold', fontsize=15)
  ax1.axhline(en_out0, 0.2, 1, linewidth=2, linestyle='-', alpha=1, color='#ffffff')
  ax1.text(text_x_pos, en_out0, ' wave_0', ha='right', va='bottom', fontweight='bold', fontsize=15)

  # ------------ ax2 ------------ #
  # ------ band ------ #
  ax2.axhline(100, color="#ffffff")
  ax2.axhline(-100, color="#ffffff")

  # ============ ylim ============ # - ax1 only
  if front_plot:
    y_lim_data = a_data[:x_max + 1, col_idx_dict['ylim_col_idxs']]  # +1 for including open_tick
  else:
    y_lim_data = a_data[:, col_idx_dict['ylim_col_idxs']]

  y_min = y_lim_data.min()
  y_max = y_lim_data.max()
  y_margin = (y_max - y_min) * y_margin_mult
  ax1.set_ylim(y_min - y_margin, y_max + y_margin)

  # ============ vline (open_tick, entry_tick, exit_tick) ============ # - add open_tick on ax2
  y0, y1 = ax1.get_ylim()
  l_data = a_data[:exit_tick + 1, col_idx_dict['ohlc_col_idxs'][2]]  # +1 for including exit_tick
  p2_ymax, open_ymax, en_ymax, ex_ymax = [(l_data[tick_] - y0) / (y1 - y0) - .01 for tick_ in
                                              [point1_tick, open_tick, entry_tick, exit_tick]]  # -.05 for margin
  ax1.axvline(point1_tick, 0, p2_ymax, alpha=1, linewidth=2, linestyle='--', color='#ff7722')
  ax1.axvline(open_tick, 0, open_ymax, alpha=1, linewidth=2, linestyle='--', color='#ffeb3b')  # 추후, tick 별 세부 정의가 달라질 수 있음을 고려해 multi_line 작성 유지
  ax2.axvline(open_tick, 0, open_ymax, alpha=1, linewidth=2, linestyle='--', color='#ffeb3b')  # 추후, tick 별 세부 정의가 달라질 수 있음을 고려해 multi_line 작성 유지
  ax1.axvline(entry_tick, 0, en_ymax, alpha=1, linewidth=2, linestyle='--', color='#ffeb3b')
  ax1.axvline(exit_tick, 0, ex_ymax, alpha=1, linewidth=2, linestyle='--', color='#ffeb3b')
  ax1.axvline(bias_info_tick, alpha=1, linewidth=2, linestyle='-', color='#ffeb3b')

  return

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
  plt.axhline(tp_line, 0.05, 1, linewidth=2, linestyle='-', alpha=1, color='#00ff00')  # ep 와 gap 비교 용이하기 위해 ex_xmin -> 0.1 사용
  plt.text(x0, tp_line, 'tpg {}'.format(config.tr_set.tp_gap), ha='right', va='center', fontweight='bold', fontsize=15, color='#00ff00')
  plt.axhline(out_line, 0.05, 1, linewidth=2, linestyle='-', alpha=1, color='#ff0000')
  plt.text(x0, out_line, 'outg {}'.format(config.tr_set.out_gap), ha='right', va='center', fontweight='bold', fontsize=15, color='#ff0000')

  # ------ wave_line ------ #
  text_x_pos = (x0 + x1) * 0.1
  plt.axhline(en_tp1, 0.2, 1, linewidth=2, linestyle='-', alpha=1, color='#ffffff')
  plt.text(text_x_pos, en_tp1, ' wave_1', ha='right', va='bottom', fontweight='bold', fontsize=15)
  plt.axhline(en_out0, 0.2, 1, linewidth=2, linestyle='-', alpha=1, color='#ffffff')
  plt.text(text_x_pos, en_out0, ' wave_0', ha='right', va='bottom', fontweight='bold', fontsize=15)

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


def plot_check_v9(res_df, config, param_zip, pr_msg, x_max, x_margin_mult, y_margin_mult, back_plot, plot_check_dir=None, **col_idx_dict):
  # start_0 = time.time()
  plt.style.use(['dark_background', 'fast'])
  fig = plt.figure(figsize=(30, 18))
  nrows, ncols = 2, 2
  gs = gridspec.GridSpec(nrows=nrows,  # row 부터 index 채우고 col 채우는 순서임 (gs_idx)
                         ncols=ncols,
                         height_ratios=[3, 1]
                         )
  for gs_idx, params in enumerate(param_zip):

    iin, iout, pr, en_p, ex_p, entry_idx, exit_idx, p1_idx, p2_idx, lvrg, fee, tp_line, out_line, tp_1, tp_0, out_1, out_0, ep2_0 = params

    # print("en_p, ex_p :", en_p, ex_p)
    # print("tp_line, out_line, ep2_0 :", tp_line, out_line, ep2_0)

    # temporary
    # if exit_idx - p1_idx < 50:
    # if exit_idx != entry_idx:
    # print("p1_idx :", p1_idx)
    # if p1_idx != 370259:
    #   break

    # ============ define ax1 & ax2 ============ #
    ax1 = fig.add_subplot(gs[gs_idx])
    ax2 = fig.add_subplot(gs[gs_idx + 2])

    # ------ date range ------ #
    if back_plot == 0:
      iout = iin + x_max
      # print("iin, iout :", iin, iout)

    a_data = res_df.iloc[int(iin):int(iout + 1)].to_numpy()
    # a_data = data[iin:iout]

    # ------------ add_col section ------------ #
    # ------ candles ------ #
    candle_plot_v2(ax1, a_data[:, col_idx_dict['ohlc_col_idxs']], alpha=1.0, wickwidth=1.0)

    # ------ add cols ------ #
    [nonstep_col_plot_v2(ax1, a_data[:, params_[0]], *params_[1:]) for params_ in col_idx_dict['nonstep_col_info']]
    [step_col_plot_v2(ax1, a_data[:, params_[0]], *params_[1:]) for params_ in col_idx_dict['step_col_info']]
    [stepmark_col_plot_v2(ax1, a_data[:, params_[0]], *params_[1:]) for params_ in col_idx_dict['stepmark_col_info']]

    [step_col_plot_v2(ax2, a_data[:, params_[0]], *params_[1:]) for params_ in col_idx_dict['step_col_info2']]

    # ------ get vp_info ------ #
    kde_factor = 0.1  # 커질 수록 전체적인 bars_shape 이 곡선이됨, 커질수록 latency 좋아짐 (0.00003s 정도)
    num_samples = 100  # plot 되는 volume bars (y_axis) 와 비례관계
    # vp_data = data[iin - 500:iin, col_idx_dict['vp_col_idxs']].T  # Todo, vp_range should be calculated by wave_point

    if tp_1 < out_0:  # SELL order
      vp_iin, vp_iout = res_df.iloc[int(p1_idx), col_idx_dict['roll_cu_idxs']].to_numpy()
    else:
      vp_iin, vp_iout = res_df.iloc[int(p1_idx), col_idx_dict['roll_co_idxs']].to_numpy()

    vp_data = res_df.iloc[int(vp_iin):int(vp_iout), col_idx_dict['vp_col_idxs']].to_numpy().T  # Todo, vp_range should be calculated by wave_point
    # print("vp_data :", vp_data)
    # vp_info = [vp_range, *vp_data, kde_factor, num_samples]
    vp_info = [*vp_data, kde_factor, num_samples]

    # ------ ep, tp + xlim ------ #
    try:
      eptp_hvline_v9_1(ax1, ax2, config, *params, back_plot, x_max, x_margin_mult, y_margin_mult, a_data, vp_info, **col_idx_dict)
    except Exception as e:
      print("error in eptp_hvline :", e)

    #     Todo    #
    #     3. outer_price plot 일 경우, gs_idx + nrows 하면 됨

    # ------ trade_info ------ #
    data_msg_list = ["\n {} : {:.3f}".format(*params_[1:], *res_df.iloc[int(p1_idx), params_[0]]) for params_ in
                     col_idx_dict['data_window_p1_col_info']]  # * for unsupported format for arr
    data_msg_list += ["\n {} : {:.3f}".format(*params_[1:], *res_df.iloc[int(p2_idx), params_[0]]) for params_ in
                      col_idx_dict['data_window_p2_col_info']]
    ps_msg_expand = pr_msg.format(p1_idx, exit_idx, pr, lvrg, fee) + ''.join(data_msg_list)

    ax1.set_title(ps_msg_expand)  # set_title on ax1

  if plot_check_dir is None:
    plt.show()
    print()
  else:
    fig_name = plot_check_dir + "/{}.png".format(int(entry_idx))
    plt.savefig(fig_name)
    print(fig_name, "saved !")
  plt.close()
  # print("elapsed time :", time.time() - start_0)

  return

def plot_check_v8(res_df, config, param_zip, pr_msg, x_max, x_margin_mult, y_margin_mult, back_plot, plot_check_dir=None, **col_idx_dict):
  # start_0 = time.time()
  plt.style.use(['dark_background', 'fast'])
  fig = plt.figure(figsize=(30, 18))
  nrows, ncols = 2, 2
  gs = gridspec.GridSpec(nrows=nrows,  # row 부터 index 채우고 col 채우는 순서임 (gs_idx)
                         ncols=ncols,
                         height_ratios=[3, 1]
                         )
  for gs_idx, params in enumerate(param_zip):

    iin, iout, pr, en_p, ex_p, entry_idx, exit_idx, p1_idx, p2_idx, lvrg, fee, tp_line, out_line, tp_1, tp_0, out_1, out_0, ep2_0 = params

    # print("en_p, ex_p :", en_p, ex_p)
    # print("tp_line, out_line, ep2_0 :", tp_line, out_line, ep2_0)

    # temporary
    # if exit_idx - p1_idx < 50:
    # if exit_idx != entry_idx:
    # print("p1_idx :", p1_idx)
    # if p1_idx != 370259:
    #   break

    # ============ define ax1 & ax2 ============ #
    ax1 = fig.add_subplot(gs[gs_idx])
    ax2 = fig.add_subplot(gs[gs_idx + 2])

    # ------ date range ------ #
    if back_plot == 0:
      iout = iin + x_max
      # print("iin, iout :", iin, iout)

    a_data = res_df.iloc[int(iin):int(iout + 1)].to_numpy()
    # a_data = data[iin:iout]

    # ------------ add_col section ------------ #
    # ------ candles ------ #
    candle_plot_v2(ax1, a_data[:, col_idx_dict['ohlc_col_idxs']], alpha=1.0, wickwidth=1.0)

    # ------ add cols ------ #
    [nonstep_col_plot_v2(ax1, a_data[:, params_[0]], *params_[1:]) for params_ in col_idx_dict['nonstep_col_info']]
    [step_col_plot_v2(ax1, a_data[:, params_[0]], *params_[1:]) for params_ in col_idx_dict['step_col_info']]
    [stepmark_col_plot_v2(ax1, a_data[:, params_[0]], *params_[1:]) for params_ in col_idx_dict['stepmark_col_info']]

    [step_col_plot_v2(ax2, a_data[:, params_[0]], *params_[1:]) for params_ in col_idx_dict['step_col_info2']]

    # ------ get vp_info ------ #
    kde_factor = 0.05  # 커질 수록 전체적인 bars_shape 이 곡선이됨, 커질수록 latency 좋아짐 (0.00003s 정도)
    num_samples = 100  # plot 되는 volume bars (y_axis) 와 비례관계
    # vp_data = data[iin - 500:iin, col_idx_dict['vp_col_idxs']].T  # Todo, vp_range should be calculated by wave_point
    vp_data = res_df.iloc[int(iin):int(p1_idx), col_idx_dict['vp_col_idxs']].to_numpy().T  # Todo, vp_range should be calculated by wave_point
    # print("vp_data :", vp_data)
    # vp_info = [vp_range, *vp_data, kde_factor, num_samples]
    vp_info = [*vp_data, kde_factor, num_samples]

    # ------ ep, tp + xlim ------ #
    try:
      eptp_hvline_v9_1(ax1, ax2, config, *params, back_plot, x_max, x_margin_mult, y_margin_mult, a_data, vp_info, **col_idx_dict)
    except Exception as e:
      print("error in eptp_hvline :", e)

    #     Todo    #
    #     3. outer_price plot 일 경우, gs_idx + nrows 하면 됨

    # ------ trade_info ------ #
    data_msg_list = ["\n {} : {:.3f}".format(*params_[1:], *res_df.iloc[int(p1_idx), params_[0]]) for params_ in
                     col_idx_dict['data_window_p1_col_info']]  # * for unsupported format for arr
    data_msg_list += ["\n {} : {:.3f}".format(*params_[1:], *res_df.iloc[int(p2_idx), params_[0]]) for params_ in
                      col_idx_dict['data_window_p2_col_info']]
    ps_msg_expand = pr_msg.format(p1_idx, exit_idx, pr, lvrg, fee) + ''.join(data_msg_list)

    ax1.set_title(ps_msg_expand)  # set_title on ax1

  if plot_check_dir is None:
    plt.show()
    print()
  else:
    fig_name = plot_check_dir + "/{}.png".format(int(entry_idx))
    plt.savefig(fig_name)
    print(fig_name, "saved !")
  plt.close()
  # print("elapsed time :", time.time() - start_0)

  return

def plot_check_v7(data, config, param_zip, pr_msg, x_max, x_margin_mult, y_margin_mult, front_plot, plot_check_dir=None, **col_idx_dict):
  # start_0 = time.time()
  plt.style.use(['dark_background', 'fast'])
  fig = plt.figure(figsize=(30, 18))
  nrows, ncols = 2, 2
  gs = gridspec.GridSpec(nrows=nrows,  # row 부터 index 채우고 col 채우는 순서임 (gs_idx)
                         ncols=ncols,
                         height_ratios=[3, 1]
                         )
  for gs_idx, params in enumerate(param_zip):

    iin, iout, pr, ep, tp, entry_idx, exit_idx, open_idx, point1_idx, lvrg, fee, tp_line, out_line, en_tp1, en_out0 = params

    # if exit_idx - open_idx < 50:  # temporary
    #   break

    # ============ define ax1 & ax2 ============ #
    ax1 = fig.add_subplot(gs[gs_idx])
    ax2 = fig.add_subplot(gs[gs_idx + 2])

    # ------ date range ------ #
    a_data = data[int(iin):int(iout + 1)]
    # a_data = data[iin:iout]

    # ------------ add_col section ------------ #
    # ------ candles ------ #
    candle_plot_v2(ax1, a_data[:, col_idx_dict['ohlc_col_idxs']], alpha=1.0, wickwidth=1.0)

    # ------ add cols ------ #
    [nonstep_col_plot_v2(ax1, a_data[:, params_[0]], *params_[1:]) for params_ in col_idx_dict['nonstep_col_info']]
    [step_col_plot_v2(ax1, a_data[:, params_[0]], *params_[1:]) for params_ in col_idx_dict['step_col_info']]
    [stepmark_col_plot_v2(ax1, a_data[:, params_[0]], *params_[1:]) for params_ in col_idx_dict['stepmark_col_info']]

    [step_col_plot_v2(ax2, a_data[:, params_[0]], *params_[1:]) for params_ in col_idx_dict['step_col_info2']]

    # ------ ep, tp + xlim ------ #
    try:
      eptp_hvline_v6(ax1, ax2, config, ep, tp, entry_idx, exit_idx, open_idx, point1_idx, tp_line, out_line, en_tp1, en_out0,
                     front_plot, iin, iout, x_max, x_margin_mult, y_margin_mult, a_data, **col_idx_dict)
    except Exception as e:
      print("error in eptp_hvline_v3 :", e)

    #     Todo    #
    #     3. outer_price plot 일 경우, gs_idx + nrows 하면 됨

    # ------ trade_info ------ #
    data_msg_list = ["\n {} : {:.3f}".format(*params_[1:], *data[int(open_idx), params_[0]]) for params_ in
                     col_idx_dict['data_window_col_info']]  # * for unsupported format for arr
    ps_msg_expand = pr_msg.format(entry_idx, exit_idx, pr, lvrg, fee) + ''.join(data_msg_list)

    ax1.set_title(ps_msg_expand)  # set_title on ax1

  if plot_check_dir is None:
    plt.show()
    print()
  else:
    fig_name = plot_check_dir + "/{}.png".format(int(entry_idx))
    plt.savefig(fig_name)
    print(fig_name, "saved !")
  plt.close()
  # print("elapsed time :", time.time() - start_0)

  return

def plot_check_v6(data, config, param_zip, pr_msg, x_max, x_margin_mult, y_margin_mult, front_plot, plot_check_dir=None, **col_idx_dict):
  # start_0 = time.time()
  plt.style.use(['dark_background', 'fast'])
  fig = plt.figure(figsize=(30, 18))
  nrows, ncols = 2, 2
  gs = gridspec.GridSpec(nrows=nrows,  # row 부터 index 채우고 col 채우는 순서임 (gs_idx)
                         ncols=ncols,
                         height_ratios=[3, 1]
                         )
  for gs_idx, params in enumerate(param_zip):

    iin, iout, pr, ep, tp, entry_idx, exit_idx, open_idx, point1_idx, lvrg, fee, tp_line, out_line, en_tp1, en_out0 = params

    # if exit_idx - open_idx < 50:  # temporary
    #   break

    ax = fig.add_subplot(gs[gs_idx])

    # ------------ add_col section ------------ #
    a_data = data[int(iin):int(iout + 1)]
    # a_data = data[iin:iout]
    # ------ candles ------ #
    candle_plot(a_data[:, col_idx_dict['ohlc_col_idxs']], ax, alpha=1.0, wickwidth=1.0)

    # ------ add cols ------ #
    [nonstep_col_plot(a_data[:, params_[0]], *params_[1:]) for params_ in col_idx_dict['nonstep_col_info']]
    [step_col_plot(a_data[:, params_[0]], *params_[1:]) for params_ in col_idx_dict['step_col_info']]
    [stepmark_col_plot(a_data[:, params_[0]], *params_[1:]) for params_ in col_idx_dict['stepmark_col_info']]

    # ------ ep, tp + xlim ------ #
    try:
      eptp_hvline_v5(config, ep, tp, entry_idx, exit_idx, open_idx, point1_idx, tp_line, out_line, en_tp1, en_out0,
                     front_plot, iin, iout, x_max, x_margin_mult, y_margin_mult, a_data, **col_idx_dict)
    except Exception as e:
      print("error in eptp_hvline_v3 :", e)

    #     Todo    #
    #     3. outer_price plot 일 경우, gs_idx + nrows 하면 됨

    # ------ trade_info ------ #
    data_msg_list = ["\n {} : {:.3f}".format(*params_[1:], *data[int(open_idx), params_[0]]) for params_ in
                     col_idx_dict['data_window_col_info']]  # * for unsupported format for arr
    ps_msg_expand = pr_msg.format(entry_idx, exit_idx, pr, lvrg, fee) + ''.join(data_msg_list)

    plt.title(ps_msg_expand)

  if plot_check_dir is None:
    plt.show()
    print()
  else:
    fig_name = plot_check_dir + "/{}.png".format(int(entry_idx))
    plt.savefig(fig_name)
    print(fig_name, "saved !")
  plt.close()
  # print("elapsed time :", time.time() - start_0)

  return

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
    fig_name = plot_check_dir + "/{}.png".format(int(entry_idx))
    plt.savefig(fig_name)
    print(fig_name, "saved !")
  plt.close()
  # print("elapsed time :", time.time() - start_0)

  return



def candle_plot(ohlc_np, ax, alpha=1.0, wickwidth=1.0):
  index = np.arange(len(ohlc_np))
  candle = np.hstack((np.reshape(index, (-1, 1)), ohlc_np))
  mf.candlestick_ohlc(ax, candle, width=0.5, colorup='#26a69a', colordown='#ef5350', alpha=alpha, wickwidth=wickwidth)


def hcandle_plot(hcandle_np, alpha=1, square=1, color='#ffffff'):
  assert hcandle_np.shape[-1] == 2, "assert hcandle_np.shape[-1] == 2"
  try:
    plt.step(np.arange(len(hcandle_np)), hcandle_np, alpha=alpha, color=color, linewidth=2 ** square)
    # plt.fill_between(np.arange(len(plot_df)), plot_df['hclose_60'].values, plot_df['hopen_60'].values,
    #                     where=1, facecolor='#ffffff', alpha=alpha)
  except Exception as e:
    print("error in hcandle_plot :", e)


