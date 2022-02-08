import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import mpl_finance as mf
from numba import jit


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
  #     0. short_pr 의 sorting
  #     2. norm_range_col - decided by a_data's col
  ep_tick = eptp_hvline(ep, tp, entry_idx, exit_idx, prev_plotsize, x_max, x_margin)

  if front_plot:
    x_max = ep_tick
  if (iout - iin) > x_max:
    plt.xlim(0 - x_margin, x_max + x_margin)

  # ------ check pr ------ #
  # if not config.lvrg_set.static_lvrg:
  plt.title(pr_msg.format(entry_idx, exit_idx, pr, lvrg, fee))
  plt.show()
  plt.close()


def candle_plot(ohlc_np, ax):
  index = np.arange(len(ohlc_np))
  candle = np.hstack((np.reshape(index, (-1, 1)), ohlc_np))
  mf.candlestick_ohlc(ax, candle, width=0.5, colorup='#26a69a', colordown='#ef5350', alpha=0.5)


@jit
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


