import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
# import mpl_finance as mf
from mplfinance.original_flavor import candlestick_ohlc
from funcs.public.idep import get_col_idxs
from scipy import stats


# from numba import jit


def strcol_tonumb(res_df, col_list):
    step_col_arr = np.array(col_list)
    if len(col_list) != 0:
        step_col_arr[:, 0] = [get_col_idxs(res_df, col_) for col_ in step_col_arr[:, 0]]  # str_col to number

    return step_col_arr


def candle_plot_v2(ax, ohlc_np, alpha=1.0, wickwidth=1.0):
    index = np.arange(len(ohlc_np))
    candle = np.hstack((np.reshape(index, (-1, 1)), ohlc_np))
    # mf.candlestick_ohlc(ax, candle, width=0.5, colorup='#26a69a', colordown='#ef5350', alpha=alpha, wickwidth=wickwidth)
    candlestick_ohlc(ax, candle, width=0.5, colorup='#26a69a', colordown='#ef5350', alpha=alpha)


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


def whole_plot_check(data, win_idxs, selected_op_idxs, selected_ex_idxs, plot_check_dir=None, **col_idx_dict):
    # start_0 = time.time()
    plt.style.use(['dark_background', 'fast'])
    fig = plt.figure(figsize=(30, 12), dpi=75)
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


def eptp_hvline_v9_1(ax1, ax2, config, iin, iout, pr, en_p, ex_p, entry_idx, exit_idx, p1_idx, p2_idx, lvrg, fee, tp_level, out_level, tp_1, tp_0,
                     out_1, out_0, ep2_0, back_plot, x_max, x_margin_mult, y_margin_mult, a_data, vp_info, **col_idx_dict):
    """
    기존 y_lim 에서 tr_set 기준 min & max 를 추가함 (indicator 에 의한 y_lim 을 유지하기 위해 추가하는 방법 사용함.)
    """

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

    # ------ get_xlim ------ #
    if (iout - iin) > x_max:
        x_margin = x_max * x_margin_mult
        ax1.set_xlim(0 - x_margin, x_max + x_margin)
        ax2.set_xlim(0 - x_margin, x_max + x_margin)
    x0, x1 = ax1.get_xlim()

    """ Axis_1 """
    # ------ entry & exit ------ #
    en_xmin = entry_tick / x1
    ex_xmin = exit_tick / x1
    ax1.text(x0, en_p, 'en_p :\n {:.3f}'.format(en_p), ha='right', va='center', fontweight='bold', fontsize=15)  # en_p line label

    ax1.axhline(ex_p, ex_xmin, 1, linewidth=2, linestyle='--', alpha=1, color='lime')  # ex_p line axhline (signal 도 포괄함, 존재 의미)
    ax1.text(x1, ex_p, 'ex_p :\n {}'.format(ex_p), ha='left', va='center', fontweight='bold', fontsize=15)  # ex_p line label

    # ------ tr_set line ------ #
    left_point = 0.1
    right_point = 1
    text_x_pos_left = (x0 + x1) * (left_point + 0.05)

    ax1.axhline(en_p, left_point, right_point, linewidth=2, linestyle='-', alpha=1, color='#005eff')  # en_p line axhline

    if config.tr_set.check_hlm in [0, 1]:
        plot_epg_tuple = ("epg1", config.tr_set.ep_gap1)
    else:
        plot_epg_tuple = ("epg2", config.tr_set.ep_gap2)
    ax1.text(text_x_pos_left, en_p, '{} {}'.format(*plot_epg_tuple), ha='right', va='bottom', fontweight='bold', fontsize=15, color='#005eff')

    ax1.axhline(tp_level, left_point, right_point, linewidth=2, linestyle='-', alpha=1, color='#00ff00')  # ep 와 gap 비교 용이하기 위해 ex_xmin -> 0.1 사용
    ax1.text(text_x_pos_left, tp_level, 'tpg {}'.format(config.tr_set.tp_gap), ha='right', va='bottom', fontweight='bold', fontsize=15, color='#00ff00')

    ax1.axhline(out_level, left_point, right_point, linewidth=2, linestyle='-', alpha=1, color='#ff0000')
    ax1.text(text_x_pos_left, out_level, 'outg {}'.format(config.tr_set.out_gap), ha='right', va='bottom', fontweight='bold', fontsize=15, color='#ff0000')

    left_point = 0.3
    right_point = 1
    text_x_pos_left = (x0 + x1) * (left_point + 0.05)
    text_x_pos_right = (x0 + x1) * right_point

    # ------ tp_box ------ #
    ax1.axhline(tp_1, left_point, right_point, linewidth=1, linestyle='-', alpha=1, color='#ffffff')
    ax1.text(text_x_pos_left, tp_1, ' tp_1', ha='right', va='bottom', fontweight='bold', fontsize=15)
    ax1.axhline(tp_0, left_point, right_point, linewidth=1, linestyle='-', alpha=1, color='#ffffff')
    ax1.text(text_x_pos_left, tp_0, ' tp_0', ha='right', va='bottom', fontweight='bold', fontsize=15)

    # ------ octa_wave_box ------ #
    wave_gap = (tp_1 - tp_0) / 8
    [ax1.axhline(tp_0 + wave_gap * gap_i, left_point, right_point, linewidth=1, linestyle='--', alpha=1, color='#ffffff') for gap_i in range(1, 8)]

    # ------ ep_box ------ #
    ax1.axhline(ep2_0, left_point, right_point, linewidth=1, linestyle='-', alpha=1, color='#ffffff')
    ax1.text(text_x_pos_left, ep2_0, ' ep2_0', ha='right', va='bottom', fontweight='bold', fontsize=15)

    # ------ out_box ------ #
    ax1.axhline(out_1, left_point, right_point, linewidth=1, linestyle='-', alpha=1, color='#ffffff')
    ax1.text(text_x_pos_right, out_1, ' out_1', ha='right', va='bottom', fontweight='bold', fontsize=15)
    ax1.axhline(out_0, left_point, right_point, linewidth=1, linestyle='-', alpha=1, color='#ffffff')
    ax1.text(text_x_pos_right, out_0, ' out_0', ha='right', va='bottom', fontweight='bold', fontsize=15)

    # ------ volume profile ------ #
    if len(vp_info) > 0:
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

    """ Axis_2 """

    # ------ cci_band ------ #
    ax2.axhline(100, color="#ffffff")
    ax2.axhline(0, color="#ffffff")
    ax2.axhline(-100, color="#ffffff")

    # ------ stoch_band ------ #
    # ax2.axhline(67, color="#ffffff")
    # ax2.axhline(33, color="#ffffff")
    # ax2.axhline(0, color="#ffffff")

    # ------ ylim ------ # - ax1 only
    # 1. ylim by tr_set
    y_min = min(tp_level, out_level, tp_1, tp_0, out_1, out_1)
    y_max = max(tp_level, out_level, tp_1, tp_0, out_1, out_1)

    # 2. ylim by indicator
    if len(col_idx_dict['ylim_col_idxs']) != 0:
        if back_plot:
            y_lim_data = a_data[:x_max + 1, col_idx_dict['ylim_col_idxs']]  # +1 for including p1_tick
        else:
            y_lim_data = a_data[:, col_idx_dict['ylim_col_idxs']]

        y_min = min(y_lim_data.min(), y_min)
        y_max = max(y_lim_data.max(), y_max)

    y_margin = (y_max - y_min) * y_margin_mult
    ax1.set_ylim(y_min - y_margin, y_max + y_margin)

    # ------ vline (p1_tick, entry_tick, exit_tick) ------ # - add p1_tick on ax2
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


def plot_check_v9(res_df, config, param_zip, pr_msg, x_max, x_margin_mult, y_margin_mult, back_plot, plot_check_dir=None, **col_idx_dict):
    # start_0 = time.time()
    plt.style.use(['dark_background', 'fast'])
    fig = plt.figure(figsize=(30, 18), dpi=60)
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

        """1. vp by lookback"""
        # vp_lookback = 500
        # vp_data = res_df.iloc[int(p1_idx - 500):int(p1_idx), col_idx_dict['vp_col_idxs']].to_numpy().T

        """2. vp by wave_point"""
        #     if tp_1 < out_0:  # SELL order
        #       post_co_idx = res_df.iloc[int(p1_idx), col_idx_dict['post_co_idx']]
        #       # vp_iin = res_df.iloc[int(p1_idx) - 1, col_idx_dict['post_cu_idx']].to_numpy()  # Todo, co_idx 와 co_post_idx 의 차별을 위해서 -1 해줌 <-- 중요 point
        #       vp_iin = res_df.iloc[post_co_idx, col_idx_dict['post_cu_idx']].to_numpy() # post_co_idx 에 있는 post_cu_idx ?
        #     else:
        #       post_cu_idx = res_df.iloc[int(p1_idx), col_idx_dict['post_cu_idx']]
        #       # vp_iin = res_df.iloc[int(p1_idx) - 1, col_idx_dict['post_co_idx']].to_numpy()
        #       vp_iin = res_df.iloc[int(post_cu_idx), col_idx_dict['post_co_idx']].to_numpy()
        #       # print("post_cu_idx, vp_iin :", post_cu_idx, vp_iin)

        #     vp_data = res_df.iloc[int(vp_iin):int(p1_idx), col_idx_dict['vp_col_idxs']].to_numpy().T   # vp : ~ post_cu / co_idx 까지

        #     vp_info = [*vp_data, kde_factor, num_samples]
        vp_info = []

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


def candle_plot(ohlc_np, ax, alpha=1.0, wickwidth=1.0):
    index = np.arange(len(ohlc_np))
    candle = np.hstack((np.reshape(index, (-1, 1)), ohlc_np))
    candlestick_ohlc(ax, candle, width=0.5, colorup='#26a69a', colordown='#ef5350', alpha=alpha, wickwidth=wickwidth)


def hcandle_plot(hcandle_np, alpha=1, square=1, color='#ffffff'):
    assert hcandle_np.shape[-1] == 2, "assert hcandle_np.shape[-1] == 2"
    try:
        plt.step(np.arange(len(hcandle_np)), hcandle_np, alpha=alpha, color=color, linewidth=2 ** square)
        # plt.fill_between(np.arange(len(plot_df)), plot_df['hclose_60'].values, plot_df['hopen_60'].values,
        #                     where=1, facecolor='#ffffff', alpha=alpha)
    except Exception as e:
        print("error in hcandle_plot :", e)
