# import os
# from basic_v1.back_pr.back_pr_strat.highlow_cloudlb import back_pr_check
# import pandas as pd
# import pickle
import matplotlib.pyplot as plt
from strat.basic_v1.utils_v2 import *
from matplotlib import gridspec
import mpl_finance as mf


# # inversion = True
# inversion = False
#
# if inversion:
#     plot_pr_list = rev_np_pr
# else:
#     plot_pr_list = np_pr

basic_list = ['open', 'high', 'low', 'close', 'minor_ST1_Up', 'minor_ST1_Down',
              'minor_ST2_Up', 'minor_ST2_Down', 'minor_ST3_Up', 'minor_ST3_Down',
              'middle_line', 'min_upper', 'max_lower']
# senkoua_list = ['senkou_a1', 'senkou_a2', 'senkou_a3', 'senkou_a4', 'senkou_a5']
# senkoub_list = ['senkou_b1', 'senkou_b2', 'senkou_b3', 'senkou_b4', 'senkou_b5']
senkoua_list = ['senkou_a1']
senkoub_list = ['senkou_b1']
# sar_list = ['sar1', 'sar2', 'sar3', 'sar4', 'sar5']
# macd_list = ['macd_hist1', 'macd_hist2', 'macd_hist3', 'macd_hist4', 'macd_hist5']
# trix_list = ['trix1', 'trix2', 'trix3', 'trix4', 'trix5']

input_colname = basic_list + senkoua_list + senkoub_list
# input_colname = basic_list + senkoua_list + senkoub_list + sar_list + macd_list + trix_list
# input_cols = basic_cols + sar_cols + ichimoku_cols + macd_cols


def plot_check(res_df, i, j, prev_plotsize=50):

    gap = 0.00005

    # plot_df = res_df.iloc[i - prev_plotsize:j + 1, input_cols]
    plot_df = res_df.iloc[i - prev_plotsize:j + 1][input_colname]

    #       keep off-color st with another variable         #
    st_trend_plot_df = res_df.iloc[i - prev_plotsize:j + 1, [7, 10, 13]]

    # y_max = np.max(plot_df.iloc[:, [4, 6, 8]])
    # y_min = np.min(plot_df.iloc[:, [5, 7, 9]])
    # print("y_max, y_min :", y_max, y_min)

    y_max = max(np.max(plot_df.iloc[:, [4, 6, 8]]))
    y_min = min(np.min(plot_df.iloc[:, [5, 7, 9]]))
    # print("y_max, y_min :", y_max, y_min)
    # break

    plot_df["off_color_upper_st1"] = np.where(st_trend_plot_df.iloc[:, [0]] == 1, plot_df.iloc[:, [4]], np.nan)
    plot_df["off_color_upper_st2"] = np.where(st_trend_plot_df.iloc[:, [1]] == 1, plot_df.iloc[:, [6]], np.nan)
    plot_df["off_color_upper_st3"] = np.where(st_trend_plot_df.iloc[:, [2]] == 1, plot_df.iloc[:, [8]], np.nan)
    plot_df["off_color_lower_st1"] = np.where(st_trend_plot_df.iloc[:, [0]] == -1, plot_df.iloc[:, [5]], np.nan)
    plot_df["off_color_lower_st2"] = np.where(st_trend_plot_df.iloc[:, [1]] == -1, plot_df.iloc[:, [7]], np.nan)
    plot_df["off_color_lower_st3"] = np.where(st_trend_plot_df.iloc[:, [2]] == -1, plot_df.iloc[:, [9]], np.nan)

    #       replace st values with np.nan, using st trend     #
    plot_df.iloc[:, [4]] = np.where(st_trend_plot_df.iloc[:, [0]] == 1, np.nan, plot_df.iloc[:, [4]])
    plot_df.iloc[:, [6]] = np.where(st_trend_plot_df.iloc[:, [1]] == 1, np.nan, plot_df.iloc[:, [6]])
    plot_df.iloc[:, [8]] = np.where(st_trend_plot_df.iloc[:, [2]] == 1, np.nan, plot_df.iloc[:, [8]])
    plot_df.iloc[:, [5]] = np.where(st_trend_plot_df.iloc[:, [0]] == -1, np.nan, plot_df.iloc[:, [5]])
    plot_df.iloc[:, [7]] = np.where(st_trend_plot_df.iloc[:, [1]] == -1, np.nan, plot_df.iloc[:, [7]])
    plot_df.iloc[:, [9]] = np.where(st_trend_plot_df.iloc[:, [2]] == -1, np.nan, plot_df.iloc[:, [9]])

    short_tp = res_df['middle_line'] * (1 + gap)
    long_tp = res_df['middle_line'] * (1 - gap)

    upper_ep = res_df['min_upper'] * (1 - gap)
    lower_ep = res_df['max_lower'] * (1 + gap)

    plot_upper_ep = upper_ep.iloc[i - prev_plotsize:j + 1]
    plot_lower_ep = lower_ep.iloc[i - prev_plotsize:j + 1]

    plot_upper_middle = (plot_df['middle_line'] + plot_df['min_upper']) / 2
    plot_lower_middle = (plot_df['middle_line'] + plot_df['max_lower']) / 2

    plot_short_tp = short_tp.iloc[i - prev_plotsize:j + 1]
    plot_long_tp = long_tp.iloc[i - prev_plotsize:j + 1]

    # fig = plt.figure(figsize=(12, 16))
    fig = plt.figure(figsize=(12, 16))

    gs = gridspec.GridSpec(nrows=3,  # row 몇 개
                           ncols=1,  # col 몇 개
                           height_ratios=[3, 1, 1]
                           )

    # fig = plt.figure(figsize=(8, 12))
    # ax = fig.add_subplot(111)
    # ax = fig.add_subplot(311)
    ax = fig.add_subplot(gs[0])

    # fig.show()
    # fig.canvas.draw()

    temp_ohlc = plot_df.values[:, :4]
    index = np.arange(len(temp_ohlc))
    candle = np.hstack((np.reshape(index, (-1, 1)), temp_ohlc))
    mf.candlestick_ohlc(ax, candle, width=0.5, colorup='r', colordown='b')

    # print(plot_df.values[:, 4:])
    plt.step(plot_df.values[:, [4, 6, 8]], 'r', alpha=1)  # upper on color
    # plt.plot(plot_df.values[:, [4, 6, 8]], 'r', alpha=1)  # upper on color
    plt.step(plot_df.values[:, [5, 7, 9]], 'b', alpha=1)  # lower on color
    plt.step(plot_df.values[:, [10]], 'fuchsia', alpha=1)  # middle

    plt.step(plot_df.values[:, -6:-3], 'r', alpha=1, linestyle=':')  # upper off color
    plt.step(plot_df.values[:, -3:], 'b', alpha=1, linestyle=':')  # lower off color

    plt.step(np.arange(len(plot_df)), plot_upper_ep.values, alpha=1, linestyle='--', color='y')  # ep
    plt.step(np.arange(len(plot_df)), plot_lower_ep.values, alpha=1, linestyle='--', color='y')  # ep

    plt.step(np.arange(len(plot_df)), plot_upper_middle.values, alpha=1, linestyle='--', color='g')  # middle
    plt.step(np.arange(len(plot_df)), plot_lower_middle.values, alpha=1, linestyle='--', color='g')  # middle

    plt.step(np.arange(len(plot_df)), plot_short_tp.values, alpha=1, linestyle=':', color='y')  # tp
    plt.step(np.arange(len(plot_df)), plot_long_tp.values, alpha=1, linestyle=':', color='y')  # tp

    # ---------------------- indicator part ---------------------- #

    # alpha = 1
    # markersize = 5
    # for sar in sar_list:
    #     plt.step(plot_df[sar].values, 'c*', alpha=alpha, markersize=markersize)  # sar mic
    #     markersize += 1
    #     alpha -= 0.1

    # plt.step(plot_df.values[:, [12]], 'co', alpha=1, markersize=7)  # sar mac

    alpha = 0.7
    for senkoua, senkoub in zip(senkoua_list, senkoub_list):
        plt.fill_between(np.arange(len(plot_df)), plot_df[senkoua].values, plot_df[senkoub].values,  # ichimoku
                         where=plot_df[senkoua].values >= plot_df[senkoub].values, facecolor='g',
                         alpha=alpha)  # ichimoku
        plt.fill_between(np.arange(len(plot_df)), plot_df[senkoua].values, plot_df[senkoub].values,
                         where=plot_df[senkoua].values <= plot_df[senkoub].values, facecolor='r', alpha=alpha)
        alpha -= 0.05

    # ------------------------------------------------------------- #

    plt.axvline(prev_plotsize, linestyle='--')
    plt.axhline(res_df['back_ep'].iloc[i], linestyle='-', xmin=0.75, xmax=1, linewidth=3)  # ep line axhline
    plt.axhline(res_df['back_tp'].iloc[j], linestyle='-', xmin=0.9, xmax=1, linewidth=3)  # tp line axhline
    # plt.title("%s ~ %s -> %.5f\n %s" % (i, j, plot_pr_list[t_i], tp_state_list[t_i]))

    #           y lim         #
    plt.ylim(y_min, y_max)

    # #           macd          #
    # # plt.subplot(312)
    # plt.subplot(gs[1])
    # alpha = 1
    # for macd in macd_list:
    #   plt.step(np.arange(len(plot_df)), plot_df[macd].values, 'g', alpha=alpha)
    #   # plt.fill_between(np.arange(len(plot_df)), 0, plot_df[macd].values, facecolor='g', alpha=alpha)
    #   alpha -= 0.2
    # plt.axvline(prev_plotsize, linestyle='--')
    # plt.axhline(0, linestyle='--')

    # #           trix          #
    # # plt.subplot(313)
    # plt.subplot(gs[2])
    # alpha = 1
    # for trix in trix_list:
    #   plt.step(np.arange(len(plot_df)), plot_df[trix].values, 'g', alpha=alpha)
    #   # plt.fill_between(np.arange(len(plot_df)), 0, plot_df[macd].values, facecolor='g', alpha=alpha)
    #   alpha -= 0.2
    # plt.axvline(prev_plotsize, linestyle='--')
    # plt.axhline(0, linestyle='--')

    # ---------------------- plot ---------------------- #

    plt.show()
    # plt.draw()
    plt.close()
    print()

    # break




