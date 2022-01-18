import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec


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
