import os
pwd = os.getcwd()

# print(os.path.dirname(pwd))
# print(pwd)
switch_path = os.path.dirname(pwd)
os.chdir(switch_path)

from binance_f import RequestClient
from binance_f.model import *
from binance_f.constant.test import *
from binance_f.base.printobject import *

from binance_futures_concat_candlestick import concat_candlestick

import matplotlib.pyplot as plt

from funcs.funcs_trader import *
from funcs.funcs_indicator import *
import mpl_finance as mf

# import numpy as np
# import pandas as pd

pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 2500)
pd.set_option('display.max_columns', 2500)


class LiveChart:

    def __init__(self, plot_size=45):

        self.plot_size = plot_size

    # def init_chart(self):

        self.fig = plt.figure(figsize=(8, 6))
        self.ax = self.fig.add_subplot(111)
        plt.ion()
        self.fig.show()
        self.fig.canvas.draw()

    def update(self, df, second_df):

        # self.df = df
        # self.second_df = second_df

        ha_second_df = heikinashi(second_df)
        # ha_third_df = heikinashi(third_df)
        # print(ha_second_df.tail(10))
        # quit()

        second_df['minor_ST1_Up'], second_df['minor_ST1_Down'], second_df['minor_ST1_Trend'] = supertrend(second_df, 10,
                                                                                                          2)
        second_df['minor_ST2_Up'], second_df['minor_ST2_Down'], second_df['minor_ST2_Trend'] = supertrend(ha_second_df,
                                                                                                          7,
                                                                                                          2)
        second_df['minor_ST3_Up'], second_df['minor_ST3_Down'], second_df['minor_ST3_Trend'] = supertrend(ha_second_df,
                                                                                                          7,
                                                                                                          2.5)
        # print(df.head(20))
        # quit()

        # startTime = time.time()

        df = df.join(pd.DataFrame(index=df.index, data=to_lower_tf(df, second_df, [i for i in range(-9, 0, 1)]),
                                  columns=['minor_ST1_Up', 'minor_ST1_Down', 'minor_ST1_Trend'
                                      , 'minor_ST2_Up', 'minor_ST2_Down', 'minor_ST2_Trend'
                                      , 'minor_ST3_Up', 'minor_ST3_Down', 'minor_ST3_Trend']))

        # print(df[["minor_ST1_Up", "minor_ST2_Up", "minor_ST3_Up"]].tail())
        # min_upper = np.minimum(df["minor_ST1_Up"], df["minor_ST2_Up"], df["minor_ST3_Up"])
        # max_lower = np.maximum(df["minor_ST1_Down"], df["minor_ST2_Down"], df["minor_ST3_Down"])
        min_upper = np.min(df[["minor_ST1_Up", "minor_ST2_Up", "minor_ST3_Up"]], axis=1)
        max_lower = np.max(df[["minor_ST1_Down", "minor_ST2_Down", "minor_ST3_Down"]], axis=1)

        df['middle_line'] = (min_upper + max_lower) / 2

        plot_df = df.iloc[-self.plot_size:, [0, 1, 2, 3, 5, 6, 8, 9, 11, 12, 14]]

        self.ax.clear()

        temp_ohlc = plot_df.values[:, :4]
        index = np.arange(len(temp_ohlc))
        candle = np.hstack((np.reshape(index, (-1, 1)), temp_ohlc))
        mf.candlestick_ohlc(self.ax, candle, width=0.5, colorup='r', colordown='b')

        # print(plot_df.values[:, 4:])
        self.ax.plot(plot_df.values[:, 4:])

        # plt.show()
        self.fig.canvas.draw()


if __name__=="__main__":

    interval = "1m"
    interval2 = "3m"
    symbol = "ETHUSDT"

    initial = True

    live_chart = LiveChart()

    while 1:

        df, _ = concat_candlestick(symbol, interval, days=1)
        second_df, _ = concat_candlestick(symbol, interval2, days=1)

        live_chart.update(df, second_df)


    # print(res_df[["minor_ST1_Up", "minor_ST1_Down"]].tail(50))
    # print(res_df[["minor_ST1_Up", "minor_ST2_Up", "minor_ST3_Up"]].tail(20))
    # print(res_df[["minor_ST1_Down", "minor_ST2_Down", "minor_ST3_Down"]].tail(20))

