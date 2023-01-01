import os
pwd = os.getcwd()

# print(os.path.dirname(pwd))
# print(pwd)
switch_path = os.path.dirname(pwd)
os.chdir(switch_path)

from binance_futures_concat_candlestick import concat_candlestick

import matplotlib.pyplot as plt

from funcs.olds.funcs_indicator_candlescore import *
import mpl_finance as mf

# from tqdm.notebook import tqdm

# import numpy as np
# import pandas as pd

pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 2500)
pd.set_option('display.max_columns', 2500)


def sync_check(df, second_df, third_df, fourth_df, plot_size=45, plotting=False):

    #           supertrend          #
    ha_second_df = heikinashi(second_df)
    # ha_third_df = heikinashi(third_df)
    # print(ha_second_df.tail(10))
    # quit()

    second_df['minor_ST1_Up'], second_df['minor_ST1_Down'], second_df['minor_ST1_Trend'] = supertrend(second_df, 10, 2)
    # second_df['minor_ST1_Up'], second_df['minor_ST1_Down'], second_df['minor_ST1_Trend'] = supertrend(ha_second_df, 10, 2)
    second_df['minor_ST2_Up'], second_df['minor_ST2_Down'], second_df['minor_ST2_Trend'] = supertrend(ha_second_df, 7, 2)
    # second_df['minor_ST2_Up'], second_df['minor_ST2_Down'], second_df['minor_ST2_Trend'] = supertrend(second_df, 7, 2)
    second_df['minor_ST3_Up'], second_df['minor_ST3_Down'], second_df['minor_ST3_Trend'] = supertrend(ha_second_df, 7,
                                                                                                      2.5)
    # print(df.head(20))
    # quit()o

    # startTime = time.time()

    df = df.join(pd.DataFrame(index=df.index, data=to_lower_tf(df, second_df, [i for i in range(-9, 0, 1)], backing_i=-2),
                              columns=['minor_ST1_Up', 'minor_ST1_Down', 'minor_ST1_Trend'
                                  , 'minor_ST2_Up', 'minor_ST2_Down', 'minor_ST2_Trend'
                                  , 'minor_ST3_Up', 'minor_ST3_Down', 'minor_ST3_Trend']))

    df.iloc[:, -9:] = df.iloc[:, -9:].shift(-3)

    # print(df.iloc[-200:, -9:])
    # quit()


    # print(df[["minor_ST1_Up", "minor_ST2_Up", "minor_ST3_Up"]].tail())
    # min_upper = np.minimum(df["minor_ST1_Up"], df["minor_ST2_Up"], df["minor_ST3_Up"])
    # max_lower = np.maximum(df["minor_ST1_Down"], df["minor_ST2_Down"], df["minor_ST3_Down"])
    min_upper = np.min(df[["minor_ST1_Up", "minor_ST2_Up", "minor_ST3_Up"]], axis=1)
    max_lower = np.max(df[["minor_ST1_Down", "minor_ST2_Down", "minor_ST3_Down"]], axis=1)

    df['middle_line'] = (min_upper + max_lower) / 2

    #           lucid sar              #
    second_df['sar'] = lucid_sar(second_df)
    df = df.join(pd.DataFrame(index=df.index, data=to_lower_tf(df, second_df, [-1]), columns=['sar1']))

    # third_df['sar'] = lucid_sar(third_df)
    # df = df.join(pd.DataFrame(index=df.index, data=to_lower_tf(df, third_df, [-1]), columns=['sar2']))

    fourth_df['sar'] = lucid_sar(fourth_df)
    df = df.join(pd.DataFrame(index=df.index, data=to_lower_tf(df, fourth_df, [-1]), columns=['sar2']))

    # print(df[['sar1', 'sar2']].tail(20))
    # quit()

    #           ichimoku            #
    # df['senkou_a'], df['senkou_b'] = ichimoku(df)

    second_df['senkou_a'], second_df['senkou_b'] = ichimoku(second_df)
    df = df.join( pd.DataFrame(index=df.index, data=to_lower_tf(df, second_df, [-2, -1]), columns=['senkou_a', 'senkou_b']))

    # third_df['senkou_a'], third_df['senkou_b'] = ichimoku(third_df)
    # df = df.join( pd.DataFrame(index=df.index, data=to_lower_tf(df, third_df, [-2, -1]), columns=['senkou_a', 'senkou_b']))

    # fourth_df['senkou_a'], fourth_df['senkou_b'] = ichimoku(fourth_df)
    # df = df.join(pd.DataFrame(index=df.index, data=to_lower_tf(df, fourth_df, [-2, -1]), columns=['senkou_a', 'senkou_b']))

    #           1-2. displacement           #
    # df['senkou_a'] = df['senkou_a'].shift(26 - 1)
    # df['senkou_b'] = df['senkou_b'].shift(26 - 1)

    df.iloc[:, -2:] = df.iloc[:, -2:].shift(26 - 1)

    #           macd            #
    second_df['macd_hist'] = macd(second_df)
    df = df.join(pd.DataFrame(index=df.index, data=to_lower_tf(df, second_df, [-1]), columns=['macd_hist']))

    # fourth_df['macd_hist'] = macd(fourth_df)
    # df = df.join(pd.DataFrame(index=df.index, data=to_lower_tf(df, fourth_df, [-1]), columns=['macd_hist']))

    # print(df['macd_hist'].tail(20))
    # quit()

    #          stochastic           #
    df['stoch'] = stoch(df)

    #          fisher           #
    df['fisher'] = fisher(df, 30)

    #          cctbbo           #
    df['cctbbo'] = cct_bbo(df, 21, 13)

    print(df.iloc[:, -3:].tail(20))
    quit()

    if plotting:

        plot_df = df.iloc[-plot_size:, [0, 1, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 16, 17, 18, 19]]

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)

        fig.show()
        fig.canvas.draw()

        temp_ohlc = plot_df.values[:, :4]
        index = np.arange(len(temp_ohlc))
        candle = np.hstack((np.reshape(index, (-1, 1)), temp_ohlc))
        mf.candlestick_ohlc(ax, candle, width=0.5, colorup='r', colordown='b')

        # print(plot_df.values[:, 4:])
        plt.plot(plot_df.values[:, [4, 6, 8]], 'r', alpha=1)  # upper
        plt.plot(plot_df.values[:, [5, 7, 9]], 'b', alpha=1)  # lower
        plt.plot(plot_df.values[:, [10]], 'g', alpha=1)  # middle

        plt.plot(plot_df.values[:, [11]], 'c*', alpha=1, markersize=5)  # sar mic
        plt.plot(plot_df.values[:, [12]], 'co', alpha=1, markersize=7)  # sar mac

        # plt.plot(plot_df.values[:, [13]], 'c', alpha=1)  # senkou a
        # plt.plot(plot_df.values[:, [14]], 'fuchsia', alpha=1)  # senkou b

        plt.fill_between(np.arange(len(plot_df)), plot_df.values[:, 13], plot_df.values[:, 14],
                         where=plot_df.values[:, 13] >= plot_df.values[:, 14], facecolor='g', alpha=0.5)
        plt.fill_between(np.arange(len(plot_df)), plot_df.values[:, 13], plot_df.values[:, 14],
                         where=plot_df.values[:, 13] <= plot_df.values[:, 14], facecolor='r', alpha=0.5)

        plt.show()
        # plt.draw()

        #       plot second plots       #
        plt.plot(plot_df.values[:, [15]], 'g', alpha=1)  # middle
        plt.axhline(0)
        plt.show()

        plt.close()

        # plt.pause(1e-3)

    return df


if __name__=="__main__":

    interval = "1m"
    interval2 = "3m"
    interval3 = "5m"
    interval4 = "15m"
    symbol = "ETHUSDT"
    # symbol = "NEOUSDT"

    # initial = True
    # while 1:

    df_, _ = concat_candlestick(symbol, interval, days=1)

    sliding_window = 200

    realtime_second_df = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
    for s_i in (range(sliding_window, len(df_))):

        df = df_.iloc[s_i + 1 - sliding_window:s_i + 1]
        second_df = to_higher_candlestick_v2(df, 3)
        # print(second_df.tail())

        realtime_second_df = realtime_second_df.append(second_df.iloc[-1])
        # print(realtime_second_df.tail())
        # quit()

    print(realtime_second_df.tail())
    # print(second_df.tail())
    quit()

    #   second df 를 이용해 마지막 행의 htf indicator value 만 추출한다   #

    third_df = to_higher_candlestick(df, 5)
    fourth_df = to_higher_candlestick(df, 15)
    # fifth_df = to_higher_candlestick(df, 30)

    res_df = sync_check(df, second_df, third_df, fourth_df, plotting=True, plot_size=100)
    # res_df = sync_check(df.iloc[-200:], second_df, third_df, fourth_df, plotting=True, plot_size=300)


    # print(res_df[["minor_ST1_Up", "minor_ST1_Down"]].tail(50))
    # print(res_df[["minor_ST1_Up", "minor_ST2_Up", "minor_ST3_Up"]].tail(20))
    # print(res_df[["minor_ST1_Down", "minor_ST2_Down", "minor_ST3_Down"]].tail(20))

