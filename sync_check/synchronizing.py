# pwd = os.getcwd()
# # print(os.path.dirname(pwd))
# # print(pwd)
# switch_path = os.path.dirname(pwd)
# os.chdir(switch_path)

# from binance_f import RequestClient
# from binance_f.model import *
# from binance_f.base.printobject import *
from funcs_binance.binance_futures_concat_candlestick_ftr import concat_candlestick
import matplotlib.pyplot as plt
from funcs.funcs_trader import *
# from funcs.olds.funcs_indicator_candlescore import *
from funcs.funcs_indicator import *
import mpl_finance as mf

# import numpy as np
# import pandas as pd

pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 2500)
pd.set_option('display.max_columns', 2500)


def sync_check(df, plot_size=45, plotting=False):


    df_5T = to_htf(df, '5T', '1h')
    df_5T['sar_5T'] = talib.SAREXT(df_5T.high.to_numpy(), df_5T.low.to_numpy(),
                               startvalue=0.0, offsetonreverse=0,
                               accelerationinitlong=0.02, accelerationlong=0.02, accelerationmaxlong=0.2,
                               accelerationinitshort=0.02, accelerationshort=0.02, accelerationmaxshort=0.2)
    # df_5T['sar_5T'] = talib.SAR(df_5T.high.to_numpy(), df_5T.low.to_numpy(), acceleration=0.02)
    df = df.join(to_lower_tf_v2(df, df_5T, [-1]), how='inner')
    # print(df_5T.sar_5T.tail(40))
    # print(df_5T.tail(40))
    print(df.tail(40))
    # print(df.sar_5T.tail(40))
    quit()

    # df['sar_T'] = talib.SAREXT(df.high.to_numpy(), df.low.to_numpy(),
    #                                startvalue=0.02, offsetonreverse=0,
    #                                accelerationinitlong=0.02, accelerationlong=0.02, accelerationmaxlong=0.2,
    #                                accelerationinitshort=0.02, accelerationshort=0.02, accelerationmaxshort=0.2)
    # print(df.sar_T.tail(40))
    # quit()

    df = stoch_v2(df)
    print(df.tail(40))
    quit()

    # ------ cci ------ #
    df = macd_hist(df)
    print(df.tail(40))
    quit()

    df = cci_v2(df, 20)
    print(df.cci_T20.tail(40))
    quit()

    # ------ lucid sar ------ #
    df_3T = to_htf(df, itv_='3T', offset='1h')
    start_0 = time.time()
    lucid_sar(df_3T)        # 0.12494134902954102s
    print(time.time() - start_0)
    start_0 = time.time()
    lucid_sar_v2(df_3T)     # 0.0
    print(time.time() - start_0)

    df = df.join(to_lower_tf_v2(df, df_3T, [-2, -1]), how='inner')
    print(df.tail())
    quit()

    # ------ ha ------ #
    # heikinashi_v2(df)
    # print(df.tail())
    # quit()

    # ------ stdev ------ #
    # df['stdev'] = stdev(df, 20)
    # print(df.stdev.tail(10))
    # quit()

    # ------ atr ------ #
    # df['atr'] = atr(df, 20)
    # print(df.stdev.tail(10))
    # quit()

    # df = dc_line(df, None, '1m', dc_period=20)
    # df = dc_line(df, third_df, '5m')
    # df = dc_line(df, fourth_df, '15m')
    #
    # df = bb_line(df, None, '1m')
    # # df = bb_line(df, third_df, '5m')
    # # df = bb_line(df, fourth_df, '15m')
    # # df = bb_line(df, fourth_df, '30m')
    # print(df.tail(5)) # 20:15:59.999  3321.18  3321.98  3320.99  3321.74  580.510  3322.939546  3318.316454
    # quit()
    res_df = df
    ticker_prcn = get_precision_by_price(res_df.close.iloc[-1]) + 1

    # slice_len_list = range(500, 5000, 100)
    slice_len_list = range(500, 10000, 100)

    start_0 = time.time()
    prev_int_, prev_pnts_ = None, None
    offset = 1

    for sample_len in slice_len_list:

        sample_df = res_df.iloc[-sample_len - offset:-offset]

        # ---------- input using indi.s ---------- #
        # res = ema_v0(sample_df['close'], 190)
        # res = rsi(sample_df, 190)

        df_5T = to_htf(sample_df, itv_='5T', offset='1h')
        sample_len2 = len(df_5T)
        res = ema(df_5T['close'], 195)

        res_last_row = res.iloc[-1]
        if pd.isnull(res_last_row):
            continue

        # sample_df = sample_df.join(to_lower_tf_v2(sample_df, df_5T, [-1]), how='inner')

        #   자리수 분할 계산    #
        int_, points_ = str(res_last_row).split('.')
        pnts_ = points_[:ticker_prcn]

        if prev_int_ == int_ and prev_pnts_ == pnts_:
            # print(sample_len, "({})".format(sample_len2), '->', int_, pnts_, end='\n\n')
            print("prev {} ({}) -> {} {}".format(sample_len, sample_len2, prev_int_, prev_pnts_))
            print("{} ({}) -> {} {}\n".format(sample_len, sample_len2, int_, points_))
            break
        else:
            prev_int_ = int_
            prev_pnts_ = pnts_
            # print("not enough slice_len for this indi.")

    print(time.time() - start_0)  # (1301)(1361)(1301)

    quit()

    start_0 = time.time()
    #
    # print(df.index[0])
    # df['ema_1m'] = ema(df['close'], 190)  # 11:20:59.999    3105.043618 11:20:59.999    3159.031942 --> days=1 로는 190 period 를 감당하기 어렵다는 이야기로 보임
    # # 11:20:59.999    3161.472017 11:20:59.999    3161.472018
    # # print(df['ema_1m'].tail(30))11:30:59.999
    # # print(df['ema_1m'].head(5))
    # print("{:.3f}".format(df['ema_1m'].loc[pd.to_datetime("2022-01-19 11:30:59.999")]))
    # quit()

    df_5T = to_htf(df, itv_='5T', offset='1h')
    print("len(df) :", len(df))
    print("len(df_5T) :", len(df_5T))
    df_5T['ema_5m'] = ema(df_5T['close'], 190)
    # df_5T['ema_5m'] = sma(df_5T['close'], 190)
    print(df_5T['ema_5m'].tail(5))
    # print(time.time() - start_0)
    quit()

    # print(df['ema_5m'].tail(20))
    # # print(df.tail(5))
    # # print(df['ema_5m'].loc["2022-01-13 20:19:59.999"])
    # quit()

    # ------ bb ------ #
    # df = bb_line(df, None, '1m')
    # df = bb_level(df, '1m', 1)
    # print(df.iloc[:, -6:].tail(40))
    # quit()

    # # ------ dc ------ #
    # df = dc_line(df, None, '1m')
    # df = dc_level(df, '1m', 1)
    # print(df.iloc[:, -8:].tail(40))
    # quit()

    # ------ rsi ------ #
    df['rsi'] = rma(df['close'], 14)    # 11:21:59.999    3166.373131 11:21:59.999    3166.373131 11:21:59.999    3166.373131
    # df['rsi'] = rsi(df, 14)
    print(df.rsi.tail(40))
    quit()


    # ------ cloud bline ------ #
    # df['cloud_bline'] = cloud_bline(df, 26)
    # third_df['cloud_bline_5m'] = cloud_bline(third_df, 26)
    # df = df.join(pd.DataFrame(index=df.index, data=to_lower_tf(df, third_df, [-1]), columns=['cloud_bline_5m']))
    # print(df.tail(200))
    # quit()
    #
    # #       normal st      #
    # df['st'] = supertrend(df, 5, 6, cal_st=True)
    # print(df.tail(40))
    # quit()

    #           supertrend          #
    df = st_price_line(df, third_df, '5m')
    df = st_level(df, '5m', 0.5)
    print(df.iloc[:, -6:].tail(60))
    quit()

    # #           ichimoku            #
    # # df['senkou_a'], df['senkou_b'] = ichimoku(df)
    #
    # second_df['senkou_a'], second_df['senkou_b'] = ichimoku(second_df)
    # df = df.join( pd.DataFrame(index=df.index, data=to_lower_tf(df, second_df, [-2, -1]), columns=['senkou_a', 'senkou_b']))
    #
    # # third_df['senkou_a'], third_df['senkou_b'] = ichimoku(third_df)
    # # df = df.join( pd.DataFrame(index=df.index, data=to_lower_tf(df, third_df, [-2, -1]), columns=['senkou_a', 'senkou_b']))
    #
    # # fourth_df['senkou_a'], fourth_df['senkou_b'] = ichimoku(fourth_df)
    # # df = df.join(pd.DataFrame(index=df.index, data=to_lower_tf(df, fourth_df, [-2, -1]), columns=['senkou_a', 'senkou_b']))
    #
    # #           1-2. displacement           #
    # # df['senkou_a'] = df['senkou_a'].shift(26 - 1)
    # # df['senkou_b'] = df['senkou_b'].shift(26 - 1)
    #
    # df.iloc[:, -2:] = df.iloc[:, -2:].shift(26 - 1)
    #
    # #           macd            #
    # second_df['macd_hist'] = macd(second_df)
    # df = df.join(pd.DataFrame(index=df.index, data=to_lower_tf(df, second_df, [-1]), columns=['macd_hist']))
    #
    # # fourth_df['macd_hist'] = macd(fourth_df)
    # # df = df.join(pd.DataFrame(index=df.index, data=to_lower_tf(df, fourth_df, [-1]), columns=['macd_hist']))
    #
    # # print(df['macd_hist'].tail(20))
    # # quit()
    #
    # #          trix         #
    # df['trix'] = trix_hist(df, 14, 1, 5)
    # # print(df['trix'].tail(15))
    # # quit()

    # -------------- ema_roc -------------- #
    df['ema_roc'] = ema_roc(df['close'], 13, 9)

    print(df.iloc[:, -3:].tail(20))
    quit()

    #          stochastic           #
    df['stoch'] = stoch(df)

    #          fisher           #
    df['fisher'] = fisher(df, 30)

    #          cctbbo           #
    df['cctbbo'], _ = cct_bbo(df, 21, 13)

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

    symbol = "ETHUSDT"

    df, _ = concat_candlestick(symbol, '1m', days=1, limit=500, show_process=True)    # limit(400) 05:01:59.999000    days(1) 14:35:00    2638.234157
    #  days(3) 16:55:00    3125.530734 days(4) 16:55:00    3126.399747 days(5) 16:55:00    3126.441473 days(6) 16:55:00    3126.443448
    #  days(7) 16:55:00    3126.443546

    #   days(1) 11:30:59.999    3159.573597  <-> days(2) len_data : 2156 11:30:59.999    3161.769875796363 days(3) len_data : 3596 11:30:59.999    3161.769876393427694
    #   days(4) 11:30:59.999    3161.769876393427694     11:30:59.999    3161.769876  days(1) len_data : 715 11:30:59.999    3159.573597
    #   days(1) len_data : 500 11:30:59.999    3140.579791  len_data : 900 3161.4217347511167

    res_df = sync_check(df, plotting=True, plot_size=300)

