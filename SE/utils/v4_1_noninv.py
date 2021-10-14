# import numpy as np
# import pandas as pd
# from binance_futures_concat_candlestick import concat_candlestick
# import matplotlib.pyplot as plt
# from plotly import subplots
# import plotly.offline as offline
# import plotly.graph_objs as go
# import pickle
from funcs.funcs_indicator import *
from funcs.funcs_for_trade import *

pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 2500)
pd.set_option('display.max_columns', 2500)


def sync_check(df, second_df, third_df=None, fourth_df=None, fifth_df=None, show_msg=False,
               sar_on=False, cloud_on=False, macd_on=False, trix_on=False):

    # --------------- st level --------------- #
    # ha_second_df = heikinashi(second_df)
    # # ha_third_df = heikinashi(third_df)
    # # print(ha_second_df.tail(10))
    # # quit()
    #
    # second_df['ST1_Up_5m'], second_df['ST1_Down_5m'], second_df['ST1_Trend_5m'] = supertrend(second_df, 10, 2)
    # second_df['ST2_Up_5m'], second_df['ST2_Down_5m'], second_df['ST2_Trend_5m'] = supertrend(ha_second_df, 7,
    #                                                                                                   2)
    # second_df['ST3_Up_5m'], second_df['ST3_Down_5m'], second_df['ST3_Trend_5m'] = supertrend(ha_second_df, 7,
    #                                                                                                   2.5)
    #
    # df = df.join(pd.DataFrame(index=df.index, data=to_lower_tf(df, second_df, [i for i in range(-9, 0, 1)]),
    #                           columns=['ST1_Up_5m', 'ST1_Down_5m', 'ST1_Trend_5m'
    #                               , 'ST2_Up_5m', 'ST2_Down_5m', 'ST2_Trend_5m'
    #                               , 'ST3_Up_5m', 'ST3_Down_5m', 'ST3_Trend_5m']))

    df = st_price_line(df, third_df, '5m')
    df = st_level(df, '5m', 0.5)

    # --------------- cloud bline --------------- #
    second_df['cloud_bline_3m'] = cloud_bline(second_df, 26)
    df = df.join(pd.DataFrame(index=df.index, data=to_lower_tf(df, second_df, [-1]), columns=['cloud_bline_3m']))

    return df


def enlist_epouttp(res_df, config):

    entry = np.where((res_df['close'].shift(1) >= res_df['st_lower_5m']) &
                     (res_df['close'] < res_df['st_lower_5m'])
                     , -1, 0)

    entry = np.where((res_df['close'].shift(1) <= res_df['st_upper_5m']) &
                     (res_df['close'] > res_df['st_upper_5m'])
                     , 1, entry)

    # ----- market entry ----- #
    short_ep = res_df['close']
    long_ep = res_df['close']

    # --------------- st level consolidation out --------------- #
    short_out = res_df['st_upper2_5m'] + res_df['st_gap_5m'] * config.out_set.out_gap
    long_out = res_df['st_lower2_5m'] - res_df['st_gap_5m'] * config.out_set.out_gap

    # --------------- st level tp --------------- #
    short_tp = res_df['st_lower3_5m'] - res_df['st_gap_5m'] * config.tp_set.tp_gap
    long_tp = res_df['st_upper3_5m'] + res_df['st_gap_5m'] * config.tp_set.tp_gap

    return entry, short_ep, long_ep, short_out, long_out, short_tp, long_tp


def interval_to_min(interval):
    if interval == '15m':
        int_minute = 15
    elif interval == '30m':
        int_minute = 30
    elif interval == '1h':
        int_minute = 60
    elif interval == '4h':
        int_minute = 240

    return int_minute


def calc_train_days(interval, use_rows):
    if interval == '15m':
        data_amt = 96
    elif interval == '30m':
        data_amt = 48
    elif interval == '1h':
        data_amt = 24
    elif interval == '4h':
        data_amt = 6
    else:
        print('interval out of range')
        quit()

    days = int(use_rows / data_amt) + 1

    return days


if __name__ == '__main__':

    os.chdir("./..")
    # print()

    from binance_futures_concat_candlestick import concat_candlestick

    days = 1
    symbol = 'ETHUSDT'
    interval = '1m'
    interval2 = '3m'

    use_rows = 200
    use_rows2 = 100

    gap = 0.00005

    # # print(calc_train_days('1h', 3000))
    # df = pd.read_excel('../candlestick_concated/%s/%s %s.xlsx' % (interval, date, symbol), index_col=0)
    # # print(df.head())
    # print(len(df))

    # new_df_, _ = concat_candlestick(symbol, interval, days=1, timesleep=0.2)
    # new_df2_, _ = concat_candlestick(symbol, interval2, days=1, timesleep=0.2)
    #
    # new_df = new_df_.iloc[-use_rows:].copy()
    # new_df2 = new_df2_.iloc[-use_rows2:].copy()
    # res_df = sync_check(new_df, new_df2)
    #
    # tp = res_df['middle_line'].iloc[-1] * (1 + gap)
    #
    # print("tp :", tp)

    # prev_tp = tp

    # if open_side == OrderSide.SELL:
    #     tp = res_df['middle_line'].iloc[self.last_index] * (1 + self.gap)
    # else:
    #     tp = res_df['middle_line'].iloc[self.last_index] * (1 - self.gap)

    # import time
    #
    # s_time = time.time()
    # #       tp_update       #
    # print(tp_update(df, plotting=False, save_path="test.png"))
    # print("elapsed time :", time.time() - s_time)
