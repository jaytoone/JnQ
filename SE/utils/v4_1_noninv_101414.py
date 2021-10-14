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


def sync_check(df, second_df=None, third_df=None, fourth_df=None, fifth_df=None, sixth_df=None, seventh_df=None):

    #       Todo        #
    #        필요한 indi. 는 enlist_epouttp & mr_check 보면서 삽입
    df = st_price_line(df, third_df, '5m')
    df = st_level(df, '5m', 0.5)

    #        sma60         #
    df['sma_1m'] = df['close'].rolling(60).mean()

    #        cloud bline         #
    fourth_df['cloud_bline_15m'] = cloud_bline(fourth_df, 26)
    df = df.join(pd.DataFrame(index=df.index, data=to_lower_tf(df, fourth_df, [-1]), columns=['cloud_bline_15m']))

    return df


def enlist_epouttp(res_df, config):

    res_df['entry'] = np.zeros(len(res_df))

    res_df['entry'] = np.where((res_df['close'].shift(1) >= res_df['st_lower_5m']) &
                     (res_df['close'] < res_df['st_lower_5m'])
                     , res_df['entry'] - 1, res_df['entry'])

    res_df['entry'] = np.where((res_df['close'].shift(1) <= res_df['st_upper_5m']) &
                     (res_df['close'] > res_df['st_upper_5m'])
                     , res_df['entry'] + 1, res_df['entry'])

    # ----- market entry ----- #
    res_df['short_ep'] = res_df['close']
    res_df['long_ep'] = res_df['close']

    # --------------- st level consolidation out --------------- #
    res_df['short_out'] = res_df['st_upper2_5m'] + res_df['st_gap_5m'] * config.out_set.out_gap
    res_df['long_out'] = res_df['st_lower2_5m'] - res_df['st_gap_5m'] * config.out_set.out_gap

    # --------------- st level tp --------------- #
    res_df['short_tp'] = res_df['st_lower3_5m'] - res_df['st_gap_5m'] * config.tp_set.tp_gap
    res_df['long_tp'] = res_df['st_upper3_5m'] + res_df['st_gap_5m'] * config.tp_set.tp_gap

    return res_df


def mr_check(res_df, config):

    #       Todo        #
    #        1. check iloc[i] --> to last_index
    #        2. if short const. copied to long, check reversion

    mr_score = 0

    # ---------------------- short ---------------------- #

    #        1. cbline const. compare with close, so 'close' data should be confirmed
    #        2. or, entry market_price would be replaced with 'close'
    if res_df['entry'][config.init_set.last_index] == config.ep_set.short_entry_score:

        #           tr scheduling        #
        if (res_df['close'].iloc[config.init_set.last_index] - res_df['short_tp'].iloc[
            config.init_set.last_index] - config.init_set.fee * res_df['close'].iloc[config.init_set.last_index]) / (
                res_df['short_out'].iloc[config.init_set.last_index] - res_df['close'].iloc[
            config.init_set.last_index] + config.init_set.fee * res_df['close'].iloc[
                    config.init_set.last_index]) >= config.ep_set.tr_thresh:
            mr_score -= 1

        #            mr             #
        prev_entry_cnt = 0
        for back_i in range(i - 1, 0, -1):
            if res_df['entry'][back_i] == 1:
                break

            elif res_df['entry'][back_i] == -1:
                prev_entry_cnt += 1

        if prev_entry_cnt <= config.ep_set.entry_incycle:
            mr_score -= 1

        if res_df['close'].iloc[i] <= res_df['sma_1m'].iloc[i]:
            mr_score -= 1

        if res_df['close'].iloc[config.init_set.last_index] <= res_df['cloud_bline_15m'].iloc[
            config.init_set.last_index]:
            mr_score -= 1

    # ---------------------- long ---------------------- #
    if res_df['entry'][config.init_set.last_index] == -config.ep_set.short_entry_score:

        if (res_df['long_tp'].iloc[config.init_set.last_index] - res_df['close'].iloc[config.init_set.last_index]
            - config.init_set.fee * res_df['close'].iloc[config.init_set.last_index]) / (
                res_df['close'].iloc[config.init_set.last_index] - res_df['long_out'].iloc[config.init_set.last_index]
                + config.init_set.fee * res_df['close'].iloc[config.init_set.last_index]) >= config.ep_set.tr_thresh:
            mr_score += 1

        #            mr             #
        prev_entry_cnt = 0
        for back_i in range(config.init_set.last_index - 1, 0, -1):
            if res_df['entry'][back_i] == -1:
                break

            elif res_df['entry'][back_i] == 1:
                prev_entry_cnt += 1

        if prev_entry_cnt <= config.ep_set.entry_incycle:
            mr_score += 1

        if res_df['close'].iloc[config.init_set.last_index] >= res_df['sma_1m'].iloc[config.init_set.last_index]:
            mr_score += 1

        if res_df['close'].iloc[config.init_set.last_index] >= res_df['cloud_bline_15m'].iloc[
            config.init_set.last_index]:
            mr_score += 1

    return mr_score


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
