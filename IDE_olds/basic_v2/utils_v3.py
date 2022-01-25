# import numpy as np
# import pandas as pd
# from binance_futures_concat_candlestick import concat_candlestick
# import matplotlib.pyplot as plt
# from plotly import subplots
# import plotly.offline as offline
# import plotly.graph_objs as go
# import pickle
from funcs.funcs_indicator_candlescore import *
from funcs.funcs_trader import *

pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 2500)
pd.set_option('display.max_columns', 2500)


def sync_check(df, second_df, third_df=None, fourth_df=None, fifth_df=None, show_msg=False,
               sar_on=False, cloud_on=False, macd_on=False, trix_on=False):

    #           supertrend          #
    ha_second_df = heikinashi(second_df)
    # ha_third_df = heikinashi(third_df)
    # print(ha_second_df.tail(10))
    # quit()

    second_df['minor_ST1_Up'], second_df['minor_ST1_Down'], second_df['minor_ST1_Trend'] = supertrend(second_df, 10, 2)
    second_df['minor_ST2_Up'], second_df['minor_ST2_Down'], second_df['minor_ST2_Trend'] = supertrend(ha_second_df, 7,
                                                                                                      2)
    second_df['minor_ST3_Up'], second_df['minor_ST3_Down'], second_df['minor_ST3_Trend'] = supertrend(ha_second_df, 7,
                                                                                                      2.5)
    # print(df.head(20))
    # quit()

    # startTime = time.time()

    min_upper = np.min(second_df[["minor_ST1_Up", "minor_ST2_Up", "minor_ST3_Up"]], axis=1)
    max_lower = np.max(second_df[["minor_ST1_Down", "minor_ST2_Down", "minor_ST3_Down"]], axis=1)

    second_df['middle_line'] = (min_upper + max_lower) / 2

    if show_msg:
        print("supertrend phase done")

    #           lucid sar              #
    if sar_on:

        df['sar1'] = lucid_sar(df)

        if second_df is not None:
            second_df['sar'] = lucid_sar(second_df)

        if third_df is not None:
            third_df['sar'] = lucid_sar(third_df)

        if fourth_df is not None:
            fourth_df['sar'] = lucid_sar(fourth_df)

        if fifth_df is not None:
            fifth_df['sar'] = lucid_sar(fifth_df)

    # print(df[['sar1', 'sar2']].tail(20))
    # print(df[['minor_ST1_Up', 'minor_ST1_Trend']].tail(20))
    # quit()

        if show_msg:
            print("sar phase done")

    #           ichimoku            #
    if cloud_on:

        displacement_cols = 2
        df['senkou_a1'], df['senkou_b1'] = ichimoku(df)

        #       cloud 는 displacement 때문에, join 을 사용하는게 맞음       #

        # if second_df is not None:
        #     second_df['senkou_a'], second_df['senkou_b'] = ichimoku(second_df)
        #     df = df.join(
        #         pd.DataFrame(index=df.index, data=to_lower_tf(df, second_df, [-2, -1]),
        #                      columns=['senkou_a2', 'senkou_b2']))
        #     displacement_cols += 2
        #
        # if third_df is not None:
        #     third_df['senkou_a'], third_df['senkou_b'] = ichimoku(third_df)
        #     df = df.join(
        #         pd.DataFrame(index=df.index, data=to_lower_tf(df, third_df, [-2, -1]),
        #                      columns=['senkou_a3', 'senkou_b3']))
        #     displacement_cols += 2
        #
        # if fourth_df is not None:
        #     fourth_df['senkou_a'], fourth_df['senkou_b'] = ichimoku(fourth_df)
        #     df = df.join(
        #         pd.DataFrame(index=df.index, data=to_lower_tf(df, fourth_df, [-2, -1]),
        #                      columns=['senkou_a4', 'senkou_b4']))
        #     displacement_cols += 2
        #
        # if fifth_df is not None:
        #     fifth_df['senkou_a'], fifth_df['senkou_b'] = ichimoku(fifth_df)
        #     df = df.join(
        #         pd.DataFrame(index=df.index, data=to_lower_tf(df, fifth_df, [-2, -1]),
        #                      columns=['senkou_a5', 'senkou_b5']))
        #     displacement_cols += 2

        #           1-2. displacement           #
        # df['senkou_a1'] = df['senkou_a1'].shift(26 - 1)
        # df['senkou_b1'] = df['senkou_b1'].shift(26 - 1)
        df.iloc[:, -displacement_cols:] = df.iloc[:, -displacement_cols:].shift(26 - 1)

        if show_msg:
            print("cloud phase done")

    #           macd            #
    if macd_on:
        df['macd_hist1'] = macd(df)

        if second_df is not None:
            second_df['macd_hist'] = macd(second_df)

        if third_df is not None:
            third_df['macd_hist'] = macd(third_df)

        if fourth_df is not None:
            fourth_df['macd_hist'] = macd(fourth_df)

        if fifth_df is not None:
            fifth_df['macd_hist'] = macd(fifth_df)

        if show_msg:
            print("macd phase done")

    #         trix        #
    if trix_on:
        df['trix1'] = trix_hist(df, 14, 1, 5)

        if second_df is not None:
            second_df['trix'] = trix_hist(second_df, 14, 1, 5)

        if third_df is not None:
            third_df['trix'] = trix_hist(third_df, 14, 1, 5)

        if fourth_df is not None:
            fourth_df['trix'] = trix_hist(fourth_df, 14, 1, 5)

        if fifth_df is not None:
            fifth_df['trix'] = trix_hist(fifth_df, 14, 1, 5)

        if show_msg:
            print("trix phase done")

    #          add for ep           #
    second_df['min_upper'] = min_upper
    second_df['max_lower'] = max_lower

    return df, second_df


# def enlist_eplvrg(df, upper_ep, lower_ep, leverage=1):
def enlist_eplvrg(df, upper_ep, lower_ep):

    # close = df['close']

    output_df = df.copy()
    # output_df['trade_state'] = np.nan
    output_df['long_ep'] = np.nan
    output_df['short_ep'] = np.nan
    # output_df['long_tp_level'] = np.nan
    # output_df['short_tp_level'] = np.nan
    output_df['sl'] = np.nan
    # output_df['tp'] = np.nan
    # output_df['leverage'] = np.nan

    output_df['short_ep'].iloc[-1] = upper_ep.iloc[-1]
    output_df['long_ep'].iloc[-1] = lower_ep.iloc[-1]
    # output_df['leverage'].iloc[-1] = leverage

    return output_df


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
    days = 1
    symbol = 'DOTUSDT'
    interval = '30m'
    date = '2021-05-17'

    # print(calc_train_days('1h', 3000))
    df = pd.read_excel('../candlestick_concated/%s/%s %s.xlsx' % (interval, date, symbol), index_col=0)
    # print(df.head())
    print(len(df))

    import time

    s_time = time.time()
    #       tp_update       #
    print(tp_update(df, plotting=False, save_path="test.png"))
    print("elapsed time :", time.time() - s_time)
