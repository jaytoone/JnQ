from binance_f import RequestClient
from binance_f.model import *
from binance_f.constant.test import *
from binance_f.base.printobject import *
import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# from datetime import datetime
# import os
# from scipy import stats
# from asq.initiators import query
# import matplotlib.ticker as ticker
# from sklearn.preprocessing import MaxAbsScaler, StandardScaler
import mpl_finance as mf
# import time
# import math

from Funcs_For_Trade import *
from Funcs_Indicator import *

pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 2500)
pd.set_option('display.max_columns', 2500)


def min_max_scaler(x):
    scaled_x = (x - x.min()) / (x.max() - x.min())
    return scaled_x


request_client = RequestClient(api_key=g_api_key, secret_key=g_secret_key)


def profitage(df, second_df, third_df, symbol=None, date=None, excel=0, get_fig=0, show_time=False, label_type='Test'):
    # limit = None  # <-- should be None, if you want all data from concated chart

    if date is None:
        date = str(datetime.now()).split(' ')[0]

    # print(df[['high', 'low']].tail())
    # quit()

    df['senkou_a'], df['senkou_b'] = ichimoku(df)

    # second_df['senkou_a'], second_df['senkou_b'] = ichimoku(second_df)
    # df = df.join(pd.DataFrame(index=df.index, data=to_lower_tf(df, second_df, [-2, -1], interval=3),
    #                           columns=['h_senkou_a', 'h_senkou_b']))

    # df['ema'] = ema(df['close'], 200)

    # second_df['sar'] = lucid_sar(second_df)
    # df = df.join(pd.DataFrame(index=df.index, data=to_lower_tf(df, second_df, [-1], interval=3), columns=['sar']))

    # df['bbw'] = bb_width(df, 20, 2)
    second_df['b_upper'], second_df['b_lower'], second_df['bbw'] = bb_width(second_df, 20, 2)
    df = df.join(pd.DataFrame(index=df.index, data=to_lower_tf(df, second_df, [i for i in range(-3, 0, 1)], interval=3),
                              columns=['b_upper', 'b_lower', 'bbw']))

    # print(df.tail(20))
    # quit()

    # df['trix'] = trix_hist(df, 14, 1, 9)
    second_df['trix'] = trix_hist(second_df, 14, 1, 9)
    df = df.join(pd.DataFrame(index=df.index, data=to_lower_tf(df, second_df, [-1], interval=3), columns=['trix']))

    # ha_df = heikinashi(df)

    # _, _, df['ST_Trend'] = supertrend(ha_df, 7, 2.5)

    # df['fisher'] = fisher(df, 30)

    # ha_second_df = heikinashi(second_df)
    # # ha_third_df = heikinashi(third_df)
    # # print(ha_second_df.tail(10))
    # # quit()
    #
    # second_df['minor_ST1_Up'], second_df['minor_ST1_Down'], second_df['minor_ST1_Trend'] = supertrend(second_df, 10, 2)
    # second_df['minor_ST2_Up'], second_df['minor_ST2_Down'], second_df['minor_ST2_Trend'] = supertrend(ha_second_df, 7,
    #                                                                                                   2)
    # second_df['minor_ST3_Up'], second_df['minor_ST3_Down'], second_df['minor_ST3_Trend'] = supertrend(ha_second_df, 7,
    #                                                                                                   2.5)
    # # print(df.head(20))
    # # quit()
    #
    # startTime = time.time()
    #
    # df = df.join(pd.DataFrame(index=df.index, data=to_lower_tf(df, second_df, [i for i in range(-9, 0, 1)], 5),
    #                           columns=['minor_ST1_Up', 'minor_ST1_Down', 'minor_ST1_Trend'
    #                               , 'minor_ST2_Up', 'minor_ST2_Down', 'minor_ST2_Trend'
    #                               , 'minor_ST3_Up', 'minor_ST3_Down', 'minor_ST3_Trend']))
    #
    # if show_time:
    #     print('time consumed by to_lower_tf :', time.time() - startTime)

    # print(df.tail())
    # print(df.iloc[:, -8:-2].tail(20))
    # quit()

    # if label_type == 'Train':

    long_marker_x = list()
    long_marker_y = list()
    short_marker_x = list()
    short_marker_y = list()

    df['trade_state'] = np.nan
    df['ep'] = np.nan
    df['sl_level'] = np.nan
    df['tp_level'] = np.nan
    df['leverage'] = np.nan

    #           Trading Fee          #
    fee = 0.0002 * 2

    ema_lookback = 50
    ichimoku_lookback = 30
    candle_lookback = 3

    target_sl_percent = 0.05
    tp_gap = 1
    fixed_leverage = 20

    tp_pips = 5
    price_precision = 2

    for i in range(ichimoku_lookback, len(df)):

        #           BBW Set         #
        if df['bbw'].iloc[i] > 0.015:

            #           Long            #

            #           SAR Set         #
            # if df['close'].iloc[i] > df['sar'].iloc[i]:

            #           EMA Set         #
            # if df['low'].rolling(ema_lookback).min().iloc[i] >= df['ema'].iloc[i]:

            #           Trix Set            #
            # if df['trix'].iloc[i] > 0:

            #           HTF  Ichimoku Set           #
            # if np.minimum(df['senkou_a'].iloc[i], df['senkou_b'].iloc[i]) >=\
            #         np.minimum(df['h_senkou_a'].iloc[i], df['h_senkou_b'].iloc[i]):
            # if df['close'].iloc[i - 1] <= np.max(df[['h_senkou_a', 'h_senkou_b', 'senkou_a', 'senkou_b']].iloc[i - 1].values) and \
            #         df['close'].iloc[i] > np.max(df[['h_senkou_a', 'h_senkou_b', 'senkou_a', 'senkou_b']].iloc[i].values):

                #           Ichimoku Set            #
                # if df['close'].iloc[i - 1] <= np.maximum(df['senkou_a'].iloc[i - 1], df['senkou_b'].iloc[i - 1]) and \
                #         df['close'].iloc[i] > np.maximum(df['senkou_a'].iloc[i], df['senkou_b'].iloc[i]):  # and \
                    # df['close'].iloc[i] >= df['open'].iloc[i]:

                    for j in range(i + 1 - ichimoku_lookback, i + 1):

                        if df['close'].iloc[j - 1] <= np.minimum(df['senkou_a'].iloc[j - 1], df['senkou_b'].iloc[j - 1]) and \
                                df['close'].iloc[j] > np.minimum(df['senkou_a'].iloc[j], df['senkou_b'].iloc[j]):  # and \
                            # df['close'].iloc[j] >= df['open'].iloc[j]:

                            if 1 not in df['trade_state'].iloc[j:i].values:
                                df['trade_state'].iloc[i] = 1
                                df['ep'].iloc[i] = df['close'].iloc[i]
                                df['sl_level'].iloc[i] = df['low'].iloc[i + 1 - candle_lookback:i + 1].min()

                                #       tp by gap_ratio     #
                                df['tp_level'].iloc[i] = df['close'].iloc[i] + \
                                                         (df['close'].iloc[i] - df['low'].iloc[
                                                                                i + 1 - candle_lookback:i + 1].min()) * tp_gap

                                #       tp by pips      #
                                # df['tp_level'].iloc[i] = df['ep'].iloc[i] + (10 ** -price_precision) * tp_pips

                                #       leverage by sl      #
                                # df['leverage'].iloc[i] = int(target_sl_percent / (
                                #             abs(df['ep'].iloc[i] - df['sl_level'].iloc[i]) / df['ep'].iloc[i]))
                                # df['leverage'].iloc[i] = min(50, df['leverage'].iloc[i])

                                #       fixed leverage      #
                                df['leverage'].iloc[i] = fixed_leverage
                                long_marker_x.append(i)
                                long_marker_y.append(df['close'].iloc[i])
                                break

            #           Short            #

            #           SAR Set         #
            # else:

            #           EMA Set         #
            # if df['high'].rolling(ema_lookback).max().iloc[i] <= df['ema'].iloc[i]:

            #           Trix Set        #
            # else:

            #           HTF  Ichimoku Set           #
            # if np.maximum(df['senkou_a'].iloc[i], df['senkou_b'].iloc[i]) <= np.maximum(
            #     df['h_senkou_a'].iloc[i], df['h_senkou_b'].iloc[i]):
            # if df['close'].iloc[i - 1] >= np.min(df[['h_senkou_a', 'h_senkou_b', 'senkou_a', 'senkou_b']].iloc[i - 1].values) and \
            #         df['close'].iloc[i] < np.min(df[['h_senkou_a', 'h_senkou_b', 'senkou_a', 'senkou_b']].iloc[i].values):

                #           Ichimoku Set            #
                # if df['close'].iloc[i - 1] >= np.minimum(df['senkou_a'].iloc[i - 1], df['senkou_b'].iloc[i - 1]) and \
                #         df['close'].iloc[i] < np.minimum(df['senkou_a'].iloc[i], df['senkou_b'].iloc[i]):  # and \
                    # df['close'].iloc[i] <= df['open'].iloc[i]:

                    for j in range(i + 1 - ichimoku_lookback, i + 1):

                        if df['close'].iloc[j - 1] >= np.maximum(df['senkou_a'].iloc[j - 1], df['senkou_b'].iloc[j - 1]) and \
                                df['close'].iloc[j] < np.maximum(df['senkou_a'].iloc[j], df['senkou_b'].iloc[j]):  # and \
                            # df['close'].iloc[i] <= df['open'].iloc[i]:

                            if 0 not in df['trade_state'].iloc[j:i].values:
                                df['trade_state'].iloc[i] = 0
                                df['ep'].iloc[i] = df['close'].iloc[i]
                                df['sl_level'].iloc[i] = df['high'].iloc[i + 1 - candle_lookback:i + 1].max()

                                #       tp by gap_ratio     #
                                df['tp_level'].iloc[i] = df['close'].iloc[i] + \
                                                         (df['close'].iloc[i] - df['high'].iloc[
                                                                                i + 1 - candle_lookback:i + 1].max()) * tp_gap

                                #       tp by pips      #
                                # df['tp_level'].iloc[i] = df['ep'].iloc[i] - (10 ** -price_precision) * tp_pips

                                # df['leverage'].iloc[i] = int(target_sl_percent / (
                                #             abs(df['ep'].iloc[i] - df['sl_level'].iloc[i]) / df['ep'].iloc[i]))
                                # df['leverage'].iloc[i] = min(50, df['leverage'].iloc[i])

                                df['leverage'].iloc[i] = fixed_leverage
                                short_marker_x.append(i)
                                short_marker_y.append(df['close'].iloc[i])
                                break

    # print(df.trade_state)
    # quit()
    return df


if __name__ == "__main__":

    home_dir = os.path.expanduser('~')
    dir = './database/'

    import pickle
    import random

    #       when we use Realtime Data       #
    with open('future_coin.p', 'rb') as f:
        coin_list = pickle.load(f)
        # random.shuffle(ohlcv_list)

    #       when we use Stacked Data        #
    coin_list = os.listdir(dir + '1m/')

    selected_date = '12-15'

    # print(ohlcv_list)
    # quit()

    # coin_list = ['COMP']

    # for file in ohlcv_list:
    #     coin = file.split()[1].split('.')[0]
    #     date = file.split()[0]
    date = str(datetime.now()).split()[0]
    # date = str(datetime.now()).split()[0] + '_110ver'
    excel_list = os.listdir('./BestSet/Test_ohlc/')
    total_df = pd.DataFrame(
        columns=['long_signal_value', 'short_signal_value', 'period1', 'short_period', 'total_profit_avg',
                 'plus_profit_avg', 'minus_profit_avg', 'avg_profit_avg',
                 'min_profit_avg', 'exit_fluc_mean'])

    #       Make folder      #
    try:
        os.mkdir("./chart_check/%s/" % (date))

    except Exception as e:
        print(e)

    total_profit = 0
    plus_profit = 0
    minus_profit = 0
    avg_profit = 0
    min_profit = 0
    exit_fluc_mean = 0
    for coin in coin_list:

        start = time.time()
        # try:
        if 'ohlcv' in coin:
            continue
        if selected_date not in coin:
            continue

        result = profitage(coin, date=date, get_fig=0, excel=1)
        # quit()
        print(coin, result)
        continue
        # if result[0] == 1.:  # 거래가 없는 코인을 데이터프레임에 저장할 필요가 없다.
        #     continue
        # except Exception as e:
        #     continue
        # quit()
        total_profit += result[0]
        plus_profit += result[1]
        minus_profit += result[2]
        avg_profit += result[3]
        min_profit += result[4]
        exit_fluc_mean += result[5]
    total_profit_avg = total_profit / len(excel_list)
    plus_profit_avg = plus_profit / len(excel_list)
    minus_profit_avg = minus_profit / len(excel_list)
    avg_profit_avg = avg_profit / len(excel_list)
    min_profit_avg = min_profit / len(excel_list)
    exit_fluc_avg = exit_fluc_mean / len(excel_list)
#     print(long_signal_value, short_signal_value, period1, short_period, total_profit_avg,
#           plus_profit_avg, minus_profit_avg, avg_profit_avg,
#           min_profit_avg, exit_fluc_avg, '%.3f second' % (time.time() - start))
#
#     result_df = pd.DataFrame(data=[
#         [long_signal_value, short_signal_value, period1, short_period, total_profit_avg,
#          plus_profit_avg, minus_profit_avg, avg_profit_avg,
#          min_profit_avg, exit_fluc_avg]],
#         columns=['long_signal_value', 'short_signal_value', 'period1', 'short_period',
#                  'total_profit_avg', 'plus_profit_avg',
#                  'minus_profit_avg', 'avg_profit_avg', 'min_profit_avg', 'exit_fluc_avg'])
#     total_df = total_df.append(result_df)
#     print()
#
# total_df.to_excel('./BestSet/total_df %s.xlsx' % long_signal_value)
# break
