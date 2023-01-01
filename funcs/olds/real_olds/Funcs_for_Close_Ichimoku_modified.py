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

    # df['senkou_a'], df['senkou_b'] = ichimoku(df)

    # second_df['senkou_a'], second_df['senkou_b'] = ichimoku(second_df)
    # df = df.join(pd.DataFrame(index=df.index, data=to_lower_tf(df, second_df, [-2, -1], interval=3),
    #                           columns=['h_senkou_a', 'h_senkou_b']))

    # df['ema'] = ema(df['close'], 200)

    # second_df['sar'] = lucid_sar(second_df)
    # df = df.join(pd.DataFrame(index=df.index, data=to_lower_tf(df, second_df, [-1], interval=3), columns=['sar']))

    # df['bbw'] = bb_width(df, 20, 2)
    # second_df['b_upper'], second_df['b_lower'], second_df['bbw'] = bb_width(second_df, 20, 2)
    # df = df.join(pd.DataFrame(index=df.index, data=to_lower_tf(df, second_df, [i for i in range(-3, 0, 1)], interval=3),
    #                           columns=['b_upper', 'b_lower', 'bbw']))

    # print(df.tail(20))
    # quit()

    # df['trix'] = trix_hist(df, 14, 1, 9)
    second_df['trix'] = trix_hist(second_df, 14, 1, 9)
    df = df.join(pd.DataFrame(index=df.index, data=to_lower_tf(df, second_df, [-1], interval=3), columns=['trix']))

    # print(df.tail())
    # print(df.iloc[:, -8:-2].tail(20))
    # quit()

    #           Trading Fee          #
    fee = 0.0002 * 2

    #           Set Params          #
    ema_lookback = 50
    ichimoku_lookback = 30
    candle_lookback = 3

    target_sl_percent = 0.05
    tp_gap = 1
    fixed_leverage = 20

    tp_pips = 5
    price_precision = 2

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
