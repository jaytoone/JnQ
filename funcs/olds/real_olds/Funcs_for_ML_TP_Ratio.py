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
# import mpl_finance as mf
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

    chart_height = (df['high'].max() - df['low'].min())
    chart_gap = (df['high'].max() / df['low'].min())
    ha_second_df = heikinashi(second_df)
    ha_third_df = heikinashi(third_df)
    # print(ha_second_df.tail(10))
    # quit()

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
    # df = df.join(pd.DataFrame(index=df.index, data=to_lower_tf(df, second_df, [i for i in range(-9, 0, 1)], 3),
    #                           columns=['minor_ST1_Up', 'minor_ST1_Down', 'minor_ST1_Trend'
    #                               , 'minor_ST2_Up', 'minor_ST2_Down', 'minor_ST2_Trend'
    #                               , 'minor_ST3_Up', 'minor_ST3_Down', 'minor_ST3_Trend']))
    #
    # if show_time:
    #     print('time consumed by to_lower_tf :', time.time() - startTime)
    #
    # df['minor_ST1'] = np.where(df['minor_ST1_Trend'] == 1, df['minor_ST1_Down'], df['minor_ST1_Up'])
    # df['minor_ST2'] = np.where(df['minor_ST2_Trend'] == 1, df['minor_ST2_Down'], df['minor_ST2_Up'])
    # df['minor_ST3'] = np.where(df['minor_ST3_Trend'] == 1, df['minor_ST3_Down'], df['minor_ST3_Up'])
    #
    # startTime = time.time()
    #
    # third_df['major_ST1_Up'], third_df['major_ST1_Down'], third_df['major_ST1_Trend'] = supertrend(third_df, 10, 2)
    # third_df['major_ST2_Up'], third_df['major_ST2_Down'], third_df['major_ST2_Trend'] = supertrend(ha_third_df, 7,
    #                                                                                                2)
    # third_df['major_ST3_Up'], third_df['major_ST3_Down'], third_df['major_ST3_Trend'] = supertrend(ha_third_df, 7,
    #                                                                                                2.5)
    #
    # df = df.join(pd.DataFrame(index=df.index, data=to_lower_tf(df, third_df, [i for i in range(-9, 0, 1)], 30),
    #                           columns=['major_ST1_Up', 'major_ST1_Down', 'major_ST1_Trend'
    #                               , 'major_ST2_Up', 'major_ST2_Down', 'major_ST2_Trend'
    #                               , 'major_ST3_Up', 'major_ST3_Down', 'major_ST3_Trend']))
    #
    # df['major_ST1'] = np.where(df['major_ST1_Trend'] == 1, df['major_ST1_Down'], df['major_ST1_Up'])
    # df['major_ST2'] = np.where(df['major_ST2_Trend'] == 1, df['major_ST2_Down'], df['major_ST2_Up'])
    # df['major_ST3'] = np.where(df['major_ST3_Trend'] == 1, df['major_ST3_Down'], df['major_ST3_Up'])
    #
    # if show_time:
    #     print('time consumed by to_lower_tf :', time.time() - startTime)
    #
    # print('elapsed time :', time.time() - start)

    df['MA'] = df['close'].rolling(15).mean()

    macd(df, 19, 26, 9)

    h_close = df.close.rolling(14).max()
    l_close = df.close.rolling(14).min()
    df['Stochastic'] = (df.close - l_close) / (h_close - l_close) * 100

    df['Stochastic_K'] = df.Stochastic.rolling(3).mean()
    df['Stochastic_D'] = df.Stochastic_K.rolling(3).mean()

    df['RSI'] = rsi(df, 14)

    startTime = time.time()

    df['30EMA'] = ema(df['close'], 30)
    df['100EMA'] = ema(df['close'], 100)

    df['CB'], df['EMA_CB'] = cct_bbo(df, 18, 13)
    # print(df.tail(10))
    # quit()
    if show_time:
        print('time consumed by ema, cb :', time.time() - startTime)
    startTime = time.time()

    df['Fisher1'] = fisher(df, period=30)
    df['Fisher2'] = fisher(df, period=60)
    df['Fisher3'] = fisher(df, period=120)

    if show_time:
        print('time consumed by fisher series :', time.time() - startTime)
    startTime = time.time()

    df['Trix'] = trix_hist(df, 14, 1, 9)
    second_df['Trix'] = trix_hist(second_df, period=14, multiplier=1, signal_period=9)
    df = df.join(pd.DataFrame(index=df.index, data=to_lower_tf(df, second_df, [-1], 3), columns=['minor_Trix']))

    if show_time:
        print('time consumed by trix :', time.time() - startTime)

    # print(df.tail())
    # print(df.iloc[:, -8:-2].tail(20))
    # quit()

    #                               Open Condition                              #
    #       Open Long / Short = 1 / 2       #
    #       Close Long / Short = -1 / -2        #
    upper = 0.5
    lower = -upper

    long_tp_marker_x = list()
    long_tp_marker_y = list()
    long_sl_marker_x = list()
    long_sl_marker_y = list()
    short_tp_marker_x = list()
    short_tp_marker_y = list()
    short_sl_marker_x = list()
    short_sl_marker_y = list()

    tick_in_box = 30
    sl_least_gap_ratio = 1 / 10
    tp_ratio = 5
    df['ep'] = df['close']
    df['box_high'] = df['high'].rolling(tick_in_box).max()
    df['box_low'] = df['low'].rolling(tick_in_box).min()
    df['long_sl'] = df['ep'] - np.minimum(abs(df['ep'] - df['box_high']), abs(df['ep'] - df['box_low']))
    df['long_tp'] = df['ep'] + abs(df['ep'] - df['long_sl']) * tp_ratio

    df['short_sl'] = df['ep'] + np.minimum(abs(df['ep'] - df['box_high']), abs(df['ep'] - df['box_low']))
    df['short_tp'] = df['ep'] - abs(df['ep'] - df['short_sl']) * tp_ratio
    df['trade_state_long'] = np.nan
    df['trade_state_short'] = np.nan

    if label_type == 'Train':

        startTime = time.time()
        for i in range(tick_in_box, len(df)):

            if df['box_low'].iloc[i] + abs(df['box_high'].iloc[i] - df['box_low'].iloc[i]) * \
                    sl_least_gap_ratio <= df['ep'].iloc[i]:

                for j in range(i + 1, len(df)):

                    #       Long side       #
                    if df['low'].iloc[j] < df['long_sl'].iloc[i]:
                        max_high = df['high'].iloc[i:j].max()
                        df['trade_state_long'].iloc[i] = (max_high - df['ep'].iloc[i]) / (df['ep'].iloc[i] - df['long_sl'].iloc[i])
                        break

            if df['ep'].iloc[i] <= df['box_high'].iloc[i] - \
                    abs(df['box_high'].iloc[i] - df['box_low'].iloc[i]) * sl_least_gap_ratio:

                for k in range(i + 1, len(df)):

                    #       Short side       #
                    if df['high'].iloc[k] > df['short_sl'].iloc[i]:
                        min_low = df['low'].iloc[i:k].min()
                        df['trade_state_short'].iloc[i] = (min_low - df['ep'].iloc[i]) / (df['ep'].iloc[i] - df['short_sl'].iloc[i])
                        break

        # print(df.trade_state)
        # quit()
        if show_time:
            print('time consumed by labeling :', time.time() - startTime)

        #           Show Result Image         #
        if get_fig:
            try:
                df = df.reset_index(drop=True)

                plt.subplot(211)
                plt.plot(df['close'])
                # plt.plot(df['long_sl'])
                # plt.plot(df['long_tp'])
                plt.plot(long_tp_marker_x, long_tp_marker_y, 'o', color='blue')
                plt.plot(long_sl_marker_x, long_sl_marker_y, 'o', color='red')
                plt.plot(short_tp_marker_x, short_tp_marker_y, 'o', color='fuchsia')
                plt.plot(short_sl_marker_x, short_sl_marker_y, 'o', color='cyan')

                plt.subplot(212)
                plt.plot(df['trade_state_long'], 'r')
                plt.plot(-df['trade_state_short'], 'b')
                plt.show()
                # plt.savefig("./chart_check/%s/%s.png" % (date, symbol), dpi=300)
                # plt.close()

            except Exception as e:
                print('Error in plotting :', e)

    #       Save Excel      #
    if excel:

        save_path = "labeled_data/TP_Ratio_TVT/%s/" % tick_in_box

        #       Make folder      #
        try:
            os.makedirs(save_path)

        except Exception as e:
            print(e)

        if symbol.endswith('.xlsx'):
            # df.to_excel("labeled_data/%s" % coin)
            df.to_excel(save_path + '%s' % symbol)
        else:
            df.to_excel(save_path + "%s %s.xlsx" % (date, symbol))

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
