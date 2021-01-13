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

    # quit()

    df['senkou_a'], df['senkou_b'] = ichimoku(df)

    df['ema'] = ema(df['close'], 200)

    #           Open Condition Define           #

    # print(df.tail())
    # print(df.iloc[:, -8:-2].tail(20))
    # quit()

    # long_tp_marker_x = list()
    # long_tp_marker_y = list()
    # long_sl_marker_x = list()
    # long_sl_marker_y = list()
    # short_tp_marker_x = list()
    # short_tp_marker_y = list()
    # short_sl_marker_x = list()
    # short_sl_marker_y = list()

    # if label_type == 'Train':

    #           Show Result Image         #
    # if get_fig:
    #     try:
    #         df = df.reset_index(drop=True)
    #
    #         plt.subplot(211)
    #         plt.plot(df['close'])
    #         # plt.plot(df['long_sl'])
    #         # plt.plot(df['long_tp'])
    #         plt.plot(long_tp_marker_x, long_tp_marker_y, 'o', color='blue')
    #         plt.plot(long_sl_marker_x, long_sl_marker_y, 'o', color='red')
    #         plt.plot(short_tp_marker_x, short_tp_marker_y, 'o', color='fuchsia')
    #         plt.plot(short_sl_marker_x, short_sl_marker_y, 'o', color='cyan')
    #
    #         plt.subplot(212)
    #         plt.plot(df['trade_state_long'], 'r')
    #         plt.plot(-df['trade_state_short'], 'b')
    #         plt.show()
    #         # plt.savefig("./ChartCheck/%s/%s.png" % (date, symbol), dpi=300)
    #         # plt.close()
    #
    #     except Exception as e:
    #         print('Error in plotting :', e)

    #       Save Excel      #
    # if excel:
    #
    #     save_path = "Labeled_Data/TP_Ratio_TVT/%s/" % tick_in_box
    #
    #     #       Make folder      #
    #     try:
    #         os.makedirs(save_path)
    #
    #     except Exception as e:
    #         print(e)
    #
    #     if symbol.endswith('.xlsx'):
    #         # df.to_excel("Labeled_Data/%s" % coin)
    #         df.to_excel(save_path + '%s' % symbol)
    #     else:
    #         df.to_excel(save_path + "%s %s.xlsx" % (date, symbol))
    #
    # return df
    #                               Open Condition                              #
    #       Open Long / Short = 1 / 2       #
    #       Close Long / Short = -1 / -2        #

    long_marker_x = list()
    long_marker_y = list()
    short_marker_x = list()
    short_marker_y = list()

    df['trade_state'] = np.nan
    df['ep'] = np.nan
    df['sl_level'] = np.nan
    df['tp_level'] = np.nan
    df['leverage'] = np.nan

    ema_lookback = 50
    ichimoku_lookback = 30
    candle_lookback = 5

    target_sl_percent = 0.05
    tp_gap = 1
    fixed_leverage = 20

    for i in range(ichimoku_lookback, len(df)):

        if df['senkou_a'].iloc[i] > df['senkou_b'].iloc[i]:

            #           Long            #
            #           EMA Set         #
            # if df['low'].rolling(ema_lookback).min().iloc[i] >= df['ema'].iloc[i]:

                #           Ichimoku Set            #
                if df['close'].iloc[i - 1] <= df['senkou_a'].iloc[i - 1] and \
                        df['close'].iloc[i] > df['senkou_a'].iloc[i] and \
                        df['close'].iloc[i] >= df['open'].iloc[i]:

                    for j in range(i + 1 - ichimoku_lookback, i + 1):

                        if df['close'].iloc[j - 1] <= df['senkou_b'].iloc[j - 1] and \
                                df['close'].iloc[j] > df['senkou_b'].iloc[j] and \
                                df['close'].iloc[j] >= df['open'].iloc[j]:

                            if 1 not in df['trade_state'].iloc[j:i].values:
                                df['trade_state'].iloc[i] = 1
                                df['ep'].iloc[i] = df['close'].iloc[i]
                                df['sl_level'].iloc[i] = df['low'].iloc[i + 1 - candle_lookback:i + 1].min()
                                df['tp_level'].iloc[i] = df['close'].iloc[i] + \
                                                         (df['close'].iloc[i] - df['low'].iloc[
                                                                                i + 1 - candle_lookback:i + 1].min()) * tp_gap
                                # df['leverage'].iloc[i] = int(target_sl_percent / (
                                #             abs(df['ep'].iloc[i] - df['sl_level'].iloc[i]) / df['ep'].iloc[i]))
                                # df['leverage'].iloc[i] = min(50, df['leverage'].iloc[i])
                                df['leverage'].iloc[i] = fixed_leverage
                                long_marker_x.append(i)
                                long_marker_y.append(df['close'].iloc[i])
                                break

            #           Short            #
            #           EMA Set         #
            # elif df['high'].rolling(ema_lookback).max().iloc[i] <= df['ema'].iloc[i]:

                #           Ichimoku Set            #
                if df['close'].iloc[i - 1] >= df['senkou_b'].iloc[i - 1] and \
                        df['close'].iloc[i] < df['senkou_b'].iloc[i] and \
                        df['close'].iloc[i] <= df['open'].iloc[i]:

                    for j in range(i + 1 - ichimoku_lookback, i + 1):

                        if df['close'].iloc[j - 1] >= df['senkou_a'].iloc[j - 1] and \
                                df['close'].iloc[j] < df['senkou_a'].iloc[j] and \
                                df['close'].iloc[i] <= df['open'].iloc[i]:

                            if 0 not in df['trade_state'].iloc[j:i].values:
                                df['trade_state'].iloc[i] = 0
                                df['ep'].iloc[i] = df['close'].iloc[i]
                                df['sl_level'].iloc[i] = df['high'].iloc[i + 1 - candle_lookback:i + 1].max()
                                df['tp_level'].iloc[i] = df['close'].iloc[i] + \
                                                         (df['close'].iloc[i] - df['high'].iloc[
                                                                                i + 1 - candle_lookback:i + 1].max()) * tp_gap
                                # df['leverage'].iloc[i] = int(target_sl_percent / (
                                #             abs(df['ep'].iloc[i] - df['sl_level'].iloc[i]) / df['ep'].iloc[i]))
                                # df['leverage'].iloc[i] = min(50, df['leverage'].iloc[i])
                                df['leverage'].iloc[i] = fixed_leverage
                                short_marker_x.append(i)
                                short_marker_y.append(df['close'].iloc[i])
                                break

        else:
            #           Long            #
            #           EMA Set         #
            # if df['low'].rolling(ema_lookback).min().iloc[i] >= df['ema'].iloc[i]:

                #           Ichimoku Set            #
                if df['close'].iloc[i - 1] <= df['senkou_b'].iloc[i - 1] and \
                        df['close'].iloc[i] > df['senkou_b'].iloc[i] and \
                        df['close'].iloc[i] >= df['open'].iloc[i]:

                    for j in range(i + 1 - ichimoku_lookback, i + 1):

                        if df['close'].iloc[j - 1] <= df['senkou_a'].iloc[j - 1] and \
                                df['close'].iloc[j] > df['senkou_a'].iloc[j] and \
                                df['close'].iloc[i] >= df['open'].iloc[i]:

                            if 1 not in df['trade_state'].iloc[j:i].values:
                                df['trade_state'].iloc[i] = 1
                                df['ep'].iloc[i] = df['close'].iloc[i]
                                df['sl_level'].iloc[i] = df['low'].iloc[i + 1 - candle_lookback:i + 1].min()
                                df['tp_level'].iloc[i] = df['close'].iloc[i] + \
                                                         (df['close'].iloc[i] - df['low'].iloc[
                                                                                i + 1 - candle_lookback:i + 1].min()) * tp_gap
                                # df['leverage'].iloc[i] = int(target_sl_percent / (
                                #             abs(df['ep'].iloc[i] - df['sl_level'].iloc[i]) / df['ep'].iloc[i]))
                                # df['leverage'].iloc[i] = min(50, df['leverage'].iloc[i])
                                df['leverage'].iloc[i] = fixed_leverage
                                long_marker_x.append(i)
                                long_marker_y.append(df['close'].iloc[i])
                                break

            #           Short            #
            #           EMA Set         #
            # elif df['high'].rolling(ema_lookback).max().iloc[i] <= df['ema'].iloc[i]:

                #           Ichimoku Set            #
                if df['close'].iloc[i - 1] >= df['senkou_a'].iloc[i - 1] and \
                        df['close'].iloc[i] < df['senkou_a'].iloc[i] and \
                        df['close'].iloc[i] <= df['open'].iloc[i]:

                    for j in range(i + 1 - ichimoku_lookback, i + 1):

                        if df['close'].iloc[j - 1] >= df['senkou_b'].iloc[j - 1] and \
                                df['close'].iloc[j] < df['senkou_b'].iloc[j] and \
                                df['close'].iloc[i] <= df['open'].iloc[i]:

                            if 0 not in df['trade_state'].iloc[j:i].values:
                                df['trade_state'].iloc[i] = 0
                                df['ep'].iloc[i] = df['close'].iloc[i]
                                df['sl_level'].iloc[i] = df['high'].iloc[i + 1 - candle_lookback:i + 1].max()
                                df['tp_level'].iloc[i] = df['close'].iloc[i] + \
                                                         (df['close'].iloc[i] - df['high'].iloc[
                                                                                i + 1 - candle_lookback:i + 1].max()) * tp_gap
                                # df['leverage'].iloc[i] = int(target_sl_percent / (
                                #             abs(df['ep'].iloc[i] - df['sl_level'].iloc[i]) / df['ep'].iloc[i]))
                                # df['leverage'].iloc[i] = min(50, df['leverage'].iloc[i])
                                df['leverage'].iloc[i] = fixed_leverage
                                short_marker_x.append(i)
                                short_marker_y.append(df['close'].iloc[i])
                                break

    # print(df.trade_state)
    # quit()

    # 거래 수수료
    fee = 0.0004 * 2

    # DatetimeIndex 를 지정해주기 번거로운 상황이기 때문에 틱을 기준으로 거래한다.
    # DatetimeIndex = df.axes[0]

    # ------------------- 상향 / 하향 매도 여부와 이익률 계산 -------------------#

    length = len(df.index) - 1  # 데이터 갯수 = 1, m = 0  >> 데이터 갯수가 100 개면 m 번호는 99 까지 ( 1 - 100 )

    # 병합할 dataframe 초기화
    close_state = pd.DataFrame(index=df.index, columns=['close_state'])
    condition = pd.DataFrame(index=df.index, columns=["Condition"])
    Profits = pd.DataFrame(index=df.index, columns=["Profits"])

    Profits.Profits = 1.0
    Loss = 1.0
    exit_fluc = 0
    exit_count = 0

    #                           Open Order Succeed 된 곳 표기                       #
    m = 0
    while m <= length:

        #           Open Order Condition 만족된 곳 찾기       #
        while True:
            if pd.notnull(df['ep'].iloc[m]):
                break
            m += 1
            if m > length:
                break

        if (m > length) or pd.isnull(df['ep'].iloc[m]):
            break

        # ep = df.iloc[m]['entry_price']
        # eprally["Eprally"].iloc[m] = ep

        condition.iloc[m] = 'Order Opened'

        m += 1
        if m > length:
            break

        #           Wait Oder execution          #
        # while True:
        #
        #     # 매수 체결 조건
        #     if df.iloc[m]['low'] <= eprally["Eprally"].iloc[m]:
        #         # and (df.iloc[m]['high'] != df.iloc[m]['low']):  # 조건을 만족할 경우 spp 생성
        #         # print(df['low'].iloc[m], '<=', bp, '?')
        #
        #         condition.iloc[m] = '매수 체결'
        #         # eprally["Eprally"].iloc[m] = eprally["Eprally"].iloc[m - 1]
        #
        #         m += 1
        #         break
        #     else:
        #
        #         m += 1
        #         if m > length:
        #             break
        #         if m - start_m >= wait_tick:
        #             break
        #
        #         condition.iloc[m] = '매수 대기'
        #         eprally["Eprally"].iloc[m] = eprally["Eprally"].iloc[m - 1]

    # if excel == 1:
    #     df = pd.merge(df, condition, how='outer', left_index=True, right_index=True)
    #     df = pd.merge(df, Profits, how='outer', left_index=True, right_index=True)
    #     df.to_excel("./ExcelCheck/%s BackTest %s.xlsx" % (Date, Coin))
    #     quit()

    #                               Close Order & TP Check                                  #
    m = 0
    spanlist_win = list()
    spanlist_lose = list()
    tp_marker_x = list()
    tp_marker_y = list()
    sl_marker_x = list()
    sl_marker_y = list()
    while m <= length:

        #           Check Opened Order          #
        while True:
            if condition.iloc[m]['Condition'] == 'Order Opened':
                break
            m += 1
            if m > length:
                break

        if (m > length) or pd.isnull(condition.iloc[m]['Condition']):
            break

        start_m = m
        open_signal_m = start_m

        order_type = None

        #           Check Order Type            #
        if df['trade_state'].iloc[start_m] not in [1, 0]:
            while True:
                open_signal_m -= 1
                if df['trade_state'].iloc[open_signal_m] in [1, 0]:
                    break
        if df['trade_state'].iloc[open_signal_m] == 1:
            order_type = 'Long'
        elif df['trade_state'].iloc[open_signal_m] == 0:
            order_type = 'Short'

        ep = df['ep'].iloc[open_signal_m]
        sl_level = df['sl_level'].iloc[open_signal_m]
        tp_level = df['tp_level'].iloc[open_signal_m]
        leverage = df['leverage'].iloc[open_signal_m]

        #       n TP        #
        tp_count = 3
        remain_qty = 1.
        profit = .0

        tp_list = list()
        if order_type == 'Long':

            for part_i in range(tp_count, 0, -1):
                tmp_tp_level = ep + abs(tp_level - ep) * (
                        part_i / tp_count)
                tp_list.append(tmp_tp_level)
        else:

            for part_i in range(tp_count, 0, -1):
                tmp_tp_level = ep - abs(tp_level - ep) * (
                        part_i / tp_count)
                tp_list.append(tmp_tp_level)

        qty_divider = 1.5
        qty_list = list()
        for qty_i in range(tp_count):

            if qty_i == tp_count - 1:
                qty_list.append(remain_qty)
            else:
                qty = remain_qty / qty_divider
                qty_list.append(qty)
                remain_qty -= qty

        while True:
            TP_switch = False
            SL_switch = False

            tp_level = tp_list[tp_count - 1]

            #       close_count == 1인 경우에만 Wait Close 기입해줄 필요가 발생한다. (in Excel)       #
            # if tp_count != 2:
            #     condition.iloc[m] = 'Wait Close'
            #     eprally["Eprally"].iloc[m] = eprally["Eprally"].iloc[m - 1]

            #               Check Close Condition            #
            #               Long Type           #
            if order_type == 'Long':

                #           TP by Level            #
                if df['high'].iloc[m] > tp_level:
                    close_state.iloc[m] = 'Level TP'
                    TP_switch = True

                #           SL by Signal            #
                # if df['trade_state'].iloc[m] == 0:
                #     SL_switch = True

                if df['close'].iloc[m - 1] >= np.minimum(df['senkou_a'].iloc[m - 1], df['senkou_b'].iloc[m - 1]) and \
                        df['close'].iloc[m] < np.minimum(df['senkou_a'].iloc[m], df['senkou_b'].iloc[m]):   # and \
                        # df['close'].iloc[m] <= df['open'].iloc[m]:
                    SL_switch = True

                #            SL by Price           #
                # if df['low'].iloc[m] < sl_level:
                #     close_state.iloc[m] = 'Price SL'
                #     SL_switch = True

            #               Short Type              #
            elif order_type == 'Short':

                #           TP by Level            #
                if df['low'].iloc[m] < tp_level:
                    close_state.iloc[m] = 'Level TP'
                    TP_switch = True

                #           SL by Signal            #
                # if df['trade_state'].iloc[m] == 1:
                #     SL_switch = True

                if df['close'].iloc[m - 1] <= np.maximum(df['senkou_a'].iloc[m - 1], df['senkou_b'].iloc[m - 1]) and \
                        df['close'].iloc[m] > np.maximum(df['senkou_a'].iloc[m], df['senkou_b'].iloc[m]):   # and \
                        # df['close'].iloc[m] >= df['open'].iloc[m]:
                    SL_switch = True

                #            SL by Price           #
                # if df['high'].iloc[m] > sl_level:
                #     close_state.iloc[m] = 'Price SL'
                #     SL_switch = True

            if not (TP_switch or SL_switch):

                m += 1
                if m > length:
                    break

                condition.iloc[m] = 'Wait Close'
                continue

            else:
                condition.iloc[m] = "Order Closed"
                exit_price = df['close'].iloc[m]

                if m == start_m:
                    close_m = m
                else:
                    close_m = m - 1

                if TP_switch:
                    tp_count -= 1
                    tp_marker_x.append(m)
                    tp_marker_y.append(tp_level)

                    profit_gap = ((tp_level / ep - fee) - 1) * qty_list[tp_count] * leverage
                    profit += profit_gap

                else:
                    tp_count = 0
                    sl_marker_x.append(m)
                    sl_marker_y.append(df['close'].iloc[m])

                    profit_gap = ((exit_price / ep - fee) - 1) * qty_list[tp_count] * leverage
                    profit += profit_gap

                if tp_count == 0:

                    if order_type == 'Long':
                        Profits.iloc[m] = 1 + profit
                    else:
                        Profits.iloc[m] = 1 - profit

                    if float(Profits.iloc[m]) < 1:
                        # print('Minus Profits at %s %.3f' % (m, float(Profits.iloc[m])))
                        Loss *= float(Profits.iloc[m])

                        #           Check Final Close       #
                        try:
                            spanlist_lose.append((start_m, m))

                        except Exception as e:
                            pass

                    else:
                        exit_count += 1

                        try:
                            spanlist_win.append((start_m, m))

                        except Exception as e:
                            pass

                    #           Trade Done          #
                    break

                # #           Partial 시, 같은 곳에 Close 두번하지 않기 위함          #
                # m += 1
                # if m > length:
                #     break

        if m > length:
            break

    df = pd.merge(df, close_state, how='outer', left_index=True, right_index=True)
    # df = pd.merge(df, leverage, how='outer', left_index=True, right_index=True)
    # df = pd.merge(df, sl_level, how='outer', left_index=True, right_index=True)
    df = pd.merge(df, condition, how='outer', left_index=True, right_index=True)
    df = pd.merge(df, Profits, how='outer', left_index=True, right_index=True)

    if excel == 1:
        # df.to_excel("./BackTest/%s BackTest %s.xlsx" % (Date, Coin))
        df.to_excel("ExcelCheck/%s BackTest.xlsx" % symbol)

    profits = Profits.cumprod()  # 해당 열까지의 누적 곱!

    if exit_count != 0:
        exit_fluc_mean = exit_fluc / exit_count
    else:
        exit_fluc_mean = 1.

    if np.isnan(profits.iloc[-1].item()):
        return 1.0, 1.0, 1.0, 1.0, 1.0, 1.0

    # [-1] 을 사용하려면 데이터가 존재해야 되는데 데이터 전체가 결측치인 경우가 존재한다.
    if len(profits) == 0:
        return 1.0, 1.0, 1.0, 1.0, 1.0, 1.0

    elif get_fig == 1:

        df = df.reset_index(drop=True)
        # elif get_fig == 1 and float(profits.iloc[-1]) != 1:

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(211)
        ohlc = df.iloc[:, :4]
        # ohlc = df2.iloc[:, :4]
        index = np.arange(len(ohlc))
        ohlc = np.hstack((np.reshape(index, (-1, 1)), ohlc))
        mf.candlestick_ohlc(ax, ohlc, width=0.5, colorup='r', colordown='b')

        plt.plot(df['senkou_a'], 'g')
        plt.plot(df['senkou_b'], 'fuchsia')
        plt.plot(long_marker_x, long_marker_y, '^', color='limegreen', markersize=10)
        plt.plot(short_marker_x, short_marker_y, 'v', color='orange', markersize=10)
        plt.plot(sl_marker_x, sl_marker_y, 'o', color='red')
        plt.plot(tp_marker_x, tp_marker_y, 'o', color='b')

        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        # plt.xlim(df.index[0], df.index[-1])
        # plt.legend(loc='auto', fontsize=10)
        plt.title('%.3f %.3f %.3f' % (float(profits.iloc[-1]),
                                      float(profits.iloc[-1]) / Loss,
                                      Loss))

        for trade_num in range(len(spanlist_win)):
            plt.axvspan(spanlist_win[trade_num][0], spanlist_win[trade_num][1], facecolor='c', alpha=0.5)
        for trade_num in range(len(spanlist_lose)):
            plt.axvspan(spanlist_lose[trade_num][0], spanlist_lose[trade_num][1], facecolor='m', alpha=0.5)

        plt.subplot(212)
        plt.plot(df['Profits'].cumprod())

        # plot 저장 & 닫기
        plt.show()
        # plt.savefig("./ChartCheck/%s.png" % symbol, dpi=300)
        #
        # plt.show(block=False)
        # plt.pause(3)
        plt.close()

    if profits.values[-1] != 1.:
        profits_sum = 0
        for i in range(len(Profits)):
            if Profits.values[i] != 1.:
                profits_sum += Profits.iloc[i]
        profits_avg = profits_sum / sum(Profits.values != 1.)

    else:
        profits_avg = [1.]

    # print(Profits.values.min())
    # quit()
    # print(Profits.values != 1.)
    # print(profits_avg)
    # quit()
    # std = np.std(max_abs_scaler(df[df['trade_state'] == 2.]['MACD_OSC']))
    # print(std)

    return float(profits.iloc[-1]), float(profits.iloc[-1]) / Loss, Loss, profits_avg[
        0], Profits.values.min(), exit_fluc_mean


if __name__ == "__main__":

    home_dir = os.path.expanduser('~')
    dir = './candlestick_concated/'

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
        os.mkdir("./ChartCheck/%s/" % (date))

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
