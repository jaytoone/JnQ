from binance_f import RequestClient
from binance_f.model import *
from binance_f.constant.test import *
from binance_f.base.printobject import *
import numpy as np
import pandas as pd
from datetime import datetime
import os
from scipy import stats
from asq.initiators import query
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.preprocessing import MaxAbsScaler, StandardScaler
import mpl_finance as mf
import time
from Funcs_For_Trade import *
from Funcs_Indicator import *
import math

pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 2500)
pd.set_option('display.max_columns', 2500)


def min_max_scaler(x):
    scaled_x = (x - x.min()) / (x.max() - x.min())
    return scaled_x


request_client = RequestClient(api_key=g_api_key, secret_key=g_secret_key)


def profitage(coin, period1=30, period2=60, period3=120, date='2019-09-25', excel=0, get_fig=0):
    global length

    if not coin.endswith('.xlsx'):
        coin = coin + 'USDT'
        df = request_client.get_candlestick_data(symbol=coin, interval=CandlestickInterval.MIN1,
                                                 startTime=None, endTime=None, limit=1500)
        second_df = request_client.get_candlestick_data(symbol=coin, interval=CandlestickInterval.MIN3,
                                                        startTime=None, endTime=None, limit=600)
        third_df = request_client.get_candlestick_data(symbol=coin, interval=CandlestickInterval.MIN30,
                                                       startTime=None, endTime=None, limit=70)
    else:
        # limit = None  # <-- should be None, if you want all data from concated chart
        df = pd.read_excel(os.path.join(dir, '1m', coin), index_col=0)
        second_df = pd.read_excel(os.path.join(dir, '3m', coin), index_col=0)
        third_df = pd.read_excel(os.path.join(dir, '30m', coin), index_col=0)

    chart_height = (df['high'].max() - df['low'].min())
    chart_gap = (df['high'].max() / df['low'].min())
    ha_second_df = heikinashi(second_df)
    ha_third_df = heikinashi(third_df)

    tc_upper = 0.35
    tc_lower = -tc_upper
    # print(ha_second_df.tail(10))
    # quit()

    second_df['minor_ST1_Up'], second_df['minor_ST1_Down'], second_df['minor_ST1_Trend'] = supertrend(second_df, 10, 2)
    second_df['minor_ST2_Up'], second_df['minor_ST2_Down'], second_df['minor_ST2_Trend'] = supertrend(ha_second_df, 7,
                                                                                                      2)
    second_df['minor_ST3_Up'], second_df['minor_ST3_Down'], second_df['minor_ST3_Trend'] = supertrend(ha_second_df, 7,
                                                                                                      2.5)
    # print(df.head(20))
    # quit()
    # start = time.time()

    df = df.join(pd.DataFrame(index=df.index, data=to_lower_tf(df, second_df, [i for i in range(-9, 0, 1)], 3),
                              columns=['minor_ST1_Up', 'minor_ST1_Down', 'minor_ST1_Trend'
                                  , 'minor_ST2_Up', 'minor_ST2_Down', 'minor_ST2_Trend'
                                  , 'minor_ST3_Up', 'minor_ST3_Down', 'minor_ST3_Trend']))

    third_df['major_ST1_Up'], third_df['major_ST1_Down'], third_df['major_ST1_Trend'] = supertrend(third_df, 10, 2)
    third_df['major_ST2_Up'], third_df['major_ST2_Down'], third_df['major_ST2_Trend'] = supertrend(ha_third_df, 7,
                                                                                                   2)
    third_df['major_ST3_Up'], third_df['major_ST3_Down'], third_df['major_ST3_Trend'] = supertrend(ha_third_df, 7,
                                                                                                   2.5)

    df = df.join(pd.DataFrame(index=df.index, data=to_lower_tf(df, third_df, [i for i in range(-9, 0, 1)], 30),
                              columns=['major_ST1_Up', 'major_ST1_Down', 'major_ST1_Trend'
                                  , 'major_ST2_Up', 'major_ST2_Down', 'major_ST2_Trend'
                                  , 'major_ST3_Up', 'major_ST3_Down', 'major_ST3_Trend']))

    # print('elapsed time :', time.time() - start)

    # atrDown = np.max(df.iloc[:, [-8, -5, -2]].values, axis=1)
    # atrUp = np.min(df.iloc[:, [-9, -6, -3]].values, axis=1)
    # df['ST_Osc'] = (df['close'] - atrDown) / (atrUp - atrDown) * 100
    # df['ST1_Osc'] = (df['close'] - df['minor_ST1_Down']) / (df['minor_ST1_Up'] - df['minor_ST1_Down']) * 100

    df['30EMA'] = ema(df['close'], 30)
    df['100EMA'] = ema(df['close'], 100)

    df['CB'], df['EMA_CB'] = cct_bbo(df, 18, 13)
    # print(df.tail(10))
    # quit()

    df['Fisher1'] = fisher(df, period=period1)
    df['Fisher2'] = fisher(df, period=period2)
    df['Fisher3'] = fisher(df, period=period3)

    df['Trix'] = trix_hist(df, 14, 1, 9)
    second_df['Trix'] = trix_hist(second_df, period=14, multiplier=1, signal_period=9)
    df = df.join(pd.DataFrame(index=df.index, data=to_lower_tf(df, second_df, [-1], 3), columns=['minor_Trix']))

    # print(df.tail())
    # print(df.iloc[:, -8:-2].tail(20))
    # quit()

    #                               Open Condition                              #
    #       Open Long / Short = 1 / 2       #
    #       Close Long / Short = -1 / -2        #
    upper = 0.5
    lower = -upper

    long_marker_x = list()
    long_marker_y = list()
    short_marker_x = list()
    short_marker_y = list()

    tick_in_box = 500
    sl_least_gap_ratio = 1 / 10
    tp_ratio = 3
    df['ep'] = df['close']
    df['long_sl'] = df['low'].rolling(tick_in_box).min()
    df['long_tp'] = df['ep'] + abs(df['ep'] - df['long_sl']) * tp_ratio

    df['short_sl'] = df['high'].rolling(tick_in_box).max()
    df['short_tp'] = df['ep'] - abs(df['ep'] - df['short_sl']) * tp_ratio
    df['trade_state'] = np.nan

    # for i in range(tick_in_box, len(df)):
    #
    #     #           Long            #
    #     if df['ep'].iloc[i] >= df['long_sl'].iloc[i] + (
    #             df['short_sl'].iloc[i] - df['long_sl'].iloc[i]) * sl_least_gap_ratio:
    #
    #         for j in range(i + 1, len(df)):
    #             if df['high'].iloc[j] > df['long_tp'].iloc[i]:
    #                 df['trade_state'].iloc[i] = 1
    #                 long_marker_x.append(i), long_marker_y.append(df['ep'].iloc[i])
    #                 break
    #             elif df['low'].iloc[j] < df['long_sl'].iloc[i]:
    #                 df['trade_state'].iloc[i] = 0
    #                 short_marker_x.append(i), short_marker_y.append(df['ep'].iloc[i])
    #                 break
    #
    #     #           Short            #
    #     # if df['ep'].iloc[i] <= df['short_sl'].iloc[i] - \
    #     #         abs(df['long_sl'].iloc[i] - df['short_sl'].iloc[i]) * sl_least_gap_ratio:
    #     #
    #     #     for j in range(i + 1, len(df)):
    #     #         if df['low'].iloc[j] < df['short_tp'].iloc[i]:
    #     #             df['trade_state'].iloc[i] = 1
    #     #             long_marker_x.append(i), long_marker_y.append(df['ep'].iloc[i])
    #     #             break
    #     #         elif df['high'].iloc[j] > df['short_sl'].iloc[i]:
    #     #             df['trade_state'].iloc[i] = 0
    #     #             short_marker_x.append(i), short_marker_y.append(df['ep'].iloc[i])
    #     #             break
    #
    # # print(df.trade_state)
    # # quit()
    #
    # #           Show Result Image         #
    # if get_fig:
    #     df = df.reset_index(drop=True)
    #
    #     plt.plot(df['close'])
    #     # plt.plot(df['long_sl'])
    #     # plt.plot(df['long_tp'])
    #     plt.plot(long_marker_x, long_marker_y, 'o', color='blue')
    #     plt.plot(short_marker_x, short_marker_y, 'o', color='red')
    #     plt.show()

    #       Save Excel      #
    if excel:
        if coin.endswith('.xlsx'):
            # df.to_excel("Labeled_Data/%s" % coin)
            df.to_excel("Labeled_Data/TP_Ratio_Test/%s/%s" % (tick_in_box, coin))
        else:
            df.to_excel("Labeled_Data/%s %s.xlsx" % (date, coin))

    return

    df['entry_price'] = np.where((df['trade_state'] == 1) | (df['trade_state'] == 2), df['close'], np.nan)
    # df['entry_price'] = df['entry_price'].apply(clearance)

    # 거래 수수료
    fee = 0.0004 * 2

    # DatetimeIndex 를 지정해주기 번거로운 상황이기 때문에 틱을 기준으로 거래한다.
    # DatetimeIndex = df.axes[0]

    # ------------------- 상향 / 하향 매도 여부와 이익률 계산 -------------------#

    length = len(df.index) - 1  # 데이터 갯수 = 1, m = 0  >> 데이터 갯수가 100 개면 m 번호는 99 까지 ( 1 - 100 )

    # 병합할 dataframe 초기화
    trade_state_close = pd.DataFrame(index=df.index, columns=['trade_state'])
    leverage = pd.DataFrame(index=df.index, columns=['Leverage'])
    eprally = pd.DataFrame(index=df.index, columns=["Eprally"])
    sl_level = pd.DataFrame(index=df.index, columns=["SL_Level"])
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
            if pd.notnull(df.iloc[m]['entry_price']):
                break
            m += 1
            if m > length:
                break

        if (m > length) or pd.isnull(df.iloc[m]['entry_price']):
            break

        ep = df.iloc[m]['entry_price']
        eprally["Eprally"].iloc[m] = ep

        #                   Set Leverage                    #
        #       Long -> Find Lowest Low / Short -> Find Highest High        #
        leverage_target_level = None
        # offset_percentage = 1 / 2
        target_percentage = 0.1

        if df['trade_state'].iloc[m] == 1:
            leverage_target_level = atrUp[m]
        else:
            leverage_target_level = atrDown[m]

        #       주문 등록 완료, 체결 대기     #
        # start_m = m
        # print('leverage_target_level :', leverage_target_level)
        # sl_level['SL_Level'].iloc[m] = leverage_target_level
        leverage['Leverage'].iloc[m] = int(target_percentage / (abs(ep - leverage_target_level) / ep))
        if leverage['Leverage'].iloc[m] > 50:
            leverage['Leverage'].iloc[m] = 50
        condition.iloc[m] = 'Order Opened'

        m += 1
        if m > length:
            break

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
    #     df = pd.merge(df, leverage, how='outer', left_index=True, right_index=True)
    #     df = pd.merge(df, eprally, how='outer', left_index=True, right_index=True)
    #     df = pd.merge(df, sl_level, how='outer', left_index=True, right_index=True)
    #     df = pd.merge(df, condition, how='outer', left_index=True, right_index=True)
    #     df = pd.merge(df, Profits, how='outer', left_index=True, right_index=True)
    #     df.to_excel("./ExcelCheck/%s BackTest %s.xlsx" % (date, coin))
    #     quit()

    #                               Close Order & TP Check                                  #
    m = 0
    spanlist_win = []
    spanlist_lose = []
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
        if df['trade_state'].iloc[start_m] not in [1, 2]:
            while True:
                open_signal_m -= 1
                if df['trade_state'].iloc[open_signal_m] in [1, 2]:
                    break
        if df['trade_state'].iloc[open_signal_m] == 1:
            order_type = 'Long'
        elif df['trade_state'].iloc[open_signal_m] == 2:
            order_type = 'Short'

        sl_level_ = sl_level['SL_Level'].iloc[open_signal_m]
        leverage_ = leverage['Leverage'].iloc[open_signal_m]

        #       2 TP        #
        close_count = 2
        tp_count = 0
        p1, p2 = 0.0, 0.0
        # trix_out = False
        # fisher_cross_zero = False
        # sl_fisher_level_none_tp = 1
        # sl_fisher_level = 0
        while True:
            TP_switch = False
            SL_switch = False

            #       close_count == 1인 경우에만 Wait Close 기입해줄 필요가 발생한다. (in Excel)       #
            if close_count != 2:
                condition.iloc[m] = 'Wait Close'
                eprally["Eprally"].iloc[m] = eprally["Eprally"].iloc[m - 1]

            #               Check Close Condition            #
            #               Long Type           #
            if order_type == 'Long':

                #           TP by Signal            #
                if close_count == 2:
                    if df['high'].iloc[m] > atrUp[m]:
                        trade_state_close.iloc[m] = 'Level TP'
                        TP_switch = True

                elif close_count == 1:
                    if trade_state_close.iloc[m - 1].item() != 'Level TP':
                        if df['ST1_Osc'].iloc[m - 2] < df['ST1_Osc'].iloc[m - 1] > df['ST1_Osc'].iloc[m]:
                            if df['ST1_Osc'].iloc[m - 1] >= 95:
                                trade_state_close.iloc[m] = 'Signal TP'
                                TP_switch = True

                #           SL by Signal            #
                if df['close'].iloc[m] <= df['30EMA'].iloc[m]:
                    trade_state_close.iloc[m] = 'Signal SL'
                    SL_switch = True

                #            SL by Price           #
                # if df['low'].iloc[m] < sl_level_:
                #     trade_state_close.iloc[m] = 'Price SL'
                #     SL_switch = True

            #               Short Type              #
            elif order_type == 'Short':

                #           TP by Signal            #
                if close_count == 2:
                    if df['low'].iloc[m] < atrDown[m]:
                        trade_state_close.iloc[m] = 'Level TP'
                        TP_switch = True

                elif close_count == 1:
                    if trade_state_close.iloc[m - 1].item() != 'Level TP':
                        if df['ST1_Osc'].iloc[m - 2] > df['ST1_Osc'].iloc[m - 1] < df['ST1_Osc'].iloc[m]:
                            if df['ST1_Osc'].iloc[m - 1] <= 5:
                                trade_state_close.iloc[m] = 'Signal TP'
                                TP_switch = True

                #           SL by Signal            #
                if df['close'].iloc[m] > df['30EMA'].iloc[m]:
                    trade_state_close.iloc[m] = 'Signal SL'
                    SL_switch = True

                #           SL by Price           #
                # if df['high'].iloc[m] > sl_level_:
                #     trade_state_close.iloc[m] = 'Price SL'
                #     SL_switch = True

            if not (TP_switch or SL_switch):

                m += 1
                if m > length:
                    break

                condition.iloc[m] = 'Wait Close'
                eprally["Eprally"].iloc[m] = eprally["Eprally"].iloc[m - 1]
                continue

            else:
                condition.iloc[m] = "Order Closed"

                if TP_switch:
                    close_count -= 1
                    tp_count += 1
                    tp_marker_x.append(m)
                    tp_marker_y.append(df['ST1_Osc'].iloc[m])
                else:
                    close_count = 0
                    sl_marker_x.append(m)
                    sl_marker_y.append(df['30EMA'].iloc[m])

                exit_price = df['close'].iloc[m]
                if m == start_m:
                    close_m = m
                else:
                    close_m = m - 1

                if close_count == 1:
                    p1 = ((exit_price / eprally["Eprally"].iloc[close_m] - fee) - 1) / 2
                elif close_count == 0:
                    p2 = ((exit_price / eprally["Eprally"].iloc[close_m] - fee) - 1) / 2

                    #       1. 1 TP + SL 을 생각해야한다.         #
                    #       2. 0 TP + SL 을 생각해야한다.         #
                    if tp_count == 0:
                        p1 = p2

                if close_count == 0:

                    Profits.iloc[m] = 1 + p1 + p2

                    #           Consider Short Type Profit          #
                    #           결과가 반대로 나오면.. Long.. ?          #
                    if order_type == 'Short':
                        Profits.iloc[m] = 1 - (p1 + p2)

                    #           Adjust Leverage             #
                    Profits.iloc[m] = (Profits.iloc[m] - 1) * leverage_ + 1

                    if float(Profits.iloc[m]) < 1:
                        # print('Minus Profits at %s %.3f' % (m, float(Profits.iloc[m])))
                        Loss *= float(Profits.iloc[m])

                        #           Check Final Close       #
                        try:
                            spanlist_lose.append((start_m, m))

                        except Exception as e:
                            pass

                    else:
                        exit_fluc += df.low.iloc[start_m:m].min() / eprally["Eprally"].iloc[m - 1]
                        exit_count += 1

                        try:
                            spanlist_win.append((start_m, m))

                        except Exception as e:
                            pass

                #           Partial 시, 같은 곳에 Close 두번하지 않기 위함          #
                m += 1
                if m > length:
                    break

                if close_count == 0:
                    break

        if m > length:
            # print(condition.iloc[m - 1])
            # condition.iloc[m - 1] = "Order Closed"
            #
            # #       매수 체결 후 바로 매도 체결 되어버리는 경우       #
            # if m == start_m:
            #     Profits.iloc[m - 1] = df.iloc[m - 1]['close'] / eprally["Eprally"].iloc[m] - fee
            # else:
            #     Profits.iloc[m - 1] = df.iloc[m - 1]['close'] / eprally["Eprally"].iloc[m - 1] - fee
            #
            # if float(Profits.iloc[m - 1]) < 1:
            #     Loss *= float(Profits.iloc[m - 1])
            #
            #     try:
            #         spanlist_lose.append((start_m, m - 1))
            #
            #     except Exception as e:
            #         pass
            #
            # else:
            #     exit_fluc += df.low.iloc[start_m:m - 1].min() / eprally["Eprally"].iloc[m - 2]
            #     exit_count += 1
            #
            #     try:
            #         spanlist_win.append((start_m, m - 1))
            #
            #     except Exception as e:
            #         pass
            break
        # else:
        #     m += 1
        #     if m > length:
        #         break
        #     condition.iloc[m] = np.nan

    df = pd.merge(df, trade_state_close, how='outer', left_index=True, right_index=True)
    df = pd.merge(df, leverage, how='outer', left_index=True, right_index=True)
    df = pd.merge(df, eprally, how='outer', left_index=True, right_index=True)
    df = pd.merge(df, sl_level, how='outer', left_index=True, right_index=True)
    df = pd.merge(df, condition, how='outer', left_index=True, right_index=True)
    df = pd.merge(df, Profits, how='outer', left_index=True, right_index=True)

    if excel == 1:
        # df.to_excel("./BackTest/%s BackTest %s.xlsx" % (date, coin))
        df.to_excel("ExcelCheck/%s BackTest %s.xlsx" % (date, coin))

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
        ax = fig.add_subplot(411)
        ohlc = df.iloc[:, :4]
        # ohlc = df2.iloc[:, :4]
        index = np.arange(len(ohlc))
        ohlc = np.hstack((np.reshape(index, (-1, 1)), ohlc))
        mf.candlestick_ohlc(ax, ohlc, width=0.5, colorup='r', colordown='b')

        plt.plot(df['30EMA'])
        plt.plot(df['100EMA'])
        plt.plot(sl_marker_x, sl_marker_y, 'o', color='red')

        # plt.plot(df.iloc[:, -14:-11], '.', markersize=2, color='orangered')
        # plt.plot(df.iloc[:, -11:-8], '.', markersize=4, color='olive')
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        # plt.xlim(df.index[0], df.index[-1])
        # plt.legend(loc='auto', fontsize=10)
        plt.title('2ST Combo Checker - %s %.3f %.3f %.3f\n chart_height = %.3f' % (coin,
                                                                                   float(profits.iloc[-1]),
                                                                                   float(profits.iloc[-1]) / Loss,
                                                                                   Loss, chart_gap))

        for trade_num in range(len(spanlist_win)):
            plt.axvspan(spanlist_win[trade_num][0], spanlist_win[trade_num][1], facecolor='c', alpha=0.5)
        for trade_num in range(len(spanlist_lose)):
            plt.axvspan(spanlist_lose[trade_num][0], spanlist_lose[trade_num][1], facecolor='m', alpha=0.5)

        plt.subplot(412)
        plt.plot(df['EMA_CB'], color='b')
        plt.plot(long_marker_x, long_marker_y, 'o', color='c')
        plt.plot(short_marker_x, short_marker_y, 'o', color='m', alpha=0.7)
        plt.axhline(50, linestyle='--')

        for trade_num in range(len(spanlist_win)):
            plt.axvspan(spanlist_win[trade_num][0], spanlist_win[trade_num][1], facecolor='c', alpha=0.5)
        for trade_num in range(len(spanlist_lose)):
            plt.axvspan(spanlist_lose[trade_num][0], spanlist_lose[trade_num][1], facecolor='m', alpha=0.5)

        plt.subplot(413)
        plt.plot(df['Fisher1'], color='lime')
        plt.axhline(0, linestyle='--')

        for trade_num in range(len(spanlist_win)):
            plt.axvspan(spanlist_win[trade_num][0], spanlist_win[trade_num][1], facecolor='c', alpha=0.5)
        for trade_num in range(len(spanlist_lose)):
            plt.axvspan(spanlist_lose[trade_num][0], spanlist_lose[trade_num][1], facecolor='m', alpha=0.5)

        plt.subplot(414)
        plt.plot(df['ST1_Osc'], color='lime')
        plt.plot(df['ST_Osc'], color='olive')
        plt.plot(tp_marker_x, tp_marker_y, 'o', color='blue')
        plt.axhline(75, linestyle='--')
        plt.axhline(25, linestyle='--')
        plt.axhline(95, linestyle='--')
        plt.axhline(5, linestyle='--')

        for trade_num in range(len(spanlist_win)):
            plt.axvspan(spanlist_win[trade_num][0], spanlist_win[trade_num][1], facecolor='c', alpha=0.5)
        for trade_num in range(len(spanlist_lose)):
            plt.axvspan(spanlist_lose[trade_num][0], spanlist_lose[trade_num][1], facecolor='m', alpha=0.5)

        # plot 저장 & 닫기
        plt.show()
        plt.savefig("./ChartCheck/%s/%s.png" % (date, coin), dpi=300)
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
