import pybithumb
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
from scipy.ndimage.filters import gaussian_filter1d
from finta import TA
from Funcs_For_Trade import *
from Funcs_Indicator import *

pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 2500)
pd.set_option('display.max_columns', 2500)


def min_max_scaler(x):
    scaled_x = (x - x.min()) / (x.max() - x.min())
    return scaled_x


def profitage(Coin, long_signal_value, short_signal_value, long_period, short_period, wait_tick=3, Date='2019-09-25',
              excel=0, get_fig=0):
    global length

    if not Coin.endswith('.xlsx'):
        df = pybithumb.get_candlestick(Coin, 'KRW', '1m')
        second_df = pybithumb.get_candlestick(Coin, 'KRW', '5m')
        third_df = pybithumb.get_candlestick(Coin, 'KRW', '3m')
    else:
        df = pd.read_excel(dir + '%s' % Coin, index_col=0)
        second_df = pd.read_excel(dir.replace('ohlcv', 'ohlcv_5m') + '%s' % Coin, index_col=0)
        third_df = pd.read_excel(dir.replace('ohlcv', 'ohlcv_3m') + '%s' % Coin, index_col=0)

    chart_height = (df['high'].max() - df['low'].min())
    chart_gap = (df['high'].max() / df['low'].min())

    df['MA_Short'] = df['close'].rolling(20).mean()
    df['MA_Long'] = df['close'].rolling(120).mean()

    df['Fisher_Long'] = TA.FISH(df, period=long_period)
    df['FISHER_TREND'] = np.where(df['Fisher_Long'] > df['Fisher_Long'].shift(1), 'UP', 'DOWN')

    second_df['MA_Long'] = second_df['close'].rolling(120).mean()
    second_df['Fisher_Long'] = TA.FISH(second_df, period=long_period)

    third_df['MA_Long'] = third_df['close'].rolling(120).mean()
    third_df['Fisher_Long'] = TA.FISH(third_df, period=long_period)

    df2 = convert_df(df, second_df, 5)
    df3 = convert_df(df, third_df, 3)

    support_line(df, 5)
    offset_upper, offset_lower = 0.05, 0.02
    df['Support_Line_Offset_Upper'] = df['Support_Line'] + offset_upper * chart_height
    df['Support_Line_Offset_Lower'] = df['Support_Line'] - offset_lower * chart_height

    #           Realtime Curve          #
    # df['Gaussian_close_Trend'] = np.NaN
    # period = 30
    # sigma = 10
    # for i in range(period, len(df)):
    #     y = df['close'].values[i + 1 - period:i + 1]
    #     y_smooth = gaussian_filter1d(y, sigma=sigma)
    #     if y_smooth[-1] > y_smooth[-2]:
    #         df['Gaussian_close_Trend'].iloc[i] = 1
    #     else:
    #         df['Gaussian_close_Trend'].iloc[i] = 0

    detect_close_value = long_signal_value

    #                               LONG SHORT                              #
    trade_state = [0] * len(df)
    for i in range(2, len(df)):

        if df['Fisher_Long'].iloc[i - 1] <= long_signal_value:
            if df['FISHER_TREND'].iloc[i] == 'UP' and df['FISHER_TREND'].iloc[i - 1] == 'DOWN':
                trade_state[i] = 1.
                if df['close'].iloc[i] >= df2['MA_Long'].iloc[i]:
                    #       Check Trend     #
                    if df['Gaussian_close_Trend'].iloc[i]:
                        copy_i = i
                        while copy_i >= 1:
                            copy_i -= 1
                            if df['Support_Line'].iloc[copy_i] != df['Support_Line'].iloc[i]:
                                df['Support_Line'].iloc[copy_i + 1:i + 1] = df['Support_Line'].iloc[copy_i]
                                df['Support_Line_Offset_Upper'].iloc[copy_i + 1:i + 1] = df['Support_Line_Offset_Upper'].iloc[copy_i]
                                df['Support_Line_Offset_Lower'].iloc[copy_i + 1:i + 1] = df['Support_Line_Offset_Lower'].iloc[copy_i]
                                # print('copy_i, i  :', copy_i, i)
                                break

                    #       Check Price Under S.P       #
                    # copy_i = i
                    # while copy_i >= 1:

                    if df['Support_Line_Offset_Upper'].iloc[i] > df['close'].iloc[i] > df['Support_Line_Offset_Lower'].iloc[i]:
                        trade_state[i] = 1.5

        if df['Fisher_Long'].iloc[i - 1] >= short_signal_value:
            if df['FISHER_TREND'].iloc[i] == 'DOWN' and df['FISHER_TREND'].iloc[i - 1] == 'UP':
                trade_state[i] = 2
    df['trade_state'] = trade_state
    # print(df.trade_state)
    # quit()

    span_list_entry_candis = list()
    span_list_entry = list()
    span_list_fisher_gaussian_exit = list()
    span_list_gaussian_entry = list()
    span_list_bull = list()
    span_list_bear = list()
    target = df['trade_state']
    for i in range(1, len(target)):
        if target.iloc[i] == 1:
            span_list_entry_candis.append((i, i + 1))
        if target.iloc[i] == 1.5:
            span_list_entry.append((i, i + 1))
        if df['Gaussian_close_Trend'].iloc[i]:
            span_list_bull.append((i, i + 1))
        else:
            span_list_bear.append((i, i + 1))

    # 매수 시점 = 급등 예상시, 매수가 = 이전 종가
    # df['BuyPrice'] = np.where(df['trade_state']== 1., df['close'], np.nan)
    df['BuyPrice'] = np.where(df['trade_state'] == 1.5, df['close'], np.nan)
    # df['BuyPrice'] = np.where(df['trade_state'] == 1.5, df['close'].shift(1), np.nan)
    # df['BuyPrice'] = df['BuyPrice'].apply(clearance)

    # 거래 수수료
    fee = 0.005

    # DatetimeIndex 를 지정해주기 번거로운 상황이기 때문에 틱을 기준으로 거래한다.
    # DatetimeIndex = df.axes[0]

    # ------------------- 상향 / 하향 매도 여부와 이익률 계산 -------------------#

    # high 가 SPP 를 건드리거나, low 가 SPM 을 건드리면 매도 체결 [ 매도 체결될 때까지 SPP 와 SPM 은 유지 !! ]
    length = len(df.index) - 1  # 데이터 갯수 = 1, m = 0  >> 데이터 갯수가 100 개면 m 번호는 99 까지 ( 1 - 100 )

    # 병합할 dataframe 초기화
    bprelay = pd.DataFrame(index=df.index, columns=['bprelay'])
    condition = pd.DataFrame(index=df.index, columns=["Condition"])
    Profits = pd.DataFrame(index=df.index, columns=["Profits"])
    support = pd.DataFrame(index=df.index, columns=["support_line"])
    # price_point = pd.DataFrame(index=np.arange(len(df)), columns=['Price_point'])

    Profits.Profits = 1.0
    Minus_Profits = 1.0
    exit_fluc = 0
    exit_count = 0

    # 오더라인과 매수가가 정해진 곳에서부터 일정시간까지 오더라인과 매수가를 만족할 때까지 대기  >> 일정시간,
    m = 0
    while m <= length:

        while True:  # bp 찾기
            if pd.notnull(df.iloc[m]['BuyPrice']):
                break
            m += 1
            if m > length:
                break

        if (m > length) or pd.isnull(df.iloc[m]['BuyPrice']):
            break

        bp = df.iloc[m]['BuyPrice']
        bprelay["bprelay"].iloc[m] = bp

        # sp = min(df['low'].iloc[ip_before:ip_after])
        # bp = clearance(sp * 1.0025)

        #       매수 등록 완료, 매수 체결 대기     #
        start_m = m
        while True:
            # support["support_line"].iloc[m] = sp

            # 매수 체결 조건
            if df.iloc[m]['low'] <= bprelay["bprelay"].iloc[m]:
                # and (df.iloc[m]['high'] != df.iloc[m]['low']):  # 조건을 만족할 경우 spp 생성
                # print(df['low'].iloc[m], '<=', bp, '?')

                condition.iloc[m] = '매수 체결'
                # bprelay["bprelay"].iloc[m] = bprelay["bprelay"].iloc[m - 1]

                m += 1
                break
            else:
                #   SINGAL 다음 시점에서 CLOSE와 매수가의 GAP이 0.6% 미만이면, CLOSE로 매수한다.
                if df['close'].iloc[m] / bp <= 1.006:
                    bprelay["bprelay"].iloc[m] = df['close'].iloc[m]
                    continue

                m += 1
                if m > length:
                    break
                if m - start_m >= wait_tick:
                    break

                condition.iloc[m] = '매수 대기'
                bprelay["bprelay"].iloc[m] = clearance(bprelay["bprelay"].iloc[m - 1] * 1.0015)
                support["support_line"].iloc[m] = support["support_line"].iloc[m - 1]

    # if excel == 1:
    #     df = pd.merge(df, condition, how='outer', left_index=True, right_index=True)
    #     df = pd.merge(df, bprelay, how='outer', left_index=True, right_index=True)
    #     df = pd.merge(df, support, how='outer', left_index=True, right_index=True)
    #     df.to_excel("./excel_check/%s BackTest %s.xlsx" % (Date, Coin))
    #     quit()

    #           지정 매도가 표시 완료           #

    #                               수익성 검사                                  #
    m = 0
    spanlist = []
    spanlist_limit = []
    spanlist_breakaway = []
    while m <= length:  # 초반 시작포인트 찾기

        while True:  # SPP 와 SPM 찾긴
            if condition.iloc[m]['Condition'] == '매수 체결':
                # and type(df.iloc[m]['SPP']) != str:  # null 이 아니라는 건 오더라인과 매수가로 캡쳐했다는 거
                break
            m += 1
            if m > length:  # 차트가 끊나는 경우, 만족하는 spp, spm 이 없는 경우
                break

        if (m > length) or pd.isnull(condition.iloc[m]['Condition']):
            break

        start_m = m
        buy_signal_m = start_m

        dc_descend_cnt = 0
        limit_sell = False
        sell_switch = False

        if df['trade_state'].iloc[start_m] != 1.5:
            while True:
                buy_signal_m -= 1
                if df['trade_state'].iloc[buy_signal_m] == 1.5:
                    break

        while True:

            df['Support_Line_Offset_Lower'].iloc[m] = df['Support_Line_Offset_Lower'].iloc[start_m]

            #       매도 시그널       #

            if df['close'].iloc[m] <= df['Support_Line_Offset_Lower'].iloc[m]:
                sell_switch = True
                break

            if df['trade_state'].iloc[m] == 2:
                sell_switch = True
                break

            m += 1
            if m > length:
                break

            condition.iloc[m] = '매도 대기'
            bprelay["bprelay"].iloc[m] = bprelay["bprelay"].iloc[m - 1]
            support["support_line"].iloc[m] = support["support_line"].iloc[m - 1]

        if m > length:
            # print(condition.iloc[m - 1])
            condition.iloc[m - 1] = "매도 체결"

            #       매수 체결 후 바로 매도 체결 되어버리는 경우       #
            if m == start_m:
                Profits.iloc[m - 1] = df.iloc[m - 1]['close'] / bprelay["bprelay"].iloc[m - 1] - fee
            else:
                Profits.iloc[m - 1] = df.iloc[m - 1]['close'] / bprelay["bprelay"].iloc[m - 1] - fee

            if float(Profits.iloc[m - 1]) < 1:
                Minus_Profits *= float(Profits.iloc[m - 1])

                try:
                    spanlist.append((start_m, m - 1))
                    spanlist_breakaway.append((start_m, m - 1))

                except Exception as e:
                    pass

            else:
                exit_fluc += df.low.iloc[start_m:m - 1].min() / bprelay["bprelay"].iloc[m - 2]
                exit_count += 1

                try:
                    spanlist.append((start_m, m - 1))
                    spanlist_limit.append((start_m, m - 1))

                except Exception as e:
                    pass
            break

        elif sell_switch:
            condition.iloc[m] = "매도 체결"

            #       매수 체결 후 바로 매도 체결 되어버리는 경우       #
            if limit_sell:
                sell_price = bprelay["bprelay"].iloc[m] * 1.015
            else:
                sell_price = df['close'].iloc[m]
            if m == start_m:
                Profits.iloc[m] = sell_price / bprelay["bprelay"].iloc[m] - fee

            else:
                Profits.iloc[m] = sell_price / bprelay["bprelay"].iloc[m - 1] - fee

            if float(Profits.iloc[m]) < 1:
                print('Minus Profits at %s %.3f' % (m, float(Profits.iloc[m])))
                Minus_Profits *= float(Profits.iloc[m])

                try:
                    spanlist.append((start_m, m))
                    spanlist_breakaway.append((start_m, m))

                except Exception as e:
                    pass

            else:
                exit_fluc += df.low.iloc[start_m:m].min() / bprelay["bprelay"].iloc[m - 1]
                exit_count += 1

                try:
                    spanlist.append((start_m, m))
                    spanlist_limit.append((start_m, m))

                except Exception as e:
                    pass

        # DC LOW 손절시, 탐색 종료 구간 벗어날때까지 m += 1
        if dc_descend_cnt == 2:
            while True:
                m += 1
                if m >= length:
                    break

                if df['Fisher_Long'].iloc[m] >= detect_close_value:
                    break
        else:
            m += 1
            if m > length:
                break
            condition.iloc[m] = np.NaN

    df = pd.merge(df, bprelay, how='outer', left_index=True, right_index=True)
    df = pd.merge(df, support, how='outer', left_index=True, right_index=True)
    df = pd.merge(df, condition, how='outer', left_index=True, right_index=True)
    df = pd.merge(df, Profits, how='outer', left_index=True, right_index=True)
    # df = pd.merge(df, price_point, how='outer', left_index=True, right_index=True)
    df = df.reset_index(drop=True)
    df2 = df2.reset_index(drop=True)
    df3 = df3.reset_index(drop=True)

    if excel == 1:
        # df.to_excel("./BackTest/%s BackTest %s.xlsx" % (Date, Coin))
        df.to_excel("excel_check/%s BackTest %s.xlsx" % (Date, Coin))

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
        # elif get_fig == 1 and float(profits.iloc[-1]) != 1:

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(211)
        ochl = df.iloc[:, :4]
        # ochl = df2.iloc[:, :4]
        index = np.arange(len(ochl))
        ochl = np.hstack((np.reshape(index, (-1, 1)), ochl))
        mf.candlestick_ochl(ax, ochl, width=0.5, colorup='r', colordown='b')

        # plt.subplot(211)
        # plt.plot(df['close'], label='first')
        # plt.plot(df2['close'], label="df2['close']", color='gold', alpha=0.5)
        plt.plot(df['Gaussian_close'], 'yellow')
        plt.plot(df['Support_Line'], 'red')
        # plt.plot(df['Support_Line_Offset_Upper'], 'fuchsia')
        # plt.plot(df['Support_Line_Offset_Lower'], 'fuchsia')
        plt.fill_between(df.index, df['Support_Line_Offset_Upper'], df['Support_Line_Offset_Lower'], alpha=0.5, color='limegreen')
        plt.plot(df['MA_Short'])
        plt.plot(df['MA_Long'], label="df['MA_Long']", color='g')
        plt.plot(df2['MA_Long'], label="df2['MA_Long']", color='r')
        plt.plot(df3['MA_Long'], label="df3['MA_Long']", color='b')
        # plt.plot(df2['MA_Long_Offset'], 'fuchsia')
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        # plt.xlim(df.index[0], df.index[-1])
        # plt.legend(loc='auto', fontsize=10)
        plt.title('Fisher Checker - %s %.3f %.3f %.3f\n chart_height = %.3f' % (Coin,
                                                                                float(profits.iloc[-1]),
                                                                                float(profits.iloc[-1]) / Minus_Profits,
                                                                                Minus_Profits, chart_gap))

        for trade_num in range(len(span_list_entry)):
            plt.axvspan(span_list_entry[trade_num][0], span_list_entry[trade_num][1], facecolor='blue', alpha=0.7)
        for trade_num in range(len(span_list_entry_candis)):
            plt.axvspan(span_list_entry_candis[trade_num][0], span_list_entry_candis[trade_num][1], facecolor='gold',
                        alpha=0.5)
        # for trade_num in range(len(span_list_gaussian_entry)):
        #     plt.axvspan(span_list_gaussian_entry[trade_num][0], span_list_gaussian_entry[trade_num][1], facecolor='green', alpha=0.7)
        # for trade_num in range(len(span_list_fisher_gaussian_exit)):
        #     plt.axvspan(span_list_fisher_gaussian_exit[trade_num][0], span_list_fisher_gaussian_exit[trade_num][1], facecolor='red', alpha=0.7)
        # for trade_num in range(len(span_list_bull)):
        #     plt.axvspan(span_list_bull[trade_num][0], span_list_bull[trade_num][1],
        #                 facecolor='green', alpha=0.4)
        # for trade_num in range(len(span_list_bear)):
        #     plt.axvspan(span_list_bear[trade_num][0], span_list_bear[trade_num][1],
        #                 facecolor='red', alpha=0.4)

        for trade_num in range(len(spanlist_limit)):
            plt.axvspan(spanlist_limit[trade_num][0], spanlist_limit[trade_num][1], facecolor='c', alpha=0.5)
        for trade_num in range(len(spanlist_breakaway)):
            plt.axvspan(spanlist_breakaway[trade_num][0], spanlist_breakaway[trade_num][1], facecolor='m', alpha=0.5)

        plt.subplot(212)
        plt.plot(df['Fisher_Long'])
        # plt.plot(df2['Fisher_Long'])
        # plt.plot(df['TRIX_GAUSSIAN'], color='m')
        # plt.plot(df.Fisher_short2, color='gold')
        # plt.plot(df.Fisher_short_gaussian)
        plt.axhline(long_signal_value)
        plt.axhline(short_signal_value)
        # plt.xlim(df.index[0], df.index[-1])

        for trade_num in range(len(span_list_entry)):
            plt.axvspan(span_list_entry[trade_num][0], span_list_entry[trade_num][1], facecolor='orange', alpha=0.7)
        for trade_num in range(len(span_list_entry_candis)):
            plt.axvspan(span_list_entry_candis[trade_num][0], span_list_entry_candis[trade_num][1], facecolor='gold',
                        alpha=0.5)
        # for trade_num in range(len(span_list_fisher_gaussian_exit)):
        #     plt.axvspan(span_list_fisher_gaussian_exit[trade_num][0], span_list_fisher_gaussian_exit[trade_num][1],
        #                 facecolor='red', alpha=0.7)
        # for trade_num in range(len(span_list_gaussian_entry)):
        #     plt.axvspan(span_list_gaussian_entry[trade_num][0], span_list_gaussian_entry[trade_num][1],
        #                 facecolor='green',
        #                 alpha=0.7)
        for trade_num in range(len(spanlist_limit)):
            plt.axvspan(spanlist_limit[trade_num][0], spanlist_limit[trade_num][1], facecolor='c', alpha=0.5)
        for trade_num in range(len(spanlist_breakaway)):
            plt.axvspan(spanlist_breakaway[trade_num][0], spanlist_breakaway[trade_num][1], facecolor='m', alpha=0.5)
        # for trade_num in range(len(spanlist_limit)):
        #     plt.axvspan(df.index[spanlist_limit[trade_num][0]], df.index[spanlist_limit[trade_num][1]], facecolor='c',
        #                 alpha=0.5)
        # for trade_num in range(len(spanlist_breakaway)):
        #     plt.axvspan(df.index[spanlist_breakaway[trade_num][0]], df.index[spanlist_breakaway[trade_num][1]],
        #                 facecolor='m', alpha=0.5)

        # plot 저장 & 닫기
        plt.show()
        # plt.savefig("./chart_check/%s/%s.png" % (Date, Coin), dpi=300)
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

    return float(profits.iloc[-1]), float(profits.iloc[-1]) / Minus_Profits, Minus_Profits, profits_avg[
        0], Profits.values.min(), exit_fluc_mean


if __name__ == "__main__":

    home_dir = os.path.expanduser('~')
    dir = home_dir + '/OneDrive/CoinBot/ohlcv/'
    # ohlcv_list = os.listdir(dir)

    import pickle

    with open('top20.txt', 'rb') as f:
        ohlcv_list = pickle.load(f)
    # print(ohlcv_list[:10])
    # quit()

    import Selenium_Funcs
    # TopCoin = Selenium_Funcs.open_coin_list(coinvolume=10)
    # quit()
    # TopCoin = pybithumb.get_tickers()
    # TopCoin = ['BTC', 'LRC', 'ZRX', 'ANW', 'EOS', 'BORA', 'SNT', 'QBZ', 'DASH', 'SOC']

    import random

    random.shuffle(ohlcv_list)
    TopCoin = ohlcv_list
    # TopCoin = ['2020-08-08 QBZ ohlcv.xlsx']

    # for file in ohlcv_list:
    #     Coin = file.split()[1].split('.')[0]
    #     Date = file.split()[0]
    Date = str(datetime.now()).split()[0]
    # Date = str(datetime.now()).split()[0] + '_110ver'
    excel_list = os.listdir('./BestSet/Test_ohlc/')
    total_df = pd.DataFrame(
        columns=['long_signal_value', 'short_signal_value', 'long_period', 'short_period', 'total_profit_avg',
                 'plus_profit_avg', 'minus_profit_avg', 'avg_profit_avg',
                 'min_profit_avg', 'exit_fluc_mean'])

    for long_signal_value in np.arange(-4, -1.5, 0.5):
        for short_signal_value in np.arange(1.5, 4, 0.5):
            for short_period in np.arange(60, 500, 100):

                long_signal_value, short_signal_value = -2.5, 2.5
                long_period, short_period = 60, 60

                #       Make folder      #
                try:
                    os.mkdir("./chart_check/%s/" % (Date))

                except Exception as e:
                    print(e)
                #
                for Coin in TopCoin:
                    # try:
                    # if Coin.endswith('.xlsx'):
                    #     if Coin.split('-')[1] not in ['07']: # or Coin.split('-')[2].split()[0] not in ['27, 29']:
                    #         continue

                    print(Coin,
                          profitage(Coin, long_signal_value, short_signal_value, long_period, short_period, Date=Date,
                                    get_fig=1, excel=0))
                # except Exception as e:
                #     print('Error in profitage :', e)
                quit()

                total_profit = 0
                plus_profit = 0
                minus_profit = 0
                avg_profit = 0
                min_profit = 0
                exit_fluc_mean = 0
                for Coin in excel_list:
                    start = time.time()
                    try:
                        result = profitage(Coin, long_signal_value, short_signal_value, long_period, short_period,
                                           Date=Date, get_fig=1)
                        if result[0] == 1.:  # 거래가 없는 코인을 데이터프레임에 저장할 필요가 없다.
                            continue
                    except Exception as e:
                        continue
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
                print(long_signal_value, short_signal_value, long_period, short_period, total_profit_avg,
                      plus_profit_avg, minus_profit_avg, avg_profit_avg,
                      min_profit_avg, exit_fluc_avg, '%.3f second' % (time.time() - start))

                result_df = pd.DataFrame(data=[
                    [long_signal_value, short_signal_value, long_period, short_period, total_profit_avg,
                     plus_profit_avg, minus_profit_avg, avg_profit_avg,
                     min_profit_avg, exit_fluc_avg]],
                    columns=['long_signal_value', 'short_signal_value', 'long_period', 'short_period',
                             'total_profit_avg', 'plus_profit_avg',
                             'minus_profit_avg', 'avg_profit_avg', 'min_profit_avg', 'exit_fluc_avg'])
                total_df = total_df.append(result_df)
                print()

            total_df.to_excel('./BestSet/total_df %s.xlsx' % long_signal_value)
            break
