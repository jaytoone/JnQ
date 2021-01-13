import pybithumb
import numpy as np
import pandas as pd
from datetime import datetime
import os
from scipy import stats
from asq.initiators import query
import matplotlib.pyplot as plt
from sklearn.preprocessing import MaxAbsScaler
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
        # df_short = pybithumb.get_candlestick(Coin, 'KRW', '1m')
        # df_short2 = pybithumb.get_ohlcv(Coin, 'KRW', 'minute5')
    else:
        df = pd.read_excel(dir + '%s' % Coin, index_col=0)
    # df = pd.read_excel('./BestSet/Test_ohlc/%s' % Coin, index_col=0)

    #           Lunde and Timmermann Formula            #
    lambda_bear = 0.06
    lambda_bull = 0.08
    df['LT_Trend'] = np.NaN
    t_zero = 0
    trend_state = None
    span_list_bull_start = list()
    span_list_bear_start = list()

    for i in range(1, len(df)):
        pmax = df['close'].iloc[t_zero:i].max()
        pmin = df['close'].iloc[t_zero:i].min()
        delta_bear = (pmax - df['close'].iloc[i]) / pmax
        delta_bull = (df['close'].iloc[i] - pmin) / pmin
        # print(pmax, pmin, delta_bear, delta_bull)

        if delta_bear > lambda_bear and trend_state != 'Bear':
            t_peak = df['close'].iloc[t_zero:i].idxmax()
            t_peak = df.index.to_list().index(t_peak)
            # t_peak = i
            # if trend_state == 'Bear':
            span_list_bear_start.append((i, i + 1))
            print('delta_bear > lambda_bear', i, 't_zero :', t_zero)
            df['LT_Trend'].iloc[t_zero + 1:t_peak + 1] = 'Bull'
            t_zero = t_peak
            trend_state = 'Bear'

        elif delta_bull > lambda_bull and trend_state != 'Bull':
            t_trough = df['close'].iloc[t_zero:i].idxmin()
            t_trough = df.index.to_list().index(t_trough)
            # t_trough = i
            # if trend_state != 'Bear':
            span_list_bull_start.append((i, i + 1))
            print('delta_bull > lambda_bull', i, 't_zero :', t_zero)
            df['LT_Trend'].iloc[t_zero + 1:t_trough + 1] = 'Bear'
            t_zero = t_trough
            trend_state = 'Bull'

        if i == len(df) - 1:
            if pd.isnull(df['LT_Trend'].iloc[i]):
                back_i = i
                while True:
                    back_i -= 1
                    if not pd.isnull(df['LT_Trend'].iloc[back_i]):
                        if df['LT_Trend'].iloc[back_i] == 'Bull':
                            df['LT_Trend'].iloc[back_i + 1:] = 'Bear'
                        else:
                            df['LT_Trend'].iloc[back_i + 1:] = 'Bull'
                        break

            df['LT_Trend'].iloc[0] = df['LT_Trend'].iloc[1]

    # print(df['LT_Trend'])
    # quit()

    chart_height = (df['high'].max() - df['low'].min())

    df['MA_SHORT'] = df['close'].rolling(20).mean()
    df['MA_SHORT_FOR_SLOPE'] = (df['MA_SHORT'] - df['low'].min()) / chart_height
    df['MA_SHORT_SLOPE'] = np.NaN
    slope_period = 5
    for s in range(slope_period - 1, len(df)):
        df['MA_SHORT_SLOPE'].iloc[s] = \
            stats.linregress([i for i in range(slope_period)], df['MA_SHORT_FOR_SLOPE'][s + 1 - slope_period:s + 1])[0]
    df['MA_SHORT_SLOPE'] *= 1e3

    df['MA_LONG'] = df['close'].rolling(120).mean()
    df['MA_LONG_FOR_SLOPE'] = (df['MA_LONG'] - df['low'].min()) / chart_height
    df['MA_LONG_SLOPE'] = np.NaN
    slope_period = 5
    for s in range(slope_period - 1, len(df)):
        df['MA_LONG_SLOPE'].iloc[s] = \
            stats.linregress([i for i in range(slope_period)], df['MA_LONG_FOR_SLOPE'][s + 1 - slope_period:s + 1])[0]
    df['MA_LONG_SLOPE'] *= 1e3

    # df['SUPERTREND'] = Fun

    df['TRIX'] = TA.TRIX(df, period=10)
    df['TRIX'] = np.where(np.isnan(df['TRIX']), 0, df['TRIX'])
    df['TRIX_TREND'] = np.where(df['TRIX'].shift(1) <= df['TRIX'], 'UP', 'DOWN')

    # df['RSI'] = TA.RSI(df, period=60)
    # for i in range(60, 0, -1):
    #     df['RSI'].iloc[i - 1] = df['RSI'].iloc[i]
    df['FISHER_LONG'] = TA.FISH(df, period=long_period)
    fisher_height = df['FISHER_LONG'].max() - df['FISHER_LONG'].min()

    df['SCALED_FISHER'] = (df['FISHER_LONG'] - df['FISHER_LONG'].min()) / fisher_height
    df['FISHER_TRIX'] = TA.TRIX(df, period=18, column='SCALED_FISHER')
    # print(df['FISHER_TRIX'].head(20))
    # quit()
    df['FISHER_TRIX_TREND'] = np.where(df['FISHER_TRIX'].shift(1) <= df['FISHER_TRIX'], 'UP', 'DOWN')

    # df['TRIX_LATER'] = TA.TRIX(df, period=20)
    # df['TRIX_LATER'] = np.where(np.isnan(df['TRIX_LATER']), 0, df['TRIX_LATER'])

    df['FISHER_TREND'] = np.where(df['FISHER_LONG'] > df['FISHER_LONG'].shift(1), 'UP', 'DOWN')
    df['FIHSER_IP'] = np.where(df['FISHER_TREND'].shift(1) != df['FISHER_TREND'], 1, 0)
    # df['Fisher_long_gaussian'] = df.FISHER_LONG.rolling(60).mean()
    # df['Fisher_short_gaussian'] = df.FISHER_SHORT.rolling(60).mean()

    #           REALTIME CURVE          #
    smoothed_curve = [np.nan] * len(df)
    smoothed_curve_short = [np.nan] * len(df)
    close_list = [np.nan] * len(df)
    period = 5
    sigma = 5
    df['TRIX_FOR_GAUSSIAN'] = TA.TRIX(df, period=16)
    for i in range(period, len(df)):
        y = df['FISHER_LONG'].values[i + 1 - period:i + 1]
        y_short = df['TRIX_FOR_GAUSSIAN'].values[i + 1 - period:i + 1]
        y_smooth = gaussian_filter1d(y, sigma=sigma)
        y_smooth_short = gaussian_filter1d(y_short, sigma=sigma)
        smoothed_curve[i] = y_smooth[-1]
        smoothed_curve_short[i] = y_smooth_short[-1]

        # close = df.close.values[i + 1 - period:i + 1]
        # close = gaussian_filter1d(close, sigma=sigma, mode='reflect')
        # close_list[i] = close[-1]

    df['FISHER_GAUSSIAN'] = smoothed_curve
    df['FISHER_GAUSSIAN_TREND'] = np.where(df['FISHER_GAUSSIAN'].shift(1) <= df['FISHER_GAUSSIAN'], 'UP', 'DOWN')
    #
    df['TRIX_GAUSSIAN'] = smoothed_curve_short
    df['TRIX_GAUSSIAN'] = np.where(np.isnan(df['TRIX_GAUSSIAN']), 0, df['TRIX_GAUSSIAN'])
    df['TRIX_GAUSSIAN_TREND'] = np.where(df['TRIX_GAUSSIAN'].shift(1) <= df['TRIX_GAUSSIAN'], 'UP', 'DOWN')

    trix_entry_previous_gap = 0.0005
    trix_entry_vertical_gap = 0.06
    trix_horizontal_ticks = 5
    trix_exit_vertical_gap = 0.17
    FISHER_trix_exit_vertical_gap = 0.8
    trix_exit_gap = -0.01
    trix_exit_previous_gap = -0.002
    upper_line = 10
    lower_line = -15

    # long_signal_value = 0
    # short_signal_value = 2.0
    long_ic_signal_value = -2.0
    detect_close_value = long_signal_value
    high_check_value = 2
    high_check_value_exit = 1.7
    high_check_value2 = 0
    high_check_value2_exit = 0
    exit_short_signal_value = -1.0

    #           SMMA DC          #
    smma_period = 200
    smma_slope_range = -0.2
    smma_accept_range = -0.08
    dc_accept_range = 0.05

    sell_price_gap = 0.25
    dc_lowprice_accept_gap_range = 0.03
    dc_entry_gap = -0.02
    # dc_exit_gap = -0.02
    dc_exit_gap = -0.035

    dc_horizontal_gap = 30
    dc_low_entry_low_accept_gap_range = 0.05
    dc_descend_accept_gap_range = -0.03

    df['SMMA'] = TA.SMMA(df, period=smma_period)
    df['SMMA_FOR_SLOPE'] = (df['SMMA'] - df['low'].min()) / chart_height
    df['SMMA_SLOPE'] = np.NaN
    slope_period = 20
    for s in range(slope_period - 1, len(df)):
        df['SMMA_SLOPE'].iloc[s] = stats.linregress([i for i in range(slope_period)], df['SMMA_FOR_SLOPE'][s + 1 - slope_period:s + 1])[0]
    df['SMMA_SLOPE'] *= 1e3

    dc_period_entry = 20
    df['DC_LOW_ENTRY'] = TA.DO(df, lower_period=dc_period_entry, upper_period=dc_period_entry)['LOWER']
    df['SMMA_DC_GAP'] = (df['DC_LOW_ENTRY'] - df['SMMA']) / chart_height

    #           DC CLOSE           #
    dc_period = 100
    # dc_period = 120
    dc_slow_period = 110
    df['DC_LOW'] = TA.DO(df, lower_period=dc_period, upper_period=dc_period)['LOWER']
    df['DC_LOW_SLOW'] = TA.DO(df, lower_period=dc_slow_period, upper_period=dc_slow_period)['LOWER']
    df['LOW_DC_GAP'] = (df['low'] - df['DC_LOW']) / chart_height
    df['LOW_SLOW_DC_GAP'] = (df['low'] - df['DC_LOW_SLOW']) / chart_height
    df['DC_TREND'] = np.where(df['DC_LOW'].shift(1) <= df['DC_LOW'], 'UP', 'DOWN')
    df['DC_SLOW_TREND'] = np.where(df['DC_LOW_SLOW'].shift(1) <= df['DC_LOW_SLOW'], 'UP', 'DOWN')
    for i in range(1, len(df)):
        # if np.isnan(df['DC_LOW'].iloc[i - 1]):
        #     df['DC_TREND'].iloc[i] = 'UP'
        if df['DC_LOW'].iloc[i] == df['DC_LOW'].iloc[i - 1]:
            df['DC_TREND'].iloc[i] = df['DC_TREND'].iloc[i - 1]
        if df['DC_LOW_SLOW'].iloc[i] == df['DC_LOW_SLOW'].iloc[i - 1]:
            df['DC_SLOW_TREND'].iloc[i] = df['DC_SLOW_TREND'].iloc[i - 1]

    # print(df)
    # quit()

    #                               LONG SHORT                              #
    trade_state = [0] * len(df)
    for i in range(2, len(df)):

        if df['FISHER_LONG'].iloc[i - 1] <= long_signal_value:
            if df['FISHER_TREND'].iloc[i] == 'UP' and df['FISHER_TREND'].iloc[i - 1] == 'DOWN':
                trade_state[i] = 1.5

        if df['FISHER_LONG'].iloc[i - 1] >= short_signal_value:
            if df['FISHER_TREND'].iloc[i] == 'DOWN' and df['FISHER_TREND'].iloc[i - 1] == 'UP':
                trade_state[i] = 2

    df['trade_state'] = trade_state
    # print(df.trade_state)
    # quit()

    span_list_entry_candis = list()
    span_list_entry = list()
    span_list_fisher_gaussian_exit = list()
    span_list_gaussian_entry = list()
    span_list_fisher_trix_up = list()
    span_list_fisher_trix_down = list()
    span_list_bull = list()
    span_list_bear = list()
    target = df['trade_state']
    for i in range(len(target)):
        if target.iloc[i] == 1:
            span_list_entry_candis.append((i, i + 1))
        if target.iloc[i] == 1.5:
            span_list_entry.append((i, i + 1))
        if df['LT_Trend'].iloc[i] == 'Bull':
            span_list_bull.append((i, i + 1))
        if df['LT_Trend'].iloc[i] == 'Bear':
            span_list_bear.append((i, i + 1))
        # if df['FISHER_GAUSSIAN_TREND'].iloc[i] == 'DOWN' and df['FISHER_GAUSSIAN_TREND'].iloc[i - 1] == 'UP':
        #     span_list_fisher_gaussian_exit.append((i, i + 1))
        # elif df['FISHER_GAUSSIAN_TREND'].iloc[i] == 'UP' and df['FISHER_GAUSSIAN_TREND'].iloc[i - 1] == 'DOWN':
        #     span_list_gaussian_entry.append((i, i + 1))
        # if df['FISHER_TRIX_TREND'].iloc[i] == 'DOWN' and df['FISHER_TRIX_TREND'].iloc[i - 1] == 'UP':
        #     if df['FISHER_TRIX'].iloc[i] >= 0:
        #         span_list_fisher_trix_down.append((i, i + 1))
        # elif df['FISHER_TRIX_TREND'].iloc[i] == 'UP' and df['FISHER_TRIX_TREND'].iloc[i - 1] == 'DOWN':
        #     if df['FISHER_TRIX'].iloc[i] < 0:
        #         span_list_fisher_trix_up.append((i, i + 1))

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

            #       매수가와 DC GAP 비교      #
            #       조건 불만족시 진입하지 않는다.       #
            # print('dc_bp_accept_gap_range :', m, (bprelay["bprelay"].iloc[m] - df['DC_LOW'].iloc[m]) / chart_height)
            # print('DC LOW ENTRY LOW GAP :', m, (df['DC_LOW_ENTRY'].iloc[m] - df['DC_LOW'].iloc[m]) / chart_height)
            # if (bprelay["bprelay"].iloc[m] - df['DC_LOW'].iloc[m]) / chart_height >= dc_lowprice_accept_gap_range:
            #     m += 1
            #     break
            # if (df['DC_LOW_ENTRY'].iloc[m] - df['DC_LOW'].iloc[m]) / chart_height < dc_low_entry_low_accept_gap_range:
            #     m += 1
            #     break

            # 매수 체결 조건
            if df.iloc[m]['low'] <= bprelay["bprelay"].iloc[m]:  # and (df.iloc[m]['high'] != df.iloc[m]['low']):  # 조건을 만족할 경우 spp 생성
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
    #     df.to_excel("./ExcelCheck/%s BackTest %s.xlsx" % (Date, Coin))
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

            #       매도 시그널       #

            # if df['high'].iloc[m] >= bprelay['bprelay'].iloc[m] * 1.015:
            #     limit_sell = True
            #     sell_switch = True
            #     break

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

                if df['FISHER_LONG'].iloc[m] >= detect_close_value:
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

    if excel == 1:
        # df.to_excel("./BackTest/%s BackTest %s.xlsx" % (Date, Coin))
        df.to_excel("ExcelCheck/%s BackTest %s.xlsx" % (Date, Coin))

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

    # elif get_fig == 1:
    elif get_fig == 1 and float(profits.iloc[-1]) != 1:

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(211)
        ochl = df.iloc[:, :4]
        index = np.arange(len(ochl))
        ochl = np.hstack((np.reshape(index, (-1, 1)), ochl))
        mf.candlestick_ochl(ax, ochl, width=0.5, colorup='r', colordown='b')
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.plot(df['MA_SHORT'])
        plt.plot(df['MA_LONG'])
        # plt.legend(loc='upper right', fontsize=5)
        plt.title('FISHER CHECKER - %s %.3f %.3f %.3f' % (Coin,
        float(profits.iloc[-1]), float(profits.iloc[-1]) / Minus_Profits, Minus_Profits))
        for trade_num in range(len(span_list_entry)):
            plt.axvspan(span_list_entry[trade_num][0], span_list_entry[trade_num][1], facecolor='orange', alpha=0.7)
        for trade_num in range(len(span_list_entry_candis)):
            plt.axvspan(span_list_entry_candis[trade_num][0], span_list_entry_candis[trade_num][1], facecolor='gold', alpha=0.5)

        for trade_num in range(len(span_list_bull)):
            plt.axvspan(span_list_bull[trade_num][0], span_list_bull[trade_num][1], facecolor='green', alpha=0.5)
        for trade_num in range(len(span_list_bear)):
            plt.axvspan(span_list_bear[trade_num][0], span_list_bear[trade_num][1], facecolor='red', alpha=0.5)
        for trade_num in range(len(span_list_bull_start)):
            plt.axvspan(span_list_bull_start[trade_num][0], span_list_bull_start[trade_num][1],
                        facecolor='blue', alpha=0.7)
        for trade_num in range(len(span_list_bear_start)):
            plt.axvspan(span_list_bear_start[trade_num][0], span_list_bear_start[trade_num][1],
                        facecolor='red', alpha=0.7)

        # for trade_num in range(len(spanlist_limit)):
        #     plt.axvspan(spanlist_limit[trade_num][0], spanlist_limit[trade_num][1], facecolor='c', alpha=0.5)
        # for trade_num in range(len(spanlist_breakaway)):
        #     plt.axvspan(spanlist_breakaway[trade_num][0], spanlist_breakaway[trade_num][1], facecolor='m', alpha=0.5)

        plt.subplot(212)
        plt.plot(df.FISHER_LONG)
        # plt.plot(df['FISHER_TRIX'])
        # plt.plot(df['TRIX_GAUSSIAN'], color='m')
        # plt.plot(df.Fisher_short2, color='gold')
        # plt.plot(df.Fisher_short_gaussian)
        plt.axhline(long_signal_value)
        plt.axhline(short_signal_value)
        for trade_num in range(len(span_list_entry_candis)):
            plt.axvspan(span_list_entry_candis[trade_num][0], span_list_entry_candis[trade_num][1], facecolor='gold',
                        alpha=0.5)
        for trade_num in range(len(span_list_entry)):
            plt.axvspan(span_list_entry[trade_num][0], span_list_entry[trade_num][1], facecolor='orange', alpha=0.7)
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

        # plt.subplot(413)
        # plt.plot(df.TRIX)
        # # plt.axhline(high_check_value, color='orange', alpha=0.5)
        # # plt.axhline(long_signal_value * (df['TRIX'].max() - df['TRIX'].min()), color='red', alpha=0.5)
        # plt.axhline(0)
        # # plt.axhline(lower_line)
        #
        # plt.subplot(414)
        # plt.plot(df['FISHER_TRIX'])
        # plt.axhline(0)
        # for trade_num in range(len(span_list_fisher_trix_up)):
        #     plt.axvspan(span_list_fisher_trix_up[trade_num][0], span_list_fisher_trix_up[trade_num][1], facecolor='gold', alpha=0.7)
        # for trade_num in range(len(span_list_fisher_trix_down)):
        #     plt.axvspan(span_list_fisher_trix_down[trade_num][0], span_list_fisher_trix_down[trade_num][1], facecolor='limegreen', alpha=0.7)

        # plot 저장 & 닫기
        plt.show()
        # plt.show(block=False)
        # plt.pause(3)
        # plt.savefig("./ChartCheck/%s/%s.png" % (Date, Coin), dpi=300)
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
    ohlcv_list = os.listdir(dir)

    import random

    random.shuffle(ohlcv_list)

    # import Selenium_Funcs
    # TopCoin = Selenium_Funcs.open_coin_list(coinvolume=10)
    # quit()
    # TopCoin = pybithumb.get_tickers()
    TopCoin = ['LINK', 'ANKR', 'CHR', 'WAVES', 'GXC', 'VALOR', 'ALGO', 'ARPA', 'THETA']
    TopCoin = [ 'ANKR', 'CHR', 'WAVES', 'GXC', 'VALOR', 'ALGO', 'ARPA', 'THETA']
    # TopCoin = ['ARPA', 'THETA']

    # TopCoin = ohlcv_list
    # TopCoin = ['2020-07-03 EM ohlcv.xlsx']

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
                    os.mkdir("./ChartCheck/%s/" % (Date))

                except Exception as e:
                    print(e)
                #
                for Coin in TopCoin:

                    if Coin.endswith('.xlsx'):
                        if Coin.split('-')[1] not in ['07']: # or Coin.split('-')[2].split()[0] not in ['27, 29']:
                            continue
                    try:
                        print(Coin, profitage(Coin, long_signal_value, short_signal_value, long_period, short_period, Date=Date,
                                        get_fig=1, excel=0))
                    except Exception as e:
                        print('Error in profitage :', e)
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