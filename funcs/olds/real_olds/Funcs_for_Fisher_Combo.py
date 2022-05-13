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


def profitage(Coin, period1=30, period2=60, period3=120, wait_tick=3, Date='2019-09-25', excel=0, get_fig=0):
    global length

    if not Coin.endswith('.xlsx'):
        Coin = Coin + 'USDT'
        df = request_client.get_candlestick_data(symbol=Coin, interval=CandlestickInterval.MIN1,
                                                 startTime=None, endTime=None, limit=1500)
        second_df = request_client.get_candlestick_data(symbol=Coin, interval=CandlestickInterval.MIN3,
                                                        startTime=None, endTime=None, limit=600)
        third_df = request_client.get_candlestick_data(symbol=Coin, interval=CandlestickInterval.MIN30,
                                                       startTime=None, endTime=None, limit=70)
    else:
        df = pd.read_excel(dir + '%s' % Coin, index_col=0)
        second_df = pd.read_excel(dir.replace('ohlcv', 'ohlcv_5m') + '%s' % Coin, index_col=0)
        third_df = pd.read_excel(dir.replace('ohlcv', 'ohlcv_3m') + '%s' % Coin, index_col=0)

    chart_height = (df['high'].max() - df['low'].min())
    chart_gap = (df['high'].max() / df['low'].min())
    ha_second_df = heikinashi(second_df)
    ha_third_df = heikinashi(third_df)

    # df['CB'], df['EMA_CB'] = cct_bbo(df, 18, 13)
    # print(df.tail(10))
    # quit()

    df['Fisher1'] = fisher(df, period=period1)
    df['Fisher2'] = fisher(df, period=period2)
    df['Fisher3'] = fisher(df, period=period3)

    tc_upper = 0.35
    tc_lower = -tc_upper

    df['Fisher1_Trend'] = fisher_trend(df, 'Fisher1', 0, 0)
    df['Fisher2_Trend'] = fisher_trend(df, 'Fisher2', tc_upper, tc_lower)
    df['Fisher3_Trend'] = fisher_trend(df, 'Fisher3', tc_upper, tc_lower)

    # print(ha_second_df.tail(10))
    # quit()

    # print(second_df.tail(20))
    # quit()

    # print(df.tail())
    # print(df.iloc[:, -8:-2].tail(20))
    # quit()

    #                               Open Condition                              #
    #       Open Long / Short = 1 / 2       #
    #       Close Long / Short = -1 / -2        #
    trade_state = [0] * len(df)
    upper = 0.5
    lower = -upper

    long_marker_x = list()
    long_marker_y = list()
    short_marker_x = list()
    short_marker_y = list()
    for i in range(2, len(df)):

        # if df['Trix'].iloc[i] > 0:
        # if df['Fisher2_Trend'].iloc[i] == 'Long':
        # if df['Fisher3_Trend'].iloc[i] == 'Long':
        if df['Fisher2_Trend'].iloc[i] == 'Long' and df['Fisher3_Trend'].iloc[i] == 'Long':
            if df['Fisher1'].iloc[i - 2] > df['Fisher1'].iloc[i - 1] < df['Fisher1'].iloc[i]:
                if df['Fisher1'].iloc[i - 1] < lower:
                    trade_state[i] = 1
                    long_marker_x.append(i)
                    long_marker_y.append(df['Fisher1'].iloc[i])

        # if df['Trix'].iloc[i] < 0:
        # if df['Fisher2_Trend'].iloc[i] == 'Short':
        # if df['Fisher3_Trend'].iloc[i] == 'Short':
        if df['Fisher2_Trend'].iloc[i] == 'Short' and df['Fisher3_Trend'].iloc[i] == 'Short':
            if df['Fisher1'].iloc[i - 2] < df['Fisher1'].iloc[i - 1] > df['Fisher1'].iloc[i]:
                if df['Fisher1'].iloc[i - 1] > upper:
                    trade_state[i] = 2
                    short_marker_x.append(i)
                    short_marker_y.append(df['Fisher1'].iloc[i])

    df['trade_state'] = trade_state
    # print(df.trade_state)
    # quit()

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
        #       Entry Price 와 HL offseted gap 이 5 % 가 되도록 Leverage 설정        #
        hl_level = None
        offset_percentage = 1 / 2
        sl_percentage = 0.05
        for i in range(m, 0, -1):
            if df['Fisher1_Trend'].iloc[i] != df['Fisher1_Trend'].iloc[i - 1]:

                low = min(df['low'].iloc[i:m + 1])
                high = max(df['high'].iloc[i:m + 1])

                if df['trade_state'].iloc[m] == 1:
                    hl_level = low - (high - low) * offset_percentage
                else:
                    hl_level = high + (high - low) * offset_percentage
                break

        #       주문 등록 완료, 체결 대기     #
        #       시장가 진입으로 설정해서 조건 변경이 요구된다.      #
        # start_m = m
        # print('hl_level :', hl_level)
        sl_level['SL_Level'].iloc[m] = hl_level
        leverage['Leverage'].iloc[m] = int(sl_percentage / (abs(ep - hl_level) / ep))
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
    #     df.to_excel("./excel_check/%s BackTest %s.xlsx" % (Date, Coin))
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
        close_count = 2
        tp_count = 0
        p1, p2 = None, None
        trix_out = False
        fisher_cross_zero = False
        sl_fisher_level_none_tp = 1
        sl_fisher_level = 0

        #           바로 체결되는 경우 회피한다.        #
        m += 1
        condition.iloc[m] = 'Wait Close'
        eprally["Eprally"].iloc[m] = eprally["Eprally"].iloc[m - 1]
        if m > length:
            break

        while True:
            TP_switch = False
            SL_switch = False

            if close_count != 2:
                condition.iloc[m] = 'Wait Close'
                eprally["Eprally"].iloc[m] = eprally["Eprally"].iloc[m - 1]

            #               Check Close Condition            #
            #               Long Type           #
            if order_type == 'Long':

                #           TP by Signal            #
                if df['Fisher1'].iloc[m - 2] < df['Fisher1'].iloc[m - 1] > df['Fisher1'].iloc[m]:
                    if df['Fisher1'].iloc[m - 1] > upper:
                        trade_state_close.iloc[m] = 'Signal TP'
                        TP_switch = True

                #            SL by Price           #
                if df['low'].iloc[m] < sl_level_:
                    trade_state_close.iloc[m] = 'Price SL'
                    SL_switch = True

                #           SL by Signal            #
                if close_count == 2:
                    # if df['Fisher1'].iloc[m - 1] > -sl_fisher_level_none_tp > df['Fisher1'].iloc[m]:
                    #     trade_state_close.iloc[m] = 'Signal SL'
                    #     SL_switch = True
                    if df['Fisher2_Trend'].iloc[m] == 'Short':
                        trade_state_close.iloc[m] = 'Signal SL'
                        SL_switch = True

                if close_count == 1:
                    if df['Fisher1'].iloc[m - 1] > -sl_fisher_level > df['Fisher1'].iloc[m]:
                        trade_state_close.iloc[m] = 'Signal SL'
                        SL_switch = True

                # if not trix_out:
                #     if df['Trix'].iloc[m] < 0:
                #         trix_out = True

            #               Short Type              #
            elif order_type == 'Short':

                #           TP by Signal            #
                if df['Fisher1'].iloc[m - 2] > df['Fisher1'].iloc[m - 1] < df['Fisher1'].iloc[m]:
                    if df['Fisher1'].iloc[m - 1] < lower:
                        trade_state_close.iloc[m] = 'Signal TP'
                        TP_switch = True

                #           SL by Price           #
                if df['high'].iloc[m] > sl_level_:
                    trade_state_close.iloc[m] = 'Price SL'
                    SL_switch = True

                #           SL by Signal            #
                if close_count == 2:
                    # if df['Fisher1'].iloc[m - 1] < sl_fisher_level_none_tp < df['Fisher1'].iloc[m]:
                    #     trade_state_close.iloc[m] = 'Signal SL'
                    #     SL_switch = True
                    if df['Fisher2_Trend'].iloc[m] == 'Long':
                        trade_state_close.iloc[m] = 'Signal SL'
                        SL_switch = True

                if close_count == 1:
                    if df['Fisher1'].iloc[m - 1] < sl_fisher_level < df['Fisher1'].iloc[m]:
                        trade_state_close.iloc[m] = 'Signal SL'
                        SL_switch = True


                # if not trix_out:
                #     if df['Trix'].iloc[m] > 0:
                #         trix_out = True

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
                    if trix_out:
                        close_count = 0
                    else:
                        tp_count += 1
                    tp_marker_x.append(m)
                    tp_marker_y.append(df['Fisher1'].iloc[m])
                else:
                    close_count = 0
                    sl_marker_x.append(m)
                    sl_marker_y.append(df['Fisher1'].iloc[m])

                exit_price = df['close'].iloc[m]
                if m == start_m:
                    close_m = m
                else:
                    close_m = m - 1

                if close_count == 1:
                    p1 = ((exit_price / eprally["Eprally"].iloc[close_m]) - 1) / 2
                elif close_count == 0:
                    p2 = ((exit_price / eprally["Eprally"].iloc[close_m]) - 1) / 2

                    if tp_count == 0:
                        p1 = p2

                if close_count == 0:

                    Profits.iloc[m] = 1 + (p1 + p2)
                    #           Consider Short Type Profit          #
                    if order_type == 'Short':
                        Profits.iloc[m] = 1 - (p1 + p2)

                    #           If Swap,        #
                    Profits.iloc[m] = 1 - (Profits.iloc[m] - 1)

                    #           Adjust Leverage             #
                    Profits.iloc[m] = (Profits.iloc[m] - 1 - fee) * leverage_ + 1

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
            break

    df = pd.merge(df, trade_state_close, how='outer', left_index=True, right_index=True)
    df = pd.merge(df, leverage, how='outer', left_index=True, right_index=True)
    df = pd.merge(df, eprally, how='outer', left_index=True, right_index=True)
    df = pd.merge(df, sl_level, how='outer', left_index=True, right_index=True)
    df = pd.merge(df, condition, how='outer', left_index=True, right_index=True)
    df = pd.merge(df, Profits, how='outer', left_index=True, right_index=True)

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

        df = df.reset_index(drop=True)
        # elif get_fig == 1 and float(profits.iloc[-1]) != 1:

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(211)
        ohlc = df.iloc[:, :4]
        # ohlc = df2.iloc[:, :4]
        index = np.arange(len(ohlc))
        ohlc = np.hstack((np.reshape(index, (-1, 1)), ohlc))
        mf.candlestick_ohlc(ax, ohlc, width=0.5, colorup='r', colordown='b')
        # ax.xaxis.set_major_locator(ticker.MaxNLocator(6))
        # def mdate(x, pos):
        #     try:
        #         return df.index[int(x)]
        #     except IndexError:
        #         return ''
        # ax.xaxis.set_major_formatter(ticker.FuncFormatter(mdate))

        # plt.plot(df.iloc[:, -14:-11], '.', markersize=2, color='orangered')
        # plt.plot(df.iloc[:, -11:-8], '.', markersize=4, color='olive')
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        # plt.xlim(df.index[0], df.index[-1])
        # plt.legend(loc='auto', fontsize=10)
        plt.title('Fisher Checker - %s %.3f %.3f %.3f\n chart_height = %.3f' % (Coin,
                                                                                float(profits.iloc[-1]),
                                                                                float(profits.iloc[-1]) / Loss,
                                                                                Loss, chart_gap))

        for trade_num in range(len(spanlist_win)):
            plt.axvspan(spanlist_win[trade_num][0], spanlist_win[trade_num][1], facecolor='c', alpha=0.5)
        for trade_num in range(len(spanlist_lose)):
            plt.axvspan(spanlist_lose[trade_num][0], spanlist_lose[trade_num][1], facecolor='m', alpha=0.5)

        plt.subplot(212)
        plt.plot(df['Fisher1'], color='lime')
        plt.plot(df['Fisher2'], color='#90caf9')
        plt.plot(df['Fisher3'], color='#f44336')
        plt.plot(long_marker_x, long_marker_y, 'o', color='c')
        plt.plot(short_marker_x, short_marker_y, 'o', color='m', alpha=0.7)
        plt.plot(tp_marker_x, tp_marker_y, 'o', color='blue')
        plt.plot(sl_marker_x, sl_marker_y, 'o', color='red')
        plt.axhline(tc_upper, linestyle='--')
        plt.axhline(tc_lower, linestyle='--')
        plt.axhline(0, linestyle='--')
        plt.axhline(1, linestyle='--')
        plt.axhline(-1, linestyle='--')

        # for trade_num in range(len(span_list_long)):
        #     plt.axvspan(span_list_long[trade_num][0], span_list_long[trade_num][1], facecolor='red', alpha=0.7)
        # for trade_num in range(len(span_list_short)):
        #     plt.axvspan(span_list_short[trade_num][0], span_list_short[trade_num][1], facecolor='blue',
        #                 alpha=0.5)p
        for trade_num in range(len(spanlist_win)):
            plt.axvspan(spanlist_win[trade_num][0], spanlist_win[trade_num][1], facecolor='c', alpha=0.5)
        for trade_num in range(len(spanlist_lose)):
            plt.axvspan(spanlist_lose[trade_num][0], spanlist_lose[trade_num][1], facecolor='m', alpha=0.5)

        # plt.subplot(313)
        # # plt.plot(df['Trix'])
        # plt.axhline(0, linestyle='--')
        #
        # for trade_num in range(len(spanlist_win)):
        #     plt.axvspan(spanlist_win[trade_num][0], spanlist_win[trade_num][1], facecolor='c', alpha=0.5)
        # for trade_num in range(len(spanlist_lose)):
        #     plt.axvspan(spanlist_lose[trade_num][0], spanlist_lose[trade_num][1], facecolor='m', alpha=0.5)

        # plot 저장 & 닫기
        plt.show()
        plt.savefig("./chart_check/%s/%s.png" % (Date, Coin), dpi=300)

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
    dir = home_dir + '/OneDrive/CoinBot/ohlcv/'
    # ohlcv_list = os.listdir(dir)

    import pickle
    import random

    with open('future_coin.p', 'rb') as f:
        ohlcv_list = pickle.load(f)
        # random.shuffle(ohlcv_list)

    # print(ohlcv_list)
    # quit()

    TopCoin = ohlcv_list
    TopCoin = ['BTC']

    # for file in ohlcv_list:
    #     Coin = file.split()[1].split('.')[0]
    #     Date = file.split()[0]
    Date = str(datetime.now()).split()[0]
    # Date = str(datetime.now()).split()[0] + '_110ver'
    excel_list = os.listdir('./BestSet/Test_ohlc/')
    total_df = pd.DataFrame(
        columns=['long_signal_value', 'short_signal_value', 'period1', 'short_period', 'total_profit_avg',
                 'plus_profit_avg', 'minus_profit_avg', 'avg_profit_avg',
                 'min_profit_avg', 'exit_fluc_mean'])

    #       Make folder      #
    try:
        os.mkdir("./chart_check/%s/" % (Date))

    except Exception as e:
        print(e)

    total_profit = 0
    plus_profit = 0
    minus_profit = 0
    avg_profit = 0
    min_profit = 0
    exit_fluc_mean = 0
    for Coin in TopCoin:
        start = time.time()
        # try:
        result = profitage(Coin, Date=Date, get_fig=1, excel=0)
        # quit()
        print(Coin, result)
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
