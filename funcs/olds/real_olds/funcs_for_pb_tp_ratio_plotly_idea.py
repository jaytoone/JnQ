from binance_f import RequestClient
from binance_f.model import *
from binance_f.constant.test import *
from plotly import subplots
import plotly.graph_objs as go
# import numpy as np
# import pandas as pd
# from datetime import datetime
# import os
# from scipy import stats
# from asq.initiators import query
# import matplotlib.ticker as ticker
# from sklearn.preprocessing import MaxAbsScaler, StandardScaler
# import time
# import math

from funcs.funcs_trader import *
from funcs.olds.funcs_indicator_candlescore import *

pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 2500)
pd.set_option('display.max_columns', 2500)

request_client = RequestClient(api_key=g_api_key, secret_key=g_secret_key)


def profitage(df, second_df, third_df, fourth_df=None, symbol=None, date=None, save_path=None, excel=0, get_fig=0, only_profit=False, show_time=False, label_type='Profit'):
    # limit = None  # <-- should be None, if you want all data from concated chart

    if date is None:
        date = str(datetime.now()).split(' ')[0]

    # print(df[['high', 'low']].tail())
    # df['ad'] = ad(df)
    # df['ad_ema'] = ema(df['ad'], 60)

    # df['macd_hist'] = macd(df)
    third_df['macd_hist'] = macd(third_df)
    df = df.join(pd.DataFrame(index=df.index, data=to_lower_tf(df, third_df, [-1]), columns=['macd_hist']))
    macd_interval = 15

    # print(df.tail(10))
    # quit()

    # df['senkou_a'], df['senkou_b'] = ichimoku(df)
    second_df['senkou_a'], second_df['senkou_b'] = ichimoku(second_df)
    df = df.join(pd.DataFrame(index=df.index, data=to_lower_tf(df, second_df, [-2, -1]), columns=['senkou_a', 'senkou_b']))

    # print(df.tail(20))
    # quit()

    # second_df['senkou_a'], second_df['senkou_b'] = ichimoku(second_df)
    # df = df.join(pd.DataFrame(index=df.index, data=to_lower_tf(df, second_df, [-2, -1]),
    #                           columns=['h_senkou_a', 'h_senkou_b']))

    df['ema'] = ema(df['close'], 200)

    # df['sar'] = lucid_sar(df)
    second_df['sar'] = lucid_sar(second_df, 0.02, 0.02)
    df = df.join(pd.DataFrame(index=df.index, data=to_lower_tf(df, second_df, [-1]), columns=['sar']))

    third_df['h_sar'] = lucid_sar(third_df)
    df = df.join(pd.DataFrame(index=df.index, data=to_lower_tf(df, third_df, [-1]), columns=['h_sar']))

    df['b_upper'], df['b_lower'], df['bbw'] = bb_width(df, 100, 2)
    # df = df.join(pd.DataFrame(index=df.index, data=to_lower_tf(df, second_df, [i for i in range(-3, 0, 1)]),
    #                           columns=['b_upper', 'b_lower', 'bbw']))
    df['b_basis'] = (df['b_upper'] + df['b_lower']) / 2

    # df['bbw'] = bb_width(df, 20, 2)
    # if fourth_df is not None:
    #     fourth_df['b_upper'], fourth_df['b_lower'], fourth_df['bbw'] = bb_width(fourth_df, 20, 2)
    #     df = df.join(pd.DataFrame(index=df.index, data=to_lower_tf(df, fourth_df, [i for i in range(-3, 0, 1)]),
    #                               columns=['b_upper', 'b_lower', 'bbw']))

    # print(df.tail(20))
    # quit()

    # df['trix'] = trix_hist(df, 14, 1, 9)
    # second_df['trix'] = trix_hist(second_df, 14, 1, 9)
    # df = df.join(pd.DataFrame(index=df.index, data=to_lower_tf(df, second_df, [-1]), columns=['trix']))

    # ha_df = heikinashi(df)

    # _, _, df['ST_Trend'] = supertrend(ha_df, 7, 2.5)

    df['fisher'] = fisher(df, 30)
    fisher_upper = 1.5
    fisher_lower = -1.5
    df['fisher_state'] = np.nan
    df['fisher_state'] = np.where((df['fisher'] > df['fisher'].shift(1)) & (df['fisher'].shift(1) < df['fisher'].shift(2)) & (df['fisher'].shift(1) < fisher_lower), 1, df['fisher_state'])
    df['fisher_state'] = np.where((df['fisher'] < df['fisher'].shift(1)) & (df['fisher'].shift(1) > df['fisher'].shift(2)) & (df['fisher'].shift(1) > fisher_upper), 0, df['fisher_state'])

    third_df['h_fisher'] = fisher(third_df, 30)
    # print(third_df['h_fisher'].tail(50))
    df = df.join(pd.DataFrame(index=df.index, data=to_lower_tf(df, third_df, [-1]), columns=['h_fisher']))
    # print(df['h_fisher'].tail(100))
    # quit()

    # ha_second_df = heikinashi(second_df)
    # # ha_third_df = heikinashi(third_df)
    # # print(ha_second_df.tail(10))
    # # quit()
    #

    if fourth_df is not None:
        fourth_df['major_ST1_Up'], fourth_df['major_ST1_Down'], fourth_df['major_ST1_Trend'] = supertrend(fourth_df, 10, 2)
        df = df.join(pd.DataFrame(index=df.index, data=to_lower_tf(df, fourth_df, [-3, -2, -1]),
                                  columns=['major_ST1_Up', 'major_ST1_Down', 'major_ST1_Trend']))

        df['major_ST1'] = np.where(df['major_ST1_Trend'] == 1, df['major_ST1_Down'], df['major_ST1_Up'])

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

    # print(df.head())
    # print(df.iloc[:, -8:-2].tail(20))
    # quit()

    if label_type == 'Close':
        return df

    long_marker_x = list()
    long_marker_y = list()
    short_marker_x = list()
    short_marker_y = list()

    df['trade_state'] = np.nan
    df['ep'] = np.nan
    df['sl_level'] = np.nan
    df['tp_level'] = np.nan
    df['leverage'] = np.nan

    #           Order Type          #
    order_type = OrderType.LIMIT

    #           Trading Fee          #
    fee = 0.0002 * 3

    ema_lookback = 50
    fisher_lookback = 30
    ichimoku_lookback = 30
    candle_lookback = 1

    target_sl_percent = 0.05
    target_tp_percent = 0.005
    tp_ratio = 1.5
    tp_gap = 1
    tp_pips = 100
    price_precision = 3

    fixed_leverage = 20
    max_leverage = 50

    for i in range(ichimoku_lookback, len(df) - 1):

        #           BBW Set         #
        # if df['bbw'].iloc[i] > 0.015:

        #           Long            #

        #           Fisher Set          #
        if df['fisher_state'].iloc[i] == 1:

            for back_i in range(i - 1, 0, -1):
                if df['fisher'].iloc[back_i] > fisher_lower:
                    break
            if back_i == 1:
                continue

            else:

            #       Fisher Set 2        @
            # if not np.sum(df['fisher_state'].iloc[back_i:i + 1] == 1) >= 2:
            #     continue
            #
            # else:

            #           SAR Set         #
            # if df['low'].iloc[i] >= df['sar'].iloc[i]:

            #          MACD Set         #
            # if df['macd_hist'].iloc[i] > 0:
            #
            #         if not df['macd_hist'].iloc[i] >= df['macd_hist'].iloc[i - macd_interval]:
            #             continue

                #           EMA Set         #
                # if np.minimum(df['close'].iloc[back_i:i], df['ema'].iloc[back_i:i]).equals(other=df['ema'].iloc[back_i:i]):

                #           Ichimoku Set            #
                minimum_cloud = np.minimum(df['senkou_a'].iloc[i + 1 - ichimoku_lookback:i + 1], df['senkou_b'].iloc[i + 1 - ichimoku_lookback:i + 1])
                if np.minimum(df['low'].iloc[i + 1 - ichimoku_lookback:i + 1], minimum_cloud).equals(other=minimum_cloud):

                #          AD Set          #
                # if df['ad'].iloc[i] > df['ad_ema'].iloc[i]:

                    #       if you use market order     #
                    if order_type == OrderType.MARKET:
                        i += 1
                        df['ep'].iloc[i] = df['open'].iloc[i]

                    else:
                        df['ep'].iloc[i] = df['close'].iloc[i]

                    df['trade_state'].iloc[i] = 1

                    #   ----------  SL  ----------  #

                    # df['sl_level'].iloc[i] = df['low'].iloc[back_i:i + 1].min()
                    df['sl_level'].iloc[i] = df['low'].iloc[i + 1 - fisher_lookback:i + 1].min()

                    #   ----------  TP  ----------  #
                    #       tp by gap_ratio     #
                    # df['tp_level'].iloc[i] = df['close'].iloc[i] + \
                    #                          (df['close'].iloc[i] - df['low'].iloc[
                    #                                                 i + 1 - candle_lookback:i + 1].min()) * tp_gap

                    #       tp by tp_ratio      #
                    # df['tp_level'].iloc[i] = df['ep'].iloc[i] + abs(
                    #     df['ep'].iloc[i] - df['sl_level'].iloc[i]) * tp_ratio

                    #       tp by pips      #
                    # df['tp_level'].iloc[i] = df['ep'].iloc[i] + (10 ** -price_precision) * tp_pips

                    #       tp by target_tp     #
                    # df['tp_level'].iloc[i] = df['ep'].iloc[i] + target_tp_percent * df['ep'].iloc[i]

                    #   ----------  Leverage  ----------  #

                    #       leverage by sl      #
                    # try:
                    #     df['leverage'].iloc[i] = int(target_sl_percent / (
                    #             abs(df['ep'].iloc[i] - df['sl_level'].iloc[i]) / df['ep'].iloc[i]))
                    # except Exception as e:
                    #     df['leverage'].iloc[i] = max_leverage
                    # else:
                    #     df['leverage'].iloc[i] = min(max_leverage, df['leverage'].iloc[i])

                    #       leverage by tp      #
                    # try:
                    #     df['leverage'].iloc[i] = int(target_tp_percent / (
                    #             abs(df['ep'].iloc[i] - df['tp_level'].iloc[i]) / df['ep'].iloc[i]))
                    # except Exception as e:
                    #     df['leverage'].iloc[i] = 75
                    # else:
                    #     df['leverage'].iloc[i] = min(75, df['leverage'].iloc[i])

                    #       fixed leverage      #
                    df['leverage'].iloc[i] = fixed_leverage

                    long_marker_x.append(df.index[i])
                    long_marker_y.append(df['close'].iloc[i])

        #           Short            #

        #           Fisher Set          #
        if df['fisher_state'].iloc[i] == 0:

            for back_i in range(i - 1, 0, -1):
                if df['fisher'].iloc[back_i] < fisher_upper:
                    break
            if back_i == 1:
                continue

            #       Fisher Set 2        @
            # if not np.sum(df['fisher_state'].iloc[back_i:i + 1] == 0) >= 2:
            #     continue
            #
            # else:

            #           SAR Set         #
            # if df['high'].iloc[i] <= df['sar'].iloc[i]:

            #           MACD Set         #
            # if df['macd_hist'].iloc[i] < 0:
            #
            #         if not df['macd_hist'].iloc[i] <= df['macd_hist'].iloc[i - macd_interval]:
            #             continue

            else:

                #           EMA Set         #
                # if np.maximum(df['close'].iloc[back_i:i], df['ema'].iloc[back_i:i]).equals(other=df['ema'].iloc[back_i:i]):

                #           Ichimoku Set            #
                maximum_cloud = np.maximum(df['senkou_a'].iloc[i + 1 - ichimoku_lookback:i + 1],
                                           df['senkou_b'].iloc[i + 1 - ichimoku_lookback:i + 1])
                if np.maximum(df['low'].iloc[i + 1 - ichimoku_lookback:i + 1], maximum_cloud).equals(
                        other=maximum_cloud):

                #           AD Set          #
                # if df['ad'].iloc[i] < df['ad_ema'].iloc[i]:

                    #       if you use market order     #
                    if order_type == OrderType.MARKET:
                        i += 1
                        df['ep'].iloc[i] = df['open'].iloc[i]

                    else:
                        df['ep'].iloc[i] = df['close'].iloc[i]

                    df['trade_state'].iloc[i] = 0

                    #   ----------  SL  ----------  #

                    # df['sl_level'].iloc[i] = df['high'].iloc[back_i:i + 1].max()
                    df['sl_level'].iloc[i] = df['high'].iloc[i + 1 - fisher_lookback:i + 1].max()

                    #   ----------  TP  ----------  #

                    #       tp by gap_ratio     #
                    # df['tp_level'].iloc[i] = df['close'].iloc[i] + \
                    #                          (df['close'].iloc[i] - df['high'].iloc[
                    #                                                 i + 1 - candle_lookback:i + 1].max()) * tp_gap

                    #       tp by tp_ratio      #
                    # df['tp_level'].iloc[i] = df['ep'].iloc[i] - abs(
                    #     df['ep'].iloc[i] - df['sl_level'].iloc[i]) * tp_ratio

                    #       tp by pips      #
                    # df['tp_level'].iloc[i] = df['ep'].iloc[i] - (10 ** -price_precision) * tp_pips

                    #       tp by target_tp     #
                    # df['tp_level'].iloc[i] = df['ep'].iloc[i] - target_tp_percent * df['ep'].iloc[i]

                    #   ----------  Leverage  ----------  #

                    #       leverage by sl      #
                    # try:
                    #     df['leverage'].iloc[i] = int(target_sl_percent / (
                    #             abs(df['ep'].iloc[i] - df['sl_level'].iloc[i]) / df['ep'].iloc[i]))
                    # except Exception as e:
                    #     df['leverage'].iloc[i] = max_leverage
                    # else:
                    #     df['leverage'].iloc[i] = min(max_leverage, df['leverage'].iloc[i])

                    #       leverage by tp      #
                    # try:
                    #     df['leverage'].iloc[i] = int(target_tp_percent / (
                    #             abs(df['ep'].iloc[i] - df['tp_level'].iloc[i]) / df['ep'].iloc[i]))
                    # except Exception as e:
                    #     df['leverage'].iloc[i] = 75
                    # else:
                    #     df['leverage'].iloc[i] = min(75, df['leverage'].iloc[i])

                    df['leverage'].iloc[i] = fixed_leverage

                    short_marker_x.append(df.index[i])
                    short_marker_y.append(df['close'].iloc[i])

    # print(df.trade_state)
    # quit()

    if label_type == 'Open':
        return df

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
    #     df.to_excel("./excel_check/%s BackTest %s.xlsx" % (Date, Coin))
    #     quit()

    #                               Close Order & TP Check                                  #
    m = 0
    long_win_cnt = 0
    long_lose_cnt = 0
    short_win_cnt = 0
    short_lose_cnt = 0
    long_profit, short_profit = list(), list()
    spanlist_win, win_profit = list(), list()
    spanlist_lose, lose_profit = list(), list()
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
        total_tp_cnt = 1
        enable_duplicate_tp = True
        tp_count = total_tp_cnt
        remain_qty = 1.
        profit = .0

        if not pd.isna(tp_level):
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

            if not pd.isna(tp_level):
                tp_level = tp_list[tp_count - 1]

            #       close_count == 1인 경우에만 Wait Close 기입해줄 필요가 발생한다. (in Excel)       #
            # if tp_count != 2:
            #     condition.iloc[m] = 'Wait Close'
            #     eprally["Eprally"].iloc[m] = eprally["Eprally"].iloc[m - 1]

            #               Check Close Condition            #
            #               Long Type           #
            if order_type == 'Long':

                # -------------------- TP --------------------  #
                #           By Price        #
                # if df['high'].iloc[m] > tp_level:
                #     close_state.iloc[m] = 'Level TP'
                #     TP_switch = True

                #           By fisher           #
                if total_tp_cnt != 1:

                    if tp_count == 2:
                        # if df['fisher'].iloc[m] > 0:
                        if df['fisher_state'].iloc[m] == 0:
                            TP_switch = True

                    elif tp_count == 1:
                        if df['fisher_state'].iloc[m] == 0:
                            TP_switch = True

                else:
                    if df['fisher_state'].iloc[m] == 0:
                    # if df['fisher_state'].iloc[m] == 0 and df['sar'].iloc[m] <= df['low'].iloc[m]:
                    #     if df['close'].iloc[m] >= ep:
                            TP_switch = True

                # -------------------- SL -------------------- #
                #           By SAR          #
                # if df['close'].iloc[m - 1] > df['sar'].iloc[m - 1]:
                #     if df['low'].iloc[m] < df['sar'].iloc[m]:
                #         SL_switch = True

                #           By ST           #
                # if df['ST_Trend'].iloc[m - 1] > 0 > df['ST_Trend'].iloc[m]:
                #     SL_switch = True

                #           By Trix         #
                # if df['trix'].iloc[m - 1] > 0 >= df['trix'].iloc[m]:
                #     SL_switch = True

                #           By Ichimoku         #
                # if df['close'].iloc[m - 1] >= np.minimum(df['senkou_a'].iloc[m - 1], df['senkou_b'].iloc[m - 1]) and \
                if df['close'].iloc[m] < np.minimum(df['senkou_a'].iloc[m], df['senkou_b'].iloc[m]):  # and \
                #     # df['close'].iloc[m] <= df['open'].iloc[m]:
                    SL_switch = True

                #            By Price           #
                # if df['low'].iloc[m] < sl_level:
                # if df['close'].iloc[m] < sl_level:
                #     close_state.iloc[m] = 'Price SL'
                #     SL_switch = True

            #               Short Type              #
            elif order_type == 'Short':

                # -------------------- TP --------------------  #
                #           By Price        #
                # if df['low'].iloc[m] < tp_level:
                #     close_state.iloc[m] = 'Level TP'
                #     TP_switch = True

                #           By fisher           #
                if total_tp_cnt != 1:

                    if tp_count == 2:
                        # if df['fisher'].iloc[m] < 0:
                        if df['fisher_state'].iloc[m] == 1:
                            TP_switch = True

                    elif tp_count == 1:
                        if df['fisher_state'].iloc[m] == 1:
                            TP_switch = True

                else:
                    if df['fisher_state'].iloc[m] == 1:
                    # if df['fisher_state'].iloc[m] == 1 and df['sar'].iloc[m] >= df['high'].iloc[m]:
                    #     if df['close'].iloc[m] <= ep:
                            TP_switch = True

                # -------------------- SL -------------------- #
                #           By SAR          #
                # if df['close'].iloc[m - 1] < df['sar'].iloc[m - 1]:
                #     if df['high'].iloc[m] > df['sar'].iloc[m]:
                #         SL_switch = True

                #           By ST           #
                # if df['ST_Trend'].iloc[m - 1] < 0 < df['ST_Trend'].iloc[m]:
                #     SL_switch = True

                #           By Trix         #
                # if df['trix'].iloc[m - 1] < 0 <= df['trix'].iloc[m]:
                #     SL_switch = True

                #           By Ichimoku     #
                # if df['close'].iloc[m - 1] <= np.maximum(df['senkou_a'].iloc[m - 1], df['senkou_b'].iloc[m - 1]) and \
                if df['close'].iloc[m] > np.maximum(df['senkou_a'].iloc[m], df['senkou_b'].iloc[m]):  # and \
                #     # df['close'].iloc[m] >= df['open'].iloc[m]:
                    SL_switch = True

                #            By Price           #
                # if df['high'].iloc[m] > sl_level:
                # if df['close'].iloc[m] > sl_level:
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

                if order_type == OrderType.MARKET:
                    m += 1
                    if m > length:
                        break
                    exit_price = df['open'].iloc[m]
                else:
                    exit_price = df['close'].iloc[m]

                if order_type == 'Short':
                    trade_fee = -fee
                else:
                    trade_fee = fee

                if TP_switch:

                    tp_count -= 1
                    tp_marker_x.append(df.index[m])

                    if not pd.isna(tp_level):
                        profit_gap = ((tp_level / ep - trade_fee) - 1) * qty_list.pop(tp_count) * leverage
                        tp_marker_y.append(tp_level)

                    else:
                        # print('exit_price used')
                        # print(exit_price, ep, qty_list, leverage, end=' ')
                        profit_gap = ((exit_price / ep - trade_fee) - 1) * qty_list.pop(tp_count) * leverage
                        # print(profit_gap)
                        tp_marker_y.append(exit_price)

                    profit += profit_gap

                else:
                    tp_count = 0
                    sl_marker_x.append(df.index[m])

                    #           SL 을 Limit 으로 한다는 설정이 무리가 있음을 확인        #
                    # if not pd.isna(sl_level):
                    #     profit_gap = ((sl_level / ep - trade_fee) - 1) * sum(qty_list) * leverage
                    #     sl_marker_y.append(sl_level)
                    #
                    # else:
                    # print(exit_price, ep, qty_list, leverage, end=' ')
                    profit_gap = ((exit_price / ep - trade_fee) - 1) * sum(qty_list) * leverage
                    # print(profit_gap)
                    sl_marker_y.append(exit_price)

                    profit += profit_gap

                if tp_count == 0:

                    if order_type == 'Long':
                        Profits.iloc[m] = 1 + profit
                    else:
                        Profits.iloc[m] = 1 - profit

                    #           Lose           #
                    if float(Profits.iloc[m]) < 1:
                        # print('Minus Profits at %s %.3f' % (m, float(Profits.iloc[m])))
                        Loss *= float(Profits.iloc[m])

                        if order_type == 'Long':
                            long_lose_cnt += 1
                            long_profit.append(Profits.iloc[m])
                        else:
                            short_lose_cnt += 1
                            short_profit.append(Profits.iloc[m])

                        #           Check Final Close       #
                        try:
                            spanlist_lose.append((start_m, m))
                            lose_profit.append(Profits.iloc[m])

                        except Exception as e:
                            pass

                    #           Win          #
                    else:
                        exit_count += 1

                        if order_type == 'Long':
                            long_win_cnt += 1
                            long_profit.append(Profits.iloc[m])
                        else:
                            short_win_cnt += 1
                            short_profit.append(Profits.iloc[m])

                        try:
                            spanlist_win.append((start_m, m))
                            win_profit.append(Profits.iloc[m])

                        except Exception as e:
                            pass

                    #           Trade Done          #
                    break

                elif not enable_duplicate_tp:
                    m += 1
                    if m > length:
                        break

        if m > length:
            break

    df = pd.merge(df, close_state, how='outer', left_index=True, right_index=True)
    df = pd.merge(df, condition, how='outer', left_index=True, right_index=True)
    df = pd.merge(df, Profits, how='outer', left_index=True, right_index=True)

    if excel == 1:
        # df.to_excel("./BackTest/%s BackTest %s.xlsx" % (Date, Coin))
        df.to_excel("../excel_check/%s BackTest.xlsx" % symbol)

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

        startTime = time.time()

        # df = df.reset_index(drop=True)
        # elif get_fig == 1 and float(profits.iloc[-1]) != 1:

        win_count = len(spanlist_win)
        lose_count = len(spanlist_lose)
        long_win_ratio = calc_win_ratio(long_win_cnt, long_lose_cnt)
        short_win_ratio = calc_win_ratio(short_win_cnt, short_lose_cnt)
        total_win_ratio = calc_win_ratio(win_count, lose_count)

        if len(long_profit) != 0:
            cum_long_profit = np.cumprod(long_profit)[-1]
        else:
            cum_long_profit = 0
        if len(long_profit) != 0:
            cum_short_profit = np.cumprod(short_profit)[-1]
        else:
            cum_short_profit = 0

        price_data = list()
        if not only_profit:

            #           ohlc          #
            ohlc_ply = go.Candlestick(x=df.index, open=df['open'], close=df['close'], high=df['high'], low=df['low'],
                                      name='ohlc')
            price_data.append(ohlc_ply)

            #       indicator       #
            senkou_a_ply = go.Scatter(x=df.index, y=df['senkou_a'], name='senkou_a', line={'color': 'green'})
            senkou_b_ply = go.Scatter(x=df.index, y=df['senkou_b'], name='senkou_b', line={'color': 'red'})
            price_data.append(senkou_a_ply)
            price_data.append(senkou_b_ply)

            sar_ply = go.Scatter(x=df.index, y=df['sar'], name='sar', mode='markers', marker=dict(size=3, color='dodgerblue'))
            # h_sar_ply = go.Scatter(x=df.index, y=df['h_sar'], name='h_sar', line={'color': 'dodgerblue'})
            price_data.append(sar_ply)

            b_upper_ply = go.Scatter(x=df.index, y=df['b_upper'], name='b_upper', line={'color': 'white'})
            b_lower_ply = go.Scatter(x=df.index, y=df['b_lower'], name='b_lower', line={'color': 'white'})
            b_basis_ply = go.Scatter(x=df.index, y=df['b_basis'], name='b_basis', line={'color': 'white'})
            price_data.append(b_upper_ply)
            price_data.append(b_lower_ply)
            price_data.append(b_basis_ply)

            # st_ply = go.Scatter(x=df.index, y=df['major_ST1'], name='major_ST1', line={'color': 'gold'})

            fisher_ply = go.Scatter(x=df.index, y=df['fisher'], name='fisher', line={'color': 'lime'})
            h_fisher_ply = go.Scatter(x=df.index, y=df['h_fisher'], name='h_fisher', line={'color': 'lime'})

            # ad_ply = go.Scatter(x=df.index, y=df['ad'], name='ad', line={'color': 'lime'})
            # ad_ema_ply = go.Scatter(x=df.index, y=df['ad_ema'], name='ad_ema', line={'color': 'deepskyblue'})

            macd_ply = go.Scatter(x=df.index, y=df['macd_hist'], name='macd_hist', line={'color': 'dodgerblue'})

            # bbw_ply = go.Scatter(x=df.index, y=df['bbw'], name='bbw', line={'color': 'dodgerblue'})

            # ad_data = [ad_ply, ad_ema_ply]


        profit_ply = go.Scatter(x=df.index, y=df['Profits'].cumprod(), name='profits', fill='tozeroy')
        profit_ma_ply = go.Scatter(x=df.index, y=df['Profits'].cumprod().rolling(2000).mean(), name='profits_ma', line={'color': 'lime'})

        if not only_profit:

            #       marker      #
            l_marker_ply = go.Scatter(x=long_marker_x, y=long_marker_y, mode='markers',
                                      marker=dict(symbol=5, size=10, color='lime'), name='long')
            s_marker_ply = go.Scatter(x=short_marker_x, y=short_marker_y, mode='markers',
                                      marker=dict(symbol=6, size=10, color='yellow'), name='short')
            tp_marker_ply = go.Scatter(x=tp_marker_x, y=tp_marker_y, mode='markers', marker=dict(size=10, color='blue'), name='tp')
            sl_marker_ply = go.Scatter(x=sl_marker_x, y=sl_marker_y, mode='markers', marker=dict(size=10, color='red'), name='sl')

            price_data.append(l_marker_ply)
            price_data.append(s_marker_ply)
            price_data.append(tp_marker_ply)
            price_data.append(sl_marker_ply)

        #           plot declaration        #
        if only_profit:
            row, col = 1, 1
            fig = subplots.make_subplots(row, col, shared_xaxes=True)
        else:
            row, col = 4, 1
            fig = subplots.make_subplots(row, col, shared_xaxes=True, row_width=[0.2, 0.2, 0.2, 0.4])
        title_text = 'Profit (Total, L, S, Max, Min) : %.3f %.3f %.3f %.3f %.3f, Win Ratio (L, S, Total) : %.3f %.3f %.3f' \
                                                            % (float(profits.iloc[-1]), cum_long_profit, cum_short_profit,
                                                               np.max(df['Profits']), np.min(df['Profits']),
                                                               long_win_ratio, short_win_ratio, total_win_ratio)
        fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False, title={'text': title_text, 'x': 0.5}, dragmode='pan')

        if not only_profit:

            row, col = 1, 1
            for trace in price_data:
                fig.append_trace(trace, row, col)

            for trade_num in range(len(spanlist_win)):
                x0, x1 = spanlist_win[trade_num]
                fig.add_vrect(x0=df.index[x0], x1=df.index[x1], annotation_text='%.3f (%s)' % (win_profit[trade_num], int(df['leverage'].iloc[x0])), annotation_position='top left',
                              fillcolor="cyan", opacity=0.25, line_width=0, row=row, col=col)
            for trade_num in range(len(spanlist_lose)):
                x0, x1 = spanlist_lose[trade_num]
                fig.add_vrect(x0=df.index[x0], x1=df.index[x1], annotation_text='%.3f (%s)' % (lose_profit[trade_num], int(df['leverage'].iloc[x0])), annotation_position='top left',
                              fillcolor="magenta", opacity=0.25, line_width=0, row=row, col=col)

            row, col = 2, 1
            fig.append_trace(fisher_ply, row, col)
            fig.append_trace(h_fisher_ply, row, col)
            # fig.add_shape(fisher_layout)
            fig.add_hline(y=fisher_upper, row=row, col=col, line_dash="dash", line_width=1)
            fig.add_hline(y=0., row=row, col=col, line_dash="dash", line_width=1)
            fig.add_hline(y=fisher_lower, row=row, col=col, line_dash="dash", line_width=1)

            for trade_num in range(len(spanlist_win)):
                x0, x1 = spanlist_win[trade_num]
                fig.add_vrect(x0=df.index[x0], x1=df.index[x1], annotation_text='%.3f' % win_profit[trade_num], annotation_position='top left',
                              fillcolor="cyan", opacity=0.25, line_width=0, row=row, col=col)
            for trade_num in range(len(spanlist_lose)):
                x0, x1 = spanlist_lose[trade_num]
                fig.add_vrect(x0=df.index[x0], x1=df.index[x1], annotation_text='%.3f' % lose_profit[trade_num], annotation_position='top left',
                              fillcolor="magenta", opacity=0.25, line_width=0, row=row, col=col)

            row, col = 3, 1
            # fig.append_trace(macd_ply, row, col)
            # fig.add_hline(y=0., row=row, col=col, line_dash="dash", line_width=1)
            # fig.append_trace(bbw_ply, row, col)
            fig.add_hline(y=0.015, row=row, col=col, line_dash="dash", line_width=1)
            # for trace in ad_data:
            #     fig.append_trace(trace, row, col)

        row, col = 4, 1
        if only_profit:
            row, col = 1, 1

        fig.append_trace(profit_ply, row, col)
        fig.append_trace(profit_ma_ply, row, col)

        # offline.offline.build_save_image_post_script(script_decorator(offline.offline.build_save_image_post_script))
        # offline.iplot(fig)
        if save_path is not None:
            fig.write_html('../%s/%s.html' % (save_path, date), auto_open=True)
        else:
            fig.write_html('../invest_stability/Temp/%s.html' % date, auto_open=True)

        if show_time:
            print('time consumed by plotly :', time.time() - startTime)

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
        0], np.min(Profits.values), exit_fluc_mean


if __name__ == "__main__":

    home_dir = os.path.expanduser('~')
    dir = './database/'

    import pickle

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
