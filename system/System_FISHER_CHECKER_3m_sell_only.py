import pybithumb
import Funcs_MACD_OSC
import time
from datetime import datetime
import random
from finta import TA
import numpy as np
from scipy import stats
from scipy.ndimage.filters import gaussian_filter1d
# import pandas as pd
# import os
import warnings

warnings.filterwarnings("ignore")
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import System_TRIX_TRIX_Funcs
from selenium import webdriver
from fake_useragent import UserAgent


#       KEY SET     #
with open("Keys.txt") as f:
    lines = f.readlines()
    key = lines[0].strip()
    secret = lines[1].strip()
    bithumb = pybithumb.Bithumb(key, secret)

#       INDICATOR SET       #
fisher_period = 70
long_signal_value = -2.0
short_signal_value = 2.5

smma_period = 200
smma_slope_range = -0.2

dc_period_entry = 20
dc_period = 80
dc_descend_accept_gap_range = -0.01

trix_period = 10
trix_fisher_period = 11

#       TRADE SET       #
CoinVolume = 7
buy_wait = 5  # minute
profits = 1.0
headless = 1

#           SELL SIGNAL CHECK           #
Coin = 'ZEC'
sell_switch = 0
while True:

    #       CHECK THE TIME      #
    #       종가를 서둘러 확인해버리면 dc_descend_cnt 오차 생길 수 있다.       #
    current_datetime = datetime.now()
    if current_datetime.minute % 3 == 2 and current_datetime.second >= 59:

        while True:
            #        바로 전 데이터를 찾을때까지 반복      #
            #        이전 종가로 매수가를 지정하기 위함        #
            #           interval time = 3m          #
            try:
                df = pybithumb.get_candlestick(Coin, chart_instervals='3m')
                current_min_minus1 = datetime.now().minute - 1
                df_last_min = int(str(df.index[-1]).split()[1].split(':')[1])
                if current_min_minus1 < 0:
                    current_min_minus1 += 60
                if current_min_minus1 <= df_last_min or current_min_minus1 - df_last_min > 1:
                    break
                time.sleep(1)

            except Exception as e:
                print('Error in get_candlestick :', e)

        chart_height = df['high'].max() - df['low'].min()

        df['FISHER_LONG'] = TA.FISH(df, period=fisher_period)
        df['FISHER_TREND'] = np.where(df['FISHER_LONG'] > df['FISHER_LONG'].shift(1), 'UP', 'DOWN')

        #           REALTIME CURVE          #
        smoothed_curve = [np.nan] * len(df)
        period = 5
        sigma = 5
        for i in range(period, len(df)):
            y = df['FISHER_LONG'].values[i + 1 - period:i + 1]
            y_smooth = gaussian_filter1d(y, sigma=sigma)
            smoothed_curve[i] = y_smooth[-1]

        df['FISHER_GAUSSIAN'] = smoothed_curve
        df['FISHER_GAUSSIAN_TREND'] = np.where(df['FISHER_GAUSSIAN'].shift(1) <= df['FISHER_GAUSSIAN'], 'UP',
                                               'DOWN')

        df['TRIX'] = TA.TRIX(df, period=trix_period)
        df['TRIX'] = np.where(np.isnan(df['TRIX']), 0, df['TRIX'])
        df['TRIX_TREND'] = np.where(df['TRIX'].shift(1) <= df['TRIX'], 'UP', 'DOWN')

        fisher_height = df['FISHER_LONG'].max() - df['FISHER_LONG'].min()

        df['SCALED_FISHER'] = (df['FISHER_LONG'] - df['FISHER_LONG'].min()) / fisher_height
        df['FISHER_TRIX'] = TA.TRIX(df, period=11, column='SCALED_FISHER')
        df['FISHER_TRIX_TREND'] = np.where(df['FISHER_TRIX'].shift(1) <= df['FISHER_TRIX'], 'UP', 'DOWN')

        df['DC_LOW'] = TA.DO(df, lower_period=dc_period, upper_period=dc_period)['LOWER']
        df['DC_TREND'] = np.where(df['DC_LOW'].shift(1) <= df['DC_LOW'], 'UP', 'DOWN')
        for i in range(1, len(df)):
            if df['DC_LOW'].iloc[i] == df['DC_LOW'].iloc[i - 1]:
                df['DC_TREND'].iloc[i] = df['DC_TREND'].iloc[i - 1]

        #           EXPOSE INDICATOR VALUE          #
        print(current_datetime.now())
        print('PREVIOUS FISHER | GAUSSIAN | TRIX | FISHER TRIX :',
              df['FISHER_LONG'].iloc[-2], df['FISHER_GAUSSIAN'].iloc[-2], df['TRIX'].iloc[-2], df['FISHER_TRIX'].iloc[-2])
        print('CURRENT FISHER | GAUSSIAN | TRIX | FISHER TRIX :',
              df['FISHER_LONG'].iloc[-1], df['FISHER_GAUSSIAN'].iloc[-1], df['TRIX'].iloc[-1], df['FISHER_TRIX'].iloc[-1])

        #       매도 조건       #
        if df['FISHER_LONG'].iloc[-2] >= short_signal_value:
            if df['FISHER_TREND'].iloc[-1] == 'DOWN' and df['FISHER_TREND'].iloc[-2] == 'UP':
                sell_switch = 1
                break

        if df['FISHER_GAUSSIAN'].iloc[-1] > 0:
            if df['FISHER_GAUSSIAN_TREND'].iloc[-1] == 'DOWN' and df['FISHER_GAUSSIAN_TREND'].iloc[-2] == 'UP':
                sell_switch = 1
                break

        if df['FISHER_GAUSSIAN_TREND'].iloc[-1] == 'DOWN' and df['FISHER_GAUSSIAN_TREND'].iloc[-2] == 'UP':
            if df['TRIX'].iloc[-2] >= 0:
                sell_switch = 1
                break

        if df['FISHER_TRIX'].iloc[-1] >= 4:
            if df['FISHER_TRIX_TREND'].iloc[-1] == 'DOWN' and df['FISHER_TRIX_TREND'].iloc[-2] == 'UP':
                sell_switch = 1
                break

        if df['DC_TREND'].iloc[-1] == 'DOWN':
            copy_m = -1
            while True:
                copy_m -= 1
                if df['DC_LOW'].iloc[copy_m] != df['DC_LOW'].iloc[copy_m + 1]:
                    break
                if copy_m == 0:
                    break
            if (df['DC_LOW'].iloc[-1] - df['DC_LOW'].iloc[copy_m]) / chart_height < dc_descend_accept_gap_range:
                print('DC GAP :', copy_m, -1, (df['DC_LOW'].iloc[-1] - df['DC_LOW'].iloc[copy_m]) / chart_height)
                sell_switch = 0.5

        if sell_switch == 0.5:
            if df['FISHER_GAUSSIAN_TREND'].iloc[-1] == 'DOWN' and df['FISHER_GAUSSIAN_TREND'].iloc[-2] == 'UP':
                sell_switch = 1
                break

        #       59초와 60초 사이에 여러번 반복되는 걸 방지하기 위한 TERM      #
        time.sleep(0.5)

#           매도 등록           #
limit_sell_price = Funcs_MACD_OSC.clearance(df['close'].iloc[-1])
while True:

    try:
        #   매도 진행
        balance = bithumb.get_balance(Coin)
        sellunit = int((balance[0]) * 10000) / 10000.0
        SellOrder = bithumb.sell_limit_order(Coin, limit_sell_price,
                                             sellunit, 'KRW')
        print("    %s %s KRW 지정가 매도     " % (Coin, limit_sell_price), end=' ')
        # print("    %s 시장가 매도     " % Coin, end=' ')
        # SellOrder = bithumb.sell_limit_order(Coin, limit_sell_pricePlus, sellunit, "KRW")
        print(SellOrder)
        break

    except Exception as e:
        print('Error in %s first sell order :' % Coin, e)

resell_switch = 0
resell_term = 60  # second
sell_type = 'limit'
sell_start_time = time.time()
#           매도 대기           #
while True:

    #                   매도 재등록                    #
    try:
        if resell_switch == 1:

            #   SellOrder Initializing
            CancelOrder = bithumb.cancel_order(SellOrder)
            if CancelOrder is False:  # 남아있는 매도 주문이 없다. 취소되었거나 체결완료.
                #       체결 완료       #
                if bithumb.get_order_completed(SellOrder)['data']['order_status'] != 'Cancel':
                    print("    매도 체결    ", end=' ')


                    break
                #       취소      #
                else:
                    pass
            elif CancelOrder is None:  # SellOrder = none 인 경우
                # 매도 재등록 해야함 ( 등록될 때까지 )
                pass
            else:
                print("    매도 취소    ", end=' ')
                print(CancelOrder)
            print()

            balance = bithumb.get_balance(Coin)
            sellunit = int((balance[0]) * 10000) / 10000.0
            if sell_type == 'limit':
                SellOrder = bithumb.sell_limit_order(Coin,
                                                     limit_sell_price, sellunit,
                                                     "KRW")
                print("    %s %s KRW 지정가 매도     " % (Coin, limit_sell_price), end=' ')
            elif sell_type == 'market':
                while True:
                    #       시장가 매도 경우 55초 이후에 진행    #
                    if datetime.now().second >= 55:
                        break
                SellOrder = bithumb.sell_market_order(Coin, sellunit, 'KRW')
                print("    %s 시장가 매도     " % Coin, end=' ')
            print(SellOrder)

            resell_switch = -1
            time.sleep(1 / 130)  # 너무 빠른 거래로 인한 오류 방지

            # 지정 매도 에러 처리
            if type(SellOrder) in [tuple, str]:
                pass
            elif SellOrder is None:  # 매도 함수 자체 오류 ( 서버 문제 같은 경우 )
                resell_switch = 1
                continue
            else:  # dictionary
                # 체결 여부 확인 로직
                ordersucceed = bithumb.get_balance(Coin)
                #   매도 주문 넣었을 때의 잔여 코인과 거래 후 코인이 다르고 사용중인 코인이 없으면 매도 체결
                #   거래 중인 코인이 있으면 매도체결이 출력되지 않는다.
                if (ordersucceed[0] != balance[0] and ordersucceed[1] == 0.0) or (
                        '최소 주문금액' in SellOrder['message']):
                    print("    매도 체결    ", end=' ')


                    break
                else:
                    resell_switch = 1
                    continue

    except Exception as e:
        print("매도 재등록 중 에러 발생 :", e)
        resell_switch = 1
        continue

    #       매도 상태 Check >> 취소 / 체결완료 / SellOrder = None / dictionary      #
    try:
        if bithumb.get_outstanding_order(SellOrder) is None:
            # 서버 에러에 의해 None 값이 발생할 수 도 있음..
            if type(SellOrder) in [tuple, str]:  # 서버에러는 except 로 가요..
                try:
                    # 체결 여부 확인 로직
                    ordersucceed = bithumb.get_balance(Coin)
                    if ordersucceed[0] != balance[0] and ordersucceed[1] == 0.0:
                        print("    매도 체결    ", end=' ')


                        break
                    elif bithumb.get_outstanding_order(SellOrder) is not None:  # 혹시 모르는 미체결
                        continue
                    else:
                        print("매도 주문이 취소되었습니다.\n")

                        if resell_switch in [0, -1]:
                            resell_switch = 1
                            # elif resell_switch == 0:
                            #     sppswitch = 1
                            time.sleep(random.random() * 5)
                        continue

                except Exception as e:
                    print('SellOrder in [tuple, str] ? 에서 에러발생 :', e)

                    time.sleep(random.random() * 5)  # 서버 에러인 경우
                    continue

            # 매도 등록 에러라면 ? 제대로 등록 될때까지 재등록 ! 지정 에러, 하향 에러
            elif SellOrder is None:  # limit_sell_order 가 아예 안되는 경우
                if resell_switch in [0, -1]:
                    resell_switch = 1
                    # elif resell_switch == 0:
                    #     sppswitch = 1
                    time.sleep(random.random() * 5)
                continue

            else:  # dictionary
                # 체결 여부 확인 로직
                ordersucceed = bithumb.get_balance(Coin)
                if (ordersucceed[0] != balance[0] and ordersucceed[1] == 0.0) or (
                        '최소 주문금액' in SellOrder['message']):
                    print("    매도 체결    ", end=' ')


                    break
                else:
                    if resell_switch in [0, -1]:
                        resell_switch = 1
                        # elif resell_switch == 0:
                        #     sppswitch = 1
                        time.sleep(random.random() * 5)
                    continue

        #       매도 / 손절 미체결 시        #
        #       시간이 지남에 따라 매도가 / 손절가를 낮춘다.      #
        #       손절 매도 등록하고 지정한 초가 지나면, 55초 이후로 시장가 매도 #
        #       시장가 미체결시 계속해서 매도 등록한다.      #
        else:
            try:
                #       3M FISHER SHORT가 아니라면 RESELL TERM은 30초      #
                if sell_switch != 1:
                    resell_term = 30

                if time.time() - sell_start_time >= resell_term:
                    limit_sell_price = Funcs_MACD_OSC.clearance(limit_sell_price * 0.9975)
                    resell_switch = 1
                    # sell_type = 'market'
                    sell_start_time = time.time()

            except Exception as e:
                pass

    except Exception as e:  # 지정 매도 대기 중 에러나서 이탈 매도가로 팔아치우는거 방지하기 위함.
        print('취소 / 체결완료 / SellOrder = None / dict 확인중 에러 발생 :', e)
        continue