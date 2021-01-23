import pybithumb
import Funcs_MACD_OSC
import time
from datetime import datetime
import random
from finta import TA
# import numpy as np
# import pandas as pd
# import os
import warnings
warnings.filterwarnings("ignore")
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains

#   Chrome Setting  #
path = "C:/Users/Lenovo/PycharmProjects/Project_System_Trading/Rapid_Ascend/chromedriver.exe"
driver = webdriver.Chrome(path)

#       TRADE IFNO      #
interval = 'minute1'
interval_key1 = Keys.NUMPAD1
# interval_key2 = Keys.NUMPAD0
init = 1
init_tab_close = 1

#       KEY SETTING     #
with open("Keys.txt") as f:
    lines = f.readlines()
    key = lines[0].strip()
    secret = lines[1].strip()
    bithumb = pybithumb.Bithumb(key, secret)

#       Trade Info      #
long_signal_value = -2.5
short_signal_value = 2.5
CoinVolume = 15
buy_wait = 5  # minute
profits = 1.0
proxy = 'on'

#                                           CHECK INVEST_KRW                                             #
while True:

    #               Finding Buy Signal              #
    buy_signal = 0
    while True:

        #           Making TopCoin List         #
        try:
            TopCoin = pybithumb.get_top_coin(CoinVolume)
        except Exception as e:
            print('Error in getting TopCoin :', e)
            continue
        # TopCoin = list(map(str.upper, ['dvp']))

        for Coin in TopCoin:

            while True:

                try:
                    df = pybithumb.get_ohlcv(Coin, interval='minute1', proxy=proxy)
                    print('Finding %s indicator value...' % Coin)
                    proxy = 'on'
                    if df is not None:
                        break
                    else:
                        proxy = None

                except Exception as e:
                    print('Error in get_ohlcv :', e)
                    proxy = None

            #       GET INDICATOR VALUE         #
            Fisher = TA.FISH(df, period=60)

            if Fisher.iloc[-1] <= long_signal_value:

                print('Fisher value at long position :', Coin, Fisher.iloc[-1])
                while True:
                    #        바로 전 데이터를 찾을때까지 반복      #
                    try:
                        df = pybithumb.get_ohlcv(Coin, interval='minute1', proxy=proxy)
                        current_min_minus1 = datetime.now().minute - 1
                        df_last_min = int(str(df.index[-1]).split()[1].split(':')[1])
                        if current_min_minus1 < 0:
                            current_min_minus1 += 60
                        if current_min_minus1 == df_last_min or current_min_minus1 - df_last_min > 1:
                            break
                        time.sleep(1)
                        proxy = 'on'

                    except Exception as e:
                        print('Error in get_ohlcv :', e)
                        proxy = None

                buy_signal = 1
                break   # GET OUT OF THE FOR LOOP

            else:
                time.sleep(1 / 130)

        if buy_signal == 1:
            break

    #                매수 등록                  #
    try:
        # 호가단위, 매수가, 수량
        limit_buy_price = Funcs_MACD_OSC.clearance(df.close.iloc[-1])
        # limit_sell_price = Funcs_MACD_OSC.clearance(limit_buy_price * 1.015)
        # limit_exit_price = Funcs_MACD_OSC.clearance(limit_buy_price * 0.99)

        # -------------- 보유 원화 확인 --------------#
        balance = bithumb.get_balance(Coin)
        start_coin = balance[0]
        krw = balance[2]
        print()
        print("보유 원화 :", krw, end=' ')
        # invest_krw = krw * 0.996
        invest_krw = 10000
        print("주문 원화 :", invest_krw)
        if krw < 1000:
            print("거래가능 원화가 부족합니다.\n")
            continue
        print()

        # 매수량
        buyunit = int((invest_krw / limit_buy_price) * 10000) / 10000.0

        # 매수 등록
        BuyOrder = bithumb.buy_limit_order(Coin, limit_buy_price, buyunit, "KRW")
        print("    %s %s KRW 매수 등록    " % (Coin, limit_buy_price), end=' ')
        print(BuyOrder)

    except Exception as e:
        print("매수 등록 중 에러 발생 :", e)
        continue

    #                   매수 대기                    #
    start = time.time()
    Complete = 0
    while True:
        try:
            #       BuyOrder is dict / in Error 걸러주기        #
            #       체결되어도 buy_wait은 기다려야한다.
            if type(BuyOrder) != tuple:
                balance = bithumb.get_balance(Coin)
                if (balance[0] - start_coin) * limit_buy_price > 1000:
                    pass
                else:
                    print('BuyOrder is not tuple')
                    break

            #   매수가 취소된 경우를 걸러주어야 한다.
            else:
                if bithumb.get_outstanding_order(BuyOrder) is None:
                    balance = bithumb.get_balance(Coin)
                    #   체결량이 존재하는 경우는 buy_wait 까지 기다린다.
                    if (balance[0] - start_coin) * limit_buy_price > 1000:
                        pass
                    else:
                        print("매수가 취소되었습니다.\n")
                        remain = krw - balance[2]
                        print("거래해야할 원화 : ", remain)
                        break

            #   반 이상 체결된 경우 : 체결된 코인량이 매수 코인량의 반 이상인경우
            if (balance[0] - start_coin) / buyunit >= 0.7:
                Complete = 1
                #   부분 체결되지 않은 미체결 잔량을 주문 취소
                print("    매수 체결    ", end=' ')
                CancelOrder = bithumb.cancel_order(BuyOrder)
                print("부분 체결 :", CancelOrder)
                print()
                time.sleep(1 / 130)
                break

            #   최대 10분 동안 매수 체결을 대기한다.
            if time.time() - start > 60 * buy_wait:
                balance = bithumb.get_balance(Coin)
                if (balance[0] - start_coin) * limit_buy_price > 1000:
                    Complete = 1
                    print("    매수 체결    ")
                    CancelOrder = bithumb.cancel_order(BuyOrder)
                else:
                    # outstanding >> None 출력되는 이유 : BuyOrder == None, 체결 완료
                    if bithumb.get_outstanding_order(BuyOrder) is None:
                        if type(BuyOrder) == tuple:
                            # 한번 더 검사 (get_outstanding_order 찍으면서 체결되는 경우가 존재한다.)
                            if (balance[0] - start_coin) * limit_buy_price > 1000:
                                Complete = 1
                                print("    매수 체결    ")
                                CancelOrder = bithumb.cancel_order(BuyOrder)
                        else:
                            # BuyOrder is None 에도 체결된 경우가 존재함
                            # 1000 원은 거래 가능 최소 금액
                            if (balance[0] - start_coin) * limit_buy_price > 1000:
                                Complete = 1
                                print("    매수 체결    ")
                                CancelOrder = bithumb.cancel_order(BuyOrder)
                            else:
                                print("미체결 또는 체결량 1000 KRW 이하\n")
                                print()
                    else:
                        if type(BuyOrder) == tuple:
                            CancelOrder = bithumb.cancel_order(BuyOrder)
                            print("미체결 또는 체결량 1000 KRW 이하")
                            print(CancelOrder)
                            print()
                break

        except Exception as e:
            print('매수 체결 여부 확인중 에러 발생 :', e)

    # 지정 시간 초과로 루프 나온 경우
    if Complete == 0:
        print()
        continue
    #                    매수 체결 완료                    #

    else:
        #           SELL SIGNAL CHECK           #
        # sell_signal = 0
        while True:

            try:
                while True:
                    #        바로 전 데이터를 찾을때까지 반복      #
                    try:
                        df = pybithumb.get_ohlcv(Coin, interval='minute1', proxy=proxy)
                        current_min_minus1 = datetime.now().minute - 1
                        df_last_min = int(str(df.index[-1]).split()[1].split(':')[1])
                        if current_min_minus1 < 0:
                            current_min_minus1 += 60
                        if current_min_minus1 == df_last_min or current_min_minus1 - df_last_min > 1:
                            break
                        time.sleep(1)
                        proxy = 'on'

                    except Exception as e:
                        print('Error in get_ohlcv :', e)
                        proxy = None

                Fisher = TA.FISH(df, period=60)
                if Fisher.iloc[-1] >= short_signal_value:
                    print('Fisher value at short position :', Fisher.iloc[-1])
                    break

            except Exception as e:
                print('Error in sell signal check :', e)

        #           매도 등록           #
        limit_sell_price = df.close.iloc[-1]
        #           매도 대기           #
        while True:

            try:
                #   매도 진행
                balance = bithumb.get_balance(Coin)
                sellunit = int((balance[0]) * 10000) / 10000.0
                SellOrder = bithumb.sell_limit_order(Coin, limit_sell_price,
                                                     sellunit, 'KRW')
                print("    %s 지정가 매도     " % Coin, end=' ')
                # print("    %s 시장가 매도     " % Coin, end=' ')
                # SellOrder = bithumb.sell_limit_order(Coin, limit_sell_pricePlus, sellunit, "KRW")
                print(SellOrder)
                break

            except Exception as e:
                print('Error in %s high predict :' % Coin, e)

        sell_switch = 0
        exit_count = 0
        while True:

            #                   매도 재등록                    #
            try:
                if sell_switch == 1:

                    #   SellOrder Initializing
                    CancelOrder = bithumb.cancel_order(SellOrder)
                    if CancelOrder is False:  # 남아있는 매도 주문이 없다. 취소되었거나 체결완료.
                        #       체결 완료       #
                        if bithumb.get_order_completed(SellOrder)['data']['order_status'] != 'Cancel':
                            print("    매도 체결    ", end=' ')

                            #   최종 원화 - (보유 원화 - 주문 원화) / 주문 원화 = 변화한 원화량 / 주문 원화량
                            profits *= (bithumb.get_balance(Coin)[2] - (
                                    krw - invest_krw)) / invest_krw
                            print("Accumulated Profits : %.6f\n" % profits)
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
                    if exit_count != -1:
                        SellOrder = bithumb.sell_limit_order(Coin,
                                                             limit_sell_price, sellunit,
                                                             "KRW")
                        print("    %s 지정가 매도     " % Coin, end=' ')
                    else:
                        while True:
                            if datetime.now().second >= 55:
                                break
                        SellOrder = bithumb.sell_market_order(Coin, sellunit, 'KRW')
                        print("    %s 시장가 매도     " % Coin, end=' ')
                    print(SellOrder)

                    sell_switch = -1
                    time.sleep(1 / 130)  # 너무 빠른 거래로 인한 오류 방지

                    # 지정 매도 에러 처리
                    if type(SellOrder) in [tuple, str]:
                        pass
                    elif SellOrder is None:  # 매도 함수 자체 오류 ( 서버 문제 같은 경우 )
                        sell_switch = 1
                        continue
                    else:  # dictionary
                        # 체결 여부 확인 로직
                        ordersucceed = bithumb.get_balance(Coin)
                        #   매도 주문 넣었을 때의 잔여 코인과 거래 후 코인이 다르고 사용중인 코인이 없으면 매도 체결
                        #   거래 중인 코인이 있으면 매도체결이 출력되지 않는다.
                        if ordersucceed[0] != balance[0] and ordersucceed[1] == 0.0:
                            print("    매도 체결    ", end=' ')

                            profits *= (bithumb.get_balance(Coin)[2] - (
                                    krw - invest_krw)) / invest_krw
                            print("Accumulated Profits : %.6f\n" % profits)
                            break
                        else:
                            sell_switch = 1
                            continue

            except Exception as e:
                print("매도 재등록 중 에러 발생 :", e)
                sell_switch = 1
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

                                profits *= (bithumb.get_balance(Coin)[2] - (
                                        krw - invest_krw)) / invest_krw
                                print("Accumulated Profits : %.6f\n" % profits)
                                break
                            elif bithumb.get_outstanding_order(SellOrder) is not None:  # 혹시 모르는 미체결
                                continue
                            else:
                                print("매도 주문이 취소되었습니다.\n")

                                if sell_switch in [0, -1]:
                                    sell_switch = 1
                                    # elif sell_switch == 0:
                                    #     sppswitch = 1
                                    time.sleep(random.random() * 5)
                                continue

                        except Exception as e:
                            print('SellOrder in [tuple, str] ? 에서 에러발생 :', e)

                            time.sleep(random.random() * 5)  # 서버 에러인 경우
                            continue

                    # 매도 등록 에러라면 ? 제대로 등록 될때까지 재등록 ! 지정 에러, 하향 에러
                    elif SellOrder is None:  # limit_sell_order 가 아예 안되는 경우
                        if sell_switch in [0, -1]:
                            sell_switch = 1
                            # elif sell_switch == 0:
                            #     sppswitch = 1
                            time.sleep(random.random() * 5)
                        continue

                    else:  # dictionary
                        # 체결 여부 확인 로직
                        ordersucceed = bithumb.get_balance(Coin)
                        if ordersucceed[0] != balance[0] and ordersucceed[1] == 0.0:
                            print("    매도 체결    ", end=' ')

                            profits *= (bithumb.get_balance(Coin)[2] - (
                                    krw - invest_krw)) / invest_krw
                            print("Accumulated Profits : %.6f\n" % profits)
                            break
                        else:
                            if sell_switch in [0, -1]:
                                sell_switch = 1
                                # elif sell_switch == 0:
                                #     sppswitch = 1
                                time.sleep(random.random() * 5)
                            continue

                #       미체결 대기 / 손절 시그널 체킹        #
                else:
                    while True:
                        #        바로 전 데이터를 찾을때까지 반복      #
                        try:
                            df = pybithumb.get_ohlcv(Coin, interval='minute1', proxy=proxy)
                            current_min_minus1 = datetime.now().minute - 1
                            df_last_min = int(str(df.index[-1]).split()[1].split(':')[1])
                            if current_min_minus1 < 0:
                                current_min_minus1 += 60
                            if current_min_minus1 == df_last_min or current_min_minus1 - df_last_min > 1:
                                break
                            time.sleep(1)
                            proxy = 'on'

                        except Exception as e:
                            print('Error in get_ohlcv :', e)
                            proxy = None

                    # if pybithumb.get_current_price(Coin) < limit_exit_price and exit_count == 0:
                    Fisher = TA.FISH(df, period=60)
                    if Fisher.iloc[-1] < short_signal_value and exit_count == 0:
                        print('Fisher value at exit position :', Fisher.iloc[-1])
                        limit_sell_price = Funcs_MACD_OSC.clearance(limit_sell_price * 0.9975)
                        sell_switch = 1
                        exit_count = 1
                        exit_start = time.time()

                    else:
                        time.sleep(1 / 130)

                    #       손절 매도 미체결 시      #
                    #       손절 매도 등록하고 지정한 초가 지나면, 55초 이후로 시장가 매도 #
                    #       시장가 미체결시 계속해서 매도 등록한다.      #
                    try:
                        if time.time() - exit_start >= 30:
                            limit_sell_price = Funcs_MACD_OSC.clearance(limit_sell_price * 0.9975)
                            sell_switch = 1
                            # exit_count = -1     # '-1'은 시장가 매도를 의미한다.
                            exit_start = time.time()

                    except Exception as e:
                        pass

            except Exception as e:  # 지정 매도 대기 중 에러나서 이탈 매도가로 팔아치우는거 방지하기 위함.
                print('취소 / 체결완료 / SellOrder = None / dict 확인중 에러 발생 :', e)
                continue

