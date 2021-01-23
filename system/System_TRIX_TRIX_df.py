import pybithumb
import Funcs_MACD_OSC
import time
from datetime import datetime
import random
from finta import TA
import numpy as np
# import pandas as pd
# import os
import warnings

warnings.filterwarnings("ignore")
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import System_TRIX_TRIX_Funcs
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from fake_useragent import UserAgent

#       TRADE INFO      #
interval_key1 = 3
proxy = 'on'

#       KEY SET     #
with open("Keys.txt") as f:
    lines = f.readlines()
    key = lines[0].strip()
    secret = lines[1].strip()
    bithumb = pybithumb.Bithumb(key, secret)

#       SIGNAL VALUE SET      #
long_signal_value = -0.8
short_signal_value = 2.45
high_check_value = 1.7
high_check_value_exit = 0.2
high_check_value2 = 5.0
high_check_value2_exit = 1.2

#       INDICATOR SET       #
fisher_short_period = 60
dc_period_entry = 20
dc_period = 80
dc_close_accept_gap_range = 0.11
dc_accept_gap_range = -0.03

trix_vertical_gap = 0.13
trix_horizontal_ticks = 5
trix_exit_gap = -0.01

# dc_accept_range = 0.05
# smma_period = 200
# smma_accept_range = -0.08

#       TRADE SET       #
CoinVolume = 7
buy_wait = 5  # minute
profits = 1.0
headless = 1

#       CHROME HEADLESS SETTING     #
options = webdriver.ChromeOptions()
options.add_argument('headless')
options.add_argument('window-size=1920x1080')
# options.add_argument("disable-gpu")
ua = UserAgent()
options.add_argument("user-agent=%s" % ua.random)
options.add_argument("lang=ko_KR")

#   CHROME SETTING  #
path = "C:/Users/Lenovo/PycharmProjects/Project_System_Trading/Rapid_Ascend/chromedriver.exe"
class_type = "<class 'selenium.webdriver.remote.webelement.WebElement'>"
if headless == 1:
    driver = webdriver.Chrome(path, chrome_options=options)
else:
    driver = webdriver.Chrome(path)

System_TRIX_TRIX_Funcs.open_coin_list(driver)

#                                           CHECK INVEST_KRW                                              #
print('#                    TRADE START                     #')
#               거래 종료 후 다시 시작         #
while True:

    #           INFINITE LOOP FOR BUY SIGNAL           #
    buy_switch = 0
    while True:

        #              GET TOP COIN LIST             #
        try:
            #           TOP COIN LIST TAB으로 가서      #
            driver.switch_to.window(driver.window_handles[1])
            driver.implicitly_wait(3)
            #           주기적으로 SORTING 작업을 한다.         #
            sort_btn = driver.find_elements_by_class_name('sorting')[2]
            sort_btn.click()
            sort_btn.click()
            driver.implicitly_wait(3)

            coin_list = driver.find_element_by_class_name('coin_list')
            coins_info = coin_list.find_elements_by_tag_name('tr')
            TopCoin = list()
            for coin_info in coins_info:
                TopCoin.append(coin_info.find_element_by_class_name('sort_coin').text.split('/')[0])
                if len(TopCoin) >= CoinVolume:
                    break
            print(TopCoin)

        except Exception as e:
            print('Error in getting TopCoin :', e)
            continue
        # TopCoin = list(map(str.upper, ['dvp']))

        for Coin in TopCoin:

            while True:
                #           GET DF VALUE         #
                try:
                    df = pybithumb.get_ohlcv(Coin, interval='minute3', proxy=proxy)
                    if df is not None:
                        break
                    else:
                        proxy = None

                except Exception as e:
                    print('Error in get_ohlcv :', e)

            #       TRADE CONDITION     #
            chart_height = (df['high'].max() - df['low'].min())

            df['TRIX'] = TA.TRIX(df, period=18)
            df['TRIX'] = np.where(np.isnan(df['TRIX']), 0, df['TRIX'])
            df['TRIX_TREND'] = np.where(df['TRIX'].shift(1) <= df['TRIX'], 'UP', 'DOWN')

            df['TRIX_LATER'] = TA.TRIX(df, period=20)
            df['TRIX_LATER'] = np.where(np.isnan(df['TRIX_LATER']), 0, df['TRIX_LATER'])

            df['Fisher_short'] = TA.FISH(df, period=fisher_short_period)

            df['DC_LOW_ENTRY'] = TA.DO(df, lower_period=dc_period_entry, upper_period=dc_period_entry)['LOWER']

            df['DC_LOW'] = TA.DO(df, lower_period=dc_period, upper_period=dc_period)['LOWER']
            df['DC_TREND'] = np.where(df['DC_LOW'].shift(1) <= df['DC_LOW'], 'UP', 'DOWN')
            for i in range(1, len(df)):
                if df['DC_LOW'].iloc[i] == df['DC_LOW'].iloc[i - 1]:
                    df['DC_TREND'].iloc[i] = df['DC_TREND'].iloc[i - 1]

            #       TRADE CONDITION     #
            if df['TRIX_LATER'].iloc[-1] < long_signal_value:
                if df['TRIX'].iloc[-1] > df['TRIX'].iloc[-2] < df['TRIX'].iloc[-3]:

                    #       매수하려는 가격과 DC를 비교한다.     #
                    if (df['close'].iloc[-1] - df['DC_LOW_ENTRY'].iloc[-1]) / chart_height < dc_close_accept_gap_range:
                        if df['DC_TREND'].iloc[-1] == 'UP':
                            buy_switch = 1
                            break  # GET OUT OF THE FOR LOOP

                        else:
                            current_index = -1
                            while True:
                                current_index -= 1
                                if df['DC_TREND'].iloc[current_index] == 'UP':
                                    break
                            if (df['DC_LOW'].iloc[-1] - df['DC_LOW'].iloc[current_index]) / chart_height > dc_accept_gap_range:
                                buy_switch = 1
                                break   # GET OUT OF FOR LOOP

            #   TOTAL ELSE IN FOR LOOP  #
            time.sleep(1 / 130)

        if buy_switch == 1:
            print('%s indicator value at long position' % Coin)
            print('TRIX :', df['TRIX'].iloc[-1], df['TRIX'].iloc[-2], df['TRIX'].iloc[-3])
            print('DC_TREND :', df['DC_TREND'].iloc[-1])
            print('DC CLOSE GAP :', (df['close'].iloc[-1] - df['DC_LOW_ENTRY'].iloc[-1]) / chart_height)
            print('DC GAP :', (df['DC_LOW'].iloc[-1] - df['DC_LOW'].iloc[current_index]) / chart_height)

            while True:
                #        바로 전 데이터를 찾을때까지 반복      #
                #        이전 종가로 매수가를 지정하기 위함        #
                #           interval time = 1m          #
                try:
                    df = pybithumb.get_ohlcv(Coin, interval='minute1')
                    current_min_minus1 = datetime.now().minute - 1
                    df_last_min = int(str(df.index[-1]).split()[1].split(':')[1])
                    if current_min_minus1 < 0:
                        current_min_minus1 += 60
                    if current_min_minus1 == df_last_min or current_min_minus1 - df_last_min > 1:
                        break
                    time.sleep(1)

                except Exception as e:
                    print('Error in get_ohlcv :', e)

            break   # GET OUT OF THE WHILE LOOP

    #                   KRW TO COIN CONVERTING PROCESS                  #
    start = time.time()
    limit_buy_price = df.close.iloc[-1]
    first_buy = 1
    buy_enroll = 1
    complete = 0
    while True:

        #                매수 등록                  #
        try:
            if buy_switch == 1:
                # 호가단위, 매수가, 수량
                limit_buy_price = Funcs_MACD_OSC.clearance(limit_buy_price)

                #           보유 원화 확인          #
                balance = bithumb.get_balance(Coin)
                #       첫 매수 등록시 시작 잔여 코인만 확인한다.    #
                if first_buy:
                    start_coin = balance[0]
                    first_buy = 0
                krw = balance[2]
                print()
                print("보유 원화 :", krw, end=' ')
                # invest_krw = krw * 0.996
                #       TODO MONEY CHECK      #
                invest_krw = 3000
                print("주문 원화 :", invest_krw)
                if krw < 1000:
                    #   체결량이 존재하는 경우 매도 / 실제로 거래가능 원화가 없는 경우 #
                    if (balance[0] - start_coin) * limit_buy_price > 1000:
                        buy_enroll = 0
                    else:
                        print("거래가능 원화가 부족합니다.\n")
                        break
                print()

                if buy_enroll == 1:
                    #   매수량
                    buyunit = int((invest_krw / limit_buy_price) * 10000) / 10000.0

                    #   매수 등록
                    BuyOrder = bithumb.buy_limit_order(Coin, limit_buy_price, buyunit, "KRW")
                    print("    %s %s KRW 매수 등록    " % (Coin, limit_buy_price), end=' ')
                    print(BuyOrder)

                buy_switch = -1
                #   57초부터 60초 사이 여러번 매수 등록하는 걸 방지하기 위함  #
                time.sleep(3)

        except Exception as e:
            print("매수 등록 중 에러 발생 :", e)
            continue

        #           지정 시간이 경과할 때마다 매수가 변경              #
        try:
            #       종가가 첫 매수가의 0.6% 이하이면 해당 종가로 매수한다.       #
            #       빠른 종가에 들어가면 매수 체결 확률을 높일거란 생각에 57초 진입       #
            if datetime.now().second >= 57:

                #       매수 미체결 시        #
                balance = bithumb.get_balance(Coin)
                if (balance[0] - start_coin) / buyunit < 0.7:

                    indicator_value = driver.find_elements_by_class_name('pane-legend-item-value-container')
                    close = float(indicator_value[0].text.split('\n')[3].replace("종", ""))
                    if close / df.close.iloc[-1] <= 1.006:
                        limit_buy_price = close
                    else:
                        limit_buy_price = limit_buy_price * 1.0015

                    #         미체결 매수 취소         #
                    CancelOrder = bithumb.cancel_order(BuyOrder)
                    print('    limit_buy_price change, CancelOrder :', CancelOrder)
                    buy_switch = 1
                    continue

        except Exception as e:
            print('Error in change limit_buy_price :', e)

        #               매수 체결 여부 확인             #
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
                complete = 1
                #   부분 체결되지 않은 미체결 잔량을 주문 취소
                print("    매수 체결    ", end=' ')
                CancelOrder = bithumb.cancel_order(BuyOrder)
                print("부분 체결 :", CancelOrder)
                print()
                time.sleep(1 / 130)
                break

            #       buy_wait 동안의 대기가 끝난 경우        #
            if time.time() - start > 60 * buy_wait:
                balance = bithumb.get_balance(Coin)
                if (balance[0] - start_coin) * limit_buy_price > 1000:
                    complete = 1
                    print("    매수 체결    ")
                    CancelOrder = bithumb.cancel_order(BuyOrder)
                else:
                    # outstanding >> None 출력되는 이유 : BuyOrder == None, 체결 완료
                    if bithumb.get_outstanding_order(BuyOrder) is None:
                        if type(BuyOrder) == tuple:
                            # 한번 더 검사 (get_outstanding_order 찍으면서 체결되는 경우가 존재한다.)
                            if (balance[0] - start_coin) * limit_buy_price > 1000:
                                complete = 1
                                print("    매수 체결    ")
                                CancelOrder = bithumb.cancel_order(BuyOrder)
                        else:
                            # BuyOrder is None 에도 체결된 경우가 존재함
                            # 1000 원은 거래 가능 최소 금액
                            if (balance[0] - start_coin) * limit_buy_price > 1000:
                                complete = 1
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
    if complete == 0:
        print()
        continue
    #                    매수 체결 완료                    #

    else:
        #           SELL SIGNAL CHECK           #
        sell_signal = 0
        exit_signal = 0
        dc_descend_cnt = 0
        previous_fisher = None
        previous_interval_time = None
        previous_interval_key = interval_key1
        previous_data = [np.NaN] * 3
        data = [np.NaN] * 3

        while True:

            #       CHECK THE TIME      #
            #       종가를 서둘러 확인해버리면 dc_descend_cnt 오차 생길 수 있다.       #
            current_datetime = datetime.now()
            if current_datetime.minute % 3 == 2 and current_datetime.second >= 59:

                while True:
                    try:
                        #           GET DATA FROM THE CHARTS            #
                        #       CHECK THE INTERVAL      #
                        interval_time = driver.find_element_by_class_name('pane-legend-title__interval').text
                        indicator_value = driver.find_elements_by_class_name('pane-legend-item-value-container')
                        close = float(indicator_value[0].text.split('\n')[3].replace("종", ""))
                        value1 = float(indicator_value[1].text)
                        value2 = float(indicator_value[2].text.split('\n')[0])
                        value3 = float(indicator_value[2].text.split('\n')[0])
                        value4 = float(indicator_value[3].text.split('\n')[0].replace("−", "-"))

                        #           GET INDICATOR VALUES        #
                        data[0] = dc
                        data[1] = fisher
                        break
                        #          GETTING DATA START FROM INTERVAL_TIME2       #
                        # if interval_time == ', 1':
                        #     data[0] = dc
                        #     data[1] = fisher
                        #     driver.switch_to.window(driver.window_handles[2])
                        #     iFrames = driver.find_elements_by_tag_name('iframe')
                        #     driver.implicitly_wait(3)
                        #     driver.switch_to.frame(iFrames[1])
                        #     driver.implicitly_wait(3)
                        #     break
                        #
                        # else:  # interval time = 3 min
                        #     data[2] = fisher
                        #     driver.switch_to.window(driver.window_handles[0])
                        #     iFrames = driver.find_elements_by_tag_name('iframe')
                        #     driver.implicitly_wait(3)
                        #     driver.switch_to.frame(iFrames[1])
                        #     driver.implicitly_wait(3)

                    except Exception as e:
                        print('Error in getting indicator value :', e)
                        driver.implicitly_wait(3)

                #           EXPOSE INDICATOR VALUE          #
                print(current_datetime.now())
                print('previous_data :', previous_data)
                print('data :', data)

                #       매도 조건       #
                #       3M FISHER는 종가가 완성될때만 검사한다.       #
                # if current_datetime.minute % 3 == 2:
                #     if previous_data[2] >= short_signal_value:
                #         if data[2] < previous_data[2]:
                #             if sell_signal == 0:
                #                 print('Above data be on short position')
                #             sell_signal = 1
                #             break

                #       3M FISHER SHORT        #
                if previous_data[1] >= short_signal_value:
                    if data[1] < previous_data[1]:
                        if sell_signal == 0:
                            print('Above data be on short position')
                        sell_signal = 1
                        break

                #       손절 조건       #
                #       고점을 지나고 3M FISHER SHORT를 찍지 못할 경우       #
                # if data[1] >= high_check_value:
                #     if exit_signal == 0:
                #         print('Above data be on high passed position')
                #         print()
                #     exit_signal = 1
                #
                # elif data[1] >= high_check_value2:
                #     if exit_signal == 0:
                #         print('Above data be on high passed position')
                #         print()
                #     exit_signal = 1.5
                #
                # if exit_signal == 1 and data[1] < high_check_value_exit:
                #     break
                #
                # elif exit_signal == 1.5 and data[1] < high_check_value2_exit:
                #     break

                #       DC LOW LINE DOWN        #
                if data[0] < previous_data[0]:
                    dc_descend_cnt += 1
                    exit_signal = 2
                    break

                # if dc_descend_cnt >= 2:
                #     break

                #       RENEWER PREVIOUS FISHER     #
                #       IF USE SAME LIST FORMAT WITH DATA,
                #       IT PREVIOUS DATA VALUE CHANGES WHILE CRAWL PROCESS      #
                previous_data = list()
                for value in data:
                    previous_data.append(value)
                time.sleep(0.5)

        #           매도 등록           #
        limit_sell_price = Funcs_MACD_OSC.clearance(close)
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

                            profits *= (bithumb.get_balance(Coin)[2] - (
                                    krw - invest_krw)) / invest_krw
                            print("Accumulated Profits : %.6f\n" % profits)
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

                                profits *= (bithumb.get_balance(Coin)[2] - (
                                        krw - invest_krw)) / invest_krw
                                print("Accumulated Profits : %.6f\n" % profits)
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

                            profits *= (bithumb.get_balance(Coin)[2] - (
                                    krw - invest_krw)) / invest_krw
                            print("Accumulated Profits : %.6f\n" % profits)
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
                        if sell_signal != 1:
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
