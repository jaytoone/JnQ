import pybithumb
import Funcs_CNN2
import time
from datetime import datetime
import random
import numpy as np
import pandas as pd
from keras.models import load_model
import os
from Make_X_total import low_high
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


#       KEY SETTING     #
with open("Keys.txt") as f:
    lines = f.readlines()
    key = lines[0].strip()
    secret = lines[1].strip()
    bithumb = pybithumb.Bithumb(key, secret)

#       Params      #
input_data_length = 54
model_num = '122_ohlc'

#       Trade Info      #
#                                           Check The Money                                              #
CoinVolume = 10
buy_wait = 5  # minute
Profits = 1.0

#       Model Fitting       #
model = load_model('./model/rapid_ascending %s_%s.hdf5' % (input_data_length, model_num))

while True:

    #               Finding Buy Signal              #
    while True:

        #           Making TopCoin List         #
        Coinlist = pybithumb.get_tickers()
        Fluclist = []
        while True:
            try:
                for Coin in Coinlist:
                    tickerinfo = pybithumb.PublicApi.ticker(Coin)
                    data = tickerinfo['data']
                    fluctate = data['fluctate_rate_24H']
                    Fluclist.append(fluctate)
                    time.sleep(1 / 90)
                break

            except Exception as e:
                Fluclist.append(None)
                print('Error in making Topcoin :', e)
                continue

        Fluclist = list(map(float, list(filter(None, Fluclist))))
        series = pd.Series(Fluclist, Coinlist)
        series = series.sort_values(ascending=False)

        series = series[:CoinVolume]
        TopCoin = list(series.index)
        # TopCoin = list(map(str.upper, ['dvp']))

        for Coin in TopCoin:
            try:
                while True:
                    if datetime.now().second >= 5:
                        break

                #   User-Agent 설정 & IP 변경으로 크롤링 우회한다.
                print('Loading %s low_high' % Coin)
                time.sleep(random.random() * 5)

                #               predict 조건              #
                #   ohlcv_data 의 price_gap 가 1.07 이하이면 predict 하지 않는다.
                #   closeprice 가 MinMaxScaler() 로 0.3 보다 크면 predict 하지 않는다.
                #   ohlcv_data_length 가 100 이하이면 predict 하지 않는다.
                #   예측 저점이 이전 저점 보다 0.1 이상이 아니면 거래하지 않는다.
                if (datetime.now().minute % 5) in [0, 1, 2]:
                    X_test, buy_price, _, start_datetime_list = low_high(Coin, input_data_length, crop_size=input_data_length, lowhigh_point='on')  # TopCoin 으로 제약조건을 걸어서 trade_limit==0
                else:
                    X_test, buy_price, _, start_datetime_list = low_high(Coin, input_data_length, 'proxy', crop_size=input_data_length, lowhigh_point='on')

                if X_test is not None:
                    Y_pred_ = model.predict(X_test, verbose=1)
                    Y_pred = np.argmax(Y_pred_, axis=1)
                    # max_value = np.max(Y_pred_, axis=0)
                    # limit_line = limit_line_low

                    if (Y_pred[-1] > 0.5) and (Y_pred[-1] < 1.5):
                        break

            except Exception as e:
                print('Error in %s low predict :' % Coin, e)
                continue

        if (Y_pred[-1] > 0.5) and (Y_pred[-1] < 1.5):
            break

    #                매수 등록                  #
    try:
        # 호가단위, 매수가, 수량
        limit_buy_price = Funcs_CNN2.clearance(buy_price)

        # -------------- 보유 원화 확인 --------------#
        balance = bithumb.get_balance(Coin)
        start_coin = balance[0]
        krw = balance[2]
        print()
        print("보유 원화 :", krw, end=' ')
        # money = krw * 0.996
        money = 10000
        print("주문 원화 :", money)
        if krw < 1000:
            print("거래가능 원화가 부족합니다.\n")
            continue
        print()

        # 매수량
        buyunit = int((money / limit_buy_price) * 10000) / 10000.0

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
            #       체결되어도 10분은 기다려야한다.
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
                        break

            #   반 이상 체결된 경우 : 체결된 코인량이 매수 코인량의 반 이상인경우
            # if (balance[0] - start_coin) / buyunit >= 0.5:
            #     Complete = 1
            #     #   부분 체결되지 않은 미체결 잔량을 주문 취소
            #     print("    매수 체결    ", end=' ')
            #     CancelOrder = bithumb.cancel_order(BuyOrder)
            #     print("부분 체결 :", CancelOrder)
            #     print()
            #     time.sleep(1 / 80)
            #     break

            #   최대 10분 동안 매수 체결을 대기한다.
            if time.time() - start > buy_wait * 60:
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

    #                    매수 체결                     #
    else:

        find_new_low = 1
        check_new_low = 0
        # limit_line = limit_line_high
        #           매도 대기           #
        while True:
            # 분당 한번 high_check predict 했을때, 1 결과값이 출력되면 매도 진행
            try:
                #   이전 종가가 dataframe 으로 완성될 시기
                if datetime.now().second == 55:
                    while True:
                        try:
                            # if find_new_low == 1:
                            #     X_test_high, _, _, end_datetime_list = low_high(Coin, input_data_length,
                            #                                                     crop_size=crop_size_high)
                            #
                            #     #       Start_datetime 의 인덱스 번호 찾기      #
                            #     for i in range(len(end_datetime_list)):
                            #         if end_datetime_list[i] == start_datetime_list[-1]:
                            #
                            #             #       len(end_datetime_list) - 시작 인덱스 > check_span 이면 저점 갱신 확인       #
                            #             if len(end_datetime_list) - i > check_span:
                            #                 X_test_low, _, _, _ = low_high(Coin, input_data_length,
                            #                                                                      crop_size=crop_size_low)
                            #                 check_new_low = 1
                            #             break
                            X_test_high, _, _, _ = low_high(Coin, input_data_length,
                                                                                crop_size=input_data_length, lowhigh_point='on')
                            break

                        except Exception as e:
                            print('Error in getting low_high data :', e)
                            time.sleep(random.random() * 5)

                    #       check_span 지나면 저점 갱신 확인        #
                    # if check_new_low == 1:
                    #     Y_pred_low_ = model.predict(X_test_low, verbose=3)
                    #     max_value_low = np.max(Y_pred_low_, axis=0)
                    #
                    #     if Y_pred_low_[-1][1] > max_value_low[1] * limit_line_low:
                    #         limit_line = limit_line_sudden_death
                    #         X_test_high, _, _, _ = low_high(Coin, input_data_length,
                    #                                                         crop_size=crop_size_sudden_death)
                    #         find_new_low = 0
                    #         check_new_low = 0

                    #       X_test_high None 이 나오면 이전 저점보다 종가가 낮아지고 있는 상황       #
                    if X_test_high is None:
                        balance = bithumb.get_balance(Coin)
                        sellunit = int((balance[0]) * 10000) / 10000.0
                        SellOrder = bithumb.sell_market_order(Coin, sellunit, 'KRW')
                        print("    %s 시장가 매도     " % Coin, end=' ')
                        # SellOrder = bithumb.sell_limit_order(Coin, limit_sell_pricePlus, sellunit, "KRW")
                        # print("##### %s %s KRW 지정 매도 재등록 #####" % (Coin, limit_sell_pricePlus))
                        print(SellOrder)
                        break

                    # else:
                    Y_pred_high_ = model.predict(X_test_high, verbose=3)
                    # max_value_high = np.max(Y_pred_high_, axis=0)
                    Y_pred_high = np.argmax(Y_pred_high_, axis=1)

                    #   매도 진행
                    if (Y_pred_high[-1] > 1.5) and (Y_pred_high[-1] < 2.5):
                        balance = bithumb.get_balance(Coin)
                        sellunit = int((balance[0]) * 10000) / 10000.0
                        SellOrder = bithumb.sell_market_order(Coin, sellunit, 'KRW')
                        print("    %s 시장가 매도     " % Coin, end=' ')
                        # SellOrder = bithumb.sell_limit_order(Coin, limit_sell_pricePlus, sellunit, "KRW")
                        # print("##### %s %s KRW 지정 매도 재등록 #####" % (Coin, limit_sell_pricePlus))
                        print(SellOrder)
                        break

                    #   Check Manual Trade
                    # else:
                    #     ordersucceed = bithumb.get_balance(Coin)
                    #     if ordersucceed[0] != balance[0]:
                    #         print("    매도 체결    ", end=' ')
                    #         Profits *= (bithumb.get_balance(Coin)[2] - (krw - money)) / money
                    #         print("Accumulated Profits : %.6f\n" % Profits)
                    #         break

            except Exception as e:
                print('Error in %s high predict :' % Coin, e)

        sell_switch = 0
        while True:

            #                   매도 재등록                    #
            try:
                if sell_switch == 1:

                    #   SellOrder Initializing
                    CancelOrder = bithumb.cancel_order(SellOrder)
                    if CancelOrder is False:  # 남아있는 매도 주문이 없다. 취소되었거나 체결완료.
                        print("    매도 체결    ", end=' ')

                        #   최종 원화 - (보유 원화 - 주문 원화) / 주문 원화 = 변화한 원화량 / 주문 원화량
                        Profits *= (bithumb.get_balance(Coin)[2] - (krw - money)) / money
                        print("Accumulated Profits : %.6f\n" % Profits)
                        break
                    elif CancelOrder is None:  # SellOrder = none 인 경우
                        # 매도 재등록 해야함 ( 등록될 때까지 )
                        pass
                    else:
                        print("    매도 취소    ", end=' ')
                        print(CancelOrder)
                    print()

                    balance = bithumb.get_balance(Coin)
                    sellunit = int((balance[0]) * 10000) / 10000.0
                    SellOrder = bithumb.sell_market_order(Coin, sellunit, 'KRW')
                    print("    %s 시장가 매도     " % Coin, end=' ')
                    # SellOrder = bithumb.sell_limit_order(Coin, limit_sell_pricePlus, sellunit, "KRW")
                    # print("##### %s %s KRW 지정 매도 재등록 #####" % (Coin, limit_sell_pricePlus))
                    print(SellOrder)

                    sell_switch = -1
                    time.sleep(1 / 80)  # 너무 빠른 거래로 인한 오류 방지

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
                            Profits *= (bithumb.get_balance(Coin)[2] - (krw - money)) / money
                            print("Accumulated Profits : %.6f\n" % Profits)
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
                                Profits *= (bithumb.get_balance(Coin)[2] - (krw - money)) / money
                                print("Accumulated Profits : %.6f\n" % Profits)
                                break
                            elif bithumb.get_outstanding_order(SellOrder) is not None:  # 혹시 모르는 미체결
                                continue
                            else:
                                print("매도 주문이 이미 취소되었습니다.\n")
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
                            Profits *= (bithumb.get_balance(Coin)[2] - (krw - money)) / money
                            print("Accumulated Profits : %.6f\n" % Profits)
                            break
                        else:
                            if sell_switch in [0, -1]:
                                sell_switch = 1
                            # elif sell_switch == 0:
                            #     sppswitch = 1
                                time.sleep(random.random() * 5)
                            continue

                else:  # 미체결량 대기 파트
                    if sell_switch == 0:  # 이탈 매도 대기
                        sell_switch = 1
                        time.sleep(random.random() * 5)
                    # else:  # 지정 매도 대기
                    #     time.sleep(1 / 80)

            except Exception as e:  # 지정 매도 대기 중 에러나서 이탈 매도가로 팔아치우는거 방지하기 위함.
                print('취소 / 체결완료 / SellOrder = None / dict 확인중 에러 발생 :', e)
                continue

