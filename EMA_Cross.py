import pybithumb
import Funcs_CNN2
import time
from datetime import datetime
import random
import numpy as np
import pandas as pd
from keras.models import load_model
import os
from Funcs_CNN4 import macd, cmo, ema_ribbon
from Make_X_Cross_Long_no_candle import made_x
import Make_X_Cross_no_candle
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
input_data_length = 30
input_data_length_cl = 30
model_num = '57_ohlc'
model_num_cl = '73_cmo'
# crop_size_low = 500     # 적당히 크면 안정적
# crop_size_high = 300    # 작을수록 손절모드에 적합하다 : LIMIT_LINE 과 조합해서 사용해야한다.
# crop_size_sudden_death = 100    # 작을수록 손절모드에 적합하다 : LIMIT_LINE 과 조합해서 사용해야한다.
# limit_line_low = 0.97   # 최저점 선정 모드
# limit_line_high = 0.8   # 익절 모드
# limit_line_sudden_death = 0.45  # 손절 모드
# check_span = 50

#       Trade Info      #
#                                           Check The Money                                              #
CoinVolume = 10
buy_wait = 4  # minute
Profits = 1.0

#       Model Fitting       #
model = load_model('./model/rapid_ascending %s_%s.hdf5' % (input_data_length, model_num))
model_cl = load_model('./model/rapid_ascending %s_%s.hdf5' % (input_data_length_cl, model_num_cl))

while True:

    #               Finding Buy Signal              #
    buy = 0
    while True:

        #           Making TopCoin List         #
        Coinlist = pybithumb.get_tickers()
        Fluclist = []

        try:
            for Coin in Coinlist:
                tickerinfo = pybithumb.PublicApi.ticker(Coin)
                data = tickerinfo['data']
                fluctate = data['fluctate_rate_24H']
                Fluclist.append(fluctate)
                time.sleep(1 / 90)

        except Exception as e:
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
                    if datetime.now().second >= 6:
                        break

                #   User-Agent 설정 & IP 변경으로 크롤링 우회한다.
                print('Loading %s MACD_Cross' % Coin)
                time.sleep(random.random() * 5)

                #               진입 조건              #
                #           EMA CROSSOVER             #
                if (datetime.now().minute % 5) in [0, 1, 2]:
                    ohlcv_excel = pybithumb.get_ohlcv(Coin, 'KRW', 'minute1')
                else:
                    ohlcv_excel = pybithumb.get_ohlcv(Coin, 'KRW', 'minute1', proxy='on')

                macd(ohlcv_excel, short=15, long=60)
                ohlcv_excel['CMO'] = cmo(ohlcv_excel)

                closeprice = ohlcv_excel['close'].iloc[-1]
                # ema15 = ohlcv_excel['EMA_1'].iloc[-1]
                # if closeprice < ema15 * 1.015:
                buy_price = closeprice + 2 * Funcs_CNN2.GetHogaunit(closeprice)
                # else:
                #     buy_price = ema15 * 1.01

                trade_state = [0] * len(ohlcv_excel)
                for i in range(1, len(ohlcv_excel)):
                    if ohlcv_excel['MACD'][i - 1] <= 0:
                        if ohlcv_excel['MACD'][i] > 0:
                            trade_state[i] = 1

                    if ohlcv_excel['MACD'][i - 1] >= 0:
                        if ohlcv_excel['MACD'][i] < 0:
                            trade_state[i] = 2

                if trade_state[-1] == 1:

                    # result = made_x(ohlcv_excel, input_data_length, model_num, crop_size=input_data_length)
                    # X_test = np.array(result[0])
                    # row = X_test.shape[1]
                    # col = X_test.shape[2]
                    # # col2 = X_test.shape[2]
                    #
                    # X_test = X_test.astype('float32').reshape(-1, row, col, 1)[:, :, [0, 1, 2, 3, 7]]
                    # Y_pred_ = model_cl.predict(X_test, verbose=1)
                    # Y_pred = np.argmax(Y_pred_, axis=1)
                    #
                    # if Y_pred[-1] == 1.:
                    buy = 1
                    break

            except Exception as e:
                print('Error in %s low predict :' % Coin, e)
                continue

        if buy == 1:
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
        money = krw * 0.996
        # money = 10000
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
            #       체결되어도 buy_wait 은 기다려야한다.
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
            balance = bithumb.get_balance(Coin)
            if (balance[0] - start_coin) / buyunit >= 0.5:
                Complete = 1
                #   부분 체결되지 않은 미체결 잔량을 주문 취소
                CancelOrder = bithumb.cancel_order(BuyOrder)
                print("부분 체결 :", CancelOrder, '사용 원화 :', (balance[0] - start_coin) * limit_buy_price)
                print()
                time.sleep(1 / 80)
                break

            #   buy_wait 동안 매수 체결을 대기한다.
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
        #           매도 대기           #
        while True:

            try:
                #   이전 종가가 완성된 시기
                #   EMA CROSSUNDER CHECK
                if datetime.now().second == 6:
                    while True:
                        try:
                            ohlcv_excel = pybithumb.get_ohlcv(Coin, 'KRW', 'minute1')
                            break

                        except Exception as e:
                            print('Error in getting ohlcv_excel :', e)
                            time.sleep(random.random() * 5)

                    # ema_ribbon(ohlcv_excel, ema_1=3, ema_2=5, ema_3=8)
                    macd(ohlcv_excel, short=15, long=60)
                    trade_state = [0] * len(ohlcv_excel)
                    # for i in range(1, len(ohlcv_excel)):
                    #     if ohlcv_excel['EMA_1'][i - 1] <= ohlcv_excel['EMA_3'][i - 1]:
                    #         if ohlcv_excel['EMA_1'][i] > ohlcv_excel['EMA_3'][i]:
                    #             trade_state[i] = 1
                    #
                    #     if ohlcv_excel['EMA_1'][i - 1] >= ohlcv_excel['EMA_3'][i - 1]:
                    #         if ohlcv_excel['EMA_1'][i] < ohlcv_excel['EMA_3'][i]:
                    #             trade_state[i] = 2
                    for i in range(1, len(ohlcv_excel)):
                        if ohlcv_excel['MACD'][i - 1] <= ohlcv_excel['MACD_Signal'][i - 1]:
                            if ohlcv_excel['MACD'][i] > ohlcv_excel['MACD_Signal'][i]:
                                trade_state[i] = 1

                        if ohlcv_excel['MACD'][i - 1] >= ohlcv_excel['MACD_Signal'][i - 1]:
                            if ohlcv_excel['MACD'][i] < ohlcv_excel['MACD_Signal'][i]:
                                trade_state[i] = 2

                    if trade_state[-1] == 2:
                        balance = bithumb.get_balance(Coin)
                        sellunit = int((balance[0]) * 10000) / 10000.0
                        SellOrder = bithumb.sell_market_order(Coin, sellunit, 'KRW')
                        print("    %s 시장가 매도     " % Coin, end=' ')
                        # SellOrder = bithumb.sell_limit_order(Coin, limit_sell_pricePlus, sellunit, "KRW")
                        # print("##### %s %s KRW 지정 매도 재등록 #####" % (Coin, limit_sell_pricePlus))
                        print(SellOrder)
                        break

                #   현재 종가가 dataframe 으로 완성될 시기
                #   고점 예측
                if datetime.now().second == 55:
                    while True:
                        try:
                            ohlcv_excel = pybithumb.get_ohlcv(Coin, 'KRW', 'minute1')
                            X_test_high, _ = Make_X_Cross_no_candle.made_x(ohlcv_excel, input_data_length, model_num,
                                                                                crop_size=input_data_length)

                            X_test_high = np.array(X_test_high)
                            row = X_test_high.shape[1]
                            col = X_test_high.shape[2]
                            X_test_high = X_test_high.astype('float32').reshape(-1, row, col, 1)[:, :, :4]

                            Y_pred_high_ = model.predict(X_test_high, verbose=3)
                            Y_pred_high = np.argmax(Y_pred_high_, axis=1)
                            break

                        except Exception as e:
                            print('Error in getting low_high data :', e)
                            time.sleep(random.random() * 5)

                    # max_value_high = np.max(Y_pred_high_, axis=0)
                    # Y_pred_high = np.zeros(len(Y_pred_high_))
                    # for i in range(len(Y_pred_high_)):
                    #     if Y_pred_high_[i][1] > max_value_high[1] * 0.9:
                    #         Y_pred_high[i] = 2

                    #   매도 진행
                    if Y_pred_high[-1] == 1.:
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

