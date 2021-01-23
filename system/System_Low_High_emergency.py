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
model_num = 23
crop_size_low = 500     # 적당히 크면 안정적
crop_size_high = 300    # 작을수록 손절모드에 적합하다 : LIMIT_LINE 과 조합해서 사용해야한다.
crop_size_sudden_death = 100    # 작을수록 손절모드에 적합하다 : LIMIT_LINE 과 조합해서 사용해야한다.
limit_line_low = 0.97   # 최저점 선정 모드
limit_line_high = 0.65   # 익절 모드
limit_line_sudden_death = 0.45  # 손절 모드
check_span = 1

#       Trade Info      #
#                                           Check The Money                                              #
CoinVolume = 10
buy_wait = 10  # minute
Profits = 1.0

#       Model Fitting       #
model = load_model('./model/rapid_ascending %s_%s - 4.39.hdf5' % (input_data_length, model_num))


find_new_low = 1
check_new_low = 0
limit_line = limit_line_high
Coin = 'enj'.upper()

#           매도 대기           #
while True:
    # 분당 한번 high_check predict 했을때, 1 결과값이 출력되면 매도 진행
    try:
        #   이전 종가가 dataframe 으로 완성될 시기
        if datetime.now().second == 55:
            while True:
                try:
                    if find_new_low == 1:
                        X_test_high, _, _, end_datetime_list = low_high(Coin, input_data_length,
                                                                        crop_size=crop_size_high,
                                                                        sudden_death=0.)
                        X_test_low, _, _, _ = low_high(Coin, input_data_length,
                                                                             crop_size=crop_size_low,
                                                                             sudden_death=0.)
                        check_new_low = 1
                        #     X_test, _, _, end_datetime_list = low_high(Coin, input_data_length,
                        #     crop_size=crop_size_high, sudden_death=0.)
                        break
                    else:
                        X_test_high, _, _, _ = low_high(Coin, input_data_length,
                                                                        crop_size=crop_size_sudden_death,
                                                                        sudden_death=0.)
                    break

                except Exception as e:
                    print('Error in getting low_high data :', e)
                    time.sleep(random.random() * 5)

            #       check_span 지나면 저점 갱신 확인        #
            if check_new_low == 1:
                Y_pred_low_ = model.predict(X_test_low, verbose=1)
                max_value_low = np.max(Y_pred_low_, axis=0)

                if Y_pred_low_[-1][1] > max_value_low[1] * limit_line_low:
                    limit_line = limit_line_sudden_death
                    X_test_high, _, _, _ = low_high(Coin, input_data_length,
                                                                    crop_size=crop_size_sudden_death, sudden_death=0.)
                    find_new_low = 0
                    check_new_low = 0

            Y_pred_high_ = model.predict(X_test_high, verbose=1)
            max_value_high = np.max(Y_pred_high_, axis=0)

            #   매도 진행
            if Y_pred_high_[-1][2] > max_value_high[2] * limit_line:
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