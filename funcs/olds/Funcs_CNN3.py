import pybithumb
import numpy as np
import pandas as pd
from datetime import datetime
import os
from scipy import stats
from asq.initiators import query
import matplotlib.pyplot as plt
from sklearn.preprocessing import MaxAbsScaler


def transh_hour(realtime, numb):
    Hour = realtime[numb].split(':')[0]
    return int(Hour)


def transh_min(realtime, numb):
    Minute = realtime[numb].split(':')[1]
    return int(Minute)


# print(transh_min(-1))
def transh_fluc(Coin):
    try:
        TransH = pybithumb.transaction_history(Coin)
        TransH = TransH['data']
        Realtime = query(TransH).select(lambda item: item['transaction_date'].split(' ')[1]).to_list()

        # 거래 활발한지 검사
        if (transh_hour(Realtime, -1) - transh_hour(Realtime, 0)) < 0:
            if 60 + (transh_min(Realtime, -1) - transh_min(Realtime, 0)) > 30:
                return 0, 0
        elif 60 * (transh_hour(Realtime, -1) - transh_hour(Realtime, 0)) + (transh_min(Realtime, -1) - transh_min(Realtime, 0)) > 30:
            return 0, 0

        # 1분 동안의 거래 이력을 조사하는 루프, 0 - 59 와 같은 음수처리를 해주어야한다.
        i = 1
        while True:
            i += 1
            if i > len(Realtime):
                m = i
                break
            # 음수 처리
            if (transh_min(Realtime, -1) - transh_min(Realtime, -i)) < 0:
                if (60 + transh_min(Realtime, -1) - transh_min(Realtime, -i)) > 1:
                    m = i - 1
                    break
            elif (transh_min(Realtime, -1) - transh_min(Realtime, -i)) > 1:
                m = i - 1
                break

        # Realtime = query(TransH[-i:]).select(lambda item: item['transaction_date'].split(' ')[1]).to_list()
        Price = list(map(float, query(TransH[-m:]).select(lambda item: item['price']).to_list()))

        # print(Realtime)
        # print(Price)
        fluc = max(Price) / min(Price)
        if TransH[-1]['type'] == 'ask':
            fluc = -fluc
        return fluc, min(Price)

    except Exception as e:
        print("Error in transh_fluc :", e)
        return 0, 0


def realtime_transaction(Coin, display=5):
    Transaction_history = pybithumb.transaction_history(Coin)
    Realtime = query(Transaction_history['data'][-display:]).select(lambda item: item['transaction_date'].split(' ')[1]).to_list()
    Realtime_Price = list(map(float,query(Transaction_history['data'][-display:]).select(lambda item: item['price']).to_list()))
    Realtime_Volume = list(map(float,query(Transaction_history['data'][-display:]).select(lambda item: item['units_traded']).to_list()))

    print("##### 실시간 체결 #####")
    print("{:^10} {:^10} {:^20}".format('시간', '가격', '거래량'))
    for i in reversed(range(display)):
        print("%-10s %10.2f %20.3f" % (Realtime[i], Realtime_Price[i], Realtime_Volume[i]))
    return


def realtime_hogachart(Coin, display=3):
    Hogachart = pybithumb.get_orderbook(Coin)

    print("##### 실시간 호가창 #####")
    print("{:^10} {:^20}".format('가격', '거래량'))
    for i in reversed(range(display)):
        print("%10.2f %20.3f" % (Hogachart['asks'][i]['price'], Hogachart['asks'][i]['quantity']))
    print('-' * 30)
    for j in range(display):
        print("%10.2f %20.3f" % (Hogachart['bids'][j]['price'], Hogachart['bids'][j]['quantity']))


def realtime_volume(Coin):
    Transaction_history = pybithumb.transaction_history(Coin)
    Realtime_Volume = query(Transaction_history['data']).where(lambda item: item['type'] == 'bid').select(lambda item: item['units_traded']).to_list()
    Realtime_Volume = sum(list(map(float, Realtime_Volume)))
    return Realtime_Volume


def realtime_volume_ratio(Coin):
    Transaction_history = pybithumb.transaction_history(Coin)
    Realtime_bid = query(Transaction_history['data']).where(lambda item: item['type'] == 'bid').select(lambda item: item['units_traded']).to_list()
    Realtime_ask = query(Transaction_history['data']).where(lambda item: item['type'] == 'ask').select(lambda item: item['units_traded']).to_list()
    Realtime_bid = sum(list(map(float, Realtime_bid)))
    Realtime_ask = sum(list(map(float, Realtime_ask)))
    Realtime_Volume_Ratio = Realtime_bid / Realtime_ask
    return Realtime_Volume_Ratio


def topcoinlist(Date):
    temp = []
    dir = 'C:/Users/장재원/OneDrive/Hacking/CoinBot/ohlcv/'
    ohlcv_list = os.listdir(dir)

    for file in ohlcv_list:
        if file.find(Date) is not -1:  # 해당 파일이면 temp[i] 에 넣겠다.
            filename = os.path.splitext(file)
            temp.append(filename[0].split(" ")[1])
    return temp


def get_ma_min(Coin):

    df = pybithumb.get_ohlcv(Coin, "KRW", 'minute1')

    df['MA20'] = df['close'].rolling(20).mean()

    DatetimeIndex = df.axes[0]
    period = 20
    if inthour(DatetimeIndex[-1]) - inthour(DatetimeIndex[-period]) < 0:
        if 60 + (intmin(DatetimeIndex[-1]) - intmin(DatetimeIndex[-period])) > 30:
            return 0
    elif 60 * (inthour(DatetimeIndex[-1]) - inthour(DatetimeIndex[-period])) + intmin(DatetimeIndex[-1]) - intmin(
            DatetimeIndex[-period]) > 30:
        return 0
    slope, intercept, r_value, p_value, stderr = stats.linregress([i for i in range(period)], df.MA20[-period:])

    return slope


def get_ma20_min(Coin):

    df = pybithumb.get_ohlcv(Coin, "KRW", 'minute1')

    maxAbsScaler = MaxAbsScaler()

    df['MA20'] = df['close'].rolling(20).mean()
    MA_array = np.array(df['MA20']).reshape(len(df.MA20), 1)
    maxAbsScaler.fit(MA_array)
    scaled_MA = maxAbsScaler.transform(MA_array)

    period = 5
    slope, intercept, r_value, p_value, stderr = stats.linregress([i for i in range(period)], scaled_MA[-period:])

    return slope


def get_obv_min(Coin):

    df = pybithumb.get_ohlcv(Coin, "KRW", "minute1")

    obv = [0] * len(df.index)
    for m in range(1, len(df.index)):
        if df['close'].iloc[m] > df['close'].iloc[m - 1]:
            obv[m] = obv[m - 1] + df['volume'].iloc[m]
        elif df['close'].iloc[m] == df['close'].iloc[m - 1]:
            obv[m] = obv[m - 1]
        else:
            obv[m] = obv[m - 1] - df['volume'].iloc[m]
    df['OBV'] = obv

    # 24시간의 obv를 잘라서 box 높이를 만들어주어야한다.
    DatetimeIndex = df.axes[0]
    boxheight = [0] * len(df.index)
    whaleincome = [0] * len(df.index)
    for m in range(len(df.index)):
        # 24시간 시작행 찾기, obv 데이터가 없으면 stop
        n = m
        while True:
            n -= 1
            if n < 0:
                n = 0
                break
            if inthour(DatetimeIndex[m]) - inthour(DatetimeIndex[n]) < 0:
                if 60 - (intmin(DatetimeIndex[m]) - intmin(DatetimeIndex[n])) >= 60 * 24:
                    break
            elif 60 * (inthour(DatetimeIndex[m]) - inthour(DatetimeIndex[n])) + intmin(DatetimeIndex[m]) - intmin(
                    DatetimeIndex[n]) >= 60 * 24:
                break
        obv_trim = obv[n:m]
        if len(obv_trim) != 0:
            boxheight[m] = max(obv_trim) - min(obv_trim)
            if obv[m] - min(obv_trim) != 0:
                whaleincome[m] = abs(max(obv_trim) - obv[m]) / abs(obv[m] - min(obv_trim))

    df['BoxHeight'] = boxheight
    df['Whaleincome'] = whaleincome

    period = 0
    while True:
        period += 1
        if period >= len(DatetimeIndex):
            break
        if inthour(DatetimeIndex[-1]) - inthour(DatetimeIndex[-period]) < 0:
            if 60 + (intmin(DatetimeIndex[-1]) - intmin(DatetimeIndex[-period])) >= 10:
                break
        elif 60 * (inthour(DatetimeIndex[-1]) - inthour(DatetimeIndex[-period])) + intmin(DatetimeIndex[-1]) - intmin(
                DatetimeIndex[-period]) >= 10:
            break

    slope, intercept, r_value, p_value, stderr = stats.linregress([i for i in range(period)], df.OBV[-period:])
    if period < 3:
        df['Whaleincome'].iloc[-1], slope = 0, 0
    else:
        slope = slope / df['BoxHeight'].iloc[-1]

    return df['Whaleincome'].iloc[-1], slope


# def get_ohlcv_min_proxy(Coin):
#
#     df = pybithumb.get_ohlcv_proxy(Coin, "KRW", "minute1")
#
#     obv = [0] * len(df.index)
#     for m in range(1, len(df.index)):
#         if df['close'].iloc[m] > df['close'].iloc[m - 1]:
#             obv[m] = obv[m - 1] + df['volume'].iloc[m]
#         elif df['close'].iloc[m] == df['close'].iloc[m - 1]:
#             obv[m] = obv[m - 1]
#         else:
#             obv[m] = obv[m - 1] - df['volume'].iloc[m]
#     df['OBV'] = obv
#
#     # 24시간의 obv를 잘라서 box 높이를 만들어주어야한다.
#     DatetimeIndex = df.axes[0]
#     boxheight = [0] * len(df.index)
#     whaleincome = [0] * len(df.index)
#     for m in range(len(df.index)):
#         # 24시간 시작행 찾기, obv 데이터가 없으면 stop
#         n = m
#         while True:
#             n -= 1
#             if n < 0:
#                 n = 0
#                 break
#             if 60 * (inthour(DatetimeIndex[m]) - inthour(DatetimeIndex[n])) + intmin(DatetimeIndex[m]) - intmin(
#                     DatetimeIndex[n]) >= 60 * 24:
#                 break
#         obv_trim = obv[n:m]
#         if len(obv_trim) != 0:
#             boxheight[m] = max(obv_trim) - min(obv_trim)
#             if obv[m] - min(obv_trim) != 0:
#                 whaleincome[m] = (max(obv_trim) - obv[m]) / (obv[m] - min(obv_trim))
#
#     df['BoxHeight'] = boxheight
#     df['Whaleincome'] = whaleincome
#     df['OBVGap'] = np.where(df['Whaleincome'] > 3, (df['OBV'] - df['OBV'].shift(1)) / df['BoxHeight'], np.nan)
#
#     return df['OBVGap']


def GetHogaunit(Hoga):

    if Hoga < 1:
        Hogaunit = 0.0001
    elif 1 <= Hoga < 10:
        Hogaunit = 0.001
    elif 10 <= Hoga < 100:
        Hogaunit = 0.01
    elif 100 <= Hoga < 1000:
        Hogaunit = 0.1
    elif 1000 <= Hoga < 5000:
        Hogaunit = 1
    elif 5000 <= Hoga < 10000:
        Hogaunit = 5
    elif 10000 <= Hoga < 50000:
        Hogaunit = 10
    elif 50000 <= Hoga < 100000:
        Hogaunit = 50
    elif 100000 <= Hoga < 500000:
        Hogaunit = 100
    elif 500000 <= Hoga < 1000000:
        Hogaunit = 500
    else:
        Hogaunit = 1000
    return Hogaunit


def clearance(price):

    try:
        Hogaunit = GetHogaunit(price)
        Htype = type(Hogaunit)
        if Hogaunit == 0.1:
            price2 = int(price * 10) / 10.0
        elif Hogaunit == 0.01:
            price2 = int(price * 100) / 100.0
        elif Hogaunit == 0.001:
            price2 = int(price * 1000) / 1000.0
        elif Hogaunit == 0.0001:
            price2 = int(price * 10000.0) / 10000.0
        else:
            return int(price) // Hogaunit * Hogaunit
        return Htype(price2)

    except Exception as e:
        return np.nan


def inthour(date):
    date = str(date)
    date = date.split(' ')
    hour = int(date[1].split(':')[0])  # 시
    return hour


def intmin(date):
    date = str(date)
    date = date.split(' ')
    min = int(date[1].split(':')[1])  # 분
    return min


def realtime_cmo(Coin, closeprice, period=9):

    try:
        df = pybithumb.get_ohlcv(Coin, "KRW", 'minute1')
        del df['open']
        del df['high']
        del df['low']
        del df['volume']
        df.loc[len(df)] = [closeprice]

        df['closegap_cunsum'] = (df['close'] - df['close'].shift(1)).cumsum()
        df['closegap_abs_cumsum'] = abs(df['close'] - df['close'].shift(1)).cumsum()
        # print(df)

        df['CMO'] = (df['closegap_cunsum'] - df['closegap_cunsum'].shift(period)) / (
                df['closegap_abs_cumsum'] - df['closegap_abs_cumsum'].shift(period)) * 100

        return df['CMO'].iloc[-1]

    except Exception as e:
        print("Error in realtime_cmo :", e)
        return 0


def profitage(Coin, input_data_length, model_num, wait_tick=3, over_tick=10, Date='2019-09-25', excel=0):

    try :
        df = pd.read_excel('./pred_ohlcv/%s_%s/%s %s ohlcv.xlsx' % (input_data_length, model_num, Date, Coin), index_col=0)

    except Exception as e:
        print('Error in loading ohlcv_data :', e)
        return 1.0, 1.0, 1.0

    # period = 9
    # df['closegap_cunsum'] = (df['close'] - df['close'].shift(1)).cumsum()
    # df['closegap_abs_cumsum'] = abs(df['close'] - df['close'].shift(1)).cumsum()
    # # print(df)
    #
    # df['CMO'] = (df['closegap_cunsum'] - df['closegap_cunsum'].shift(period)) / (
    #         df['closegap_abs_cumsum'] - df['closegap_abs_cumsum'].shift(period)) * 100

    # 매수 시점 = 급등 예상시, 매수가 = 이전 종가
    df['BuyPrice'] = np.where((df['trade_state'] > 1.5) & (df['trade_state'] < 2.5), df['close'].shift(1), np.nan)
    # df['BuyPrice'] = df['BuyPrice'].apply(clearance)

    # 거래 수수료
    fee = 0.005

    # DatetimeIndex 를 지정해주기 번거로운 상황이기 때문에 틱을 기준으로 거래한다.
    # DatetimeIndex = df.axes[0]

    # ------------------- 상향 / 하향 매도 여부와 이익률 계산 -------------------#

    # high 가 SPP 를 건드리거나, low 가 SPM 을 건드리면 매도 체결 [ 매도 체결될 때까지 SPP 와 SPM 은 유지 !! ]
    length = len(df.index) - 1  # 데이터 갯수 = 1, m = 0  >> 데이터 갯수가 100 개면 m 번호는 99 까지 ( 1 - 100 )

    if length < 100:
        return 1.0, 1.0, 1.0

    # 병합할 dataframe 초기화
    bprelay = pd.DataFrame(index=np.arange(len(df)), columns=['bprelay'])
    condition = pd.DataFrame(index=np.arange(len(df)), columns=["Condition"])
    Profits = pd.DataFrame(index=np.arange(len(df)), columns=["Profits"])
    price_point = pd.DataFrame(index=np.arange(len(df)), columns=['Price_point'])

    Profits.Profits = 1.0
    Minus_Profits = 1.0

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

        #       매수 등록 완료, 매수 체결 대기     #

        start_m = m
        while True:
            bprelay["bprelay"].iloc[m] = bp

            # 매수 체결 조건
            if df.iloc[m]['low'] <= bp and (df.iloc[m]['high'] != df.iloc[m]['low']):  # 조건을 만족할 경우 spp 생성
                # print(df['low'].iloc[m], '<=', bp, '?')

                condition.iloc[m] = '매수 체결'
                bprelay["bprelay"].iloc[m] = bprelay["bprelay"].iloc[m - 1]

                m += 1
                break
            else:
                m += 1
                if m > length:
                    break
                if m - start_m >= wait_tick:
                    break

                condition.iloc[m] = '매수 대기'
                bprelay["bprelay"].iloc[m] = bprelay["bprelay"].iloc[m - 1]

    # df = pd.merge(df, condition, how='outer', left_index=True, right_index=True)
    # df = pd.merge(df, bprelay, how='outer', left_index=True, right_index=True)

    # if excel == 1:
    #     df.to_excel("./BackTest/%s BackTest %s.xlsx" % (Date, Coin))
    #     quit()

    #           지정 매도가 표시 완료           #

    #                               수익성 검사                                  #

    m = 0
    spanlist = []
    spanlist_limit = []
    spanlist_breakaway = []
    while m <= length:  # 초반 시작포인트 찾기

        while True: # SPP 와 SPM 찾긴
            if condition.iloc[m]['Condition'] == '매수 체결':
                # and type(df.iloc[m]['SPP']) != str:  # null 이 아니라는 건 오더라인과 매수가로 캡쳐했다는 거
                break
            m += 1
            if m > length: # 차트가 끊나는 경우, 만족하는 spp, spm 이 없는 경우
                break

        if (m > length) or pd.isnull(condition.iloc[m]['Condition']):
            break

        start_m = m
        start_span = df.index[m]

        #           매수 체결 지점 확인          #

        #       Predict value == 1 지점으로 부터 일정 시간 (10분) 지난 후 checking      #
        while m - start_m <= over_tick:
            m += 1
            if m > length:
                break
            condition.iloc[m] = '매도 대기'
            bprelay["bprelay"].iloc[m] = bprelay["bprelay"].iloc[m - 1]

        if m > length:
            break

        while True:

            #       고점 예측시 매도       #
            if df.iloc[m]['trade_state'] > 2.5:
                break

            m += 1
            if m > length:
                break

            condition.iloc[m] = '매도 대기'
            bprelay["bprelay"].iloc[m] = bprelay["bprelay"].iloc[m - 1]

        if m > length:
            # print(condition.iloc[m - 1])
            if condition.iloc[m - 1].item() == '매도 대기':
                print('매수 체결 / 매도 신호 미포착 : ', Date, Coin)
                try:
                    spanlist.append((start_m, m - 1))
                    spanlist_breakaway.append((start_span, df.index[m - 1]))

                except Exception as e:
                    pass
            break

        elif df.iloc[m]['trade_state'] > 2.5:
            condition.iloc[m] = "매도 체결"
            Profits.iloc[m] = df.iloc[m]['close'] / bprelay["bprelay"].iloc[m - 1] - fee

            if float(Profits.iloc[m]) < 1:
                Minus_Profits *= float(Profits.iloc[m])
                try:
                    spanlist.append((start_m, m))
                    spanlist_breakaway.append((start_span, df.index[m]))

                except Exception as e:
                    pass

            else:
                try:
                    spanlist.append((start_m, m))
                    spanlist_limit.append((start_span, df.index[m]))

                except Exception as e:
                    pass

                if float(Profits.iloc[m]) > 1.05:
                    price_point.iloc[m] = df['high'][:start_m].max() / df['low'][:start_m].min()

        # 체결시 재시작
        m += 1

    df = pd.merge(df, bprelay, how='outer', left_index=True, right_index=True)
    df = pd.merge(df, condition, how='outer', left_index=True, right_index=True)
    df = pd.merge(df, Profits, how='outer', left_index=True, right_index=True)
    df = pd.merge(df, price_point, how='outer', left_index=True, right_index=True)

    if excel == 1:
        df.to_excel("./BackTest/%s BackTest %s.xlsx" % (Date, Coin))

    profits = Profits.cumprod()  # 해당 열까지의 누적 곱!

    if np.isnan(profits.iloc[-1].item()):
        return 1.0, 1.0, 1.0

    # [-1] 을 사용하려면 데이터가 존재해야 되는데 데이터 전체가 결측치인 경우가 존재한다.
    if len(profits) == 0:
        return 1.0, 1.0, 1.0

    return float(profits.iloc[-1]), float(profits.iloc[-1]) / Minus_Profits, Minus_Profits


if __name__=="__main__":
    # Best Value #########################################################
    # ------- FEED MACHINE -------#
    excel_file = input("Input File : ")

    # ----------- PARAMETER -----------#
    Coin = excel_file.split()[1].split('.')[0]
    Date = excel_file.split()[0]
    input_data_length = 54
    # model_num = input('Press model num : ')
    model_num = 20
    wait_tick = 3
    over_tick = 10
    ######################################################################

    print(profitage(Coin, input_data_length, model_num, wait_tick, over_tick, Date, 1))

