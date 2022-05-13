import pybithumb
import numpy as np
import pandas as pd
from datetime import datetime
import os
from scipy import stats
from asq.initiators import query
import matplotlib.pyplot as plt
from sklearn.preprocessing import MaxAbsScaler


def profitage(Coin, input_data_length, model_num, wait_tick=3, check_span=10, Date='2019-09-25', excel=0):

    try:
        df = pd.read_excel(dir + '%s %s ohlcv.xlsx' % (Date, Coin), index_col=0)

    except Exception as e:
        print('Error in loading ohlcv_data :', e)
        return 1.0, 1.0, 1.0

    ema_ribbon(df, 15, 30, 60)

    ema_state = [0] * len(df)
    trade_state = [0] * len(df)
    #   역순, 정순 라벨링
    for i in range(len(df)):
        if not ((df['EMA_1'][i] > df['EMA_2'][i]) and (df['EMA_1'][i] > df['EMA_3'][i])):
            ema_state[i] = 1

        elif not ((df['EMA_1'][i] < df['EMA_2'][i]) and (df['EMA_1'][i] < df['EMA_3'][i])):
            ema_state[i] = 2

    for i in range(check_span, len(df)):
        #   이전에 역순이 존재하고 바로 전이 정순이면,
        if 1 in ema_state[i - check_span:i]:
            if (df['EMA_1'][i] > df['EMA_2'][i]) and (df['EMA_1'][i] > df['EMA_3'][i]):
                trade_state[i] = 1

        if 2 in ema_state[i - check_span:i]:
            if (df['EMA_1'][i] < df['EMA_2'][i]) and (df['EMA_1'][i] < df['EMA_3'][i]):
                trade_state[i] = 2

    df['trade_state'] = trade_state

    # 매수 시점 = 급등 예상시, 매수가 = 이전 종가
    df['BuyPrice'] = np.where((df['trade_state'] > 0.5) & (df['trade_state'] < 1.5), df['open'], np.nan)
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
    # price_point = pd.DataFrame(index=np.arange(len(df)), columns=['Price_point'])

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
            if df.iloc[m]['low'] <= bp: #and (df.iloc[m]['high'] != df.iloc[m]['low']):  # 조건을 만족할 경우 spp 생성
                # print(df['low'].iloc[m], '<=', bp, '?')

                condition.iloc[m] = '매수 체결'
                # bprelay["bprelay"].iloc[m] = bprelay["bprelay"].iloc[m - 1]

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

        #           매수 체결 지점 확인          #

        #       Predict value == 1 지점으로 부터 일정 시간 (10분) 지난 후 checking      #
        # while m - start_m <= over_tick:
        #     m += 1
        #     if m > length:
        #         break
        #     condition.iloc[m] = '매도 대기'
        #     bprelay["bprelay"].iloc[m] = bprelay["bprelay"].iloc[m - 1]
        # if m > length:
        #     break

        while True:

            #       고점 예측시 매도       #
            if df.iloc[m]['trade_state'] > 1.5:
                break

            m += 1
            if m > length:
                break

            condition.iloc[m] = '매도 대기'
            bprelay["bprelay"].iloc[m] = bprelay["bprelay"].iloc[m - 1]

        if m > length:
            # print(condition.iloc[m - 1])
            break

        elif df.iloc[m]['trade_state'] > 1.5:
            condition.iloc[m] = "매도 체결"
            Profits.iloc[m] = df.iloc[m]['close'] / bprelay["bprelay"].iloc[m - 1] - fee

            if float(Profits.iloc[m]) < 1:
                Minus_Profits *= float(Profits.iloc[m])
                try:
                    spanlist.append((start_m, m))
                    spanlist_breakaway.append((start_m, m))

                except Exception as e:
                    pass

            else:
                try:
                    spanlist.append((start_m, m))
                    spanlist_limit.append((start_m, m))

                except Exception as e:
                    pass

        # 체결시 재시작
        m += 1

    df = pd.merge(df, bprelay, how='outer', left_index=True, right_index=True)
    df = pd.merge(df, condition, how='outer', left_index=True, right_index=True)
    df = pd.merge(df, Profits, how='outer', left_index=True, right_index=True)
    # df = pd.merge(df, price_point, how='outer', left_index=True, right_index=True)
    df = df.reset_index(drop=True)

    if excel == 1:
        df.to_excel("./BackTest/%s BackTest %s.xlsx" % (Date, Coin))

    profits = Profits.cumprod()  # 해당 열까지의 누적 곱!

    if np.isnan(profits.iloc[-1].item()):
        return 1.0, 1.0, 1.0

    # [-1] 을 사용하려면 데이터가 존재해야 되는데 데이터 전체가 결측치인 경우가 존재한다.
    if len(profits) == 0:
        return 1.0, 1.0, 1.0

    elif float(profits.iloc[-1]) != 1.0:

        # 거래 체결마다 subplot 1,2 저장
        plt.figure(figsize=(5, 5))
        plt.subplot(111)
        plt.plot(df[['close']], 'gold', label='close', linewidth=1.0)
        plt.plot(df[['EMA_1']], 'y', label='ema1', linewidth=1.0)
        plt.plot(df[['EMA_2']], 'g', label='ema2', linewidth=1.0)
        plt.plot(df[['EMA_3']], 'b', label='ema3', linewidth=1.0)
        plt.xticks(fontsize=5)
        plt.yticks(fontsize=5)
        plt.legend(loc='upper right', fontsize=5)
        plt.title('%.2f %.2f %.2f' % (float(profits.iloc[-1]), float(profits.iloc[-1]) / Minus_Profits, Minus_Profits))

        for trade_num in range(len(spanlist_limit)):
            plt.axvspan(spanlist_limit[trade_num][0], spanlist_limit[trade_num][1], facecolor='c', alpha=0.5)

        for trade_num in range(len(spanlist_breakaway)):
            plt.axvspan(spanlist_breakaway[trade_num][0], spanlist_breakaway[trade_num][1], facecolor='m', alpha=0.5)

        # plt.subplot(211)
        # plt.plot(df[['close']], 'gold', label='close', linewidth=5.0)
        # plt.plot(df[['EMA_1']], 'y', label='ema1', linewidth=5.0)
        # plt.plot(df[['EMA_2']], 'g', label='ema2', linewidth=5.0)
        # plt.plot(df[['EMA_3']], 'b', label='ema3', linewidth=5.0)
        # plt.xticks(fontsize=10)
        # plt.yticks(fontsize=10)
        # plt.legend(loc='upper right', fontsize=20)
        # spanlist_low = []
        # spanlist_high = []
        #
        # for m in range(len(trade_state)):
        #     if (trade_state[m] > 0.5) and (trade_state[m] < 1.5):
        #         if m + 1 < len(trade_state):
        #             spanlist_low.append((df.index[m], df.index[m + 1]))
        #         else:
        #             spanlist_low.append((df.index[m - 1], df.index[m]))
        #
        # for m in range(len(trade_state)):
        #     if (trade_state[m] > 1.5) and (trade_state[m] < 2.5):
        #         if m + 1 < len(trade_state):
        #             spanlist_high.append((df.index[m], df.index[m + 1]))
        #         else:
        #             spanlist_high.append((df.index[m - 1], df.index[m]))
        #
        # for i in range(len(spanlist_low)):
        #     plt.axvspan(spanlist_low[i][0], spanlist_low[i][1], facecolor='c', alpha=0.7)
        # for i in range(len(spanlist_high)):
        #     plt.axvspan(spanlist_high[i][0], spanlist_high[i][1], facecolor='m', alpha=0.7)

        # plot 저장 & 닫기
        # plt.show()
        plt.savefig("./Figure_pred/%s_%s/%s %s.png" % (input_data_length, model_num, Date, Coin), dpi=300)
        plt.close()

    return float(profits.iloc[-1]), float(profits.iloc[-1]) / Minus_Profits, Minus_Profits


if __name__=="__main__":

    home_dir = os.path.expanduser('~')
    dir = home_dir + '/OneDrive/CoinBot/ohlcv/'
    ohlcv_list = os.listdir(dir)

    # ohlcv_list = ['2019-10-05 LAMB ohlcv.xlsx']

    for file in ohlcv_list:
        Coin = file.split()[1].split('.')[0]
        Date = file.split()[0]
        input_data_length = 30
        model_num = 65
        # Coin = 'LUNA'
        # Date = '2020-02-27'
        wait_tick = 3
        # over_tick = 10

        #       Make folder      #
        try:
            os.mkdir('./Figure_pred/%s_%s/' % (input_data_length, model_num))

        except Exception as e:
            pass

        print(profitage(Coin, input_data_length, model_num, wait_tick, Date=Date, excel=1))

