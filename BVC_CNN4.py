import numpy as np
import pandas as pd
import Funcs_CNN4
import os
from multiprocessing import Pool
from functools import partial
import warnings
warnings.filterwarnings("ignore")


input_data_length = 54
model_num = 23
dir = './pred_ohlcv/%s_%s' % (input_data_length, model_num)
ohlcv_list = os.listdir(dir)

Datelist = []
Date = ''
for file in ohlcv_list:
    New_Date = str(file.split()[0])
    if Date != New_Date:
        Datelist.append(New_Date)
        Date = New_Date

Coinlist = [None] * len(Datelist)
filename = None
temp = []

for i in range(0, len(Datelist)) :
    for file in ohlcv_list :
        if file.find(Datelist[i]) is not -1 : # 해당 파일이면 temp[i] 에 넣겠다.
            filename = os.path.splitext(file)
            temp.append(filename[0].split(" ")[1])
    Coinlist[i] = temp
    temp = [] # temp 초기화

#               날짜별 TopCoinList 받아오기 완료             #


def multi(wait_tick, over_tick, i):
    TotalProfits = 1.0
    Plus_Profits = 1.0
    Minus_Profits = 1.0

    print()
    print('wait_tick :', wait_tick, "over_tick :", over_tick,"End Date :", Datelist[-1])

    for Coin in Coinlist[i]:
        try:
            EndProfits = Funcs_CNN4.profitage(Coin, input_data_length, model_num, wait_tick, over_tick, Datelist[i], 1)
            TotalProfits *= EndProfits[0]
            Plus_Profits *= EndProfits[1]
            Minus_Profits *= EndProfits[2]

        except Exception as e:
            print('Error in Function :', e)
            continue
    # TotalProfits = Profit_Check.profit_check(Datelist[i], 0)
    df2 = pd.DataFrame(data=[[Datelist[i], TotalProfits, Minus_Profits]], columns=['Date', 'TotalProfits', 'Minus_Profits'])
    print(df2)
    return df2


Result_df = pd.DataFrame(columns=['DatesProfits', 'DatesProfits_Minus', 'wait_tick', 'over_tick'])

if __name__ == '__main__':

    wait_tick = 3
    over_tick = 10

    pool = Pool(processes=8)
    multi2 = partial(multi, wait_tick, over_tick)
    result = pool.map(multi2, [i for i in range(len(Datelist))])
    pool.close()
    pool.join()

    df3 = result[0]
    for i in range(1, len(Datelist)):
        df3 = pd.concat([df3, result[i]])

    DatesProfits = df3.TotalProfits.cumsum().iloc[-1]
    DatesProfits_Minus = df3.Minus_Profits.cumsum().iloc[-1]

    DatesProfits = DatesProfits / len(Datelist)
    DatesProfits_Minus = DatesProfits_Minus / len(Datelist)

    df3.to_excel("./BVC/%s %.3f %.3f by %s %s %s.xlsx" % (Datelist[-1], DatesProfits, DatesProfits_Minus, model_num, wait_tick, over_tick))
    Result_df = Result_df.append(pd.DataFrame(data=[[DatesProfits, DatesProfits_Minus, wait_tick, over_tick]],
                                              columns=['DatesProfits', 'DatesProfits_Minus', 'wait_tick', 'over_tick']))
    print("Profit per day : ", DatesProfits, DatesProfits_Minus)



