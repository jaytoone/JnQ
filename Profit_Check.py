import pandas as pd
import numpy as np
import os


def profit_check(Date, model_num) :
    temp = []
    # input_data_length = input('input data length : ')
    input_data_length = 54
    dir = './pred_ohlcv/{}_{}'.format(input_data_length, model_num)
    ohlcv_list = os.listdir(dir)

    for file in ohlcv_list:
        if file.find(Date) is not -1:  # 해당 파일이면 temp[i] 에 넣겠다.
            filename = os.path.splitext(file)
            temp.append(filename[0].split(" ")[1])

    TotalProfits = 1.0
    profit_list = []
    gap_list = []
    for Coin in temp :
        try:
            df = pd.read_excel("./BackTest/" + "%s BackTest %s.xlsx" % (Date, Coin))
            Profits = df.Profits.cumprod().iloc[-1]
            Price_point = df.Price_point.max()

            # if Profits > 1:
            print(Coin, Profits, Price_point)

            profit_list.append(Profits)
            gap_list.append(Price_point)
            TotalProfits *= Profits
        except Exception as e:
            print(e)

    return TotalProfits, profit_list, gap_list


input_data_length = 54
model_num = 13
dir = './pred_ohlcv/{}_{}'.format(input_data_length, model_num)
ohlcv_list = os.listdir(dir)

Datelist = []
Date = ''
for file in ohlcv_list:
    New_Date = str(file.split()[0])
    if Date != New_Date:
        Datelist.append(New_Date)
        Date = New_Date

result_profit = []
result_gap = []
for Date in Datelist:
    print(Date)
    result = profit_check(Date, model_num)
    result_profit += result[1]
    result_gap += result[2]
    # print(result_profit)
    print()

# result_df = pd.DataFrame({'profit': result_profit, 'gap': result_gap})
# result_df.to_excel('result_df.xlsx')
# print(result_df.info())
