import pandas as pd
from datetime import datetime
import numpy as np
from binance_futures_concat_candlestick import concat_candlestick
from fishing_prev_close.utils import interval_to_min, calc_train_days, tp_update
import os
import pickle


print(datetime.now())
# key_abspath = os.path.abspath("private_info/binance_key.p")
# key_abspath = os.path.abspath("private_info/api_for_bot.txt")
# key_abspath = os.path.abspath("private_info/api_for_bot.pickle")
#
# #           txt file to pickle      #
#
# # with open(key_abspath, 'rb') as f:
# #     # api_list = f.readlines()
# #     api_list = f.read().splitlines()
# with open(key_abspath, 'wb') as f:
#     pickle.dump(api_list, f)
#
# with open(key_abspath, 'rb') as f:
#     api_list = pickle.load(f)
#
# print(api_list[0])
# symbol = 'DOTUSDT'
# interval = "30m"
#
# # with open('updown_rnn/survey_logger/%s.txt' % symbol, 'w') as log_file:
# #     log_file.write(str(datetime.now().timestamp()))
#
# # with open('updown_rnn/survey_logger/%s.txt' % symbol, 'r') as log_file:
# #     # print(type(log_file.readline()))
# #     print(float(log_file.readline()))
#
# h_ohlcv_path = './fishing_prev_close/%s_history.xlsx' % symbol
#
#
# #       update tp       #
# # try:
# #       Todo : stack historical ohlcv       #
# #       0. history 불러오기
# h_ohlcv = pd.read_excel(h_ohlcv_path, index_col=0)
#
# #       0-1. h_ohlcv 의 last_row 는 imcomplete 함
# complete_h_ohlcv = h_ohlcv.iloc[:-1, :]
#
# #       1. 효율성을 따졌을 땐, back_df 를 사용하는게 맞고
# back_df, _ = concat_candlestick(symbol, interval, days=1)
#
# #       Todo : last_index 와 현재 시간의 gap 추정
# index_gap = datetime.timestamp(back_df.index[-1]) - datetime.timestamp(h_ohlcv.index[-1])
# print("index_gap :", index_gap)
# h_use_rows = int(index_gap / (60 * interval_to_min(interval)))
# days = calc_train_days(interval=interval, use_rows=h_use_rows + 1)
# print("days :", days)
# # quit()
#
# #           1-1. back_df 를 사용하지 못하는 경우는 ? 채워주어야겠지
# #           => h_ohlcv 의 마지막 time_index 부터 현재 시간까지의 days 를 계산해,
# #           ohlcv 를 새로 불러온다
# if days > 1:
#     back_df, _ = concat_candlestick(symbol, interval, days=days, timesleep=0.2)
#
# #       2. 중복절은 제거
# h_ohlcv = complete_h_ohlcv.append(back_df)
# h_ohlcv = h_ohlcv[~h_ohlcv.index.duplicated(keep='first')]
#
# tp = tp_update(h_ohlcv, wr_threshold=0.64, save_path=h_ohlcv_path.replace("xlsx", "png"))
# print("tp updated to %.3f" % tp)

#       4. save new history
# h_ohlcv.to_excel(h_ohlcv_path)

# except Exception as e:
#     print("Error in tp_update :", e)



# import time
# st = datetime.now().timestamp()
# print(st)
# time.sleep(5)
# print(datetime.now().timestamp() - st)



# if 1 and not pd.isna(1):
#     print('break')
# price = 1.0074
#
# print('something new', end=' ')
# print('something')
# tlist = [1, 2, 3, 4]
# a, b = tlist[:2]
# print(a, b)
# tlist = [tlist, tlist]
# print(list(map(lambda x: sum(x), tlist)))
#
# import sys
# import inspect
#
# clsmembers = inspect.getmembers(sys.modules[__name__], inspect.isclass)
# print(clsmembers)

# error_msg = ('ExecuteError', '[Executing] -4003: Quantity less than zero.')
# print('Quantity less than zero' in str(error_msg))
# try:
#     precision = len(str(price).split('.')[1])
# except:
#     precision = 0
#
# print('precision :', precision)
#
# for back_i in range(5, 0, -1):
#     pass
# print(back_i)

# print(datetime.fromtimestamp(1613759400.004374))
# print(int('08'))
# import os

# tlist = [2, 2, 1, 4, 2]
# print(np.cumprod(tlist))
# # print(tlist.pop(2))
# # print(sum(tlist))
# print(tlist[0:1])
# print(10 ** -2)
