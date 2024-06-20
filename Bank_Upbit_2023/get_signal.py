import os, sys

pkg_path = 'D:\\Projects\\System_Trading\\JnQ\\'

os.chdir(pkg_path)

# mpl_finance_path = 'D:\\python\\python38_1\\projects\\JnQ\\mpl_finance'
# ta_lib_path = 'D:\\python\\python38_1\\projects\\JnQ\\ta_lib'
funcs_path = pkg_path + 'funcs'

if funcs_path not in sys.path:

  try:
    # sys.path.insert(0, '/content/drive/My Drive/Colab Notebooks/JnQ')
    sys.path.insert(0, pkg_path + 'Bank')
    sys.path.insert(0, funcs_path)
    # sys.path.insert(0, mpl_finance_path)
    # sys.path.insert(0, ta_lib_path)
    
  except Exception as e:
    print(e)
    
    
import os
import talib
from funcs.public.idep import *
from funcs.public.plot_check import *
from funcs.public.en_ex_pairing import *
from funcs.public.indicator import *
from funcs.public.broker import *
from funcs.public.ds import *
from ast import literal_eval
import logging
import importlib


import matplotlib.pyplot as plt
from matplotlib import gridspec

# import torch

import numpy as np
import pandas as pd
import scipy.stats as stats
# from sklearn.metrics.pairwise import cosine_similarity

# import bz2
import pickle
# import _pickle as cPickle
import shutil
import json
from easydict import EasyDict
import copy

import datetime
from datetime import datetime
import random
import time
# import warnings

from IPython.display import clear_output
# warnings.simplefilter("ignore", category=RuntimeWarning)

np.seterr(invalid="ignore")
np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=2000) 

pd.set_option('mode.chained_assignment',  None)

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)



import telegram
from telegram.ext import Updater
from telegram.ext import MessageHandler, Filters

def echo(self, update, context):
    self.user_text = update.message.text
    
#       i. Telegram logger
#               1. chat_id 는 env 동일.
token = "6717940201:AAFc62YReeED3mJKw5drHnT7jp6dzXy0rPE"
msg_bot = telegram.Bot(token=token)
chat_id = "5320962614"

#       ii. Telegram messenger
#           1. init.
user_text = None

updater = Updater(token=token, use_context=True)
dispatcher = updater.dispatcher

echo_handler = MessageHandler(Filters.text & (~Filters.command), echo)
dispatcher.add_handler(echo_handler)

#           2. polling for messenger.
updater.start_polling()


from funcs.upbit.concat_candlestick_ftr_v2 import concat_candlestick
from pyupbit import get_tickers

"""
1. upbit 에서 당일 기준 이전 데이터를 조회할지라도 None 은 성립하지 않는다. --> end_date = today 로 변경 요구.
"""


days = 1  # 330 3

end_date = None  # None 2023-07-04 # "2023-01-06" "2021-04-12" "2021-03-23"
    
intervals = [ 'D']  # ['1m', '3m', '5m', '15m', '30m', '1h', '4h'] - old
intervals = ['30m', '1h', '2h']  # ['1m', '3m', '5m', '15m', '30m', '1h', '4h'] - old
intervals = ['4h', '1d']  # ['1m', '3m', '5m', '15m', '30m', '1h', '4h'] - old

time_schedule = {'4h': [21, 17, 13, 9, 5, 1], '1d': [9]}
    
show_process = False
show_process = True



with open("signal/upbit_fluc_max.pkl", 'rb') as f:
    fluc_dict = pickle.load(f)
    
# hour_prev = None    
symbol_list = get_tickers()

while 1:

    time.sleep(1)  # term for small fan sound.
    
    target_symbol_total = []
    for interval in intervals:

        datetime_now = datetime.now()
        timestamp = datetime_now.timestamp()
        hour_current = datetime.fromtimestamp(timestamp).hour
        hour_prev = datetime.fromtimestamp(timestamp - 100).hour  # 100 seconds before.
        
        print("hour_current :", hour_current)
        print("hour_prev :", hour_prev)
        print("time_schedule[interval] :", time_schedule[interval])
        print()
        
        if not ((hour_current in time_schedule[interval]) and hour_current != hour_prev):
            continue

        target_symbol = []
        for s_i, symbol in enumerate(symbol_list):

            print('s_i :', s_i)

            if 'KRW' not in symbol:
                continue

            try:
                concated_df, end_date = concat_candlestick(symbol, 
                                                           interval, 
                                                           days, 
                                                           limit=limit_by_itv(interval),
                                                           end_date=end_date, 
                                                           show_process=show_process, 
                                                           timesleep=0.1)


                concated_df2 = fisher_v2(concated_df, 30, itv=interval)    
                # display(concated_df2.tail())
                # print(concated_df2.tail())
                # break

                target_data = concated_df2.iloc[-2:, -1].to_numpy()

                """
                condition
                """
                # 1. fisher band
                fisher_lower = -1.5
                if target_data[0] < fisher_lower < target_data[1]:
                    target_symbol.append(symbol)

                clear_output(wait=True)          



                # break
                # save_path = os.path.join(save_dir, save_name)
                # concated_df.reset_index().to_feather(save_path, compression='lz4')
                # print(save_path, "saved.\n")
            except Exception as e:
                print("error in save to_feather :", e)
                continue

        target_symbol_total = np.array([[s_.replace('KRW-', ''), fluc_dict[s_]] for s_ in target_symbol])
        if len(target_symbol_total) > 0:
            target_symbol_total = target_symbol_total[target_symbol_total[:, 1].astype(float).argsort()[::-1]]
            print("target_symbol_total :", target_symbol_total)

            msg = "target_symbol_total : {} {}".format(interval, target_symbol_total)
            try:
                msg_bot.sendMessage(chat_id=chat_id, text=msg)
            except Exception as e:
                print("error in msg_bot : {}".format(e))
        # break