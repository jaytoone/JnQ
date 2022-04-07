import pandas as pd
from datetime import datetime
import numpy as np
# from binance_futures_concat_candlestick import concat_candlestick
# from fishing_prev_close.utils import interval_to_min, calc_train_days, tp_update
from funcs.funcs_trader import intmin
import os
import pickle
import math
from pathlib import Path
import json
from easydict import EasyDict
import time
# import logging
import logging.config
from logging.handlers import RotatingFileHandler
import ast

# from funcs_binance.binance_futures_concat_candlestick_ftr import concat_candlestick
from funcs.funcs_trader import load_bin, preproc_bin

short_bin = load_bin("C:\\Users\\Lenovo\\PycharmProjects\\System_Trading\\JnQ\\IDE\\res_bin\\ID3_1_all_05_5_ETHUSDT_short.bin")
# while 1:
#     print(datetime.now().second)
#     print(datetime.now().minute)
#     # print(datetime.now().min)
#     # quit()
#     time.sleep(1)
# print(short_bin)
ini_list = "[11, 21, 19, 46, 29]"
# ini_list = '[11, 21, 19, 46, 29]'
# ini_list = "['1h', '1h', '1h', '1h', '1h']"
res = ast.literal_eval(ini_list)
# res = json.loads(ini_list)
print(res)
print(type(res[0]))
quit()

inval_list = [-11, -1, -11, -1, -1, -1]
# input_arr = np.array([9, 1, 9, 1, 0, 2])
# feature_arr = np.array([9.0, 5, -8.0, 7, 13, 3])
feature_arr = [9.0, 5, -8.0, 7, 13, 2]

preproc_bin(short_bin, inval_list, feature_arr)
print(len(short_bin))
print(short_bin[0])
print(any(np.equal(feature_arr, short_bin).all(axis=1)))
# print(any(np.equal(feature_arr, short_bin).all(axis=1)))

