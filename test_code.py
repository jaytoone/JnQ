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

# from funcs_binance.binance_futures_concat_candlestick_ftr import concat_candlestick
from funcs.funcs_trader import load_bin

load_arr = load_bin("C:\\Users\\Lenovo\\PycharmProjects\\System_Trading\\JnQ\\IDE\\res_bin\\ID3_1_all_05_5_SOLUSDT_long.bin")
# print(any(np.equal([-3.0, 1, -11, -1, 15, -1], arr_).all(1)))
# feature_arr = np.array([ 9,  6, -9,  5, 13,  4.])
feature_arr = np.array([-3.0, 1, -11, -1, 15, -1])
feature_arr = np.array([-10, -1, 9, 9, 9, 4])
print(feature_arr)
# print(any(np.equal(feature_arr, load_arr).all(axis=1)))
print(any((feature_arr == load_arr).all(axis=1)))
print(len(load_arr))
print(load_arr)

