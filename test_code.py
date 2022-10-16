import pandas as pd
# import numpy as np
# from binance_futures_concat_candlestick import concat_candlestick
# from fishing_prev_close.utils import interval_to_min, calc_train_days, tp_update
# from funcs.funcs_trader import intmin
import os
# import pickle
# import math
# from pathlib import Path
# import json
# from easydict import EasyDict
# import time
# import logging
# import logging.config
# from logging.handlers import RotatingFileHandler
# from ast import literal_eval
from datetime import datetime

from funcs_binance.binance_futures_concat_candlestick_ftr import concat_candlestick
# from funcs.funcs_trader import load_bin, preproc_bin
# from funcs_binance.funcs_trader_modules import read_write_cfg_list


res_df_ = pd.read_feather(r"C:\Users\Lenovo\PycharmProjects\System_Trading\JnQ\candlestick_concated\database_bn\2022-07-19\2022-07-19 ETHUSDT_1m.ftr",
                                       columns=None, use_threads=True).set_index("index")
print(res_df_.tail())

# concated_df, end_date = concat_candlestick("ETHUSDT", '1m', days, limit=1500,
#                                                               end_date=None, show_process=True, timesleep=0.2)
str_ts = str(datetime.now())
# str_ts[-9:] = '12.354235'
# print(str_ts.replace(str_ts[-9:], "59.999000"))
print(str_ts.split(':')[-1])
print(str_ts.replace(str_ts.split(':')[-1], "59.999000"))
# print(concated_df.tail())

# config_list = read_write_cfg_list([r"C:\Users\Lenovo\PycharmProjects\System_Trading\JnQ\IDE\config\0405_wave_config_3_5.json"])
# p_ranges, p_qty = literal_eval(config_list[0].tp_set.p_ranges), literal_eval(config_list[0].tp_set.p_qty)
# print(type(p_ranges))

# short_bin = load_bin("C:\\Users\\Lenovo\\PycharmProjects\\System_Trading\\JnQ\\IDE\\res_bin\\ID3_1_all_05_5_ETHUSDT_short.bin")
# # while 1:
# #     print(datetime.now().second)
# #     print(datetime.now().minute)
# #     # print(datetime.now().min)
# #     # quit()
# #     time.sleep(1)
# # print(short_bin)
# ini_list = "[11, 21, 19, 46, 29]"
# # ini_list = '[11, 21, 19, 46, 29]'
# # ini_list = "['1h', '1h', '1h', '1h', '1h']"
# res = ast.literal_eval(ini_list)
# # res = json.loads(ini_list)
# print(res)
# print(type(res[0]))
# quit()
#
# inval_list = [-11, -1, -11, -1, -1, -1]
# # input_arr = np.array([9, 1, 9, 1, 0, 2])
# # feature_arr = np.array([9.0, 5, -8.0, 7, 13, 3])
# feature_arr = [9.0, 5, -8.0, 7, 13, 2]
#
# preproc_bin(short_bin, inval_list, feature_arr)
# print(len(short_bin))
# print(short_bin[0])
# print(any(np.equal(feature_arr, short_bin).all(axis=1)))
# print(any(np.equal(feature_arr, short_bin).all(axis=1)))

