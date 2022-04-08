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
from ast import literal_eval

# from funcs_binance.binance_futures_concat_candlestick_ftr import concat_candlestick
from funcs.funcs_trader import load_bin, preproc_bin
# from funcs_binance.funcs_trader_modules import read_write_cfg_list
def read_write_cfg_list(cfg_path_list, mode='r', edited_cfg_list=None):
    try:
        cfg_file_list = [open(cfg_path, mode) for cfg_path in cfg_path_list]
        if mode == 'r':
            cfg_list = [EasyDict(json.load(cfg_)) for cfg_ in cfg_file_list]
        elif mode == 'w':
            assert edited_cfg_list is not None, "assert edited_cfg_list is not None"
            _ = [json.dump(cfg_, cfg_file_, indent=2) for cfg_, cfg_file_ in zip(edited_cfg_list, cfg_file_list)]
        else:
            assert mode in ['r', 'w'], "assert mode in ['r', 'w']"

        #       opened files should be closed --> 닫지 않으면 reopen 시 error occurs         #
        _ = [cfg_.close() for cfg_ in cfg_file_list]

        if mode == 'r':
            return cfg_list
        else:
            return
    except Exception as e:
        print("error in read_write_cfg_list :", e)

# config_list = read_write_cfg_list([r"C:\Users\Lenovo\PycharmProjects\System_Trading\JnQ\IDE\config\0405_wave_config_3_5.json"])
# p_ranges, p_qty = literal_eval(config_list[0].tp_set.p_ranges), literal_eval(config_list[0].tp_set.p_qty)
# print(type(p_ranges))
t_list = [1]
print(eval("t_list[-1] == 0"))
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

