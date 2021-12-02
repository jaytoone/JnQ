import pandas as pd
from datetime import datetime
import numpy as np
# from binance_futures_concat_candlestick import concat_candlestick
# from fishing_prev_close.utils import interval_to_min, calc_train_days, tp_update
from funcs.funcs_for_trade import intmin
import os
import pickle
import math
from pathlib import Path
import json
from easydict import EasyDict
import time
# import logging
import logging.config
# from logging.handlers import RotatingFileHandler

from binance_funcs.binance_futures_concat_candlestick_ftr import concat_candlestick


# def log_str(logger_, idx):
#
#     logger_.info("rot_codes %s" % idx)
#     # logger_.debug("rot_codes")
#
#     return
#
#
# # for i in range(5):
#
# with open("test_log2.json", 'rt') as f:
#     log_cfg = EasyDict(json.load(f))
#
# # print(log_cfg.handlers.file_rot.filename)
# # quit()
# log_cfg.handlers.file_rot.filename = "test_log_dir/mod_log.log"
# logging.config.dictConfig(log_cfg)
#
# # logger = logging.getLogger()
# logger = logging.getLogger("JnQ")
#
# logger.error("error occursion {} {}\n"
#              .format("hell", 1.12312412))
# logger.error(None)
# print("", end)
# for i in range(5):
#     # print("passed")
#     log_str(logger, i)

