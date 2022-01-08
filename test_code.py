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
from logging.handlers import RotatingFileHandler

# from funcs_binance.binance_futures_concat_candlestick_ftr import concat_candlestick

sys_log_path = r"C:\Users\Lenovo\PycharmProjects\System_Trading\JnQ\AT\sys_log\test_cfg.json"
# with open(sys_log_path, 'rt') as sys_cfg:
#     # sys_log_cfg = EasyDict(json.load(sys_cfg))
#     sys_log_cfg = json.load(sys_cfg)

# sys_log_path = r"C:\Users\Lenovo\PycharmProjects\System_Trading\JnQ\AT\sys_log\test_cfg.cfg"
# logging.config.fileConfig(sys_log_path)

# sys_log_cfg.handlers.file_rot.filename = sys_log_path.replace("test_cfg.json", "test_log.log")

# res = logging.config.dictConfig(sys_log_cfg)


# create and start listener on port 9999
t = logging.config.listen(9999)
t.start()

logging.getLogger("apscheduler.executors.default").propagate = False

# logging.config.stopListening()
# t.join()
# quit()
# sys_log = logging.getLogger()

initial_run = 1
while 1:
    print(logging.getLogger("apscheduler.executors.default").propagate)

    with open(sys_log_path, 'r') as sys_cfg:
        sys_log_cfg = EasyDict(json.load(sys_cfg))
        # sys_log_cfg = json.load(sys_cfg)

        if initial_run:
            logging.getLogger("apscheduler.executors.default").propagate = True
            initial_run = 0

        # sys_log_cfg.handlers.file_rot.filename = sys_log_path.replace("test_cfg.json", "test.log")

        logging.config.dictConfig(sys_log_cfg)
        sys_log = logging.getLogger()

    # print("sys_log_cfg.handlers.file_rot.filename :", sys_log_cfg.handlers.file_rot.filename)
    # logging.config.fileConfig(sys_log_path)

    # t.join()

    # with open("test_log2.json", 'rt') as f:
    #     log_cfg = EasyDict(json.load(f))
    sys_log.info("rot_codes %s" % time.time())
    time.sleep(1)


# print(log_cfg.handlers.file_rot.filename)
# quit()
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

