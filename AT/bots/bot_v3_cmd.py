import importlib
import os
import sys

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
super_dir = os.path.dirname(parent_dir)
print("super_dir :", super_dir)

sys.path.insert(0, super_dir)   # exists for importlib in cmd
os.chdir(super_dir)     # exists for pycharm env

if __name__ == '__main__':

    trader_name = "AT.traders.AT_v3_1207_cmd.py".replace(".py", "")
    utils_name = "AT.utils.utils_v3_logger_chk.py".replace(".py", "")
    config_name = "config_v3_logger_chk.json"

    trader_lib = importlib.import_module(trader_name)
    utils_lib = importlib.import_module(utils_name)

    trader = trader_lib.Trader(utils_lib=utils_lib, config_name=config_name)
    trader.run()
