import importlib

trader_name = "AT.traders.AT_v3n5_2n7_3_0101_mrlog.py".replace(".py", "")

#       rtc & tr format     #
utils1_name = "AT.utils.utils_v3_1216.py".replace(".py", "")
utils2_name = "AT.utils.utils_v5_2_1216.py".replace(".py", "")
utils_public_name = "AT.utils.utils_public_0106_v7_3_candle2_1.py".replace(".py", "")

#       format setter       #
config1_name = "config_v3_1231_v7_3.json"
config2_name = "config_v5_2_1231_v7_3.json"
config3_name = "config_v7_3_1231_v7_3.json"

trader_lib = importlib.import_module(trader_name)
utils_public_lib = importlib.import_module(utils_public_name)
utils1_lib = importlib.import_module(utils1_name)
utils2_lib = importlib.import_module(utils2_name)

utils_list = [utils_public_lib, utils1_lib, utils2_lib]
config_list = [config1_name, config2_name, config3_name]


if __name__ == '__main__':

    trader = trader_lib.Trader(utils_list=utils_list, config_list=config_list)
    trader.run()
