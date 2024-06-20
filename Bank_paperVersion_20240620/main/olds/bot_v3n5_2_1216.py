import importlib

trader_name = "AT.traders.AT_v3n5_2_1217_posset.py".replace(".py", "")
utils_public_name = "AT.utils.utils_public_1216.py".replace(".py", "")
utils1_name = "AT.utils.utils_v3_1216.py".replace(".py", "")
utils2_name = "AT.utils.utils_v5_2_1216.py".replace(".py", "")
config1_name = "config_v3_1216_chk.json"
config2_name = "config_v5_2_1216_chk.json"

trader_lib = importlib.import_module(trader_name)
utils_public_lib = importlib.import_module(utils_public_name)
utils1_lib = importlib.import_module(utils1_name)
utils2_lib = importlib.import_module(utils2_name)

utils_list = [utils_public_lib, utils1_lib, utils2_lib]
config_list = [config1_name, config2_name]


if __name__ == '__main__':

    # trader_name = "AT.traders.AT_v3n5_2_1217_posset.py".replace(".py", "")
    # utils_public_name = "AT.utils.utils_public_1216.py".replace(".py", "")
    # utils1_name = "AT.utils.utils_v3_1216.py".replace(".py", "")
    # utils2_name = "AT.utils.utils_v5_2_1216.py".replace(".py", "")
    # config1_name = "config_v3_1216.json"
    # config2_name = "config_v5_2_1216.json"
    #
    # trader_lib = importlib.import_module(trader_name)
    # utils_public_lib = importlib.import_module(utils_public_name)
    # utils1_lib = importlib.import_module(utils1_name)
    # utils2_lib = importlib.import_module(utils2_name)
    #
    # utils_list = [utils_public_lib, utils1_lib, utils2_lib]
    # config_list = [config1_name, config2_name]

    trader = trader_lib.Trader(utils_list=utils_list, config_list=config_list)
    trader.run()
