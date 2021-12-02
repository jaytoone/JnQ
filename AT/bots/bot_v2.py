import importlib

if __name__ == '__main__':

    trader_name = "AT.traders.AT_v2_1116_toftr.py".replace(".py", "")
    utils_name = "AT.utils.utils_v2.py".replace(".py", "")
    config_name = "config_v2.json"

    trader_lib = importlib.import_module(trader_name)
    utils_lib = importlib.import_module(utils_name)

    trader = trader_lib.Trader(utils_lib=utils_lib, config_name=config_name)
    trader.run()
