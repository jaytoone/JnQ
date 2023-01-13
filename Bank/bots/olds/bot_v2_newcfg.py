import importlib

if __name__ == '__main__':

    trader_name = "AT.traders.AT_v2_111618_newcfg.py".replace(".py", "")
    utils_name = "AT.utils.utils_v2_newcfg.py".replace(".py", "")
    config_name = "config_v2_newcfg.json"

    trader_lib = importlib.import_module(trader_name)
    utils_lib = importlib.import_module(utils_name)

    trader = trader_lib.Trader(utils_lib=utils_lib, config_name=config_name)
    trader.run()
