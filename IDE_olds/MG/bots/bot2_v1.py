import importlib

if __name__ == '__main__':

    trader_name = "MG.traders.MG_v1_112417_marketfunc.py".replace(".py", "")
    utils_name = "MG.utils.utils_v1.py".replace(".py", "")
    config_name = "config2_v1.json"

    trader_lib = importlib.import_module(trader_name)
    utils_lib = importlib.import_module(utils_name)

    trader = trader_lib.Trader(utils_lib=utils_lib, config_name=config_name)
    trader.run()
