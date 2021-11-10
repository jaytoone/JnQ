import importlib

if __name__ == '__main__':

    trader_name = "SE.traders.SEv4_1_longinv_1028_initaset"
    utils_name = "SE.utils.utilsv4_1_longinv_101523"
    config_name = "configv4_1_longinv_1028.json"

    trader_lib = importlib.import_module(trader_name)
    utils_lib = importlib.import_module(utils_name)

    trader = trader_lib.Trader(utils_lib=utils_lib, config_name=config_name)
    trader.run()
