import importlib

if __name__ == '__main__':

    trader_name = "SE.traders.SEv15_3_1028"
    utils_name = "SE.utils.utilsv15_3"
    config_name = "configv15_3_1028.json"

    trader_lib = importlib.import_module(trader_name)
    utils_lib = importlib.import_module(utils_name)

    trader = trader_lib.Trader(utils_lib=utils_lib, config_name=config_name)
    trader.run()