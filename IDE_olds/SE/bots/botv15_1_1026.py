import importlib

if __name__ == '__main__':

    trader_name = "SE.traders.SEv15_1_1026"
    utils_name = "SE.utils.utilsv15_1"
    config_name = "configv15_1_1026.json"

    trader_lib = importlib.import_module(trader_name)
    utils_lib = importlib.import_module(utils_name)

    trader = trader_lib.Trader(utils_lib=utils_lib, config_name=config_name)
    trader.run()