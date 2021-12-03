import importlib

if __name__ == '__main__':

    trader_name = "SE.traders.SEv15_102117_importlib"
    utils_name = "SE.utils.utilsv15"
    config_name = "configv15.json"

    trader_lib = importlib.import_module(trader_name)
    utils_lib = importlib.import_module(utils_name)

    trader = trader_lib.Trader(initial_asset=1700, utils_lib=utils_lib, config_name=config_name)
    trader.run()