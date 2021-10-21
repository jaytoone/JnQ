from SE.traders.SEv15_102115_loaddf import *

if __name__ == '__main__':
    trader = Trader(initial_asset=10, config_name="configv15.json")
    trader.run()
