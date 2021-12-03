from strat.SE.traders.SEv4_1_longinv_1021_feecalc import *

if __name__ == '__main__':
    trader = Trader(initial_asset=110, config_name="configv4_1_longinv_1021.json")
    trader.run()
