from strat.SE.traders.SEv4_1_noninv_101413_strat_asfunc import *

if __name__ == '__main__':
    trader = Trader(initial_asset=10, config_name="v4_1_noninv.json")
    trader.run()
