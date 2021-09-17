from basic_v1.traders.trader_cloud_closelb_sma_tp2log_0903 import *

if __name__ == '__main__':
    arima_bot = Trader(symbol='ETHUSDT', interval='1m', interval2='3m', leverage_=5, initial_asset=559)
    arima_bot.run()
