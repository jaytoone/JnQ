from basic_v1.traders.trader_close_cloudlb_partial_tp_0822_calcpr_modi import *

if __name__ == '__main__':
    arima_bot = Trader(symbol='ETHUSDT', interval='1m', interval2='3m', leverage_=6, initial_asset=22)
    arima_bot.run()
