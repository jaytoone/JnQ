from basic_v1.trader_cloudlb import *

if __name__ == '__main__':
    arima_bot = Trader(symbol='ETHUSDT', interval='1m', interval2='3m', leverage_=5, initial_asset=10)
    arima_bot.run()
