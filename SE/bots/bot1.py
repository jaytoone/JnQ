from SE.traders.SEv4_1_100713_partialtp import *

if __name__ == '__main__':
    trader = Trader(symbol='ETHUSDT', interval='1m', interval2='3m', interval3='5m',  leverage_=1, initial_asset=10)
    trader.run()
