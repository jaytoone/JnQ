import pybithumb
from datetime import datetime

interval_list = ['1m', '3m', '5m']
folder = None
Date = str(datetime.now()).split()[0]
for index, interval in enumerate(interval_list):
    Coin_list = pybithumb.get_tickers()
    for i, Coin in enumerate(Coin_list):
        try:
            df = pybithumb.get_candlestick(Coin, chart_instervals=interval)

            if index == 0:
                folder = 'ohlcv'
            elif index == 1:
                folder = 'ohlcv_min3'
            elif index == 1:
                folder = 'ohlcv_min5'

            df.to_excel("C:/Users/Lenovo/OneDrive/CoinBot/%s/%s %s ohlcv.xlsx" % (folder, Date, Coin))
            print(Coin, '%.2f' % ((i + 1) / len(Coin_list) * 100))

        except Exception as e:
            print(e)