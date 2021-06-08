from datetime import datetime
from binance_f import RequestClient
from binance_f.model import *
from binance_f.constant.test import *
from binance_f.base.printobject import *

import pickle
import random

with open('future_coin.p', 'rb') as f:
    coin_list = pickle.load(f)

request_client = RequestClient(api_key=g_api_key, secret_key=g_secret_key)


interval_list = ['1m', '3m', '5m']
folder = None
date = str(datetime.now()).split()[0]
for index, interval in enumerate(interval_list):
    for i, coin in enumerate(coin_list):

        try:
            df = request_client.get_candlestick_data(symbol=coin + 'USDT', interval=interval,
                                                         startTime=None, endTime=None, limit=10)
            if interval == CandlestickInterval.MIN1:
                folder = 'ohlcv'
            elif interval == CandlestickInterval.MIN3:
                folder = 'ohlcv_min3'
            elif interval == CandlestickInterval.MIN5:
                folder = 'ohlcv_min5'

            df.to_excel("C:/Users/Lenovo/OneDrive/CoinBot/%s/%s %s ohlcv.xlsx" % (folder, date, coin))
            print(coin, '%.2f' % ((i + 1) / len(coin_list) * 100))

        except Exception as e:
            print('Error in get_candlestick_data :', e)
