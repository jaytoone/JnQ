from binance_f import RequestClient
from binance_f.model import *
from binance_f.constant.test import *
from binance_f.base.printobject import *

request_client = RequestClient(api_key=g_api_key, secret_key=g_secret_key)


# import json
from datetime import datetime
import pandas as pd

import pickle
import random

date = '2020-01-04'
startTime = datetime.timestamp(pd.to_datetime('%s 00:00:00' % date)) * 1000
endTime = datetime.timestamp(pd.to_datetime('%s 23:59:59' % date)) * 1000
# print(startTime, endTime)
# print(datetime.timestamp(pd.to_datetime(startTime)))

with open('future_coin.p', 'rb') as f:
    coin_list = pickle.load(f)

for coin in coin_list:

    try:
        print(coin)
        result = request_client.get_candlestick_data(symbol=coin + 'USDT', interval=CandlestickInterval.MIN1,
                                                 startTime=startTime, endTime=endTime, limit=1500)
        print(result.tail())
        # quit()
    except Exception as e:
        print(e)
