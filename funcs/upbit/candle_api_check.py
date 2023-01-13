import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

coin_list = ['BTC', 'ETH', 'BSV']
minute_units = [1, 3, 5, 15, 30, 60, 240]


def to_norm_datetime(unormed_datetime):

    ts = datetime.timestamp(pd.to_datetime(unormed_datetime))

    return datetime.fromtimestamp(ts)


for coin in coin_list:

    for minute in minute_units:

        # end_date = "2021-11-22"
        end_date = "2020-11-22 23:17:00"
        # endTime = datetime.timestamp(pd.to_datetime('{} 23:59:59'.format(end_date))) * 1000
        # endTime = int(endTime)
        # print(type(endTime))

        url = f"https://crix-api-endpoint.upbit.com/v1/crix/candles/minutes/{minute}?" \
              f"code=CRIX.UPBIT.KRW-{coin}&count=1000&to={end_date}"
            # f"code=CRIX.UPBIT.KRW-{coin}&count=400&"

        # print(url)
        # quit()
        req = requests.get(url)
        data = req.json()
        result = []

        # print(data)
        # quit()
        for i, candle in enumerate(data):
            result.append({
                # 'time': data[i]["candleDateTime"],    # utc
                # 'time': data[i]["candleDateTimeKst"],

                # edit to simple datetime format
                'time': to_norm_datetime(data[i]["candleDateTimeKst"]),
                'open': data[i]["openingPrice"],
                'high': data[i]["highPrice"],
                'low': data[i]["lowPrice"],
                'close': data[i]["tradePrice"],
                'volume': data[i]["candleAccTradeVolume"],
                # 'acc_close': data[i]["candleAccTradePrice"]
            })

        coin_data = pd.DataFrame(result).set_index("time").iloc[::-1]
        print(coin_data.head(20))
        print(coin_data.tail(20))
        quit()