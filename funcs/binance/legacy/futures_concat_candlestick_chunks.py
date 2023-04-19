import pandas as pd
import os
import pickle
from datetime import datetime
from binance_f import RequestClient
from binance_f.model import *
from binance_f.constant.test import *
from binance_f.base.printobject import *
import time

request_client = RequestClient(api_key=g_api_key, secret_key=g_secret_key)

pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 2500)


#           1 day = 86399000. (timestamp)       #
a_day = 3600 * 24 * 1000


def concat_candlestick(symbol, interval, days, end_date=None, show_process=False, timesleep=None):

    if end_date is None:
        end_date = str(datetime.now()).split(' ')[0]

    startTime = datetime.timestamp(pd.to_datetime('{} 00:00:00'.format(end_date))) * 1000
    if interval != '1m':
        startTime -= a_day
    endTime = datetime.timestamp(pd.to_datetime('{} 23:59:59'.format(end_date))) * 1000

    if show_process:
        print(symbol)

    for day_cnt in range(days):

        if day_cnt != 0:
            startTime -= a_day
            endTime -= a_day

        if show_process:
            print(datetime.fromtimestamp(startTime / 1000), end=' --> ')
            print(datetime.fromtimestamp(endTime / 1000))

        try:
            df = request_client.get_candlestick_data(symbol=symbol,
                                                     interval=interval,
                                                     startTime=startTime, endTime=endTime, limit=1500)
            # print(result.tail())
            # quit()

            assert len(df) != 0, "len(df) == 0"

            if day_cnt == 0:
                sum_df = df
            else:
                # print(df.head())
                # sum_df = pd.concat([sum_df, df])
                sum_df = df.append(sum_df)  # <-- -a_day 이기 때문에 sum_df 와 df 의 위치가 좌측과 같다.
                # print(sum_df)
                # quit()

            if timesleep is not None:
                time.sleep(timesleep)

        except Exception as e:
            print('error in get_candlestick_data :', e)

            if len(df) == 0:
                # quit()
                break

    # end_date = str(datetime.fromtimestamp(endTime / 1000)).split(' ')[0]
    # print(len(sum_df[~sum_df.index.duplicated(keep='first')]))

    # keep = 'last' 로 해야 중복기준 최신 df 를 넣는건데, 왜 first 로 해놓은거지
    return sum_df[~sum_df.index.duplicated(keep='last')], end_date


if __name__ == '__main__':

    days = 30
    # days = 21
    end_date = '2021-05-17'
    # end_date = '2021-04-18'
    # end_date = '2020-10-20'
    # end_date = '2021-06-15'
    # end_date = '2019-12-08'
    # end_date = None

    days_iter = 14

    intervals = ['30m']
    intervals = ['1m']
    # intervals = ['3m']

    for interval in intervals:

        try:
            os.makedirs(os.path.join('./database', interval))
        except Exception as e:
            print('Error in makedirs :', e)

    with open('future_coin.p', 'rb') as f:
        coin_list = pickle.load(f)

    # coin_list = coin_list[11:]
    coin_list = ['NEO']
    coin_list = ['ETH']
    # print(coin_list)
    # coin_list = ['THETA']
    # coin_list.remove('BTC')
    # coin_list.remove('ETH')

    for interval in intervals:

        for coin in coin_list:

            # Todo : chuck by end_date  #
            for iter_i in range(days_iter):

                if iter_i != 0:
                    # end_date = (initial) end_date - days * index --> timestamp 에서 datetime 으로 변경해주어야함 #
                    next_endTime = datetime.timestamp(pd.to_datetime('{} 23:59:59'.format(end_date))) * 1000 - a_day * days
                    end_date = str(datetime.fromtimestamp(next_endTime / 1000)).split(" ")[0]
                    print("end_date :", end_date)
                    print()

                try:
                    concated_excel, end_date = concat_candlestick(coin + 'USDT', interval, days, end_date=end_date, show_process=True, timesleep=0.2)

                # try:
                    concated_excel.to_excel('./database/%s/%s %s.xlsx' % (interval, end_date, coin + 'USDT'))
                except Exception as e:
                    print('Error in to_excel :', e)
                    continue

            # print(concated_excel.tail())
            #
            # print('df[-1] timestamp :', datetime.timestamp(concated_excel.index[-1]))
            # print('current timestamp :', datetime.now().timestamp())
            # print(datetime.timestamp(concated_excel.index[-1]) < datetime.now().timestamp())
            # quit()
