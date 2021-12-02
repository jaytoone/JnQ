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


def concat_candlestick(symbol, interval, days, limit=1500, end_date=None, show_process=False, timesleep=None):

    if end_date is None:
        end_date = str(datetime.now()).split(' ')[0]

    startTime_ = datetime.timestamp(pd.to_datetime('{} 00:00:00'.format(end_date))) * 1000

    # if interval != '1m':
    #       trader 에서 자정 지나면 data 부족해지는 문제로 days >= 2 적용    #
    if interval == '1d':
        startTime_ -= a_day

    endTime = datetime.timestamp(pd.to_datetime('{} 23:59:59'.format(end_date))) * 1000

    if show_process:
        print(symbol)

    for day_cnt in range(days):

        if day_cnt != 0:
            startTime_ -= a_day
            endTime -= a_day

        if show_process:
            print(datetime.fromtimestamp(startTime_ / 1000), end=' --> ')
            print(datetime.fromtimestamp(endTime / 1000))

        try:
            startTime = int(startTime_)
            endTime = int(endTime)

            #       Todo        #
            #        추후에 문제없으면, 없앨 예정 -> limit != 1500 에 startTime = None 기입        #
            if limit != 1500:
                startTime = None

            df = request_client.get_candlestick_data(symbol=symbol,
                                                     interval=interval,
                                                     startTime=startTime, endTime=endTime, limit=limit)

            # print("endTime :", endTime)
            # print(df.tail())
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

    days = 300
    # days = 45

    end_date = '2021-07-01'
    end_date = '2021-10-10'
    end_date = None

    # intervals = ['5m', '15m', '30m']
    intervals = ['1m', '3m', '5m', '15m', '30m', '1h', '4h']

    concat_path = '../candlestick_concated/database_bn'

    #       Todo        #
    #        higher timeframe 에 대해서는 days 를 충분히 할당해야할 것      #

    # for interval in intervals:
        # try:
        #     os.makedirs(os.path.join(concat_path, interval))
        # except Exception as e:
        #     print('Error in makedirs :', e)

    # with open('future_coin.p', 'rb') as f:
    #     coin_list = pickle.load(f)
    # with open('ticker_in_futures.txt', 'r') as f:
    #     coin_list = list(f.read())
    with open('../ticker_list/ticker_in_futures.pkl', 'rb') as f:
        coin_list = pickle.load(f)

    #       custom list for yearly survey 0701      #
    # coin_list = ['ETHUSDT', 'BTCUSDT', 'ETCUSDT', 'ADAUSDT', 'XLMUSDT', 'LINKUSDT', 'LTCUSDT', 'EOSUSDT', 'XRPUSDT',
    #              'BCHUSDT']
    coin_list = ['ZECUSDT', 'BNBUSDT', 'RUNEUSDT']
    # print(coin_list)
    # quit()

    # coin_list = coin_list[:10]
    # coin_list = ['ETHUSDT']
    # print(coin_list)
    # coin_list = ['THETA']
    # coin_list.remove('BTC')
    # coin_list.remove('ETH')

    for interval in intervals:

        for coin in coin_list:

            # print(coin)

            #       check existing file     #
            #       Todo        #
            #        1. this phase require valid end_date       #
            save_dir = os.path.join(concat_path)
            # save_dir = os.path.join(concat_path, '%s' % interval)
            save_name = '%s %s_%s.ftr' % (end_date, coin, interval)
            exist_files = os.listdir(save_dir)
            # print(exist_files)
            # print(save_name)
            # quit()

            # if save_name in exist_files:
            #     print(save_name, 'exist !')
            #     continue

            try:
                concated_excel, end_date = concat_candlestick(coin, interval, days,
                                                              end_date=end_date, show_process=True, timesleep=0.3)
                # concated_excel, end_date = concat_candlestick(coin, interval, days, limit=1500, by_limit=False,
                #                                               end_date=end_date, show_process=True, timesleep=0.2)

                # print("str(concated_excel.index[-1]) :", str(concated_excel.index[-1]))
                # quit()

            # try:
                concated_excel.reset_index().to_feather(os.path.join(concat_path, '%s %s_%s.ftr' % (end_date, coin, interval)), compression='lz4')
                # concated_excel.reset_index().to_feather(os.path.join(concat_path, '%s/%s %s.ftr' % (interval, end_date, coin)), compression='lz4')
                # concated_excel.to_excel('./candlestick_concated/%s/%s %s.xlsx' % (interval, end_date, coin + 'USDT'))
            except Exception as e:
                print('Error in save to_excel :', e)
                continue

            # print(concated_excel.tail())
            # print(len(concated_excel))
            # print(concated_excel.head())
            # quit()
            #
            # print('df[-1] timestamp :', datetime.timestamp(concated_excel.index[-1]))
            # print('current timestamp :', datetime.now().timestamp())
            # print(datetime.timestamp(concated_excel.index[-1]) < datetime.now().timestamp())
            # quit()

